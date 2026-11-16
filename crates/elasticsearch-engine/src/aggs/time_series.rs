//! `time_series` bucket aggregation (one bucket per `_tsid` of a
//! `time_series` index) and the counter `rate` computed inside it.

use super::super::{apply_parent_pipelines, Col, Ctx, Docs, Key};
use super::STACK;
use crate::error::{EsError, Result};
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// Java `Map.toString` of the (sorted) dimension map: `{a=x, b=y}`.
fn key_string(key: &Map<String, Value>) -> String {
    let parts: Vec<String> = key
        .iter()
        .map(|(k, v)| {
            let s = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            format!("{k}={s}")
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// ES's legacy tsid of a dimension map (`TimeSeriesIdFieldMapper
/// .buildLegacyTsid`), the `time_series` buckets' sort key: the dimension
/// count (vint), then per dimension in name order its name (vint length +
/// UTF-8) and value (`s` + vint length + UTF-8, `l` + big-endian long,
/// `u` + the shifted unsigned long, `b` + `t` / `f`).
fn legacy_tsid(key: &Map<String, Value>) -> Vec<u8> {
    fn vint(mut n: usize, out: &mut Vec<u8>) {
        while n >= 0x80 {
            out.push((n as u8 & 0x7f) | 0x80);
            n >>= 7;
        }
        out.push(n as u8);
    }
    let mut entries: Vec<(&String, &Value)> = key.iter().collect();
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut out = Vec::with_capacity(64);
    vint(entries.len(), &mut out);
    for (name, v) in entries {
        vint(name.len(), &mut out);
        out.extend_from_slice(name.as_bytes());
        match v {
            Value::Bool(b) => out.extend_from_slice(if *b { b"bt" } else { b"bf" }),
            Value::Number(n) if n.as_i64().is_some() => {
                out.push(b'l');
                out.extend_from_slice(&n.as_i64().unwrap_or(0).to_be_bytes());
            }
            Value::Number(n) if n.as_u64().is_some() => {
                out.push(b'u');
                out.extend_from_slice(&((n.as_u64().unwrap_or(0) ^ (1 << 63)) as i64).to_be_bytes());
            }
            other => {
                let text = match other {
                    Value::String(t) => t.clone(),
                    o => o.to_string(),
                };
                out.push(b's');
                vint(text.len(), &mut out);
                out.extend_from_slice(text.as_bytes());
            }
        }
    }
    out
}

/// Enclosing aggregation type `n` levels up (0 = the current one).
fn enclosing(n: usize) -> Option<String> {
    STACK.with(|s| {
        let s = s.borrow();
        s.len().checked_sub(n + 1).and_then(|i| s.get(i).cloned())
    })
}

impl<'a, 'b> Ctx<'a, 'b> {
    fn tsid_of(&self, s: usize, d: DocId) -> Option<String> {
        let f = self.field(crate::tsdb::TSID_FIELD).ok()?;
        match self.values(&f, s, d).into_iter().next() {
            Some(Key::Str(t)) => Some(t),
            _ => None,
        }
    }

    pub(super) fn time_series(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        if enclosing(1).as_deref() == Some("auto_date_histogram") {
            return Err(EsError::illegal_argument("Wrapping a time-series rate aggregation within a DeferableBucketAggregator is not supported"));
        }
        let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(true);
        // ES `TimeSeriesAggregationBuilder.DEFAULT_SIZE`: the default max buckets.
        let size = o.get("size").and_then(Value::as_i64).unwrap_or(65_536);
        if size <= 0 {
            return Err(EsError::illegal_argument(format!("[size] must be greater than 0. Found [{size}] in [{name}]")));
        }
        let tsid_field = self.field(crate::tsdb::TSID_FIELD)?;
        let mut groups: HashMap<String, Docs> = HashMap::new();
        for (s, v) in docs.iter().enumerate() {
            if self.segs[s].lookup.get(crate::tsdb::TSID_FIELD).is_none() {
                continue;
            }
            // A sealed TSID column is single-valued. Group by its local
            // ordinal before decoding the term, rather than allocating and
            // translating the same series ID for every matching document.
            if let Some(col) = self.col(&tsid_field, s) {
                if let Col::Kw(reader) = &*col {
                    let mut local: HashMap<u32, Vec<(DocId, f32)>> = HashMap::new();
                    for &(d, sc) in v {
                        let ord = reader.ord(d);
                        if ord != rustic_core::codec::fastfields::MISSING_ORD {
                            local.entry(ord).or_default().push((d, sc));
                        }
                    }
                    for (ord, selected) in local {
                        if let Some(stored) = reader.term(ord) {
                            let t = crate::tsdb::tsid_display(&stored);
                            groups.entry(t).or_insert_with(|| vec![Vec::new(); docs.len()])[s].extend(selected);
                        }
                    }
                    continue;
                }
            }
            for &(d, sc) in v {
                if let Some(Key::Str(t)) = self.values(&tsid_field, s, d).into_iter().next() {
                    groups.entry(t).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc));
                }
            }
        }
        // ES orders the buckets by the legacy tsid of their dimension map
        // (`TimeSeriesAggregator`: `buildLegacyTsid`, then a sort by those
        // bytes), not by `_tsid` (a hash); `size` keeps the first ones.
        // Groups of one map (series of several indices) merge.
        let gs: Vec<(String, Docs)> = groups.into_iter().collect();
        // The dimension fields of each segment's mapping, resolved once (not
        // per series: a mapping has thousands of fields).
        let mut dims_of: HashMap<usize, std::rc::Rc<Vec<(String, super::super::FieldRef)>>> = HashMap::new();
        let mut keyed_groups: Vec<(Vec<u8>, String, Map<String, Value>, Docs)> = Vec::with_capacity(gs.len());
        for (_, d) in gs {
            let Some((s, doc)) = d.iter().enumerate().find_map(|(s, v)| v.first().map(|x| (s, x.0))) else { continue };
            let lookup_at = self.segs[s].lookup as *const _ as usize;
            let dims = dims_of.entry(lookup_at).or_insert_with(|| std::rc::Rc::new(self.dimension_fields(s))).clone();
            let key = self.dimension_key_of(&dims, s, doc);
            keyed_groups.push((legacy_tsid(&key), key_string(&key), key, d));
        }
        keyed_groups.sort_by(|a, b| a.0.cmp(&b.0));
        // Groups merge only when their structured dimension maps are equal
        // (the legacy tsid: typed, length-prefixed, collision-free), never
        // because their display strings match (`{a=x, b=y, b=z}` renders
        // distinct maps alike).
        let mut by_key: Vec<(Vec<u8>, String, Map<String, Value>, Docs)> = Vec::new();
        for (lt, ks, key, d) in keyed_groups {
            match by_key.last_mut() {
                Some(last) if last.0 == lt => {
                    for (i, v) in d.into_iter().enumerate() {
                        last.3[i].extend(v);
                        last.3[i].sort_by_key(|x| x.0);
                    }
                }
                _ => by_key.push((lt, ks, key, d)),
            }
        }
        by_key.truncate(size as usize);
        // A keyed response names each bucket by its display string: two
        // distinct series with one display string cannot both be returned
        // that way (one would overwrite the other).
        if keyed {
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            if let Some(dup) = by_key.iter().find(|b| !seen.insert(b.1.as_str())) {
                return Err(EsError::illegal_argument(format!(
                    "time_series aggregation [{name}]: distinct time series have the same bucket key [{}]; use [keyed: false] to return them as separate buckets",
                    dup.1
                )));
            }
        }
        self.add_buckets(by_key.len())?;
        let mut out: Vec<(String, Map<String, Value>)> = Vec::new();
        for (_, ks, key, d) in by_key {
            let mut m = Map::new();
            m.insert("key".into(), Value::Object(key));
            m.insert("doc_count".into(), json!(super::super::count(&d)));
            m.extend(self.subs(sub, &d)?);
            out.push((ks, m));
        }
        let mut maps: Vec<Map<String, Value>> = out.into_iter().map(|x| x.1).collect();
        apply_parent_pipelines(sub, &mut maps)?;
        if keyed {
            // Each retained bucket named by its own key: a `bucket_sort`
            // may have reordered or paged them.
            let obj: Map<String, Value> = maps
                .into_iter()
                .map(|m| (m.get("key").and_then(Value::as_object).map(key_string).unwrap_or_default(), Value::Object(m)))
                .collect();
            Ok(json!({ "buckets": obj }))
        } else {
            Ok(json!({ "buckets": maps }))
        }
    }

    /// The dimension fields of segment `s`'s mapping, by name.
    fn dimension_fields(&self, s: usize) -> Vec<(String, super::super::FieldRef)> {
        let lookup = self.segs[s].lookup;
        let mut dims: Vec<String> = lookup.fields.iter().filter(|(_, f)| f.time_series_dimension).map(|(n, _)| n.clone()).collect();
        // A flattened field's `time_series_dimensions` keys are dimensions
        // too (ES `RootFlattenedFieldType`), named `<field>.<key>`.
        for (n, f) in &lookup.fields {
            if f.declared != crate::mapping::EsType::Flattened {
                continue;
            }
            let keys = f.ext.as_ref().and_then(|e| e.params.get("time_series_dimensions").and_then(Value::as_array).cloned()).unwrap_or_default();
            dims.extend(keys.iter().filter_map(Value::as_str).map(|k| format!("{n}.{k}")));
        }
        dims.sort();
        dims.dedup();
        dims.into_iter().filter_map(|n| { let f = self.field(&n).ok()?; Some((n, f)) }).collect()
    }

    /// The dimension values of a doc, by field name ([`Self::dimension_fields`]).
    fn dimension_key_of(&self, dims: &[(String, super::super::FieldRef)], s: usize, d: DocId) -> Map<String, Value> {
        let mut m = Map::new();
        for (n, f) in dims {
            let vals = self.values(f, s, d);
            let Some(v) = vals.into_iter().next() else { continue };
            let j = match (v, f.ty) {
                (Key::Bool(b), _) => json!(b),
                (k, _) => super::super::key_json(&k),
            };
            m.insert(n.clone(), j);
        }
        m
    }

    /// `rate` of a `counter` metric inside a `time_series` bucket: the
    /// increase (with counter resets) between the first and last sample
    /// over their time span; the last sample before the bucket (from the
    /// same series) starts the span, as ES's time-ordered collection does.
    pub(super) fn counter_rate(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Option<Value>> {
        if enclosing(1).as_deref() != Some("time_series") {
            return Ok(None);
        }
        let Some(field) = o.get("field").and_then(Value::as_str) else { return Ok(None) };
        let Some(rf) = self.segs.iter().find_map(|s| s.lookup.get(field)) else { return Ok(None) };
        if rf.time_series_metric.as_deref() != Some("counter") {
            return Ok(None);
        }
        let unit_ms: f64 = match o.get("unit").and_then(Value::as_str) {
            None | Some("second" | "1s") => 1_000.0,
            Some("minute" | "1m") => 60_000.0,
            Some("hour" | "1h") => 3_600_000.0,
            Some("day" | "1d") => 86_400_000.0,
            Some("week" | "1w") => 604_800_000.0,
            Some("month" | "1M") => 2_629_746_000.0,
            Some("quarter" | "1q") => 7_889_238_000.0,
            Some("year" | "1y") => 31_556_952_000.0,
            Some(u) => return Err(EsError::illegal_argument(format!("Unsupported unit {u}"))),
        };
        let _ = name;
        let f = self.field(field)?;
        let ts = self.field("@timestamp")?;
        let sample = |s: usize, d: DocId| -> Option<(i64, f64)> {
            let t = match self.values(&ts, s, d).into_iter().next()? {
                Key::Long(t) => t,
                k => k.as_f64()? as i64,
            };
            let v = self.values(&f, s, d).into_iter().find_map(|k| k.as_f64())?;
            Some((t, v))
        };
        let mut pts: Vec<(i64, f64)> = Vec::new();
        let mut tsid = None;
        for (s, v) in docs.iter().enumerate() {
            for &(d, _) in v {
                if tsid.is_none() {
                    tsid = self.tsid_of(s, d);
                }
                if let Some(p) = sample(s, d) {
                    pts.push(p);
                }
            }
        }
        pts.sort_by_key(|p| p.0);
        if let (Some(first), Some(tsid)) = (pts.first().copied(), tsid) {
            // The latest matching sample of the series before the bucket.
            let mut prev: Option<(i64, f64)> = None;
            for (s, seg) in self.segs.iter().enumerate() {
                for &d in &seg.docs {
                    if let Some(p) = sample(s, d) {
                        if p.0 < first.0 && prev.map(|x| p.0 > x.0).unwrap_or(true) && self.tsid_of(s, d).as_deref() == Some(tsid.as_str()) {
                            prev = Some(p);
                        }
                    }
                }
            }
            if let Some(p) = prev {
                pts.insert(0, p);
            }
        }
        let mut increase = 0.0;
        for w in pts.windows(2) {
            increase += if w[1].1 >= w[0].1 { w[1].1 - w[0].1 } else { w[1].1 };
        }
        let mut out = if pts.len() < 2 || pts[pts.len() - 1].0 == pts[0].0 {
            json!({ "value": "NaN" })
        } else {
            let dt = (pts[pts.len() - 1].0 - pts[0].0) as f64;
            json!({ "value": crate::values::java_double(increase / dt * 1000.0 * (unit_ms / 1000.0)) })
        };
        // ES `InternalResetTrackingRate`: a shard response carries the
        // series' end points and its increase so the coordinator merges a
        // series that spans backing indices (on several shards) before
        // normalising once, instead of adding per-shard rates.
        if crate::search::aggregation_shard_response() {
            if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
                out["__p"] = json!({ "counter": { "st": first.0, "sv": first.1, "et": last.0, "ev": last.1, "inc": increase, "unit_ms": unit_ms } });
            }
        }
        Ok(Some(out))
    }
}

/// Coordinator reduce of counter `rate`s inside `time_series` buckets (ES
/// `InternalResetTrackingRate.reduce`): the shards' pieces of one series in
/// time order, the increase across each boundary (a lower start than the
/// previous end is a counter reset), normalised once over the whole span.
/// `None` when a shard response lacks the state (an older node) or none
/// has samples.
pub fn reduce_counter_rates(vals: &[&Value]) -> Option<Value> {
    let mut parts: Vec<(i64, f64, i64, f64, f64, f64)> = Vec::new();
    for v in vals {
        match v.get("__p").and_then(|p| p.get("counter")) {
            Some(c) => parts.push((c["st"].as_i64()?, c["sv"].as_f64()?, c["et"].as_i64()?, c["ev"].as_f64()?, c["inc"].as_f64()?, c["unit_ms"].as_f64()?)),
            // A shard without samples of the series.
            None if v.get("value").and_then(Value::as_str) == Some("NaN") => {}
            None => return None,
        }
    }
    if parts.is_empty() {
        // No shard has samples of the series (all NaN).
        return Some(json!({ "value": "NaN" }));
    }
    parts.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
    let mut increase: f64 = parts.iter().map(|p| p.4).sum();
    for w in parts.windows(2) {
        let (prev_end, next_start) = (w[0].3, w[1].1);
        increase += if next_start >= prev_end { next_start - prev_end } else { next_start };
    }
    let (start, end, unit_ms) = (parts[0].0, parts.iter().map(|p| p.2).max().unwrap_or(parts[0].2), parts[0].5);
    if end == start {
        return Some(json!({ "value": "NaN" }));
    }
    Some(json!({ "value": crate::values::java_double(increase / (end - start) as f64 * unit_ms) }))
}

#[cfg(test)]
mod identity_tests {
    use crate::node::Node;
    use serde_json::{json, Value};

    /// Two series whose dimension maps render alike: `{a=x, b=y, b=z}`.
    fn fixture(node: &Node) {
        let now = crate::date::now_millis();
        let base = now - now.rem_euclid(3_600_000) - 3_600_000;
        node.create_index("t218", &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": ["a", "b"], "index.number_of_replicas": 0,
                "index.time_series.start_time": crate::ilm::iso_date(base - 3_600_000), "index.time_series.end_time": crate::ilm::iso_date(now + 3_600_000)},
            "mappings": {"properties": {"@timestamp": {"type": "date"},
                "a": {"type": "keyword", "time_series_dimension": true}, "b": {"type": "keyword", "time_series_dimension": true},
                "v": {"type": "double", "time_series_metric": "gauge"}}}}))
            .unwrap();
        let svc = node.get_index("t218").unwrap();
        let docs = [json!({"a": "x, b=y", "b": "z", "v": 1.0}), json!({"a": "x", "b": "y, b=z", "v": 100.0})];
        let reqs = docs
            .iter()
            .map(|d| {
                let mut d = d.clone();
                d["@timestamp"] = json!(crate::ilm::iso_date(base + 1000));
                crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }
            })
            .collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
    }

    fn agg(node: &Node, keyed: bool) -> crate::error::Result<Value> {
        let svc = node.get_index("t218").unwrap();
        let body = json!({"size": 0, "aggs": {"ts": {"time_series": {"keyed": keyed}, "aggs": {"s": {"sum": {"field": "v"}}}}}});
        let req = crate::search::parse_request(&body, &std::collections::HashMap::new())?;
        let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = crate::search::execute(&[crate::search::Target::new(svc)], &req, &opts)?;
        Ok(serde_json::from_str(&out.json)?)
    }

    /// Review finding 218: series whose dimension maps share a display
    /// string stay two buckets with their own metrics (keyed: false); a
    /// keyed response, which cannot name both, refuses instead of merging
    /// or overwriting them; ES|QL `_tsid` metadata stays distinct.
    #[test]
    fn display_string_collisions_never_merge_series() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        fixture(&node);
        let v = agg(&node, false).unwrap();
        let buckets = v["aggregations"]["ts"]["buckets"].as_array().cloned().unwrap_or_default();
        assert_eq!(buckets.len(), 2, "{v}");
        let mut sums: Vec<f64> = buckets.iter().map(|b| b["s"]["value"].as_f64().unwrap_or(f64::NAN)).collect();
        sums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sums, vec![1.0, 100.0], "{v}");
        assert!(buckets.iter().all(|b| b["doc_count"] == 1), "{v}");
        assert!(agg(&node, true).is_err(), "a keyed response cannot hold both buckets");
        let out = crate::esql::run(&node, "FROM t218 METADATA _tsid | KEEP _tsid", crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }).unwrap();
        let mut ids: Vec<String> = out.table.rows.iter().map(|r| r[0].to_text()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 2, "distinct _tsid metadata: {ids:?}");
    }
}
