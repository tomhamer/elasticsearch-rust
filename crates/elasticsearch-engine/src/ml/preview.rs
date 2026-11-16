//! `_ml/datafeeds[/{id}]/_preview`: the data a datafeed would send to its
//! job — source documents (scroll datafeeds) or flattened aggregation
//! buckets (`AggregationToJsonProcessor`).

use super::anomaly::parse_datafeed_time;
use super::store::service;
use super::xc::{bad_request, Pos};
use super::{datafeed, job};
use crate::error::{EsError, Result};
use crate::node::Node;
use serde_json::{json, Map, Value};

/// Preview a stored datafeed (`id`) or an inline `datafeed_config` /
/// `job_config` body.
pub fn preview(node: &Node, id: Option<&str>, body: &Value, start: Option<&str>, end: Option<&str>) -> Result<Value> {
    let st = service(node).read();
    let dfb = body.get("datafeed_config").filter(|v| !v.is_null()).cloned();
    let jb = body.get("job_config").filter(|v| !v.is_null()).cloned();
    let (df, job) = match id {
        Some(i) => {
            if dfb.is_some() || jb.is_some() {
                return Err(bad_request("Params [datafeed_id] and [datafeed_config|job_config] are mutually exclusive"));
            }
            let df = st.datafeeds.get(i).cloned().ok_or_else(|| EsError::resource_not_found(format!("No datafeed with id [{i}] exists")))?;
            let jid = df["job_id"].as_str().unwrap_or("").to_string();
            let job = st.jobs.get(&jid).cloned().ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{jid}'")))?;
            (df, job)
        }
        None => {
            let nested = jb.as_ref().and_then(|j| j.get("datafeed_config")).filter(|v| !v.is_null()).cloned();
            let d = match (dfb, nested) {
                (Some(_), Some(_)) => return Err(bad_request("[datafeed_config] must not be present when a [job_config.datafeed_config] is present")),
                (Some(d), None) | (None, Some(d)) => d,
                (None, None) => return Err(bad_request("[datafeed_config] must be present when a [job_config.datafeed_config] is not present")),
            };
            let job = match &jb {
                Some(j) => {
                    let mut j = j.clone();
                    if let Some(o) = j.as_object_mut() {
                        o.remove("datafeed_config");
                        o.remove("job_id");
                    }
                    let raw = j.to_string();
                    let nj = job::parse_put("preview", &j, &raw)?;
                    job::finish("preview", &nj, job::DEFAULT_MML_MB, 0)
                }
                None => {
                    let jid = d.get("job_id").and_then(Value::as_str).ok_or_else(|| bad_request("[job_id] must be set in the datafeed_config when no job_config is supplied"))?;
                    st.jobs.get(jid).cloned().ok_or_else(|| EsError::resource_not_found(format!("No known job with id '{jid}'")))?
                }
            };
            let mut o = d.as_object().cloned().unwrap_or_default();
            o.insert("job_id".into(), json!(job["job_id"].as_str().unwrap_or("preview")));
            o.remove("datafeed_id");
            (datafeed::build("preview-datafeed", &o, Pos::empty(), &Map::new())?, job)
        }
    };
    drop(st);
    let start = start.map(|s| parse_datafeed_time(s, "start")).transpose()?;
    let end = end.map(|s| parse_datafeed_time(s, "end")).transpose()?;
    run(node, &df, &job, start, end)
}

fn analysis_fields(job: &Value) -> Vec<String> {
    let mut fields: Vec<String> = Vec::new();
    let mut add = |f: &str| {
        if f != "mlcategory" && !fields.iter().any(|x| x == f) {
            fields.push(f.to_string());
        }
    };
    for d in job["analysis_config"]["detectors"].as_array().into_iter().flatten() {
        for k in ["field_name", "by_field_name", "over_field_name", "partition_field_name"] {
            if let Some(f) = d[k].as_str() {
                add(f);
            }
        }
    }
    for k in ["summary_count_field_name", "categorization_field_name"] {
        if let Some(f) = job["analysis_config"][k].as_str() {
            add(f);
        }
    }
    for i in job["analysis_config"]["influencers"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        add(i);
    }
    fields
}

fn run(node: &Node, df: &Value, job: &Value, start: Option<i64>, end: Option<i64>) -> Result<Value> {
    fetch(node, df, job, start, end, 1000)
}

/// The documents a datafeed sends for `[start, end)` (time field as epoch
/// millis), at most `size` of them in time order (the preview: the first
/// page). The datafeed runner pages through a window with [`Extractor`].
pub fn fetch(node: &Node, df: &Value, job: &Value, start: Option<i64>, end: Option<i64>, size: usize) -> Result<Value> {
    if df.get("aggregations").is_some_and(Value::is_object) {
        // The preview: the first page of a composite aggregation.
        return fetch_aggregated(node, df, job, start, end, None).map(|(docs, _)| Value::Array(docs));
    }
    check_indices(node, df)?;
    let src = Source::new(df, job);
    let resp = super::search(node, &src.target, &src.body(df, start, end, size))?;
    Ok(Value::Array(resp["hits"]["hits"].as_array().into_iter().flatten().map(|h| src.doc(h)).collect()))
}

/// A concrete missing index is a 404 (ES fails the preview search).
fn check_indices(node: &Node, df: &Value) -> Result<()> {
    for i in df["indices"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        if !i.contains('*') && node.resolve(i, crate::node::ResolveOpts { allow_no_indices: true, ..Default::default() }).map(|t| t.is_empty()).unwrap_or(true) {
            return Err(EsError::index_not_found(i));
        }
    }
    Ok(())
}

fn time_query(df: &Value, time_field: &str, start: Option<i64>, end: Option<i64>) -> Value {
    let mut range = Map::new();
    if let Some(s) = start {
        range.insert("gte".into(), json!(s));
    }
    if let Some(e) = end {
        range.insert("lt".into(), json!(e));
    }
    range.insert("format".into(), json!("epoch_millis"));
    json!({"bool": {"filter": [df["query"].clone(), {"range": {time_field.to_string(): range}}]}})
}

/// The source side of a scroll datafeed: its indices, the fields it sends
/// and how a hit becomes a document.
struct Source {
    target: String,
    time_field: String,
    fields: Vec<String>,
}

impl Source {
    fn new(df: &Value, job: &Value) -> Source {
        let time_field = job["data_description"]["time_field"].as_str().unwrap_or("time").to_string();
        let target = df["indices"].as_array().into_iter().flatten().filter_map(Value::as_str).collect::<Vec<_>>().join(",");
        let mut fields = vec![time_field.clone()];
        fields.extend(analysis_fields(job).into_iter().filter(|f| *f != time_field));
        Source { target, time_field, fields }
    }

    /// Time-ordered search of `[start, end)`.
    fn body(&self, df: &Value, start: Option<i64>, end: Option<i64>, size: usize) -> Value {
        let tf = &self.time_field;
        let mut search = json!({"size": size, "query": time_query(df, tf, start, end), "sort": [{tf.clone(): {"order": "asc"}}], "_source": false,
            "fields": self.fields.iter().map(|f| if f == tf { json!({"field": f, "format": "epoch_millis"}) } else { json!({"field": f}) }).collect::<Vec<_>>()});
        if let Some(rm) = df.get("runtime_mappings") {
            search["runtime_mappings"] = rm.clone();
        }
        if let Some(sf) = df.get("script_fields") {
            search["script_fields"] = sf.clone();
        }
        search
    }

    fn doc(&self, h: &Value) -> Value {
        let mut doc = Map::new();
        for f in &self.fields {
            let Some(arr) = h["fields"].get(f).and_then(Value::as_array) else { continue };
            let v = if *f == self.time_field {
                let first = arr.first().cloned().unwrap_or(Value::Null);
                match &first {
                    Value::String(s) => s.split('.').next().and_then(|x| x.parse::<i64>().ok()).map(|x| json!(x)).unwrap_or(first),
                    other => other.clone(),
                }
            } else if arr.len() == 1 {
                arr[0].clone()
            } else {
                Value::Array(arr.clone())
            };
            doc.insert(f.clone(), v);
        }
        Value::Object(doc)
    }
}

/// Extracts one datafeed search window `[start, end)` (ES
/// `ScrollDataExtractor` / `AggregationDataExtractor`): a scroll in time
/// order, so documents sharing a timestamp are never skipped or repeated
/// across pages. A failed page leaves the extractor where it was; the next
/// call retries it.
pub struct Extractor {
    df: Value,
    end: i64,
    start: i64,
    size: usize,
    aggregated: bool,
    agg_done: bool,
    /// A composite aggregation's `after_key` of the last page handed out.
    after: Option<Value>,
    source: Source,
    scroll: Option<super::Scroll>,
    /// Timestamp of the last document handed out, and the identities
    /// (`_index` / `_id`) of the documents with it that were handed out, to
    /// resume a lost scroll exactly: a reopened search from that timestamp
    /// skips exactly those, so a deleted or newly inserted tied document
    /// neither hides an unread one nor is repeated.
    last: Option<(i64, std::collections::HashSet<String>)>,
    /// A reopened scroll is still at `last`'s timestamp.
    resuming: bool,
}

/// A hit's identity across searches (`_index` and `_id`).
fn identity(h: &Value) -> String {
    format!("{}\u{0}{}", h["_index"].as_str().unwrap_or(""), h["_id"].as_str().unwrap_or(""))
}

impl Extractor {
    pub fn new(df: &Value, job: &Value, start: i64, end: i64, size: usize) -> Extractor {
        let aggregated = df.get("aggregations").is_some_and(Value::is_object);
        Extractor { df: df.clone(), end, start, size, aggregated, agg_done: false, after: None, source: Source::new(df, job), scroll: None, last: None, resuming: false }
    }

    /// The next documents of the window; empty once it is exhausted.
    pub fn next(&mut self, node: &Node, job: &Value) -> Result<Vec<Value>> {
        if self.aggregated {
            // One search (one page of a composite aggregation, resumed
            // after the last page's `after_key`) per call; a failed or
            // partial page leaves the position where it was.
            while !self.agg_done {
                let (docs, after) = fetch_aggregated(node, &self.df, job, Some(self.start), Some(self.end), self.after.as_ref())?;
                match after {
                    Some(a) => self.after = Some(a),
                    None => self.agg_done = true,
                }
                if !docs.is_empty() {
                    return Ok(docs);
                }
            }
            return Ok(Vec::new());
        }
        loop {
            if self.scroll.is_none() {
                check_indices(node, &self.df)?;
                let from = self.last.as_ref().map(|(t, _)| *t).unwrap_or(self.start);
                let body = self.source.body(&self.df, Some(from), Some(self.end), self.size);
                self.scroll = Some(super::Scroll::new(&self.source.target, body));
            }
            let scroll = self.scroll.as_mut().expect("scroll opened above");
            let hits = match scroll.next_page(node) {
                Ok(h) => h,
                Err(e) => {
                    if scroll.lost() {
                        // Reopen from the last timestamp, skipping the
                        // documents at it that were already handed out.
                        self.scroll = None;
                        self.resuming = self.last.is_some();
                    }
                    return Err(e);
                }
            };
            if hits.is_empty() {
                return Ok(Vec::new());
            }
            let mut docs = Vec::with_capacity(hits.len());
            for h in &hits {
                let doc = self.source.doc(h);
                let ts = doc.get(&self.source.time_field).and_then(Value::as_i64);
                let id = identity(h);
                if self.resuming {
                    match (&self.last, ts) {
                        (Some((t, seen)), Some(x)) if x == *t => {
                            if seen.contains(&id) {
                                continue;
                            }
                        }
                        _ => self.resuming = false,
                    }
                }
                if let Some(t) = ts {
                    match &mut self.last {
                        Some((l, seen)) if *l == t => {
                            seen.insert(id);
                        }
                        last => *last = Some((t, std::iter::once(id).collect())),
                    }
                }
                docs.push(doc);
            }
            if !docs.is_empty() {
                return Ok(docs);
            }
        }
    }
}

impl Extractor {
    /// Tests: lose the open scroll's position.
    #[cfg(test)]
    pub(crate) fn lose_scroll(&mut self) {
        if let Some(s) = self.scroll.as_mut() {
            s.lose();
        }
    }
}

/// The datafeed's top-level composite aggregation: its name and the names
/// of its `date_histogram` sources.
fn composite_of(aggs: &Value) -> Option<(String, Vec<String>)> {
    aggs.as_object()?.iter().find_map(|(name, a)| {
        let sources = a.get("composite")?["sources"].as_array().cloned().unwrap_or_default();
        let dates = sources.iter().filter_map(Value::as_object).flat_map(|s| s.iter()).filter(|(_, v)| v.get("date_histogram").is_some()).map(|(k, _)| k.clone()).collect();
        Some((name.clone(), dates))
    })
}

/// One search of an aggregated datafeed over `[start, end)` (a composite
/// aggregation: the page after `after`): the flattened rows, and the
/// composite's next `after_key` (`None` once its pages are exhausted, and
/// for other aggregations). A partial response (shard failures, timeout)
/// is an error: the runner retries the window rather than analysing part
/// of it.
fn fetch_aggregated(node: &Node, df: &Value, job: &Value, start: Option<i64>, end: Option<i64>, after: Option<&Value>) -> Result<(Vec<Value>, Option<Value>)> {
    check_indices(node, df)?;
    let time_field = job["data_description"]["time_field"].as_str().unwrap_or("time").to_string();
    let target = df["indices"].as_array().into_iter().flatten().filter_map(Value::as_str).collect::<Vec<_>>().join(",");
    let mut aggs = df["aggregations"].clone();
    let composite = composite_of(&aggs);
    if let (Some((name, _)), Some(after)) = (&composite, after) {
        aggs[name.as_str()]["composite"]["after"] = after.clone();
    }
    let mut search = json!({"size": 0, "query": time_query(df, &time_field, start, end), "aggs": aggs});
    if let Some(rm) = df.get("runtime_mappings") {
        search["runtime_mappings"] = rm.clone();
    }
    #[allow(unused_mut)]
    let mut resp = super::search(node, &target, &search)?;
    #[cfg(test)]
    if partial_left(&format!("{}#{target}", node.data.display())) {
        // A 200 response one shard of which failed.
        resp["_shards"]["failed"] = json!(1);
    }
    super::require_complete(&resp)?;
    let summary = job["analysis_config"]["summary_count_field_name"].as_str().map(str::to_string);
    let include_doc_count = summary.as_deref() == Some("doc_count");
    let mut out = Vec::new();
    let mut next = None;
    for (name, agg) in resp["aggregations"].as_object().into_iter().flatten() {
        let buckets = agg["buckets"].as_array().cloned().unwrap_or_default();
        let dates: Option<&Vec<String>> = composite.as_ref().filter(|(n, _)| n == name).map(|(_, d)| d);
        if dates.is_some() && !buckets.is_empty() {
            next = agg.get("after_key").filter(|k| !k.is_null()).cloned();
        }
        for b in &buckets {
            if b["doc_count"].as_i64().unwrap_or(0) == 0 {
                continue;
            }
            let mut base = Map::new();
            let key_time = match dates {
                // Composite: the date_histogram source's key is the bucket
                // time; every other source's key is a field of the row.
                Some(dates) => {
                    for (src, v) in b["key"].as_object().into_iter().flatten() {
                        if !dates.contains(src) && !v.is_null() {
                            base.insert(src.clone(), v.clone());
                        }
                    }
                    dates.iter().find_map(|d| b["key"][d.as_str()].as_f64())
                }
                None => b["key"].as_f64(),
            };
            let t = b.get(&time_field).and_then(|m| m["value"].as_f64()).or(key_time).unwrap_or(0.0);
            base.insert(time_field.clone(), json!(t as i64));
            flatten_bucket(b, base, &time_field, include_doc_count, &mut out);
        }
    }
    Ok((out, next))
}

fn flatten_bucket(b: &Value, base: Map<String, Value>, time_field: &str, include_doc_count: bool, out: &mut Vec<Value>) {
    let mut doc = base;
    let mut multi: Vec<(&String, &Value)> = Vec::new();
    for (k, v) in b.as_object().into_iter().flatten() {
        if matches!(k.as_str(), "key" | "key_as_string" | "doc_count" | "from" | "to" | "from_as_string" | "to_as_string") || k == time_field {
            continue;
        }
        if v.get("buckets").is_some() {
            multi.push((k, v));
        } else if let Some(val) = v.get("value") {
            if !val.is_null() {
                doc.insert(k.clone(), val.clone());
            }
        }
    }
    if multi.is_empty() {
        if include_doc_count {
            doc.insert("doc_count".into(), b["doc_count"].clone());
        }
        out.push(Value::Object(doc));
        return;
    }
    for (name, agg) in multi {
        for sb in agg["buckets"].as_array().into_iter().flatten() {
            if sb["doc_count"].as_i64().unwrap_or(0) == 0 {
                continue;
            }
            let mut d = doc.clone();
            d.insert(name.clone(), sb["key"].clone());
            flatten_bucket(sb, d, time_field, include_doc_count, out);
        }
    }
}

/// Test builds: searches of these targets (`<node data>#<target>`) answer
/// partially (one shard failed) this many more times.
#[cfg(test)]
pub(crate) static PARTIAL: parking_lot::Mutex<Vec<(String, u32)>> = parking_lot::Mutex::new(Vec::new());

#[cfg(test)]
fn partial_left(target: &str) -> bool {
    let mut p = PARTIAL.lock();
    match p.iter_mut().find(|(t, n)| t == target && *n > 0) {
        Some(e) => {
            e.1 -= 1;
            true
        }
        None => false,
    }
}
