//! Analytics and ML aggregations (x-pack `analytics` / `ml` plugins):
//! `matrix_stats`, `counted_terms`, `random_sampler`, `time_series`,
//! `frequent_item_sets` (`frequent_items`), `categorize_text`, the parent
//! pipelines `normalize`, `cumulative_cardinality`, `moving_percentiles` and
//! the sibling pipelines `bucket_correlation`, `bucket_count_ks_test`,
//! `change_point`; plus `histogram`-field support in `range` / `rate` /
//! `percentiles` (`hdr`).
//!
//! A child module of `aggs` so it can use the framework's private context
//! (`Ctx`, `Key`, `Docs`, helpers). Registry (in `aggs/mod.rs`): [`AGGS`],
//! [`PARENT`], [`SIBLING`] and the hooks [`Ctx::analytics_agg`],
//! [`Ctx::analytics_override`], [`Ctx::analytics_prepare`],
//! [`parent_pipeline`], [`sibling_pipeline`].

use super::{count, fnum, jfmt, key_json, opt_num, resolve_path, Ctx, Docs, FieldRef, Key};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

#[path = "categorize.rs"]
pub(crate) mod categorize;
#[path = "frequent_items.rs"]
pub mod frequent_items;
#[path = "ml_pipelines.rs"]
mod ml_pipelines;
#[path = "time_series.rs"]
mod time_series;

/// Bucket / metric aggregations implemented here.
pub(super) const AGGS: &[&str] = &["matrix_stats", "counted_terms", "random_sampler", "time_series", "frequent_item_sets", "frequent_items", "categorize_text"];
/// Parent pipelines implemented here (also listed in `PARENT_PIPELINES`).
pub(crate) const PARENT: &[&str] = &["normalize", "cumulative_cardinality", "moving_percentiles", "serial_diff", "moving_fn"];
/// Sibling pipelines implemented here (also listed in `SIBLING_PIPELINES`).
pub(crate) const SIBLING: &[&str] = &["bucket_correlation", "bucket_count_ks_test", "change_point"];
/// Aggregations that reject sub-aggregations.
const NO_SUBS: &[&str] = &["frequent_item_sets", "frequent_items", "matrix_stats"];

/// Key of the per-bucket sketch payload a parent pipeline needs (the raw
/// cardinality set / percentiles values of the referenced metric); never
/// rendered: [`parent_pipeline`] strips it after the pipelines ran.
pub(crate) const HIDDEN: &str = "\u{1}p5";
/// Histogram totals up to which values are expanded into the t-digest.
const EXPAND_LIMIT: u64 = 100_000;

thread_local! {
    /// Types of the enclosing aggregations, innermost last.
    static STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Marks `t` as the enclosing aggregation while its sub-aggregations run.
pub(super) struct AggGuard;

impl AggGuard {
    pub(super) fn push(t: &str) -> AggGuard {
        STACK.with(|s| s.borrow_mut().push(t.to_string()));
        AggGuard
    }
}

impl Drop for AggGuard {
    fn drop(&mut self) {
        STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

#[allow(dead_code)]
fn parent_type() -> Option<String> {
    STACK.with(|s| s.borrow().last().cloned())
}

fn validation(msg: impl Into<String>) -> EsError {
    let m: String = msg.into();
    EsError::action_validation(&[m.as_str()])
}

/// First aggregation name of a `buckets_path` (`a>b`, `a.metric`, `a[x]`).
fn path_head(p: &str) -> &str {
    let first = p.split('>').next().unwrap_or(p);
    let first = first.split('[').next().unwrap_or(first);
    first.split('.').next().unwrap_or(first)
}

fn buckets_path(body: &Value) -> Result<String> {
    match body.get("buckets_path") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => Ok(a[0].as_str().unwrap().to_string()),
        Some(Value::Object(m)) if m.len() == 1 => Ok(m.values().next().and_then(Value::as_str).unwrap_or_default().to_string()),
        _ => Err(EsError::illegal_argument("Required [buckets_path]")),
    }
}

// ---------------------------------------------------------------------------
// java.text.DecimalFormat (the `format` of pipeline values).

/// `java.text.DecimalFormat(pattern).format(x)` for the common patterns
/// (`0`, `#`, `,`, `.`, literal prefix/suffix, `%`, `‰`), HALF_EVEN.
pub(crate) fn decimal_format(pattern: &str, x: f64) -> String {
    let pat = pattern.split(';').next().unwrap_or(pattern);
    let is_num = |c: char| matches!(c, '0' | '#' | ',' | '.');
    let start = pat.find(is_num).unwrap_or(pat.len());
    let end = pat[start..].find(|c: char| !is_num(c)).map(|i| start + i).unwrap_or(pat.len());
    let (prefix, num, suffix) = (&pat[..start], &pat[start..end], &pat[end..]);
    let unquote = |s: &str| s.replace('\'', "");
    if x.is_nan() {
        return "NaN".into();
    }
    let mut v = x;
    if prefix.contains('%') || suffix.contains('%') {
        v *= 100.0;
    } else if prefix.contains('‰') || suffix.contains('‰') {
        v *= 1000.0;
    }
    if v.is_infinite() {
        return format!("{}{}∞{}", if v < 0.0 { "-" } else { "" }, unquote(prefix), unquote(suffix));
    }
    let (int_p, frac_p) = num.split_once('.').unwrap_or((num, ""));
    let min_frac = frac_p.chars().filter(|c| *c == '0').count();
    let max_frac = frac_p.chars().filter(|c| *c == '0' || *c == '#').count();
    let min_int = int_p.chars().filter(|c| *c == '0').count();
    let group = int_p.rfind(',').map(|i| int_p[i + 1..].chars().filter(|c| *c == '0' || *c == '#').count()).filter(|g| *g > 0);
    let s = format!("{:.*}", max_frac, v.abs());
    let (i, f) = s.split_once('.').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((s.clone(), String::new()));
    let mut f = f;
    while f.len() > min_frac && f.ends_with('0') {
        f.pop();
    }
    let mut i = i.trim_start_matches('0').to_string();
    while i.len() < min_int {
        i.insert(0, '0');
    }
    if let Some(g) = group {
        let b: Vec<char> = i.chars().collect();
        let mut out = String::new();
        for (n, c) in b.iter().enumerate() {
            if n > 0 && (b.len() - n) % g == 0 {
                out.push(',');
            }
            out.push(*c);
        }
        i = out;
    }
    if i.is_empty() && f.is_empty() {
        i.push('0');
    }
    // Java prints the sign even when the value rounds to zero ("-0.0").
    let negative = v.is_sign_negative();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    out.push_str(&unquote(prefix));
    out.push_str(&i);
    if !f.is_empty() {
        out.push('.');
        out.push_str(&f);
    }
    out.push_str(&unquote(suffix));
    out
}

/// `{"value": v}` (+ `value_as_string` under a `format`), ES
/// `InternalSimpleValue`: non-finite values render as `null`.
fn simple_value(v: f64, format: Option<&str>) -> Value {
    let mut m = Map::new();
    m.insert("value".into(), opt_num(v.is_finite().then_some(v)));
    if let (Some(f), true) = (format, v.is_finite()) {
        m.insert("value_as_string".into(), json!(decimal_format(f, v)));
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// Java HashMap iteration order (matrix_stats renders its per-field maps in
// `HashMap<String, _>` order).

fn java_string_hash(s: &str) -> i32 {
    let mut h: i32 = 0;
    for u in s.encode_utf16() {
        h = h.wrapping_mul(31).wrapping_add(u as i32);
    }
    h
}

/// The iteration order of a `java.util.HashMap` holding `keys` (inserted in
/// this order).
pub(super) fn java_hashmap_order(keys: &[String]) -> Vec<String> {
    let mut cap = 16usize;
    while keys.len() as f64 > cap as f64 * 0.75 {
        cap *= 2;
    }
    let mut idx: Vec<(usize, usize, &String)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let h = java_string_hash(k) as u32;
            let spread = h ^ (h >> 16);
            ((spread as usize) & (cap - 1), i, k)
        })
        .collect();
    idx.sort_by_key(|x| (x.0, x.1));
    idx.into_iter().map(|x| x.2.clone()).collect()
}

fn xparse(agg: &str, field: Option<&str>, cause: EsError) -> EsError {
    let reason = match field {
        Some(f) => format!("[{agg}] failed to parse field [{f}]"),
        None => format!("Failed to build [{agg}] after last required field arrived"),
    };
    EsError::new(400, "x_content_parse_exception", reason).caused_by(cause)
}

/// Request-level validation of an aggregation tree (what ES rejects while
/// parsing / validating the request). `parent` is the enclosing type.
pub(super) fn validate_tree(spec: &Map<String, Value>, parent: Option<&str>) -> Result<()> {
    let mut entries: Vec<(&str, &str, &Value, Option<&Map<String, Value>>)> = Vec::new();
    for (name, def) in spec {
        let Some(d) = def.as_object() else { continue };
        let sub = d.get("aggs").or(d.get("aggregations")).and_then(Value::as_object);
        let Some((t, body)) = d.iter().find(|(k, _)| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")) else { continue };
        entries.push((name.as_str(), t.as_str(), body, sub));
    }
    let names: HashSet<&str> = entries.iter().map(|e| e.0).collect();
    for &(name, t, body, sub) in &entries {
        if NO_SUBS.contains(&t) && sub.is_some_and(|m| !m.is_empty()) {
            return Err(EsError::illegal_argument(format!("Aggregator [{name}] of type [{t}] cannot accept sub-aggregations")));
        }
        match t {
            "random_sampler" => {
                if let Some(p) = body.get("probability").and_then(Value::as_f64) {
                    if !((p > 0.0 && p <= 0.5) || p == 1.0) {
                        return Err(xparse(t, Some("probability"), EsError::illegal_argument(format!("[probability] must be between 0.0 and 0.5 or exactly 1.0, was [{}]", jfmt(p)))));
                    }
                }
            }
            "categorize_text" => {
                if let Some(th) = body.get("similarity_threshold").and_then(Value::as_i64) {
                    if !(1..=100).contains(&th) {
                        return Err(xparse(t, Some("similarity_threshold"), EsError::new(400, "status_exception", format!("[similarity_threshold] must be in the range [1, 100]. Found [{th}] in [{name}]"))));
                    }
                }
                if let Some(o) = body.as_object() {
                    let fi = o.keys().position(|k| k == "categorization_filters");
                    let ai = o.keys().position(|k| k == "categorization_analyzer");
                    if let (Some(fi), Some(ai)) = (fi, ai) {
                        let later = if fi > ai { "categorization_filters" } else { "categorization_analyzer" };
                        return Err(xparse(t, Some(later), EsError::new(400, "status_exception", "[categorization_filters] cannot be used with [categorization_analyzer] - instead specify them as pattern_replace char_filters in the analyzer")));
                    }
                }
            }
            "inference" => {
                // `InferencePipelineAggregationBuilder.validate`.
                let mut errs: Vec<String> = Vec::new();
                if parent.is_none() {
                    errs.push(format!("inference aggregation [{name}] must be declared inside of another aggregation"));
                }
                if body.get("model_id").is_none_or(Value::is_null) {
                    errs.push("[model_id] must be set".into());
                }
                if let Some((kind, cfg)) = body.get("inference_config").and_then(Value::as_object).and_then(|o| o.iter().next()) {
                    if let Some(rf) = cfg.get("results_field").and_then(Value::as_str).filter(|s| !s.is_empty() && *s != "value") {
                        errs.push(format!("setting option [results_field] to [{rf}] is not valid for inference aggregations"));
                    }
                    if kind == "classification" {
                        if let Some(tc) = cfg.get("top_classes_results_field").and_then(Value::as_str).filter(|s| !s.is_empty() && *s != "top_classes") {
                            errs.push(format!("setting option [top_classes] to [{tc}] is not valid for inference aggregations"));
                        }
                    }
                }
                if !errs.is_empty() {
                    let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
                    return Err(EsError::action_validation(&refs));
                }
            }
            _ => {}
        }
        if matches!(t, "bucket_selector" | "bucket_script") {
            // ES `AggregatorFactories` resolves every path's first element
            // against the sibling aggregations.
            let paths: Vec<&str> = match body.get("buckets_path") {
                Some(Value::Object(m)) => m.values().filter_map(Value::as_str).collect(),
                Some(Value::String(p)) => vec![p.as_str()],
                _ => Vec::new(),
            };
            for p in paths {
                let head = path_head(p);
                if !matches!(head, "_count" | "_key") && !names.contains(head) {
                    return Err(EsError::illegal_argument(format!("No aggregation found for path [{p}]")));
                }
            }
        }
        if PARENT.contains(&t) || SIBLING.contains(&t) {
            if t == "normalize" {
                let m = body.get("method").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("Required [method]"))?;
                if !matches!(m, "rescale_0_1" | "rescale_0_100" | "percent_of_sum" | "mean" | "z-score" | "softmax") {
                    return Err(xparse(t, None, EsError::illegal_argument(format!("invalid method [{m}]"))));
                }
            }
            if t == "moving_percentiles" {
                let w = body.get("window").and_then(Value::as_i64).unwrap_or(0);
                if w <= 0 {
                    return Err(xparse(t, Some("window"), xparse(t, None, EsError::illegal_argument("[window] must be a positive, non-zero integer."))));
                }
            }
            if t == "bucket_count_ks_test" && body.get("alternative").and_then(Value::as_array).is_some_and(|a| a.is_empty()) {
                return Err(xparse(t, None, EsError::illegal_argument(format!("[alternative] must not be empty for aggregation [{name}]"))));
            }
            let path = buckets_path(body)?;
            if SIBLING.contains(&t) {
                ml_pipelines::validate(name, t, body)?;
            }
            let head = path_head(&path);
            if !(PARENT.contains(&t) && matches!(head, "_count" | "_key")) && !names.contains(head) {
                return Err(validation(format!("No aggregation found for path [{path}]")));
            }
            if t == "normalize" {
                if parent.is_none() {
                    return Err(validation(format!("{t} aggregation [{name}] must be declared inside of another aggregation")));
                }
            } else if PARENT.contains(&t) {
                match parent {
                    None => return Err(validation(format!("{t} aggregation [{name}] must have a histogram, date_histogram or auto_date_histogram as parent but doesn't have a parent"))),
                    Some("histogram" | "date_histogram" | "auto_date_histogram") => {}
                    Some(_) => return Err(validation(format!("{t} aggregation [{name}] must have a histogram, date_histogram or auto_date_histogram as parent"))),
                }
            }
        }
        if let Some(sub) = sub {
            validate_tree(sub, Some(t))?;
        }
    }
    Ok(())
}

impl<'a, 'b> Ctx<'a, 'b> {
    /// After this level's aggregations ran: attach the sketch payloads the
    /// `cumulative_cardinality` / `moving_percentiles` pipelines of the
    /// enclosing multi-bucket aggregation merge across buckets.
    pub(super) fn analytics_prepare(&self, entries: &[(String, String, Value, bool)], results: &mut HashMap<String, Value>, docs: &Docs) -> Result<()> {
        if super::partial::active() {
            return Ok(());
        }
        for (_, t, body, _) in entries {
            if t != "cumulative_cardinality" && t != "moving_percentiles" {
                continue;
            }
            let path = buckets_path(body)?;
            let target = path_head(&path);
            let Some((_, tt, tbody, _)) = entries.iter().find(|e| e.0 == target) else { continue };
            let Some(o) = tbody.as_object() else { continue };
            let payload = match (t.as_str(), tt.as_str()) {
                ("cumulative_cardinality", "cardinality") => {
                    let f = self.source(o, target, "cardinality")?;
                    let missing = self.missing_key(&f, o.get("missing"))?;
                    let mut set: Vec<String> = Vec::new();
                    let mut seen = HashSet::new();
                    for (s, v) in docs.iter().enumerate() {
                        for &(d, _) in v {
                            for k in self.values_or(&f, s, d, &missing) {
                                let ks = format!("{k:?}");
                                if seen.insert(ks.clone()) {
                                    set.push(ks);
                                }
                            }
                        }
                    }
                    // A shard response ships a sketch of a large set (ES merges
                    // the cardinality's HLL++); the coordinator mixes both.
                    let threshold = o.get("precision_threshold").and_then(Value::as_u64).unwrap_or(3000) as usize;
                    if crate::search::aggregation_shard_response() && set.len() > threshold {
                        let mut h = ValueSet::default();
                        h.add(&json!(set));
                        h.into_sketch(threshold);
                        h.payload()
                    } else {
                        json!(set)
                    }
                }
                ("moving_percentiles", "percentiles") => self.percentile_payload(target, o, docs)?,
                _ => continue,
            };
            if let Some(Value::Object(r)) = results.get_mut(target) {
                r.insert(HIDDEN.into(), payload);
            }
        }
        Ok(())
    }

    /// Raw `(value, count)` pairs of a `percentiles` aggregation plus its
    /// sketch configuration.
    fn percentile_payload(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let mut pairs: BTreeMap<u64, (f64, u64)> = BTreeMap::new();
        let hist = self.histogram_field(o);
        if let Some(fname) = &hist {
            for (v, c) in self.histogram_values(fname, docs) {
                pairs.entry(v.to_bits()).or_insert((v, 0)).1 += c;
            }
        } else {
            let f = self.source(o, name, "percentiles")?;
            let missing = self.missing_key(&f, o.get("missing"))?;
            for (s, v) in docs.iter().enumerate() {
                for &(d, _) in v {
                    for x in self.values_or(&f, s, d, &missing).iter().filter_map(Key::as_f64) {
                        pairs.entry(x.to_bits()).or_insert((x, 0)).1 += 1;
                    }
                }
            }
        }
        let hdr = o.get("hdr").map(|h| h.get("number_of_significant_value_digits").and_then(Value::as_u64).unwrap_or(3));
        // HDR auto-ranging depends on recording order: replay doc order.
        let hdr_state = match hdr {
            Some(d) => {
                let mut h = super::sketch::Hdr::new(d.min(5) as u32);
                if let Some(fname) = &hist {
                    for (v, c) in self.histogram_values(fname, docs) {
                        h.add_count(v, c);
                    }
                } else {
                    let f = self.source(o, name, "percentiles")?;
                    let missing = self.missing_key(&f, o.get("missing"))?;
                    for (s, v) in docs.iter().enumerate() {
                        for &(d, _) in v {
                            for x in self.values_or(&f, s, d, &missing).iter().filter_map(Key::as_f64) {
                                h.add(x);
                            }
                        }
                    }
                }
                Some(h.state())
            }
            None => None,
        };
        let comp = o.get("tdigest").and_then(|t| t.get("compression")).and_then(Value::as_f64).unwrap_or(100.0);
        let percents: Vec<f64> = match o.get("percents") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_f64).collect(),
            _ => vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0],
        };
        // A shard response ships a digest's centroids (weights are whole
        // counts) instead of every distinct value once there are many: ES
        // merges the shards' digests too.
        let mut v: Vec<(f64, u64)> = pairs.values().copied().collect();
        if crate::search::aggregation_shard_response() && v.len() as f64 > 20.0 * comp {
            let mut td = super::sketch::TDigest::new(comp);
            for (x, c) in &v {
                for _ in 0..*c {
                    td.add(*x);
                }
            }
            let st = td.state();
            v = st.means.iter().zip(&st.weights).map(|(m, w)| (*m, w.round() as u64)).collect();
        }
        Ok(json!({
            "v": v.iter().map(|(v, c)| json!([v, c])).collect::<Vec<_>>(),
            "hdr": hdr,
            "hs": hdr_state,
            "c": comp,
            "h": hist.is_some(),
            "p": percents,
            "k": o.get("keyed").and_then(Value::as_bool).unwrap_or(true),
        }))
    }

    /// The field of `o` when it is mapped as a pre-aggregated `histogram`.
    fn histogram_field(&self, o: &Map<String, Value>) -> Option<String> {
        let name = o.get("field").and_then(Value::as_str)?;
        if o.contains_key("script") {
            return None;
        }
        let f = self.segs.iter().find_map(|s| s.lookup.get(name))?;
        // `exponential_histogram` (in any index) is aggregated by
        // `fieldtypes::aggs` (`HistogramUnionState` semantics).
        if self.segs.iter().any(|s| s.lookup.get(name).is_some_and(|x| x.declared == crate::mapping::EsType::ExponentialHistogram)) {
            return None;
        }
        crate::fieldtypes::is_histogram(f.declared).then(|| name.to_string())
    }

    /// `(value, count)` centroids of a `histogram` field over `docs`.
    fn histogram_values(&self, field: &str, docs: &Docs) -> Vec<(f64, u64)> {
        let mut out = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, _) in v {
                out.extend(self.doc_histogram(field, s, d));
            }
        }
        out
    }

    fn doc_histogram(&self, field: &str, s: usize, d: DocId) -> Vec<(f64, u64)> {
        let seg = &self.segs[s];
        let Some(rf) = seg.lookup.get(field) else { return Vec::new() };
        let Some(id) = rf.core_id else { return Vec::new() };
        seg.leaf.keyword_values(id, d).iter().flat_map(|enc| rustic_core::fieldtypes::histogram::decode(enc)).collect()
    }

    /// Analytics aggregations that take over a generic aggregation for
    /// some inputs (`histogram` fields in `range` / `rate` / `percentiles`
    /// with `hdr`). `None`: not handled here.
    pub(super) fn analytics_override(&self, name: &str, t: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Option<Value>> {
        if !matches!(t, "range" | "rate" | "percentiles") {
            return Ok(None);
        }
        if t == "rate" {
            if let Some(v) = self.counter_rate(name, o, docs)? {
                return Ok(Some(v));
            }
        }
        let Some(field) = self.histogram_field(o) else { return Ok(None) };
        match t {
            "range" => self.histogram_range(name, &field, o, sub, docs).map(Some),
            "rate" => {
                // ES sums the histogram's values (not weighted by their counts).
                let mut total = 0.0f64;
                let mut comp = 0.0f64;
                // Compensated (Kahan) summation, as ES's `CompensatedSum`.
                for (v, _) in self.histogram_values(&field, docs) {
                    let y = v - comp;
                    let t = total + y;
                    comp = (t - total) - y;
                    total = t;
                }
                self.rate_with_total(name, o, docs, total).map(Some)
            }
            _ => {
                // ES feeds every (value, count) of the histogram into its
                // digest; small totals are expanded exactly here.
                let total: u64 = self.histogram_values(&field, docs).iter().map(|x| x.1).sum();
                if !o.contains_key("hdr") && total > EXPAND_LIMIT {
                    return Ok(None);
                }
                let pl = self.percentile_payload(name, o, docs)?;
                Ok(Some(percentiles_from_payload(&[&pl], name, false)?))
            }
        }
    }

    /// `range` over a `histogram` field: each bucket counts the histogram
    /// counts whose value falls in the range.
    fn histogram_range(&self, name: &str, field: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let ranges = o.get("ranges").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument(format!("No [ranges] specified for the [{name}] aggregation")))?;
        let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(false);
        let bound = |v: Option<&Value>| -> Result<Option<f64>> {
            match v {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Number(n)) => Ok(n.as_f64()),
                Some(Value::String(s)) => s.parse::<f64>().map(Some).map_err(|_| EsError::illegal_argument(format!("For input string: \"{s}\""))),
                Some(_) => Err(EsError::parsing_at("[from]/[to] must be a number or a string", 1, 1)),
            }
        };
        let mut rs = Vec::new();
        for r in ranges {
            rs.push((bound(r.get("from"))?, bound(r.get("to"))?, r.get("key").and_then(Value::as_str).map(str::to_string)));
        }
        rs.sort_by(|a, b| a.0.unwrap_or(f64::NEG_INFINITY).total_cmp(&b.0.unwrap_or(f64::NEG_INFINITY)).then(a.1.unwrap_or(f64::INFINITY).total_cmp(&b.1.unwrap_or(f64::INFINITY))));
        self.add_buckets(rs.len())?;
        let per_doc: Vec<Vec<(DocId, f32, Vec<(f64, u64)>)>> =
            docs.iter().enumerate().map(|(s, v)| v.iter().map(|&(d, sc)| (d, sc, self.doc_histogram(field, s, d))).collect()).collect();
        let mut out: Vec<(String, Map<String, Value>)> = Vec::new();
        for (from, to, key) in rs {
            let inr = |x: f64| from.map(|a| x >= a).unwrap_or(true) && to.map(|b| x < b).unwrap_or(true);
            let mut n = 0u64;
            let mut d: Docs = vec![Vec::new(); docs.len()];
            for (s, v) in per_doc.iter().enumerate() {
                for (doc, sc, h) in v {
                    let c: u64 = h.iter().filter(|(x, _)| inr(*x)).map(|x| x.1).sum();
                    if c > 0 {
                        n += c;
                        d[s].push((*doc, *sc));
                    }
                }
            }
            let key = key.unwrap_or_else(|| format!("{}-{}", from.map(jfmt).unwrap_or("*".into()), to.map(jfmt).unwrap_or("*".into())));
            let mut m = Map::new();
            if !keyed {
                m.insert("key".into(), json!(key));
            }
            if let Some(a) = from {
                m.insert("from".into(), fnum(a));
            }
            if let Some(b) = to {
                m.insert("to".into(), fnum(b));
            }
            m.insert("doc_count".into(), json!(n));
            m.extend(self.subs(sub, &d)?);
            out.push((key, m));
        }
        Ok(if keyed {
            json!({ "buckets": Value::Object(out.into_iter().map(|(k, m)| (k, Value::Object(m))).collect()) })
        } else {
            json!({ "buckets": out.into_iter().map(|(_, m)| Value::Object(m)).collect::<Vec<_>>() })
        })
    }

    pub(super) fn analytics_agg(&self, name: &str, t: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        match t {
            "matrix_stats" => self.matrix_stats(name, o, docs),
            "counted_terms" => self.counted_terms(name, o, sub, docs),
            "random_sampler" => self.random_sampler(name, o, sub, docs),
            "time_series" => self.time_series(name, o, sub, docs),
            "frequent_item_sets" | "frequent_items" => self.frequent_item_sets(name, t, o, docs),
            "categorize_text" => self.categorize_text(name, o, sub, docs),
            _ => Err(EsError::parsing_at(format!("Unknown aggregation type [{t}]"), 1, 1)),
        }
    }

    // -----------------------------------------------------------------------
    // matrix_stats

    fn matrix_stats(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let fields: Vec<String> = match o.get("fields") {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            Some(Value::String(s)) => vec![s.clone()],
            _ => Vec::new(),
        };
        if fields.is_empty() {
            return Err(EsError::illegal_argument(format!("Required [fields] for [matrix_stats] aggregation [{name}]")));
        }
        // ES: every field must be mapped in every searched index.
        let mapping_context: Vec<_> = if self.env.mapping_context.is_empty() {
            self.segs.iter().map(|s|(s.index,s.lookup)).collect()
        } else { self.env.mapping_context.to_vec() };
        let mut indices: Vec<&str> = Vec::new();
        for (index, _) in &mapping_context {
            if !indices.contains(index) { indices.push(index); }
        }
        if indices.len() > 1 {
            let missing: Vec<&String> = fields.iter().filter(|f| mapping_context.iter().any(|(_,lookup)| lookup.get(f).is_none())).collect();
            let any_mapped = fields.iter().any(|f| mapping_context.iter().any(|(_,lookup)| lookup.get(f).is_some()));
            if !missing.is_empty() && any_mapped {
                return Err(EsError::illegal_argument(format!(
                    "Aggregation [{name}] all fields must exist in all indices, but some indices are missing these fields [{}]",
                    missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
        }
        let mut refs: Vec<(FieldRef, Option<Key>)> = Vec::new();
        for f in &fields {
            let fr = self.field(f)?;
            if let Some(ty) = fr.ty {
                if !ty.is_numeric() && !ty.is_date() && ty != EsType::Boolean {
                    return Err(EsError::illegal_argument(format!("Field [{f}] of type [{}] is not supported for aggregation [matrix_stats]", ty.name())));
                }
            }
            let m = self.missing_key(&fr, o.get("missing").and_then(|m| m.get(f)))?;
            refs.push((fr, m));
        }
        let mode = o.get("mode").and_then(Value::as_str).unwrap_or("avg");
        let reduce = |vs: &[f64]| -> f64 {
            match mode {
                "min" => vs.iter().copied().fold(f64::INFINITY, f64::min),
                "max" => vs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "sum" => vs.iter().sum(),
                "median" => {
                    let mut s = vs.to_vec();
                    s.sort_by(|a, b| a.total_cmp(b));
                    let n = s.len();
                    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
                }
                _ => vs.iter().sum::<f64>() / vs.len() as f64,
            }
        };
        // Rows: docs with a value for every field.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            'doc: for &(d, _) in v {
                let mut row = Vec::with_capacity(refs.len());
                for (f, m) in &refs {
                    let vs: Vec<f64> = self.values_or(f, s, d, m).iter().filter_map(Key::as_f64).collect();
                    if vs.is_empty() {
                        continue 'doc;
                    }
                    row.push(reduce(&vs));
                }
                rows.push(row);
            }
        }
        let mut m = Moments::new(fields.len());
        for row in &rows {
            m.add(row);
        }
        let mut v = render_matrix_stats(&fields, &m);
        // A coordinator reduce follows: ship the moments (ES `RunningStats`).
        if crate::search::aggregation_shard_response() {
            // Each shard normally sees only one index. Preserve mapping coverage
            // so the coordinator can enforce the multi-index field contract.
            let mapped: Vec<bool> = fields.iter().map(|f| mapping_context.iter().any(|(_,lookup)| lookup.get(f).is_some())).collect();
            v["__p"] = json!({ "fields": fields, "moments": m.state(), "mapped": mapped, "indices": indices });
        }
        Ok(v)
    }

    // -----------------------------------------------------------------------
    // counted_terms

    /// Terms of a `counted_keyword` field, each value counted as often as
    /// it occurs in the documents (sub-aggregations collect a document once
    /// per occurrence, as ES does).
    fn counted_terms(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument(format!("Required [field] for [counted_terms] aggregation [{name}]")))?;
        if let Some(f) = self.segs.iter().find_map(|s| s.lookup.get(field)) {
            if f.declared != EsType::CountedKeyword {
                return Err(EsError::illegal_argument(format!("Field [{field}] of type [{}] is not supported for aggregation [counted_terms]", f.declared.name())));
            }
        }
        let size = o.get("size").and_then(Value::as_i64).unwrap_or(10);
        if size <= 0 {
            return Err(EsError::illegal_argument(format!("[size] must be greater than 0. Found [{size}] in [{name}]")));
        }
        let spec = json!({ "_source": { "includes": [field] } });
        let mut groups: HashMap<String, (u64, Docs)> = HashMap::new();
        for (s, v) in docs.iter().enumerate() {
            if self.segs[s].lookup.get(field).is_none() {
                continue;
            }
            for &(d, sc) in v {
                let hit = (self.env.fetch_hit)(s, d, sc, &spec)?;
                let mut vals = Vec::new();
                collect_strings(source_path(hit.get("_source"), field), &mut vals);
                for t in vals {
                    let e = groups.entry(t).or_insert_with(|| (0, vec![Vec::new(); docs.len()]));
                    e.0 += 1;
                    e.1[s].push((d, sc));
                }
            }
        }
        let mut bs: Vec<(String, u64, Docs)> = groups.into_iter().map(|(k, (c, d))| (k, c, d)).collect();
        let total: u64 = bs.iter().map(|b| b.1).sum();
        bs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_bytes().cmp(b.0.as_bytes())));
        bs.truncate(size as usize);
        self.add_buckets(bs.len())?;
        let kept: u64 = bs.iter().map(|b| b.1).sum();
        let mut out = Vec::new();
        for (k, c, d) in bs {
            let mut m = Map::new();
            m.insert("key".into(), json!(k));
            m.insert("doc_count".into(), json!(c));
            m.extend(self.subs(sub, &d)?);
            out.push(m);
        }
        super::apply_parent_pipelines(sub, &mut out)?;
        Ok(json!({ "doc_count_error_upper_bound": 0, "sum_other_doc_count": total - kept, "buckets": out }))
    }

    // -----------------------------------------------------------------------
    // random_sampler

    fn random_sampler(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let p = o.get("probability").and_then(Value::as_f64).ok_or_else(|| EsError::illegal_argument(format!("[random_sampler] aggregation [{name}] must have [probability] set")))?;
        if !((p > 0.0 && p <= 0.5) || p == 1.0) {
            return Err(EsError::new(400, "x_content_parse_exception", "[random_sampler] failed to parse field [probability]")
                .caused_by(EsError::illegal_argument(format!("[probability] must be between 0.0 and 0.5 or exactly 1.0, was [{}]", jfmt(p)))));
        }
        if sub.map(|s| s.is_empty()).unwrap_or(true) {
            let m = format!("[random_sampler] aggregation [{name}] must have sub-aggregations");
            return Err(EsError::illegal_argument(m.clone()).caused_by(EsError::illegal_argument(m)));
        }
        let seed = match o.get("seed").and_then(Value::as_i64) {
            Some(s) => s as i32,
            None => {
                let mut b = [0u8; 4];
                let _ = getrandom::getrandom(&mut b);
                i32::from_le_bytes(b)
            }
        };
        let sampled: Docs = if p >= 1.0 {
            docs.clone()
        } else {
            docs.iter()
                .enumerate()
                .map(|(s, v)| {
                    v.iter()
                        .filter(|(d, _)| {
                            let h = super::sketch::murmur3_64(&[seed.to_le_bytes(), (s as u32).to_le_bytes(), d.to_le_bytes()].concat());
                            ((h >> 11) as f64 / (1u64 << 53) as f64) < p
                        })
                        .copied()
                        .collect()
                })
                .collect()
        };
        let mut m = Map::new();
        m.insert("seed".into(), json!(seed));
        m.insert("probability".into(), fnum(p));
        m.insert("doc_count".into(), json!(count(&sampled)));
        let mut subs = self.subs(sub, &sampled)?;
        if p < 1.0 {
            if let Some(spec) = sub {
                scale_sampled(&mut subs, spec, 1.0 / p);
            }
        }
        m.extend(subs);
        Ok(Value::Object(m))
    }
}

/// ES `InternalAggregation.finalizeSampling`: counts (and sums) of the
/// sub-aggregations of a `random_sampler` are scaled by the inverse
/// probability.
fn scale_sampled(out: &mut Map<String, Value>, spec: &Map<String, Value>, f: f64) {
    let scale_u = |v: &mut Value| {
        if let Some(n) = v.as_f64() {
            *v = json!((n * f).round() as i64);
        }
    };
    for (name, def) in spec {
        let Some(defo) = def.as_object() else { continue };
        let Some(t) = defo.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")) else { continue };
        let subspec = defo.get("aggs").or(defo.get("aggregations")).and_then(Value::as_object);
        let Some(Value::Object(r)) = out.get_mut(name) else { continue };
        match t.as_str() {
            "value_count" | "cardinality" => {
                if let Some(v) = r.get_mut("value") {
                    scale_u(v);
                }
            }
            "sum" => {
                if let Some(x) = r.get("value").and_then(Value::as_f64) {
                    r.insert("value".into(), fnum(x * f));
                }
            }
            "stats" | "extended_stats" => {
                if let Some(v) = r.get_mut("count") {
                    scale_u(v);
                }
                if let Some(x) = r.get("sum").and_then(Value::as_f64) {
                    r.insert("sum".into(), fnum(x * f));
                }
            }
            _ => {}
        }
        for k in ["doc_count", "sum_other_doc_count"] {
            if let Some(v) = r.get_mut(k) {
                scale_u(v);
            }
        }
        let mut scale_bucket = |b: &mut Value| {
            if let Value::Object(bm) = b {
                if let Some(v) = bm.get_mut("doc_count") {
                    scale_u(v);
                }
                if let Some(ss) = subspec {
                    scale_sampled(bm, ss, f);
                }
            }
        };
        match r.get_mut("buckets") {
            Some(Value::Array(a)) => a.iter_mut().for_each(&mut scale_bucket),
            Some(Value::Object(o)) => o.values_mut().for_each(&mut scale_bucket),
            _ => {
                if let Some(ss) = subspec {
                    scale_sampled(r, ss, f);
                }
            }
        }
    }
}

/// The value at a dotted path of a `_source` object (objects and arrays
/// of objects are walked).
fn source_path<'v>(src: Option<&'v Value>, path: &str) -> Vec<&'v Value> {
    let Some(src) = src else { return Vec::new() };
    if let Some(v) = src.get(path) {
        return vec![v];
    }
    let mut out = Vec::new();
    if let Value::Object(m) = src {
        for (k, v) in m {
            if let Some(rest) = path.strip_prefix(k.as_str()).and_then(|r| r.strip_prefix('.')) {
                match v {
                    Value::Array(a) => a.iter().for_each(|x| out.extend(source_path(Some(x), rest))),
                    _ => out.extend(source_path(Some(v), rest)),
                }
            }
        }
    }
    out
}

fn collect_strings(vs: Vec<&Value>, out: &mut Vec<String>) {
    for v in vs {
        match v {
            Value::String(s) => out.push(s.clone()),
            Value::Array(a) => collect_strings(a.iter().collect(), out),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Parent pipelines.

/// Percentiles (as the `percentiles` aggregation renders them) of the merged
/// payloads of [`Ctx::percentile_payload`].
///
/// `window`: the payloads are a moving window, which ES (like HDR's
/// reduce) folds into a fresh histogram; otherwise a single payload's
/// histogram is used as recorded.
fn percentiles_from_payload(payloads: &[&Value], name: &str, window: bool) -> Result<Value> {
    let first = payloads.first().ok_or_else(|| EsError::illegal_argument(format!("no percentiles for [{name}]")))?;
    let hdr = first.get("hdr").and_then(Value::as_u64);
    let comp = first.get("c").and_then(Value::as_f64).unwrap_or(100.0);
    let hist = first.get("h").and_then(Value::as_bool).unwrap_or(false);
    let keyed = first.get("k").and_then(Value::as_bool).unwrap_or(true);
    let percents: Vec<f64> = first.get("p").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
    let mut pairs: Vec<(f64, u64)> = Vec::new();
    for p in payloads {
        for e in p.get("v").and_then(Value::as_array).into_iter().flatten() {
            if let (Some(v), Some(c)) = (e.get(0).and_then(Value::as_f64), e.get(1).and_then(Value::as_u64)) {
                pairs.push((v, c));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: u64 = pairs.iter().map(|x| x.1).sum();
    let value_at: Box<dyn Fn(f64) -> Option<f64>> = if total == 0 {
        Box::new(|_| None)
    } else if let Some(d) = hdr {
        let states: Vec<super::sketch::Hdr> = payloads
            .iter()
            .filter_map(|p| p.get("hs").cloned().and_then(|s| serde_json::from_value::<super::sketch::HdrState>(s).ok()))
            .map(|s| super::sketch::Hdr::from_state(&s))
            .collect();
        let h = if window || states.len() != 1 {
            let mut m = super::sketch::Hdr::new(d.min(5) as u32);
            states.iter().for_each(|s| m.merge(s));
            m
        } else {
            states.into_iter().next().unwrap()
        };
        Box::new(move |p| h.value_at_percentile(p))
    } else if hist && total > EXPAND_LIMIT {
        let mut merged: Vec<(f64, u64)> = Vec::new();
        for (v, c) in pairs {
            match merged.last_mut() {
                Some(l) if l.0 == v => l.1 += c,
                _ => merged.push((v, c)),
            }
        }
        Box::new(move |p| crate::fieldtypes::aggs::weighted_quantile(&merged, p / 100.0))
    } else {
        let mut td = super::sketch::TDigest::new(comp);
        for (v, c) in &pairs {
            for _ in 0..*c {
                td.add(*v);
            }
        }
        let td = RefCell::new(td);
        Box::new(move |p| td.borrow_mut().quantile(p / 100.0))
    };
    if keyed {
        let mut m = Map::new();
        for p in &percents {
            m.insert(jfmt(*p), opt_num(value_at(*p)));
        }
        Ok(json!({ "values": m }))
    } else {
        Ok(json!({ "values": percents.iter().map(|p| json!({"key": fnum(*p), "value": opt_num(value_at(*p))})).collect::<Vec<_>>() }))
    }
}

/// Apply one of this module's parent pipelines to its parent's buckets.
pub(super) fn parent_pipeline(name: &str, t: &str, body: &Value, buckets: &mut [Map<String, Value>]) -> Result<()> {
    let path = buckets_path(body)?;
    let format = body.get("format").and_then(Value::as_str);
    match t {
        "normalize" => {
            let method = body.get("method").and_then(Value::as_str).unwrap_or("");
            let vals: Vec<f64> = buckets.iter().map(|b| resolve_path(b, &path).unwrap_or(f64::NAN)).collect();
            let fin: Vec<f64> = vals.iter().copied().filter(|v| !v.is_nan()).collect();
            let n = fin.len() as f64;
            let min = fin.iter().copied().fold(f64::INFINITY, f64::min);
            let max = fin.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = fin.iter().sum();
            let mean = sum / n;
            let sq: f64 = fin.iter().map(|v| (v - mean) * (v - mean)).sum();
            let stdev = (sq / n).sqrt();
            let exp_sum: f64 = fin.iter().map(|v| v.exp()).sum();
            for (b, v) in buckets.iter_mut().zip(vals) {
                let x = if v.is_nan() {
                    f64::NAN
                } else {
                    match method {
                        "rescale_0_1" => (v - min) / (max - min),
                        "rescale_0_100" => 100.0 * (v - min) / (max - min),
                        "percent_of_sum" => v / sum,
                        "mean" => (v - mean) / (max - min),
                        "z-score" => (v - mean) / stdev,
                        "softmax" => v.exp() / exp_sum,
                        _ => f64::NAN,
                    }
                };
                b.insert(name.to_string(), simple_value(x, format));
            }
        }
        "cumulative_cardinality" => {
            let target = path_head(&path).to_string();
            let mut seen = ValueSet::default();
            for b in buckets.iter_mut() {
                let Some(agg) = b.get(&target).and_then(Value::as_object) else {
                    return Err(EsError::illegal_argument(format!("buckets_path must reference a cardinality aggregation, got: [{target}]")));
                };
                let Some(payload) = agg.get(HIDDEN).filter(|p| p.is_array() || p.get("hll").is_some()) else {
                    if super::partial::active() || agg.contains_key("value") {
                        return Err(EsError::illegal_argument(format!(
                            "aggregation [{name}] of type [{t}] does not support a cross-shard reduce yet; it can only run on a single shard"
                        )));
                    }
                    return Err(EsError::illegal_argument(format!("buckets_path must reference a cardinality aggregation, got: [{target}]")));
                };
                seen.add(payload);
                let n = seen.count();
                let mut m = Map::new();
                m.insert("value".into(), json!(n));
                if let Some(f) = format {
                    m.insert("value_as_string".into(), json!(decimal_format(f, n as f64)));
                }
                b.insert(name.to_string(), Value::Object(m));
            }
        }
        "moving_percentiles" => {
            let target = path_head(&path).to_string();
            let window = body.get("window").and_then(Value::as_i64).unwrap_or(1);
            let shift = body.get("shift").and_then(Value::as_i64).unwrap_or(0);
            let payloads: Vec<Option<Value>> = buckets.iter().map(|b| b.get(&target).and_then(|a| a.get(HIDDEN)).cloned()).collect();
            if !buckets.is_empty() && payloads.iter().all(Option::is_none) {
                return Err(EsError::illegal_argument(format!(
                    "aggregation [{name}] of type [{t}] does not support a cross-shard reduce yet; it can only run on a single shard"
                )));
            }
            let len = buckets.len() as i64;
            let clamp = |i: i64| i.clamp(0, len) as usize;
            for i in 0..buckets.len() {
                let from = clamp(i as i64 - window + shift);
                let to = clamp(i as i64 + shift);
                if from >= to {
                    continue;
                }
                // A coordinator's merged payload holds every shard's (`parts`).
                let ps: Vec<&Value> = payloads[from..to].iter().flatten().flat_map(|p| match p.get("parts").and_then(Value::as_array) {
                    Some(a) => a.iter().collect::<Vec<_>>(),
                    None => vec![p],
                }).collect();
                if ps.is_empty() {
                    continue;
                }
                let v = percentiles_from_payload(&ps, name, true)?;
                buckets[i].insert(name.to_string(), v);
            }
        }
        "serial_diff" => {
            // ES `SerialDiffPipelineAggregator`: the value `lag` buckets back,
            // gaps included in the lag window.
            let lag = match body.get("lag") {
                None => 1,
                Some(v) => match v.as_i64() {
                    Some(l) if l > 0 => l as usize,
                    _ => return Err(EsError::illegal_argument(format!("[lag] must be a positive integer: [{name}]"))),
                },
            };
            let policy = gap_policy(body)?;
            let vals: Vec<f64> = buckets.iter().map(|b| gap_value(b, &path, policy)).collect();
            for (i, b) in buckets.iter_mut().enumerate() {
                let (cur, back) = (vals[i], if i >= lag { vals[i - lag] } else { f64::NAN });
                if !cur.is_nan() && !back.is_nan() {
                    b.insert(name.to_string(), simple_value(cur - back, format));
                }
            }
        }
        "moving_fn" => {
            // ES `MovFnPipelineAggregator`: the window is over the buckets
            // that have a value, `[i - window + shift, i + shift)` clamped.
            let window = match body.get("window").and_then(Value::as_i64) {
                Some(w) if w > 0 => w,
                _ => return Err(EsError::illegal_argument("[window] must be a positive, non-zero integer.")),
            };
            let shift = body.get("shift").and_then(Value::as_i64).unwrap_or(0);
            let (src, params) = match body.get("script") {
                Some(Value::String(s)) => (s.clone(), None),
                Some(Value::Object(o)) => match o.get("source").and_then(Value::as_str) {
                    Some(s) => (s.to_string(), o.get("params").and_then(Value::as_object)),
                    None => return Err(EsError::illegal_argument(format!("[moving_fn] [{name}] supports inline scripts only"))),
                },
                _ => return Err(EsError::illegal_argument("[script] must not be null: [moving_fn]")),
            };
            let f = MovFn::parse(&src).ok_or_else(|| {
                EsError::illegal_argument(format!("[moving_fn] script [{src}] is not supported; use MovingFunctions functions over `values`"))
            })?;
            let policy = gap_policy(body)?;
            let vals: Vec<f64> = buckets.iter().map(|b| gap_value(b, &path, policy)).collect();
            let present: Vec<f64> = vals.iter().copied().filter(|v| !v.is_nan()).collect();
            let len = present.len() as i64;
            let clamp = |i: i64| i.clamp(0, len) as usize;
            let mut index = 0i64;
            for (b, v) in buckets.iter_mut().zip(&vals) {
                if v.is_nan() {
                    continue;
                }
                let (from, to) = (clamp(index - window + shift), clamp(index + shift));
                let x = f.eval(&present[from..to.max(from)], params).map_err(EsError::illegal_argument)?;
                b.insert(name.to_string(), simple_value(x, format));
                index += 1;
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GapPolicy {
    Skip,
    InsertZeros,
    KeepValues,
}

fn gap_policy(body: &Value) -> Result<GapPolicy> {
    match body.get("gap_policy").and_then(Value::as_str) {
        None | Some("skip") => Ok(GapPolicy::Skip),
        Some("insert_zeros") => Ok(GapPolicy::InsertZeros),
        Some("keep_values") => Ok(GapPolicy::KeepValues),
        Some(p) => Err(EsError::illegal_argument(format!("Invalid gap policy: [{p}], accepted values: [skip, insert_zeros, keep_values]"))),
    }
}

/// A bucket's value under a gap policy (ES `BucketHelpers.resolveBucketValue`),
/// NaN for none: missing, non-finite and empty-bucket values are gaps.
fn gap_value(b: &Map<String, Value>, path: &str, policy: GapPolicy) -> f64 {
    let v = resolve_path(b, path);
    // BucketHelpers: _count is always present, including zero-count buckets.
    let empty = path != "_count" && b.get("doc_count").and_then(Value::as_u64) == Some(0);
    match v {
        Some(x) if x.is_finite() && !empty => x,
        _ => match policy {
            GapPolicy::InsertZeros => 0.0,
            GapPolicy::KeepValues => v.filter(|x| x.is_finite()).unwrap_or(f64::NAN),
            GapPolicy::Skip => f64::NAN,
        },
    }
}

/// A `moving_fn` script: a `MovingFunctions` call over `values`, whose
/// arguments are numbers, `params.*` or nested calls (ES `MovingFunctions`).
enum MovFn {
    Values,
    Num(f64),
    Param(String),
    Call(String, Vec<MovFn>),
}

impl MovFn {
    fn parse(src: &str) -> Option<MovFn> {
        let s = src.trim();
        let s = s.strip_prefix("return ").unwrap_or(s).trim().trim_end_matches(';').trim();
        let (e, rest) = Self::expr(s)?;
        rest.trim().is_empty().then_some(e)
    }

    fn expr(s: &str) -> Option<(MovFn, &str)> {
        let s = s.trim_start();
        if let Some(r) = s.strip_prefix("MovingFunctions.") {
            let end = r.find('(')?;
            let (f, mut r) = (r[..end].trim().to_string(), &r[end + 1..]);
            let mut args = Vec::new();
            loop {
                r = r.trim_start();
                if let Some(rest) = r.strip_prefix(')') {
                    return Some((MovFn::Call(f, args), rest));
                }
                if !args.is_empty() {
                    r = r.strip_prefix(',')?;
                }
                let (a, rest) = Self::expr(r)?;
                args.push(a);
                r = rest;
            }
        }
        let end = s.find(|c: char| c == ',' || c == ')' || c.is_whitespace()).unwrap_or(s.len());
        let (tok, rest) = (&s[..end], &s[end..]);
        if tok == "values" {
            Some((MovFn::Values, rest))
        } else if let Some(p) = tok.strip_prefix("params.") {
            Some((MovFn::Param(p.to_string()), rest))
        } else {
            Some((MovFn::Num(tok.trim_end_matches(['d', 'D', 'f', 'F']).parse().ok()?), rest))
        }
    }

    fn num(&self, values: &[f64], params: Option<&Map<String, Value>>) -> std::result::Result<f64, String> {
        match self {
            MovFn::Num(x) => Ok(*x),
            MovFn::Param(p) => params.and_then(|m| m.get(p)).and_then(Value::as_f64).ok_or_else(|| format!("[moving_fn] missing numeric param [{p}]")),
            MovFn::Values => Err("[moving_fn] `values` is not a number".into()),
            MovFn::Call(..) => self.eval(values, params),
        }
    }

    fn eval(&self, values: &[f64], params: Option<&Map<String, Value>>) -> std::result::Result<f64, String> {
        let MovFn::Call(f, args) = self else { return self.num(values, params) };
        if !matches!(args.first(), Some(MovFn::Values)) {
            return Err(format!("[moving_fn] MovingFunctions.{f} takes `values` first"));
        }
        let arg = |i: usize| args.get(i).ok_or_else(|| format!("[moving_fn] MovingFunctions.{f} needs {} arguments", i + 1)).and_then(|a| a.num(values, params));
        let live = || values.iter().copied().filter(|v| !v.is_nan());
        Ok(match (f.as_str(), args.len()) {
            ("max", 1) => live().reduce(f64::max).unwrap_or(f64::NAN),
            ("min", 1) => live().reduce(f64::min).unwrap_or(f64::NAN),
            ("sum", 1) => live().sum(),
            ("unweightedAvg", 1) => {
                let n = live().count();
                if n == 0 { f64::NAN } else { live().sum::<f64>() / n as f64 }
            }
            ("stdDev", 2) => {
                let avg = arg(1)?;
                if avg.is_nan() {
                    f64::NAN
                } else {
                    let n = live().count();
                    (live().map(|v| (v - avg).powi(2)).sum::<f64>() / n as f64).sqrt()
                }
            }
            ("linearWeightedAvg", 1) => {
                let (mut avg, mut total, mut w) = (0.0, 1.0, 1.0);
                for v in live() {
                    avg += v * w;
                    total += w;
                    w += 1.0;
                }
                if total == 1.0 { f64::NAN } else { avg / (total - 1.0) }
            }
            ("ewma", 2) => {
                let alpha = arg(1)?;
                live().reduce(|avg, v| v * alpha + avg * (1.0 - alpha)).unwrap_or(f64::NAN)
            }
            ("holt", 3) => {
                let (alpha, beta) = (arg(1)?, arg(2)?);
                let (mut s, mut b, mut first) = (f64::NAN, 0.0, true);
                for v in live() {
                    if first {
                        s = v;
                        first = false;
                    } else {
                        let last_s = s;
                        s = alpha * v + (1.0 - alpha) * (last_s + b);
                        b = beta * (s - last_s) + (1.0 - beta) * b;
                    }
                }
                s
            }
            _ => return Err(format!("[moving_fn] MovingFunctions.{f} with {} arguments is not supported", args.len())),
        })
    }
}

/// Drop the sketch payloads of [`Ctx::analytics_prepare`] from rendered buckets.
pub(super) fn strip_hidden(buckets: &mut [Map<String, Value>]) {
    for b in buckets {
        for v in b.values_mut() {
            if let Value::Object(m) = v {
                m.shift_remove(HIDDEN);
            }
        }
    }
}

/// A sibling pipeline of this module over the computed aggregations of its level.
pub(super) fn sibling_pipeline(name: &str, t: &str, body: &Value, results: &HashMap<String, Value>) -> Result<Value> {
    ml_pipelines::sibling(name, t, body, results)
}

#[allow(dead_code)]
fn key_value(k: &Key) -> Value {
    key_json(k)
}

/// `matrix_stats` online moments (ES `RunningStats`: Welford / Terriberry),
/// mergeable across shards.
pub(super) struct Moments {
    cnt: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
    m3: Vec<f64>,
    m4: Vec<f64>,
    /// Co-moments, upper triangle (`cov[i][j]`, i < j).
    cov: Vec<Vec<f64>>,
}

impl Moments {
    pub(super) fn new(k: usize) -> Moments {
        Moments { cnt: 0.0, mean: vec![0.0; k], m2: vec![0.0; k], m3: vec![0.0; k], m4: vec![0.0; k], cov: vec![vec![0.0; k]; k] }
    }

    fn add(&mut self, row: &[f64]) {
        let k = self.mean.len();
        self.cnt += 1.0;
        let cnt = self.cnt;
        let mut delta = vec![0.0f64; k];
        for i in 0..k {
            let x = row[i];
            let d = x - self.mean[i];
            let dn = d / cnt;
            let dn2 = dn * dn;
            let term1 = d * dn * (cnt - 1.0);
            self.m4[i] += term1 * dn2 * (cnt * cnt - 3.0 * cnt + 3.0) + 6.0 * dn2 * self.m2[i] - 4.0 * dn * self.m3[i];
            self.m3[i] += term1 * dn * (cnt - 2.0) - 3.0 * dn * self.m2[i];
            self.m2[i] += term1;
            delta[i] = d;
            self.mean[i] += dn;
        }
        for i in 0..k {
            for j in (i + 1)..k {
                self.cov[i][j] += (cnt - 1.0) / cnt * delta[i] * delta[j];
            }
        }
    }

    /// ES `RunningStats.merge`: the pairwise update of every moment.
    pub(super) fn merge(&mut self, o: &Moments) {
        if o.cnt == 0.0 {
            return;
        }
        if self.cnt == 0.0 {
            *self = Moments { cnt: o.cnt, mean: o.mean.clone(), m2: o.m2.clone(), m3: o.m3.clone(), m4: o.m4.clone(), cov: o.cov.clone() };
            return;
        }
        let (na, nb) = (self.cnt, o.cnt);
        let n = na + nb;
        let k = self.mean.len();
        let d: Vec<f64> = (0..k).map(|i| o.mean[i] - self.mean[i]).collect();
        for i in 0..k {
            let (d1, d2) = (d[i], d[i] * d[i]);
            let (d3, d4) = (d2 * d1, d2 * d2);
            let (m2a, m3a, m2b, m3b) = (self.m2[i], self.m3[i], o.m2[i], o.m3[i]);
            self.m4[i] += o.m4[i] + d4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n) + 6.0 * d2 * (na * na * m2b + nb * nb * m2a) / (n * n) + 4.0 * d1 * (na * m3b - nb * m3a) / n;
            self.m3[i] = m3a + m3b + d3 * na * nb * (na - nb) / (n * n) + 3.0 * d1 * (na * m2b - nb * m2a) / n;
            self.m2[i] = m2a + m2b + d2 * na * nb / n;
            self.mean[i] = (na * self.mean[i] + nb * o.mean[i]) / n;
        }
        for i in 0..k {
            for j in (i + 1)..k {
                self.cov[i][j] += o.cov[i][j] + d[i] * d[j] * na * nb / n;
            }
        }
        self.cnt = n;
    }

    pub(super) fn state(&self) -> Value {
        json!({ "n": self.cnt, "mean": self.mean, "m2": self.m2, "m3": self.m3, "m4": self.m4, "cov": self.cov })
    }

    pub(super) fn from_state(v: &Value) -> Option<Moments> {
        let vec = |k: &str| -> Option<Vec<f64>> { v.get(k)?.as_array()?.iter().map(Value::as_f64).collect() };
        let cov: Vec<Vec<f64>> = v.get("cov")?.as_array()?.iter().map(|r| r.as_array()?.iter().map(Value::as_f64).collect()).collect::<Option<_>>()?;
        Some(Moments { cnt: v.get("n")?.as_f64()?, mean: vec("mean")?, m2: vec("m2")?, m3: vec("m3")?, m4: vec("m4")?, cov })
    }
}

/// The `matrix_stats` response of `fields` (request order) from their moments.
pub(super) fn render_matrix_stats(fields: &[String], m: &Moments) -> Value {
    let n = m.cnt as u64;
    let mut out = Map::new();
    out.insert("doc_count".into(), json!(n));
    if n == 0 {
        out.insert("fields".into(), json!([]));
        return Value::Object(out);
    }
    let cnt = m.cnt;
    let var = |i: usize| m.m2[i] / (cnt - 1.0);
    let covariance = |i: usize, j: usize| -> f64 {
        if i == j {
            var(i)
        } else {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            m.cov[a][b] / (cnt - 1.0)
        }
    };
    let correlation = |i: usize, j: usize| -> f64 {
        if i == j {
            return 1.0;
        }
        let c = covariance(i, j);
        let d = (var(i) * var(j)).sqrt();
        if d == 0.0 { f64::NAN } else { c / d }
    };
    let order = java_hashmap_order(fields);
    let pos = |f: &str| fields.iter().position(|x| x == f).unwrap();
    let num = |v: f64| if v.is_nan() { json!("NaN") } else { fnum(v) };
    let mut list = Vec::new();
    for f in &order {
        let i = pos(f);
        let mut e = Map::new();
        e.insert("name".into(), json!(f));
        e.insert("count".into(), json!(n));
        e.insert("mean".into(), num(m.mean[i]));
        e.insert("variance".into(), num(var(i)));
        let skew = cnt.sqrt() * m.m3[i] / m.m2[i].powf(1.5);
        let kurt = cnt * m.m4[i] / (m.m2[i] * m.m2[i]);
        e.insert("skewness".into(), num(skew));
        e.insert("kurtosis".into(), num(kurt));
        let mut cv = Map::new();
        let mut cr = Map::new();
        for g in &order {
            let j = pos(g);
            cv.insert(g.clone(), num(covariance(i, j)));
            cr.insert(g.clone(), num(correlation(i, j)));
        }
        e.insert("covariance".into(), Value::Object(cv));
        e.insert("correlation".into(), Value::Object(cr));
        list.push(Value::Object(e));
    }
    out.insert("fields".into(), Value::Array(list));
    Value::Object(out)
}

/// Distinct values of `cumulative_cardinality` payloads: exact strings, or
/// an HLL++ of their hashes once a sketch payload (a large shard set) joins.
#[derive(Default)]
pub(crate) struct ValueSet {
    exact: HashSet<String>,
    hll: Option<super::sketch::HyperLogLog>,
}

impl ValueSet {
    fn hash(s: &str) -> u64 {
        super::sketch::murmur3_64(s.as_bytes())
    }

    fn into_sketch(&mut self, threshold: usize) {
        let mut h = self.hll.take().unwrap_or_else(|| super::sketch::HyperLogLog::new(threshold));
        self.exact.drain().for_each(|s| h.add_hash(Self::hash(&s)));
        self.hll = Some(h);
    }

    /// Add a payload: an array of values, `{"hll": state}` or `{"parts": [..]}`.
    pub(crate) fn add(&mut self, p: &Value) {
        if let Some(parts) = p.get("parts").and_then(Value::as_array) {
            parts.iter().for_each(|x| self.add(x));
        } else if let Some(st) = p.get("hll").and_then(|h| serde_json::from_value::<super::sketch::HllState>(h.clone()).ok()) {
            let other = super::sketch::HyperLogLog::from_state(&st);
            self.into_sketch(st.threshold);
            self.hll.as_mut().unwrap().merge(&other);
        } else {
            let vals = p.as_array().into_iter().flatten().filter_map(Value::as_str);
            match &mut self.hll {
                Some(h) => vals.for_each(|s| h.add_hash(Self::hash(s))),
                None => self.exact.extend(vals.map(str::to_string)),
            }
        }
    }

    pub(crate) fn count(&self) -> u64 {
        self.hll.as_ref().map(|h| h.count()).unwrap_or(self.exact.len() as u64)
    }

    pub(crate) fn payload(&self) -> Value {
        match &self.hll {
            Some(h) => json!({ "hll": h.state() }),
            None => json!(self.exact.iter().collect::<Vec<_>>()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series() -> Vec<Map<String, Value>> {
        // doc_count, sum: the second bucket is empty (a gap).
        [(2, 10.0), (0, 0.0), (1, 4.0), (3, 7.0), (1, 1.0)]
            .iter()
            .map(|&(dc, v)| json!({"doc_count": dc, "s": {"value": v}}).as_object().unwrap().clone())
            .collect()
    }

    fn run(t: &str, body: Value) -> Vec<Option<Option<f64>>> {
        let mut b = series();
        parent_pipeline("p", t, &body, &mut b).unwrap();
        b.iter().map(|b| b.get("p").map(|v| v["value"].as_f64())).collect()
    }

    #[test]
    fn serial_diff_matches_es() {
        let (n, v) = (None, |x: f64| Some(Some(x)));
        assert_eq!(run("serial_diff", json!({"buckets_path": "s"})), [n, n, n, v(3.0), v(-6.0)]);
        assert_eq!(run("serial_diff", json!({"buckets_path": "s", "gap_policy": "insert_zeros"})), [n, v(-10.0), v(4.0), v(3.0), v(-6.0)]);
        assert_eq!(run("serial_diff", json!({"buckets_path": "s", "lag": 2})), [n, n, v(-6.0), n, v(-3.0)]);
        let mut b = series();
        assert!(parent_pipeline("p", "serial_diff", &json!({"buckets_path": "s", "lag": 0}), &mut b).is_err());
    }

    #[test]
    fn count_pipeline_keeps_empty_histogram_buckets() {
        assert_eq!(run("serial_diff", json!({"buckets_path": "_count"})),
            [None, Some(Some(-2.0)), Some(Some(1.0)), Some(Some(2.0)), Some(Some(-2.0))]);
        assert_eq!(run("moving_fn", json!({"buckets_path": "_count", "window": 2,
            "script": "MovingFunctions.unweightedAvg(values)"})),
            [Some(None), Some(Some(2.0)), Some(Some(1.0)), Some(Some(0.5)), Some(Some(2.0))]);
    }

    #[test]
    fn moving_fn_matches_es() {
        let (n, v) = (None, |x: f64| Some(Some(x)));
        let avg = |extra: Value| {
            let mut body = json!({"buckets_path": "s", "window": 2, "script": "MovingFunctions.unweightedAvg(values)"});
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            run("moving_fn", body)
        };
        // The first bucket's window is empty: a null value, as ES.
        assert_eq!(avg(json!({})), [Some(None), n, v(10.0), v(7.0), v(5.5)]);
        assert_eq!(avg(json!({"shift": 1})), [v(10.0), n, v(7.0), v(5.5), v(4.0)]);
        // Window 3 at the last bucket covers [10, 4, 7].
        let last = |script: Value| run("moving_fn", json!({"buckets_path": "s", "window": 3, "script": script}))[4].unwrap().unwrap();
        assert_eq!(last(json!("MovingFunctions.max(values)")), 10.0);
        assert_eq!(last(json!("MovingFunctions.min(values)")), 4.0);
        assert_eq!(last(json!("return MovingFunctions.sum(values);")), 21.0);
        assert_eq!(last(json!("MovingFunctions.linearWeightedAvg(values)")), 6.5);
        assert_eq!(last(json!({"source": "MovingFunctions.ewma(values, params.a)", "params": {"a": 0.5}})), 7.0);
        assert!((last(json!("MovingFunctions.stdDev(values, MovingFunctions.unweightedAvg(values))")) - 6f64.sqrt()).abs() < 1e-12);
        assert_eq!(last(json!("MovingFunctions.holt(values, 0.5, 0.5)")), 6.25);
        let mut b = series();
        assert!(parent_pipeline("p", "moving_fn", &json!({"buckets_path": "s", "window": 2, "script": "values[0] * 2"}), &mut b).is_err());
        assert!(parent_pipeline("p", "moving_fn", &json!({"buckets_path": "s", "script": "MovingFunctions.max(values)"}), &mut b).is_err());
    }

    #[test]
    fn decimal_formats() {
        assert_eq!(decimal_format("00.00%", 0.5), "50.00%");
        assert_eq!(decimal_format("00.00%", 1.0 / 3.0), "33.33%");
        assert_eq!(decimal_format("00.00%", 1.0 / 6.0), "16.67%");
        assert_eq!(decimal_format("00", 2.0), "02");
        assert_eq!(decimal_format("#,##0.0#", 1234567.891), "1,234,567.89");
        assert_eq!(decimal_format("0.0", -0.04), "-0.0");
    }

    #[test]
    fn hashmap_order() {
        let k = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(java_hashmap_order(&k(&["value1", "value2"])), k(&["value2", "value1"]));
        assert_eq!(java_hashmap_order(&k(&["value1", "value2", "value3"])), k(&["value2", "value1", "value3"]));
    }
}
