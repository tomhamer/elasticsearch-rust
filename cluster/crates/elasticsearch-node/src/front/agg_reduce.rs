//! Coordinator-side reduce of aggregations for cross-node searches.
//!
//! The engine only returns final-rendered aggregations (no
//! `InternalAggregation`-style partial state, see `docs/ENGINE_HOOKS.md` #6),
//! so the reduce works on rendered JSON. Where the rendered form is not
//! mergeable, [`rewrite`] adds a mergeable helper aggregation next to the
//! original in the per-node request (`avg` gets a `stats` sibling) and
//! [`reduce`] consumes and drops it. Sketch aggregations (`cardinality`,
//! percentiles, `geo_bounds`) instead carry their mergeable shard state
//! (`__p`, `SHARD_STATE_AGGS`) and are reduced by the engine's reducers.
//! `terms` is asked for `shard_size` buckets per node and reduced with ES's
//! `doc_count_error_upper_bound` / `sum_other_doc_count` rules, treating
//! each node response as one shard result.
//!
//! Anything not listed in [`unsupported`]'s allow-list is rejected up front.
//! See `cluster/docs/CROSS_NODE_SEARCH.md`.

use super::fanout::{cmp_values, hit_order, sort_keys};
use elasticsearch_engine::aggs::expr as eexpr;
use elasticsearch_engine::date as edate;
use elasticsearch_engine::script::{ScriptContext, Val};
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

/// Helper-aggregation name prefixes (never returned to the client).
const AVG_HELPER: &str = "__fanout_avg__";
/// `terms` over the same values as a `rare_terms` (global per-key counts).
/// Root-level `global > filter > terms` giving a `significant_terms`'s
/// background counts per key.
const SIG_BG_HELPER: &str = "__fanout_sigbg__";
/// Per-level exact subset counts of a `significant_terms` field.
const SIG_SUB_HELPER: &str = "__fanout_sigsub__";
/// `filter > sum` of a `weighted_avg`'s weights over the docs it averages.
const WAVG_HELPER: &str = "__fanout_wavg__";
/// Buckets a node is asked for when every bucket is needed for exactness.
const ALL_LIMIT: u64 = 10_000;
const PARENT_PIPELINES: &[&str] = &["bucket_sort", "bucket_selector", "bucket_script", "cumulative_sum", "derivative"];
const SIBLING_PIPELINES: &[&str] = &["avg_bucket", "sum_bucket", "max_bucket", "min_bucket", "stats_bucket", "extended_stats_bucket", "percentiles_bucket"];

thread_local! {
    /// `search.max_buckets` of the current reduce.
    static MAX_BUCKETS: std::cell::Cell<usize> = const { std::cell::Cell::new(65_536) };
    /// First reduce error of the current request (status, type, reason).
    static ERR: RefCell<Option<(u16, String, String)>> = const { RefCell::new(None) };
    /// The reduce's circuit breaker trip, whole (`bytes_wanted`,
    /// `bytes_limit`, `durability` included).
    static BREAKING: RefCell<Option<elasticsearch_engine::error::EsError>> = const { RefCell::new(None) };
    static REDUCE_MEMORY: RefCell<Vec<elasticsearch_engine::memory::Reservation>> = const { RefCell::new(Vec::new()) };
    /// Merged background counts of each `significant_terms`, by agg name:
    /// (`key` JSON -> count, complete).
    static SIG_BG: RefCell<HashMap<String, (HashMap<String, u64>, bool)>> = RefCell::new(HashMap::new());
}

/// Keep growth charges attached to the active reduction, including unwind.
/// Nested root reductions get their own slot and cannot release the outer one.
struct ReduceMemory(bool);
impl ReduceMemory {
    fn enter(memory:elasticsearch_engine::memory::Reservation)->Self { REDUCE_MEMORY.with(|stack|stack.borrow_mut().push(memory));Self(true) }
    fn finish(mut self)->elasticsearch_engine::memory::Reservation {self.0=false;REDUCE_MEMORY.with(|stack|stack.borrow_mut().pop().unwrap())}
}
impl Drop for ReduceMemory {fn drop(&mut self){if self.0{REDUCE_MEMORY.with(|stack|{stack.borrow_mut().pop();});}}}
fn admit_reduce_growth(bytes:u64)->bool {
    let result=REDUCE_MEMORY.with(|stack|match stack.borrow_mut().last_mut(){Some(memory)=>memory.grow(bytes,"<reduce_aggs>"),None=>Ok(())});
    match result {Ok(())=>true,Err(error)=>{BREAKING.with(|slot|*slot.borrow_mut()=Some(error.clone()));fail_es(error);false}}
}
#[cfg(test)]
thread_local! {static FILLED_BUCKETS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}

fn fail(reason: impl Into<String>) {
    fail_with(400, "illegal_argument_exception", reason)
}

fn fail_with(status: u16, kind: &str, reason: impl Into<String>) {
    ERR.with(|e| {
        let mut e = e.borrow_mut();
        if e.is_none() {
            *e = Some((status, kind.into(), reason.into()));
        }
    });
}

fn fail_es(e: elasticsearch_engine::error::EsError) {
    fail_with(e.status, &e.kind, e.reason)
}

/// The first error raised while reducing (cleared by the call).
pub fn take_error() -> Option<(u16, String, String)> {
    BREAKING.with(|e| e.borrow_mut().take());
    ERR.with(|e| e.borrow_mut().take())
}

/// The reduce's circuit breaker trip with all its fields (cleared, with
/// [`take_error`]'s, by the call).
pub fn take_breaking() -> Option<elasticsearch_engine::error::EsError> {
    let e = BREAKING.with(|e| e.borrow_mut().take());
    if e.is_some() {
        ERR.with(|e| e.borrow_mut().take());
    }
    e
}


/// Java `Double.toString`.
fn jfmt(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let a = v.abs();
    if a == 0.0 || (1e-3..1e7).contains(&a) {
        let s = format!("{v}");
        if s.contains('.') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
        format!("{m}E{e}")
    }
}

fn fnum(v: f64) -> Value {
    elasticsearch_engine::values::java_double(v)
}

fn opt_num(v: Option<f64>) -> Value {
    v.map(fnum).unwrap_or(Value::Null)
}
/// Upper bound on synthesized empty histogram buckets.

const PIPELINE: &[&str] = &[
    "avg_bucket", "max_bucket", "min_bucket", "sum_bucket", "stats_bucket", "extended_stats_bucket", "percentiles_bucket", "derivative", "cumulative_sum",
    "cumulative_cardinality", "bucket_script", "bucket_selector", "bucket_sort", "moving_fn", "moving_avg", "moving_percentiles", "serial_diff", "normalize",
    "inference", "bucket_count_ks_test", "bucket_correlation", "change_point",
];

/// `(type, config, sub-aggregations)` of one named aggregation.
fn parse(entry: &Value) -> Option<(&str, &Value, Option<&Value>)> {
    let mut ty = None;
    let mut subs = None;
    for (k, v) in entry.as_object()? {
        match k.as_str() {
            "aggs" | "aggregations" => subs = Some(v),
            "meta" => {}
            _ => ty = Some((k.as_str(), v)),
        }
    }
    let (t, c) = ty?;
    Some((t, c, subs))
}

/// The first aggregation in the tree that cannot be reduced here.
pub fn unsupported(aggs: &Value) -> Option<&'static str> {
    for (_, e) in aggs.as_object()? {
        // Malformed entries go to the nodes, which answer with ES's error.
        let Some((ty, _, subs)) = parse(e) else { continue };
        match ty {
            "min" | "max" | "sum" | "avg" | "value_count" | "stats" | "extended_stats" | "cardinality" | "top_hits" | "terms" | "histogram" | "date_histogram"
            | "filter" | "filters" | "global" | "missing" | "nested" | "reverse_nested" | "sampler" | "range" | "date_range" | "ip_range" | "geo_distance" => {}
            "percentiles" | "percentile_ranks" | "composite" | "significant_terms" | "significant_text" | "rare_terms" | "multi_terms" | "frequent_item_sets" | "frequent_items" => {}
            "rate" | "weighted_avg" | "geo_centroid" | "cartesian_centroid" | "geo_bounds" | "cartesian_bounds" | "top_metrics" | "string_stats" => {}
            "auto_date_histogram" | "variable_width_histogram" | "counted_terms" | "categorize_text" => {}
            "median_absolute_deviation" | "boxplot" | "matrix_stats" | "t_test" | "geo_line" => {}
            t if elasticsearch_engine::aggs::partial::ENGINE_PARENT_PIPELINES.contains(&t) || elasticsearch_engine::aggs::partial::ENGINE_SIBLING_PIPELINES.contains(&t) => {}
            "geohash_grid" | "geotile_grid" | "geohex_grid" | "adjacency_matrix" | "random_sampler" | "diversified_sampler" | "children" | "parent" | "time_series" | "scripted_metric" => {}
            t if PARENT_PIPELINES.contains(&t) || SIBLING_PIPELINES.contains(&t) => {}
            t if PIPELINE.contains(&t) => return Some("this pipeline aggregation"),
            _ => return Some("this aggregation type"),
        }
        if let Some(why) = subs.and_then(unsupported) {
            return Some(why);
        }
    }
    None
}

fn uint(v: Option<&Value>) -> Option<u64> {
    v.and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
}

fn terms_sizes(conf: &Value) -> (usize, usize) {
    let size = uint(conf.get("size")).unwrap_or(10) as usize;
    let shard = uint(conf.get("shard_size")).map(|s| s as usize).unwrap_or(size * 3 / 2 + 10).max(size);
    (size, shard)
}

/// The aggregation tree each node is asked for.
pub fn rewrite(aggs: &Value) -> Value {
    let Some(m) = aggs.as_object() else { return aggs.clone() };
    let mut out = Map::new();
    for (name, e) in m {
        let Some((ty, conf, _)) = parse(e) else {
            out.insert(name.clone(), e.clone());
            continue;
        };
        let ty = ty.to_string();
        let mut entry = e.as_object().cloned().unwrap_or_default();
        let mut helper: Option<(String, Value)> = None;
        let mut c = conf.clone();
        if let Some(o) = c.as_object_mut() {
            match ty.as_str() {
                "terms" => {
                    let (_, shard) = terms_sizes(conf);
                    o.shift_remove("shard_size");
                    o.insert("size".into(), json!(shard));
                    if uint(conf.get("min_doc_count")).unwrap_or(1) > 1 {
                        o.insert("min_doc_count".into(), json!(uint(conf.get("shard_min_doc_count")).unwrap_or(0).max(1)));
                    }
                }
                "histogram" | "date_histogram" => {
                    if uint(conf.get("min_doc_count")).unwrap_or(0) > 1 {
                        o.insert("min_doc_count".into(), json!(1));
                    }
                }
                "top_hits" => {
                    let total = uint(conf.get("from")).unwrap_or(0) + uint(conf.get("size")).unwrap_or(3);
                    o.insert("from".into(), json!(0));
                    o.insert("size".into(), json!(total));
                }
                "avg" => {
                    let mut s = o.clone();
                    s.shift_remove("format");
                    helper = Some((format!("{AVG_HELPER}{name}"), json!({ "stats": s })));
                }
                // `cardinality` / percentiles / `geo_bounds`: the shards attach
                // their mergeable sketch state (`SHARD_STATE_AGGS`), no helper.
                "geohash_grid" | "geotile_grid" | "geohex_grid" => {
                    let size = uint(conf.get("size")).unwrap_or(10000) as usize;
                    let shard = uint(conf.get("shard_size")).map(|s| s as usize).unwrap_or(size * 3 / 2 + 10).max(size);
                    o.shift_remove("shard_size");
                    o.insert("size".into(), json!(shard));
                }
                "string_stats" => {
                    o.insert("show_distribution".into(), json!(true));
                }
                "scripted_metric" => {
                    // Each node returns its shards' combine states; the
                    // coordinator runs the request's reduce_script over all.
                    o.insert("reduce_script".into(), json!({"source": "return states;"}));
                }
                "weighted_avg" => {
                    let weight = conf.get("weight").and_then(Value::as_object).map(|w| pick(w, &["field", "script", "missing"], 0)).unwrap_or_default();
                    let mut weight = weight;
                    weight.shift_remove("size");
                    // Only docs with a value (or a `missing` for it) are averaged.
                    let value = &conf["value"];
                    let filter = match (value.get("field").and_then(Value::as_str), value.get("missing")) {
                        (Some(f), None) => json!({"exists": {"field": f}}),
                        _ => json!({"match_all": {}}),
                    };
                    helper = Some((format!("{WAVG_HELPER}{name}"), json!({"filter": filter, "aggs": {"w": {"sum": weight}}})));
                }
                // `rare_terms`: shards attach a filter of their frequent terms.
                "counted_terms" => {
                    let (_, shard) = terms_sizes(conf);
                    o.shift_remove("shard_size");
                    o.insert("size".into(), json!(shard));
                }
                "categorize_text" => {
                    // ES: shards return `shard_size` categories.
                    let size = uint(conf.get("size")).unwrap_or(10);
                    let shard = uint(conf.get("shard_size")).unwrap_or(size * 3 / 2 + 10).max(size);
                    o.insert("size".into(), json!(shard));
                    o.insert("min_doc_count".into(), json!(uint(conf.get("shard_min_doc_count")).unwrap_or(0).max(1)));
                    o.shift_remove("shard_size");
                    o.shift_remove("shard_min_doc_count");
                }
                "variable_width_histogram" => {
                    // ES: shards cluster into `shard_size` (buckets * 50) buckets.
                    let buckets = uint(conf.get("buckets")).unwrap_or(10);
                    let shard = uint(conf.get("shard_size")).unwrap_or(buckets * 50).max(buckets);
                    o.shift_remove("shard_size");
                    o.insert("buckets".into(), json!(shard));
                }
                "multi_terms" => {
                    let (_, shard) = terms_sizes(conf);
                    o.shift_remove("shard_size");
                    o.insert("size".into(), json!(shard.max(1)));
                    if uint(conf.get("min_doc_count")).unwrap_or(1) > 1 {
                        o.insert("min_doc_count".into(), json!(1));
                    }
                }
                "significant_terms" => {
                    // ES: each shard's `shard_size` best candidates under the
                    // request's heuristic, with their subset/superset counts;
                    // scoring and thresholds happen on the merged counts.
                    let size = uint(conf.get("size")).unwrap_or(10);
                    let shard = uint(conf.get("shard_size")).unwrap_or(size * 3 / 2 + 10).max(size);
                    o.shift_remove("shard_size");
                    o.insert("size".into(), json!(shard));
                    o.insert("min_doc_count".into(), json!(uint(conf.get("shard_min_doc_count")).unwrap_or(0).max(1)));
                    o.shift_remove("shard_min_doc_count");
                    // Exact subset counts while the field has few terms.
                    let t = pick(conf.as_object().unwrap_or(&Map::new()), &["field", "include", "exclude"], ALL_LIMIT);
                    helper = Some((format!("{SIG_SUB_HELPER}{name}"), json!({ "terms": t })));
                }
                _ => {}
            }
        }
        entry.insert(ty, c);
        for k in ["aggs", "aggregations"] {
            if let Some(s) = entry.get(k).cloned() {
                // Parent pipelines that drop or reorder buckets run on the
                // coordinator only, over the merged buckets.
                let mut s = rewrite(&s);
                if let Some(m) = s.as_object_mut() {
                    m.retain(|_, v| !matches!(parse(v), Some(("bucket_sort" | "bucket_selector", _, _))));
                }
                entry.insert(k.into(), s);
            }
        }
        out.insert(name.clone(), Value::Object(entry));
        if let Some((n, h)) = helper {
            out.insert(n, h);
        }
    }
    Value::Object(out)
}

/// [`rewrite`] for the request's top level: also adds the root-level
/// background-count helpers of every `significant_terms` in the tree.
pub fn rewrite_root(aggs: &Value) -> Value {
    let mut out = rewrite(aggs);
    let mut sigs = Vec::new();
    collect_sig(aggs, &mut sigs);
    if let Some(m) = out.as_object_mut() {
        for (name, conf) in sigs {
            let key = format!("{SIG_BG_HELPER}{name}");
            if m.contains_key(&key) {
                continue;
            }
            let t = pick(conf.as_object().unwrap_or(&Map::new()), &["field", "include", "exclude"], ALL_LIMIT);
            let filter = conf.get("background_filter").cloned().unwrap_or_else(|| json!({"match_all": {}}));
            m.insert(key, json!({"global": {}, "aggs": {"f": {"filter": filter, "aggs": {"t": {"terms": t}}}}}));
        }
    }
    out
}

fn collect_sig(aggs: &Value, out: &mut Vec<(String, Value)>) {
    let Some(m) = aggs.as_object() else { return };
    for (name, e) in m {
        let Some((ty, conf, subs)) = parse(e) else { continue };
        if ty == "significant_terms" {
            out.push((name.clone(), conf.clone()));
        }
        if let Some(s) = subs {
            collect_sig(s, out);
        }
    }
}

/// A `terms` config over the same values as `o`.
fn pick(o: &Map<String, Value>, keys: &[&str], size: u64) -> Map<String, Value> {
    let mut t = Map::new();
    for k in keys {
        if let Some(v) = o.get(*k) {
            t.insert((*k).into(), v.clone());
        }
    }
    t.insert("size".into(), json!(size));
    t
}

const HEURISTICS: &[&str] = &["jlh", "mutual_information", "chi_square", "gnd", "percentage", "script_heuristic"];

/// The `significant_terms` scoring heuristic and its parameters.
fn heuristic(conf: &Value) -> (&'static str, Map<String, Value>) {
    for h in HEURISTICS {
        if let Some(v) = conf.get(*h) {
            return (*h, v.as_object().cloned().unwrap_or_default());
        }
    }
    ("jlh", Map::new())
}

/// A named aggregation result in one rendered level (`name` or `type#name`).
fn find<'a>(part: &'a Value, name: &str) -> Option<(&'a str, &'a Value)> {
    let m = part.as_object()?;
    if let Some((k, v)) = m.get_key_value(name) {
        return Some((k.as_str(), v));
    }
    m.iter().find(|(k, _)| k.split_once('#').map(|(_, n)| n == name).unwrap_or(false)).map(|(k, v)| (k.as_str(), v))
}

/// Reduce one level: `spec` is the request's aggregation map, `parts` the
/// rendered maps (a response's `aggregations`, or a bucket) from each node.
pub fn reduce(spec: &Value, parts: &[&Value], typed: bool) -> Value {
    let mut out = Map::new();
    let Some(m) = spec.as_object() else { return Value::Object(out) };
    // Plain-name results of this level, for sibling pipelines.
    let mut results: HashMap<String, Value> = HashMap::new();
    let mut siblings = Vec::new();
    for (name, e) in m {
        let Some((ty, conf, subs)) = parse(e) else { continue };
        let found: Vec<(&str, &Value)> = parts.iter().filter_map(|p| find(p, name)).collect();
        if PARENT_PIPELINES.contains(&ty) || elasticsearch_engine::aggs::partial::ENGINE_PARENT_PIPELINES.contains(&ty) {
            // Inserted into the buckets by the enclosing bucket reducer.
            continue;
        }
        if SIBLING_PIPELINES.contains(&ty) || elasticsearch_engine::aggs::partial::ENGINE_SIBLING_PIPELINES.contains(&ty) {
            siblings.push((name, e, ty, conf, found.first().map(|f| f.0.to_string())));
            continue;
        }
        let vals: Vec<&Value> = found.iter().map(|f| f.1).collect();
        let helper = |prefix: &str| -> Vec<Option<&Value>> {
            let h = format!("{prefix}{name}");
            parts.iter().filter(|p| find(p, name).is_some()).map(|p| find(p, &h).map(|f| f.1)).collect()
        };
        let v = match ty {
            "avg" => reduce_avg(conf, &vals, &helper(AVG_HELPER)),
            "rare_terms" => reduce_rare(name, conf, subs, &vals, typed),
            "variable_width_histogram" => reduce_variable_width(conf, subs, &vals, typed),
            "categorize_text" => reduce_categorize(name, conf, subs, &vals, typed),
            "counted_terms" => reduce_terms(conf, subs, &vals, typed),
            t if elasticsearch_engine::aggs::partial::SHARD_STATE_AGGS.contains(&t) => {
                match elasticsearch_engine::aggs::partial::reduce_shard_states(name, t, conf.as_object().unwrap_or(&Map::new()), &vals) {
                    Some(Ok(v)) => v,
                    Some(Err(e)) => { fail_es(e); json!({}) }
                    None if t == "geo_bounds" => reduce_bounds(t, &vals),
                    None => {
                        fail_with(500, "illegal_state_exception", format!("[{t}] aggregation [{name}] needs mergeable sketch state from every shard (a node predates it)"));
                        json!({})
                    }
                }
            }

            "significant_terms" => reduce_sig(name, conf, subs, &vals, &helper(SIG_SUB_HELPER), typed),
            // The coordinator checks the complete output tree using the
            // configured limit after all sibling/child reducers finish.
            "significant_text" => match elasticsearch_engine::aggs::significant_text::reduce(conf.as_object().unwrap_or(&Map::new()), &vals, usize::MAX, true) {
                Ok(v) => v,
                Err(e) => { fail_es(e); json!({}) }
            },
            "frequent_item_sets" | "frequent_items" => match elasticsearch_engine::aggs::frequent_items::reduce(conf.as_object().unwrap_or(&Map::new()), &vals, true) {
                Ok(v) => v,
                Err(e) => { fail_es(e); json!({}) }
            },
            "multi_terms" => reduce_multi_terms(conf, subs, &vals, typed),
            "composite" => reduce_composite(conf, subs, &vals, typed),
            "weighted_avg" => reduce_weighted_avg(&vals, &helper(WAVG_HELPER)),
            "scripted_metric" => reduce_scripted_metric(name, conf, &vals),
            _ => reduce_one(ty, conf, subs, &vals, typed),
        };
        let mut v = v;
        // The payload a parent pipeline of the enclosing buckets reads.
        let payloads: Vec<&Value> = vals.iter().filter_map(|x| x.get(elasticsearch_engine::aggs::partial::PIPELINE_PAYLOAD)).collect();
        if let (Some(o), Some(p)) = (v.as_object_mut(), elasticsearch_engine::aggs::partial::merge_pipeline_payloads(&payloads)) {
            o.insert(elasticsearch_engine::aggs::partial::PIPELINE_PAYLOAD.into(), p);
        }
        results.insert(name.clone(), v.clone());
        // Unmapped/empty nodes may render a different typed prefix
        // (`sterms` for an empty `terms`); prefer a node with buckets.
        let has_buckets = |v: &Value| match v.get("buckets") {
            Some(Value::Array(a)) => !a.is_empty(),
            Some(Value::Object(o)) => !o.is_empty(),
            _ => false,
        };
        let key = if typed {
            found.iter().find(|f| has_buckets(f.1)).or(found.first()).map(|f| f.0.to_string()).unwrap_or_else(|| format!("{ty}#{name}"))
        } else {
            name.clone()
        };
        out.insert(key, with_meta(e, v));
    }
    for (name, e, ty, conf, node_key) in siblings {
        let v = if elasticsearch_engine::aggs::partial::ENGINE_SIBLING_PIPELINES.contains(&ty) {
            let names = spec_names(Some(spec));
            let plain: HashMap<String, Value> = results.iter().map(|(k, v)| (k.clone(), untyped(v, &names))).collect();
            elasticsearch_engine::aggs::partial::engine_sibling_pipeline(name, ty, conf, &plain).unwrap_or_else(|e| {
                fail_es(e);
                json!({})
            })
        } else {
            sibling_pipeline(name, ty, conf, &results).unwrap_or_else(|err| {
                fail(err);
                json!({})
            })
        };
        results.insert(name.clone(), v.clone());
        let key = if typed { node_key.unwrap_or_else(|| format!("{ty}#{name}")) } else { name.clone() };
        out.insert(key, with_meta(e, v));
    }
    Value::Object(out)
}

/// ES renders `meta` ahead of the aggregation's own fields.
fn with_meta(e: &Value, mut v: Value) -> Value {
    if let (Some(meta), Some(o)) = (e.get("meta"), v.as_object_mut()) {
        o.shift_remove("meta");
        let mut first = Map::new();
        first.insert("meta".into(), meta.clone());
        first.extend(std::mem::take(o));
        *o = first;
    }
    v
}

/// Reduce the top level of a search's aggregations. Reduce errors are
/// collected for [`take_error`].
pub fn reduce_root(spec: &Value, parts: &[&Value], typed: bool) -> Value {
    reduce_root_with_limit(spec, parts, typed, 65_536)
}

pub fn reduce_root_with_limit(spec: &Value, parts: &[&Value], typed: bool, max_buckets: usize) -> Value {
    ERR.with(|e| *e.borrow_mut() = None);
    BREAKING.with(|e| *e.borrow_mut() = None);
    // ES `QueryPhaseResultConsumer`: the nodes' partials are held on the
    // request breaker during the reduce, with half as much again for its result.
    let held=parts.iter().fold(0u64,|n,p|n.saturating_add(elasticsearch_engine::memory::json_bytes(p)));
    // The partials move into this charge rather than being counted twice.
    elasticsearch_engine::memory::consume_retained_partials(held);
    let nested=elasticsearch_engine::memory::partials_mark();
    let mem=match elasticsearch_engine::memory::breakers().reserve_request(held.saturating_add(held/2),"<reduce_aggs>") {
        Ok(mem)=>mem,
        Err(error)=>{BREAKING.with(|b|*b.borrow_mut()=Some(error.clone()));fail_es(error);return Value::Null;},
    };
    let memory=ReduceMemory::enter(mem);
    MAX_BUCKETS.with(|m| m.set(max_buckets));
    let mut bg: HashMap<String, (HashMap<String, u64>, bool)> = HashMap::new();
    let mut sigs = Vec::new();
    collect_sig(spec, &mut sigs);
    for (name, _) in sigs {
        let h = format!("{SIG_BG_HELPER}{name}");
        let entry = bg.entry(name).or_insert_with(|| (HashMap::new(), true));
        for p in parts {
            let terms = find(p, &h).and_then(|(_, g)| find(g, "f")).and_then(|(_, f)| find(f, "t")).map(|(_, t)| t);
            let Some(t) = terms else {
                entry.1 = false;
                continue;
            };
            if count_of(t, "sum_other_doc_count") > 0 {
                entry.1 = false;
            }
            for b in t["buckets"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                *entry.0.entry(b["key"].to_string()).or_insert(0) += count_of(b, "doc_count");
            }
        }
    }
    SIG_BG.with(|s| *s.borrow_mut() = bg);
    let mut out = reduce(spec, parts, typed);
    SIG_BG.with(|s| s.borrow_mut().clear());
    if let Some(results)=out.as_object_mut(){for result in results.values_mut(){strip_shard_state(result);}}
    // ES's final reduce counts every bucket against `search.max_buckets`.
    if let Err(e) = elasticsearch_engine::aggs::significant_text::check_final_bucket_limit(spec, &out, max_buckets) { fail_es(e); }
    let mut mem=memory.finish();
    if BREAKING.with(|slot|slot.borrow().is_some()){return Value::Null;}
    let output=elasticsearch_engine::memory::json_bytes(&out);
    if output>mem.bytes(){
        if let Err(error)=mem.grow(output-mem.bytes(),"<reduce_aggs>") {
            BREAKING.with(|b|*b.borrow_mut()=Some(error.clone()));fail_es(error);return Value::Null;
        }
    } else {mem.shrink(mem.bytes()-output);}
    elasticsearch_engine::memory::consume_partials_since(nested);
    mem.retain_for_request();
    out
}

fn num(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

fn count_of(v: &Value, k: &str) -> u64 {
    v.get(k).map(|c| c.as_u64().unwrap_or_else(|| c.as_f64().unwrap_or(0.0) as u64)).unwrap_or(0)
}

fn fmt_date(conf: &Value, millis: f64) -> Option<String> {
    let spec = conf.get("format").and_then(Value::as_str).unwrap_or("strict_date_optional_time");
    Some(edate::DateFormatter::new(spec).ok()?.format(millis as i64, edate::Tz::UTC))
}

fn reduce_one(ty: &str, conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    match ty {
        "min" | "max" => {
            let mut best: Option<(&Value, f64)> = None;
            for v in vals {
                if let Some(x) = num(v, "value") {
                    if best.map_or(true, |(_, y)| if ty == "min" { x < y } else { x > y }) {
                        best = Some((*v, x));
                    }
                }
            }
            best.map(|b| b.0.clone()).or_else(|| vals.first().map(|v| (*v).clone())).unwrap_or_else(|| json!({"value": null}))
        }
        "sum" => json!({"value": vals.iter().filter_map(|v| num(v, "value")).sum::<f64>()}),
        "value_count" => json!({"value": vals.iter().map(|v| count_of(v, "value")).sum::<u64>()}),
        "stats" | "extended_stats" => reduce_stats(ty, conf, vals),
        "top_hits" => reduce_top_hits(conf, vals),
        "terms" => reduce_terms(conf, subs, vals, typed),
        "histogram" | "date_histogram" => reduce_histogram(ty, conf, subs, vals, typed),
        "auto_date_histogram" => reduce_auto_date_histogram(conf, subs, vals, typed),
        "filters" | "range" | "date_range" | "ip_range" | "geo_distance" => reduce_fixed(subs, vals, typed),
        "rate" => reduce_rate(vals),
        "geo_centroid" | "cartesian_centroid" => reduce_centroid(ty, vals),
        "geo_bounds" | "cartesian_bounds" => reduce_bounds(ty, vals),
        "top_metrics" => reduce_top_metrics(conf, vals),
        "string_stats" => reduce_string_stats(conf, vals),
        "geohash_grid" | "geotile_grid" | "geohex_grid" => reduce_grid(ty, conf, subs, vals, typed),
        "adjacency_matrix" => reduce_adjacency(subs, vals, typed),
        "time_series" => reduce_time_series(subs, vals, typed),
        // filter, global, missing, nested, reverse_nested, sampler,
        // random_sampler, diversified_sampler, children, parent
        _ => Value::Object(merge_bucket(vals, subs, typed)),
    }
}

/// `scripted_metric`: the nodes' combine states (rewritten to come back
/// unreduced) in node order, reduced by the request's `reduce_script`.
fn reduce_scripted_metric(name: &str, conf: &Value, vals: &[&Value]) -> Value {
    let Some(o) = conf.as_object() else { return json!({"value": null}) };
    match elasticsearch_engine::aggs::scripted_metric_reduce_source(name, o, vals) {
        Ok(v) => v,
        Err(e) => {
            fail_es(e);
            json!({"value": null})
        }
    }
}

/// `rate`: every node divides by the same unit (the enclosing bucket's or
/// the requested one), so the node rates add up.
fn reduce_rate(vals: &[&Value]) -> Value {
    if vals.len() == 1 {
        return vals[0].clone();
    }
    json!({"value": fnum(vals.iter().filter_map(|v| num(v, "value")).sum::<f64>())})
}

/// `weighted_avg`: each node's average weighted by the summed weights of
/// the docs it averaged (the helper).
fn reduce_weighted_avg(vals: &[&Value], helpers: &[Option<&Value>]) -> Value {
    let (mut num_sum, mut weights) = (0f64, 0f64);
    let mut live = Vec::new();
    for (i, v) in vals.iter().enumerate() {
        let w = helpers.get(i).copied().flatten().and_then(|h| find(h, "w")).and_then(|(_, w)| num(w, "value")).unwrap_or(0.0);
        if let (Some(x), true) = (num(v, "value"), w != 0.0) {
            num_sum += x * w;
            weights += w;
            live.push(i);
        }
    }
    match live.len() {
        0 => vals.first().map(|v| (*v).clone()).unwrap_or_else(|| json!({"value": null})),
        1 => vals[live[0]].clone(),
        _ => json!({"value": fnum(num_sum / weights)}),
    }
}

/// `geo_centroid` / `cartesian_centroid`: node centroids weighted by count.
fn reduce_centroid(ty: &str, vals: &[&Value]) -> Value {
    let live: Vec<&Value> = vals.iter().copied().filter(|v| count_of(v, "count") > 0).collect();
    if live.len() <= 1 {
        return live.first().or(vals.first()).map(|v| (*v).clone()).unwrap_or_else(|| json!({"count": 0}));
    }
    let (a, b) = if ty == "geo_centroid" { ("lat", "lon") } else { ("x", "y") };
    let n: u64 = live.iter().map(|v| count_of(v, "count")).sum();
    let mean = |k: &str| live.iter().map(|v| v["location"][k].as_f64().unwrap_or(0.0) * count_of(v, "count") as f64).sum::<f64>() / n as f64;
    json!({"location": {a: fnum(mean(a)), b: fnum(mean(b))}, "count": n})
}

/// `geo_bounds` / `cartesian_bounds`: the box covering every node's box.
fn reduce_bounds(ty: &str, vals: &[&Value]) -> Value {
    let live: Vec<&Value> = vals.iter().copied().filter(|v| v.get("bounds").is_some()).collect();
    if live.len() <= 1 {
        return live.first().or(vals.first()).map(|v| (*v).clone()).unwrap_or_else(|| json!({}));
    }
    let (y, x) = if ty == "geo_bounds" { ("lat", "lon") } else { ("y", "x") };
    let get = |corner: &str, k: &str| live.iter().filter_map(|v| v["bounds"][corner][k].as_f64()).collect::<Vec<f64>>();
    let max = |v: Vec<f64>| v.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let min = |v: Vec<f64>| v.into_iter().fold(f64::INFINITY, f64::min);
    let top_left = json!({y: fnum(max(get("top_left", y))), x: fnum(min(get("top_left", x)))});
    let bottom_right = json!({y: fnum(min(get("bottom_right", y))), x: fnum(max(get("bottom_right", x)))});
    let (tl, br) = if ty == "geo_bounds" { (top_left, bottom_right) } else {
        (json!({"x": top_left["x"], "y": top_left["y"]}), json!({"x": bottom_right["x"], "y": bottom_right["y"]}))
    };
    json!({"bounds": {"top_left": tl, "bottom_right": br}})
}

/// `top_metrics`: every node's top docs merged by their sort value.
fn reduce_top_metrics(conf: &Value, vals: &[&Value]) -> Value {
    let size = uint(conf.get("size")).unwrap_or(1) as usize;
    let desc = match &conf["sort"] {
        Value::Object(o) => o.iter().next().map(|(k, v)| match v {
            Value::String(d) => d == "desc",
            Value::Object(m) => m.get("order").and_then(Value::as_str).map(|d| d == "desc").unwrap_or(k == "_score"),
            _ => k == "_score",
        }).unwrap_or(false),
        Value::Array(a) => a.first().and_then(Value::as_object).and_then(|o| o.iter().next()).map(|(k, v)| v.as_str().map(|d| d == "desc").or_else(|| v.get("order").and_then(Value::as_str).map(|d| d == "desc")).unwrap_or(k == "_score")).unwrap_or(false),
        Value::String(k) => k == "_score",
        _ => false,
    };
    let mut top: Vec<Value> = vals.iter().flat_map(|v| v["top"].as_array().cloned().unwrap_or_default()).collect();
    top.sort_by(|a, b| {
        let o = cmp_key(&a["sort"][0], &b["sort"][0]);
        if desc { o.reverse() } else { o }
    });
    top.truncate(size);
    json!({"top": top})
}

/// `string_stats`: lengths combined by count; entropy from the merged
/// character distribution (every node is asked for its distribution).
fn reduce_string_stats(conf: &Value, vals: &[&Value]) -> Value {
    let show = conf.get("show_distribution").and_then(Value::as_bool).unwrap_or(false);
    let live: Vec<&Value> = vals.iter().copied().filter(|v| count_of(v, "count") > 0).collect();
    let strip = |v: &Value| {
        let mut v = v.clone();
        if !show {
            if let Some(o) = v.as_object_mut() { o.shift_remove("distribution"); }
        }
        v
    };
    if live.len() <= 1 {
        return live.first().or(vals.first()).map(|v| strip(v)).unwrap_or_else(|| json!({"count": 0}));
    }
    let n: u64 = live.iter().map(|v| count_of(v, "count")).sum();
    let chars: Vec<f64> = live.iter().map(|v| num(v, "avg_length").unwrap_or(0.0) * count_of(v, "count") as f64).collect();
    let total: f64 = chars.iter().sum();
    let mut dist: HashMap<String, f64> = HashMap::new();
    for (v, c) in live.iter().zip(&chars) {
        for (k, p) in v["distribution"].as_object().map(|o| o.iter().collect::<Vec<_>>()).unwrap_or_default() {
            *dist.entry(k.clone()).or_insert(0.0) += p.as_f64().unwrap_or(0.0) * c;
        }
    }
    let mut dist: Vec<(String, f64)> = dist.into_iter().map(|(k, c)| (k, if total > 0.0 { c / total } else { 0.0 })).collect();
    dist.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let entropy: f64 = -dist.iter().filter(|(_, p)| *p > 0.0).map(|(_, p)| p * p.log2()).sum::<f64>();
    let min = live.iter().filter_map(|v| v["min_length"].as_u64()).min();
    let max = live.iter().filter_map(|v| v["max_length"].as_u64()).max();
    let mut out = json!({"count": n, "min_length": min, "max_length": max, "avg_length": fnum(total / n as f64), "entropy": fnum(entropy)});
    if show {
        let mut m = Map::new();
        for (k, p) in dist {
            m.insert(k, fnum(p));
        }
        out["distribution"] = Value::Object(m);
    }
    out
}

/// Geo grids: buckets merged by cell, ordered by count (ties by key), cut
/// to `size`.
fn reduce_grid(ty: &str, conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let size = uint(conf.get("size")).unwrap_or(10000) as usize;
    let mut buckets: Vec<Value> = group_buckets(vals).iter().map(|g| Value::Object(merge_bucket(g, subs, typed))).collect();
    // ES breaks count ties by the cell's long encoding, highest first.
    let cell = |k: &Value| -> (u64, u64, u64) {
        let k = k.as_str().unwrap_or_default();
        match ty {
            "geotile_grid" => {
                let mut zxy = k.split('/').map(|p| p.parse::<u64>().unwrap_or(0));
                (zxy.next().unwrap_or(0), zxy.next().unwrap_or(0), zxy.next().unwrap_or(0))
            }
            "geohex_grid" => (u64::from_str_radix(k, 16).unwrap_or(0), 0, 0),
            _ => (geohash_long(k), 0, 0),
        }
    };
    buckets.sort_by(|a, b| count_of(b, "doc_count").cmp(&count_of(a, "doc_count")).then_with(|| cell(&b["key"]).cmp(&cell(&a["key"]))));
    buckets.truncate(size);
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets})
}

/// A geohash as ES's `Geohash.longEncode` orders it (bits, then length).
fn geohash_long(hash: &str) -> u64 {
    const BASE32: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";
    let mut bits = 0u64;
    for c in hash.bytes().take(12) {
        let v = BASE32.iter().position(|b| *b == c).unwrap_or(0) as u64;
        bits = (bits << 5) | v;
    }
    let len = hash.len().min(12) as u64;
    (bits << (4 + (12 - len) * 5)) | len
}

/// `time_series`: every series is routed to one shard, so buckets merge by
/// key (`keyed` renders an object).
fn reduce_time_series(subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    if vals.first().is_some_and(|v| v["buckets"].is_object()) {
        return reduce_fixed(subs, vals, typed);
    }
    let mut buckets: Vec<Value> = group_buckets(vals).iter().map(|g| Value::Object(merge_bucket(g, subs, typed))).collect();
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets})
}

/// `adjacency_matrix`: buckets merged by key, in key order.
fn reduce_adjacency(subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let mut buckets: Vec<Value> = group_buckets(vals).iter().map(|g| Value::Object(merge_bucket(g, subs, typed))).collect();
    buckets.sort_by(|a, b| cmp_key(&a["key"], &b["key"]));
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets})
}

fn reduce_avg(conf: &Value, vals: &[&Value], helpers: &[Option<&Value>]) -> Value {
    let live: Vec<usize> = (0..vals.len()).filter(|i| helpers.get(*i).copied().flatten().map(|h| count_of(h, "count") > 0).unwrap_or(false)).collect();
    match live.len() {
        0 => vals.first().map(|v| (*v).clone()).unwrap_or_else(|| json!({"value": null})),
        // One node has all the values: its rendering is exact.
        1 => vals[live[0]].clone(),
        _ => {
            let (mut n, mut s) = (0u64, 0f64);
            for i in &live {
                let h = helpers[*i].unwrap();
                n += count_of(h, "count");
                s += num(h, "sum").unwrap_or(0.0);
            }
            let avg = s / n as f64;
            let mut out = json!({ "value": avg });
            if vals.iter().any(|v| v.get("value_as_string").is_some()) {
                if let Some(f) = fmt_date(conf, avg) {
                    out["value_as_string"] = json!(f);
                }
            }
            out
        }
    }
}

fn reduce_stats(ty: &str, conf: &Value, vals: &[&Value]) -> Value {
    let live: Vec<&Value> = vals.iter().copied().filter(|v| count_of(v, "count") > 0).collect();
    if live.len() == 1 {
        return live[0].clone();
    }
    if live.is_empty() {
        if let Some(v) = vals.first() {
            return (*v).clone();
        }
    }
    let n: u64 = live.iter().map(|v| count_of(v, "count")).sum();
    let sum: f64 = live.iter().filter_map(|v| num(v, "sum")).sum();
    let pick = |k: &str, less: bool| -> Option<(&Value, f64)> {
        let mut best: Option<(&Value, f64)> = None;
        for v in &live {
            if let Some(x) = num(v, k) {
                if best.map_or(true, |(_, y)| if less { x < y } else { x > y }) {
                    best = Some((*v, x));
                }
            }
        }
        best
    };
    let (min, max) = (pick("min", true), pick("max", false));
    let opt = |x: Option<f64>| x.map(|x| json!(x)).unwrap_or(Value::Null);
    let avg = if n > 0 { Some(sum / n as f64) } else { None };
    let mut out = Map::new();
    out.insert("count".into(), json!(n));
    out.insert("min".into(), opt(min.map(|m| m.1)));
    out.insert("max".into(), opt(max.map(|m| m.1)));
    out.insert("avg".into(), opt(avg));
    out.insert("sum".into(), json!(sum));
    if ty == "extended_stats" {
        let sq: f64 = live.iter().filter_map(|v| num(v, "sum_of_squares")).sum();
        let sigma = num(conf, "sigma").unwrap_or(2.0);
        let (vp, vs) = if n > 0 {
            let d = (sq - sum * sum / n as f64).max(0.0);
            (Some(d / n as f64), if n > 1 { Some(d / (n - 1) as f64) } else { None })
        } else {
            (None, None)
        };
        let (sp, ss) = (vp.map(f64::sqrt), vs.map(f64::sqrt));
        out.insert("sum_of_squares".into(), if n > 0 { json!(sq) } else { Value::Null });
        out.insert("variance".into(), opt(vp));
        out.insert("variance_population".into(), opt(vp));
        out.insert("variance_sampling".into(), opt(vs));
        out.insert("std_deviation".into(), opt(sp));
        out.insert("std_deviation_population".into(), opt(sp));
        out.insert("std_deviation_sampling".into(), opt(ss));
        let b = |s: Option<f64>, sign: f64| opt(avg.zip(s).map(|(a, s)| a + sign * sigma * s));
        out.insert(
            "std_deviation_bounds".into(),
            json!({"upper": b(sp, 1.0), "lower": b(sp, -1.0), "upper_population": b(sp, 1.0), "lower_population": b(sp, -1.0), "upper_sampling": b(ss, 1.0), "lower_sampling": b(ss, -1.0)}),
        );
    }
    if vals.iter().any(|v| v.get("min_as_string").is_some()) {
        for (k, src) in [("min_as_string", min), ("max_as_string", max)] {
            if let Some(s) = src.and_then(|m| m.0.get(k)) {
                out.insert(k.into(), s.clone());
            }
        }
        for (k, x) in [("avg_as_string", avg), ("sum_as_string", Some(sum))] {
            if let Some(f) = x.and_then(|x| fmt_date(conf, x)) {
                out.insert(k.into(), json!(f));
            }
        }
    }
    Value::Object(out)
}

fn reduce_top_hits(conf: &Value, vals: &[&Value]) -> Value {
    let from = uint(conf.get("from")).unwrap_or(0) as usize;
    let size = uint(conf.get("size")).unwrap_or(3) as usize;
    let keys = sort_keys(conf, &HashMap::new());
    let mut hits: Vec<Value> = Vec::new();
    let (mut total, mut max_score) = (0u64, None::<f64>);
    for v in vals {
        let h = &v["hits"];
        total += h["total"]["value"].as_u64().or_else(|| h["total"].as_u64()).unwrap_or(0);
        if let Some(s) = h["max_score"].as_f64() {
            max_score = Some(max_score.map_or(s, |m| m.max(s)));
        }
        hits.extend(h["hits"].as_array().cloned().unwrap_or_default());
    }
    hits.sort_by(|a, b| hit_order(a, b, keys.as_deref()).then_with(|| a["_index"].as_str().cmp(&b["_index"].as_str())));
    let page: Vec<Value> = hits.into_iter().skip(from).take(size).collect();
    json!({"hits": {"total": {"value": total, "relation": "eq"}, "max_score": max_score, "hits": page}})
}

/// Whether a rendered bucket key belongs to a sub-aggregation result.
fn is_sub_key(k: &str, subs: Option<&Value>) -> bool {
    let name = k.split_once('#').map(|(_, n)| n).unwrap_or(k);
    if k.starts_with("__fanout_") || name.starts_with("__fanout_") {
        return true;
    }
    subs.and_then(Value::as_object).map(|m| m.contains_key(k) || m.contains_key(name)).unwrap_or(false)
}

/// Merge the same bucket from several nodes: `doc_count` summed,
/// sub-aggregations reduced, everything else from the first.
fn merge_bucket(bs: &[&Value], subs: Option<&Value>, typed: bool) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(first) = bs.first().and_then(|b| b.as_object()) {
        for (k, v) in first {
            if k != "doc_count" && !is_sub_key(k, subs) {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    out.insert("doc_count".into(), json!(bs.iter().map(|b| count_of(b, "doc_count")).sum::<u64>()));
    if let Some(Value::Object(r)) = subs.map(|s| reduce(s, bs, typed)) {
        out.extend(r);
    }
    out
}

/// `order` as `(path, descending)` pairs.
fn order_spec(o: Option<&Value>, default: (&str, bool)) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut add = |m: &Map<String, Value>| {
        for (k, v) in m {
            out.push((k.clone(), v.as_str() == Some("desc")));
        }
    };
    match o {
        Some(Value::Object(m)) => add(m),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_object).for_each(&mut add),
        _ => {}
    }
    if out.is_empty() {
        out.push((default.0.to_string(), default.1));
    }
    out
}

fn is_key_path(p: &str) -> bool {
    p == "_key" || p == "_term"
}

fn order_val(b: &Value, path: &str) -> Value {
    if path == "_count" {
        return b["doc_count"].clone();
    }
    if is_key_path(path) {
        return b["key"].clone();
    }
    let mut cur = b;
    let segs: Vec<&str> = path.split('>').collect();
    for (i, s) in segs.iter().enumerate() {
        if let Some((_, v)) = find(cur, s) {
            cur = v;
            if i + 1 == segs.len() {
                return v.get("value").or_else(|| v.get("doc_count")).cloned().unwrap_or(Value::Null);
            }
        } else if let Some((n, metric)) = s.rsplit_once('.') {
            return find(cur, n).and_then(|(_, v)| v.get(metric)).cloned().unwrap_or(Value::Null);
        } else {
            return Value::Null;
        }
    }
    Value::Null
}

fn cmp_buckets(a: &Value, b: &Value, order: &[(String, bool)]) -> Ordering {
    for (p, desc) in order {
        let o = cmp_values(&order_val(a, p), &order_val(b, p));
        let o = if *desc { o.reverse() } else { o };
        if o != Ordering::Equal {
            return o;
        }
    }
    cmp_values(&a["key"], &b["key"])
}

fn reduce_terms(conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let (size, shard_size) = terms_sizes(conf);
    let min_doc_count = uint(conf.get("min_doc_count")).unwrap_or(1);
    let order = order_spec(conf.get("order"), ("_count", true));
    let show_err = conf.get("show_term_doc_count_error").and_then(Value::as_bool).unwrap_or(false);
    let key_order = is_key_path(&order[0].0);
    let count_desc = order[0].0 == "_count" && order[0].1;
    let mut groups: Vec<(Vec<&Value>, Vec<usize>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut node_err: Vec<i64> = Vec::new();
    let mut other = 0u64;
    for (i, v) in vals.iter().enumerate() {
        let bs = v["buckets"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
        other += v["sum_other_doc_count"].as_u64().unwrap_or(0);
        // ES InternalTerms: a shard that returned fewer than shard_size
        // terms (or is key-ordered) has seen everything; count-descending
        // can have missed at most its last bucket's count; else unknown.
        let trunc: i64 = if vals.len() == 1 || bs.len() < shard_size || key_order {
            0
        } else if count_desc {
            bs.last().map(|b| count_of(b, "doc_count") as i64).unwrap_or(0)
        } else {
            -1
        };
        let own = v["doc_count_error_upper_bound"].as_i64().unwrap_or(0);
        node_err.push(if own < 0 || trunc < 0 { -1 } else { own + trunc });
        for b in bs {
            let g = *index.entry(b["key"].to_string()).or_insert_with(|| {
                groups.push((Vec::new(), Vec::new()));
                groups.len() - 1
            });
            groups[g].0.push(b);
            groups[g].1.push(i);
        }
    }
    let mut buckets: Vec<Value> = Vec::new();
    for (bs, nodes) in &groups {
        let mut m = merge_bucket(bs, subs, typed);
        m.shift_remove("doc_count_error_upper_bound");
        if show_err {
            let mut e = 0i64;
            for (i, ne) in node_err.iter().enumerate() {
                let add = match nodes.iter().position(|n| *n == i) {
                    Some(p) => bs[p]["doc_count_error_upper_bound"].as_i64().unwrap_or(0),
                    None => *ne,
                };
                e = if e < 0 || add < 0 { -1 } else { e + add };
            }
            m.insert("doc_count_error_upper_bound".into(), json!(e));
        }
        if count_of(&Value::Object(m.clone()), "doc_count") >= min_doc_count {
            buckets.push(Value::Object(m));
        }
    }
    buckets.sort_by(|a, b| cmp_buckets(a, b, &order));
    if buckets.len() > size {
        other += buckets[size..].iter().map(|b| count_of(b, "doc_count")).sum::<u64>();
        buckets.truncate(size);
    }
    apply_parents(subs, &mut buckets, typed);
    let err = node_err.iter().fold(0i64, |a, e| if a < 0 || *e < 0 { -1 } else { a + e });
    json!({"doc_count_error_upper_bound": err, "sum_other_doc_count": other, "buckets": buckets})
}

/// The next bucket key of a (date_)histogram, if its interval is understood.
fn stepper(ty: &str, conf: &Value) -> Option<Box<dyn Fn(f64) -> f64>> {
    if ty == "histogram" {
        let iv = num(conf, "interval").filter(|i| *i > 0.0)?;
        return Some(Box::new(move |k| k + iv));
    }
    // Adding fixed UTC milliseconds invents a bucket at the old offset
    // across DST. Shards and coordinator must use identical rounding.
    let next = elasticsearch_engine::aggs::date_histogram_stepper(conf.as_object()?).ok()?;
    Some(Box::new(move |k| next(k as i64) as f64))
}

/// A `key_as_string` renderer that reproduces the nodes' own rendering.
fn key_formatter(conf: &Value, sample: Option<&Value>) -> Option<(edate::DateFormatter, edate::Tz)> {
    let sample = sample?;
    let want = sample.get("key_as_string")?.as_str()?;
    let key = sample["key"].as_i64()?;
    let tz = conf.get("time_zone").and_then(Value::as_str).and_then(|z| edate::Tz::parse(z).ok()).unwrap_or(edate::Tz::UTC);
    let mut first = None;
    for spec in [conf.get("format").and_then(Value::as_str), Some("strict_date_optional_time"), Some("strict_date_optional_time_nanos"), Some("epoch_millis"), Some("epoch_second")].into_iter().flatten() {
        if let Ok(f) = edate::DateFormatter::new(spec) {
            if f.format(key, tz) == want {
                return Some((f, tz));
            }
            first.get_or_insert(f);
        }
    }
    first.map(|f| (f, tz))
}

/// `InternalAutoDateHistogram.reduce`: re-choose the rounding over every
/// shard's buckets (never finer than the coarsest shard's interval), merge
/// the buckets that share a rounded key and fill the gaps.
fn reduce_auto_date_histogram(conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    use elasticsearch_engine::aggs::auto_date;
    let all: Vec<&Value> = vals.iter().flat_map(|v| v["buckets"].as_array().into_iter().flatten()).collect();
    let floor = vals.iter().filter_map(|v| v["interval"].as_str().and_then(auto_date::ladder_index)).max().unwrap_or(0);
    let keys: Vec<i64> = all.iter().filter_map(|b| b["key"].as_i64()).collect();
    let (Some(lo), Some(hi)) = (keys.iter().min().copied(), keys.iter().max().copied()) else {
        let interval = vals.iter().find_map(|v| v["interval"].as_str()).unwrap_or("1s");
        return json!({"buckets": [], "interval": interval});
    };
    let target = uint(conf.get("buckets")).unwrap_or(10).max(1);
    let tz = conf.get("time_zone").and_then(Value::as_str).and_then(|z| edate::Tz::parse(z).ok()).unwrap_or(edate::Tz::UTC);
    let min_unit = auto_date::minimum_unit(conf.get("minimum_interval").and_then(Value::as_str)).unwrap_or(edate::CalUnit::Second);
    let choice = auto_date::choose(lo, hi, target, min_unit, floor, tz);
    let mut groups: std::collections::BTreeMap<i64, Vec<&Value>> = std::collections::BTreeMap::new();
    for b in &all {
        if let Some(k) = b["key"].as_i64() {
            groups.entry(choice.bucket_of(k)).or_default().push(*b);
        }
    }
    let fmt = key_formatter(conf, all.first().copied());
    let last = choice.bucket_of(hi);
    let mut buckets = Vec::new();
    for b in 0..=last.max(0) {
        let bucket_bytes=groups.get(&b).map(|parts|parts.iter().map(|value|elasticsearch_engine::memory::json_bytes(value)).fold(0u64,u64::saturating_add)).unwrap_or_else(||subs.map(elasticsearch_engine::memory::json_bytes).unwrap_or(0).saturating_mul(16));
        if !admit_reduce_growth(bucket_bytes.saturating_mul(2).saturating_add(1024)){return Value::Null;}
        let mut m = match groups.get(&b) {
            Some(bs) => merge_bucket(bs, subs, typed),
            None => {
                let mut m = Map::new();
                m.insert("doc_count".into(), json!(0));
                if let Some(Value::Object(r)) = subs.map(|s| reduce(s, &[], typed)) {
                    m.extend(r);
                }
                m
            }
        };
        let k = choice.key_of(b);
        m.insert("key".into(), json!(k));
        if let Some((f, tz)) = &fmt {
            m.insert("key_as_string".into(), json!(f.format(k, *tz)));
        }
        // Keep `key_as_string`, `key`, `doc_count` first, as the engine renders.
        let mut ordered = Map::new();
        for f in ["key_as_string", "key", "doc_count"] {
            if let Some(v) = m.shift_remove(f) {
                ordered.insert(f.into(), v);
            }
        }
        ordered.extend(m);
        buckets.push(Value::Object(ordered));
    }
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets, "interval": choice.interval()})
}

/// Bucket groups of a histogram reduce, keyed as the shards' keys compare:
/// a numeric key by its `f64` value, anything else by its JSON text. The
/// `f64`'s bits stand for its shortest decimal text (Rust's `Display`
/// round-trips, so distinct finite values print differently, `-0` apart
/// from `0` included): the same groups as keying by `f.to_string()`,
/// without formatting a string for every bucket of every shard.
#[derive(Default)]
struct HistogramKeyIndex {
    numeric: HashMap<u64, usize>,
    other: HashMap<String, usize>,
}

impl HistogramKeyIndex {
    /// The group of `key`, opening a new (empty) one in `groups` if needed.
    fn group<'v>(&mut self, key: &Value, groups: &mut Vec<Vec<&'v Value>>) -> usize {
        let mut open = || {
            groups.push(Vec::new());
            groups.len() - 1
        };
        match key.as_f64() {
            Some(f) => *self.numeric.entry(f.to_bits()).or_insert_with(&mut open),
            None => *self.other.entry(key.to_string()).or_insert_with(open),
        }
    }
}

fn reduce_histogram(ty: &str, conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let keyed = conf.get("keyed").and_then(Value::as_bool).unwrap_or(false) || vals.iter().any(|v| v["buckets"].is_object());
    let min_doc_count = uint(conf.get("min_doc_count")).unwrap_or(0);
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut index = HistogramKeyIndex::default();
    for v in vals {
        let bs: Vec<&Value> = match &v["buckets"] {
            Value::Array(a) => a.iter().collect(),
            Value::Object(m) => m.values().collect(),
            _ => Vec::new(),
        };
        for b in bs {
            let g = index.group(&b["key"], &mut groups);
            groups[g].push(b);
        }
    }
    let mut buckets: Vec<Value> = groups.iter().map(|bs| Value::Object(merge_bucket(bs, subs, typed))).collect();
    buckets.sort_by(|a, b| cmp_values(&a["key"], &b["key"]));
    if min_doc_count == 0 {
        if let (Some(next), true) = (stepper(ty, conf), buckets.len() >= 2) {
            let template = buckets.iter().find(|b| count_of(b, "doc_count") == 0).cloned();
            let fmt = if ty == "date_histogram" { key_formatter(conf, buckets.first()) } else { None };
            let int_keys = buckets[0]["key"].is_i64() || buckets[0]["key"].is_u64();
            let mut filled: Vec<Value> = Vec::with_capacity(buckets.len());
            // ES counts filled empty buckets against `search.max_buckets`.
            let limit = MAX_BUCKETS.with(|m| m.get());
            let mut budget = limit;
            for b in buckets {
                if let Some(prev) = filled.last().and_then(|p| p["key"].as_f64()) {
                    let target = b["key"].as_f64().unwrap_or(prev);
                    let mut k = next(prev);
                    let eps = (k - prev).abs() * 1e-6;
                    while k < target - eps {
                        if budget == 0 {
                            fail_es(elasticsearch_engine::error::EsError::too_many_buckets(limit));
                            break;
                        }
                        let template_bytes=template.as_ref().map(elasticsearch_engine::memory::json_bytes).unwrap_or_else(||subs.map(elasticsearch_engine::memory::json_bytes).unwrap_or(0).saturating_mul(16));
                        if !admit_reduce_growth(template_bytes.saturating_mul(2).saturating_add(1024)){return Value::Null;}
                        #[cfg(test)]
                        FILLED_BUCKETS.with(|count|count.set(count.get()+1));
                        let mut e = match &template {
                            Some(Value::Object(t)) => t.clone(),
                            _ => {
                                let mut m = Map::new();
                                m.insert("doc_count".into(), json!(0));
                                if let Some(Value::Object(r)) = subs.map(|s| reduce(s, &[], typed)) {
                                    m.extend(r);
                                }
                                m
                            }
                        };
                        e.insert("key".into(), if int_keys { json!(k as i64) } else { json!(k) });
                        match &fmt {
                            Some((f, tz)) => {
                                e.insert("key_as_string".into(), json!(f.format(k as i64, *tz)));
                            }
                            None => {
                                e.shift_remove("key_as_string");
                            }
                        }
                        filled.push(Value::Object(e));
                        budget -= 1;
                        let n = next(k);
                        if n <= k {
                            break;
                        }
                        k = n;
                    }
                }
                filled.push(b);
            }
            buckets = filled;
        }
    } else {
        buckets.retain(|b| count_of(b, "doc_count") >= min_doc_count);
    }
    if conf.get("order").is_some() {
        let order = order_spec(conf.get("order"), ("_key", false));
        buckets.sort_by(|a, b| cmp_buckets(a, b, &order));
    }
    apply_parents(subs, &mut buckets, typed);
    let mut out = vals.first().and_then(|v| v.as_object()).cloned().unwrap_or_default();
    if keyed {
        let mut m = Map::new();
        for b in buckets {
            let k = b.get("key_as_string").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| match b["key"].as_f64() {
                Some(f) => format!("{f:?}"),
                None => b["key"].to_string(),
            });
            m.insert(k, b);
        }
        out.insert("buckets".into(), Value::Object(m));
    } else {
        out.insert("buckets".into(), Value::Array(buckets));
    }
    Value::Object(out)
}

/// Aggregations whose bucket list is fixed by the request (`filters`,
/// ranges): merged by position, or by name when keyed.
fn reduce_fixed(subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let Some(first) = vals.first() else { return json!({"buckets": []}) };
    let mut out = first.as_object().cloned().unwrap_or_default();
    let merged = match &first["buckets"] {
        Value::Object(_) => {
            let mut m = Map::new();
            for v in vals {
                for k in v["buckets"].as_object().map(|o| o.keys().collect::<Vec<_>>()).unwrap_or_default() {
                    if !m.contains_key(k) {
                        let bs: Vec<&Value> = vals.iter().filter_map(|v| v["buckets"].get(k)).collect();
                        m.insert(k.clone(), Value::Object(merge_bucket(&bs, subs, typed)));
                    }
                }
            }
            // Pipelines run over a list; the bucket name rides along.
            const TAG: &str = "__fanout_bucket_name";
            let mut list: Vec<Value> = m
                .into_iter()
                .map(|(k, mut b)| {
                    if let Some(o) = b.as_object_mut() {
                        o.insert(TAG.into(), json!(k));
                    }
                    b
                })
                .collect();
            apply_parents(subs, &mut list, typed);
            let mut m = Map::new();
            for mut b in list {
                let k = b.as_object_mut().and_then(|o| o.shift_remove(TAG)).and_then(|k| k.as_str().map(str::to_string)).unwrap_or_default();
                m.insert(k, b);
            }
            Value::Object(m)
        }
        _ => {
            let n = vals.iter().map(|v| v["buckets"].as_array().map(|a| a.len()).unwrap_or(0)).max().unwrap_or(0);
            let mut list: Vec<Value> = (0..n).map(|i| Value::Object(merge_bucket(&vals.iter().filter_map(|v| v["buckets"].get(i)).collect::<Vec<_>>(), subs, typed))).collect();
            apply_parents(subs, &mut list, typed);
            Value::Array(list)
        }
    };
    out.insert("buckets".into(), merged);
    Value::Object(out)
}

/// The engine's bucket-key order: null first, numbers numerically
/// (integers exactly), strings bytewise and after numbers.
fn cmp_key(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::String(x), Value::String(y)) => x.as_bytes().cmp(y.as_bytes()),
        (Value::String(_), _) => Ordering::Greater,
        (_, Value::String(_)) => Ordering::Less,
        (Value::Array(x), Value::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                let o = cmp_key(p, q);
                if o != Ordering::Equal {
                    return o;
                }
            }
            x.len().cmp(&y.len())
        }
        _ => {
            if let (Some(x), Some(y)) = (a.as_i64(), b.as_i64()) {
                return x.cmp(&y);
            }
            let n = |v: &Value| match v {
                Value::Bool(t) => if *t { 1.0 } else { 0.0 },
                _ => v.as_f64().unwrap_or(f64::NAN),
            };
            n(a).total_cmp(&n(b))
        }
    }
}

/// A sub-result of a rendered bucket by plain name (`name` or `type#name`).
fn mget<'a>(m: &'a Map<String, Value>, name: &str) -> Option<&'a Value> {
    m.get(name).or_else(|| m.iter().find(|(k, _)| k.split_once('#').map(|(_, n)| n == name).unwrap_or(false)).map(|(_, v)| v))
}

/// A `buckets_path` value inside one bucket.
fn resolve_path(bucket: &Map<String, Value>, path: &str) -> Option<f64> {
    let mut node = bucket;
    let parts: Vec<&str> = path.split('>').collect();
    for p in &parts[..parts.len() - 1] {
        node = mget(node, p)?.as_object()?;
    }
    let last = parts[parts.len() - 1];
    match last {
        "_count" => return node.get("doc_count").and_then(Value::as_f64),
        "_key" => return node.get("key").and_then(Value::as_f64),
        _ => {}
    }
    let (name, metric) = if let Some(i) = last.find('[') {
        (&last[..i], Some(last[i + 1..].trim_end_matches(']')))
    } else if let Some((n, m)) = last.split_once('.') {
        (n, Some(m))
    } else {
        (last, None)
    };
    let agg = mget(node, name)?.as_object()?;
    match metric {
        None => agg.get("value").and_then(Value::as_f64).or_else(|| agg.get("doc_count").and_then(Value::as_f64)),
        Some("_count") => agg.get("doc_count").and_then(Value::as_f64),
        Some(m) => agg.get(m).and_then(Value::as_f64).or_else(|| {
            let vals = agg.get("values")?;
            let mf: Option<f64> = m.parse().ok();
            match vals {
                Value::Object(o) => o.get(m).or_else(|| mf.and_then(|f| o.get(&jfmt(f)))).and_then(Value::as_f64),
                Value::Array(a) => a.iter().find(|e| e.get("key").and_then(Value::as_f64) == mf).and_then(|e| e.get("value")).and_then(Value::as_f64),
                _ => None,
            }
        }),
    }
}

fn buckets_of(v: &Value) -> Vec<Map<String, Value>> {
    match v.get("buckets") {
        Some(Value::Array(a)) => a.iter().filter_map(|b| b.as_object().cloned()).collect(),
        Some(Value::Object(o)) => o
            .iter()
            .filter_map(|(k, b)| {
                let mut m = b.as_object()?.clone();
                m.entry("key".to_string()).or_insert(json!(k));
                Some(m)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn bucket_key_string(b: &Map<String, Value>) -> String {
    if let Some(s) = b.get("key_as_string").and_then(Value::as_str) {
        return s.to_string();
    }
    match b.get("key") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) if n.is_f64() => jfmt(n.as_f64().unwrap_or(0.0)),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn path_of(body: &Value, t: &str) -> Result<String, String> {
    body.get("buckets_path").and_then(Value::as_str).map(str::to_string).ok_or_else(|| format!("Required parameter [buckets_path] missing for [{t}]"))
}

/// A sibling pipeline over the merged results of its level (engine
/// `sibling_pipeline`).
fn sibling_pipeline(name: &str, t: &str, body: &Value, results: &HashMap<String, Value>) -> Result<Value, String> {
    let path = path_of(body, t)?;
    let (agg, rest) = path.split_once('>').ok_or_else(|| format!("No aggregation found for path [{path}]"))?;
    let target = results.get(agg).ok_or_else(|| format!("No aggregation found for path [{path}]"))?;
    if target.get("buckets").is_none() {
        return Err(format!("buckets_path must reference either a number value or a single value numeric metric aggregation, got: [{agg}] for [{name}]"));
    }
    let insert_zeros = body.get("gap_policy").and_then(Value::as_str) == Some("insert_zeros");
    let mut vals: Vec<(String, f64)> = Vec::new();
    for b in buckets_of(target) {
        match resolve_path(&b, rest) {
            Some(v) if v.is_finite() => vals.push((bucket_key_string(&b), v)),
            _ if insert_zeros => vals.push((bucket_key_string(&b), 0.0)),
            _ => {}
        }
    }
    let n = vals.len();
    let sum: f64 = vals.iter().map(|x| x.1).sum();
    Ok(match t {
        "avg_bucket" => json!({ "value": if n > 0 { fnum(sum / n as f64) } else { Value::Null } }),
        "sum_bucket" => json!({ "value": fnum(sum) }),
        "max_bucket" | "min_bucket" => {
            let best = vals.iter().map(|x| x.1).fold(None, |acc: Option<f64>, v| {
                Some(match acc {
                    None => v,
                    Some(a) if t == "max_bucket" => a.max(v),
                    Some(a) => a.min(v),
                })
            });
            let keys: Vec<String> = match best {
                Some(b) => vals.iter().filter(|x| x.1 == b).map(|x| x.0.clone()).collect(),
                None => vec![],
            };
            json!({ "value": opt_num(best), "keys": keys })
        }
        "stats_bucket" | "extended_stats_bucket" => {
            let min = vals.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
            let max = vals.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            let mut m = json!({
                "count": n,
                "min": if n > 0 { fnum(min) } else { Value::Null },
                "max": if n > 0 { fnum(max) } else { Value::Null },
                "avg": if n > 0 { fnum(sum / n as f64) } else { Value::Null },
                "sum": fnum(sum),
            });
            if t == "extended_stats_bucket" && n > 0 {
                let sos: f64 = vals.iter().map(|x| x.1 * x.1).sum();
                let avg = sum / n as f64;
                let var = (sos / n as f64 - avg * avg).max(0.0);
                m["sum_of_squares"] = fnum(sos);
                m["variance"] = fnum(var);
                m["std_deviation"] = fnum(var.sqrt());
            }
            m
        }
        "percentiles_bucket" => {
            let mut sorted: Vec<f64> = vals.iter().map(|x| x.1).collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let percents: Vec<f64> = body.get("percents").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_else(|| vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]);
            let mut m = Map::new();
            for p in percents {
                let v = if sorted.is_empty() { None } else { Some(sorted[((p / 100.0 * sorted.len() as f64).ceil() as usize).saturating_sub(1).min(sorted.len() - 1)]) };
                m.insert(jfmt(p), opt_num(v));
            }
            json!({ "values": m })
        }
        _ => json!({}),
    })
}

fn script_source(body: &Value) -> Result<String, String> {
    match body.get("script") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Object(o)) => o.get("source").or(o.get("inline")).and_then(Value::as_str).map(str::to_string).ok_or_else(|| "script source is required".to_string()),
        _ => Err("[script] is required".into()),
    }
}

fn script_vars(body: &Value, b: &Map<String, Value>) -> Option<HashMap<String, f64>> {
    let mut vars = HashMap::new();
    match body.get("buckets_path") {
        Some(Value::Object(paths)) => {
            for (k, p) in paths {
                vars.insert(k.clone(), resolve_path(b, p.as_str()?)?);
            }
        }
        Some(Value::String(p)) => {
            vars.insert("_value".into(), resolve_path(b, p)?);
        }
        _ => {}
    }
    if let Some(Value::Object(params)) = body.get("script").and_then(|s| s.get("params")) {
        for (k, v) in params {
            if let Some(f) = v.as_f64() {
                vars.entry(k.clone()).or_insert(f);
            }
        }
    }
    Some(vars)
}

/// Run the parent pipelines declared in `subs` over a merged bucket list
/// (engine `apply_parent_pipelines`), in declaration order.
fn apply_parents(subs: Option<&Value>, buckets: &mut Vec<Value>, typed: bool) {
    let Some(sub) = subs.and_then(Value::as_object) else { return };
    for (name, def) in sub {
        let Some((t, body, _)) = parse(def) else { continue };
        if elasticsearch_engine::aggs::partial::ENGINE_PARENT_PIPELINES.contains(&t) {
            // The engine's implementation over plain-keyed copies.
            let names = spec_names(Some(&Value::Object(sub.clone())));
            let mut plain: Vec<Map<String, Value>> = buckets.iter().map(|b| untyped(b, &names).as_object().cloned().unwrap_or_default()).collect();
            if let Err(e) = elasticsearch_engine::aggs::partial::engine_parent_pipeline(name, t, body, &mut plain) {
                fail_es(e);
                return;
            }
            // The engine's typed prefixes (`search.rs`).
            let prefix = match t {
                "normalize" | "serial_diff" | "moving_fn" => "simple_value",
                "cumulative_cardinality" => "simple_long_value",
                other => other,
            };
            let key = if typed { format!("{prefix}#{name}") } else { name.clone() };
            for (b, p) in buckets.iter_mut().zip(plain) {
                if let (Some(o), Some(v)) = (b.as_object_mut(), p.get(name)) {
                    o.insert(key.clone(), v.clone());
                }
            }
            continue;
        }
        if !PARENT_PIPELINES.contains(&t) {
            continue;
        }
        if let Err(e) = apply_parent(name, t, body, buckets, typed) {
            fail(e);
            return;
        }
    }
}

/// A rendered aggregation with `typed_keys` prefixes (`sterms#name`)
/// removed wherever `name` is one of the request's aggregation `names`.
fn untyped(v: &Value, names: &std::collections::HashSet<String>) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| {
            let plain = match k.split_once('#') {
                Some((_, n)) if names.contains(n) => n.to_string(),
                _ => k.clone(),
            };
            (plain, untyped(v, names))
        }).collect()),
        Value::Array(a) => Value::Array(a.iter().map(|x| untyped(x, names)).collect()),
        other => other.clone(),
    }
}

/// Every aggregation name in a request (sub-)tree.
fn spec_names(spec: Option<&Value>) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut stack: Vec<&Value> = spec.into_iter().collect();
    while let Some(level) = stack.pop() {
        for (name, e) in level.as_object().into_iter().flatten() {
            out.insert(name.clone());
            if let Some(s) = e.get("aggs").or_else(|| e.get("aggregations")) {
                stack.push(s);
            }
        }
    }
    out
}

/// Drop shard-only state ([`PIPELINE_PAYLOAD`], `__p`) from a reduced tree.
fn strip_shard_state(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.shift_remove(elasticsearch_engine::aggs::partial::PIPELINE_PAYLOAD);
            // Raw values the final render writes verbatim (a scripted
            // metric's value, a `top_hits` hit kept as text) are not shard
            // state: kept.
            let raw: Map<String, Value> = m.get("__p").and_then(Value::as_object).map(|p| p.iter().filter(|(k, _)| matches!(k.as_str(), "utf16_metric_value" | "raw_json")).map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
            m.shift_remove("__p");
            let bucket=m.contains_key("doc_count");
            for(key,value)in m.iter_mut(){if !matches!(key.as_str(),"meta"|"key")&&(key!="value"||bucket){strip_shard_state(value);}}
            if !raw.is_empty(){m.insert("__p".into(),Value::Object(raw));}
        }
        Value::Array(a) => a.iter_mut().for_each(strip_shard_state),
        _ => {}
    }
}

/// `variable_width_histogram` (ES `InternalVariableWidthHistogram.reduce`):
/// every node's buckets by centroid, the closest neighbours merged until
/// the requested number of buckets remains.
fn reduce_variable_width(conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let target = uint(conf.get("buckets")).unwrap_or(10).max(1) as usize;
    let mut groups: Vec<(f64, Vec<&Value>)> = vals.iter().flat_map(|v| buckets_of_ref(v)).map(|b| (num(b, "key").unwrap_or(0.0), vec![b])).collect();
    groups.sort_by(|a, b| a.0.total_cmp(&b.0));
    let weight = |g: &[&Value]| g.iter().map(|b| count_of(b, "doc_count")).sum::<u64>() as f64;
    while groups.len() > target {
        let i = (0..groups.len() - 1).min_by(|&a, &b| (groups[a + 1].0 - groups[a].0).total_cmp(&(groups[b + 1].0 - groups[b].0))).unwrap();
        let (k2, g2) = groups.remove(i + 1);
        let (w1, w2) = (weight(&groups[i].1), weight(&g2));
        groups[i].0 = if w1 + w2 > 0.0 { (groups[i].0 * w1 + k2 * w2) / (w1 + w2) } else { groups[i].0 };
        groups[i].1.extend(g2);
    }
    let mut buckets: Vec<Value> = groups.into_iter().map(|(key, g)| {
        let mut m = merge_bucket(&g, subs, typed);
        m.insert("min".into(), fnum(g.iter().filter_map(|b| num(b, "min")).fold(f64::INFINITY, f64::min)));
        m.insert("key".into(), fnum(key));
        m.insert("max".into(), fnum(g.iter().filter_map(|b| num(b, "max")).fold(f64::NEG_INFINITY, f64::max)));
        Value::Object(m)
    }).collect();
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets})
}

/// `categorize_text`: the shards' categories re-categorized (ES
/// `InternalCategorizationAggregation.reduce`), sub-aggregations reduced
/// over each final category's member buckets.
fn reduce_categorize(name: &str, conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let nodes: Vec<Vec<&Value>> = vals.iter().map(|v| buckets_of_ref(v).collect()).collect();
    let list = match elasticsearch_engine::aggs::partial::categorize_reduce(name, conf.as_object().unwrap_or(&Map::new()), &nodes) {
        Ok(Some(l)) => l,
        Ok(None) => {
            fail_with(500, "illegal_state_exception", format!("[categorize_text] aggregation [{name}] needs the categories' token state from every shard (a node predates it)"));
            return json!({});
        }
        Err(e) => {
            fail_es(e);
            return json!({});
        }
    };
    let mut buckets: Vec<Value> = list.into_iter().map(|(mut m, members)| {
        let bs: Vec<&Value> = members.iter().map(|(n, b)| nodes[*n][*b]).collect();
        for (k, v) in merge_bucket(&bs, subs, typed) {
            if !matches!(k.as_str(), "doc_count" | "key" | "regex" | "max_matching_length" | "__p") {
                m.insert(k, v);
            }
        }
        Value::Object(m)
    }).collect();
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets})
}

fn buckets_of_ref(v: &Value) -> impl Iterator<Item = &Value> {
    v["buckets"].as_array().into_iter().flatten()
}

fn apply_parent(name: &str, t: &str, body: &Value, buckets: &mut Vec<Value>, typed: bool) -> Result<(), String> {
    let key = if typed { format!("{t}#{name}") } else { name.to_string() };
    let empty = Map::new();
    let obj = |b: &Value| b.as_object().cloned().unwrap_or_default();
    let set = |b: &mut Value, v: Value| {
        if let Some(o) = b.as_object_mut() {
            o.insert(key.clone(), v);
        }
    };
    match t {
        "cumulative_sum" => {
            let path = path_of(body, t)?;
            let mut acc = 0.0;
            for b in buckets.iter_mut() {
                acc += resolve_path(b.as_object().unwrap_or(&empty), &path).filter(|v| v.is_finite()).unwrap_or(0.0);
                set(b, json!({ "value": fnum(acc) }));
            }
        }
        "derivative" => {
            let path = path_of(body, t)?;
            let mut prev: Option<f64> = None;
            for b in buckets.iter_mut() {
                let cur = resolve_path(b.as_object().unwrap_or(&empty), &path).filter(|v| v.is_finite());
                if let (Some(p), Some(c)) = (prev, cur) {
                    set(b, json!({ "value": fnum(c - p) }));
                }
                if cur.is_some() {
                    prev = cur;
                }
            }
        }
        "bucket_script" => {
            let script=body.get("script").ok_or_else(||"[script] is required".to_string())?;
            for b in buckets.iter_mut() {
                if let Some(vars) = script_vars(body, &obj(b)) {
                    let v = match eexpr::eval_search_script(script, &vars, ScriptContext::BucketScript) {
                        Ok(v) => v,
                        Err(e) => {
                            fail_es(e);
                            return Ok(());
                        }
                    };
                    if v.is_null() {
                        continue;
                    }
                    let x = v.as_f64().ok_or_else(|| format!("buckets_path must reference either a number value or a single value numeric metric aggregation, but [{name}] returned a {}", v.type_name()))?;
                    set(b, json!({ "value": fnum(x) }));
                }
            }
        }
        "bucket_selector" => {
            let script=body.get("script").ok_or_else(||"[script] is required".to_string())?;
            let mut keep = Vec::new();
            for b in buckets.drain(..) {
                match script_vars(body, &obj(&b)) {
                    Some(vars) => {
                        let v = match eexpr::eval_search_script(script, &vars, ScriptContext::BucketSelector) {
                            Ok(v) => v,
                            Err(e) => {
                                fail_es(e);
                                return Ok(());
                            }
                        };
                        match v {
                            Val::Bool(true) => keep.push(b),
                            Val::Bool(false) => {}
                            other => return Err(format!("Script for [{name}] must return a boolean, got [{}]", other.type_name())),
                        }
                    }
                    None => keep.push(b),
                }
            }
            *buckets = keep;
        }
        "bucket_sort" => {
            let mut crit: Vec<(String, bool)> = Vec::new();
            if let Some(Value::Array(a)) = body.get("sort") {
                for s in a {
                    match s {
                        Value::String(p) => crit.push((p.clone(), true)),
                        Value::Object(m) => {
                            for (p, d) in m {
                                let asc = match d {
                                    Value::String(x) => x != "desc",
                                    Value::Object(x) => x.get("order").and_then(Value::as_str) != Some("desc"),
                                    _ => true,
                                };
                                crit.push((p.clone(), asc));
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !crit.is_empty() {
                buckets.sort_by(|a, b| {
                    let (a, b) = (a.as_object().unwrap_or(&empty), b.as_object().unwrap_or(&empty));
                    for (p, asc) in &crit {
                        let c = match (resolve_path(a, p), resolve_path(b, p)) {
                            (Some(x), Some(y)) if *asc => x.total_cmp(&y),
                            (Some(x), Some(y)) => y.total_cmp(&x),
                            (None, Some(_)) => Ordering::Greater,
                            (Some(_), None) => Ordering::Less,
                            _ => Ordering::Equal,
                        };
                        if c != Ordering::Equal {
                            return c;
                        }
                    }
                    Ordering::Equal
                });
            }
            let from = body.get("from").and_then(Value::as_u64).unwrap_or(0) as usize;
            let size = body.get("size").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(usize::MAX);
            *buckets = buckets.drain(..).skip(from).take(size).collect();
        }
        _ => {}
    }
    Ok(())
}

/// Bucket groups by key across nodes, in first-seen order.
fn group_buckets<'a>(vals: &[&'a Value]) -> Vec<Vec<&'a Value>> {
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for v in vals {
        for b in v["buckets"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            let g = *index.entry(b["key"].to_string()).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(b);
        }
    }
    groups
}

/// `rare_terms`: a key is rare when its count summed over every node (from
/// the helper `terms`) is at most `max_doc_count`.
/// `rare_terms`: every shard's rare candidates, minus the terms some shard
/// holds more than `max_doc_count` times (its Bloom filter, as ES's cuckoo
/// filter: a false positive can drop a rare term, never keep a frequent one).
fn reduce_rare(name: &str, conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let max = uint(conf.get("max_doc_count")).unwrap_or(1);
    let mut frequent = Vec::new();
    for v in vals {
        match v.get("__p") {
            Some(st) => frequent.extend(st.get("frequent").and_then(elasticsearch_engine::aggs::sketch::KeyBloom::from_state)),
            // A shard without the field returns no buckets and no state.
            None if v["buckets"].as_array().is_none_or(|b| b.is_empty()) => {}
            None if vals.len() > 1 => {
                fail_with(500, "illegal_state_exception", format!("[rare_terms] aggregation [{name}] needs the frequent-terms state from every shard (a node predates it)"));
                return json!({});
            }
            None => {}
        }
    }
    let mut buckets: Vec<Value> = Vec::new();
    for bs in group_buckets(vals) {
        let k = bs[0]["key"].to_string();
        if frequent.iter().any(|f| f.contains(&k)) {
            continue;
        }
        let m = merge_bucket(&bs, subs, typed);
        if count_of(&Value::Object(m.clone()), "doc_count") <= max {
            buckets.push(Value::Object(m));
        }
    }
    buckets.sort_by(|a, b| count_of(a, "doc_count").cmp(&count_of(b, "doc_count")).then_with(|| cmp_key(&a["key"], &b["key"])));
    apply_parents(subs, &mut buckets, typed);
    json!({"buckets": buckets})
}

/// `significant_terms` score of one term (engine heuristics).
fn sig_score(h: &str, p: &Map<String, Value>, sub_f: f64, sub_n: f64, sup_f: f64, sup_n: f64) -> f64 {
    let flag = |k: &str, d: bool| p.get(k).and_then(Value::as_bool).unwrap_or(d);
    match h {
        "percentage" => {
            if sup_f == 0.0 {
                0.0
            } else {
                sub_f / sup_f
            }
        }
        "gnd" => {
            let (fx, fy, fxy) = (sup_f, sub_n, sub_f);
            if fx == 0.0 || fy == 0.0 || fxy == 0.0 {
                return 0.0;
            }
            if fx == fy && fy == fxy {
                return 1.0;
            }
            let num = fx.ln().max(fy.ln()) - fxy.ln();
            let den = sup_n.ln() - fx.ln().min(fy.ln());
            (-(num / den)).exp()
        }
        "mutual_information" | "chi_square" => {
            let superset = flag("background_is_superset", true);
            let (n11, n10, n01, n00) = if superset {
                (sub_f, sup_f - sub_f, sub_n - sub_f, (sup_n - sub_n) - (sup_f - sub_f))
            } else {
                (sub_f, sup_f, sub_n - sub_f, sup_n - sup_f)
            };
            let n = n11 + n10 + n01 + n00;
            let (n1_, n0_, n_1, n_0) = (n10 + n11, n00 + n01, n01 + n11, n10 + n00);
            if !flag("include_negatives", false) && n_1 > 0.0 && n_0 > 0.0 && n11 / n_1 < n10 / n_0 {
                return f64::NEG_INFINITY;
            }
            if h == "chi_square" {
                let den = n_1 * n1_ * n0_ * n_0;
                if den == 0.0 {
                    return 0.0;
                }
                let d = n11 * n00 - n10 * n01;
                return n * d * d / den;
            }
            let term = |nxy: f64, nx: f64, ny: f64| if nxy == 0.0 || nx == 0.0 || ny == 0.0 || n == 0.0 { 0.0 } else { nxy / n * (n * nxy / (nx * ny)).log2() };
            term(n11, n1_, n_1) + term(n01, n0_, n_1) + term(n10, n1_, n_0) + term(n00, n0_, n_0)
        }
        _ => {
            if sub_f == 0.0 || sub_n == 0.0 || sup_n == 0.0 || sup_f == 0.0 {
                return 0.0;
            }
            let sp = sub_f / sub_n;
            let bp = sup_f / sup_n;
            let abs = sp - bp;
            if abs <= 0.0 {
                0.0
            } else {
                abs * (sp / bp)
            }
        }
    }
}

/// `significant_terms`: candidate counts summed over nodes, background
/// counts from the root-level helper, scored once on the merged counts.
fn reduce_sig(name: &str, conf: &Value, subs: Option<&Value>, vals: &[&Value], helpers: &[Option<&Value>], typed: bool) -> Value {
    let (h, params) = heuristic(conf);
    if h == "script_heuristic" {
        fail(format!("[script_heuristic] is not supported by this server: [{name}]"));
        return json!({});
    }
    // Exact background counts when the root-level helper holds every term
    // of every node; otherwise ES's: the sum over nodes that returned it.
    let bg_entry = SIG_BG.with(|s| s.borrow().get(name).cloned());
    let bg_exact = bg_entry.as_ref().is_some_and(|b| b.1);
    let bg = bg_entry.filter(|b| b.1).map(|b| b.0).unwrap_or_default();
    // Every node's subset counts, when each helper holds all its terms: with
    // exact background counts too, scoring matches one shard exactly.
    let exact_sub: Option<Vec<(String, u64, &Value)>> = (bg_exact && !helpers.is_empty() && helpers.iter().all(|h| h.is_some_and(|h| count_of(h, "sum_other_doc_count") == 0 && h["buckets"].as_array().is_some_and(|b| (b.len() as u64) < ALL_LIMIT)))).then(|| {
        let mut order: Vec<(String, u64, &Value)> = Vec::new();
        let mut at: HashMap<String, usize> = HashMap::new();
        for h in helpers.iter().flatten() {
            for b in h["buckets"].as_array().into_iter().flatten() {
                let k = b["key"].to_string();
                let i = *at.entry(k.clone()).or_insert_with(|| { order.push((k, 0, b)); order.len() - 1 });
                order[i].1 += count_of(b, "doc_count");
            }
        }
        order
    });
    let sub_counts: HashMap<String, u64> = exact_sub.iter().flatten().map(|(k, c, _)| (k.clone(), *c)).collect();
    let subset: u64 = vals.iter().map(|v| count_of(v, "doc_count")).sum();
    let superset: u64 = vals.iter().map(|v| count_of(v, "bg_count")).sum();
    let is_superset = params.get("background_is_superset").and_then(Value::as_bool).unwrap_or(true);
    let min_doc_count = uint(conf.get("min_doc_count")).unwrap_or(3);
    let size = uint(conf.get("size")).map(|s| s as usize).unwrap_or(10);
    let mut scored: Vec<(f64, Value)> = Vec::new();
    let mut groups = group_buckets(vals);
    // Without sub-aggregations every term is a candidate (helper buckets
    // carry key and doc_count only).
    if subs.is_none() {
        let seen: std::collections::HashSet<String> = groups.iter().map(|bs| bs[0]["key"].to_string()).collect();
        groups.extend(exact_sub.iter().flatten().filter(|(k, _, _)| !seen.contains(k)).map(|(_, _, b)| vec![*b]));
    }
    for bs in groups {
        let mut merged = merge_bucket(&bs, subs, typed);
        let k = bs[0]["key"].to_string();
        if let Some(c) = sub_counts.get(&k) {
            merged.insert("doc_count".into(), json!(c));
        }
        let c = count_of(&Value::Object(merged.clone()), "doc_count");
        let node_bg: u64 = bs.iter().map(|b| count_of(b, "bg_count")).sum();
        let b = bg.get(&k).copied().unwrap_or(node_bg).max(if is_superset { c } else { 0 });
        let score = sig_score(h, &params, c as f64, subset as f64, b as f64, superset as f64);
        if !(score > 0.0) || c < min_doc_count {
            continue;
        }
        let mut out = Map::new();
        out.insert("key".into(), merged.get("key").cloned().unwrap_or(Value::Null));
        if let Some(s) = merged.get("key_as_string") {
            out.insert("key_as_string".into(), s.clone());
        }
        out.insert("doc_count".into(), json!(c));
        out.insert("score".into(), fnum(score));
        out.insert("bg_count".into(), json!(b));
        for (k, v) in merged {
            if !matches!(k.as_str(), "key" | "key_as_string" | "doc_count" | "score" | "bg_count") {
                out.insert(k, v);
            }
        }
        scored.push((score, Value::Object(out)));
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| cmp_key(&a.1["key"], &b.1["key"])));
    scored.truncate(size);
    let mut buckets: Vec<Value> = scored.into_iter().map(|s| s.1).collect();
    apply_parents(subs, &mut buckets, typed);
    json!({"doc_count": subset, "bg_count": superset, "buckets": buckets})
}

fn cmp_multi(a: &Value, b: &Value, order: &[(String, bool)]) -> Ordering {
    let empty = Map::new();
    for (p, desc) in order {
        let o = if p == "_count" {
            count_of(a, "doc_count").cmp(&count_of(b, "doc_count"))
        } else if is_key_path(p) {
            cmp_key(&a["key"], &b["key"])
        } else {
            match (resolve_path(a.as_object().unwrap_or(&empty), p), resolve_path(b.as_object().unwrap_or(&empty), p)) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (Some(_), None) => return Ordering::Less,
                (None, Some(_)) => return Ordering::Greater,
                _ => Ordering::Equal,
            }
        };
        let o = if *desc { o.reverse() } else { o };
        if o != Ordering::Equal {
            return o;
        }
    }
    cmp_key(&a["key"], &b["key"])
}

fn reduce_multi_terms(conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let (size, shard_size) = terms_sizes(conf);
    let min_doc_count = uint(conf.get("min_doc_count")).unwrap_or(1);
    let order = order_spec(conf.get("order"), ("_count", true));
    let show_err = conf.get("show_term_doc_count_error").and_then(Value::as_bool).unwrap_or(false);
    let mut other: u64 = vals.iter().map(|v| count_of(v, "sum_other_doc_count")).sum();
    // As for `terms` (ES InternalMultiTerms shares InternalTerms' reduce):
    // a node that returned `shard_size` buckets ordered by descending count
    // can have missed up to its last count; any other order, an unknown amount.
    let node_err: Vec<i64> = vals.iter().map(|v| {
        let bs = v["buckets"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
        let trunc = if bs.len() < shard_size || is_key_path(&order[0].0) {
            0
        } else if order[0].0 == "_count" && order[0].1 {
            bs.last().map(|b| count_of(b, "doc_count") as i64).unwrap_or(0)
        } else {
            -1
        };
        let own = v["doc_count_error_upper_bound"].as_i64().unwrap_or(0);
        if own < 0 || trunc < 0 { -1 } else { own + trunc }
    }).collect();
    let add_err = |a: i64, e: i64| if a < 0 || e < 0 { -1 } else { a + e };
    let mut buckets: Vec<Value> = group_buckets(vals)
        .iter()
        .map(|bs| {
            let mut m = merge_bucket(bs, subs, typed);
            m.shift_remove("doc_count_error_upper_bound");
            if show_err {
                // Nodes that returned the key report its own error; the others
                // could have missed it entirely.
                let key = bs[0]["key"].to_string();
                let e = vals.iter().zip(&node_err).fold(0i64, |acc, (v, ne)| {
                    let own = v["buckets"].as_array().and_then(|a| a.iter().find(|b| b["key"].to_string() == key));
                    add_err(acc, own.map(|b| b["doc_count_error_upper_bound"].as_i64().unwrap_or(0)).unwrap_or(*ne))
                });
                m.insert("doc_count_error_upper_bound".into(), json!(e));
            }
            Value::Object(m)
        })
        .filter(|b| count_of(b, "doc_count") >= min_doc_count)
        .collect();
    buckets.sort_by(|a, b| cmp_multi(a, b, &order));
    if buckets.len() > size {
        other += buckets[size..].iter().map(|b| count_of(b, "doc_count")).sum::<u64>();
        buckets.truncate(size);
    }
    apply_parents(subs, &mut buckets, typed);
    let err = node_err.iter().fold(0i64, |a, e| add_err(a, *e));
    json!({"doc_count_error_upper_bound": err, "sum_other_doc_count": other, "buckets": buckets})
}

/// `composite`: nodes already skipped past `after`, so the page is the
/// first `size` keys of the union in source order.
fn reduce_composite(conf: &Value, subs: Option<&Value>, vals: &[&Value], typed: bool) -> Value {
    let size = uint(conf.get("size")).map(|s| s as usize).unwrap_or(10);
    // (name, descending, nulls first): ES `MissingOrder` puts missing values
    // first ascending and last descending unless `missing_order` says.
    let mut sources: Vec<(String, bool, bool)> = Vec::new();
    for s in conf.get("sources").and_then(Value::as_array).map(|a| a.as_slice()).unwrap_or(&[]) {
        if let Some((name, def)) = s.as_object().and_then(|o| o.iter().next()) {
            let c = def.as_object().and_then(|d| d.values().next());
            let desc = c.and_then(|c| c.get("order")).and_then(Value::as_str) == Some("desc");
            let nulls_first = match c.and_then(|c| c.get("missing_order")).and_then(Value::as_str) {
                Some("first") => true,
                Some("last") => false,
                _ => !desc,
            };
            sources.push((name.clone(), desc, nulls_first));
        }
    }
    let mut buckets: Vec<Value> = group_buckets(vals).iter().map(|bs| Value::Object(merge_bucket(bs, subs, typed))).collect();
    buckets.sort_by(|a, b| {
        for (n, desc, nulls_first) in &sources {
            let (x, y) = (&a["key"][n.as_str()], &b["key"][n.as_str()]);
            let c = match (x.is_null(), y.is_null()) {
                (true, true) => Ordering::Equal,
                (true, false) => if *nulls_first { Ordering::Less } else { Ordering::Greater },
                (false, true) => if *nulls_first { Ordering::Greater } else { Ordering::Less },
                _ if *desc => cmp_key(x, y).reverse(),
                _ => cmp_key(x, y),
            };
            if c != Ordering::Equal {
                return c;
            }
        }
        Ordering::Equal
    });
    buckets.truncate(size);
    apply_parents(subs, &mut buckets, typed);
    let mut out = Map::new();
    if let Some(last) = buckets.last() {
        out.insert("after_key".into(), last["key"].clone());
    }
    out.insert("buckets".into(), Value::Array(buckets));
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn auto_date_histogram_rechooses_rounding_across_shards() {
        const DAY: i64 = 86_400_000;
        let lo = 1_672_531_200_000i64; // 2023-01-01
        // Shard A has days 0 and 1, shard B has day 9: ten daily buckets fit.
        let a = json!({"interval": "1d", "buckets": [{"key": lo, "doc_count": 2}, {"key": lo + DAY, "doc_count": 1}]});
        let b = json!({"interval": "1d", "buckets": [{"key": lo + 9 * DAY, "doc_count": 4}]});
        let r = reduce_auto_date_histogram(&json!({"field": "d", "buckets": 10}), None, &[&a, &b], false);
        assert_eq!(r["interval"], "1d");
        let bs = r["buckets"].as_array().unwrap();
        assert_eq!(bs.len(), 10);
        assert_eq!((bs[0]["doc_count"].clone(), bs[1]["doc_count"].clone(), bs[5]["doc_count"].clone(), bs[9]["doc_count"].clone()), (json!(2), json!(1), json!(0), json!(4)));
        // Only 2 buckets allowed: weekly, counts merged.
        let r = reduce_auto_date_histogram(&json!({"field": "d", "buckets": 2}), None, &[&a, &b], false);
        assert_eq!(r["interval"], "7d");
        let total: u64 = r["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).sum();
        assert_eq!(total, 7);
        // A shard already at 7d keeps the result at 7d.
        let c = json!({"interval": "7d", "buckets": [{"key": lo - 6 * DAY, "doc_count": 1}]});
        assert_eq!(reduce_auto_date_histogram(&json!({"field": "d"}), None, &[&a, &c], false)["interval"], "7d");
        assert_eq!(unsupported(&json!({"h": {"auto_date_histogram": {"field": "d"}}})), None);
    }

    use super::*;

    /// Keying histogram groups by the key's `f64` bits forms exactly the
    /// groups the former `f.to_string()` keys formed (integer and fractional
    /// keys, `-0.0`/`0.0`, keys equal only after `f64` conversion, non-numeric
    /// keys), and the reduce of a monthly date_histogram is unchanged.
    #[test]
    fn histogram_key_index_groups_like_decimal_text() {
        let keys = [json!(1), json!(1.0), json!(-0.0), json!(0), json!(0.1), json!(0.1 + 0.2), json!(0.30000000000000004), json!(9007199254740993i64), json!(9007199254740992i64), json!(1_420_070_400_000i64), json!(null), json!("a"), json!(1.5e300)];
        let mut by_bits = HistogramKeyIndex::default();
        let mut groups: Vec<Vec<&Value>> = Vec::new();
        let mut by_text: HashMap<String, usize> = HashMap::new();
        let mut text_groups = 0usize;
        for k in &keys {
            let g = by_bits.group(k, &mut groups);
            let t = k.as_f64().map(|f| f.to_string()).unwrap_or_else(|| k.to_string());
            let tg = *by_text.entry(t).or_insert_with(|| { text_groups += 1; text_groups - 1 });
            assert_eq!(g, tg, "{k}");
        }
        assert_eq!(groups.len(), text_groups);
        let spec = json!({"d": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}});
        let month = |m: i64| 1_420_070_400_000i64 + m * 31 * 86_400_000;
        let part = |ms: &[(i64, u64)]| json!({"d": {"buckets": ms.iter().map(|(m, c)| json!({"key": month(*m), "key_as_string": format!("k{m}"), "doc_count": c})).collect::<Vec<_>>()}});
        let a = part(&[(0, 3), (1, 1)]);
        let b = part(&[(1, 2), (0, 4)]);
        let r = reduce_root(&spec, &[&a, &b], false);
        let counts: Vec<u64> = r["d"]["buckets"].as_array().unwrap().iter().map(|b| b["doc_count"].as_u64().unwrap()).collect();
        assert_eq!(counts[0], 7);
        assert_eq!(*counts.last().unwrap(), 3);
    }

    /// Timing of the coordinator reduce of `articles_monthly_agg`: 5 shard
    /// partials of `PMC_MONTHS` (default 120) monthly buckets.
    /// Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn monthly_histogram_reduce_timing() {
        let months: i64 = std::env::var("PMC_MONTHS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let spec = json!({"articles_over_time": {"date_histogram": {"field": "timestamp", "calendar_interval": "month"}}});
        let rounding = elasticsearch_engine::aggs::date_histogram_stepper(spec["articles_over_time"]["date_histogram"].as_object().unwrap()).unwrap();
        let fmt = edate::DateFormatter::new("yyyy-MM-dd HH:mm:ss").unwrap();
        let mut keys = vec![1_199_145_600_000i64];
        for _ in 1..months { let k = rounding(*keys.last().unwrap()); keys.push(k); }
        let parts: Vec<Value> = (0..5).map(|s| json!({"articles_over_time": {"buckets": keys.iter().map(|k| json!({"key_as_string": fmt.format(*k, edate::Tz::UTC), "key": k, "doc_count": 900 + s})).collect::<Vec<_>>()}})).collect();
        let refs: Vec<&Value> = parts.iter().collect();
        let mut t: Vec<_> = (0..51).map(|_| { let s = std::time::Instant::now(); let r = reduce_root(&spec, &refs, false); assert_eq!(r["articles_over_time"]["buckets"].as_array().unwrap().len(), months as usize); s.elapsed() }).collect();
        t.sort();
        eprintln!("reduce of 5 x {months} monthly buckets: median {:?}", t[25]);
    }


    #[test]
    fn significant_text_fanout_preserves_shard_parameters_and_reduces_raw_counts() {
        let spec = json!({"sample": {"sampler": {"shard_size": 100}, "aggs": {
            "sig": {"significant_text": {"field": "text", "size": 1, "shard_size": 20, "min_doc_count": 3, "p_value": {}}}
        }}});
        assert_eq!(unsupported(&spec), None);
        // No global-vocabulary helper or percentage-heuristic replacement.
        assert_eq!(rewrite_root(&spec), spec);
        let a = json!({"sample": {"doc_count": 2, "sig": {"doc_count": 2, "bg_count": 10, "buckets": [
            {"key": "rare", "doc_count": 2, "bg_count": 2, "score": 0.0}]}}});
        let b = json!({"sample": {"doc_count": 2, "sig": {"doc_count": 2, "bg_count": 10, "buckets": [
            {"key": "rare", "doc_count": 2, "bg_count": 2, "score": 0.0}]}}});
        let out = red(spec, vec![a, b]);
        assert_eq!(out["sample"]["doc_count"], 4);
        assert_eq!(out["sample"]["sig"]["doc_count"], 4);
        assert_eq!(out["sample"]["sig"]["bg_count"], 20);
        assert_eq!(out["sample"]["sig"]["buckets"][0]["key"], "rare");
        assert_eq!(out["sample"]["sig"]["buckets"][0]["doc_count"], 4);
        assert_eq!(out["sample"]["sig"]["buckets"][0]["bg_count"], 4);
        assert!(out["sample"]["sig"]["buckets"][0]["score"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn significant_text_same_name_in_different_parent_buckets_is_isolated() {
        let spec = json!({"a": {"filter": {"match_all": {}}, "aggs": {"sig": {"significant_text": {"field": "text", "min_doc_count": 1}}}},
            "b": {"filter": {"match_all": {}}, "aggs": {"sig": {"significant_text": {"field": "text", "min_doc_count": 1}}}}});
        let sig = |key: &str| json!({"doc_count": 1, "bg_count": 10, "buckets": [{"key": key, "doc_count": 1, "bg_count": 1, "score": 9.0}]});
        let part = json!({"a": {"doc_count": 1, "sig": sig("alpha")}, "b": {"doc_count": 1, "sig": sig("beta")}});
        let out = red(spec, vec![part.clone(), part]);
        assert_eq!(out["a"]["sig"]["buckets"][0]["key"], "alpha");
        assert_eq!(out["b"]["sig"]["buckets"][0]["key"], "beta");
        assert_eq!(out["a"]["sig"]["buckets"][0]["bg_count"], 2);
    }

    #[test]
    fn significant_text_coordinator_uses_configured_limit_for_the_whole_tree() {
        let spec = json!({"a": {"significant_text": {"field": "text", "size": 1, "min_doc_count": 1, "percentage": {}}},
            "b": {"significant_text": {"field": "text", "size": 1, "min_doc_count": 1, "percentage": {}}}});
        let sig = json!({"doc_count": 2, "bg_count": 10, "buckets": [
            {"key": "alpha", "doc_count": 1, "bg_count": 1, "score": 1.0},
            {"key": "beta", "doc_count": 1, "bg_count": 1, "score": 1.0}]});
        let part = json!({"a": sig, "b": sig});
        let result = reduce_root_with_limit(&spec, &[&part], false, 2);
        assert_eq!(result["a"]["buckets"].as_array().unwrap().len(), 1);
        assert!(take_error().is_none());
        reduce_root_with_limit(&spec, &[&part], false, 1);
        assert_eq!(take_error().unwrap().1, "too_many_buckets_exception");
    }

    fn red(spec: Value, parts: Vec<Value>) -> Value {
        let refs: Vec<&Value> = parts.iter().collect();
        reduce(&spec, &refs, false)
    }

    #[test]
    fn metric_reducers_combine_node_results() {
        // rate: node rates over the same unit add up.
        let r = red(json!({"r": {"rate": {"field": "v", "unit": "hour"}}}), vec![json!({"r": {"value": 2.0}}), json!({"r": {"value": 3.5}})]);
        assert_eq!(r["r"]["value"], json!(5.5));
        // weighted_avg: averages weighted by each node's summed weights.
        let spec = json!({"w": {"weighted_avg": {"value": {"field": "v"}, "weight": {"field": "w"}}}});
        let rw = rewrite(&spec);
        assert_eq!(rw["__fanout_wavg__w"]["filter"], json!({"exists": {"field": "v"}}));
        let r = red(spec, vec![
            json!({"w": {"value": 2.0}, "__fanout_wavg__w": {"doc_count": 1, "w": {"value": 1.0}}}),
            json!({"w": {"value": 5.0}, "__fanout_wavg__w": {"doc_count": 2, "w": {"value": 2.0}}}),
        ]);
        assert_eq!(r["w"]["value"], json!(4.0));
        // geo_centroid: weighted by count.
        let r = red(json!({"c": {"geo_centroid": {"field": "p"}}}), vec![
            json!({"c": {"location": {"lat": 10.0, "lon": 0.0}, "count": 1}}),
            json!({"c": {"location": {"lat": 40.0, "lon": 30.0}, "count": 2}}),
            json!({"c": {"count": 0}}),
        ]);
        assert_eq!(r["c"], json!({"location": {"lat": 30.0, "lon": 20.0}, "count": 3}));
        // geo_bounds: the covering box; empty nodes render no bounds.
        let r = red(json!({"b": {"geo_bounds": {"field": "p"}}}), vec![
            json!({"b": {"bounds": {"top_left": {"lat": 10.0, "lon": -5.0}, "bottom_right": {"lat": 0.0, "lon": 5.0}}}}),
            json!({"b": {}}),
            json!({"b": {"bounds": {"top_left": {"lat": 20.0, "lon": 1.0}, "bottom_right": {"lat": 5.0, "lon": 8.0}}}}),
        ]);
        assert_eq!(r["b"]["bounds"], json!({"top_left": {"lat": 20.0, "lon": -5.0}, "bottom_right": {"lat": 0.0, "lon": 8.0}}));
        // top_metrics: merged by sort value, cut to size.
        let r = red(json!({"t": {"top_metrics": {"metrics": {"field": "m"}, "sort": {"s": "desc"}, "size": 2}}}), vec![
            json!({"t": {"top": [{"sort": [5], "metrics": {"m": 1}}, {"sort": [1], "metrics": {"m": 2}}]}}),
            json!({"t": {"top": [{"sort": [3], "metrics": {"m": 3}}]}}),
        ]);
        assert_eq!(r["t"]["top"], json!([{"sort": [5], "metrics": {"m": 1}}, {"sort": [3], "metrics": {"m": 3}}]));
        // string_stats: lengths by count, entropy from the merged distribution.
        let spec = json!({"s": {"string_stats": {"field": "k"}}});
        assert_eq!(rewrite(&spec)["s"]["string_stats"]["show_distribution"], json!(true));
        let r = red(spec, vec![
            json!({"s": {"count": 1, "min_length": 2, "max_length": 2, "avg_length": 2.0, "entropy": 0.0, "distribution": {"a": 1.0}}}),
            json!({"s": {"count": 1, "min_length": 2, "max_length": 2, "avg_length": 2.0, "entropy": 0.0, "distribution": {"b": 1.0}}}),
        ]);
        assert_eq!(r["s"]["entropy"], json!(1.0));
        assert_eq!(r["s"]["count"], json!(2));
        assert!(r["s"].get("distribution").is_none());
    }

    #[test]
    fn scripted_metric_reduces_every_nodes_combine_states() {
        let spec = json!({"s": {"scripted_metric": {
            "init_script": "state.n = 0", "map_script": "state.n += 1", "combine_script": "return state.n",
            "reduce_script": "double t = 0; for (s in states) { t += s } return t"}}});
        assert_eq!(rewrite(&spec)["s"]["scripted_metric"]["reduce_script"]["source"], "return states;");
        // Two nodes, holding 2 and 1 shards.
        let r = red(spec, vec![json!({"s": {"value": [3, 4]}}), json!({"s": {"value": [5]}})]);
        assert_eq!(r["s"]["value"], json!(12.0));
        assert!(take_error().is_none());
    }

    #[test]
    fn grid_and_series_buckets_merge_by_key() {
        // Count ties are broken by the cell's long encoding, highest first.
        let r = red(json!({"g": {"geohash_grid": {"field": "p", "precision": 5}}}), vec![
            json!({"g": {"buckets": [{"key": "u09tu", "doc_count": 1}, {"key": "u0abc", "doc_count": 3}]}}),
            json!({"g": {"buckets": [{"key": "u155k", "doc_count": 1}, {"key": "u0abc", "doc_count": 1}]}}),
        ]);
        let keys: Vec<&str> = r["g"]["buckets"].as_array().unwrap().iter().map(|b| b["key"].as_str().unwrap()).collect();
        assert_eq!(keys, ["u0abc", "u155k", "u09tu"]);
        assert_eq!(r["g"]["buckets"][0]["doc_count"], json!(4));
        let r = red(json!({"g": {"geotile_grid": {"field": "p", "size": 2}}}), vec![
            json!({"g": {"buckets": [{"key": "8/1/2", "doc_count": 1}, {"key": "8/2/1", "doc_count": 1}, {"key": "8/0/9", "doc_count": 1}]}}),
        ]);
        assert_eq!(r["g"]["buckets"].as_array().unwrap().len(), 2);
        assert_eq!(r["g"]["buckets"][0]["key"], json!("8/2/1"));
        // time_series: each series lives on one shard; sub-aggregations
        // (here a rate) are that shard's own.
        let r = red(json!({"ts": {"time_series": {"keyed": false}, "aggs": {"r": {"rate": {"field": "c"}}}}}), vec![
            json!({"ts": {"buckets": [{"key": {"host": "a"}, "doc_count": 2, "r": {"value": 1.0}}]}}),
            json!({"ts": {"buckets": [{"key": {"host": "b"}, "doc_count": 3, "r": {"value": 7.0}}]}}),
        ]);
        assert_eq!(r["ts"]["buckets"].as_array().unwrap().len(), 2);
        assert_eq!(r["ts"]["buckets"][1]["r"]["value"], json!(7.0));
        assert!(unsupported(&json!({"ts": {"time_series": {}, "aggs": {"r": {"rate": {"field": "c"}}}}})).is_none());
        assert!(unsupported(&json!({"m": {"boxplot": {"field": "c"}}})).is_none());
        assert!(unsupported(&json!({"m": {"categorize_text": {"field": "c"}}})).is_none());
        assert!(unsupported(&json!({"m": {"no_such_aggregation": {"field": "c"}}})).is_some());
    }

    #[test]
    fn rewrite_asks_for_shard_size_and_helpers() {
        let r = rewrite(&json!({
            "t": {"terms": {"field": "k", "size": 2, "min_doc_count": 3}, "aggs": {"a": {"avg": {"field": "v"}}, "c": {"cardinality": {"field": "u"}}}},
            "h": {"top_hits": {"size": 2, "from": 1}}
        }));
        assert_eq!(r["t"]["terms"], json!({"field": "k", "size": 13, "min_doc_count": 1}));
        assert_eq!(r["t"]["aggs"]["a"], json!({"avg": {"field": "v"}}));
        assert_eq!(r["t"]["aggs"]["__fanout_avg__a"], json!({"stats": {"field": "v"}}));
        assert_eq!(r["t"]["aggs"]["c"], json!({"cardinality": {"field": "u"}}));
        assert!(r["t"]["aggs"].get("__fanout_card__c").is_none(), "cardinality carries shard state: {r}");
        assert_eq!(r["h"]["top_hits"], json!({"size": 3, "from": 0}));
    }

    #[test]
    fn unsupported_is_recursive() {
        assert_eq!(unsupported(&json!({"a": {"terms": {"field": "k"}, "aggs": {"m": {"max": {"field": "v"}}}}})), None);
        assert_eq!(unsupported(&json!({"a": {"terms": {"field": "k"}, "aggs": {"p": {"percentiles": {"field": "v"}}}}})), None);
        assert_eq!(unsupported(&json!({"a": {"date_histogram": {"field": "d"}, "aggs": {"p": {"derivative": {"buckets_path": "_count"}}}}})), None);
        assert!(unsupported(&json!({"a": {"terms": {"field": "k"}, "aggs": {"b": {"no_such_aggregation": {"field": "v"}}}}})).is_some());
        assert_eq!(unsupported(&json!({"a": {"histogram": {"field": "v", "interval": 1}, "aggs": {"m": {"moving_fn": {"buckets_path": "_count"}}}}})), None);
        assert_eq!(unsupported(&json!({"a": {"histogram": {"field": "v", "interval": 1}, "aggs": {"m": {"inference": {"buckets_path": {}}}}}})), Some("this pipeline aggregation"));
    }

    #[test]
    fn rewrite_root_adds_helpers_and_strips_coordinator_pipelines() {
        let r = rewrite_root(&json!({
            "p": {"percentiles": {"field": "v", "percents": [50]}},
            "r": {"rare_terms": {"field": "k"}},
            "h": {"histogram": {"field": "v", "interval": 1}, "aggs": {"s": {"bucket_sort": {"size": 1}}, "c": {"cumulative_sum": {"buckets_path": "_count"}}}},
            "t": {"terms": {"field": "g"}, "aggs": {"sig": {"significant_terms": {"field": "k", "chi_square": {}}}}}
        }));
        assert!(r.as_object().unwrap().keys().all(|k| !k.starts_with("__fanout_pct__") && !k.starts_with("__fanout_card__")),
            "sketch aggregations carry shard state, no helper terms: {r}");
        assert!(r.get("__fanout_rare__r").is_none(), "rare_terms carries shard state: {r}");
        assert!(r["h"]["aggs"].get("s").is_none());
        assert!(r["h"]["aggs"].get("c").is_some());
        assert_eq!(r["t"]["aggs"]["sig"]["significant_terms"], json!({"field": "k", "chi_square": {}, "size": 25, "min_doc_count": 1}));
        assert_eq!(r["__fanout_sigbg__sig"]["aggs"]["f"]["aggs"]["t"], json!({"terms": {"field": "k", "size": 10000}}));
    }

    fn values(pairs: &[(f64, u64)]) -> Value {
        json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": pairs.iter().map(|(k, c)| json!({"key": k, "doc_count": c})).collect::<Vec<_>>()})
    }

    fn digest_state(xs: impl Iterator<Item = f64>) -> Value {
        let mut td = elasticsearch_engine::aggs::sketch::TDigest::new(100.0);
        let mut n = 0;
        for x in xs {
            td.add(x);
            n += 1;
        }
        json!({"sketch": {"td": td.state()}, "n": n, "date": false, "fmt": ""})
    }

    #[test]
    fn percentiles_merge_shard_digests() {
        // Each shard has 50 distinct values: far more than any exact-values
        // helper could carry at scale, merged through the shards' t-digests.
        let spec = json!({"p": {"percentiles": {"field": "v", "percents": [50, 100]}}, "r": {"percentile_ranks": {"field": "v", "values": [25], "keyed": false}}});
        let a = json!({"p": {"values": {"50.0": 25.5, "100.0": 50.0}, "__p": digest_state((1..=50).map(f64::from))},
            "r": {"values": [{"key": 25.0, "value": 50.0}], "__p": digest_state((1..=50).map(f64::from))}});
        let b = json!({"p": {"values": {"50.0": 75.5, "100.0": 100.0}, "__p": digest_state((51..=100).map(f64::from))},
            "r": {"values": [{"key": 25.0, "value": 0.0}], "__p": digest_state((51..=100).map(f64::from))}});
        let out = red(spec, vec![a, b]);
        assert_eq!(out["p"]["values"]["100.0"], 100.0);
        let med = out["p"]["values"]["50.0"].as_f64().unwrap();
        assert!((48.0..=53.0).contains(&med), "{med}");
        let rank = out["r"]["values"][0]["value"].as_f64().unwrap();
        assert!((23.0..=27.0).contains(&rank), "{rank}");
        assert!(out["p"].get("__p").is_none() && out["r"].get("__p").is_none(), "shard state never reaches clients: {out}");
        // A top_hits hit kept as raw text survives the reduce (the final
        // render writes it verbatim).
        let hit = |n: u64| json!({"_id": n.to_string(), "_index": "i", "_score": null, "sort": [n], "__p": {"raw_json": format!(r#"{{"_index":"i","_id":"{n}","_source":{{"t":"x\ud800"}},"sort":[{n}]}}"#)}});
        let spec = json!({"t": {"top_hits": {"size": 2, "sort": [{"n": "desc"}]}}});
        let part = |a: u64, b: u64| json!({"t": {"hits": {"total": {"value": 2, "relation": "eq"}, "max_score": null, "hits": [hit(a), hit(b)]}}});
        let out = reduce_root_with_limit(&spec, &[&part(9, 3), &part(7, 1)], false, 65_536);
        let hits = out["t"]["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.iter().map(|h| h["_id"].as_str().unwrap()).collect::<Vec<_>>(), ["9", "7"], "{out}");
        assert!(hits.iter().all(|h| h["__p"]["raw_json"].as_str().is_some_and(|r| r.contains(r#"\ud800"#))), "{out}");
    }

    #[test]
    fn cardinality_merges_shard_sketches() {
        // 3 shards x 20,000 distinct values: the old 10,000-key helper could
        // only report a lower bound (~20,000).
        let spec = json!({"c": {"cardinality": {"field": "k"}}});
        let shard = |offset: u64| {
            let mut h = elasticsearch_engine::aggs::sketch::HyperLogLog::new(3000);
            for i in 0..20_000u64 {
                h.add_bytes(&(offset + i).to_le_bytes());
            }
            json!({"c": {"value": h.count(), "__p": {"hll": h.state()}}})
        };
        let out = red(spec, vec![shard(0), shard(20_000), shard(40_000)]);
        let n = out["c"]["value"].as_u64().unwrap() as f64;
        assert!((n - 60_000.0).abs() / 60_000.0 < 0.05, "{n}");
        assert!(out["c"].get("__p").is_none());
    }

    #[test]
    fn sketch_aggregations_refuse_responses_without_state() {
        // A node that predates shard state: an explicit error, never a guess.
        let spec = json!({"c": {"cardinality": {"field": "k"}}});
        let _ = take_error();
        red(spec, vec![json!({"c": {"value": 3}}), json!({"c": {"value": 4}})]);
        assert_eq!(take_error().map(|e| e.0), Some(500));
    }

    #[test]
    fn rare_terms_use_global_counts() {
        let spec = json!({"r": {"rare_terms": {"field": "k"}}});
        let bloom = |keys: &[&str]| {
            let mut b = elasticsearch_engine::aggs::sketch::KeyBloom::with_capacity(keys.len(), 0.001);
            for k in keys {
                b.insert(&json!(k).to_string());
            }
            json!({"frequent": b.state()})
        };
        // "x" is rare on both nodes but occurs twice overall; "a" is rare on
        // one node and frequent on the other.
        let a = json!({"r": {"buckets": [{"key": "x", "doc_count": 1}, {"key": "y", "doc_count": 1}], "__p": bloom(&["a", "z"])}});
        let b = json!({"r": {"buckets": [{"key": "x", "doc_count": 1}, {"key": "a", "doc_count": 1}, {"key": "b", "doc_count": 1}], "__p": {}}});
        let out = red(spec.clone(), vec![a, b.clone()]);
        assert_eq!(out["r"], json!({"buckets": [{"key": "b", "doc_count": 1}, {"key": "y", "doc_count": 1}]}));
        // A node without the field contributes nothing; one that predates the
        // state fails the reduce.
        let out = red(spec.clone(), vec![json!({"r": {"buckets": []}}), b.clone()]);
        assert_eq!(out["r"]["buckets"].as_array().unwrap().len(), 3);
        ERR.with(|e| *e.borrow_mut() = None);
        let _ = reduce(&spec, &[&json!({"r": {"buckets": [{"key": "q", "doc_count": 1}]}}), &b], false);
        assert!(take_error().is_some());
    }

    #[test]
    fn significant_terms_rescored_on_merged_counts() {
        let spec = json!({"s": {"significant_terms": {"field": "k", "min_doc_count": 1}}});
        let part = |n: u64, bg: u64, bs: Value| json!({"s": {"doc_count": n, "bg_count": bg, "buckets": bs}});
        let a = part(5, 50, json!([{"key": "x", "doc_count": 3, "score": 0.1, "bg_count": 4}, {"key": "y", "doc_count": 1, "score": 0.1, "bg_count": 20}]));
        let b = part(5, 50, json!([{"key": "x", "doc_count": 2, "score": 0.1, "bg_count": 4}]));
        let parts = vec![a, b];
        let refs: Vec<&Value> = parts.iter().collect();
        SIG_BG.with(|s| {
            s.borrow_mut().insert("s".into(), (HashMap::from([(json!("x").to_string(), 8u64), (json!("y").to_string(), 20u64)]), true));
        });
        let out = reduce(&spec, &refs, false);
        SIG_BG.with(|s| s.borrow_mut().clear());
        let s = &out["s"];
        assert_eq!(s["doc_count"], 10);
        assert_eq!(s["bg_count"], 100);
        let bs = s["buckets"].as_array().unwrap();
        // y: sp = 0.1, bp = 0.2 -> not significant.
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0]["key"], "x");
        assert_eq!(bs[0]["doc_count"], 5);
        assert_eq!(bs[0]["bg_count"], 8);
        // jlh: sp = 0.5, bp = 0.08 -> 0.42 * 6.25
        assert!((bs[0]["score"].as_f64().unwrap() - 2.625).abs() < 1e-9);
        let keys: Vec<&str> = bs[0].as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, vec!["key", "doc_count", "score", "bg_count"]);
    }

    #[test]
    fn significant_terms_heuristics() {
        let p = Map::new();
        assert_eq!(sig_score("percentage", &p, 2.0, 10.0, 4.0, 100.0), 0.5);
        assert_eq!(sig_score("jlh", &p, 1.0, 10.0, 20.0, 100.0), 0.0);
        assert!(sig_score("chi_square", &p, 5.0, 10.0, 8.0, 100.0) > 0.0);
        assert!(sig_score("mutual_information", &p, 5.0, 10.0, 8.0, 100.0) > 0.0);
        assert!(sig_score("gnd", &p, 5.0, 10.0, 8.0, 100.0) > 0.0);
        assert_eq!(sig_score("chi_square", &p, 1.0, 10.0, 50.0, 100.0), f64::NEG_INFINITY);
    }

    #[test]
    fn multi_terms_merge() {
        let spec = json!({"m": {"multi_terms": {"terms": [{"field": "a"}, {"field": "b"}], "size": 2}}});
        let part = |bs: Value, other: u64| json!({"m": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": other, "buckets": bs}});
        let a = part(json!([{"key": ["x", 1], "key_as_string": "x|1", "doc_count": 3}, {"key": ["y", 2], "key_as_string": "y|2", "doc_count": 1}]), 0);
        let b = part(json!([{"key": ["y", 2], "key_as_string": "y|2", "doc_count": 2}, {"key": ["x", 0], "key_as_string": "x|0", "doc_count": 2}]), 1);
        let out = red(spec, vec![a, b]);
        assert_eq!(out["m"], json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": 3, "buckets": [
            {"key": ["x", 1], "key_as_string": "x|1", "doc_count": 3}, {"key": ["y", 2], "key_as_string": "y|2", "doc_count": 3}]}));
    }

    #[test]
    fn composite_pages_in_source_order() {
        let spec = json!({"c": {"composite": {"size": 3, "sources": [{"k": {"terms": {"field": "k", "order": "desc"}}}, {"n": {"terms": {"field": "n"}}}]}}});
        let part = |bs: Value| json!({"c": {"after_key": bs.as_array().unwrap().last().unwrap()["key"].clone(), "buckets": bs}});
        let a = part(json!([{"key": {"k": "b", "n": 1}, "doc_count": 1}, {"key": {"k": "a", "n": 1}, "doc_count": 2}]));
        let b = part(json!([{"key": {"k": "b", "n": 0}, "doc_count": 1}, {"key": {"k": "a", "n": 1}, "doc_count": 1}, {"key": {"k": "a", "n": 2}, "doc_count": 1}]));
        let out = red(spec, vec![a, b]);
        assert_eq!(out["c"], json!({"after_key": {"k": "a", "n": 1}, "buckets": [
            {"key": {"k": "b", "n": 0}, "doc_count": 1}, {"key": {"k": "b", "n": 1}, "doc_count": 1}, {"key": {"k": "a", "n": 1}, "doc_count": 3}]}));
    }

    #[test]
    fn parent_pipelines_run_on_merged_buckets() {
        let spec = json!({"h": {"histogram": {"field": "v", "interval": 1}, "aggs": {
            "s": {"sum": {"field": "v"}},
            "d": {"derivative": {"buckets_path": "s"}},
            "c": {"cumulative_sum": {"buckets_path": "_count"}},
            "sel": {"bucket_selector": {"buckets_path": {"n": "_count"}, "script": "params.n > 1"}},
            "so": {"bucket_sort": {"sort": [{"s": "desc"}], "size": 2}}}},
            "avg": {"avg_bucket": {"buckets_path": "h>s"}},
            "mx": {"max_bucket": {"buckets_path": "h>_count"}}});
        // Node copies of d/c are stale and must be recomputed.
        let a = json!({"h": {"buckets": [{"key": 0.0, "doc_count": 1, "s": {"value": 1.0}, "c": {"value": 1.0}}, {"key": 1.0, "doc_count": 1, "s": {"value": 2.0}, "d": {"value": 1.0}, "c": {"value": 2.0}}]},
            "avg": {"value": 1.5}, "mx": {"value": 1.0, "keys": ["0.0"]}});
        let b = json!({"h": {"buckets": [{"key": 1.0, "doc_count": 2, "s": {"value": 5.0}, "c": {"value": 2.0}}, {"key": 2.0, "doc_count": 3, "s": {"value": 3.0}, "d": {"value": -2.0}, "c": {"value": 5.0}}]},
            "avg": {"value": 4.0}, "mx": {"value": 3.0, "keys": ["2.0"]}});
        let out = red(spec, vec![a, b]);
        // Merged: 0 -> (1, s1), 1 -> (3, s7), 2 -> (3, s3). Selector drops 0; sort by s desc.
        assert_eq!(out["h"]["buckets"], json!([
            {"key": 1.0, "doc_count": 3, "s": {"value": 7.0}, "d": {"value": 6.0}, "c": {"value": 4.0}},
            {"key": 2.0, "doc_count": 3, "s": {"value": 3.0}, "d": {"value": -4.0}, "c": {"value": 7.0}}]));
        // Siblings see the final (selected and sorted) buckets.
        assert_eq!(out["avg"], json!({"value": 5.0}));
        assert_eq!(out["mx"], json!({"value": 3.0, "keys": ["1.0", "2.0"]}));
    }

    #[test]
    fn pipeline_errors_surface() {
        let spec = json!({"h": {"histogram": {"field": "v", "interval": 1}, "aggs": {"sel": {"bucket_selector": {"buckets_path": {"n": "_count"}, "script": "params.n + 1"}}}}});
        let a = json!({"h": {"buckets": [{"key": 0.0, "doc_count": 1}]}});
        let _ = take_error();
        red(spec, vec![a.clone(), a]);
        let e = take_error().unwrap();
        assert_eq!(e.0, 400);
        assert!(e.2.contains("must return a boolean"), "{e:?}");
        let _ = take_error();
        red(json!({"x": {"avg_bucket": {"buckets_path": "nope>_count"}}}), vec![json!({}), json!({})]);
        assert!(take_error().unwrap().2.contains("No aggregation found for path [nope>_count]"));
    }

    #[test]
    fn metrics() {
        let spec = json!({"mn": {"min": {"field": "v"}}, "mx": {"max": {"field": "v"}}, "s": {"sum": {"field": "v"}}, "n": {"value_count": {"field": "v"}},
            "a": {"avg": {"field": "v"}}, "st": {"stats": {"field": "v"}}});
        let part = |mn: f64, mx: f64, s: f64, n: u64| {
            json!({"mn": {"value": mn}, "mx": {"value": mx}, "s": {"value": s}, "n": {"value": n}, "a": {"value": s / n as f64},
                "__fanout_avg__a": {"count": n, "min": mn, "max": mx, "avg": s / n as f64, "sum": s},
                "st": {"count": n, "min": mn, "max": mx, "avg": s / n as f64, "sum": s}})
        };
        let out = red(spec, vec![part(1.0, 5.0, 9.0, 3), part(-2.0, 4.0, 3.0, 1)]);
        assert_eq!(out, json!({"mn": {"value": -2.0}, "mx": {"value": 5.0}, "s": {"value": 12.0}, "n": {"value": 4}, "a": {"value": 3.0},
            "st": {"count": 4, "min": -2.0, "max": 5.0, "avg": 3.0, "sum": 12.0}}));
    }

    #[test]
    fn empty_nodes_do_not_disturb_metrics() {
        let spec = json!({"mn": {"min": {"field": "v"}}, "a": {"avg": {"field": "v"}}, "st": {"extended_stats": {"field": "v"}}});
        let full = json!({"mn": {"value": 1.0}, "a": {"value": 2.5}, "__fanout_avg__a": {"count": 2, "sum": 5.0}, "st": {"count": 2, "sum": 5.0, "marker": true}});
        let empty = json!({"mn": {"value": null}, "a": {"value": null}, "__fanout_avg__a": {"count": 0, "sum": 0.0}, "st": {"count": 0, "sum": 0.0}});
        let out = red(spec, vec![empty, full]);
        assert_eq!(out, json!({"mn": {"value": 1.0}, "a": {"value": 2.5}, "st": {"count": 2, "sum": 5.0, "marker": true}}));
    }

    #[test]
    fn extended_stats_merge() {
        // {1, 3} and {5}: mean 3, population variance 8/3.
        let out = red(
            json!({"e": {"extended_stats": {"field": "v"}}}),
            vec![json!({"e": {"count": 2, "min": 1.0, "max": 3.0, "sum": 4.0, "sum_of_squares": 10.0}}), json!({"e": {"count": 1, "min": 5.0, "max": 5.0, "sum": 5.0, "sum_of_squares": 25.0}})],
        );
        let e = &out["e"];
        assert_eq!(e["count"], 3);
        assert_eq!(e["avg"], 3.0);
        assert!((e["variance"].as_f64().unwrap() - 8.0 / 3.0).abs() < 1e-9);
        assert!((e["variance_sampling"].as_f64().unwrap() - 4.0).abs() < 1e-9);
        assert!((e["std_deviation_bounds"]["upper_sampling"].as_f64().unwrap() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn terms_merge_with_error_bounds_and_sub_aggs() {
        let spec = json!({"t": {"terms": {"field": "k", "size": 2, "shard_size": 2, "show_term_doc_count_error": true}, "aggs": {"m": {"max": {"field": "v"}}}}});
        let a = json!({"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 3, "buckets": [
            {"key": "x", "doc_count": 10, "m": {"value": 1.0}}, {"key": "y", "doc_count": 4, "m": {"value": 7.0}}]}});
        let b = json!({"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 1, "buckets": [
            {"key": "z", "doc_count": 6, "m": {"value": 2.0}}, {"key": "x", "doc_count": 5, "m": {"value": 9.0}}]}});
        let out = red(spec, vec![a, b]);
        assert_eq!(out["t"], json!({"doc_count_error_upper_bound": 9, "sum_other_doc_count": 8, "buckets": [
            {"key": "x", "doc_count": 15, "m": {"value": 9.0}, "doc_count_error_upper_bound": 0},
            {"key": "z", "doc_count": 6, "m": {"value": 2.0}, "doc_count_error_upper_bound": 4}]}));
    }

    #[test]
    fn single_shard_terms_truncation_has_no_count_error() {
        let spec = json!({"t": {"terms": {"field": "k", "size": 1, "shard_size": 2, "show_term_doc_count_error": true}}});
        let part = json!({"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 3, "buckets": [
            {"key": "x", "doc_count": 10}, {"key": "y", "doc_count": 4}]}});
        let out = red(spec, vec![part]);
        assert_eq!(out["t"], json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": 7,
            "buckets": [{"key": "x", "doc_count": 10, "doc_count_error_upper_bound": 0}]}));
    }

    #[test]
    fn terms_orders_and_min_doc_count() {
        let part = |bs: Value| json!({"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": bs}});
        let a = part(json!([{"key": 1, "doc_count": 1, "s": {"value": 5.0}}, {"key": 2, "doc_count": 2, "s": {"value": 1.0}}]));
        let b = part(json!([{"key": 2, "doc_count": 1, "s": {"value": 1.0}}, {"key": 3, "doc_count": 1, "s": {"value": 9.0}}]));
        let sub = json!({"s": {"sum": {"field": "v"}}});
        let keys = |v: &Value| v["t"]["buckets"].as_array().unwrap().iter().map(|b| b["key"].as_i64().unwrap()).collect::<Vec<_>>();
        let out = red(json!({"t": {"terms": {"field": "k", "order": {"_key": "desc"}}, "aggs": sub}}), vec![a.clone(), b.clone()]);
        assert_eq!(keys(&out), vec![3, 2, 1]);
        assert_eq!(out["t"]["doc_count_error_upper_bound"], 0);
        let out = red(json!({"t": {"terms": {"field": "k", "order": [{"s": "desc"}]}, "aggs": sub}}), vec![a.clone(), b.clone()]);
        assert_eq!(keys(&out), vec![3, 1, 2]);
        // Default order: count desc, then key asc.
        let out = red(json!({"t": {"terms": {"field": "k"}, "aggs": sub}}), vec![a.clone(), b.clone()]);
        assert_eq!(keys(&out), vec![2, 1, 3]);
        let out = red(json!({"t": {"terms": {"field": "k", "min_doc_count": 2}, "aggs": sub}}), vec![a, b]);
        assert_eq!(keys(&out), vec![2]);
        assert_eq!(out["t"]["buckets"][0]["s"]["value"], 2.0);
    }

    #[test]
    fn histogram_gap_growth_trips_before_materializing_all_empty_buckets() {
        let breakers=elasticsearch_engine::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        elasticsearch_engine::memory::scoped(Some(elasticsearch_engine::memory::RequestScope::new("histogram fill",breakers)),|| {
            let spec=json!({"h":{"histogram":{"field":"v","interval":1}}});
            let a=json!({"h":{"buckets":[{"key":0,"doc_count":1}]}});
            let b=json!({"h":{"buckets":[{"key":100000,"doc_count":1}]}});
            super::FILLED_BUCKETS.with(|count|count.set(0));
            let out=super::reduce_root_with_limit(&spec,&[&a,&b],false,200000);
            assert!(out.is_null());
            assert_eq!(super::take_breaking().unwrap().status,429);
            assert!(super::FILLED_BUCKETS.with(|count|count.get())<1000);
            assert_eq!(breakers.request.reserved(),0);
        });
    }

    #[test]
    fn histogram_fills_gaps_between_nodes() {
        let spec = json!({"h": {"histogram": {"field": "v", "interval": 10}, "aggs": {"m": {"min": {"field": "v"}}, "n": {"value_count": {"field": "v"}}}}});
        let a = json!({"h": {"buckets": [{"key": 0.0, "doc_count": 2, "m": {"value": 1.0}, "n": {"value": 2}}]}});
        let b = json!({"h": {"buckets": [{"key": 30.0, "doc_count": 1, "m": {"value": 31.0}, "n": {"value": 1}}, {"key": 0.0, "doc_count": 1, "m": {"value": 0.5}, "n": {"value": 1}}]}});
        let out = red(spec.clone(), vec![a.clone(), b.clone()]);
        assert_eq!(out["h"]["buckets"], json!([
            {"key": 0.0, "doc_count": 3, "m": {"value": 0.5}, "n": {"value": 2 + 1}},
            {"doc_count": 0, "m": {"value": null}, "n": {"value": 0}, "key": 10.0},
            {"doc_count": 0, "m": {"value": null}, "n": {"value": 0}, "key": 20.0},
            {"key": 30.0, "doc_count": 1, "m": {"value": 31.0}, "n": {"value": 1}}]));
        let mut s2 = spec;
        s2["h"]["histogram"]["min_doc_count"] = json!(2);
        assert_eq!(red(s2, vec![a, b])["h"]["buckets"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn date_histogram_fixed_gap_fill_crosses_dst_without_extra_buckets() {
        // Exact ES9 full-corpus keys around the 2008 New York fall-back.
        let spec = json!({"d":{"date_histogram":{"field":"ts","fixed_interval":"30d",
            "time_zone":"America/New_York","offset":"6h"}}});
        let a = json!({"d":{"buckets":[{"key":1223460000000i64,"key_as_string":"2008-10-08T06:00:00.000-04:00","doc_count":1}]}});
        let b = json!({"d":{"buckets":[{"key":1226055600000i64,"key_as_string":"2008-11-07T06:00:00.000-05:00","doc_count":2}]}});
        let out = red(spec, vec![a, b]);
        let buckets = out["d"]["buckets"].as_array().unwrap();
        assert_eq!(buckets.len(), 2, "{out}");
        assert_eq!(buckets[1]["key"], 1226055600000i64);
        assert_eq!(buckets[1]["doc_count"], 2);
    }

    #[test]
    fn date_histogram_calendar_gap_fill() {
        let spec = json!({"d": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}});
        let a = json!({"d": {"buckets": [{"key_as_string": "2024-01-01T00:00:00.000Z", "key": 1704067200000i64, "doc_count": 1}]}});
        let b = json!({"d": {"buckets": [{"key_as_string": "2024-04-01T00:00:00.000Z", "key": 1711929600000i64, "doc_count": 2}]}});
        let out = red(spec, vec![a, b]);
        let bs = out["d"]["buckets"].as_array().unwrap();
        assert_eq!(bs.len(), 4);
        assert_eq!(bs[1], json!({"doc_count": 0, "key": 1706745600000i64, "key_as_string": "2024-02-01T00:00:00.000Z"}));
        assert_eq!(bs[2]["key_as_string"], "2024-03-01T00:00:00.000Z");
        assert_eq!(bs[3]["doc_count"], 2);
    }

    #[test]
    fn single_and_fixed_buckets() {
        let spec = json!({"f": {"filter": {"term": {"k": "a"}}, "aggs": {"s": {"sum": {"field": "v"}}}},
            "r": {"range": {"field": "v", "ranges": [{"to": 5}, {"from": 5}]}},
            "fs": {"filters": {"filters": {"x": {"match_all": {}}}}}});
        let part = |n: u64| json!({"f": {"doc_count": n, "s": {"value": n as f64}}, "r": {"buckets": [{"key": "*-5.0", "to": 5.0, "doc_count": n}, {"key": "5.0-*", "from": 5.0, "doc_count": 1}]},
            "fs": {"buckets": {"x": {"doc_count": n}}}});
        let out = red(spec, vec![part(2), part(3)]);
        assert_eq!(out["f"], json!({"doc_count": 5, "s": {"value": 5.0}}));
        assert_eq!(out["r"]["buckets"], json!([{"key": "*-5.0", "to": 5.0, "doc_count": 5}, {"key": "5.0-*", "from": 5.0, "doc_count": 2}]));
        assert_eq!(out["fs"], json!({"buckets": {"x": {"doc_count": 5}}}));
    }

    #[test]
    fn typed_keys_and_meta_and_top_hits() {
        let spec = json!({"t": {"terms": {"field": "k"}, "meta": {"a": 1}}, "h": {"top_hits": {"size": 2, "sort": [{"v": "desc"}]}}});
        let part = |k: &str, v: i64| json!({"sterms#t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": [{"key": k, "doc_count": 1}]},
            "top_hits#h": {"hits": {"total": {"value": 2, "relation": "eq"}, "max_score": null, "hits": [{"_index": k, "_id": "1", "sort": [v]}, {"_index": k, "_id": "2", "sort": [v - 5]}]}}});
        let parts = vec![part("a", 10), part("b", 8)];
        let refs: Vec<&Value> = parts.iter().collect();
        let out = reduce(&spec, &refs, true);
        assert_eq!(out["sterms#t"]["meta"], json!({"a": 1}));
        assert_eq!(out["sterms#t"]["buckets"].as_array().unwrap().len(), 2);
        let h = &out["top_hits#h"]["hits"];
        assert_eq!(h["total"]["value"], 4);
        assert_eq!(h["hits"].as_array().unwrap().iter().map(|x| x["sort"][0].as_i64().unwrap()).collect::<Vec<_>>(), vec![10, 8]);
    }
}
