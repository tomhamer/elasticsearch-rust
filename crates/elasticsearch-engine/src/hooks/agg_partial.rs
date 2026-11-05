//! Hook 10: per-shard aggregation partials and the coordinator-side reduce.
//!
//! * [`shard_partial`] runs the aggregation tree over one shard's matching
//!   docs in *partial mode*: bucket aggregations return `shard_size` buckets
//!   (not `size`), pipeline aggregations are skipped (`skip_pipeline`), and
//!   metrics whose rendered value cannot be merged carry their mergeable
//!   state next to it under a hidden `__p` key (`avg`: sum + count,
//!   `cardinality`: HyperLogLog++ hashes / registers, `percentiles`: t-digest
//!   centroids, `significant_terms`: background counts, `geo_bounds`: signed
//!   longitude extents, ...).
//! * [`AggPartial::reduce`] merges the partials of all shards the way ES's
//!   `InternalAggregation.reduce` does and then runs the pipeline
//!   aggregations over the reduced tree, producing the final `aggregations`
//!   JSON (typed keys applied when requested).
//!
//! Compiled as a child module of `crate::aggs` (`aggs::partial`) so it can
//! use the aggregation framework's private context and helpers.

use super::sketch::{Hdr, HdrState, HllState, HyperLogLog, TDigest, TDigestState};
use super::*;
use serde::{Deserialize, Serialize};

/// The aggregation part of a search request, as the coordinator reduces it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggRequest {
    /// The request's `aggs` object.
    pub aggs: Value,
    pub typed_keys: bool,
    pub max_buckets: usize,
    /// `now` for date math (same value the shards used).
    pub now: i64,
}

/// One shard's aggregation state: the rendered tree of that shard plus the
/// hidden mergeable state of non-additive metrics. Serializable as JSON
/// ([`AggPartial::to_json`]) or bytes ([`AggPartial::to_bytes`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggPartial {
    /// Format version of `tree`.
    pub version: u32,
    pub tree: Value,
}

const PARTIAL_VERSION: u32 = 1;

impl AggPartial {
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    pub fn from_json(v: Value) -> Result<AggPartial> {
        Ok(serde_json::from_value(v)?)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn from_bytes(b: &[u8]) -> Result<AggPartial> {
        Ok(serde_json::from_slice(b)?)
    }

    /// Reduce shard partials into the final `aggregations` object.
    ///
    /// Works for any number of partials (a single one included: pipeline
    /// aggregations skipped on the shard run here). Aggregation types without
    /// a mergeable partial pass a lone partial through and reject several.
    pub fn reduce(parts: Vec<AggPartial>, req: &AggRequest) -> Result<Value> {
        // ES `QueryPhaseResultConsumer`: the partials are held on the request
        // breaker during the reduce, with half as much again for its result.
        let held = parts.iter().fold(0u64, |n,p|n.saturating_add(crate::memory::json_bytes(&p.tree)));
        // The partials move into this charge: any retained for the request
        // are released, not counted a second time (and not kept after).
        crate::memory::consume_retained_partials(held);
        let mut mem = crate::memory::breakers().reserve_request(held.saturating_add(held / 2), "<reduce_aggs>")?;
        let nested = crate::memory::partials_mark();
        let trees: Vec<&Value> = parts.iter().map(|p| &p.tree).collect();
        let rc = Rc2 { now: req.now, max_buckets: req.max_buckets };
        let spec = req.aggs.as_object().ok_or_else(|| EsError::parsing_at("Expected [START_OBJECT] under [aggregations], but got a [VALUE_STRING]", 1, 1))?;
        let mut out = Value::Object(reduce_level(spec, &trees, &rc)?);
        if (super::significant_text::present(&req.aggs) || super::frequent_items::present(&req.aggs)) && !crate::search::aggregation_shard_response() {
            super::significant_text::check_final_bucket_limit(&req.aggs, &out, req.max_buckets)?;
        }
        if req.typed_keys {
            out = crate::search::add_typed_keys_pub(out, &req.aggs);
        }
        // The result escapes the reducer and remains live while the
        // coordinator builds/serializes its response. Retain only that live
        // tree after the partial/scratch ownership ends.
        let output=crate::memory::json_bytes(&out);
        if output>mem.bytes(){mem.grow(output-mem.bytes(),"<reduce_aggs>")?;}
        else {mem.shrink(mem.bytes()-output);}
        // Outputs of nested reduces (scripted metrics) are inside `out`, now
        // charged whole.
        crate::memory::consume_partials_since(nested);
        mem.retain_for_request();
        Ok(out)
    }
}

/// Aggregations whose cluster shard responses carry their mergeable state
/// (`__p`): HLL++ for `cardinality`, t-digest / HDR for the percentiles,
/// t-digest for `boxplot` / `median_absolute_deviation`, signed longitude
/// extents for `geo_bounds`, moments for `matrix_stats` / `t_test`; and the
/// sort values of `geo_line` points. Their rendered values are not mergeable
/// (ES reduces the same states).
pub const SHARD_STATE_AGGS: &[&str] = &[
    "cardinality", "percentiles", "percentile_ranks", "geo_bounds", "boxplot", "median_absolute_deviation", "matrix_stats", "t_test", "geo_line",
];

/// Coordinator reduce of one [`SHARD_STATE_AGGS`] aggregation from shard
/// responses carrying their state; `None` when a response lacks it (an
/// older node). The result has no state; `meta` is the caller's.
pub fn reduce_shard_states(name: &str, t: &str, conf: &Map<String, Value>, vals: &[&Value]) -> Option<Result<Value>> {
    if vals.is_empty() {
        return None;
    }
    if t == "geo_line" {
        // Shards keep their points' sort values (`include_sort`).
        let lines: Vec<&Value> = vals.iter().copied().filter(|v| v.get("geometry").is_some()).collect();
        if lines.iter().any(|v| v["properties"].get("sort_values").is_none()) {
            return None;
        }
        return Some(Ok(if lines.is_empty() { (*vals[0]).clone() } else { reduce_geo_line(conf, &lines) }));
    }
    if vals.iter().any(|v| state(v).is_none()) {
        return None;
    }
    let mut out = match t {
        "cardinality" => Ok(reduce_cardinality(vals)),
        "percentiles" | "percentile_ranks" => reduce_percentiles(name, t, conf, vals),
        "geo_bounds" => Ok(reduce_geo_bounds(conf, vals)),
        "boxplot" | "median_absolute_deviation" => Ok(reduce_quantile_metric(t, vals)),
        "t_test" => Ok(reduce_t_test(vals)),
        "matrix_stats" => reduce_matrix_stats(name, vals),
        _ => return None,
    };
    // Formatted like the shards rendered it.
    if let (Ok(v), "cardinality", true) = (&mut out, t, vals[0].get("value_as_string").is_some()) {
        if let (Some(f), Some(n)) = (conf.get("format").and_then(Value::as_str), v.get("value").and_then(Value::as_f64)) {
            v["value_as_string"] = json!(super::analytics::decimal_format(f, n));
        }
    }
    Some(out)
}

/// Parent pipelines the coordinator runs with the engine's implementation
/// over merged buckets: `normalize` reads rendered values; the cardinality
/// sets / percentile values `cumulative_cardinality` / `moving_percentiles`
/// need ride on the referenced metric under [`PIPELINE_PAYLOAD`].
pub const ENGINE_PARENT_PIPELINES: &[&str] = super::analytics::PARENT;
/// Sibling pipelines the coordinator runs with the engine's implementation.
pub const ENGINE_SIBLING_PIPELINES: &[&str] = super::analytics::SIBLING;
/// Key of a metric's pipeline payload in shard responses (never rendered).
pub const PIPELINE_PAYLOAD: &str = super::analytics::HIDDEN;

/// Merge the shards' [`PIPELINE_PAYLOAD`]s of one metric: the union of
/// cardinality sets (an HLL++ past a shard's precision threshold), or every
/// shard's percentile values.
pub fn merge_pipeline_payloads(parts: &[&Value]) -> Option<Value> {
    if parts.is_empty() {
        return None;
    }
    // Cardinality sets (arrays, or a sketch once large): their union.
    if parts.iter().all(|p| p.is_array() || p.get("hll").is_some()) {
        let mut set = super::analytics::ValueSet::default();
        parts.iter().for_each(|p| set.add(p));
        return Some(set.payload());
    }
    let flat: Vec<Value> = parts.iter().flat_map(|p| match p.get("parts").and_then(Value::as_array) {
        Some(a) => a.clone(),
        None => vec![(*p).clone()],
    }).collect();
    (!flat.is_empty()).then(|| json!({ "parts": flat }))
}

/// Run one [`ENGINE_PARENT_PIPELINES`] pipeline over merged buckets (plain
/// keys); payloads stay for the caller to strip.
pub fn engine_parent_pipeline(name: &str, t: &str, body: &Value, buckets: &mut [Map<String, Value>]) -> Result<()> {
    super::analytics::parent_pipeline(name, t, body, buckets)
}

/// Run one [`ENGINE_SIBLING_PIPELINES`] pipeline over a level's reduced
/// aggregations (plain keys).
pub fn engine_sibling_pipeline(name: &str, t: &str, body: &Value, results: &HashMap<String, Value>) -> Result<Value> {
    super::analytics::sibling_pipeline(name, t, body, results)
}

/// Coordinator reduce of `categorize_text` buckets carrying their shard
/// categories: each final bucket's fields and its `(node, bucket)` members.
pub fn categorize_reduce(name: &str, conf: &Map<String, Value>, nodes: &[Vec<&Value>]) -> Result<Option<Vec<(Map<String, Value>, Vec<(usize, usize)>)>>> {
    super::analytics::categorize::reduce(name, conf, nodes)
}

struct Rc2 {
    #[allow(dead_code)]
    now: i64,
    max_buckets: usize,
}

// ---------------------------------------------------------------------------
// Partial mode (shard side).

thread_local! {
    /// 0 = off, 1 = partial, 2 = partial and skip pipeline aggregations.
    static MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

pub(super) fn active() -> bool {
    MODE.with(|m| m.get() != 0)
}

pub(super) fn skip_pipeline() -> bool {
    MODE.with(|m| m.get() == 2)
}

struct ModeGuard(u8);

impl ModeGuard {
    fn set(mode: u8) -> ModeGuard {
        ModeGuard(MODE.with(|m| m.replace(mode)))
    }
}

impl Drop for ModeGuard {
    fn drop(&mut self) {
        MODE.with(|m| m.set(self.0));
    }
}

/// Run the aggregations over one shard's segments and return its partial.
pub(crate) fn shard_partial(spec: &Value, segs: &[AggSeg], env: &AggEnv, _req: &AggRequest, skip_pipelines: bool) -> Result<AggPartial> {
    let _g = ModeGuard::set(if skip_pipelines { 2 } else { 1 });
    let tree = run_aggregations(spec, segs, env)?;
    Ok(AggPartial { version: PARTIAL_VERSION, tree })
}

fn terms_sizes(o: &Map<String, Value>) -> (usize, usize) {
    let size = o.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
    let shard = o.get("shard_size").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(size * 3 / 2 + 10).max(size);
    (size, shard)
}

fn nums(ctx: &Ctx, f: &FieldRef, docs: &Docs, missing: &Option<Key>) -> Result<Vec<f64>> {
    ctx.num_values(f, docs, missing)
}

impl<'a, 'b> Ctx<'a, 'b> {
    /// Partial-mode override of one aggregation (`None`: compute it normally).
    pub(super) fn partial_one(&self, name: &str, t: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Option<Value>> {
        Ok(Some(match t {
            "significant_text" => self.significant_text(name, o, sub, docs)?,
            "terms" => {
                let (_, shard) = terms_sizes(o);
                let mut b = o.clone();
                b.insert("size".into(), json!(shard));
                b.insert("min_doc_count".into(), json!(o.get("shard_min_doc_count").and_then(Value::as_u64).unwrap_or(0)));
                self.terms(name, &b, sub, docs)?
            }
            "histogram" | "date_histogram" => {
                let mut b = o.clone();
                if o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(0) > 1 {
                    b.insert("min_doc_count".into(), json!(1));
                }
                let mut v = if t == "histogram" { self.histogram(name, &b, sub, docs)? } else { self.date_histogram(name, &b, sub, docs)? };
                if t == "date_histogram" {
                    let f = self.source(o, name, t)?;
                    let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
                    if let Value::Object(m) = &mut v {
                        m.insert("__p".into(), json!({ "fmt": fmt.spec }));
                    }
                }
                v
            }
            "top_hits" => {
                let from = o.get("from").and_then(Value::as_u64).unwrap_or(0);
                let size = o.get("size").and_then(Value::as_u64).unwrap_or(3);
                let mut b = o.clone();
                b.insert("from".into(), json!(0));
                b.insert("size".into(), json!(from + size));
                self.top_hits(&Value::Object(b), docs)?
            }
            "avg" => {
                let mut v = self.metric(name, t, o, docs)?;
                let f = self.source(o, name, t)?;
                let missing = self.missing_key(&f, o.get("missing"))?;
                let vals = nums(self, &f, docs, &missing)?;
                let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
                let is_date = f.ty.map(|t| t.is_date()).unwrap_or(false);
                set_state(&mut v, json!({ "sum": vals.iter().sum::<f64>(), "count": vals.len(), "date": is_date, "fmt": fmt.spec }));
                v
            }
            "cardinality" => {
                // One sketch for the rendered value and the state.
                let f = self.source(o, name, t)?;
                let missing = self.missing_key(&f, o.get("missing"))?;
                let mut hll = HyperLogLog::new(o.get("precision_threshold").and_then(Value::as_u64).unwrap_or(3000) as usize);
                self.add_value_hashes(&f, docs, &missing, &mut hll);
                let mut v = json!({ "value": hll.count() });
                set_state(&mut v, json!({ "hll": hll.state() }));
                v
            }
            "percentiles" | "percentile_ranks" => {
                // Every reduce of a partial reads the sketch, never the
                // shard's rendered percentiles: skip computing them, keeping
                // the request checks `metric` makes.
                if o.get("hdr").and_then(|h| h.get("number_of_significant_value_digits")).and_then(Value::as_u64).is_some_and(|d| d > 5) {
                    return Err(EsError::illegal_argument("numberOfSignificantValueDigits must be between 0 and 5"));
                }
                if t == "percentile_ranks" && o.get("values").and_then(Value::as_array).is_none() {
                    return Err(EsError::illegal_argument(format!("[values] must not be null: [{name}]")));
                }
                if let (true, Some(Value::Array(a))) = (t == "percentiles", o.get("percents")) {
                    if let Some(p) = a.iter().filter_map(Value::as_f64).find(|p| !(0.0..=100.0).contains(p)) {
                        return Err(EsError::illegal_argument(format!("percent must be in [0,100], got [{p}]: [{name}]")));
                    }
                }
                let mut v = json!({});
                let f = self.source(o, name, t)?;
                let missing = self.missing_key(&f, o.get("missing"))?;
                let comp = o.get("tdigest").and_then(|t| t.get("compression")).and_then(Value::as_f64).unwrap_or(100.0);
                let vals = nums(self, &f, docs, &missing)?;
                let mut td = TDigest::new(comp);
                let mut hdr = o.get("hdr").map(|h| Hdr::new(h.get("number_of_significant_value_digits").and_then(Value::as_u64).unwrap_or(3).min(5) as u32));
                for x in &vals {
                    match &mut hdr {
                        Some(h) => h.add(*x),
                        None => td.add(*x),
                    }
                }
                let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
                let is_date = f.ty.map(|t| t.is_date()).unwrap_or(false);
                let sketch = match &hdr {
                    Some(h) => json!({ "hdr": h.state() }),
                    None => json!({ "td": td.state() }),
                };
                set_state(&mut v, json!({ "sketch": sketch, "n": vals.len(), "date": is_date, "fmt": fmt.spec }));
                v
            }
            "boxplot" | "median_absolute_deviation" => {
                // As for percentiles: the reduce reads only the digest
                // (the checks are `quantile_metric`'s).
                let mut v = json!({});
                let f = self.source(o, name, t)?;
                let missing = self.missing_key(&f, o.get("missing"))?;
                let comp = o.get("compression").and_then(Value::as_f64).unwrap_or(if t == "boxplot" { 100.0 } else { 1000.0 });
                if comp < 0.0 {
                    return Err(EsError::illegal_argument(format!("[compression] must be greater than or equal to 0. Found [{}] in [{name}]", jfmt(comp))));
                }
                if let Some(h) = o.get("execution_hint").and_then(Value::as_str).filter(|h| !matches!(*h, "default" | "high_accuracy")) {
                    return Err(EsError::illegal_argument(format!("Invalid execution_hint [{h}], valid values are [default, high_accuracy]")));
                }
                let mut td = TDigest::new(comp);
                for x in nums(self, &f, docs, &missing)? {
                    td.add(x);
                }
                set_state(&mut v, json!({ "td": td.state() }));
                v
            }
            "geo_bounds" => {
                let mut v = self.geo_nested_agg(name, t, o, sub, docs)?;
                let f = field_name(o, name, t)?;
                // Points and shapes alike (`aggs/geo_shape.rs`).
                if let Some(st) = self.geo_extent(f, docs).state() {
                    set_state(&mut v, st);
                }
                v
            }
            "geotile_grid" | "geohash_grid" | "geohex_grid" => {
                // Shards return `shard_size` cells (ES `suggestShardSideQueueSize`).
                let size = o.get("size").and_then(Value::as_u64).unwrap_or(10_000) as usize;
                let shard = o.get("shard_size").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(size * 3 / 2 + 10).max(size);
                let mut b = o.clone();
                b.insert("size".into(), json!(shard));
                self.geo_nested_agg(name, t, &b, sub, docs)?
            }
            "geo_line" => {
                // Shards keep their sort values for the merge.
                let mut b = o.clone();
                b.insert("include_sort".into(), json!(true));
                self.geo_nested_agg(name, t, &b, sub, docs)?
            }
            "significant_terms" => {
                let size = o.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
                // ES doubles the usual shard queue for significant terms
                // (`2 * suggestShardSideQueueSize(size)`): a significant term
                // may have few docs on each shard.
                let shard = o.get("shard_size").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(2 * (size * 3 / 2 + 10)).max(size);
                let mut b = o.clone();
                b.insert("size".into(), json!(shard));
                b.insert("min_doc_count".into(), json!(o.get("shard_min_doc_count").and_then(Value::as_u64).unwrap_or(1).max(1)));
                // A shard's candidates are its best `shard_size` by its own
                // score, kept whatever their sign: the reduce scores them
                // against every shard's counts and drops the non-positive
                // ones there (ES filters `score > 0` only at the reduce).
                b.insert(SHARD_CANDIDATES.into(), json!(true));
                let mut v = self.extra_agg(name, t, &b, sub, docs)?;
                // Background counts of every term: a term absent from this
                // shard's candidates still has background docs here.
                // Counted once per shard request (the same for every parent
                // bucket), shared with significant_terms' own scoring.
                let f = self.field(field_name(o, name, t)?)?;
                let bg_key = format!("{}\n{}", field_name(o, name, t)?, o.get("background_filter").map(|q| q.to_string()).unwrap_or_default());
                let background = self.background_counts(bg_key, || {
                    let all: Docs = (0..self.segs.len()).map(|s| (self.env.all_docs)(s).into_iter().map(|d| (d, 0.0)).collect()).collect();
                    let bg: Docs = match o.get("background_filter") {
                        Some(q) => self.filter_docs(q, &all)?,
                        None => all,
                    };
                    Ok((self.count_by(&f, &bg, &None)?, count(&bg)))
                })?;
                let groups = &background.0;
                let list: Vec<Value> = if groups.len() <= 100_000 { groups.iter().map(|(k, n)| json!([key_json(k), n])).collect() } else { Vec::new() };
                set_state(&mut v, json!({ "bg": list }));
                v
            }
            _ => return Ok(None),
        }))
    }
}

fn set_state(v: &mut Value, st: Value) {
    if let Value::Object(m) = v {
        m.insert("__p".into(), st);
    }
}

// ---------------------------------------------------------------------------
// Reduce (coordinator side).

fn cmp_json(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Number(x), Value::Number(y)) => x.as_f64().unwrap_or(0.0).total_cmp(&y.as_f64().unwrap_or(0.0)),
        (Value::String(x), Value::String(y)) => x.as_bytes().cmp(y.as_bytes()),
        (Value::String(_), _) => Ordering::Greater,
        (_, Value::String(_)) => Ordering::Less,
        (x, y) => x.to_string().cmp(&y.to_string()),
    }
}

fn num(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

fn count_of(v: &Value, k: &str) -> u64 {
    v.get(k).map(|c| c.as_u64().unwrap_or_else(|| c.as_f64().unwrap_or(0.0) as u64)).unwrap_or(0)
}

fn state<'v>(v: &'v Value) -> Option<&'v Value> {
    v.get("__p")
}

fn parse_entry<'s>(name: &'s String, def: &'s Value) -> Option<(String, &'s Value, Option<&'s Map<String, Value>>, Option<&'s Value>)> {
    let _ = name;
    let def = def.as_object()?;
    let mut ty: Option<(&String, &Value)> = None;
    let mut sub = None;
    let mut meta = None;
    for (k, v) in def {
        match k.as_str() {
            "aggs" | "aggregations" => sub = v.as_object(),
            "meta" => meta = Some(v),
            _ => ty = Some((k, v)),
        }
    }
    let (t, body) = ty?;
    Some((t.clone(), body, sub, meta))
}

/// The aggregation types [`reduce_one`] reduces across shards.
const REDUCIBLE: &[&str] = &["frequent_item_sets", "frequent_items", "min", "max", "sum", "value_count", "stats", "extended_stats", "avg",
    "cardinality", "percentiles", "percentile_ranks", "top_hits", "scripted_metric", "terms", "histogram", "date_histogram", "range", "date_range",
    "ip_range", "geo_distance", "filters", "filter", "global", "missing", "nested", "reverse_nested", "sampler", "diversified_sampler", "children",
    "parent", "composite", "significant_terms", "significant_text", "geohash_grid", "geotile_grid", "geohex_grid", "geo_line", "cartesian_bounds",
    "cartesian_centroid", "geo_centroid", "geo_bounds"];

/// Whether every aggregation of `spec` (sub-aggregations included) has a
/// cross-shard reduce: otherwise per-shard partials would be built only to
/// be thrown away for the union path (`multi_terms` aggregated twice).
pub fn reducible(spec: &Value) -> bool {
    let Some(spec) = spec.as_object() else { return false };
    spec.iter().all(|(name, def)| match parse_entry(name, def) {
        Some((t, _, sub, _)) => {
            (PARENT_PIPELINES.contains(&t.as_str()) || SIBLING_PIPELINES.contains(&t.as_str()) || REDUCIBLE.contains(&t.as_str()))
                && sub.is_none_or(|m| reducible(&Value::Object(m.clone())))
        }
        None => false,
    })
}

/// Reduce one level of the aggregation tree: `spec` is the request's aggs
/// map, `parts` the matching partial maps from every shard.
fn reduce_level(spec: &Map<String, Value>, parts: &[&Value], rc: &Rc2) -> Result<Map<String, Value>> {
    let mut results: HashMap<String, Value> = HashMap::new();
    let mut entries: Vec<(&String, String, &Value, Option<&Map<String, Value>>, Option<&Value>)> = Vec::new();
    for (name, def) in spec {
        let (t, body, sub, meta) = parse_entry(name, def).ok_or_else(|| EsError::parsing_at(format!("Missing definition for aggregation [{name}]"), 1, 1))?;
        entries.push((name, t, body, sub, meta));
    }
    for (name, t, body, sub, _) in &entries {
        if PARENT_PIPELINES.contains(&t.as_str()) || SIBLING_PIPELINES.contains(&t.as_str()) {
            continue;
        }
        let vals: Vec<&Value> = parts.iter().filter_map(|p| p.get(name.as_str())).collect();
        let v = reduce_one(name, t, body, *sub, &vals, rc)?;
        results.insert((*name).clone(), v);
    }
    for (name, t, body, _, _) in &entries {
        if SIBLING_PIPELINES.contains(&t.as_str()) {
            let r = sibling_pipeline(name, t, body, &results)?;
            results.insert((*name).clone(), r);
        }
    }
    let mut out = Map::new();
    for (name, t, _, _, meta) in entries {
        if PARENT_PIPELINES.contains(&t.as_str()) {
            continue;
        }
        let mut v = results.remove(name).unwrap_or(Value::Null);
        if let (Some(m), Value::Object(o)) = (meta, &mut v) {
            let mut with_meta = Map::new();
            with_meta.insert("meta".into(), m.clone());
            for (k, x) in std::mem::take(o) {
                with_meta.insert(k, x);
            }
            *o = with_meta;
        }
        out.insert(name.clone(), v);
    }
    Ok(out)
}

fn strip_state(v: &Value) -> Value {
    let mut v = v.clone();
    if let Value::Object(m) = &mut v {
        m.shift_remove("__p");
    }
    v
}

fn reduce_one(name: &str, t: &str, body: &Value, sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let conf = body.as_object().cloned().unwrap_or_default();
    Ok(match t {
        "frequent_item_sets" | "frequent_items" => super::frequent_items::reduce(&conf, vals, !crate::search::aggregation_shard_response())?,
        "min" | "max" => reduce_minmax(t, vals),
        "sum" => {
            let sum: f64 = vals.iter().filter_map(|v| num(v, "value")).sum();
            let mut m = Map::new();
            m.insert("value".into(), fnum(sum));
            let sample = vals.iter().find_map(|v| Some((num(v, "value")?, v.get("value_as_string")?.as_str()?)));
            if let Some(f) = fmt_like(&conf, sample) {
                m.insert("value_as_string".into(), json!(f.format(sum as i64, Tz::UTC)));
            }
            Value::Object(m)
        }
        "value_count" => json!({ "value": vals.iter().map(|v| count_of(v, "value")).sum::<u64>() }),
        "stats" | "extended_stats" => reduce_stats(name, t, &conf, vals)?,
        "avg" => reduce_avg(vals),
        "cardinality" => reduce_cardinality(vals),
        "percentiles" | "percentile_ranks" => reduce_percentiles(name, t, &conf, vals)?,
        "top_hits" => reduce_top_hits(&conf, vals),
        "scripted_metric" => super::scripted::reduce_partials(name, &conf, vals)?,
        "terms" => reduce_terms(&conf, sub, vals, rc)?,
        "histogram" | "date_histogram" => reduce_histogram(name, t, &conf, sub, vals, rc)?,
        "range" | "date_range" | "ip_range" | "geo_distance" | "filters" => reduce_fixed(sub, vals, rc)?,
        "filter" | "global" | "missing" | "nested" | "reverse_nested" | "sampler" | "diversified_sampler" | "children" | "parent" => Value::Object(merge_bucket(vals, sub, rc)?),
        "composite" => reduce_composite(&conf, sub, vals, rc)?,
        "significant_terms" => reduce_significant(&conf, sub, vals, rc)?,
        "significant_text" => super::significant_text::reduce(&conf, vals, rc.max_buckets, !crate::search::aggregation_shard_response())?,
        // geohex keys are fixed-width hex: string order is H3 index order.
        "geohash_grid" | "geotile_grid" | "geohex_grid" => reduce_grid(&conf, t == "geotile_grid", sub, vals, rc)?,
        "geo_line" => reduce_geo_line(&conf, vals),
        "cartesian_bounds" => reduce_cartesian_bounds(vals),
        "cartesian_centroid" => reduce_cartesian_centroid(vals),
        "geo_centroid" => reduce_geo_centroid(vals),
        "geo_bounds" => reduce_geo_bounds(&conf, vals),
        _ => match vals.len() {
            0 => Value::Null,
            1 => strip_state(vals[0]),
            _ => {
                return Err(EsError::illegal_argument(format!(
                    "aggregation [{name}] of type [{t}] does not support a cross-shard reduce yet; it can only run on a single shard"
                )))
            }
        },
    })
}

fn reduce_minmax(t: &str, vals: &[&Value]) -> Value {
    let mut best: Option<(&Value, f64)> = None;
    for v in vals {
        if let Some(x) = num(v, "value") {
            if best.map_or(true, |(_, y)| if t == "min" { x < y } else { x > y }) {
                best = Some((*v, x));
            }
        }
    }
    best.map(|b| b.0.clone()).or_else(|| vals.first().map(|v| (*v).clone())).unwrap_or_else(|| json!({ "value": null }))
}

/// A formatter that reproduces `sample` (`(millis, rendered)`) from `conf`'s
/// `format` or the usual defaults.
fn fmt_like(conf: &Map<String, Value>, sample: Option<(f64, &str)>) -> Option<DateFormatter> {
    let (ms, want) = sample?;
    let mut first = None;
    for spec in [conf.get("format").and_then(Value::as_str), Some("strict_date_optional_time"), Some("strict_date_optional_time_nanos"), Some("epoch_millis"), Some("epoch_second")].into_iter().flatten() {
        if let Ok(f) = DateFormatter::new(spec) {
            if f.format(ms as i64, Tz::UTC) == want {
                return Some(f);
            }
            first.get_or_insert(f);
        }
    }
    first
}

fn reduce_stats(name: &str, t: &str, conf: &Map<String, Value>, vals: &[&Value]) -> Result<Value> {
    let live: Vec<&Value> = vals.iter().copied().filter(|v| count_of(v, "count") > 0).collect();
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
    let ok = n > 0;
    let mut m = Map::new();
    m.insert("count".into(), json!(n));
    m.insert("min".into(), opt_num(min.map(|x| x.1)));
    m.insert("max".into(), opt_num(max.map(|x| x.1)));
    let avg = (n > 0).then(|| sum / n as f64);
    m.insert("avg".into(), opt_num(avg));
    m.insert("sum".into(), fnum(sum));
    if t == "extended_stats" {
        let sigma = conf.get("sigma").and_then(Value::as_f64).unwrap_or(2.0);
        if sigma < 0.0 {
            return Err(EsError::illegal_argument(format!("[sigma] must be greater than or equal to 0. Found [{sigma}] in [{name}]")));
        }
        let sos: f64 = live.iter().filter_map(|v| num(v, "sum_of_squares")).sum();
        let nf = n as f64;
        let a = sum / nf;
        let var_p = (sos / nf - a * a).max(0.0);
        let var_s = if n > 1 { (sos - nf * a * a).max(0.0) / (nf - 1.0) } else { f64::NAN };
        let o2 = |v: f64| if ok && v.is_finite() { fnum(v) } else { Value::Null };
        m.insert("sum_of_squares".into(), if ok { fnum(sos) } else { Value::Null });
        m.insert("variance".into(), o2(var_p));
        m.insert("variance_population".into(), o2(var_p));
        m.insert("variance_sampling".into(), o2(var_s));
        m.insert("std_deviation".into(), o2(var_p.sqrt()));
        m.insert("std_deviation_population".into(), o2(var_p.sqrt()));
        m.insert("std_deviation_sampling".into(), o2(var_s.sqrt()));
        m.insert(
            "std_deviation_bounds".into(),
            json!({
                "upper": o2(a + sigma * var_p.sqrt()),
                "lower": o2(a - sigma * var_p.sqrt()),
                "upper_population": o2(a + sigma * var_p.sqrt()),
                "lower_population": o2(a - sigma * var_p.sqrt()),
                "upper_sampling": o2(a + sigma * var_s.sqrt()),
                "lower_sampling": o2(a - sigma * var_s.sqrt()),
            }),
        );
    }
    if ok && live.iter().any(|v| v.get("min_as_string").is_some()) {
        for (k, src) in [("min_as_string", min), ("max_as_string", max)] {
            if let Some(s) = src.and_then(|x| x.0.get(k)) {
                m.insert(k.into(), s.clone());
            }
        }
        let sample = min.and_then(|x| Some((x.1, x.0.get("min_as_string")?.as_str()?)));
        if let Some(f) = fmt_like(conf, sample) {
            for (k, x) in [("avg", avg.unwrap_or(0.0)), ("sum", sum)] {
                m.insert(format!("{k}_as_string"), json!(f.format(x as i64, Tz::UTC)));
            }
        }
    }
    // Key order of the single-shard rendering.
    let order = ["count", "min", "max", "avg", "sum"];
    let mut out = Map::new();
    for k in order {
        if let Some(v) = m.shift_remove(k) {
            out.insert(k.into(), v);
        }
    }
    out.extend(m);
    Ok(Value::Object(out))
}

fn reduce_avg(vals: &[&Value]) -> Value {
    let (mut n, mut s) = (0u64, 0f64);
    let mut date = None;
    for v in vals {
        if let Some(p) = state(v) {
            n += count_of(p, "count");
            s += num(p, "sum").unwrap_or(0.0);
            if p["date"].as_bool() == Some(true) {
                date = p["fmt"].as_str().map(str::to_string);
            }
        }
    }
    let avg = (n > 0).then(|| s / n as f64);
    let mut m = Map::new();
    m.insert("value".into(), opt_num(avg));
    if let (Some(x), Some(spec)) = (avg, date) {
        if let Ok(f) = DateFormatter::new(&spec) {
            m.insert("value_as_string".into(), json!(f.format(x as i64, Tz::UTC)));
        }
    }
    Value::Object(m)
}

fn reduce_cardinality(vals: &[&Value]) -> Value {
    let mut acc: Option<HyperLogLog> = None;
    for v in vals {
        let Some(st) = state(v).and_then(|p| p.get("hll")).and_then(|h| serde_json::from_value::<HllState>(h.clone()).ok()) else { continue };
        let h = HyperLogLog::from_state(&st);
        match &mut acc {
            None => acc = Some(h),
            Some(a) => a.merge(&h),
        }
    }
    json!({ "value": acc.map(|a| a.count()).unwrap_or(0) })
}

fn reduce_percentiles(name: &str, t: &str, o: &Map<String, Value>, vals: &[&Value]) -> Result<Value> {
    let mut acc: Option<TDigest> = None;
    let mut hdr_parts: Vec<Hdr> = Vec::new();
    let (mut n, mut date, mut fmt) = (0u64, false, String::new());
    for v in vals {
        let Some(p) = state(v) else { continue };
        n += count_of(p, "n");
        date = p["date"].as_bool().unwrap_or(false);
        fmt = p["fmt"].as_str().unwrap_or_default().to_string();
        if let Some(st) = p["sketch"].get("hdr").and_then(|h| serde_json::from_value::<HdrState>(h.clone()).ok()) {
            hdr_parts.push(Hdr::from_state(&st));
        } else if let Some(st) = p["sketch"].get("td").and_then(|h| serde_json::from_value::<TDigestState>(h.clone()).ok()) {
            let mut d = TDigest::from_state(&st);
            match &mut acc {
                None => acc = Some(d),
                Some(a) => a.merge(&mut d),
            }
        }
    }
    // ES reduce: a lone shard result is used as is; otherwise a fresh
    // histogram shaped like the first state gets every state `add`ed.
    let hdr_acc: Option<Hdr> = if hdr_parts.len() <= 1 {
        hdr_parts.pop()
    } else {
        let mut m = hdr_parts[0].empty_like();
        hdr_parts.iter().for_each(|h| m.merge(h));
        Some(m)
    };
    let mut td = acc.unwrap_or_else(|| TDigest::new(o.get("tdigest").and_then(|t| t.get("compression")).and_then(Value::as_f64).unwrap_or(100.0)));
    let df = if date { DateFormatter::new(&fmt).ok() } else { None };
    let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(true);
    let points: Vec<f64> = if t == "percentiles" {
        match o.get("percents") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_f64).collect(),
            _ => vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0],
        }
    } else {
        o.get("values").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).ok_or_else(|| EsError::illegal_argument(format!("[values] must not be null: [{name}]")))?
    };
    let mut entries: Vec<(f64, Option<f64>)> = Vec::new();
    for p in points {
        let v = match (&hdr_acc, t == "percentiles") {
            (_, _) if n == 0 => None,
            (Some(h), true) => h.value_at_percentile(p),
            (Some(h), false) => h.percentile_at_or_below(p),
            (None, true) => td.quantile(p / 100.0),
            (None, false) => td.cdf(p).map(|c| c * 100.0),
        };
        entries.push((p, v));
    }
    Ok(if keyed {
        let mut m = Map::new();
        for (p, v) in &entries {
            m.insert(jfmt(*p), opt_num(*v));
            if let (Some(v), Some(f)) = (v, &df) {
                if t == "percentiles" {
                    m.insert(format!("{}_as_string", jfmt(*p)), json!(f.format(*v as i64, Tz::UTC)));
                }
            }
        }
        json!({ "values": m })
    } else {
        json!({ "values": entries.iter().map(|(p, v)| json!({"key": fnum(*p), "value": opt_num(*v)})).collect::<Vec<_>>() })
    })
}

/// The shards' merged t-digest of a `boxplot` / `median_absolute_deviation`.
fn merged_digest(vals: &[&Value]) -> Option<TDigest> {
    let mut acc: Option<TDigest> = None;
    for v in vals {
        let Some(st) = state(v).and_then(|p| p.get("td")).and_then(|h| serde_json::from_value::<TDigestState>(h.clone()).ok()) else { continue };
        let mut d = TDigest::from_state(&st);
        match &mut acc {
            None => acc = Some(d),
            Some(a) => a.merge(&mut d),
        }
    }
    acc
}

/// ES `InternalBoxplot` / `InternalMedianAbsoluteDeviation` over the merged
/// digest: quartiles from it, whiskers and deviations from its centroids
/// (every value while the digest still keeps exact samples).
fn reduce_quantile_metric(t: &str, vals: &[&Value]) -> Value {
    let Some(mut td) = merged_digest(vals).filter(|d| d.count() > 0) else {
        return if t == "boxplot" {
            json!({"min": "Infinity", "max": "-Infinity", "q1": "NaN", "q2": "NaN", "q3": "NaN", "lower": "NaN", "upper": "-Infinity"})
        } else {
            json!({"value": null})
        };
    };
    let st = td.state();
    if t == "median_absolute_deviation" {
        let med = td.quantile(0.5).unwrap_or(0.0);
        let mut dev: Vec<(f64, f64)> = st.means.iter().zip(&st.weights).map(|(m, w)| ((m - med).abs(), *w)).collect();
        dev.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (means, weights): (Vec<f64>, Vec<f64>) = dev.into_iter().unzip();
        let (min, max) = (means.first().copied(), means.last().copied());
        let mut d = TDigest::from_state(&TDigestState { compression: st.compression, count: weights.iter().sum(), means, weights, min, max });
        return json!({"value": opt_num(d.quantile(0.5))});
    }
    let (min, max) = (st.min.unwrap_or(f64::NAN), st.max.unwrap_or(f64::NAN));
    let q1 = td.quantile(0.25).unwrap_or(min);
    let q2 = td.quantile(0.5).unwrap_or(min);
    let q3 = td.quantile(0.75).unwrap_or(max);
    let iqr = q3 - q1;
    let lower = st.means.iter().copied().filter(|v| *v >= q1 - 1.5 * iqr).fold(f64::INFINITY, f64::min).max(min);
    let upper = st.means.iter().copied().filter(|v| *v <= q3 + 1.5 * iqr).fold(f64::NEG_INFINITY, f64::max).min(max);
    json!({"min": fnum(min), "max": fnum(max), "q1": fnum(q1), "q2": fnum(q2), "q3": fnum(q3), "lower": fnum(lower), "upper": fnum(upper)})
}

/// `t_test` over the shards' summed (count, sum, sum of squares).
fn reduce_t_test(vals: &[&Value]) -> Value {
    let first = state(vals[0]).unwrap();
    let ty = first["type"].as_str().unwrap_or("heteroscedastic");
    let tails = first["tails"].as_i64().unwrap_or(2);
    let side = |k: &str| {
        vals.iter().filter_map(|v| state(v)?.get(k)?.as_array().cloned()).fold((0u64, 0.0, 0.0), |acc, a| {
            (acc.0 + a.first().and_then(Value::as_u64).unwrap_or(0), acc.1 + a.get(1).and_then(Value::as_f64).unwrap_or(0.0), acc.2 + a.get(2).and_then(Value::as_f64).unwrap_or(0.0))
        })
    };
    super::extra::t_test_value(ty, tails, side("a"), side("b"))
}

/// `matrix_stats` over the shards' merged moments (ES `RunningStats.merge`).
fn reduce_matrix_stats(name: &str, vals: &[&Value]) -> Result<Value> {
    let first = state(vals[0]).unwrap();
    let fields: Vec<String> = first["fields"].as_array().map(|a| a.iter().filter_map(|f| f.as_str().map(str::to_string)).collect()).unwrap_or_default();
    // Older partials have no coverage metadata: retain their compatibility
    // behavior rather than inventing unmapped fields during rolling upgrades.
    if vals.iter().all(|v| state(v).is_some_and(|p| p["mapped"].is_array() && p["indices"].is_array())) {
        let mut indices = std::collections::BTreeSet::new();
        let mut any_mapped = false;
        let mut missing = vec![false; fields.len()];
        for v in vals {
            let p = state(v).unwrap();
            for index in p["indices"].as_array().unwrap().iter().filter_map(Value::as_str) {
                indices.insert(index);
            }
            for (i, absent) in missing.iter_mut().enumerate() {
                let mapped = p["mapped"][i].as_bool().unwrap_or(false);
                any_mapped |= mapped;
                *absent |= !mapped;
            }
        }
        if indices.len() > 1 && any_mapped && missing.iter().any(|v| *v) {
            let missing = fields.iter().zip(missing).filter_map(|(f, absent)| absent.then_some(f.as_str())).collect::<Vec<_>>().join(", ");
            return Err(EsError::illegal_argument(format!("Aggregation [{name}] all fields must exist in all indices, but some indices are missing these fields [{missing}]")));
        }
    }
    let mut acc = super::analytics::Moments::new(fields.len());
    for v in vals {
        if let Some(m) = state(v).and_then(|p| p.get("moments")).and_then(super::analytics::Moments::from_state) {
            acc.merge(&m);
        }
    }
    Ok(super::analytics::render_matrix_stats(&fields, &acc))
}

fn reduce_top_hits(conf: &Map<String, Value>, vals: &[&Value]) -> Value {
    let from = conf.get("from").and_then(Value::as_u64).unwrap_or(0) as usize;
    let size = conf.get("size").and_then(Value::as_u64).unwrap_or(3) as usize;
    let mut crit: Vec<(String, bool)> = Vec::new();
    if let Some(s) = conf.get("sort") {
        let items: Vec<Value> = match s {
            Value::Array(a) => a.clone(),
            other => vec![other.clone()],
        };
        for it in items {
            match it {
                Value::String(f) => crit.push((f.clone(), f == "_score")),
                Value::Object(m) => {
                    for (f, v) in m {
                        let desc = match v {
                            Value::String(s) => s == "desc",
                            Value::Object(x) => x.get("order").and_then(Value::as_str).map(|s| s == "desc").unwrap_or(f == "_score"),
                            _ => false,
                        };
                        crit.push((f.clone(), desc));
                    }
                }
                _ => {}
            }
        }
    }
    let by_score = crit.is_empty() || (crit.len() == 1 && crit[0].0 == "_score");
    let mut hits: Vec<(usize, Value)> = Vec::new();
    let (mut total, mut max_score) = (0u64, None::<f64>);
    for (si, v) in vals.iter().enumerate() {
        let h = &v["hits"];
        total += h["total"]["value"].as_u64().unwrap_or(0);
        if let Some(s) = h["max_score"].as_f64() {
            max_score = Some(max_score.map_or(s, |m| m.max(s)));
        }
        hits.extend(h["hits"].as_array().cloned().unwrap_or_default().into_iter().map(|x| (si, x)));
    }
    hits.sort_by(|(sa, a), (sb, b)| {
        if by_score {
            return b["_score"].as_f64().unwrap_or(0.0).total_cmp(&a["_score"].as_f64().unwrap_or(0.0)).then(sa.cmp(sb));
        }
        for (i, (_, desc)) in crit.iter().enumerate() {
            let (x, y) = (&a["sort"][i], &b["sort"][i]);
            let c = match (x, y) {
                (Value::Null, Value::Null) => Ordering::Equal,
                (Value::Null, _) => Ordering::Greater,
                (_, Value::Null) => Ordering::Less,
                _ => {
                    let c = cmp_json(x, y);
                    if *desc { c.reverse() } else { c }
                }
            };
            if c != Ordering::Equal {
                return c;
            }
        }
        sa.cmp(sb)
    });
    let page: Vec<Value> = hits.into_iter().skip(from).take(size).map(|(_, h)| h).collect();
    json!({ "hits": { "total": { "value": total, "relation": "eq" }, "max_score": if by_score { max_score.map(fnum).unwrap_or(Value::Null) } else { Value::Null }, "hits": page } })
}

/// Merge the same bucket from several shards: `doc_count` summed,
/// sub-aggregations reduced, everything else (key, from, to, ...) from the first.
fn merge_bucket(bs: &[&Value], sub: Option<&Map<String, Value>>, rc: &Rc2) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    if let Some(first) = bs.first().and_then(|b| b.as_object()) {
        for (k, v) in first {
            if k != "doc_count" && k != "__p" && !sub.is_some_and(|s| s.contains_key(k)) {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    // `doc_count` right after the key fields, then the sub-aggregations.
    out.insert("doc_count".into(), json!(bs.iter().map(|b| count_of(b, "doc_count")).sum::<u64>()));
    if let Some(s) = sub {
        out.extend(reduce_level(s, bs, rc)?);
    }
    Ok(out)
}

fn buckets_of(v: &Value) -> Vec<&Value> {
    match &v["buckets"] {
        Value::Array(a) => a.iter().collect(),
        Value::Object(m) => m.values().collect(),
        _ => Vec::new(),
    }
}

fn cmp_buckets(a: &Map<String, Value>, b: &Map<String, Value>, order: &[(String, bool)]) -> Ordering {
    for (k, asc) in order {
        let c = match k.as_str() {
            "_count" => count_of(&Value::Object(a.clone()), "doc_count").cmp(&count_of(&Value::Object(b.clone()), "doc_count")),
            "_key" => cmp_json(&a["key"], &b["key"]),
            path => match (resolve_path(a, path), resolve_path(b, path)) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (None, Some(_)) => return Ordering::Greater,
                (Some(_), None) => return Ordering::Less,
                _ => Ordering::Equal,
            },
        };
        let c = if *asc { c } else { c.reverse() };
        if c != Ordering::Equal {
            return c;
        }
    }
    Ordering::Equal
}

fn reduce_terms(conf: &Map<String, Value>, sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let (size, shard_size) = terms_sizes(conf);
    let min_doc_count = conf.get("min_doc_count").and_then(Value::as_u64).unwrap_or(1);
    let order = parse_order(conf.get("order"), &[("_count", false)])?;
    let show_err = conf.get("show_term_doc_count_error").and_then(Value::as_bool).unwrap_or(false);
    let key_order = order[0].0 == "_key";
    let count_desc = order[0].0 == "_count" && !order[0].1;
    let mut groups: Vec<(Vec<&Value>, Vec<usize>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut shard_err: Vec<i64> = Vec::new();
    let mut other = 0u64;
    for (i, v) in vals.iter().enumerate() {
        let bs = buckets_of(v);
        other += count_of(v, "sum_other_doc_count");
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
        shard_err.push(if own < 0 || trunc < 0 { -1 } else { own + trunc });
        for b in bs {
            let g = *index.entry(b["key"].to_string()).or_insert_with(|| {
                groups.push((Vec::new(), Vec::new()));
                groups.len() - 1
            });
            groups[g].0.push(b);
            groups[g].1.push(i);
        }
    }
    let mut buckets: Vec<Map<String, Value>> = Vec::new();
    for (bs, shards) in &groups {
        let mut m = merge_bucket(bs, sub, rc)?;
        m.shift_remove("doc_count_error_upper_bound");
        if show_err {
            let mut e = 0i64;
            for (i, se) in shard_err.iter().enumerate() {
                let add = match shards.iter().position(|n| *n == i) {
                    Some(p) => bs[p]["doc_count_error_upper_bound"].as_i64().unwrap_or(0),
                    None => *se,
                };
                e = if e < 0 || add < 0 { -1 } else { e + add };
            }
            m.insert("doc_count_error_upper_bound".into(), json!(e));
        }
        if count_of(&Value::Object(m.clone()), "doc_count") >= min_doc_count {
            buckets.push(m);
        }
    }
    buckets.sort_by(|a, b| cmp_buckets(a, b, &order));
    if buckets.len() > size {
        other += buckets[size..].iter().map(|b| count_of(&Value::Object(b.clone()), "doc_count")).sum::<u64>();
        buckets.truncate(size);
    }
    apply_parent_pipelines(sub, &mut buckets)?;
    let err = shard_err.iter().fold(0i64, |a, e| if a < 0 || *e < 0 { -1 } else { a + e });
    Ok(json!({ "doc_count_error_upper_bound": err, "sum_other_doc_count": other, "buckets": buckets }))
}

fn reduce_histogram(name: &str, t: &str, conf: &Map<String, Value>, sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let min_doc_count = conf.get("min_doc_count").and_then(Value::as_u64).unwrap_or(0);
    let keyed = conf.get("keyed").and_then(Value::as_bool).unwrap_or(false);
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut index: HashMap<u64, usize> = HashMap::new();
    for v in vals {
        for b in buckets_of(v) {
            let k = b["key"].as_f64().unwrap_or(0.0);
            let g = *index.entry(k.to_bits()).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(b);
        }
    }
    let mut buckets: Vec<Map<String, Value>> = Vec::new();
    for bs in &groups {
        buckets.push(merge_bucket(bs, sub, rc)?);
    }
    buckets.sort_by(|a, b| cmp_json(&a["key"], &b["key"]));
    let date = t == "date_histogram";
    let fmt = vals.iter().find_map(|v| state(v)).and_then(|p| p["fmt"].as_str()).and_then(|s| DateFormatter::new(s).ok());
    if min_doc_count == 0 && buckets.len() >= 2 {
        // Fill the gaps between buckets of different shards.
        enum Step {
            Num { interval: f64, offset: f64 },
            Date(Rounding),
        }
        let step = if date {
            Step::Date(Rounding::parse(conf, name)?)
        } else {
            Step::Num { interval: conf.get("interval").and_then(Value::as_f64).unwrap_or(1.0), offset: conf.get("offset").and_then(Value::as_f64).unwrap_or(0.0) }
        };
        let next = |k: f64| -> f64 {
            match &step {
                Step::Num { interval, .. } => k + interval,
                Step::Date(r) => r.next(k as i64) as f64,
            }
        };
        let _ = |s: &Step| matches!(s, Step::Num { offset, .. } if *offset == 0.0);
        let int_keys = buckets[0]["key"].is_i64() || buckets[0]["key"].is_u64();
        let mut empty = Map::new();
        empty.insert("doc_count".into(), json!(0));
        if let Some(s) = sub {
            empty.extend(reduce_level(s, &[], rc)?);
        }
        let mut filled: Vec<Map<String, Value>> = Vec::with_capacity(buckets.len());
        for b in buckets {
            if let Some(prev) = filled.last().and_then(|p| p["key"].as_f64()) {
                let target = b["key"].as_f64().unwrap_or(prev);
                let mut k = next(prev);
                let eps = (k - prev).abs() * 1e-9;
                while k < target - eps {
                    if filled.len() > rc.max_buckets {
                        return Err(EsError::too_many_buckets(rc.max_buckets));
                    }
                    let mut e = Map::new();
                    if date {
                        e.insert("key_as_string".into(), json!(fmt.clone().unwrap_or_default().format(k as i64, match Rounding::parse(conf, name) { Ok(r) => r.tz, Err(_) => Tz::UTC })));
                        e.insert("key".into(), json!(k as i64));
                    } else {
                        e.insert("key".into(), if int_keys { json!(k as i64) } else { fnum(k) });
                    }
                    e.extend(empty.clone());
                    filled.push(e);
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
    buckets.retain(|b| count_of(&Value::Object(b.clone()), "doc_count") >= min_doc_count);
    let order = parse_order(conf.get("order"), &[("_key", true)])?;
    buckets.sort_by(|a, b| cmp_buckets(a, b, &order));
    apply_parent_pipelines(sub, &mut buckets)?;
    if keyed {
        let mut km = Map::new();
        for m in buckets {
            let k = m.get("key_as_string").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| match &m["key"] {
                Value::Number(n) => n.as_f64().map(jfmt).unwrap_or_default(),
                v => v.to_string(),
            });
            km.insert(k, Value::Object(m));
        }
        return Ok(json!({ "buckets": km }));
    }
    Ok(json!({ "buckets": buckets }))
}

/// Aggregations whose bucket list is fixed by the request (`filters`,
/// ranges): merged by position, or by name when keyed.
fn reduce_fixed(sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let Some(first) = vals.first() else { return Ok(json!({ "buckets": [] })) };
    let mut out = first.as_object().cloned().unwrap_or_default();
    out.shift_remove("__p");
    let merged = match &first["buckets"] {
        Value::Object(m) => {
            let mut o = Map::new();
            for k in m.keys() {
                let bs: Vec<&Value> = vals.iter().filter_map(|v| v["buckets"].get(k)).collect();
                o.insert(k.clone(), Value::Object(merge_bucket(&bs, sub, rc)?));
            }
            Value::Object(o)
        }
        _ => {
            let n = vals.iter().map(|v| v["buckets"].as_array().map(|a| a.len()).unwrap_or(0)).max().unwrap_or(0);
            let mut arr = Vec::new();
            for i in 0..n {
                let bs: Vec<&Value> = vals.iter().filter_map(|v| v["buckets"].get(i)).collect();
                arr.push(Value::Object(merge_bucket(&bs, sub, rc)?));
            }
            // Parent pipelines run over the merged list.
            let mut maps: Vec<Map<String, Value>> = arr.iter().filter_map(|b| b.as_object().cloned()).collect();
            apply_parent_pipelines(sub, &mut maps)?;
            Value::Array(maps.into_iter().map(Value::Object).collect())
        }
    };
    out.insert("buckets".into(), merged);
    Ok(Value::Object(out))
}

fn reduce_composite(conf: &Map<String, Value>, sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let size = conf.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
    // Per source: name, descending.
    let mut srcs: Vec<(String, bool)> = Vec::new();
    for s in conf.get("sources").and_then(Value::as_array).cloned().unwrap_or_default() {
        if let Some((sname, def)) = s.as_object().and_then(|m| m.iter().next()) {
            let desc = def.as_object().and_then(|d| d.values().next()).and_then(|d| d.get("order")).and_then(Value::as_str) == Some("desc");
            srcs.push((sname.clone(), desc));
        }
    }
    let cmp = |a: &Value, b: &Value| -> Ordering {
        for (n, desc) in &srcs {
            let (x, y) = (&a[n.as_str()], &b[n.as_str()]);
            let c = cmp_json(x, y);
            let c = if *desc && !x.is_null() && !y.is_null() { c.reverse() } else { c };
            if c != Ordering::Equal {
                return c;
            }
        }
        Ordering::Equal
    };
    let mut groups: Vec<(Value, Vec<&Value>)> = Vec::new();
    for v in vals {
        for b in buckets_of(v) {
            match groups.iter_mut().find(|(k, _)| cmp(k, &b["key"]) == Ordering::Equal) {
                Some((_, g)) => g.push(b),
                None => groups.push((b["key"].clone(), vec![b])),
            }
        }
    }
    groups.sort_by(|a, b| cmp(&a.0, &b.0));
    groups.truncate(size);
    let mut buckets = Vec::new();
    for (_, bs) in &groups {
        buckets.push(merge_bucket(bs, sub, rc)?);
    }
    apply_parent_pipelines(sub, &mut buckets)?;
    let mut r = Map::new();
    if let Some(l) = buckets.last() {
        r.insert("after_key".into(), l["key"].clone());
    }
    r.insert("buckets".into(), Value::Array(buckets.into_iter().map(Value::Object).collect()));
    Ok(Value::Object(r))
}

fn reduce_grid(conf: &Map<String, Value>, tile: bool, sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let size = conf.get("size").and_then(Value::as_u64).unwrap_or(10_000) as usize;
    let mut groups: Vec<(Vec<&Value>, String)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for v in vals {
        for b in buckets_of(v) {
            let key = b["key"].as_str().unwrap_or_default().to_string();
            let g = *index.entry(key.clone()).or_insert_with(|| {
                groups.push((Vec::new(), key));
                groups.len() - 1
            });
            groups[g].0.push(b);
        }
    }
    let tile_key = |k: &str| -> (u64, u64) {
        let mut it = k.split('/').skip(1).map(|x| x.parse::<u64>().unwrap_or(0));
        (it.next().unwrap_or(0), it.next().unwrap_or(0))
    };
    let mut buckets: Vec<(String, Map<String, Value>)> = Vec::new();
    for (bs, key) in &groups {
        buckets.push((key.clone(), merge_bucket(bs, sub, rc)?));
    }
    buckets.sort_by(|(ka, a), (kb, b)| {
        count_of(&Value::Object(b.clone()), "doc_count")
            .cmp(&count_of(&Value::Object(a.clone()), "doc_count"))
            .then_with(|| if tile { tile_key(kb).cmp(&tile_key(ka)) } else { kb.cmp(ka) })
    });
    buckets.truncate(size);
    Ok(json!({ "buckets": buckets.into_iter().map(|(_, m)| Value::Object(m)).collect::<Vec<_>>() }))
}

/// `geo_line` across shards: every shard's best `size` points (with their
/// sort values), merged in sort order (ES `InternalGeoLine.reduce`).
fn reduce_geo_line(conf: &Map<String, Value>, vals: &[&Value]) -> Value {
    let size = conf.get("size").and_then(Value::as_u64).unwrap_or(10_000) as usize;
    let desc = conf.get("sort_order").and_then(Value::as_str).is_some_and(|s| s.eq_ignore_ascii_case("desc"));
    let include_sort = conf.get("include_sort").and_then(Value::as_bool).unwrap_or(false);
    let mut complete = true;
    let mut pts: Vec<(f64, Value)> = Vec::new();
    for v in vals {
        complete &= v["properties"]["complete"].as_bool().unwrap_or(true);
        let sorts = v["properties"]["sort_values"].as_array().cloned().unwrap_or_default();
        let coords = match v["geometry"]["type"].as_str() {
            Some("Point") => vec![v["geometry"]["coordinates"].clone()],
            _ => v["geometry"]["coordinates"].as_array().cloned().unwrap_or_default(),
        };
        for (c, s) in coords.into_iter().zip(sorts) {
            pts.push((s.as_f64().unwrap_or(0.0), c));
        }
    }
    pts.sort_by(|a, b| if desc { b.0.total_cmp(&a.0) } else { a.0.total_cmp(&b.0) });
    complete &= pts.len() <= size;
    pts.truncate(size);
    let mut coords: Vec<Value> = pts.iter().map(|p| p.1.clone()).collect();
    let geometry = if coords.len() == 1 { json!({ "coordinates": coords.pop().unwrap(), "type": "Point" }) } else { json!({ "coordinates": coords, "type": "LineString" }) };
    let mut props = Map::new();
    props.insert("complete".into(), json!(complete));
    if include_sort {
        props.insert("sort_values".into(), Value::Array(pts.iter().map(|p| fnum(p.0)).collect()));
    }
    json!({ "type": "Feature", "geometry": geometry, "properties": props })
}

fn reduce_cartesian_bounds(vals: &[&Value]) -> Value {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for v in vals {
        let b = &v["bounds"];
        let (Some(l), Some(t), Some(r), Some(bt)) = (b["top_left"]["x"].as_f64(), b["top_left"]["y"].as_f64(), b["bottom_right"]["x"].as_f64(), b["bottom_right"]["y"].as_f64()) else { continue };
        x0 = x0.min(l);
        x1 = x1.max(r);
        y0 = y0.min(bt);
        y1 = y1.max(t);
    }
    if x0.is_infinite() {
        return json!({});
    }
    json!({ "bounds": { "top_left": { "x": fnum(x0), "y": fnum(y1) }, "bottom_right": { "x": fnum(x1), "y": fnum(y0) } } })
}

fn reduce_cartesian_centroid(vals: &[&Value]) -> Value {
    let (mut n, mut sx, mut sy) = (0u64, 0f64, 0f64);
    for v in vals {
        let c = count_of(v, "count");
        if c > 0 {
            n += c;
            sx += v["location"]["x"].as_f64().unwrap_or(0.0) * c as f64;
            sy += v["location"]["y"].as_f64().unwrap_or(0.0) * c as f64;
        }
    }
    if n == 0 {
        return json!({ "count": 0 });
    }
    json!({ "location": { "x": fnum(sx / n as f64), "y": fnum(sy / n as f64) }, "count": n })
}

fn reduce_geo_centroid(vals: &[&Value]) -> Value {
    let (mut n, mut slat, mut slon) = (0u64, 0f64, 0f64);
    for v in vals {
        let c = count_of(v, "count");
        if c > 0 {
            n += c;
            slat += v["location"]["lat"].as_f64().unwrap_or(0.0) * c as f64;
            slon += v["location"]["lon"].as_f64().unwrap_or(0.0) * c as f64;
        }
    }
    if n == 0 {
        return json!({ "count": 0 });
    }
    json!({ "location": { "lat": fnum(slat / n as f64), "lon": fnum(slon / n as f64) }, "count": n })
}

fn reduce_geo_bounds(conf: &Map<String, Value>, vals: &[&Value]) -> Value {
    let wrap = conf.get("wrap_longitude").and_then(Value::as_bool).unwrap_or(true);
    let (mut top, mut bottom) = (f64::NEG_INFINITY, f64::INFINITY);
    let (mut pl, mut pr, mut nl, mut nr) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for v in vals {
        let Some(p) = state(v) else { continue };
        top = top.max(num(p, "top").unwrap_or(f64::NEG_INFINITY));
        bottom = bottom.min(num(p, "bottom").unwrap_or(f64::INFINITY));
        pl = pl.min(num(p, "pl").unwrap_or(f64::INFINITY));
        pr = pr.max(num(p, "pr").unwrap_or(f64::NEG_INFINITY));
        nl = nl.min(num(p, "nl").unwrap_or(f64::INFINITY));
        nr = nr.max(num(p, "nr").unwrap_or(f64::NEG_INFINITY));
    }
    if top.is_infinite() {
        return json!({});
    }
    let (left, right) = if pl.is_infinite() {
        (nl, nr)
    } else if nl.is_infinite() {
        (pl, pr)
    } else if wrap {
        let unwrapped = pr - nl;
        let wrapped = (180.0 - pl) - (-180.0 - nr);
        if unwrapped <= wrapped { (nl, pr) } else { (pl, nr) }
    } else {
        (nl, pr)
    };
    json!({ "bounds": {
        "top_left": { "lat": fnum(top), "lon": fnum(left) },
        "bottom_right": { "lat": fnum(bottom), "lon": fnum(right) },
    }})
}

pub(super) fn sig_score(heuristic: &str, include_negatives: bool, bg_superset: bool, normalize_above: u64, sub_f: u64, sup_f: u64, sub_n: u64, sup_n: u64) -> f64 {
    if heuristic == "p_value" {
        return crate::aggs::pvalue::score(sub_f, sub_n, sup_f, sup_n, bg_superset, normalize_above);
    }
    let (sub_f, sup_f, sub_n, sup_n) = (sub_f as f64, sup_f as f64, sub_n as f64, sup_n as f64);
    match heuristic {
        "percentage" => {
            if sup_f == 0.0 { 0.0 } else { sub_f / sup_f }
        }
        "gnd" => {
            let (fx, fy, fxy) = (sup_f, sub_n, sub_f);
            if fxy == 0.0 || fx == 0.0 || fy == 0.0 {
                return 0.0;
            }
            if fx == fy && fy == fxy {
                return 1.0;
            }
            let s = (fx.ln().max(fy.ln()) - fxy.ln()) / (sup_n.ln() - fx.ln().min(fy.ln()));
            (-s).exp()
        }
        "mutual_information" | "chi_square" => {
            let (n11, n10, n01, n00) = if bg_superset {
                (sub_f, sup_f - sub_f, sub_n - sub_f, (sup_n - sub_n) - (sup_f - sub_f))
            } else {
                (sub_f, sup_f, sub_n - sub_f, sup_n - sup_f)
            };
            let n = n11 + n10 + n01 + n00;
            let (n1_, n0_, n_1, n_0) = (n10 + n11, n00 + n01, n01 + n11, n10 + n00);
            if !include_negatives && n_1 > 0.0 && n_0 > 0.0 && n11 / n_1 < n10 / n_0 {
                return f64::NEG_INFINITY;
            }
            if heuristic == "chi_square" {
                let den = n_1 * n1_ * n0_ * n_0;
                if den == 0.0 { 0.0 } else { n * (n11 * n00 - n10 * n01).powi(2) / den }
            } else {
                let term = |nxy: f64, nx: f64, ny: f64| -> f64 {
                    if nxy == 0.0 || nx == 0.0 || ny == 0.0 { 0.0 } else { nxy / n * ((n * nxy) / (nx * ny)).log2() }
                };
                term(n11, n1_, n_1) + term(n01, n0_, n_1) + term(n10, n1_, n_0) + term(n00, n0_, n_0)
            }
        }
        _ => {
            if sub_f == 0.0 || sub_n == 0.0 || sup_n == 0.0 || sup_f == 0.0 {
                return 0.0;
            }
            let sp = sub_f / sub_n;
            let bp = sup_f / sup_n;
            let abs = sp - bp;
            if abs <= 0.0 { 0.0 } else { abs * (sp / bp) }
        }
    }
}

fn reduce_significant(conf: &Map<String, Value>, sub: Option<&Map<String, Value>>, vals: &[&Value], rc: &Rc2) -> Result<Value> {
    let size = conf.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
    let min_doc_count = conf.get("min_doc_count").and_then(Value::as_u64).unwrap_or(3);
    let mut heuristic = "jlh";
    let (mut include_negatives, mut bg_superset, mut normalize_above) = (false, true, 0u64);
    for h in ["jlh", "mutual_information", "chi_square", "gnd", "percentage", "p_value"] {
        if let Some(cfg) = conf.get(h) {
            heuristic = h;
            include_negatives = cfg.get("include_negatives").and_then(Value::as_bool).unwrap_or(false);
            bg_superset = cfg.get("background_is_superset").and_then(Value::as_bool).unwrap_or(true);
            normalize_above = cfg.get("normalize_above").and_then(Value::as_u64).unwrap_or(0);
        }
    }
    let subset_size: u64 = vals.iter().map(|v| count_of(v, "doc_count")).sum();
    let superset_size: u64 = vals.iter().map(|v| count_of(v, "bg_count")).sum();
    // Background count of every term, summed over shards.
    let mut bg: HashMap<String, u64> = HashMap::new();
    let mut have_bg = true;
    for v in vals {
        match state(v).and_then(|p| p["bg"].as_array()) {
            Some(list) if !list.is_empty() || count_of(v, "bg_count") == 0 => {
                for e in list {
                    *bg.entry(e[0].to_string()).or_default() += e[1].as_u64().unwrap_or(0);
                }
            }
            _ => have_bg = false,
        }
    }
    // ES's reduce: buckets keyed by their key string in a paged hash map
    // (shards in order, each shard's buckets in its order), iterated by
    // slot into a priority queue of `size` by score alone, listed as it
    // pops. Equal scores keep the order ES gives them.
    let key_str = |b: &Value| match b.get("key_as_string").and_then(Value::as_str) {
        Some(s) => s.to_string(),
        None => match &b["key"] { Value::String(s) => s.clone(), other => other.to_string() },
    };
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for v in vals {
        for b in buckets_of(v) {
            let k = key_str(b);
            let g = match index.get(&k) {
                Some(&g) => g,
                None => {
                    groups.push(Vec::new());
                    names.push(k.clone());
                    index.insert(k, groups.len() - 1);
                    groups.len() - 1
                }
            };
            groups[g].push(b);
        }
    }
    let leader = vals.iter().map(|v| buckets_of(v).len()).next().unwrap_or(0);
    let order = super::extra::java_paged_map_order(leader, &names);
    let mut scored: Vec<(f64, Map<String, Value>)> = Vec::new();
    for bs in order.iter().map(|&g| &groups[g]) {
        let mut m = merge_bucket(bs, sub, rc)?;
        let c = count_of(&Value::Object(m.clone()), "doc_count");
        let bgc = if have_bg { bg.get(&bs[0]["key"].to_string()).copied().unwrap_or(0) } else { bs.iter().map(|b| count_of(b, "bg_count")).sum() };
        let bgc = bgc.max(if bg_superset { c } else { 0 });
        let sc = sig_score(heuristic, include_negatives, bg_superset, normalize_above, c, bgc, subset_size, superset_size);
        if sc > 0.0 && c >= min_doc_count {
            m.shift_remove("score");
            m.shift_remove("bg_count");
            // key, [key_as_string], doc_count, score, bg_count, subs
            let mut out = Map::new();
            for (k, v) in &m {
                if k == "key" || k == "key_as_string" || k == "doc_count" {
                    out.insert(k.clone(), v.clone());
                }
            }
            out.insert("score".into(), fnum(sc));
            out.insert("bg_count".into(), json!(bgc));
            for (k, v) in m {
                if !out.contains_key(&k) {
                    out.insert(k, v);
                }
            }
            scored.push((sc, out));
        }
    }
    let mut heap = super::extra::LuceneHeap::new(size.min(groups.len()), |a: &(f64, Map<String, Value>), b: &(f64, Map<String, Value>)| a.0 < b.0);
    for e in scored {
        heap.insert_with_overflow(e);
    }
    let mut buckets: Vec<Map<String, Value>> = heap.into_list().into_iter().map(|(_, m)| m).collect();
    apply_parent_pipelines(sub, &mut buckets)?;
    Ok(json!({ "doc_count": subset_size, "bg_count": superset_size, "buckets": buckets }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisRegistry;
    use crate::document::parse_with_dynamic;
    use crate::mapping::Mappings;
    use rustic_core::types::IndexSettings;
    use rustic_core::Index;

    struct Fx {
        index: Index,
        lookup: FieldLookup,
    }

    fn mapping() -> Value {
        json!({"properties": {
            "tag": {"type": "keyword"}, "price": {"type": "double"}, "qty": {"type": "long"}, "ts": {"type": "date"},
            "ok": {"type": "boolean"}, "ip": {"type": "ip"}, "loc": {"type": "geo_point"}, "grp": {"type": "keyword"}
        }})
    }

    fn doc(i: i64) -> Value {
        let mut d = json!({
            "tag": format!("t{}", i % 5),
            "qty": i * 7 % 101,
            "ts": format!("2020-{:02}-{:02}T10:00:00Z", 1 + (i % 9) / 2 * 2, 1 + i % 27),
            "ok": i % 2 == 0,
            "ip": format!("10.{}.{}.{}", i / 65536, (i / 256) % 256, i % 256),
            "loc": {"lat": 10.0 + (i % 7) as f64, "lon": -170.0 + (i * 13 % 340) as f64},
            "grp": if i % 4 == 0 { "rare" } else { "common" },
        });
        if i % 6 != 0 {
            d["price"] = json!((i * 5 % 60) as f64);
        }
        if i % 7 == 0 {
            d["tag"] = json!([format!("t{}", i % 5), "multi"]);
        }
        d
    }

    fn fixture(ids: &[i64], refresh_mid: bool) -> Fx {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&mapping(), &reg).unwrap();
        m.assign_core_ids();
        let lookup = FieldLookup::build(&m, &reg);
        let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
        for (n, i) in ids.iter().enumerate() {
            let (pd, upd) = parse_with_dynamic(&doc(*i), &i.to_string(), &m, &lookup, 1000).unwrap();
            assert!(upd.is_none());
            index.add_document(&pd.doc).unwrap();
            if refresh_mid && n == ids.len() / 2 {
                index.refresh().unwrap();
            }
        }
        Fx { index, lookup }
    }

    fn with_segs<R>(f: &Fx, pick: &[usize], body: impl FnOnce(&[AggSeg], &AggEnv) -> R) -> R {
        let leaves = f.index.leaves();
        let segs: Vec<AggSeg> = pick
            .iter()
            .map(|&li| {
                let l = &leaves[li];
                AggSeg { leaf: *l, lookup: &f.lookup, index: "idx", max_regex_length: 1000, docs: (0..l.num_docs()).collect(), scores: (0..l.num_docs()).map(|d| d as f32).collect() }
            })
            .collect();
        let fetch = |si: usize, d: DocId, sc: f32, _spec: &Value| -> Result<Value> {
            let l = &segs[si].leaf;
            let q = l.i64_values(f.lookup.get("qty").unwrap().core_id.unwrap(), d);
            Ok(json!({"_index": "idx", "_id": format!("{}", q.first().copied().unwrap_or(-1)), "_score": sc}))
        };
        let all = |si: usize| -> Vec<DocId> { (0..segs[si].leaf.num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 65536, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        body(&segs, &env)
    }

    fn partial(f: &Fx, pick: &[usize], spec: &Value) -> AggPartial {
        let req = AggRequest { aggs: spec.clone(), typed_keys: false, max_buckets: 65536, now: 0 };
        let p = with_segs(f, pick, |segs, env| shard_partial(spec, segs, env, &req, true).unwrap());
        // Survive the wire.
        AggPartial::from_bytes(&p.to_bytes()).unwrap()
    }

    fn single(f: &Fx, spec: &Value) -> Value {
        let all: Vec<usize> = (0..f.index.leaves().len()).collect();
        with_segs(f, &all, |segs, env| run_aggregations(spec, segs, env).unwrap())
    }

    fn req(spec: &Value) -> AggRequest {
        AggRequest { aggs: spec.clone(), typed_keys: false, max_buckets: 65536, now: 0 }
    }

    fn specs() -> Vec<(&'static str, Value)> {
        vec![
            ("metrics", json!({
                "mn": {"min": {"field": "price"}}, "mx": {"max": {"field": "qty"}}, "s": {"sum": {"field": "price"}}, "a": {"avg": {"field": "price"}},
                "vc": {"value_count": {"field": "tag"}}, "st": {"stats": {"field": "qty"}}, "es": {"extended_stats": {"field": "price"}},
                "c": {"cardinality": {"field": "tag"}}, "cq": {"cardinality": {"field": "qty"}}, "e": {"avg": {"field": "nope"}}, "d": {"max": {"field": "ts"}}
            })),
            ("terms", json!({
                "t": {"terms": {"field": "tag"}, "aggs": {"p": {"avg": {"field": "price"}}, "mx": {"max": {"field": "qty"}}}},
                "t2": {"terms": {"field": "tag", "size": 2}},
                "t3": {"terms": {"field": "qty", "size": 5, "order": {"_key": "desc"}}},
                "t4": {"terms": {"field": "tag", "order": {"p": "desc"}}, "aggs": {"p": {"sum": {"field": "price"}}}},
                "t5": {"terms": {"field": "tag", "min_doc_count": 7}},
                "t6": {"terms": {"field": "ok"}}, "t7": {"terms": {"field": "ts", "size": 20}}
            })),
            ("histograms", json!({
                "h": {"histogram": {"field": "qty", "interval": 10}, "aggs": {"s": {"sum": {"field": "price"}}}},
                "hk": {"histogram": {"field": "qty", "interval": 25, "keyed": true, "min_doc_count": 3}},
                "dh": {"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {"a": {"avg": {"field": "price"}}}},
                "dh2": {"date_histogram": {"field": "ts", "fixed_interval": "10d", "format": "yyyy-MM-dd", "min_doc_count": 2}},
                "dh3": {"date_histogram": {"field": "ts", "calendar_interval": "month", "extended_bounds": {"min": "2019-11-01", "max": "2020-12-31"}}}
            })),
            ("buckets", json!({
                "r": {"range": {"field": "qty", "ranges": [{"to": 20}, {"from": 20, "to": 60}, {"from": 60}]}, "aggs": {"s": {"sum": {"field": "price"}}}},
                "dr": {"date_range": {"field": "ts", "ranges": [{"to": "2020-04-01"}, {"from": "2020-04-01"}], "keyed": true}},
                "f": {"filters": {"filters": {"a": {"term": {"tag": "t1"}}, "b": {"range": {"qty": {"gt": 50}}}}, "other_bucket": true}},
                "fl": {"filter": {"term": {"ok": true}}, "aggs": {"m": {"min": {"field": "qty"}}}},
                "ms": {"missing": {"field": "price"}}, "g": {"global": {}, "aggs": {"c": {"value_count": {"field": "qty"}}}},
                "ipr": {"ip_range": {"field": "ip", "ranges": [{"to": "10.0.0.30"}, {"from": "10.0.0.30"}]}}
            })),
            ("composite", json!({
                "c": {"composite": {"size": 4, "sources": [{"t": {"terms": {"field": "tag"}}}, {"g": {"terms": {"field": "grp", "order": "desc"}}}]}, "aggs": {"s": {"sum": {"field": "qty"}}}},
                "c2": {"composite": {"size": 3, "sources": [{"q": {"histogram": {"field": "qty", "interval": 20}}}]}}
            })),
            ("top_hits_geo", json!({
                "th": {"top_hits": {"size": 3, "from": 1, "sort": [{"qty": "desc"}]}},
                "gc": {"geo_centroid": {"field": "loc"}}, "gb": {"geo_bounds": {"field": "loc"}},
                "gh": {"geohash_grid": {"field": "loc", "precision": 2}}, "gt": {"geotile_grid": {"field": "loc", "precision": 3}}
            })),
            ("sketches_and_significance", json!({
                "c10": {"cardinality": {"field": "qty", "precision_threshold": 10}}, "ct": {"cardinality": {"field": "ip", "precision_threshold": 5}},
                "f": {"filter": {"term": {"grp": "rare"}}, "aggs": {
                    "sig": {"significant_terms": {"field": "tag", "min_doc_count": 1}},
                    "sigp": {"significant_terms": {"field": "tag", "min_doc_count": 1, "percentage": {}}}
                }}
            })),
            ("pipelines", json!({
                "dh": {"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {
                    "s": {"sum": {"field": "qty"}}, "cs": {"cumulative_sum": {"buckets_path": "s"}}, "d": {"derivative": {"buckets_path": "s"}},
                    "bs": {"bucket_script": {"buckets_path": {"x": "s", "n": "_count"}, "script": "params.x / params.n"}}
                }},
                "t": {"terms": {"field": "tag"}, "aggs": {"s": {"sum": {"field": "qty"}}, "sel": {"bucket_selector": {"buckets_path": {"s": "s"}, "script": "params.s > 100"}}}},
                "ts": {"terms": {"field": "tag", "size": 10}, "aggs": {"s": {"sum": {"field": "price"}}, "bsort": {"bucket_sort": {"sort": [{"s": "desc"}], "size": 2}}}},
                "sb": {"sum_bucket": {"buckets_path": "t>_count"}}, "mb": {"max_bucket": {"buckets_path": "dh>s"}}
            })),
        ]
    }

    fn assert_same(label: &str, want: &Value, got: &Value) {
        let (w, g) = (serde_json::to_string_pretty(want).unwrap(), serde_json::to_string_pretty(got).unwrap());
        if w != g {
            let (wl, gl): (Vec<&str>, Vec<&str>) = (w.lines().collect(), g.lines().collect());
            let at = (0..wl.len().max(gl.len())).find(|i| wl.get(*i) != gl.get(*i)).unwrap_or(0);
            let ctx = |l: &Vec<&str>| l[at.saturating_sub(12)..(at + 6).min(l.len())].join("\n");
            panic!("{label}: first difference at line {at}\n--- want\n{}\n--- got\n{}", ctx(&wl), ctx(&gl));
        }
    }

    #[test]
    fn matrix_mapping_coverage_survives_shard_reduction() {
        let moment = super::super::analytics::Moments::new(2).state();
        let a = json!({"__p":{"fields":["x","y"],"moments":moment,"mapped":[true,true],"indices":["a"]}});
        let b = json!({"__p":{"fields":["x","y"],"moments":moment,"mapped":[true,false],"indices":["b"]}});
        let err = reduce_matrix_stats("m", &[&a,&b]).unwrap_err();
        assert_eq!(err.status, 400);
        assert!(err.reason.contains("missing these fields [y]"));
        assert!(reduce_matrix_stats("m", &[&a,&a]).is_ok());
        let mut old = b.clone();
        old["__p"].as_object_mut().unwrap().remove("mapped");
        assert!(reduce_matrix_stats("m", &[&a,&old]).is_ok());
        let mut unmapped = a.clone();
        unmapped["__p"]["mapped"] = json!([false,false]);
        let mut unmapped_b = b.clone();
        unmapped_b["__p"]["mapped"] = json!([false,false]);
        assert!(reduce_matrix_stats("m", &[&unmapped,&unmapped_b]).is_ok());
    }

    /// Review finding 11: partials retained for the request (scripted
    /// metric states) are released by the reduce that consumes them; after
    /// it only the reduced output stays charged, so a multi-shard request
    /// does not hold partials + 1.5x partials + output.
    #[test]
    fn reduce_releases_the_partials_it_consumes() {
        let b = crate::memory::Breakers::leak_for_tests(1 << 30);
        b.set_real_memory_for_tests(Some(0));
        let spec = json!({"s": {"sum": {"field": "x"}}});
        let parts: Vec<AggPartial> = (0..8).map(|i| AggPartial { version: PARTIAL_VERSION, tree: json!({"s": {"value": f64::from(i)}}) }).collect();
        let held: u64 = parts.iter().map(|p| crate::memory::json_bytes(&p.tree)).sum();
        let scope = crate::memory::RequestScope::new("multi-shard reduce", b);
        let out = crate::memory::scoped(Some(scope.clone()), || {
            for _ in &parts { b.reserve_request(held / 8, "shard partial").unwrap().retain_partial_for_request(); }
            assert_eq!(b.request.used(), held / 8 * 8);
            AggPartial::reduce(parts, &req(&spec)).unwrap()
        });
        assert_eq!(out["s"]["value"], json!(28.0));
        assert_eq!(b.request.used(), crate::memory::json_bytes(&out), "only the output stays charged");
        drop(scope);
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn single_partial_reduces_to_the_single_node_output() {
        let all: Vec<i64> = (0..60).collect();
        let f = fixture(&all, true);
        for (label, spec) in specs() {
            let want = single(&f, &spec);
            let p = partial(&f, &[0, 1], &spec);
            let got = AggPartial::reduce(vec![p], &req(&spec)).unwrap();
            assert_same(label, &want, &got);
        }
    }

    #[test]
    fn splitting_segments_into_partials_reduces_to_the_same_result() {
        let all: Vec<i64> = (0..60).collect();
        let f = fixture(&all, true);
        for (label, spec) in specs() {
            let want = single(&f, &spec);
            let parts = vec![partial(&f, &[0], &spec), partial(&f, &[1], &spec)];
            let got = AggPartial::reduce(parts, &req(&spec)).unwrap();
            assert_same(&format!("{label} (2 partials)"), &want, &got);
        }
    }

    #[test]
    fn three_shards_reduce_to_the_combined_index() {
        let all: Vec<i64> = (0..60).collect();
        let whole = fixture(&all, false);
        let shards: Vec<Fx> = (0..3).map(|s| fixture(&all.iter().copied().filter(|i| i % 3 == s).collect::<Vec<_>>(), s == 1)).collect();
        for (label, spec) in specs() {
            let want = single(&whole, &spec);
            let parts: Vec<AggPartial> = shards.iter().map(|s| partial(s, &(0..s.index.leaves().len()).collect::<Vec<_>>(), &spec)).collect();
            let got = AggPartial::reduce(parts, &req(&spec)).unwrap();
            assert_same(&format!("{label} (3 shards)"), &want, &got);
        }
    }

    #[test]
    fn percentiles_merge_sketches() {
        let all: Vec<i64> = (0..60).collect();
        let whole = fixture(&all, false);
        let shards: Vec<Fx> = (0..3).map(|s| fixture(&all.iter().copied().filter(|i| i % 3 == s).collect::<Vec<_>>(), false)).collect();
        let spec = json!({
            "p": {"percentiles": {"field": "qty", "percents": [1, 25, 50, 75, 99]}}, "r": {"percentile_ranks": {"field": "qty", "values": [10, 50, 90]}},
            "pk": {"percentiles": {"field": "price", "keyed": false}}, "pe": {"percentiles": {"field": "nope"}}
        });
        let want = single(&whole, &spec);
        let parts: Vec<AggPartial> = shards.iter().map(|s| partial(s, &[0], &spec)).collect();
        let got = AggPartial::reduce(parts, &req(&spec)).unwrap();
        // 20 values per shard stay exact samples: merged equals single.
        assert_same("percentiles", &want, &got);
        assert_eq!(got["pe"]["values"]["50.0"], Value::Null);
        // HDR sketches merge bucket-wise (ES `DoubleHistogram.add`): close to the single one.
        let spec = json!({"ph": {"percentiles": {"field": "qty", "percents": [25, 50, 75], "hdr": {"number_of_significant_value_digits": 3}}}});
        let want = single(&whole, &spec);
        let got = AggPartial::reduce(shards.iter().map(|s| partial(s, &[0], &spec)).collect(), &req(&spec)).unwrap();
        for k in ["25.0", "50.0", "75.0"] {
            let (a, b) = (want["ph"]["values"][k].as_f64().unwrap(), got["ph"]["values"][k].as_f64().unwrap());
            assert!((a - b).abs() <= 3.0, "hdr {k}: {a} vs {b}");
        }
        // Many values: the merged digest approximates the single one.
        let ids: Vec<i64> = (0..3000).collect();
        let big = fixture(&ids, false);
        let halves: Vec<Fx> = (0..2).map(|s| fixture(&ids.iter().copied().filter(|i| i % 2 == s).collect::<Vec<_>>(), false)).collect();
        let spec = json!({"p": {"percentiles": {"field": "qty", "percents": [5, 50, 95]}}});
        let want = single(&big, &spec);
        let got = AggPartial::reduce(halves.iter().map(|s| partial(s, &[0], &spec)).collect(), &req(&spec)).unwrap();
        for k in ["5.0", "50.0", "95.0"] {
            let (a, b) = (want["p"]["values"][k].as_f64().unwrap(), got["p"]["values"][k].as_f64().unwrap());
            assert!((a - b).abs() <= 2.0, "{k}: {a} vs {b}");
        }
    }

    #[test]
    fn cardinality_merges_hll_registers_exactly_above_the_threshold() {
        let ids: Vec<i64> = (0..30_000).collect();
        // `ip` has 30k distinct values: well past the 10k exact-set limit.
        let whole = fixture(&ids, false);
        let shards: Vec<Fx> = (0..3).map(|s| fixture(&ids.iter().copied().filter(|i| i % 3 == s).collect::<Vec<_>>(), false)).collect();
        let spec = json!({"c": {"cardinality": {"field": "ip", "precision_threshold": 40000}}, "d": {"cardinality": {"field": "ip"}}});
        let want = single(&whole, &spec);
        let got = AggPartial::reduce(shards.iter().map(|s| partial(s, &[0], &spec)).collect(), &req(&spec)).unwrap();
        assert_same("cardinality", &want, &got);
        assert_eq!(got["c"]["value"], 30_000, "exact below the 40k threshold");
        let d = got["d"]["value"].as_u64().unwrap();
        assert!((29_000..31_000).contains(&d), "{d}");
    }

    #[test]
    fn typed_keys_meta_and_unmergeable_aggs() {
        let all: Vec<i64> = (0..30).collect();
        let a = fixture(&all[..15], false);
        let b = fixture(&all[15..], false);
        let spec = json!({"t": {"terms": {"field": "tag"}, "meta": {"x": 1}}, "m": {"min": {"field": "qty"}}});
        let r = AggRequest { aggs: spec.clone(), typed_keys: true, max_buckets: 100, now: 0 };
        let got = AggPartial::reduce(vec![partial(&a, &[0], &spec), partial(&b, &[0], &spec)], &r).unwrap();
        assert_eq!(got["sterms#t"]["meta"], json!({"x": 1}));
        assert!(got["min#m"]["value"].is_number());
        // Types without a mergeable partial pass a lone partial through and reject several.
        let spec = json!({"rt": {"rare_terms": {"field": "tag", "max_doc_count": 100}}});
        let one = AggPartial::reduce(vec![partial(&a, &[0], &spec)], &req(&spec)).unwrap();
        assert!(one["rt"]["buckets"].is_array());
        let err = AggPartial::reduce(vec![partial(&a, &[0], &spec), partial(&b, &[0], &spec)], &req(&spec)).unwrap_err();
        assert_eq!(err.kind, "illegal_argument_exception");
        assert!(err.reason.contains("rare_terms"), "{}", err.reason);
    }

    #[test]
    fn frequent_items_distributed_matches_whole_index() {
        let ids: Vec<i64> = (0..80).collect();
        let all = fixture(&ids, false);
        let shards = [fixture(&ids[..13], false), fixture(&ids[13..], false), fixture(&[], false)];
        for alias in ["frequent_item_sets", "frequent_items"] {
            for filter in [json!({"match_all":{}}), json!({"term":{"ok":true}})] {
                let spec = json!({"sets": {alias: {"fields":[{"field":"tag","exclude":["t4"]},{"field":"ok"},{"field":"ts"}], "minimum_support":0.04,"minimum_set_size":2,"size":20,"filter":filter}}});
                let want = single(&all, &spec);
                let got = AggPartial::reduce(shards.iter().map(|s| partial(s, &(0..s.index.leaves().len()).collect::<Vec<_>>(), &spec)).collect(), &req(&spec)).unwrap();
                assert_eq!(want, got, "{spec}");
                assert!(!got.to_string().contains("__p"));
            }
        }
    }

    #[test]
    fn coordinator_result_charge_outlives_reducer() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        let scope=crate::memory::RequestScope::new("coordinator result",breakers);
        let spec=json!({"sum":{"sum":{"field":"value"}}});
        let out=crate::memory::scoped(Some(scope.clone()),||AggPartial::reduce(vec![AggPartial{version:1,tree:json!({"sum":{"value":3.0}})}],&req(&spec))).unwrap();
        assert_eq!(out["sum"]["value"],3.0);
        assert!(breakers.request.reserved()>=crate::memory::json_bytes(&out));
        drop(out);drop(scope);
        assert_eq!(breakers.request.reserved(),0);
    }

    #[test]
    fn terms_error_bounds_follow_es_when_shards_truncate() {
        let all: Vec<i64> = (0..60).collect();
        let shards: Vec<Fx> = (0..2).map(|s| fixture(&all.iter().copied().filter(|i| i % 2 == s).collect::<Vec<_>>(), false)).collect();
        // 27 distinct dates, shard_size 3 → both shards truncate.
        let spec = json!({"t": {"terms": {"field": "ts", "size": 2, "shard_size": 3, "show_term_doc_count_error": true}}});
        let got = AggPartial::reduce(shards.iter().map(|s| partial(s, &[0], &spec)).collect(), &req(&spec)).unwrap();
        let t = &got["t"];
        assert_eq!(t["buckets"].as_array().unwrap().len(), 2);
        assert!(t["doc_count_error_upper_bound"].as_i64().unwrap() > 0);
        assert!(t["buckets"][0].get("doc_count_error_upper_bound").is_some());
    }
}
