//! Executing a transform function over the source: composite-aggregation
//! paging (pivot and latest), bucket → document conversion (ES
//! `AggregationResultUtils` / latest's top hit), destination mappings
//! deduction (ES `SchemaUtil` / `TransformAggregations`) and document ids
//! (ES `IdGenerator`).

use super::config::{Config, Function, GroupSource};
use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use crate::search::{self, ExecOpts, Target};
use base64::Engine;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

/// The composite aggregation name used for paging.
const COMPOSITE: &str = "_transform";

/// One page of results.
pub struct Page {
    /// `(id, document)` per bucket.
    pub docs: Vec<(String, Map<String, Value>)>,
    /// Source documents covered by the page (sum of bucket doc counts).
    pub docs_processed: u64,
    pub after_key: Option<Value>,
}

/// Resolve the source indices (missing concrete names → `no such index`).
pub fn source_targets(node: &Node, cfg: &Config, lenient: bool) -> Result<Vec<Target>> {
    let opts = ResolveOpts { ignore_unavailable: lenient, allow_no_indices: true, ..Default::default() };
    node.resolve(&cfg.source_indices.join(","), opts)
}

/// Group sources in composite order: date histograms first (ES puts the
/// date histogram group first so checkpoints page in time order).
pub fn ordered_groups(groups: &[GroupSource]) -> Vec<&GroupSource> {
    let mut out: Vec<&GroupSource> = groups.iter().filter(|g| g.kind == "date_histogram").collect();
    out.extend(groups.iter().filter(|g| g.kind != "date_histogram"));
    out
}

fn composite_sources(cfg: &Config) -> Vec<Value> {
    match &cfg.function {
        Function::Pivot { groups, .. } => ordered_groups(groups)
            .into_iter()
            .map(|g| {
                let mut p = g.params.clone();
                if g.kind == "date_histogram" {
                    p.remove("format");
                }
                json!({ g.name.clone(): { g.kind.clone(): p } })
            })
            .collect(),
        Function::Latest { unique_key, .. } => unique_key.iter().map(|k| json!({ k.clone(): {"terms": {"field": k, "missing_bucket": true}} })).collect(),
    }
}

/// The search body for one page.
pub fn page_body(cfg: &Config, query: &Value, size: usize, after: Option<&Value>) -> Value {
    let mut comp = Map::new();
    comp.insert("size".into(), json!(size));
    comp.insert("sources".into(), Value::Array(composite_sources(cfg)));
    if let Some(a) = after {
        comp.insert("after".into(), a.clone());
    }
    let subs = match &cfg.function {
        Function::Pivot { aggs, .. } => Value::Object(aggs.clone()),
        Function::Latest { sort, .. } if is_runtime(cfg, sort) => {
            // Runtime sort fields: pick the maximum client-side from the field values.
            json!({"_top_hits": {"top_hits": {"size": 100, "fields": [sort]}}})
        }
        Function::Latest { sort, .. } => json!({"_top_hits": {"top_hits": {"size": 1, "sort": [{ sort.clone(): {"order": "desc"} }]}}}),
    };
    let mut b = json!({
        "size": 0,
        "track_total_hits": false,
        "query": query,
        "aggs": { COMPOSITE: { "composite": Value::Object(comp), "aggs": subs } }
    });
    if let Some(rm) = &cfg.runtime_mappings {
        b["runtime_mappings"] = rm.clone();
    }
    b
}

fn is_runtime(cfg: &Config, f: &str) -> bool {
    cfg.runtime_mappings.as_ref().and_then(|m| m.get(f)).is_some()
}

fn cmp_json(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        _ => a.as_str().unwrap_or("").cmp(b.as_str().unwrap_or("")),
    }
}

/// Run one search request over the targets; returns the response JSON.
pub fn search(node: &Node, targets: &[Target], body: &Value) -> Result<Value> {
    let req = search::parse_request(body, &HashMap::new())?;
    let out = search::execute(targets, &req, &ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets().max(65_536), scroll_id: None, pit_id: None })?;
    serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))
}

/// Source field types (first mapping wins), runtime fields included.
pub fn field_types(cfg: &Config, targets: &[Target]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Some(rm) = cfg.runtime_mappings.as_ref().and_then(Value::as_object) {
        for (k, v) in rm {
            if let Some(t) = v.get("type").and_then(Value::as_str) {
                out.insert(k.clone(), t.to_string());
            }
        }
    }
    let mut wanted: Vec<String> = Vec::new();
    match &cfg.function {
        Function::Pivot { groups, aggs, .. } => {
            wanted.extend(groups.iter().filter_map(|g| g.field().map(str::to_string)));
            fn walk(aggs: &Map<String, Value>, out: &mut Vec<String>) {
                for def in aggs.values() {
                    if let Some(o) = def.as_object() {
                        for (k, v) in o {
                            if k == "aggs" || k == "aggregations" {
                                if let Some(s) = v.as_object() {
                                    walk(s, out);
                                }
                            } else if let Some(f) = v.get("field").and_then(Value::as_str) {
                                out.push(f.to_string());
                            } else if let Some(ms) = v.get("metrics") {
                                let list = match ms {
                                    Value::Array(a) => a.clone(),
                                    o => vec![o.clone()],
                                };
                                for m in list {
                                    if let Some(f) = m.get("field").and_then(Value::as_str) {
                                        out.push(f.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            walk(aggs, &mut wanted);
        }
        Function::Latest { unique_key, sort } => {
            wanted.extend(unique_key.iter().cloned());
            wanted.push(sort.clone());
        }
    }
    if let Some((f, _)) = &cfg.sync {
        wanted.push(f.clone());
    }
    for t in targets {
        let lookup = t.svc.lookup();
        for f in &wanted {
            if out.contains_key(f) {
                continue;
            }
            if let Some(rf) = lookup.get(f) {
                out.insert(f.clone(), rf.declared.name().to_string());
            }
        }
    }
    out
}

fn put_path(doc: &mut Map<String, Value>, path: &str, v: Value) {
    let parts: Vec<&str> = path.split('.').collect();
    let mut cur = doc;
    for p in &parts[..parts.len() - 1] {
        let e = cur.entry(p.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if !e.is_object() {
            *e = Value::Object(Map::new());
        }
        cur = e.as_object_mut().unwrap();
    }
    cur.insert(parts[parts.len() - 1].to_string(), v);
}

/// ES `OutputFieldNameConverter.fromDouble`.
fn double_name(d: f64) -> String {
    if d.fract() == 0.0 && d.abs() < 1e15 {
        format!("{}", d as i64)
    } else {
        format!("{d}").replace('.', "_")
    }
}

fn is_date_type(t: Option<&String>) -> bool {
    matches!(t.map(String::as_str), Some("date") | Some("date_nanos"))
}

/// Convert one aggregation result to its document value.
fn agg_value(kind: &str, def: &Value, res: &Value, types: &HashMap<String, String>, epoch: bool) -> Value {
    let field = def.get(kind).and_then(|d| d.get("field")).and_then(Value::as_str).map(str::to_string);
    let sub = def.get("aggs").or_else(|| def.get("aggregations")).and_then(Value::as_object);
    match kind {
        "avg" | "sum" | "weighted_avg" | "median_absolute_deviation" | "bucket_script" | "scripted_metric" => res.get("value").cloned().unwrap_or(Value::Null),
        "value_count" | "cardinality" => res.get("value").cloned().unwrap_or(json!(0)),
        "min" | "max" => {
            let v = res.get("value").cloned().unwrap_or(Value::Null);
            if !v.is_null() && is_date_type(field.as_ref().and_then(|f| types.get(f))) {
                if epoch {
                    return json!(v.as_f64().unwrap_or(0.0) as i64);
                }
                return res.get("value_as_string").cloned().unwrap_or(v);
            }
            v
        }
        "percentiles" => {
            let mut m = Map::new();
            match res.get("values") {
                Some(Value::Object(vals)) => {
                    for (k, v) in vals {
                        if k.ends_with("_as_string") {
                            continue;
                        }
                        let name = k.parse::<f64>().map(double_name).unwrap_or_else(|_| k.replace('.', "_"));
                        m.insert(name, v.clone());
                    }
                }
                Some(Value::Array(a)) => {
                    for e in a {
                        let k = e.get("key").and_then(Value::as_f64).unwrap_or(0.0);
                        m.insert(double_name(k), e.get("value").cloned().unwrap_or(Value::Null));
                    }
                }
                _ => {}
            }
            Value::Object(m)
        }
        "stats" | "extended_stats" | "boxplot" => {
            let mut m = Map::new();
            if let Some(o) = res.as_object() {
                for (k, v) in o {
                    if k.ends_with("_as_string") || k == "meta" {
                        continue;
                    }
                    m.insert(k.clone(), v.clone());
                }
            }
            Value::Object(m)
        }
        "filter" | "missing" => match sub {
            Some(s) if !s.is_empty() => Value::Object(sub_values(s, res, types, epoch)),
            _ => res.get("doc_count").cloned().unwrap_or(json!(0)),
        },
        "range" | "date_range" | "filters" | "terms" | "rare_terms" | "multi_terms" => {
            let mut m = Map::new();
            let buckets: Vec<(String, Value)> = match res.get("buckets") {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|b| {
                        let k = b.get("key_as_string").or_else(|| b.get("key")).map(|k| match k {
                            Value::String(s) => s.clone(),
                            Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect::<Vec<_>>().join("|"),
                            other => other.to_string(),
                        });
                        (k.unwrap_or_default(), b.clone())
                    })
                    .collect(),
                Some(Value::Object(o)) => o.iter().map(|(k, b)| (k.clone(), b.clone())).collect(),
                _ => vec![],
            };
            for (k, b) in buckets {
                let v = match sub {
                    Some(s) if !s.is_empty() => Value::Object(sub_values(s, &b, types, epoch)),
                    _ => b.get("doc_count").cloned().unwrap_or(json!(0)),
                };
                m.insert(k.replace('.', "_"), v);
            }
            Value::Object(m)
        }
        "top_metrics" => {
            let mut m = Map::new();
            if let Some(t) = res.get("top").and_then(Value::as_array).and_then(|a| a.first()) {
                if let Some(ms) = t.get("metrics").and_then(Value::as_object) {
                    for (k, v) in ms {
                        m.insert(k.clone(), v.clone());
                    }
                }
            } else if let Some(ms) = def.get("top_metrics").and_then(|d| d.get("metrics")) {
                let list = match ms {
                    Value::Array(a) => a.clone(),
                    o => vec![o.clone()],
                };
                for x in list {
                    if let Some(f) = x.get("field").and_then(Value::as_str) {
                        m.insert(f.to_string(), Value::Null);
                    }
                }
            }
            Value::Object(m)
        }
        "geo_centroid" => match res.get("location") {
            Some(l) => json!(format!("{}, {}", l["lat"].as_f64().unwrap_or(0.0), l["lon"].as_f64().unwrap_or(0.0))),
            None => Value::Null,
        },
        "geo_bounds" => match res.get("bounds") {
            Some(b) => {
                let (top, left) = (b["top_left"]["lat"].as_f64().unwrap_or(0.0), b["top_left"]["lon"].as_f64().unwrap_or(0.0));
                let (bottom, right) = (b["bottom_right"]["lat"].as_f64().unwrap_or(0.0), b["bottom_right"]["lon"].as_f64().unwrap_or(0.0));
                if top == bottom && left == right {
                    json!({"type": "point", "coordinates": [left, top]})
                } else if top == bottom || left == right {
                    json!({"type": "linestring", "coordinates": [[left, top], [right, bottom]]})
                } else {
                    json!({"type": "polygon", "coordinates": [[[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]]})
                }
            }
            None => Value::Null,
        },
        _ => res.get("value").cloned().unwrap_or_else(|| res.clone()),
    }
}

fn agg_kind(def: &Value) -> Option<String> {
    def.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "aggs" | "aggregations" | "meta")).cloned())
}

fn sub_values(aggs: &Map<String, Value>, bucket: &Value, types: &HashMap<String, String>, epoch: bool) -> Map<String, Value> {
    let mut m = Map::new();
    for (name, def) in aggs {
        let Some(kind) = agg_kind(def) else { continue };
        if kind == "bucket_selector" {
            continue;
        }
        let res = bucket.get(name).cloned().unwrap_or(Value::Null);
        put_path(&mut m, name, agg_value(&kind, def, &res, types, epoch));
    }
    m
}

fn format_date(ms: i64) -> String {
    crate::date::DateFormatter::default().format(ms, crate::date::Tz::UTC)
}

/// Group-by key value as written to the document.
fn group_value(g: &GroupSource, v: &Value, epoch: bool) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    match g.kind.as_str() {
        "date_histogram" => {
            let ms = v.as_f64().map(|f| f as i64);
            match (ms, epoch) {
                (Some(ms), true) => json!(ms),
                (Some(ms), false) => match g.params.get("format").and_then(Value::as_str).and_then(|f| crate::date::DateFormatter::new(f).ok()) {
                    Some(f) => json!(f.format(ms, crate::date::Tz::UTC)),
                    None => json!(format_date(ms)),
                },
                _ => v.clone(),
            }
        }
        "histogram" => json!(v.as_f64().unwrap_or(0.0)),
        "geotile_grid" => match v.as_str().and_then(crate::ingest::util::geotile_bbox) {
            Some(b) => crate::ingest::util::bbox_shape(b, false),
            None => v.clone(),
        },
        _ => v.clone(),
    }
}

/// Murmur3 x64 128 (Lucene/ES `MurmurHash3.hash128`).
fn murmur3_x64_128(data: &[u8], seed: u64) -> (u64, u64) {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    fn fmix(mut k: u64) -> u64 {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^= k >> 33;
        k
    }
    let (mut h1, mut h2) = (seed, seed);
    let nblocks = data.len() / 16;
    for i in 0..nblocks {
        let b = &data[i * 16..i * 16 + 16];
        let mut k1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(27).wrapping_add(h2).wrapping_mul(5).wrapping_add(0x52dc_e729);
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2.rotate_left(31).wrapping_add(h1).wrapping_mul(5).wrapping_add(0x3849_5ab5);
    }
    let tail = &data[nblocks * 16..];
    let (mut k1, mut k2) = (0u64, 0u64);
    for (i, b) in tail.iter().enumerate().rev() {
        if i >= 8 {
            k2 ^= (*b as u64) << ((i - 8) * 8);
        } else {
            k1 ^= (*b as u64) << (i * 8);
        }
    }
    if tail.len() > 8 {
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
    }
    h1 ^= data.len() as u64;
    h2 ^= data.len() as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (h1, h2)
}

fn id_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::Null => b"__NULL_VALUE__".to_vec(),
        Value::String(s) if s.is_empty() => b"__EMPTY_VALUE__".to_vec(),
        Value::String(s) => s.as_bytes().to_vec(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_be_bytes().to_vec(),
            None => n.as_f64().unwrap_or(0.0).to_bits().to_be_bytes().to_vec(),
        },
        Value::Bool(b) => vec![*b as u8],
        other => other.to_string().into_bytes(),
    }
}

/// ES `IdGenerator`: keys sorted by name, values joined with `$`, murmur3
/// (seed 19) appended to the first byte of the first values, base64url.
pub fn doc_id(keys: &BTreeMap<String, Value>) -> String {
    let mut buf = Vec::new();
    let mut prefix = Vec::new();
    for v in keys.values() {
        let b = id_bytes(v);
        if prefix.len() < 5 {
            if let Some(f) = b.first() {
                prefix.push(*f);
            }
        }
        buf.extend_from_slice(&b);
        buf.push(b'$');
    }
    let (h1, h2) = murmur3_x64_128(&buf, 19);
    prefix.extend_from_slice(&h1.to_be_bytes());
    prefix.extend_from_slice(&h2.to_be_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(prefix)
}

/// Turn a composite response into documents.
pub fn convert_page(cfg: &Config, resp: &Value, types: &HashMap<String, String>) -> Result<Page> {
    let comp = &resp["aggregations"][COMPOSITE];
    let buckets = comp["buckets"].as_array().cloned().unwrap_or_default();
    let mut docs = Vec::with_capacity(buckets.len());
    let mut processed = 0u64;
    let epoch = cfg.settings.dates_as_epoch_millis;
    for b in &buckets {
        let dc = b["doc_count"].as_u64().unwrap_or(0);
        processed += dc;
        let key = b["key"].as_object().cloned().unwrap_or_default();
        let id_keys: BTreeMap<String, Value> = key.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let id = doc_id(&id_keys);
        let doc = match &cfg.function {
            Function::Pivot { groups, aggs, .. } => {
                let mut d = Map::new();
                for g in groups {
                    put_path(&mut d, &g.name, group_value(g, key.get(&g.name).unwrap_or(&Value::Null), epoch));
                }
                for (name, def) in aggs {
                    let Some(kind) = agg_kind(def) else { continue };
                    if kind == "bucket_selector" {
                        continue;
                    }
                    let res = b.get(name).cloned().unwrap_or(Value::Null);
                    put_path(&mut d, name, agg_value(&kind, def, &res, types, epoch));
                }
                d
            }
            Function::Latest { sort, .. } => {
                let hits = b["_top_hits"]["hits"]["hits"].as_array().cloned().unwrap_or_default();
                let hit = if is_runtime(cfg, sort) {
                    hits.iter().max_by(|x, y| cmp_json(&x["fields"][sort.as_str()][0], &y["fields"][sort.as_str()][0])).cloned().unwrap_or(Value::Null)
                } else {
                    hits.first().cloned().unwrap_or(Value::Null)
                };
                let hit = &hit;
                let mut d = hit["_source"].as_object().cloned().unwrap_or_default();
                d.retain(|k, _| !k.starts_with('_'));
                d
            }
        };
        docs.push((id, doc));
    }
    let after_key = if buckets.is_empty() { None } else { comp.get("after_key").cloned() };
    Ok(Page { docs, docs_processed: processed, after_key })
}

/// Mapping type of a source field.
fn src_type(f: Option<&str>, types: &HashMap<String, String>) -> Option<String> {
    f.and_then(|f| types.get(f)).cloned()
}

/// Deduced destination `properties` (flat dotted names, as ES).
pub fn deduce_properties(cfg: &Config, types: &HashMap<String, String>) -> Map<String, Value> {
    let mut props = Map::new();
    let Function::Pivot { groups, aggs, .. } = &cfg.function else { return props };
    let epoch = cfg.settings.dates_as_epoch_millis;
    let mut put = |name: String, t: &str| {
        let t = if epoch && (t == "date" || t == "date_nanos") { "long" } else { t };
        props.insert(name, json!({"type": t}));
    };
    for g in groups {
        match g.kind.as_str() {
            "terms" => {
                if let Some(t) = src_type(g.field(), types) {
                    let t = if t == "text" { "keyword".to_string() } else { t };
                    put(g.name.clone(), &t);
                }
            }
            "histogram" => put(g.name.clone(), "double"),
            "date_histogram" => put(g.name.clone(), "date"),
            "geotile_grid" => put(g.name.clone(), "geo_shape"),
            _ => {}
        }
    }
    fn walk(prefix: &str, aggs: &Map<String, Value>, types: &HashMap<String, String>, put: &mut dyn FnMut(String, &str)) {
        for (name, def) in aggs {
            let full = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
            let Some(kind) = agg_kind(def) else { continue };
            let field = def.get(&kind).and_then(|d| d.get("field")).and_then(Value::as_str);
            let sub = def.get("aggs").or_else(|| def.get("aggregations")).and_then(Value::as_object);
            match kind.as_str() {
                "avg" | "sum" | "weighted_avg" | "median_absolute_deviation" => put(full, "double"),
                "value_count" | "cardinality" => put(full, "long"),
                "min" | "max" => match src_type(field, types) {
                    Some(t) => put(full, &t),
                    None => put(full, "double"),
                },
                "percentiles" => {
                    let pcts: Vec<f64> = def[&kind]["percents"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_else(|| vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]);
                    for p in pcts {
                        put(format!("{full}.{}", double_name(p)), "double");
                    }
                }
                "stats" => {
                    for (k, t) in [("count", "long"), ("min", "double"), ("max", "double"), ("avg", "double"), ("sum", "double")] {
                        put(format!("{full}.{k}"), t);
                    }
                }
                "filter" | "missing" => match sub {
                    Some(s) if !s.is_empty() => walk(&full, s, types, put),
                    _ => put(full, "long"),
                },
                "terms" | "rare_terms" => put(full, "flattened"),
                "geo_centroid" => put(full, "geo_point"),
                "geo_bounds" | "geo_line" => put(full, "geo_shape"),
                "top_metrics" => {
                    let ms = def[&kind]["metrics"].clone();
                    let list = match ms {
                        Value::Array(a) => a,
                        o => vec![o],
                    };
                    for m in list {
                        if let Some(f) = m.get("field").and_then(Value::as_str) {
                            if let Some(t) = types.get(f) {
                                put(format!("{full}.{f}"), t);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    walk("", aggs, types, &mut put);
    props
}

/// `_meta` written into the destination mapping.
pub fn dest_meta(id: &str, version: &str) -> Value {
    json!({
        "_transform": {"transform": id, "version": {"created": version}, "creation_date_in_millis": crate::date::now_millis()},
        "created_by": "transform"
    })
}

/// `generated_dest_index` of a preview / the create-index body.
pub fn dest_index_body(cfg: &Config, types: &HashMap<String, String>, version: &str, for_preview: bool) -> Value {
    let mut mappings = Map::new();
    mappings.insert("_meta".into(), dest_meta(&cfg.id, version));
    let deduce = cfg.settings.deduce_mappings;
    if deduce || for_preview {
        let props = if deduce { deduce_properties(cfg, types) } else { Map::new() };
        if for_preview || !props.is_empty() {
            mappings.insert("properties".into(), Value::Object(props));
        }
    }
    let mut aliases = Map::new();
    for a in &cfg.dest_aliases {
        if let Some(n) = a.get("alias").and_then(Value::as_str) {
            aliases.insert(n.to_string(), json!({}));
        }
    }
    json!({
        "mappings": Value::Object(mappings),
        "settings": {"index": {"number_of_shards": "1", "auto_expand_replicas": "0-1"}},
        "aliases": Value::Object(aliases)
    })
}

/// Field types the latest function cannot group on.
pub fn check_latest_keys(cfg: &Config, types: &HashMap<String, String>) -> Result<()> {
    if let Function::Latest { unique_key, .. } = &cfg.function {
        for k in unique_key {
            if let Some(t) = types.get(k) {
                if matches!(t.as_str(), "aggregate_metric_double" | "text" | "geo_shape" | "dense_vector" | "histogram") {
                    return Err(EsError::illegal_argument(format!("Field [{k}] of type [{t}] is not supported for aggregation [terms]")));
                }
            }
        }
    }
    Ok(())
}
