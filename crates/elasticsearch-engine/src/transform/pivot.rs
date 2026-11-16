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

/// Hits a runtime-sorted latest bucket returns as candidates.
const LATEST_CANDIDATES: usize = 100;

/// A latest bucket whose candidates may not include its latest document
/// (more source documents than candidates): its position in `Page::docs`,
/// its unique key and the group's maximum sort value.
pub struct Unresolved {
    pub pos: usize,
    pub key: Map<String, Value>,
    pub max: f64,
}

/// One page of results.
pub struct Page {
    /// `(id, document)` per bucket.
    pub docs: Vec<(String, Map<String, Value>)>,
    /// The ids older encodings gave the page's groups, deleted when a
    /// transform on an older encoding is rebuilt.
    pub legacy_ids: Vec<String>,
    /// Source documents covered by the page (sum of bucket doc counts).
    pub docs_processed: u64,
    /// The composite continuation cursor, whatever survived bucket
    /// pipelines (a `bucket_selector` may filter a whole page while later
    /// pages still hold eligible groups).
    pub after_key: Option<Value>,
    /// Latest buckets to look up (see [`Unresolved`]).
    pub unresolved: Vec<Unresolved>,
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
            // Runtime sort fields: pick the maximum client-side from the
            // field values of up to 100 hits; the group's true maximum
            // (`_latest_max`) tells whether a larger group needs the
            // follow-up lookup of `Page::unresolved`.
            json!({"_top_hits": {"top_hits": {"size": LATEST_CANDIDATES, "fields": [sort]}}, "_latest_max": {"max": {"field": sort}}})
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
    let resp: Value = serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))?;
    check_complete(&resp)?;
    Ok(resp)
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

/// Destination id encoding of this build: 3, ES 9.5.5's `IDGenerator`
/// byte for byte (see [`doc_id`]), except for the key tuples ES itself
/// encodes ambiguously, which get a typed id of a different length.
/// Encoding 1 (rustic's first transform ids) was ES's scheme with a shorter
/// prefix and no buffer padding, encoding 2 the typed scheme for every
/// tuple. A transform whose stored state is on an older encoding rebuilds
/// its destination on its next checkpoint, deleting both older ids of every
/// group (see `TransformService::checkpoint`).
pub const ID_ENCODING: u32 = 3;

/// ES `IDGenerator.getBytes`: the bytes of one key value as ES sees the
/// composite key object (`Long` for integers, `Double` for floating point
/// keys, a one-byte `Boolean`, UTF-8 strings, `"__NULL_VALUE__"` for null).
fn es_id_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::Null => b"__NULL_VALUE__".to_vec(),
        Value::String(s) => s.as_bytes().to_vec(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_be_bytes().to_vec(),
            None => n.as_f64().unwrap_or(0.0).to_bits().to_be_bytes().to_vec(),
        },
        Value::Bool(b) => vec![*b as u8],
        other => other.to_string().into_bytes(),
    }
}

/// Lucene `ArrayUtil.oversize(min, 1)` on a 64-bit JVM.
fn lucene_oversize(min: usize) -> usize {
    if min == 0 {
        return 0;
    }
    let extra = (min >> 3).max(3);
    (min + extra + 7) & !7
}

/// Lucene `BytesRefBuilder`: appends grow the backing array with
/// `ArrayUtil.grow`; `bytes()` is the whole backing array (zero padded).
#[derive(Default)]
struct JavaBytesRefBuilder {
    bytes: Vec<u8>,
    len: usize,
}

impl JavaBytesRefBuilder {
    fn grow(&mut self, capacity: usize) {
        if self.bytes.len() < capacity {
            self.bytes.resize(lucene_oversize(capacity), 0);
        }
    }
    fn append(&mut self, b: &[u8]) {
        self.grow(self.len + b.len());
        self.bytes[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
    }
}

/// Whether ES's encoding is ambiguous for this tuple: a string value that
/// contains the `$` delimiter, or equals the null/empty markers, can share
/// ES's hash input with a different tuple (`("x$x","x")` and
/// `("x","x$x")` both hash `x$x$x$`; null and `"__NULL_VALUE__"` both
/// hash `__NULL_VALUE__$`), and ES then overwrites one group's document
/// with the other's.
fn es_ambiguous(keys: &BTreeMap<String, Value>) -> bool {
    keys.values().any(|v| matches!(v, Value::String(s) if s.contains('$') || s == "__NULL_VALUE__" || s == "__EMPTY_VALUE__"))
}

/// The ES `IDGenerator` id of a tuple: keys in name order, each value's
/// bytes (empty ones as `"__EMPTY_VALUE__"`) followed by `$`, murmur3 x64
/// 128 (seed 19) of that buffer appended to the first byte of the first
/// six values, and the builder's whole backing array (zero padded, as
/// `BytesRefBuilder.bytes()` returns it) in base64url without padding.
/// Always 24 bytes, so 32 characters.
pub fn es_doc_id(keys: &BTreeMap<String, Value>) -> String {
    let mut buffer = JavaBytesRefBuilder::default();
    let mut hashed = JavaBytesRefBuilder::default();
    for v in keys.values() {
        let mut b = es_id_bytes(v);
        if b.is_empty() {
            b = b"__EMPTY_VALUE__".to_vec();
        }
        buffer.append(&b);
        buffer.append(b"$");
        if hashed.len <= 5 {
            hashed.append(&b[..1]);
        }
    }
    let (h1, h2) = murmur3_x64_128(&buffer.bytes[..buffer.len], 19);
    hashed.append(&h1.to_be_bytes());
    hashed.append(&h2.to_be_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&hashed.bytes)
}

/// Destination document id of a group: ES's id ([`es_doc_id`]), so a
/// transform writes the same `_id`s as Elasticsearch, except for a tuple
/// ES encodes ambiguously ([`es_ambiguous`]), which gets the typed,
/// length-delimited id ([`typed_doc_id`]) instead: it is never 32
/// characters long, so it cannot meet an ES-shaped id.
pub fn doc_id(keys: &BTreeMap<String, Value>) -> String {
    if es_ambiguous(keys) {
        typed_doc_id(keys)
    } else {
        es_doc_id(keys)
    }
}

/// Encoding-1 bytes of one key value.
fn legacy_id_bytes(v: &Value) -> Vec<u8> {
    match v {
        Value::Null => b"__NULL_VALUE__".to_vec(),
        Value::String(s) if s.is_empty() => b"__EMPTY_VALUE__".to_vec(),
        other => es_id_bytes(other),
    }
}

/// The unpadded id shape: prefix bytes, then the murmur3 x64 128 (seed 19)
/// of `buf`, base64url without padding.
fn id_of(prefix: Vec<u8>, buf: &[u8]) -> String {
    let mut out = prefix;
    let (h1, h2) = murmur3_x64_128(buf, 19);
    out.extend_from_slice(&h1.to_be_bytes());
    out.extend_from_slice(&h2.to_be_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(out)
}

/// Encoding-1 id: kept only to find and delete the documents an encoding-1
/// transform wrote, when it is rebuilt.
pub fn legacy_doc_id(keys: &BTreeMap<String, Value>) -> String {
    let mut buf = Vec::new();
    let mut prefix = Vec::new();
    for v in keys.values() {
        let b = legacy_id_bytes(v);
        if prefix.len() < 5 {
            if let Some(f) = b.first() {
                prefix.push(*f);
            }
        }
        buf.extend_from_slice(&b);
        buf.push(b'$');
    }
    id_of(prefix, &buf)
}

/// One length-delimited field: `tag`, 4-byte big-endian length, bytes.
fn put_field(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) {
    out.push(tag);
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
}

/// Type tag and bytes of one key value: every JSON type has its own tag, so
/// null, `""`, `"__NULL_VALUE__"`, `1`, `1.0` and `"1"` all differ.
fn typed_value(v: &Value) -> (u8, Vec<u8>) {
    match v {
        Value::Null => (0, Vec::new()),
        Value::Bool(b) => (1, vec![*b as u8]),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => (2, i.to_be_bytes().to_vec()),
            (None, Some(u)) => (3, u.to_be_bytes().to_vec()),
            _ => (4, n.as_f64().unwrap_or(0.0).to_bits().to_be_bytes().to_vec()),
        },
        Value::String(s) => (5, s.as_bytes().to_vec()),
        other => (6, other.to_string().into_bytes()),
    }
}

/// The typed, length-delimited id (encoding 2 for every tuple; encoding 3
/// for the tuples ES encodes ambiguously): keys in name order, each written
/// as its length-delimited name and its typed, length-delimited value, so
/// no two distinct tuples share the hash input. At most 21 bytes (28
/// characters), never an ES id's 32.
pub fn typed_doc_id(keys: &BTreeMap<String, Value>) -> String {
    let mut buf = Vec::new();
    let mut prefix = Vec::new();
    for (name, v) in keys {
        let (tag, bytes) = typed_value(v);
        put_field(&mut buf, b'k', name.as_bytes());
        put_field(&mut buf, tag, &bytes);
        if prefix.len() < 5 {
            prefix.push(bytes.first().copied().unwrap_or(tag));
        }
    }
    id_of(prefix, &buf)
}

/// The ids older encodings gave a group (deleted when its transform is
/// rebuilt; the current id is never among them).
pub fn older_doc_ids(keys: &BTreeMap<String, Value>) -> Vec<String> {
    let current = doc_id(keys);
    let mut out: Vec<String> = [legacy_doc_id(keys), typed_doc_id(keys)].into_iter().filter(|i| *i != current).collect();
    out.dedup();
    out
}

/// A source search response that is not complete fails the checkpoint (ES
/// runs transform searches with `allow_partial_search_results=false` and
/// retries the checkpoint): a shard failure or a timeout would otherwise
/// complete it without the groups of the missing data.
pub fn check_complete(resp: &Value) -> Result<()> {
    let shards = &resp["_shards"];
    let failed = shards["failed"].as_u64().unwrap_or(0);
    if failed > 0 {
        let total = shards["total"].as_u64().unwrap_or(0);
        let mut e = EsError::new(503, "search_phase_execution_exception", format!("Partial shards failure ({failed} of {total} shards failed); the transform checkpoint needs complete source results"));
        if let Some(f) = shards["failures"].as_array().and_then(|a| a.first()) {
            let reason = &f["reason"];
            e = e.caused_by(EsError::from_error_json(f["status"].as_u64().unwrap_or(503) as u16, reason));
        }
        return Err(e);
    }
    if resp["timed_out"].as_bool() == Some(true) {
        return Err(EsError::new(503, "search_phase_execution_exception", "Search timed out; the transform checkpoint needs complete source results"));
    }
    Ok(())
}

/// Turn a composite response into documents.
pub fn convert_page(cfg: &Config, resp: &Value, types: &HashMap<String, String>) -> Result<Page> {
    let comp = &resp["aggregations"][COMPOSITE];
    let buckets = comp["buckets"].as_array().cloned().unwrap_or_default();
    let mut docs = Vec::with_capacity(buckets.len());
    let mut legacy_ids = Vec::with_capacity(buckets.len());
    let mut unresolved = Vec::new();
    let mut processed = 0u64;
    let epoch = cfg.settings.dates_as_epoch_millis;
    for b in &buckets {
        let dc = b["doc_count"].as_u64().unwrap_or(0);
        processed += dc;
        let key = b["key"].as_object().cloned().unwrap_or_default();
        let id_keys: BTreeMap<String, Value> = key.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let id = doc_id(&id_keys);
        legacy_ids.extend(older_doc_ids(&id_keys));
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
                if is_runtime(cfg, sort) && dc as usize > hits.len() {
                    if let Some(max) = b["_latest_max"]["value"].as_f64() {
                        unresolved.push(Unresolved { pos: docs.len(), key: key.clone(), max });
                    }
                }
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
    // The cursor is kept even when every bucket of the page was filtered.
    let after_key = comp.get("after_key").cloned().filter(|a| !a.is_null());
    Ok(Page { docs, legacy_ids, docs_processed: processed, after_key, unresolved })
}

/// Whether a pivot filters its buckets (`bucket_selector`): a short or empty
/// page then says nothing about later pages.
pub fn filters_buckets(cfg: &Config) -> bool {
    match &cfg.function {
        Function::Pivot { aggs, .. } => aggs.values().any(|d| agg_kind(d).as_deref() == Some("bucket_selector")),
        Function::Latest { .. } => false,
    }
}

/// The search finding the latest document of an [`Unresolved`] bucket: its
/// unique key, at the group's maximum sort value.
pub fn latest_lookup(cfg: &Config, query: &Value, u: &Unresolved) -> Option<Value> {
    let Function::Latest { sort, .. } = &cfg.function else { return None };
    let mut filters = vec![query.clone()];
    for (field, v) in &u.key {
        filters.push(if v.is_null() { json!({"bool": {"must_not": {"exists": {"field": field}}}}) } else { json!({"term": {field.clone(): v}}) });
    }
    let is_date = cfg.runtime_mappings.as_ref().and_then(|m| m.get(sort.as_str())).and_then(|d| d.get("type")).and_then(Value::as_str) == Some("date");
    let range = if is_date {
        let ms = u.max as i64;
        json!({"range": {sort.clone(): {"gte": ms, "lte": ms, "format": "epoch_millis"}}})
    } else {
        json!({"range": {sort.clone(): {"gte": u.max, "lte": u.max}}})
    };
    filters.push(range);
    let mut body = json!({"size": 1, "track_total_hits": false, "query": {"bool": {"filter": filters}}});
    if let Some(rm) = &cfg.runtime_mappings {
        body["runtime_mappings"] = rm.clone();
    }
    Some(body)
}

/// The destination document of a latest hit (its `_source` without
/// metadata fields).
pub fn latest_doc(hit: &Value) -> Option<Map<String, Value>> {
    let mut d = hit["_source"].as_object().cloned()?;
    d.retain(|k, _| !k.starts_with('_'));
    Some(d)
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

#[cfg(test)]
mod id_tests {
    use super::*;

    fn keys(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn delimiter_inside_values_no_longer_collides() {
        let a = keys(&[("a", json!("x$x")), ("b", json!("x"))]);
        let b = keys(&[("a", json!("x")), ("b", json!("x$x"))]);
        assert_eq!(legacy_doc_id(&a), legacy_doc_id(&b), "encoding 1 was ambiguous for these keys");
        assert_eq!(es_doc_id(&a), es_doc_id(&b), "so is ES's");
        assert_ne!(doc_id(&a), doc_id(&b));
    }

    #[test]
    fn null_is_not_the_null_marker_string_and_types_are_distinct() {
        let null = keys(&[("a", Value::Null)]);
        let marker = keys(&[("a", json!("__NULL_VALUE__"))]);
        assert_eq!(legacy_doc_id(&null), legacy_doc_id(&marker), "encoding 1 was ambiguous for these keys");
        assert_ne!(doc_id(&null), doc_id(&marker));
        let empty = keys(&[("a", json!(""))]);
        let empty_marker = keys(&[("a", json!("__EMPTY_VALUE__"))]);
        assert_ne!(doc_id(&empty), doc_id(&empty_marker));
        let variants = [json!(1), json!(1.0), json!("1"), json!(true), json!(null), json!(""), json!([1])];
        let ids: std::collections::HashSet<String> = variants.iter().map(|v| doc_id(&keys(&[("a", v.clone())]))).collect();
        assert_eq!(ids.len(), variants.len());
    }

    /// `_id`s Elasticsearch 9.5.5 wrote for these groups (batch pivot
    /// transforms run on a real 9.5.5 node, 9 Nov 2026).
    #[test]
    fn ids_are_the_ids_elasticsearch_writes() {
        let cases: Vec<(BTreeMap<String, Value>, &str)> = vec![
            (keys(&[("a", json!("")), ("b", json!("e"))]), "X2X1ntn1ERb7y1eSYGEe5f62AAAAAAAA"),
            (keys(&[("a", json!("foo")), ("b", json!("bar"))]), "ZmKcNTRLxPl7G_pBVjBEFPK9AAAAAAAA"),
            (keys(&[("a", json!("\u{e9}t\u{e9}")), ("b", json!("z"))]), "w3rDu5_tv8fyvHsy4JlGobAFAAAAAAAA"),
            // long, double, boolean, date_histogram (epoch millis), histogram.
            (keys(&[("bo", json!(true)), ("d", json!(0.0)), ("dh", json!(0)), ("h", json!(0.0)), ("l", json!(36))]), "AQAAAAAeV7qEQHy9WPK6apOqCP-DAAAA"),
            (keys(&[("bo", json!(true)), ("d", json!(1.5)), ("dh", json!(1_487_379_600_000i64)), ("h", json!(0.0)), ("l", json!(42))]), "AT8AAAAsfPIvxC5blRKYQOolRSsJAAAA"),
            (keys(&[("bo", json!(false)), ("d", json!(12.0)), ("dh", json!(1_487_386_800_000i64)), ("h", json!(10.0)), ("l", json!(-7))]), "AEAAQP8u1b9NfC2Xp544UGp1aZExAAAA"),
            // Seven keys: the prefix holds the first byte of the first six.
            (keys(&[("k1", json!("1")), ("k2", json!("2")), ("k3", json!("3")), ("k4", json!("4")), ("k5", json!("5")), ("k6", json!("6")), ("k7", json!("7"))]), "MTIzNDU2MqkKF3Hy97xKJBHAnD-o8gAA"),
            (keys(&[("k1", json!("b")), ("k2", json!("cc")), ("k3", json!("ddd")), ("k4", json!("eeee")), ("k5", json!("fffff")), ("k6", json!("gggggg")), ("k7", json!("hhhhhhh"))]), "YmNkZWZnlyEJupH6M3u-l9SiTdwGyAAA"),
        ];
        for (k, es) in &cases {
            assert_eq!(es_doc_id(k), *es, "{k:?}");
            assert_eq!(doc_id(k), *es, "an unambiguous tuple gets the id ES writes: {k:?}");
        }
    }

    /// The same run shows ES itself merging distinct groups: it wrote one
    /// document for `("x$x","x")` and `("x","x$x")`, and one for null and
    /// `"__NULL_VALUE__"`. `es_doc_id` reproduces those ids exactly; `doc_id`
    /// keeps the groups apart with typed ids no ES id can equal.
    #[test]
    fn tuples_es_merges_keep_their_own_documents() {
        let a = keys(&[("a", json!("x$x")), ("b", json!("x"))]);
        let b = keys(&[("a", json!("x")), ("b", json!("x$x"))]);
        assert_eq!(es_doc_id(&a), "eHjktIEmGEbxkJ82sNFdqgw2AAAAAAAA");
        assert_eq!(es_doc_id(&b), "eHjktIEmGEbxkJ82sNFdqgw2AAAAAAAA");
        let null = keys(&[("a", Value::Null), ("b", json!("n"))]);
        let marker = keys(&[("a", json!("__NULL_VALUE__")), ("b", json!("n"))]);
        assert_eq!(es_doc_id(&null), "X27T7iu6vNuvZt4nWKq_a3PsAAAAAAAA");
        assert_eq!(es_doc_id(&marker), "X27T7iu6vNuvZt4nWKq_a3PsAAAAAAAA");
        let ids: std::collections::HashSet<String> = [&a, &b, &null, &marker].iter().map(|k| doc_id(k)).collect();
        assert_eq!(ids.len(), 4);
        assert_eq!(doc_id(&null), es_doc_id(&null), "null itself keeps the ES id");
        for k in [&a, &b, &marker] {
            assert_ne!(doc_id(k).len(), 32, "a typed id never has an ES id's length");
        }
        // Rebuilds delete the older encodings' ids, never the current one.
        for k in [&a, &b, &null, &marker] {
            assert!(!older_doc_ids(k).contains(&doc_id(k)));
        }
    }

    #[test]
    fn partial_or_timed_out_source_results_are_rejected() {
        assert!(check_complete(&json!({"_shards": {"total": 2, "successful": 2, "failed": 0}, "timed_out": false})).is_ok());
        let e = check_complete(&json!({"_shards": {"total": 2, "successful": 1, "failed": 1, "failures": [{"shard": 1, "status": 503, "reason": {"type": "no_shard_available_action_exception", "reason": "gone"}}]}})).unwrap_err();
        assert_eq!(e.status, 503);
        assert_eq!(e.kind, "search_phase_execution_exception");
        assert_eq!(e.caused_by.as_ref().map(|c| c.kind.as_str()), Some("no_shard_available_action_exception"));
        let e = check_complete(&json!({"_shards": {"total": 1, "successful": 1, "failed": 0}, "timed_out": true})).unwrap_err();
        assert_eq!(e.status, 503);
    }
}
