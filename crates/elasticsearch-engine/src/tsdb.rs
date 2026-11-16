//! Time series (`index.mode=time_series`) validation: settings
//! combinations, `routing_path` / dimension mapping rules and the
//! `@timestamp` bounds enforced on writes.

use crate::date::DateFormatter;
use crate::error::{EsError, Result};
use crate::mapping::{EsType, Mapper, Mappings, ObjectMapping};
use crate::settings::FlatSettings;
use serde_json::Value;

pub const MODE: &str = "index.mode";
pub const ROUTING_PATH: &str = "index.routing_path";
pub const START_TIME: &str = "index.time_series.start_time";
pub const END_TIME: &str = "index.time_series.end_time";

fn setting_str(s: &FlatSettings, k: &str) -> Option<String> {
    match s.get(k)? {
        Value::String(x) => Some(x.clone()),
        Value::Null => None,
        Value::Array(a) => Some(a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(",")),
        other => Some(other.to_string()),
    }
}

pub fn is_time_series(s: &FlatSettings) -> bool {
    setting_str(s, MODE).as_deref() == Some("time_series")
}

fn routing_path(s: &FlatSettings) -> Vec<String> {
    match s.get(ROUTING_PATH) {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).map(str::to_string).filter(|x| !x.is_empty()).collect(),
        Some(Value::String(x)) => x.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => Vec::new(),
    }
}

fn parse_bound(k: &str, v: &str) -> Result<i64> {
    if v.trim().is_empty() {
        return Err(EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{v}]: cannot parse empty date")));
    }
    DateFormatter::new(crate::date::DEFAULT_FORMAT)?
        .parse(v)
        .map_err(|_| EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{v}]: failed to parse date field [{v}] with format [{}]", crate::date::DEFAULT_FORMAT)))
}

/// `Instant#toString`-like rendering of epoch nanos.
pub fn instant_string(nanos: i128) -> String {
    let secs = nanos.div_euclid(1_000_000_000) as i64;
    let frac = nanos.rem_euclid(1_000_000_000) as u32;
    let base = DateFormatter::new("yyyy-MM-dd'T'HH:mm:ss").map(|f| f.format(secs * 1000, crate::date::Tz::UTC)).unwrap_or_default();
    let f = if frac == 0 {
        String::new()
    } else if frac % 1_000_000 == 0 {
        format!(".{:03}", frac / 1_000_000)
    } else if frac % 1000 == 0 {
        format!(".{:06}", frac / 1000)
    } else {
        format!(".{frac:09}")
    };
    format!("{base}{f}Z")
}

struct Leaf<'a> {
    path: String,
    ty: Option<EsType>,
    params: &'a serde_json::Map<String, Value>,
    nested: bool,
    passthrough_dim: bool,
    /// Source path a multi-field reads its parent's values from.
    src: Option<String>,
}

fn collect<'a>(o: &'a ObjectMapping, prefix: &str, nested: bool, pt_dim: bool, out: &mut Vec<Leaf<'a>>) {
    for (k, m) in &o.properties {
        let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        match m {
            Mapper::Field(f) => {
                // Dimension multi-fields take their parent's values.
                for (sub, sf) in &f.fields {
                    if sf.params.get("time_series_dimension").and_then(Value::as_bool) == Some(true) {
                        out.push(Leaf { path: format!("{full}.{sub}"), ty: Some(sf.ty), params: &sf.params, nested, passthrough_dim: false, src: Some(full.clone()) });
                    }
                }
                out.push(Leaf { path: full, ty: Some(f.ty), params: &f.params, nested, passthrough_dim: pt_dim, src: None });
            }
            Mapper::Object(c) => {
                let pd = pt_dim || c.params.get("time_series_dimension").and_then(Value::as_bool) == Some(true);
                out.push(Leaf { path: full.clone(), ty: None, params: &c.params, nested: nested || c.nested, passthrough_dim: pd, src: None });
                collect(c, &full, nested || c.nested, pd, out);
            }
        }
    }
}

fn dimension_params(ty: Option<EsType>, params: &serde_json::Map<String, Value>, passthrough: bool) -> bool {
    match ty {
        Some(EsType::Flattened) => params.get("time_series_dimensions").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) && !params.contains_key("script"),
        Some(_) => (params.get("time_series_dimension").and_then(Value::as_bool) == Some(true) || passthrough) && !params.contains_key("script"),
        None => false,
    }
}
fn is_dimension(l: &Leaf) -> bool { dimension_params(l.ty, l.params, l.passthrough_dim) }

/// Write-time consumers only need dimensions or position metrics. Reuse the
/// traversal path and allocate owned paths only for selected leaves, instead
/// of materializing every metric path on every document.
fn collect_selected<'a, const DIMENSIONS: bool>(o: &'a ObjectMapping, path: &mut String, nested: bool, pt_dim: bool, out: &mut Vec<Leaf<'a>>) {
    fn wanted<const D: bool>(ty: Option<EsType>, p: &serde_json::Map<String, Value>, pt: bool) -> bool {
        if D { dimension_params(ty, p, pt) } else { p.get("time_series_metric").and_then(Value::as_str) == Some("position") }
    }
    for (name, mapper) in &o.properties {
        let parent_len = path.len();
        if parent_len > 0 { path.push('.'); }
        path.push_str(name);
        match mapper {
            Mapper::Field(field) => {
                for (sub, sf) in &field.fields {
                    if sf.params.get("time_series_dimension").and_then(Value::as_bool) == Some(true)
                        && wanted::<DIMENSIONS>(Some(sf.ty), &sf.params, false) {
                        let source = path.clone();
                        let len = path.len(); path.push('.'); path.push_str(sub);
                        out.push(Leaf { path: path.clone(), ty: Some(sf.ty), params: &sf.params, nested, passthrough_dim: false, src: Some(source) });
                        path.truncate(len);
                    }
                }
                if wanted::<DIMENSIONS>(Some(field.ty), &field.params, pt_dim) {
                    out.push(Leaf { path: path.clone(), ty: Some(field.ty), params: &field.params, nested, passthrough_dim: pt_dim, src: None });
                }
            }
            Mapper::Object(child) => {
                let pd = pt_dim || child.params.get("time_series_dimension").and_then(Value::as_bool) == Some(true);
                let nested = nested || child.nested;
                if wanted::<DIMENSIONS>(None, &child.params, pd) {
                    out.push(Leaf { path: path.clone(), ty: None, params: &child.params, nested, passthrough_dim: pd, src: None });
                }
                collect_selected::<DIMENSIONS>(child, path, nested, pd, out);
            }
        }
        path.truncate(parent_len);
    }
}

/// Immutable selected fields, rebuilt with the mapping/FieldLookup snapshot.
/// Callers must pair this with the mapping from which it was prepared.
#[derive(Clone, Default, Debug)]
pub struct PreparedFields {
    dimensions: Vec<PreparedDimension>,
    positions: Vec<String>,
}
#[derive(Clone, Debug)]
struct PreparedDimension {
    path: String,
    ty: Option<EsType>,
    params: serde_json::Map<String, Value>,
    src: Option<String>,
}
impl PreparedFields {
    pub fn new(m: &Mappings) -> Self {
        let mut leaves = Vec::new();
        collect_selected::<true>(&m.root, &mut String::new(), false, false, &mut leaves);
        let dimensions = leaves.into_iter().map(|l| PreparedDimension { path: l.path, ty: l.ty, params: l.params.clone(), src: l.src }).collect();
        let mut leaves = Vec::new();
        collect_selected::<false>(&m.root, &mut String::new(), false, false, &mut leaves);
        Self { dimensions, positions: leaves.into_iter().map(|l| l.path).collect() }
    }
    pub fn build_ids(&self, m: &Mappings, s: &FlatSettings, src: &Value) -> Result<TsIds> {
        // Only selected dimensions are materialized; no full mapping walk.
        let leaves: Vec<_> = self.dimensions.iter().map(|l| Leaf { path: l.path.clone(), ty: l.ty, params: &l.params, nested: false, passthrough_dim: false, src: l.src.clone() }).collect();
        build_ids_from_leaves(m, s, src, &leaves)
    }
    /// The routing hash an id-less write of `src` is routed by (see
    /// [`routing_hash`]).
    pub fn routing_hash(&self, m: &Mappings, s: &FlatSettings, src: &Value) -> Result<i32> {
        let leaves: Vec<_> = self.dimensions.iter().map(|l| Leaf { path: l.path.clone(), ty: l.ty, params: &l.params, nested: false, passthrough_dim: false, src: l.src.clone() }).collect();
        routing_hash_from_leaves(m, s, src, &leaves)
    }
    pub fn check_positions(&self, src: &Value) -> Result<()> {
        check_position_paths(self.positions.iter().map(String::as_str), src)
    }
}

/// Mapping rules that hold for every index.
pub fn validate_mappings_any(m: &Mappings) -> Result<()> {
    let mut leaves = Vec::new();
    collect(&m.root, "", false, false, &mut leaves);
    for l in &leaves {
        if l.nested && l.ty.is_some() && l.params.get("time_series_dimension").and_then(Value::as_bool) == Some(true) {
            return Err(EsError::mapper_parsing(format!("time_series_dimension can't be configured in nested field [{}]", l.path)));
        }
    }
    Ok(())
}

/// Validate settings + mappings at index creation.
pub fn validate_create(s: &FlatSettings, m: &Mappings) -> Result<()> {
    validate_mappings_any(m)?;
    let ts = is_time_series(s);
    if !ts {
        if !routing_path(s).is_empty() && !crate::indexsort::routes_on_sort_fields(s) {
            return Err(EsError::illegal_argument(format!("[{ROUTING_PATH}] requires [{MODE}=time_series]")));
        }
        for k in [START_TIME, END_TIME] {
            if s.get(k).is_some_and(|v| !v.is_null()) {
                return Err(EsError::illegal_argument(format!("[{k}] requires [{MODE}=time_series]")));
            }
        }
        return Ok(());
    }
    for k in ["index.sort.field", "index.sort.order", "index.sort.mode", "index.sort.missing", "index.routing_partition_size"] {
        if s.get(k).is_some_and(|v| !v.is_null()) {
            return Err(EsError::illegal_argument(format!("[{MODE}=time_series] is incompatible with [{k}]")));
        }
    }
    if m.routing_required() {
        return Err(EsError::illegal_argument(format!("routing is forbidden on CRUD operations that target indices in [{MODE}=time_series]")));
    }
    let (mut start, mut end) = (None, None);
    if let Some(v) = setting_str(s, START_TIME) {
        start = Some(parse_bound(START_TIME, &v)?);
    }
    if let Some(v) = setting_str(s, END_TIME) {
        end = Some(parse_bound(END_TIME, &v)?);
    }
    if let (Some(a), Some(b)) = (start, end) {
        if b <= a {
            return Err(EsError::illegal_argument(format!("index.time_series.end_time [{}] must be larger than index.time_series.start_time [{}]", instant_string(b as i128 * 1_000_000), instant_string(a as i128 * 1_000_000))));
        }
    }
    let mut leaves = Vec::new();
    collect(&m.root, "", false, false, &mut leaves);
    // @timestamp must be date / date_nanos when mapped.
    if let Some(l) = leaves.iter().find(|l| l.path == "@timestamp") {
        if let Some(t) = l.ty {
            if !t.is_date() {
                return Err(EsError::illegal_argument(format!("data stream timestamp field [@timestamp] is of type [{}], but [date,date_nanos] is expected", t.name())));
            }
        }
    }
    let rp = routing_path(s);
    if rp.is_empty() {
        // A pass-through object with `time_series_dimension` routes on its leaves.
        let pt_dim = leaves.iter().any(|l| l.ty.is_none() && l.params.get("type").and_then(Value::as_str) == Some("passthrough") && l.passthrough_dim);
        if !leaves.iter().any(is_dimension) && !pt_dim {
            return Err(EsError::illegal_argument(format!("[{MODE}=time_series] requires a non-empty [{ROUTING_PATH}]")));
        }
        return Ok(());
    }
    for pat in &rp {
        for l in leaves.iter().filter(|l| if pat.contains('*') { l.ty.is_some() && crate::mapping::wildcard_match(pat, &l.path) } else { *pat == l.path }) {
            if !is_dimension(l) {
                let was = match l.ty {
                    None => " was [object]".to_string(),
                    Some(_) => " was not a dimension".to_string(),
                };
                let tail = if l.ty.is_none() { format!("[{}]{was}.", l.path) } else { format!("[{}]{was}.", l.path) };
                return Err(EsError::illegal_argument(format!(
                    "All fields that match routing_path must be configured with [time_series_dimension: true] or flattened fields with a list of dimensions in [time_series_dimensions] and without the [script] parameter. {tail}"
                )));
            }
        }
    }
    Ok(())
}

/// Bounds `(start_ms, end_ms)` if this is a time series index.
pub fn bounds(s: &FlatSettings) -> Option<(Option<i64>, Option<i64>)> {
    if !is_time_series(s) {
        return None;
    }
    let p = |k: &str| setting_str(s, k).and_then(|v| parse_bound(k, &v).ok());
    Some((p(START_TIME), p(END_TIME)))
}

/// Epoch nanos of a document's `@timestamp` value.
fn timestamp_nanos(v: &Value) -> Option<i128> {
    let fmt = DateFormatter::new(crate::date::DEFAULT_FORMAT).ok()?;
    match v {
        Value::String(s) => {
            let ms = fmt.parse(s).ok()?;
            let sub = s.find('.').map(|i| {
                let d: String = s[i + 1..].chars().take_while(|c| c.is_ascii_digit()).collect();
                let mut d = d;
                while d.len() < 9 {
                    d.push('0');
                }
                d[3..9].parse::<i128>().unwrap_or(0)
            });
            Some(ms as i128 * 1_000_000 + sub.unwrap_or(0))
        }
        Value::Number(n) => Some(n.as_i64()? as i128 * 1_000_000),
        _ => None,
    }
}

/// Reject a document whose `@timestamp` is outside the index time range.
pub fn check_timestamp(src: &Value, bounds: (Option<i64>, Option<i64>)) -> Result<()> {
    let Some(ts) = src.get("@timestamp").and_then(timestamp_nanos) else { return Ok(()) };
    let fail = |msg: String| {
        EsError::document_parsing(format!("[1:1] failed to parse: {msg}")).caused_by(EsError::illegal_argument(msg))
    };
    let shown = instant_string(ts);
    if let Some(a) = bounds.0 {
        if ts < a as i128 * 1_000_000 {
            return Err(fail(format!("time series index @timestamp value [{shown}] must be larger than {}", instant_string(a as i128 * 1_000_000))));
        }
    }
    if let Some(b) = bounds.1 {
        if ts >= b as i128 * 1_000_000 {
            return Err(fail(format!("time series index @timestamp value [{shown}] must be smaller than {}", instant_string(b as i128 * 1_000_000))));
        }
    }
    Ok(())
}

/// `PUT /{index}/_settings` rules for the time series settings.
pub fn validate_put_settings(cur: &FlatSettings, upd: &FlatSettings, open: bool, name: &str) -> Result<()> {
    let ts = is_time_series(cur);
    if let Some(v) = upd.get(END_TIME).filter(|v| !v.is_null()) {
        if !ts {
            return Err(EsError::illegal_argument(format!("[{END_TIME}] requires [{MODE}=time_series]")));
        }
        let new = parse_bound(END_TIME, v.as_str().unwrap_or(&v.to_string()))?;
        if let Some(Some(old)) = bounds(cur).map(|b| b.1) {
            if new < old {
                return Err(EsError::illegal_argument(format!("{END_TIME} must be larger than current value [{old}] but was [{new}]")));
            }
        }
    }
    if !open {
        for k in [START_TIME, ROUTING_PATH, MODE] {
            if upd.contains_key(k) {
                return Err(EsError::illegal_argument(format!("final {name} setting [{k}], not updateable")));
            }
        }
    }
    Ok(())
}

/// Time series indices always map `@timestamp` as a date and carry the
/// `_data_stream_timestamp` meta field.
pub fn prepare_mappings(s: &FlatSettings, m: &mut Mappings, reg: &crate::analysis::AnalysisRegistry) -> Result<()> {
    if !is_time_series(s) {
        return Ok(());
    }
    if m.runtime.as_ref().and_then(|r| r.get("@timestamp")).is_some() {
        return Err(EsError::illegal_argument("docvalues not found for index sort field:[@timestamp]"));
    }
    let (has_ts, has_tsid) = {
        let mut leaves = Vec::new();
        collect(&m.root, "", false, false, &mut leaves);
        (leaves.iter().any(|l| l.path == "@timestamp"), leaves.iter().any(|l| l.path == TSID_FIELD))
    };
    if !has_ts {
        let add = Mappings::parse(&serde_json::json!({"properties": {"@timestamp": {"type": "date"}}}), reg)?;
        m.merge(&add)?;
    }
    m.data_stream_timestamp = true;
    // A synthetic `_id` is rebuilt from `_tsid`, `@timestamp` and the
    // routing hash (ES `TimeSeriesRoutingHashFieldMapper`: doc values only).
    if synthetic_id(s) && !m.root.properties.contains_key(ROUTING_HASH_FIELD) {
        let add = Mappings::parse(&serde_json::json!({"properties": {ROUTING_HASH_FIELD: {"type": "long", "index": false}}}), reg)?;
        m.merge(&add)?;
    }
    if !has_tsid {
        // ES `TimeSeriesIdFieldMapper`: doc values only (sorted, and the
        // index sort's first key), never searchable, so no postings: they
        // cost 1.6-1.9 GB per tsdb_k8s_queries backing index.
        let add = Mappings::parse(&serde_json::json!({"properties": {TSID_FIELD: {"type": "keyword", "index": false}}}), reg)?;
        m.merge(&add)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// _tsid / _id generation (ES 8.13+ hashed `_tsid`).

pub const TSID_FIELD: &str = "_tsid";
/// The hidden routing-hash column of a synthetic-id time series index.
pub const ROUTING_HASH_FIELD: &str = "_ts_routing_hash";

/// [`rustic_core::synthetic_id`] codec of synthetic time series ids
/// (`TsidExtractingIdFieldMapper.createSyntheticId`: the `_tsid` bytes,
/// `Long.MAX_VALUE - timestamp` and the routing hash, big endian, URL-safe
/// base64): an id's stored `_tsid` value and timestamp, and back.
pub fn install_synthetic_id_codec() {
    fn decode(id: &[u8]) -> Option<(Vec<u8>, i64)> {
        use base64::Engine;
        let b = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id).ok()?;
        let tsid_len = b.len().checked_sub(12).filter(|&n| n > 0)?;
        let rev = i64::from_be_bytes(b[tsid_len..tsid_len + 8].try_into().ok()?);
        let tsid = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&b[..tsid_len]);
        Some((tsid_stored(&tsid).into_bytes(), i64::MAX - rev))
    }
    fn encode(tsid: &[u8], ts: i64, rh: i32) -> Option<String> {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let mut idb = b64.decode(tsid_display(std::str::from_utf8(tsid).ok()?)).ok()?;
        idb.extend_from_slice(&(i64::MAX - ts).to_be_bytes());
        idb.extend_from_slice(&rh.to_be_bytes());
        Some(b64.encode(&idb))
    }
    rustic_core::synthetic_id::install(rustic_core::synthetic_id::Codec { decode, encode });
}

/// The hidden fields a time series document carries besides its source:
/// `_tsid` and, for a synthetic `_id`, its routing hash.
///
/// Each carries exactly the one generated value: any other value of those
/// columns in the parsed document (the source parser rejects them as
/// metadata fields; this guards every other path) is dropped first, since
/// a second value would turn the single-valued identity column the
/// synthetic `_id` is rebuilt and looked up from into a multi-valued one.
pub fn push_hidden_fields(lookup: &crate::mapping::FieldLookup, ids: &TsIds, doc: &mut rustic_core::types::Document) {
    if let Some(field) = lookup.get(TSID_FIELD).and_then(|f| f.core_id) {
        doc.fields.retain(|(f, _)| *f != field);
        doc.fields.push((field, rustic_core::FieldValue::Keyword(tsid_stored(&ids.tsid))));
    }
    if let Some(field) = lookup.get(ROUTING_HASH_FIELD).and_then(|f| f.core_id) {
        doc.fields.retain(|(f, _)| *f != field);
        if let Some(rh) = routing_hash_of_id(&ids.id) {
            doc.fields.push((field, rustic_core::FieldValue::I64(rh as i64)));
        }
    }
}

/// URL-safe base64 alphabet (how `_tsid` is displayed) and an alphabet in
/// ascending ASCII order: the `_tsid` doc value is stored in the latter so
/// that comparing the strings compares the raw tsid bytes (ES sorts the
/// `_tsid` `BytesRef`), and translated back for display.
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const B64_SORTABLE: &[u8; 64] = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

/// Byte `b` -> `to[i]` where `from[i] == b`; every other byte unchanged.
const fn translation(from: &[u8; 64], to: &[u8; 64]) -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut b = 0;
    while b < 256 {
        t[b] = b as u8;
        b += 1;
    }
    // The first position of each byte wins (as a search from the start).
    let mut i = 64;
    while i > 0 {
        i -= 1;
        t[from[i] as usize] = to[i];
    }
    t
}

const URL_TO_SORTABLE: [u8; 256] = translation(B64_URL, B64_SORTABLE);
const SORTABLE_TO_URL: [u8; 256] = translation(B64_SORTABLE, B64_URL);

/// `s` with each byte mapped through `table` (base64 alphabets map ASCII to
/// ASCII; other bytes are unchanged), a table lookup per byte.
fn translate(s: &str, table: &[u8; 256]) -> String {
    s.bytes().map(|b| table[b as usize] as char).collect()
}

/// The stored (order-preserving) form of a displayed `_tsid`.
pub fn tsid_stored(display: &str) -> String {
    translate(display, &URL_TO_SORTABLE)
}

/// The displayed (URL-safe base64) form of a stored `_tsid` doc value.
pub fn tsid_display(stored: &str) -> String {
    translate(stored, &SORTABLE_TO_URL)
}

#[cfg(test)]
mod translate_tests {
    use super::*;

    /// The tables map exactly as searching the alphabets did, both ways,
    /// every byte (non-ASCII bytes of a string pass through as before).
    #[test]
    fn translation_tables_match_the_alphabet_search() {
        let search = |b: u8, from: &[u8; 64], to: &[u8; 64]| from.iter().position(|c| *c == b).map(|i| to[i]).unwrap_or(b);
        for b in 0..=255u8 {
            assert_eq!(URL_TO_SORTABLE[b as usize], search(b, B64_URL, B64_SORTABLE), "{b}");
            assert_eq!(SORTABLE_TO_URL[b as usize], search(b, B64_SORTABLE, B64_URL), "{b}");
        }
        let id = "LBRk0D5uktQNwLv5DMr2K5AATBIbAF2VRQxMDTouOYmk7qtVNkC4SOA0VmOs-_";
        assert_eq!(tsid_display(&tsid_stored(id)), id);
    }
}

fn murmur_fmix(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^ (k >> 33)
}

/// MurmurHash3 x64 128-bit (seed 0 unless given), `(h1, h2)`.
pub fn murmur3_128(data: &[u8], seed: u64) -> (u64, u64) {
    const C1: u64 = 0x87c37b91114253d5;
    const C2: u64 = 0x4cf5ad432745937f;
    let (mut h1, mut h2) = (seed, seed);
    let n = data.len();
    for ch in data.chunks_exact(16) {
        let mut k1 = u64::from_le_bytes(ch[..8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(ch[8..].try_into().unwrap());
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(27).wrapping_add(h2).wrapping_mul(5).wrapping_add(0x52dce729);
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2.rotate_left(31).wrapping_add(h1).wrapping_mul(5).wrapping_add(0x38495ab5);
    }
    let t = &data[n / 16 * 16..];
    let (mut k1, mut k2) = (0u64, 0u64);
    for i in (8..t.len()).rev() {
        k2 ^= (t[i] as u64) << ((i - 8) * 8);
    }
    if t.len() > 8 {
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
    }
    for i in (0..t.len().min(8)).rev() {
        k1 ^= (t[i] as u64) << (i * 8);
    }
    if !t.is_empty() {
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
    }
    h1 ^= n as u64;
    h2 ^= n as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = murmur_fmix(h1);
    h2 = murmur_fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (h1, h2)
}

/// MurmurHash3 x86 32-bit (Lucene `StringHelper.murmurhash3_x86_32`).
pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;
    let mut h = seed;
    for ch in data.chunks_exact(4) {
        let mut k = u32::from_le_bytes(ch.try_into().unwrap());
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe6546b64);
    }
    let t = &data[data.len() / 4 * 4..];
    let mut k = 0u32;
    for (i, b) in t.iter().enumerate().rev() {
        k ^= (*b as u32) << (i * 8);
    }
    if !t.is_empty() {
        h ^= k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
    }
    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^ (h >> 16)
}

fn vint(mut n: usize, out: &mut Vec<u8>) {
    while n >= 0x80 {
        out.push((n & 0x7f) as u8 | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

fn oversize(n: usize) -> usize {
    (n + (n >> 3).max(3) + 7) & !7
}

/// Backing-array length of a `BytesStreamOutput` after the given writes
/// (ES hashes the whole backing array, trailing zeros included).
fn grown_len(writes: &[usize]) -> usize {
    let (mut cap, mut count) = (0usize, 0usize);
    for w in writes {
        count += w;
        if count > cap {
            cap = oversize(count);
        }
    }
    cap
}

struct DimVal {
    bytes: Vec<u8>,
    padded: Vec<u8>,
}

fn string_val(s: &str) -> DimVal {
    let mut bytes = vec![b's'];
    vint(s.len(), &mut bytes);
    bytes.extend_from_slice(s.as_bytes());
    let mut vl = Vec::new();
    vint(s.len(), &mut vl);
    let cap = grown_len(&[1, vl.len(), s.len()]);
    let mut padded = bytes.clone();
    padded.resize(cap, 0);
    DimVal { bytes, padded }
}

fn long_val(v: i64) -> DimVal {
    let mut bytes = vec![b'l'];
    bytes.extend_from_slice(&v.to_be_bytes());
    let mut padded = bytes.clone();
    padded.resize(grown_len(&[1, 8]), 0);
    DimVal { bytes, padded }
}

fn unsigned_val(v: i64) -> DimVal {
    let mut bytes = vec![b'u'];
    bytes.extend_from_slice(&v.to_be_bytes());
    let mut padded = bytes.clone();
    padded.resize(grown_len(&[1, 8]), 0);
    DimVal { bytes, padded }
}

fn bool_val(v: bool) -> DimVal {
    let bytes = vec![b'b', if v { b't' } else { b'f' }];
    let mut padded = bytes.clone();
    padded.resize(grown_len(&[1, 1]), 0);
    DimVal { bytes, padded }
}

pub struct TsIds {
    pub tsid: String,
    pub id: String,
}

fn scalar_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn add_str(dims: &mut std::collections::BTreeMap<Vec<u8>, Vec<DimVal>>, routing: &mut Vec<(Vec<u8>, i32, usize)>, routed: bool, path: &str, text: &str) {
    dims.entry(path.as_bytes().to_vec()).or_default().push(string_val(text));
    if routed {
        let order = routing.len();
        routing.push((path.as_bytes().to_vec(), murmur3_32(text.as_bytes(), 0) as i32, order));
    }
}

/// Compute the `_tsid` and time series `_id` of a document.
/// The routing hash of a time series `_id` (`IndexRouting.ExtractFromSource`
/// routes a get / delete by it): the first 4 bytes (little endian) of a
/// standard 20-byte id, the last 4 (big endian) of a synthetic id.
pub fn routing_hash_of_id(id: &str) -> Option<i32> {
    use base64::Engine;
    let b = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id).ok()?;
    if b.len() == 20 {
        let h: [u8; 4] = b.get(..4)?.try_into().ok()?;
        return Some(i32::from_le_bytes(h));
    }
    let h: [u8; 4] = b.get(b.len().checked_sub(4)?..)?.try_into().ok()?;
    Some(i32::from_be_bytes(h))
}

/// `index.mapping.synthetic_id`: explicit, else on for time series indices
/// created on or after `TIME_SERIES_USE_SYNTHETIC_ID_DEFAULT_PROD` with the
/// default / best_compression codec.
pub fn synthetic_id(s: &FlatSettings) -> bool {
    use crate::version::IndexVersion;
    if let Some(v) = s.get("index.mapping.synthetic_id") {
        return v.as_bool().unwrap_or_else(|| v.as_str() == Some("true"));
    }
    if !is_time_series(s) {
        return false;
    }
    let created = s.get("index.version.created").map_or(IndexVersion::CURRENT, |v| IndexVersion::from_setting(Some(v)));
    let codec_ok = s.get("index.codec").and_then(Value::as_str).is_none_or(|c| matches!(c, "default" | "best_compression"));
    codec_ok && created.on_or_after(IndexVersion::TIME_SERIES_USE_SYNTHETIC_ID_DEFAULT_PROD)
}

/// The routing hash of a document of a time series index (its
/// `routing_path` dimensions), from the index's mapping JSON and settings:
/// what a coordinator routes an id-less write by.
pub fn source_routing_hash(mappings: &Value, s: &FlatSettings, src: &Value) -> Result<i32> {
    let m = Mappings::parse(mappings, &crate::analysis::AnalysisRegistry::default())?;
    routing_hash(&m, s, src)
}

/// The routing hash of an id-less write of `src` into a time series index
/// (`IndexRouting.ExtractFromSource`): the `routing_path` values of the
/// source, mapped or not yet mapped (a dimension a dynamic template maps
/// only once the document is parsed on its shard routes the same as it
/// will once mapped). Equals the routing hash of the `_id` the shard
/// generates once those fields are mapped as dimensions.
pub fn routing_hash(m: &Mappings, s: &FlatSettings, src: &Value) -> Result<i32> {
    let mut leaves = Vec::new();
    collect_selected::<true>(&m.root, &mut String::new(), false, false, &mut leaves);
    routing_hash_from_leaves(m, s, src, &leaves)
}

fn routing_hash_from_leaves(m: &Mappings, s: &FlatSettings, src: &Value, leaves: &[Leaf<'_>]) -> Result<i32> {
    let (dims, mut routing) = collect_identity(m, s, src, leaves)?;
    if dims.is_empty() && routing.is_empty() {
        return Err(EsError::illegal_argument("Error extracting routing: source didn't contain any routing fields"));
    }
    Ok(fold_routing(&mut routing))
}

type IdentityDims = std::collections::BTreeMap<Vec<u8>, Vec<DimVal>>;
/// (name bytes, value hash, order) of the routing path matches.
type RoutingHashes = Vec<(Vec<u8>, i32, usize)>;

/// ES `RoutingHashBuilder.buildHash`: the matches sorted by name (then
/// order), folded.
fn fold_routing(routing: &mut RoutingHashes) -> i32 {
    routing.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
    let mut rh: i32 = 0;
    for (name, h, _) in routing.iter() {
        rh = rh.wrapping_mul(31).wrapping_add((murmur3_32(name, 0) as i32) ^ h);
    }
    rh
}

/// Is `path` a mapped field or object (a flattened field's keys and a
/// field's multi-fields included)? Keys may themselves be dotted.
fn mapped_at(o: &ObjectMapping, path: &str) -> bool {
    for (k, m) in &o.properties {
        if path == k {
            return true;
        }
        let Some(rest) = path.strip_prefix(k.as_str()).and_then(|r| r.strip_prefix('.')) else { continue };
        let found = match m {
            Mapper::Object(c) => mapped_at(c, rest),
            Mapper::Field(f) => f.ty == EsType::Flattened || f.fields.contains_key(rest),
        };
        if found {
            return true;
        }
    }
    false
}

/// Source values matching `routing_path` that no mapping covers yet,
/// hashed as their text (`IndexRouting.ExtractFromSource` hashes every
/// matching source value, mapped or not): a dynamic dimension routes the
/// same before and after its mapping exists.
fn add_unmapped_routing(root: &ObjectMapping, rp: &[String], v: &Value, path: &str, routing: &mut RoutingHashes) {
    match v {
        Value::Object(o) => {
            for (k, x) in o {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                add_unmapped_routing(root, rp, x, &p, routing);
            }
        }
        Value::Array(a) => {
            for x in a {
                add_unmapped_routing(root, rp, x, path, routing);
            }
        }
        Value::Null => {}
        scalar => {
            if !path.is_empty() && rp.iter().any(|p| crate::mapping::wildcard_match(p, path)) && !mapped_at(root, path) {
                let order = routing.len();
                routing.push((path.as_bytes().to_vec(), murmur3_32(scalar_text(scalar).as_bytes(), 0) as i32, order));
            }
        }
    }
}

/// Does `src` hold an explicit `null` (or only nulls) at `path` (dotted
/// keys at any level)? Such a field indexes its `null_value`.
fn explicit_null(src: &Value, path: &str) -> bool {
    match src {
        Value::Array(a) => a.iter().any(|x| explicit_null(x, path)),
        Value::Object(o) => {
            if let Some(x) = o.get(path) {
                return match x {
                    Value::Null => true,
                    Value::Array(a) => !a.is_empty() && a.iter().all(Value::is_null),
                    _ => false,
                };
            }
            path.match_indices('.').any(|(i, _)| o.get(&path[..i]).is_some_and(|c| explicit_null(c, &path[i + 1..])))
        }
        _ => false,
    }
}

/// The `@timestamp` value a time series `_id` encodes: in the field's own
/// resolution (epoch nanos for `date_nanos`, as its indexed column holds
/// them and synthetic ids are looked up and rebuilt from that column),
/// else epoch millis.
fn id_timestamp(m: &Mappings, src: &Value) -> Option<i64> {
    let v = src.get("@timestamp")?;
    match m.root.properties.get("@timestamp") {
        Some(Mapper::Field(f)) if f.ty == EsType::DateNanos => {
            use rustic_core::fieldtypes::numeric;
            let fmt = DateFormatter::new(f.params.get("format").and_then(Value::as_str).unwrap_or(crate::date::DEFAULT_FORMAT)).ok()?;
            match v {
                Value::String(s) => {
                    let ms = fmt.parse(s).ok()?;
                    match numeric::fraction_nanos(s) {
                        Some(frac) => numeric::nanos_from_parts(ms.div_euclid(1000), frac).ok(),
                        None => numeric::nanos_from_millis(ms, 0).ok(),
                    }
                }
                Value::Number(n) => {
                    let x = n.as_f64()?;
                    let ms = fmt.parse_number(x.trunc()).ok()?;
                    numeric::nanos_from_millis(ms, ((x - x.trunc()) * 1e6).round() as u32).ok()
                }
                _ => None,
            }
        }
        _ => match v {
            Value::String(s) => DateFormatter::new(crate::date::DEFAULT_FORMAT).ok()?.parse(s).ok(),
            Value::Number(n) => n.as_i64(),
            _ => None,
        },
    }
}

pub fn build_ids(m: &Mappings, s: &FlatSettings, src: &Value) -> Result<TsIds> {
    let mut leaves = Vec::new();
    collect_selected::<true>(&m.root, &mut String::new(), false, false, &mut leaves);
    build_ids_from_leaves(m, s, src, &leaves)
}
fn build_ids_from_leaves(m: &Mappings, s: &FlatSettings, src: &Value, leaves: &[Leaf<'_>]) -> Result<TsIds> {
    let (dims, mut routing) = collect_identity(m, s, src, leaves)?;
    build_ids_from_identity(m, s, src, dims, &mut routing)
}

/// The dimension values (`_tsid` input, in canonical indexed form) and the
/// routing hashes of a time series document.
fn collect_identity(m: &Mappings, s: &FlatSettings, src: &Value, leaves: &[Leaf<'_>]) -> Result<(IdentityDims, RoutingHashes)> {
    let rp = routing_path(s);
    if rp.iter().any(|p| p.contains('*')) {
        check_unmapped_routing(m, &rp, src)?;
    }
    let in_routing = |path: &str| rp.is_empty() || rp.iter().any(|p| crate::mapping::wildcard_match(p, path));
    let mut dims: IdentityDims = IdentityDims::new();
    let mut routing: RoutingHashes = Vec::new();
    for l in leaves {
        match l.ty {
            Some(EsType::Flattened) => {
                let names: Vec<&str> = l.params.get("time_series_dimensions").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                for n in names {
                    let path = format!("{}.{n}", l.path);
                    for v in crate::highlight::source_values(src, &path) {
                        add_str(&mut dims, &mut routing, in_routing(&path), &path, &scalar_text(v));
                    }
                }
            }
            Some(EsType::Boolean) => {
                for v in crate::highlight::source_values(src, l.src.as_deref().unwrap_or(&l.path)) {
                    let b = match v {
                        Value::Bool(b) => Some(*b),
                        Value::String(x) if x == "true" || x == "false" => Some(x == "true"),
                        _ => None,
                    };
                    if let Some(b) = b {
                        if in_routing(&l.path) {
                            let order = routing.len();
                            routing.push((l.path.as_bytes().to_vec(), murmur3_32(if b { b"true" } else { b"false" }, 0) as i32, order));
                        }
                        dims.entry(l.path.as_bytes().to_vec()).or_default().push(bool_val(b));
                    }
                }
            }
            // ES `addUnsignedLong`: a long when it fits one, otherwise 'u'
            // and the value shifted by Long.MIN_VALUE.
            Some(EsType::UnsignedLong) => {
                for v in crate::highlight::source_values(src, l.src.as_deref().unwrap_or(&l.path)) {
                    let n = match v {
                        Value::Number(n) => n.as_u64(),
                        Value::String(x) => x.trim().parse::<u64>().ok(),
                        _ => None,
                    };
                    if let Some(n) = n {
                        let dv = if n <= i64::MAX as u64 { long_val(n as i64) } else { unsigned_val((n ^ (1u64 << 63)) as i64) };
                        dims.entry(l.path.as_bytes().to_vec()).or_default().push(dv);
                    }
                }
            }
            // The value the document indexes (`parse_integer`, the
            // indexing parser: coercion truncates numeric strings and
            // fractions, `null_value` stands in for null); a value it
            // would not index is not part of the identity either.
            Some(t) if t.is_integer() => {
                let (min, max): (i64, i64) = match t {
                    EsType::Byte => (i8::MIN as i64, i8::MAX as i64),
                    EsType::Short => (i16::MIN as i64, i16::MAX as i64),
                    EsType::Integer => (i32::MIN as i64, i32::MAX as i64),
                    _ => (i64::MIN, i64::MAX),
                };
                let coerce = match l.params.get("coerce") {
                    Some(Value::Bool(b)) => *b,
                    Some(Value::String(x)) => x == "true",
                    _ => true,
                };
                let path = l.src.as_deref().unwrap_or(&l.path);
                let mut vals = crate::highlight::source_values(src, path);
                if vals.is_empty() {
                    if let Some(nv) = l.params.get("null_value").filter(|v| !v.is_null() && explicit_null(src, path)) {
                        vals.push(nv);
                    }
                }
                for v in vals {
                    if let Ok(n) = crate::document::parse_integer(v, coerce, min, max, t) {
                        dims.entry(l.path.as_bytes().to_vec()).or_default().push(long_val(n));
                    }
                }
            }
            Some(t) => {
                for v in crate::highlight::source_values(src, l.src.as_deref().unwrap_or(&l.path)) {
                    let mut text = scalar_text(v);
                    if t == EsType::Ip {
                        if let Ok(ip) = text.parse::<std::net::IpAddr>() {
                            text = match ip {
                                std::net::IpAddr::V6(v6) => v6.to_ipv4_mapped().map(|v4| v4.to_string()).unwrap_or_else(|| v6.to_string()),
                                other => other.to_string(),
                            };
                        }
                    }
                    add_str(&mut dims, &mut routing, in_routing(&l.path), &l.path, &text);
                }
            }
            None => {}
        }
    }
    // `routing_path` values no mapping covers yet (a dynamic dimension).
    if !rp.is_empty() && rp.iter().any(|p| p.contains('*') || !mapped_at(&m.root, p)) {
        add_unmapped_routing(&m.root, &rp, src, "", &mut routing);
    }
    Ok((dims, routing))
}

fn build_ids_from_identity(m: &Mappings, s: &FlatSettings, src: &Value, dims: IdentityDims, routing: &mut RoutingHashes) -> Result<TsIds> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    if dims.is_empty() {
        return Err(EsError::illegal_argument("Error extracting routing: source didn't contain any routing fields"));
    }
    // _tsid
    let n_fields = dims.len().min(16);
    let hash_len = 16 + 16 + 4 * n_fields;
    let mut out = Vec::with_capacity(hash_len + 2);
    vint(hash_len, &mut out);
    let mut names = Vec::new();
    for k in dims.keys() {
        let chars = String::from_utf8_lossy(k).chars().count();
        let mut b = k.clone();
        b.resize(chars * 3, 0);
        names.extend_from_slice(&b);
    }
    let (h1, h2) = murmur3_128(&names, 0);
    out.extend_from_slice(&h1.to_le_bytes());
    out.extend_from_slice(&h2.to_le_bytes());
    let start = out.len();
    for vals in dims.values() {
        if out.len() - start >= 4 * n_fields {
            break;
        }
        out.extend_from_slice(&murmur3_32(&vals[0].bytes, 0).to_le_bytes());
    }
    let mut all = Vec::new();
    for vals in dims.values() {
        for v in vals {
            all.extend_from_slice(&v.padded);
        }
    }
    let (v1, v2) = murmur3_128(&all, 0);
    out.extend_from_slice(&v1.to_le_bytes());
    out.extend_from_slice(&v2.to_le_bytes());
    let tsid = b64.encode(&out);
    let rh = fold_routing(routing);
    let ts_ms = id_timestamp(m, src).ok_or_else(|| EsError::illegal_argument("data stream timestamp field [@timestamp] is missing"))?;
    if synthetic_id(s) {
        // `TsidExtractingIdFieldMapper.createSyntheticId`: _tsid bytes,
        // (Long.MAX_VALUE - timestamp) and the routing hash, big endian.
        let mut idb = Vec::with_capacity(out.len() + 12);
        idb.extend_from_slice(&out);
        idb.extend_from_slice(&(i64::MAX - ts_ms).to_be_bytes());
        idb.extend_from_slice(&rh.to_be_bytes());
        return Ok(TsIds { tsid, id: b64.encode(&idb) });
    }
    let (t1, _) = murmur3_128(&out, 0);
    let mut idb = Vec::with_capacity(20);
    idb.extend_from_slice(&rh.to_le_bytes());
    idb.extend_from_slice(&t1.to_le_bytes());
    idb.extend_from_slice(&ts_ms.to_be_bytes());
    Ok(TsIds { tsid, id: b64.encode(&idb) })
}

/// Runtime fields (mapping or request level) of a time series index must not shadow
/// dimensions / metrics nor match the `routing_path`.
pub fn check_runtime_shadow(s: &FlatSettings, m: &Mappings, runtime: &Value) -> Result<()> {
    let Some(rt) = runtime.as_object().filter(|_| is_time_series(s)) else { return Ok(()) };
    let mut leaves = Vec::new();
    collect(&m.root, "", false, false, &mut leaves);
    let rp = routing_path(s);
    for name in rt.keys() {
        if let Some(l) = leaves.iter().find(|l| l.ty.is_some() && l.path == *name) {
            if is_dimension(l) {
                return Err(EsError::illegal_argument(format!("Field [{name}] attempted to shadow a time_series_dimension")));
            }
            if l.params.contains_key("time_series_metric") {
                return Err(EsError::illegal_argument(format!("Field [{name}] attempted to shadow a time_series_metric")));
            }
        }
        if rp.iter().any(|p| crate::mapping::wildcard_match(p, name)) {
            return Err(EsError::illegal_argument(format!("runtime fields may not match [routing_path] but [{name}] matched")));
        }
    }
    Ok(())
}

/// `time_series_metric: position` fields accept a single value only.
pub fn check_position_fields(m: &Mappings, src: &Value) -> Result<()> {
    let mut leaves = Vec::new();
    collect_selected::<false>(&m.root, &mut String::new(), false, false, &mut leaves);
    check_position_paths(leaves.iter().map(|l| l.path.as_str()), src)
}
fn check_position_paths<'a>(paths: impl Iterator<Item = &'a str>, src: &Value) -> Result<()> {
    for path in paths {
        let mut cur = Some(src);
        for part in path.split('.') {
            cur = cur.and_then(|c| c.get(part));
        }
        // `[lon, lat]` is one point; several values (points) are not allowed.
        if let Some(Value::Array(a)) = cur {
            if a.len() > 1 && !a.iter().all(Value::is_number) {
                return Err(EsError::document_parsing(format!("failed to parse: field type for [{}] does not accept more than single value", path)));
            }
        }
    }
    Ok(())
}

/// A document of a time series index whose source has a field matching a
/// wildcard `routing_path` that cannot be a dimension (under a `dynamic:
/// false` or `dynamic: runtime` object). Checked before the document's
/// dynamic mapping update is requested, as ES's document parser fails it:
/// otherwise the update (a runtime field matching `routing_path`) reaches
/// the master first.
pub fn check_routing_source(m: &Mappings, s: &FlatSettings, src: &Value) -> Result<()> {
    let rp = routing_path(s);
    if rp.iter().any(|p| p.contains('*')) {
        check_unmapped_routing(m, &rp, src)?;
    }
    Ok(())
}

/// A source field matching a wildcard `routing_path` that has no dimension mapping
/// (`dynamic: false` / `dynamic: runtime` objects).
fn check_unmapped_routing(m: &Mappings, rp: &[String], src: &Value) -> Result<()> {
    use crate::mapping::Dynamic;
    fn walk(m: &Mappings, rp: &[String], prefix: &str, v: &Value, obj: Option<&ObjectMapping>, dynamic: Option<Dynamic>) -> Result<()> {
        let Some(map) = v.as_object() else { return Ok(()) };
        for (k, x) in map {
            let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            let mapper = obj.and_then(|o| o.properties.get(k));
            match mapper {
                Some(Mapper::Object(c)) => walk(m, rp, &path, x, Some(c), c.dynamic.or(dynamic))?,
                Some(Mapper::Field(_)) => {}
                None if rp.iter().any(|p| p.contains('*') && crate::mapping::wildcard_match(p, &path)) => match dynamic {
                    Some(Dynamic::False) => {
                        return Err(EsError::illegal_argument(format!("All fields matching [routing_path] must be mapped but [{path}] was declared as [dynamic: false]")));
                    }
                    Some(Dynamic::Runtime) => {
                        let ty = m
                            .runtime
                            .as_ref()
                            .and_then(|r| r.get(&path))
                            .and_then(|f| f.get("type"))
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| match x {
                                Value::Bool(_) => "boolean".into(),
                                Value::Number(n) if n.is_i64() => "long".into(),
                                Value::Number(_) => "double".into(),
                                _ => "keyword".into(),
                            });
                        return Err(EsError::illegal_argument(format!(
                            "All fields that match routing_path must be configured with [time_series_dimension: true] or flattened fields with a list of dimensions in [time_series_dimensions] and without the [script] parameter. [{path}] was a runtime [{ty}]."
                        )));
                    }
                    _ => {}
                },
                None => {}
            }
        }
        Ok(())
    }
    walk(m, rp, "", src, Some(&m.root), m.root.dynamic)
}

/// Is `id` a time series `_id` (url-safe base64 of 20 bytes)?
pub fn valid_id(id: &str) -> bool {
    use base64::Engine;
    // Standard ids are 20 bytes; synthetic ids carry the _tsid plus 12.
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(id).map(|b| b.len() == 20 || b.len() > 12 + 16).unwrap_or(false)
}

/// Does an aggregation tree contain a `terms` aggregation on `_id`?
pub fn aggs_terms_on_id(aggs: &Value) -> bool {
    match aggs {
        Value::Object(o) => o.iter().any(|(k, v)| {
            (k == "terms" && v.get("field").and_then(Value::as_str) == Some("_id")) || aggs_terms_on_id(v)
        }),
        Value::Array(a) => a.iter().any(aggs_terms_on_id),
        _ => false,
    }
}

pub fn mode_error(op: &str, index: &str) -> EsError {
    EsError::illegal_argument(format!("{op} is not supported because the destination index [{index}] is in time_series mode"))
}

pub fn routing_forbidden() -> EsError {
    EsError::illegal_argument(format!("routing is forbidden on CRUD operations that target indices in [{MODE}=time_series]"))
}

#[cfg(test)]
mod selective_leaf_tests {
    use super::*;
    #[test]
    fn selected_paths_match_full_walk_for_dimensions_and_positions() {
        let mut mapping = serde_json::json!({"properties":{
            "flat":{"type":"flattened","time_series_dimensions":["region","zone"]},
            "host":{"type":"keyword","time_series_dimension":true},
            "title":{"type":"text","fields":{"dim":{"type":"keyword","time_series_dimension":true}}},
            "scripted":{"type":"keyword","time_series_dimension":true},
            "location":{"type":"geo_point","time_series_metric":"position"},
            "nested":{"type":"nested","properties":{"tag":{"type":"keyword","time_series_dimension":true}}},
            "pass":{"properties":{"child":{"type":"keyword"},"inner":{"properties":{"é":{"type":"keyword"}}}}}
        }});
        for i in 0..512 { mapping["properties"][format!("metric{i}")] = serde_json::json!({"type":"long"}); }
        let mut m = Mappings::parse(&mapping, &crate::analysis::AnalysisRegistry::default()).unwrap();
        if let Mapper::Object(o) = m.root.properties.get_mut("pass").unwrap() {
            o.params.insert("time_series_dimension".into(), Value::Bool(true));
        }
        if let Mapper::Field(f) = m.root.properties.get_mut("scripted").unwrap() {
            f.params.insert("script".into(), serde_json::json!({"source":"emit('x')"}));
        }
        let encode = |l: &Leaf| serde_json::json!({"path":l.path,"type":format!("{:?}",l.ty),"params":l.params,"nested":l.nested,"passthrough":l.passthrough_dim,"source":l.src});
        let mut all = Vec::new(); collect(&m.root,"",false,false,&mut all);
        let mut dimensions = Vec::new(); let mut path = String::new();
        collect_selected::<true>(&m.root,&mut path,false,false,&mut dimensions);
        assert!(path.is_empty());
        assert_eq!(dimensions.iter().map(&encode).collect::<Vec<_>>(), all.iter().filter(|l| is_dimension(l)).map(&encode).collect::<Vec<_>>());
        assert!(dimensions.len() < 10);
        let mut positions = Vec::new();
        collect_selected::<false>(&m.root,&mut path,false,false,&mut positions);
        assert!(path.is_empty());
        assert_eq!(positions.iter().map(&encode).collect::<Vec<_>>(), all.iter().filter(|l| l.params.get("time_series_metric").and_then(Value::as_str)==Some("position")).map(&encode).collect::<Vec<_>>());
        assert_eq!(positions.len(),1);
        let prepared = PreparedFields::new(&m);
        let settings: FlatSettings = serde_json::from_value(serde_json::json!({"index.mode":"time_series","index.routing_path":["host"]})).unwrap();
        for source in [
            serde_json::json!({"@timestamp":"2026-01-01T00:00:00Z","host":"one","title":"multi","flat":{"region":"west"},"pass":{"child":"v","inner":{"é":"x"}},"location":[1,2]}),
            serde_json::json!({"@timestamp":"2026-01-01T00:00:00Z","host":"two","location":[[1,2],[3,4]]}),
            serde_json::json!({"host":"missing-time"}),
        ] {
            let encode_id = |r: Result<TsIds>| r.map(|v| (v.id,v.tsid)).map_err(|e| e.to_string());
            assert_eq!(encode_id(build_ids(&m,&settings,&source)), encode_id(prepared.build_ids(&m,&settings,&source)));
            assert_eq!(check_position_fields(&m,&source).map_err(|e| e.to_string()), prepared.check_positions(&source).map_err(|e| e.to_string()));
        }
    }
}

/// Review findings 137–140: a time series document's identity (routing,
/// `_tsid`, `_id`) agrees with what it indexes and where it lands.
#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::index::WriteReq;
    use crate::node::Node;
    use crate::shard::WriteCond;
    use serde_json::json;

    fn settings(routing_path: &[&str], extra: Value) -> FlatSettings {
        let mut v = json!({"index.mode": "time_series", "index.routing_path": routing_path});
        for (k, x) in extra.as_object().into_iter().flatten() {
            v[k] = x.clone();
        }
        v.as_object().unwrap().iter().map(|(k, x)| (k.clone(), x.clone())).collect()
    }

    fn mappings(v: Value) -> Mappings {
        Mappings::parse(&v, &crate::analysis::AnalysisRegistry::default()).unwrap()
    }

    fn ts_index(node: &Node, name: &str, shards: u32, extra: Value, mapping: Value) -> std::sync::Arc<crate::index::IndexService> {
        let mut s = json!({"index.mode": "time_series", "index.number_of_shards": shards, "index.number_of_replicas": 0,
            "index.time_series.start_time": "2026-01-01T00:00:00Z", "index.time_series.end_time": "2027-01-01T00:00:00Z"});
        for (k, x) in extra.as_object().into_iter().flatten() {
            s[k] = x.clone();
        }
        node.create_index(name, &json!({"settings": s, "mappings": mapping})).unwrap()
    }

    fn index(svc: &crate::index::IndexService, src: Value, create: bool) -> Result<crate::shard::WriteResult> {
        let cond = WriteCond { create, ..Default::default() };
        svc.write(vec![WriteReq::Index { id: None, source: src.to_string(), routing: None, cond }], true).pop().unwrap()
    }

    /// 137: with `routing_path: labels.*` and a dynamic template mapping
    /// `labels.*` to keyword dimensions (none mapped yet), every first
    /// write routes to the shard its final `_id` names: the id GETs, an
    /// identical create conflicts. A partially mapped path (one mapped
    /// dimension plus a new dynamic one) too. The source routing hash
    /// before the mapping equals the `_id`'s once it is mapped.
    #[test]
    fn dynamic_routing_dimensions_route_where_their_id_does() {
        let before = mappings(json!({"dynamic_templates": [{"labels": {"path_match": "labels.*", "mapping": {"type": "keyword", "time_series_dimension": true}}}],
            "properties": {"@timestamp": {"type": "date"}}}));
        let after = mappings(json!({"properties": {"@timestamp": {"type": "date"}, "labels": {"properties": {"host": {"type": "keyword", "time_series_dimension": true}}}}}));
        let s = settings(&["labels.*"], json!({}));
        for i in 0..8 {
            let src = json!({"@timestamp": "2026-06-01T00:00:00Z", "labels": {"host": format!("h{i}")}});
            let ids = build_ids(&after, &s, &src).unwrap();
            assert_eq!(routing_hash(&before, &s, &src).unwrap(), routing_hash_of_id(&ids.id).unwrap(), "h{i}");
            assert_eq!(routing_hash(&after, &s, &src).unwrap(), routing_hash_of_id(&ids.id).unwrap(), "h{i}");
        }
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = ts_index(&node, "dyn", 4, json!({"index.routing_path": ["labels.*"]}), json!({
            "dynamic_templates": [{"labels": {"path_match": "labels.*", "mapping": {"type": "keyword", "time_series_dimension": true}}}],
            "properties": {"@timestamp": {"type": "date"}, "labels": {"properties": {"region": {"type": "keyword", "time_series_dimension": true}}}}}));
        let mut shards = std::collections::BTreeSet::new();
        for i in 0..16 {
            // Even: only dynamic leaves; odd: a mapped one plus a new dynamic one.
            let labels = if i % 2 == 0 { json!({format!("k{i}"): format!("v{i}")}) } else { json!({"region": "eu", format!("k{i}"): format!("v{i}")}) };
            let src = json!({"@timestamp": "2026-06-01T00:00:00Z", "labels": labels, "v": i});
            let w = index(&svc, src.clone(), false).unwrap_or_else(|e| panic!("doc {i}: {e:?}"));
            shards.insert(svc.route(&w.id, None).unwrap());
            assert!(svc.get(&w.id).unwrap().is_some(), "doc {i}: the acknowledged id [{}] GETs", w.id);
            let again = index(&svc, src, true).expect_err("an identical create conflicts");
            assert_eq!(again.status, 409, "doc {i}: {again:?}");
        }
        assert!(shards.iter().any(|s| *s != 0), "the fixture reaches a nonzero shard: {shards:?}");
    }

    /// 138: integer dimensions take the value the document indexes:
    /// coerced "1.5" and "2.5" (indexed 1 and 2) are distinct series;
    /// 1, "1", " 1 " and "1.5" (all indexed 1) are one; `null_value`
    /// stands in for null.
    #[test]
    fn coerced_integer_dimensions_keep_series_apart() {
        let m = mappings(json!({"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true},
            "sensor": {"type": "long", "time_series_dimension": true, "coerce": true, "null_value": 7}}}));
        let s = settings(&["host"], json!({}));
        let tsid = |v: Value| build_ids(&m, &s, &json!({"@timestamp": "2026-06-01T00:00:00Z", "host": "a", "sensor": v})).unwrap().tsid;
        assert_ne!(tsid(json!("1.5")), tsid(json!("2.5")));
        let one = tsid(json!(1));
        for v in [json!("1"), json!(" 1 "), json!("1.5"), json!(1.9)] {
            assert_eq!(tsid(v.clone()), one, "{v}");
        }
        assert_eq!(tsid(Value::Null), tsid(json!(7)));
        assert_ne!(tsid(json!(" 2 ")), one);
        // End to end: the two coerced samples are two documents.
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = ts_index(&node, "coerce", 1, json!({"index.routing_path": ["host"]}), json!({"properties": {"@timestamp": {"type": "date"},
            "host": {"type": "keyword", "time_series_dimension": true}, "sensor": {"type": "long", "time_series_dimension": true, "coerce": true}}}));
        let a = index(&svc, json!({"@timestamp": "2026-06-01T00:00:00Z", "host": "a", "sensor": "1.5"}), true).unwrap();
        let b = index(&svc, json!({"@timestamp": "2026-06-01T00:00:00Z", "host": "a", "sensor": "2.5"}), true).unwrap();
        assert_ne!(a.id, b.id);
        assert_eq!((a.result, b.result), ("created", "created"));
        // "1" indexes as 1.5 did: the same series and timestamp conflicts.
        assert_eq!(index(&svc, json!({"@timestamp": "2026-06-01T00:00:00Z", "host": "a", "sensor": "1"}), true).unwrap_err().status, 409);
    }

    /// 139: a `date_nanos` `@timestamp` is encoded in the id in nanos, as
    /// the column holds it: samples of one series within a millisecond are
    /// distinct documents, and every acknowledged synthetic id still GETs
    /// after the segment is sealed (flushed).
    #[test]
    fn date_nanos_ids_encode_the_indexed_nanos() {
        let m = mappings(json!({"properties": {"@timestamp": {"type": "date_nanos"}, "host": {"type": "keyword", "time_series_dimension": true}}}));
        let s = settings(&["host"], json!({"index.mapping.synthetic_id": true}));
        let id = |ts: &str| build_ids(&m, &s, &json!({"@timestamp": ts, "host": "a"})).unwrap().id;
        let (a, b) = (id("2026-06-01T00:00:00.000000001Z"), id("2026-06-01T00:00:00.000000002Z"));
        assert_ne!(a, b);
        use base64::Engine;
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&a).unwrap();
        let rev = i64::from_be_bytes(raw[raw.len() - 12..raw.len() - 4].try_into().unwrap());
        assert_eq!(i64::MAX - rev, 1_780_272_000_000_000_001, "the id carries epoch nanos");
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = ts_index(&node, "nanos", 1, json!({"index.routing_path": ["host"], "index.mapping.synthetic_id": true}),
            json!({"properties": {"@timestamp": {"type": "date_nanos"}, "host": {"type": "keyword", "time_series_dimension": true}, "v": {"type": "long"}}}));
        let w1 = index(&svc, json!({"@timestamp": "2026-06-01T00:00:00.000000001Z", "host": "a", "v": 1}), true).unwrap();
        let w2 = index(&svc, json!({"@timestamp": "2026-06-01T00:00:00.000000002Z", "host": "a", "v": 2}), true).unwrap();
        assert_ne!(w1.id, w2.id);
        assert_eq!((w1.result, w2.result), ("created", "created"));
        svc.refresh().unwrap();
        svc.flush().unwrap();
        for w in [&w1, &w2] {
            assert!(svc.get(&w.id).unwrap().is_some(), "[{}] GETs after the seal", w.id);
        }
    }

    /// 140: `_tsid` and `_ts_routing_hash` are metadata fields: a source
    /// carrying either (alone or together) is refused before any write,
    /// and earlier acknowledged ids still GET after a seal. The hidden
    /// columns get exactly one value even if a parsed document had one.
    #[test]
    fn hidden_identity_fields_are_reserved() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        let svc = ts_index(&node, "hidden", 1, json!({"index.routing_path": ["host"], "index.mapping.synthetic_id": true}),
            json!({"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true}}}));
        let ok = index(&svc, json!({"@timestamp": "2026-06-01T00:00:00Z", "host": "a"}), true).unwrap();
        for (bad, field) in [
            (json!({"@timestamp": "2026-06-01T00:00:01Z", "host": "b", "_tsid": "x"}), "_tsid"),
            (json!({"@timestamp": "2026-06-01T00:00:01Z", "host": "b", "_ts_routing_hash": 0}), "_ts_routing_hash"),
            (json!({"@timestamp": "2026-06-01T00:00:01Z", "host": "b", "_tsid": "x", "_ts_routing_hash": 0}), "_ts"),
        ] {
            let e = index(&svc, bad, false).unwrap_err();
            assert_eq!((e.status, e.kind.as_str()), (400, "document_parsing_exception"), "{e:?}");
            assert!(e.reason.contains(&format!("Field [{field}")) && e.reason.contains("is a metadata field and cannot be added inside a document"), "{}", e.reason);
        }
        let after = index(&svc, json!({"@timestamp": "2026-06-01T00:00:02Z", "host": "c"}), true).unwrap();
        assert_eq!(after.seq_no, ok.seq_no + 1, "the refused writes took no sequence number");
        svc.refresh().unwrap();
        svc.flush().unwrap();
        for id in [&ok.id, &after.id] {
            assert!(svc.get(id).unwrap().is_some(), "[{id}] GETs after the seal");
        }
        // Defensive: a pre-existing value of a hidden column is replaced.
        let lookup = svc.lookup();
        let tsid_field = lookup.get(TSID_FIELD).and_then(|f| f.core_id).unwrap();
        let rh_field = lookup.get(ROUTING_HASH_FIELD).and_then(|f| f.core_id).unwrap();
        let mut doc = rustic_core::types::Document::default();
        doc.fields.push((tsid_field, rustic_core::FieldValue::Keyword("x".into())));
        doc.fields.push((rh_field, rustic_core::FieldValue::I64(0)));
        let m = svc.meta().mappings;
        let ids = build_ids(&m, &svc.meta().settings, &json!({"@timestamp": "2026-06-01T00:00:00Z", "host": "a"})).unwrap();
        push_hidden_fields(&lookup, &ids, &mut doc);
        assert_eq!(doc.fields.iter().filter(|(f, _)| *f == tsid_field).count(), 1);
        assert_eq!(doc.fields.iter().filter(|(f, _)| *f == rh_field).count(), 1);
    }
}
