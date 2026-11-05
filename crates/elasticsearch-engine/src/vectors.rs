//! `dense_vector` beyond plain floats, and the kNN query / section:
//!
//! * element types `float`, `byte` (integers in `[-128, 127]`, hex strings)
//!   and `bit` (packed bytes); `byte` and `bit` rows are stored packed in
//!   core (`rustic_core::vector::packed`: `dims` bytes / `dims / 8` bytes,
//!   scored by `i8` kernels / Hamming), see [`Packing`];
//! * `index_options` (`hnsw`, `int8_hnsw`, `int4_hnsw`, `bbq_hnsw`, `flat`,
//!   `int8_flat`, `int4_flat`, `bbq_flat`) validation, defaults and allowed
//!   updates, mapped onto the core's per-field [`VectorFieldOptions`]
//!   (quantization stored at seal time, graph or flat, `confidence_interval`,
//!   default `rescore_vector.oversample`) through the core schema;
//! * kNN parsing with ES's validation and errors, `similarity` thresholds in
//!   the raw-similarity space, `boost`, pre-`filter`, `rescore_vector`,
//!   ES scores for `byte` `dot_product` and `bit` vectors, and nested
//!   (`nested.*`) vector fields (top `k` parents, best child per parent).

use crate::error::{EsError, Result};
use crate::mapping::{EsType, ResolvedField};
use crate::query::dsl::{parse_query, QueryContext};
use crate::version::IndexVersion;
use rustic_core::query::query::{CustomQuery, ScoredDocs, SearchContext};
use rustic_core::query::{DocSet, Explanation, LeafReader, MinShouldMatch, Query, Scorer, TERMINATED};
use rustic_core::types::DocId;
use rustic_core::vector::quantize::Quantization;
use rustic_core::vector::store::VectorFieldOptions;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    Float,
    Byte,
    Bit,
    /// ES 9 `bfloat16`: floats kept at bfloat16 precision (values are
    /// rounded to the nearest bfloat16 when indexed, as ES stores them).
    BFloat16,
}

impl ElementType {
    pub fn parse(s: &str) -> Option<ElementType> {
        Some(match s {
            "float" => ElementType::Float,
            "byte" => ElementType::Byte,
            "bit" => ElementType::Bit,
            "bfloat16" => ElementType::BFloat16,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            ElementType::Float => "float",
            ElementType::Byte => "byte",
            ElementType::Bit => "bit",
            ElementType::BFloat16 => "bfloat16",
        }
    }
    /// `float` or `bfloat16`: float-valued vectors (quantizable, unit-length
    /// checks, float scoring).
    pub fn is_float(self) -> bool {
        matches!(self, ElementType::Float | ElementType::BFloat16)
    }
}

/// ES `BFloat16Support.floatToBFloat16` + `bFloat16ToFloat`: `f` rounded to
/// the nearest bfloat16 (ties to even), as an `f32`.
#[inline]
pub fn bf16_round(f: f32) -> f32 {
    let bits = f.to_bits() as i32;
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(bias) as u32) & 0xFFFF_0000)
}

/// A stored vector as ES's `binary` doc value format shows it: base64 of
/// its big-endian `float`s / bfloat16s, or of its bytes (`byte`, packed
/// `bit`).
pub fn binary_doc_value(v: &[f32], et: ElementType, dims: usize) -> String {
    use base64::Engine;
    let bytes: Vec<u8> = match Packing::for_type(et, dims) {
        // Packed rows: their bytes.
        Some(p) => p.bytes(v).to_vec(),
        None if et == ElementType::BFloat16 => v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_be_bytes()).collect(),
        None => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
    };
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decode big-endian bfloat16 values.
fn bf16_from_be(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|c| f32::from_bits(((c[0] as u32) << 24) | ((c[1] as u32) << 16))).collect()
}

/// Decode big-endian `float` values.
fn f32_from_be(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Java `Base64.getDecoder().decode` (RFC 4648 basic alphabet, padding
/// optional at the end, no whitespace).
pub fn java_base64(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let cfg = base64::engine::GeneralPurposeConfig::new()
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent)
        .with_decode_allow_trailing_bits(true);
    base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, cfg).decode(s).ok()
}

/// Java `HexFormat.of().parseHex`.
fn java_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

fn param<'a>(f: &'a ResolvedField, k: &str) -> Option<&'a Value> {
    f.ext.as_ref().and_then(|e| e.params.get(k))
}

pub fn element_type(f: &ResolvedField) -> ElementType {
    param(f, "element_type").and_then(Value::as_str).and_then(ElementType::parse).unwrap_or(ElementType::Float)
}

/// How a `byte` / `bit` field's core rows are packed
/// (`rustic_core::vector::packed`), with the field's ES `dims`. Readers
/// that need the values (scripts, `_source`, `fields`) unpack them into
/// the form ES computes on: one value per byte (`byte`), one 0/1 value
/// per bit (`bit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    Bits(usize),
    Bytes(usize),
}

impl Packing {
    pub fn of(f: &ResolvedField) -> Option<Packing> {
        Self::for_type(element_type(f), f.dims.unwrap_or(0))
    }
    pub fn for_type(et: ElementType, dims: usize) -> Option<Packing> {
        match et {
            ElementType::Bit => Some(Packing::Bits(dims)),
            ElementType::Byte => Some(Packing::Bytes(dims)),
            _ => None,
        }
    }
    /// The row's values (`dims` 0: every byte the row holds).
    pub fn unpack(self, row: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.unpack_into(row, &mut out);
        out
    }
    pub fn unpack_into(self, row: &[f32], out: &mut Vec<f32>) {
        use rustic_core::vector::packed;
        out.clear();
        let bytes = packed::as_bytes(row);
        match self {
            Packing::Bits(dims) => {
                let bits = if dims == 0 { bytes.len() * 8 } else { dims.min(bytes.len() * 8) };
                out.extend((0..bits).map(|i| ((bytes[i / 8] >> (7 - i % 8)) & 1) as f32));
            }
            Packing::Bytes(dims) => {
                let n = if dims == 0 { bytes.len() } else { dims.min(bytes.len()) };
                out.extend(bytes[..n].iter().map(|&b| f32::from(b as i8)));
            }
        }
    }
    /// The row's bytes as ES holds them (`byte` values / packed bits).
    pub fn bytes(self, row: &[f32]) -> &[u8] {
        let bytes = rustic_core::vector::packed::as_bytes(row);
        let n = match self {
            Packing::Bits(dims) => dims / 8,
            Packing::Bytes(dims) => dims,
        };
        if n == 0 { bytes } else { &bytes[..n.min(bytes.len())] }
    }
}

/// The core similarity of a vector field: its ES `similarity`, or for
/// packed `byte` / `bit` rows the packed variant (`byte_*`, `bit_hamming`).
pub fn core_similarity(et: ElementType, similarity: &str) -> String {
    match et {
        ElementType::Bit => "bit_hamming".into(),
        ElementType::Byte => format!("byte_{similarity}"),
        _ => similarity.to_string(),
    }
}

/// The core dimension (row lanes) of `dims` ES dims of `et`.
pub fn core_dims(et: ElementType, dims: usize) -> usize {
    match et {
        ElementType::Bit => rustic_core::vector::packed::words(dims / 8),
        ElementType::Byte => rustic_core::vector::packed::words(dims),
        _ => dims,
    }
}

/// Bytes of one stored row of `f` (packed for `byte` / `bit`; `bfloat16`
/// values are kept as `f32`).
pub fn stored_row_bytes(f: &ResolvedField) -> u64 {
    core_dims(element_type(f), f.dims.unwrap_or(0)) as u64 * 4
}

/// Pack byte values (floats in `[-128, 127]`, `check_bytes`-validated)
/// into a core row.
fn pack_byte_values(values: &[f32]) -> Vec<f32> {
    let bytes: Vec<u8> = values.iter().map(|x| *x as i8 as u8).collect();
    rustic_core::vector::packed::pack(&bytes)
}

fn indexed(f: &ResolvedField) -> bool {
    match param(f, "index") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s != "false",
        _ => true,
    }
}

/// `index_options.rescore_vector.oversample` of a vector field mapping.
pub fn mapped_oversample(f: &ResolvedField) -> Option<f64> {
    param(f, "index_options").and_then(|o| o.get("rescore_vector")).and_then(|r| r.get("oversample")).and_then(Value::as_f64)
}

/// The index type (`index_options.type`) of a vector field.
pub fn index_type(f: &ResolvedField) -> Option<String> {
    param(f, "index_options").and_then(|o| o.get("type")).and_then(Value::as_str).map(str::to_string)
}

/// Core quantization a field's index type asks for.
pub fn quantization(index_type: Option<&str>) -> Quantization {
    match index_type {
        Some("int8_hnsw") | Some("int8_flat") => Quantization::Int8,
        Some("int4_hnsw") | Some("int4_flat") => Quantization::Int4,
        Some("bbq_hnsw") | Some("bbq_flat") | Some("bbq_disk") => Quantization::Binary,
        Some(_) => Quantization::None,
        None => Quantization::Auto,
    }
}

/// The core per-field vector policy of a `dense_vector` mapping
/// (`index_options.type` / `m` / `ef_construction` / `confidence_interval`
/// and `rescore_vector.oversample`), as registered on the shard's vector
/// store. `None` for unindexed fields. Only `float` / `bfloat16` vectors
/// are quantized; `byte` / `bit` fields keep the graph / flat choice and
/// store `f32`. A field whose default index type is not chosen yet (no
/// dims) gets ES's eventual default for small vectors.
pub fn field_options(f: &ResolvedField) -> Option<VectorFieldOptions> {
    if f.ty != EsType::DenseVector || !indexed(f) {
        return None;
    }
    let io = param(f, "index_options");
    let t = io.and_then(|o| o.get("type")).and_then(Value::as_str).unwrap_or("int8_hnsw");
    let et = element_type(f);
    let mut q = quantization(Some(t));
    if t == "bbq_disk" {
        // DiskBBQ's code width (`bits`).
        q = match io.and_then(|o| o.get("bits")).and_then(Value::as_i64).unwrap_or(1) {
            1 => Quantization::Binary,
            7 => Quantization::Int8,
            _ => Quantization::Int4,
        };
    }
    // ES's GPU plugin builds `hnsw` / `int8_hnsw` graphs of `float`
    // vectors on the GPU (when the node allows it).
    let gpu = crate::gpu::format_supported(t, et == ElementType::Float, &f.similarity);
    if !et.is_float() {
        q = Quantization::None;
    }
    // ES 9 scores `int8_*` / `int4_*` estimates with OSQ.
    let mut opts = VectorFieldOptions::quantized_hnsw(q).with_osq(true).with_gpu(gpu);
    if t.ends_with("_flat") || t == "flat" {
        opts = opts.flat();
    }
    if t == "bbq_disk" && et.is_float() {
        // DiskBBQ: no graph; sealed and merged segments carry an IVF index
        // (`ES950DiskBBQVectorsFormat`).
        let o = io.and_then(Value::as_object);
        let int = |k: &str| o.and_then(|o| o.get(k)).and_then(node_int);
        let flag = |k: &str| o.and_then(|o| o.get(k)).and_then(Value::as_bool).unwrap_or(false);
        let ivf = rustic_core::vector::ivf::IvfOptions {
            cluster_size: int("cluster_size").unwrap_or(DEFAULT_VECTORS_PER_CLUSTER).clamp(64, 65536) as u32,
            flat_index_threshold: int("flat_index_threshold").unwrap_or(-1).clamp(-1, i64::from(i32::MAX)) as i32,
            default_visit_percentage: o.and_then(|o| o.get("default_visit_percentage")).and_then(Value::as_f64).unwrap_or(0.0) as f32,
            bits: int("bits").filter(|b| matches!(b, 1 | 2 | 4 | 7)).unwrap_or(1) as u8,
            on_disk_rescore: flag("on_disk_rescore"),
            precondition: flag("precondition"),
            // ES sizes the collector for the mapping's oversample, 3 when
            // `rescore_vector` is absent (bits 4 and 7 rescore nothing then).
            budget_oversample: o.and_then(|o| o.get("rescore_vector")).and_then(|r| r.get("oversample")).and_then(Value::as_f64).unwrap_or(3.0) as f32,
            explicit_oversample: o.and_then(|o| o.get("rescore_vector")).and_then(|r| r.get("oversample")).is_some(),
        };
        opts = opts.flat().with_ivf(Some(ivf));
    }
    if let Some(o) = io {
        if let (Some(m), Some(ef)) = (o.get("m").and_then(Value::as_u64), o.get("ef_construction").and_then(Value::as_u64)) {
            opts = opts.with_hnsw(m as usize, ef as usize);
        }
        if let Some(ci) = o.get("confidence_interval").and_then(Value::as_f64) {
            opts = opts.with_confidence_interval(Some(ci as f32));
        }
        // `rescore_vector` absent: no rescoring, as ES 9.5 (bbq's 3.0
        // default is materialized into the mapping by `normalize_mapping`;
        // int8/int4 have none). An explicit `oversample` rescores.
        opts = opts.with_oversample(o.get("rescore_vector").and_then(|r| r.get("oversample")).and_then(Value::as_f64).unwrap_or(0.0) as f32);
    } else {
        opts = opts.with_oversample(0.0);
    }
    Some(opts)
}

// ---------------------------------------------------------------------------
// Mapping.

pub const INDEX_TYPES: &[&str] = &["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw", "flat", "int8_flat", "int4_flat", "bbq_flat", "bbq_disk"];

/// ES `DenseVectorFieldMapper.BBQ_DIMS_DEFAULT_THRESHOLD`: from this many
/// dims a 9.x index defaults to `bbq_hnsw`.
pub const BBQ_DIMS_DEFAULT_THRESHOLD: u64 = 384;
/// ES `DenseVectorFieldMapper.MAX_DIMS_COUNT`: most dims of a float,
/// bfloat16 or byte vector.
pub const MAX_DIMS_COUNT: u64 = 4096;
/// ES `MAX_DIMS_COUNT_BIT`: most dims of a bit vector (4096 bytes).
pub const MAX_DIMS_COUNT_BIT: u64 = 4096 * 8;

/// The largest `dims` a `dense_vector` of `element_type` accepts (ES
/// applies the bit limit to `bit`, the general one to every other type,
/// unknown ones included: those fail on the element type).
pub fn max_dims(element_type: Option<&str>) -> u64 {
    if element_type == Some("bit") { MAX_DIMS_COUNT_BIT } else { MAX_DIMS_COUNT }
}
/// `ES940DiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER`.
pub const DEFAULT_VECTORS_PER_CLUSTER: i64 = 384;

fn mp(reason: impl Into<String>) -> EsError {
    EsError::mapper_parsing(reason)
}

/// What a mapping is parsed against: the index's creation version (every
/// default that changed between releases is keyed on it) and whether
/// parse-time deprecation warnings are reported (not when re-normalizing
/// an already stored mapping for a merge).
#[derive(Clone, Copy, Debug)]
pub struct VecMappingCtx {
    pub created: IndexVersion,
    pub warn: bool,
    /// `index.mode: vectordb_document`: `bfloat16` is the default element
    /// type (and always shown).
    pub vectordb: bool,
}

impl VecMappingCtx {
    pub fn new(created: IndexVersion) -> Self {
        VecMappingCtx { created, warn: true, vectordb: false }
    }
    pub fn quiet(created: IndexVersion) -> Self {
        VecMappingCtx { created, warn: false, vectordb: false }
    }
    /// The context of an index's mapping parse.
    pub fn of(reg: &crate::analysis::AnalysisRegistry) -> Self {
        VecMappingCtx { created: reg.created_version, warn: true, vectordb: reg.index_mode.as_deref() == Some("vectordb_document") }
    }
    /// Index created by a 9.x release: 9.x `index_options` parameters
    /// (`on_disk_rescore`, `flat_index_threshold`, `bbq_disk`) and lazy,
    /// dims-dependent defaults.
    fn v9(&self) -> bool {
        self.created.is_9x()
    }
}

fn allowed_index_params(t: &str, v9: bool) -> &'static [&'static str] {
    if !v9 {
        return match t {
            "hnsw" => &["type", "m", "ef_construction"],
            "int8_hnsw" | "int4_hnsw" => &["type", "m", "ef_construction", "confidence_interval", "rescore_vector"],
            "bbq_hnsw" => &["type", "m", "ef_construction", "rescore_vector"],
            "flat" => &["type"],
            "int8_flat" | "int4_flat" => &["type", "confidence_interval", "rescore_vector"],
            "bbq_flat" => &["type", "rescore_vector"],
            _ => &["type"],
        };
    }
    match t {
        "hnsw" => &["type", "m", "ef_construction", "flat_index_threshold", "on_disk_rescore"],
        "int8_hnsw" | "int4_hnsw" => &["type", "m", "ef_construction", "confidence_interval", "on_disk_rescore", "flat_index_threshold", "rescore_vector"],
        "bbq_hnsw" => &["type", "m", "ef_construction", "on_disk_rescore", "flat_index_threshold", "rescore_vector"],
        "flat" => &["type", "on_disk_rescore"],
        "int8_flat" | "int4_flat" => &["type", "on_disk_rescore", "confidence_interval", "rescore_vector"],
        "bbq_flat" => &["type", "rescore_vector", "on_disk_rescore"],
        "bbq_disk" => &["type", "cluster_size", "rescore_vector", "flat_index_threshold", "default_visit_percentage", "on_disk_rescore", "bits", "precondition", "auto_calibrate"],
        _ => &["type"],
    }
}

/// ES `XContentMapValues.nodeIntegerValue`.
fn node_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok().or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64)),
        _ => None,
    }
}

/// ES `XContentMapValues.nodeBooleanValue`.
fn node_bool(name: &str, k: &str, v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        other => Err(mp(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", crate::query::dsl::scalar_string(other))).caused_by(EsError::illegal_argument(format!("Failed to parse value [{}] for [{k}] of field [{name}]", crate::query::dsl::scalar_string(other))))),
    }
}

/// Java `(float) x` widened back to `double` (how ES keeps `float` options).
fn as_java_float(x: f64) -> Value {
    json!((x as f32) as f64)
}

/// Validate a `dense_vector` mapping and fill in what ES reports as
/// defaults (`index: true`, `similarity`, the default `index_options` of
/// the index's creation version), in ES's serialization order.
pub fn normalize_mapping(name: &str, p: &mut Map<String, Value>, ctx: VecMappingCtx) -> Result<()> {
    if ctx.vectordb && !p.contains_key("element_type") {
        p.insert("element_type".into(), json!("bfloat16"));
    }
    let et = match p.get("element_type") {
        None => ElementType::Float,
        Some(v) => v.as_str().and_then(ElementType::parse).ok_or_else(|| mp(format!("invalid element_type [{}]; available types are [byte, bfloat16, bit, float]", crate::query::dsl::scalar_string(v))))?,
    };
    let indexed = match p.get("index") {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) if s == "true" || s == "false" => s == "true",
        Some(v) => return Err(mp(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", crate::query::dsl::scalar_string(v)))),
    };
    let dims = p.get("dims").and_then(Value::as_u64);
    if let Some(d) = dims {
        // ES `DenseVectorFieldMapper` dims validator: the range first
        // ([8, MAX_DIMS_COUNT_BIT] for bit), then the multiple of 8.
        if et == ElementType::Bit {
            if !(8..=MAX_DIMS_COUNT_BIT).contains(&d) {
                return Err(mp(format!("The number of dimensions should be in the range [8, {MAX_DIMS_COUNT_BIT}] but was [{d}]")));
            }
            if d % 8 != 0 {
                return Err(mp(format!("The number of dimensions for field [{name}] should be a multiple of 8 but was [{d}]")));
            }
        }
    }
    if !indexed {
        if p.contains_key("similarity") {
            return Err(crate::fieldtypes::params::iae("Field [similarity] can only be specified for a field of type [dense_vector] when it is indexed"));
        }
        if p.contains_key("index_options") {
            return Err(crate::fieldtypes::params::iae("Field [index_options] can only be specified for a field of type [dense_vector] when it is indexed"));
        }
        p.insert("index".into(), Value::Bool(false));
        return Ok(());
    }
    p.insert("index".into(), Value::Bool(true));
    let default_sim = if et == ElementType::Bit { "l2_norm" } else { "cosine" };
    let sim = p.get("similarity").and_then(Value::as_str).unwrap_or(default_sim).to_string();
    if !matches!(sim.as_str(), "l2_norm" | "dot_product" | "cosine" | "max_inner_product") {
        return Err(mp(format!("Unknown value [{sim}] for field [similarity] - accepted values are [l2_norm, cosine, dot_product, max_inner_product]")));
    }
    if et == ElementType::Bit && sim != "l2_norm" {
        return Err(mp(format!("The [{sim}] similarity can only be used with [element_type] [float] or [byte]; [element_type] [bit] only supports [l2_norm]")));
    }
    p.insert("similarity".into(), Value::String(sim));
    match p.get("index_options").cloned() {
        Some(Value::Object(io)) => {
            let io = parse_index_options(name, io, et, dims, ctx)?;
            p.insert("index_options".into(), Value::Object(io));
        }
        Some(other) => return Err(mp(format!("Expected map for property [index_options] on field [{name}] but got [{other}]"))),
        None => {
            if let Some(d) = default_index_options(et, dims, ctx.created) {
                p.insert("index_options".into(), d);
            }
        }
    }
    Ok(())
}

/// ES `Builder.defaultIndexOptions` for a stateful node: `bbq_disk` is never
/// a default there (the DiskBBQ provider only allows it as a default on
/// stateless nodes, whatever the licence); 9.x indices choose `bbq_hnsw`
/// (>= 384 dims) or `int8_hnsw` once the dims are known; older ones
/// `int8_hnsw` right away.
fn default_index_options(et: ElementType, dims: Option<u64>, created: IndexVersion) -> Option<Value> {
    let bf16_defaults = created.on_or_after(IndexVersion::DENSE_VECTOR_BFLOAT16_DEFAULT_INDEX_OPTIONS)
        || created.between(IndexVersion::DENSE_VECTOR_BFLOAT16_DEFAULT_INDEX_OPTIONS_BACKPORT, IndexVersion::KEYWORD_FIELDS_KEEP_DUPLICATES_IN_BINARY_DOC_VALUES);
    match et {
        ElementType::Float => {}
        ElementType::BFloat16 if bf16_defaults => {}
        _ => return None,
    }
    if created.on_or_after(IndexVersion::DEFAULT_DENSE_VECTOR_TO_BBQ_HNSW) {
        let d = dims?;
        if d >= BBQ_DIMS_DEFAULT_THRESHOLD {
            return Some(json!({ "type": "bbq_hnsw", "m": 16, "ef_construction": 100, "rescore_vector": { "oversample": 3.0 } }));
        }
    }
    Some(json!({ "type": "int8_hnsw", "m": 16, "ef_construction": 100 }))
}

/// Parse and validate an explicit `index_options` (ES
/// `VectorIndexType.parseIndexOptions` + the element type / dims
/// validators) into ES's serialized form.
fn parse_index_options(name: &str, mut io: Map<String, Value>, et: ElementType, dims: Option<u64>, ctx: VecMappingCtx) -> Result<Map<String, Value>> {
    let v9 = ctx.v9();
    let t = match io.get("type") {
        None | Some(Value::Null) => return Err(mp("[index_options] requires field [type] to be configured")),
        Some(v) => crate::query::dsl::scalar_string(v),
    };
    if !INDEX_TYPES.contains(&t.as_str()) || (t == "bbq_disk" && !v9) {
        return Err(mp(format!("Unknown vector index options type [{t}] for field [{name}]")));
    }
    io.remove("type");
    let hnsw_family = t.ends_with("hnsw");
    let quantized = t != "hnsw" && t != "flat";
    // Parameters are consumed in ES's order; a bad value fails before the
    // unknown-parameter check.
    let mut m = 16i64;
    let mut ef = 100i64;
    if hnsw_family {
        for k in ["m", "ef_construction"] {
            if let Some(v) = io.remove(k) {
                // 9.x: `nodeIntegerValue` (range checked by the codec when
                // the first vector is written); 8.x: a positive integer.
                let n = if v9 { node_int(&v).ok_or_else(|| crate::fieldtypes::params::nfe(&crate::query::dsl::scalar_string(&v)))? } else { v.as_i64().filter(|n| *n > 0).ok_or_else(|| mp(format!("[{k}] must be a positive integer, got [{v}]")))? };
                if k == "m" {
                    m = n
                } else {
                    ef = n
                }
            }
        }
    }
    let mut ci: Option<f64> = None;
    if matches!(t.as_str(), "int8_hnsw" | "int4_hnsw" | "int8_flat" | "int4_flat") {
        if let Some(v) = io.remove("confidence_interval") {
            let x = v.as_f64().ok_or_else(|| mp(format!("[confidence_interval] must be a number, got [{v}]")))?;
            ci = Some(((x as f32) as f64).max(f64::MIN));
            if ctx.warn && ctx.created.on_or_after(IndexVersion::UPGRADE_TO_LUCENE_10_4_0) {
                crate::warnings::add(format!("Parameter [confidence_interval] in [index_options] for dense_vector field [{name}] is deprecated and will be removed in a future version"));
            }
        }
    }
    let mut on_disk_rescore = false;
    let mut flat_index_threshold: i64 = -1;
    let mut cluster_size = DEFAULT_VECTORS_PER_CLUSTER;
    let mut visit_pct = 0f64;
    let mut bits: i64 = 1;
    let mut precondition = false;
    let mut auto_calibrate = false;
    if v9 {
        if t == "bbq_disk" {
            if let Some(v) = io.remove("cluster_size") {
                cluster_size = node_int(&v).ok_or_else(|| mp(format!("cluster_size must be an integer, got: {v}")))?;
                if !(64..=65536).contains(&cluster_size) {
                    return Err(mp(format!("cluster_size must be between 64 and 65536, got: {cluster_size}")));
                }
            }
        }
        if hnsw_family || t == "bbq_disk" {
            if let Some(v) = io.remove("flat_index_threshold") {
                flat_index_threshold = node_int(&v).ok_or_else(|| mp(format!("flat_index_threshold must be an integer, got: {v}")))?;
                if t == "bbq_disk" && flat_index_threshold < -1 {
                    return Err(mp(format!("flat_index_threshold must be -1 (dynamic), 0 (disabled), or > 0, got: {flat_index_threshold} for field [{name}]")));
                }
            }
        }
        if t == "bbq_disk" {
            if let Some(v) = io.remove("default_visit_percentage") {
                let x = v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| mp(format!("default_visit_percentage must be a number, got: {v}")))?;
                visit_pct = (x as f32) as f64;
                if !(0.0..=100.0).contains(&visit_pct) {
                    return Err(mp(format!("default_visit_percentage must be between 0.0 and 100.0, got: {} for field [{name}]", crate::fieldtypes::params::jdouble(visit_pct))));
                }
            }
        }
        if let Some(v) = io.remove("on_disk_rescore") {
            on_disk_rescore = node_bool(name, "on_disk_rescore", &v)?;
        }
        if on_disk_rescore && !(quantized && (hnsw_family || t == "bbq_disk")) {
            return Err(mp("on_disk_rescore is only supported for indexed and quantized vector types"));
        }
        if t == "bbq_disk" {
            if let Some(v) = io.remove("bits") {
                bits = node_int(&v).ok_or_else(|| mp(format!("'bits' must be 1, 2, 4 or 7, got: {v} for field [{name}]")))?;
            }
            if !matches!(bits, 1 | 2 | 4 | 7) {
                return Err(mp(format!("'bits' must be 1, 2, 4 or 7, got: {bits} for field [{name}]")));
            }
            if let Some(v) = io.remove("precondition") {
                precondition = node_bool(name, "precondition", &v)?;
            }
            if let Some(v) = io.remove("auto_calibrate") {
                auto_calibrate = node_bool(name, "auto_calibrate", &v)?;
            }
        }
    }
    let mut rescore: Option<f64> = None;
    if quantized {
        if let Some(rv) = io.remove("rescore_vector") {
            let Some(os) = rv.get("oversample").and_then(Value::as_f64) else {
                return Err(mp("Invalid rescore_vector value. Missing required field oversample"));
            };
            if os != 0.0 && os < 1.0 {
                return Err(mp("oversample must be greater than 1 or exactly 0"));
            }
            if os > 10.0 {
                return Err(mp("oversample must be less than or equal to 10"));
            }
            rescore = Some((os as f32) as f64);
        }
    }
    let allowed = allowed_index_params(&t, v9);
    if !io.is_empty() {
        // ES `XContentMapValues.checkNoRemainingFields`: ` [k : v]` per key.
        let list: String = io.iter().map(|(k, v)| format!(" [{k} : {}]", crate::fieldtypes::params::jstr(v))).collect();
        let _ = allowed;
        return Err(mp(format!("Mapping definition for [{name}] has unsupported parameters: {list}")));
    }
    // The validators: element type, then dims.
    if quantized && !et.is_float() {
        return Err(mp(format!("[element_type] cannot be [{}] when using index type [{t}]", et.name())));
    }
    if let Some(d) = dims {
        if t.starts_with("int4") && d % 2 != 0 {
            return Err(mp(format!("{t} only supports even dimensions; provided={d}")));
        }
        if (t == "bbq_hnsw" || t == "bbq_flat") && d < 64 {
            return Err(mp(format!("{t} does not support dimensions fewer than 64; provided={d}")));
        }
    }
    // Defaults the parsed options carry.
    if rescore.is_none() {
        match t.as_str() {
            // bbq rescoring defaults to 3x oversampling (ES 8.18+).
            "bbq_hnsw" | "bbq_flat" => rescore = Some(3.0),
            "bbq_disk" => {
                if bits == 2 {
                    rescore = Some(1.5)
                } else if bits == 1 {
                    rescore = Some(3.0)
                }
            }
            _ => {}
        }
    }
    // int4 on 8.x indices defaults to the dynamic confidence interval (`0`),
    // which ES 8 reports in the mapping.
    if !v9 && t.starts_with("int4") && ci.is_none() {
        ci = Some(0.0);
    }
    // ES's serialization order per type.
    let mut out = Map::new();
    out.insert("type".into(), json!(t));
    match t.as_str() {
        "bbq_disk" => {
            out.insert("cluster_size".into(), json!(cluster_size));
            out.insert("flat_index_threshold".into(), json!(flat_index_threshold));
            out.insert("default_visit_percentage".into(), json!(visit_pct));
            if on_disk_rescore {
                out.insert("on_disk_rescore".into(), json!(true));
            }
            if let Some(os) = rescore {
                out.insert("rescore_vector".into(), json!({ "oversample": os }));
            }
            out.insert("bits".into(), json!(bits));
            if precondition {
                out.insert("precondition".into(), json!(true));
            }
            if auto_calibrate {
                out.insert("auto_calibrate".into(), json!(true));
            }
        }
        _ => {
            if hnsw_family {
                out.insert("m".into(), json!(m));
                out.insert("ef_construction".into(), json!(ef));
            }
            if let Some(c) = ci {
                out.insert("confidence_interval".into(), json!(c));
            }
            if on_disk_rescore {
                out.insert("on_disk_rescore".into(), json!(true));
            }
            if let Some(os) = rescore {
                out.insert("rescore_vector".into(), json!({ "oversample": os }));
            }
            if hnsw_family && flat_index_threshold >= 0 {
                out.insert("flat_index_threshold".into(), json!(flat_index_threshold));
            }
        }
    }
    Ok(out)
}

/// Whether `index_options` may change from `old` to `new` (both normalized
/// by [`normalize_mapping`]). ES (`DenseVectorFieldMapper.IndexOptions
/// .updatableTo`):
///
/// * `flat` -> anything; `int8_flat` -> `int8_flat` / `int4_flat` /
///   `bbq_flat` / any hnsw; `int4_flat` -> `int4_flat` / `bbq_flat` / any
///   hnsw; `bbq_flat` -> `bbq_flat` / `bbq_hnsw`. Never back to a weaker
///   flat type.
/// * hnsw family: `hnsw` -> any hnsw, `int8_hnsw` -> `int8_hnsw` /
///   `int4_hnsw` / `bbq_hnsw`, `int4_hnsw` -> `int4_hnsw` / `bbq_hnsw`,
///   `bbq_hnsw` -> `bbq_hnsw`; never to a flat type; `m` may not shrink
///   (`ef_construction` may).
/// * `bbq_disk` -> `bbq_disk` with the same `precondition` /
///   `auto_calibrate`; only `flat` may become `bbq_disk`.
/// * Same quantized hnsw type: an explicit `confidence_interval` may not
///   change (int4's defaults to `0`, so it is always explicit); flat types
///   may change it freely.
pub fn index_options_updatable(old: Option<&Value>, new: Option<&Value>) -> bool {
    let rank = |t: &str| -> Option<(u8, u8)> {
        // (family: 0 flat / 1 hnsw / 2 ivf, quantization level)
        Some(match t {
            "flat" => (0, 0),
            "int8_flat" => (0, 1),
            "int4_flat" => (0, 2),
            "bbq_flat" => (0, 3),
            "hnsw" => (1, 0),
            "int8_hnsw" => (1, 1),
            "int4_hnsw" => (1, 2),
            "bbq_hnsw" => (1, 3),
            "bbq_disk" => (2, 3),
            _ => return None,
        })
    };
    let ty = |v: Option<&Value>| v.and_then(|o| o.get("type")).and_then(Value::as_str).map(str::to_string);
    // No options yet (a 9.x default waits for the dims): anything goes.
    if old.is_none_or(Value::is_null) {
        return true;
    }
    let (Some(ot), Some(nt)) = (ty(old), ty(new)) else { return false };
    let (Some((of, oq)), Some((nf, nq))) = (rank(&ot), rank(&nt)) else { return false };
    let get = |v: Option<&Value>, k: &str| v.and_then(|o| o.get(k)).and_then(Value::as_f64);
    let flag = |v: Option<&Value>, k: &str| v.and_then(|o| o.get(k)).and_then(Value::as_bool).unwrap_or(false);
    match (of, nf) {
        (2, 2) => flag(old, "precondition") == flag(new, "precondition") && flag(old, "auto_calibrate") == flag(new, "auto_calibrate"),
        (2, _) => false,
        // Only plain `flat` may become `bbq_disk`.
        (_, 2) => ot == "flat",
        // hnsw -> flat: never.
        (1, 0) => false,
        // hnsw -> hnsw: same or more quantization, `m` may not shrink, an
        // explicit confidence interval is frozen within the same type.
        (1, 1) => {
            if nq < oq || get(new, "m").unwrap_or(16.0) < get(old, "m").unwrap_or(16.0) {
                return false;
            }
            if ot == nt && (oq == 1 || oq == 2) {
                if let Some(oci) = get(old, "confidence_interval") {
                    if get(new, "confidence_interval") != Some(oci) {
                        return false;
                    }
                }
            }
            true
        }
        // flat -> hnsw: bbq_flat only to bbq_hnsw.
        (0, 1) => oq < 3 || nq == 3,
        // flat -> flat: same or more quantization.
        _ => nq >= oq,
    }
}

// ---------------------------------------------------------------------------
// Values.

fn check_bytes(v: &[f32]) -> std::result::Result<(), String> {
    for (i, x) in v.iter().enumerate() {
        if x.fract() != 0.0 || *x < -128.0 || *x > 127.0 {
            let shown = if x.fract() == 0.0 { format!("{}", *x as i64) } else { format!("{x}") };
            return Err(format!(
                "element_type [byte] vectors only support integers between [-128, 127] but found [{shown}] at dim [{i}]; Preview [{}]",
                v.iter().take(5).map(|x| format!("{x}")).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    Ok(())
}

fn bytes_as_f32(b: Vec<u8>) -> Vec<f32> {
    b.into_iter().map(|x| x as i8 as f32).collect()
}

/// A `byte` / `bit` document string (ES `ByteElement.parseKnnVector`): of
/// length `2 * dims` it is hex (else base64), otherwise base64 (else hex).
/// Without dims (learnt from this document), a length not divisible by 4
/// is tried as hex first (`parseDimensionCount`).
fn decode_byte_string(s: &str, dims_bytes: Option<usize>) -> std::result::Result<Vec<u8>, EsError> {
    let hex_first = match dims_bytes {
        Some(d) => s.len() == d * 2,
        None => s.len() % 4 != 0,
    };
    let first = if hex_first { java_hex(s) } else { java_base64(s) };
    if let Some(b) = first {
        return Ok(b);
    }
    let second = if hex_first { java_base64(s) } else { java_hex(s) };
    second.ok_or_else(|| {
        if hex_first {
            EsError::illegal_argument(format!("Illegal base64 character {:x}", s.bytes().find(|c| !(c.is_ascii_alphanumeric() || *c == b'+' || *c == b'/' || *c == b'=')).unwrap_or(b'?')))
        } else {
            EsError::illegal_argument(format!("Failed to parse object: hex string [{s}] is not a valid hex encoding of a byte vector"))
        }
    })
}

/// A `float` / `bfloat16` document string (ES `FloatElement`): base64 of
/// big-endian `float`s or bfloat16s (`dims * 4` or `dims * 2` bytes).
fn decode_float_string(s: &str, dims: Option<usize>) -> std::result::Result<Vec<f32>, EsError> {
    let bytes = java_base64(s).ok_or_else(|| {
        let bad = s.bytes().find(|c| !(c.is_ascii_alphanumeric() || *c == b'+' || *c == b'/' || *c == b'=')).unwrap_or(b'=');
        EsError::illegal_argument(format!("Illegal base64 character {bad:x}"))
    })?;
    match dims {
        Some(d) if bytes.len() == d * 4 => Ok(f32_from_be(&bytes)),
        Some(d) if bytes.len() == d * 2 => Ok(bf16_from_be(&bytes)),
        Some(d) => Err(EsError::parsing(format!(
            "Failed to parse object: Base64 decoded vector byte length [{}] does not match the expected length of [{}] or [{}] for dimension count [{d}]",
            bytes.len(),
            d * 4,
            d * 2
        ))),
        None if bytes.len() % 4 != 0 => Err(EsError::parsing(format!("Failed to parse object: Base64 decoded vector byte length [{}] is not a multiple of [4]", bytes.len()))),
        None => Ok(f32_from_be(&bytes)),
    }
}

/// One encoded vector of a multi-vector (`rank_vectors`) document value:
/// base64 floats / bfloat16s, or hex / base64 bytes (values as numbers).
pub fn decode_vector_string(s: &str, et: &str, dims: Option<usize>) -> std::result::Result<Vec<f64>, EsError> {
    match et {
        "float" | "bfloat16" => Ok(decode_float_string(s, dims)?.into_iter().map(|x| x as f64).collect()),
        _ => {
            let dims_bytes = dims.map(|d| if et == "bit" { d / 8 } else { d });
            Ok(decode_byte_string(s, dims_bytes)?.into_iter().map(|b| b as i8 as f64).collect())
        }
    }
}

/// Parse a document value of a vector field into the core representation
/// (`byte` / `bit`: a packed row, see [`Packing`]; `bfloat16`: rounded to
/// bfloat16). `Err` carries the cause message.
/// Peak producer storage before the parsed row is attached to its Document.
/// This enumerates byte-decoder scratch, raw f32 rows, and the separate bit /
/// bfloat16 conversion output; it is not a serialized-size heap estimate.
/// Byte strings conservatively allow one decoded byte per input byte plus
/// base64 padding, including geometric hex-collector replacement overlap.
pub(crate) fn document_vector_storage(f:&ResolvedField,v:&Value)->Result<usize> {
    let overflow=||EsError::new(429,"circuit_breaking_exception","document vector storage overflow");
    let (rows,decoder)=match v {
        Value::Array(a)=>(a.len(),0),
        Value::String(s)=>{
            let bound=s.len().checked_add(3).ok_or_else(overflow)?;
            (bound,bound.checked_mul(3).and_then(|n|n.checked_add(32)).ok_or_else(overflow)?)
        }
        _=>return Ok(0),
    };
    let raw=rows.checked_mul(std::mem::size_of::<f32>()).ok_or_else(overflow)?;
    // `bfloat16` rounds into a second row; `byte` / `bit` pack into one at
    // most a quarter the size.
    let converted=match element_type(f){ElementType::BFloat16=>raw,ElementType::Bit|ElementType::Byte=>raw/4+4,_=>0};
    decoder.checked_add(raw).and_then(|n|n.checked_add(converted)).ok_or_else(overflow)
}

pub fn doc_vector(f: &ResolvedField, v: &Value) -> std::result::Result<Vec<f32>, EsError> {
    doc_vector_with_dims(f, v).map(|(row, _)| row)
}

/// [`doc_vector`] with the vector's ES dimension count (`bit`: bits).
pub fn doc_vector_with_dims(f: &ResolvedField, v: &Value) -> std::result::Result<(Vec<f32>, usize), EsError> {
    let et = element_type(f);
    let dims = f.dims;
    let raw: Vec<f32> = match v {
        Value::Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for x in a {
                match x.as_f64() {
                    Some(n) => out.push(n as f32),
                    None => return Err(EsError::illegal_argument("dense_vector values must be numbers")),
                }
            }
            out
        }
        Value::String(s) if et.is_float() => decode_float_string(s, dims)?,
        Value::String(s) => {
            let dims_bytes = dims.map(|d| if et == ElementType::Bit { d / 8 } else { d });
            bytes_as_f32(decode_byte_string(s, dims_bytes)?)
        }
        _ => {
            return Err(EsError::parsing(format!("Failed to parse object: expecting token of type [START_ARRAY] but found [{}]", if v.is_string() { "VALUE_STRING" } else { "VALUE_NUMBER" })))
        }
    };
    if !et.is_float() {
        check_bytes(&raw).map_err(EsError::illegal_argument)?;
    } else if let Some((i, _)) = raw.iter().enumerate().find(|(_, x)| !x.is_finite()) {
        return Err(EsError::illegal_argument(format!("[dense_vector] field [{}] has an invalid value at dim [{i}]", f.name)));
    }
    Ok(match et {
        ElementType::Bit => (pack_byte_values(&raw), raw.len() * 8),
        ElementType::Byte => (pack_byte_values(&raw), raw.len()),
        ElementType::BFloat16 => { let n = raw.len(); (bf16_store(&raw, f.similarity == "cosine"), n) }
        ElementType::Float => { let n = raw.len(); (raw, n) }
    })
}

/// A `bfloat16` vector as ES stores it and reads it back: `cosine` vectors
/// not already (leniently) unit length are normalized, stored as bfloat16
/// and multiplied back by their `float` magnitude on read
/// (`DenormalizedCosineFloatVectorValues`); others are just rounded.
fn bf16_store(raw: &[f32], cosine: bool) -> Vec<f32> {
    if cosine {
        let mut sq = 0f32;
        for x in raw {
            sq += x * x;
        }
        if (sq - 1.0).abs() >= 0.02 && sq > 0.0 {
            let len = (sq as f64).sqrt() as f32;
            return raw.iter().map(|x| bf16_round(x / len) * len).collect();
        }
    }
    raw.iter().map(|x| bf16_round(*x)).collect()
}

/// The `dot_product` unit-length tolerance on the squared magnitude
/// (`Element.isUnitVector`): `None` when the check does not apply (only
/// float-valued vectors are checked; bfloat16 is more lenient).
pub fn unit_length_tolerance(f: &ResolvedField) -> Option<f32> {
    if f.similarity != "dot_product" {
        return None;
    }
    match element_type(f) {
        ElementType::Float => Some(1e-3),
        ElementType::BFloat16 => Some(0.02),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// kNN.

fn knn_err(field: &str, cause: EsError) -> EsError {
    EsError::x_content_parse(format!("[1:1] [knn] failed to parse field [{field}]")).caused_by(cause)
}

fn shard_err(ctx: &QueryContext, reason: impl Into<String>) -> EsError {
    EsError::query_shard(ctx.index_name, ctx.index_uuid, reason)
}

/// ES score for a raw similarity value (`VectorSimilarity.score`).
pub fn score_of_similarity(sim: f32, similarity: &str, et: ElementType, dims: usize) -> f32 {
    match (similarity, et) {
        ("l2_norm", ElementType::Bit) => (dims as f32 - sim) / dims as f32,
        ("l2_norm", _) => 1.0 / (1.0 + sim * sim),
        ("dot_product", ElementType::Byte) => 0.5 + sim / (dims as f32 * 32768.0),
        ("max_inner_product", _) => {
            if sim < 0.0 {
                1.0 / (1.0 - sim)
            } else {
                sim + 1.0
            }
        }
        _ => (1.0 + sim) / 2.0,
    }
}

/// Core score → ES score where the two differ (`byte` `dot_product`,
/// `bit` Hamming).
#[derive(Clone, Copy, Debug)]
enum ScoreMap {
    ByteDot { dims: f32 },
    Hamming { bits: f32 },
}

impl ScoreMap {
    fn apply(self, s: f32) -> f32 {
        match self {
            ScoreMap::ByteDot { dims } => 0.5 + (2.0 * s - 1.0) / (dims * 32768.0),
            ScoreMap::Hamming { bits } => {
                let h = if s > 0.0 { (1.0 / s - 1.0).round() } else { bits };
                (bits - h) / bits
            }
        }
    }
}

/// Wraps a core kNN node: maps its scores to ES's and applies the
/// similarity threshold in ES score space.
#[derive(Debug)]
struct MappedKnn {
    inner: Query,
    map: Option<ScoreMap>,
    threshold: Option<f32>,
    boost: f32,
    /// Quantized index types (`int8_*`): the threshold is checked against the
    /// score of the int8-quantized vectors, as ES/Lucene collect approximate
    /// scores (the returned score stays the exact one).
    quant: Option<QuantThreshold>,
}

#[derive(Debug)]
struct QuantThreshold {
    field: u16,
    query: Vec<f32>,
    similarity: rustic_core::vector::Similarity,
    /// `confidence_interval`; `None` is Lucene's default
    /// `max(0.9, 1 - 1 / (dims + 1))`, `Some(0.0)` the dynamic interval.
    confidence_interval: Option<f32>,
    /// Lucene code bits: 7 for `int8_*`, 4 for `int4_*`.
    bits: u8,
}

impl QuantThreshold {
    /// Per-leaf filter: keep the hits whose Lucene-quantized score reaches
    /// `threshold` (ES checks `similarity` against the quantized index's
    /// scores). The quantizer is fitted on the leaf's vectors the way Lucene
    /// fits a segment's (cosine: on the unit vectors).
    fn filter(&self, leaf: &LeafReader<'_>, hits: Vec<(DocId, f32)>, threshold: f32) -> Vec<(DocId, f32)> {
        use rustic_core::vector::es_quant::{EsQuant, EsQuantScorer};
        let n = leaf.num_docs();
        let sample: Vec<Vec<f32>> = (0..n).filter_map(|d| leaf.vector(self.field, d)).collect();
        if sample.is_empty() {
            return hits;
        }
        // ES 9 scores int8 / int4 with OSQ (`ES94ScalarQuantizedVectorsFormat`).
        let spec = EsQuant { bits: self.bits, confidence_interval: self.confidence_interval, osq: true };
        let q = spec.fit(self.similarity, sample.len(), |i| sample[i].as_slice());
        let mut scorer = EsQuantScorer::new(q, self.similarity, &self.query);
        hits.into_iter()
            .filter(|(d, exact)| match leaf.vector(self.field, *d) {
                Some(v) => scorer.score(&v) >= threshold,
                None => *exact >= threshold,
            })
            .collect()
    }
}

struct MappedScorer {
    inner: Box<dyn Scorer>,
    map: Option<ScoreMap>,
    threshold: Option<f32>,
    boost: f32,
    score: f32,
}

impl MappedScorer {
    fn settle(&mut self, mut d: DocId) -> DocId {
        while d != TERMINATED {
            let s = self.map.map_or(self.inner.score(), |m| m.apply(self.inner.score()));
            if self.threshold.is_none_or(|t| s >= t) {
                self.score = s * self.boost;
                return d;
            }
            d = self.inner.advance();
        }
        d
    }
}

impl DocSet for MappedScorer {
    fn advance(&mut self) -> DocId {
        let d = self.inner.advance();
        self.settle(d)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        let d = self.inner.seek(target);
        self.settle(d)
    }
    fn doc(&self) -> DocId {
        self.inner.doc()
    }
    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl Scorer for MappedScorer {
    fn score(&mut self) -> f32 {
        self.score
    }
}

impl CustomQuery for MappedKnn {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut inner = self.inner.scorer(leaf, ctx)?;
        if let (Some(qt), Some(t)) = (&self.quant, self.threshold) {
            let mut hits = Vec::new();
            loop {
                let d = inner.advance();
                if d == TERMINATED {
                    break;
                }
                hits.push((d, inner.score()));
            }
            let hits: Vec<(DocId, f32)> = qt.filter(leaf, hits, t).into_iter().map(|(d, s)| (d, s * self.boost)).collect();
            if hits.is_empty() {
                return None;
            }
            return Some(Box::new(ScoredDocs::new(hits)));
        }
        Some(Box::new(MappedScorer { inner, map: self.map, threshold: self.threshold, boost: self.boost, score: 0.0 }))
    }
    fn children(&self) -> Vec<&Query> {
        vec![&self.inner]
    }
    fn name(&self) -> &'static str {
        "knn"
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        // `KnnScoreDocQuery` (the kNN winners as fixed docs).
        Explanation::leaf(score, "within top k documents".to_string())
    }
}

/// Precomputed per-leaf `(doc, score)` lists (nested kNN parents), keyed by
/// leaf identity of the searched snapshot.
#[derive(Debug)]
struct FixedDocs {
    per_leaf: HashMap<usize, Vec<(DocId, f32)>>,
    /// Every candidate child of the matched parents (`inner_hits`).
    children: HashMap<usize, Vec<(DocId, f32)>>,
    /// `inner_hits` of a nested kNN: an exact kNN over the children of
    /// whichever parent the hit is (see [`ExactInnerKnn`]); replaces
    /// `children` when set.
    exact: Option<Arc<ExactInnerKnn>>,
}

impl CustomQuery for FixedDocs {
    fn scorer(&self, leaf: &LeafReader<'_>, _ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let v = self.per_leaf.get(&leaf.id())?;
        if v.is_empty() {
            return None;
        }
        Some(Box::new(ScoredDocs::new(v.clone())))
    }
    fn inner_hits_scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        if let Some(exact) = &self.exact {
            return exact.scorer(leaf);
        }
        let v = self.children.get(&leaf.id()).filter(|v| !v.is_empty());
        match v {
            Some(v) => Some(Box::new(ScoredDocs::new(v.clone()))),
            None => self.scorer(leaf, ctx),
        }
    }
    fn name(&self) -> &'static str {
        "knn"
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, "within top k documents".to_string())
    }
}

/// ES `KnnVectorQueryBuilder.doRewrite` under an `InnerHitsRewriteContext`:
/// for `inner_hits` a nested kNN becomes an `ExactKnnQueryBuilder` over the
/// children of the parent being rendered, whichever parent that is (a kNN
/// winner, a parent beyond `k`, or one matched only by another clause), the
/// filters kept (`should(filter, ToChildBlockJoin(filter))`: the child-level
/// filter on the child, the parent-level one on its parent) and the
/// `similarity` threshold applied. Scored lazily, only for the children of
/// the parents whose inner hits are rendered.
#[derive(Debug)]
struct ExactInnerKnn {
    path: String,
    field: u16,
    vector: Vec<f32>,
    sim: rustic_core::vector::Similarity,
    map: Option<ScoreMap>,
    threshold: Option<f32>,
    boost: f32,
    /// Parent-level filter matches by leaf identity (`None`: no filter).
    parent_bits: Option<HashMap<usize, Vec<u64>>>,
    /// Child-level filter matches by leaf identity (`None`: no filter).
    child_bits: Option<HashMap<usize, Vec<u64>>>,
}

fn bit_set(bits: Option<&Vec<u64>>, d: DocId) -> bool {
    bits.and_then(|b| b.get(d as usize / 64)).is_some_and(|w| (w >> (d % 64)) & 1 == 1)
}

impl ExactInnerKnn {
    fn scorer(self: &Arc<Self>, leaf: &LeafReader<'_>) -> Option<Box<dyn Scorer>> {
        // SAFETY: the scorer never outlives the search borrowing this leaf
        // (the `CustomQuery::scorer` contract).
        let leaf = unsafe { std::mem::transmute::<LeafReader<'_>, LeafReader<'static>>(*leaf) };
        let info = leaf.nested_info()?;
        let ord = info.path_ord(&self.path)?;
        let depth = info.ord_depth(ord);
        Some(Box::new(ExactInnerScorer { q: self.clone(), leaf, info, ord, depth, pos: 0, doc: TERMINATED, score: 0.0, parent: None, parent_ok: false }))
    }

    /// The exact (boosted) ES score of child `d`, `None` when it has no
    /// vector or falls under the `similarity` threshold.
    fn score(&self, leaf: &LeafReader<'_>, d: DocId) -> Option<f32> {
        let raw = leaf.with_vector(self.field, d, |v| (v.len() == self.vector.len()).then(|| self.sim.score(&self.vector, v)))??;
        let score = match self.map {
            Some(m) => m.apply(raw),
            None => raw,
        };
        if !score.is_finite() || self.threshold.is_some_and(|t| score < t) {
            return None;
        }
        Some(score * self.boost)
    }
}

/// [`ExactInnerKnn`] on one leaf: the passing children of the path in doc
/// order. Every doc shallower than the path (a parent) is also returned, as
/// a non-matching stop: `Query::nested_inner_hits_with` stops at the parent
/// and keeps only the path's docs, so a parent whose children all fail
/// never scans (and scores) the children of later parents.
struct ExactInnerScorer {
    q: Arc<ExactInnerKnn>,
    leaf: LeafReader<'static>,
    info: &'static rustic_core::nested::NestedInfo,
    ord: u16,
    depth: u16,
    /// Next doc to examine.
    pos: DocId,
    doc: DocId,
    score: f32,
    /// The parent of the children being scanned and whether it passes the
    /// parent-level filter.
    parent: Option<DocId>,
    parent_ok: bool,
}

impl ExactInnerScorer {
    fn passes(&mut self, d: DocId) -> bool {
        let id = self.leaf.id();
        if let Some(cb) = &self.q.child_bits {
            if !bit_set(cb.get(&id), d) {
                return false;
            }
        }
        if let Some(pb) = &self.q.parent_bits {
            if self.parent.is_none_or(|p| d >= p) {
                self.parent = self.info.parent(d);
                self.parent_ok = self.parent.is_some_and(|p| bit_set(pb.get(&id), p));
            }
            if !self.parent_ok {
                return false;
            }
        }
        true
    }

    fn settle(&mut self, mut d: DocId) -> DocId {
        let n = self.leaf.num_docs();
        while d < n {
            if self.info.depth(d) < self.depth {
                self.doc = d;
                self.score = 0.0;
                self.pos = d + 1;
                return d;
            }
            if self.info.ord(d) == self.ord && !self.leaf.is_deleted(d) && self.passes(d) {
                if let Some(s) = self.q.score(&self.leaf, d) {
                    self.doc = d;
                    self.score = s;
                    self.pos = d + 1;
                    return d;
                }
            }
            d += 1;
        }
        self.pos = n;
        self.doc = TERMINATED;
        TERMINATED
    }
}

impl DocSet for ExactInnerScorer {
    fn advance(&mut self) -> DocId {
        self.settle(self.pos)
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if target == TERMINATED {
            self.doc = TERMINATED;
            return TERMINATED;
        }
        self.settle(target.max(self.pos))
    }
    fn doc(&self) -> DocId {
        self.doc
    }
    fn size_hint(&self) -> u32 {
        self.leaf.num_docs().saturating_sub(self.pos)
    }
}

impl Scorer for ExactInnerScorer {
    fn score(&mut self) -> f32 {
        self.score
    }
}

/// Parse a query vector for a field of element type `et`: an array of
/// numbers (or numeric strings), a single number, or an encoded string,
/// which is decoded later against the field's dims ([`decode_query_vector`]).
enum QueryVector {
    Values(Vec<f32>),
    Encoded(String),
}

fn query_vector(v: &Value, et: ElementType) -> Result<QueryVector> {
    let raw: Vec<f32> = match v {
        Value::Array(a) => a
            .iter()
            .map(|x| {
                x.as_f64()
                    .or_else(|| x.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
                    .map(|f| f as f32)
                    .ok_or_else(|| knn_err("query_vector", EsError::illegal_argument("[query_vector] must be an array of numbers")))
            })
            .collect::<Result<_>>()?,
        Value::Number(n) => vec![n.as_f64().unwrap_or(0.0) as f32],
        Value::String(s) => return Ok(QueryVector::Encoded(s.clone())),
        _ => return Err(knn_err("query_vector", EsError::illegal_argument("[query_vector] must be an array or a hex string"))),
    };
    if !et.is_float() {
        check_bytes(&raw).map_err(EsError::illegal_argument)?;
    }
    Ok(QueryVector::Values(raw))
}

/// ES `VectorData.decodeQueryVector`: hex when its length matches the dims
/// (bit: the dims in bytes), else base64 when its decoded length fits the
/// element type (`float`: 4 bytes per dim, `bfloat16`: 2 or 4, `byte`: 1,
/// `bit`: 1 per 8 dims). The error is the IllegalArgumentException message.
fn decode_query_vector(s: &str, et: ElementType, dims: usize) -> std::result::Result<Vec<f32>, String> {
    let hex = java_hex(s);
    if let Some(h) = &hex {
        if h.len() == dims || (et == ElementType::Bit && h.len() == dims / 8) {
            return Ok(bytes_as_f32(h.clone()));
        }
    }
    let b64 = java_base64(s);
    if hex.is_none() && b64.is_none() {
        return Err("failed to parse field [query_vector]: [query_vector] must be a valid base64 or hex string".into());
    }
    if let Some(b) = &b64 {
        let ok = match et {
            ElementType::Byte => b.len() == dims,
            ElementType::Bit => b.len() == dims / 8,
            ElementType::Float => b.len() == dims * 4,
            ElementType::BFloat16 => b.len() == dims * 4 || b.len() == dims * 2,
        };
        if ok {
            return Ok(match et {
                ElementType::Byte | ElementType::Bit => bytes_as_f32(b.clone()),
                ElementType::BFloat16 if b.len() == dims * 2 => bf16_from_be(b),
                _ => f32_from_be(b),
            });
        }
    }
    if let Some(h) = hex {
        return Err(format!("The query vector has a different number of dimensions [{}] than the document vectors [{dims}].", h.len()));
    }
    let expected = match et {
        ElementType::Byte | ElementType::Bit => "byte",
        ElementType::Float => "float",
        ElementType::BFloat16 => "float or bfloat16",
    };
    Err(format!(
        "failed to parse field [query_vector]: [query_vector] must contain a valid Base64-encoded {expected} vector, but the decoded bytes length [{}] is not compatible with the expected vector length",
        b64.map_or(0, |b| b.len())
    ))
}

/// ES 9 `LookupQueryVectorBuilder`: the query vector is the `path` vector
/// of document `id` in `index` (read like its doc value: decoded, `byte` /
/// `bit` values as floats).
fn lookup_query_vector(o: &Map<String, Value>, spec: &Value, ctx: &QueryContext) -> Result<Map<String, Value>> {
    let bad = |m: &str| EsError::x_content_parse("[1:1] [knn] failed to parse field [query_vector_builder]").caused_by(EsError::x_content_parse(format!("[lookup] {m}")));
    let sp = spec.as_object().ok_or_else(|| bad("expected an object"))?;
    for k in sp.keys() {
        if !matches!(k.as_str(), "id" | "index" | "path" | "routing") {
            return Err(bad(&format!("unknown field [{k}]")));
        }
    }
    let get = |k: &str| sp.get(k).map(crate::query::dsl::scalar_string);
    let (Some(id), Some(index), Some(path)) = (get("id"), get("index"), get("path")) else {
        let missing: Vec<&str> = ["id", "index", "path"].into_iter().filter(|k| !sp.contains_key(*k)).collect();
        return Err(bad(&format!("Required [{}]", missing.join(", "))));
    };
    let routing = get("routing");
    let lookup = ctx.terms_lookup.ok_or_else(|| EsError::illegal_argument("[lookup] query vector builder is not available here"))?;
    let values = lookup(&index, &id, &path, routing.as_deref())?;
    if values.is_empty() {
        let reason = if ctx.lookup.get(&path).is_some() || index != ctx.index_name {
            format!("Document with id [{id}] not found in index [{index}] for lookup query vector builder.")
        } else {
            format!("Field [{path}] not found in document with id [{id}] for lookup query vector builder.")
        };
        return Err(EsError::new(404, "status_exception", reason));
    }
    if let Some(f) = ctx.lookup.get(&path) {
        if f.ty != EsType::DenseVector {
            let class = match values.first() {
                Some(Value::String(_)) | None => "java.lang.String",
                Some(Value::Bool(_)) => "java.lang.Boolean",
                Some(_) => "java.lang.Long",
            };
            return Err(EsError::illegal_argument(format!("Field [{path}] in document with id [{id}] is of unsupported type [{class}] for lookup query vector builder.")));
        }
    }
    // A base64 / hex source value decodes against the queried field.
    let qv = match values.as_slice() {
        [Value::String(s)] => {
            let field = o.get("field").and_then(Value::as_str).unwrap_or(&path);
            let f = ctx.lookup.get(field).or_else(|| ctx.lookup.get(&path));
            let (et, dims) = f.map(|f| (element_type(&f), f.dims)).unwrap_or((ElementType::Float, None));
            let dims = dims.unwrap_or(s.len());
            let v = decode_query_vector(s, et, dims).map_err(EsError::illegal_argument)?;
            Value::Array(v.into_iter().map(|x| json!(x)).collect())
        }
        vals => Value::Array(vals.to_vec()),
    };
    let mut out = o.clone();
    out.shift_remove("query_vector_builder");
    out.insert("query_vector".into(), qv);
    Ok(out)
}

fn int_param(o: &Map<String, Value>, k: &str) -> Result<Option<i64>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_i64()
            .or_else(|| v.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            .map(Some)
            .ok_or_else(|| knn_err(k, EsError::illegal_argument(format!("[{k}] must be an integer")))),
    }
}

/// `knn` query or top-level section entry. `default_k` is the request size
/// for the top-level section (`None` for the query).
pub fn parse_knn(o: &Map<String, Value>, ctx: &QueryContext, default_k: Option<usize>) -> Result<Query> {
    for k in o.keys() {
        if !matches!(
            k.as_str(),
            "field"
                | "query_vector"
                | "query_vector_builder"
                | "k"
                | "num_candidates"
                | "filter"
                | "similarity"
                | "boost"
                | "_name"
                | "inner_hits"
                | "rescore_vector"
                | "visit_percentage"
                | "_inner_hits_children"
        ) {
            return Err(EsError::x_content_parse(format!("[1:1] [knn] unknown field [{k}]")));
        }
    }
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::x_content_parse("[1:1] [knn] failed to parse: Required [field]"))?;
    if o.contains_key("query_vector") && o.contains_key("query_vector_builder") {
        return Err(knn_err("query_vector_builder", EsError::illegal_argument("cannot provide both [query_vector_builder] and [query_vector]")));
    }
    if let Some(f) = ctx.lookup.get(field).filter(|f| f.declared.is_semantic()) {
        // `SemanticKnnVectorQueryRewriteInterceptor`: a nested kNN over the chunks.
        let k = match (int_param(o, "k")?, default_k) {
            (Some(k), _) => k as usize,
            (None, Some(d)) => d,
            (None, None) => int_param(o, "num_candidates")?.map(|n| (n as usize).min(10)).unwrap_or(10),
        };
        let q = crate::inference::semantic_query::knn(f, o, ctx, k)?;
        return Ok(match o.get("boost").and_then(Value::as_f64) {
            Some(b) if b != 1.0 => Query::Boost { query: Box::new(q), boost: b as f32 },
            _ => q,
        });
    }
    let resolved_builder;
    let o = if let Some(lk) = o.get("query_vector_builder").and_then(|b| b.get("lookup")) {
        resolved_builder = lookup_query_vector(o, lk, ctx)?;
        &resolved_builder
    } else if let Some(b) = o.get("query_vector_builder") {
        resolved_builder = crate::inference::semantic::resolve_knn_builder(o, b)?;
        &resolved_builder
    } else {
        o
    };
    let k = int_param(o, "k")?;
    let nc = int_param(o, "num_candidates")?;
    if let Some(k) = k {
        if k < 1 {
            return Err(knn_err("k", EsError::illegal_argument("[k] must be greater than 0")));
        }
    }
    if let Some(n) = nc {
        if n > 10_000 {
            return Err(knn_err("num_candidates", EsError::illegal_argument("[num_candidates] cannot exceed [10000]")));
        }
        if n < 1 {
            return Err(knn_err("num_candidates", EsError::illegal_argument("[num_candidates] must be greater than 0")));
        }
    }
    let k: usize = match (k, default_k) {
        (Some(k), _) => k as usize,
        (None, Some(d)) => d,
        // knn query: k defaults to the request size (10); never above an
        // explicit num_candidates.
        (None, None) => nc.map(|n| (n as usize).min(10)).unwrap_or(10),
    };
    let num_candidates = match nc {
        Some(n) => n as usize,
        None => ((k as f64 * 1.5).round() as usize).clamp(k.min(10_000), 10_000),
    };
    if num_candidates < k {
        return Err(knn_err("num_candidates", EsError::illegal_argument("[num_candidates] cannot be less than [k]")));
    }
    // ES `KnnSearchBuilder` / `KnnVectorQueryBuilder`: a float in [0, 100].
    let visit_percentage: Option<f32> = match o.get("visit_percentage") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let x = v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
                .ok_or_else(|| knn_err("visit_percentage", EsError::illegal_argument("[visit_percentage] must be a number")))? as f32;
            if !(0.0..=100.0).contains(&x) {
                let range = if default_k.is_some() { "0 and 100" } else { "0.0 and 100.0" };
                return Err(knn_err("visit_percentage", EsError::illegal_argument(format!("[visit_percentage] must be between {range}"))));
            }
            Some(x)
        }
    };
    let oversample: Option<f32> = match o.get("rescore_vector") {
        None | Some(Value::Null) => None,
        Some(rv) => {
            let os = rv.get("oversample").and_then(Value::as_f64);
            match os {
                // ES `RescoreVectorBuilder`: >= 1.0, or 0 to disable (the
                // mapping alone caps it at 10; candidates stop at 10,000).
                Some(x) if x == 0.0 || (x >= 1.0 && x.is_finite()) => Some(x as f32),
                _ => return Err(knn_err("rescore_vector", EsError::illegal_argument("[oversample] must be >= 1.0 or 0"))),
            }
        }
    };
    let qv = match o.get("query_vector") {
        Some(v) => v,
        None => {
            if o.contains_key("query_vector_builder") {
                return Err(shard_err(ctx, "[query_vector_builder] requires an inference endpoint, which is not available"));
            }
            return Err(knn_err("query_vector", EsError::illegal_argument("either [query_vector] or [query_vector_builder] must be provided")));
        }
    };
    let Some(f) = ctx.lookup.get(field) else {
        // An unmapped field matches nothing (as ES), unless the index
        // forbids unmapped fields in queries.
        if !ctx.allow_unmapped_fields {
            return Err(shard_err(ctx, format!("No field mapping can be found for the field with name [{field}]")));
        }
        return Ok(Query::MatchNone);
    };
    if f.ty != EsType::DenseVector {
        return Err(shard_err(ctx, format!("[knn] queries are only supported on [dense_vector] fields, but field [{field}] is of type [{}]", f.ty.name())));
    }
    if !indexed(&f) {
        return Err(shard_err(ctx, format!("to perform knn search on field [{field}], its mapping must have [index] set to [true]")));
    }
    let et = element_type(&f);
    let parsed = query_vector(qv, et)?;
    let Some(id) = f.core_id else { return Ok(Query::MatchNone) };
    let raw = match parsed {
        QueryVector::Values(v) => v,
        QueryVector::Encoded(s) => {
            // ES resolves an encoded vector against the mapped dims; none yet
            // (nothing indexed) matches nothing.
            let Some(d) = f.dims else { return Ok(Query::MatchNone) };
            decode_query_vector(&s, et, d).map_err(|m| shard_err(ctx, format!("failed to create query: {m}")))?
        }
    };
    // ES dims of the query (`bit`: bits); `byte` / `bit` rows are packed
    // as the field's rows are (`Packing`).
    let dims = if et == ElementType::Bit { raw.len() * 8 } else { raw.len() };
    if let Some(d) = f.dims {
        if d != dims {
            let (q, dd) = if et == ElementType::Bit { (raw.len(), d / 8) } else { (dims, d) };
            return Err(shard_err(ctx, format!("failed to create query: The query vector has a different number of dimensions [{q}] than the document vectors [{dd}].")));
        }
    }
    let sim_name = f.similarity.clone();
    if sim_name == "cosine" && raw.iter().all(|x| *x == 0.0) {
        return Err(shard_err(ctx, "failed to create query: The [cosine] similarity does not support vectors with zero magnitude. Preview of invalid vector: [0.0, ...]"));
    }
    let filter_json: Vec<&Value> = match o.get("filter") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.iter().collect(),
        Some(q) => vec![q],
    };
    // A nested vector field: filters that can only match docs of its nested
    // path filter the children directly (inside the graph search); the
    // others are parent filters (ES `NestedHelper.mightMatchNonNestedDocs`).
    let mut filters: Vec<Query> = Vec::new();
    let mut child_filters: Vec<Query> = Vec::new();
    for q in filter_json {
        // A knn query inside a `nested` query: a `nested` filter on a path
        // outside the field's nested object joins to the wrong parents (ES
        // resolves it within the current nested scope) and matches nothing.
        if let (None, Some(path), Some(fp)) = (default_k, &f.nested_path, q.get("nested").and_then(|n| n.get("path")).and_then(Value::as_str)) {
            if fp != path.as_str() && !fp.starts_with(&format!("{path}.")) {
                return Ok(Query::MatchNone);
            }
        }
        let parsed = parse_query(q, ctx)?;
        match &f.nested_path {
            Some(path) if !might_match_non_nested(q, path, ctx.lookup) => child_filters.push(parsed),
            _ => filters.push(parsed),
        }
    }
    if let Some(filter) = ctx.knn_prefilter_query()? {
        filters.push(filter);
    }
    let all = |mut fs: Vec<Query>| match fs.len() {
        0 => None,
        1 => Some(Box::new(fs.pop().unwrap())),
        _ => Some(Box::new(Query::Bool { must: vec![], should: vec![], must_not: vec![], filter: fs, minimum_should_match: MinShouldMatch::Count(0) })),
    };
    let filter = all(filters);
    let child_filter = all(child_filters);
    let vector = if Packing::for_type(et, dims).is_some() { pack_byte_values(&raw) } else { raw.clone() };
    let boost = match o.get("boost") {
        None => 1.0,
        Some(v) => v.as_f64().ok_or_else(|| knn_err("boost", EsError::illegal_argument("[boost] must be a number")))? as f32,
    };
    let es_threshold = o.get("similarity").and_then(Value::as_f64).map(|s| score_of_similarity(s as f32, &sim_name, et, dims));
    let map = match (et, sim_name.as_str()) {
        (ElementType::Bit, _) => Some(ScoreMap::Hamming { bits: dims as f32 }),
        (ElementType::Byte, "dot_product") => Some(ScoreMap::ByteDot { dims: dims as f32 }),
        _ => None,
    };
    if let Some(path) = f.nested_path.clone() {
        let all_children = o.get("_inner_hits_children").and_then(Value::as_bool).unwrap_or(false);
        let similarity = rustic_core::codec::vectors::similarity_by_name(&core_similarity(et, &sim_name));
        return nested_knn(ctx, &path, id, vector, k, num_candidates, filter.map(|b| *b), child_filter, es_threshold, map, boost, default_k.is_some(), all_children, similarity, oversample, visit_percentage);
    }
    // A `similarity` threshold applies to the scores the kNN query returns
    // (ES `VectorSimilarityQuery` wraps the possibly rescored query): f32
    // similarities once rescored, else the quantized estimates of the
    // Lucene segment (`ES94ScalarQuantizedVectorsFormat`: OSQ fitted on the
    // segment's own vectors and centroid, refitted by every merge). The
    // mutable graph's row codes are calibrated once per shared field index
    // (their calibration epoch), not per segment, so their estimates can
    // fall on the other side of a threshold ES's segment estimate clears
    // (41_knn_search_byte_quantized: l2 10.44 vs `similarity` 10.51).
    // Unrescored scalar-quantized fields (hnsw and flat) are therefore
    // checked against a per-leaf fit (Lucene 7-bit for int8, 4-bit for
    // int4).
    let ty = index_type(&f);
    let rescored = quantization(ty.as_deref()).is_quantized() && oversample.map(f64::from).or_else(|| mapped_oversample(&f)).unwrap_or(0.0) > 0.0;
    let bits = match ty.as_deref() {
        _ if rescored => None,
        Some("int8_hnsw" | "int8_flat") => Some(7u8),
        Some("int4_hnsw" | "int4_flat") => Some(4u8),
        _ => None,
    };
    let quant = match (es_threshold, bits) {
        (Some(_), Some(bits)) if et.is_float() => rustic_core::codec::vectors::similarity_by_name(&sim_name).map(|similarity| QuantThreshold {
            field: id,
            query: vector.clone(),
            similarity,
            confidence_interval: param(&f, "index_options").and_then(|o| o.get("confidence_interval")).and_then(Value::as_f64).map(|c| c as f32),
            bits,
        }),
        _ => None,
    };
    Ok(match (map, quant) {
        (None, None) => Query::Knn { field: id, vector, k, num_candidates, filter, similarity_threshold: es_threshold, boost, oversample, visit_percentage },
        (map, quant) => Query::Custom(Arc::new(MappedKnn {
            inner: Query::Knn { field: id, vector, k, num_candidates, filter, similarity_threshold: None, boost: 1.0, oversample, visit_percentage },
            map,
            threshold: es_threshold,
            boost,
            quant,
        })),
    })
}

/// ES `NestedHelper.mightMatchNonNestedDocs` over a filter's JSON: whether
/// it can match documents other than the children of `path` (parents or
/// other nested objects). Only term-level queries on mapped fields and
/// booleans of them are known to stay inside the path.
fn might_match_non_nested(q: &Value, path: &str, lookup: &crate::mapping::FieldLookup) -> bool {
    let Some((kind, body)) = q.as_object().and_then(|o| o.iter().next()) else { return true };
    let field_case = |field: &str| -> bool {
        if field.starts_with('_') {
            return true;
        }
        let Some(f) = lookup.get(field) else { return false };
        match &f.nested_path {
            None => true,
            Some(p) if !p.starts_with(path) => true,
            // Not `include_in_parent` / `include_in_root` here.
            Some(p) if p == path => false,
            Some(_) => true,
        }
    };
    match kind.as_str() {
        "match_all" => true,
        "match_none" => false,
        "constant_score" => body.get("filter").map_or(true, |f| might_match_non_nested(f, path, lookup)),
        "term" | "terms" | "range" | "match" => {
            let field = body.as_object().and_then(|b| b.keys().find(|k| !matches!(k.as_str(), "boost" | "_name"))).cloned().unwrap_or_default();
            field_case(&field)
        }
        "bool" => {
            let list = |k: &str| -> Vec<&Value> {
                match body.get(k) {
                    Some(Value::Array(a)) => a.iter().collect(),
                    Some(v) if v.is_object() => vec![v],
                    _ => vec![],
                }
            };
            let required: Vec<&Value> = list("must").into_iter().chain(list("filter")).collect();
            if !required.is_empty() {
                return required.iter().all(|c| might_match_non_nested(c, path, lookup));
            }
            let should = list("should");
            if should.is_empty() {
                // Only `must_not`: ES requires a match_all.
                return true;
            }
            should.iter().any(|c| might_match_non_nested(c, path, lookup))
        }
        _ => true,
    }
}

/// kNN over a vector field inside a `nested` object (ES diversified kNN):
/// the top `k` *parents*, each scored by its best child, `filter` applying
/// to the parents. Evaluated eagerly over the searched snapshot's leaves,
/// widening the child candidate set until `k` distinct parents are found.
/// Returns child-level docs (only the best child per parent unless
/// `all_children`, used for `inner_hits`); `top_level` joins them to the
/// parents (score = best child).
#[allow(clippy::too_many_arguments)]
fn nested_knn(
    ctx: &QueryContext,
    path: &str,
    id: u16,
    vector: Vec<f32>,
    k: usize,
    num_candidates: usize,
    filter: Option<Query>,
    child_filter: Option<Box<Query>>,
    threshold: Option<f32>,
    map: Option<ScoreMap>,
    boost: f32,
    top_level: bool,
    all_children: bool,
    similarity: Option<rustic_core::vector::Similarity>,
    oversample: Option<f32>,
    visit_percentage: Option<f32>,
) -> Result<Query> {
    let leaves = ctx.leaves;
    let fbits: Option<Vec<Vec<u64>>> = filter.map(|f| -> Result<Vec<Vec<u64>>> {
        let sctx = SearchContext::for_query(&f, leaves, rustic_core::query::Bm25Params::default());
        sctx.execution_result()?;
        let bits = leaves
            .iter()
            .map(|leaf| {
                let mut b = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
                if let Some(mut s) = f.scorer(leaf, &sctx) {
                    loop {
                        let d = s.advance();
                        if d == TERMINATED {
                            break;
                        }
                        if let Some(w) = b.get_mut(d as usize / 64) {
                            *w |= 1 << (d % 64);
                        }
                    }
                }
                b
            })
            .collect();
        sctx.execution_result()?;
        Ok(bits)
    }).transpose()?;
    let total_children: usize = leaves.iter().map(|l| l.num_docs() as usize).sum();
    let mut kc = num_candidates.max(k).saturating_mul(4).max(64);
    // (score, leaf, child, parent)
    let cands: Vec<(f32, usize, DocId, DocId)> = loop {
        let child: Query = match map {
            Some(m) => Query::Custom(Arc::new(MappedKnn {
                inner: Query::Knn { field: id, vector: vector.clone(), k: kc, num_candidates: kc, filter: child_filter.clone(), similarity_threshold: None, boost: 1.0, oversample, visit_percentage },
                map: Some(m),
                threshold,
                boost: 1.0,
                quant: None,
            })),
            None => Query::Knn { field: id, vector: vector.clone(), k: kc, num_candidates: kc, filter: child_filter.clone(), similarity_threshold: threshold, boost: 1.0, oversample, visit_percentage },
        };
        let sctx = SearchContext::for_query_deadline(&child, leaves, rustic_core::query::Bm25Params::default(), ctx.deadline);
        sctx.execution_result()?;
        if sctx.vector_timed_out.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(EsError::new(408, "timeout_exception", "nested kNN collection timed out"));
        }
        let mut out = Vec::new();
        let mut parents = std::collections::HashSet::new();
        for (li, leaf) in leaves.iter().enumerate() {
            let Some(info) = leaf.nested_info() else { continue };
            let Some(mut s) = child.scorer(leaf, &sctx) else { continue };
            loop {
                let d = s.advance();
                if d == TERMINATED {
                    break;
                }
                if leaf.is_deleted(d) {
                    continue;
                }
                let Some(p) = info.parent(d) else { continue };
                if let Some(fb) = &fbits {
                    if fb[li].get(p as usize / 64).is_none_or(|w| (w >> (p % 64)) & 1 == 0) {
                        continue;
                    }
                }
                parents.insert((li, p));
                out.push((s.score(), li, d, p));
            }
        }
        if parents.len() >= k || kc >= total_children {
            break out;
        }
        kc = kc.saturating_mul(4).min(total_children.max(1));
    };
    let mut cands = cands;
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut chosen: Vec<(usize, DocId)> = Vec::new();
    for c in &cands {
        if chosen.len() >= k {
            break;
        }
        if !chosen.contains(&(c.1, c.3)) {
            chosen.push((c.1, c.3));
        }
    }
    // Inner hits (ES `KnnVectorQueryBuilder` rewritten for inner hits into
    // an `ExactKnnQueryBuilder`): every child of the rendered parent scored
    // exactly (the filters applied, the `similarity` threshold kept),
    // collected or not, so inner-hit scores and order are exact even on
    // quantized fields; `inner_hits.size` then keeps the best of them
    // (100_knn_nested_search "inner_hits size > 1"). This covers EVERY
    // parent hit, not only the kNN winners: a parent matched by another
    // clause, or ranked beyond `k`, gets its children too. Parent selection
    // above is unchanged.
    let exact = match (all_children, similarity) {
        (true, Some(sim)) => {
            let child_bits: Option<HashMap<usize, Vec<u64>>> = child_filter.as_deref().map(|f| -> Result<HashMap<usize, Vec<u64>>> {
                let sctx = SearchContext::for_query(f, leaves, rustic_core::query::Bm25Params::default());
                sctx.execution_result()?;
                let mut bits = HashMap::new();
                for leaf in leaves.iter().filter(|l| l.nested_info().is_some()) {
                    let mut b = vec![0u64; (leaf.num_docs() as usize).div_ceil(64)];
                    if let Some(mut sc) = f.scorer(leaf, &sctx) {
                        loop {
                            let d = sc.advance();
                            if d == TERMINATED { break; }
                            if let Some(w) = b.get_mut(d as usize / 64) { *w |= 1 << (d % 64); }
                        }
                    }
                    bits.insert(leaf.id(), b);
                }
                sctx.execution_result()?;
                Ok(bits)
            }).transpose()?;
            let parent_bits = fbits.map(|fb| leaves.iter().zip(fb).map(|(leaf, b)| (leaf.id(), b)).collect::<HashMap<_, _>>());
            Some(Arc::new(ExactInnerKnn { path: path.to_string(), field: id, vector: vector.clone(), sim, map, threshold, boost, parent_bits, child_bits }))
        }
        _ => None,
    };
    let mut per_leaf: HashMap<usize, Vec<(DocId, f32)>> = HashMap::new();
    let mut children: HashMap<usize, Vec<(DocId, f32)>> = HashMap::new();
    let mut seen: Vec<(usize, DocId)> = Vec::new();
    for (s, li, d, p) in cands {
        if !chosen.contains(&(li, p)) {
            continue;
        }
        children.entry(leaves[li].id()).or_default().push((d, s * boost));
        if !all_children {
            if seen.contains(&(li, p)) {
                continue;
            }
            seen.push((li, p));
        }
        per_leaf.entry(leaves[li].id()).or_default().push((d, s * boost));
    }
    let docs = Query::Custom(Arc::new(FixedDocs { per_leaf, children, exact }));
    Ok(if top_level {
        Query::Nested { path: path.to_string(), query: Box::new(docs), score_mode: rustic_core::nested::NestedScoreMode::Max, filter: false }
    } else {
        docs
    })
}

/// The `query` to scan for `inner_hits` definitions: the request query plus,
/// for every top-level `knn` section on a nested vector field with
/// `inner_hits`, an equivalent `nested` query (as ES does).
pub fn inner_hits_query(query: Option<&Value>, knn: &[Value], size: usize, lookup: &crate::mapping::FieldLookup) -> Option<Value> {
    let mut extra = Vec::new();
    for k in knn {
        let Some(o) = k.as_object() else { continue };
        let Some(ih) = o.get("inner_hits") else { continue };
        let Some(field) = o.get("field").and_then(Value::as_str) else { continue };
        let Some(path) = lookup.get(field).and_then(|f| f.nested_path.clone()) else { continue };
        let mut spec = o.clone();
        spec.remove("inner_hits");
        spec.entry("k").or_insert(json!(size.max(1)));
        spec.insert("_inner_hits_children".into(), Value::Bool(true));
        extra.push(json!({ "nested": { "path": path, "query": { "knn": spec }, "score_mode": "max", "inner_hits": ih } }));
    }
    if extra.is_empty() {
        return None;
    }
    if let Some(q) = query {
        extra.push(q.clone());
    }
    Some(json!({ "bool": { "should": extra } }))
}

#[cfg(test)]
mod default_index_options_tests {
    use super::*;

    /// The default never sends vectors to the disk IVF (`bbq_disk`), whatever
    /// the dimension: `bbq_hnsw` from 384 dims, `int8_hnsw` below.
    #[test]
    fn defaults_never_choose_bbq_disk() {
        for et in [ElementType::Float, ElementType::BFloat16] {
            for dims in [1u64, 128, 383, 384, 768, 1536, 4096] {
                let options = default_index_options(et, Some(dims), IndexVersion::CURRENT).expect("float default");
                let ty = options["type"].as_str().unwrap();
                assert_ne!(ty, "bbq_disk", "{dims} dims");
                assert_eq!(ty, if dims >= BBQ_DIMS_DEFAULT_THRESHOLD { "bbq_hnsw" } else { "int8_hnsw" }, "{dims} dims");
            }
        }
    }
}
