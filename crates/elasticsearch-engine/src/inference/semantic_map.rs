//! `semantic_text` mapping parameters (`SemanticTextFieldMapper.Builder`):
//! parse-time canonicalisation / validation of `inference_id`,
//! `search_inference_id`, `chunking_settings` and `index_options`, the
//! checks that need the inference endpoint registry (index options against
//! the endpoint's model settings), and the put-mapping merge rules.

use crate::error::{EsError, Result};
use crate::mapping::{FieldLookup, Mappings};
use serde_json::{json, Map, Value};

fn ia(msg: impl Into<String>) -> EsError {
    EsError::illegal_argument(msg)
}

fn validation(errors: &[String]) -> EsError {
    let mut s = String::from("Validation Failed: ");
    for (i, e) in errors.iter().enumerate() {
        s.push_str(&format!("{}: {e};", i + 1));
    }
    EsError::new(400, "validation_exception", s)
}

/// Java `Arrays.toString(Object[])` of keys.
fn keys_str(keys: &[&String]) -> String {
    format!("[{}]", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", "))
}

/// `ServiceUtils.removeAsType(map, k, Integer.class)` + the positive checks.
fn required_positive_int(m: &mut Map<String, Value>, k: &str, errs: &mut Vec<String>) -> Option<i64> {
    let scope = "chunking_settings";
    match m.shift_remove(k) {
        None | Some(Value::Null) => {
            errs.push(format!("[{scope}] does not contain the required setting [{k}]"));
            None
        }
        Some(v) => match v.as_i64().filter(|_| v.is_i64() || v.is_u64()) {
            Some(n) if n <= 0 => {
                errs.push(format!("[{scope}] Invalid value [{n}]. [{k}] must be a positive integer"));
                None
            }
            Some(n) => Some(n),
            None => {
                let shown = match &v {
                    Value::String(s) => s.clone(),
                    o => o.to_string(),
                };
                errs.push(format!("field [{k}] is not of the expected type. The value [{shown}] cannot be converted to a [Integer]"));
                None
            }
        },
    }
}

/// `ChunkingSettingsBuilder.fromMap(map, false)`: `None` for a missing or
/// empty map, else the canonical settings (`toXContent` order).
pub fn chunking_settings(v: &Value) -> Result<Option<Value>> {
    let m = match v {
        Value::Null => return Ok(None),
        Value::Object(m) if m.is_empty() => return Ok(None),
        Value::Object(m) => m.clone(),
        other => return Err(ia(format!("Expected a map for [chunking_settings], got [{other}]"))),
    };
    let Some(strategy) = m.get("strategy") else {
        return Err(ia("Can't generate Chunker without ChunkingStrategy provided"));
    };
    let strategy = match strategy {
        Value::String(s) => s.clone(),
        o => o.to_string(),
    };
    let mut errs = Vec::new();
    let valid: &[&str] = match strategy.to_ascii_lowercase().as_str() {
        "none" => &["strategy"],
        "word" => &["strategy", "max_chunk_size", "overlap"],
        "sentence" => &["strategy", "max_chunk_size", "sentence_overlap"],
        "recursive" => &["strategy", "max_chunk_size", "separator_group", "separators"],
        _ => return Err(ia(format!("Invalid chunkingStrategy {strategy}"))),
    };
    let invalid: Vec<&String> = m.keys().filter(|k| !valid.contains(&k.as_str())).collect();
    let mut rest = m.clone();
    rest.shift_remove("strategy");
    let s = strategy.to_ascii_lowercase();
    if !invalid.is_empty() {
        errs.push(match s.as_str() {
            "none" => format!("When chunking is disabled (none), settings can not have the following: {}", keys_str(&invalid)),
            "word" => format!("Word based chunking settings can not have the following settings: {}", keys_str(&invalid)),
            "sentence" => format!("Sentence based chunking settings can not have the following settings: {}", keys_str(&invalid)),
            _ => format!("Recursive chunking settings can not have the following settings: {}", keys_str(&invalid)),
        });
    }
    let between = |n: Option<i64>, lo: i64, hi: i64, errs: &mut Vec<String>| -> Option<i64> {
        let n = n?;
        if n < lo {
            errs.push(format!("[chunking_settings] Invalid value [{}]. [max_chunk_size] must be greater than or equal to [{}]", n as f64, lo as f64));
            return None;
        }
        if n > hi {
            errs.push(format!("[chunking_settings] Invalid value [{}]. [max_chunk_size] must be less than or equal to [{}]", n as f64, hi as f64));
            return None;
        }
        Some(n)
    };
    let out = match s.as_str() {
        "none" => json!({ "strategy": "none" }),
        "word" => {
            let max = required_positive_int(&mut rest, "max_chunk_size", &mut errs);
            let max = between(max, 10, 300, &mut errs);
            let mut overlap = None;
            if let Some(mx) = max {
                overlap = required_positive_int(&mut rest, "overlap", &mut errs);
                if let Some(o) = overlap {
                    if o > mx / 2 {
                        errs.push(format!("[chunking_settings] Invalid value [{}]. [overlap] must be less than or equal to [{}]", o as f64, (mx / 2) as f64));
                    }
                }
            }
            json!({ "strategy": "word", "max_chunk_size": max, "overlap": overlap })
        }
        "sentence" => {
            let max = required_positive_int(&mut rest, "max_chunk_size", &mut errs);
            let max = between(max, 20, 300, &mut errs);
            let so = match rest.shift_remove("sentence_overlap") {
                None | Some(Value::Null) => 1,
                Some(v) => match v.as_i64() {
                    Some(n) if (0..=1).contains(&n) => n,
                    Some(n) => {
                        errs.push(format!("sentence_overlap[{n}] must be either 0 or 1"));
                        0
                    }
                    None => {
                        errs.push(format!("field [sentence_overlap] is not of the expected type. The value [{v}] cannot be converted to a [Integer]"));
                        0
                    }
                },
            };
            json!({ "strategy": "sentence", "max_chunk_size": max, "sentence_overlap": so })
        }
        _ => {
            let max = required_positive_int(&mut rest, "max_chunk_size", &mut errs);
            let max = between(max, 10, 300, &mut errs);
            let mut o = Map::new();
            o.insert("strategy".into(), json!("recursive"));
            o.insert("max_chunk_size".into(), json!(max));
            if let Some(sep) = m.get("separators") {
                o.insert("separators".into(), sep.clone());
            } else {
                o.insert("separator_group".into(), m.get("separator_group").cloned().unwrap_or(json!("plaintext")));
            }
            Value::Object(o)
        }
    };
    if !errs.is_empty() {
        return Err(validation(&errs));
    }
    Ok(Some(out))
}

thread_local! {
    /// The `index.version.created` of the index whose mapping is being
    /// parsed / merged (defaults that depend on it, e.g. bfloat16 embeddings).
    static CREATED: std::cell::Cell<Option<crate::version::IndexVersion>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`created_version`] = `v`.
pub fn with_created<T>(v: crate::version::IndexVersion, f: impl FnOnce() -> T) -> T {
    let prev = CREATED.with(|c| c.replace(Some(v)));
    struct Restore(Option<crate::version::IndexVersion>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CREATED.with(|c| c.set(self.0));
        }
    }
    let _r = Restore(prev);
    f()
}

/// The creation version of the index being mapped (a new index when unset).
pub fn created_version() -> crate::version::IndexVersion {
    CREATED.with(|c| c.get()).unwrap_or(crate::version::IndexVersion::CURRENT)
}

/// ES 9 semantics for this request (`compatible-with=8` keeps the 8.x ones).
fn v9() -> bool {
    !crate::version::current_rest_api_version().is_v8()
}

/// `MapperParsingException("Failed to parse mapping: ..")` around `e` (ES
/// wraps the type parser's failures).
fn mapping_failure(e: EsError) -> EsError {
    EsError::mapper_parsing(format!("Failed to parse mapping: {}", e.reason)).caused_by(e)
}

/// `DenseVectorFieldMapper.namesToElementType`.
const ELEMENT_TYPES: &[&str] = &["byte", "bfloat16", "bit", "float"];

/// 9.x `SemanticIndexOptions.parseDenseVectorIndexOptionsFromMap`:
/// `element_type` and/or a `dense_vector` index type.
fn dense_index_options_v9(name: &str, body: &Value) -> Result<Value> {
    let mut io = body.as_object().cloned().unwrap_or_default();
    let element_type = match io.shift_remove("element_type") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let s = crate::query::dsl::scalar_string(&v);
            if !ELEMENT_TYPES.contains(&s.as_str()) {
                return Err(mapping_failure(ia(format!("Invalid element_type [{s}]; available types are [{}]", ELEMENT_TYPES.join(", ")))));
            }
            Some(s)
        }
    };
    let base = match io.get("type") {
        None | Some(Value::Null) => {
            io.shift_remove("type");
            if !io.is_empty() {
                return Err(mapping_failure(ia("[type] is required when specifying more params than [element_type]")));
            }
            None
        }
        Some(t) => {
            let ty = t.as_str().map(str::to_string).unwrap_or_else(|| t.to_string());
            if !crate::vectors::INDEX_TYPES.contains(&ty.as_str()) {
                return Err(mapping_failure(ia(format!("Unsupported index options type {ty}"))));
            }
            io.insert("type".into(), json!(ty));
            let mut p = Map::new();
            p.insert("index_options".into(), Value::Object(io));
            let leaf = name.rsplit('.').next().unwrap_or(name);
            crate::vectors::normalize_mapping(leaf, &mut p, crate::vectors::VecMappingCtx::new(created_version())).map_err(mapping_failure)?;
            p.get("index_options").cloned()
        }
    };
    if element_type.is_none() && base.is_none() {
        return Err(mapping_failure(ia("Must specify at least [element_type] or [type]")));
    }
    let mut out = base.and_then(|b| b.as_object().cloned()).unwrap_or_default();
    if let Some(et) = element_type {
        out.insert("element_type".into(), json!(et));
    }
    Ok(Value::Object(out))
}

/// `SparseVectorFieldMapper.SparseVectorIndexOptions.parseFromMap` (kept as
/// given: explicit default values are still serialized).
fn sparse_index_options(body: &Value) -> Result<Value> {
    let o = body.as_object().cloned().unwrap_or_default();
    for k in o.keys() {
        if !matches!(k.as_str(), "prune" | "pruning_config") {
            return Err(mapping_failure(EsError::x_content_parse(format!("[0:0] [index_options] unknown field [{k}]"))));
        }
    }
    let prune = match o.get("prune") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(b)) => Some(*b),
        Some(Value::String(s)) if s == "true" || s == "false" => Some(s == "true"),
        Some(v) => return Err(mapping_failure(ia(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", crate::query::dsl::scalar_string(v))))),
    };
    let config = match o.get("pruning_config") {
        None | Some(Value::Null) => None,
        Some(c) => Some(crate::fieldtypes::params::token_pruning_config(c).map_err(mapping_failure)?),
    };
    if config.is_some() && prune != Some(true) {
        return Err(mapping_failure(ia("[index_options] field [pruning_config] should only be set if [prune] is set to true")));
    }
    let mut m = Map::new();
    if let Some(p) = prune {
        m.insert("prune".into(), json!(p));
    }
    if let Some(c) = config {
        m.insert("pruning_config".into(), c);
    }
    Ok(Value::Object(m))
}

/// `SemanticTextIndexOptions` parse: `{"dense_vector": {..}}` (ES 9 also
/// `{"sparse_vector": {prune, pruning_config}}`).
fn index_options(name: &str, v: &Value) -> Result<Option<Value>> {
    let m = match v {
        Value::Null => return Ok(None),
        Value::Object(m) => m,
        other => return Err(EsError::mapper_parsing(format!("Expected map for property [index_options] on field [{name}] but got [{other}]"))),
    };
    if m.len() != 1 {
        let keys: Vec<&String> = m.keys().collect();
        return Err(ia(format!("Too many index options provided, found [{}]", keys_str(&keys))));
    }
    let (kind, body) = m.iter().next().expect("one entry");
    if v9() {
        return match kind.as_str() {
            "dense_vector" => Ok(Some(json!({ "dense_vector": dense_index_options_v9(name, body)? }))),
            "sparse_vector" => Ok(Some(json!({ "sparse_vector": sparse_index_options(body)? }))),
            _ => Err(ia(format!("Unknown index options type [{kind}]"))),
        };
    }
    if kind != "dense_vector" {
        return Err(ia(format!("Unknown index options type [{kind}]")));
    }
    // 8.x: `parseDenseVectorIndexOptionsFromMap` wraps its failures in an
    // `ElasticsearchException` (reason: the cause's `toString`).
    let wrap = |e: EsError| EsError::new(500, "exception", format!("org.elasticsearch.index.mapper.MapperParsingException: {}", e.reason)).caused_by(e);
    let mut io = body.as_object().cloned().unwrap_or_default();
    let ty = match io.get("type") {
        None | Some(Value::Null) => return Err(EsError::new(400, "exception", "java.lang.IllegalArgumentException: Required type").caused_by(ia("Required type"))),
        Some(t) => t.as_str().map(str::to_string).unwrap_or_else(|| t.to_string()),
    };
    if !crate::vectors::INDEX_TYPES.contains(&ty.as_str()) {
        let msg = format!("Unsupported index options type {ty}");
        return Err(EsError::new(400, "exception", format!("java.lang.IllegalArgumentException: {msg}")).caused_by(ia(msg)));
    }
    io.insert("type".into(), json!(ty));
    let mut p = Map::new();
    p.insert("index_options".into(), Value::Object(io));
    let leaf = name.rsplit('.').next().unwrap_or(name);
    crate::vectors::normalize_mapping(leaf, &mut p, crate::vectors::VecMappingCtx::new(created_version())).map_err(|e| {
        let mut w = wrap(e);
        w.status = 400;
        w
    })?;
    Ok(Some(json!({ "dense_vector": p.get("index_options").cloned().unwrap_or(Value::Null) })))
}

/// The embeddings element type a `semantic_text` field resolves to
/// (`DenseVectorMapperConfigurator.configure`): an `index_options`
/// override, else the model's, defaulting `float` to `bfloat16` on indices
/// created since `SEMANTIC_TEXT_DEFAULTS_TO_BFLOAT16`.
pub fn resolved_element_type(params: &Map<String, Value>, created: crate::version::IndexVersion) -> Option<String> {
    let ms = resolved_model_settings(params)?;
    if !matches!(ms.get("task_type").and_then(Value::as_str), Some("text_embedding" | "embedding")) {
        return None;
    }
    if let Some(et) = params.get("index_options").and_then(|io| io.get("dense_vector")).and_then(|d| d.get("element_type")).and_then(Value::as_str) {
        return Some(et.to_string());
    }
    let model = ms.get("element_type").and_then(Value::as_str).unwrap_or("float");
    if model == "float" && created.on_or_after(crate::version::IndexVersion::SEMANTIC_TEXT_DEFAULTS_TO_BFLOAT16) {
        return Some("bfloat16".into());
    }
    Some(model.to_string())
}

/// `SemanticTextFieldMapper.defaultIndexOptions`: what `include_defaults`
/// shows for a field without `index_options`.
pub fn default_index_options(params: &Map<String, Value>, created: crate::version::IndexVersion) -> Option<Value> {
    let ms = resolved_model_settings(params)?;
    match ms.get("task_type").and_then(Value::as_str) {
        Some("text_embedding" | "embedding") => {
            let model = ms.get("element_type").and_then(Value::as_str).unwrap_or("float");
            (model == "float" && created.on_or_after(crate::version::IndexVersion::SEMANTIC_TEXT_DEFAULTS_TO_BFLOAT16)).then(|| json!({ "dense_vector": { "element_type": "bfloat16" } }))
        }
        Some("sparse_embedding") => Some(json!({ "sparse_vector": { "prune": true, "pruning_config": { "tokens_freq_ratio_threshold": 5.0, "tokens_weight_threshold": 0.4 } } })),
        _ => None,
    }
}

/// Parse-time canonicalisation of a `semantic_text` field's parameters.
pub fn normalize_mapping(name: &str, ty: crate::mapping::EsType, p: &mut Map<String, Value>) -> Result<()> {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    let tname = ty.name();
    match p.get("inference_id") {
        // `semantic` has no default inference id.
        None | Some(Value::Null) if ty == crate::mapping::EsType::Semantic => {
            return Err(mapping_failure(ia(format!("[inference_id] on mapper [{leaf}] of type [{tname}] must not be empty"))));
        }
        None | Some(Value::Null) => {
            p.insert("inference_id".into(), json!(super::semantic::DEFAULT_INFERENCE_ID));
        }
        Some(Value::String(s)) if s.is_empty() => {
            return Err(ia(format!("[inference_id] on mapper [{leaf}] of type [{tname}] must not be empty")));
        }
        _ => {}
    }
    match p.get("search_inference_id") {
        Some(Value::Null) => {
            p.shift_remove("search_inference_id");
        }
        Some(Value::String(s)) if s.is_empty() => {
            return Err(ia(format!("[search_inference_id] on mapper [{leaf}] of type [{tname}] must not be empty")));
        }
        _ => {}
    }
    if let Some(c) = p.get("chunking_settings").cloned() {
        match chunking_settings(&c)? {
            Some(c) => {
                p.insert("chunking_settings".into(), c);
            }
            None => {
                p.shift_remove("chunking_settings");
            }
        }
    }
    if let Some(io) = p.get("index_options").cloned() {
        match index_options(name, &io)? {
            Some(io) => {
                p.insert("index_options".into(), io);
            }
            None => {
                p.shift_remove("index_options");
            }
        }
    }
    if p.get("model_settings").is_some_and(Value::is_null) {
        p.shift_remove("model_settings");
    }
    if let Some(ms) = p.get("model_settings") {
        let task = ms.get("task_type").and_then(Value::as_str).unwrap_or("");
        if v9() {
            if !matches!(task, "text_embedding" | "embedding" | "sparse_embedding") {
                return Err(ia(format!("Wrong [task_type], expected text_embedding, embedding, or sparse_embedding, got {}", task.to_ascii_uppercase())));
            }
        } else if !matches!(task, "text_embedding" | "sparse_embedding") {
            return Err(ia(format!("Wrong [task_type], expected text_embedding or sparse_embedding, got {}", task.to_ascii_uppercase())));
        }
    }
    Ok(())
}

/// The model settings a field resolves to: its own `model_settings`, else
/// the registered endpoint's (`getResolvedModelSettings`).
pub fn resolved_model_settings(params: &Map<String, Value>) -> Option<Value> {
    if let Some(ms) = params.get("model_settings").filter(|v| !v.is_null()) {
        return Some(ms.clone());
    }
    let id = params.get("inference_id").and_then(Value::as_str).unwrap_or(super::semantic::DEFAULT_INFERENCE_ID);
    let svc = super::current()?;
    let ep = svc.get_endpoint(id)?;
    super::model_settings(&ep)
}

/// `Builder.validateIndexOptions` (needs the endpoint registry).
fn validate_index_options(name: &str, params: &Map<String, Value>) -> Result<()> {
    let Some(all) = params.get("index_options").filter(|v| !v.is_null()) else { return Ok(()) };
    let id = params.get("inference_id").and_then(Value::as_str).unwrap_or(super::semantic::DEFAULT_INFERENCE_ID);
    let Some(ms) = resolved_model_settings(params) else {
        let e = ia(format!("Model settings must be set to validate index options for inference ID [{id}]"));
        return Err(if v9() { mapping_failure(e) } else { e });
    };
    let task = ms.get("task_type").and_then(Value::as_str).unwrap_or("");
    if all.get("sparse_vector").is_some() {
        if task != "sparse_embedding" {
            return Err(mapping_failure(ia(format!("Invalid task type for index options, required [sparse_embedding] but was [{task}]"))));
        }
        return Ok(());
    }
    let Some(io) = all.get("dense_vector") else { return Ok(()) };
    if v9() {
        if !matches!(task, "text_embedding" | "embedding") {
            return Err(mapping_failure(ia(format!("Invalid task type for index options, required [text_embedding] or [embedding] but was [{task}]"))));
        }
    } else if task != "text_embedding" {
        return Err(ia(format!("Invalid task type for index options, required [text_embedding] but was [{task}]")));
    }
    // Only the index type (with the overriding element type) is checked
    // against the model dimensions.
    if io.get("type").is_none() {
        return Ok(());
    }
    let mut io = io.clone();
    let et_override = io.as_object_mut().and_then(|o| o.shift_remove("element_type"));
    let mut p = Map::new();
    if let Some(d) = ms.get("dimensions").filter(|d| !d.is_null()) {
        p.insert("dims".into(), d.clone());
    }
    // bfloat16 vectors are validated as floats.
    let et = et_override.or_else(|| ms.get("element_type").filter(|d| !d.is_null()).cloned()).map(|e| if e == "bfloat16" { json!("float") } else { e });
    if let Some(e) = et {
        p.insert("element_type".into(), e);
    }
    p.insert("index_options".into(), io);
    let leaf = name.rsplit('.').next().unwrap_or(name);
    crate::vectors::normalize_mapping(leaf, &mut p, crate::vectors::VecMappingCtx::new(created_version())).map(|_| ())
}

/// Registry-dependent and cross-object checks of the `semantic_text`
/// fields in `only` (every one when `None`): index options against the
/// endpoint's model settings, and never inside a `subobjects: false` object.
pub fn validate_fields(m: &Mappings, lk: &FieldLookup, only: Option<&[String]>) -> Result<()> {
    if !lk.may_have_semantic_fields { return Ok(()); }
    let mut names: Vec<&String> = lk.fields.iter().filter(|(_, f)| f.declared.is_semantic()).map(|(n, _)| n).collect();
    names.sort();
    for name in names {
        if only.is_some_and(|o| !o.iter().any(|x| x == name)) {
            continue;
        }
        let f = &lk.fields[name];
        let path = if f.is_multi_field { name.rsplit_once('.').map(|x| x.0).unwrap_or(name) } else { name.as_str() };
        if let Some((parent, _)) = path.rsplit_once('.') {
            if m.no_subobjects_at(parent) {
                return Err(ia(format!("semantic_text field [{name}] cannot be in an object field with subobjects disabled")));
            }
        }
        if let Some(params) = f.ext.as_ref().map(|e| &e.params) {
            validate_index_options(name, params)?;
        }
    }
    Ok(())
}

/// The fields a mapping update body declares (dotted paths).
pub fn declared_fields(body: &Value) -> Vec<String> {
    fn walk(props: &Map<String, Value>, prefix: &str, out: &mut Vec<String>) {
        for (k, v) in props {
            let full = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            if let Some(p) = v.get("properties").and_then(Value::as_object) {
                walk(p, &full, out);
            }
            if let Some(f) = v.get("fields").and_then(Value::as_object) {
                for sub in f.keys() {
                    out.push(format!("{full}.{sub}"));
                }
            }
            out.push(full);
        }
    }
    let mut out = Vec::new();
    let top = body.get("mappings").unwrap_or(body);
    let top = top.get("_doc").unwrap_or(top);
    if let Some(p) = top.get("properties").and_then(Value::as_object) {
        walk(p, "", &mut out);
    }
    out
}

fn show(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "null".into(),
        Some(Value::String(s)) => s.clone(),
        Some(o) => java_map(o),
    }
}

/// `Objects.toString` of a parsed map value (`{k=v, ..}`).
fn java_map(v: &Value) -> String {
    match v {
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, x)| format!("{k}={}", java_map(x))).collect::<Vec<_>>().join(", ")),
        Value::Array(a) => format!("[{}]", a.iter().map(java_map).collect::<Vec<_>>().join(", ")),
        Value::String(s) => s.clone(),
        o => o.to_string(),
    }
}

/// `MinimalServiceSettings.toString`.
fn settings_string(ms: &Value) -> String {
    let mut out = format!("service={}, task_type={}", show(ms.get("service")), show(ms.get("task_type")));
    for k in ["dimensions", "similarity", "element_type"] {
        if let Some(v) = ms.get(k).filter(|v| !v.is_null()) {
            out.push_str(&format!(", {k}={}", show(Some(v))));
        }
    }
    out.push_str(", metadata=EndpointMetadata[heuristics=null, internal=null, display=null]");
    out
}

/// `canMergeModelSettings` / `MinimalServiceSettings.canMergeWith`.
fn settings_compatible(a: &Value, b: &Value) -> bool {
    ["task_type", "dimensions", "similarity", "element_type"].iter().all(|k| a.get(*k) == b.get(*k))
}

/// Put-mapping merge of a `semantic_text` field's parameters (`new` is the
/// update's canonical parameters). Returns the merged parameters.
pub fn merge(name: &str, old: &Map<String, Value>, new: &Map<String, Value>) -> Result<Map<String, Value>> {
    let mut conflicts = Vec::new();
    let mut out = old.clone();
    let leaf = name.rsplit('.').next().unwrap_or(name);
    // The model settings the incoming mapping carries (or adopts).
    let mut incoming_ms = new.get("model_settings").filter(|v| !v.is_null()).cloned();
    if old.get("inference_id") != new.get("inference_id") {
        if !v9() {
            conflicts.push(format!("Cannot update parameter [inference_id] from [{}] to [{}]", show(old.get("inference_id")), show(new.get("inference_id"))));
        } else {
            // ES 9.3+: `inference_id` is updatable (`SemanticFieldMapper.Builder.mergeFromBuilder`).
            let new_id = new.get("inference_id").and_then(Value::as_str).unwrap_or(super::semantic::DEFAULT_INFERENCE_ID).to_string();
            if let Some(cur) = old.get("model_settings").filter(|v| !v.is_null()) {
                let updated = super::current().and_then(|svc| svc.get_endpoint(&new_id)).and_then(|ep| super::model_settings(&ep));
                let Some(updated) = updated else {
                    return Err(EsError::resource_not_found(format!("{new_id} does not exist in this cluster.")));
                };
                if !settings_compatible(cur, &updated) {
                    return Err(ia(format!(
                        "Cannot update [semantic_text] field [{leaf}] because inference endpoint [{}] with model settings [{}] is not compatible with new inference endpoint [{new_id}] with model settings [{}].",
                        show(old.get("inference_id")),
                        settings_string(cur),
                        settings_string(&updated)
                    )));
                }
                if incoming_ms.is_none() {
                    incoming_ms = Some(updated);
                }
                out.insert("model_settings".into(), incoming_ms.clone().unwrap_or(Value::Null));
            } else {
                // No model settings yet: switch freely (the field has no embeddings).
                out.shift_remove("model_settings");
                if let Some(ms) = &incoming_ms {
                    out.insert("model_settings".into(), ms.clone());
                }
            }
            out.insert("inference_id".into(), json!(new_id));
        }
    } else if v9() {
        // The embeddings sub-field cannot change its element type
        // (`DenseVectorFieldMapper` merge, reported without the sub-field path).
        let created = created_version();
        let mut after = new.clone();
        if after.get("model_settings").is_none() {
            if let Some(ms) = old.get("model_settings") {
                after.insert("model_settings".into(), ms.clone());
            }
        }
        if let (Some(a), Some(b)) = (resolved_element_type(old, created), resolved_element_type(&after, created)) {
            if a != b {
                conflicts.push(format!("Cannot update parameter [element_type] from [{a}] to [{b}]"));
            }
        }
    }
    // Updatable parameters: the update's value (or its absence) wins.
    for k in ["search_inference_id", "chunking_settings", "meta"] {
        match new.get(k) {
            Some(v) => {
                out.insert(k.into(), v.clone());
            }
            None => {
                out.shift_remove(k);
            }
        }
    }
    // Model settings: set once, then only compatible re-statements.
    let switched = old.get("inference_id") != out.get("inference_id") || old.get("inference_id") != new.get("inference_id");
    match (if switched { None } else { old.get("model_settings") }, if switched { None } else { new.get("model_settings") }) {
        (Some(a), Some(b)) if !settings_compatible(a, b) => {
            conflicts.push(format!("Cannot update parameter [model_settings] from [{}] to [{}]", java_map(a), java_map(b)));
        }
        (None, Some(b)) => {
            out.insert("model_settings".into(), b.clone());
        }
        _ => {}
    }
    match (old.get("index_options"), new.get("index_options")) {
        (a, b) if a == b => {}
        (Some(a), Some(b)) if a.get("sparse_vector").is_some() || b.get("sparse_vector").is_some() => {
            // Sparse vector pruning options are updatable.
            out.insert("index_options".into(), b.clone());
        }
        (Some(a), Some(b)) if v9() && a.get("dense_vector").and_then(|d| d.get("type")).is_none() && b.get("dense_vector").and_then(|d| d.get("type")).is_some() => {
            out.insert("index_options".into(), b.clone());
        }
        (Some(a), Some(b)) => {
            let strip = |v: Option<&Value>| v.map(|d| {
                let mut d = d.clone();
                if let Some(o) = d.as_object_mut() {
                    o.shift_remove("element_type");
                }
                d
            });
            if crate::vectors::index_options_updatable(strip(a.get("dense_vector")).as_ref(), strip(b.get("dense_vector")).as_ref()) {
                out.insert("index_options".into(), b.clone());
            } else {
                conflicts.push(format!("Cannot update parameter [index_options] from [{}] to [{}]", java_map(a), java_map(b)));
            }
        }
        (None, Some(b)) => {
            out.insert("index_options".into(), b.clone());
        }
        (Some(_), None) => {
            out.shift_remove("index_options");
        }
        _ => {}
    }
    if !conflicts.is_empty() {
        return Err(ia(format!("Mapper for [{name}] conflicts with existing mapper:\n\t{}", conflicts.join("\n\t"))));
    }
    Ok(out)
}
