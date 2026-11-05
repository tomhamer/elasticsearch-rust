//! Mapping-parameter canonicalisation and ES-exact errors.
//!
//! Elasticsearch parses every mapping parameter into a typed value and
//! prints back only the ones that differ from their default, in canonical
//! form (`"false"` → `false`, `copy_to: "b"` → `["b"]`, `script: "x"` →
//! `{source, lang}`, ...). [`normalize`] does the per-parameter step while a
//! mapping is parsed and [`finish`] the whole-field step (typed
//! `null_value`, defaults stripped, always-printed defaults added).
//!
//! Errors follow ES's three shapes (see `wrap_parse_err` in `mapping.rs`):
//! * [`mpe`] – a `MapperParsingException` (the cause repeats the message),
//! * [`iae`] – an `IllegalArgumentException` under `Failed to parse mapping`,
//! * [`nested_mpe`] – `Error parsing [x] on field [f]: ..` wrapping an IAE.

use crate::error::{EsError, Result};
use crate::mapping::EsType;
use serde_json::{json, Map, Number, Value};

pub fn mpe(msg: impl Into<String>) -> EsError {
    EsError::mapper_parsing(msg)
}

/// `Failed to parse mapping: m` caused by an `illegal_argument_exception: m`.
pub fn iae(msg: impl Into<String>) -> EsError {
    let m: String = msg.into();
    EsError::mapper_parsing(m.clone()).caused_by(EsError::illegal_argument(m))
}

/// Like [`iae`] with a further cause below the IAE.
pub fn iae_caused(msg: impl Into<String>, inner: EsError) -> EsError {
    let m: String = msg.into();
    EsError::mapper_parsing(m.clone()).caused_by(EsError::illegal_argument(m).caused_by(inner))
}

/// `Error parsing [param] on field [f]: inner` (a mapper exception) wrapping
/// an `illegal_argument_exception: inner`.
pub fn nested_mpe(param: &str, field: &str, inner: &str) -> EsError {
    EsError::mapper_parsing(format!("Error parsing [{param}] on field [{field}]: {inner}")).caused_by(EsError::illegal_argument(inner.to_string()))
}

const LANGUAGES: &str = "aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik in io is it iu iw ja ji jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mo mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg sh si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu";

/// A mapping `locale` must name a known language (`en`, `fr_CA`, `ROOT`).
pub fn check_locale(l: &str) -> Result<()> {
    if l.is_empty() || l.eq_ignore_ascii_case("root") {
        return Ok(());
    }
    let lang = l.split(['_', '-']).next().unwrap_or(l).to_ascii_lowercase();
    if lang.len() == 3 || LANGUAGES.split(' ').any(|c| c == lang) {
        return Ok(());
    }
    let inner = EsError::new(400, "missing_resource_exception", format!("Couldn't find 3-letter language code for {lang}"));
    Err(iae_caused(format!("Unknown language: {lang}"), inner))
}

/// The Java `toString` of a JSON value (maps `{a=1}`, lists `[a, b]`).
pub fn jstr(v: &Value) -> String {
    crate::document::preview(v)
}

/// The Java class and value ES prints in `meta` errors (`Integer[1]`).
pub fn jclass_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("String[{s}]"),
        Value::Bool(b) => format!("Boolean[{b}]"),
        Value::Number(n) if n.is_f64() => format!("Double[{n}]"),
        Value::Number(n) => match n.as_i64() {
            Some(i) if i32::try_from(i).is_ok() => format!("Integer[{n}]"),
            _ => format!("Long[{n}]"),
        },
        Value::Array(_) => format!("ArrayList[{}]", jstr(v)),
        Value::Object(_) => format!("LinkedHashMap[{}]", jstr(v)),
        Value::Null => "null".into(),
    }
}

pub fn to_bool(v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        other => Err(iae(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", jstr(other)))),
    }
}

/// A boolean option of an object mapping (`enabled`, `include_in_parent`,
/// ..): `Could not convert [<what>] to boolean`.
pub fn object_bool(what: &str, v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        other => {
            let inner = EsError::illegal_argument(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", jstr(other)));
            Err(iae_caused(format!("Could not convert [{what}] to boolean"), inner))
        }
    }
}

const BOOL_PARAMS: &[&str] = &[
    "index",
    "doc_values",
    "store",
    "norms",
    "eager_global_ordinals",
    "coerce",
    "ignore_malformed",
    "ignore_z_value",
    "enable_position_increments",
    "index_phrases",
    "disable_templating",
    "fielddata",
    "split_queries_on_whitespace",
    "time_series_dimension",
    "preserve_separators",
    "preserve_position_increments",
    "positive_score_impact",
];

pub(crate) fn nfe(s: &str) -> EsError {
    let m = format!("For input string: \"{s}\"");
    EsError::mapper_parsing(m.clone()).caused_by(EsError::new(400, "number_format_exception", m))
}

/// Java `Double.toString` for the values that show up in messages.
pub fn jdouble(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e7 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

/// One parameter, canonicalised. `Ok(None)` drops it.
pub fn normalize(ty: EsType, k: &str, v: &Value) -> Result<Option<Value>> {
    if ty == EsType::DenseVector || ty == EsType::RankVectors || ty.is_semantic() {
        return Ok(Some(v.clone()));
    }
    if BOOL_PARAMS.contains(&k) && !v.is_null() {
        return Ok(Some(Value::Bool(to_bool(v)?)));
    }
    Ok(Some(match k {
        "copy_to" => match v {
            Value::String(s) => json!([s]),
            Value::Array(a) => Value::Array(a.iter().map(|x| Value::String(match x {
                Value::String(s) => s.clone(),
                o => jstr(o),
            })).collect()),
            Value::Null => json!([]),
            other => json!([jstr(other)]),
        },
        "script" => match v {
            Value::String(s) => json!({"source": s, "lang": "painless"}),
            Value::Object(o) => {
                let mut m = Map::new();
                for key in ["source", "id"] {
                    if let Some(x) = o.get(key) {
                        m.insert(key.into(), x.clone());
                    }
                }
                m.insert("lang".into(), o.get("lang").cloned().unwrap_or_else(|| json!("painless")));
                for (key, x) in o {
                    if !m.contains_key(key) {
                        m.insert(key.clone(), x.clone());
                    }
                }
                Value::Object(m)
            }
            other => other.clone(),
        },
        "orientation" => match v.as_str().map(str::to_ascii_lowercase).as_deref() {
            Some("right" | "ccw" | "counterclockwise") => json!("RIGHT"),
            Some("left" | "cw" | "clockwise") => json!("LEFT"),
            _ => return Err(iae(format!("Unknown orientation [{}]", jstr(v)))),
        },
        "scaling_factor" => match v {
            Value::String(t) => match t.trim().parse::<f64>() {
                Ok(x) => Number::from_f64(x).map(Value::Number).unwrap_or_else(|| v.clone()),
                Err(_) => return Err(nfe(t)),
            },
            Value::Number(n) if !n.is_f64() => Number::from_f64(n.as_f64().unwrap_or(0.0)).map(Value::Number).unwrap_or_else(|| v.clone()),
            other => other.clone(),
        },
        "ignore_above" | "position_increment_gap" | "max_input_length" | "depth_limit" | "max_shingle_size" => match v {
            Value::String(s) => match s.trim().parse::<i64>() {
                Ok(n) => json!(n),
                Err(_) => return Err(nfe(s)),
            },
            other => other.clone(),
        },
        _ => v.clone(),
    }))
}

fn default_date_format() -> &'static str {
    "strict_date_optional_time||epoch_millis"
}

/// `true` when `v` is the default of `k` for `ty` (ES then omits it).
fn is_default(ty: EsType, k: &str, v: &Value) -> bool {
    use EsType as T;
    let text_like = matches!(ty, T::Text | T::SearchAsYouType | T::AnnotatedText);
    match k {
        "meta" => v.as_object().is_some_and(Map::is_empty),
        "copy_to" => v.as_array().is_some_and(Vec::is_empty),
        "null_value" => v.is_null(),
        "index" => v == &json!(true) && !matches!(ty, T::Alias | T::DenseVector),
        "doc_values" => {
            let dflt = !matches!(ty, T::Binary | T::SearchAsYouType | T::Text | T::MatchOnlyText);
            ty != T::SearchAsYouType && v == &json!(dflt)
        }
        "store" => v == &json!(false),
        "norms" => v == &json!(text_like),
        "eager_global_ordinals" => v == &json!(false) && ty != T::Join,
        "split_queries_on_whitespace" | "index_phrases" | "fielddata" | "time_series_dimension" => v == &json!(false),
        "ignore_malformed" => v == &json!(false) && matches!(ty, T::Date | T::DateNanos | T::Boolean | T::Ip | T::Long | T::Integer | T::Short | T::Byte | T::Double | T::Float | T::HalfFloat | T::ScaledFloat | T::UnsignedLong | T::GeoPoint | T::Point | T::GeoShape | T::Shape),
        "ignore_above" => v == &json!(2147483647),
        "position_increment_gap" => v == &json!(100),
        "index_options" => match ty {
            T::Keyword | T::Flattened | T::PatternText => v == &json!("docs"),
            T::Text | T::SearchAsYouType => v == &json!("positions"),
            _ => false,
        },
        "term_vector" => v == &json!("no"),
        "analyzer" if ty == T::PatternText => v == &json!("delimiter"),
        "disable_templating" if ty == T::PatternText => v == &json!(false),
        "analyzer" | "search_analyzer" => v == &json!("default") && text_like,
        "depth_limit" => v == &json!(20),
        "enable_position_increments" => v == &json!(true),
        "positive_score_impact" => v == &json!(true),
        "element_type" => v == &json!("float"),
        "on_script_error" => v == &json!("fail"),
        "fielddata_frequency_filter" => v.as_object().is_some_and(Map::is_empty),
        "locale" => matches!(ty, T::Date | T::DateNanos | T::DateRange | T::IntegerRange | T::LongRange | T::FloatRange | T::DoubleRange | T::IpRange) && (v == &json!("en") || (ty != T::DateRange && ty != T::Date && ty != T::DateNanos)),
        "format" => (matches!(ty, T::Date | T::DateRange) && v == &json!(default_date_format())) || (ty == T::DateNanos && v == &json!("strict_date_optional_time_nanos||epoch_millis")),
        _ => false,
    }
}

/// Typed, validated `null_value` (field `name`, given `params`).
fn null_value(name: &str, ty: EsType, nv: &Value, params: &Map<String, Value>) -> Result<Value> {
    use EsType as T;
    if nv.is_null() {
        return Ok(Value::Null);
    }
    let s = match nv {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let out_of_range = |what: &str| iae(format!("Value [{s}] is out of range for {what}"));
    Ok(match ty {
        T::Long | T::Integer | T::Short | T::Byte | T::TokenCount => {
            let n: i64 = match nv {
                Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).ok_or_else(|| out_of_range("a long"))?,
                Value::String(t) => t.trim().parse::<i64>().or_else(|_| t.trim().parse::<f64>().map(|f| f.trunc() as i64)).map_err(|_| nfe(t))?,
                _ => return Err(iae(format!("Failed to parse value [{}]", jstr(nv)))),
            };
            let (lo, hi, what) = match ty {
                T::Byte => (i8::MIN as i64, i8::MAX as i64, "a byte"),
                T::Short => (i16::MIN as i64, i16::MAX as i64, "a short"),
                T::Integer | T::TokenCount => (i32::MIN as i64, i32::MAX as i64, "an integer"),
                _ => (i64::MIN, i64::MAX, "a long"),
            };
            if n < lo || n > hi {
                return Err(out_of_range(what));
            }
            json!(n)
        }
        T::Double | T::Float | T::HalfFloat | T::ScaledFloat => match nv {
            Value::Number(_) => nv.clone(),
            Value::String(t) => Number::from_f64(t.trim().parse::<f64>().map_err(|_| nfe(t))?).map(Value::Number).unwrap_or_else(|| nv.clone()),
            _ => nv.clone(),
        },
        T::UnsignedLong => {
            let t = s.trim().trim_start_matches('+');
            let ok = t.parse::<u64>().ok().or_else(|| t.parse::<f64>().ok().filter(|f| *f >= 0.0 && *f <= u64::MAX as f64).map(|f| f as u64));
            match ok {
                Some(u) => Value::String(u.to_string()),
                None => return Err(nested_mpe("null_value", name, &format!("Value [{s}] is out of range for unsigned long."))),
            }
        }
        T::Boolean => Value::Bool(to_bool(nv)?),
        T::Date | T::DateNanos => {
            let fmt = params.get("format").and_then(Value::as_str).unwrap_or(default_date_format());
            if let Value::String(t) = nv {
                let ok = crate::date::DateFormatter::new(fmt).ok().is_some_and(|df| df.parse(t).is_ok());
                if !ok {
                    let dtpe = EsError::new(400, "date_time_parse_exception", "Failed to parse with all enclosed parsers");
                    let inner = format!("failed to parse date field [{t}] with format [{fmt}]");
                    return Err(EsError::mapper_parsing(format!("Error parsing [null_value] on field [{name}]: {inner}")).caused_by(EsError::illegal_argument(inner).caused_by(dtpe)));
                }
            }
            nv.clone()
        }
        T::Ip => {
            if rustic_core::fieldtypes::ip::parse(&s).is_none() {
                return Err(nested_mpe("null_value", name, &format!("'{s}' is not an IP string literal.")));
            }
            nv.clone()
        }
        T::GeoPoint => match crate::values::parse_geo_point(nv) {
            Some((lat, lon)) => json!({"lat": lat, "lon": lon}),
            None => {
                let m = format!("unsupported symbol [{}] in geohash [{s}]", s.chars().find(|c| !"0123456789bcdefghjkmnpqrstuvwxyz".contains(*c)).unwrap_or('?'));
                return Err(EsError::mapper_parsing(m.clone()).caused_by(EsError::new(400, "parse_exception", m.clone()).caused_by(EsError::illegal_argument(m))));
            }
        },
        T::Point => match crate::fieldtypes::index::parse_xy_point(nv, &rustic_core::fieldtypes::shape::ParseOpts { geo: false, coerce: false, ignore_z: true }) {
            Ok((x, y)) => json!({"x": x, "y": y}),
            Err(e) => return Err(iae(e)),
        },
        T::Keyword | T::Wildcard | T::Flattened | T::Text | T::MatchOnlyText => match nv {
            Value::String(_) => nv.clone(),
            other => Value::String(jstr(other)),
        },
        _ => nv.clone(),
    })
}

/// A text field with a custom search / quote analyzer prints the (default) index
/// analyzer too.
pub fn derive_analyzers(ty: EsType, params: &mut Map<String, Value>) {
    let sqa = params.contains_key("search_quote_analyzer");
    if ty.is_text() && (sqa || params.contains_key("search_analyzer")) {
        params.entry("analyzer").or_insert(json!("default"));
        if sqa {
            params.entry("search_analyzer").or_insert(json!("default"));
        }
    }
}

/// Whole-field canonicalisation after all parameters are parsed.
pub fn finish(name: &str, ty: EsType, params: &mut Map<String, Value>) -> Result<()> {
    use EsType as T;
    if let Some(nv) = params.get("null_value").cloned() {
        let typed = null_value(name, ty, &nv, params)?;
        params.insert("null_value".into(), typed);
    }
    // `index_prefixes: {}` prints its defaults; `fielddata_frequency_filter` too.
    if let Some(Value::Object(ip)) = params.get_mut("index_prefixes") {
        let min = ip.get("min_chars").cloned().unwrap_or(json!(2));
        let max = ip.get("max_chars").cloned().unwrap_or(json!(5));
        let (mn, mx) = (min.as_i64().unwrap_or(2), max.as_i64().unwrap_or(5));
        if mn <= 0 {
            return Err(iae(format!("min_chars [{mn}] must be greater than zero")));
        }
        if mn >= mx {
            return Err(iae(format!("min_chars [{mn}] must be less than max_chars [{mx}]")));
        }
        if mx >= 20 {
            return Err(iae(format!("max_chars [{mx}] must be less than 20")));
        }
        ip.insert("min_chars".into(), min);
        ip.insert("max_chars".into(), max);
    }
    if let Some(Value::Object(f)) = params.get_mut("fielddata_frequency_filter") {
        if !f.is_empty() {
            f.entry("min").or_insert(json!(0.0));
            f.entry("max").or_insert(json!(1.7976931348623157e308));
            f.entry("min_segment_size").or_insert(json!(0));
        }
    }
    if params.get("index_phrases") == Some(&json!(true)) {
        let io = params.get("index_options").and_then(Value::as_str).unwrap_or("positions");
        if params.get("index") == Some(&json!(false)) || !matches!(io, "positions" | "offsets") {
            return Err(iae(format!("Cannot set index_phrases on field [{name}] if positions are not enabled")));
        }
    }
    match ty {
        T::Completion => {
            for (k, d) in [("analyzer", json!("simple")), ("preserve_separators", json!(true)), ("preserve_position_increments", json!(true)), ("max_input_length", json!(50))] {
                params.entry(k).or_insert(d);
            }
            if let Some(Value::Array(ctxs)) = params.get_mut("contexts") {
                for c in ctxs {
                    if let Some(t) = c.get("type").and_then(Value::as_str).map(str::to_ascii_uppercase) {
                        c["type"] = json!(t);
                    }
                }
            }
        }
        T::Join => {
            params.entry("eager_global_ordinals").or_insert(json!(true));
            params.entry("relations").or_insert(json!({}));
        }
        T::DenseVector => {
            params.entry("index").or_insert(json!(true));
        }
        T::SparseVector => {
            if let Some(io) = params.get("index_options").cloned() {
                match sparse_index_options(&io)? {
                    Some(canon) => params.insert("index_options".into(), canon),
                    None => params.remove("index_options"),
                };
            }
        }
        T::SearchAsYouType => {
            params.entry("doc_values").or_insert(json!(false));
            params.entry("max_shingle_size").or_insert(json!(3));
        }
        _ => {}
    }
    let sqa = ty.is_text() && params.contains_key("search_quote_analyzer");
    let drop: Vec<String> = params.iter().filter(|(k, v)| is_default(ty, k, v)).map(|(k, _)| k.clone()).collect();
    for k in drop {
        params.remove(&k);
    }
    derive_analyzers(ty, params);
    // `locale: ROOT` prints as the empty root locale.
    if matches!(ty, T::Date | T::DateNanos | T::DateRange) && params.get("locale").and_then(Value::as_str) == Some("ROOT") {
        params.insert("locale".into(), json!(""));
    }
    Ok(())
}

/// A float parameter as Java renders it after `parser.floatValue()`.
fn java_float(x: f32) -> Value {
    format!("{x}").parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::Null)
}

/// `TokenPruningConfig` (`pruning_config`) parsed and validated: the
/// canonical rendering.
pub fn token_pruning_config(v: &Value) -> Result<Value> {
    let o = v.as_object().ok_or_else(|| iae("[pruning_config] must be an object"))?;
    let num = |k: &str, d: f32| -> Result<f32> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or(d as f64) as f32),
            Some(Value::String(s)) => s.trim().parse::<f32>().map_err(|_| nfe(s)),
            Some(other) => Err(iae(format!("[{k}] must be a number, got [{}]", jstr(other)))),
        }
    };
    for k in o.keys() {
        if !matches!(k.as_str(), "tokens_freq_ratio_threshold" | "tokens_weight_threshold" | "only_score_pruned_tokens") {
            return Err(iae(format!("[pruning_config] unknown field [{k}]")));
        }
    }
    let freq = num("tokens_freq_ratio_threshold", 5.0)?;
    let weight = num("tokens_weight_threshold", 0.4)?;
    let only = match o.get("only_score_pruned_tokens") {
        None | Some(Value::Null) => false,
        Some(b) => to_bool(b)?,
    };
    if !(1.0..=100.0).contains(&freq) {
        return Err(iae(format!("[tokens_freq_ratio_threshold] must be between [1] and [100], got {}", jdouble(freq as f64))));
    }
    if !(0.0..=1.0).contains(&weight) {
        return Err(iae("[tokens_weight_threshold] must be between 0 and 1"));
    }
    let mut m = Map::new();
    m.insert("tokens_freq_ratio_threshold".into(), java_float(freq));
    m.insert("tokens_weight_threshold".into(), java_float(weight));
    if only {
        m.insert("only_score_pruned_tokens".into(), json!(true));
    }
    Ok(Value::Object(m))
}

/// `sparse_vector` `index_options` (`SparseVectorFieldMapper.IndexOptions`):
/// `None` when they equal the defaults (`prune: true` with the default
/// pruning config), which ES does not serialize.
fn sparse_index_options(v: &Value) -> Result<Option<Value>> {
    if v.is_null() {
        return Ok(None);
    }
    let o = v.as_object().ok_or_else(|| iae("[index_options] must be an object"))?;
    for k in o.keys() {
        if !matches!(k.as_str(), "prune" | "pruning_config") {
            return Err(iae(format!("[index_options] unknown field [{k}]")));
        }
    }
    let prune = match o.get("prune") {
        None | Some(Value::Null) => None,
        Some(b) => Some(to_bool(b)?),
    };
    let config = match o.get("pruning_config") {
        None | Some(Value::Null) => None,
        Some(c) => Some(token_pruning_config(c)?),
    };
    if config.is_some() && prune != Some(true) {
        return Err(iae("[index_options] field [pruning_config] should only be set if [prune] is set to true"));
    }
    let default_config = json!({"tokens_freq_ratio_threshold": 5.0, "tokens_weight_threshold": 0.4});
    if prune == Some(true) && config.as_ref() == Some(&default_config) {
        return Ok(None);
    }
    let mut m = Map::new();
    if let Some(p) = prune {
        m.insert("prune".into(), json!(p));
    }
    if let Some(c) = config {
        m.insert("pruning_config".into(), c);
    }
    Ok(Some(Value::Object(m)))
}

// ---------------------------------------------------------------------------
// Mapping updates.

/// ES's parameter declaration order per mapper (conflicts are reported in it).
pub fn param_order(ty: EsType) -> &'static [&'static str] {
    use EsType as T;
    match ty {
        T::Keyword => &["index", "doc_values", "store", "null_value", "eager_global_ordinals", "ignore_above", "index_options", "norms", "similarity", "normalizer", "normalizer_skip_store_original_value", "split_queries_on_whitespace", "meta", "time_series_dimension", "script", "on_script_error"],
        T::Text | T::MatchOnlyText => &["index", "doc_values", "store", "index_options", "norms", "term_vector", "analyzer", "search_analyzer", "search_quote_analyzer", "similarity", "position_increment_gap", "fielddata", "fielddata_frequency_filter", "eager_global_ordinals", "index_phrases", "index_prefixes", "meta"],
        T::Long | T::Integer | T::Short | T::Byte | T::Double | T::Float | T::HalfFloat | T::ScaledFloat | T::UnsignedLong => &["scaling_factor", "index", "doc_values", "store", "ignore_malformed", "coerce", "null_value", "script", "on_script_error", "meta", "time_series_dimension", "time_series_metric"],
        T::Date | T::DateNanos => &["index", "doc_values", "store", "format", "locale", "ignore_malformed", "null_value", "script", "on_script_error", "meta"],
        T::Boolean => &["doc_values", "index", "null_value", "store", "meta", "script", "on_script_error", "time_series_dimension"],
        T::Ip => &["index", "doc_values", "store", "ignore_malformed", "null_value", "script", "on_script_error", "meta", "time_series_dimension"],
        T::Flattened => &["null_value", "ignore_above", "index", "doc_values", "depth_limit", "eager_global_ordinals", "index_options", "split_queries_on_whitespace", "meta"],
        _ => &[],
    }
}

fn parent_children(v: Option<&Value>) -> Vec<(String, Vec<String>)> {
    v.and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .map(|(p, c)| {
                    let ch = match c {
                        Value::String(s) => vec![s.clone()],
                        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                        _ => vec![],
                    };
                    (p.clone(), ch)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `Conflict in parameter [relations]: ..` lines of a join `relations` update.
pub fn join_conflicts(old: Option<&Value>, new: Option<&Value>) -> Vec<String> {
    let (o, n) = (parent_children(old), parent_children(new));
    let mut out = Vec::new();
    for (p, kids) in &o {
        match n.iter().find(|(np, _)| np == p) {
            None => out.push(format!("Conflict in parameter [relations]: Cannot remove parent [{p}]")),
            Some((_, nk)) => {
                for c in kids {
                    if !nk.contains(c) {
                        if n.iter().any(|(_, k)| k.contains(c)) {
                            out.push(format!("Conflict in parameter [relations]: Cannot change parent of [{c}]"));
                        } else {
                            out.push(format!("Conflict in parameter [relations]: Cannot remove child [{c}]"));
                        }
                    }
                }
            }
        }
    }
    for (p, kids) in &o {
        if n.iter().any(|(np, _)| np == p) {
            continue;
        }
        for c in kids {
            if n.iter().any(|(_, k)| k.contains(c)) {
                out.push(format!("Conflict in parameter [relations]: Cannot change parent of [{c}]"));
            }
        }
    }
    out
}

/// May an update change `k` from `old` to `new` in place?
pub fn updatable(ty: EsType, k: &str, old: Option<&Value>, new: Option<&Value>) -> bool {
    use EsType as T;
    match k {
        "ignore_above" | "search_analyzer" | "search_quote_analyzer" | "meta" | "eager_global_ordinals" | "copy_to" | "ignore_malformed" | "coerce" | "fielddata" | "fielddata_frequency_filter" | "split_queries_on_whitespace" | "depth_limit" | "orientation" | "ignore_z_value" => true,
        "norms" => {
            // Norms can be dropped, never added.
            let o = old.and_then(Value::as_bool).unwrap_or(matches!(ty, T::Text | T::MatchOnlyText | T::SearchAsYouType));
            let n = new.and_then(Value::as_bool).unwrap_or(matches!(ty, T::Text | T::MatchOnlyText | T::SearchAsYouType));
            o || !n
        }
        "analyzer" => ty == T::TokenCount,
        "path" => ty == T::Alias,
        "value" => ty == T::ConstantKeyword && old.is_none_or(Value::is_null),
        // rank_vectors without `dims` takes them from the first document
        // (a dynamic mapping update); once set they never change.
        "dims" => ty == T::RankVectors && old.is_none_or(Value::is_null),
        "relations" => ty == T::Join && join_conflicts(old, new).is_empty(),
        _ => false,
    }
}

fn nanos(ty: EsType) -> bool {
    ty == EsType::DateNanos
}

/// A parameter value as it appears in an update conflict (`None` shows the default).
pub fn display(ty: EsType, k: &str, v: Option<&Value>) -> String {
    use EsType as T;
    match v {
        Some(Value::Object(o)) if k == "index_prefixes" => format!("{{ min_chars={}, max_chars={} }}", o.get("min_chars").map(jstr).unwrap_or_default(), o.get("max_chars").map(jstr).unwrap_or_default()),
        Some(Value::Object(o)) if k == "script" => format!(
            "Script{{type=inline, lang='{}', idOrCode='{}', options={{}}, params={{}}}}",
            o.get("lang").map(jstr).unwrap_or_else(|| "painless".into()),
            o.get("source").map(jstr).unwrap_or_default()
        ),
        Some(Value::Object(_)) if k == "relations" => {
            let rel = parent_children(v);
            format!("[{}]", rel.iter().map(|(p, c)| format!("Relations[parent={p}, children=[{}]]", c.join(", "))).collect::<Vec<_>>().join(", "))
        }
        Some(x) => jstr(x),
        None => match k {
            "index" | "doc_values" if !(k == "doc_values" && ty == T::Binary) => "true".into(),
            "doc_values" => "false".into(),
            "norms" => (matches!(ty, T::Text | T::MatchOnlyText | T::SearchAsYouType)).to_string(),
            "coerce" | "enable_position_increments" | "positive_score_impact" => "true".into(),
            "store" | "eager_global_ordinals" | "split_queries_on_whitespace" | "index_phrases" | "fielddata" | "time_series_dimension" | "ignore_malformed" | "ignore_z_value" => "false".into(),
            "ignore_above" => "2147483647".into(),
            "position_increment_gap" => "100".into(),
            "depth_limit" => "20".into(),
            "index_options" => if matches!(ty, T::Keyword | T::Flattened) { "docs".into() } else { "positions".into() },
            "term_vector" => "no".into(),
            "format" => if nanos(ty) { "strict_date_optional_time_nanos||epoch_millis".into() } else { default_date_format().into() },
            "locale" => "en".into(),
            "analyzer" | "search_analyzer" | "search_quote_analyzer" => "default".into(),
            "meta" => "{}".into(),
            "copy_to" | "fields" => "[]".into(),
            "element_type" => "float".into(),
            _ => "null".into(),
        },
    }
}
