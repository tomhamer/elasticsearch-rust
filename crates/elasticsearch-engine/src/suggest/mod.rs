//! The `suggest` section of `_search`: `term`, `phrase` and `completion`
//! suggesters. Completion lives with its field type in
//! [`crate::fieldtypes::suggest`]; index-time context handling is in
//! [`index`].

pub mod distance;
pub mod index;
pub mod phrase;
pub mod term;

use crate::error::{EsError, Result};
use crate::index::IndexService;
use crate::mapping::{FieldLookup, ResolvedField};
use rustic_core::analysis::Analyzer;
use rustic_core::query::LeafReader;
use serde_json::{Map, Value};
use std::sync::Arc;

/// One searched index for the suggesters.
pub struct SuggestTarget<'a> {
    pub index: &'a str,
    pub leaves: &'a [LeafReader<'a>],
    pub lookup: &'a FieldLookup,
    /// The index (analyzers, settings, collate queries).
    pub svc: Option<&'a IndexService>,
}

/// Run the `suggest` section; returns the response `suggest` object
/// (`typed_keys` prefixes names with `term#` / `phrase#` / `completion#`).
pub fn run(spec: &Value, targets: &[SuggestTarget], typed_keys: bool) -> Result<Value> {
    run_with_source(spec, targets, typed_keys, &crate::search::SourceFilter::All)
}

/// [`run`] with the request's `_source` filter applied to completion hits.
pub fn run_with_source(spec: &Value, targets: &[SuggestTarget], typed_keys: bool, source: &crate::search::SourceFilter) -> Result<Value> {
    let o = spec.as_object().ok_or_else(|| EsError::parsing("[suggest] must be an object"))?;
    // ES reads `text` with `parser.text()`: a number is its literal text.
    let as_text = |v: &Value| match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    };
    let global_text = o.get("text").and_then(as_text);
    let mut out = Map::new();
    for (name, def) in o {
        if name == "text" {
            continue;
        }
        let d = def.as_object().ok_or_else(|| EsError::parsing(format!("suggestion [{name}] must be an object")))?;
        let text = d
            .get("prefix")
            .or_else(|| d.get("regex"))
            .or_else(|| d.get("text"))
            .and_then(as_text)
            .or_else(|| global_text.clone());
        let kind = ["completion", "term", "phrase"].into_iter().find(|k| d.contains_key(*k));
        let Some(kind) = kind else {
            let bad = d.keys().find(|k| !matches!(k.as_str(), "text" | "prefix" | "regex"));
            return Err(match bad {
                Some(k) => EsError::new(400, "x_content_parse_exception", format!("[1:1] unknown field [{k}]"))
                    .caused_by(EsError::new(400, "named_object_not_found_exception", format!("[1:1] unknown field [{k}]"))),
                None => EsError::illegal_argument(format!("suggestion [{name}] does not have a known suggester type")),
            });
        };
        let body = d.get(kind).and_then(Value::as_object).ok_or_else(|| EsError::parsing(format!("[{kind}] suggester must be an object")))?;
        let text = match text {
            Some(t) => t.to_string(),
            None => return Err(EsError::illegal_argument("The required text option is missing")),
        };
        let res = match kind {
            "completion" => {
                let regex = d.get("regex").and_then(Value::as_str).is_some() && d.get("prefix").is_none();
                Value::Array(vec![crate::fieldtypes::suggest::completion(&text, regex, body, targets, source)?])
            }
            "term" => term::run(&text, body, targets)?,
            _ => phrase::run(&text, body, targets)?,
        };
        let key = if typed_keys { format!("{kind}#{name}") } else { name.clone() };
        out.insert(key, res);
    }
    Ok(Value::Object(out))
}

/// ES `SuggestionBuilder.fromXContent`: unknown keys are a parsing error.
pub fn check_keys(kind: &str, body: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    match body.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(EsError::parsing(format!("suggester[{kind}] doesn't support field [{k}]"))),
        None => Ok(()),
    }
}

/// Java float rendering (`(float) x` printed as the shortest round-trip).
pub fn jfloat(x: f32) -> Value {
    let s = format!("{x}");
    s.parse::<f64>().ok().and_then(serde_json::Number::from_f64).map(Value::Number).unwrap_or(Value::Null)
}

/// Analyzer for suggesting over `f`: the explicit `analyzer`, else the
/// field's search analyzer (keyword fields: their normalizer).
pub fn field_analyzer(t: &SuggestTarget, f: &ResolvedField, name: Option<&str>) -> Result<Arc<Analyzer>> {
    if let Some(n) = name {
        let reg = t.svc.map(|s| s.registry());
        if let Some(a) = reg.as_ref().and_then(|r| r.analyzer(n)) {
            return Ok(a);
        }
        if let Some(a) = Analyzer::by_name(n) {
            return Ok(Arc::new(a));
        }
        return Err(EsError::illegal_argument(format!("analyzer [{n}] doesn't exists")));
    }
    if f.ty.is_keyword_like() {
        return Ok(f.normalizer.clone().unwrap_or_else(|| Arc::new(Analyzer::keyword())));
    }
    if let Some(a) = &f.search_analyzer {
        return Ok(a.clone());
    }
    Ok(match &f.index_analyzer {
        Some(crate::analysis::IndexAnalyzer::Custom(a)) => a.clone(),
        Some(crate::analysis::IndexAnalyzer::Builtin(n)) => Arc::new(Analyzer::by_name(n).unwrap_or_else(Analyzer::standard)),
        None => Arc::new(Analyzer::standard()),
    })
}

/// An analyzed token with UTF-16 offsets (as ES reports them).
#[derive(Clone, Debug)]
pub struct Tok {
    pub text: String,
    pub position: u32,
    pub offset: usize,
    pub length: usize,
}

pub fn tokens(a: &Analyzer, text: &str) -> Vec<Tok> {
    let u16_at = |b: usize| -> usize {
        let mut b = b.min(text.len());
        while !text.is_char_boundary(b) {
            b -= 1;
        }
        text[..b].encode_utf16().count()
    };
    a.analyze(text)
        .into_iter()
        .map(|t| {
            let s = u16_at(t.offset_from as usize);
            let e = u16_at(t.offset_to as usize);
            Tok { text: t.text, position: t.position, offset: s, length: e.saturating_sub(s) }
        })
        .collect()
}
