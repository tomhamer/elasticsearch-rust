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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisRegistry;
    use crate::document::parse_with_dynamic;
    use crate::mapping::Mappings;
    use rustic_core::types::IndexSettings;
    use rustic_core::Index;
    use serde_json::json;

    struct Fx {
        index: Index,
        lookup: FieldLookup,
    }

    fn fx(mapping: Value, docs: &[Value]) -> Fx {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&mapping, &reg).unwrap();
        m.assign_core_ids();
        let lookup = FieldLookup::build(&m, &reg);
        let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
        for (i, d) in docs.iter().enumerate() {
            let (pd, upd) = parse_with_dynamic(d, &i.to_string(), &m, &lookup, 1000).unwrap();
            assert!(upd.is_none());
            index.add_document(&pd.doc).unwrap();
        }
        index.refresh().unwrap();
        Fx { index, lookup }
    }

    fn with_target<T>(f: &Fx, run: impl FnOnce(&[SuggestTarget]) -> T) -> T {
        let leaves = f.index.leaves();
        let t = SuggestTarget { index: "idx", leaves: &leaves, lookup: &f.lookup, svc: None };
        run(std::slice::from_ref(&t))
    }

    /// Six misspelt tokens, each with four one-edit corrections indexed.
    fn grid() -> (Fx, String) {
        let words: Vec<String> = (1..=6).flat_map(|i| ["b", "c", "d", "e"].map(|s| format!("tok{i}{s}"))).collect();
        let f = fx(json!({"properties": {"body": {"type": "text"}}}), &[json!({"body": words.join(" ")})]);
        let text = (1..=6).map(|i| format!("tok{i}a")).collect::<Vec<_>>().join(" ");
        (f, text)
    }

    /// #167: the correction tree is explored under a path budget, polls
    /// cancellation, and refuses an excessive depth; within the budget the
    /// suggestions are unchanged.
    #[test]
    fn phrase_correction_search_is_bounded() {
        let (f, text) = grid();
        let o = |errors: u64| json!({"field": "body", "max_errors": errors, "confidence": 0.0, "direct_generator": [{"field": "body", "suggest_mode": "always"}]});
        let one_error = |budget: Option<u64>| {
            phrase::PATH_BUDGET.with(|b| b.set(budget));
            let r = with_target(&f, |t| phrase::run(&text, o(1).as_object().unwrap(), t));
            phrase::PATH_BUDGET.with(|b| b.set(None));
            r
        };
        // One error: ~90 paths, the same options under a 200-path budget.
        let full = one_error(None).unwrap();
        assert!(!full[0]["options"].as_array().unwrap().is_empty(), "{full}");
        assert_eq!(one_error(Some(200)).unwrap(), full);
        // Six errors: 5^6 paths; a 200-path budget stops the search.
        phrase::PATH_BUDGET.with(|b| b.set(Some(200)));
        let e = with_target(&f, |t| phrase::run(&text, o(6).as_object().unwrap(), t)).unwrap_err();
        phrase::PATH_BUDGET.with(|b| b.set(None));
        assert_eq!(e.status, 400);
        assert!(e.reason.contains("explores more than [200] correction paths"), "{}", e.reason);
        // The same search under the default budget completes, and stops
        // when its task is cancelled.
        with_target(&f, |t| phrase::run(&text, o(6).as_object().unwrap(), t)).unwrap();
        let gone = rustic_core::cancel::Token::cancelled_with("by user request");
        let e = rustic_core::cancel::with_search(Some(gone), || with_target(&f, |t| phrase::run(&text, o(6).as_object().unwrap(), t))).unwrap_err();
        assert_eq!(e.kind, "task_cancelled_exception");
        // Depth: more tokens than the search recurses over, and token_limit 0.
        let long = vec!["tok1a"; phrase::MAX_PHRASE_TOKENS + 1].join(" ");
        let deep = json!({"field": "body", "token_limit": 5000, "direct_generator": [{"field": "body", "suggest_mode": "always"}]});
        let e = with_target(&f, |t| phrase::run(&long, deep.as_object().unwrap(), t)).unwrap_err();
        assert!(e.reason.contains("corrects at most"), "{}", e.reason);
        let e = with_target(&f, |t| phrase::run(&text, json!({"field": "body", "token_limit": 0}).as_object().unwrap(), t)).unwrap_err();
        assert_eq!(e.reason, "token_limit must be >= 1");
    }

    /// #168: a term suggestion for a 16,000-char keyword one edit from an
    /// indexed one is found with banded distance work, and polls cancellation.
    #[test]
    fn term_suggestion_of_a_long_keyword_is_banded() {
        let base: String = (0..16_000).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let f = fx(json!({"properties": {"kw": {"type": "keyword"}}}), &[json!({"kw": base.clone()})]);
        let mut near: Vec<char> = base.chars().collect();
        near[15_990] = '#';
        let text: String = near.into_iter().collect();
        let o = json!({"field": "kw"});
        let started = std::time::Instant::now();
        let r = with_target(&f, |t| term::run(&text, o.as_object().unwrap(), t)).unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(5), "{:?}", started.elapsed());
        let opts = r[0]["options"].as_array().unwrap();
        assert_eq!(opts.len(), 1);
        assert_eq!(opts[0]["text"].as_str().unwrap().len(), 16_000);
        assert_eq!(opts[0]["text"].as_str().unwrap(), base);
        // Two edits apart (default max_edits 2) is found; three is not.
        let mut two: Vec<char> = base.chars().collect();
        two[100] = '#';
        two[15_000] = '#';
        let r = with_target(&f, |t| term::run(&two.iter().collect::<String>(), o.as_object().unwrap(), t)).unwrap();
        assert_eq!(r[0]["options"].as_array().unwrap().len(), 1);
        two[9_000] = '#';
        let r = with_target(&f, |t| term::run(&two.iter().collect::<String>(), o.as_object().unwrap(), t)).unwrap();
        assert!(r[0]["options"].as_array().unwrap().is_empty());
        let gone = rustic_core::cancel::Token::cancelled_with("by user request");
        let e = rustic_core::cancel::with_search(Some(gone), || with_target(&f, |t| term::run(&text, o.as_object().unwrap(), t))).unwrap_err();
        assert_eq!(e.kind, "task_cancelled_exception");
    }
}
