//! Search highlighting (`highlight` in `_search` and `inner_hits`) over
//! `_source` values, mirroring Elasticsearch's three highlighters:
//!
//! - [`unified`] (default): Lucene `UnifiedHighlighter` + ES
//!   `CustomUnifiedHighlighter` — sentence passages bounded by
//!   `fragment_size` (ES `BoundedBreakIteratorScanner`), BM25 passage
//!   scoring, weight-matches mode (a phrase match is one `<em>`), word /
//!   whole-value break iterators, `no_match_size` to a word boundary.
//! - [`plain`]: Lucene `Highlighter` + `QueryScorer` + `SimpleSpanFragmenter`
//!   (token-by-token fragments, score = distinct matched terms).
//! - [`fvh`]: Lucene `FastVectorHighlighter` (`SimpleFragListBuilder`
//!   margins, `SimpleBoundaryScanner` chars / break-iterator scanners,
//!   discrete multi-value fragments, multi-tag `seqnum`); requires
//!   `term_vector: with_positions_offsets`.
//!
//! Matches come from re-analyzing each value with the field's index
//! analyzer and matching the query against those tokens
//! ([`matches`]): phrase / span / interval queries only highlight where they
//! match positionally, `must_not` never highlights. Offsets are UTF-16 code
//! units as in Java, so `fragment_size` etc. count the same characters.

mod breaks;
mod fvh;
pub mod matches;
mod plain;
mod unified;

use crate::error::{EsError, Result};
use crate::mapping::{FieldLookup, Mapper, ObjectMapping, ResolvedField};
use matches::{Match, Stream, Tok};
use rustic_core::analysis::Analyzer;
use rustic_core::query::Query;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Paragraph separator joining multiple values (unified highlighter).
pub(crate) const MULTIVAL_SEP: u16 = 0x2029;
const MAX_FRAGMENTS: i64 = 10_000;
const DEFAULT_MAX_ANALYZED_OFFSET: usize = 1_000_000;

/// How a field was indexed (decides the unified highlighter's offset source
/// and whether fvh can run).
#[derive(Clone, Copy, Default, Debug)]
pub struct FieldFlags {
    /// `index_options: offsets`.
    pub postings_offsets: bool,
    /// `term_vector: with_*offsets*`.
    pub tv_offsets: bool,
}

/// Everything the fetch phase needs to highlight hits of one index: the
/// query (or `highlight_query` overrides), index settings and per-field
/// index options.
#[derive(Clone)]
pub struct QueryTerms {
    query: Query,
    global_hq: Option<Query>,
    field_hq: HashMap<String, Query>,
    weight_matches_setting: bool,
    index_max_offset: usize,
    flags: Arc<HashMap<String, FieldFlags>>,
    index: String,
    synthetic_source: bool,
}

impl std::fmt::Debug for QueryTerms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryTerms").field("index", &self.index).finish()
    }
}

impl QueryTerms {
    /// Highlight state for `q` with default index settings.
    pub fn collect(q: &Query) -> QueryTerms {
        QueryTerms {
            query: q.clone(),
            global_hq: None,
            field_hq: HashMap::new(),
            weight_matches_setting: true,
            index_max_offset: DEFAULT_MAX_ANALYZED_OFFSET,
            flags: Arc::new(HashMap::new()),
            index: String::new(),
            synthetic_source: false,
        }
    }

    /// Highlight state for one searched index: parses the request's
    /// `highlight_query` (global and per field) with `parse`.
    pub fn for_index(q: &Query, spec: &Map<String, Value>, svc: &crate::index::IndexService, parse: &dyn Fn(&Value) -> Result<Query>) -> Result<QueryTerms> {
        let mut t = QueryTerms::collect(q);
        t.index = svc.name.clone();
        t.index_max_offset = svc.setting_usize("index.highlight.max_analyzed_offset", DEFAULT_MAX_ANALYZED_OFFSET);
        t.weight_matches_setting = svc.setting("index.highlight.weight_matches_mode.enabled").map(|s| s != "false").unwrap_or(true);
        let mut flags = HashMap::new();
        let meta = svc.meta();
        collect_flags(&meta.mappings.root, "", &mut flags);
        t.synthetic_source = svc.setting("index.mapping.source.mode").map(|m| m == "synthetic").unwrap_or(false)
            || meta.mappings.source.as_ref().and_then(|s| s.get("mode")).and_then(Value::as_str) == Some("synthetic");
        t.flags = Arc::new(flags);
        if let Some(hq) = spec.get("highlight_query") {
            t.global_hq = Some(parse(hq)?);
        }
        for (name, fo) in field_specs(spec) {
            if let Some(hq) = fo.get("highlight_query") {
                t.field_hq.insert(name, parse(hq)?);
            }
        }
        Ok(t)
    }

    /// The query a field pattern highlights against (`highlight_query` overrides).
    pub(crate) fn query_for(&self, pattern: &str) -> &Query {
        self.field_hq.get(pattern).or(self.global_hq.as_ref()).unwrap_or(&self.query)
    }
}

fn collect_flags(o: &ObjectMapping, prefix: &str, out: &mut HashMap<String, FieldFlags>) {
    for (k, m) in &o.properties {
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        match m {
            Mapper::Object(child) => collect_flags(child, &path, out),
            Mapper::Field(f) => {
                fn add(f: &crate::mapping::FieldMapping, path: &str, out: &mut HashMap<String, FieldFlags>) {
                    let tv = f.params.get("term_vector").and_then(Value::as_str).unwrap_or("no");
                    out.insert(
                        path.to_string(),
                        FieldFlags { postings_offsets: f.params.get("index_options").and_then(Value::as_str) == Some("offsets"), tv_offsets: tv.contains("offsets") },
                    );
                    for (sk, sf) in &f.fields {
                        add(sf, &format!("{path}.{sk}"), out);
                    }
                }
                add(f, &path, out);
            }
        }
    }
}

/// Options of one highlighted field (field-level values over global ones).
#[derive(Clone, Debug)]
pub struct HighlightOpts {
    pub pre: Vec<String>,
    pub post: Vec<String>,
    pub fragment_size: usize,
    pub number_of_fragments: usize,
    pub no_match_size: usize,
    pub require_field_match: bool,
    pub order_score: bool,
    pub html_encode: bool,
    pub ty: Option<String>,
    pub boundary_scanner: Option<String>,
    pub boundary_chars: Vec<u16>,
    pub boundary_max_scan: usize,
    pub fragment_offset: Option<usize>,
    pub fragmenter: Option<String>,
    pub matched_fields: Vec<String>,
    pub max_analyzed_offset: Option<i64>,
}

fn as_usize(v: Option<&Value>) -> Option<usize> {
    match v? {
        Value::Number(n) => n.as_u64().map(|x| x as usize).or_else(|| n.as_f64().map(|x| x.max(0.0) as usize)),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn as_i64(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|x| x as i64)),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn as_bool(v: Option<&Value>) -> Option<bool> {
    match v? {
        Value::Bool(b) => Some(*b),
        Value::String(s) => Some(s == "true"),
        _ => None,
    }
}

const STYLED_PRE: [&str; 10] = [
    "<em class=\"hlt1\">",
    "<em class=\"hlt2\">",
    "<em class=\"hlt3\">",
    "<em class=\"hlt4\">",
    "<em class=\"hlt5\">",
    "<em class=\"hlt6\">",
    "<em class=\"hlt7\">",
    "<em class=\"hlt8\">",
    "<em class=\"hlt9\">",
    "<em class=\"hlt10\">",
];

impl HighlightOpts {
    pub fn from_json(o: &Map<String, Value>, parent: Option<&HighlightOpts>) -> HighlightOpts {
        let d = parent.cloned().unwrap_or(HighlightOpts {
            pre: vec!["<em>".into()],
            post: vec!["</em>".into()],
            fragment_size: 100,
            number_of_fragments: 5,
            no_match_size: 0,
            require_field_match: true,
            order_score: false,
            html_encode: false,
            ty: None,
            boundary_scanner: None,
            boundary_chars: ".,!? \t\n".encode_utf16().collect(),
            boundary_max_scan: 20,
            fragment_offset: None,
            fragmenter: None,
            matched_fields: Vec::new(),
            max_analyzed_offset: None,
        });
        let list = |k: &str| o.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect::<Vec<_>>());
        let styled = o.get("tags_schema").and_then(Value::as_str) == Some("styled");
        let pre = list("pre_tags").unwrap_or_else(|| if styled { STYLED_PRE.iter().map(|s| s.to_string()).collect() } else { d.pre.clone() });
        let post = list("post_tags").unwrap_or_else(|| if styled { vec!["</em>".into()] } else { d.post.clone() });
        HighlightOpts {
            pre,
            post,
            fragment_size: as_usize(o.get("fragment_size")).unwrap_or(d.fragment_size),
            number_of_fragments: as_usize(o.get("number_of_fragments")).unwrap_or(d.number_of_fragments),
            no_match_size: as_usize(o.get("no_match_size")).unwrap_or(d.no_match_size),
            require_field_match: as_bool(o.get("require_field_match")).unwrap_or(d.require_field_match),
            order_score: o.get("order").and_then(Value::as_str).map(|s| s == "score").unwrap_or(d.order_score),
            html_encode: o.get("encoder").and_then(Value::as_str).map(|s| s == "html").unwrap_or(d.html_encode),
            ty: o.get("type").and_then(Value::as_str).map(str::to_string).or(d.ty),
            boundary_scanner: o.get("boundary_scanner").and_then(Value::as_str).map(|s| s.to_ascii_lowercase()).or(d.boundary_scanner),
            boundary_chars: o.get("boundary_chars").and_then(Value::as_str).map(|s| s.encode_utf16().collect()).unwrap_or(d.boundary_chars),
            boundary_max_scan: as_usize(o.get("boundary_max_scan")).unwrap_or(d.boundary_max_scan),
            fragment_offset: as_usize(o.get("fragment_offset")).or(d.fragment_offset),
            fragmenter: o.get("fragmenter").and_then(Value::as_str).map(str::to_string).or(d.fragmenter),
            matched_fields: list("matched_fields").unwrap_or_default(),
            max_analyzed_offset: as_i64(o.get("max_analyzed_offset")).or(d.max_analyzed_offset),
        }
    }

    fn pre_tag(&self, seq: usize) -> &str {
        if self.pre.is_empty() {
            "<em>"
        } else {
            &self.pre[seq % self.pre.len()]
        }
    }

    fn post_tag(&self, seq: usize) -> &str {
        if self.post.is_empty() {
            "</em>"
        } else {
            &self.post[seq % self.post.len()]
        }
    }
}

pub(crate) fn field_specs(spec: &Map<String, Value>) -> Vec<(String, Map<String, Value>)> {
    match spec.get("fields") {
        Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.as_object().cloned().unwrap_or_default())).collect(),
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_object()).flat_map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_object().cloned().unwrap_or_default()))).collect(),
        _ => Vec::new(),
    }
}

fn parse_err(field: &str, reason: &str) -> EsError {
    EsError::x_content_parse(format!("[highlight] failed to parse field [{field}]")).caused_by(EsError::illegal_argument(reason))
}

/// Request-time validation of a `highlight` object (ES `HighlightBuilder`).
pub fn validate_spec(spec: &Map<String, Value>) -> Result<()> {
    fn check(o: &Map<String, Value>) -> Result<()> {
        if let Some(n) = as_i64(o.get("number_of_fragments")) {
            if !(0..=MAX_FRAGMENTS).contains(&n) {
                return Err(parse_err("number_of_fragments", "[number_of_fragments] must be between [0] and [10000]"));
            }
        }
        if let Some(v) = o.get("max_analyzed_offset") {
            if !v.is_null() {
                let n = as_i64(Some(v)).unwrap_or(0);
                if n == 0 || n < -1 {
                    return Err(parse_err("max_analyzed_offset", "[max_analyzed_offset] must be a positive integer, or -1"));
                }
            }
        }
        for k in ["pre_tags", "post_tags"] {
            if let Some(Value::Array(a)) = o.get(k) {
                if a.is_empty() {
                    return Err(EsError::illegal_argument("pre_tags or post_tags must not be empty"));
                }
            }
        }
        Ok(())
    }
    check(spec)?;
    let global = HighlightOpts::from_json(spec, None);
    for (_, fo) in field_specs(spec) {
        check(&fo)?;
        let o = HighlightOpts::from_json(&fo, Some(&global));
        let ty = o.ty.as_deref().unwrap_or("unified");
        if !o.matched_fields.is_empty() && !o.require_field_match && ty == "unified" {
            return Err(EsError::illegal_argument("Matched fields are not supported when [require_field_match] is set to [false]"));
        }
    }
    Ok(())
}

/// Lucene `SimpleHTMLEncoder` (`encoder: html`) or the identity.
pub(crate) fn encode(s: &str, html: bool) -> String {
    if !html {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("&quot;"),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\'' => out.push_str("&#x27;"),
            '/' => out.push_str("&#x2F;"),
            other => out.push(other),
        }
    }
    out
}

pub(crate) fn u16s(s: &[u16]) -> String {
    String::from_utf16_lossy(s)
}

/// Java `String.trim()` (removes code units `<= ' '`).
pub(crate) fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| (c as u32) <= 0x20)
}

/// Values of `path` in a source document (arrays flattened).
pub fn source_values<'v>(src: &'v Value, path: &str) -> Vec<&'v Value> {
    let mut out = Vec::new();
    fn walk<'v>(v: &'v Value, parts: &[&str], out: &mut Vec<&'v Value>) {
        if parts.is_empty() {
            match v {
                Value::Array(a) => a.iter().for_each(|x| walk(x, parts, out)),
                Value::Null => {}
                other => out.push(other),
            }
            return;
        }
        match v {
            Value::Object(m) => {
                // Support dotted keys at any level.
                for i in (1..=parts.len()).rev() {
                    let key = parts[..i].join(".");
                    if let Some(c) = m.get(&key) {
                        walk(c, &parts[i..], out);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, parts, out)),
            _ => {}
        }
    }
    if !path.contains('.') {
        // A single component needs neither an allocated split vector nor an
        // allocated joined key at every object. Array/null semantics stay in walk.
        fn flat<'v>(v: &'v Value, path: &str, out: &mut Vec<&'v Value>) {
            match v {
                Value::Object(map) => {
                    if let Some(value) = map.get(path) { walk(value, &[], out); }
                }
                Value::Array(values) => values.iter().for_each(|value| flat(value, path, out)),
                _ => {}
            }
        }
        flat(src, path, &mut out);
    } else {
        let parts: Vec<&str> = path.split('.').collect();
        walk(src, &parts, &mut out);
    }
    out
}

/// The joined content of a field: values separated by `sep`.
pub(crate) struct Content {
    pub text: Vec<u16>,
    /// `(start, end)` of each value in `text`.
    pub values: Vec<(usize, usize)>,
}

impl Content {
    fn new(vals: &[String], sep: u16) -> Content {
        let mut text = Vec::new();
        let mut values = Vec::new();
        for (i, v) in vals.iter().enumerate() {
            if i > 0 {
                text.push(sep);
            }
            let s = text.len();
            text.extend(v.encode_utf16());
            values.push((s, text.len()));
        }
        Content { text, values }
    }
}

/// Analyze every value of the content with `analyzer` (+ `normalizer` for
/// keywords); positions continue across values with `gap`.
fn tokens(vals: &[String], content: &Content, analyzer: &Analyzer, normalizer: Option<&Analyzer>, gap: u32) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut next_pos: i64 = 0;
    for (vi, v) in vals.iter().enumerate() {
        // byte offset -> UTF-16 offset within the value
        let mut b2u = vec![0usize; v.len() + 1];
        let mut u = 0usize;
        for (bi, c) in v.char_indices() {
            b2u[bi] = u;
            for k in 1..c.len_utf8() {
                b2u[bi + k] = u;
            }
            u += c.len_utf16();
        }
        b2u[v.len()] = u;
        let base = content.values[vi].0;
        let toks = analyzer.analyze(v);
        let first = next_pos;
        let mut last = first - 1;
        for t in toks {
            let s = (t.offset_from as usize).min(v.len());
            let e = (t.offset_to as usize).min(v.len());
            let text = match normalizer {
                Some(n) => n.analyze(&t.text).into_iter().map(|x| x.text).collect::<String>(),
                None => t.text,
            };
            let pos = first + t.position as i64;
            last = last.max(pos);
            out.push(Tok { text, pos: pos.max(0) as u32, start: base + b2u[s], end: base + b2u[e].max(b2u[s]) });
        }
        next_pos = last + 1 + gap as i64;
    }
    out
}

pub(crate) fn analyzer_for(f: &ResolvedField) -> Arc<Analyzer> {
    if f.ty.is_keyword_like() {
        return Arc::new(Analyzer::keyword());
    }
    match &f.index_analyzer {
        Some(crate::analysis::IndexAnalyzer::Custom(a)) => a.clone(),
        Some(crate::analysis::IndexAnalyzer::Builtin(n)) => Arc::new(Analyzer::by_name(n).unwrap_or_else(Analyzer::standard)),
        None => f.search_analyzer.clone().unwrap_or_else(|| Arc::new(Analyzer::standard())),
    }
}

fn max_offset_error(len: usize, field: &str, doc: u32, index: &str, max: usize) -> EsError {
    EsError::illegal_argument(format!(
        "The length [{len}] of field [{field}] in doc[{doc}]/index[{index}] exceeds the [index.highlight.max_analyzed_offset] limit [{max}]. To avoid this error, set the query parameter [max_analyzed_offset] to a value less than index setting [{max}] and this will tolerate long field values by truncating them."
    ))
}

/// Build the `highlight` object of one hit (errors are dropped).
pub fn highlight_hit(spec: &Map<String, Value>, src: &Value, lookup: &FieldLookup, terms: &QueryTerms) -> Map<String, Value> {
    try_highlight_hit(spec, src, lookup, terms, 0).unwrap_or_default()
}

/// Build the `highlight` object of one hit; `doc` is the hit's segment doc
/// (error messages only).
pub fn try_highlight_hit(spec: &Map<String, Value>, src: &Value, lookup: &FieldLookup, terms: &QueryTerms, doc: u32) -> Result<Map<String, Value>> {
    let global = HighlightOpts::from_json(spec, None);
    let mut out = Map::new();
    for (pattern, fo) in field_specs(spec) {
        let o = HighlightOpts::from_json(&fo, Some(&global));
        let wildcard = pattern.contains('*');
        let names = if wildcard { lookup.matching(&pattern) } else { vec![pattern.clone()] };
        let q = terms.query_for(&pattern);
        for name in names {
            if out.contains_key(&name) {
                continue;
            }
            let Some(f) = lookup.get(&name) else { continue };
            if !(f.ty.is_text() || f.ty.is_keyword_like()) {
                continue;
            }
            // semantic_text fields and the `semantic` type: the semantic highlighter.
            if f.declared.is_semantic() || crate::inference::highlight::handles(lookup, &name, &o) {
                continue;
            }
            if wildcard && f.alias_path.is_some() {
                continue;
            }
            if let Some(frags) = highlight_field(&name, f, &o, src, lookup, terms, q, doc)? {
                out.insert(name, Value::Array(frags.into_iter().map(Value::String).collect()));
            }
        }
    }
    Ok(out)
}

/// The `highlight` of one nested inner hit: `child_src` (the nested
/// object's source) is highlighted as if it were the document's only value
/// at `nested_path`.
pub fn inner_hit(spec: &Map<String, Value>, q: &Query, svc: &crate::index::IndexService, lookup: &FieldLookup, nested_path: &str, child_src: Option<Value>, doc: u32) -> Result<Map<String, Value>> {
    let Some(mut v) = child_src else { return Ok(Map::new()) };
    for part in nested_path.rsplit('.') {
        if part.is_empty() {
            continue;
        }
        let mut m = Map::new();
        m.insert(part.to_string(), v);
        v = Value::Object(m);
    }
    let terms = QueryTerms::for_index(q, spec, svc, &|_| Err(EsError::illegal_argument("[highlight_query] is not supported in inner_hits")))?;
    try_highlight_hit(spec, &v, lookup, &terms, doc)
}

#[allow(clippy::too_many_arguments)]
fn highlight_field(name: &str, f: &ResolvedField, o: &HighlightOpts, src: &Value, lookup: &FieldLookup, terms: &QueryTerms, q: &Query, doc: u32) -> Result<Option<Vec<String>>> {
    let Some(id) = f.core_id else { return Ok(None) };
    let src_path = if f.is_multi_field { name.rsplit_once('.').map(|x| x.0.to_string()).unwrap_or(name.to_string()) } else { name.to_string() };
    let mut vals: Vec<String> = Vec::new();
    for v in source_values(src, &src_path) {
        let s = match v {
            Value::String(s) => s.clone(),
            Value::Object(_) | Value::Array(_) => continue,
            other => other.to_string(),
        };
        if f.ty.is_keyword_like() {
            if let Some(ia) = f.ignore_above {
                if s.chars().count() > ia {
                    continue;
                }
            }
        }
        vals.push(s);
    }
    if vals.is_empty() {
        return Ok(None);
    }
    let ty = o.ty.as_deref().unwrap_or("unified");
    let ty = match ty {
        "highlighter" => "plain",
        "fast-vector-highlighter" => "fvh",
        "postings" | "postings-highlighter" => "unified",
        t => t,
    };
    let flags = terms.flags.get(name).copied().unwrap_or_default();
    let tokenized = f.ty.is_text();
    // max_analyzed_offset
    let qmax: Option<usize> = match o.max_analyzed_offset {
        Some(-1) => Some(terms.index_max_offset),
        Some(n) if n > 0 => Some(n as usize),
        _ => None,
    };
    let analysis_check = |len: usize| -> Result<()> {
        if qmax.map(|m| m > terms.index_max_offset).unwrap_or(true) && len > terms.index_max_offset {
            return Err(max_offset_error(len, name, doc, &terms.index, terms.index_max_offset));
        }
        Ok(())
    };
    let weighted = ty == "unified" && terms.weight_matches_setting && !matches::disables_weight_matches(q);
    let sep = if ty == "unified" { MULTIVAL_SEP } else { b' ' as u16 };
    let content = Content::new(&vals, sep);
    // token streams: the field itself (+ matched_fields)
    let mut stream_fields: Vec<(Option<u16>, Arc<Analyzer>, Option<Arc<Analyzer>>)> = vec![(if o.require_field_match { Some(id) } else { None }, analyzer_for(f), f.normalizer.clone())];
    if ty != "plain" && o.require_field_match {
        for mf in &o.matched_fields {
            if mf == name {
                continue;
            }
            if let Some(m) = lookup.get(mf) {
                if let Some(mid) = m.core_id {
                    stream_fields.push((Some(mid), analyzer_for(m), m.normalizer.clone()));
                }
            }
        }
    }
    let gap = f.position_increment_gap;
    let limit = qmax;
    let mut tok_lists: Vec<Vec<Tok>> = Vec::new();
    for (_, a, n) in &stream_fields {
        let mut t = tokens(&vals, &content, a, n.as_deref(), gap);
        if let Some(m) = limit {
            if ty == "plain" {
                // LimitTokenOffsetFilter per value
                t.retain(|x| {
                    let vs = content.values.iter().rev().find(|v| v.0 <= x.start).map(|v| v.0).unwrap_or(0);
                    x.start - vs <= m
                });
            } else {
                t.retain(|x| x.start <= m);
            }
        }
        tok_lists.push(t);
    }
    match ty {
        "plain" => {
            for v in &content.values {
                analysis_check(v.1 - v.0)?;
            }
        }
        "fvh" => {
            if !flags.tv_offsets {
                return Err(EsError::illegal_argument(format!("the field [{name}] should be indexed with term vector with position offsets to be used with fast vector highlighter")));
            }
            if terms.synthetic_source && vals.len() > 1 {
                return Err(EsError::illegal_argument(format!(
                    "The fast vector highlighter doesn't support loading multi-valued fields from _source in index [{}] because _source can reorder field values",
                    terms.index
                )));
            }
        }
        _ => {
            if !flags.postings_offsets && !flags.tv_offsets && tokenized {
                analysis_check(content.text.len())?;
            }
        }
    }
    let streams: Vec<Stream> = stream_fields.iter().zip(&tok_lists).map(|((fid, _, _), t)| Stream { field: *fid, toks: t }).collect();
    // plain / fvh need phrase groups (parts); unified splits them unless weighted
    let grouped = if ty == "unified" { weighted } else { true };
    let ms: Vec<Match> = matches::find(q, &streams, grouped);
    let frags = match ty {
        "plain" => plain::highlight(&content, &tok_lists[0], &ms, o),
        "fvh" => fvh::highlight(&content, &ms, o)?,
        _ => unified::highlight(&content, &ms, o, tokenized),
    };
    Ok(if frags.is_empty() { None } else { Some(frags) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_paths() {
        let v = serde_json::json!({"a": {"b": ["x", "y"]}, "c.d": 1});
        assert_eq!(source_values(&v, "a.b").len(), 2);
        assert_eq!(source_values(&v, "c.d"), vec![&serde_json::json!(1)]);
    }

    #[test]
    fn flat_source_paths_keep_array_and_null_semantics() {
        let source = serde_json::json!([
            {"field": [1, null, [2, 3]], "": 4},
            [{"field": {"nested": 5}}, {"other": 6}],
            null,
            {"field": false}
        ]);
        assert_eq!(source_values(&source, "field").into_iter().cloned().collect::<Vec<_>>(),
            vec![serde_json::json!(1), serde_json::json!(2), serde_json::json!(3), serde_json::json!({"nested": 5}), serde_json::json!(false)]);
        assert_eq!(source_values(&source, "").into_iter().cloned().collect::<Vec<_>>(), vec![serde_json::json!(4)]);
        assert!(source_values(&source, "absent").is_empty());
    }

    #[test]
    fn validation() {
        let m = |v: Value| v.as_object().cloned().unwrap();
        assert!(validate_spec(&m(serde_json::json!({"number_of_fragments": 10001, "fields": {"a": {}}}))).is_err());
        assert!(validate_spec(&m(serde_json::json!({"number_of_fragments": 10000, "fields": {"a": {}}}))).is_ok());
        assert!(validate_spec(&m(serde_json::json!({"max_analyzed_offset": 0, "fields": {"a": {}}}))).is_err());
        assert!(validate_spec(&m(serde_json::json!({"max_analyzed_offset": -1, "fields": {"a": {}}}))).is_ok());
        assert!(validate_spec(&m(serde_json::json!({"pre_tags": [], "fields": {"a": {}}}))).is_err());
    }
}
