//! Term vectors API (`_termvectors`, `_mtermvectors`).
//!
//! Term vectors are generated on the fly by re-analyzing the document's
//! `_source` (or an artificial `doc`) with the field's index analyzer, like
//! ES does for fields without stored term vectors. Positions follow Lucene's
//! inversion (`position_increment_gap` between values) and offsets are UTF-16
//! char offsets with an offset gap of 1 between values. Term / field
//! statistics come from the shard's term dictionaries (deleted docs
//! included, like Lucene).

use crate::analysis::IndexAnalyzer;
use crate::error::{EsError, Result};
use crate::highlight::source_values;
use crate::index::IndexService;
use crate::mapping::{FieldLookup, ResolvedField};
use crate::node::Node;
use rustic_core::analysis::Analyzer;
use rustic_core::query::LeafReader;
use rustic_core::types::Term;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// Options of one term vectors request (URL params merged with the body).
#[derive(Clone, Debug)]
pub struct TvRequest {
    pub fields: Option<Vec<String>>,
    pub offsets: bool,
    pub positions: bool,
    pub payloads: bool,
    pub term_statistics: bool,
    pub field_statistics: bool,
    pub routing: Option<String>,
    pub version: Option<i64>,
    pub version_type: Option<String>,
    pub realtime: bool,
    pub doc: Option<Value>,
    pub per_field_analyzer: HashMap<String, String>,
    pub filter: Option<TvFilter>,
}

impl Default for TvRequest {
    fn default() -> Self {
        TvRequest {
            fields: None,
            offsets: true,
            positions: true,
            payloads: true,
            term_statistics: false,
            field_statistics: true,
            routing: None,
            version: None,
            version_type: None,
            realtime: true,
            doc: None,
            per_field_analyzer: HashMap::new(),
            filter: None,
        }
    }
}

/// `filter` section: keep the best tf-idf scored terms.
#[derive(Clone, Debug)]
pub struct TvFilter {
    pub max_num_terms: usize,
    pub min_term_freq: u32,
    pub max_term_freq: u32,
    pub min_doc_freq: u64,
    pub max_doc_freq: u64,
    pub min_word_length: usize,
    pub max_word_length: usize,
}

fn as_bool(_k: &str, v: &Value) -> Result<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "true" || s.is_empty() => Ok(true),
        Value::String(s) if s == "false" => Ok(false),
        _ => Err(EsError::illegal_argument(format!("Failed to parse value [{}] as only [true] or [false] are allowed.", crate::query::dsl::scalar_string(v)))),
    }
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a.iter().map(crate::query::dsl::scalar_string).collect(),
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        _ => vec![],
    }
}

fn num<T: std::str::FromStr>(k: &str, v: &Value) -> Result<T> {
    let s = crate::query::dsl::scalar_string(v);
    s.parse().map_err(|_| EsError::illegal_argument(format!("failed to parse [{k}] value [{s}]")))
}

impl TvRequest {
    /// Apply one option (URL param or body key). Returns false for unknown keys.
    pub fn set(&mut self, k: &str, v: &Value) -> Result<bool> {
        match k {
            "fields" => self.fields = Some(str_list(v)),
            "offsets" => self.offsets = as_bool(k, v)?,
            "positions" => self.positions = as_bool(k, v)?,
            "payloads" => self.payloads = as_bool(k, v)?,
            "term_statistics" => self.term_statistics = as_bool(k, v)?,
            "field_statistics" => self.field_statistics = as_bool(k, v)?,
            "routing" => self.routing = Some(crate::query::dsl::scalar_string(v)),
            "version" => self.version = Some(num(k, v)?),
            "version_type" => self.version_type = Some(crate::query::dsl::scalar_string(v)),
            "realtime" => self.realtime = as_bool(k, v)?,
            "preference" => {}
            "doc" => self.doc = Some(v.clone()),
            "per_field_analyzer" => {
                let o = v.as_object().ok_or_else(|| EsError::parsing("[per_field_analyzer] must be an object"))?;
                for (f, a) in o {
                    self.per_field_analyzer.insert(f.clone(), crate::query::dsl::scalar_string(a));
                }
            }
            "filter" => {
                let o = v.as_object().ok_or_else(|| EsError::parsing("[filter] must be an object"))?;
                let mut f = TvFilter {
                    max_num_terms: 25,
                    min_term_freq: 0,
                    max_term_freq: u32::MAX,
                    min_doc_freq: 0,
                    max_doc_freq: u64::MAX,
                    min_word_length: 0,
                    max_word_length: 0,
                };
                for (fk, fv) in o {
                    match fk.as_str() {
                        "max_num_terms" => f.max_num_terms = num(fk, fv)?,
                        "min_term_freq" => f.min_term_freq = num(fk, fv)?,
                        "max_term_freq" => f.max_term_freq = num(fk, fv)?,
                        "min_doc_freq" => f.min_doc_freq = num(fk, fv)?,
                        "max_doc_freq" => f.max_doc_freq = num(fk, fv)?,
                        "min_word_length" => f.min_word_length = num(fk, fv)?,
                        "max_word_length" => f.max_word_length = num(fk, fv)?,
                        other => return Err(EsError::new(400, "parse_exception", format!("failed to parse term vectors request. the field [{other}] is not valid for filter parameter for term vector request"))),
                    }
                }
                self.filter = Some(f);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// From URL params (unknown params ignored) then a request body.
    pub fn from_parts(params: &HashMap<String, String>, body: &Value) -> Result<TvRequest> {
        let mut r = TvRequest::default();
        for (k, v) in params {
            r.set(k, &Value::String(v.clone()))?;
        }
        if let Value::Object(o) = body {
            for (k, v) in o {
                if k == "fields" && !v.is_array() {
                    return Err(EsError::new(400, "parse_exception", "failed to parse term vectors request. field [fields] must be an array"));
                }
                if k == "_id" || k == "_index" || k == "_routing" || k == "_version" || k == "_version_type" {
                    continue;
                }
                if !r.set(k, v)? {
                    return Err(EsError::new(400, "parse_exception", format!("failed to parse term vectors request. unknown field [{k}]")));
                }
            }
        }
        Ok(r)
    }
}

/// Resolve a single concrete index for a read op (alias must point to one index).
pub fn single_index(node: &Node, name: &str) -> Result<Arc<IndexService>> {
    if let Some(s) = node.get_index(name) {
        return Ok(s);
    }
    let mut cands: Vec<Arc<IndexService>> = node.all_indices().into_iter().filter(|s| s.aliases().contains_key(name)).collect();
    match cands.len() {
        0 => node.write_index(name, false),
        1 => Ok(cands.remove(0)),
        _ => {
            let names: Vec<String> = cands.iter().map(|s| s.name.clone()).collect();
            Err(EsError::illegal_argument(format!(
                "Alias [{name}] has more than one index associated with it [[{}]], can't execute a single index op",
                names.join(", ")
            )))
        }
    }
}


/// The mapping JSON of field `name` (dotted path, multi-fields included).
fn field_mapping_json(mappings: &Value, name: &str) -> Option<Value> {
    let parts: Vec<&str> = name.split('.').collect();
    fn walk(node: &Value, parts: &[&str]) -> Option<Value> {
        if parts.is_empty() {
            return Some(node.clone());
        }
        for i in (1..=parts.len()).rev() {
            let key = parts[..i].join(".");
            if let Some(c) = node.get("properties").and_then(|p| p.get(&key)) {
                if let Some(r) = walk(c, &parts[i..]) {
                    return Some(r);
                }
            }
            if let Some(c) = node.get("fields").and_then(|p| p.get(&key)) {
                if let Some(r) = walk(c, &parts[i..]) {
                    return Some(r);
                }
            }
        }
        None
    }
    walk(mappings, &parts)
}

fn index_analyzer(f: &ResolvedField) -> Arc<Analyzer> {
    if f.ty.is_keyword_like() {
        return f.normalizer.clone().unwrap_or_else(|| Arc::new(Analyzer::keyword()));
    }
    match &f.index_analyzer {
        Some(IndexAnalyzer::Custom(a)) => a.clone(),
        Some(IndexAnalyzer::Builtin(n)) => Arc::new(Analyzer::by_name(n).unwrap_or_else(Analyzer::standard)),
        None => f.search_analyzer.clone().unwrap_or_else(|| Arc::new(Analyzer::standard())),
    }
}

struct Tok {
    pos: u32,
    start: u32,
    end: u32,
}

/// Inverted view of one field: term → tokens.
fn invert(values: &[String], analyzer: &Analyzer, keyword: bool, ignore_above: Option<usize>, gap: u32) -> BTreeMap<String, Vec<Tok>> {
    let mut out: BTreeMap<String, Vec<Tok>> = BTreeMap::new();
    let mut pos_base: i64 = 0;
    let mut off_base: u32 = 0;
    let mut first = true;
    for v in values {
        let u16len = v.encode_utf16().count() as u32;
        if keyword {
            if ignore_above.map(|m| v.chars().count() > m).unwrap_or(false) {
                continue;
            }
            let toks = analyzer.analyze(v);
            let t: String = if toks.is_empty() { v.clone() } else { toks.into_iter().map(|t| t.text).collect() };
            if !first {
                pos_base += gap as i64 + 1;
            }
            out.entry(t).or_default().push(Tok { pos: pos_base as u32, start: off_base, end: off_base + u16len });
            first = false;
            off_base += u16len + 1;
            continue;
        }
        let toks = analyzer.analyze(v);
        // byte offset → utf16 offset
        let mut map: Vec<u32> = vec![0; v.len() + 1];
        let mut acc = 0u32;
        let mut bi = 0usize;
        for ch in v.chars() {
            for k in 0..ch.len_utf8() {
                map[bi + k] = acc;
            }
            bi += ch.len_utf8();
            acc += ch.len_utf16() as u32;
        }
        map[v.len()] = acc;
        let base = if first { 0 } else { pos_base + gap as i64 + 1 };
        let mut last = base - 1;
        for t in &toks {
            let p = base + t.position as i64;
            last = last.max(p);
            let s = map[(t.offset_from as usize).min(v.len())];
            let e = map[(t.offset_to as usize).min(v.len())];
            out.entry(t.text.clone()).or_default().push(Tok { pos: p as u32, start: off_base + s, end: off_base + e });
        }
        if !toks.is_empty() || !first {
            pos_base = last.max(base - 1);
        }
        if !toks.is_empty() {
            first = false;
        }
        off_base += u16len + 1;
    }
    out
}

struct FieldStatsTv {
    sum_doc_freq: u64,
    doc_count: u64,
    sum_ttf: u64,
}

fn field_stats(leaves: &[LeafReader<'_>], id: u16) -> FieldStatsTv {
    let mut s = FieldStatsTv { sum_doc_freq: 0, doc_count: 0, sum_ttf: 0 };
    for l in leaves {
        let terms = l.term_range(id, None, None);
        let mut docs: HashSet<u32> = HashSet::new();
        for (t, loc) in &terms {
            s.sum_doc_freq += loc.doc_freq as u64;
            s.sum_ttf += loc.total_term_freq;
            let k = Term::from_key(t);
            let term = if k.field == id && t.len() >= 2 && t[..2] == id.to_le_bytes() { k } else { Term::new(id, t.clone()) };
            if let Some(mut p) = l.postings_with(&term, loc) {
                while let Some(d) = p.next() {
                    docs.insert(d);
                }
            }
        }
        s.doc_count += docs.len() as u64;
    }
    s
}

fn term_stats(leaves: &[LeafReader<'_>], id: u16, term: &str) -> (u64, u64) {
    let t = Term::new(id, term.as_bytes().to_vec());
    leaves.iter().fold((0, 0), |(df, ttf), l| (df + l.doc_freq(&t) as u64, ttf + l.total_term_freq(&t)))
}

fn tv_flags(mapping: Option<&Value>) -> Option<(bool, bool)> {
    let tv = mapping.and_then(|m| m.get("term_vector")).and_then(Value::as_str)?;
    match tv {
        "no" => None,
        "yes" => Some((false, false)),
        "with_positions" | "with_positions_payloads" => Some((true, false)),
        "with_offsets" => Some((false, true)),
        _ => Some((true, true)),
    }
}

/// Render the term vectors of the given field values.
/// Term vectors of `source` with the term / field statistics of shard copy `svc` (ES reports per-shard statistics).
fn render(svc: &crate::index::IndexShard, lookup: &FieldLookup, source: &Value, req: &TvRequest, artificial: bool) -> Result<Map<String, Value>> {
    let mappings = svc.meta().mappings.to_json();
    let registry = svc.registry();
    // Candidate fields.
    let mut names: Vec<String> = Vec::new();
    match &req.fields {
        Some(list) => {
            for p in list {
                for n in lookup.matching(p) {
                    if !names.contains(&n) {
                        names.push(n);
                    }
                }
                if !p.contains('*') && lookup.get(p).is_none() && !names.contains(p) && artificial {
                    names.push(p.clone());
                }
            }
        }
        None => {
            let mut all: Vec<String> = lookup.matching("*");
            all.sort();
            for n in all {
                let fm = field_mapping_json(&mappings, &n);
                if artificial || tv_flags(fm.as_ref()).is_some() {
                    names.push(n);
                }
            }
            if artificial {
                // unmapped string fields of an artificial doc are dynamic text
                let mut extra = Vec::new();
                collect_paths(source, "", &mut extra);
                for p in extra {
                    if lookup.get(&p).is_none() && !names.contains(&p) {
                        names.push(p);
                    }
                }
            }
        }
    }
    names.sort();
    // Field / term statistics over the published reader view (no shard lock).
    let view = svc.shard.reader_view();
    let leaves = view.leaves();
    let mut rendered: Vec<(String, Value)> = Vec::new();
    for name in names {
        let f = lookup.get(&name).cloned();
        let (is_kw, id, gap, ignore_above) = match &f {
            Some(f) => {
                if !(f.ty.is_text() || f.ty.is_keyword_like()) {
                    continue;
                }
                (f.ty.is_keyword_like(), f.core_id, f.position_increment_gap, f.ignore_above)
            }
            None => (false, None, 100, None),
        };
        let fm = field_mapping_json(&mappings, &name);
        let src_path = match &f {
            Some(f) if f.is_multi_field => name.rsplit_once('.').map(|x| x.0.to_string()).unwrap_or(name.clone()),
            _ => name.clone(),
        };
        let values: Vec<String> = source_values(source, &src_path)
            .into_iter()
            .filter(|v| !v.is_object())
            .map(|v| match v {
                Value::String(s) => s.clone(),
                o => o.to_string(),
            })
            .collect();
        if values.is_empty() {
            continue;
        }
        let analyzer: Arc<Analyzer> = match req.per_field_analyzer.get(&name) {
            Some(a) => registry.analyzer(a).or_else(|| Analyzer::by_name(a).map(Arc::new)).ok_or_else(|| EsError::illegal_argument(format!("failed to find analyzer [{a}]")))?,
            None => match &f {
                Some(f) => index_analyzer(f),
                None => Arc::new(Analyzer::standard()),
            },
        };
        let inv = invert(&values, &analyzer, is_kw && !req.per_field_analyzer.contains_key(&name), ignore_above, gap);
        if inv.is_empty() {
            continue;
        }
        // Stored term vectors restrict positions/offsets; generated ones follow the request.
        let (has_pos, has_off) = match tv_flags(fm.as_ref()) {
            Some((p, o)) if !artificial && req.per_field_analyzer.get(&name).is_none() => (p && req.positions, o && req.offsets),
            _ => (req.positions, req.offsets),
        };
        let mut fobj = Map::new();
        if req.field_statistics {
            if let Some(id) = id {
                let s = field_stats(&leaves, id);
                fobj.insert("field_statistics".into(), json!({"sum_doc_freq": s.sum_doc_freq, "doc_count": s.doc_count, "sum_ttf": s.sum_ttf}));
            }
        }
        // Optional filter scoring.
        let mut keep: Option<HashMap<String, f32>> = None;
        if let Some(flt) = &req.filter {
            let num_docs = id.map(|id| field_stats(&leaves, id).doc_count).unwrap_or(0);
            let mut scored: Vec<(String, f32)> = Vec::new();
            for (t, toks) in &inv {
                let freq = toks.len() as u32;
                let len = t.chars().count();
                if (flt.min_word_length > 0 && len < flt.min_word_length)
                    || (flt.max_word_length > 0 && len > flt.max_word_length)
                    || (flt.min_term_freq > 0 && freq < flt.min_term_freq)
                    || freq > flt.max_term_freq
                {
                    continue;
                }
                let df = id.map(|id| term_stats(&leaves, id, t).0).unwrap_or(0);
                if df < flt.min_doc_freq || df > flt.max_doc_freq {
                    continue;
                }
                let idf = ((num_docs as f64 + 1.0) / (df as f64 + 1.0)).ln() + 1.0;
                scored.push((t.clone(), (freq as f64 * idf) as f32));
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(flt.max_num_terms);
            keep = Some(scored.into_iter().collect());
        }
        let mut terms = Map::new();
        for (t, toks) in &inv {
            let score = match &keep {
                Some(k) => match k.get(t) {
                    Some(s) => Some(*s),
                    None => continue,
                },
                None => None,
            };
            let mut tobj = Map::new();
            if req.term_statistics {
                if let Some(id) = id {
                    let (df, ttf) = term_stats(&leaves, id, t);
                    if df > 0 {
                        tobj.insert("doc_freq".into(), json!(df));
                        tobj.insert("ttf".into(), json!(ttf));
                    }
                }
            }
            tobj.insert("term_freq".into(), json!(toks.len()));
            if has_pos || has_off {
                let arr: Vec<Value> = toks
                    .iter()
                    .map(|k| {
                        let mut o = Map::new();
                        if has_pos {
                            o.insert("position".into(), json!(k.pos));
                        }
                        if has_off {
                            o.insert("start_offset".into(), json!(k.start));
                            o.insert("end_offset".into(), json!(k.end));
                        }
                        Value::Object(o)
                    })
                    .collect();
                tobj.insert("tokens".into(), Value::Array(arr));
            }
            if let Some(s) = score {
                tobj.insert("score".into(), crate::suggest::jfloat(s));
            }
            terms.insert(t.clone(), Value::Object(tobj));
        }
        fobj.insert("terms".into(), Value::Object(terms));
        rendered.push((name, Value::Object(fobj)));
    }
    // ES iterates a `HashMap` of the selected fields (sorted insertion).
    let cap = crate::fieldtypes::suggest::java_expected_capacity(rendered.len());
    let mut out = Map::new();
    for (k, v) in crate::fieldtypes::suggest::java_order(rendered, cap) {
        out.insert(k, v);
    }
    Ok(out)
}

fn collect_paths(v: &Value, prefix: &str, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, c) in m {
                let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                collect_paths(c, &p, out);
            }
        }
        Value::Array(a) => {
            for c in a {
                collect_paths(c, prefix, out);
            }
        }
        Value::String(_) => {
            if !prefix.is_empty() && !out.contains(&prefix.to_string()) {
                out.push(prefix.to_string());
            }
        }
        _ => {}
    }
}

/// `GET /{index}/_termvectors[/{id}]`.
pub fn term_vectors(node: &Node, index: &str, id: Option<&str>, req: &TvRequest) -> Result<Value> {
    let start = std::time::Instant::now();
    let svc = single_index(node, index)?;
    svc.check_read_block()?;
    let lookup = svc.lookup();
    let took = |s: &std::time::Instant| s.elapsed().as_millis() as u64;
    // The shard serving the request: the document's (or the routing's;
    // an artificial document without routing uses shard 0).
    let shard_for = |id: Option<&str>| -> Result<std::sync::Arc<crate::index::IndexShard>> {
        let n = match (id, req.routing.as_deref()) {
            (Some(id), r) => svc.route(id, r)?,
            (None, Some(r)) => svc.route(r, Some(r))?,
            (None, None) => 0,
        };
        svc.shard(n).ok_or_else(|| EsError::shard_not_found(&svc.name, &svc.uuid, n))
    };
    if let Some(doc) = &req.doc {
        if !doc.is_object() {
            return Err(EsError::new(500, "exception", "failed to execute term vector request").caused_by(EsError::document_parsing("Malformed content, must start with an object")));
        }
        let sh = shard_for(None)?;
        let tv = render(&sh, &lookup, doc, req, true)?;
        return Ok(json!({"_index": svc.name, "_version": 0, "found": true, "took": took(&start), "term_vectors": tv}));
    }
    let id = id.ok_or_else(|| EsError::action_validation(&["id or doc is missing"]))?;
    if req.routing.is_none() && svc.meta().mappings.routing_required() {
        return Err(EsError::new(400, "routing_missing_exception", format!("routing is required for [{}]/[{id}]", svc.name)).with("index", svc.name.clone()));
    }
    let not_found = |svc: &IndexService| json!({"_index": svc.name, "_id": id, "_version": 0, "found": false, "took": took(&start)});
    let sh = shard_for(Some(id))?;
    svc.check_open()?;
    let Some(g) = sh.shard.get_mode(id, req.realtime)? else { return Ok(not_found(&svc)) };
    if let Some(v) = req.version {
        let vt = req.version_type.as_deref().unwrap_or("internal");
        let ok = match vt {
            "external" => (g.version as i64) <= v,
            "external_gte" => (g.version as i64) <= v,
            _ => g.version as i64 == v,
        };
        if !ok && v != -3 {
            return Err(EsError::version_conflict(&svc.name, &svc.uuid, id, format!("version conflict, current version [{}] is different than the one provided [{v}]", g.version)));
        }
    }
    let src: Value = match g.source.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => serde_json::from_str(s)?,
        None => Value::Object(Map::new()),
    };
    let tv = render(&sh, &lookup, &src, req, false)?;
    for f in tv.keys() {
        crate::field_usage::bump(&svc.uuid, f, "term_vectors");
    }
    let mut m = Map::new();
    m.insert("_index".into(), json!(svc.name));
    m.insert("_id".into(), json!(id));
    m.insert("_version".into(), json!(g.version));
    m.insert("found".into(), json!(true));
    m.insert("took".into(), json!(took(&start)));
    m.insert("term_vectors".into(), Value::Object(tv));
    Ok(Value::Object(m))
}

fn item_error(e: &EsError) -> Value {
    let mut v = e.error_object(false);
    if let Some(o) = v.as_object_mut() {
        o.remove("root_cause");
    }
    v
}

/// `_mtermvectors`: `docs` (per-doc `_index`/`_id`/options) or `ids`.
pub fn mterm_vectors(node: &Node, index: Option<&str>, params: &HashMap<String, String>, body: &Value) -> Result<Value> {
    // Request-level options (URL params, except the multi-request ones).
    let mut base = TvRequest::default();
    let mut ids_param: Option<Vec<String>> = None;
    for (k, v) in params {
        if k == "ids" {
            ids_param = Some(str_list(&Value::String(v.clone())));
            continue;
        }
        base.set(k, &Value::String(v.clone()))?;
    }
    let mut items: Vec<(Option<String>, Option<String>, TvRequest)> = Vec::new();
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if !matches!(k.as_str(), "docs" | "ids" | "parameters") || (k == "docs" && !v.is_array()) {
                return Err(EsError::new(400, "parse_exception", format!("_mtermvectors: Parameter [{k}] not supported")));
            }
        }
    }
    if body.as_object().is_some_and(|o| !o.contains_key("docs") && !o.contains_key("ids")) && !params.contains_key("ids") {
        return Err(EsError::action_validation(&["multi term vectors: no documents requested"]));
    }
    let docs = body.get("docs").and_then(Value::as_array);
    let ids = body.get("ids").and_then(Value::as_array).map(|a| a.iter().map(crate::query::dsl::scalar_string).collect::<Vec<_>>()).or(ids_param);
    if let Some(docs) = docs {
        for d in docs {
            let o = d.as_object().ok_or_else(|| EsError::parsing("docs array element should include an object"))?;
            let mut r = base.clone();
            let mut idx = index.map(str::to_string);
            let mut id = None;
            for (k, v) in o {
                match k.as_str() {
                    "_index" | "index" => idx = Some(crate::query::dsl::scalar_string(v)),
                    "_id" | "id" => id = Some(crate::query::dsl::scalar_string(v)),
                    "_routing" => r.routing = Some(crate::query::dsl::scalar_string(v)),
                    "_version" => r.version = Some(num(k, v)?),
                    _ => {
                        if !r.set(k, v)? {
                            return Err(EsError::new(400, "parse_exception", format!("failed to parse term vectors request. unknown field [{k}]")));
                        }
                    }
                }
            }
            items.push((idx, id, r));
        }
    } else if let Some(ids) = ids {
        let mut r = base.clone();
        if let Some(p) = body.get("parameters").and_then(Value::as_object) {
            for (k, v) in p {
                if !r.set(k, v)? {
                    return Err(EsError::parsing(format!("failed to parse term vectors request. unknown field [{k}]")));
                }
            }
        }
        for id in ids {
            items.push((index.map(str::to_string), Some(id), r.clone()));
        }
    } else {
        return Err(EsError::action_validation(&["multi term vectors: no documents requested"]));
    }
    let mut out = Vec::new();
    for (i, (idx, id, r)) in items.into_iter().enumerate() {
        let Some(idx) = idx else {
            return Err(EsError::action_validation(&[&format!("index is missing for doc {i}")]));
        };
        let res = term_vectors(node, &idx, id.as_deref(), &r);
        match res {
            Ok(v) => out.push(v),
            Err(e) => {
                let concrete = single_index(node, &idx).map(|s| s.name.clone()).unwrap_or(idx.clone());
                let mut m = Map::new();
                m.insert("_index".into(), json!(concrete));
                if let Some(id) = id {
                    m.insert("_id".into(), json!(id));
                }
                m.insert("error".into(), item_error(&e));
                out.push(Value::Object(m));
            }
        }
    }
    Ok(json!({ "docs": out }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_positions_and_offsets() {
        let a = Analyzer::standard();
        let inv = invert(&["The quick brown fox is brown.".into()], &a, false, None, 100);
        let b = &inv["brown"];
        assert_eq!(b.len(), 2);
        assert_eq!((b[0].pos, b[0].start, b[0].end), (2, 10, 15));
        assert_eq!((b[1].pos, b[1].start, b[1].end), (5, 23, 28));
        let inv = invert(&["a b".into(), "c".into()], &a, false, None, 100);
        assert_eq!(inv["c"][0].pos, 1 + 100 + 1);
        assert_eq!(inv["c"][0].start, 4);
    }
}
