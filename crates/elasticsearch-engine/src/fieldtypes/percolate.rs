//! `percolator` field + `percolate` query: stored queries are parsed when
//! indexed (validation), and at search time every candidate's query runs
//! against an in-memory index holding the given document(s) (Lucene
//! `MemoryIndex` per candidate).

use crate::error::{EsError, Result};
use crate::mapping::{EsType, FieldLookup, ID_FIELD};
use crate::query::dsl::{parse_query, QueryContext};
use rustic_core::fieldtypes::filter::FilterQuery;
use rustic_core::query::{CustomQuery, LeafReader, Query, Scorer, SearchContext, DocSet, TERMINATED};
use rustic_core::types::{DocId, IndexSettings};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::Mutex;

/// Per-request matches, retained from verification for the fetch phase.
type Verify = Arc<dyn Fn(&str) -> Option<f32> + Send + Sync>;

struct PercolateQuery {
    filter: FilterQuery,
    candidate: Query,
    version_field: Option<u16>,
    marked_candidate: Option<Query>,
    verify: Verify,
    field: u16,
    slot_field: String,
    slots: Arc<Mutex<HashMap<String, Vec<u32>>>>,
    documents: Vec<Value>,
    memory: Arc<rustic_core::Index>,
    lookup: Arc<FieldLookup>,
    max_regex_length: usize,
}

impl CustomQuery for PercolateQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        if let LeafReader::Segment(segment) = leaf {
            if let Some(column) = segment.owned_keyword_column(self.field) {
                let all_marked = self.version_field.is_some_and(|f|
                    leaf.doc_freq(&rustic_core::types::Term::new_text(f, EXTRACTION_VERSION)) == leaf.num_docs());
                let candidate = if all_marked { self.marked_candidate.as_ref().unwrap_or(&self.candidate) } else { &self.candidate };
                let memo = vec![None; column.num_terms().min(4096)];
                return Some(Box::new(LazyPercolateScorer {
                    candidates: candidate.scorer(leaf, ctx)?, column, memo,
                    verify: self.verify.clone(), live: Box::new(leaf.live_docs()),
                    current: TERMINATED, score: 0.0,
                }));
            }
        }
        self.filter.scorer(leaf, ctx)
    }
    fn children(&self) -> Vec<&Query> { self.filter.children() }
    fn name(&self) -> &'static str { "percolate" }
    fn as_any(&self) -> Option<&dyn std::any::Any> { Some(self) }
}

impl std::fmt::Debug for PercolateQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PercolateQuery").field("field", &self.field).finish()
    }
}

/// Verify only candidates consumed by the collector. Retains the immutable
/// column file, never a borrowed leaf or a copy of every stored query.
struct LazyPercolateScorer {
    candidates: Box<dyn Scorer>,
    column: rustic_core::codec::fastfields::OwnedKeywordColumn,
    memo: Vec<Option<Option<f32>>>,
    verify: Verify,
    live: Box<dyn Fn(DocId) -> bool + Send + Sync>,
    current: DocId,
    score: f32,
}
impl LazyPercolateScorer {
    fn settle(&mut self, mut doc: DocId) -> DocId {
        while doc != TERMINATED {
            if (self.live)(doc) {
                let ord = self.column.ord(doc) as usize;
                let score = match self.memo.get(ord).copied().flatten() {
                    Some(score) => score,
                    None => {
                        let score = self.column.term(ord as u32).and_then(|source| (self.verify)(&source));
                        if let Some(slot) = self.memo.get_mut(ord) { *slot = Some(score); }
                        score
                    }
                };
                if let Some(score) = score { self.current = doc; self.score = score; return doc; }
            }
            doc = self.candidates.advance();
        }
        self.current = TERMINATED;
        TERMINATED
    }
}
impl DocSet for LazyPercolateScorer {
    fn advance(&mut self) -> DocId { let doc = self.candidates.advance(); self.settle(doc) }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.current != TERMINATED && self.current >= target { return self.current; }
        let doc = self.candidates.seek(target); self.settle(doc)
    }
    fn doc(&self) -> DocId { self.current }
    fn size_hint(&self) -> u32 { self.candidates.size_hint() }
}
impl Scorer for LazyPercolateScorer { fn score(&mut self) -> f32 { self.score } }

fn collect<'a>(query: &'a Query, out: &mut Vec<&'a PercolateQuery>) {
    if let Query::Custom(custom) = query {
        if let Some(percolate) = custom.as_any().and_then(|q| q.downcast_ref::<PercolateQuery>()) {
            out.push(percolate);
        }
    }
    for child in query.children() { collect(child, out); }
}

/// Fetch slot metadata without re-parsing and evaluating stored queries.
pub(crate) fn add_hit_fields(query: &Query, leaf: &LeafReader<'_>, doc: DocId, fields: &mut Map<String, Value>) {
    let mut queries = Vec::new();
    collect(query, &mut queries);
    for percolate in &queries {
        if let Some(source) = leaf.keyword_value(percolate.field, doc) {
            if let Some(slots) = percolate.slots.lock().get(&source) {
                let field = if queries.len() == 1 { "_percolator_document_slot" } else { &percolate.slot_field };
                fields.insert(field.into(), serde_json::json!(slots));
            }
        }
    }
}

/// Highlight the percolated document with the stored query, rather than the
/// stored query document with the outer percolate query (ES's fetch subphase).
pub(crate) fn highlight_hit(query: &Query, leaf: &LeafReader<'_>, doc: DocId,
    spec: &Map<String, Value>, svc: &crate::index::IndexService) -> Result<Map<String, Value>> {
    let mut queries = Vec::new();
    collect(query, &mut queries);
    let mut result = Map::new();
    for percolate in &queries {
        let Some(source) = leaf.keyword_value(percolate.field, doc) else { continue };
        let slots = percolate.slots.lock().get(&source).cloned().unwrap_or_default();
        if slots.is_empty() { continue; }
        let stored: Value = serde_json::from_str(&source).map_err(|e| EsError::internal(e.to_string()))?;
        let leaves = percolate.memory.leaves();
        let mut ctx = qctx(&percolate.lookup, &leaves);
        ctx.max_regex_length = percolate.max_regex_length;
        let stored_query = parse_query(&stored, &ctx)?;
        let terms = crate::highlight::QueryTerms::for_index(&stored_query, spec, svc, &|q| parse_query(q, &ctx))?;
        for slot in slots {
            let Some(document) = percolate.documents.get(slot as usize) else { continue };
            let highlights = crate::highlight::try_highlight_hit(spec, document, &percolate.lookup, &terms, slot)?;
            for (field, fragments) in highlights {
                let mut name = String::new();
                if queries.len() > 1 {
                    name.push_str(percolate.slot_field.strip_prefix("_percolator_document_slot_").unwrap_or(""));
                    name.push('_');
                }
                if percolate.documents.len() > 1 { name.push_str(&format!("{slot}_")); }
                name.push_str(&field);
                result.insert(name, fragments);
            }
        }
    }
    Ok(result)
}

pub(crate) fn qctx<'a>(lookup: &'a FieldLookup, leaves: &'a [LeafReader<'a>]) -> QueryContext<'a> {
    QueryContext {
        deadline: None,
        lookup,
        index_name: "",
        index_uuid: "",
        aliases: Vec::new(),
        now: chrono::Utc::now().timestamp_millis(),
        leaves,
        default_fields: vec!["*".into()],
        terms_lookup: None,
        max_terms_count: 65536,
        max_regex_length: 1000,
        max_result_window: 10_000,
        seq_no_disabled: false,
        allow_unmapped_fields: true,
        knn_prefilter: None,
        knn_prefilter_lookup: None,
        allow_expensive_queries: true,
        mappings: None,
    }
}

/// A stored query must parse against the index mapping.
pub fn validate_stored_query(q: &Value, lookup: &FieldLookup, max_regex_length: usize) -> Result<()> {
    let leaves: Vec<LeafReader<'_>> = Vec::new();
    let mut ctx = qctx(lookup, &leaves);
    ctx.max_regex_length = max_regex_length;
    parse_query(q, &ctx).map(|_| ())
}

// Versioned auxiliary terms distinguish conservative extraction from older
// indexes whose unqualified terms did not prove a necessary match condition.
const EXTRACTION_VERSION: &str = "\0percolator-v1";
const UNPRUNABLE: &str = "\0percolator-fallback";
const BODY_TERM_PREFIX: &str = "\0body:";

/// Extract a necessary term condition from exact terms and analyzed match
/// queries. Unsupported forms stay candidates, including fuzzy matches and
/// zero-terms match-all. Analysis is the same as actual query verification.
pub fn extract_terms(q: &Value, lookup: &FieldLookup) -> Vec<String> {
    fn necessary(q: &Query, field: u16) -> Option<Vec<Vec<u8>>> {
        match q {
            Query::Term(t) if t.field == field => Some(vec![t.text.clone()]),
            Query::TermSet { field: f, terms, .. } if *f == field => Some(terms.clone()),
            Query::Boost { query, .. } | Query::ConstScore { query, .. } => necessary(query, field),
            Query::Bool { must, filter, should, .. } => {
                if !must.is_empty() || !filter.is_empty() {
                    // Any required clause is a necessary condition. Pick the
                    // smallest extractable one; never use optional shoulds.
                    must.iter().chain(filter).filter_map(|q| necessary(q, field)).min_by_key(Vec::len)
                } else if !should.is_empty() {
                    // A pure should requires at least one branch. Every branch
                    // must be extractable, otherwise it could match without a term.
                    should.iter().map(|q| necessary(q, field)).collect::<Option<Vec<_>>>()
                        .map(|parts| parts.into_iter().flatten().collect())
                } else { None }
            }
            Query::MatchNone => Some(Vec::new()),
            _ => None,
        }
    }
    let terms = lookup.get("body").filter(|f| f.declared == EsType::Text).and_then(|f| f.core_id)
        .and_then(|field| {
            let object = q.as_object()?;
            if object.len() != 1 || !(object.contains_key("term") || object.contains_key("match")) { return None; }
            let leaves = Vec::new();
            necessary(&parse_query(q, &qctx(lookup, &leaves)).ok()?, field)
        });
    let mut out = vec![EXTRACTION_VERSION.into()];
    match terms {
        Some(terms) => {
            for term in terms {
                let Ok(term) = String::from_utf8(term) else { return vec![EXTRACTION_VERSION.into(), UNPRUNABLE.into()]; };
                out.push(format!("{BODY_TERM_PREFIX}{term}"));
            }
            out.sort_unstable();
            out.dedup();
        }
        None => out.push(UNPRUNABLE.into()),
    }
    out
}

/// The `percolate` query.
pub fn query(o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[percolate] query is missing required [field] parameter"))?;
    let docs: Vec<Value> = match (o.get("document"), o.get("documents")) {
        (Some(d), _) => vec![d.clone()],
        (None, Some(Value::Array(a))) => a.clone(),
        _ => {
            if o.contains_key("id") || o.contains_key("index") {
                let index = o.get("index").and_then(Value::as_str).unwrap_or(ctx.index_name);
                let id = o.get("id").map(crate::query::dsl::scalar_string).unwrap_or_default();
                let lk = ctx.terms_lookup.ok_or_else(|| EsError::illegal_argument("percolating an indexed document is not available here"))?;
                let src = lk(index, &id, "_source", o.get("routing").and_then(Value::as_str))?;
                src
            } else {
                return Err(EsError::illegal_argument("[document] or [documents] must be specified"));
            }
        }
    };
    let Some(f) = ctx.lookup.get(field).cloned() else {
        return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("field [{field}] does not exist")));
    };
    if f.declared != EsType::Percolator {
        return Err(EsError::query_shard(ctx.index_name, ctx.index_uuid, format!("expected field [{field}] to be of type [percolator], but is of type [{}]", f.declared.name())));
    }
    let Some(qid) = f.core_id else { return Ok(Query::MatchNone) };
    // Memory index with the documents, built with this index's mapping.
    let mut m = ctx.mappings.map(|svc| svc.meta().mappings).unwrap_or_default();
    let mut lookup = Arc::new((*ctx.lookup).clone());
    let mut extended = false;
    // Resolve dynamic fields in the percolate doc(s) so the in-memory index
    // schema includes them (e.g. `body` when the index mapping is minimal).
    let mut parsed_docs: Vec<(usize, crate::document::ParsedDoc)> = Vec::with_capacity(docs.len());
    for (i, d) in docs.iter().enumerate() {
        let mut parsed = crate::document::parse_document(d, &i.to_string(), &m, &lookup)?;
        if let crate::document::ParseOutcome::NeedsMapping(adds) = &parsed {
            let mut next = m.clone();
            for (path, mapper) in adds {
                next.insert_at_with(path, mapper.clone(), Some(&lookup.registry));
            }
            next.assign_core_ids();
            m = next;
            lookup = Arc::new(crate::mapping::FieldLookup::build(&m, &lookup.registry));
            extended = true;
            parsed = crate::document::parse_document(d, &i.to_string(), &m, &lookup)?;
        }
        if let crate::document::ParseOutcome::Doc(pd) = parsed {
            parsed_docs.push((i, pd));
        }
    }
    // Unextended, the clone has the context lookup's fields: reuse its schema.
    let schema = if extended { lookup.core_schema() } else { (*ctx.lookup.core_schema_cached()).clone() };
    let mut mem = rustic_core::Index::new("_percolate", schema, IndexSettings::default());
    for (i, pd) in &parsed_docs {
        let mut doc = pd.doc.try_clone()?;
        doc.add_keyword(ID_FIELD, &i.to_string());
        mem.add_document(&doc).map_err(|e| EsError::internal(e.to_string()))?;
    }
    mem.refresh().map_err(|e| EsError::internal(e.to_string()))?;
    let mem = Arc::new(mem);
    let candidate_terms = lookup.get("body").filter(|field| field.declared == EsType::Text)
        .and_then(|field| field.core_id).map(|body| {
            let mut terms = vec![UNPRUNABLE.as_bytes().to_vec()];
            for leaf in mem.leaves() {
                for (token, _) in leaf.term_range(body, None, None) {
                    let mut key = BODY_TERM_PREFIX.as_bytes().to_vec();
                    key.extend(rustic_core::types::Term::from_key(&token).text);
                    terms.push(key);
                }
            }
            terms.sort_unstable();
            terms.dedup();
            terms
        });
    let lk2 = lookup.clone();
    let highlight_memory = mem.clone();
    let slots = Arc::new(Mutex::new(HashMap::new()));
    let recorded_slots = slots.clone();
    // Repeated stored queries share verification within this request. Keep a
    // bound so a corpus of unique queries cannot grow an unbounded cache.
    let verified = Mutex::new(HashMap::<String, Option<f32>>::new());
    let max_regex_length = ctx.max_regex_length;
    // Capture the request's existing typed scorer-error channel, rather than
    // consulting thread-local state while scoring or converting errors to misses.
    let request_error = crate::script::env().map(|env| env.error.clone())
        .ok_or_else(|| EsError::illegal_argument("percolate query requires a search request error context"))?;
    let verify: Verify = Arc::new(move |src: &str| -> Option<f32> {
        if let Some(score) = verified.lock().get(src).copied() { return score; }
        let qv: Value = serde_json::from_str(&src).ok()?;
        let leaves = mem.leaves();
        let mut ctx = qctx(&lk2, &leaves);
        ctx.max_regex_length = max_regex_length;
        let q = match parse_query(&qv, &ctx) {
            Ok(query) => query,
            Err(error) => {
                request_error.lock().get_or_insert(error);
                return None;
            }
        };
        let context = SearchContext::with_similarity(&q, &leaves,
            Arc::new(rustic_core::query::similarity::LuceneBm25), HashMap::new());
        if let Err(error) = context.execution_result() {
            request_error.lock().get_or_insert(error.into());
            return None;
        }
        let mut hits = Vec::new();
        for (li, leaf) in leaves.iter().enumerate() {
            let Some(mut scorer) = q.scorer(leaf, &context) else { continue };
            let live = leaf.live_docs();
            loop {
                let doc = scorer.advance();
                if doc == TERMINATED { break; }
                if live(doc) { hits.push((li, doc, scorer.score())); }
            }
        }
        if let Err(error) = context.execution_result() {
            request_error.lock().get_or_insert(error.into());
            return None;
        }
        let score = hits.iter().map(|h| h.2).fold(None, |m: Option<f32>, s| Some(m.map_or(s, |m| m.max(s))));
        if score.is_some() {
            let mut matched_slots: Vec<u32> = hits.iter().filter_map(|(li, d, _)|
                leaves[*li].keyword_value(ID_FIELD, *d)?.parse().ok()).collect();
            matched_slots.sort_unstable();
            matched_slots.dedup();
            recorded_slots.lock().entry(src.to_string()).or_insert(matched_slots);
        }
        let mut cache = verified.lock();
        if cache.len() < 4096 { cache.insert(src.to_string(), score); }
        score
    });
    let check_verify = verify.clone();
    let check = move |leaf: &LeafReader<'_>, d: DocId| -> Option<f32> {
        leaf.keyword_value(qid, d).and_then(|src| check_verify(&src))
    };
    // Include all unsupported queries and all legacy extraction formats.
    // Using actual indexed terms preserves analyzer, punctuation, arrays,
    // and copy_to behavior instead of guessing tokens from raw JSON.
    // A lazy disjunction avoids const_union's eager collect/sort of all IDs.
    let marked_candidate = match (f.aux_id("terms"), candidate_terms.as_ref()) {
        (Some(field), Some(terms)) => {
            let mut builder = Query::bool_query();
            for term in terms { builder = builder.should(Query::Term(rustic_core::types::Term::new(field, term.clone()))); }
            Some(builder.minimum_should_match(1usize).build())
        }
        _ => None,
    };
    let candidate = match (f.aux_id("terms"), candidate_terms) {
        (Some(tid), Some(terms)) => {
            let legacy = Query::bool_query().filter(Query::Exists { field: qid })
                .must_not(Query::term(tid, EXTRACTION_VERSION)).build();
            Query::bool_query().should(Query::TermSet { field: tid, terms, boost: 1.0 })
                .should(legacy).minimum_should_match(1usize).build()
        }
        _ => Query::Exists { field: qid },
    };
    let slot_field = o.get("name").and_then(Value::as_str)
        .map(|name| format!("_percolator_document_slot_{name}"))
        .unwrap_or_else(|| "_percolator_document_slot".into());
    Ok(Query::Custom(Arc::new(PercolateQuery {
        filter: FilterQuery::new("percolate", Some(candidate.clone()), Arc::new(check)),
        candidate, marked_candidate, version_field: f.aux_id("terms"), verify,
        field: qid, slot_field, slots, documents: docs, memory: highlight_memory, lookup, max_regex_length,
    })))
}

#[cfg(test)]
mod limit_tests {
    use super::*;
    use crate::{index::WriteReq, node::Node, search::{ExecOpts, Target}, shard::WriteCond};
    use serde_json::json;

    #[test]
    fn configured_regex_cap_applies_to_stored_queries_and_lazy_verification() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("percolator-limits", &json!({
            "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.max_regex_length": 3},
            "mappings": {"properties": {"query": {"type": "percolator"}, "tag": {"type": "keyword"}}}
        })).unwrap();
        let write = |id: &str, pattern: &str| svc.write(vec![WriteReq::Index {
            id: Some(id.into()), source: json!({"query": {"regexp": {"tag": pattern}}}).to_string(), routing: None, cond: WriteCond::default()
        }], true).remove(0);
        write("kept", "😀a").unwrap();
        svc.refresh().unwrap();
        let error = write("rejected", "😀ab").unwrap_err();
        assert!(error.to_json().to_string().contains("index.max_regex_length"), "{error}");
        assert!(svc.get("rejected").unwrap().is_none());
        let search = || {
            let req = crate::search::parse_request(&json!({"query": {"percolate": {
                "field": "query", "document": {"tag": "😀a"}
            }}}), &HashMap::new()).unwrap();
            crate::search::execute(&[Target::new(svc.clone())], &req, &ExecOpts {
                terms_lookup: None, cursors: None, max_buckets: 100, scroll_id: None, pit_id: None
            })
        };
        assert_eq!(search().unwrap().total, 1);
        svc.put_settings(crate::settings::normalize_index_settings(&json!({"max_regex_length": 2})), false).unwrap();
        let error = search().err().expect("lowered cap must fail, not silently return zero matches");
        assert!(error.to_json().to_string().contains("index.max_regex_length"), "{error}");
        svc.put_settings(crate::settings::normalize_index_settings(&json!({"max_regex_length": 1001})), false).unwrap();
        write("long", &"a".repeat(1001)).unwrap();
        // The failed request's captured error cannot contaminate the next request.
        assert_eq!(search().unwrap().total, 1);
    }
}
