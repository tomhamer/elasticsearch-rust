//! Percolate POC: replicates the current engine percolate code path so we can
//! iterate on it fast without a full ES build.
//!
//! Current path (elasticsearch-engine/src/fieldtypes/percolate.rs):
//!   1. Build an in-memory index holding the percolate document(s).
//!   2. Candidate query = Exists{percolator_field} -> every stored-query doc.
//!   3. For each candidate: read stored query JSON, parse it, run it against
//!      the in-memory index, keep the max score.
//!
//! This POC reproduces that path with a synthetic percolator corpus (N stored
//! match queries over a `body` text field) and the Rally percolate query
//! shapes, so we can measure the cost and iterate on candidate pruning.
//!
//! Usage:
//!   es-path-poc percolate-index CORPUS INDEX [N]
//!   es-path-poc percolate-bench INDEX [WARMUP] [ROUNDS]
//!
//! CORPUS: JSONL, one stored query per line: {"id":"...","query":{...}}
//! N: number of stored queries to index (default: all lines).

use super::*;
use std::collections::BTreeMap;

const ID: u16 = 0;
const BODY: u16 = 1;
const QUERY: u16 = 2; // percolator field: stored query JSON
const TERMS: u16 = 3; // extracted terms from the stored query (candidate pruning)

fn percolate_schema() -> Vec<Field> {
    vec![
        {
            let mut f = Field::new(ID, "_id", FieldType::Keyword);
            f.stored = true;
            f
        },
        {
            let mut f = Field::new(BODY, "body", FieldType::Text);
            f.stored = false;
            f
        },
        {
            let mut f = Field::new(QUERY, "query", FieldType::Keyword);
            f.stored = true;
            f.indexed = true;
            f
        },
        {
            // Multi-valued keyword: the terms extracted from the stored query.
            let mut f = Field::new(TERMS, "terms", FieldType::Keyword);
            f.stored = false;
            f.indexed = true;
            f
        },
    ]
}

/// Extract the terms from a stored query (for candidate pruning). For a
/// `match` query, the terms are the tokens of the query string. This mirrors
/// what ES does with `extracted_terms` on the percolator field.
fn extract_terms(q: &Value) -> Vec<String> {
    let o = match q.as_object() {
        Some(o) => o,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for (_kind, def) in o.iter() {
        match def {
            Value::Object(d) => {
                for (_field, val) in d.iter() {
                    let text = val
                        .as_str()
                        .or_else(|| val.get("query").and_then(Value::as_str))
                        .or_else(|| val.get("value").and_then(Value::as_str));
                    if let Some(t) = text {
                        for tok in t.split_whitespace() {
                            let tok = tok.to_lowercase();
                            if !tok.is_empty() {
                                out.push(tok);
                            }
                        }
                    }
                }
            }
            Value::String(s) => {
                for tok in s.split_whitespace() {
                    let tok = tok.to_lowercase();
                    if !tok.is_empty() {
                        out.push(tok);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Tokenize a percolate document body into terms (for the candidate query).
/// Strips non-alphanumeric characters so "stroke;" -> "stroke" (matching how
/// the standard ES analyzer tokenizes).
fn tokenize(body: &str) -> Vec<String> {
    body.split_whitespace()
        .map(|t| {
            t.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|t| !t.is_empty())
        .collect()
}

/// Index N stored queries. Each line: {"id":"...","query":{...}}
pub(super) fn index(args: &[String]) {
    let corpus = &args[2];
    let index_path = &args[3];
    let n: Option<usize> = args.get(4).and_then(|s| s.parse().ok());
    let mut index = Index::create(
        std::path::Path::new(index_path),
        "percolate-poc",
        percolate_schema(),
        IndexSettings::default(),
    )
    .unwrap();
    let mut count = 0usize;
    for line in BufReader::new(std::fs::File::open(corpus).unwrap()).lines() {
        if let Some(n) = n {
            if count >= n {
                break;
            }
        }
        let raw = line.unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        let mut d = Document::new();
        d.add_keyword(ID, v["id"].as_str().unwrap_or(&count.to_string()));
        let qjson = v["query"].to_string();
        d.add_keyword(QUERY, &qjson);
        // Extract terms for candidate pruning (mirrors ES extracted_terms).
        for t in extract_terms(&v["query"]) {
            d.add_keyword(TERMS, &t);
        }
        index.add_document(&d).unwrap();
        count += 1;
    }
    index.refresh().unwrap();
    index.commit().unwrap();
    index.force_merge().unwrap();
    index.commit().unwrap();
    println!("indexed {count} stored queries -> {index_path}");
}

/// The Rally percolate query shapes (document bodies from the audit).
fn percolate_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("president_bush", "president bush"),
        ("saddam_hussein", "saddam hussein"),
        ("hurricane_katrina", "hurricane katrina"),
        ("google", "google"),
        ("ignore_me", "ignore me"),
        (
            "sharon_stroke",
            "Israeli prime minister Ariel Sharon suffers a massive stroke; he is replaced by acting prime minister Ehud Olmert",
        ),
    ]
}

/// Build the in-memory index for a percolate document (replicates the engine).
fn build_mem_index(body: &str) -> Index {
    let mut mem = Index::new(
        "_percolate",
        percolate_schema(),
        IndexSettings::default(),
    );
    let mut d = Document::new();
    d.add_keyword(ID, "0");
    d.add_text(BODY, body);
    mem.add_document(&d).unwrap();
    mem.refresh().unwrap();
    mem
}

/// Parse a stored query JSON into a rustic Query (replicates the engine's
/// parse_query against the in-memory index). For the POC we support the
/// `match` query shape used by the percolator corpus.
fn parse_stored_query(q: &Value, _leaves: &[LeafReader<'_>]) -> Option<Query> {
    let o = q.as_object()?;
    let (kind, def) = o.iter().next()?;
    match kind.as_str() {
        "match" => {
            let d = def.as_object()?;
            let (_field, val) = d.iter().next()?;
            let text = val.as_str().or_else(|| val.get("query").and_then(Value::as_str))?;
            // match on body -> OR of the query's terms (a real match, not a
            // single multi-word term). This is what the engine's match query
            // does after analysis.
            let terms: Vec<String> = text.split_whitespace().map(|t| t.to_lowercase()).filter(|t| !t.is_empty()).collect();
            if terms.is_empty() {
                return None;
            }
            if terms.len() == 1 {
                Some(Query::term(BODY, &terms[0]))
            } else {
                Some(Query::Bool {
                    must: vec![],
                    should: terms.iter().map(|t| Query::term(BODY, t)).collect(),
                    must_not: vec![],
                    filter: vec![],
                    minimum_should_match: rustic_core::query::MinShouldMatch::Count(1),
                })
            }
        }
        "match_all" => Some(Query::MatchAll),
        _ => None,
    }
}

/// Run a query against leaves, return max score (replicates join::run).
fn run_max_score(q: &Query, leaves: &[LeafReader<'_>]) -> Option<f32> {
    let sctx = rustic_core::query::SearchContext::for_query(q, leaves, rustic_core::query::Bm25Params::default());
    let mut max: Option<f32> = None;
    for leaf in leaves {
        let Some(mut s) = q.scorer(leaf, &sctx) else { continue };
        let live = leaf.live_docs();
        loop {
            let d = s.advance();
            if d == rustic_core::query::TERMINATED {
                break;
            }
            if live(d) {
                let sc = s.score();
                max = Some(max.map_or(sc, |m| m.max(sc)));
            }
        }
    }
    max
}

/// The current percolate path: for each stored-query doc, parse + run against
/// the in-memory index. Returns (count, max_score).
fn percolate_current(
    index: &Index,
    mem: &Index,
    _body: &str,
) -> (usize, Option<f32>) {
    let leaves = index.leaves();
    let mem_leaves = mem.leaves();
    let mut count = 0usize;
    let mut max: Option<f32> = None;
    for leaf in &leaves {
        let num = leaf.num_docs();
        for d in 0..num {
            if leaf.is_deleted(d) {
                continue;
            }
            // Exists candidate: every doc with a percolator field value.
            let Some(qjson) = leaf.keyword_value(QUERY, d) else { continue };
            let qv: Value = match serde_json::from_str(&qjson) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(q) = parse_stored_query(&qv, &mem_leaves) else { continue };
            if let Some(sc) = run_max_score(&q, &mem_leaves) {
                count += 1;
                max = Some(max.map_or(sc, |m| m.max(sc)));
            }
        }
    }
    (count, max)
}

/// Debug: list stored queries that the current path matches but the pruned
/// path's candidate set misses (i.e. their extracted terms don't overlap the
/// document tokens, yet the query still matches the document).
pub(super) fn debug_misses(args: &[String]) {
    let index_path = &args[2];
    let body = args.get(3).map(String::as_str).unwrap_or("president bush");
    let index = Index::open(std::path::Path::new(index_path)).unwrap();
    let mem = build_mem_index(body);
    let leaves = index.leaves();
    let mem_leaves = mem.leaves();
    let doc_terms: Vec<String> = tokenize(body);
    let doc_set: std::collections::HashSet<&str> = doc_terms.iter().map(|s| s.as_str()).collect();
    let terms_query = Query::TermSet {
        field: TERMS,
        terms: doc_terms.iter().map(|t| t.as_bytes().to_vec()).collect(),
        boost: 1.0,
    };
    let sctx = rustic_core::query::SearchContext::for_query(&terms_query, &leaves, rustic_core::query::Bm25Params::default());
    let mut pruned_docs: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for leaf in &leaves {
        let Some(mut s) = terms_query.scorer(leaf, &sctx) else { continue };
        let live = leaf.live_docs();
        loop {
            let d = s.advance();
            if d == rustic_core::query::TERMINATED { break; }
            if live(d) { pruned_docs.insert(d); }
        }
    }
    let mut missed = 0usize;
    let mut shown = 0usize;
    for leaf in &leaves {
        let num = leaf.num_docs();
        for d in 0..num {
            if leaf.is_deleted(d) { continue; }
            let Some(qjson) = leaf.keyword_value(QUERY, d) else { continue };
            let qv: Value = match serde_json::from_str(&qjson) { Ok(v) => v, Err(_) => continue };
            let Some(q) = parse_stored_query(&qv, &mem_leaves) else { continue };
            if run_max_score(&q, &mem_leaves).is_some() {
                // Current path matches this stored query.
                if !pruned_docs.contains(&d) {
                    missed += 1;
                    if shown < 10 {
                        // Show the query and its extracted terms.
                        let terms = extract_terms(&qv);
                        let overlap: Vec<&String> = terms.iter().filter(|t| doc_set.contains(t.as_str())).collect();
                        println!("MISSED doc={d} query={qjson} extracted={terms:?} overlap={overlap:?}");
                        shown += 1;
                    }
                }
            }
        }
    }
    println!("total missed: {missed}");
}

/// The pruned percolate path: candidate query = terms query over the
/// percolate document's tokens on the `terms` field. Only stored queries whose
/// extracted terms overlap the document are parsed and run.
fn percolate_pruned(
    index: &Index,
    mem: &Index,
    body: &str,
) -> (usize, Option<f32>, usize) {
    let leaves = index.leaves();
    let mem_leaves = mem.leaves();
    let doc_terms = tokenize(body);
    if doc_terms.is_empty() {
        return (0, None, 0);
    }
    // Build a terms query over the document's tokens on the `terms` field.
    let terms_query = Query::TermSet {
        field: TERMS,
        terms: doc_terms.iter().map(|t| t.as_bytes().to_vec()).collect(),
        boost: 1.0,
    };
    let sctx = rustic_core::query::SearchContext::for_query(&terms_query, &leaves, rustic_core::query::Bm25Params::default());
    let mut count = 0usize;
    let mut max: Option<f32> = None;
    let mut candidates = 0usize;
    for leaf in &leaves {
        let Some(mut s) = terms_query.scorer(leaf, &sctx) else { continue };
        let live = leaf.live_docs();
        loop {
            let d = s.advance();
            if d == rustic_core::query::TERMINATED {
                break;
            }
            if !live(d) {
                continue;
            }
            candidates += 1;
            let Some(qjson) = leaf.keyword_value(QUERY, d) else { continue };
            let qv: Value = match serde_json::from_str(&qjson) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(q) = parse_stored_query(&qv, &mem_leaves) else { continue };
            if let Some(sc) = run_max_score(&q, &mem_leaves) {
                count += 1;
                max = Some(max.map_or(sc, |m| m.max(sc)));
            }
        }
    }
    (count, max, candidates)
}

pub(super) fn bench(args: &[String]) {
    let index_path = &args[2];
    let warmup: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let rounds: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);
    let index = Index::open(std::path::Path::new(index_path)).unwrap();
    let queries = percolate_queries();
    let mut samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut per_query = Vec::new();
    for (name, body) in &queries {
        let mem = build_mem_index(body);
        let mut t_cur: Vec<f64> = Vec::new();
        let mut t_pruned: Vec<f64> = Vec::new();
        for r in 0..warmup + rounds {
            let start = Instant::now();
            let (c, m) = percolate_current(&index, &mem, body);
            let us = start.elapsed().as_secs_f64() * 1e6;
            if r >= warmup {
                t_cur.push(us);
            }
            let _ = (c, m);
            let start = Instant::now();
            let (c, m, cand) = percolate_pruned(&index, &mem, body);
            let us = start.elapsed().as_secs_f64() * 1e6;
            if r >= warmup {
                t_pruned.push(us);
            }
            let _ = (c, m, cand);
        }
        // Final verification run: both paths must agree on matches + max score.
        let (count, max) = percolate_current(&index, &mem, body);
        let (pcount, pmax, candidates) = percolate_pruned(&index, &mem, body);
        assert_eq!(count, pcount, "{name}: match count mismatch");
        assert_eq!(max, pmax, "{name}: max score mismatch");
        per_query.push(json!({"name":name,"body":body,"matches":count,"max_score":max,"pruned_candidates":candidates,"pruned_matches":pcount,"pruned_max_score":pmax}));
        samples.insert(format!("{name}/current"), t_cur);
        samples.insert(format!("{name}/pruned"), t_pruned);
    }
    let stats = |v: &Vec<f64>| -> Value {
        let mut s = v.clone();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        json!({"n":n,"p50_us":if n%2==0 {(s[n/2-1]+s[n/2])/2.0} else {s[n/2]},"p95_us":s[(n as f64*0.95).ceil() as usize-1],"mean_us":s.iter().sum::<f64>()/n as f64})
    };
    let out = json!({
        "index": index_path,
        "warmup": warmup,
        "rounds": rounds,
        "per_query": per_query,
        "stats": samples.iter().map(|(k, v)| (k.clone(), stats(v))).collect::<BTreeMap<_, _>>(),
    });
    println!("{}", out);
}
