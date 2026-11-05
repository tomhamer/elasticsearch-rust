//! ES9 query construction limits. The legacy max_clause_count setting is a
//! deprecated no-op; SearchUtils derives the effective cap from heap/search CPUs.
use crate::error::{EsError, Result};
use rustic_core::query::Query;
use serde_json::Value;

pub fn clause_limit() -> usize {
    let cpus = crate::processors::get().max(1);
    calculate_clause_limit(crate::memory::es_heap() >> 20, cpus * 3 / 2 + 1)
}
fn calculate_clause_limit(heap_mb: u64, search_threads: usize) -> usize {
    if search_threads == 0 || heap_mb == 0 { return 1024; }
    (heap_mb.saturating_mul(64) / search_threads as u64).clamp(1024, i32::MAX as u64) as usize
}
pub fn check_clauses(query: &Query) -> Result<()> { check_clauses_with_limit(query, clause_limit()) }
/// ES `MaxClauseCountQueryVisitor` (see [`Query::clause_count`]).
fn check_clauses_with_limit(query: &Query, max: usize) -> Result<()> {
    if query.clause_count(&|_| 1) > max {
        return Err(EsError::new(400, "too_many_nested_clauses", format!("Query contains too many nested clauses; maxClauseCount is set to {max}")));
    }
    Ok(())
}
/// `indices.query.bool.max_nested_depth` (node scope, ES default 30, min 1).
static MAX_NESTED_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(30);
pub const MAX_NESTED_DEPTH_SETTING: &str = "indices.query.bool.max_nested_depth";
pub fn max_nested_depth() -> usize { MAX_NESTED_DEPTH.load(std::sync::atomic::Ordering::Relaxed) }
/// Apply the node setting at startup (a static setting, as in ES).
pub fn configure_nested_depth(node_settings: &serde_json::Map<String, Value>) -> Result<()> {
    let Some(value) = node_settings.get(MAX_NESTED_DEPTH_SETTING) else { return Ok(()) };
    let text = value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string());
    let depth = text.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{text}] for setting [{MAX_NESTED_DEPTH_SETTING}]")))?;
    if depth < 1 { return Err(EsError::illegal_argument(format!("Failed to parse value [{text}] for setting [{MAX_NESTED_DEPTH_SETTING}] must be >= 1"))); }
    if depth > i32::MAX as i64 { return Err(EsError::illegal_argument(format!("Failed to parse value [{text}] for setting [{MAX_NESTED_DEPTH_SETTING}] must be <= {}", i32::MAX))); }
    MAX_NESTED_DEPTH.store(depth as usize, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub fn nested_depth_error() -> EsError {
    EsError::illegal_argument(format!("The nested depth of the query exceeds the maximum nested depth for queries set in [{MAX_NESTED_DEPTH_SETTING}]"))
}
/// Nesting depth of the query being parsed on this thread. `enter` fails
/// past the limit before the nested query is parsed, so a deeply nested
/// request cannot exhaust the stack; `is_root` tells the outermost query.
pub struct Depth(&'static std::thread::LocalKey<std::cell::Cell<usize>>);
impl Depth {
    pub fn enter(slot: &'static std::thread::LocalKey<std::cell::Cell<usize>>) -> Result<Self> {
        let depth = slot.with(|d| { let n = d.get() + 1; d.set(n); n });
        let guard = Depth(slot);
        if depth > max_nested_depth() { return Err(nested_depth_error()); }
        Ok(guard)
    }
    pub fn is_root(&self) -> bool { self.0.with(|d| d.get() == 1) }
}
impl Drop for Depth {
    fn drop(&mut self) { self.0.with(|d| d.set(d.get() - 1)); }
}
thread_local! {
    pub static PARSE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub static CHECK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub fn check_regex(kind: &str, value: &str, max: usize) -> Result<()> {
    let len = value.encode_utf16().count();
    if len <= max { return Ok(()); }
    let used = match kind {
        "prefix" => format!("prefix [{len}] used in the Prefix Query request"),
        // `QueryStringQueryParser.getRegexpQuery`.
        "query_string" => format!("regex [{len}] used in the [query_string]"),
        _ => format!("regex [{len}] used in the Regexp Query request"),
    };
    Err(EsError::illegal_argument(format!("The length of {used} has exceeded the allowed maximum of [{max}]. This maximum can be set by changing the [index.max_regex_length] index level setting.")))
}
pub fn check_regex_body(kind: &str, body: &Value, max: usize) -> Result<()> {
    if let Some(fields) = body.as_object() {
        for (field, spec) in fields {
            if matches!(field.as_str(), "boost" | "_name") { continue; }
            if let Some(value) = spec.as_str().or_else(|| spec.get("value").and_then(Value::as_str)) { check_regex(kind, value, max)?; }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regex_limit_counts_java_utf16_units() {
        assert!(check_regex("regexp", "😀", 2).is_ok());
        assert!(check_regex("regexp", "😀", 1).unwrap_err().reason.contains("[2]"));
        assert!(check_regex("prefix", "ab", 1).unwrap_err().reason.contains("Prefix"));
        assert_eq!(check_regex("query_string", "abc", 2).unwrap_err().reason, "The length of regex [3] used in the [query_string] has exceeded the allowed maximum of [2]. This maximum can be set by changing the [index.max_regex_length] index level setting.");
    }
    #[test]
    fn nested_bools_cannot_escape_total_leaf_limit() {
        let pair = || Query::bool_query().should(Query::MatchAll).should(Query::MatchAll).build();
        let q = Query::bool_query().must(pair()).should(pair()).build();
        assert!(check_clauses_with_limit(&q, 4).is_ok());
        assert_eq!(check_clauses_with_limit(&q, 3).unwrap_err().kind, "too_many_nested_clauses");
        assert_eq!(calculate_clause_limit(30 * 1024, 73), 26932);
        // `terms` on a numeric field is one point-set clause however many values it has.
        let set = |point_set| Query::ConstScore { query: Box::new(Query::Bool { must: vec![], should: (0..10).map(|_| Query::MatchAll).collect(), must_not: vec![], filter: vec![], minimum_should_match: rustic_core::query::MinShouldMatch::Count(1) }), score: 1.0, point_set };
        let points = || set(true);
        assert!(check_clauses_with_limit(&points(), 1).is_ok());
        // The same shape from a user `constant_score` counts every clause.
        assert!(check_clauses_with_limit(&set(false), 9).is_err());
        assert!(check_clauses_with_limit(&set(false), 10).is_ok());
        let two = Query::bool_query().must(points()).must(points()).build();
        assert!(check_clauses_with_limit(&two, 2).is_ok());
        assert!(check_clauses_with_limit(&two, 1).is_err());
        assert_eq!(calculate_clause_limit(1, 100), 1024);
    }
    #[test]
    fn nested_depth_guard_fails_past_the_limit_and_unwinds() {
        thread_local! { static D: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
        let mut held = Vec::new();
        for _ in 0..max_nested_depth() { held.push(Depth::enter(&D).unwrap()); }
        assert!(!held[0].is_root() || held.len() == 1);
        let error = Depth::enter(&D).err().unwrap();
        assert_eq!(error.reason, "The nested depth of the query exceeds the maximum nested depth for queries set in [indices.query.bool.max_nested_depth]");
        drop(held);
        assert_eq!(D.with(|d| d.get()), 0);
        assert!(Depth::enter(&D).unwrap().is_root());
        assert!(configure_nested_depth(&serde_json::from_str(r#"{"indices.query.bool.max_nested_depth":"0"}"#).unwrap()).is_err());
    }
    fn nested_bools(levels: usize) -> Value {
        let mut q = serde_json::json!({"match_all": {}});
        for _ in 1..levels { q = serde_json::json!({"bool": {"must": q}}); }
        q
    }
    #[test]
    fn request_queries_deeper_than_the_limit_are_rejected_before_recursion() {
        // 30 query objects deep is allowed (ES default), 31 is not.
        crate::search_strict::check_query(&nested_bools(30), "query", None).unwrap();
        let error = crate::search_strict::check_query(&nested_bools(31), "query", None).unwrap_err();
        assert!(format!("{error:?}").contains("The nested depth of the query exceeds the maximum nested depth for queries set in [indices.query.bool.max_nested_depth]"), "{error:?}");
        assert!(crate::search_strict::check_query(&nested_bools(330), "query", None).is_err());
        assert_eq!(CHECK_DEPTH.with(|d| d.get()), 0);
    }
    #[test]
    fn phrase_terms_each_count_and_nested_clause_message_matches_lucene() {
        let phrase = Query::Phrase { terms: (0..3).map(|i| rustic_core::types::Term::new_text(0, &format!("t{i}"))).collect(), slop: 0 };
        assert!(check_clauses_with_limit(&phrase, 3).is_ok());
        assert!(check_clauses_with_limit(&phrase, 2).is_err());
        let error = EsError::from(rustic_core::RusticError::TooManyClauses { max: 5, nested: true });
        assert_eq!((error.kind.as_str(), error.reason.as_str()), ("too_many_nested_clauses", "Query contains too many nested clauses; maxClauseCount is set to 5"));
    }
}
