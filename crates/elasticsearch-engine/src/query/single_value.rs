//! `esql_single_value`: ES|QL's `SingleValueQuery`. A scalar ES|QL
//! predicate pushed to the index (`v == 1`, ranges, `IN`, `LIKE`) holds
//! only on documents where its field has exactly one value: on a
//! multivalued field the comparison is NULL (the row evaluator's result),
//! which neither the predicate nor its negation selects. Leaves whose
//! column holds at most one value per document run the inner query as is.

use super::dsl::QueryContext;
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use rustic_core::query::explain::Explanation;
use rustic_core::query::numeric_dv::NumericDv;
use rustic_core::query::{CustomQuery, LeafReader, Query, Scorer, SearchContext};
use rustic_core::types::DocId;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Keyword,
    Numeric,
}

#[derive(Debug)]
struct SingleValueQuery {
    field: u16,
    kind: Kind,
    inner: Query,
}

/// `{"esql_single_value": {"field": f, "query": q}}`.
/// This process's token for internally built `esql_single_value` queries
/// (random per process, never sent to users): the query is internal to
/// ES|QL's pushdown, and a request naming it without the token is an
/// unknown query, as in ES.
pub fn internal_token() -> &'static str {
    static TOKEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| uuid::Uuid::new_v4().simple().to_string());
    TOKEN.as_str()
}

/// The key of [`internal_token`] in the query body.
pub const TOKEN_KEY: &str = "_internal";

/// Whether `body` (an `esql_single_value` body) was built by this process.
pub fn is_internal(body: &Value) -> bool {
    body.get(TOKEN_KEY).and_then(Value::as_str) == Some(internal_token())
}

/// Re-stamp every `esql_single_value` in `q` with this process's token: a
/// pushed-down ES|QL query another cluster node built (its own token) and
/// sent over the internal fetch action.
pub fn restamp(q: &mut Value) {
    match q {
        Value::Object(o) => {
            if let Some(Value::Object(b)) = o.get_mut("esql_single_value") {
                b.insert(TOKEN_KEY.into(), Value::String(internal_token().to_string()));
            }
            for v in o.values_mut() {
                restamp(v);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(restamp),
        _ => {}
    }
}

pub fn parse(body: &Value, ctx: &QueryContext) -> Result<Query> {
    if !is_internal(body) {
        return Err(EsError::unknown_named("unknown query", "esql_single_value", None, super::strict::QUERY_NAMES));
    }
    let field = body["field"].as_str().ok_or_else(|| EsError::parsing("[esql_single_value] requires a [field]"))?;
    let inner = super::dsl::parse_query(body.get("query").ok_or_else(|| EsError::parsing("[esql_single_value] requires a [query]"))?, ctx)?;
    let Some(f) = ctx.lookup.get(field) else { return Ok(inner) };
    let Some(id) = f.core_id else { return Ok(inner) };
    let kind = match f.ty {
        EsType::Keyword | EsType::Ip | EsType::Wildcard | EsType::Version => Kind::Keyword,
        EsType::Long | EsType::Integer | EsType::Short | EsType::Byte | EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat | EsType::UnsignedLong
        | EsType::Date | EsType::DateNanos | EsType::Boolean => Kind::Numeric,
        _ => return Ok(inner),
    };
    Ok(Query::Custom(Arc::new(SingleValueQuery { field: id, kind, inner })))
}

impl SingleValueQuery {
    /// Whether this leaf may hold several values of the field for a doc.
    fn multi(&self, leaf: &LeafReader<'_>) -> bool {
        match self.kind {
            Kind::Keyword => match leaf {
                LeafReader::Segment(s) => s.multi_keyword_reader(self.field).is_some(),
                LeafReader::Memtable(w, ..) => w.fast_fields().get_multi_keyword(self.field).is_some(),
            },
            Kind::Numeric => !NumericDv::open(leaf, self.field).is_single_valued(),
        }
    }

    /// The number of values of `doc` (keywords: distinct values, as their
    /// sorted-set doc values hold them).
    fn count(&self, leaf: &LeafReader<'_>, dv: &NumericDv<'_>, doc: DocId) -> usize {
        match self.kind {
            Kind::Keyword => {
                let mut v = leaf.keyword_values(self.field, doc);
                v.sort();
                v.dedup();
                v.len()
            }
            Kind::Numeric => dv.for_each(doc, |_| {}),
        }
    }
}

impl CustomQuery for SingleValueQuery {
    fn scorer(&self, leaf: &LeafReader<'_>, ctx: &SearchContext) -> Option<Box<dyn Scorer>> {
        let mut inner = self.inner.scorer(leaf, ctx)?;
        if !self.multi(leaf) {
            return Some(inner);
        }
        let dv = NumericDv::open(leaf, self.field);
        let mut docs: Vec<(DocId, f32)> = Vec::new();
        loop {
            let d = inner.advance();
            if d == rustic_core::query::TERMINATED {
                break;
            }
            if self.count(leaf, &dv, d) == 1 {
                docs.push((d, inner.score()));
            }
        }
        if docs.is_empty() {
            return None;
        }
        Some(Box::new(rustic_core::query::query::ScoredDocs::new(docs)))
    }
    fn children(&self) -> Vec<&Query> {
        vec![&self.inner]
    }
    fn name(&self) -> &'static str {
        "esql_single_value"
    }
    fn explain(&self, _leaf: &LeafReader<'_>, _ctx: &SearchContext, _doc: DocId, score: f32) -> Explanation {
        Explanation::leaf(score, "single value query match".to_string())
    }
}

#[cfg(test)]
mod internal_only_tests {
    use serde_json::{json, Value};

    /// `esql_single_value` is ES|QL-internal: a user search naming it is an
    /// unknown query (as in ES), with or without a guessed token, while the
    /// ES|QL pushdown that builds it still answers; a body re-stamped by
    /// [`super::restamp`] (a fetch from another node) is accepted.
    #[test]
    fn user_requests_cannot_name_the_internal_query() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        node.create_index("sv", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"v": {"type": "long"}}}})).unwrap();
        let svc = node.get_index("sv").unwrap();
        let reqs = [json!({"v": 1}), json!({"v": [1, 2]})].iter().map(|d| crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let search = |q: Value| -> crate::error::Result<usize> {
            let req = crate::search::parse_request(&json!({"query": q}), &std::collections::HashMap::new())?;
            let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
            let out = crate::search::execute(&[crate::search::Target::new(node.get_index("sv").unwrap())], &req, &opts)?;
            let v: Value = serde_json::from_str(&out.json)?;
            Ok(v["hits"]["hits"].as_array().map_or(0, Vec::len))
        };
        for body in [json!({"field": "v", "query": {"term": {"v": 1}}}), json!({"field": "v", "query": {"term": {"v": 1}}, "_internal": "guess"})] {
            let e = search(json!({"esql_single_value": body})).unwrap_err();
            assert!(e.reason.contains("unknown query [esql_single_value]"), "{e:?}");
            assert_eq!(e.status, 400, "{e:?}");
        }
        // Also refused where a query skips the strict request check.
        let e = crate::search::count(&[crate::search::Target::new(node.get_index("sv").unwrap())], Some(&json!({"esql_single_value": {"field": "v", "query": {"term": {"v": 1}}}})), None).unwrap_err();
        assert!(e.reason.contains("unknown query [esql_single_value]") || e.caused_by.as_ref().is_some_and(|c| c.reason.contains("unknown query [esql_single_value]")), "{e:?}");
        // The internal form (as ES|QL builds it, or re-stamped) works.
        let mut q = json!({"esql_single_value": {"field": "v", "query": {"term": {"v": 1}}, "_internal": "another-node"}});
        super::restamp(&mut q);
        assert_eq!(search(q).unwrap(), 1, "the multivalued document does not match");
        let out = crate::esql::run(&node, "FROM sv | WHERE v == 1 | STATS c = COUNT(*)", crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() }).unwrap();
        assert_eq!(out.table.rows[0][0], crate::esql::Val::Long(1));
    }
}
