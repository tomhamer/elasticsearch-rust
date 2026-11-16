//! SQL and ES|QL semantics regressions: SQL projection scoping, three-valued
//! filter pushdown, SQL results past a million rows, and multi-valued
//! grouping keys (duplicate membership and combination admission).

use super::exec::{run, run_local, EsqlOptions};
use super::parse::Params;
use super::Val;
use crate::index::WriteReq;
use crate::node::Node;
use serde_json::{json, Value};
use std::sync::Arc;

fn index(name: &str, mappings: Value, docs: impl IntoIterator<Item = (String, Value)>) -> (tempfile::TempDir, Arc<Node>) {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    add_index(&node, name, mappings, docs);
    (dir, node)
}

fn add_index(node: &Node, name: &str, mappings: Value, docs: impl IntoIterator<Item = (String, Value)>) {
    let svc = node.create_index(name, &json!({"settings": {"number_of_shards": 1}, "mappings": mappings})).unwrap();
    let docs: Vec<(String, Value)> = docs.into_iter().collect();
    for chunk in docs.chunks(20_000) {
        let w = chunk.iter().map(|(id, s)| WriteReq::Index { id: Some(id.clone()), source: s.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(w, false) {
            r.unwrap();
        }
    }
    svc.refresh().unwrap();
}

fn sql_rows(node: &Node, q: &str) -> Vec<Vec<Val>> {
    crate::sql::execute(node, q, None, None).unwrap_or_else(|e| panic!("{q}: {e}")).rows
}

/// The rows of the streaming SQL scan the HTTP handler opens (`None`: the
/// query does not stream).
fn sql_scan_rows(node: &Node, q: &str) -> Option<Vec<Value>> {
    let mut scan = crate::sql::scan_with(node, q, None, None, None, None, None).unwrap_or_else(|e| panic!("{q}: {e}"))?;
    let mut out = Vec::new();
    loop {
        let page = scan.page(node, 2).unwrap();
        let done = page.is_empty() || scan.len() == 0;
        out.extend(page);
        if done {
            return Some(out);
        }
    }
}

fn esql_rows(node: &Node, q: &str) -> Vec<Vec<Val>> {
    run(node, q, EsqlOptions::default()).unwrap_or_else(|e| panic!("{q}: {e}")).table.rows
}

fn ints(vals: &[i64]) -> Vec<Val> {
    vals.iter().map(|v| Val::Int(*v)).collect()
}

// ------------------------------------------------------------ #78

/// SQL evaluates every projection against the input row: an alias never
/// changes what a later projection of the same SELECT reads.
#[test]
fn sql_projection_aliases_bind_the_input_row() {
    let (_dir, node) = index("proj", json!({"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}), [("1".to_string(), json!({"a": 1, "b": 2}))]);
    let cases: [(&str, &[i64]); 5] = [
        ("SELECT a AS b, b AS a FROM proj", &[1, 2]),
        ("SELECT b AS a, a AS b FROM proj", &[2, 1]),
        ("SELECT a AS b, b + 10 AS c FROM proj", &[1, 12]),
        ("SELECT a + 1 AS b, b * 10 AS a FROM proj", &[2, 20]),
        ("SELECT a AS b, b AS a FROM proj ORDER BY a", &[1, 2]),
    ];
    for (q, want) in cases {
        // Materialized execution.
        assert_eq!(sql_rows(&node, q), vec![ints(want)], "{q}");
        // The streaming scan, where the query streams.
        if let Some(rows) = sql_scan_rows(&node, q) {
            assert_eq!(rows, vec![json!(want)], "{q} (streamed)");
        }
    }
    // The plain swap streams.
    assert!(sql_scan_rows(&node, "SELECT a AS b, b AS a FROM proj").is_some());
    // `SELECT *` with a shadowing alias keeps no temporary column.
    let t = crate::sql::execute(&node, "SELECT *, a AS b, b AS a FROM proj", None, None).map(|t| t.cols.iter().map(|c| c.name.clone()).collect::<Vec<_>>());
    if let Ok(names) = t {
        assert!(names.iter().all(|n| !n.starts_with("__sql_projection")), "{names:?}");
    }
}

// ------------------------------------------------------------ #79

fn nulls_index() -> (tempfile::TempDir, Arc<Node>) {
    index(
        "nulls",
        json!({"properties": {"id": {"type": "keyword"}, "amount": {"type": "integer"}, "k": {"type": "keyword"}}}),
        [
            ("1".to_string(), json!({"id": "d1", "amount": 3, "k": "x"})),
            ("2".to_string(), json!({"id": "d2", "amount": 7, "k": "y"})),
            ("3".to_string(), json!({"id": "d3"})),
            ("4".to_string(), json!({"id": "d4", "amount": null, "k": null})),
        ],
    )
}

fn ids(rows: Vec<Vec<Val>>) -> Vec<String> {
    let mut out: Vec<String> = rows.into_iter().map(|r| r[0].to_text()).collect();
    out.sort();
    out
}

/// ES SQL pushes a field predicate down to Query DSL two-valued (`NOT p` and
/// `<>` / `NOT IN` / `NOT LIKE` / `NOT BETWEEN` are `must_not`, so null and
/// missing values match; `NOT` of a comparison negates it), while the same
/// predicate over an expression runs per row as ES's script does, where
/// NULL is not a match. ES|QL keeps three-valued logic either way.
/// `{a}` / `{k}` are the fields themselves (pushed into the index) or an
/// expression of them (evaluated per row).
#[test]
fn pushed_down_predicates_keep_three_valued_logic() {
    let (_dir, node) = nulls_index();
    // (template, rows on the fields, rows on the expressions)
    let sql: [(&str, &[&str], &[&str]); 14] = [
        ("NOT ({a} > 5)", &["d1"], &["d1"]),
        ("NOT (NOT ({a} > 5))", &["d2"], &["d2"]),
        ("{k} NOT IN ('x', NULL)", &["d2", "d3", "d4"], &[]),
        ("NOT ({k} IN ('x', NULL))", &["d2", "d3", "d4"], &[]),
        ("{k} IN ('x', NULL)", &["d1"], &["d1"]),
        ("NOT ({k} IN ('x'))", &["d2", "d3", "d4"], &["d2"]),
        ("NOT ({a} > 5 AND {k} = 'y')", &["d1", "d3", "d4"], &["d1"]),
        ("NOT ({a} > 5 OR {k} = 'x')", &["d3", "d4"], &[]),
        ("NOT ({a} < 5 OR {k} = 'q')", &["d2", "d3", "d4"], &["d2"]),
        ("NOT ({a} IS NULL)", &["d1", "d2"], &["d1", "d2"]),
        ("NOT ({a} IS NOT NULL)", &["d3", "d4"], &["d3", "d4"]),
        ("NOT ({a} = 3)", &["d2", "d3", "d4"], &["d2"]),
        ("{a} != 3", &["d2", "d3", "d4"], &["d2"]),
        ("NOT ({a} BETWEEN 1 AND 5)", &["d2", "d3", "d4"], &["d2"]),
    ];
    for (template, fields, exprs) in sql {
        for (a, k, want) in [("amount", "k", fields), ("(amount + 0)", "LCASE(k)", exprs)] {
            let q = format!("SELECT id FROM nulls WHERE {}", template.replace("{a}", a).replace("{k}", k));
            assert_eq!(ids(sql_rows(&node, &q)), want.to_vec(), "{q}");
        }
    }
    let esql: [(&str, &[&str]); 9] = [
        ("NOT ({a} > 5)", &["d1"]),
        ("NOT (NOT ({a} > 5))", &["d2"]),
        ("{k} NOT IN (\"x\", null)", &[]),
        ("NOT ({k} IN (\"x\", null))", &[]),
        ("NOT ({k} IN (\"x\"))", &["d2"]),
        ("NOT ({a} > 5 AND {k} == \"y\")", &["d1"]),
        ("NOT ({a} IS NULL)", &["d1", "d2"]),
        ("NOT ({k} LIKE \"x*\")", &["d2"]),
        ("{a} != 3", &["d2"]),
    ];
    for (template, want) in esql {
        for (a, k) in [("amount", "k"), ("(amount + 0)", "TO_LOWER(k)")] {
            let q = format!("FROM nulls | WHERE {} | KEEP id", template.replace("{a}", a).replace("{k}", k));
            assert_eq!(ids(esql_rows(&node, &q)), want.to_vec(), "{q}");
        }
    }
}

// ------------------------------------------------------------ #80

/// SQL has no row cap: a materialized (computed-sort) result past a million
/// rows keeps every row up to its LIMIT, and an unlimited one keeps all.
#[test]
fn sql_materialized_results_are_not_cut_at_a_million_rows() {
    const N: i64 = 1_000_001;
    let (_dir, node) = index("big", json!({"properties": {"x": {"type": "integer"}}}), (0..N).map(|i| (i.to_string(), json!({"x": i}))));
    for q in ["SELECT x FROM big ORDER BY x + 0 LIMIT 1000001", "SELECT x FROM big ORDER BY x + 0", "SELECT x FROM big ORDER BY x DESC"] {
        let rows = sql_rows(&node, q);
        assert_eq!(rows.len(), N as usize, "{q}");
        let (first, last) = if q.contains("DESC") { (N - 1, 0) } else { (0, N - 1) };
        assert_eq!(rows[0], vec![Val::Int(first)], "{q}");
        assert_eq!(rows[rows.len() - 1], vec![Val::Int(last)], "{q}");
    }
    // A LIMIT still bounds the result.
    assert_eq!(sql_rows(&node, "SELECT x FROM big ORDER BY x + 0 LIMIT 1000000").len(), 1_000_000);
}

// ------------------------------------------------------------ #81

/// A repeated grouping value puts the row in its group once; distinct
/// values put it in one group each.
#[test]
fn repeated_group_key_values_count_a_row_once() {
    let t = run_local("ROW price = 10, color = [\"blue\", \"blue\"] | STATS s = SUM(price), c = COUNT(*) BY color", &Params::default()).unwrap();
    assert_eq!(t.rows, vec![vec![Val::Long(10), Val::Long(1), Val::Str("blue".into())]]);
    let t = run_local("ROW price = 10, color = [\"blue\", \"red\", \"blue\"], size = [1, 1] | STATS s = SUM(price), c = COUNT(*) BY color, size | SORT color", &Params::default()).unwrap();
    assert_eq!(
        t.rows,
        vec![
            vec![Val::Long(10), Val::Long(1), Val::Str("blue".into()), Val::Int(1)],
            vec![Val::Long(10), Val::Long(1), Val::Str("red".into()), Val::Int(1)],
        ]
    );
    // Indexed numeric arrays keep their repeated values in doc values.
    let (_dir, node) = index(
        "mv",
        json!({"properties": {"price": {"type": "long"}, "n": {"type": "long"}, "k": {"type": "keyword"}}}),
        [("1".to_string(), json!({"price": 10, "n": [1, 1, 2], "k": ["a", "a"]})), ("2".to_string(), json!({"price": 5, "n": [2, 2]}))],
    );
    let rows = esql_rows(&node, "FROM mv | STATS s = SUM(price), c = COUNT(*) BY n | SORT n");
    assert_eq!(rows, vec![vec![Val::Long(10), Val::Long(1), Val::Long(1)], vec![Val::Long(15), Val::Long(2), Val::Long(2)]]);
    let rows = esql_rows(&node, "FROM mv | WHERE k IS NOT NULL | STATS s = SUM(price), c = COUNT(*) BY k, n | SORT n");
    assert_eq!(rows, vec![vec![Val::Long(10), Val::Long(1), Val::Str("a".into()), Val::Long(1)], vec![Val::Long(10), Val::Long(1), Val::Str("a".into()), Val::Long(2)]]);
    // The partial (per-shard) aggregation groups the same way.
    use super::partial::{partial_groups, AggKind, AggState};
    let rows = vec![vec![Val::Long(10), Val::Multi(vec![Val::Str("blue".into()), Val::Str("blue".into()), Val::Str("red".into())])]];
    let groups = partial_groups(&rows, &[1], &[(AggKind::Count, None), (AggKind::Sum, Some(0))]);
    assert_eq!(groups.len(), 2);
    for g in &groups {
        assert_eq!(g.states[0], AggState::Count(1), "{g:?}");
        assert!(matches!(&g.states[1], AggState::Sum { n: 1, lsum: Some(10), .. }), "{g:?}");
    }
}

// ------------------------------------------------------------ #82

/// The combinations of several multi-valued grouping keys are admitted on
/// the request breaker before any is built: a small request whose product
/// is huge is rejected at once, holding nothing.
#[test]
fn group_key_combinations_are_admitted_before_expansion() {
    let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
    breakers.apply_settings(&|key| match key {
        "indices.breaker.request.limit" => Some("64kb".into()),
        "indices.breaker.total.limit" => Some("-1".into()),
        _ => None,
    });
    // 40^5 (about 10^8) combinations of five 40-value keys.
    let values = format!("[{}]", (0..40).map(|i| i.to_string()).collect::<Vec<_>>().join(", "));
    let q = format!("ROW a = {values}, b = {values}, c = {values}, d = {values}, e = {values} | STATS n = COUNT(*) BY a, b, c, d, e");
    crate::memory::scoped(Some(crate::memory::RequestScope::new("group combinations", breakers)), || {
        let start = std::time::Instant::now();
        let err = run_local(&q, &Params::default()).unwrap_err();
        assert_eq!(err.status, 429, "{err}");
        assert_eq!(err.kind, "circuit_breaking_exception", "{err}");
        assert!(start.elapsed() < std::time::Duration::from_secs(5), "rejected only after {:?}", start.elapsed());
        assert_eq!(breakers.request.reserved(), 0);
        // A modest product still groups.
        let t = run_local("ROW a = [1, 2], b = [3, 4] | STATS n = COUNT(*) BY a, b", &Params::default()).unwrap();
        assert_eq!(t.rows.len(), 4);
    });
}

// ------------------------------------------------------------ #79 parity

/// ES 9.5.5's answers on the reviewer's documents (k1 {a:1,b:true},
/// k2 {a:null,b:false}, k3 {}, k4 {a:2,b:null}, k5 {a:3}).
fn parity_index() -> (tempfile::TempDir, Arc<Node>) {
    index(
        "par",
        json!({"properties": {"id": {"type": "keyword"}, "a": {"type": "long"}, "b": {"type": "boolean"}}}),
        [
            ("1".to_string(), json!({"id": "k1", "a": 1, "b": true})),
            ("2".to_string(), json!({"id": "k2", "a": null, "b": false})),
            ("3".to_string(), json!({"id": "k3"})),
            ("4".to_string(), json!({"id": "k4", "a": 2, "b": null})),
            ("5".to_string(), json!({"id": "k5", "a": 3})),
        ],
    )
}

#[test]
fn sql_where_matches_es_on_null_and_missing_values() {
    let (_dir, node) = parity_index();
    let all_but_k1: &[&str] = &["k2", "k3", "k4", "k5"];
    let cases: [(&str, &[&str]); 8] = [
        ("NOT (a = 1)", all_but_k1),
        // The NULL member is ignored; null and missing values match.
        ("a NOT IN (1, NULL)", all_but_k1),
        ("a <> 1", all_but_k1),
        ("NOT (a = 1 OR b)", all_but_k1),
        ("NOT (a = 1 AND b)", all_but_k1),
        ("NOT (a > 1)", &["k1"]),
        ("NOT (a IS NULL)", &["k1", "k4", "k5"]),
        ("b IS NULL", &["k3", "k4", "k5"]),
    ];
    for (cond, want) in cases {
        let q = format!("SELECT id FROM par WHERE {cond}");
        assert_eq!(ids(sql_rows(&node, &q)), want.to_vec(), "{q}");
        // The streaming scan the HTTP handler opens agrees.
        if let Some(rows) = sql_scan_rows(&node, &q) {
            let mut got: Vec<String> = rows.iter().map(|r| r[0].as_str().unwrap().to_string()).collect();
            got.sort();
            assert_eq!(got, want.to_vec(), "scan {q}");
        }
    }
    let q = "SELECT COUNT(*) FROM par WHERE NOT (a IN (1, 2))";
    assert_eq!(sql_rows(&node, q)[0][0].as_i64(), Some(3), "{q}");
    // `<>` translates to must_not of the term, as ES SQL's translation does.
    let tr = crate::sql::translate(&node, "SELECT id FROM par WHERE a <> 1", None, 1000).unwrap();
    let text = tr["query"].to_string();
    assert!(text.contains("must_not") && !text.contains("exists"), "{tr}");
}

/// ES 9.5.5 fails `WHERE NOT b` on a boolean field with HTTP 500
/// `sql_illegal_argument_exception` ("Don't know how to translate
/// FieldAttribute .."); so does every SQL path here.
#[test]
fn sql_where_not_boolean_field_is_refused() {
    let (_dir, node) = parity_index();
    let q = "SELECT id FROM par WHERE NOT b";
    let e = crate::sql::execute(&node, q, None, None).err().expect("refused");
    assert_eq!((e.status, e.kind.as_str()), (500, "sql_illegal_argument_exception"), "{e}");
    assert_eq!(e.reason, "Don't know how to translate FieldAttribute b");
    assert!(crate::sql::scan_with(&node, q, None, None, None, None, None).is_err());
    assert!(crate::sql::translate(&node, q, None, 1000).is_err());
}

#[test]
fn sql_boolean_scalars_match_es() {
    let (_dir, node) = parity_index();
    let cases: [(&str, Val); 7] = [
        ("NOT NULL", Val::Null),
        ("NULL AND FALSE", Val::Bool(false)),
        ("NULL AND TRUE", Val::Null),
        ("NULL OR TRUE", Val::Bool(true)),
        ("NULL OR FALSE", Val::Null),
        ("NOT (NULL = 1)", Val::Null),
        ("NOT (1 = 1)", Val::Bool(false)),
    ];
    for (expr, want) in cases {
        let q = format!("SELECT {expr}");
        assert_eq!(sql_rows(&node, &q)[0][0], want, "{q}");
    }
    let rows = sql_rows(&node, "SELECT id, NOT b FROM par ORDER BY id");
    let got: Vec<Val> = rows.into_iter().map(|r| r[1].clone()).collect();
    assert_eq!(got, vec![Val::Bool(false), Val::Bool(true), Val::Null, Val::Null, Val::Null]);
}

/// ES|QL keeps three-valued logic in WHERE (its pushed-down single-value
/// queries never match a null or missing value): a NULL condition drops the row.
#[test]
fn esql_where_keeps_three_valued_logic_on_the_parity_documents() {
    let (_dir, node) = parity_index();
    let cases: [(&str, &[&str]); 7] = [
        ("NOT (a == 1)", &["k4", "k5"]),
        ("a != 1", &["k4", "k5"]),
        ("NOT (a > 1)", &["k1"]),
        ("NOT (a IS NULL)", &["k1", "k4", "k5"]),
        ("b IS NULL", &["k3", "k4", "k5"]),
        ("NOT b", &["k2"]),
        ("NOT (a IN (1, 2))", &["k5"]),
    ];
    for (cond, want) in cases {
        let q = format!("FROM par | WHERE {cond} | KEEP id");
        assert_eq!(ids(esql_rows(&node, &q)), want.to_vec(), "{q}");
    }
    let rows = esql_rows(&node, "FROM par | EVAL nb = NOT b | SORT id | KEEP nb");
    let got: Vec<Val> = rows.into_iter().map(|r| r[0].clone()).collect();
    assert_eq!(got, vec![Val::Bool(false), Val::Bool(true), Val::Null, Val::Null, Val::Null]);
}

// ------------------------------------------------------------ batch 6

/// Review finding 439: SQL ISNULL(a, b) is the two-argument fallback;
/// GREATEST / LEAST skip NULL operands; in projections and WHERE.
#[test]
fn sql_conditional_null_functions() {
    let (_dir, node) = index("bills", json!({"properties": {"id": {"type": "keyword"}, "amount": {"type": "integer"}}}), [
        ("1".to_string(), json!({"id": "b1", "amount": 25})),
        ("2".to_string(), json!({"id": "b2"})),
    ]);
    let rows = sql_rows(&node, "SELECT id, ISNULL(amount, 0) AS amount FROM bills ORDER BY id");
    assert_eq!(rows.iter().map(|r| r[1].as_i64()).collect::<Vec<_>>(), vec![Some(25), Some(0)]);
    assert_eq!(ids(sql_rows(&node, "SELECT id FROM bills WHERE ISNULL(amount, 0) > 0")), vec!["b1"]);
    for (q, want) in [
        ("SELECT GREATEST(NULL, 25) AS v", Some(25)),
        ("SELECT LEAST(NULL, 25) AS v", Some(25)),
        ("SELECT GREATEST(3, NULL, 7) AS v", Some(7)),
        ("SELECT LEAST(3, NULL, 7) AS v", Some(3)),
        ("SELECT GREATEST(NULL, NULL) AS v", None),
    ] {
        assert_eq!(sql_rows(&node, q)[0][0].as_i64(), want, "{q}");
    }
    assert_eq!(ids(sql_rows(&node, "SELECT id FROM bills WHERE GREATEST(amount, 1) > 10")), vec!["b1"]);
}

/// Review finding 508: SQL string functions keep SQL's semantics: LOCATE
/// takes the pattern first, REPLACE is literal, CONCAT reads NULL as '',
/// LENGTH excludes trailing blanks; in WHERE too.
#[test]
fn sql_string_functions_keep_sql_semantics() {
    let (_dir, node) = index("names", json!({"properties": {"id": {"type": "keyword"}, "name": {"type": "keyword"}}}), [
        ("1".to_string(), json!({"id": "n1", "name": "foobar"})),
        ("2".to_string(), json!({"id": "n2", "name": "baz"})),
    ]);
    let one = |q: &str| sql_rows(&node, q)[0][0].clone();
    assert_eq!(one("SELECT LOCATE('bar', 'foobar') AS v").as_i64(), Some(4));
    assert_eq!(one("SELECT LOCATE('o', 'foobar', 3) AS v").as_i64(), Some(3));
    assert_eq!(one("SELECT REPLACE('a.b', '.', '-') AS v").to_text(), "a-b");
    assert_eq!(one("SELECT CONCAT('customer:', NULL) AS v").to_text(), "customer:");
    assert_eq!(one("SELECT LENGTH('ab  ') AS v").as_i64(), Some(2));
    assert_eq!(ids(sql_rows(&node, "SELECT id FROM names WHERE LOCATE('bar', name) > 0")), vec!["n1"]);
}

/// Review finding 440: a global STATS over no matching document answers
/// its one row on the columnar path, as on the row path.
#[test]
fn columnar_global_stats_answer_the_empty_row() {
    let (_dir, node) = index("metrics440", json!({"properties": {"v": {"type": "long"}}}), (0..20).map(|i| (i.to_string(), json!({"v": i}))));
    let q = "FROM metrics440 | WHERE v > 100 | STATS n = COUNT(*), mean = AVG(v) | WHERE n == 0";
    let rows = esql_rows(&node, q);
    assert_eq!(rows, vec![vec![Val::Long(0), Val::Null]], "{q}");
    let q2 = "FROM metrics440 | WHERE v > 100 | STATS n = COUNT(*), s = SUM(v), d = COUNT_DISTINCT(v)";
    let columnar = esql_rows(&node, q2);
    super::exec::fast_switches::no_columnar(true);
    let rows_path = esql_rows(&node, q2);
    super::exec::fast_switches::no_columnar(false);
    assert_eq!(columnar, rows_path);
    assert_eq!(columnar.len(), 1);
}

/// Review finding 441: TOP(field, limit, order, output) returns the output
/// column's values of the top rows, typed as the output.
#[test]
fn top_returns_its_output_column() {
    let (_dir, node) = index("status441", json!({"properties": {"@timestamp": {"type": "date"}, "status": {"type": "keyword"}, "service": {"type": "keyword"}}}), [
        ("1".to_string(), json!({"@timestamp": "2026-01-01T00:00:00Z", "status": "healthy", "service": "api"})),
        ("2".to_string(), json!({"@timestamp": "2026-01-01T00:05:00Z", "status": "failed", "service": "api"})),
        ("3".to_string(), json!({"@timestamp": "2026-01-01T00:01:00Z", "status": "healthy", "service": "web"})),
    ]);
    let out = run(&node, "FROM status441 | STATS latest = TOP(@timestamp, 1, \"desc\", status) BY service | SORT service", EsqlOptions::default()).unwrap();
    assert_eq!(out.table.cols[0].ty, "keyword");
    assert_eq!(out.table.rows.iter().map(|r| r[0].to_text()).collect::<Vec<_>>(), vec!["failed", "healthy"]);
    assert_eq!(ids(esql_rows(&node, "FROM status441 | STATS latest = TOP(@timestamp, 1, \"desc\", status) BY service | WHERE latest == \"failed\" | KEEP service")), vec!["api"]);
    let rows = esql_rows(&node, "FROM status441 | STATS first2 = TOP(@timestamp, 2, \"asc\", status)");
    assert_eq!(rows[0][0], Val::Multi(vec![Val::Str("healthy".into()), Val::Str("healthy".into())]));
}

/// Review finding 442: integer MV_PERCENTILE interpolates as ES does, in
/// the integer type.
#[test]
fn integer_mv_percentile_stays_integral() {
    let (_dir, node) = index("p442", json!({"properties": {"x": {"type": "keyword"}}}), [("1".to_string(), json!({"x": "a"}))]);
    let rows = esql_rows(&node, "ROW v = [1, 2] | EVAL p = MV_PERCENTILE(v, 50)");
    assert_eq!(rows[0][1], Val::Int(1));
    assert!(esql_rows(&node, "ROW v = [1, 2] | EVAL p = MV_PERCENTILE(v, 50) | WHERE p > 1").is_empty());
    assert_eq!(esql_rows(&node, "ROW v = [-3, 10, 2] | EVAL p = MV_PERCENTILE(v, 75)")[0][1], Val::Int(6));
    assert_eq!(esql_rows(&node, "ROW v = [1.0, 2.0] | EVAL p = MV_PERCENTILE(v, 50)")[0][1], Val::Double(1.5));
}

/// Review finding 443: PRESENT / ABSENT over aggregate_metric_double.
#[test]
fn present_and_absent_over_aggregate_metric_double() {
    let (_dir, node) = index("amd443", json!({"properties": {"service": {"type": "keyword"}, "metric": {"type": "aggregate_metric_double", "metrics": ["min", "max", "sum", "value_count"], "default_metric": "max"}}}), [
        ("1".to_string(), json!({"service": "a", "metric": {"min": 1.0, "max": 5.0, "sum": 6.0, "value_count": 2}})),
        ("2".to_string(), json!({"service": "b"})),
    ]);
    let rows = esql_rows(&node, "FROM amd443 | STATS missing = ABSENT(metric), available = PRESENT(metric) BY service | SORT service");
    assert_eq!(rows, vec![vec![Val::Bool(false), Val::Bool(true), Val::Str("a".into())], vec![Val::Bool(true), Val::Bool(false), Val::Str("b".into())]]);
    assert_eq!(ids(esql_rows(&node, "FROM amd443 | STATS missing = ABSENT(metric) BY service | WHERE missing | KEEP service")), vec!["b"]);
}

/// Review finding 467: the source fallback reads what the field indexed:
/// null_value substitutes an explicit null, an out-of-range integer that
/// was ignored is no value, scaled_float rounds as the mapper does.
#[test]
fn source_fallback_is_mapping_faithful() {
    let (_dir, node) = index("m467", json!({"properties": {
        "status": {"type": "keyword", "null_value": "unknown"},
        "b": {"type": "byte", "ignore_malformed": true},
        "s": {"type": "scaled_float", "scaling_factor": 10}}}), [
        ("1".to_string(), json!({"status": null, "b": 300, "s": -1.25})),
        ("2".to_string(), json!({"status": "ok", "b": 7, "s": 2.0})),
    ]);
    let rows = esql_rows(&node, "FROM m467 METADATA _id | WHERE status == \"unknown\" | KEEP _id, status, b, s");
    assert_eq!(rows, vec![vec![Val::Str("1".into()), Val::Str("unknown".into()), Val::Null, Val::Double(-1.2)]]);
    let rows = esql_rows(&node, "FROM m467 METADATA _id | STATS c = COUNT(*) BY status | SORT status");
    assert_eq!(rows, vec![vec![Val::Long(1), Val::Str("ok".into())], vec![Val::Long(1), Val::Str("unknown".into())]]);
}

/// Review findings 474 / 475: CIDR_MATCH compares IPv4 and IPv4-mapped
/// IPv6 alike; TO_LONG / TO_INTEGER round doubles (Java Math.round).
#[test]
fn cidr_mapped_addresses_and_rounding_conversions() {
    let (_dir, node) = index("x474", json!({"properties": {"x": {"type": "keyword"}}}), [("1".to_string(), json!({"x": "a"}))]);
    let one = |q: &str| esql_rows(&node, q)[0].last().cloned().unwrap();
    assert_eq!(one("ROW ip = TO_IP(\"::ffff:192.0.2.1\") | EVAL m = CIDR_MATCH(ip, \"192.0.2.0/24\")"), Val::Bool(true));
    assert_eq!(one("ROW ip = TO_IP(\"192.0.2.1\") | EVAL m = CIDR_MATCH(ip, \"::/0\")"), Val::Bool(true));
    assert_eq!(one("ROW ip = TO_IP(\"192.0.3.1\") | EVAL m = CIDR_MATCH(ip, \"::ffff:192.0.2.0/120\")"), Val::Bool(false));
    assert_eq!(one("ROW ip = TO_IP(\"::ffff:192.0.2.1\") | EVAL s = TO_STRING(ip)"), Val::Str("192.0.2.1".into()));
    for (x, want) in [("3.75", 4), ("-3.75", -4), ("2.5", 3), ("-2.5", -2), ("3.25", 3)] {
        let r = esql_rows(&node, &format!("ROW x = {x} | EVAL n = TO_LONG(x), i = TO_INTEGER(x)"));
        assert_eq!((r[0][1].clone(), r[0][2].clone()), (Val::Long(want), Val::Int(want)), "{x}");
    }
    assert!(esql_rows(&node, "ROW x = 3.75 | WHERE TO_INTEGER(x) == 3").is_empty());
}

/// Review findings 506 / 507: GROK and DISSECT parse every value of a
/// multivalued input; GROK has ES's built-in pattern bank (composite log
/// formats with their nested ECS captures).
#[test]
fn grok_and_dissect_multivalues_and_builtin_patterns() {
    let (_dir, node) = index("g506", json!({"properties": {"x": {"type": "keyword"}}}), [("1".to_string(), json!({"x": "a"}))]);
    let multi = |s: &[&str]| Val::Multi(s.iter().map(|x| Val::Str(x.to_string())).collect());
    let out = run(&node, "ROW m = [\"GET /a\", \"POST /b\"] | GROK m \"%{WORD:verb} %{NOTSPACE:path}\" | KEEP verb, path", EsqlOptions::default()).unwrap();
    assert_eq!(out.table.rows[0], vec![multi(&["GET", "POST"]), multi(&["/a", "/b"])]);
    let out = run(&node, "ROW m = [\"GET /a\", \"POST /b\"] | DISSECT m \"%{verb} %{path}\" | KEEP verb, path", EsqlOptions::default()).unwrap();
    assert_eq!(out.table.rows[0], vec![multi(&["GET", "POST"]), multi(&["/a", "/b"])]);
    let rows = esql_rows(&node, "ROW m = [\"GET /a\", \"nope\"] | GROK m \"%{WORD:verb} %{NOTSPACE:path}\" | KEEP verb");
    assert_eq!(rows[0][0], Val::Str("GET".into()));
    let log = r#"127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] \"GET /apache_pb.gif HTTP/1.0\" 200 2326"#;
    let q = format!("ROW m = \"{log}\" | GROK m \"%{{HTTPD_COMMONLOG}}\" | KEEP `http.request.method`, `http.response.status_code`, `url.original`");
    let out = run(&node, &q, EsqlOptions::default()).unwrap_or_else(|e| panic!("{q}: {e}"));
    assert_eq!(out.table.rows[0], vec![Val::Str("GET".into()), Val::Int(200), Val::Str("/apache_pb.gif".into())], "{:?}", out.table.cols);
    assert_eq!(out.table.cols[1].ty, "integer");
}

/// Review finding 509: REPLACE expands `$n` / `${name}` capture references
/// (Java's replacement grammar); distinct keys stay apart.
#[test]
fn replace_expands_capture_references() {
    let (_dir, node) = index("r509", json!({"properties": {"code": {"type": "keyword"}}}), [
        ("1".to_string(), json!({"code": "widget-123"})),
        ("2".to_string(), json!({"code": "gadget-456"})),
        ("3".to_string(), json!({"code": "widget-9"})),
    ]);
    let rows = esql_rows(&node, "FROM r509 | EVAL family = REPLACE(code, \"([a-z]+)-[0-9]+\", \"$1\") | STATS c = COUNT(*) BY family | SORT family");
    assert_eq!(rows, vec![vec![Val::Long(1), Val::Str("gadget".into())], vec![Val::Long(2), Val::Str("widget".into())]]);
    let one = |q: &str| esql_rows(&node, q)[0].last().cloned().unwrap();
    assert_eq!(one("ROW s = \"ab-12\" | EVAL r = REPLACE(s, \"(?<w>[a-z]+)-(\\\\d+)\", \"${w}:$2\")"), Val::Str("ab:12".into()));
    assert_eq!(one("ROW s = \"a.b\" | EVAL r = REPLACE(s, \"\\\\.\", \"\\\\$\")"), Val::Str("a$b".into()));
    assert_eq!(one("ROW s = \"ab\" | EVAL r = REPLACE(s, \"a\", \"$9\")"), Val::Null);
}

/// Review finding 513: scalar RLIKE is a Lucene regular expression, as
/// its pushed-down regexp query: `@`, intersection, quoted literals.
#[test]
fn scalar_rlike_is_lucene_regex() {
    let (_dir, node) = index("r513", json!({"properties": {"code": {"type": "keyword"}}}), [("1".to_string(), json!({"code": "foo"}))]);
    for (pat, keep) in [("@", true), ("foo.*&.*o", true), ("f.*&b.*", false), ("\\\"foo\\\"", true), ("~(bar)", true), ("~(foo)", false), ("a|f.o", true), ("fo", false)] {
        let row = format!("ROW code = \"foo\" | WHERE code RLIKE \"{pat}\"");
        assert_eq!(!esql_rows(&node, &row).is_empty(), keep, "{row}");
        // The same predicate after EVAL agrees with the indexed one.
        let indexed = format!("FROM r513 | WHERE code RLIKE \"{pat}\" | KEEP code");
        let computed = format!("FROM r513 | EVAL c2 = CONCAT(code, \"\") | WHERE c2 RLIKE \"{pat}\" | KEEP code");
        assert_eq!(esql_rows(&node, &indexed).len(), esql_rows(&node, &computed).len(), "{pat}");
    }
}

/// Review findings 482 / 483 / 484: the spatial envelope rewrite and
/// filter hoisting keep EVAL's meaning: two shape fields aggregate, a
/// computed envelope filters by its own value, a replaced shape column is
/// read as replaced (row path forced onto the streaming optimization).
#[test]
fn spatial_rewrites_keep_eval_semantics() {
    let (_dir, node) = index("sh482", json!({"properties": {"id": {"type": "keyword"}, "include": {"type": "boolean"}, "shape_a": {"type": "geo_shape"}, "shape_b": {"type": "geo_shape"}}}), [
        ("1".to_string(), json!({"id": "s1", "include": true, "shape_a": "POLYGON ((-0.5 -0.5, 0.5 -0.5, 0.5 0.5, -0.5 0.5, -0.5 -0.5))", "shape_b": "POINT (10 10)"})),
        ("2".to_string(), json!({"id": "s2", "include": false, "shape_a": "POLYGON ((20 20, 21 20, 21 21, 20 21, 20 20))", "shape_b": "POINT (11 11)"})),
    ]);
    let old = super::exec::fast_switches::residual_min(0);
    let two = esql_rows(&node, "FROM sh482 | EVAL a = ST_NPOINTS(ST_ENVELOPE(shape_a)), b = ST_NPOINTS(ST_ENVELOPE(shape_b)) | STATS s = SUM(a), t = SUM(b)");
    let hoisted = esql_rows(&node, "FROM sh482 | EVAL env = ST_ENVELOPE(shape_a) | WHERE ST_INTERSECTS(env, TO_GEOSHAPE(\"BBOX (-1, 1, 1, -1)\")) | KEEP id");
    let replaced = esql_rows(&node, "FROM sh482 | EVAL shape_a = CASE(include, shape_a, TO_GEOSHAPE(null)) | EVAL n = ST_NPOINTS(ST_ENVELOPE(shape_a)) | STATS included = COUNT(n)");
    super::exec::fast_switches::residual_min(old);
    assert_eq!(two.len(), 1, "{two:?}");
    assert_eq!(ids(hoisted), vec!["s1"]);
    assert_eq!(replaced, vec![vec![Val::Long(1)]]);
    // The row path agrees.
    super::exec::fast_switches::disabled(true);
    let two_rows = esql_rows(&node, "FROM sh482 | EVAL a = ST_NPOINTS(ST_ENVELOPE(shape_a)), b = ST_NPOINTS(ST_ENVELOPE(shape_b)) | STATS s = SUM(a), t = SUM(b)");
    super::exec::fast_switches::disabled(false);
    assert_eq!(two, two_rows);
}

// ------------------------------------------------------------ schema cache

fn col_types(node: &Node, q: &str) -> Vec<(String, String)> {
    let out = run(node, q, EsqlOptions::default()).unwrap_or_else(|e| panic!("{q}: {e}"));
    out.table.cols.iter().map(|c| (c.name.clone(), c.ty.clone())).collect()
}

/// The cached schema resolution is never stale: a mapping update, a new or
/// deleted index and an alias move between two queries are seen at once.
#[test]
fn schema_resolution_sees_every_metadata_change_at_once() {
    let (_dir, node) = index("sc", json!({"properties": {"a": {"type": "long"}}}), [("1".to_string(), json!({"a": 1}))]);
    let q = "FROM sc* | LIMIT 1";
    assert_eq!(col_types(&node, q), vec![("a".to_string(), "long".to_string())]);
    assert_eq!(col_types(&node, q), vec![("a".to_string(), "long".to_string())]);
    // A mapping update.
    node.get_index("sc").unwrap().put_mapping(&json!({"properties": {"b": {"type": "keyword"}}})).unwrap();
    assert_eq!(col_types(&node, q), vec![("a".to_string(), "long".to_string()), ("b".to_string(), "keyword".to_string())]);
    // A dynamic mapping update from indexing.
    let svc = node.get_index("sc").unwrap();
    for r in svc.write(vec![WriteReq::Index { id: Some("2".into()), source: json!({"c": true}).to_string(), routing: None, cond: Default::default() }], false) {
        r.unwrap();
    }
    svc.refresh().unwrap();
    assert!(col_types(&node, q).contains(&("c".to_string(), "boolean".to_string())));
    // A new index with a conflicting type, then its deletion.
    add_index(&node, "sc2", json!({"properties": {"a": {"type": "keyword"}}}), [("1".to_string(), json!({"a": "x"}))]);
    assert!(col_types(&node, q).iter().any(|(n, t)| n == "a" && t.starts_with("unsupported")), "{:?}", col_types(&node, q));
    node.delete_index("sc2").unwrap();
    assert!(col_types(&node, q).contains(&("a".to_string(), "long".to_string())));
    // An alias moved to another index.
    add_index(&node, "other", json!({"properties": {"z": {"type": "double"}}}), [("1".to_string(), json!({"z": 1.5}))]);
    node.update_aliases(&[json!({"add": {"index": "sc", "alias": "al"}})]).unwrap();
    assert!(col_types(&node, "FROM al | LIMIT 1").iter().any(|(n, _)| n == "a"));
    node.update_aliases(&[json!({"remove": {"index": "sc", "alias": "al"}}), json!({"add": {"index": "other", "alias": "al"}})]).unwrap();
    assert_eq!(col_types(&node, "FROM al | LIMIT 1"), vec![("z".to_string(), "double".to_string())]);
    // A deleted and recreated index of the same name.
    node.delete_index("other").unwrap();
    add_index(&node, "other", json!({"properties": {"y": {"type": "keyword"}}}), [("1".to_string(), json!({"y": "v"}))]);
    assert_eq!(col_types(&node, "FROM other | LIMIT 1"), vec![("y".to_string(), "keyword".to_string())]);
}
