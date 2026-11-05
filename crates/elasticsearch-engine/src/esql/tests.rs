//! ES|QL unit tests (node-less pipelines).

use super::exec::run_local;
use super::parse::Params;
use super::Val;

fn run(q: &str) -> super::Table {
    run_local(q, &Params::default()).unwrap_or_else(|e| panic!("{q}: {e}"))
}

fn names(t: &super::Table) -> Vec<String> {
    t.cols.iter().map(|c| c.name.clone()).collect()
}

#[test]
fn row_eval_where_sort_limit() {
    let t = run("ROW a = [3, 1, 2] | MV_EXPAND a | EVAL b = a * 2, c = CONCAT(\"x\", TO_STRING(a)) | WHERE b > 2 | SORT a DESC | LIMIT 5");
    assert_eq!(names(&t), vec!["a", "b", "c"]);
    assert_eq!(t.rows, vec![vec![Val::Int(3), Val::Int(6), Val::Str("x3".into())], vec![Val::Int(2), Val::Int(4), Val::Str("x2".into())]]);
    assert_eq!(t.cols[1].ty, "integer");
    assert_eq!(t.cols[2].ty, "keyword");
}

#[test]
fn abs_is_a_known_function() {
    // ES 9.5.5 defines ABS in the separate `x-pack-esql-function-math`
    // definitions, which the generated table had missed: "Unknown function [abs]".
    let t = run("ROW i = ABS(-3), l = ABS(-5000000000), d = ABS(-2.5), n = ABS(null)");
    assert_eq!(t.rows, vec![vec![Val::Int(3), Val::Long(5000000000), Val::Double(2.5), Val::Null]]);
    assert_eq!(t.cols[..3].iter().map(|c| c.ty.as_str()).collect::<Vec<_>>(), vec!["integer", "long", "double"]);
    let e = run_local("ROW x = ABS()", &Params::default()).err().expect("arity");
    assert!(e.reason.contains("expects exactly one argument"), "{}", e.reason);
}

#[test]
fn stats_by() {
    let t = run("ROW k = [\"a\", \"b\", \"a\"], v = 1 | MV_EXPAND k | STATS n = COUNT(*), s = SUM(v), avg = AVG(v) BY k | SORT k");
    assert_eq!(names(&t), vec!["n", "s", "avg", "k"]);
    assert_eq!(t.rows[0], vec![Val::Long(2), Val::Long(2), Val::Double(1.0), Val::Str("a".into())]);
    assert_eq!(t.cols[0].ty, "long");
    let t = run("ROW x = [1, 2, 3, 4] | MV_EXPAND x | STATS MAX(x), m = MEDIAN(x), p = PERCENTILE(x, 50), c = COUNT_DISTINCT(x), v = VALUES(x)");
    assert_eq!(names(&t), vec!["MAX(x)", "m", "p", "c", "v"]);
    assert_eq!(t.rows[0][0], Val::Int(4));
    assert_eq!(t.rows[0][1], Val::Double(2.5));
    assert_eq!(t.rows[0][3], Val::Long(4));
}

#[test]
fn functions_and_null_semantics() {
    let t = run("ROW s = \"Hello World\", n = null, d = TO_DATETIME(\"2024-03-15T10:20:30Z\") | EVAL l = LENGTH(s), u = TO_UPPER(s), sub = SUBSTRING(s, 1, 5), c = COALESCE(n, 7), cs = CASE(l > 5, \"long\", \"short\"), y = DATE_EXTRACT(\"year\", d), t = DATE_TRUNC(1 month, d), plus = d + 1 day, isn = n IS NULL, lk = s LIKE \"Hello*\", r = ROUND(3.14159, 2)");
    let row = &t.rows[0];
    let col = |n: &str| row[t.cols.iter().position(|c| c.name == n).unwrap()].clone();
    assert_eq!(col("l"), Val::Int(11));
    assert_eq!(col("u"), Val::Str("HELLO WORLD".into()));
    assert_eq!(col("sub"), Val::Str("Hello".into()));
    assert_eq!(col("c"), Val::Int(7));
    assert_eq!(col("cs"), Val::Str("long".into()));
    assert_eq!(col("y"), Val::Long(2024));
    assert_eq!(col("t").to_json(), serde_json::json!("2024-03-01T00:00:00.000Z"));
    assert_eq!(col("plus").to_json(), serde_json::json!("2024-03-16T10:20:30.000Z"));
    assert_eq!(col("isn"), Val::Bool(true));
    assert_eq!(col("lk"), Val::Bool(true));
    assert_eq!(col("r"), Val::Double(3.14));
}

#[test]
fn dissect_grok_keep_drop_rename() {
    let t = run("ROW m = \"2024-01-01 ERROR disk full\" | DISSECT m \"%{date} %{level} %{msg}\" | KEEP level, msg | RENAME level AS lvl");
    assert_eq!(names(&t), vec!["lvl", "msg"]);
    assert_eq!(t.rows[0], vec![Val::Str("ERROR".into()), Val::Str("disk full".into())]);
    let t = run("ROW m = \"took 42 ms from 10.0.0.1\" | GROK m \"took %{NUMBER:took:int} ms from %{IP:ip}\" | DROP m");
    assert_eq!(names(&t), vec!["took", "ip"]);
    assert_eq!(t.rows[0], vec![Val::Int(42), Val::Str("10.0.0.1".into())]);
}

#[test]
fn errors() {
    let e = run_local("ROW a = 1 | EVAL b = nope + 1", &Params::default()).unwrap_err();
    assert_eq!(e.kind, "verification_exception");
    assert!(e.reason.contains("Unknown column [nope]"), "{}", e.reason);
    let e = run_local("ROW a = 1 | FOO", &Params::default()).unwrap_err();
    assert_eq!(e.kind, "parsing_exception");
    let e = run_local("ROW a = 1 | EVAL b = nofn(a)", &Params::default()).unwrap_err();
    assert!(e.reason.contains("Unknown function [nofn]"), "{}", e.reason);
}

#[test]
fn params() {
    let p = Params::from_json(Some(&serde_json::json!([10, {"name": "x"}]))).unwrap();
    let t = run_local("ROW a = ? | EVAL b = ?name", &p).unwrap();
    assert_eq!(t.rows[0], vec![Val::Int(10), Val::Str("x".into())]);
}

#[test]
fn template_render() {
    let v = crate::template::render_to_json("{\"size\": {{size}}{{^size}}10{{/size}}, \"q\": {{#toJson}}q{{/toJson}}}", &serde_json::json!({"q": {"match_all": {}}})).unwrap();
    assert_eq!(v, serde_json::json!({"size": 10, "q": {"match_all": {}}}));
}

#[test]
fn multivalue_keys_preserve_nested_boundaries() {
    let a = Val::Multi(vec![Val::Str("x\u{1}sy".into()), Val::Str("z".into())]);
    let b = Val::Multi(vec![Val::Str("x".into()), Val::Str("y\u{1}sz".into())]);
    assert_ne!(a.key(), b.key());
    assert_eq!(Val::Multi(vec![Val::Int(7)]).key(), Val::Multi(vec![Val::Long(7)]).key());
    assert_ne!(Val::Multi(vec![]).key(), Val::Multi(vec![Val::Str(String::new())]).key());
}

/// ES 9.5.5 `spatial-jts.csv-spec` ST_SIMPLIFY cases through the pipeline:
/// the result keeps the input's spatial type; null in, null out; the
/// tolerance must be a non-negative constant (`StSimplify.getInputTolerance`).
#[test]
fn st_simplify_follows_es() {
    let sq = "TO_GEOSHAPE(\"POLYGON((0 0, 1 0.1, 2 0, 2 2, 1 1.9, 0 2, 0 0))\")";
    // stSimplifyFoldableGeometry / stSimplifyEmptySimplification / stSimplifyWithIntegerTolerance.
    let t = run(&format!("ROW a = ST_SIMPLIFY({sq}, 0.2), b = ST_SIMPLIFY({sq}, 2.0), c = ST_SIMPLIFY({sq}, 2)"));
    assert_eq!(
        t.rows,
        vec![vec![Val::Str("POLYGON ((0.0 0.0, 2.0 0.0, 2.0 2.0, 0.0 2.0, 0.0 0.0))".into()), Val::Str("POLYGON EMPTY".into()), Val::Str("POLYGON EMPTY".into())]]
    );
    assert_eq!(t.cols.iter().map(|c| c.ty.as_str()).collect::<Vec<_>>(), vec!["geo_shape"; 3]);
    // stSimplifyCartesianShape.
    let t = run("ROW wkt = [\"POINT(97.11 75.53)\", \"POLYGON((0 0, 1 0.1, 2 0, 2 2, 1 1.9, 0 2, 0 0))\"] | MV_EXPAND wkt | EVAL geom = TO_CARTESIANSHAPE(wkt) | EVAL result = ST_SIMPLIFY(geom, 0.2) | KEEP result");
    assert_eq!(t.rows, vec![vec![Val::Str("POINT (97.11 75.53)".into())], vec![Val::Str("POLYGON ((0.0 0.0, 2.0 0.0, 2.0 2.0, 0.0 2.0, 0.0 0.0))".into())]]);
    assert_eq!(t.cols[0].ty, "cartesian_shape");
    // stSimplifyNull.
    let t = run("ROW geo_shape = NULL | EVAL result = ST_SIMPLIFY(geo_shape, 2.0) | KEEP result");
    assert_eq!(t.rows, vec![vec![Val::Null]]);
    // The Rally st-simplify-count-esql shape: collapsed polygons have no points.
    let t = run("ROW w = [\"POLYGON((2.3 48.8, 2.3002 48.8, 2.3002 48.8002, 2.3 48.8002, 2.3 48.8))\", \"POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))\", \"LINESTRING(0 0, 0.001 0.0001, 1 0)\"] | MV_EXPAND w | EVAL shape = TO_GEOSHAPE(w) | EVAL simplified = ST_SIMPLIFY(shape, 0.01), points = ST_NPOINTS(simplified) | WHERE points > 0 | STATS count = COUNT(*) BY points | SORT points");
    assert_eq!(t.rows, vec![vec![Val::Long(1), Val::Int(2)], vec![Val::Long(1), Val::Int(5)]]);
    let e = run_local(&format!("ROW a = ST_SIMPLIFY({sq}, -0.5)"), &Params::default()).err().expect("negative tolerance");
    assert_eq!((e.status, e.kind.as_str(), e.reason.as_str()), (400, "illegal_argument_exception", "tolerance must not be negative"));
}

/// ES 9.5.5 `ToGeohash|ToGeotile|ToGeohex`: longs are cell longs, strings
/// print canonically, unparsable strings are null (`spatial-grid.csv-spec`
/// `toGeohashToLong`, `geohashLongToString`); `ST_NPOINTS` of a `BBOX` is 4
/// (`GeometryPointCountVisitor.visit(Rectangle)`).
#[test]
fn grid_conversions_and_npoints_follow_es() {
    let t = run("ROW h = TO_GEOHASH(13686180::long), s = TO_STRING(TO_GEOHASH(13686180::long)), h2 = TO_GEOHASH(\"u3bu\"), l = TO_LONG(TO_GEOHASH(\"u3bu\"))");
    assert_eq!(t.rows, vec![vec![Val::Str("u3bu".into()), Val::Str("u3bu".into()), Val::Str("u3bu".into()), Val::Long(13686180)]]);
    assert_eq!(t.cols[0].ty, "geohash");
    let tile = (6i64 << 58) | (32 << 29) | 22;
    let t = run(&format!("ROW a = TO_GEOTILE(\"06/032/22\"), b = TO_GEOTILE({tile}::long), c = TO_GEOHEX(\"811FBFFFFFFFFFF\"), d = TO_GEOHEX(581672437419081727::long), e = TO_GEOTILE(\"bad\"), f = TO_GEOHEX(\"zz\")"));
    assert_eq!(
        t.rows,
        vec![vec![Val::Str("6/32/22".into()), Val::Str("6/32/22".into()), Val::Str("811fbffffffffff".into()), Val::Str("81283ffffffffff".into()), Val::Null, Val::Null]]
    );
    let t = run("ROW n = ST_NPOINTS(TO_GEOSHAPE(\"BBOX (0, 2, 2, 0)\")), p = ST_NPOINTS(TO_GEOSHAPE(\"POLYGON((0 0, 1 0, 1 1, 0 0))\"))");
    assert_eq!(t.rows, vec![vec![Val::Int(4), Val::Int(4)]]);
}
