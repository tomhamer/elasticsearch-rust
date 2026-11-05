//! Usage telemetry of an ES|QL query (`GET /_xpack/usage` → `esql`),
//! as ES's `Verifier.gatherMetrics` collects it: the commands (features)
//! and the distinct functions of each successfully verified query.

use super::parse::{Cmd, Expr, Field};
use std::collections::BTreeSet;

/// Function names `esql.functions` reports (`EsqlFunctionRegistry.functions()`
/// of a release build; one entry per function, aliases excluded).
pub const FUNCTIONS: &[&str] = &[
    "abs",
    "absent",
    "absent_over_time",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "avg",
    "avg_over_time",
    "bit_length",
    "bucket",
    "byte_length",
    "case",
    "categorize",
    "cbrt",
    "ceil",
    "chicken",
    "chunk",
    "cidr_match",
    "clamp",
    "clamp_max",
    "clamp_min",
    "coalesce",
    "concat",
    "contains",
    "copy_sign",
    "cos",
    "cosh",
    "count",
    "count_distinct",
    "count_distinct_over_time",
    "count_over_time",
    "date_diff",
    "date_extract",
    "date_format",
    "date_parse",
    "date_trunc",
    "date_unit_count",
    "day_name",
    "decay",
    "delta",
    "deriv",
    "e",
    "earliest",
    "embedding",
    "ends_with",
    "exp",
    "field_extract",
    "first",
    "first_over_time",
    "floor",
    "from_base64",
    "greatest",
    "hash",
    "hypot",
    "idelta",
    "increase",
    "ip_prefix",
    "irate",
    "json_extract",
    "knn",
    "kql",
    "last",
    "last_over_time",
    "latest",
    "least",
    "left",
    "length",
    "locate",
    "log",
    "log10",
    "ltrim",
    "match",
    "match_phrase",
    "max",
    "max_over_time",
    "md5",
    "median",
    "median_absolute_deviation",
    "min",
    "min_over_time",
    "month_name",
    "mv_append",
    "mv_avg",
    "mv_concat",
    "mv_contains",
    "mv_count",
    "mv_dedupe",
    "mv_difference",
    "mv_first",
    "mv_intersection",
    "mv_intersects",
    "mv_last",
    "mv_max",
    "mv_median",
    "mv_median_absolute_deviation",
    "mv_min",
    "mv_percentile",
    "mv_pseries_weighted_sum",
    "mv_slice",
    "mv_sort",
    "mv_sum",
    "mv_union",
    "mv_zip",
    "network_direction",
    "now",
    "percentile",
    "percentile_over_time",
    "pi",
    "pow",
    "present",
    "present_over_time",
    "qstr",
    "range_contains",
    "range_intersects",
    "range_max",
    "range_min",
    "range_within",
    "rate",
    "repeat",
    "replace",
    "reverse",
    "right",
    "round",
    "round_to",
    "rtrim",
    "sample",
    "scalb",
    "score",
    "sha1",
    "sha256",
    "signum",
    "sin",
    "sinh",
    "space",
    "sparkline",
    "split",
    "sqrt",
    "st_buffer",
    "st_centroid_agg",
    "st_contains",
    "st_difference",
    "st_dimension",
    "st_disjoint",
    "st_distance",
    "st_envelope",
    "st_extent_agg",
    "st_geohash",
    "st_geohex",
    "st_geometrytype",
    "st_geotile",
    "st_intersection",
    "st_intersects",
    "st_isempty",
    "st_npoints",
    "st_simplify",
    "st_simplifypreservetopology",
    "st_symdifference",
    "st_union",
    "st_within",
    "st_x",
    "st_xmax",
    "st_xmin",
    "st_y",
    "st_ymax",
    "st_ymin",
    "starts_with",
    "std_dev",
    "stddev_over_time",
    "substring",
    "sum",
    "sum_over_time",
    "tan",
    "tanh",
    "tau",
    "tbucket",
    "text_embedding",
    "to_aggregate_metric_double",
    "to_base64",
    "to_boolean",
    "to_cartesianpoint",
    "to_cartesianshape",
    "to_counter",
    "to_date_nanos",
    "to_date_range",
    "to_dateperiod",
    "to_datetime",
    "to_degrees",
    "to_dense_vector",
    "to_double",
    "to_exponential_histogram",
    "to_gauge",
    "to_geohash",
    "to_geohex",
    "to_geopoint",
    "to_geoshape",
    "to_geotile",
    "to_integer",
    "to_ip",
    "to_long",
    "to_lower",
    "to_radians",
    "to_range",
    "to_string",
    "to_tdigest",
    "to_text",
    "to_timeduration",
    "to_unsigned_long",
    "to_upper",
    "to_version",
    "top",
    "top_snippets",
    "trange",
    "trim",
    "url_decode",
    "url_encode",
    "url_encode_component",
    "v_cosine",
    "v_dot_product",
    "v_hamming",
    "v_l1_norm",
    "v_l2_norm",
    "values",
    "variance",
    "variance_over_time",
    "weighted_avg",
    "without",
];

/// Feature names `esql.features` reports (`FeatureMetric`).
pub const FEATURES: &[&str] = &[
    "change_point",
    "completion",
    "dedup",
    "dissect",
    "drop",
    "enrich",
    "eval",
    "explain",
    "external",
    "fork",
    "from",
    "fuse",
    "grok",
    "highlight",
    "in_subquery",
    "inline_stats",
    "ip_location",
    "keep",
    "limit",
    "limit_by",
    "lookup",
    "lookup_join",
    "lookup_join_on_expression",
    "metrics_info",
    "mmr",
    "mv_expand",
    "promql",
    "registered_domain",
    "rename",
    "rerank",
    "row",
    "sample",
    "show",
    "sort",
    "stats",
    "subquery",
    "ts",
    "ts_info",
    "uri_parts",
    "user_agent",
    "view",
    "where",
];

/// Query settings `esql.settings` counts.
pub const SETTINGS: &[&str] = &["approximation", "column_metadata", "time_zone", "unmapped_fields"];

/// `esql.took` buckets: (name, upper bound in millis).
pub const TOOK: &[(&str, u64)] = &[
    ("lt_10ms", 10),
    ("lt_100ms", 100),
    ("lt_1s", 1000),
    ("lt_10s", 10_000),
    ("lt_1m", 60_000),
    ("lt_10m", 600_000),
    ("lt_1h", 3_600_000),
    ("lt_10h", 36_000_000),
    ("lt_1d", 86_400_000),
    ("gt_1d", u64::MAX),
];

/// The registered name of a function as the query spells it (aliases resolved).
fn canonical(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    let n = match n.as_str() {
        "bin" => "bucket",
        "to_aggregatemetricdouble" => "to_aggregate_metric_double",
        "to_bool" => "to_boolean",
        "to_dt" => "to_datetime",
        "to_datenanos" => "to_date_nanos",
        "to_dbl" => "to_double",
        "to_int" => "to_integer",
        "to_str" => "to_string",
        "to_ulong" | "to_ul" => "to_unsigned_long",
        "to_ver" => "to_version",
        o => o,
    };
    FUNCTIONS.iter().copied().find(|f| *f == n)
}

/// The conversion function behind a `::type` cast.
fn cast_function(ty: &str) -> Option<&'static str> {
    Some(match ty.to_ascii_lowercase().as_str() {
        "bool" | "boolean" => "to_boolean",
        "datetime" | "date" => "to_datetime",
        "date_nanos" => "to_date_nanos",
        "double" => "to_double",
        "int" | "integer" => "to_integer",
        "long" => "to_long",
        "ip" => "to_ip",
        "keyword" | "text" | "string" => "to_string",
        "unsigned_long" => "to_unsigned_long",
        "version" => "to_version",
        "geo_point" => "to_geopoint",
        "geo_shape" => "to_geoshape",
        "cartesian_point" => "to_cartesianpoint",
        "cartesian_shape" => "to_cartesianshape",
        "date_period" => "to_dateperiod",
        "time_duration" => "to_timeduration",
        _ => return None,
    })
}

fn expr_functions(e: &Expr, out: &mut BTreeSet<&'static str>) {
    match e {
        Expr::Call(n, args, _) => {
            if let Some(f) = canonical(n) {
                out.insert(f);
            }
            // Time-series functions are translated to the aggregations computing them.
            let base = match n.as_str() {
                "AVG_OVER_TIME" => Some("avg"),
                "SUM_OVER_TIME" => Some("sum"),
                "MIN_OVER_TIME" => Some("min"),
                "MAX_OVER_TIME" => Some("max"),
                "COUNT_OVER_TIME" => Some("count"),
                "COUNT_DISTINCT_OVER_TIME" => Some("count_distinct"),
                "STDDEV_OVER_TIME" => Some("std_dev"),
                "VARIANCE_OVER_TIME" | "STDVAR_OVER_TIME" => Some("variance"),
                "PERCENTILE_OVER_TIME" => Some("percentile"),
                _ => None,
            };
            if let Some(b) = base.and_then(canonical) {
                out.insert(b);
            }
            args.iter().for_each(|a| expr_functions(a, out));
        }
        Expr::Cast(a, t) => {
            if let Some(f) = cast_function(t) {
                out.insert(f);
            }
            expr_functions(a, out);
        }
        Expr::Bin(_, a, b) => {
            expr_functions(a, out);
            expr_functions(b, out);
        }
        Expr::Neg(a) | Expr::Not(a) | Expr::IsNull(a, _) | Expr::Like(a, ..) => expr_functions(a, out),
        Expr::In(a, l, _) => {
            expr_functions(a, out);
            l.iter().for_each(|x| expr_functions(x, out));
        }
        Expr::Between(a, b, c, _) => {
            expr_functions(a, out);
            expr_functions(b, out);
            expr_functions(c, out);
        }
        _ => {}
    }
}

fn fields_functions(fields: &[Field], out: &mut BTreeSet<&'static str>) {
    for f in fields {
        expr_functions(&f.expr, out);
        if let Some(w) = &f.filter {
            expr_functions(w, out);
        }
    }
}

fn collect(cmds: &[Cmd], features: &mut BTreeSet<&'static str>, functions: &mut BTreeSet<&'static str>) {
    for c in cmds {
        let feature = match c {
            Cmd::From { ts, subqueries, .. } => {
                if !subqueries.is_empty() {
                    features.insert("subquery");
                    subqueries.iter().for_each(|b| collect(b, features, functions));
                }
                if *ts {
                    "ts"
                } else {
                    "from"
                }
            }
            Cmd::Row(f) => {
                fields_functions(f, functions);
                "row"
            }
            Cmd::ShowInfo => "show",
            Cmd::Promql(_) => "promql",
            Cmd::Where(e, _) => {
                expr_functions(e, functions);
                "where"
            }
            Cmd::Eval(f) => {
                fields_functions(f, functions);
                "eval"
            }
            Cmd::Keep(..) => "keep",
            Cmd::Drop(..) => "drop",
            Cmd::Rename(..) => "rename",
            Cmd::Sort(keys) => {
                keys.iter().for_each(|k| expr_functions(&k.expr, functions));
                "sort"
            }
            Cmd::Limit(_) => "limit",
            Cmd::LimitBy(_, keys) => {
                keys.iter().for_each(|k| expr_functions(k, functions));
                "limit_by"
            }
            Cmd::SqlPivot { aggregate, .. } => { expr_functions(aggregate, functions); "pivot" }
            Cmd::Stats { aggs, by, inline, .. } => {
                fields_functions(aggs, functions);
                fields_functions(by, functions);
                if *inline { "inline_stats" } else { "stats" }
            }
            Cmd::MvExpand(..) => "mv_expand",
            Cmd::Dissect { field, .. } => {
                expr_functions(field, functions);
                "dissect"
            }
            Cmd::Grok { field, .. } => {
                expr_functions(field, functions);
                "grok"
            }
            Cmd::LookupJoin { .. } => "lookup_join",
            Cmd::Info(true, _) => "ts_info",
            Cmd::Info(false, _) => "metrics_info",
            Cmd::Enrich { .. } => "enrich",
            Cmd::Sample(_) => "sample",
            Cmd::Fork(branches) => {
                branches.iter().for_each(|b| collect(b, features, functions));
                "fork"
            }
        };
        features.insert(feature);
    }
}

/// `(features, functions)` a query uses, each counted once per query.
pub fn usage(cmds: &[Cmd]) -> (BTreeSet<&'static str>, BTreeSet<&'static str>) {
    let (mut features, mut functions) = (BTreeSet::new(), BTreeSet::new());
    collect(cmds, &mut features, &mut functions);
    (features, functions)
}
