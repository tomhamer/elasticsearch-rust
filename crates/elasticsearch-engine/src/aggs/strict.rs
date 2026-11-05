//! Strict structure checks for the `aggs` section
//! (`AggregatorFactories.parseAggregators` + each builder's parser) with
//! ES's exception types, messages and token positions (`jsonpos` hints).

use crate::error::{EsError, Result};
use crate::jsonpos::token_name;
use serde_json::{Map, Value};

/// Every aggregation name ES 8.19 registers (candidates of `did you mean`).
pub const AGG_NAMES: &[&str] = &[
    "adjacency_matrix", "auto_date_histogram", "avg", "avg_bucket", "boxplot", "bucket_correlation", "bucket_count_ks_test", "bucket_script", "bucket_selector", "bucket_sort",
    "cardinality", "cartesian_bounds", "cartesian_centroid", "categorize_text", "change_point", "children", "composite", "cumulative_cardinality", "cumulative_sum", "date_histogram", "date_range", "derivative",
    "diversified_sampler", "extended_stats", "extended_stats_bucket", "filter", "filters", "frequent_item_sets", "geo_bounds", "geo_centroid", "geo_distance", "geo_line",
    "geohash_grid", "geohex_grid", "geotile_grid", "global", "histogram", "inference", "ip_prefix", "ip_range", "matrix_stats", "max", "max_bucket", "median_absolute_deviation",
    "min", "min_bucket", "missing", "moving_fn", "moving_percentiles", "multi_terms", "nested", "normalize", "parent", "percentile_ranks", "percentiles", "percentiles_bucket",
    "random_sampler", "range", "rare_terms", "rate", "reverse_nested", "sampler", "scripted_metric", "serial_diff", "significant_terms", "significant_text", "stats",
    "stats_bucket", "string_stats", "sum", "sum_bucket", "t_test", "terms", "time_series", "top_hits", "top_metrics", "value_count", "variable_width_histogram", "weighted_avg",
];

const VS_COMMON: &[&str] = &["field", "script", "missing", "format", "time_zone", "value_type"];

/// How a builder reports an unknown field.
enum Style {
    /// ObjectParser: `x_content_parse_exception "[l:c] [type] unknown field [f]"`.
    Object(&'static str),
    /// `Unknown key for a VALUE_NUMBER in [name]: [f].` (hand-written loops).
    UnknownKeyIn,
    /// `Unknown key for a VALUE_NUMBER in [f].`
    UnknownKey,
    /// `Unexpected token VALUE_NUMBER in [name].`
    UnexpectedIn,
    /// `Unexpected token VALUE_NUMBER [f] in [name].`
    UnexpectedFieldIn,
    /// `Unexpected token VALUE_NUMBER [f] in [name]` (sibling pipelines, no period).
    UnexpectedFieldInNoDot,
    /// `Unsupported property "f" for aggregation "name` (sampler).
    Sampler,
    /// `Expected [FIELD_NAME] under a [START_OBJECT], but got a [VALUE_NUMBER] in [name]` (global).
    Global,
    /// `Expected [START_OBJECT] under [f], but got a [VALUE_NUMBER] in [name]` (filter: the body is a query).
    Filter,
}

fn spec(t: &str) -> Option<(Style, &'static [&'static str])> {
    use Style::*;
    Some(match t {
        "terms" => (Object("terms"), &["field", "script", "missing", "format", "value_type", "size", "shard_size", "min_doc_count", "shard_min_doc_count", "show_term_doc_count_error", "execution_hint", "order", "include", "exclude", "collect_mode"]),
        "rare_terms" => (Object("rare_terms"), &["field", "script", "missing", "format", "value_type", "max_doc_count", "precision", "include", "exclude"]),
        "multi_terms" => (Object("multi_terms"), &["terms", "size", "shard_size", "min_doc_count", "shard_min_doc_count", "show_term_doc_count_error", "order", "collect_mode"]),
        "significant_terms" => (Object("significant_terms"), &["field", "script", "missing", "format", "value_type", "size", "shard_size", "min_doc_count", "shard_min_doc_count", "execution_hint", "background_filter", "include", "exclude", "jlh", "mutual_information", "chi_square", "gnd", "percentage", "script_heuristic", "p_value"]),
        "significant_text" => (Object("significant_text"), &["field", "size", "shard_size", "min_doc_count", "shard_min_doc_count", "background_filter", "include", "exclude", "source_fields", "filter_duplicate_text", "jlh", "mutual_information", "chi_square", "gnd", "percentage", "script_heuristic", "p_value"]),
        "avg" | "sum" | "min" | "max" | "value_count" | "stats" | "geo_centroid" | "geo_bounds" | "boxplot" | "median_absolute_deviation" | "string_stats" => (Object(leak(t)), &["field", "script", "missing", "format", "time_zone", "value_type", "wrap_longitude", "compression", "show_distribution", "execution_hint"]),
        "extended_stats" => (Object("extended_stats"), &["field", "script", "missing", "format", "time_zone", "value_type", "sigma"]),
        "cardinality" => (Object("cardinality"), &["field", "script", "missing", "format", "time_zone", "value_type", "precision_threshold", "rehash", "execution_hint"]),
        "percentiles" => (Object("percentiles"), &["field", "script", "missing", "format", "time_zone", "value_type", "percents", "keyed", "tdigest", "hdr"]),
        "percentile_ranks" => (Object("percentile_ranks"), &["field", "script", "missing", "format", "time_zone", "value_type", "values", "keyed", "tdigest", "hdr"]),
        "weighted_avg" => (Object("weighted_avg"), &["value", "weight", "format", "value_type"]),
        "t_test" => (Object("t_test"), &["a", "b", "type"]),
        "rate" => (Object("rate"), &["field", "script", "missing", "format", "value_type", "unit", "mode"]),
        "top_metrics" => (Object("top_metrics"), &["metrics", "sort", "size"]),
        "scripted_metric" => (Object("scripted_metric"), &["init_script", "map_script", "combine_script", "reduce_script", "params"]),
        "histogram" => (Object("histogram"), &["field", "script", "missing", "format", "time_zone", "value_type", "interval", "offset", "min_doc_count", "keyed", "order", "extended_bounds", "hard_bounds"]),
        "date_histogram" => (Object("date_histogram"), &["field", "script", "missing", "format", "time_zone", "value_type", "interval", "calendar_interval", "fixed_interval", "offset", "min_doc_count", "keyed", "order", "extended_bounds", "hard_bounds", "downsampled_results_offset"]),
        "auto_date_histogram" => (Object("auto_date_histogram"), &["field", "script", "missing", "format", "time_zone", "value_type", "buckets", "minimum_interval"]),
        "variable_width_histogram" => (Object("variable_width_histogram"), &["field", "script", "missing", "format", "time_zone", "value_type", "buckets", "shard_size", "initial_buffer"]),
        "range" => (Object("range"), &["field", "script", "missing", "format", "time_zone", "value_type", "ranges", "keyed"]),
        "date_range" => (Object("date_range"), &["field", "script", "missing", "format", "time_zone", "value_type", "ranges", "keyed"]),
        "ip_range" => (Object("ip_range"), &["field", "script", "missing", "format", "value_type", "ranges", "keyed"]),
        "geo_distance" => (Object("geo_distance"), &["field", "script", "missing", "format", "value_type", "ranges", "keyed", "origin", "unit", "distance_type"]),
        "geohash_grid" | "geotile_grid" | "geohex_grid" => (Object(leak(t)), &["field", "script", "missing", "format", "value_type", "precision", "size", "shard_size", "bounds"]),
        "ip_prefix" => (Object("ip_prefix"), &["field", "script", "missing", "format", "value_type", "prefix_length", "is_ipv6", "append_prefix_length", "keyed", "min_doc_count"]),
        "missing" => (Object("missing"), &["field", "script", "missing", "format", "value_type"]),
        "composite" => (Object("composite"), &["sources", "size", "after"]),
        "diversified_sampler" => (Object("diversified_sampler"), &["field", "script", "missing", "format", "value_type", "shard_size", "max_docs_per_value", "execution_hint"]),
        "random_sampler" => (Object("random_sampler"), &["probability", "seed", "shard_seed"]),
        "adjacency_matrix" => (Object("adjacency_matrix"), &["filters", "separator"]),
        "bucket_sort" => (Object("bucket_sort"), &["sort", "from", "size", "gap_policy"]),
        "bucket_script" => (Object("bucket_script"), &["buckets_path", "script", "format", "gap_policy"]),
        "cumulative_sum" => (Object("cumulative_sum"), &["buckets_path", "format"]),
        "cumulative_cardinality" => (Object("cumulative_cardinality"), &["buckets_path", "format"]),
        "moving_fn" => (Object("moving_fn"), &["buckets_path", "script", "format", "gap_policy", "window", "shift"]),
        "moving_percentiles" => (Object("moving_percentiles"), &["buckets_path", "window", "shift"]),
        "normalize" => (Object("normalize"), &["buckets_path", "format", "method"]),
        "inference" => (Object("inference"), &["model_id", "inference_config", "buckets_path"]),
        "categorize_text" => (Object("categorize_text"), &["field", "max_unique_tokens", "max_matched_tokens", "similarity_threshold", "categorization_filters", "categorization_analyzer", "shard_size", "size", "min_doc_count", "shard_min_doc_count"]),
        "frequent_item_sets" | "frequent_items" => (Object("frequent_item_sets"), &["fields", "minimum_set_size", "minimum_support", "size", "filter", "execution_hint"]),
        "geo_line" => (Object("geo_line"), &["point", "sort", "include_sort", "sort_order", "size"]),
        "time_series" => (Object("time_series"), &["keyed", "size"]),
        "change_point" | "bucket_correlation" | "bucket_count_ks_test" => (Object(leak(t)), &["buckets_path", "function", "alternative", "fractions", "sampling_method"]),
        "filters" => (UnknownKeyIn, &["filters", "other_bucket", "other_bucket_key", "keyed"]),
        "nested" => (UnexpectedIn, &["path"]),
        "reverse_nested" => (UnexpectedIn, &["path"]),
        "children" => (UnexpectedIn, &["type"]),
        "parent" => (UnexpectedIn, &["type"]),
        "top_hits" => (UnknownKey, &["from", "size", "sort", "_source", "stored_fields", "docvalue_fields", "fields", "script_fields", "highlight", "explain", "version", "seq_no_primary_term", "track_scores"]),
        "sampler" => (Sampler, &["shard_size"]),
        "global" => (Global, &[]),
        "filter" => (Filter, &[]),
        "matrix_stats" => (UnexpectedFieldIn, &["fields", "missing", "mode"]),
        "derivative" => (UnexpectedIn, &["buckets_path", "format", "gap_policy", "unit"]),
        "bucket_selector" => (UnexpectedIn, &["buckets_path", "script", "gap_policy"]),
        "serial_diff" => (UnknownKeyIn, &["buckets_path", "format", "gap_policy", "lag"]),
        "avg_bucket" | "sum_bucket" | "max_bucket" | "min_bucket" | "stats_bucket" | "extended_stats_bucket" | "percentiles_bucket" => (UnexpectedFieldInNoDot, &["buckets_path", "format", "gap_policy", "sigma", "percents", "keyed"]),
        _ => return None,
    })
}

fn leak(s: &str) -> &'static str {
    match s {
        "avg" => "avg",
        "sum" => "sum",
        "min" => "min",
        "max" => "max",
        "value_count" => "value_count",
        "stats" => "stats",
        "geo_centroid" => "geo_centroid",
        "geo_bounds" => "geo_bounds",
        "boxplot" => "boxplot",
        "median_absolute_deviation" => "median_absolute_deviation",
        "string_stats" => "string_stats",
        "geohash_grid" => "geohash_grid",
        "geotile_grid" => "geotile_grid",
        "geohex_grid" => "geohex_grid",
        "change_point" => "change_point",
        "bucket_correlation" => "bucket_correlation",
        "bucket_count_ks_test" => "bucket_count_ks_test",
        _ => "aggregation",
    }
}

fn numeric(parser: &str, field: &str, v: &Value, within: Option<&str>) -> Result<()> {
    match v {
        Value::String(s) if s.trim().parse::<f64>().is_err() => Err(EsError::field_failure(parser, field, within, EsError::number_format(s))),
        Value::Object(_) | Value::Array(_) | Value::Null | Value::Bool(_) => Err(EsError::unsupported_type_xc(parser, field, v, within)),
        _ => Ok(()),
    }
}

/// `Expected [START_OBJECT] under [x], but got a [T] in [name]` at the value of `x`.
fn expected_object(name: &str, key: &str, v: &Value) -> EsError {
    EsError::parsing(format!("Expected [START_OBJECT] under [{key}], but got a [{}] in [{name}]", token_name(v))).at_key(key, Some(name)).with_lc()
}

/// Validate an `aggs` object; `parent` is the enclosing aggregation name.
pub fn check_tree(spec_v: &Value, parent: Option<&str>) -> Result<()> {
    let Some(m) = spec_v.as_object() else {
        return Ok(());
    };
    for (name, def) in m {
        if name.is_empty() || name.contains(['[', ']', '>']) {
            return Err(EsError::parsing(format!("Invalid aggregation name [{name}]. Aggregation names can contain any character except '[', ']', and '>'")).at_key(name, parent).cursor("key").with_lc());
        }
        let Some(d) = def.as_object() else {
            return Err(EsError::parsing(format!("Aggregation definition for [{name} starts with a [{}], expected a [START_OBJECT].", token_name(def))).at_key(name, parent).with_lc());
        };
        let mut agg_type: Option<(&String, &Value)> = None;
        let mut subs: Option<&Value> = None;
        for (k, v) in d {
            match k.as_str() {
                "aggs" | "aggregations" => {
                    if !v.is_object() {
                        return Err(expected_object(name, k, v));
                    }
                    if subs.is_some() {
                        return Err(EsError::parsing(format!("Found two sub aggregation definitions under [{name}]")).at_key(k, Some(name)).with_lc());
                    }
                    subs = Some(v);
                }
                "meta" => {
                    if !v.is_object() {
                        return Err(expected_object(name, k, v));
                    }
                }
                _ => {
                    if !v.is_object() {
                        return Err(expected_object(name, k, v));
                    }
                    if let Some((first, _)) = agg_type {
                        return Err(EsError::parsing(format!("Found two aggregation type definitions in [{name}]: [{first}] and [{k}]")).at_key(k, Some(name)).with_lc());
                    }
                    agg_type = Some((k, v));
                }
            }
        }
        let Some((t, body)) = agg_type else {
            return Err(EsError::parsing(format!("Missing definition for aggregation [{name}]")).at_key(name, parent).cursor("end").with_lc());
        };
        let bo = body.as_object().unwrap();
        if !AGG_NAMES.contains(&t.as_str()) && !matches!(t.as_str(), "counted_terms" | "frequent_items" | "inference" | "geo_line" | "exact_knn") {
            return Err(EsError::unknown_named("Unknown aggregation type", t, Some(name), AGG_NAMES));
        }
        check_body(name, t, bo)?;
        const NO_SUBS: &[&str] = &["significant_text", "avg", "sum", "min", "max", "value_count", "stats", "extended_stats", "cardinality", "percentiles", "percentile_ranks", "top_hits", "geo_bounds", "geo_centroid", "scripted_metric", "weighted_avg", "median_absolute_deviation", "boxplot", "string_stats", "t_test", "rate", "top_metrics", "matrix_stats", "avg_bucket", "sum_bucket", "max_bucket", "min_bucket", "stats_bucket", "extended_stats_bucket", "percentiles_bucket", "derivative", "cumulative_sum", "bucket_script", "bucket_selector", "bucket_sort", "moving_fn", "serial_diff", "cumulative_cardinality", "moving_percentiles", "normalize", "inference", "bucket_correlation", "bucket_count_ks_test", "change_point"];
        if let Some(s) = subs {
            if NO_SUBS.contains(&t.as_str()) && s.as_object().is_some_and(|m| !m.is_empty()) {
                return Err(EsError::illegal_argument(format!("Aggregator [{name}] of type [{t}] cannot accept sub-aggregations")));
            }
            check_tree(s, Some(name))?;
        }
    }
    Ok(())
}

fn check_body(name: &str, t: &str, bo: &Map<String, Value>) -> Result<()> {
    let Some((style, allowed)) = spec(t) else { return Ok(()) };
    for (k, v) in bo {
        if allowed.contains(&k.as_str()) {
            continue;
        }
        return Err(match style {
            Style::Object(p) => {
                let e = EsError::unknown_field_xc(p, k, Some(t), allowed);
                // Significance heuristics are named objects: the cause is the registry lookup.
                if matches!(t, "significant_terms" | "significant_text") {
                    e.caused_by(EsError::new(400, "named_object_not_found_exception", format!("unknown field [{k}]")).at_key(k, Some(t)).prefixed("cursor"))
                } else {
                    e
                }
            }
            Style::UnknownKeyIn => EsError::parsing(format!("Unknown key for a {} in [{name}]: [{k}].", token_name(v))).at_key(k, Some(t)).with_lc(),
            Style::UnknownKey => EsError::unknown_key(k, v, Some(t)),
            Style::UnexpectedIn if v.is_string() => EsError::parsing(format!("Unknown key for a {} in [{name}]: [{k}].", token_name(v))).at_key(k, Some(t)).with_lc(),
            Style::UnexpectedIn => EsError::parsing(format!("Unexpected token {} in [{name}].", token_name(v))).at_key(k, Some(t)).with_lc(),
            Style::UnexpectedFieldIn => EsError::parsing(format!("Unexpected token {} [{k}] in [{name}].", token_name(v))).at_key(k, Some(t)).with_lc(),
            Style::UnexpectedFieldInNoDot => EsError::parsing(format!("Unexpected token {} [{k}] in [{name}]", token_name(v))).at_key(k, Some(t)).with_lc(),
            Style::Sampler => EsError::parsing(format!("Unsupported property \"{k}\" for aggregation \"{name}")).at_key(k, Some(t)).with_lc(),
            Style::Global => EsError::parsing(format!("Expected [FIELD_NAME] under a [START_OBJECT], but got a [{}] in [{name}]", token_name(v))).at_key(k, Some(t)).with_lc(),
            Style::Filter => {
                if !v.is_object() {
                    EsError::parsing(format!("Expected [START_OBJECT] under [{k}], but got a [{}] in [{name}]", token_name(v))).at_key(k, Some(name)).with_lc()
                } else {
                    continue;
                }
            }
        });
    }
    // Field type checks ES's ObjectParsers perform.
    if let Style::Object(p) = style {
        if allowed.contains(&"value_type") {
            if let Some(value) = bo.get("value_type") {
                let Some(hint) = value.as_str() else {
                    return Err(EsError::unsupported_type_xc(p, "value_type", value, Some(t)));
                };
                // These are the public value-source hints, not mapping types.
                // In particular, geo_point is a valid mapping but not a hint
                // accepted by ES's ValueType.lenientParse.
                if !matches!(hint, "string" | "double" | "float" | "number" | "numeric" | "long" | "integer" | "short" | "byte" | "date" | "ip" | "boolean") {
                    return Err(EsError::field_failure(p, "value_type", Some(t), EsError::illegal_argument(format!("Unknown value type [{hint}]"))));
                }
            }
        }
        for k in ["size", "shard_size", "min_doc_count", "shard_min_doc_count", "interval", "precision_threshold", "sigma", "probability", "window", "shift", "lag", "from", "max_doc_count", "buckets", "compression"] {
            if let Some(v) = bo.get(k) {
                if k == "precision_threshold" {
                    if let Value::String(s) = v {
                        if s.trim().parse::<i64>().is_err() {
                            return Err(EsError::field_failure(p, k, Some(t), EsError::illegal_argument(format!("For input string: \"{s}\""))));
                        }
                    }
                    continue;
                }
                if k == "interval" && t == "date_histogram" {
                    continue;
                }
                numeric(p, k, v, Some(t))?;
            }
        }
        if VS_COMMON.contains(&"field") && allowed.contains(&"field") {
            if let Some(v) = bo.get("field") {
                if !v.is_string() {
                    return Err(EsError::unsupported_type_xc(p, "field", v, Some(t)));
                }
            }
        }
        if let Some(v) = bo.get("order") {
            if !(v.is_object() || v.is_array()) {
                return Err(EsError::unsupported_type_xc(p, "order", v, Some(t)));
            }
        }
        if let Some(v) = bo.get("percents") {
            if let Value::String(s) = v {
                return Err(EsError::field_failure(p, "percents", Some(t), EsError::number_format(s)));
            }
        }
        if let Some(v) = bo.get("ranges") {
            let Value::Array(a) = v else {
                return Err(EsError::unsupported_type_xc(p, "ranges", v, Some(t)));
            };
            for r in a {
                if let Some(ro) = r.as_object() {
                    let allowed_r: &[&str] = match t {
                        "ip_range" => &["from", "to", "key", "mask"],
                        _ => &["from", "to", "key"],
                    };
                    for (rk, _) in ro {
                        if !allowed_r.contains(&rk.as_str()) {
                            // ES's Range parser reports the field a column late.
                            return Err(EsError::field_failure(p, "ranges", Some(t), EsError::x_content_parse(format!("[range] unknown field [{rk}]")).at_key(rk, Some("ranges")).cursor("end").prefixed("key").nth_key(0)).cursor("inherit"));
                        }
                    }
                }
            }
        }
        if t == "date_histogram" {
            if let Some(Value::String(ci)) = bo.get("calendar_interval") {
                if !matches!(ci.as_str(), "minute" | "1m" | "hour" | "1h" | "day" | "1d" | "week" | "1w" | "month" | "1M" | "quarter" | "1q" | "year" | "1y") {
                    return Err(EsError::field_failure(p, "calendar_interval", Some(t), EsError::illegal_argument(format!("The supplied interval [{ci}] could not be parsed as a calendar interval."))));
                }
            }
        }
        if t == "composite" {
            if let Some(Value::Array(sources)) = bo.get("sources") {
                for s in sources {
                    if let Some(so) = s.as_object() {
                        for (_, sdef) in so {
                            if let Some(sd) = sdef.as_object() {
                                for (st, sbody) in sd {
                                    if !matches!(st.as_str(), "terms" | "histogram" | "date_histogram" | "geotile_grid") {
                                        return Err(EsError::field_failure(p, "sources", Some(t), EsError::parsing(format!("invalid source type: {st}")).at_key(st, None).cursor("inherit").with_lc()).cursor("inherit"));
                                    }
                                    if let Some(sb) = sbody.as_object() {
                                        let allowed_s: &[&str] = match st.as_str() {
                                            "terms" => &["field", "script", "missing_bucket", "missing_order", "order", "value_type", "format"],
                                            "histogram" => &["field", "script", "missing_bucket", "missing_order", "order", "value_type", "format", "interval"],
                                            "date_histogram" => &["field", "script", "missing_bucket", "missing_order", "order", "value_type", "format", "interval", "calendar_interval", "fixed_interval", "time_zone", "offset"],
                                            _ => &["field", "script", "missing_bucket", "missing_order", "order", "value_type", "format", "precision", "bounds"],
                                        };
                                        for (sk, _) in sb {
                                            if !allowed_s.contains(&sk.as_str()) {
                                                return Err(EsError::field_failure(p, "sources", Some(t), EsError::unknown_field_xc(st, sk, Some(st), allowed_s)));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if t == "multi_terms" {
            if let Some(Value::Array(a)) = bo.get("terms") {
                for term in a {
                    if let Some(to) = term.as_object() {
                        for (tk, _) in to {
                            if !matches!(tk.as_str(), "field" | "script" | "missing" | "value_type" | "format") {
                                return Err(EsError::field_failure(p, "terms", Some(t), EsError::unknown_field_xc("field_config", tk, Some("terms"), &["field", "script", "missing", "value_type", "format"])));
                            }
                        }
                    }
                }
                if a.len() < 2 {
                    return Err(EsError::field_failure(p, "terms", Some(t), EsError::illegal_argument(format!("The [terms] parameter in the aggregation [{name}] must be present and have at least 2 fields or scripts. For a single field user terms aggregation.")).at_key("terms", Some(t)).cursor("end")));
                }
            }
        }
        if let Some(Value::String(h)) = bo.get("execution_hint") {
            let ok: &[&str] = match t {
                "terms" | "significant_terms" | "rare_terms" => &["map", "global_ordinals"],
                "cardinality" => &["save_memory_heuristic", "save_time_heuristic", "direct", "ordinals", "segment_ordinals"],
                _ => &[],
            };
            if !ok.is_empty() && !ok.contains(&h.as_str()) {
                return Err(crate::search_strict::shard_failure(&format!("Unknown `execution_hint`: [{h}], expected any of [{}]", ok.join(", "))));
            }
        }
        if t == "weighted_avg" {
            for part in ["value", "weight"] {
                if let Some(Value::Object(po)) = bo.get(part) {
                    for (pk, _) in po {
                        if !matches!(pk.as_str(), "field" | "script" | "missing" | "value_type" | "format") {
                            return Err(EsError::field_failure(p, part, Some(t), EsError::unknown_field_xc("field_config", pk, Some(part), &["field", "script", "missing", "value_type", "format"])));
                        }
                    }
                }
            }
        }
    }
    // Sub-structures parsed as queries / sorts.
    match t {
        "filter" => crate::search_strict::check_query(&Value::Object(bo.clone()), name, None).map_err(|e| e.default_key(name, None))?,
        "filters" => {
            if let Some(f) = bo.get("filters") {
                let items: Vec<(&str, &Value)> = match f {
                    Value::Object(m) => m.iter().map(|(k, v)| (k.as_str(), v)).collect(),
                    Value::Array(a) => a.iter().map(|v| ("filters", v)).collect(),
                    _ => vec![],
                };
                for (k, q) in items {
                    crate::search_strict::check_query(q, k, Some("filters"))?;
                }
            }
        }
        "top_hits" => {
            if let Some(s) = bo.get("sort") {
                crate::search_strict::check_sort(s)?;
            }
        }
        "significant_terms" | "significant_text" | "diversified_sampler" => {
            if let Some(q) = bo.get("background_filter") {
                crate::search_strict::check_query(q, "background_filter", Some(t))?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod value_type_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn value_source_hints_reject_mapping_types_before_field_resolution() {
        for field in ["location", "unmapped"] {
            for hint in ["geo_point", "keyword", "range", "LONG", ""] {
                let error = check_tree(&json!({"points": {"value_count": {"field": field, "missing": "0,0", "value_type": hint}}}), None).unwrap_err();
                assert_eq!(error.status, 400);
                assert_eq!(error.kind, "x_content_parse_exception");
                let cause = error.caused_by.as_ref().unwrap();
                assert_eq!(cause.kind, "illegal_argument_exception");
                assert_eq!(cause.reason, format!("Unknown value type [{hint}]"));
            }
        }
        for hint in ["string", "double", "float", "number", "numeric", "long", "integer", "short", "byte", "date", "ip", "boolean"] {
            check_tree(&json!({"values": {"value_count": {"field": "unmapped", "value_type": hint}}}), None).unwrap();
        }
        for hint in [Value::Null, json!(true), json!(1), json!([]), json!({})] {
            let error = check_tree(&json!({"values": {"value_count": {"field": "unmapped", "value_type": hint}}}), None).unwrap_err();
            assert_eq!(error.status, 400);
            assert_eq!(error.kind, "x_content_parse_exception");
        }
    }
}
