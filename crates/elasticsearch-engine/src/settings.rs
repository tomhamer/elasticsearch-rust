//! Index settings: ES keeps settings as a flat `key -> string` map
//! (`index.number_of_shards: "1"`); requests may use nested or flat form,
//! with or without the `index.` prefix. Arrays stay arrays of strings.

use crate::error::{EsError, Result};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub type FlatSettings = BTreeMap<String, Value>;

/// Flatten a settings object into `a.b.c -> value` (scalars stringified).
pub fn flatten(v: &Value) -> FlatSettings {
    let mut out = BTreeMap::new();
    fn walk(prefix: &str, v: &Value, out: &mut FlatSettings) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    walk(&key, v, out);
                }
            }
            Value::Array(a) => {
                out.insert(prefix.to_string(), Value::Array(a.iter().map(stringify).collect()));
            }
            other => {
                out.insert(prefix.to_string(), stringify(other));
            }
        }
    }
    walk("", v, &mut out);
    out
}

fn stringify(v: &Value) -> Value {
    match v {
        Value::String(_) | Value::Null => v.clone(),
        Value::Bool(b) => Value::String(b.to_string()),
        Value::Number(n) => Value::String(n.to_string()),
        Value::Array(a) => Value::Array(a.iter().map(stringify).collect()),
        Value::Object(_) => Value::String(v.to_string()),
    }
}

/// Normalize index settings: flatten and prefix every key with `index.`.
pub fn normalize_index_settings(v: &Value) -> FlatSettings {
    let mut v = v.clone();
    // `{"settings": {...}}` wrapper is accepted by update settings.
    if let Some(inner) = v.get("settings").cloned() {
        if v.as_object().map(|m| m.len() == 1).unwrap_or(false) {
            v = inner;
        }
    }
    flatten(&v)
        .into_iter()
        .map(|(k, v)| if k.starts_with("index.") || k.starts_with("archived.") { (k, v) } else { (format!("index.{k}"), v) })
        .collect()
}

/// Rebuild the nested form of flat settings (ES `Settings.getAsStructuredMap`):
/// keys nest on `.`; where a key is both a value and a prefix of other keys,
/// the remainder of the longer keys stays dotted at that level
/// (`watermark: {"low": "85%", "low.max_headroom": "200GB"}`).
pub fn unflatten(flat: &FlatSettings) -> Value {
    fn process(map: &mut Map<String, Value>, prefix: &str, setting: &str, value: Value) {
        match setting.find('.') {
            None => {
                let key = format!("{prefix}{setting}");
                if let Some(Value::Object(inner)) = map.get(&key).cloned() {
                    for (k, v) in inner {
                        map.insert(format!("{key}.{k}"), v);
                    }
                }
                map.insert(key, value);
            }
            Some(i) => {
                let (k, rest) = (&setting[..i], &setting[i + 1..]);
                let key = format!("{prefix}{k}");
                match map.get_mut(&key) {
                    None => {
                        let mut m = Map::new();
                        process(&mut m, "", rest, value);
                        map.insert(key, Value::Object(m));
                    }
                    Some(Value::Object(inner)) => process(inner, "", rest, value),
                    Some(_) => process(map, &format!("{key}."), rest, value),
                }
            }
        }
    }
    let mut root = Map::new();
    for (k, v) in flat {
        process(&mut root, "", k, v.clone());
    }
    Value::Object(root)
}

pub fn as_flat_json(flat: &FlatSettings) -> Value {
    Value::Object(flat.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

pub fn get_str<'a>(flat: &'a FlatSettings, key: &str) -> Option<&'a str> {
    flat.get(key).and_then(Value::as_str)
}

pub fn get_bool(flat: &FlatSettings, key: &str, default: bool) -> bool {
    get_str(flat, key).map(|s| s == "true").unwrap_or(default)
}

pub fn get_usize(flat: &FlatSettings, key: &str, default: usize) -> usize {
    get_str(flat, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Settings that can be changed on an open index.
pub fn is_dynamic(key: &str) -> bool {
    const DYNAMIC: &[&str] = &[
        "index.number_of_replicas",
        "index.refresh_interval",
        "index.max_result_window",
        "index.max_inner_result_window",
        "index.max_rescore_window",
        "index.max_terms_count",
        "index.max_ngram_diff",
        "index.max_shingle_diff",
        "index.blocks.read_only",
        "index.blocks.read_only_allow_delete",
        "index.blocks.read",
        "index.blocks.write",
        "index.blocks.metadata",
        "index.auto_expand_replicas",
        "index.translog.durability",
        "index.translog.sync_interval",
        "index.translog.flush_threshold_size",
        "index.translog.flush_threshold_age",
        "index.mapping.total_fields.limit",
        "index.mapping.depth.limit",
        "index.mapping.nested_fields.limit",
        "index.mapping.nested_objects.limit",
        "index.mapping.field_name_length.limit",
        "index.mapping.dimension_fields.limit",
        "index.mapping.total_fields.ignore_dynamic_beyond_limit",
        "index.mapping.field_name_length.ignore_dynamic_beyond_limit",
        "index.mapping.array_objects.limit",
        "index.mapping.dynamic_strings.auto_text",
        "index.mapping.ignore_malformed",
        "index.mapping.coerce",
        "index.default_pipeline",
        "index.final_pipeline",
        "index.search.slowlog.threshold.query.warn",
        "index.highlight.max_analyzed_offset",
        "index.highlight.weight_matches_mode.enabled",
        "index.max_script_fields",
        "index.max_docvalue_fields_search",
        "index.query.default_field",
        "index.query.parse.allow_unmapped_fields",
        "index.routing.allocation.include._tier_preference",
        "index.hidden",
        "index.priority",
        "index.lifecycle.name",
        "index.lifecycle.rollover_alias",
        "index.lifecycle.indexing_complete",
        "index.requests.cache.enable",
        "index.write.wait_for_active_shards",
        "index.max_slices_per_scroll",
        "index.max_regex_length",
        "index.analyze.max_token_count",
        "index.analyze.max_char_count",
        "index.max_adjacency_matrix_filters",
        "index.max_refresh_listeners",
        "index.search.idle.after",
        "index.gc_deletes",
        "index.unassigned.node_left.delayed_timeout",
        "index.time_series.end_time",
        "index.top_metrics_max_size",
        // `MergeSchedulerConfig` (dynamic; ES 9 clamps an inverted pair at apply time).
        "index.merge.scheduler.max_thread_count",
        "index.merge.scheduler.max_merge_count",
        "index.merge.scheduler.auto_throttle",
    ];
    key.starts_with("archived.") || DYNAMIC.contains(&key)
        || key.starts_with("index.routing.allocation.")
        || key.starts_with("index.search.slowlog.")
        || key.starts_with("index.indexing.slowlog.")
        || key.starts_with("index.lifecycle.")
        || key.starts_with("index.blocks.")
}

/// Validate the values of known settings.
pub fn validate(flat: &FlatSettings) -> Result<()> {
    for (k, v) in flat {
        // `null` resets a setting; there is nothing to validate.
        if v.is_null() {
            continue;
        }
        if k.starts_with("archived.") {
            return Err(EsError::illegal_argument(format!("archived setting [{k}] may only be removed")));
        }
        let s = v.as_str().unwrap_or("");
        match k.as_str() {
            "index.translog.flush_threshold_size" => {
                let bytes = crate::ingest::util::parse_byte_size_value(s, k)?;
                if bytes < 56 { return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}] must be >= 56b"))); }
            }
            "index.translog.flush_threshold_age" => {
                let millis = crate::date::parse_time_value(s).ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{s}] as a time value")))?;
                if !(1000..=3_600_000).contains(&millis) { return Err(EsError::illegal_argument(format!("setting [{k}] must be between [1s] and [1h]"))); }
            }
            "index.translog.retention.age" | "index.translog.retention.size" => {
                return Err(EsError::illegal_argument(
                    "Translog retention settings [index.translog.retention.age] and [index.translog.retention.size] are no longer supported. Please do not specify values for these settings",
                ));
            }
            "index.soft_deletes.enabled" if s == "false" || v.as_bool() == Some(false) => {
                return Err(EsError::illegal_argument(
                    "Creating indices with soft-deletes disabled is no longer supported. Please do not specify a value for setting [index.soft_deletes.enabled].",
                ));
            }
            "index.mapping.ignore_above" => {
                let n: i64 = s.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.mapping.ignore_above]")))?;
                if n < 0 {
                    return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.mapping.ignore_above] must be >= 0")));
                }
            }
            "index.number_of_shards" => {
                let n: i64 = s.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.number_of_shards]")))?;
                if n < 1 {
                    return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.number_of_shards] must be >= 1")));
                }
            }
            "index.number_of_replicas" => {
                let n: i64 = s.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.number_of_replicas]")))?;
                if n < 0 {
                    return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [index.number_of_replicas] must be >= 0")));
                }
            }
            "index.refresh_interval" => {
                if crate::date::parse_time_value(s).is_none() {
                    return Err(EsError::illegal_argument(format!(
                        "failed to parse setting [index.refresh_interval] with value [{s}] as a time value: unit is missing or unrecognized"
                    )));
                }
            }
            "index.mapping.array_objects.limit" => {
                let n: i64 = s.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}]")))?;
                if n < 1 {
                    return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}] must be >= 1")));
                }
            }
            "index.merge.scheduler.max_thread_count" | "index.merge.scheduler.max_merge_count" | "index.merge.scheduler.auto_throttle"
            | "index.merge.policy.segments_per_tier" | "index.merge.policy.max_merge_at_once" | "index.merge.policy.floor_segment"
            | "index.merge.policy.max_merged_segment" | "index.merge.policy.deletes_pct_allowed" | "index.merge.policy.expunge_deletes_allowed"
            | "index.merge.policy.type" | "index.merge.policy.merge_factor" => {
                // Each value on its own (the pair is checked below).
                let text = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                let one = |key: &str| if key == k.as_str() { Some(text.clone()) } else { None };
                crate::shard::MergeConfig::from_settings(one)?;
            }
            "index.max_regex_length" => {
                // ES `intSetting(.., 1000, 1, ..)`.
                match s.parse::<i32>() {
                    Err(_) => return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}]"))),
                    Ok(value) if value < 1 => return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}] must be >= 1"))),
                    Ok(_) => {}
                }
            }
            "index.max_result_window" | "index.mapping.total_fields.limit" => {
                if s.parse::<u64>().is_err() {
                    return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}]")));
                }
            }
            _ => {}
        }
    }
    // ES `MergeSchedulerConfig`: both given in one update must be ordered.
    let int = |k: &str| flat.get(k).and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_u64().map(|n| n.to_string()))).and_then(|v| v.parse::<u64>().ok());
    if let (Some(threads), Some(merges)) = (int("index.merge.scheduler.max_thread_count"), int("index.merge.scheduler.max_merge_count")) {
        if threads > merges {
            return Err(EsError::illegal_argument(format!("maxThreadCount (= {threads}) should be <= maxMergeCount (= {merges})")));
        }
    }
    Ok(())
}


// ---------------------------------------------------------------- registry

/// ES 8.19's registered settings and their defaults: every cluster-scope
/// setting with its default and whether it is dynamic, and every index-scope
/// setting's default (captured from a default 8.19 node's
/// `include_defaults`). Machine-specific values (paths, names, processors,
/// memory) are overridden by the caller.
struct Registry {
    cluster: BTreeMap<String, (Value, bool)>,
    index: BTreeMap<String, Value>,
}

static REGISTRY: std::sync::LazyLock<Registry> = std::sync::LazyLock::new(|| {
    let v: Value = serde_json::from_str(include_str!("settings_registry.json")).unwrap_or(Value::Null);
    let mut cluster = BTreeMap::new();
    if let Some(m) = v["cluster"].as_object() {
        for (k, e) in m {
            cluster.insert(k.clone(), (e[0].clone(), e[1].as_bool().unwrap_or(true)));
        }
    }
    let index = v["index"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
    Registry { cluster, index }
});

/// Cluster-scope settings registered by prefix (ES affix / group settings):
/// `(prefix, dynamic)`.
const CLUSTER_AFFIX: &[(&str, bool)] = &[
    ("cluster.routing.allocation.include.", true),
    ("cluster.routing.allocation.exclude.", true),
    ("cluster.routing.allocation.require.", true),
    ("cluster.routing.allocation.awareness.force.", true),
    ("cluster.remote.", true),
    ("cluster.search.remote.", true),
    ("search.remote.", true),
    ("logger.", true),
    ("xpack.notification.", true),
    ("xpack.monitoring.exporters.", true),
    ("cluster.metadata.", true),
    ("node.attr.", false),
    ("thread_pool.", false),
    ("xpack.security.authc.realms.", false),
    ("transport.profiles.", false),
];

/// Default values of every registered cluster setting (flat keys).
pub fn cluster_defaults() -> FlatSettings {
    REGISTRY.cluster.iter().map(|(k, (v, _))| (k.clone(), v.clone())).collect()
}

/// `Some(dynamic)` when `key` is a registered cluster setting.
pub fn cluster_setting_kind(key: &str) -> Option<bool> {
    if crate::script::cache_policy::context_setting(key).is_some() { return Some(true); }
    if let Some((_, d)) = REGISTRY.cluster.get(key) {
        return Some(*d);
    }
    CLUSTER_AFFIX.iter().find(|(p, _)| key.starts_with(p) && key.len() > p.len()).map(|(_, d)| *d)
}

/// ES `ClusterSettings.archiveUnknownOrInvalidSettings`, as applied when
/// the persisted metadata loads (`ClusterStateUpdaters
/// .upgradeAndArchiveUnknownOrInvalidSettings`): a persisted `scope`
/// (`persistent` / `transient`) setting this version does not know, or
/// whose value no longer validates, moves to `archived.<key>` (value kept)
/// with a warning, instead of failing the start or being applied. A known
/// setting with a valid value stays, also a non-dynamic one (such as
/// `indices.breaker.total.use_real_memory`, which an older version of this
/// server accepted dynamically): ES keeps it and nothing applies it, as
/// only the node's own setting counts. Archived keys stay archived; a
/// nested object value (not a flat key) is left alone. Returns the
/// archived keys.
/// ES `ByteSizeValue.parseBytesSizeValue`: a number with a unit (`b`, `k`/
/// `kb`, `m`/`mb`, `g`/`gb`, `t`/`tb`, `p`/`pb`, any case), or a bare `0` /
/// `-1`. `None` when malformed. Fractions are accepted (ES deprecates them).
pub fn parse_byte_size(v: &str) -> Option<i64> {
    let t = v.trim().to_ascii_lowercase();
    if t == "0" || t == "-1" {
        return t.parse().ok();
    }
    let units: [(&str, i64); 11] = [("pb", 1 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10),
        ("p", 1 << 50), ("t", 1 << 40), ("g", 1 << 30), ("m", 1 << 20), ("k", 1 << 10), ("b", 1)];
    let (num, mult) = units.iter().find_map(|(u, m)| t.strip_suffix(u).map(|n| (n.trim(), *m)))?;
    let n: f64 = num.parse().ok().filter(|n: &f64| n.is_finite())?;
    (n >= 0.0 || (n == -1.0 && mult == 1)).then(|| (n * mult as f64) as i64)
}

pub fn archive_unknown_or_invalid_settings(scope: &str, map: &mut serde_json::Map<String, Value>) -> Vec<String> {
    let mut archive = Vec::new();
    for (k, v) in map.iter() {
        if k.starts_with("archived.") || v.is_null() || v.is_object() {
            continue;
        }
        if cluster_setting_kind(k).is_none() {
            tracing::warn!("ignoring unknown {scope} setting: [{k}] with value [{v}]; archiving");
        } else if let Err(e) = validate_cluster_value(k, v) {
            tracing::warn!("ignoring invalid {scope} setting: [{k}] with value [{v}]; archiving: {}", e.reason);
        } else {
            continue;
        }
        archive.push(k.clone());
    }
    for k in &archive {
        if let Some(v) = map.remove(k) {
            map.insert(format!("archived.{k}"), v);
        }
    }
    archive
}

/// Upgrade the old `name.0`, `name.1` representation used by persisted
/// Settings builders. ES9.5.5 Settings.Builder.processLegacyLists converts the
/// contiguous prefix to an array and rejects a simultaneous `name` value.
/// This compatibility load path additionally logs the requested deprecation;
/// it does not invent a Warning header for ordinary numeric JSON settings.
pub fn upgrade_legacy_setting_lists(map: &mut Map<String, Value>) -> Result<Vec<String>> {
    let heads: Vec<String> = map.keys().filter_map(|key| key.strip_suffix(".0").map(str::to_string)).collect();
    let mut replacements = Vec::new();
    for prefix in heads {
        if map.contains_key(&prefix) {
            return Err(EsError::new(400, "illegal_state_exception", format!(
                "settings builder can't contain values for [{prefix}={}] and [{prefix}.0={}]", map[&prefix], map[&format!("{prefix}.0")]
            )));
        }
        let mut values = Vec::new();
        loop {
            let key = format!("{prefix}.{}", values.len());
            match map.get(&key) {
                None | Some(Value::Null) => break,
                Some(value) if value.is_object() || value.is_array() => return Err(EsError::illegal_argument(format!("legacy list setting [{key}] must contain scalar values"))),
                Some(value) => values.push(stringify(value)),
            }
        }
        replacements.push((prefix, values));
    }
    let mut upgraded = Vec::new();
    for (prefix, values) in replacements {
        for i in 0..values.len() { map.remove(&format!("{prefix}.{i}")); }
        // Persisted nulls represent resets, not another list element. Remove
        // a terminating reset so a second load cannot see both base and .0.
        let terminal = format!("{prefix}.{}", values.len());
        if map.get(&terminal).is_some_and(Value::is_null) { map.remove(&terminal); }
        map.insert(prefix.clone(), Value::Array(values));
        tracing::warn!(setting = %prefix, "deprecated old-format indexed-list setting; use an array value instead");
        upgraded.push(prefix);
    }
    Ok(upgraded)
}

/// Archive persisted index settings rejected by the index value validators.
/// Only loading stored metadata calls this: new requests still return errors.
/// Unknown keys are deliberately not inferred from the default-value registry:
/// it is not the complete set of plugin/group/private setting registrations.
/// ES reference: IndexMetadataVerifier.archiveOrDeleteBrokenIndexSettings.
pub fn archive_invalid_index_settings(index: &str, map: &mut Map<String, Value>) -> Vec<String> {
    let mut invalid = BTreeMap::new();
    for (key, value) in map.iter() {
        if key.starts_with("archived.") || value.is_null() { continue; }
        // Older persisted JSON may contain scalar numbers/booleans rather than
        // the canonical Settings strings. Test their value, not their JSON type.
        let one = [(key.clone(), stringify(value))].into_iter().collect();
        if let Err(error) = validate(&one) { invalid.insert(key.clone(), error.reason); }
    }
    // These settings constrain one another. Neither is safe to apply if the
    // persisted pair is inconsistent, even when both parse independently.
    let pair: FlatSettings = ["index.merge.scheduler.max_thread_count", "index.merge.scheduler.max_merge_count"]
        .into_iter().filter_map(|key| map.get(key).map(|value| (key.to_string(), stringify(value)))).collect();
    if pair.len() == 2 && pair.keys().all(|key| !invalid.contains_key(key)) {
        if let Err(error) = validate(&pair) {
            for key in pair.keys() { invalid.insert(key.clone(), error.reason.clone()); }
        }
    }
    let mut archived = Vec::new();
    for (key, reason) in invalid {
        let value = map.remove(&key).expect("persisted setting exists");
        map.insert(format!("archived.{key}"), value);
        tracing::warn!(index, setting = %key, %reason, "archiving invalid persisted index setting");
        archived.push(key);
    }
    archived
}

/// Validate a cluster settings update (`PUT _cluster/settings`) the way ES's
/// `AbstractScopedSettings.validate` does: unknown keys, static keys and
/// malformed values are rejected (`scope` is `persistent` / `transient`).
pub fn validate_cluster_update(scope: &str, flat: &FlatSettings) -> Result<()> {
    for (k, v) in flat {
        // `prefix.*: null` resets a group of settings.
        if v.is_null() && k.ends_with('*') {
            continue;
        }
        // Archived settings (see `archive_unknown_or_invalid_settings`) can only be
        // removed (`archived.*: null`).
        if k.starts_with("archived.") {
            if v.is_null() {
                continue;
            }
            return Err(EsError::illegal_argument(format!("{scope} setting [{k}], not recognized")));
        }
        let Some(dynamic) = cluster_setting_kind(k) else {
            return Err(EsError::illegal_argument(format!("{scope} setting [{k}], not recognized")));
        };
        if !dynamic {
            return Err(EsError::illegal_argument(format!("{scope} setting [{k}], not dynamically updateable")));
        }
        validate_cluster_value(k, v)?;
    }
    Ok(())
}

/// The value checks of [`validate_cluster_update`] for one known setting
/// (`null`: a reset, always valid).
fn validate_cluster_value(k: &str, v: &Value) -> Result<()> {
    if v.is_null() {
        return Ok(());
    }
    if k == "xpack.applications.rules.max_rules_per_ruleset" {
        let value = v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()));
        if !value.is_some_and(|n| (1..=1000).contains(&n)) {
            return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] for setting [{k}] must be between 1 and 1000")));
        }
    }
    let s = match v {
        Value::String(s) => s.clone(),
        Value::Array(_) if !k.starts_with("script.") => return Ok(()),
        other => other.to_string(),
    };
    crate::memory::validate_setting(k, &s)?;
    crate::script::cache_policy::validate_value(k, v)?;
    let enum_err = |class: &str| EsError::illegal_argument(format!("No enum constant {class}.{}", s.to_uppercase()));
    match k {
        "cluster.routing.allocation.enable" if !matches!(s.to_lowercase().as_str(), "all" | "primaries" | "new_primaries" | "none") => {
            return Err(enum_err("org.elasticsearch.cluster.routing.allocation.decider.EnableAllocationDecider.Allocation"));
        }
        "cluster.routing.rebalance.enable" if !matches!(s.to_lowercase().as_str(), "all" | "primaries" | "replicas" | "none") => {
            return Err(enum_err("org.elasticsearch.cluster.routing.allocation.decider.EnableAllocationDecider.Rebalance"));
        }
        // `RecoverySettings`: byte sizes (`max_bytes_per_sec` 0 or -1
        // disables throttling) and the bounded chunk concurrency.
        "indices.recovery.max_bytes_per_sec" | "indices.recovery.chunk_size" if parse_byte_size(&s).is_none() => {
            return Err(EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{s}] as a size in bytes: unit is missing or unrecognized"))
                .caused_by(EsError::new(400, "parse_exception", format!("failed to parse setting [{k}] with value [{s}] as a size in bytes: unit is missing or unrecognized"))));
        }
        "indices.recovery.chunk_size" if parse_byte_size(&s).is_some_and(|b| b < 1) => {
            return Err(EsError::illegal_argument(format!("failed to parse value [{s}] for setting [{k}], must be >= [1b]")));
        }
        "indices.recovery.max_concurrent_file_chunks" => match s.trim().parse::<i64>() {
            Ok(n) if n < 1 => return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}] must be >= 1"))),
            Ok(n) if n > 8 => return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}] must be <= 8"))),
            _ => {}
        },
        // ES 9: `BalancedShardsAllocator.THRESHOLD_SETTING` has a minimum of 1.0.
        "cluster.routing.allocation.balance.threshold" if s.parse::<f64>().is_ok_and(|t| t < 1.0) => {
            return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}] must be >= 1.0")));
        }
        _ => {}
    }
    let Some((def, _)) = REGISTRY.cluster.get(k) else { return Ok(()) };
    match def.as_str() {
        Some("true") | Some("false") if s != "true" && s != "false" => {
            return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] as only [true] or [false] are allowed.")));
        }
        Some(d) if d.parse::<i64>().is_ok() && s.parse::<i64>().is_err() && !s.chars().any(|c| c.is_ascii_digit()) => {
            return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}]"))
                .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))));
        }
        Some(d) => validate_typed_value(k, d, &s)?,
        None => {}
    }
    Ok(())
}

/// The kind of value a setting takes, read off its registered default (the
/// registry carries no types), checked as ES's `Setting.get` parses it,
/// with its errors: an `illegal_argument_exception`, caused by the
/// parser's `parse_exception` where the parser throws one.
/// - A time value: `TimeValue.parseTimeValue` ([`check_time_value`]).
/// - A byte size: `ByteSizeValue.parseBytesSizeValue`.
/// - A percentage default: a disk watermark takes a ratio, a percentage or
///   a byte size (`RelativeByteSizeValue`); a write-load threshold a ratio
///   or a percentage (`RatioValue`); anything else is a memory size, a
///   percentage of the heap or a byte size
///   (`MemorySizeValue.parseBytesSizeValueOrHeapRatio`).
/// - A float: `Float.parseFloat`.
/// Other defaults say nothing.
fn validate_typed_value(k: &str, default: &str, s: &str) -> Result<()> {
    let number = |t: &str| t.parse::<f64>().is_ok_and(f64::is_finite);
    let unit = |d: &str, units: &[&str]| units.iter().any(|u| d.len() > u.len() && d.to_ascii_lowercase().ends_with(u) && number(&d[..d.len() - u.len()]));
    // `Setting.get`: a parser's ElasticsearchParseException becomes an
    // IllegalArgumentException with its message, caused by it.
    let wrapped = |e: EsError| EsError::illegal_argument(e.reason.clone()).caused_by(e);
    let parse = |m: String| EsError::new(400, "parse_exception", m);
    if unit(default, &["nanos", "micros", "ms", "s", "m", "h", "d"]) {
        check_time_value(k, s)?;
    } else if unit(default, &["kb", "mb", "gb", "tb", "pb", "b"]) {
        crate::ingest::util::parse_byte_size_value(s, k).map_err(wrapped)?;
    } else if default.ends_with('%') && number(&default[..default.len() - 1]) {
        if k.contains(".disk.watermark.") {
            // `RelativeByteSizeValue.parseRelativeByteSizeValue`.
            if ratio_value(s).is_err() {
                crate::ingest::util::parse_byte_size_value(s, k)
                    .map_err(|e| wrapped(parse(format!("unable to parse [{k}={s}] as either percentage or bytes")).caused_by(e)))?;
            }
        } else if k.contains(".write_load_decider.") {
            ratio_value(s).map_err(|m| wrapped(parse(m)))?;
        } else if let Some(p) = s.strip_suffix('%') {
            // `MemorySizeValue.parseHeapRatio`.
            match p.parse::<f64>() {
                Ok(pct) if !(0.0..=100.0).contains(&pct) => return Err(wrapped(parse(format!("percentage should be in [0-100], got [{p}]")))),
                Ok(_) => {}
                Err(_) => {
                    return Err(wrapped(parse(format!("failed to parse [{p}] as a double"))
                        .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{p}\"")))));
                }
            }
        } else {
            crate::ingest::util::parse_byte_size_value(s, k).map_err(wrapped)?;
        }
    } else if default.contains('.') && number(default) && !number(s.trim()) {
        return Err(EsError::illegal_argument(format!("Failed to parse value [{s}] for setting [{k}]"))
            .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))));
    }
    Ok(())
}

/// ES `TimeValue.parseTimeValue` as a setting's parser, with its errors
/// (`IllegalArgumentException`s, thrown as they are by `Setting.get`):
/// units are case-insensitive except minutes (`m`: `M` would be months, so
/// `5M` has no unit), `-1` and `0` need none, and a negative value other
/// than `-1` or a fractional one is refused.
pub fn check_time_value(k: &str, s: &str) -> Result<()> {
    let normalized = s.to_lowercase();
    let normalized = normalized.trim();
    let suffix = if normalized.ends_with("nanos") {
        "nanos"
    } else if normalized.ends_with("micros") {
        "micros"
    } else if normalized.ends_with("ms") {
        "ms"
    } else if normalized.ends_with('s') {
        "s"
    } else if s.ends_with('m') {
        "m"
    } else if normalized.ends_with('h') {
        "h"
    } else if normalized.ends_with('d') {
        "d"
    } else {
        let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b == b'0');
        // `-0*1` and `0+`.
        if normalized.strip_prefix('-').is_some_and(|t| t.strip_suffix('1').is_some_and(|z| z.is_empty() || digits(z))) || digits(normalized) {
            return Ok(());
        }
        return Err(EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{s}] as a time value: unit is missing or unrecognized")));
    };
    let n = normalized[..normalized.len() - suffix.len()].trim();
    let nfe = || EsError::new(400, "number_format_exception", format!("For input string: \"{n}\""));
    match n.parse::<i64>() {
        Ok(v) if v < -1 => Err(EsError::illegal_argument(format!("failed to parse setting [{k}] with value [{s}] as a time value: negative durations are not supported"))),
        Ok(_) => Ok(()),
        Err(_) if n.parse::<f64>().is_ok() => Err(EsError::illegal_argument(format!("failed to parse [{s}], fractional time values are not supported")).caused_by(nfe())),
        Err(_) => Err(EsError::illegal_argument(format!("failed to parse [{s}]")).caused_by(nfe())),
    }
}

/// ES `RatioValue.parseRatioValue`: a percentage in [0-100] or a ratio in
/// [0-1.0]; `Err` holds its `ElasticsearchParseException` message.
fn ratio_value(s: &str) -> std::result::Result<f64, String> {
    if let Some(p) = s.strip_suffix('%') {
        return match p.parse::<f64>() {
            Ok(pct) if !(0.0..=100.0).contains(&pct) => Err(format!("Percentage should be in [0-100], got [{p}]")),
            Ok(pct) => Ok(pct),
            Err(_) => Err(format!("Failed to parse [{p}] as a double")),
        };
    }
    match s.parse::<f64>() {
        Ok(r) if !(0.0..=1.0).contains(&r) => Err(format!("Ratio should be in [0-1.0], got [{}]", java_double(r))),
        Ok(r) => Ok(100.0 * r),
        Err(_) => Err(format!("Invalid ratio or percentage [{s}]")),
    }
}

/// Java's `Double.toString` for the plain values a ratio takes (`1.5`, `2.0`).
fn java_double(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e7 { format!("{v:.1}") } else { v.to_string() }
}

/// `index.mapping.nested_fields.limit` default (`MapperService`): 100 for
/// indices created on or after `NESTED_PATH_LIMIT`, 50 before.
/// `IndexSettings.USE_DOC_VALUES_SKIPPER` default: on for time series
/// indices (from `STATELESS_SKIPPERS_ENABLED_FOR_TSDB`) and logsdb / strict
/// columnar ones (from `SKIPPERS_ENABLED_BY_DEFAULT_IN_LOGSDB`).
pub fn use_doc_values_skipper_default(mode: Option<&str>, created: crate::version::IndexVersion) -> bool {
    use crate::version::IndexVersion;
    match mode.map(str::to_ascii_lowercase).as_deref() {
        Some("time_series") => created.on_or_after(IndexVersion::STATELESS_SKIPPERS_ENABLED_FOR_TSDB),
        Some("logsdb" | "columnar" | "logsdb_columnar") => created.on_or_after(IndexVersion::SKIPPERS_ENABLED_BY_DEFAULT_IN_LOGSDB),
        _ => false,
    }
}

pub fn nested_fields_limit_default(created: crate::version::IndexVersion) -> usize {
    if created.on_or_after(crate::version::IndexVersion::NESTED_PATH_LIMIT) { 100 } else { 50 }
}

/// The defaults `GET {index}/_settings?include_defaults` reports for an
/// index with settings `own`: every registered index setting it does not set.
pub fn index_defaults(own: &FlatSettings) -> FlatSettings {
    use crate::version::IndexVersion;
    let mode = get_str(own, "index.mode").unwrap_or("standard").to_lowercase();
    let created = IndexVersion::from_setting(own.get("index.version.created"));
    let mut out = FlatSettings::new();
    for (k, v) in &REGISTRY.index {
        if own.contains_key(k) {
            continue;
        }
        let v = match k.as_str() {
            "index.number_of_routing_shards" => Value::String(get_str(own, "index.number_of_shards").unwrap_or("1").to_string()),
            "index.version.compatibility" => own.get("index.version.created").cloned().unwrap_or_else(|| v.clone()),
            // Defaults keyed on the index's creation version (ES setting default functions).
            "index.mapping.nested_fields.limit" => Value::String(nested_fields_limit_default(created).to_string()),
            "index.mapping.nested_parents.limit" => {
                Value::String(if created.on_or_after(IndexVersion::NESTED_PATH_LIMIT) { "50".into() } else { i32::MAX.to_string() })
            }
            "index.mapping.exclude_source_vectors" => Value::String(created.on_or_after(IndexVersion::EXCLUDE_SOURCE_VECTORS_DEFAULT).to_string()),
            "index.mapping.total_fields.ignore_dynamic_beyond_limit" if mode == "logsdb" => Value::String("true".into()),
            "index.mapping.ignore_above" if mode == "logsdb" || mode == "logsdb_columnar" => Value::String("8191".into()),
            "index.mapping.source.mode" if matches!(mode.as_str(), "logsdb" | "time_series" | "columnar" | "logsdb_columnar") => Value::String("SYNTHETIC".into()),
            // Strict columnar index modes (`IndexMode.isStrictColumnar`).
            "index.mapping.index_disabled_by_default" if matches!(mode.as_str(), "columnar" | "logsdb_columnar") => Value::String("true".into()),
            "index.mapping.use_columnar_id_mode_by_default" if matches!(mode.as_str(), "columnar" | "logsdb_columnar") && created.on_or_after(IndexVersion::MAPPING_ID_MODE_DEFAULT) => Value::String("true".into()),
            "index.mapping.use_doc_values_skipper" => Value::String(use_doc_values_skipper_default(Some(mode.as_str()), created).to_string()),
            "index.disable_sequence_numbers" if matches!(mode.as_str(), "columnar" | "logsdb_columnar") && created.on_or_after(IndexVersion::DISABLE_SEQUENCE_NUMBERS) => Value::String("true".into()),
            "index.mapping.synthetic_id" => Value::String(crate::tsdb::synthetic_id(own).to_string()),
            // `IndexMode.isColumnar`.
            "index.use_time_series_doc_values_format" if matches!(mode.as_str(), "time_series" | "logsdb" | "columnar" | "logsdb_columnar") => Value::String("true".into()),
            "index.merge.scheduler.max_thread_count" => {
                let n = crate::processors::get();
                Value::String((n / 2).clamp(1, 4).to_string())
            }
            _ => v.clone(),
        };
        out.insert(k.clone(), v);
    }
    // `IndexSortConfigDefaults`: the index mode's sort, or one derived from
    // an explicit `index.sort.field`.
    if !own.contains_key("index.sort.field") {
        let t = |v: &[&str]| Value::Array(v.iter().map(|x| Value::String(x.to_string())).collect());
        let (f, o, m, mi) = crate::indexsort::mode_default_sort(own);
        for (k, v) in [("index.sort.field", f), ("index.sort.order", o), ("index.sort.mode", m), ("index.sort.missing", mi)] {
            if !own.contains_key(k) {
                out.insert(k.into(), t(&v));
            }
        }
    } else {
        let fields: Vec<String> = match own.get("index.sort.field") {
            Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect(),
            Some(Value::String(v)) => v.split(',').map(|x| x.trim().to_string()).collect(),
            _ => vec![],
        };
        let orders: Option<Vec<String>> = match own.get("index.sort.order") {
            Some(Value::Array(a)) => Some(a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()),
            Some(Value::String(v)) => Some(v.split(',').map(|x| x.trim().to_string()).collect()),
            _ => None,
        };
        let n = fields.len();
        let arr = |f: &dyn Fn(usize) -> String| Value::Array((0..n).map(|i| Value::String(f(i))).collect());
        if !own.contains_key("index.sort.order") {
            out.insert("index.sort.order".into(), arr(&|_| "asc".into()));
        }
        if !own.contains_key("index.sort.mode") {
            out.insert("index.sort.mode".into(), arr(&|i| if orders.as_ref().and_then(|o| o.get(i)).map(String::as_str) == Some("desc") { "max".into() } else { "min".into() }));
        }
        if !own.contains_key("index.sort.missing") {
            out.insert("index.sort.missing".into(), arr(&|_| "_last".into()));
        }
    }
    // `index.mapping.synthetic_source_keep` defaults to `arrays` on
    // `logsdb` indices (ES `LogsDbIndexMode`), `none` elsewhere.
    if mode == "logsdb" {
        out.insert("index.mapping.synthetic_source_keep".into(), Value::String("arrays".into()));
    }
    // `index.recovery.use_synthetic_source` defaults to `true` when the
    // effective source mode is synthetic (the setting, else the mapping's
    // `_source.mode`, else the index-mode default).
    let src_mode = crate::synthetic::source_mode(
        get_str(own, "index.mapping.source.mode"),
        &crate::mapping::Mappings::default(),
        get_str(own, "index.mode"),
    );
    if src_mode == crate::synthetic::SourceMode::Synthetic {
        out.insert("index.recovery.use_synthetic_source".into(), Value::String("true".into()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_setting_lists_upgrade_atomically_and_keep_array_order() {
        let mut map = serde_json::json!({"index.query.default_field.0":"first", "index.query.default_field.1":"second", "index.query.default_field.3":"gap", "unchanged":["x"]}).as_object().unwrap().clone();
        assert_eq!(upgrade_legacy_setting_lists(&mut map).unwrap(), vec!["index.query.default_field"]);
        assert_eq!(map["index.query.default_field"], serde_json::json!(["first", "second"]));
        assert_eq!(map["index.query.default_field.3"], "gap", "ES converts only the contiguous prefix");
        assert_eq!(map["unchanged"], serde_json::json!(["x"]));
        assert!(upgrade_legacy_setting_lists(&mut map).unwrap().is_empty());
        let mut ambiguous = serde_json::json!({"a.0":"ok", "z":["existing"], "z.0":"conflict"}).as_object().unwrap().clone();
        let before = ambiguous.clone();
        assert_eq!(upgrade_legacy_setting_lists(&mut ambiguous).unwrap_err().kind, "illegal_state_exception");
        assert_eq!(ambiguous, before, "one ambiguous list must not partially upgrade other keys");
    }

    #[test]
    fn max_regex_length_is_at_least_one() {
        let reason = |v: serde_json::Value| validate(&normalize_index_settings(&serde_json::json!({"index.max_regex_length": v}))).unwrap_err().reason;
        assert_eq!(reason(serde_json::json!(0)), "Failed to parse value [0] for setting [index.max_regex_length] must be >= 1");
        assert_eq!(reason(serde_json::json!("x")), "Failed to parse value [x] for setting [index.max_regex_length]");
        assert!(validate(&normalize_index_settings(&serde_json::json!({"index.max_regex_length": 1}))).is_ok());
    }

    #[test]
    fn invalid_persisted_index_settings_are_archived_and_removable() {
        let mut map = serde_json::json!({
            "index.max_regex_length": "broken", "index.number_of_replicas": 0,
            "index.merge.scheduler.max_thread_count": "3", "index.merge.scheduler.max_merge_count": "2",
            "index.analysis.analyzer.mine.tokenizer": "standard", "archived.index.old": "retained"
        }).as_object().unwrap().clone();
        let archived = archive_invalid_index_settings("old", &mut map);
        assert_eq!(archived.len(), 3);
        assert_eq!(map["archived.index.max_regex_length"], "broken");
        assert_eq!(map["index.number_of_replicas"], 0);
        assert_eq!(map["index.analysis.analyzer.mine.tokenizer"], "standard");
        assert_eq!(map["archived.index.old"], "retained");
        assert!(archive_invalid_index_settings("old", &mut map).is_empty());
        let removal = normalize_index_settings(&serde_json::json!({"archived.*": null}));
        assert!(removal.contains_key("archived.*"));
        assert!(is_dynamic("archived.*"));
        assert!(validate(&removal).is_ok());
        assert!(validate(&normalize_index_settings(&serde_json::json!({"archived.index.old": "replacement"}))).is_err());
    }

    /// Combined review r3: the typed checks follow ES's parsers exactly
    /// (`TimeValue`, `ByteSizeValue`, `RatioValue`, `RelativeByteSizeValue`,
    /// `MemorySizeValue`) and `Setting.get`'s errors.
    #[test]
    fn typed_setting_values_parse_as_es_parses_them() {
        let check = |k: &str, v: &str| validate_cluster_value(k, &Value::String(v.into()));
        // TimeValue: case-insensitive units except minutes; -1 and 0 bare.
        for ok in ["5m", "5S", "5MS", "5H", "5D", "10nanos", "3MICROS", "-1", "-01", "0", "000", "-1s", " 5s "] {
            assert!(check("search.max_keep_alive", ok).is_ok(), "{ok}");
        }
        let e = check("search.max_keep_alive", "5M").unwrap_err();
        assert_eq!((e.kind.as_str(), e.reason.as_str()), ("illegal_argument_exception", "failed to parse setting [search.max_keep_alive] with value [5M] as a time value: unit is missing or unrecognized"));
        assert!(e.caused_by.is_none());
        let e = check("search.max_keep_alive", "-2s").unwrap_err();
        assert_eq!(e.reason, "failed to parse setting [search.max_keep_alive] with value [-2s] as a time value: negative durations are not supported");
        let e = check("search.max_keep_alive", "1.5s").unwrap_err();
        assert_eq!(e.reason, "failed to parse [1.5s], fractional time values are not supported");
        assert_eq!(e.caused_by.as_deref().map(|c| c.kind.as_str()), Some("number_format_exception"));
        let e = check("search.max_keep_alive", "abcs").unwrap_err();
        assert_eq!(e.reason, "failed to parse [abcs]");
        // ByteSizeValue: illegal_argument caused by the parse_exception.
        let e = check("indices.recovery.max_bytes_per_sec", "lots").unwrap_err();
        assert_eq!(e.kind, "illegal_argument_exception");
        assert_eq!(e.reason, "failed to parse setting [indices.recovery.max_bytes_per_sec] with value [lots] as a size in bytes: unit is missing or unrecognized");
        assert_eq!(e.caused_by.as_deref().map(|c| (c.kind.as_str(), c.reason.as_str())), Some(("parse_exception", e.reason.as_str())));
        assert!(check("indices.recovery.max_bytes_per_sec", "40MB").is_ok());
        // RelativeByteSizeValue (watermarks): ratio <= 1.0, percentage or bytes.
        let wm = "cluster.routing.allocation.disk.watermark.low";
        for ok in ["85%", "0.85", "1.0", "0", "500mb"] {
            assert!(check(wm, ok).is_ok(), "{ok}");
        }
        let e = check(wm, "1.5").unwrap_err();
        assert_eq!(e.kind, "illegal_argument_exception");
        assert_eq!(e.reason, format!("unable to parse [{wm}=1.5] as either percentage or bytes"));
        assert_eq!(e.caused_by.as_deref().map(|c| c.kind.as_str()), Some("parse_exception"));
        assert!(check(wm, "101%").is_err());
        // RatioValue (write-load thresholds): no byte sizes.
        let wl = "cluster.routing.allocation.write_load_decider.hotspot_utilization_threshold";
        assert!(check(wl, "0.5").is_ok() && check(wl, "50%").is_ok());
        assert_eq!(check(wl, "1.5").unwrap_err().reason, "Ratio should be in [0-1.0], got [1.5]");
        assert_eq!(check(wl, "1mb").unwrap_err().reason, "Invalid ratio or percentage [1mb]");
        // MemorySizeValue: heap percentage or bytes, never a ratio.
        let mem = "indices.queries.cache.size";
        assert!(check(mem, "10%").is_ok() && check(mem, "512mb").is_ok());
        assert_eq!(check(mem, "150%").unwrap_err().reason, "percentage should be in [0-100], got [150]");
        assert_eq!(check(mem, "0.1").unwrap_err().reason, format!("failed to parse setting [{mem}] with value [0.1] as a size in bytes: unit is missing or unrecognized"));
        // A valid persisted value in another case is not archived.
        let mut m: Map<String, Value> = serde_json::from_value(serde_json::json!({"search.max_keep_alive": "2H"})).unwrap();
        assert!(archive_unknown_or_invalid_settings("persistent", &mut m).is_empty());
        assert_eq!(crate::date::parse_time_value("2H"), Some(7_200_000), "and it applies as ES reads it");
        assert_eq!(crate::date::parse_time_value("2M"), None);
    }

    /// ES `archiveUnknownOrInvalidSettings`: unknown keys and invalid
    /// values are archived, whatever the setting; known valid ones stay,
    /// non-dynamic ones included (`use_real_memory`, which nothing applies
    /// from cluster settings).
    #[test]
    fn unknown_or_invalid_persisted_settings_are_archived_not_rejected() {
        let mut m: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "indices.breaker.total.use_real_memory": false,
            "indices.breaker.request.limit": "abc",
            "indices.breaker.request.overhead": "-1",
            "indices.breaker.fielddata.limit": "30%",
            "cluster.routing.allocation.enable": "sometimes",
            "cluster.routing.allocation.balance.threshold": "2.0",
            "cluster.no_such_setting": "x",
            "archived.cluster.old": "y"
        })).unwrap();
        let mut archived = archive_unknown_or_invalid_settings("persistent", &mut m);
        archived.sort();
        assert_eq!(archived, ["cluster.no_such_setting", "cluster.routing.allocation.enable", "indices.breaker.request.limit", "indices.breaker.request.overhead"]);
        assert_eq!(m["archived.cluster.no_such_setting"], "x");
        assert_eq!(m["archived.cluster.routing.allocation.enable"], "sometimes");
        assert_eq!(m["archived.indices.breaker.request.limit"], "abc");
        assert_eq!(m["archived.cluster.old"], "y");
        assert_eq!(m["indices.breaker.total.use_real_memory"], false, "a known non-dynamic setting is kept, as ES");
        assert_eq!(m["indices.breaker.fielddata.limit"], "30%");
        assert_eq!(m["cluster.routing.allocation.balance.threshold"], "2.0");
        // Archiving is idempotent.
        assert!(archive_unknown_or_invalid_settings("transient", &mut m).is_empty());
        // Generic value checks, by the registered default's kind.
        let mut typed: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "search.max_keep_alive": "abc",
            "search.default_keep_alive": "5m",
            "indices.recovery.max_bytes_per_sec": "lots",
            "cluster.routing.allocation.disk.watermark.low": "much",
            "cluster.routing.allocation.disk.watermark.high": "0.9",
            "cluster.routing.allocation.disk.watermark.flood_stage": "500mb",
            "cluster.routing.allocation.balance.shard": "x"
        })).unwrap();
        let mut archived = archive_unknown_or_invalid_settings("persistent", &mut typed);
        archived.sort();
        assert_eq!(archived, ["cluster.routing.allocation.balance.shard", "cluster.routing.allocation.disk.watermark.low", "indices.recovery.max_bytes_per_sec", "search.max_keep_alive"]);
        assert_eq!(typed["search.default_keep_alive"], "5m");
        let flat = |v: Value| flatten(&v);
        let e = validate_cluster_update("persistent", &flat(serde_json::json!({"search.max_keep_alive": "abc"}))).unwrap_err();
        assert!(e.reason.contains("as a time value"), "{e:?}");
        assert!(validate_cluster_update("persistent", &flat(serde_json::json!({"indices.recovery.max_bytes_per_sec": "40mb", "search.max_keep_alive": "-1"}))).is_ok());
        // Archived settings can be removed, not set.
        assert!(validate_cluster_update("persistent", &flat(serde_json::json!({"archived.*": null}))).is_ok());
        assert!(validate_cluster_update("persistent", &flat(serde_json::json!({"archived.indices.breaker.request.limit": null}))).is_ok());
        assert!(validate_cluster_update("persistent", &flat(serde_json::json!({"archived.indices.breaker.request.limit": "1gb"}))).is_err());
    }
    use serde_json::json;

    #[test]
    fn flatten_and_prefix() {
        let f = normalize_index_settings(&json!({"number_of_shards": 1, "index": {"refresh_interval": "5s"}, "analysis": {"analyzer": {"a": {"filter": ["lowercase"]}}}}));
        assert_eq!(f["index.number_of_shards"], json!("1"));
        assert_eq!(f["index.refresh_interval"], json!("5s"));
        assert_eq!(f["index.analysis.analyzer.a.filter"], json!(["lowercase"]));
        let nested = unflatten(&f);
        assert_eq!(nested["index"]["analysis"]["analyzer"]["a"]["filter"], json!(["lowercase"]));
        assert_eq!(nested["index"]["number_of_shards"], json!("1"));
    }

    #[test]
    fn recovery_byte_size_settings_validate_like_es() {
        let put = |k: &str, v: Value| validate_cluster_update("persistent", &[(k.to_string(), v)].into_iter().collect());
        for ok in ["4mb", "40MB", "512kb", "1.5gb", "0", "-1", "100b", "4m"] {
            assert!(put("indices.recovery.max_bytes_per_sec", json!(ok)).is_ok(), "{ok}");
        }
        let e = put("indices.recovery.max_bytes_per_sec", json!("fast")).unwrap_err();
        assert_eq!(e.reason, "failed to parse setting [indices.recovery.max_bytes_per_sec] with value [fast] as a size in bytes: unit is missing or unrecognized");
        assert!(put("indices.recovery.max_bytes_per_sec", json!("40")).is_err(), "a unit is required");
        assert!(put("indices.recovery.chunk_size", json!("0")).is_err());
        assert!(put("indices.recovery.max_concurrent_file_chunks", json!(9)).is_err());
        assert!(put("indices.recovery.max_concurrent_file_chunks", json!("0")).is_err());
        assert!(put("indices.recovery.max_concurrent_file_chunks", json!(4)).is_ok());
        assert!(put("indices.recovery.max_bytes_per_sec", Value::Null).is_ok(), "reset to the 40mb default");
        assert_eq!(parse_byte_size("40mb"), Some(40 << 20));
        assert_eq!(parse_byte_size("-1"), Some(-1));
        assert_eq!(cluster_defaults()["indices.recovery.max_bytes_per_sec"], json!("40mb"));
    }

    #[test]
    fn validation() {
        assert!(validate(&normalize_index_settings(&json!({"number_of_shards": 0}))).is_err());
        assert!(validate(&normalize_index_settings(&json!({"refresh_interval": "5x"}))).is_err());
        assert!(validate(&normalize_index_settings(&json!({"refresh_interval": "-1"}))).is_ok());
        assert!(is_dynamic("index.refresh_interval"));
        assert!(!is_dynamic("index.number_of_shards"));
    }
}
