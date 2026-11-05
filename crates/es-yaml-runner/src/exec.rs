//! Execution of YAML test sections: prerequisites, `do`, assertions, stash.

use crate::spec::Specs;
use crate::suite::TestFile;
use elasticsearch_engine::node::Node;
use serde_json::{Map, Value};
use std::time::Duration;

/// The version our server reports (`version.number`), for `skip.version`
/// ranges and the synthetic `gte_v*` cluster features.
pub const SERVER_VERSION: (u64, u64, u64) = {
    let (a, b, c) = elasticsearch_engine::version::ES_VERSION_TRIPLE;
    (a as u64, b as u64, c as u64)
};

/// Runner features (`skip.features` / `requires.test_runner_features`) we
/// implement: exactly ES's `Features.SUPPORTED`, so a test requiring anything
/// else (`regex`, `skip_os`, `fips_140`, ...) is skipped as it is in ES.
/// `xpack` / `no_xpack` are not runner features (see `xpack_skip`).
const SUPPORTED_FEATURES: &[&str] = &[
    "catch_unauthorized",
    "default_shards",
    "embedded_stash_key",
    "headers",
    "node_selector",
    "stash_in_key",
    "stash_in_path",
    "warnings",
    "warnings_regex",
    "yaml",
    "contains",
    "transform_and_set",
    "arbitrary_key",
    "allowed_warnings",
    "allowed_warnings_regex",
    "close_to",
    "is_after",
    "capabilities",
];

/// `--no-xpack`: the suite's cluster has no x-pack modules (ES's OSS
/// `rest-api-spec` suite runs on the INTEG_TEST distribution), which is what
/// ES's `hasXPack()` reports to the `xpack` / `no_xpack` skip features.
pub static NO_XPACK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// ES `PrerequisiteSection.parseFeatureField`: `xpack` requires x-pack and
/// `no_xpack` skips when it is present. `None` for any other feature.
fn xpack_skip(feature: &str) -> Option<Option<String>> {
    let has_xpack = !NO_XPACK.load(std::sync::atomic::Ordering::Relaxed);
    match feature {
        "xpack" => Some((!has_xpack).then(|| "requires xpack".to_string())),
        "no_xpack" => Some(has_xpack.then(|| "skip no_xpack".to_string())),
        _ => None,
    }
}

/// `--security`: start every node with security on, as ES's x-pack YAML
/// cluster does (`XPackRestIT`): trial license, token + API key services,
/// file user `x_pack_rest_user` (superuser), `elastic/fleet-server/token1`.
pub static SECURITY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The suite runs against ES's DEFAULT distribution (x-pack main / security
/// suites, `XPackRestIT`): every built-in template registry is installed.
pub static DEFAULT_DISTRIBUTION: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Built-in template registries present in the test cluster a file runs
/// against: (stack, apm-data, otel-data, prometheus). The stack plugin
/// also carries the query-logging registry. The x-pack main/security
/// suites and the suites whose cluster uses `DistributionType.DEFAULT`
/// (including the data-streams module's `DataStreamsClientYamlTestSuiteIT`)
/// have them all; the other plugin suites list their modules
/// (`x-pack-stack`, `x-pack-apm-data`, `x-pack-otel-data`); the OSS suite
/// (`rest-api-spec`, INTEG_TEST distribution) has none.
fn builtin_registries(dir: &str) -> (bool, bool, bool, bool) {
    if DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        return (true, true, true, true);
    }
    match dir {
        "downsample" | "ent-search" | "eql" | "inference" | "kql" | "logsdb" | "rank-rrf" | "searchable-snapshots" | "snapshot-repo-test-kit" | "prometheus" | "data_stream" => (true, true, true, true),
        "apm-data" => (true, true, false, false),
        "otel-data" => (true, false, true, false),
        "stack" => (true, false, false, false),
        _ => (false, false, false, false),
    }
}

pub const XPACK_USER: &str = "x_pack_rest_user";
pub const XPACK_PASSWORD: &str = "x-pack-test-password";

fn security_config() -> elasticsearch_engine::security::SecConfig {
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    static HASHES: std::sync::OnceLock<(String, String)> = std::sync::OnceLock::new();
    let (user_hash, token_hash) = HASHES.get_or_init(|| (Hasher::Pbkdf2(1000).hash(XPACK_PASSWORD), Hasher::Pbkdf2(1000).hash("r5wdbdboQSe9vGOKwaJGAw"))).clone();
    let mut c = SecConfig::default();
    c.enabled = true;
    c.token_enabled = true;
    c.api_key_enabled = true;
    c.trial = true;
    c.bootstrap_password = Some(XPACK_PASSWORD.into());
    c.file_users.insert(XPACK_USER.into(), user_hash);
    c.file_user_roles.insert(XPACK_USER.into(), vec!["superuser".into()]);
    c.file_service_tokens.insert("elastic/fleet-server/token1".into(), token_hash);
    c
}

fn basic(user: &str, pass: &str) -> String {
    use base64::Engine;
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{user}:{pass}")))
}

/// `WatcherYamlRestWithSecurityIT`: the watcher `security` tests run on a
/// cluster with security on, the `roles.yml` of the watcher YAML tests, the
/// client authenticated as `watcher_manager`, email account settings to
/// test settings filtering on, and one doc in `index_not_allowed_to_read`.
fn watcher_security_config() -> elasticsearch_engine::security::SecConfig {
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    let h = Hasher::Pbkdf2(1000).hash(XPACK_PASSWORD);
    let mut c = SecConfig::default();
    c.enabled = true;
    c.trial = true;
    c.api_key_enabled = true;
    for (u, role) in [("test_admin", "superuser"), ("watcher_manager", "watcher_manager"), ("x_pack_rest_user", "watcher_manager")] {
        c.file_users.insert(u.into(), h.clone());
        c.file_user_roles.insert(u.into(), vec![role.into()]);
    }
    let roles = serde_json::json!({
        "admin": {"cluster": ["all"], "indices": [{"names": ["*"], "privileges": ["all"]}]},
        "watcher_manager": {"cluster": ["manage"], "indices": [{"names": [".watch*"], "privileges": ["all"]}, {"names": ["my_test_index"], "privileges": ["all"]}],
                            "run_as": ["powerless_user", "watcher_manager", "x_pack_rest_user"]},
        "watcher_monitor": {"cluster": ["monitor"], "indices": [{"names": [".watcher-history-*"], "privileges": ["read"]}]},
        "crappy_role": {"cluster": ["cluster:monitor/nodes/info", "cluster:monitor/health", "cluster:monitor/nodes/liveness", "cluster:monitor/main"]}
    });
    for (k, v) in roles.as_object().unwrap() {
        c.file_roles.insert(k.clone(), v.clone());
    }
    c
}

fn is_watcher_security(file: &TestFile) -> bool {
    file.rel.starts_with("watcher/security/")
}

/// `FleetRestIT`'s cluster: security on, a basic license, the
/// `elastic_admin` superuser (the client's credentials) and
/// `fleet_unprivileged_secrets` with the suite's `roles.yml` role.
fn fleet_security_config() -> (elasticsearch_engine::security::SecConfig, String) {
    use base64::Engine;
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    static HASHES: std::sync::OnceLock<(String, String)> = std::sync::OnceLock::new();
    let (admin, unpriv) = HASHES.get_or_init(|| (Hasher::Pbkdf2(1000).hash("admin-password"), Hasher::Pbkdf2(1000).hash("password"))).clone();
    let mut c = SecConfig::default();
    c.enabled = true;
    c.api_key_enabled = true;
    c.file_users.insert("elastic_admin".into(), admin);
    c.file_user_roles.insert("elastic_admin".into(), vec!["superuser".into()]);
    c.file_users.insert("fleet_unprivileged_secrets".into(), unpriv);
    c.file_user_roles.insert("fleet_unprivileged_secrets".into(), vec!["unprivileged_secrets".into()]);
    c.file_roles.insert(
        "unprivileged_secrets".into(),
        serde_json::json!({"cluster": ["monitor"], "indices": [{"names": [".fleet-secrets*"], "privileges": ["write", "read"]}]}),
    );
    (c, format!("Basic {}", base64::engine::general_purpose::STANDARD.encode("elastic_admin:admin-password")))
}

/// `EnterpriseSearchRestIT`'s cluster: security on, a trial license, the
/// suite's `roles.yml` and four file users; the client authenticates as
/// `entsearch-admin`.
/// `InferenceWithSecurityRestIT` (the plugin suite's `inference` dir):
/// security on, a trial license, its `roles.yml`, the admin user
/// `x_pack_rest_user`, and the client authenticated as `inference_user`
/// (`monitor` + `manage_inference`); `monitor_inference_user` has only
/// `monitor_inference`.
fn inference_security_config() -> (elasticsearch_engine::security::SecConfig, String) {
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    static HASHES: std::sync::OnceLock<Vec<(&'static str, &'static str, &'static str, String)>> = std::sync::OnceLock::new();
    let users = HASHES.get_or_init(|| {
        [
            (XPACK_USER, XPACK_PASSWORD, "superuser"),
            ("inference_user", "inference_user_password", "monitor_only_user"),
            ("monitor_inference_user", "monitor_inference_password", "monitor_inference_only_user"),
        ]
        .into_iter()
        .map(|(u, p, r)| (u, p, r, Hasher::Pbkdf2(1000).hash(p)))
        .collect()
    });
    let mut c = SecConfig::default();
    c.enabled = true;
    c.trial = true;
    c.api_key_enabled = true;
    for (u, _, r, h) in users {
        c.file_users.insert((*u).into(), h.clone());
        c.file_user_roles.insert((*u).into(), vec![(*r).into()]);
    }
    let roles = serde_json::json!({
        "monitor_only_user": {"cluster": ["monitor", "manage_inference"], "indices": [{"names": ["test-index*"], "privileges": ["all"]}]},
        "monitor_inference_only_user": {"cluster": ["monitor_inference"]}
    });
    for (k, v) in roles.as_object().unwrap() {
        c.file_roles.insert(k.clone(), v.clone());
    }
    (c, basic("inference_user", "inference_user_password"))
}

fn entsearch_security_config() -> (elasticsearch_engine::security::SecConfig, String) {
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    static HASHES: std::sync::OnceLock<Vec<(&'static str, &'static str, String)>> = std::sync::OnceLock::new();
    let users = HASHES.get_or_init(|| {
        [("entsearch-superuser", "superuser"), ("entsearch-admin", "admin"), ("entsearch-user", "user"), ("entsearch-unprivileged", "unprivileged")]
            .into_iter()
            .map(|(u, r)| (u, r, Hasher::Pbkdf2(1000).hash(&format!("{u}-password"))))
            .collect()
    });
    let mut c = SecConfig::default();
    c.enabled = true;
    c.trial = true;
    c.api_key_enabled = true;
    for (u, r, h) in users {
        c.file_users.insert((*u).into(), h.clone());
        c.file_user_roles.insert((*u).into(), vec![(*r).into()]);
    }
    let roles = serde_json::json!({
        "admin": {
            "cluster": ["manage_search_application", "manage_behavioral_analytics", "manage", "monitor"],
            "indices": [{"names": ["test-*", "another-test-search-application", ".elastic-connectors-v1", ".elastic-connectors-sync-jobs-v1"], "privileges": ["manage", "write", "read"]}]
        },
        "user": {
            "cluster": ["post_behavioral_analytics_event", "manage_api_key", "read_connector_secrets", "write_connector_secrets"],
            "indices": [{"names": ["test-index1", "test-search-application", "test-search-application-1", "test-search-application-with-aggs",
                                   "test-search-application-with-list", "test-search-application-with-list-invalid", ".elastic-connectors-v1",
                                   ".elastic-connectors-sync-jobs-v1"], "privileges": ["read"]}]
        },
        "unprivileged": {
            "indices": [{"names": ["test-*", "another-test-search-application"], "privileges": ["manage", "write", "read"]}]
        }
    });
    for (k, v) in roles.as_object().unwrap() {
        c.file_roles.insert(k.clone(), v.clone());
    }
    (c, basic("entsearch-admin", "entsearch-admin-password"))
}

/// `SearchableSnapshotsClientYamlTestSuiteIT`'s cluster: security on, a
/// trial license, the client authenticated as the `admin` superuser.
fn admin_superuser_config() -> (elasticsearch_engine::security::SecConfig, String) {
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    static HASH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let h = HASH.get_or_init(|| Hasher::Pbkdf2(1000).hash("admin-password")).clone();
    let mut c = SecConfig::default();
    c.enabled = true;
    c.trial = true;
    c.api_key_enabled = true;
    c.file_users.insert("admin".into(), h);
    c.file_user_roles.insert("admin".into(), vec!["superuser".into()]);
    (c, basic("admin", "admin-password"))
}

/// `CcrRestIT`: security on, a trial license and `ccr-user` (superuser)
/// as the REST client's user.
fn ccr_security_config() -> (elasticsearch_engine::security::SecConfig, String) {
    use elasticsearch_engine::security::{hash::Hasher, SecConfig};
    static HASH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let h = HASH.get_or_init(|| Hasher::Pbkdf2(1000).hash("ccr-user-password")).clone();
    let mut c = SecConfig::default();
    c.enabled = true;
    c.trial = true;
    c.api_key_enabled = true;
    c.file_users.insert("ccr-user".into(), h);
    c.file_user_roles.insert("ccr-user".into(), vec!["superuser".into()]);
    (c, basic("ccr-user", "ccr-user-password"))
}

fn default_auth() -> Option<String> {
    if SECURITY.load(std::sync::atomic::Ordering::Relaxed) {
        use base64::Engine;
        Some(format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{XPACK_USER}:{XPACK_PASSWORD}"))))
    } else {
        None
    }
}

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
pub enum Outcome {
    Pass,
    Fail { msg: String, category: String },
    Skip(String),
}

pub struct Failure {
    pub msg: String,
    pub category: String,
}

fn fail(msg: impl Into<String>, category: impl Into<String>) -> Failure {
    Failure { msg: msg.into(), category: category.into() }
}

fn trunc(s: &str, n: usize) -> String {
    // `ES_YAML_FULL=1`: untruncated failure messages (debugging aid).
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *FULL.get_or_init(|| std::env::var("ES_YAML_FULL").is_ok_and(|v| v == "1")) || s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push('…');
        t
    }
}

const KEEP_AFTER: &[&str] = &["type", "query", "parameter", "parameters", "aggregation", "field_type", "mapper", "setting", "processor", "aggregations", "method", "key"];

/// Normalise a free-text message so similar failures group together.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut depth = 0usize;
    while let Some(c) = chars.next() {
        match c {
            '[' => {
                // keep short names after words that identify a feature (type/query/parameter...)
                let prev = out.trim_end().rsplit(' ').next().unwrap_or("").to_string();
                let keep = depth == 0 && KEEP_AFTER.contains(&prev.as_str());
                if keep {
                    let mut inner = String::new();
                    let mut closed = false;
                    for c2 in chars.by_ref() {
                        if c2 == ']' {
                            closed = true;
                            break;
                        }
                        inner.push(c2);
                    }
                    if closed && inner.len() <= 40 {
                        out.push('[');
                        out.push_str(&inner);
                        out.push(']');
                    } else {
                        out.push_str("[*]");
                    }
                    continue;
                }
                if depth == 0 {
                    out.push_str("[*]");
                }
                depth += 1;
            }
            ']' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            '0'..='9' => {
                while matches!(chars.peek(), Some('0'..='9') | Some('.')) {
                    chars.next();
                }
                out.push('N');
            }
            _ => out.push(c),
        }
    }
    trunc(&out, 160)
}

/// Historical features: behaviours every 8.19 node has, which real ES has dropped
/// from its node-feature list (ES's test framework keeps the same table in
/// `historical-features.json`). Each is implemented and exercised by the suite.
const HISTORICAL_FEATURES: &[&str] = &[
    // vectors / kNN
    "search.vectors.no_dimensions_bugfix",
    "search.vectors.nested_knn_inner_hits_match_query_phase_scoring",
    "search.knn.query.bugfix.130254",
    "mapper.dense_vector.rescore_vector",
    "mapper.dense_vector.rescore_zero_vector",
    "mapper.dense_vector.default_oversample_value_for_bbq",
    "mapper.dense_vector.dynamic_template_nested_object_fix",
    "mapper.npe_on_dims_update_fix",
    // retrievers
    "search.retriever.rescorer.enabled",
    "standard_retriever.fix_for_empty_query_in_compound_retrievers",
    "rrf_retriever.multi_fields_query_format_support",
    "linear_retriever.l2_norm",
    "linear_retriever_minscore_fix",
    "linear_retriever.minmax_single_doc_fix",
    "linear_retriever.multi_fields_query_format_support",
    "retriever.negative_rank_window_size_fix",
    "nested_retriever_min_score_total_hits_fix",
    "inner_retrievers_filter_support",
    "pinned_retriever_supported",
    "text_similarity_reranker_minscore_fix",
    "text_similarity_reranker_alias_handling_fix",
    // synthetic _source / stored fields
    "mapper.source.mode_from_index_setting",
    "mapper.ignored_source.dont_expand_dots",
    "mapper.ignored_source.always_store_object_arrays_in_nested",
    "mapper.synthetic_recovery_source",
    "mapper.range.date_range_indexing_fix",
    // mapping layer
    "mapper.subobjects_false_mapping_update_fix",
    "mapper.multi_field.unicode_optimisation_fix",
    "mapper.copy_to.dynamic_handling",
    "mapper.unknown_field_mapping_update_error_message",
    "search.completion_field.duplicate.support",
    "index.shadowing_dimensions_and_metrics_is_valid_in_non_tsdb",
    "mapper.tsdb_nested_field_support",
    "index.synonyms_set_lenient_on_non_existing",
    "synonyms_set.get.return_empty_synonym_sets",
    "mapper.fix_parsing_subobjects_false_dynamic_false",
    "mapper.range.invalid_date_fix",
    "search.sort.date_format_missing_as_null",
    "search.sort.int_sort_for_int_short_byte_fields",
    "meta_fetch_fields_error_code_changed",
    "index.logsdb_no_host_name_field",
    "mapper.logsdb_default_ignore_dynamic_beyond_limit",
    "routing.logsb_route_on_sort_fields",
    "mapper.sparse_vector.store_support",
    "mapper.nested.sorting_fields_check_fix",
];

/// Debug aid for finding what a claim would unlock: `ES_YAML_CLAIM_ALL=1` treats every
/// named feature and capability as present. Never set in normal runs.
fn claim_all() -> bool {
    std::env::var_os("ES_YAML_CLAIM_ALL").is_some()
}

fn has_cluster_feature(f: &str, node_features: &std::collections::HashSet<String>) -> bool {
    if claim_all() && !f.starts_with("gte_v") {
        return true;
    }
    match f.strip_prefix("gte_v") {
        Some(v) => parse_version(v).map(|x| x <= SERVER_VERSION).unwrap_or(false),
        None => node_features.contains(f) || HISTORICAL_FEATURES.contains(&f) || crate::compat::assumed_feature(f),
    }
}

fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut it = s.split(|c: char| c == '.' || c == '-').map(|p| p.trim().parse::<u64>());
    let a = it.next()?.ok()?;
    let b = it.next().and_then(|r| r.ok()).unwrap_or(0);
    let c = it.next().and_then(|r| r.ok()).unwrap_or(0);
    Some((a, b, c))
}

fn version_in_ranges(spec: &str) -> bool {
    let spec = spec.trim();
    if spec == "all" {
        return true;
    }
    for range in spec.split(',') {
        let r = range.trim();
        let (lo, hi) = match r.split_once('-') {
            Some((l, h)) => (parse_version(l).unwrap_or((0, 0, 0)), parse_version(h).unwrap_or((u64::MAX, u64::MAX, u64::MAX))),
            None => match parse_version(r) {
                Some(v) => (v, v),
                None => continue,
            },
        };
        if lo <= SERVER_VERSION && SERVER_VERSION <= hi {
            return true;
        }
    }
    false
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => vec![],
    }
}

fn is_prereq(step: &Value) -> bool {
    step.as_object().map(|o| o.contains_key("skip") || o.contains_key("requires")).unwrap_or(false)
}

fn split_path(p: &str) -> Vec<String> {
    // ES `ObjectPath.parsePath`: a backslash is dropped and escapes the next
    // `.` (wherever it comes); empty segments are skipped.
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut escape = false;
    for c in p.chars() {
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '.' {
            if escape {
                escape = false;
            } else {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                continue;
            }
        }
        cur.push(c);
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

fn scalar_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}

fn is_regex(s: &str) -> Option<&str> {
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('/') && t.ends_with('/') {
        Some(&t[1..t.len() - 1])
    } else {
        None
    }
}

/// Render a JSON value the way Java's `Map.toString()` / `List.toString()` would,
/// since the ES yaml runner matches `catch` regexes against `error.toString()`
/// where `error` is a parsed `LinkedHashMap`. Maps render as `{k=v, k2=v2}`
/// (space after `{` and after each `,`), lists as `[a, b]`, strings unquoted.
fn java_map_to_string(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let parts: Vec<String> = m.iter().map(|(k, val)| format!("{k}={}", java_map_to_string(val))).collect();
            format!("{{ {} }}", parts.join(", "))
        }
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(java_map_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
    }
}

/// `comments`: Java `Pattern.COMMENTS` (used by `match`, not by `catch`).
fn regex_find(pat: &str, text: &str, comments: bool) -> Result<bool, String> {
    let re = regex::Regex::new(&if comments { format!("(?x){pat}") } else { pat.to_string() }).map_err(|e| format!("unsupported regex: {}", trunc(&e.to_string(), 80)))?;
    // Java's `$` also matches right before a final line terminator; Rust's does not.
    Ok(re.is_match(text) || text.strip_suffix('\n').map(|t| re.is_match(t)).unwrap_or(false))
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn json_eq(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let same = match (a.as_i64(), e.as_i64()) {
                (Some(x), Some(y)) => x == y,
                _ => match (a.as_u64(), e.as_u64()) {
                    (Some(x), Some(y)) => x == y,
                    _ => a.as_f64() == e.as_f64(),
                },
            };
            if same { Ok(()) } else { Err(format!("at [{path}] expected {e} but was {a}")) }
        }
        (Value::Object(a), Value::Object(e)) => {
            for (k, ev) in e {
                match a.get(k) {
                    Some(av) => json_eq(av, ev, &format!("{path}.{k}"))?,
                    None => {
                        if !ev.is_null() {
                            return Err(format!("at [{path}] missing key [{k}]"));
                        }
                    }
                }
            }
            for k in a.keys() {
                if !e.contains_key(k) {
                    return Err(format!("at [{path}] unexpected key [{k}]"));
                }
            }
            Ok(())
        }
        (Value::Array(a), Value::Array(e)) => {
            if a.len() != e.len() {
                return Err(format!("at [{path}] expected {} elements but was {}", e.len(), a.len()));
            }
            for (i, (av, ev)) in a.iter().zip(e).enumerate() {
                json_eq(av, ev, &format!("{path}.{i}"))?;
            }
            Ok(())
        }
        (a, e) if a == e => Ok(()),
        (a, Value::String(s)) if is_regex(s).is_some() => {
            let pat = is_regex(s).unwrap_or("");
            match regex_find(pat, &scalar_string(a), true) {
                Ok(true) => Ok(()),
                Ok(false) => Err(format!("at [{path}] {} does not match {}", trunc(&scalar_string(a), 200), trunc(s, 200))),
                Err(e) => Err(e),
            }
        }
        (a, e) => Err(format!("at [{path}] expected {} but was {}", trunc(&e.to_string(), 200), trunc(&a.to_string(), 200))),
    }
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(v) => {
            let s = scalar_string(v);
            !(s.is_empty() || s.eq_ignore_ascii_case("false") || s == "0")
        }
    }
}

struct Last {
    body: Value,
    text: String,
    api: String,
}

pub struct Ctx<'a> {
    specs: &'a Specs,
    client: reqwest::Client,
    base: String,
    stash: Map<String, Value>,
    last: Option<Last>,
    /// Features the node reports (`nodes_features` of `GET _cluster/state/nodes`), fetched lazily.
    node_features: Option<std::collections::HashSet<String>>,
    required_coverage: Vec<String>,
    /// `Authorization` the client sends by default (ES's `restClientSettings`).
    auth: Option<String>,
}

fn param_string(v: &Value) -> String {
    match v {
        Value::Array(a) => a.iter().map(scalar_string).collect::<Vec<_>>().join(","),
        other => scalar_string(other),
    }
}

fn encode(s: &str, keep: &[u8]) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) || keep.contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

struct Built {
    method: String,
    url: String,
    body: Option<Vec<u8>>,
    content_type: &'static str,
    ignore: Vec<u16>,
}

impl<'a> Ctx<'a> {
    fn new(specs: &'a Specs, base: String) -> Self {
        let client = reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build().expect("http client");
        Ctx { specs, client, base, stash: Map::from_iter([("_project_id_prefix_".into(), Value::String(String::new()))]), last: None, node_features: None, required_coverage: Vec::new(), auth: default_auth() }
    }


    async fn get_json(&self, path: &str) -> Option<Value> {
        let mut rb = self.client.get(format!("{}{}", self.base, path));
        if let Some(a) = self.auth.clone() {
            rb = rb.header("Authorization", a);
        }
        rb.send().await.ok()?.json().await.ok()
    }

    /// Ask the node which features it has, like ES's runner does via cluster state.
    async fn features(&mut self) -> &std::collections::HashSet<String> {
        if self.node_features.is_none() {
            let mut set = std::collections::HashSet::new();
            if let Some(v) = self.get_json("/_cluster/state/nodes").await {
                for n in v["nodes_features"].as_array().into_iter().flatten() {
                    for f in n["features"].as_array().into_iter().flatten() {
                        if let Some(f) = f.as_str() {
                            set.insert(f.to_string());
                        }
                    }
                }
            }
            self.node_features = Some(set);
        }
        self.node_features.as_ref().expect("set above")
    }

    /// `GET _capabilities`: does the node support this method/path/parameters/capabilities?
    async fn has_capabilities(&self, c: &Value) -> bool {
        if claim_all() {
            return true;
        }
        let list = |k: &str| str_list(&c[k]).join(",");
        let method = c["method"].as_str().unwrap_or("GET");
        let path = encode(c["path"].as_str().unwrap_or("/"), b"/");
        let url = format!(
            "/_capabilities?method={}&path={}&parameters={}&capabilities={}",
            method,
            path,
            encode(&list("parameters"), b","),
            encode(&list("capabilities"), b",")
        );
        self.get_json(&url).await.map(|v| v["supported"] == Value::Bool(true)).unwrap_or(false)
    }

    /// If `step` is a `skip`/`requires` prerequisite that says to skip, the reason.
    async fn prereq_skip(&mut self, step: &Value) -> Option<String> {
        let obj = step.as_object()?;
        for (op, body) in obj {
            match op.as_str() {
                "skip" => {
                    for f in str_list(&body["features"]) {
                        if let Some(skip) = xpack_skip(&f) {
                            if skip.is_some() {
                                return skip;
                            }
                            continue;
                        }
                        if !SUPPORTED_FEATURES.contains(&f.as_str()) {
                            return Some(format!("runner feature `{f}`"));
                        }
                    }
                    if let Some(v) = body["version"].as_str() {
                        if version_in_ranges(v) {
                            return Some("version range".into());
                        }
                    }
                    for f in str_list(&body["cluster_features"]) {
                        let nf = self.features().await.clone();
                        if has_cluster_feature(&f, &nf) {
                            return Some(format!("skip cluster feature `{f}`"));
                        }
                    }
                    if let Some(ki) = body["known_issues"].as_array() {
                        let nf = self.features().await.clone();
                        for i in ki {
                            let cf = i["cluster_feature"].as_str().unwrap_or("");
                            let fx = i["fixed_by"].as_str().unwrap_or("");
                            if has_cluster_feature(cf, &nf) && !has_cluster_feature(fx, &nf) {
                                return Some("known issue".into());
                            }
                        }
                    }
                    if !body["awaits_fix"].is_null() {
                        return Some("awaits_fix".into());
                    }
                    if let Some(caps) = body["capabilities"].as_array() {
                        for c in caps {
                            if self.has_capabilities(c).await {
                                return Some(format!("skip capability `{}`", str_list(&c["capabilities"]).join(",")));
                            }
                        }
                    }
                }
                "requires" => {
                    let mut feats = str_list(&body["test_runner_features"]);
                    feats.extend(str_list(&body["features"]));
                    for f in feats {
                        if let Some(skip) = xpack_skip(&f) {
                            if skip.is_some() {
                                return skip;
                            }
                            continue;
                        }
                        if !SUPPORTED_FEATURES.contains(&f.as_str()) {
                            return Some(format!("required runner feature unsupported: `{f}`"));
                        }
                    }
                    for f in str_list(&body["cluster_features"]) {
                        let nf = self.features().await.clone();
                        if !has_cluster_feature(&f, &nf) {
                            if has_cluster_feature(&f, crate::compat::reference_features()) {
                                self.required_coverage.push(format!("missing reference-required cluster feature `{f}`"));
                            } else {
                                return Some(format!("reference-inapplicable cluster feature `{f}`"));
                            }
                        }
                    }
                    if let Some(caps) = body["capabilities"].as_array() {
                        for c in caps {
                            if !self.has_capabilities(c).await {
                                let names = str_list(&c["capabilities"]);
                                let what = if names.is_empty() { format!("{} params `{}`", c["path"].as_str().unwrap_or(""), str_list(&c["parameters"]).join(",")) } else { names.join(",") };
                                if crate::compat::reference_capability(c) == Some(false) {
                                    return Some(format!("reference-inapplicable capability `{what}`"));
                                }
                                self.required_coverage.push(format!("missing or unclassified required capability `{what}`"));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn body(&self) -> Value {
        self.last.as_ref().map(|l| l.body.clone()).unwrap_or(Value::Null)
    }

    fn api(&self) -> String {
        self.last.as_ref().map(|l| l.api.clone()).unwrap_or_else(|| "-".into())
    }

    /// Look up `$name.rest.of.path` in the stash.
    fn stash_get(&self, key: &str) -> Result<Value, String> {
        let segs = split_path(key);
        let root = match self.stash.get(&segs[0]) {
            Some(v) => v.clone(),
            None => return Err(format!("stashed value not found for key [{key}]")),
        };
        if segs.len() == 1 {
            return Ok(root);
        }
        let rest = segs[1..].join(".");
        self.eval_segments(&root, &split_path(&rest)).ok_or_else(|| format!("stashed path not found [{key}]"))
    }

    fn interpolate(&self, s: &str) -> Result<String, String> {
        if !s.contains("${") {
            return Ok(s.to_string());
        }
        let mut out = String::new();
        let mut rest = s;
        while let Some(i) = rest.find("${") {
            out.push_str(&rest[..i]);
            let after = &rest[i + 2..];
            match after.find('}') {
                Some(j) => {
                    out.push_str(&scalar_string(&self.stash_get(&after[..j])?));
                    rest = &after[j + 1..];
                }
                None => {
                    out.push_str(&rest[i..]);
                    rest = "";
                }
            }
        }
        out.push_str(rest);
        Ok(out)
    }

    fn unstash(&self, v: &Value) -> Result<Value, String> {
        Ok(match v {
            Value::String(s) => {
                if s.len() > 1 && s.starts_with('$') && !s.starts_with("${") {
                    self.stash_get(&s[1..])?
                } else {
                    Value::String(self.interpolate(s)?)
                }
            }
            Value::Array(a) => Value::Array(a.iter().map(|x| self.unstash(x)).collect::<Result<_, _>>()?),
            Value::Object(m) => {
                let mut out = Map::new();
                for (k, x) in m {
                    let key = if k.len() > 1 && k.starts_with('$') && !k.starts_with("${") {
                        scalar_string(&self.stash_get(&k[1..])?)
                    } else {
                        self.interpolate(k)?
                    };
                    out.insert(key, self.unstash(x)?);
                }
                Value::Object(out)
            }
            other => other.clone(),
        })
    }

    fn eval_segments(&self, root: &Value, segs: &[String]) -> Option<Value> {
        let mut cur = root.clone();
        for (i, raw) in segs.iter().enumerate() {
            if raw.is_empty() && segs.len() == 1 {
                return Some(cur);
            }
            let seg = if raw.len() > 1 && raw.starts_with('$') && !raw.starts_with("${") {
                scalar_string(&self.stash.get(&raw[1..])?.clone())
            } else {
                self.interpolate(raw).ok()?
            };
            if seg == "_arbitrary_key_" {
                let (k, v) = cur.as_object()?.iter().next().map(|(k, v)| (k.clone(), v.clone()))?;
                if i + 1 == segs.len() {
                    return Some(Value::String(k));
                }
                cur = v;
                continue;
            }
            cur = match &cur {
                Value::Object(m) => m.get(&seg)?.clone(),
                Value::Array(a) => a.get(seg.parse::<usize>().ok()?)?.clone(),
                _ => return None,
            };
        }
        Some(cur)
    }

    /// Evaluate an assertion path against the last response.
    fn eval(&self, path: &str) -> Option<Value> {
        if path.is_empty() || path == "$body" {
            return self.last.as_ref().map(|l| l.body.clone());
        }
        let segs = split_path(path);
        if segs[0] == "$body" {
            return self.eval_segments(&self.body(), &segs[1..]);
        }
        // A bare stashed variable (`contains: { $composite_id: ... }`) evaluates to its value.
        if segs.len() == 1 && segs[0].len() > 1 && segs[0].starts_with('$') && !segs[0].starts_with("${") {
            if let Some(v) = self.stash.get(&segs[0][1..]) {
                return Some(v.clone());
            }
        }
        // ES `Stash.getValue("$profile.uid")`: a path that starts with a
        // stashed object evaluates the rest of the path inside it.
        if segs.len() > 1 && segs[0].len() > 1 && segs[0].starts_with('$') && !segs[0].starts_with("${") {
            if let Some(v @ (Value::Object(_) | Value::Array(_))) = self.stash.get(&segs[0][1..]) {
                return self.eval_segments(v, &segs[1..]);
            }
        }
        self.eval_segments(&self.body(), &segs)
    }

    fn build(&self, api_name: &str, args: &Map<String, Value>) -> Result<Built, Failure> {
        let api = self.specs.apis.get(api_name).ok_or_else(|| fail(format!("unknown api [{api_name}]"), format!("runner: unknown api [{api_name}]")))?;
        let mut params: Map<String, Value> = args.clone();
        // ES's client drops null-valued params (e.g. `user: null` selects the path without `{user}`).
        params.retain(|k, v| k == "body" || !v.is_null());
        let body = params.remove("body");
        // `method` overrides the HTTP method unless the API has a real `method` param.
        let method_override = if api.params.contains("method") { None } else { params.remove("method").map(|m| scalar_string(&m).to_uppercase()) };
        let ignore: Vec<u16> = match params.remove("ignore") {
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|n| n as u16)).collect(),
            Some(v) => num(&v).map(|n| vec![n as u16]).unwrap_or_default(),
            None => vec![],
        };
        let all_parts: std::collections::HashSet<&str> = api.paths.iter().flat_map(|p| p.parts.iter().map(String::as_str)).collect();
        for k in params.keys() {
            if !api.params.contains(k) && !self.specs.common.contains(k) && !all_parts.contains(k.as_str()) {
                return Err(fail(format!("param [{k}] not supported by [{api_name}]"), "client: unknown param".to_string()));
            }
        }
        // ES's `ClientYamlTestExecutionContext.callApi` asks for error traces
        // unless the test sets `error_trace` (`false` drops the parameter).
        match params.get("error_trace").map(scalar_string).as_deref() {
            None => {
                params.insert("error_trace".into(), Value::String("true".into()));
            }
            Some("false") => {
                params.remove("error_trace");
            }
            Some(_) => {}
        }
        let fits = |p: &&crate::spec::ApiPath| p.parts.iter().all(|x| params.contains_key(x));
        let maxp = api.paths.iter().filter(fits).map(|p| p.parts.len()).max()
            .ok_or_else(|| fail(format!("no path of [{api_name}] matches the given parts"), "client: no matching path".to_string()))?;
        let chosen = api.paths.iter().filter(fits).find(|p| p.parts.len() == maxp).expect("max exists");
        let mut path = chosen.path.clone();
        for part in &chosen.parts {
            let v = params.remove(part).unwrap_or(Value::Null);
            path = path.replace(&format!("{{{part}}}"), &encode(&param_string(&v), b",*:@!$&'()+;="));
        }
        // The Java runner asks for error traces unless a test opts out
        // (`ClientYamlTestExecutionContext.callApi`).
        match params.get("error_trace").map(param_string).as_deref() {
            None => {
                params.insert("error_trace".into(), Value::String("true".into()));
            }
            Some("false") => {
                params.remove("error_trace");
            }
            _ => {}
        }
        let mut url = format!("{}{}", self.base, path);
        let mut first = true;
        for (k, v) in &params {
            url.push(if first { '?' } else { '&' });
            first = false;
            url.push_str(&encode(k, b""));
            url.push('=');
            url.push_str(&encode(&param_string(v), b",*:"));
        }
        let has_body = body.as_ref().map(|b| !b.is_null()).unwrap_or(false);
        let methods = &chosen.methods;
        let pick = |m: &str| methods.iter().any(|x| x == m);
        let method = if has_body {
            if pick("POST") {
                "POST"
            } else if pick("PUT") {
                "PUT"
            } else {
                methods.first().map(String::as_str).unwrap_or("GET")
            }
        } else if pick("GET") {
            "GET"
        } else {
            methods.first().map(String::as_str).unwrap_or("GET")
        }
        .to_string();
        let method = method_override.unwrap_or(method);
        let (body, content_type) = match body {
            None | Some(Value::Null) => (None, "application/json"),
            Some(Value::String(s)) => {
                if api.bulk_body {
                    // ES's DoSection parses a string body as a stream of JSON
                    // objects (one per bulk line), so pretty-printed documents
                    // spanning several lines are sent compacted.
                    let docs: Result<Vec<Value>, _> = serde_json::Deserializer::from_str(&s).into_iter::<Value>().collect();
                    let mut s = match docs {
                        Ok(d) if !d.is_empty() && s.trim_end().lines().count() > d.len() => d.iter().map(|v| v.to_string() + "\n").collect(),
                        _ => s,
                    };
                    if !s.ends_with('\n') {
                        s.push('\n');
                    }
                    (Some(s.into_bytes()), "application/x-ndjson")
                } else {
                    (Some(s.into_bytes()), "application/json")
                }
            }
            Some(Value::Array(a)) if api.bulk_body => {
                let mut s = String::new();
                // A YAML list item holding only a comment (`- # note`) is null: ES's runner drops it.
                for line in a.into_iter().filter(|l| !l.is_null()) {
                    match line {
                        // A folded/literal YAML scalar may span lines; one JSON document
                        // per bulk line, so compact it (as the ES test client does).
                        Value::String(l) if l.trim_end_matches('\n').contains('\n') && serde_json::from_str::<Value>(&l).is_ok() => {
                            s.push_str(&serde_json::from_str::<Value>(&l).map(|v| v.to_string()).unwrap_or_default())
                        }
                        Value::String(l) => s.push_str(l.trim_end_matches('\n')),
                        other => s.push_str(&other.to_string()),
                    }
                    s.push('\n');
                }
                (Some(s.into_bytes()), "application/x-ndjson")
            }
            // A one-element body list is that single body (ES ClientYamlTestExecutionContext.createEntity).
            Some(Value::Array(a)) if a.len() == 1 => match &a[0] {
                Value::String(s) => (Some(s.clone().into_bytes()), "application/json"),
                v => (Some(v.to_string().into_bytes()), "application/json"),
            },
            Some(v) => (Some(v.to_string().into_bytes()), if api.bulk_body { "application/x-ndjson" } else { "application/json" }),
        };
        Ok(Built { method, url, body, content_type, ignore })
    }

    async fn do_step(&mut self, d: &Map<String, Value>) -> Result<(), Failure> {
        let mut catch: Option<String> = None;
        let mut headers: Vec<(String, String)> = vec![];
        let mut call: Option<(String, Map<String, Value>)> = None;
        let mut warn_spec: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for (k, v) in d {
            match k.as_str() {
                "catch" => catch = Some(scalar_string(v)),
                "headers" => {
                    if let Some(m) = v.as_object() {
                        for (hk, hv) in m {
                            let hv = self.unstash(hv).map_err(|e| fail(e.clone(), "runner: stash"))?;
                            headers.push((hk.clone(), scalar_string(&hv)));
                        }
                    }
                }
                "warnings" | "allowed_warnings" | "warnings_regex" | "allowed_warnings_regex" => {
                    let list: Vec<String> = v.as_array().map(|a| a.iter().map(scalar_string).collect()).unwrap_or_default();
                    warn_spec.insert(k.clone(), list);
                }
                // The node is a single 8.19 node: every selector the suite uses matches it.
                "node_selector" => {}
                api => {
                    // Like ES's runner, a JSON string body is parsed before stash replacement.
                    let mut v = v.clone();
                    if let Some(Value::String(b)) = v.get("body") {
                        if b.contains('$') {
                            if let Ok(parsed @ (Value::Object(_) | Value::Array(_))) = serde_json::from_str::<Value>(b) {
                                v["body"] = parsed;
                            }
                        }
                    }
                    let args = match self.unstash(&v).map_err(|e| fail(e.clone(), "runner: stash"))? {
                        Value::Object(m) => m,
                        _ => Map::new(),
                    };
                    call = Some((api.to_string(), args));
                }
            }
        }
        let Some((api, args)) = call else { return Err(fail("do without api", "runner: do without api")) };
        // `--compat`: ES's `InjectHeaders` (every `do` but `cat.*`; replaces the test's own values).
        if let Some(major) = crate::compat::compat_major() {
            if crate::compat::injects_headers(&api) {
                headers.retain(|(k, _)| !k.eq_ignore_ascii_case("content-type") && !k.eq_ignore_ascii_case("accept"));
                headers.push(("Content-Type".into(), crate::compat::header_value(major)));
                headers.push(("Accept".into(), crate::compat::header_value(major)));
            }
        }
        let built = match self.build(&api, &args) {
            Ok(b) => b,
            Err(f) => {
                if catch.as_deref() == Some("param") {
                    self.last = None;
                    return Ok(());
                }
                return Err(f);
            }
        };
        if catch.as_deref() == Some("param") {
            return Err(fail(format!("{api}: expected client-side param error"), format!("{api}: expected `param` catch")));
        }
        if std::env::var_os("ES_YAML_DEBUG").is_some() {
            eprintln!(">> {} {} {}", built.method, built.url, built.body.as_ref().map(|b| String::from_utf8_lossy(b).chars().take(2000).collect::<String>()).unwrap_or_default());
        }
        let method = reqwest::Method::from_bytes(built.method.as_bytes()).unwrap_or(reqwest::Method::GET);
        let mut rb = self.client.request(method, &built.url);
        if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("authorization")) {
            if let Some(a) = self.auth.clone() {
                rb = rb.header("Authorization", a);
            }
        }
        let mut has_ct = false;
        for (k, v) in &headers {
            if k.eq_ignore_ascii_case("content-type") {
                has_ct = true;
            }
            rb = rb.header(k.as_str(), v.as_str());
        }
        if let Some(b) = built.body {
            if !has_ct {
                rb = rb.header("Content-Type", built.content_type);
            }
            rb = rb.body(b);
        }
        let resp = rb.send().await.map_err(|e| {
            let kind = if e.is_timeout() { "timeout" } else { "connection error (server panic?)" };
            fail(format!("{api} {} {}: {kind}: {e}", built.method, built.url), format!("{api}: request {kind}"))
        })?;
        let status = resp.status().as_u16();
        let ct = resp.headers().get("content-type").and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
        let warnings: Vec<String> = resp.headers().get_all("warning").iter().filter_map(|h| h.to_str().ok()).map(warning_message).collect();
        let bytes = resp.bytes().await.map_err(|e| fail(format!("{api}: reading body: {e}"), format!("{api}: body read error")))?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        if std::env::var_os("ES_YAML_DEBUG").is_some() {
            eprintln!("<< {status} {}", text.chars().take(4000).collect::<String>());
        }
        let body = if built.method == "HEAD" {
            Value::Bool(status == 200)
        } else if ct.contains("yaml") && !text.trim().is_empty() {
            serde_yaml::from_str::<serde_yaml::Value>(&text).map(crate::suite::yaml_to_json).unwrap_or_else(|_| Value::String(text.clone()))
        } else if ct.contains("json") && !text.trim().is_empty() {
            serde_json::from_str(&text).unwrap_or_else(|_| Value::String(text.clone()))
        } else {
            Value::String(text.clone())
        };
        let err_type = body["error"]["type"].as_str().unwrap_or("").to_string();
        let mut err_reason = body["error"]["reason"].as_str().map(str::to_string).unwrap_or_else(|| match &body["error"] {
            Value::String(s) => s.clone(),
            _ => String::new(),
        });
        let mut err_type = err_type;
        if err_reason == "all shards failed" || err_type == "search_phase_execution_exception" {
            let rc = &body["error"]["root_cause"][0];
            let fc = &body["error"]["failed_shards"][0]["reason"];
            let inner = if rc.is_object() { rc } else { fc };
            if let Some(r) = inner["reason"].as_str() {
                err_type = format!("{err_type}/{}", inner["type"].as_str().unwrap_or(""));
                err_reason = r.to_string();
            }
        }
        self.stash.insert("body".into(), body.clone());
        self.last = Some(Last { body: body.clone(), text: text.clone(), api: api.clone() });
        let path_shape = built.url.split('?').next().unwrap_or("").trim_start_matches(&self.base).to_string();
        let is_err = status >= 400 && !(built.method == "HEAD" && status == 404) && !built.ignore.contains(&status);
        match catch {
            None => {
                check_warnings(&api, &warn_spec, &warnings)?;
                if is_err {
                    return Err(fail(
                        format!("{api} {} {path_shape} -> {status} {err_type}: {}", built.method, trunc(&err_reason, 300)),
                        format!("{api}: unexpected {status} {err_type}: {}", normalize(&err_reason)),
                    ));
                }
            }
            Some(c) => {
                let ok = match c.as_str() {
                    "bad_request" => status == 400,
                    "unauthorized" => status == 401,
                    "forbidden" => status == 403,
                    "missing" => status == 404,
                    "request_timeout" => status == 408,
                    "conflict" => status == 409,
                    "unavailable" => status == 503,
                    "request" => status >= 400 && ![400, 401, 403, 404, 408, 409].contains(&status),
                    other => match is_regex(other) {
                        Some(pat) => {
                            // ES matches the regex against `error.toString()` where `error` is the
                            // parsed response's `error` object (a LinkedHashMap). Java's Map.toString
                            // renders `{key=value, ...}` (space after `{` and after each `,`), so catch
                            // patterns often begin with a leading space. Reproduce that rendering, but
                            // also fall back to the raw body text: some patterns (e.g. a leading space
                            // before the reason) only match the raw JSON, not the map rendering.
                            let err_obj = match body.get("error") {
                                Some(e) => java_map_to_string(e),
                                None => err_reason.clone(),
                            };
                            let found = regex_find(pat, &err_obj, false)
                                .or_else(|_| regex_find(pat, &text, false))
                                .map_err(|e| fail(e.clone(), format!("runner: {e}")))?;
                            status >= 400 && found
                        }
                        None => return Err(fail(format!("unknown catch [{other}]"), "runner: unknown catch")),
                    },
                };
                if !ok {
                    let cat = if is_regex(&c).is_some() { "/regex/".to_string() } else { c.clone() };
                    return Err(fail(
                        format!("{api} {} {path_shape}: expected catch [{c}] but got {status} {err_type}: {}", built.method, trunc(&err_reason, 300)),
                        if status < 400 {
                            format!("{api}: expected error `{cat}` but request succeeded ({status})")
                        } else {
                            format!("{api}: expected `{cat}` got {status} {err_type}")
                        },
                    ));
                }
            }
        }
        Ok(())
    }

    fn assertion(&mut self, op: &str, arg: &Value) -> Result<(), Failure> {
        let api = self.api();
        match op {
            "is_true" | "is_false" | "exists" | "not_exists" => {
                let path = scalar_string(arg);
                let path = if arg.is_null() { String::new() } else { path };
                let v = self.eval(&path);
                let ok = match op {
                    "is_true" => truthy(v.as_ref()),
                    "is_false" => !truthy(v.as_ref()),
                    // ES's `NotExistsAssertion` / `ExistsAssertion`: a JSON
                    // null counts as a missing value.
                    "not_exists" => v.as_ref().is_none_or(Value::is_null),
                    _ => v.as_ref().is_some_and(|x| !x.is_null()),
                };
                if !ok {
                    let shown = v.map(|x| trunc(&x.to_string(), 200)).unwrap_or_else(|| "<missing>".into());
                    return Err(fail(format!("{op} [{path}] failed after {api}: value {shown}"), format!("{op} [{}] after {api}", normalize_path(&path))));
                }
                Ok(())
            }
            "set" => {
                if let Some(m) = arg.as_object() {
                    for (path, var) in m {
                        let v = self.eval(path).ok_or_else(|| fail(format!("set: path [{path}] not found after {api}"), format!("set [{}] not found after {api}", normalize_path(path))))?;
                        self.stash.insert(scalar_string(var), v);
                    }
                }
                Ok(())
            }
            "transform_and_set" => {
                if let Some(m) = arg.as_object() {
                    for (var, expr) in m {
                        let e = scalar_string(expr);
                        let v = match e.strip_prefix("#base64EncodeCredentials(").and_then(|r| r.strip_suffix(')')).and_then(|r| r.split_once(',')) {
                            Some((a, b)) => {
                                let body = self.body();
                                let (x, y) = (scalar_string(&body[a.trim()]), scalar_string(&body[b.trim()]));
                                use base64::Engine;
                                Value::String(base64::engine::general_purpose::STANDARD.encode(format!("{x}:{y}")))
                            }
                            None => expr.clone(),
                        };
                        self.stash.insert(var.clone(), v);
                    }
                }
                Ok(())
            }
            _ => {
                let Some(m) = arg.as_object() else { return Err(fail(format!("malformed {op}"), format!("runner: malformed {op}"))) };
                for (path, exp) in m {
                    let exp = self.unstash(exp).map_err(|e| fail(e.clone(), "runner: stash"))?;
                    let actual = self.eval(path);
                    let np = normalize_path(path);
                    if std::env::var_os("ES_YAML_DEBUG_BODY").is_some() && path.contains("hits.total") {
                        eprintln!("DBGHITS {op} path=[{path}] actual={:?}", actual);
                        eprintln!("DBGHITS   body={}", self.body().to_string());
                    }
                    let res: Result<(), (String, &str)> = match op {
                        "match" => {
                            let regex = match &exp {
                                Value::String(s) => is_regex(s).map(str::to_string),
                                _ => None,
                            };
                            match regex {
                                Some(pat) => {
                                    let text = if path == "$body" || path.is_empty() {
                                        self.last.as_ref().map(|l| l.text.clone())
                                    } else {
                                        actual.as_ref().map(scalar_string)
                                    };
                                    match text {
                                        None => Err(("value missing".into(), "missing")),
                                        Some(t) => match regex_find(&pat, &t, true) {
                                            Ok(true) => Ok(()),
                                            Ok(false) => Err((format!("{} does not match /{}/", trunc(&t, 300), trunc(&pat, 200)), "regex mismatch")),
                                            Err(e) => Err((e, "unsupported regex")),
                                        },
                                    }
                                }
                                None => match &actual {
                                    None if exp.is_null() => Ok(()),
                                    None => Err((format!("missing, expected {}", trunc(&exp.to_string(), 200)), "missing")),
                                    Some(a) => json_eq(a, &exp, path).map_err(|e| (e, "differs")),
                                },
                            }
                        }
                        "length" => {
                            let want = num(&exp).unwrap_or(-1.0) as i64;
                            let got = match &actual {
                                Some(Value::Array(a)) => Some(a.len() as i64),
                                Some(Value::Object(o)) => Some(o.len() as i64),
                                Some(Value::String(s)) => Some(s.chars().count() as i64),
                                _ => None,
                            };
                            match got {
                                Some(g) if g == want => Ok(()),
                                Some(g) => Err((format!("expected length {want} but was {g}"), "differs")),
                                None => Err((format!("expected length {want} but value missing/not a collection"), "missing")),
                            }
                        }
                        "gt" | "gte" | "lt" | "lte" => match (actual.as_ref().and_then(num), num(&exp)) {
                            (Some(a), Some(e)) => {
                                let ok = match op {
                                    "gt" => a > e,
                                    "gte" => a >= e,
                                    "lt" => a < e,
                                    _ => a <= e,
                                };
                                if ok { Ok(()) } else { Err((format!("{a} {op} {e} is false"), "differs")) }
                            }
                            // Java's runner compares any two `Comparable`s: strings lexicographically.
                            _ if matches!((&actual, &exp), (Some(Value::String(_)), Value::String(_))) => {
                                let (a, e) = (actual.as_ref().and_then(Value::as_str).unwrap_or(""), exp.as_str().unwrap_or(""));
                                let ok = match op {
                                    "gt" => a > e,
                                    "gte" => a >= e,
                                    "lt" => a < e,
                                    _ => a <= e,
                                };
                                if ok { Ok(()) } else { Err((format!("{a:?} {op} {e:?} is false"), "differs")) }
                            }
                            _ => Err((format!("not numeric: actual {:?} expected {exp}", actual.as_ref().map(|x| trunc(&x.to_string(), 100))), "missing")),
                        },
                        "close_to" => {
                            let want = num(&exp["value"]).unwrap_or(f64::NAN);
                            let err = num(&exp["error"]).unwrap_or(0.0);
                            match actual.as_ref().and_then(num) {
                                Some(a) if (a - want).abs() <= err => Ok(()),
                                Some(a) => Err((format!("{a} not within {err} of {want}"), "differs")),
                                None => Err(("value missing".into(), "missing")),
                            }
                        }
                        "contains" => match &actual {
                            Some(Value::Array(a)) => {
                                let found = a.iter().any(|x| {
                                    json_eq(x, &exp, path).is_ok()
                                        || match (x, &exp) {
                                            (Value::Object(xo), Value::Object(eo)) => eo.iter().all(|(k, ev)| xo.get(k).map(|xv| json_eq(xv, ev, path).is_ok()).unwrap_or(false)),
                                            _ => false,
                                        }
                                });
                                if found { Ok(()) } else { Err((format!("array does not contain {}", trunc(&exp.to_string(), 200)), "differs")) }
                            }
                            Some(Value::String(s)) => {
                                if s.contains(&scalar_string(&exp)) { Ok(()) } else { Err((format!("{} does not contain {}", trunc(s, 200), exp), "differs")) }
                            }
                            _ => Err(("value missing / not array or string".into(), "missing")),
                        },
                        "is_after" => {
                            let p = |v: &Value| chrono::DateTime::parse_from_rfc3339(&scalar_string(v)).ok();
                            match (actual.as_ref().and_then(p), p(&exp)) {
                                (Some(a), Some(e)) if a > e => Ok(()),
                                _ => Err((format!("{:?} is not after {exp}", actual), "differs")),
                            }
                        }
                        other => return Err(fail(format!("unknown assertion [{other}]"), format!("runner: unknown assertion [{other}]"))),
                    };
                    if let Err((msg, kind)) = res {
                    return Err(fail(format!("{op} [{path}] after {api}: {msg}"), format!("{op} [{np}] after {api}: {kind}")));
                    }
                }
                Ok(())
            }
        }
    }

    async fn run_steps(&mut self, steps: &[Value]) -> Result<(), Failure> {
        for step in steps {
            if is_prereq(step) {
                continue;
            }
            let Some(obj) = step.as_object() else { return Err(fail("step is not a mapping", "runner: malformed step")) };
            for (op, arg) in obj {
                if op == "do" {
                    let m = arg.as_object().cloned().unwrap_or_default();
                    self.do_step(&m).await?;
                } else {
                    self.assertion(op, arg)?;
                }
            }
        }
        Ok(())
    }
}

/// The quoted message of a `Warning: 299 Elasticsearch-x.y.z-hash "message" "date"` header.
fn warning_message(h: &str) -> String {
    let Some(start) = h.find('"') else { return h.to_string() };
    let mut out = String::new();
    let mut chars = h[start + 1..].chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            '"' => break,
            c => out.push(c),
        }
    }
    out
}

/// ES's strict deprecation mode: every `Warning` header must be expected (`warnings*`) or allowed
/// (`allowed_warnings*`); every expected warning must be present.
fn check_warnings(api: &str, spec: &std::collections::HashMap<String, Vec<String>>, actual: &[String]) -> Result<(), Failure> {
    let get = |k: &str| spec.get(k).cloned().unwrap_or_default();
    let (exp, allowed, exp_re, allowed_re) = (get("warnings"), get("allowed_warnings"), get("warnings_regex"), get("allowed_warnings_regex"));
    let re_match = |pat: &str, text: &str| regex::Regex::new(&format!("(?s)^(?:{pat})$")).map(|r| r.is_match(text)).unwrap_or(false);
    let mut unexpected: Vec<&String> = Vec::new();
    for a in actual {
        let ok = exp.contains(a) || allowed.contains(a) || exp_re.iter().chain(allowed_re.iter()).any(|p| re_match(p, a));
        if !ok {
            unexpected.push(a);
        }
    }
    // ES 9.5.5 DoSection.checkWarningHeaders explicitly tolerates this
    // exact legacy-template warning in existing upstream YAML fixtures.
    // Keep the product warning and any explicit expectation intact.
    unexpected.retain(|warning| warning.as_str() != "Legacy index templates are deprecated in favor of composable templates.");
    let mut missing: Vec<String> = exp.iter().filter(|e| !actual.contains(e)).cloned().collect();
    missing.extend(exp_re.iter().filter(|p| !actual.iter().any(|a| re_match(p, a))).map(|p| format!("/{p}/")));
    if unexpected.is_empty() && missing.is_empty() {
        return Ok(());
    }
    let mut msg = format!("{api}: warnings mismatch");
    if !unexpected.is_empty() {
        msg.push_str(&format!("; unexpected {:?}", unexpected.iter().map(|s| trunc(s, 160)).collect::<Vec<_>>()));
    }
    if !missing.is_empty() {
        msg.push_str(&format!("; missing {:?}", missing.iter().map(|s| trunc(s, 160)).collect::<Vec<_>>()));
    }
    let cat = if !unexpected.is_empty() { format!("{api}: unexpected warning {}", normalize(&unexpected[0])) } else { format!("{api}: missing warning {}", normalize(&missing[0])) };
    Err(fail(msg, cat))
}

fn normalize_path(p: &str) -> String {
    split_path(p)
        .into_iter()
        .map(|s| if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) { "N".to_string() } else { s })
        .collect::<Vec<_>>()
        .join(".")
}

/// Run one test section against a fresh node in a fresh data dir.
pub async fn run_test(specs: &Specs, file: &TestFile, steps: &[Value]) -> Outcome {
    // `StatelessYamlTestSuiteIT` runs only against a `StatelessElasticsearchCluster`
    // (index + search nodes over an object store) with the `stateless` module,
    // which no release distribution ships.
    if file.dir == "stateless" {
        return Outcome::Skip("stateless suite: needs ES's stateless test cluster (no release build runs it)".into());
    }
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => return Outcome::Fail { msg: e.to_string(), category: "runner: tempdir".into() },
    };
    let node = match Node::open(dir.path()) {
        Ok(n) => n,
        Err(e) => return Outcome::Fail { msg: format!("node open: {e}"), category: "runner: node open".into() },
    };
    let watcher_sec = is_watcher_security(file);
    if watcher_sec {
        node.security.configure(watcher_security_config());
        let mut ns = node.node_settings.write();
        ns.insert("xpack.notification.email.account._email.smtp.host".into(), serde_json::json!("host.domain"));
        ns.insert("xpack.notification.email.account._email.smtp.port".into(), serde_json::json!("587"));
        ns.insert("xpack.notification.email.account._email.smtp.user".into(), serde_json::json!("_user"));
    } else if SECURITY.load(std::sync::atomic::Ordering::Relaxed) {
        node.security.configure(security_config());
        // ES's x-pack REST test cluster (`XPackRestIT`) enables transport
        // TLS with the `testnode` certificate in its config directory.
        let conf = dir.path().join("config");
        if std::fs::create_dir_all(&conf).is_ok() && std::fs::write(conf.join("testnode.crt"), include_str!("../resources/testnode.crt")).is_ok() {
            let mut ns = node.node_settings.write();
            ns.insert("path.conf".into(), serde_json::json!(conf.to_string_lossy()));
            ns.insert("xpack.security.transport.ssl.enabled".into(), serde_json::json!("true"));
            ns.insert("xpack.security.transport.ssl.key".into(), serde_json::json!("testnode.pem"));
            ns.insert("xpack.security.transport.ssl.certificate".into(), serde_json::json!("testnode.crt"));
            ns.insert("xpack.security.transport.ssl.verification_mode".into(), serde_json::json!("certificate"));
        }
    }
    let mut auth_override = None;
    if file.dir == "fleet" && !DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        let (cfg, auth) = fleet_security_config();
        node.security.configure(cfg);
        auth_override = Some(auth);
    }
    if file.dir == "searchable-snapshots" && !DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        let (cfg, auth) = admin_superuser_config();
        node.security.configure(cfg);
        auth_override = Some(auth);
    }
    // The x-pack main cluster (`XPackRestIT`) and the searchable snapshots
    // suite's cluster configure a 16MB shared cache of 256KB regions.
    if file.dir == "searchable-snapshots" || DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        let mut ns = node.node_settings.write();
        ns.insert("xpack.searchable.snapshot.shared_cache.size".into(), serde_json::json!("16MB"));
        ns.insert("xpack.searchable.snapshot.shared_cache.region_size".into(), serde_json::json!("256KB"));
    }
    if file.dir == "ccr" && !DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        let (cfg, auth) = ccr_security_config();
        node.security.configure(cfg);
        auth_override = Some(auth);
    }
    if file.dir == "inference" && !DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        let (cfg, auth) = inference_security_config();
        node.security.configure(cfg);
        auth_override = Some(auth);
    }
    if file.dir == "ent-search" && !DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) {
        let (cfg, auth) = entsearch_security_config();
        node.security.configure(cfg);
        auth_override = Some(auth);
    }
    // ES's REST test cluster runs with this node setting (its suites flip it
    // on per test); production ES 8 defaults it to true.
    node.node_settings.write().insert("action.destructive_requires_name".into(), serde_json::json!("false"));
    // ES's test clusters fix `node.processors: 2` (`DefaultSettingsProvider`)
    // so output such as `_cat/nodes` `ap` does not depend on the host.
    node.node_settings.write().insert("node.processors".into(), serde_json::json!("2"));
    elasticsearch_http::configure_processors(&node.node_settings.read());
    // ES's REST test clusters start with a self-generated trial
    // (`xpack.license.self_generated.type: trial`), so license-gated APIs
    // (graph, ML, watcher, inference, ...) are served; `ES_YAML_LICENSE=basic`
    // runs against a basic license instead.
    // Suites whose ES test cluster keeps the default (basic) license:
    // rank-rrf's `LicenseClientYamlTestSuiteIT` (`license/` tests).
    let basic_suite = file.rel.starts_with("rank-rrf/license/");
    let lic = std::env::var("ES_YAML_LICENSE").unwrap_or_else(|_| if basic_suite { "basic".into() } else { "trial".into() });
    // `FleetRestIT` runs on a basic license.
    let lic = if file.dir == "fleet" && !DEFAULT_DISTRIBUTION.load(std::sync::atomic::Ordering::Relaxed) { "basic".to_string() } else { lic };
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), serde_json::json!(lic));
    // ES runs the data-streams module suite on a two-node cluster
    // (`DataStreamsClientYamlTestSuiteIT`, `yaml.rest.tests.set_num_nodes`),
    // so one replica per shard allocates and its health turns green.
    // ES's async-search REST test cluster runs the deprecated-query test
    // module (`clusterModules project(":x-pack:test:deprecated-query")`).
    if file.dir == "async-search" {
        node.node_settings.write().insert(elasticsearch_engine::warnings::TEST_MODULES_SETTING.into(), serde_json::json!("x-pack-test-deprecated-query"));
    }
    if file.dir == "data_stream" {
        node.node_settings.write().insert(elasticsearch_engine::datastream::DATA_NODES_SETTING.into(), serde_json::json!(2));
    }
    {
        let (stack, apm, otel, prometheus) = builtin_registries(&file.dir);
        let mut ns = node.node_settings.write();
        ns.insert("stack.templates.enabled".into(), serde_json::json!(stack.to_string()));
        ns.insert(elasticsearch_engine::registry::QUERYLOG_REGISTRY_ENABLED.into(), serde_json::json!(stack.to_string()));
        ns.insert(elasticsearch_engine::registry::PROMETHEUS_REGISTRY_ENABLED.into(), serde_json::json!(prometheus.to_string()));
        // The logsdb plugin (`logs-*-*` default index mode) ships with the
        // default distribution and the suites that load it.
        let logsdb = prometheus || file.dir == "logsdb";
        ns.insert(elasticsearch_engine::datastream::LOGSDB_PLUGIN_SETTING.into(), serde_json::json!(logsdb.to_string()));
        ns.insert("xpack.apm_data.enabled".into(), serde_json::json!(apm.to_string()));
        ns.insert("xpack.otel_data.enabled".into(), serde_json::json!(otel.to_string()));
    }
    let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
        Ok(l) => l,
        Err(e) => return Outcome::Fail { msg: e.to_string(), category: "runner: bind".into() },
    };
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(elasticsearch_http::serve_listener(node, listener));
    let mut ctx = Ctx::new(specs, format!("http://{addr}"));
    if watcher_sec {
        // WatcherYamlRestWithSecurityIT.beforeTest (as the admin user)
        let _ = ctx
            .client
            .put(format!("http://{addr}/index_not_allowed_to_read/_doc/1"))
            .header("Authorization", basic("test_admin", XPACK_PASSWORD))
            .header("Content-Type", "application/json")
            .body(r#"{"foo":"bar"}"#)
            .send()
            .await;
        ctx.auth = Some(basic("watcher_manager", XPACK_PASSWORD));
    }
    if let Some(a) = auth_override {
        ctx.auth = Some(a);
    }
    let fut = async {
        for s in file.setup.iter().chain(file.teardown.iter()).chain(steps.iter()) {
            // A required runner feature outside ES's `Features.SUPPORTED`
            // (which `SUPPORTED_FEATURES` mirrors) skips the test, as in ES.
            if let Some(r) = ctx.prereq_skip(s).await {
                return Outcome::Skip(r);
            }
        }
        if let Err(f) = ctx.run_steps(&file.setup).await {
            return Outcome::Fail { msg: format!("setup: {}", f.msg), category: format!("setup: {}", f.category) };
        }
        ctx.last = None;
        // `ESClientYamlSuiteTestCase.test`: `${_project_id_prefix_}` expands to
        // the active project prefix (empty without multi-project).
        ctx.stash.insert("_project_id_prefix_".into(), Value::String(String::new()));
        let r = ctx.run_steps(steps).await;
        let _ = ctx.run_steps(&file.teardown).await;
        match r {
            Ok(()) if ctx.required_coverage.is_empty() => Outcome::Pass,
            Ok(()) => Outcome::Fail {
                msg: format!("Assertions passed, but required capability/feature discovery is missing: {}", ctx.required_coverage.join("; ")),
                category: "product: missing required capability discovery".into(),
            },
            Err(f) => Outcome::Fail { msg: f.msg, category: f.category },
        }
    };
    let out = match tokio::time::timeout(Duration::from_secs(180), fut).await {
        Ok(o) => o,
        Err(_) => Outcome::Fail { msg: "test timed out after 180s".into(), category: "runner: test timeout".into() },
    };
    server.abort();
    out
}

#[cfg(test)]
mod warning_contract_tests {
    use super::*;
    #[test]
    fn legacy_template_allowance_matches_upstream_without_hiding_other_warnings() {
        let legacy = "Legacy index templates are deprecated in favor of composable templates.".to_string();
        let empty = std::collections::HashMap::new();
        assert!(check_warnings("indices.put_template", &empty, &[legacy.clone()]).is_ok());
        assert!(check_warnings("indices.put_template", &empty, &[legacy.clone(), "unexpected extra".into()]).is_err());
        assert!(check_warnings("indices.put_template", &empty, &[format!("{legacy} extra")]).is_err());
        let expected = std::collections::HashMap::from([("warnings".to_string(), vec![legacy.clone()])]);
        assert!(check_warnings("indices.put_template", &expected, &[]).is_err());
        assert!(check_warnings("indices.put_template", &expected, &[legacy]).is_ok());
    }
}
