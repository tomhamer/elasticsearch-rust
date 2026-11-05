//! Node settings validation, as ES's `SettingsModule`: every key from
//! `elasticsearch.yml`, dotted environment variables and `-E` must be a
//! registered node setting; otherwise the start fails with ES's
//! `unknown setting [...]` message (with "did you mean" suggestions from the
//! registered keys, Lucene Levenshtein similarity > 0.7, as
//! `AbstractScopedSettings.findSimilarKeys`). Index-level settings are
//! rejected first ("node settings must not contain any index level
//! settings").
//!
//! The registry: every cluster-scope setting ES registers
//! (`elasticsearch_engine::settings::cluster_setting_kind`: the captured ES
//! registry, its affix families and the script context settings), the
//! settings ES filters from that capture (TLS, secure settings, repository
//! clients), affix families, and the rustic-only node settings the code
//! reads. Secure setting names (keystore entries) are known too.

use crate::config::{ConfigError, Settings};
use std::collections::BTreeSet;

/// Registered node settings that the captured registry does not list
/// (filtered or secure settings, 9.x additions) and rustic-only settings.
const EXTRA_KEYS: &[&str] = &[
    // Secure settings (keystore) and other filtered ES settings.
    "bootstrap.password",
    "keystore.seed",
    "autoconfiguration.password_hash",
    "cluster.state.encryption.active_password_id",
    "xpack.watcher.encryption_key",
    "xpack.security.authc.service_token_hashing.algorithm",
    // ES 9.x settings missing from the capture.
    "xpack.otel_data.enabled",
    "xpack.prometheus.enabled",
    "data_streams.failure_store.refresh_interval",
    "reindex.remote.allowlist",
    "xpack.watcher.history.cleaner_service.enabled",
    "xpack.watcher.history.cleaner_service.retention",
    // Coordination settings rustic reads (also accepted by older ES).
    "cluster.join.timeout",
    // Rustic-only node settings.
    "path.conf",
    "cluster.election.excluded_node_fallback",
    "indices.recovery.file_based",
    "indices.recovery.graph_checkpoint_lag_ops",
    "snapshot.restore.file_based",
    "xpack.license.enable_enterprise_features",
    "node.test_modules",
    "_internal.xpack.logsdb",
];

/// TLS contexts (`<context>.<key>`), ES `SslConfigurationKeys`.
const SSL_CONTEXTS: &[&str] = &[
    "xpack.security.transport.ssl",
    "xpack.security.http.ssl",
    "xpack.security.remote_cluster_server.ssl",
    "xpack.security.remote_cluster_client.ssl",
    "xpack.http.ssl",
    "reindex.ssl",
];
const SSL_KEYS: &[&str] = &[
    "enabled", "certificate", "certificate_authorities", "cipher_suites", "client_authentication", "handshake_timeout", "key",
    "key_passphrase", "secure_key_passphrase", "keystore.algorithm", "keystore.key_password", "keystore.password", "keystore.path",
    "keystore.secure_key_password", "keystore.secure_password", "keystore.type", "supported_protocols", "truststore.algorithm",
    "truststore.password", "truststore.path", "truststore.secure_password", "truststore.type", "verification_mode",
    "trust_restrictions.path", "trust_restrictions.x509_fields",
];

/// Affix / group settings: `*` is one key segment, a trailing `**` one or
/// more segments.
const PATTERNS: &[&str] = &[
    // Repository clients (ES `S3ClientSettings`, `GoogleCloudStorageClientSettings`,
    // `AzureStorageSettings`), including their secure settings.
    "s3.client.*.access_key", "s3.client.*.secret_key", "s3.client.*.session_token", "s3.client.*.endpoint", "s3.client.*.protocol",
    "s3.client.*.proxy.host", "s3.client.*.proxy.port", "s3.client.*.proxy.scheme", "s3.client.*.proxy.username", "s3.client.*.proxy.password",
    "s3.client.*.read_timeout", "s3.client.*.max_retries", "s3.client.*.use_throttle_retries", "s3.client.*.path_style_access",
    "s3.client.*.disable_chunked_encoding", "s3.client.*.region", "s3.client.*.signer_override", "s3.client.*.max_connections",
    "s3.client.*.connection_max_idle_time", "s3.client.*.add_purpose_custom_query_parameter", "s3.client.*.always_sign_requests",
    "s3.client.*.api_call_timeout", "s3.client.*.tenacious_retries.enabled", "s3.client.*.max_copy_size_before_multipart",
    "gcs.client.*.credentials_file", "gcs.client.*.endpoint", "gcs.client.*.project_id", "gcs.client.*.connect_timeout",
    "gcs.client.*.read_timeout", "gcs.client.*.application_name", "gcs.client.*.token_uri", "gcs.client.*.proxy.host",
    "gcs.client.*.proxy.port", "gcs.client.*.proxy.type", "gcs.client.*.max_retries", "gcs.client.*.megabytes_copied_per_chunk",
    "gcs.client.*.resumable_write_buffer_size", "gcs.client.*.tenacious_retries.enabled",
    "azure.client.*.account", "azure.client.*.key", "azure.client.*.sas_token", "azure.client.*.endpoint_suffix", "azure.client.*.timeout",
    "azure.client.*.max_retries", "azure.client.*.proxy.host", "azure.client.*.proxy.port", "azure.client.*.proxy.type",
    "azure.client.*.endpoint", "azure.client.*.secondary_endpoint",
    // Group and affix families.
    "node.attr.**",
    "logger.**",
    "thread_pool.**",
    "transport.profiles.**",
    "cluster.remote.**",
    "cluster.search.remote.**",
    "search.remote.**",
    "cluster.metadata.**",
    "cluster.routing.allocation.include.**",
    "cluster.routing.allocation.exclude.**",
    "cluster.routing.allocation.require.**",
    "cluster.routing.allocation.awareness.force.**",
    "cluster.state.encryption.password.*",
    "xpack.security.authc.realms.**",
    "xpack.security.authc.domains.**",
    "xpack.notification.**",
    "xpack.monitoring.exporters.**",
    "telemetry.agent.**",
    "tracing.apm.**",
    "script.context.*.**",
    // Rustic's stateless mode (ES serverless) settings.
    "stateless.**",
];

static EXACT: std::sync::LazyLock<BTreeSet<String>> = std::sync::LazyLock::new(|| {
    let mut keys: BTreeSet<String> = elasticsearch_engine::settings::cluster_defaults().into_keys().collect();
    keys.extend(EXTRA_KEYS.iter().map(|k| k.to_string()));
    for context in SSL_CONTEXTS {
        keys.extend(SSL_KEYS.iter().map(|k| format!("{context}.{k}")));
    }
    keys
});

fn pattern_matches(pattern: &str, key: &str) -> bool {
    let mut p = pattern.split('.').peekable();
    let mut k = key.split('.');
    while let Some(seg) = p.next() {
        if seg == "**" && p.peek().is_none() {
            // One or more remaining segments, none empty.
            let rest: Vec<&str> = k.collect();
            return !rest.is_empty() && rest.iter().all(|s| !s.is_empty());
        }
        match k.next() {
            Some(s) if seg == "*" && !s.is_empty() => {}
            Some(s) if s == seg => {}
            _ => return false,
        }
    }
    k.next().is_none()
}

/// Whether `key` is a registered node setting.
pub fn is_known(key: &str) -> bool {
    EXACT.contains(key)
        || elasticsearch_engine::settings::cluster_setting_kind(key).is_some()
        || PATTERNS.iter().any(|p| pattern_matches(p, key))
}

/// Lucene `LevenshteinDistance.getDistance` (a similarity in [0, 1]).
pub fn levenshtein_similarity(target: &str, other: &str) -> f32 {
    let sa: Vec<u16> = target.encode_utf16().collect();
    let ot: Vec<u16> = other.encode_utf16().collect();
    let (n, m) = (sa.len(), ot.len());
    if n == 0 || m == 0 {
        return if n == m { 1.0 } else { 0.0 };
    }
    let mut p: Vec<usize> = (0..=n).collect();
    let mut d: Vec<usize> = vec![0; n + 1];
    for j in 1..=m {
        let t_j = ot[j - 1];
        d[0] = j;
        for i in 1..=n {
            let cost = if sa[i - 1] == t_j { 0 } else { 1 };
            d[i] = (d[i - 1] + 1).min(p[i] + 1).min(p[i - 1] + cost);
        }
        std::mem::swap(&mut p, &mut d);
    }
    1.0f32 - (p[n] as f32 / n.max(m) as f32)
}

/// ES `findSimilarKeys`: registered keys with similarity > 0.7, best first.
pub fn similar_keys(key: &str) -> Vec<String> {
    let mut scored: Vec<(f32, &String)> = EXACT.iter().map(|k| (levenshtein_similarity(key, k), k)).filter(|(s, _)| *s > 0.7).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, k)| k.clone()).collect()
}

/// ES's message for one unknown key.
pub fn unknown_setting_message(key: &str) -> String {
    if key.starts_with("archived.") {
        return format!("unknown setting [{key}] was archived after upgrading, and must be removed. See [https://www.elastic.co/docs/deploy-manage/upgrade/deployment-or-cluster/archived-settings?version=9.5] for details.");
    }
    let similar = similar_keys(key);
    match similar.len() {
        0 => format!("unknown setting [{key}] please check that any required plugins are installed, or check the breaking changes documentation for removed settings"),
        1 => format!("unknown setting [{key}] did you mean [{}]?", similar[0]),
        _ => format!("unknown setting [{key}] did you mean any of [{}]?", similar.join(", ")),
    }
}

/// Validate the node settings (keys only), in key order as ES does.
pub fn validate_node_settings(settings: &Settings) -> Result<(), ConfigError> {
    if settings.keys().any(|k| k.starts_with("index.") && !is_known(k)) {
        return Err(ConfigError("node settings must not contain any index level settings".into()));
    }
    match settings.keys().find(|k| !is_known(k)) {
        Some(key) => Err(ConfigError(unknown_setting_message(key))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn s(keys: &[&str]) -> Settings {
        keys.iter().map(|k| (k.to_string(), Value::String("x".into()))).collect()
    }

    #[test]
    fn typos_are_rejected_with_suggestions() {
        let e = validate_node_settings(&s(&["cluster.name", "clustr.name"])).unwrap_err().0;
        assert!(e.starts_with("unknown setting [clustr.name] did you mean ") && e.contains("cluster.name") && e.ends_with('?'), "{e}");
        assert_eq!(similar_keys("clustr.name").first().map(String::as_str), Some("cluster.name"));
        let e = validate_node_settings(&s(&["xpack.security.enabeld"])).unwrap_err().0;
        assert!(e.starts_with("unknown setting [xpack.security.enabeld] did you mean"), "{e}");
        assert!(e.contains("xpack.security.enabled"), "{e}");
        let e = validate_node_settings(&s(&["totally.unrelated.thing"])).unwrap_err().0;
        assert_eq!(e, "unknown setting [totally.unrelated.thing] please check that any required plugins are installed, or check the breaking changes documentation for removed settings");
        // ES reports the first unknown key in key order.
        let e = validate_node_settings(&s(&["zz.unknown", "aa.unknown"])).unwrap_err().0;
        assert!(e.starts_with("unknown setting [aa.unknown]"), "{e}");
        assert!(unknown_setting_message("archived.foo").contains("was archived after upgrading"));
        // Removed settings are unknown.
        assert!(validate_node_settings(&s(&["discovery.zen.minimum_master_nodes"])).is_err());
    }

    #[test]
    fn index_level_settings_are_rejected() {
        let e = validate_node_settings(&s(&["index.number_of_shards"])).unwrap_err().0;
        assert_eq!(e, "node settings must not contain any index level settings");
    }

    #[test]
    fn common_affix_and_secure_settings_are_known() {
        let ok = [
            "cluster.name", "node.name", "node.roles", "path.data", "path.logs", "path.repo", "path.conf", "network.host", "http.port",
            "http.host", "transport.port", "transport.host", "discovery.seed_hosts", "discovery.type", "cluster.initial_master_nodes",
            "bootstrap.memory_lock", "action.destructive_requires_name", "xpack.license.self_generated.type",
            "xpack.security.enabled", "xpack.security.enrollment.enabled", "xpack.security.autoconfiguration.enabled",
            "xpack.security.http.ssl.enabled", "xpack.security.http.ssl.key", "xpack.security.http.ssl.certificate",
            "xpack.security.http.ssl.certificate_authorities", "xpack.security.http.ssl.keystore.secure_password",
            "xpack.security.transport.ssl.verification_mode", "xpack.security.transport.ssl.secure_key_passphrase",
            "xpack.security.authc.password_hashing.algorithm", "xpack.security.authc.token.enabled",
            // Affix families.
            "node.attr.zone", "node.attr.rustic.internal.capability", "logger.org.elasticsearch.transport", "thread_pool.write.size",
            "cluster.routing.allocation.awareness.force.zone.values", "cluster.routing.allocation.include._name",
            "cluster.remote.other.seeds", "cluster.remote.other.credentials", "transport.profiles.client.port",
            "xpack.security.authc.realms.file.file1.order", "xpack.security.authc.realms.ldap.ldap1.secure_bind_password",
            "xpack.notification.email.account._email.smtp.host", "xpack.notification.slack.account.a.secure_url",
            "script.context.score.max_compilations_rate",
            // Secure / repository client settings.
            "bootstrap.password", "keystore.seed", "s3.client.default.access_key", "s3.client.default.secret_key",
            "s3.client.default.session_token", "s3.client.minio.endpoint", "s3.client.default.path_style_access",
            "gcs.client.default.credentials_file", "azure.client.default.account", "azure.client.default.sas_token",
            "cluster.state.encryption.password.autoconfigured", "cluster.state.encryption.active_password_id",
            // Rustic-only settings the code reads.
            "stateless.enabled", "stateless.object_store.type", "indices.recovery.graph_checkpoint_lag_ops", "indices.recovery.file_based",
            "snapshot.restore.file_based", "cluster.election.excluded_node_fallback", "node.test_modules", "_internal.xpack.logsdb",
            "vectors.indexing.use_gpu", "xpack.otel_data.enabled", "data_streams.failure_store.refresh_interval",
            "cluster.routing.allocation.disk.watermark.low", "indices.breaker.total.use_real_memory", "node.portsfile",
        ];
        for k in ok {
            assert!(is_known(k), "{k} must be accepted");
        }
        assert!(validate_node_settings(&s(&ok)).is_ok());
        // Patterns match whole segments only.
        assert!(!is_known("s3.client.access_key"));
        assert!(!is_known("s3.client.default.acess_key"));
        assert!(!is_known("node.attr"));
        assert!(!is_known("xpack.security.http.ssl.keyy"));
    }

    #[test]
    fn lucene_levenshtein() {
        assert_eq!(levenshtein_similarity("", ""), 1.0);
        assert_eq!(levenshtein_similarity("a", ""), 0.0);
        assert_eq!(levenshtein_similarity("abc", "abc"), 1.0);
        assert!((levenshtein_similarity("clustr.name", "cluster.name") - (1.0 - 1.0 / 12.0)).abs() < 1e-6);
    }

    /// Every setting a real ES 9.5.5 node registers (`GET
    /// _cluster/settings?include_defaults&flat_settings`, captured from the
    /// 9.5.5 distribution on 2026-11-03) is accepted: a config that starts
    /// ES must not be refused here.
    #[test]
    fn every_es_955_registered_setting_is_known() {
        let keys: Vec<String> = serde_json::from_str(include_str!("es955_node_setting_keys.json")).unwrap();
        assert!(keys.len() > 1000, "{}", keys.len());
        let unknown: Vec<&String> = keys.iter().filter(|k| !is_known(k)).collect();
        assert!(unknown.is_empty(), "ES 9.5.5 settings rejected: {unknown:?}");
    }
}
