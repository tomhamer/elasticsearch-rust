//! Which node settings may never be rendered by an API, after ES's
//! `SettingsFilter` and `SecureSetting`.
//!
//! ES keeps secure settings (the `elasticsearch.keystore` entries) in a
//! separate `SecureSettings` holder that `Settings.toXContent` never writes,
//! and removes every setting registered with `Property.Filtered` (plus the
//! plugins' `getSettingsFilter()` patterns and `xpack.security.hide_settings`)
//! from `_nodes` info, `_cluster/settings?include_defaults`, repository
//! settings and the rest. Here keystore entries are injected into the node's
//! flat settings map at startup, so every renderer asks [`SettingsFilter`]
//! (or [`is_filtered`]) before writing a node setting.
//!
//! - [`register_secure`]: the keystore's entry names (process-wide; a name is
//!   filtered for every node of the process, which only matters for several
//!   in-process nodes and never un-hides anything).
//! - [`is_secure_only`]: settings ES defines only as `SecureSetting`s, which
//!   ES refuses to read from `elasticsearch.yml` / `-E` / the environment.
//! - [`is_filtered`]: secure settings plus the `Property.Filtered` settings
//!   (TLS key material and passwords, realm bind credentials, the
//!   deprecated insecure S3 repository `access_key` / `secret_key`, ...).

use parking_lot::RwLock;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::LazyLock;

static SECURE: LazyLock<RwLock<BTreeSet<String>>> = LazyLock::new(|| RwLock::new(BTreeSet::new()));

/// Record keystore entry names as secure settings (never rendered).
pub fn register_secure<I, S>(names: I)
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut set = SECURE.write();
    for n in names {
        set.insert(n.into());
    }
}

/// Whether `key` was loaded from a keystore of this process.
pub fn is_registered_secure(key: &str) -> bool {
    SECURE.read().contains(key)
}

/// The keystore entry names registered so far.
pub fn registered_secure() -> BTreeSet<String> {
    SECURE.read().clone()
}

/// Settings ES registers only as `SecureSetting` (namespaces are one
/// segment, as `Setting.affixKeySetting` namespaces are).
const SECURE_ONLY: &[&str] = &[
    "bootstrap.password",
    "keystore.seed",
    "s3.client.*.access_key",
    "s3.client.*.secret_key",
    "s3.client.*.session_token",
    "s3.client.*.proxy.username",
    "s3.client.*.proxy.password",
    "gcs.client.*.credentials_file",
    "azure.client.*.account",
    "azure.client.*.key",
    "azure.client.*.sas_token",
    "cluster.remote.*.credentials",
    "xpack.security.authc.realms.oidc.*.rp.client_secret",
    "xpack.security.authc.realms.jwt.*.hmac_key",
    "xpack.security.authc.realms.jwt.*.hmac_jwkset",
    "xpack.security.authc.realms.jwt.*.client_authentication.shared_secret",
    "xpack.watcher.encryption_key",
];

/// Whether ES defines `key` only as a secure (keystore) setting: the
/// [`SECURE_ONLY`] affix keys and, by ES's naming convention, every setting
/// whose last segment starts with `secure_` (`*.ssl.secure_key_passphrase`,
/// `*.ssl.keystore.secure_password`, realms' `secure_bind_password`, the
/// notification accounts' `secure_*`, ...).
pub fn is_secure_only(key: &str) -> bool {
    let last = key.rsplit('.').next().unwrap_or(key);
    last.starts_with("secure_") || SECURE_ONLY.iter().any(|p| segment_match(p, key))
}

/// `Property.Filtered` settings ES registers (as keys or affix patterns;
/// `*` matches anything, as in `Regex.simpleMatch`).
const FILTERED: &[&str] = &[
    // `S3Repository.ACCESS_KEY_SETTING` / `SECRET_KEY_SETTING`
    // (`SecureSetting.insecureString`), also matched against repository
    // settings by `GET _snapshot`.
    "access_key",
    "secret_key",
    "xpack.security.hide_settings",
    // Security's `getSettingsFilter()`.
    "transport.profiles.*.xpack.security.*",
    // `PoolingSessionFactorySettings.BIND_DN`, `SessionFactorySettings`.
    "xpack.security.authc.realms.*.bind_dn",
    "xpack.security.authc.realms.*.hostname_verification",
    // `CrossClusterApiKeySigningSettings`.
    "cluster.remote.*.signing.*",
    "cluster.remote.signing.*",
];

/// Whether `key` is a `Property.Filtered` or secure setting.
pub fn is_filtered(key: &str) -> bool {
    if key.starts_with("_internal.") || is_registered_secure(key) || is_secure_only(key) {
        return true;
    }
    let last = key.rsplit('.').next().unwrap_or(key);
    // Passwords: the deprecated insecure variants of the TLS / realm /
    // monitoring exporter / notification account passwords are all
    // `Property.Filtered`.
    if last.contains("password") || last.contains("passphrase") {
        return true;
    }
    if ssl_filtered(key) {
        return true;
    }
    FILTERED.iter().any(|p| simple_match(p, key))
}

/// `SSLConfigurationSettings` / `X509KeyPairSettings` / `ReindexSslConfig` /
/// the monitoring exporters' `ssl` group: every `*.ssl.<x>` setting
/// (keystore / truststore / key / certificate paths and types, protocols,
/// ciphers, client authentication, verification mode, trust restrictions)
/// except the plain `*.ssl.enabled` switch.
fn ssl_filtered(key: &str) -> bool {
    let mut parts = key.split('.');
    let mut seen_ssl = false;
    let mut rest: Vec<&str> = Vec::new();
    for p in parts.by_ref() {
        if seen_ssl {
            rest.push(p);
        } else if p == "ssl" {
            seen_ssl = true;
        }
    }
    seen_ssl && !rest.is_empty() && rest != ["enabled"]
}

/// ES's `SettingsFilter` for one node: the static patterns plus the
/// node's `xpack.security.hide_settings` patterns.
#[derive(Debug, Clone, Default)]
pub struct SettingsFilter {
    hide: Vec<String>,
}

impl SettingsFilter {
    /// `hide_settings` is the node's `xpack.security.hide_settings` value
    /// (a list or a comma-separated string), if any.
    pub fn new(hide_settings: Option<&Value>) -> SettingsFilter {
        let hide = match hide_settings {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            Some(Value::String(s)) => s.trim().trim_start_matches('[').trim_end_matches(']').split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
            _ => Vec::new(),
        };
        SettingsFilter { hide }
    }

    /// The filter for a flat node settings map.
    pub fn for_settings<'a, I>(settings: I) -> SettingsFilter
    where
        I: IntoIterator<Item = (&'a String, &'a Value)>,
    {
        let hide = settings.into_iter().find(|(k, _)| k.as_str() == "xpack.security.hide_settings").map(|(_, v)| v);
        SettingsFilter::new(hide)
    }

    /// Whether `key` must be left out of any rendering.
    pub fn hides(&self, key: &str) -> bool {
        is_filtered(key) || self.hide.iter().any(|p| simple_match(p, key))
    }
}

/// `Regex.simpleMatch`: `*` matches any (possibly empty) run of characters.
pub fn simple_match(pattern: &str, s: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == s;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !s.starts_with(first) || s.len() < first.len() + last.len() || !s.ends_with(last) {
        return false;
    }
    let mut rest = &s[first.len()..s.len() - last.len()];
    for mid in &parts[1..parts.len() - 1] {
        if mid.is_empty() {
            continue;
        }
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

/// Dotted-key match where `*` stands for exactly one segment.
fn segment_match(pattern: &str, key: &str) -> bool {
    let p: Vec<&str> = pattern.split('.').collect();
    let k: Vec<&str> = key.split('.').collect();
    p.len() == k.len() && p.iter().zip(&k).all(|(a, b)| *a == "*" && !b.is_empty() || a == b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_only_settings() {
        for k in [
            "bootstrap.password",
            "s3.client.default.access_key",
            "s3.client.backup-1.secret_key",
            "s3.client.default.session_token",
            "xpack.security.http.ssl.keystore.secure_password",
            "xpack.security.transport.ssl.secure_key_passphrase",
            "xpack.security.authc.realms.ldap.ldap1.secure_bind_password",
            "xpack.notification.email.account.a.smtp.secure_password",
            "gcs.client.default.credentials_file",
            "cluster.remote.r1.credentials",
        ] {
            assert!(is_secure_only(k), "{k}");
            assert!(is_filtered(k), "{k}");
        }
        for k in ["s3.client.default.endpoint", "s3.client.a.b.access_key", "access_key", "xpack.security.enabled", "cluster.remote.r1.seeds"] {
            assert!(!is_secure_only(k), "{k}");
        }
    }

    #[test]
    fn filtered_settings() {
        for k in [
            "access_key",
            "secret_key",
            "xpack.security.http.ssl.key",
            "xpack.security.http.ssl.keystore.path",
            "xpack.security.transport.ssl.certificate_authorities",
            "xpack.security.transport.ssl.verification_mode",
            "xpack.security.authc.realms.ldap.ldap1.ssl.truststore.path",
            "xpack.security.authc.realms.ldap.ldap1.bind_dn",
            "xpack.security.authc.realms.ldap.ldap1.bind_password",
            "reindex.ssl.certificate",
            "xpack.monitoring.exporters.e1.ssl.truststore.path",
            "transport.profiles.client.xpack.security.type",
            "xpack.notification.email.account.a.smtp.password",
            "_internal.anything",
        ] {
            assert!(is_filtered(k), "{k}");
        }
        for k in ["xpack.security.http.ssl.enabled", "xpack.security.enabled", "cluster.name", "path.data", "xpack.security.authc.password_hashing.algorithm", "s3.client.default.endpoint"] {
            assert!(!is_filtered(k), "{k}");
        }
    }

    #[test]
    fn registered_keystore_names_and_hide_settings() {
        assert!(!is_filtered("custom.plugin.token"));
        register_secure(["custom.plugin.token"]);
        assert!(is_filtered("custom.plugin.token"));
        let f = SettingsFilter::new(Some(&Value::String("cluster.routing.*, node.attr.secret".into())));
        assert!(f.hides("cluster.routing.allocation.awareness.attributes"));
        assert!(f.hides("node.attr.secret"));
        assert!(!f.hides("node.attr.zone"));
        let f = SettingsFilter::new(Some(&serde_json::json!(["a.*"])));
        assert!(f.hides("a.b") && !f.hides("b.a"));
    }

    #[test]
    fn simple_match_wildcards() {
        assert!(simple_match("a*b*c", "a-x-b-y-c"));
        assert!(simple_match("*", "anything"));
        assert!(simple_match("a.*", "a.b.c"));
        assert!(!simple_match("a*b", "ab-c"));
        assert!(simple_match("a*b", "ab"));
    }
}
