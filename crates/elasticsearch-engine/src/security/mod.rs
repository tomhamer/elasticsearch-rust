//! Node security (`xpack.security.enabled`, default true for a node built from settings, as in ES 8+):
//! reserved / file / native realms, API keys, OAuth2-style tokens, service
//! accounts, roles (reserved, file, native), role mappings and application
//! privileges. State persists in `_state/security.json`.
//!
//! The HTTP layer (`elasticsearch-http::security`) authenticates every
//! request through [`Security::authenticate`] and authorizes it against
//! [`Security::permission`]; with security disabled nothing here runs.

pub mod hash;
pub mod owner;
pub mod privilege;
pub mod roles;

use crate::error::EsError;
use hash::Hasher;
use parking_lot::{Mutex, RwLock};
use roles::Role;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const RESERVED_USERS: &[(&str, &[&str])] = &[
    ("elastic", &["superuser"]),
    ("kibana", &["kibana_system"]),
    ("kibana_system", &["kibana_system"]),
    ("logstash_system", &["logstash_system"]),
    ("beats_system", &["beats_system"]),
    ("apm_system", &["apm_system"]),
    ("remote_monitoring_user", &["remote_monitoring_collector", "remote_monitoring_agent"]),
];

pub fn reserved_user_roles(name: &str) -> Option<&'static [&'static str]> {
    RESERVED_USERS.iter().find(|(n, _)| *n == name).map(|(_, r)| *r)
}

fn reserved_user_metadata(name: &str) -> Value {
    if name == "kibana" {
        json!({"_reserved": true, "_deprecated": true, "_deprecated_reason": "Please use the [kibana_system] user instead."})
    } else {
        json!({"_reserved": true})
    }
}

/// `xpack.security.enabled` when a node built from settings does not set it
/// (ES 8+: true). See [`SecConfig::from_node_settings`].
pub const NODE_SECURITY_ENABLED_DEFAULT: bool = true;

/// Node-level security settings.
#[derive(Clone, Debug)]
pub struct SecConfig {
    pub enabled: bool,
    pub anonymous_username: String,
    pub anonymous_roles: Vec<String>,
    pub anonymous_authz_exception: bool,
    pub password_hashing: Hasher,
    pub token_enabled: bool,
    pub api_key_enabled: bool,
    pub token_timeout_ms: i64,
    pub bootstrap_password: Option<String>,
    /// `xpack.security.authc.reserved_realm.enabled` (default true). Off,
    /// the reserved users (`elastic`, `kibana_system`, ...) are not
    /// authenticated or looked up by the reserved realm, so other realms
    /// can define them: ECK disables it and puts `elastic` in the file realm.
    pub reserved_realm_enabled: bool,
    /// The file and native realms' names: `default_file` / `default_native`
    /// unless configured (`xpack.security.authc.realms.file.<name>.*`; ECK
    /// configures `file1` and `native1`). Reported as `authentication_realm`.
    pub file_realm_name: String,
    pub native_realm_name: String,
    /// File realm: user -> password hash.
    pub file_users: BTreeMap<String, String>,
    /// File realm: user -> roles.
    pub file_user_roles: BTreeMap<String, Vec<String>>,
    /// File-based roles (`roles.yml` equivalents), name -> descriptor.
    pub file_roles: BTreeMap<String, Value>,
    /// File service tokens: `ns/svc/name` -> hash.
    pub file_service_tokens: BTreeMap<String, String>,
    /// Whether the REST layer serves HTTPS (`xpack.security.http.ssl.*`).
    /// Read from the setting here; the node then sets it from the TLS
    /// config it actually serves (`elasticsearch_ssl::http_server`, which
    /// fails the start on invalid settings), as does the engine-only
    /// `serve_listener_with_tls`. Also the default of
    /// `xpack.security.authc.token.enabled`.
    pub http_ssl: bool,
    /// The trial license: unlocks the Standard-level security features
    /// (token service, profile collaboration). Always on: licensing is
    /// hardwired to enterprise/trial. DLS/FLS are enforced for every role
    /// that defines them, regardless of this flag.
    pub trial: bool,
}

impl Default for SecConfig {
    fn default() -> Self {
        SecConfig {
            enabled: false,
            anonymous_username: "_anonymous".into(),
            anonymous_roles: Vec::new(),
            anonymous_authz_exception: true,
            password_hashing: Hasher::Bcrypt(10),
            token_enabled: false,
            api_key_enabled: true,
            token_timeout_ms: 20 * 60 * 1000,
            bootstrap_password: None,
            reserved_realm_enabled: true,
            file_realm_name: "default_file".into(),
            native_realm_name: "default_native".into(),
            file_users: BTreeMap::new(),
            file_user_roles: BTreeMap::new(),
            file_roles: BTreeMap::new(),
            file_service_tokens: BTreeMap::new(),
            http_ssl: false,
            // Licensing is hardwired to enterprise/trial; see AGENTS.md
            // "Licensing: always enterprise".
            trial: true,
        }
    }
}

/// Settings lookup: env var with the dotted name, `ES_`-style alias, then `elasticsearch.yml`.
fn setting(yml: &BTreeMap<String, String>, key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key) {
        return Some(v);
    }
    yml.get(key).cloned()
}

/// ES `Booleans.parseBoolean`: `true` / `false` only (`tru`, `TRUE`,
/// `yes`, `1` are errors, never false).
fn parse_bool(s: &str) -> Result<bool, String> {
    crate::settings::parse_bool(s).map_err(|e| e.reason)
}

/// For the pre-start environment read ([`SecConfig::from_env`], which
/// cannot fail): an invalid value keeps `default`. The node settings read
/// ([`SecConfig::from_node_settings`]) and the start's validation reject it.
fn parse_bool_or(s: &str, default: bool) -> bool {
    parse_bool(s).unwrap_or(default)
}

fn read_flat_yml(path: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(path) else { return out };
    // Flat `a.b.c: value` lines plus one level of nesting by indentation.
    let mut stack: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let raw = line.split(" #").next().unwrap_or("");
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let indent = raw.len() - raw.trim_start().len();
        let Some((k, v)) = raw.trim().split_once(':') else { continue };
        while stack.last().map(|(i, _)| *i >= indent).unwrap_or(false) {
            stack.pop();
        }
        let full = stack.iter().map(|(_, s)| s.as_str()).chain(std::iter::once(k.trim())).collect::<Vec<_>>().join(".");
        let v = v.trim().trim_matches('"').trim_matches('\'').to_string();
        if v.is_empty() {
            stack.push((indent, k.trim().to_string()));
        } else {
            out.insert(full, v);
        }
    }
    out
}

impl SecConfig {
    /// Apply parsed node/CLI settings after environment and file-realm loading.
    pub fn from_node_settings(settings: &serde_json::Map<String, Value>) -> Result<SecConfig, String> {
        let mut c = Self::from_env();
        let value = |key: &str| settings.get(key).filter(|v| !v.is_null()).map(|v| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()));
        // ES 8/9: `xpack.security.enabled` defaults to true for a node. The
        // rustic-only `ES_SECURITY_ENABLED` environment variable still
        // overrides the default (never an explicit setting). In-process
        // embedders that use `SecConfig::default()` keep security off.
        c.enabled = match value("xpack.security.enabled") {
            Some(v) => parse_bool(&v)?,
            None => match std::env::var("ES_SECURITY_ENABLED") {
                Ok(v) => parse_bool(&v)?,
                Err(_) => NODE_SECURITY_ENABLED_DEFAULT,
            },
        };
        if let Some(v) = value("xpack.security.http.ssl.enabled") {
            c.http_ssl = parse_bool(&v)?;
            if !settings.contains_key("xpack.security.authc.token.enabled") { c.token_enabled = c.http_ssl; }
        }
        if let Some(v) = value("xpack.security.authc.token.enabled") { c.token_enabled = parse_bool(&v)?; }
        if let Some(v) = value("xpack.security.authc.api_key.enabled") { c.api_key_enabled = parse_bool(&v)?; }
        if let Some(v) = value("xpack.security.authc.token.timeout") { c.token_timeout_ms = parse_time_ms(&v, "xpack.security.authc.token.timeout").map_err(|e| e.reason)?; }
        if let Some(v) = value("xpack.security.authc.anonymous.username") { c.anonymous_username = v; }
        if let Some(v) = settings.get("xpack.security.authc.anonymous.roles") {
            c.anonymous_roles = match v {
                Value::Array(a) => a.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
                Value::String(s) => s.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect(),
                _ => return Err("xpack.security.authc.anonymous.roles must be a list of roles".into()),
            };
        }
        if let Some(v) = value("xpack.security.authc.anonymous.authz_exception") { c.anonymous_authz_exception = parse_bool(&v)?; }
        if let Some(v) = value("xpack.security.authc.password_hashing.algorithm") { c.password_hashing = Hasher::resolve(&v)?; }
        if let Some(v) = value("bootstrap.password") { c.bootstrap_password = Some(v); }
        if let Some(v) = value("xpack.security.authc.reserved_realm.enabled") { c.reserved_realm_enabled = parse_bool(&v)?; }
        for key in settings.keys() {
            if let Some(rest) = key.strip_prefix("xpack.security.authc.realms.") {
                if rest.split('.').count() == 3 && rest.ends_with(".enabled") {
                    if let Some(v) = value(key) { parse_bool(&v)?; }
                }
                match rest.split('.').collect::<Vec<_>>().as_slice() {
                    ["file", name, _, ..] => c.file_realm_name = name.to_string(),
                    ["native", name, _, ..] => c.native_realm_name = name.to_string(),
                    _ => {}
                }
            }
        }
        Ok(c)
    }

    /// Build from the environment (`ES_SECURITY_ENABLED`, `xpack.security.*`
    /// env vars, `$ES_PATH_CONF/elasticsearch.yml`, file realm files).
    pub fn from_env() -> SecConfig {
        let conf = PathBuf::from(std::env::var("ES_PATH_CONF").unwrap_or_else(|_| "config".into()));
        let yml = read_flat_yml(&conf.join("elasticsearch.yml"));
        let mut c = SecConfig::default();
        // An invalid value reads as enabled (fail closed); the start rejects it.
        c.enabled = std::env::var("ES_SECURITY_ENABLED").ok().or_else(|| setting(&yml, "xpack.security.enabled")).map(|v| parse_bool_or(&v, true)).unwrap_or(false);
        if let Some(u) = setting(&yml, "xpack.security.authc.anonymous.username") {
            c.anonymous_username = u;
        }
        if let Some(r) = setting(&yml, "xpack.security.authc.anonymous.roles") {
            c.anonymous_roles = r.trim_matches(|ch| ch == '[' || ch == ']').split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect();
        }
        if let Some(v) = setting(&yml, "xpack.security.authc.anonymous.authz_exception") {
            c.anonymous_authz_exception = parse_bool_or(&v, c.anonymous_authz_exception);
        }
        if let Some(v) = setting(&yml, "xpack.security.authc.password_hashing.algorithm") {
            if let Ok(h) = Hasher::resolve(&v) {
                c.password_hashing = h;
            }
        }
        c.http_ssl = setting(&yml, "xpack.security.http.ssl.enabled").map(|v| parse_bool_or(&v, false)).unwrap_or(false);
        c.token_enabled = setting(&yml, "xpack.security.authc.token.enabled").map(|v| parse_bool_or(&v, c.http_ssl)).unwrap_or(c.http_ssl);
        if let Some(v) = setting(&yml, "xpack.security.authc.api_key.enabled") {
            c.api_key_enabled = parse_bool_or(&v, c.api_key_enabled);
        }
        if let Some(v) = setting(&yml, "xpack.security.authc.token.timeout") {
            if let Ok(ms) = parse_time_ms(&v, "xpack.security.authc.token.timeout") {
                c.token_timeout_ms = ms;
            }
        }
        c.bootstrap_password = std::env::var("ELASTIC_PASSWORD").ok().or_else(|| std::env::var("ES_BOOTSTRAP_PASSWORD").ok()).or_else(|| setting(&yml, "bootstrap.password"));
        if let Some(v) = setting(&yml, "xpack.security.authc.reserved_realm.enabled") {
            c.reserved_realm_enabled = parse_bool_or(&v, c.reserved_realm_enabled);
        }
        c.load_files(&conf);
        c
    }

    /// Load the file realm (`users`, `users_roles`) and `service_tokens`.
    pub fn load_files(&mut self, conf: &Path) {
        if let Ok(t) = std::fs::read_to_string(conf.join("users")) {
            for l in t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
                if let Some((u, h)) = l.split_once(':') {
                    self.file_users.insert(u.to_string(), h.to_string());
                }
            }
        }
        if let Ok(t) = std::fs::read_to_string(conf.join("users_roles")) {
            for l in t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
                if let Some((r, users)) = l.split_once(':') {
                    for u in users.split(',').map(str::trim).filter(|u| !u.is_empty()) {
                        self.file_user_roles.entry(u.to_string()).or_default().push(r.trim().to_string());
                    }
                }
            }
        }
        if let Ok(t) = std::fs::read_to_string(conf.join("service_tokens")) {
            for l in t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
                if let Some((n, h)) = l.split_once(':') {
                    self.file_service_tokens.insert(n.to_string(), h.to_string());
                }
            }
        }
        // ES's file-role format is YAML. Keep the legacy JSON fixture
        // format as a fallback when the native file is absent.
        let yaml = conf.join("roles.yml");
        let descriptors = if yaml.exists() {
            std::fs::read_to_string(&yaml).ok().and_then(|text| {
                serde_yaml::from_str::<Value>(&text).map_err(|error| tracing::warn!(%error, "failed to parse file roles")).ok()
            })
        } else {
            std::fs::read_to_string(conf.join("roles.json")).ok().and_then(|text| serde_json::from_str::<Value>(&text).ok())
        };
        if let Some(Value::Object(map)) = descriptors {
            for (name, descriptor) in map {
                match roles::parse_role(&name, &descriptor, false) {
                    Ok(role) => { self.file_roles.insert(name, role); }
                    Err(error) => tracing::warn!(role=%name, reason=%error.reason, "invalid file role"),
                }
            }
        }
    }
}

/// Parse an ES time value (`1d`, `30m`, `500ms`) into milliseconds.
pub fn parse_time_ms(v: &str, setting: &str) -> Result<i64, EsError> {
    let s = v.trim();
    let err = || EsError::new(400, "parse_exception", format!("failed to parse setting [{setting}] with value [{v}] as a time value: unit is missing or unrecognized"));
    let idx = s.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').ok_or_else(err)?;
    let (num, unit) = s.split_at(idx);
    let n: f64 = num.parse().map_err(|_| err())?;
    let mult: f64 = match unit {
        "nanos" => 1e-6,
        "micros" => 1e-3,
        "ms" => 1.0,
        "s" => 1000.0,
        "m" => 60_000.0,
        "h" => 3_600_000.0,
        "d" => 86_400_000.0,
        _ => return Err(err()),
    };
    Ok((n * mult) as i64)
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeUser {
    pub password_hash: String,
    pub roles: Vec<String>,
    #[serde(default)]
    pub full_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReservedState {
    #[serde(default)]
    pub password_hash: Option<String>,
    #[serde(default = "yes")]
    pub enabled: bool,
}

impl Default for ReservedState {
    fn default() -> Self {
        Self { password_hash: None, enabled: true }
    }
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    pub hash: String,
    pub creation: i64,
    #[serde(default)]
    pub expiration: Option<i64>,
    #[serde(default)]
    pub invalidated: bool,
    #[serde(default)]
    pub invalidation: Option<i64>,
    pub username: String,
    pub realm: String,
    pub realm_type: String,
    #[serde(default)]
    pub full_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub user_metadata: Value,
    #[serde(default)]
    pub metadata: Value,
    /// name -> descriptor (assigned privileges; empty = owner's)
    #[serde(default)]
    pub role_descriptors: Map<String, Value>,
    /// Owner's role descriptors at creation time.
    #[serde(default)]
    pub limited_by: Map<String, Value>,
    #[serde(default = "rest_type")]
    pub key_type: String,
    /// `access` of a cross-cluster key (`search` / `replication`); null for rest keys.
    #[serde(default)]
    pub access: Value,
}

fn rest_type() -> String {
    "rest".into()
}

/// An OAuth2 token pair as the token store keeps it: only the SHA-256 of
/// the access and refresh tokens is stored (ES keeps hashed token ids), so
/// the security state on disk or in a snapshot yields no usable bearer
/// token. [`Security::create_token`] hands the raw values back once.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenDoc {
    /// Hex SHA-256 of the access token (the raw value only in the copy
    /// `create_token` returns).
    pub access_token: String,
    /// Hex SHA-256 of the refresh token (raw in `create_token`'s copy).
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Token ids are hashed (state written by this version). Documents
    /// from before (plaintext ids) are never accepted and are purged.
    #[serde(default)]
    pub hashed: bool,
    /// The client that obtained the token (ES `checkClientCanRefresh`:
    /// only it may refresh).
    #[serde(default)]
    pub client: Option<owner::ResourceOwner>,
    pub creation: i64,
    pub expires: i64,
    #[serde(default)]
    pub invalidated: bool,
    #[serde(default)]
    pub refresh_invalidated: bool,
    #[serde(default)]
    pub refreshed: bool,
    /// Snapshot of the authenticated user.
    pub authn: AuthnSnapshot,
}

/// The authentication a token carries (ES stores the whole `Authentication`
/// in the token document: authenticating subject, run-as target and type).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthnSnapshot {
    pub username: String,
    pub roles: Vec<String>,
    pub full_name: Option<String>,
    pub email: Option<String>,
    pub metadata: Value,
    pub realm_name: String,
    pub realm_type: String,
    /// Set when the authenticating subject is an API key: the token keeps
    /// the key's role descriptors and its owner's `limited_by` ceiling
    /// (ES `Authentication.token()` keeps the `API_KEY` subject).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<TokenApiKey>,
    /// The run-as target (effective subject) when the token was issued for a
    /// run-as authentication; the run-as privilege is checked again on use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_as: Option<Box<AuthnSnapshot>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenApiKey {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceToken {
    pub hash: String,
    pub creation: i64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SecState {
    #[serde(default)]
    native_revision: u64,
    #[serde(default)]
    pub users: BTreeMap<String, NativeUser>,
    #[serde(default)]
    pub reserved: BTreeMap<String, ReservedState>,
    #[serde(default)]
    pub roles: BTreeMap<String, Value>,
    #[serde(default)]
    pub role_mappings: BTreeMap<String, Value>,
    #[serde(default)]
    pub app_privileges: BTreeMap<String, BTreeMap<String, Value>>,
    #[serde(default)]
    pub api_keys: BTreeMap<String, ApiKey>,
    #[serde(default)]
    pub tokens: BTreeMap<String, TokenDoc>,
    #[serde(default)]
    pub service_tokens: BTreeMap<String, BTreeMap<String, ServiceToken>>,
    /// User profiles by uid.
    #[serde(default)]
    pub profiles: BTreeMap<String, Value>,
    /// `_security/settings`: flat index settings of `security`, `security-tokens`, `security-profile`.
    #[serde(default)]
    pub index_settings: BTreeMap<String, Map<String, Value>>,
}

/// What the request authenticated as.
#[derive(Clone, Debug)]
pub enum Principal {
    User,
    /// `metadata`: the key's own metadata (exposed to `set_security_user`).
    ApiKey { id: String, name: String, metadata: Value },
    ServiceAccount { token_name: String, source: &'static str },
    Anonymous,
    /// A remote cluster's user, over API key based remote cluster security
    /// (ES RCS 2.0, `cross_cluster_access`): `descriptors` are the role
    /// descriptors the querying cluster grants the user for this cluster
    /// (its roles' `remote_indices` / `remote_cluster`), limited by the
    /// cross-cluster API key `key_id`'s access.
    CrossCluster { key_id: String, descriptors: Arc<Vec<Value>> },
}

#[derive(Clone, Debug)]
pub struct Authn {
    pub username: String,
    /// Roles shown by `_authenticate` (empty for API keys / service accounts).
    pub roles: Vec<String>,
    pub full_name: Option<String>,
    pub email: Option<String>,
    pub metadata: Value,
    pub enabled: bool,
    pub realm_name: String,
    pub realm_type: String,
    /// `realm`, `api_key`, `token`, `anonymous`.
    pub auth_type: &'static str,
    pub principal: Principal,
    /// Run-as: the effective user (and its realm) when `es-security-runas-user` was honoured.
    pub run_as: Option<Box<Authn>>,
}

impl Authn {
    pub fn is_api_key(&self) -> bool {
        matches!(self.principal, Principal::ApiKey { .. })
    }
    pub fn api_key_id(&self) -> Option<&str> {
        match &self.principal {
            Principal::ApiKey { id, .. } => Some(id),
            _ => None,
        }
    }
    pub fn is_service_account(&self) -> bool {
        matches!(self.principal, Principal::ServiceAccount { .. })
    }
    /// The effective subject (run-as target if any).
    pub fn effective(&self) -> &Authn {
        self.run_as.as_deref().unwrap_or(self)
    }

    /// The token snapshot: the authenticating subject (an API key stays an
    /// API key) plus the run-as target, never the effective user alone.
    fn snapshot(&self) -> AuthnSnapshot {
        AuthnSnapshot {
            username: self.username.clone(),
            roles: self.roles.clone(),
            full_name: self.full_name.clone(),
            email: self.email.clone(),
            metadata: self.metadata.clone(),
            realm_name: self.realm_name.clone(),
            realm_type: self.realm_type.clone(),
            api_key: match &self.principal {
                Principal::ApiKey { id, name, .. } => Some(TokenApiKey { id: id.clone(), name: name.clone() }),
                _ => None,
            },
            run_as: self.run_as.as_ref().map(|target| Box::new(target.snapshot())),
        }
    }

    /// `_security/_authenticate` body.
    pub fn to_json(&self) -> Value {
        let e = self.effective();
        let mut m = Map::new();
        m.insert("username".into(), json!(e.username));
        m.insert("roles".into(), json!(e.roles));
        m.insert("full_name".into(), json!(e.full_name));
        m.insert("email".into(), json!(e.email));
        if let Principal::ServiceAccount { token_name, source } = &self.principal {
            m.insert("token".into(), json!({"name": token_name, "type": format!("_service_account_{source}")}));
        }
        m.insert("metadata".into(), if e.metadata.is_null() { json!({}) } else { e.metadata.clone() });
        m.insert("enabled".into(), json!(e.enabled));
        m.insert("authentication_realm".into(), json!({"name": self.realm_name, "type": self.realm_type}));
        m.insert("lookup_realm".into(), json!({"name": e.realm_name, "type": e.realm_type}));
        m.insert("authentication_type".into(), json!(self.auth_type));
        if let Principal::ApiKey { id, name, .. } = &self.principal {
            m.insert("api_key".into(), json!({"id": id, "name": name}));
        }
        Value::Object(m)
    }
}

/// Credentials extracted from the `Authorization` header.
#[derive(Clone)]
pub enum Credentials {
    None,
    Basic(String, String),
    ApiKey(String),
    Bearer(String),
    /// Unparseable credentials for a known scheme (message for `additional_unsuccessful_credentials`).
    Malformed(String),
}

pub enum AuthFail {
    /// No credentials and no anonymous access.
    Missing,
    /// Basic auth (or token user) failed for this principal.
    User(String),
    /// Non-user credentials failed (API key, token); messages for metadata.
    Other(Vec<String>),
    /// Expired bearer token.
    TokenExpired,
    /// Credentials rejected with a fixed 401 reason (e.g. wrong API key type).
    Reason(String),
}

/// Effective permissions of an authenticated subject.
#[derive(Clone)]
pub struct Permission {
    pub role: Arc<Role>,
    /// API keys: the owner's limiting role (both must grant).
    pub limit: Option<Arc<Role>>,
    pub effective_roles: Vec<String>,
    pub missing_roles: Vec<String>,
    pub is_superuser: bool,
}

impl Permission {
    pub fn check_cluster(&self, action: &str) -> bool {
        self.role.check_cluster(action) && self.limit.as_ref().map(|l| l.check_cluster(action)).unwrap_or(true)
    }
    pub fn check_index_action(&self, action: &str) -> bool {
        self.role.check_index_action(action) && self.limit.as_ref().map(|l| l.check_index_action(action)).unwrap_or(true)
    }
    pub fn check_index(&self, action: &str, name: &str) -> bool {
        self.check_index_component(action, name, false)
    }
    /// `action` on `name`'s `::failures` component (`failures`) or its data.
    pub fn check_index_component(&self, action: &str, name: &str, failures: bool) -> bool {
        self.role.check_index_component(action, name, failures) && self.limit.as_ref().map(|l| l.check_index_component(action, name, failures)).unwrap_or(true)
    }
    pub fn has_cluster_privilege(&self, name: &str) -> bool {
        self.role.has_cluster_privilege(name) && self.limit.as_ref().map(|l| l.has_cluster_privilege(name)).unwrap_or(true)
    }
    pub fn has_index_privilege(&self, index: &str, name: &str) -> bool {
        self.role.has_index_privilege(index, name) && self.limit.as_ref().map(|l| l.has_index_privilege(index, name)).unwrap_or(true)
    }
    /// Run-as: an API key's own `run_as` is bounded by its owner's
    /// (`limited_by`) role, as ES's `LimitedRole.checkRunAs`.
    pub fn can_run_as(&self, user: &str) -> bool {
        self.role.can_run_as(user) && self.limit.as_ref().map(|l| l.can_run_as(user)).unwrap_or(true)
    }

    /// Does the role (or an API key's limiting role) carry any DLS/FLS group?
    /// `false` guarantees no read is ever filtered for this subject.
    pub fn has_dls_fls(&self) -> bool {
        self.role.has_dls_fls() || self.limit.as_ref().is_some_and(|l| l.has_dls_fls())
    }

    /// DLS/FLS for `action` on a concrete index reached through `names`
    /// (the index name first, then the alias names the request used).
    /// API keys intersect their descriptors with the owner's (`limited_by`):
    /// filters AND together and field grants intersect. An error means the
    /// access cannot be computed and the request must be denied.
    pub fn index_access(&self, action: &str, names: &[&str], user: &Authn) -> Result<roles::IndexAccess, EsError> {
        let a = self.role.index_access(action, names, user)?;
        match &self.limit {
            Some(l) if !Arc::ptr_eq(l, &self.role) => Ok(a.and(l.index_access(action, names, user)?)),
            _ => Ok(a),
        }
    }

    /// The DLS filter (query DSL) for reads of `index`; `None` = unrestricted.
    /// Several granting groups OR together; any granting group without a query makes it `None`.
    pub fn doc_filter(&self, index: &str, user: &Authn) -> Result<Option<Value>, EsError> {
        Ok(self.index_access("indices:data/read/search", &[index], user)?.dls)
    }

    /// The FLS for reads of `index`; `None` = every field.
    pub fn field_permissions(&self, index: &str, user: &Authn) -> Result<Option<roles::FieldAccess>, EsError> {
        Ok(self.index_access("indices:data/read/search", &[index], user)?.fields)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Defer {
    Off,
    Armed,
    Deferred,
}

thread_local! {
    /// [`Security::authenticate_nonblocking`] on this thread: skip (and
    /// report) uncached expensive verification.
    static DEFER_EXPENSIVE: std::cell::Cell<Defer> = const { std::cell::Cell::new(Defer::Off) };
}

pub struct Security {
    enabled: AtomicBool,
    pub cfg: RwLock<SecConfig>,
    pub state: RwLock<SecState>,
    path: PathBuf,
    persist_lock: Mutex<()>,
    native_snapshot_seen: AtomicBool,
    /// Bumped on every change; invalidates caches.
    generation: AtomicU64,
    /// sha256(scheme:principal:secret) -> (generation at verification time,
    /// `scheme\0principal` for targeted invalidation by `_clear_cache`).
    verified: Mutex<HashMap<[u8; 32], (u64, String)>>,
    /// role-set key -> compiled role, tagged with the generation the
    /// descriptors were read under.
    role_cache: Mutex<HashMap<String, (u64, Arc<Role>)>>,
    /// Tests: runs after a permission lookup read its descriptors and before
    /// it compiles and caches them (interleaves a concurrent role change).
    #[cfg(test)]
    after_descriptors_hook: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl Security {
    /// Open with settings from the environment.
    pub fn open(data: &Path) -> Security {
        let cfg = SecConfig::from_env();
        let path = data.join("_state/security.json");
        let mut state: SecState = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        // Tokens stored before ids were hashed are invalid (clients
        // re-authenticate); their plaintext ids leave the state.
        state.tokens.retain(|_, t| t.hashed);
        Security {
            enabled: AtomicBool::new(cfg.enabled),
            cfg: RwLock::new(cfg),
            state: RwLock::new(state),
            path,
            persist_lock: Mutex::new(()),
            native_snapshot_seen: AtomicBool::new(false),
            generation: AtomicU64::new(1),
            verified: Mutex::new(HashMap::new()),
            role_cache: Mutex::new(HashMap::new()),
            #[cfg(test)]
            after_descriptors_hook: Mutex::new(None),
        }
    }

    /// Hot path: one relaxed atomic load.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Replace the configuration (tests, embedders).
    pub fn configure(&self, cfg: SecConfig) {
        self.enabled.store(cfg.enabled, Ordering::Relaxed);
        *self.cfg.write() = cfg;
        self.bump();
    }

    pub fn bump(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Re-read the file realm (`users`, `users_roles`), `service_tokens`
    /// and the file roles from `conf`, replacing the loaded ones (ES's file
    /// realm reloads its files when they change).
    pub fn reload_files(&self, conf: &Path) {
        let mut fresh = SecConfig::default();
        fresh.load_files(conf);
        {
            let mut cfg = self.cfg.write();
            cfg.file_users = fresh.file_users;
            cfg.file_user_roles = fresh.file_user_roles;
            cfg.file_service_tokens = fresh.file_service_tokens;
            cfg.file_roles = fresh.file_roles;
        }
        self.bump();
    }

    /// Watch the security files in `conf` and [`Self::reload_files`] when
    /// one changes (ES `ResourceWatcherService`; polled every `interval`).
    /// The first poll loads files that exist, so a watcher on a config
    /// directory other than the one read at startup still takes effect.
    /// The thread ends once the `Security` is dropped.
    pub fn watch_files(self: &Arc<Self>, conf: PathBuf, interval: std::time::Duration) {
        const FILES: &[&str] = &["users", "users_roles", "service_tokens", "roles.yml", "roles.json"];
        let weak = Arc::downgrade(self);
        let snapshot = move |conf: &Path| -> Vec<Option<Vec<u8>>> { FILES.iter().map(|f| std::fs::read(conf.join(f)).ok()).collect() };
        let spawned = std::thread::Builder::new().name("security-files".into()).spawn(move || {
            let mut last: Vec<Option<Vec<u8>>> = vec![None; FILES.len()];
            loop {
                let Some(security) = weak.upgrade() else { return };
                let now = snapshot(&conf);
                if now != last {
                    security.reload_files(&conf);
                    tracing::debug!(conf = %conf.display(), "reloaded the security files");
                    last = now;
                }
                drop(security);
                std::thread::sleep(interval);
            }
        });
        if let Err(error) = spawned {
            tracing::warn!(%error, "cannot watch the security files");
        }
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Persist the state after a change.
    pub fn save(&self) -> Result<(), EsError> {
        self.bump();
        let _persist = self.persist_lock.lock();
        let bytes = {
            let mut state = self.state.write();
            state.native_revision = state.native_revision.saturating_add(1);
            serde_json::to_vec(&*state)?
        };
        self.native_snapshot_seen.store(true, Ordering::Release);
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("json.tmp");
        write_private(&tmp, &bytes)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    /// Native security state is cluster-owned; file realms remain node-owned.
    pub fn cluster_state(&self) -> Value {
        let state = self.state.read();
        self.native_snapshot_seen.store(true, Ordering::Release);
        serde_json::to_value(&*state).expect("serializable security state")
    }

    pub fn restore_cluster_state(&self, value: &Value) -> Result<bool, EsError> {
        let restored: SecState = serde_json::from_value(value.clone())?;
        let _persist = self.persist_lock.lock();
        let mut state = self.state.write();
        let current = serde_json::to_value(&*state)?;
        if restored.native_revision < state.native_revision
            || (self.native_snapshot_seen.load(Ordering::Acquire) && restored.native_revision == state.native_revision && current != *value) {
            return Ok(false);
        }
        self.native_snapshot_seen.store(true, Ordering::Release);
        if current == *value { return Ok(false); }
        if let Some(parent) = self.path.parent() { std::fs::create_dir_all(parent)?; }
        let temporary = self.path.with_extension("cluster.tmp");
        let mut restored = restored;
        restored.tokens.retain(|_, t| t.hashed);
        write_private(&temporary, &serde_json::to_vec(&restored)?)?;
        std::fs::rename(&temporary, &self.path)?;
        *state = restored;
        self.bump();
        Ok(true)
    }

    /// Security feature-state reset. File-realm configuration is node-owned
    /// and survives; native users, keys, tokens, profiles and passwords do not.
    pub fn reset(&self) -> Result<(), EsError> {
        {
            let mut state = self.state.write();
            let revision = state.native_revision;
            *state = SecState { native_revision: revision, ..Default::default() };
        }
        self.save()
    }

    // ---------------------------------------------------------------- realms

    fn check_cached(&self, key: &[u8; 32]) -> bool {
        let g = self.generation();
        self.verified.lock().get(key).map(|v| v.0 == g).unwrap_or(false)
    }

    /// Cache a verified credential under the generation read before the
    /// verification (`g`); refused when the generation moved meanwhile.
    fn remember(&self, key: [u8; 32], g: u64, tag: String) {
        let mut v = self.verified.lock();
        if self.generation() != g {
            return;
        }
        if v.len() > 10_000 {
            v.clear();
        }
        v.insert(key, (g, tag));
    }

    fn verify_cached(&self, scheme: &str, principal: &str, secret: &str, hash: &str) -> bool {
        let g = self.generation();
        let key = hash::sha256(format!("{scheme}\u{0}{principal}\u{0}{secret}\u{0}{hash}").as_bytes());
        if self.check_cached(&key) {
            return true;
        }
        // On a networking thread ([`Self::authenticate_nonblocking`]), an
        // uncached bcrypt / PBKDF2 verification is not run: the caller
        // repeats the whole authentication on the crypto pool.
        if hash::is_expensive(hash) && DEFER_EXPENSIVE.with(|d| d.get() != Defer::Off) {
            DEFER_EXPENSIVE.with(|d| d.set(Defer::Deferred));
            return false;
        }
        if hash::verify(secret, hash) {
            self.remember(key, g, format!("{scheme}\u{0}{principal}"));
            true
        } else {
            false
        }
    }

    /// Cache contents: the role-cache keys and the verified-credential tags
    /// (`scheme\0principal`), for tests and diagnostics.
    #[doc(hidden)]
    pub fn cache_keys(&self) -> (Vec<String>, Vec<String>) {
        let mut roles: Vec<String> = self.role_cache.lock().keys().cloned().collect();
        let mut verified: Vec<String> = self.verified.lock().values().map(|(_, tag)| tag.clone()).collect();
        roles.sort();
        verified.sort();
        (roles, verified)
    }

    /// The `_clear_cache` security APIs on this node (the REST layer fans
    /// them out to every node). `kind`: `roles` (role names; none = all),
    /// `api_keys` (key ids), `realms` (realm names, `usernames`),
    /// `privileges` (application names: ES also clears the role cache) and
    /// `service_tokens` (`ns/service/token` names, `*` for every token).
    /// Clearing only drops cache entries: nothing is ever cached past a
    /// security change (the generation check), so a clear never loosens.
    pub fn clear_cache(&self, kind: &str, names: &[String], usernames: &[String]) -> Result<(), EsError> {
        let all = names.is_empty() || names.iter().any(|n| n == "*" || n == "_all");
        let tag_principal = |tag: &str, scheme: &str| tag.strip_prefix(scheme).and_then(|t| t.strip_prefix('\u{0}')).map(str::to_owned);
        match kind {
            "roles" => {
                let mut cache = self.role_cache.lock();
                if all {
                    cache.clear();
                } else {
                    cache.retain(|key, _| match key.strip_prefix("roles:") {
                        Some(set) => !set.split('\u{0}').any(|r| names.iter().any(|n| n == r)),
                        None => true,
                    });
                }
            }
            "privileges" => self.role_cache.lock().clear(),
            "api_keys" => {
                self.role_cache.lock().retain(|key, _| {
                    let id = key.strip_prefix("apikey-limit:").or_else(|| key.strip_prefix("apikey:"));
                    match id {
                        Some(id) => !(all || names.iter().any(|n| n == id)),
                        None => true,
                    }
                });
                self.verified.lock().retain(|_, (_, tag)| match tag_principal(tag, "apikey") {
                    Some(id) => !(all || names.iter().any(|n| *n == id)),
                    None => true,
                });
            }
            "realms" => {
                // One verified-credential cache serves every realm here, so a
                // named realm clears it for those users across realms (more,
                // never less, than ES clears).
                let every_user = usernames.is_empty() || usernames.iter().any(|u| u == "*");
                self.verified.lock().retain(|_, (_, tag)| match tag_principal(tag, "basic") {
                    Some(user) => !(every_user || usernames.iter().any(|u| *u == user)),
                    None => true,
                });
            }
            "service_tokens" => {
                self.verified.lock().retain(|_, (_, tag)| match tag_principal(tag, "svc") {
                    Some(token) => !names.iter().any(|n| privilege::glob(n, &token)),
                    None => true,
                });
            }
            other => return Err(EsError::illegal_argument(format!("unknown security cache [{other}]"))),
        }
        Ok(())
    }

    /// Look up a user (no password check) as the realms would: reserved, file, native.
    pub fn lookup_user(&self, username: &str) -> Option<Authn> {
        if let Some(roles) = reserved_user_roles(username).filter(|_| self.cfg.read().reserved_realm_enabled) {
            let st = self.state.read();
            let enabled = st.reserved.get(username).map(|r| r.enabled).unwrap_or(true);
            return Some(user_authn(username, roles.iter().map(|s| s.to_string()).collect(), None, None, reserved_user_metadata(username), enabled, "reserved", "reserved"));
        }
        let cfg = self.cfg.read();
        if cfg.file_users.contains_key(username) {
            let roles = cfg.file_user_roles.get(username).cloned().unwrap_or_default();
            return Some(user_authn(username, roles, None, None, json!({}), true, &cfg.file_realm_name, "file"));
        }
        let native = cfg.native_realm_name.clone();
        drop(cfg);
        let st = self.state.read();
        st.users.get(username).map(|u| user_authn(username, u.roles.clone(), u.full_name.clone(), u.email.clone(), u.metadata.clone(), u.enabled, &native, "native"))
    }

    /// Verify a username/password through the realm chain.
    pub fn authenticate_password(&self, user: &str, pass: &str) -> Result<Authn, AuthFail> {
        if reserved_user_roles(user).is_some() && self.cfg.read().reserved_realm_enabled {
            let st = self.state.read();
            let rs = st.reserved.get(user).cloned().unwrap_or_default();
            drop(st);
            let ok = match &rs.password_hash {
                Some(h) => self.verify_cached("basic", user, pass, h),
                None => user == "elastic" && self.cfg.read().bootstrap_password.as_deref().map(|b| !b.is_empty() && hash::ct_eq(b.as_bytes(), pass.as_bytes())).unwrap_or(false),
            };
            if !ok || !rs.enabled {
                return Err(AuthFail::User(user.to_string()));
            }
            return self.lookup_user(user).ok_or_else(|| AuthFail::User(user.to_string()));
        }
        let file_hash = self.cfg.read().file_users.get(user).cloned();
        if let Some(h) = file_hash {
            if self.verify_cached("basic", user, pass, &h) {
                return self.lookup_user(user).ok_or_else(|| AuthFail::User(user.to_string()));
            }
        }
        let native = self.state.read().users.get(user).cloned();
        if let Some(u) = native {
            if u.enabled && self.verify_cached("basic", user, pass, &u.password_hash) {
                let native = self.cfg.read().native_realm_name.clone();
                return Ok(user_authn(user, u.roles, u.full_name, u.email, u.metadata, true, &native, "native"));
            }
        }
        Err(AuthFail::User(user.to_string()))
    }

    pub fn anonymous(&self) -> Option<Authn> {
        let cfg = self.cfg.read();
        if cfg.anonymous_roles.is_empty() {
            return None;
        }
        let mut a = user_authn(&cfg.anonymous_username, cfg.anonymous_roles.clone(), None, None, json!({"_reserved": true}), true, "__anonymous", "__anonymous");
        a.auth_type = "anonymous";
        a.principal = Principal::Anonymous;
        Some(a)
    }

    /// ES's internal `_system` user: cluster-internal work that has no caller
    /// (e.g. a restore resumed after a master failover). It authenticates
    /// only from the in-process system marker, never from request
    /// credentials, and holds the reserved `superuser` role.
    pub fn system() -> Authn {
        let mut a = user_authn("_system", vec!["superuser".into()], None, None, json!({"_reserved": true}), true, "__attach", "__attach");
        a.auth_type = "internal";
        a
    }

    /// [`Self::authenticate`] when it needs no expensive (bcrypt / PBKDF2)
    /// password verification: a cached success, a cheap hash, an unknown
    /// principal, missing or malformed credentials. `None`: deciding needs
    /// expensive verification, which must run on the bounded crypto pool
    /// (`elasticsearch_http::security::edge`), not on this (networking)
    /// thread. Only successes are cached (ES `CachingUsernamePasswordRealm`).
    pub fn authenticate_nonblocking(&self, creds: &Credentials) -> Option<Result<Authn, AuthFail>> {
        let previous = DEFER_EXPENSIVE.with(|d| d.replace(Defer::Armed));
        let result = self.authenticate(creds.clone());
        let deferred = DEFER_EXPENSIVE.with(|d| d.replace(previous)) == Defer::Deferred;
        if deferred { None } else { Some(result) }
    }

    /// The principal whose password an expensive verification of `creds`
    /// would check (failed-authentication rate limiting key).
    pub fn credential_principal(creds: &Credentials) -> Option<&str> {
        match creds {
            Credentials::Basic(user, _) => Some(user),
            _ => None,
        }
    }

    /// Authenticate the request credentials.
    pub fn authenticate(&self, creds: Credentials) -> Result<Authn, AuthFail> {
        match creds {
            Credentials::None => self.anonymous().ok_or(AuthFail::Missing),
            Credentials::Basic(u, p) => self.authenticate_password(&u, &p),
            Credentials::ApiKey(raw) => self.authenticate_api_key(&raw).map_err(|m| {
                if m.starts_with("authentication expected API key type") {
                    AuthFail::Reason(m)
                } else {
                    AuthFail::Other(vec![format!("API key: {m}")])
                }
            }),
            Credentials::Bearer(tok) => self.authenticate_bearer(&tok),
            Credentials::Malformed(m) => Err(AuthFail::Other(vec![m])),
        }
    }

    // -------------------------------------------------------------- api keys

    /// Authenticate a remote cluster's request over API key based remote
    /// cluster security: `raw` must be an active cross-cluster API key
    /// (`POST /_security/cross_cluster/api_key`); the request then runs as
    /// `user` (the querying cluster's user) with `descriptors` limited by
    /// the key's access.
    pub fn authenticate_cross_cluster(&self, raw: &str, user: &str, descriptors: Vec<Value>) -> Result<Authn, String> {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD.decode(raw.trim()).ok().and_then(|b| String::from_utf8(b).ok()).ok_or_else(|| "invalid cross-cluster API key value".to_string())?;
        let (id, secret) = decoded.split_once(':').ok_or_else(|| "invalid cross-cluster API key value".to_string())?;
        let key = self.state.read().api_keys.get(id).cloned().ok_or_else(|| format!("unable to find apikey with id {id}"))?;
        if !self.verify_cached("apikey", id, secret, &key.hash) {
            return Err(format!("invalid credentials for API key [{id}]"));
        }
        if key.invalidated {
            return Err("api key has been invalidated".into());
        }
        if key.expiration.map(|e| e <= now_ms()).unwrap_or(false) {
            return Err("api key is expired".into());
        }
        if key.key_type != "cross_cluster" {
            return Err(format!("authentication expected API key type of [cross_cluster], but API key [{id}] has type [{}]", key.key_type));
        }
        Ok(Authn {
            username: user.to_string(),
            roles: vec![],
            full_name: None,
            email: None,
            metadata: json!({}),
            enabled: true,
            realm_name: "_es_cross_cluster".into(),
            realm_type: "_es_cross_cluster".into(),
            auth_type: "api_key",
            principal: Principal::CrossCluster { key_id: key.id.clone(), descriptors: Arc::new(descriptors) },
            run_as: None,
        })
    }

    pub fn authenticate_api_key(&self, raw: &str) -> Result<Authn, String> {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD.decode(raw.trim()).ok().and_then(|b| String::from_utf8(b).ok()).ok_or_else(|| "invalid ApiKey value".to_string())?;
        let (id, secret) = decoded.split_once(':').ok_or_else(|| "invalid ApiKey value".to_string())?;
        let key = self.state.read().api_keys.get(id).cloned().ok_or_else(|| format!("unable to find apikey with id {id}"))?;
        if !self.verify_cached("apikey", id, secret, &key.hash) {
            return Err(format!("invalid credentials for API key [{id}]"));
        }
        if key.invalidated {
            return Err("api key has been invalidated".into());
        }
        if key.expiration.map(|e| e <= now_ms()).unwrap_or(false) {
            return Err("api key is expired".into());
        }
        if key.key_type != "rest" {
            return Err(format!("authentication expected API key type of [rest], but API key [{id}] has type [{}]", key.key_type));
        }
        Ok(Authn {
            username: key.username.clone(),
            roles: vec![],
            full_name: key.full_name.clone(),
            email: key.email.clone(),
            metadata: if key.user_metadata.is_null() { json!({}) } else { key.user_metadata.clone() },
            enabled: true,
            realm_name: "_es_api_key".into(),
            realm_type: "_es_api_key".into(),
            auth_type: "api_key",
            principal: Principal::ApiKey { id: key.id.clone(), name: key.name.clone(), metadata: key.metadata.clone() },
            run_as: None,
        })
    }

    /// Create an API key owned by `owner`; returns (id, secret, expiration).
    pub fn create_api_key(&self, owner: &Authn, owner_descriptors: Map<String, Value>, name: String, expiration: Option<i64>, role_descriptors: Map<String, Value>, metadata: Value) -> Result<ApiKey, EsError> {
        let secret = hash::random_b64url(16);
        let id = hash::random_b64url(15);
        let owner = owner.effective();
        let key = ApiKey {
            id: id.clone(),
            name,
            hash: Hasher::Ssha256.hash(&secret),
            creation: now_ms(),
            expiration,
            invalidated: false,
            invalidation: None,
            username: owner.username.clone(),
            realm: owner.realm_name.clone(),
            realm_type: owner.realm_type.clone(),
            full_name: owner.full_name.clone(),
            email: owner.email.clone(),
            user_metadata: owner.metadata.clone(),
            metadata: if metadata.is_null() { json!({}) } else { metadata },
            role_descriptors,
            limited_by: owner_descriptors,
            key_type: "rest".into(),
            access: Value::Null,
        };
        self.state.write().api_keys.insert(id.clone(), key.clone());
        self.save()?;
        // Return the secret through `hash` of a clone (caller formats the response).
        let mut out = key;
        out.hash = secret;
        Ok(out)
    }

    /// `ApiKeyService.cloneApiKey`: the source key (`id:secret` credential)
    /// is validated like an authentication, then copied (creator, role
    /// descriptors, limited-by, type) under a new id / secret / name.
    /// `expiration`: `None` keeps the source's, `Some(None)` none, `Some(Some(ms))` that.
    /// `Err(Ok(msg))`: the credential did not authenticate; `Err(Err(e))`: another failure.
    pub fn clone_api_key(&self, id: &str, secret: &str, name: String, expiration: Option<Option<i64>>, metadata: Option<Value>) -> Result<ApiKey, std::result::Result<String, EsError>> {
        let src = self.state.read().api_keys.get(id).cloned().ok_or_else(|| Ok(format!("unable to find apikey with id {id}")))?;
        if !self.verify_cached("apikey", id, secret, &src.hash) {
            return Err(Ok(format!("invalid credentials for API key [{id}]")));
        }
        if src.key_type != "rest" {
            return Err(Ok(format!("authentication expected API key type of [rest], but API key [{id}] has type [{}]", src.key_type)));
        }
        if src.invalidated {
            return Err(Ok("api key has been invalidated".into()));
        }
        if src.expiration.map(|e| e <= now_ms()).unwrap_or(false) {
            return Err(Ok("api key is expired".into()));
        }
        let mut md = match metadata {
            Some(Value::Object(m)) => m,
            Some(_) => Map::new(),
            None => src.metadata.as_object().cloned().unwrap_or_default(),
        };
        md.insert("_cloned_from".into(), json!(id));
        let new_secret = hash::random_b64url(16);
        let new_id = hash::random_b64url(15);
        let mut key = src;
        key.id = new_id.clone();
        key.name = name;
        key.hash = Hasher::Ssha256.hash(&new_secret);
        key.creation = now_ms();
        key.expiration = match expiration {
            None => key.expiration,
            Some(e) => e,
        };
        key.invalidated = false;
        key.invalidation = None;
        key.metadata = Value::Object(md);
        self.state.write().api_keys.insert(new_id, key.clone());
        self.save().map_err(Err)?;
        let mut out = key;
        out.hash = new_secret;
        Ok(out)
    }

    // ---------------------------------------------------------------- tokens

    fn authenticate_bearer(&self, tok: &str) -> Result<Authn, AuthFail> {
        if let Some((ns_svc, name, secret)) = parse_service_token(tok) {
            return self.authenticate_service_token(&ns_svc, &name, &secret).map_err(|m| AuthFail::Other(vec![format!("Service account: {m}")]));
        }
        let enabled = {
            let c = self.cfg.read();
            c.token_enabled && c.trial
        };
        if !enabled {
            return Err(AuthFail::Other(vec![]));
        }
        // Looked up by hash; documents from before hashing never match.
        let doc = self.state.read().tokens.get(&token_hash(tok)).filter(|d| d.hashed).cloned();
        let Some(doc) = doc else { return Err(AuthFail::Other(vec!["OAuth2 token: token malformed".into()])) };
        if doc.invalidated {
            return Err(AuthFail::Other(vec!["OAuth2 token: token expired".into()]));
        }
        if doc.expires <= now_ms() {
            return Err(AuthFail::TokenExpired);
        }
        self.authn_from_snapshot(&doc.authn).map_err(|m| AuthFail::Other(vec![format!("OAuth2 token: {m}")]))
    }

    /// Rebuild a token's authentication. An API-key subject stays an API key
    /// (its descriptors and `limited_by` ceiling keep applying, and an
    /// invalidated, expired or deleted key ends the token); a run-as target
    /// is honoured only while the authenticating subject may still run as it
    /// (ES re-authorizes run-as on every request).
    pub fn authn_from_snapshot(&self, s: &AuthnSnapshot) -> Result<Authn, String> {
        let mut a = match &s.api_key {
            Some(k) => {
                let key = self.state.read().api_keys.get(&k.id).cloned().ok_or_else(|| format!("unable to find apikey with id {}", k.id))?;
                if key.invalidated {
                    return Err("api key has been invalidated".into());
                }
                if key.expiration.map(|e| e <= now_ms()).unwrap_or(false) {
                    return Err("api key is expired".into());
                }
                if key.key_type != "rest" {
                    return Err(format!("authentication expected API key type of [rest], but API key [{}] has type [{}]", key.id, key.key_type));
                }
                Authn {
                    username: key.username.clone(),
                    roles: vec![],
                    full_name: key.full_name.clone(),
                    email: key.email.clone(),
                    metadata: if key.user_metadata.is_null() { json!({}) } else { key.user_metadata.clone() },
                    enabled: true,
                    realm_name: "_es_api_key".into(),
                    realm_type: "_es_api_key".into(),
                    auth_type: "token",
                    principal: Principal::ApiKey { id: key.id.clone(), name: key.name.clone(), metadata: key.metadata.clone() },
                    run_as: None,
                }
            }
            None => user_authn(&s.username, s.roles.clone(), s.full_name.clone(), s.email.clone(), s.metadata.clone(), true, &s.realm_name, &s.realm_type),
        };
        a.auth_type = "token";
        if let Some(target) = &s.run_as {
            if !self.permission(&a).can_run_as(&target.username) {
                return Err(format!("[{}] is unauthorized to run as [{}]", a.username, target.username));
            }
            let mut t = user_authn(&target.username, target.roles.clone(), target.full_name.clone(), target.email.clone(), target.metadata.clone(), true, &target.realm_name, &target.realm_type);
            t.auth_type = "token";
            a.run_as = Some(Box::new(t));
        }
        Ok(a)
    }

    /// Issue an access (+ optional refresh) token for `authn`, obtained by
    /// `client` (the authenticated caller of the token API). The store keeps
    /// only the tokens' hashes; the returned copy carries the raw values.
    pub fn create_token(&self, authn: &Authn, with_refresh: bool, client: &Authn) -> Result<TokenDoc, EsError> {
        let timeout = self.cfg.read().token_timeout_ms;
        let now = now_ms();
        let (access, refresh) = (hash::random_b64url(48), with_refresh.then(|| hash::random_b64url(32)));
        let stored = TokenDoc {
            access_token: token_hash(&access),
            refresh_token: refresh.as_deref().map(token_hash),
            hashed: true,
            client: Some(owner::ResourceOwner::of(client)),
            creation: now,
            expires: now + timeout,
            invalidated: false,
            refresh_invalidated: false,
            refreshed: false,
            authn: authn.snapshot(),
        };
        let mut st = self.state.write();
        // Drop long-dead tokens, and any stored before ids were hashed.
        let horizon = now - 24 * 3600 * 1000;
        st.tokens.retain(|_, t| t.hashed && t.creation > horizon);
        st.tokens.insert(stored.access_token.clone(), stored.clone());
        drop(st);
        self.save()?;
        let mut out = stored;
        out.access_token = access;
        out.refresh_token = refresh;
        Ok(out)
    }

    /// The stored token documents matching a raw access and/or refresh
    /// token (`_security/oauth2/token` invalidation).
    pub fn token_matches(doc: &TokenDoc, access: Option<&str>, refresh: Option<&str>) -> bool {
        doc.hashed
            && (access.is_some_and(|a| hash::ct_eq(token_hash(a).as_bytes(), doc.access_token.as_bytes()))
                || refresh.is_some_and(|r| doc.refresh_token.as_deref().is_some_and(|h| hash::ct_eq(token_hash(r).as_bytes(), h.as_bytes()))))
    }

    // ------------------------------------------------------ service accounts

    fn authenticate_service_token(&self, ns_svc: &str, name: &str, secret: &str) -> Result<Authn, String> {
        if service_account(ns_svc).is_none() {
            return Err(format!("the [{ns_svc}] service account does not exist"));
        }
        let qualified = format!("{ns_svc}/{name}");
        let file_hash = self.cfg.read().file_service_tokens.get(&qualified).cloned();
        let (hash_str, source) = match file_hash {
            Some(h) => (h, "file"),
            None => match self.state.read().service_tokens.get(ns_svc).and_then(|m| m.get(name)).map(|t| t.hash.clone()) {
                Some(h) => (h, "index"),
                None => return Err(format!("unable to authenticate service account token [{qualified}]")),
            },
        };
        if !self.verify_cached("svc", &qualified, secret, &hash_str) {
            return Err(format!("unable to authenticate service account token [{qualified}]"));
        }
        Ok(Authn {
            username: ns_svc.to_string(),
            roles: vec![],
            full_name: Some(format!("Service account - {ns_svc}")),
            email: None,
            metadata: json!({"_elastic_service_account": true}),
            enabled: true,
            realm_name: "_service_account".into(),
            realm_type: "_service_account".into(),
            auth_type: "token",
            principal: Principal::ServiceAccount { token_name: name.to_string(), source },
            run_as: None,
        })
    }

    // ----------------------------------------------------------------- roles

    /// Resolve a role name to its descriptor (reserved, file, native).
    pub fn role_descriptor(&self, name: &str) -> Option<Value> {
        if let Some(r) = roles::reserved_roles().get(name) {
            return Some(r.clone());
        }
        if let Some(r) = self.cfg.read().file_roles.get(name) {
            return Some(r.clone());
        }
        self.state.read().roles.get(name).cloned()
    }

    /// The compiled role for `key`. `g` is the generation read *before*
    /// the descriptors `descs` returns were read: an entry is reused only
    /// at that generation, and a freshly compiled role is cached only if the
    /// generation has not moved since, so descriptors read before a
    /// concurrent role change (every change bumps the generation after
    /// writing) are never cached as current.
    fn compiled(&self, key: String, g: u64, descs: impl FnOnce() -> Vec<Value>) -> Arc<Role> {
        if let Some((cg, r)) = self.role_cache.lock().get(&key) {
            if *cg == g {
                return r.clone();
            }
        }
        let d = descs();
        let r = Arc::new(Role::compile(&d.iter().collect::<Vec<_>>()));
        let mut c = self.role_cache.lock();
        if self.generation() != g {
            return r;
        }
        if c.len() > 1000 {
            c.clear();
        }
        c.insert(key, (g, r.clone()));
        r
    }

    #[inline]
    fn descriptors_read(&self) {
        #[cfg(test)]
        {
            let hook = self.after_descriptors_hook.lock().clone();
            if let Some(hook) = hook {
                hook();
            }
        }
    }

    /// Role names assigned to the effective user, plus anonymous roles when enabled.
    pub fn assigned_roles(&self, a: &Authn) -> Vec<String> {
        let e = a.effective();
        let mut names = e.roles.clone();
        let cfg = self.cfg.read();
        if !cfg.anonymous_roles.is_empty() && !matches!(e.principal, Principal::ApiKey { .. } | Principal::ServiceAccount { .. }) {
            for r in &cfg.anonymous_roles {
                if !names.contains(r) {
                    names.push(r.clone());
                }
            }
        }
        names
    }

    /// Descriptors (name -> descriptor) of the user's roles (for API key `limited_by`).
    pub fn user_descriptors(&self, a: &Authn) -> Map<String, Value> {
        let mut m = Map::new();
        for n in self.assigned_roles(a) {
            if let Some(d) = self.role_descriptor(&n) {
                m.insert(n, d);
            }
        }
        m
    }

    /// The `limited_by` descriptors of an API key created (or granted) for
    /// `owner` (ES `ApiKeyUserRoleDescriptorResolver`): the owner's role
    /// descriptors, a service account's own descriptor, and none for an
    /// API key subject. A key derived from an API key holds no privileges
    /// (ES retains "the user of an API key authentication has no roles"):
    /// the parent's descriptors alone are not its effective permissions,
    /// which are also bounded by the parent's own `limited_by`.
    pub fn api_key_owner_descriptors(&self, owner: &Authn) -> Map<String, Value> {
        let effective = owner.effective();
        if effective.is_api_key() {
            return Map::new();
        }
        if effective.is_service_account() {
            let mut m = Map::new();
            if let Some(d) = service_account(&effective.username) {
                m.insert(effective.username.clone(), d);
            }
            return m;
        }
        self.user_descriptors(owner)
    }

    /// Workflows an API key's role descriptors restrict it to
    /// (`restriction.workflows`); `None` when unrestricted.
    pub fn workflow_restriction(&self, a: &Authn) -> Option<Vec<String>> {
        let Principal::ApiKey { id, .. } = &a.principal else { return None };
        let key = self.state.read().api_keys.get(id).cloned()?;
        let mut out: Vec<String> = Vec::new();
        for d in key.role_descriptors.values() {
            for w in d.pointer("/restriction/workflows").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                if !out.iter().any(|x| x == w) {
                    out.push(w.to_string());
                }
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// The permissions of the effective subject (the run-as target when
    /// run-as was honoured, as ES authorizes the effective subject).
    pub fn permission(&self, a: &Authn) -> Permission {
        let a = a.effective();
        // Read before any descriptor: see `compiled`.
        let g = self.generation();
        match &a.principal {
            Principal::ApiKey { id, .. } => {
                let key = self.state.read().api_keys.get(id).cloned();
                let Some(key) = key else {
                    return Permission { role: Arc::new(Role::default()), limit: None, effective_roles: vec![], missing_roles: vec![], is_superuser: false };
                };
                self.descriptors_read();
                let limit = self.compiled(format!("apikey-limit:{id}"), g, || key.limited_by.values().cloned().collect());
                let role = if key.role_descriptors.is_empty() {
                    limit.clone()
                } else {
                    self.compiled(format!("apikey:{id}"), g, || key.role_descriptors.values().cloned().collect())
                };
                let is_superuser = key.role_descriptors.is_empty() && key.limited_by.contains_key("superuser");
                Permission { role, limit: Some(limit), effective_roles: vec![], missing_roles: vec![], is_superuser }
            }
            Principal::CrossCluster { key_id, descriptors } => {
                let key = self.state.read().api_keys.get(key_id).cloned();
                let Some(key) = key else {
                    return Permission { role: Arc::new(Role::default()), limit: None, effective_roles: vec![], missing_roles: vec![], is_superuser: false };
                };
                self.descriptors_read();
                let limit = self.compiled(format!("ccx-key:{key_id}"), g, || key.role_descriptors.values().cloned().collect());
                let fingerprint = serde_json::to_string(&**descriptors).unwrap_or_default();
                let descs = descriptors.clone();
                let role = self.compiled(format!("ccx-user:{fingerprint}"), g, || (*descs).clone());
                Permission { role, limit: Some(limit), effective_roles: vec![], missing_roles: vec![], is_superuser: false }
            }
            Principal::ServiceAccount { .. } => {
                let name = a.username.clone();
                let role = self.compiled(format!("svc:{name}"), g, || service_account(&name).into_iter().collect());
                Permission { role, limit: None, effective_roles: vec![], missing_roles: vec![], is_superuser: false }
            }
            _ => {
                let names = self.assigned_roles(a);
                let mut found = Vec::new();
                let mut missing = Vec::new();
                let mut descs = Vec::new();
                for n in &names {
                    match self.role_descriptor(n) {
                        Some(d) => {
                            found.push(n.clone());
                            descs.push(d);
                        }
                        None => missing.push(n.clone()),
                    }
                }
                self.descriptors_read();
                let is_superuser = found.iter().any(|n| n == "superuser");
                let role = self.compiled(format!("roles:{}", found.join("\u{0}")), g, || descs);
                Permission { role, limit: None, effective_roles: found, missing_roles: missing, is_superuser }
            }
        }
    }

    /// Run-as: swap in the effective user named by `es-security-runas-user`.
    /// The authenticating subject's permission decides, an API key's
    /// bounded by its owner's `limited_by` role. ES `supportsRunAs`: no
    /// chained run-as, and service accounts never run as.
    pub fn run_as(&self, mut a: Authn, target: &str) -> Result<Authn, ()> {
        if a.run_as.is_some() || a.is_service_account() {
            return Err(());
        }
        let perm = self.permission(&a);
        if !perm.can_run_as(target) {
            return Err(());
        }
        match self.lookup_user(target) {
            // ES `AuthenticatorChain.finishAuthentication`: a disabled
            // effective user fails the authentication.
            Some(u) if u.enabled => {
                a.run_as = Some(Box::new(u));
                Ok(a)
            }
            _ => Err(()),
        }
    }

    /// Whether `authenticating` could run as `target` but `target` is a
    /// disabled user: ES answers that request with an authentication
    /// failure (401) rather than a run-as denial.
    pub fn run_as_target_disabled(&self, authenticating: &Authn, target: &str) -> bool {
        authenticating.run_as.is_none()
            && !authenticating.is_service_account()
            && self.permission(authenticating).can_run_as(target)
            && self.lookup_user(target).is_some_and(|u| !u.enabled)
    }

    /// ES `TokenService.refreshToken`: take the refresh token `rt` (once),
    /// under one write of the token store, and rebuild the original
    /// token's authentication for the new pair. `Err`: the
    /// `error_description` of the `invalid_grant` answer.
    pub fn take_refresh(&self, rt: &str, client: &Authn) -> Result<Authn, String> {
        let wanted = token_hash(rt);
        let snapshot = {
            let mut st = self.state.write();
            let Some(t) = st.tokens.values_mut().find(|t| t.hashed && t.refresh_token.as_deref() == Some(wanted.as_str())) else {
                return Err("could not refresh the requested token".into());
            };
            if t.refresh_invalidated {
                return Err("token has been invalidated".into());
            }
            // ES `checkClientCanRefresh`: the client that obtained the
            // token (same principal and realm, or the same API key).
            let caller = owner::ResourceOwner::of(client);
            let same = t.client.as_ref().is_some_and(|c| c.username == caller.username && c.realm_name == caller.realm_name && c.api_key_id == caller.api_key_id);
            if !same {
                return Err("tokens must be refreshed by the creating client".into());
            }
            if t.refreshed {
                return Err("token has already been refreshed".into());
            }
            if t.creation + 24 * 3600 * 1000 < now_ms() {
                return Err("refresh token is expired".into());
            }
            t.refreshed = true;
            t.authn.clone()
        };
        // The new token carries the original authentication (its subject,
        // realm and roles), never the refreshing client's or a same-named
        // user's of another realm.
        let mut a = self.authn_from_snapshot(&snapshot)?;
        if snapshot.api_key.is_none() {
            a.auth_type = "realm";
            if let Some(t) = a.run_as.as_deref_mut() {
                t.auth_type = "realm";
            }
        }
        Ok(a)
    }
}

/// Write the security state (password hashes, key and token hashes) readable
/// by the node's user only (mode 0600), as ES protects its data directory.
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut f = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    f.write_all(bytes)?;
    f.sync_all()
}

fn user_authn(username: &str, roles: Vec<String>, full_name: Option<String>, email: Option<String>, metadata: Value, enabled: bool, realm: &str, realm_type: &str) -> Authn {
    Authn {
        username: username.to_string(),
        roles,
        full_name,
        email,
        metadata: if metadata.is_null() { json!({}) } else { metadata },
        enabled,
        realm_name: realm.to_string(),
        realm_type: realm_type.to_string(),
        auth_type: "realm",
        principal: Principal::User,
        run_as: None,
    }
}

/// Hex SHA-256 of a raw OAuth2 token: its id in the token store.
pub fn token_hash(token: &str) -> String {
    hash::sha256(token.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a service account bearer token: `base64("\0\1\0\1" + "ns/svc/name:secret")`.
pub fn parse_service_token(tok: &str) -> Option<(String, String, String)> {
    use base64::Engine;
    let b = base64::engine::general_purpose::STANDARD_NO_PAD.decode(tok.trim_end_matches('=')).ok()?;
    if b.len() < 5 || b[..4] != [0, 1, 0, 1] {
        return None;
    }
    let s = String::from_utf8(b[4..].to_vec()).ok()?;
    let (q, secret) = s.split_once(':')?;
    let mut parts = q.splitn(3, '/');
    let (ns, svc, name) = (parts.next()?, parts.next()?, parts.next()?);
    Some((format!("{ns}/{svc}"), name.to_string(), secret.to_string()))
}

pub fn service_token_value(ns_svc: &str, name: &str, secret: &str) -> String {
    use base64::Engine;
    let mut b = vec![0u8, 1, 0, 1];
    b.extend_from_slice(format!("{ns_svc}/{name}:{secret}").as_bytes());
    base64::engine::general_purpose::STANDARD_NO_PAD.encode(b)
}

/// `ElasticServiceAccounts.ACCOUNTS` (ES 9 dropped `elastic/enterprise-search-server`).
pub const SERVICE_ACCOUNTS: &[&str] = &["elastic/auto-ops", "elastic/fleet-server", "elastic/fleet-server-remote", "elastic/kibana"];

/// Role descriptor of a built-in service account.
pub fn service_account(name: &str) -> Option<Value> {
    let idx = |names: &[&str], privs: &[&str], r: bool| json!({"names": names, "privileges": privs, "allow_restricted_indices": r});
    let fleet_rw = ["read", "write", "monitor", "create_index", "auto_configure", "maintenance"];
    let d = match name {
        "elastic/auto-ops" => json!({
            "cluster": ["monitor", "read_ilm", "read_slm"],
            "indices": [idx(&["*"], &["monitor", "view_index_metadata"], true)],
            "applications": [], "run_as": [], "metadata": {}, "transient_metadata": {"enabled": true}
        }),
        "elastic/fleet-server" => {
            let mut indices = vec![
                idx(&["logs-*", "metrics-*", "traces-*", ".logs-endpoint.diagnostic.collection-*", ".logs-endpoint.action.responses-*", ".logs-endpoint.heartbeat-*"], &["write", "create_index", "auto_configure"], false),
                idx(&["profiling-*"], &["read", "write"], false),
                idx(&["traces-apm.sampled-*"], &["read", "monitor", "maintenance"], false),
                idx(&[".fleet-secrets*"], &["read"], true),
            ];
            for n in [".fleet-actions*", ".fleet-agents*", ".fleet-artifacts*", ".fleet-enrollment-api-keys*", ".fleet-policies*", ".fleet-policies-leader*", ".fleet-servers*", ".fleet-fileds*"] {
                indices.push(idx(&[n], &fleet_rw, true));
            }
            indices.push(idx(&["synthetics-*"], &["read", "write", "create_index", "auto_configure"], false));
            indices.push(idx(&["agentless-*"], &["read", "write", "monitor", "create_index", "auto_configure", "maintenance", "view_index_metadata"], false));
            json!({
                "cluster": ["monitor", "manage_own_api_key", "read_fleet_secrets", "cluster:admin/xpack/connector/*"],
                "indices": indices,
                "applications": [{"application": "kibana-*", "privileges": ["reserved_fleet-setup"], "resources": ["*"]}],
                "run_as": [], "metadata": {}, "transient_metadata": {"enabled": true}
            })
        }
        "elastic/fleet-server-remote" => json!({
            "cluster": ["monitor", "manage_own_api_key"],
            "indices": [idx(&["logs-*", "metrics-*", "traces-*", ".logs-endpoint.diagnostic.collection-*", ".logs-endpoint.action.responses-*", ".logs-endpoint.heartbeat-*"], &["write", "create_index", "auto_configure"], false)],
            "applications": [], "run_as": [], "metadata": {}, "transient_metadata": {"enabled": true}
        }),
        "elastic/kibana" => {
            let mut d = roles::reserved_roles()["kibana_system"].clone();
            d["metadata"] = json!({});
            if let Some(o) = d.as_object_mut() {
                o.remove("description");
            }
            d
        }
        _ => return None,
    };
    Some(d)
}

#[cfg(test)]
mod file_role_format_tests {
    use super::*;
    #[test]
    fn native_yaml_roles_are_loaded_and_take_precedence_over_legacy_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("roles.yml"), "reader:\n  cluster: [monitor]\n  indices:\n    - names: [public-*]\n      privileges: [read]\n").unwrap();
        std::fs::write(dir.path().join("roles.json"), r#"{"reader":{"cluster":["all"]}}"#).unwrap();
        let mut cfg = SecConfig::default();
        cfg.load_files(dir.path());
        assert_eq!(cfg.file_roles["reader"]["cluster"], serde_json::json!(["monitor"]));
        assert_eq!(cfg.file_roles["reader"]["indices"][0]["names"], serde_json::json!(["public-*"]));
        std::fs::remove_file(dir.path().join("roles.yml")).unwrap();
        let mut legacy = SecConfig::default();
        legacy.load_files(dir.path());
        assert_eq!(legacy.file_roles["reader"]["cluster"], serde_json::json!(["all"]));
    }
}

#[cfg(test)]
mod cluster_security_tests {
    use super::*;

    /// ECK: `reserved_realm.enabled: false` and `elastic` in the file realm.
    #[test]
    fn reserved_users_come_from_the_file_realm_when_the_reserved_realm_is_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let hash = Hasher::Bcrypt(10).hash("file-pw");
        std::fs::write(dir.path().join("users"), format!("elastic:{hash}\nelastic-internal:{hash}\n")).unwrap();
        std::fs::write(dir.path().join("users_roles"), "superuser:elastic,elastic-internal\n").unwrap();
        let security = Security::open(dir.path());
        let mut cfg = SecConfig { enabled: true, bootstrap_password: Some("bootstrap".into()), ..Default::default() };
        cfg.load_files(dir.path());
        security.configure(cfg.clone());
        assert!(security.authenticate_password("elastic", "file-pw").is_err(), "the reserved realm answers for elastic");
        assert!(security.authenticate_password("elastic", "bootstrap").is_ok());
        cfg.reserved_realm_enabled = false;
        security.configure(cfg);
        assert!(security.authenticate_password("elastic", "bootstrap").is_err(), "no reserved realm");
        let elastic = security.authenticate_password("elastic", "file-pw").unwrap_or_else(|_| panic!("file realm elastic"));
        assert_eq!(elastic.roles, vec!["superuser".to_string()]);
        assert!(security.authenticate_password("elastic-internal", "file-pw").is_ok());
        assert!(security.lookup_user("kibana_system").is_none(), "no reserved users without the reserved realm");
    }

    #[test]
    fn api_key_propagation_and_revocation_invalidate_verified_cache() {
        let source_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let source = Security::open(source_dir.path());
        let peer = Security::open(peer_dir.path());
        source.configure(SecConfig { enabled:true, bootstrap_password:Some("bootstrap".into()), ..Default::default() });
        peer.configure(SecConfig { enabled:true, file_user_roles:BTreeMap::from([("peer-file-user".into(), vec!["superuser".into()])]), ..Default::default() });
        let owner = source.authenticate_password("elastic", "bootstrap").unwrap_or_else(|_| panic!("bootstrap authentication failed"));
        let key = source.create_api_key(&owner, Map::new(), "key".into(), None, Map::new(), json!({})).unwrap();
        use base64::Engine;
        let credentials = base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", key.id, key.hash));
        assert!(peer.authenticate_api_key(&credentials).is_err());
        assert!(peer.restore_cluster_state(&source.cluster_state()).unwrap());
        assert!(peer.authenticate_api_key(&credentials).is_ok());
        assert!(!peer.restore_cluster_state(&source.cluster_state()).unwrap());
        let old_snapshot = source.cluster_state();
        source.state.write().api_keys.get_mut(&key.id).unwrap().hash = Hasher::Ssha256.hash("rotated");
        source.save().unwrap();
        assert!(peer.restore_cluster_state(&source.cluster_state()).unwrap());
        assert!(peer.authenticate_api_key(&credentials).is_err(), "verified credential cache survived state replacement");
        let rotated = base64::engine::general_purpose::STANDARD.encode(format!("{}:rotated", key.id));
        assert!(peer.authenticate_api_key(&rotated).is_ok());
        source.state.write().api_keys.get_mut(&key.id).unwrap().invalidated = true;
        source.save().unwrap();
        assert!(peer.restore_cluster_state(&source.cluster_state()).unwrap());
        assert!(peer.authenticate_api_key(&rotated).is_err());
        assert!(!peer.restore_cluster_state(&old_snapshot).unwrap());
        assert!(peer.authenticate_api_key(&credentials).is_err());
        assert!(peer.cfg.read().file_user_roles.contains_key("peer-file-user"));
        let reopened = Security::open(peer_dir.path());
        assert!(reopened.state.read().api_keys[&key.id].invalidated);
        source.reset().unwrap();
        assert!(peer.restore_cluster_state(&source.cluster_state()).unwrap());
        assert!(peer.state.read().api_keys.is_empty());
        assert!(!peer.restore_cluster_state(&old_snapshot).unwrap());
    }
}

#[cfg(test)]
mod api_key_ceiling_and_role_cache_tests {
    use super::*;
    use base64::Engine;
    use std::sync::Barrier;

    const PUT_USER: &str = "cluster:admin/xpack/security/user/put";
    const HEALTH: &str = "cluster:monitor/health";

    fn secured() -> (tempfile::TempDir, Arc<Security>) {
        let dir = tempfile::tempdir().unwrap();
        let security = Arc::new(Security::open(dir.path()));
        security.configure(SecConfig { enabled: true, token_enabled: true, ..Default::default() });
        (dir, security)
    }

    fn put_role(security: &Security, name: &str, descriptor: Value) {
        let role = roles::parse_role(name, &descriptor, false).unwrap();
        security.state.write().roles.insert(name.into(), role);
        security.save().unwrap();
    }

    fn put_user(security: &Security, name: &str, roles: &[&str]) {
        let user = NativeUser { password_hash: Hasher::Ssha256.hash("pw"), roles: roles.iter().map(|r| r.to_string()).collect(), full_name: None, email: None, metadata: json!({}), enabled: true };
        security.state.write().users.insert(name.into(), user);
        security.save().unwrap();
    }

    /// Create a key owned by `owner` with `descriptors` (limited by the
    /// owner's roles, as `do_create_key`); its authentication.
    fn key(security: &Security, owner: &str, descriptors: Value) -> Authn {
        let owner = security.authenticate_password(owner, "pw").unwrap_or_else(|_| panic!("owner authenticates"));
        let mut rds = Map::new();
        for (name, d) in descriptors.as_object().unwrap() {
            rds.insert(name.clone(), roles::parse_role(name, d, true).unwrap());
        }
        let limited = security.user_descriptors(&owner);
        let key = security.create_api_key(&owner, limited, "k".into(), None, rds, json!({})).unwrap();
        let credential = base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", key.id, key.hash));
        security.authenticate_api_key(&credential).unwrap()
    }

    fn setup(security: &Security) {
        put_role(security, "owner", json!({"cluster": ["monitor", "manage_own_api_key", "manage_token"]}));
        put_role(security, "runner", json!({"run_as": ["target"]}));
        put_role(security, "everything", json!({"cluster": ["all"], "indices": [{"names": ["*"], "privileges": ["all"]}]}));
        put_user(security, "owner", &["owner"]);
        put_user(security, "runner", &["owner", "runner"]);
        put_user(security, "target", &["everything"]);
    }

    #[test]
    fn api_key_run_as_is_bounded_by_the_owner_role() {
        let (_dir, security) = secured();
        setup(&security);
        // The key asks for run_as, its owner has none: denied.
        let limited = key(&security, "owner", json!({"r": {"run_as": ["target"], "cluster": ["all"]}}));
        assert!(security.run_as(limited.clone(), "target").is_err(), "an API key ran as a user its owner cannot run as");
        assert!(!security.permission(&limited).check_cluster(PUT_USER));
        // Owner and key both allow it: the target's permissions apply.
        let allowed = key(&security, "runner", json!({"r": {"run_as": ["target"]}}));
        let ran = security.run_as(allowed.clone(), "target").unwrap_or_else(|_| panic!("run-as within the owner ceiling"));
        assert_eq!(ran.effective().username, "target");
        assert!(security.permission(&ran).check_cluster(PUT_USER));
        // The owner allows it but the key's own descriptors do not.
        let narrowed = key(&security, "runner", json!({"r": {"cluster": ["monitor"]}}));
        assert!(security.run_as(narrowed, "target").is_err());
        // No chained run-as.
        assert!(security.run_as(ran, "target").is_err());
    }

    #[test]
    fn tokens_issued_to_an_api_key_keep_its_limits() {
        let (_dir, security) = secured();
        setup(&security);
        // Key descriptors broader than the owner: the owner bounds them.
        let limited = key(&security, "owner", json!({"r": {"cluster": ["all"], "run_as": ["target"]}}));
        assert!(!security.permission(&limited).check_cluster(PUT_USER));
        let doc = security.create_token(&limited, false, &limited).unwrap();
        assert_eq!(doc.authn.api_key.as_ref().map(|k| k.id.as_str()), limited.api_key_id());
        let bearer = security.authenticate(Credentials::Bearer(doc.access_token.clone())).unwrap_or_else(|_| panic!("token authenticates"));
        assert!(bearer.is_api_key(), "the token lost the API key subject");
        assert_eq!(bearer.auth_type, "token");
        assert_eq!(bearer.api_key_id(), limited.api_key_id());
        let perm = security.permission(&bearer);
        assert!(perm.limit.is_some(), "the token lost the owner's limited_by ceiling");
        assert!(!perm.check_cluster(PUT_USER), "the token gained a privilege its owner lacks");
        assert!(perm.check_cluster(HEALTH));
        assert!(security.run_as(bearer, "target").is_err(), "the token ran as a user the owner cannot run as");

        // A run-as token keeps the API key as the authenticating subject and
        // is re-checked on use.
        let allowed = key(&security, "runner", json!({"r": {"run_as": ["target"]}}));
        let ran = security.run_as(allowed.clone(), "target").unwrap_or_else(|_| panic!("run-as"));
        let doc = security.create_token(&ran, false, &ran).unwrap();
        let bearer = security.authenticate(Credentials::Bearer(doc.access_token.clone())).unwrap_or_else(|_| panic!("run-as token authenticates"));
        assert!(bearer.is_api_key());
        assert_eq!(bearer.effective().username, "target");
        // The owner loses run_as (the key's limited_by changes): the token stops working.
        let id = allowed.api_key_id().unwrap().to_string();
        security.state.write().api_keys.get_mut(&id).unwrap().limited_by.remove("runner");
        security.save().unwrap();
        assert!(security.authenticate(Credentials::Bearer(doc.access_token.clone())).is_err(), "run-as token survived the loss of the run_as privilege");
        // An invalidated key ends its tokens.
        let other = key(&security, "owner", json!({"r": {"cluster": ["monitor"]}}));
        let doc = security.create_token(&other, false, &other).unwrap();
        assert!(security.authenticate(Credentials::Bearer(doc.access_token.clone())).is_ok());
        security.state.write().api_keys.get_mut(other.api_key_id().unwrap()).unwrap().invalidated = true;
        security.save().unwrap();
        assert!(security.authenticate(Credentials::Bearer(doc.access_token)).is_err());
        // Tokens without the new fields (written before) still read as users.
        let legacy: AuthnSnapshot = serde_json::from_value(json!({"username": "owner", "roles": ["owner"], "full_name": null, "email": null, "metadata": {}, "realm_name": "default_native", "realm_type": "native"})).unwrap();
        assert!(legacy.api_key.is_none() && legacy.run_as.is_none());
    }

    #[test]
    fn a_key_derived_from_an_api_key_has_no_privileges() {
        let (_dir, security) = secured();
        setup(&security);
        // A parent key whose descriptors ask for more than its owner holds:
        // its effective permissions stay under the owner.
        let parent = key(&security, "owner", json!({"r": {"cluster": ["all"], "run_as": ["*"], "indices": [{"names": ["*"], "privileges": ["all"]}]}}));
        assert!(security.run_as(parent.clone(), "target").is_err());
        assert!(!security.permission(&parent).check_cluster(PUT_USER));
        // A key created by that key (empty descriptors, as derived keys
        // must be) is limited by what the parent's authentication resolves
        // to: no roles, as ES's ApiKeyUserRoleDescriptorResolver.
        let limited = security.api_key_owner_descriptors(&parent);
        assert!(limited.is_empty(), "a derived key inherited descriptors: {limited:?}");
        let mut rds = Map::new();
        rds.insert("d".into(), roles::parse_role("d", &json!({"run_as": ["target"]}), true).unwrap());
        let derived = security.create_api_key(&parent, limited, "derived".into(), None, rds, json!({})).unwrap();
        let credential = base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", derived.id, derived.hash));
        let derived = security.authenticate_api_key(&credential).unwrap();
        let perm = security.permission(&derived);
        assert!(!perm.check_cluster(PUT_USER), "a derived key escaped its parent's owner ceiling");
        assert!(!perm.check_cluster(HEALTH), "a derived key holds privileges");
        assert!(!perm.check_index("indices:data/read/search", "anything"));
        assert!(!perm.is_superuser);
        assert!(security.run_as(derived, "target").is_err(), "a derived key ran as a user its parent's owner cannot run as");
        // Users and service accounts keep their own descriptors.
        let owner = security.authenticate_password("owner", "pw").unwrap_or_else(|_| panic!("owner"));
        assert!(security.api_key_owner_descriptors(&owner).contains_key("owner"));
    }

    #[test]
    fn run_as_a_disabled_user_fails() {
        let (_dir, security) = secured();
        setup(&security);
        let runner = security.authenticate_password("runner", "pw").unwrap_or_else(|_| panic!("runner"));
        assert!(security.run_as(runner.clone(), "target").is_ok());
        assert!(!security.run_as_target_disabled(&runner, "target"));
        security.state.write().users.get_mut("target").unwrap().enabled = false;
        security.save().unwrap();
        assert!(security.run_as(runner.clone(), "target").is_err(), "ran as a disabled user");
        assert!(security.run_as_target_disabled(&runner, "target"));
        // A subject that may not run as the user gets the run-as denial instead.
        let owner = security.authenticate_password("owner", "pw").unwrap_or_else(|_| panic!("owner"));
        assert!(!security.run_as_target_disabled(&owner, "target"));
    }

    #[test]
    fn a_refresh_keeps_the_original_authentication_and_happens_once() {
        let (_dir, security) = secured();
        setup(&security);
        let owner = security.authenticate_password("owner", "pw").unwrap_or_else(|_| panic!("owner"));
        // Obtained by a client (Kibana-style) on the user's behalf.
        let client = security.authenticate_password("runner", "pw").unwrap_or_else(|_| panic!("client"));
        let doc = security.create_token(&owner, true, &client).unwrap();
        let rt = doc.refresh_token.clone().unwrap();
        // Only the stored hashes are kept.
        assert!(security.state.read().tokens.values().all(|t| t.hashed && t.access_token != doc.access_token && t.refresh_token.as_deref() != Some(rt.as_str())));
        // Another client may not refresh it.
        let other_client = security.authenticate_password("target", "pw").unwrap_or_else(|_| panic!("other"));
        assert_eq!(security.take_refresh(&rt, &other_client).err().as_deref(), Some("tokens must be refreshed by the creating client"));
        // The user is deleted and a same-named file user appears with more
        // roles: the refreshed token keeps the original subject and roles.
        security.state.write().users.remove("owner");
        security.save().unwrap();
        let mut cfg = security.cfg.read().clone();
        cfg.file_users.insert("owner".into(), Hasher::Ssha256.hash("pw"));
        cfg.file_user_roles.insert("owner".into(), vec!["everything".into()]);
        security.configure(cfg);
        let refreshed = security.take_refresh(&rt, &client).unwrap_or_else(|why| panic!("refresh: {why}"));
        assert_eq!(refreshed.username, "owner");
        assert_eq!(refreshed.realm_type, "native");
        assert_eq!(refreshed.roles, vec!["owner".to_string()]);
        assert!(!security.permission(&refreshed).check_cluster(PUT_USER), "the refresh picked up another realm's user");
        assert_eq!(security.take_refresh(&rt, &client).err().as_deref(), Some("token has already been refreshed"));
        assert!(security.take_refresh("no-such-refresh-token", &client).is_err());
    }

    #[test]
    fn token_ids_are_hashed_at_rest_and_the_state_file_is_private() {
        let (dir, security) = secured();
        setup(&security);
        let owner = security.authenticate_password("owner", "pw").unwrap_or_else(|_| panic!("owner"));
        let doc = security.create_token(&owner, true, &owner).unwrap();
        let path = dir.path().join("_state/security.json");
        let text = String::from_utf8_lossy(&std::fs::read(&path).unwrap()).to_string();
        assert!(!text.contains(&doc.access_token), "the access token is stored in plaintext");
        assert!(!text.contains(doc.refresh_token.as_deref().unwrap()), "the refresh token is stored in plaintext");
        assert!(security.authenticate(Credentials::Bearer(doc.access_token.clone())).is_ok());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // A document from before hashing (plaintext id) never authenticates
        // and is purged when the state is next loaded.
        let mut legacy = security.state.read().tokens.values().next().unwrap().clone();
        legacy.access_token = "legacy-plain".into();
        legacy.hashed = false;
        security.state.write().tokens.insert("legacy-plain".into(), legacy);
        security.save().unwrap();
        assert!(security.authenticate(Credentials::Bearer("legacy-plain".into())).is_err());
        let reopened = Security::open(dir.path());
        assert!(reopened.state.read().tokens.values().all(|t| t.hashed));
        assert_eq!(reopened.state.read().tokens.len(), 1);
    }

    /// Interleave a permission lookup (paused after it read the descriptors)
    /// with a downgrade; `downgrade` runs while the lookup is paused.
    fn race(security: &Arc<Security>, subject: &Authn, downgrade: impl FnOnce()) -> bool {
        let barrier = Arc::new(Barrier::new(2));
        let armed = Arc::new(AtomicBool::new(true));
        let (b, a) = (barrier.clone(), armed.clone());
        *security.after_descriptors_hook.lock() = Some(Arc::new(move || {
            if a.swap(false, Ordering::SeqCst) {
                b.wait();
                b.wait();
            }
        }));
        let (s, who) = (security.clone(), subject.clone());
        let lookup = std::thread::spawn(move || s.permission(&who).check_cluster(PUT_USER));
        barrier.wait(); // the lookup holds the old descriptors
        downgrade();
        barrier.wait(); // let it compile and try to cache them
        let in_flight = lookup.join().unwrap();
        *security.after_descriptors_hook.lock() = None;
        in_flight
    }

    #[test]
    fn a_role_downgrade_racing_a_lookup_is_not_cached() {
        let (_dir, security) = secured();
        put_role(&security, "r", json!({"cluster": ["all"]}));
        put_user(&security, "u", &["r"]);
        let user = security.lookup_user("u").unwrap();
        assert!(security.permission(&user).check_cluster(PUT_USER));
        // Start from a cold cache so the lookup has to compile and insert.
        security.clear_cache("roles", &[], &[]).unwrap();
        let in_flight = race(&security, &user, || put_role(&security, "r", json!({"cluster": ["monitor"]})));
        assert!(in_flight, "the in-flight lookup read the old descriptors (interleaving happened)");
        for _ in 0..3 {
            let perm = security.permission(&user);
            assert!(!perm.check_cluster(PUT_USER), "the downgraded role's old permissions were cached");
            assert!(perm.check_cluster(HEALTH));
        }
    }

    #[test]
    fn an_api_key_downgrade_racing_a_lookup_is_not_cached() {
        let (_dir, security) = secured();
        put_role(&security, "boss", json!({"cluster": ["all"]}));
        put_user(&security, "boss", &["boss"]);
        let k = key(&security, "boss", json!({"r": {"cluster": ["all"]}}));
        assert!(security.permission(&k).check_cluster(PUT_USER));
        security.clear_cache("api_keys", &[], &[]).unwrap();
        let id = k.api_key_id().unwrap().to_string();
        let in_flight = race(&security, &k, || {
            let narrowed = roles::parse_role("r", &json!({"cluster": ["monitor"]}), true).unwrap();
            security.state.write().api_keys.get_mut(&id).unwrap().role_descriptors = Map::from_iter([("r".to_string(), narrowed)]);
            security.save().unwrap();
        });
        assert!(in_flight);
        let perm = security.permission(&k);
        assert!(!perm.check_cluster(PUT_USER), "the narrowed key's old descriptors were cached");
        assert!(perm.check_cluster(HEALTH));
    }

    #[test]
    fn clear_cache_invalidates_each_cache() {
        let (_dir, security) = secured();
        setup(&security);
        put_role(&security, "r", json!({"cluster": ["monitor"]}));
        put_user(&security, "u", &["r"]);
        let u = security.authenticate_password("u", "pw").unwrap_or_else(|_| panic!("u"));
        security.permission(&u);
        let owner = security.authenticate_password("owner", "pw").unwrap_or_else(|_| panic!("owner"));
        security.permission(&owner);
        let k = key(&security, "owner", json!({"r": {"cluster": ["monitor"]}}));
        let id = k.api_key_id().unwrap().to_string();
        security.permission(&k);
        let has = |roles: bool, entry: &str| {
            let (r, v) = security.cache_keys();
            (if roles { r } else { v }).iter().any(|x| x == entry)
        };
        assert!(has(true, "roles:r") && has(true, "roles:owner") && has(true, &format!("apikey:{id}")) && has(true, &format!("apikey-limit:{id}")));
        assert!(has(false, "basic\u{0}u") && has(false, &format!("apikey\u{0}{id}")));

        security.clear_cache("roles", &["other".into()], &[]).unwrap();
        assert!(has(true, "roles:r"));
        security.clear_cache("roles", &["r".into()], &[]).unwrap();
        assert!(!has(true, "roles:r") && has(true, "roles:owner"));

        security.clear_cache("api_keys", &[id.clone()], &[]).unwrap();
        assert!(!has(true, &format!("apikey:{id}")) && !has(true, &format!("apikey-limit:{id}")) && !has(false, &format!("apikey\u{0}{id}")));
        assert!(has(true, "roles:owner") && has(false, "basic\u{0}u"));

        security.clear_cache("realms", &["default_native".into()], &["owner".into()]).unwrap();
        assert!(has(false, "basic\u{0}u") && !has(false, "basic\u{0}owner"));
        security.clear_cache("realms", &["default_native".into()], &[]).unwrap();
        assert!(!has(false, "basic\u{0}u"));

        security.clear_cache("privileges", &["app".into()], &[]).unwrap();
        assert!(security.cache_keys().0.is_empty());

        let tokens = BTreeMap::from([("t1".to_string(), ServiceToken { hash: Hasher::Ssha256.hash("s1"), creation: 0 }), ("t2".to_string(), ServiceToken { hash: Hasher::Ssha256.hash("s2"), creation: 0 })]);
        security.state.write().service_tokens.insert("elastic/fleet-server".into(), tokens);
        security.save().unwrap();
        for (name, secret) in [("t1", "s1"), ("t2", "s2")] {
            assert!(security.authenticate(Credentials::Bearer(service_token_value("elastic/fleet-server", name, secret))).is_ok());
        }
        security.clear_cache("service_tokens", &["elastic/fleet-server/t1".into()], &[]).unwrap();
        assert!(!has(false, "svc\u{0}elastic/fleet-server/t1") && has(false, "svc\u{0}elastic/fleet-server/t2"));
        security.clear_cache("service_tokens", &["elastic/fleet-server/*".into()], &[]).unwrap();
        assert!(!has(false, "svc\u{0}elastic/fleet-server/t2"));
        assert!(security.clear_cache("bogus", &[], &[]).is_err());
    }
}

#[cfg(test)]
mod nonblocking_authentication_tests {
    use super::*;

    /// Production review #17: the networking thread never runs uncached
    /// bcrypt / PBKDF2 verification; it learns that the verdict needs the
    /// crypto pool. Only successes are cached, so a failed password is never
    /// decided (or cached) on the networking thread.
    #[test]
    fn expensive_verification_is_deferred_and_only_successes_are_cached() {
        let dir = tempfile::tempdir().unwrap();
        let bcrypt = Hasher::Bcrypt(10).hash("bcrypt-pw");
        let pbkdf2 = Hasher::Pbkdf2(10000).hash("pbkdf2-pw");
        let cheap = Hasher::Ssha256.hash("cheap-pw");
        std::fs::write(dir.path().join("users"), format!("b:{bcrypt}\np:{pbkdf2}\nc:{cheap}\n")).unwrap();
        std::fs::write(dir.path().join("users_roles"), "superuser:b,p,c\n").unwrap();
        let security = Security::open(dir.path());
        let mut cfg = SecConfig { enabled: true, bootstrap_password: Some("bootstrap".into()), ..Default::default() };
        cfg.load_files(dir.path());
        security.configure(cfg);
        let basic = |u: &str, p: &str| Credentials::Basic(u.into(), p.into());
        for (user, pw) in [("b", "bcrypt-pw"), ("p", "pbkdf2-pw")] {
            let started = std::time::Instant::now();
            assert!(security.authenticate_nonblocking(&basic(user, pw)).is_none(), "{user}: uncached expensive success deferred");
            assert!(security.authenticate_nonblocking(&basic(user, "wrong")).is_none(), "{user}: expensive failure deferred");
            assert!(started.elapsed() < std::time::Duration::from_millis(20), "{user}: no hashing on the networking thread");
            // On the crypto pool: the real verdicts.
            assert!(security.authenticate(basic(user, pw)).is_ok());
            assert!(security.authenticate(basic(user, "wrong")).is_err());
            // A success is cached and then decided without the pool...
            assert!(matches!(security.authenticate_nonblocking(&basic(user, pw)), Some(Ok(_))), "{user}: cached success");
            // ... a failure never is.
            assert!(security.authenticate_nonblocking(&basic(user, "wrong")).is_none(), "{user}: failures are not cached");
        }
        // Cheap cases are decided inline.
        assert!(matches!(security.authenticate_nonblocking(&basic("c", "cheap-pw")), Some(Ok(_))));
        assert!(matches!(security.authenticate_nonblocking(&basic("c", "nope")), Some(Err(_))));
        assert!(matches!(security.authenticate_nonblocking(&basic("nobody", "x")), Some(Err(_))));
        assert!(matches!(security.authenticate_nonblocking(&basic("elastic", "bootstrap")), Some(Ok(_))));
        assert!(matches!(security.authenticate_nonblocking(&Credentials::None), Some(Err(AuthFail::Missing))));
        // Deferral is scoped to the call: plain authentication still verifies.
        assert!(security.authenticate(basic("b", "wrong")).is_err());
        assert!(security.authenticate(basic("p", "pbkdf2-pw")).is_ok());
    }
}
