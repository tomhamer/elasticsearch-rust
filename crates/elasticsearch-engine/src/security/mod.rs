//! Node security (`xpack.security.enabled`, default true for a node built from settings, as in ES 8+):
//! reserved / file / native realms, API keys, OAuth2-style tokens, service
//! accounts, roles (reserved, file, native), role mappings and application
//! privileges. State persists in `_state/security.json`.
//!
//! The HTTP layer (`elasticsearch-http::security`) authenticates every
//! request through [`Security::authenticate`] and authorizes it against
//! [`Security::permission`]; with security disabled nothing here runs.

pub mod hash;
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

fn parse_bool(s: &str) -> bool {
    matches!(s.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "on")
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
            Some(v) => parse_bool(&v),
            None => std::env::var("ES_SECURITY_ENABLED").ok().map(|v| parse_bool(&v)).unwrap_or(NODE_SECURITY_ENABLED_DEFAULT),
        };
        if let Some(v) = value("xpack.security.http.ssl.enabled") {
            c.http_ssl = parse_bool(&v);
            if !settings.contains_key("xpack.security.authc.token.enabled") { c.token_enabled = c.http_ssl; }
        }
        if let Some(v) = value("xpack.security.authc.token.enabled") { c.token_enabled = parse_bool(&v); }
        if let Some(v) = value("xpack.security.authc.api_key.enabled") { c.api_key_enabled = parse_bool(&v); }
        if let Some(v) = value("xpack.security.authc.token.timeout") { c.token_timeout_ms = parse_time_ms(&v, "xpack.security.authc.token.timeout").map_err(|e| e.reason)?; }
        if let Some(v) = value("xpack.security.authc.anonymous.username") { c.anonymous_username = v; }
        if let Some(v) = settings.get("xpack.security.authc.anonymous.roles") {
            c.anonymous_roles = match v {
                Value::Array(a) => a.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
                Value::String(s) => s.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect(),
                _ => return Err("xpack.security.authc.anonymous.roles must be a list of roles".into()),
            };
        }
        if let Some(v) = value("xpack.security.authc.anonymous.authz_exception") { c.anonymous_authz_exception = parse_bool(&v); }
        if let Some(v) = value("xpack.security.authc.password_hashing.algorithm") { c.password_hashing = Hasher::resolve(&v)?; }
        if let Some(v) = value("bootstrap.password") { c.bootstrap_password = Some(v); }
        if let Some(v) = value("xpack.security.authc.reserved_realm.enabled") { c.reserved_realm_enabled = parse_bool(&v); }
        for key in settings.keys() {
            if let Some(rest) = key.strip_prefix("xpack.security.authc.realms.") {
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
        c.enabled = std::env::var("ES_SECURITY_ENABLED").ok().map(|v| parse_bool(&v)).or_else(|| setting(&yml, "xpack.security.enabled").map(|v| parse_bool(&v))).unwrap_or(false);
        if let Some(u) = setting(&yml, "xpack.security.authc.anonymous.username") {
            c.anonymous_username = u;
        }
        if let Some(r) = setting(&yml, "xpack.security.authc.anonymous.roles") {
            c.anonymous_roles = r.trim_matches(|ch| ch == '[' || ch == ']').split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect();
        }
        if let Some(v) = setting(&yml, "xpack.security.authc.anonymous.authz_exception") {
            c.anonymous_authz_exception = parse_bool(&v);
        }
        if let Some(v) = setting(&yml, "xpack.security.authc.password_hashing.algorithm") {
            if let Ok(h) = Hasher::resolve(&v) {
                c.password_hashing = h;
            }
        }
        c.http_ssl = setting(&yml, "xpack.security.http.ssl.enabled").map(|v| parse_bool(&v)).unwrap_or(false);
        c.token_enabled = setting(&yml, "xpack.security.authc.token.enabled").map(|v| parse_bool(&v)).unwrap_or(c.http_ssl);
        if let Some(v) = setting(&yml, "xpack.security.authc.api_key.enabled") {
            c.api_key_enabled = parse_bool(&v);
        }
        if let Some(v) = setting(&yml, "xpack.security.authc.token.timeout") {
            if let Ok(ms) = parse_time_ms(&v, "xpack.security.authc.token.timeout") {
                c.token_timeout_ms = ms;
            }
        }
        c.bootstrap_password = std::env::var("ELASTIC_PASSWORD").ok().or_else(|| std::env::var("ES_BOOTSTRAP_PASSWORD").ok()).or_else(|| setting(&yml, "bootstrap.password"));
        if let Some(v) = setting(&yml, "xpack.security.authc.reserved_realm.enabled") {
            c.reserved_realm_enabled = parse_bool(&v);
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenDoc {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthnSnapshot {
    pub username: String,
    pub roles: Vec<String>,
    pub full_name: Option<String>,
    pub email: Option<String>,
    pub metadata: Value,
    pub realm_name: String,
    pub realm_type: String,
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

    fn snapshot(&self) -> AuthnSnapshot {
        let e = self.effective();
        AuthnSnapshot {
            username: e.username.clone(),
            roles: e.roles.clone(),
            full_name: e.full_name.clone(),
            email: e.email.clone(),
            metadata: e.metadata.clone(),
            realm_name: e.realm_name.clone(),
            realm_type: e.realm_type.clone(),
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
        self.role.check_index(action, name) && self.limit.as_ref().map(|l| l.check_index(action, name)).unwrap_or(true)
    }
    pub fn has_cluster_privilege(&self, name: &str) -> bool {
        self.role.has_cluster_privilege(name) && self.limit.as_ref().map(|l| l.has_cluster_privilege(name)).unwrap_or(true)
    }
    pub fn has_index_privilege(&self, index: &str, name: &str) -> bool {
        self.role.has_index_privilege(index, name) && self.limit.as_ref().map(|l| l.has_index_privilege(index, name)).unwrap_or(true)
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

pub struct Security {
    enabled: AtomicBool,
    pub cfg: RwLock<SecConfig>,
    pub state: RwLock<SecState>,
    path: PathBuf,
    persist_lock: Mutex<()>,
    native_snapshot_seen: AtomicBool,
    /// Bumped on every change; invalidates caches.
    generation: AtomicU64,
    /// sha256(scheme:principal:secret) -> generation at verification time.
    verified: Mutex<HashMap<[u8; 32], u64>>,
    /// role-set key -> compiled role.
    role_cache: Mutex<HashMap<String, (u64, Arc<Role>)>>,
}

impl Security {
    /// Open with settings from the environment.
    pub fn open(data: &Path) -> Security {
        let cfg = SecConfig::from_env();
        let path = data.join("_state/security.json");
        let state: SecState = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
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
        self.generation.fetch_add(1, Ordering::Relaxed);
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
        self.generation.load(Ordering::Relaxed)
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
        std::fs::write(&tmp, bytes)?;
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
        std::fs::write(&temporary, serde_json::to_vec(&restored)?)?;
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
        self.verified.lock().get(key).map(|v| *v == g).unwrap_or(false)
    }

    fn remember(&self, key: [u8; 32]) {
        let mut v = self.verified.lock();
        if v.len() > 10_000 {
            v.clear();
        }
        v.insert(key, self.generation());
    }

    fn verify_cached(&self, scheme: &str, principal: &str, secret: &str, hash: &str) -> bool {
        let key = hash::sha256(format!("{scheme}\u{0}{principal}\u{0}{secret}\u{0}{hash}").as_bytes());
        if self.check_cached(&key) {
            return true;
        }
        if hash::verify(secret, hash) {
            self.remember(key);
            true
        } else {
            false
        }
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
                None => user == "elastic" && self.cfg.read().bootstrap_password.as_deref().map(|b| !b.is_empty() && b == pass).unwrap_or(false),
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
        let doc = self.state.read().tokens.get(tok).cloned();
        let Some(doc) = doc else { return Err(AuthFail::Other(vec!["OAuth2 token: token malformed".into()])) };
        if doc.invalidated {
            return Err(AuthFail::Other(vec!["OAuth2 token: token expired".into()]));
        }
        if doc.expires <= now_ms() {
            return Err(AuthFail::TokenExpired);
        }
        let s = doc.authn;
        let mut a = user_authn(&s.username, s.roles, s.full_name, s.email, s.metadata, true, &s.realm_name, &s.realm_type);
        a.auth_type = "token";
        Ok(a)
    }

    /// Issue an access (+ optional refresh) token for `authn`.
    pub fn create_token(&self, authn: &Authn, with_refresh: bool) -> Result<TokenDoc, EsError> {
        let timeout = self.cfg.read().token_timeout_ms;
        let now = now_ms();
        let doc = TokenDoc {
            access_token: format!("{}", hash::random_b64url(48)),
            refresh_token: if with_refresh { Some(hash::random_b64url(32)) } else { None },
            creation: now,
            expires: now + timeout,
            invalidated: false,
            refresh_invalidated: false,
            refreshed: false,
            authn: authn.snapshot(),
        };
        let mut st = self.state.write();
        // Drop long-dead tokens.
        let horizon = now - 24 * 3600 * 1000;
        st.tokens.retain(|_, t| t.creation > horizon);
        st.tokens.insert(doc.access_token.clone(), doc.clone());
        drop(st);
        self.save()?;
        Ok(doc)
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

    fn compiled(&self, key: String, descs: impl FnOnce() -> Vec<Value>) -> Arc<Role> {
        let g = self.generation();
        if let Some((cg, r)) = self.role_cache.lock().get(&key) {
            if *cg == g {
                return r.clone();
            }
        }
        let d = descs();
        let r = Arc::new(Role::compile(&d.iter().collect::<Vec<_>>()));
        let mut c = self.role_cache.lock();
        if c.len() > 1000 {
            c.clear();
        }
        c.insert(key, (g, r.clone()));
        r
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

    pub fn permission(&self, a: &Authn) -> Permission {
        match &a.principal {
            Principal::ApiKey { id, .. } => {
                let key = self.state.read().api_keys.get(id).cloned();
                let Some(key) = key else {
                    return Permission { role: Arc::new(Role::default()), limit: None, effective_roles: vec![], missing_roles: vec![], is_superuser: false };
                };
                let limit = self.compiled(format!("apikey-limit:{id}"), || key.limited_by.values().cloned().collect());
                let role = if key.role_descriptors.is_empty() {
                    limit.clone()
                } else {
                    self.compiled(format!("apikey:{id}"), || key.role_descriptors.values().cloned().collect())
                };
                let is_superuser = key.role_descriptors.is_empty() && key.limited_by.contains_key("superuser");
                Permission { role, limit: Some(limit), effective_roles: vec![], missing_roles: vec![], is_superuser }
            }
            Principal::ServiceAccount { .. } => {
                let name = a.username.clone();
                let role = self.compiled(format!("svc:{name}"), || service_account(&name).into_iter().collect());
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
                let is_superuser = found.iter().any(|n| n == "superuser");
                let role = self.compiled(format!("roles:{}", found.join("\u{0}")), || descs);
                Permission { role, limit: None, effective_roles: found, missing_roles: missing, is_superuser }
            }
        }
    }

    /// Run-as: swap in the effective user named by `es-security-runas-user`.
    pub fn run_as(&self, mut a: Authn, target: &str) -> Result<Authn, ()> {
        let perm = self.permission(&a);
        if !perm.role.can_run_as(target) {
            return Err(());
        }
        match self.lookup_user(target) {
            Some(u) => {
                a.run_as = Some(Box::new(u));
                Ok(a)
            }
            None => Err(()),
        }
    }
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
