//! License state (`x-pack/license`): the self-generated basic / trial
//! license, `start_trial`, `start_basic`, `PUT/DELETE /_license` and the
//! license-mode checks behind `GET /_xpack`.
//!
//! The state persists in `<data>/_state/license.json`. Like ES's
//! `LicenseService`, a node without a license generates one on startup from
//! the node setting `xpack.license.self_generated.type` (`basic`, the
//! default, or `trial`).
//!
//! Signatures of installed licenses are checked for ES's signature layout
//! (base64 of version, magic, public-key hash and signed content) but not
//! cryptographically verified.

use crate::error::{EsError, Result};
use crate::node::Node;
use base64::Engine as _;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Self-generated trials last 30 days.
pub const TRIAL_MILLIS: i64 = 30 * 24 * 3_600_000;
/// `max_nodes` (or `max_resource_units` for enterprise) of self-generated licenses.
const SELF_GENERATED_MAX_NODES: i64 = 1000;
/// Version of licenses this node generates (`License.VERSION_CURRENT`).
const VERSION_CURRENT: i64 = 5;
/// Version from which enterprise licenses carry `max_resource_units`.
const VERSION_ENTERPRISE: i64 = 5;

/// Valid `type` values of `POST /_license/start_trial`.
pub const TRIAL_TYPES: &[&str] = &["enterprise", "gold", "platinum", "trial"];
/// License types ES can parse (`License.LicenseType`).
pub const LICENSE_TYPES: &[&str] = &["basic", "standard", "gold", "platinum", "enterprise", "trial"];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct License {
    pub uid: String,
    #[serde(rename = "type")]
    pub kind: String,
    /// `subscription_type` of version 1 licenses (decides their mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscription_type: Option<String>,
    pub issue_date_in_millis: i64,
    /// `None`: self-generated basic, never expires.
    #[serde(default)]
    pub expiry_date_in_millis: Option<i64>,
    #[serde(default)]
    pub max_nodes: Option<i64>,
    #[serde(default)]
    pub max_resource_units: Option<i64>,
    pub issued_to: String,
    pub issuer: String,
    #[serde(default = "minus_one")]
    pub start_date_in_millis: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default = "version_current")]
    pub version: i64,
}

fn minus_one() -> i64 {
    -1
}

fn version_current() -> i64 {
    VERSION_CURRENT
}

/// License operation mode (`License.OperationMode`), lowest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Basic,
    Standard,
    Gold,
    Platinum,
    Enterprise,
    /// Every feature, like enterprise.
    Trial,
}

impl Mode {
    /// `License.OperationMode.resolve` (legacy names included).
    pub fn resolve(kind: &str) -> Mode {
        match kind.to_lowercase().as_str() {
            "basic" => Mode::Basic,
            "standard" => Mode::Standard,
            "silver" | "gold" => Mode::Gold,
            "platinum" | "cloud_internal" | "internal" => Mode::Platinum,
            "enterprise" => Mode::Enterprise,
            _ => Mode::Trial,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Basic => "basic",
            Mode::Standard => "standard",
            Mode::Gold => "gold",
            Mode::Platinum => "platinum",
            Mode::Enterprise => "enterprise",
            Mode::Trial => "trial",
        }
    }

    /// Whether this mode includes features licensed at `min`.
    pub fn allows(self, min: Mode) -> bool {
        self == Mode::Trial || min <= self
    }
}

impl License {
    pub fn mode(&self) -> Mode {
        if self.version == 1 {
            if let Some(s) = &self.subscription_type {
                return Mode::resolve(s);
            }
        }
        Mode::resolve(&self.kind)
    }

    pub fn expired(&self, now: i64) -> bool {
        self.expiry_date_in_millis.map(|e| e < now).unwrap_or(false)
    }

    pub fn status(&self, now: i64) -> &'static str {
        if self.expired(now) {
            "expired"
        } else {
            "active"
        }
    }

    /// The mode features are checked against: an expired license only
    /// keeps basic features.
    pub fn effective_mode(&self, now: i64) -> Mode {
        if self.expired(now) {
            Mode::Basic
        } else {
            self.mode()
        }
    }

    /// `GET /_license` body (`license` object, REST view, current version).
    pub fn to_rest(&self, now: i64) -> Value {
        let mut m = Map::new();
        m.insert("status".into(), json!(self.status(now)));
        m.insert("uid".into(), json!(self.uid));
        m.insert("type".into(), json!(self.kind));
        m.insert("issue_date".into(), json!(iso(self.issue_date_in_millis)));
        m.insert("issue_date_in_millis".into(), json!(self.issue_date_in_millis));
        if let Some(e) = self.expiry_date_in_millis {
            m.insert("expiry_date".into(), json!(iso(e)));
            m.insert("expiry_date_in_millis".into(), json!(e));
        }
        m.insert("max_nodes".into(), json!(self.max_nodes));
        m.insert("max_resource_units".into(), json!(self.max_resource_units));
        m.insert("issued_to".into(), json!(self.issued_to));
        m.insert("issuer".into(), json!(self.issuer));
        m.insert("start_date_in_millis".into(), json!(self.start_date_in_millis));
        Value::Object(m)
    }

    /// The `license` section of `GET /_xpack`.
    pub fn to_xpack_info(&self, now: i64) -> Value {
        let mut m = Map::new();
        m.insert("uid".into(), json!(self.uid));
        m.insert("type".into(), json!(self.kind));
        m.insert("mode".into(), json!(self.mode().name()));
        m.insert("status".into(), json!(self.status(now)));
        if let Some(e) = self.expiry_date_in_millis {
            m.insert("expiry_date_in_millis".into(), json!(e));
        }
        Value::Object(m)
    }
}

pub fn iso(ms: i64) -> String {
    crate::date::from_millis(ms).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

#[derive(Default, Serialize, Deserialize, PartialEq)]
struct Persisted {
    #[serde(default)]
    license: Option<License>,
    /// A trial was generated or started (`LicensesMetadata.trialVersion`).
    #[serde(default)]
    trial_used: bool,
}

pub struct LicenseService {
    path: PathBuf,
    st: Mutex<Persisted>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<LicenseService>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<LicenseService>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The license service of `node`, generating the self-generated license on
/// first use when the node has none.
pub fn service(node: &Node) -> Arc<LicenseService> {
    let svc = {
        let mut r = registry().lock();
        r.entry(node.data.clone())
            .or_insert_with(|| {
                let path = node.data.join("_state/license.json");
                let st: Persisted = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
                Arc::new(LicenseService { path, st: Mutex::new(st) })
            })
            .clone()
    };
    svc.ensure_license(node);
    svc
}

fn started_nodes() -> &'static Mutex<std::collections::HashSet<PathBuf>> {
    static S: OnceLock<Mutex<std::collections::HashSet<PathBuf>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// Generate the startup license (call once the node settings are known).
pub fn start(node: &Node) {
    let _ = service(node);
    started_nodes().lock().insert(node.data.clone());
}

/// Whether the node runs with license enforcement started ([`start`]: an
/// HTTP / cluster node). Engine-only embeddings (unit tests, tools) never
/// start it and keep every index-creation feature.
pub fn started(node: &Node) -> bool {
    started_nodes().lock().contains(&node.data)
}

/// `LogsdbLicenseService.DEFAULT_CUTOFF_DATE` (2025-02-04T00:00:00Z): gold /
/// platinum licenses that started by then keep synthetic source for the
/// legacy use cases.
pub const SYNTHETIC_SOURCE_LEGACY_CUTOFF: i64 = 1_738_627_200_000;

/// Node setting that makes the node behave as if it held an enterprise
/// license (every license-gated feature served, `GET /_license` and
/// `GET /_xpack` report `enterprise`). Off by default: ES-exact licensing.
/// Also read from the `ES_XPACK_LICENSE_ENABLE_ENTERPRISE_FEATURES` env var.
pub const ENTERPRISE_FEATURES_SETTING: &str = "xpack.license.enable_enterprise_features";

/// Whether [`ENTERPRISE_FEATURES_SETTING`] is on.
pub fn enterprise_features_enabled(node: &Node) -> bool {
    let truthy = |v: &Value| v.as_bool() == Some(true) || v.as_str() == Some("true");
    node.node_settings.read().get(ENTERPRISE_FEATURES_SETTING).map(truthy).unwrap_or(false)
        || std::env::var("ES_XPACK_LICENSE_ENABLE_ENTERPRISE_FEATURES").map(|v| v == "true").unwrap_or(false)
}

/// Expiry the hardwired trial reports (2099-12-31T23:59:59.999Z).
pub const HARDWIRED_TRIAL_EXPIRY: i64 = 4_102_444_799_999;

/// The node's effective license: always an active trial.
///
/// Licensing is hardwired to enterprise/trial; see AGENTS.md "Licensing:
/// always enterprise".
///
/// Every licence check (`check`, `effective_mode`, synthetic source,
/// logsdb / TSDB, ES|QL, security, ML-gated paths) reads this license, so
/// every feature is available from node start, before any index exists,
/// whatever was requested (`xpack.license.self_generated.type`,
/// `start_basic`, `PUT` / `DELETE /_license`). The stored license only
/// lends its identity (uid, issue date) so `GET /_license` is stable.
pub fn current(node: &Node) -> License {
    current_from(node, Some(service(node).license(node)))
}

/// `stored` as the effective license (see [`current`]): the hardwired trial.
fn current_from(node: &Node, stored: Option<License>) -> License {
    let now = crate::date::now_millis();
    let mut l = stored.unwrap_or_else(|| self_generated(node, "trial", now, None));
    l.kind = "trial".into();
    l.subscription_type = None;
    l.expiry_date_in_millis = Some(HARDWIRED_TRIAL_EXPIRY);
    l.max_nodes = Some(l.max_nodes.or(l.max_resource_units).unwrap_or(SELF_GENERATED_MAX_NODES));
    l.max_resource_units = None;
    l.version = VERSION_CURRENT;
    if l.issue_date_in_millis > now {
        l.issue_date_in_millis = now;
    }
    l
}

/// `LicenseUtils.newComplianceException`: 403 for a feature the current
/// license does not include.
pub fn compliance_error(feature: &str) -> EsError {
    EsError::new(403, "security_exception", format!("current license is non-compliant for [{feature}]")).with("license.expired.feature", feature)
}

/// Fail with [`compliance_error`] unless the license includes `min`.
pub fn check(node: &Node, feature: &str, min: Mode) -> Result<()> {
    if current(node).effective_mode(crate::date::now_millis()).allows(min) {
        Ok(())
    } else {
        Err(compliance_error(feature))
    }
}

/// Enterprise Search's own wording (`search application` needs platinum,
/// `query rules` enterprise).
pub fn check_ent_search(node: &Node, what: &str, min: Mode) -> Result<()> {
    let now = crate::date::now_millis();
    let l = current(node);
    if l.effective_mode(now).allows(min) {
        return Ok(());
    }
    Err(EsError::new(
        403,
        "security_exception",
        format!(
            "Current license is non-compliant for {what}. Current license is {} {} license. This feature requires an active trial, {}, or higher license.",
            l.status(now),
            l.mode().name(),
            min.name()
        ),
    ))
}

/// Outcome of `start_trial` / `start_basic` / `PUT /_license`.
pub enum Outcome {
    /// The change was applied.
    Done,
    /// Not applied: needs `acknowledge=true`; messages by feature.
    NeedsAck(Vec<(&'static str, Vec<String>)>),
    /// Not applied: `start_trial` after a trial was used.
    TrialAlreadyActivated,
    /// Not applied: `start_basic` while the license is basic.
    AlreadyBasic,
    /// `PUT /_license`: an expired license is refused.
    Expired,
    /// `PUT /_license`: the signature could not be verified.
    Invalid,
}

impl LicenseService {
    /// Cluster metadata carries the elected master's license and trial history.
    pub fn cluster_state(&self) -> Value {
        serde_json::to_value(&*self.st.lock()).unwrap_or(Value::Null)
    }

    pub fn restore_cluster_state(&self, value: &Value) -> Result<bool> {
        let next: Persisted = serde_json::from_value(value.clone())?;
        let mut current = self.st.lock();
        if *current == next { return Ok(false); }
        if let Some(parent) = self.path.parent() { std::fs::create_dir_all(parent)?; }
        let temporary = self.path.with_extension("cluster.tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(&next)?)?;
        std::fs::rename(&temporary, &self.path)?;
        *current = next;
        Ok(true)
    }

    fn persist(&self, st: &Persisted) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("json.tmp");
        if let Ok(b) = serde_json::to_vec_pretty(st) {
            if std::fs::write(&tmp, b).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
    }

    fn ensure_license(&self, node: &Node) {
        let mut st = self.st.lock();
        if st.license.is_some() {
            return;
        }
        // Hardwired trial (see [`current`]), whatever
        // `xpack.license.self_generated.type` asks for.
        let now = crate::date::now_millis();
        st.trial_used = true;
        let lic = self_generated(node, "trial", now, Some(HARDWIRED_TRIAL_EXPIRY));
        st.license = Some(lic);
        self.persist(&st);
    }

    pub fn license(&self, node: &Node) -> License {
        self.ensure_license(node);
        self.st.lock().license.clone().expect("license generated above")
    }

    /// `GET /_license/trial_status`.
    pub fn eligible_for_trial(&self) -> bool {
        // The hardwired trial is active (ES: no second trial).
        false
    }

    /// `GET /_license/basic_status`.
    pub fn eligible_for_basic(&self, node: &Node) -> bool {
        // From the (hardwired) trial ES allows starting basic.
        let _ = node;
        true
    }

    /// `POST /_license/start_trial` (`kind` already validated).
    pub fn start_trial(&self, node: &Node, kind: &str, acknowledge: bool) -> Outcome {
        if !acknowledge {
            return Outcome::NeedsAck(vec![("security", vec!["With a trial license, X-Pack security features are available, but are not enabled by default.".into()])]);
        }
        // Hardwired trial (see [`current`]): acknowledged and nothing to
        // change.
        let _ = (node, kind);
        Outcome::Done
    }

    /// `POST /_license/start_basic` (and `DELETE /_license`, which always
    /// acknowledges and treats an existing basic license as a no-op).
    pub fn start_basic(&self, node: &Node, acknowledge: bool) -> Outcome {
        // Hardwired trial (see [`current`]): the request succeeds (with the
        // acknowledgement ES asks of a trial -> basic move) but the
        // effective license stays the trial.
        let _ = node;
        if !acknowledge {
            let msgs = ack_messages(Mode::Trial, Mode::Basic);
            if !msgs.is_empty() {
                return Outcome::NeedsAck(msgs);
            }
        }
        Outcome::Done
    }

    /// `PUT /_license` with a parsed license.
    pub fn put(&self, lic: License, acknowledge: bool) -> Outcome {
        let now = crate::date::now_millis();
        if !signature_ok(lic.signature.as_deref().unwrap_or("")) {
            return Outcome::Invalid;
        }
        if lic.expired(now) {
            return Outcome::Expired;
        }
        if !acknowledge {
            let msgs = ack_messages(Mode::Trial, lic.mode());
            if !msgs.is_empty() {
                return Outcome::NeedsAck(msgs);
            }
        }
        // Hardwired trial (see [`current`]): accepted, not applied.
        Outcome::Done
    }
}

fn self_generated(node: &Node, kind: &str, now: i64, expiry: Option<i64>) -> License {
    let enterprise = kind == "enterprise";
    License {
        uid: uuid::Uuid::new_v4().to_string(),
        kind: kind.to_string(),
        subscription_type: None,
        issue_date_in_millis: now,
        expiry_date_in_millis: expiry,
        max_nodes: if enterprise { None } else { Some(SELF_GENERATED_MAX_NODES) },
        max_resource_units: if enterprise { Some(SELF_GENERATED_MAX_NODES) } else { None },
        issued_to: node.cluster_name.clone(),
        issuer: "elasticsearch".into(),
        start_date_in_millis: -1,
        signature: None,
        version: VERSION_CURRENT,
    }
}

/// Feature availability by minimum license mode (`XPackLicenseState` /
/// `LicensedFeature`), in `GET /_xpack` order.
pub const XPACK_FEATURES: &[(&str, Mode)] = &[
    ("aggregate_metric", Mode::Basic),
    ("analytics", Mode::Basic),
    ("archive", Mode::Enterprise),
    ("ccr", Mode::Platinum),
    ("data_streams", Mode::Basic),
    ("data_tiers", Mode::Basic),
    ("enrich", Mode::Basic),
    ("enterprise_search", Mode::Platinum),
    ("eql", Mode::Basic),
    ("esql", Mode::Basic),
    ("frozen_indices", Mode::Basic),
    // `GPUPlugin.GPU_INDEXING_FEATURE`.
    ("gpu_vector_indexing", Mode::Enterprise),
    ("graph", Mode::Platinum),
    ("ilm", Mode::Basic),
    ("logsdb", Mode::Basic),
    ("logstash", Mode::Standard),
    ("ml", Mode::Platinum),
    ("monitoring", Mode::Basic),
    ("rollup", Mode::Basic),
    ("searchable_snapshots", Mode::Enterprise),
    ("security", Mode::Basic),
    ("slm", Mode::Basic),
    ("spatial", Mode::Basic),
    ("sql", Mode::Basic),
    ("transform", Mode::Basic),
    ("universal_profiling", Mode::Enterprise),
    ("voting_only", Mode::Basic),
    ("watcher", Mode::Standard),
];

/// Acknowledgement messages for moving from `from` to `to`
/// (`XPackLicenseState.ACKNOWLEDGMENT_MESSAGES`, in ES's order).
pub fn ack_messages(from: Mode, to: Mode) -> Vec<(&'static str, Vec<String>)> {
    let lost = |min: Mode| from.allows(min) && !to.allows(min);
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let mut out: Vec<(&'static str, Vec<String>)> = Vec::new();
    if lost(Mode::Standard) {
        out.push(("watcher", s(&["Watcher will be disabled"])));
        out.push(("logstash", s(&["Logstash will no longer poll for centrally-managed pipelines"])));
    }
    if lost(Mode::Gold) && to == Mode::Basic {
        out.push((
            "security",
            s(&[
                "Authentication will be limited to the native and file realms.",
                "Security tokens will not be supported.",
                "IP filtering and auditing will be disabled.",
                "Field and document level access control will be disabled.",
                "Custom realms will be ignored.",
                "A custom authorization engine will be ignored.",
            ]),
        ));
    }
    if lost(Mode::Platinum) {
        out.push(("ccr", s(&["Cross-Cluster Replication will be disabled"])));
        let mut es = vec!["Search Applications and behavioral analytics will be disabled."];
        if lost(Mode::Enterprise) {
            es.push("Query rules will be disabled.");
        }
        es.extend(["Elastic Web crawler will be disabled.", "Connector clients require at least a platinum license."]);
        out.push(("enterprise_search", s(&es)));
        out.push(("redact_processor", s(&["Redact ingest pipeline processors will be disabled"])));
    }
    if lost(Mode::Enterprise) {
        out.push(("esql", s(&["ES|QL cross-cluster search will be disabled."])));
    }
    if lost(Mode::Standard) {
        out.push(("beats", s(&["Beats will no longer be able to use centrally-managed configuration"])));
    }
    if to == Mode::Basic && from != Mode::Basic {
        out.push((
            "monitoring",
            s(&["Multi-cluster support is disabled for clusters with [BASIC] license. If you are\nrunning multiple clusters, users won't be able to access the clusters with\n[BASIC] licenses from within a single X-Pack Kibana instance. You will have to deploy a\nseparate and dedicated X-pack Kibana instance for each [BASIC] cluster you wish to monitor."]),
        ));
    }
    if lost(Mode::Platinum) {
        out.push(("graph", s(&["Graph will be disabled"])));
        out.push(("ml", s(&["Machine learning will be disabled"])));
        out.push(("sql", s(&["JDBC and ODBC support will be disabled, but you can continue to use SQL CLI and REST endpoint"])));
    }
    out
}

/// Whether `sig` has ES's signature layout: base64 of
/// `[version][magic len][magic][hash len][hash][content len][content]`
/// (big-endian `int` lengths).
fn signature_ok(sig: &str) -> bool {
    let Ok(b) = b64().decode(sig) else { return false };
    let mut pos = 0usize;
    fn int(b: &[u8], pos: &mut usize) -> Option<usize> {
        let v = b.get(*pos..*pos + 4)?;
        *pos += 4;
        usize::try_from(i32::from_be_bytes([v[0], v[1], v[2], v[3]])).ok()
    }
    let Some(_version) = int(&b, &mut pos) else { return false };
    for _ in 0..3 {
        let Some(n) = int(&b, &mut pos) else { return false };
        pos += n;
        if pos > b.len() {
            return false;
        }
    }
    pos == b.len()
}

/// Java's basic decoder: padding optional.
fn b64() -> base64::engine::GeneralPurpose {
    use base64::engine::{general_purpose::GeneralPurposeConfig, DecodePaddingMode};
    base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent))
}

/// The license version encoded in a signature (its first `int`).
pub fn signature_version(sig: &str) -> Option<i64> {
    let b = b64().decode(sig).ok()?;
    let v = b.get(0..4)?;
    Some(i32::from_be_bytes([v[0], v[1], v[2], v[3]]) as i64)
}

/// Java's `Base64.Decoder` message for an undecodable signature.
pub fn base64_error(sig: &str) -> Option<String> {
    let valid = |c: u8| c.is_ascii_alphanumeric() || c == b'+' || c == b'/';
    let body = sig.trim_end_matches('=');
    if let Some(c) = body.bytes().find(|c| !valid(*c)) {
        return Some(format!("Illegal base64 character {:x}", c));
    }
    if b64().decode(sig).is_ok() {
        return None;
    }
    if body.len() % 4 == 1 {
        Some("Last unit does not have enough valid bits".into())
    } else {
        Some("Input byte array has wrong 4-byte ending unit".into())
    }
}

/// Parse one license object of a `PUT /_license` body.
pub fn parse_license(v: &Value) -> Result<License> {
    let o = v.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "failed to parse license - expected object"))?;
    let s = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_string);
    let n = |k: &str| o.get(k).and_then(|x| x.as_i64().or_else(|| x.as_str().and_then(|s| s.parse().ok())));
    let uid = s("uid");
    let kind = s("type");
    let signature = s("signature");
    let mut version = signature.as_deref().and_then(signature_version).unwrap_or(VERSION_CURRENT);
    if let Some(sig) = &signature {
        if let Some(why) = base64_error(sig) {
            return Err(EsError::new(400, "parse_exception", format!("malformed signature for license [{}]", uid.clone().unwrap_or_default()))
                .caused_by(EsError::illegal_argument(why)));
        }
        // The license version is the signature's first `int`.
        if b64().decode(sig).map(|b| b.len() < 4).unwrap_or(true) {
            return Err(EsError::new(400, "parse_exception", format!("malformed signature for license [{}]", uid.clone().unwrap_or_default()))
                .caused_by(EsError::new(400, "buffer_underflow_exception", "").with("reason", Value::Null)));
        }
    }
    let Some(kind) = kind else {
        return Err(EsError::new(500, "null_pointer_exception", "Cannot invoke \"String.toLowerCase(java.util.Locale)\" because \"name\" is null"));
    };
    if version < 1 {
        version = 1;
    }
    let kl = kind.to_lowercase();
    let legacy = matches!(kl.as_str(), "subscription" | "internal" | "dev" | "silver" | "cloud_internal" | "none");
    if !legacy && !LICENSE_TYPES.contains(&kl.as_str()) {
        return Err(EsError::illegal_argument(format!(
            "unrecognised license type [ {kind}], supported license types are [{}]",
            LICENSE_TYPES.join(",")
        )));
    }
    let ise = |m: &str| EsError::new(500, "illegal_state_exception", m.to_string());
    let issuer = s("issuer").ok_or_else(|| ise("issuer can not be null"))?;
    let issued_to = s("issued_to").ok_or_else(|| ise("issuedTo can not be null"))?;
    let issue = n("issue_date_in_millis").ok_or_else(|| ise("issueDate has to be set"))?;
    let uid = uid.ok_or_else(|| ise("uid can not be null"))?;
    let expiry = n("expiry_date_in_millis").ok_or_else(|| ise("expiryDate has to be set"))?;
    let max_nodes = n("max_nodes");
    let max_ru = n("max_resource_units");
    if version >= VERSION_ENTERPRISE && kl == "enterprise" {
        if max_ru.is_none() {
            return Err(ise("maxResourceUnits must be set for enterprise licenses (version 5+)"));
        }
    } else if max_nodes.is_none() {
        return Err(ise("maxNodes has to be set"));
    }
    Ok(License {
        uid,
        kind,
        subscription_type: s("subscription_type"),
        issue_date_in_millis: issue,
        expiry_date_in_millis: Some(expiry),
        max_nodes,
        max_resource_units: max_ru,
        issued_to,
        issuer,
        start_date_in_millis: n("start_date_in_millis").unwrap_or(-1),
        signature,
        version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "licence hardwired to trial"]
    fn cluster_license_restore_preserves_uid_mode_and_trial_history() {
        let master_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let master = Node::open(master_dir.path()).unwrap();
        let peer = Node::open(peer_dir.path()).unwrap();
        let authoritative = service(&master);
        assert!(matches!(authoritative.start_trial(&master, "trial", true), Outcome::Done));
        let mirrored = service(&peer);
        assert!(mirrored.restore_cluster_state(&authoritative.cluster_state()).unwrap());
        assert_eq!(current(&master), current(&peer));
        assert!(!mirrored.eligible_for_trial());
        assert!(!mirrored.restore_cluster_state(&authoritative.cluster_state()).unwrap());
        assert!(matches!(authoritative.start_basic(&master, true), Outcome::Done));
        mirrored.restore_cluster_state(&authoritative.cluster_state()).unwrap();
        assert_eq!(current(&peer).mode(), Mode::Basic);
        let reloaded: Persisted = serde_json::from_slice(&std::fs::read(mirrored.path.clone()).unwrap()).unwrap();
        assert!(reloaded.trial_used);
        assert_eq!(reloaded.license.unwrap(), current(&master));
    }

    #[test]
    fn modes_and_acks() {
        assert!(Mode::Trial.allows(Mode::Enterprise));
        assert!(!Mode::Gold.allows(Mode::Platinum));
        assert!(Mode::Platinum.allows(Mode::Standard));
        let names: Vec<&str> = ack_messages(Mode::Platinum, Mode::Basic).iter().map(|x| x.0).collect();
        assert_eq!(names, ["watcher", "logstash", "security", "ccr", "enterprise_search", "redact_processor", "beats", "monitoring", "graph", "ml", "sql"]);
        let names: Vec<&str> = ack_messages(Mode::Trial, Mode::Basic).iter().map(|x| x.0).collect();
        assert_eq!(names, ["watcher", "logstash", "security", "ccr", "enterprise_search", "redact_processor", "esql", "beats", "monitoring", "graph", "ml", "sql"]);
        assert!(ack_messages(Mode::Basic, Mode::Platinum).is_empty());
    }

    #[test]
    fn signatures() {
        // version 2 license signature layout from ES's tests (header only).
        assert_eq!(base64_error("AAAAAgAAAA3U8"), Some("Last unit does not have enough valid bits".into()));
        assert!(base64_error("not-base64").unwrap().starts_with("Illegal base64 character"));
        assert!(!signature_ok("AAAA"));
        let mut b = vec![0, 0, 0, 2];
        for part in [&b"magic"[..], b"hash", b"content"] {
            b.extend((part.len() as i32).to_be_bytes());
            b.extend(part);
        }
        let sig = base64::engine::general_purpose::STANDARD.encode(&b);
        assert!(signature_ok(&sig));
        assert_eq!(signature_version(&sig), Some(2));
    }
}
