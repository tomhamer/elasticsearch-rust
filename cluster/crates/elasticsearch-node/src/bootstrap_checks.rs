//! ES `BootstrapChecks`: production checks that fail the start when the
//! node is bound or publishing to a non-loopback transport address (and not
//! in single-node discovery), or when `es.enforce.bootstrap.checks` is
//! `true`; otherwise each failure is only logged as a warning.
//!
//! Built-in checks, in ES's order (`BootstrapChecks.checks()`), as they
//! apply to a native process:
//! - `FileDescriptorCheck` (`OsXFileDescriptorCheck` on macOS: 10240),
//! - `MlockallCheck` (`bootstrap.memory_lock` requested but not locked),
//! - `MaxNumberOfThreadsCheck` (Linux),
//! - `MaxSizeVirtualMemoryCheck` and `MaxFileSizeCheck` (Linux, macOS),
//! - `MaxMapCountCheck` (Linux, while `node.store.allow_mmap` is true),
//! - `DiscoveryConfiguredCheck`,
//! - `ByteOrderCheck`.
//!
//! Then the plugin checks: the security plugin's `TransportTLSBootstrapCheck`
//! (only installed while security is enabled). The GPU check lives in
//! `elasticsearch_engine::gpu::bootstrap_check`.
//!
//! ES checks deliberately not ported, because they test the JVM this
//! process does not run on (there is no heap to size, no collector, VM
//! flavour, `-XX:OnError` hook or early-access JDK):
//! - `HeapSizeCheck` (initial vs maximum `-Xms`/`-Xmx`),
//! - `ClientJvmCheck`, `UseSerialGCCheck`, `EarlyAccessCheck`,
//! - `OnErrorCheck`, `OnOutOfMemoryErrorCheck` (JVM hooks that fork; only
//!   meaningful with ES's seccomp filter, below),
//! - `SystemCallFilterCheck`: ES installs a seccomp/seatbelt filter blocking
//!   `fork`/`exec` from inside the JVM; this process installs none, and the
//!   check would fail every production start rather than test a setting
//!   an operator can change,
//! - `AllPermissionCheck` (Java security manager / entitlements; gone from
//!   ES 9's built-in list anyway).

use crate::config::Settings;
use crate::natives::{self, Platform, ProcessLimits, UNKNOWN, UNLIMITED};
use std::net::IpAddr;

/// ES `ReferenceDocs` links for 9.5 (`?version=<major.minor>`).
pub const BOOTSTRAP_CHECKS_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5";
pub const BOOTSTRAP_CHECK_TLS_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-tls";
pub const BOOTSTRAP_CHECK_FILE_DESCRIPTOR_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-file-descriptor";
pub const BOOTSTRAP_CHECK_MEMORY_LOCK_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-memory-lock";
pub const BOOTSTRAP_CHECK_MAX_NUMBER_THREADS_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#max-number-threads-check";
pub const BOOTSTRAP_CHECK_MAX_SIZE_VIRTUAL_MEMORY_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#max-size-virtual-memory-check";
pub const BOOTSTRAP_CHECK_MAX_FILE_SIZE_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-max-file-size";
pub const BOOTSTRAP_CHECK_MAXIMUM_MAP_COUNT_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-max-map-count";
pub const BOOTSTRAP_CHECK_DISCOVERY_CONFIGURATION_DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-discovery-configuration";

/// The system property ES reads; rustic takes it from `ES_JAVA_OPTS`
/// (`-Des.enforce.bootstrap.checks=true`) or `ES_ENFORCE_BOOTSTRAP_CHECKS`.
pub const ES_ENFORCE_BOOTSTRAP_CHECKS: &str = "es.enforce.bootstrap.checks";

/// `FileDescriptorCheck`'s limit (`OsXFileDescriptorCheck`: `OPEN_MAX`).
pub const FILE_DESCRIPTOR_LIMIT: i64 = 65535;
pub const OSX_FILE_DESCRIPTOR_LIMIT: i64 = 10240;
/// `MaxNumberOfThreadsCheck.MAX_NUMBER_OF_THREADS_THRESHOLD` (`1 << 12`).
pub const MAX_NUMBER_OF_THREADS_THRESHOLD: i64 = 1 << 12;
/// `MaxMapCountCheck.LIMIT` (`1 << 18`).
pub const MAX_MAP_COUNT_LIMIT: i64 = 1 << 18;

/// ES `NodeValidationException`: the node refuses to start (exit code 78).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeValidationError(pub String);

impl std::fmt::Display for NodeValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NodeValidationError {}

/// What the checks read from the OS (ES's `NativeAccess`, `ProcessProbe`,
/// `/proc`, `user.name`), gathered once so tests can inject any values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    pub platform: Platform,
    /// Soft `RLIMIT_NOFILE` ([`UNKNOWN`] where there is none).
    pub max_file_descriptors: i64,
    pub memory_locked: bool,
    pub limits: ProcessLimits,
    /// `vm.max_map_count` (Linux; [`UNKNOWN`] when unreadable).
    pub max_map_count: i64,
    pub user_name: String,
    pub little_endian: bool,
}

impl Probe {
    /// This process, now.
    pub fn read() -> Probe {
        let platform = Platform::current();
        Probe {
            platform,
            max_file_descriptors: natives::max_file_descriptors(),
            memory_locked: natives::is_memory_locked(),
            limits: natives::process_limits(),
            max_map_count: if platform == Platform::Linux { natives::max_map_count() } else { UNKNOWN },
            user_name: natives::user_name(),
            little_endian: cfg!(target_endian = "little"),
        }
    }
}

/// ES `BootstrapChecks.enforceLimits`: some bound address or the publish
/// address is not loopback, and discovery is not single-node.
pub fn enforce_limits(bound: &[IpAddr], publish: IpAddr, single_node: bool) -> bool {
    let all_loopback = bound.iter().all(IpAddr::is_loopback) && publish.is_loopback();
    !all_loopback && !single_node
}

/// The `es.enforce.bootstrap.checks` property: unset = false, `true` =
/// true, anything else fails the start (ES message).
pub fn explicitly_enforced(java_opts: Option<&str>, env_value: Option<&str>) -> Result<bool, String> {
    let prefix = format!("-D{ES_ENFORCE_BOOTSTRAP_CHECKS}=");
    let from_opts = java_opts.and_then(|opts| opts.split_whitespace().filter_map(|o| o.strip_prefix(prefix.as_str())).last());
    match from_opts.or(env_value) {
        None => Ok(false),
        Some("true") => Ok(true),
        Some(other) => Err(format!("[{ES_ENFORCE_BOOTSTRAP_CHECKS}] must be [true] but was [{other}]")),
    }
}

/// One failed check: the message and its reference docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub message: String,
    pub docs: &'static str,
}

fn setting_true(settings: &Settings, key: &str) -> bool {
    match settings.get(key) {
        Some(serde_json::Value::String(s)) => s == "true",
        Some(serde_json::Value::Bool(b)) => *b,
        _ => false,
    }
}

/// A boolean node setting parsed as ES's `Booleans.parseBoolean` does:
/// only `true` / `false`, anything else is an `IllegalArgumentException`.
pub fn bool_setting(settings: &Settings, key: &str, default: bool) -> Result<bool, String> {
    match settings.get(key) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(serde_json::Value::Bool(b)) => Ok(*b),
        Some(serde_json::Value::String(s)) if s == "true" => Ok(true),
        Some(serde_json::Value::String(s)) if s == "false" => Ok(false),
        Some(serde_json::Value::String(s)) => Err(format!("Failed to parse value [{s}] as only [true] or [false] are allowed.")),
        Some(other) => Err(format!("Failed to parse value [{other}] as only [true] or [false] are allowed.")),
    }
}

/// ES `BootstrapSettings.MEMORY_LOCK_SETTING` (`bootstrap.memory_lock`, default false).
pub fn memory_lock_requested(settings: &Settings) -> Result<bool, String> {
    bool_setting(settings, "bootstrap.memory_lock", false)
}

/// `FileDescriptorCheck` / `OsXFileDescriptorCheck`.
fn file_descriptor_check(probe: &Probe) -> Option<Failure> {
    let limit = if probe.platform == Platform::MacOs { OSX_FILE_DESCRIPTOR_LIMIT } else { FILE_DESCRIPTOR_LIMIT };
    (probe.max_file_descriptors != -1 && probe.max_file_descriptors < limit).then(|| Failure {
        message: format!("max file descriptors [{}] for elasticsearch process is too low, increase to at least [{limit}]", probe.max_file_descriptors),
        docs: BOOTSTRAP_CHECK_FILE_DESCRIPTOR_DOCS,
    })
}

/// `MlockallCheck`.
fn mlockall_check(memory_lock: bool, probe: &Probe) -> Option<Failure> {
    (memory_lock && !probe.memory_locked).then(|| Failure {
        message: "memory locking requested for elasticsearch process but memory is not locked".into(),
        docs: BOOTSTRAP_CHECK_MEMORY_LOCK_DOCS,
    })
}

/// `MaxNumberOfThreadsCheck` (installed on Linux only).
fn max_number_of_threads_check(probe: &Probe) -> Option<Failure> {
    let n = probe.limits.max_threads;
    (n != UNKNOWN && n < MAX_NUMBER_OF_THREADS_THRESHOLD).then(|| Failure {
        message: format!("max number of threads [{n}] for user [{}] is too low, increase to at least [{MAX_NUMBER_OF_THREADS_THRESHOLD}]", probe.user_name),
        docs: BOOTSTRAP_CHECK_MAX_NUMBER_THREADS_DOCS,
    })
}

/// `MaxSizeVirtualMemoryCheck`. ES compares with `Long.MIN_VALUE`, not
/// `ProcessLimits.UNKNOWN`, so an unreadable limit (-1) fails as `[-1]`;
/// kept as ES has it.
fn max_size_virtual_memory_check(probe: &Probe) -> Option<Failure> {
    let v = probe.limits.max_virtual_memory_size;
    (v != i64::MIN && v != UNLIMITED).then(|| Failure {
        message: format!("max size virtual memory [{v}] for user [{}] is too low, increase to [unlimited]", probe.user_name),
        docs: BOOTSTRAP_CHECK_MAX_SIZE_VIRTUAL_MEMORY_DOCS,
    })
}

/// `MaxFileSizeCheck` (the same `Long.MIN_VALUE` comparison as above).
fn max_file_size_check(probe: &Probe) -> Option<Failure> {
    let v = probe.limits.max_file_size;
    (v != i64::MIN && v != UNLIMITED).then(|| Failure {
        message: format!("max file size [{v}] for user [{}] is too low, increase to [unlimited]", probe.user_name),
        docs: BOOTSTRAP_CHECK_MAX_FILE_SIZE_DOCS,
    })
}

/// `MaxMapCountCheck` (installed on Linux only): only while
/// `node.store.allow_mmap` (default true) lets a store use mmap.
fn max_map_count_check(allow_mmap: bool, probe: &Probe) -> Option<Failure> {
    let n = probe.max_map_count;
    (allow_mmap && n != -1 && n < MAX_MAP_COUNT_LIMIT).then(|| Failure {
        message: format!("max virtual memory areas vm.max_map_count [{n}] is too low, increase to at least [{MAX_MAP_COUNT_LIMIT}]"),
        docs: BOOTSTRAP_CHECK_MAXIMUM_MAP_COUNT_DOCS,
    })
}

/// `DiscoveryConfiguredCheck`: with `discovery.type: multi-node` (the
/// default), one of the discovery settings must be present
/// (`ClusterBootstrapService.discoveryIsConfigured`: `Setting.exists`, so
/// an empty list counts).
fn discovery_configured_check(settings: &Settings) -> Option<Failure> {
    let discovery_type = crate::config::get_str(settings, "discovery.type").unwrap_or("multi-node");
    if discovery_type != "multi-node" {
        return None;
    }
    if ["discovery.seed_providers", "discovery.seed_hosts", "cluster.initial_master_nodes"].iter().any(|k| settings.contains_key(*k)) {
        return None;
    }
    Some(Failure {
        message: "the default discovery settings are unsuitable for production use; at least one of [discovery.seed_hosts, discovery.seed_providers, cluster.initial_master_nodes] must be configured".into(),
        docs: BOOTSTRAP_CHECK_DISCOVERY_CONFIGURATION_DOCS,
    })
}

/// `ByteOrderCheck`.
fn byte_order_check(probe: &Probe) -> Option<Failure> {
    (!probe.little_endian).then(|| Failure { message: "Little-endian native byte order is required to run Elasticsearch".into(), docs: BOOTSTRAP_CHECKS_DOCS })
}

/// The checks that fail for these settings and this process (`probe`), in
/// ES's order: the built-in checks, then the plugin checks
/// (`security_enabled` is the node's resolved `xpack.security.enabled`).
/// An unparsable boolean setting is ES's `IllegalArgumentException`.
pub fn failures(settings: &Settings, security_enabled: bool, probe: &Probe) -> Result<Vec<Failure>, String> {
    let memory_lock = memory_lock_requested(settings)?;
    let allow_mmap = bool_setting(settings, "node.store.allow_mmap", true)?;
    let linux = probe.platform == Platform::Linux;
    let posix = linux || probe.platform == Platform::MacOs;
    let mut out = Vec::new();
    out.extend(file_descriptor_check(probe));
    out.extend(mlockall_check(memory_lock, probe));
    if linux {
        out.extend(max_number_of_threads_check(probe));
    }
    if posix {
        out.extend(max_size_virtual_memory_check(probe));
        out.extend(max_file_size_check(probe));
    }
    if linux {
        out.extend(max_map_count_check(allow_mmap, probe));
    }
    out.extend(discovery_configured_check(settings));
    out.extend(byte_order_check(probe));
    // `TransportTLSBootstrapCheck`, installed only while security is enabled.
    if security_enabled && !setting_true(settings, "xpack.security.transport.ssl.enabled") {
        out.push(Failure {
            message: "Transport SSL must be enabled if security is enabled. Please set [xpack.security.transport.ssl.enabled] to [true] or disable security by setting [xpack.security.enabled] to [false]".into(),
            docs: BOOTSTRAP_CHECK_TLS_DOCS,
        });
    }
    Ok(out)
}

/// ES `BootstrapChecks.check`: the failures are fatal when enforced, else
/// logged as warnings.
pub fn check(failures: Vec<Failure>, enforce_limits: bool, enforced_explicitly: bool) -> Result<(), NodeValidationError> {
    if enforce_limits {
        tracing::info!("bound or publishing to a non-loopback address, enforcing bootstrap checks");
    } else if enforced_explicitly {
        tracing::info!("explicitly enforcing bootstrap checks");
    }
    let messages: Vec<String> = failures.iter().map(|f| format!("{}; for more information see [{}]", f.message, f.docs)).collect();
    if !enforce_limits && !enforced_explicitly {
        for m in &messages {
            tracing::warn!("{m}");
        }
        return Ok(());
    }
    if messages.is_empty() {
        return Ok(());
    }
    let n = messages.len();
    let mut lines = vec![format!(
        "[{n}] bootstrap checks failed. You must address the points described in the following [{n}] lines before starting Elasticsearch. For more information see [{BOOTSTRAP_CHECKS_DOCS}]"
    )];
    for (i, m) in messages.iter().enumerate() {
        lines.push(format!("bootstrap check failure [{}] of [{n}]: {m}", i + 1));
    }
    Err(NodeValidationError(lines.join("\n")))
}

/// Run the node's bootstrap checks against its transport binding.
pub fn run(settings: &Settings, security_enabled: bool, bound: &[IpAddr], publish: IpAddr, single_node: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let java_opts = std::env::var("ES_JAVA_OPTS").ok();
    let env_value = std::env::var("ES_ENFORCE_BOOTSTRAP_CHECKS").ok();
    let explicit = explicitly_enforced(java_opts.as_deref(), env_value.as_deref())?;
    let failures = failures(settings, security_enabled, &Probe::read())?;
    check(failures, enforce_limits(bound, publish, single_node), explicit)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    fn s(pairs: &[(&str, &str)]) -> Settings {
        pairs.iter().map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string()))).collect()
    }

    const LO: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
    const ANY: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
    const SITE: IpAddr = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5));

    #[test]
    fn enforced_only_when_bound_or_publishing_off_loopback_and_not_single_node() {
        assert!(!enforce_limits(&[LO], LO, false));
        assert!(!enforce_limits(&[IpAddr::V6(Ipv6Addr::LOCALHOST)], LO, false));
        assert!(enforce_limits(&[ANY], SITE, false));
        assert!(enforce_limits(&[LO], SITE, false), "publishing off loopback");
        assert!(enforce_limits(&[SITE], LO, false), "bound off loopback");
        assert!(!enforce_limits(&[ANY], SITE, true), "single-node discovery never enforces");
    }

    #[test]
    fn explicit_enforcement_property() {
        assert_eq!(explicitly_enforced(None, None), Ok(false));
        assert_eq!(explicitly_enforced(Some("-Xms1g -Des.enforce.bootstrap.checks=true"), None), Ok(true));
        assert_eq!(explicitly_enforced(None, Some("true")), Ok(true));
        assert_eq!(explicitly_enforced(Some("-Xmx1g"), None), Ok(false));
        assert_eq!(explicitly_enforced(None, Some("yes")).unwrap_err(), "[es.enforce.bootstrap.checks] must be [true] but was [yes]");
    }

    /// A Linux process that passes every check.
    fn ok_probe() -> Probe {
        Probe {
            platform: Platform::Linux,
            max_file_descriptors: 1048576,
            memory_locked: false,
            limits: ProcessLimits { max_threads: 127_000, max_virtual_memory_size: UNLIMITED, max_file_size: UNLIMITED },
            max_map_count: 1048576,
            user_name: "ubuntu".into(),
            little_endian: true,
        }
    }

    /// Discovery configured, security off: no plugin check either.
    fn configured() -> Settings {
        s(&[("discovery.seed_hosts", "10.0.0.6")])
    }

    fn messages(settings: &Settings, probe: &Probe) -> Vec<String> {
        failures(settings, false, probe).unwrap().into_iter().map(|f| format!("{}; for more information see [{}]", f.message, f.docs)).collect()
    }

    const DOCS: &str = "https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5";

    #[test]
    fn a_healthy_process_passes_every_check() {
        assert!(messages(&configured(), &ok_probe()).is_empty());
        // macOS: no thread or map count check, 10240 file descriptors suffice.
        let mac = Probe { platform: Platform::MacOs, max_file_descriptors: 10240, max_map_count: UNKNOWN, limits: ProcessLimits { max_threads: UNKNOWN, ..ok_probe().limits }, ..ok_probe() };
        assert!(messages(&configured(), &mac).is_empty());
    }

    #[test]
    fn file_descriptor_check() {
        let p = Probe { max_file_descriptors: 1024, ..ok_probe() };
        assert_eq!(messages(&configured(), &p), vec![format!("max file descriptors [1024] for elasticsearch process is too low, increase to at least [65535]; for more information see [{DOCS}#bootstrap-checks-file-descriptor]")]);
        assert!(messages(&configured(), &Probe { max_file_descriptors: 65535, ..ok_probe() }).is_empty());
        assert!(messages(&configured(), &Probe { max_file_descriptors: 65534, ..ok_probe() }).len() == 1);
        // Unknown: skipped.
        assert!(messages(&configured(), &Probe { max_file_descriptors: -1, ..ok_probe() }).is_empty());
        // macOS (`OsXFileDescriptorCheck`): at least 10240.
        let mac = |n| Probe { platform: Platform::MacOs, max_file_descriptors: n, ..ok_probe() };
        assert!(messages(&configured(), &mac(10240)).is_empty());
        assert_eq!(messages(&configured(), &mac(256))[0], format!("max file descriptors [256] for elasticsearch process is too low, increase to at least [10240]; for more information see [{DOCS}#bootstrap-checks-file-descriptor]"));
    }

    #[test]
    fn mlockall_check() {
        let lock = s(&[("discovery.seed_hosts", "x"), ("bootstrap.memory_lock", "true")]);
        assert_eq!(messages(&lock, &ok_probe()), vec![format!("memory locking requested for elasticsearch process but memory is not locked; for more information see [{DOCS}#bootstrap-checks-memory-lock]")]);
        assert!(messages(&lock, &Probe { memory_locked: true, ..ok_probe() }).is_empty());
        // Not requested: never fails.
        assert!(messages(&configured(), &ok_probe()).is_empty());
        let mut off = configured();
        off.insert("bootstrap.memory_lock".into(), serde_json::Value::Bool(false));
        assert!(messages(&off, &ok_probe()).is_empty());
        // An unparsable boolean is ES's IllegalArgumentException.
        let bad = s(&[("bootstrap.memory_lock", "yes")]);
        assert_eq!(failures(&bad, false, &ok_probe()).unwrap_err(), "Failed to parse value [yes] as only [true] or [false] are allowed.");
    }

    #[test]
    fn max_number_of_threads_check() {
        let p = Probe { limits: ProcessLimits { max_threads: 1024, ..ok_probe().limits }, ..ok_probe() };
        assert_eq!(messages(&configured(), &p), vec![format!("max number of threads [1024] for user [ubuntu] is too low, increase to at least [4096]; for more information see [{DOCS}#max-number-threads-check]")]);
        assert!(messages(&configured(), &Probe { limits: ProcessLimits { max_threads: 4096, ..ok_probe().limits }, ..ok_probe() }).is_empty());
        assert!(messages(&configured(), &Probe { limits: ProcessLimits { max_threads: UNKNOWN, ..ok_probe().limits }, ..ok_probe() }).is_empty());
        assert!(messages(&configured(), &Probe { limits: ProcessLimits { max_threads: UNLIMITED, ..ok_probe().limits }, ..ok_probe() }).is_empty());
        // Linux only.
        assert!(messages(&configured(), &Probe { platform: Platform::MacOs, ..p }).is_empty());
    }

    #[test]
    fn max_size_virtual_memory_and_max_file_size_checks() {
        let vm = Probe { limits: ProcessLimits { max_virtual_memory_size: 4294967296, ..ok_probe().limits }, ..ok_probe() };
        assert_eq!(messages(&configured(), &vm), vec![format!("max size virtual memory [4294967296] for user [ubuntu] is too low, increase to [unlimited]; for more information see [{DOCS}#max-size-virtual-memory-check]")]);
        let fsize = Probe { limits: ProcessLimits { max_file_size: 1 << 40, ..ok_probe().limits }, ..ok_probe() };
        assert_eq!(messages(&configured(), &fsize), vec![format!("max file size [1099511627776] for user [ubuntu] is too low, increase to [unlimited]; for more information see [{DOCS}#bootstrap-checks-max-file-size]")]);
        // macOS has both checks too.
        assert_eq!(messages(&configured(), &Probe { platform: Platform::MacOs, max_map_count: UNKNOWN, ..fsize.clone() }).len(), 1);
        // ES compares with Long.MIN_VALUE: an unreadable limit (-1) fails.
        let unknown = Probe { limits: ProcessLimits { max_file_size: UNKNOWN, ..ok_probe().limits }, ..ok_probe() };
        assert!(messages(&configured(), &unknown)[0].starts_with("max file size [-1] for user [ubuntu]"));
        // Other platforms: not installed.
        assert!(messages(&configured(), &Probe { platform: Platform::Other, ..fsize }).is_empty());
    }

    #[test]
    fn max_map_count_check() {
        let p = Probe { max_map_count: 65530, ..ok_probe() };
        assert_eq!(messages(&configured(), &p), vec![format!("max virtual memory areas vm.max_map_count [65530] is too low, increase to at least [262144]; for more information see [{DOCS}#bootstrap-checks-max-map-count]")]);
        assert!(messages(&configured(), &Probe { max_map_count: 262144, ..ok_probe() }).is_empty());
        assert!(messages(&configured(), &Probe { max_map_count: -1, ..ok_probe() }).is_empty());
        // `node.store.allow_mmap: false` skips it, as in ES.
        let mut no_mmap = configured();
        no_mmap.insert("node.store.allow_mmap".into(), serde_json::Value::String("false".into()));
        assert!(messages(&no_mmap, &p).is_empty());
        // Linux only.
        assert!(messages(&configured(), &Probe { platform: Platform::MacOs, ..p }).is_empty());
        assert_eq!(crate::natives::parse_max_map_count("65530"), Some(65530));
        assert_eq!(crate::natives::parse_max_map_count("x"), None);
    }

    #[test]
    fn discovery_configured_check() {
        let unset = format!("the default discovery settings are unsuitable for production use; at least one of [discovery.seed_hosts, discovery.seed_providers, cluster.initial_master_nodes] must be configured; for more information see [{DOCS}#bootstrap-checks-discovery-configuration]");
        assert_eq!(messages(&s(&[]), &ok_probe()), vec![unset.clone()]);
        assert_eq!(messages(&s(&[("discovery.type", "multi-node")]), &ok_probe()), vec![unset]);
        for key in ["discovery.seed_hosts", "discovery.seed_providers", "cluster.initial_master_nodes"] {
            assert!(messages(&s(&[(key, "a")]), &ok_probe()).is_empty(), "{key}");
        }
        // Present but empty still counts (`Setting.exists`).
        let mut empty = Settings::new();
        empty.insert("discovery.seed_hosts".into(), serde_json::json!([]));
        assert!(messages(&empty, &ok_probe()).is_empty());
        assert!(messages(&s(&[("discovery.type", "single-node")]), &ok_probe()).is_empty());
    }

    #[test]
    fn byte_order_check() {
        assert_eq!(messages(&configured(), &Probe { little_endian: false, ..ok_probe() }), vec![format!("Little-endian native byte order is required to run Elasticsearch; for more information see [{DOCS}]")]);
    }

    #[test]
    fn failures_are_numbered_in_es_order() {
        let p = Probe { max_file_descriptors: 1024, max_map_count: 65530, ..ok_probe() };
        let err = check(failures(&s(&[]), true, &p).unwrap(), enforce_limits(&[ANY], SITE, false), false).unwrap_err().0;
        let lines: Vec<&str> = err.lines().collect();
        assert_eq!(lines.len(), 5);
        assert_eq!(lines[0], format!("[4] bootstrap checks failed. You must address the points described in the following [4] lines before starting Elasticsearch. For more information see [{DOCS}]"));
        assert!(lines[1].starts_with("bootstrap check failure [1] of [4]: max file descriptors [1024]"), "{}", lines[1]);
        assert!(lines[2].starts_with("bootstrap check failure [2] of [4]: max virtual memory areas vm.max_map_count [65530]"), "{}", lines[2]);
        assert!(lines[3].starts_with("bootstrap check failure [3] of [4]: the default discovery settings are unsuitable"), "{}", lines[3]);
        assert!(lines[4].starts_with("bootstrap check failure [4] of [4]: Transport SSL must be enabled"), "{}", lines[4]);
        // Loopback / single-node: the same failures are only warnings.
        assert!(check(failures(&s(&[]), true, &p).unwrap(), enforce_limits(&[LO], LO, false), false).is_ok());
        assert!(check(failures(&s(&[]), true, &p).unwrap(), enforce_limits(&[ANY], SITE, true), false).is_ok());
        // Explicit enforcement applies on loopback too.
        assert!(check(failures(&s(&[]), true, &p).unwrap(), enforce_limits(&[LO], LO, false), true).is_err());
    }

    #[test]
    fn transport_tls_check_follows_es() {
        let probe = ok_probe();
        let fails = |settings: &Settings, security: bool| failures(settings, security, &probe).unwrap();
        let none = configured();
        // Security on, transport TLS off, non-loopback: refused with ES's message.
        let err = check(fails(&none, true), enforce_limits(&[ANY], SITE, false), false).unwrap_err();
        assert_eq!(err.0, "[1] bootstrap checks failed. You must address the points described in the following [1] lines before starting Elasticsearch. For more information see [https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5]\n\
bootstrap check failure [1] of [1]: Transport SSL must be enabled if security is enabled. Please set [xpack.security.transport.ssl.enabled] to [true] or disable security by setting [xpack.security.enabled] to [false]; for more information see [https://www.elastic.co/docs/deploy-manage/deploy/self-managed/bootstrap-checks?version=9.5#bootstrap-checks-tls]");
        // Explicit enforcement on loopback refuses too.
        assert!(check(fails(&none, true), false, true).is_err());
        // Loopback, single-node or security off: starts.
        assert!(check(fails(&none, true), enforce_limits(&[LO], LO, false), false).is_ok());
        assert!(check(fails(&none, true), enforce_limits(&[ANY], SITE, true), false).is_ok());
        assert!(check(fails(&none, false), true, false).is_ok());
        // Transport TLS on: passes.
        let tls = s(&[("discovery.seed_hosts", "x"), ("xpack.security.transport.ssl.enabled", "true")]);
        assert!(check(fails(&tls, true), true, false).is_ok());
    }
}
