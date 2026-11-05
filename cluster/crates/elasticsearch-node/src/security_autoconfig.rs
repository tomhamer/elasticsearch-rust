//! First-start security auto-configuration, as ES's `AutoConfigureNode`
//! (run by the CLI before the node starts) and
//! `InitialNodeSecurityAutoConfiguration` (run by the node once it leads).
//!
//! On the first start of a new node whose configuration does not mention
//! security, [`run`] generates a transport CA and certificate and an HTTP CA
//! and certificate under `$ES_PATH_CONF/certs/`, and appends ES's
//! `BEGIN SECURITY AUTO CONFIGURATION` block to `elasticsearch.yml`
//! (security, enrollment, HTTP TLS, transport TLS with `verification_mode:
//! certificate`, `cluster.initial_master_nodes` and `http.host: 0.0.0.0`).
//! The skip conditions and their order follow `AutoConfigureNode.execute`
//! and `checkExistingConfiguration`.
//!
//! Deviations: the certificates
//! and keys are PEM files instead of PKCS#12 keystores whose passwords live
//! in `elasticsearch.keystore` (the keys are protected by file mode 0600),
//! and the keys are ECDSA P-256 instead of RSA 4096.
//!
//! With `--enrollment-token` ([`enroll`], ES's enrollment branch of
//! `AutoConfigureNode`) the node joins an existing secured cluster instead:
//! it calls `GET /_security/enroll/node` on the token's addresses (trusting
//! only the HTTP CA the token pins), installs the cluster's transport key,
//! certificate and CA and its HTTP CA, issues its own HTTP certificate from
//! that CA, and writes the block with `discovery.seed_hosts` set to the
//! cluster's transport addresses. Every reason not to auto-configure is an
//! error then (the node does not start).
//!
//! Once the node leads a new cluster with `xpack.security.enrollment.enabled`
//! set, [`spawn_initial_node_password`] generates the `elastic` password and
//! enrollment tokens for Kibana and for other nodes and prints ES's banner
//! (password, HTTP CA fingerprint, tokens) to stdout, only when stdout is a
//! terminal, as ES does.

use crate::config::{get_list, get_str, Settings};
use elasticsearch_cluster::state::Role;
use serde_json::Value;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

pub const AUTO_CONFIGURATION_START_MARKER: &str = "#----------------------- BEGIN SECURITY AUTO CONFIGURATION -----------------------";
pub const AUTO_CONFIGURATION_END_MARKER: &str = "#----------------------- END SECURITY AUTO CONFIGURATION -------------------------";
pub const AUTO_CONFIG_HTTP_ALT_DN: &str = "Elasticsearch security auto-configuration HTTP CA";
pub const AUTO_CONFIG_TRANSPORT_ALT_DN: &str = "Elasticsearch security auto-configuration transport CA";
pub const TLS_GENERATED_CERTS_DIR_NAME: &str = "certs";
pub const KEYSTORE_FILENAME: &str = "elasticsearch.keystore";

/// Generated files in `certs/` (PEM).
pub const HTTP_CA_CERT: &str = "http_ca.crt";
pub const HTTP_CA_KEY: &str = "http_ca.key";
pub const HTTP_CERT: &str = "http.crt";
pub const HTTP_KEY: &str = "http.key";
pub const TRANSPORT_CA_CERT: &str = "transport_ca.crt";
pub const TRANSPORT_CERT: &str = "transport.crt";
pub const TRANSPORT_KEY: &str = "transport.key";

const TRANSPORT_CERTIFICATE_DAYS: i64 = 99 * 365;
const TRANSPORT_CA_CERTIFICATE_DAYS: i64 = 99 * 365;
const HTTP_CA_CERTIFICATE_DAYS: i64 = 3 * 365;
const HTTP_CERTIFICATE_DAYS: i64 = 2 * 365;

/// ES `ExitCodes` used by the auto-configuration.
pub mod exit_codes {
    pub const USAGE: i32 = 64;
    pub const DATA_ERROR: i32 = 65;
    pub const UNAVAILABLE: i32 = 69;
    pub const NOOP: i32 = 63;
    pub const CANT_CREATE: i32 = 73;
    pub const IO_ERROR: i32 = 74;
    pub const CONFIG: i32 = 78;
}

/// Auto-configuration does not apply (ES `notifyOfFailure` outside
/// enrollment mode): the node starts as configured. `verbose` messages are
/// only printed with `-v`/`--verbose` (ES `Terminal.Verbosity.VERBOSE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skip {
    pub exit_code: i32,
    pub message: String,
    pub verbose: bool,
}

/// Auto-configuration failed. ES's `ServerCli` lets the node start anyway
/// for `CANT_CREATE`, `CONFIG` and `NOOP` (printing the message), and fails
/// the start otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoConfigError {
    pub exit_code: i32,
    pub message: String,
}

impl AutoConfigError {
    fn new(exit_code: i32, message: impl Into<String>) -> Self {
        AutoConfigError { exit_code, message: message.into() }
    }

    pub fn start_anyway(&self) -> bool {
        matches!(self.exit_code, exit_codes::CANT_CREATE | exit_codes::CONFIG | exit_codes::NOOP)
    }
}

impl std::fmt::Display for AutoConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Debug)]
pub enum Outcome {
    Configured { certs_dir: PathBuf, http_ca_fingerprint: String },
    Skipped(Skip),
}

fn skip(exit_code: i32, verbose: bool, message: impl Into<String>) -> Option<Skip> {
    Some(Skip { exit_code, message: message.into(), verbose })
}

/// ES `Setting.exists`: the key is set (to anything).
fn exists(settings: &Settings, key: &str) -> bool {
    settings.contains_key(key)
}

/// ES `Settings.hasValue`: the key is set to a non-null value.
fn has_value(settings: &Settings, key: &str) -> bool {
    settings.get(key).is_some_and(|v| !v.is_null())
}

fn is_false(settings: &Settings, key: &str) -> bool {
    match settings.get(key) {
        Some(Value::String(s)) => s == "false",
        Some(Value::Bool(b)) => !*b,
        _ => false,
    }
}

/// `path.data` (a list), default `data` as in [`crate::config::NodeConfig`].
pub fn data_paths(settings: &Settings) -> Vec<PathBuf> {
    let paths = get_list(settings, "path.data");
    if paths.is_empty() { vec![PathBuf::from("data")] } else { paths.into_iter().map(PathBuf::from).collect() }
}

fn dir_is_empty(path: &Path) -> bool {
    std::fs::read_dir(path).map(|mut d| d.next().is_none()).unwrap_or(false)
}

/// ES `AutoConfigureNode.execute` pre-flight checks (data paths, the yml
/// and keystore files) followed by [`check_existing_configuration`].
pub fn preflight(settings: &Settings, conf_dir: &Path) -> Option<Skip> {
    preflight_for(settings, conf_dir, false)
}

/// [`preflight`]; `enrolling`: with `--enrollment-token` (the role checks
/// do not apply).
pub fn preflight_for(settings: &Settings, conf_dir: &Path, enrolling: bool) -> Option<Skip> {
    for data_path in data_paths(settings) {
        if data_path.is_dir() && !dir_is_empty(&data_path) {
            return skip(exit_codes::NOOP, true, "Skipping security auto configuration because it appears that the node is not starting up for the first time. The node might already be part of a cluster and this auto setup utility is designed to configure Security for new clusters only.");
        }
    }
    let yml = conf_dir.join("elasticsearch.yml");
    let regular = std::fs::symlink_metadata(&yml).map(|m| m.file_type().is_file()).unwrap_or(false);
    if !regular {
        return skip(exit_codes::CONFIG, false, format!("Skipping security auto configuration because the configuration file [{}] is missing or is not a regular file", yml.display()));
    }
    if std::fs::File::open(&yml).is_err() {
        return skip(exit_codes::NOOP, false, format!("Skipping security auto configuration because the current user does not have permission to read  configuration file [{}]", yml.display()));
    }
    let keystore = conf_dir.join(KEYSTORE_FILENAME);
    if keystore.exists() {
        let regular = std::fs::symlink_metadata(&keystore).map(|m| m.file_type().is_file()).unwrap_or(false);
        if !regular || std::fs::File::open(&keystore).is_err() {
            return skip(exit_codes::NOOP, false, format!("Skipping security auto configuration because the node keystore file [{}] is not a readable regular file", keystore.display()));
        }
    }
    check_existing_configuration_for(settings, enrolling)
}

/// The node's roles as [`crate::config::NodeConfig`] resolves them
/// (`None`: `node.roles` unset, the default roles).
fn configured_roles(settings: &Settings) -> Option<Vec<Role>> {
    if !settings.contains_key("node.roles") {
        return None;
    }
    Some(get_list(settings, "node.roles").iter().filter_map(|r| Role::parse(r)).collect())
}

/// ES `isInitialClusterNode`: single-node discovery, or no discovery
/// configuration at all, or `cluster.initial_master_nodes` naming only this
/// node.
pub fn is_initial_cluster_node(settings: &Settings) -> bool {
    let initial = get_list(settings, "cluster.initial_master_nodes");
    get_str(settings, "discovery.type") == Some("single-node")
        || (initial.is_empty() && get_list(settings, "discovery.seed_hosts").is_empty() && get_list(settings, "discovery.seed_providers").is_empty())
        || initial == vec![get_str(settings, "node.name").unwrap_or("").to_string()]
}

/// ES `checkExistingConfiguration` (non-enrollment mode).
pub fn check_existing_configuration(settings: &Settings) -> Option<Skip> {
    check_existing_configuration_for(settings, false)
}

/// ES `checkExistingConfiguration`: an enrolling node skips the role checks.
pub fn check_existing_configuration_for(settings: &Settings, enrolling: bool) -> Option<Skip> {
    if is_false(settings, "xpack.security.autoconfiguration.enabled") {
        return skip(exit_codes::NOOP, true, "Skipping security auto configuration because [xpack.security.autoconfiguration.enabled] is false");
    }
    if exists(settings, "xpack.security.enabled") {
        return skip(exit_codes::NOOP, true, "Skipping security auto configuration because it appears that security is already configured.");
    }
    if settings.keys().any(|k| k.starts_with("xpack.security.transport.ssl.") || k.starts_with("xpack.security.http.ssl.")) {
        return skip(exit_codes::NOOP, true, "Skipping security auto configuration because it appears that TLS is already configured.");
    }
    if !is_initial_cluster_node(settings) {
        return skip(exit_codes::NOOP, true, "Skipping security auto configuration because this node is configured to bootstrap or to join a multi-node cluster, which is not supported.");
    }
    if let Some(roles) = configured_roles(settings).filter(|_| !enrolling) {
        if !roles.contains(&Role::Master) || roles.contains(&Role::VotingOnly) {
            return skip(exit_codes::NOOP, true, "Skipping security auto configuration because the node is configured such that it cannot become master.");
        }
        if !roles.iter().any(|r| r.is_data()) {
            return skip(exit_codes::NOOP, true, "Skipping security auto configuration because the node is configured such that it cannot contain data.");
        }
    }
    None
}

/// The `elasticsearch.yml` block (ES writes it after the existing lines and
/// one empty line). `date` is `dd-MM-yyyy HH:mm:ss` (UTC).
pub fn config_block(settings: &Settings, date: &str) -> String {
    config_block_for(settings, date, None)
}

/// An enrolling node's block: the cluster's transport addresses, and whether
/// one of them is on another host (`transport.host` is then uncommented).
pub struct EnrollmentBlock<'a> {
    pub transport_addresses: &'a [String],
    pub remote_host: bool,
}

/// [`config_block`], for an enrolling node with `enrollment`.
pub fn config_block_for(settings: &Settings, date: &str, enrollment: Option<EnrollmentBlock<'_>>) -> String {
    let certs = TLS_GENERATED_CERTS_DIR_NAME;
    let mut b = String::new();
    let mut line = |s: &str| {
        b.push_str(s);
        b.push('\n');
    };
    line(AUTO_CONFIGURATION_START_MARKER);
    line("#");
    line("# The following settings, TLS certificates, and keys have been automatically      ");
    line(&format!("# generated to configure Elasticsearch security features on {date}"));
    line("#");
    line("# --------------------------------------------------------------------------------");
    line("");
    line("# Enable security features");
    line("xpack.security.enabled: true");
    line("");
    // Enrollment mode unless set explicitly (ES writes it unless set to
    // false; an explicit `true` is not repeated here, which would duplicate
    // a key of the same file).
    if !has_value(settings, "xpack.security.enrollment.enabled") {
        line("xpack.security.enrollment.enabled: true");
        line("");
    }
    line("# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents");
    line("xpack.security.http.ssl:");
    line("  enabled: true");
    line(&format!("  key: {certs}/{HTTP_KEY}"));
    line(&format!("  certificate: {certs}/{HTTP_CERT}"));
    line("");
    line("# Enable encryption and mutual authentication between cluster nodes");
    line("xpack.security.transport.ssl:");
    line("  enabled: true");
    line("  verification_mode: certificate");
    line(&format!("  key: {certs}/{TRANSPORT_KEY}"));
    line(&format!("  certificate: {certs}/{TRANSPORT_CERT}"));
    // As ES's last transport line: terminated by what follows.
    b.push_str(&format!("  certificate_authorities: [ \"{certs}/{TRANSPORT_CA_CERT}\" ]"));
    if let Some(e) = &enrollment {
        b.push('\n');
        b.push_str("# Discover existing nodes in the cluster\n");
        let hosts: Vec<String> = e.transport_addresses.iter().map(|a| format!("\"{a}\"")).collect();
        b.push_str(&format!("discovery.seed_hosts: [{}]\n", hosts.join(", ")));
    }
    let mut line = |s: &str| {
        b.push_str(s);
        b.push('\n');
    };
    // New certificates on the transport layer: this node cannot form a
    // multi-node cluster; without this the discovery bootstrap check trips.
    if enrollment.is_none() && get_str(settings, "discovery.type") != Some("single-node") && !exists(settings, "cluster.initial_master_nodes") {
        line("");
        line("# Create a new cluster with the current node only");
        line("# Additional nodes can still join the cluster later");
        let value = match get_str(settings, "node.name") {
            Some(name) if exists(settings, "node.name") => format!("[\"{name}\"]"),
            _ => "[\"${HOSTNAME}\"]".to_string(),
        };
        line(&format!("cluster.initial_master_nodes: {value}"));
    }
    let any_set = |keys: &[&str]| keys.iter().any(|k| has_value(settings, k));
    let network = ["network.host", "network.bind_host", "network.publish_host"];
    if !any_set(&["http.host", "http.bind_host", "http.publish_host"]) && !any_set(&network) {
        line("");
        line("# Allow HTTP API connections from anywhere");
        line("# Connections are encrypted and require user authentication");
        line("http.host: 0.0.0.0");
    }
    if !any_set(&["transport.host", "transport.bind_host", "transport.publish_host"]) && !any_set(&network) {
        line("");
        line("# Allow other nodes to join the cluster from anywhere");
        line("# Connections are encrypted and mutually authenticated");
        if enrollment.as_ref().is_some_and(|e| e.remote_host) {
            line("transport.host: 0.0.0.0");
        } else {
            line("#transport.host: 0.0.0.0");
        }
    }
    line("");
    line(AUTO_CONFIGURATION_END_MARKER);
    b
}

/// The new `elasticsearch.yml` text: the existing lines, an empty line and
/// the block.
pub fn append_block(existing: &str, block: &str) -> String {
    let mut out = String::with_capacity(existing.len() + block.len() + 2);
    out.push_str(existing);
    if !existing.is_empty() && !existing.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(block);
    out
}

/// Every address of every local interface (ES `NetworkUtils.getAllAddresses`).
pub fn all_addresses() -> Vec<IpAddr> {
    let mut out = Vec::new();
    #[cfg(unix)]
    {
        let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
        // SAFETY: getifaddrs fills `head` with a list freed below; every
        // address is read according to its family.
        unsafe {
            if libc::getifaddrs(&mut head) == 0 {
                let mut cur = head;
                while !cur.is_null() {
                    let addr = (*cur).ifa_addr;
                    if !addr.is_null() {
                        let family = (*addr).sa_family as i32;
                        if family == libc::AF_INET {
                            let sin = addr as *const libc::sockaddr_in;
                            out.push(IpAddr::V4(std::net::Ipv4Addr::from(u32::from_be((*sin).sin_addr.s_addr))));
                        } else if family == libc::AF_INET6 {
                            let sin6 = addr as *const libc::sockaddr_in6;
                            out.push(IpAddr::V6(std::net::Ipv6Addr::from((*sin6).sin6_addr.s6_addr)));
                        }
                    }
                    cur = (*cur).ifa_next;
                }
                libc::freeifaddrs(head);
            }
        }
    }
    if out.is_empty() {
        out.push(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    }
    out.sort();
    out.dedup();
    out
}

/// ES `getSubjectAltNames`: every local address, `localhost`, `$HOSTNAME`,
/// and the `network.publish_host` / `http.publish_host` values.
pub fn subject_alt_names(settings: &Settings, addresses: &[IpAddr], hostname: &str) -> Vec<rcgen::SanType> {
    let mut ips: Vec<IpAddr> = addresses.to_vec();
    let mut dns: Vec<String> = vec!["localhost".into(), hostname.to_string()];
    for key in ["network.publish_host", "http.publish_host"] {
        for host in get_list(settings, key) {
            match host.parse::<IpAddr>() {
                Ok(ip) => ips.push(ip),
                Err(_) => dns.push(host),
            }
        }
    }
    ips.sort();
    ips.dedup();
    dns.retain(|d| !d.is_empty());
    dns.sort();
    dns.dedup();
    let mut out: Vec<rcgen::SanType> = ips.into_iter().map(rcgen::SanType::IpAddress).collect();
    out.extend(dns.into_iter().filter_map(|d| rcgen::Ia5String::try_from(d).ok().map(rcgen::SanType::DnsName)));
    out
}

/// One generated file: name in `certs/`, PEM text, file mode.
pub struct GeneratedFile {
    pub name: &'static str,
    pub pem: String,
    pub mode: u32,
}

pub struct GeneratedCerts {
    pub files: Vec<GeneratedFile>,
    /// Hex SHA-256 of the HTTP CA certificate (DER), as ES prints it.
    pub http_ca_fingerprint: String,
}

fn ymd(t: chrono::DateTime<chrono::Utc>) -> (i32, u8, u8) {
    use chrono::Datelike;
    (t.year(), t.month() as u8, t.day() as u8)
}

fn cert_params(cn: &str, now: chrono::DateTime<chrono::Utc>, days: i64) -> rcgen::CertificateParams {
    let mut p = rcgen::CertificateParams::default();
    let mut dn = rcgen::DistinguishedName::new();
    dn.push(rcgen::DnType::CommonName, cn.to_string());
    p.distinguished_name = dn;
    let (y, m, d) = ymd(now);
    p.not_before = rcgen::date_time_ymd(y, m, d);
    let (y, m, d) = ymd(now + chrono::Duration::days(days));
    p.not_after = rcgen::date_time_ymd(y, m, d);
    p
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The transport CA + certificate and the HTTP CA + certificate.
pub fn generate_certs(cn: &str, http_sans: Vec<rcgen::SanType>, now: chrono::DateTime<chrono::Utc>) -> Result<GeneratedCerts, rcgen::Error> {
    use rcgen::{BasicConstraints, ExtendedKeyUsagePurpose as Eku, IsCa, KeyPair, KeyUsagePurpose as Ku};
    // Transport: a self-signed CA and a node certificate (no names: transport
    // verification is `certificate`, chain only).
    let transport_ca_key = KeyPair::generate()?;
    let mut p = cert_params(AUTO_CONFIG_TRANSPORT_ALT_DN, now, TRANSPORT_CA_CERTIFICATE_DAYS);
    p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    p.key_usages = vec![Ku::KeyCertSign, Ku::CrlSign, Ku::DigitalSignature];
    let transport_ca = p.self_signed(&transport_ca_key)?;
    let transport_key = KeyPair::generate()?;
    let mut p = cert_params(cn, now, TRANSPORT_CERTIFICATE_DAYS);
    p.key_usages = vec![Ku::DigitalSignature];
    p.extended_key_usages = vec![Eku::ServerAuth, Eku::ClientAuth];
    let transport_cert = p.signed_by(&transport_key, &transport_ca, &transport_ca_key)?;
    // HTTP: a self-signed CA and a server certificate for every local name.
    let http_ca_key = KeyPair::generate()?;
    let mut p = cert_params(AUTO_CONFIG_HTTP_ALT_DN, now, HTTP_CA_CERTIFICATE_DAYS);
    p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    p.key_usages = vec![Ku::KeyCertSign, Ku::CrlSign, Ku::DigitalSignature];
    let http_ca = p.self_signed(&http_ca_key)?;
    let (http_cert, http_key) = http_certificate(cn, http_sans, now, &http_ca, &http_ca_key)?;
    let http_ca_der: &[u8] = http_ca.der();
    let http_ca_fingerprint = hex(&elasticsearch_engine::security::hash::sha256(http_ca_der));
    Ok(GeneratedCerts {
        files: cert_files(
            transport_ca.pem(),
            transport_cert.pem(),
            transport_key.serialize_pem(),
            http_ca.pem(),
            http_ca_key.serialize_pem(),
            http_cert.pem(),
            http_key.serialize_pem(),
        ),
        http_ca_fingerprint,
    })
}

/// The node's HTTP server certificate, issued by the HTTP CA.
pub fn http_certificate(cn: &str, http_sans: Vec<rcgen::SanType>, now: chrono::DateTime<chrono::Utc>, ca: &rcgen::Certificate, ca_key: &rcgen::KeyPair) -> Result<(rcgen::Certificate, rcgen::KeyPair), rcgen::Error> {
    use rcgen::{ExtendedKeyUsagePurpose as Eku, KeyPair, KeyUsagePurpose as Ku};
    let http_key = KeyPair::generate()?;
    let mut p = cert_params(cn, now, HTTP_CERTIFICATE_DAYS);
    p.subject_alt_names = http_sans;
    p.key_usages = vec![Ku::DigitalSignature];
    p.extended_key_usages = vec![Eku::ServerAuth];
    let http_cert = p.signed_by(&http_key, ca, ca_key)?;
    Ok((http_cert, http_key))
}

/// The files of `certs/` (PEM texts).
fn cert_files(transport_ca: String, transport_cert: String, transport_key: String, http_ca: String, http_ca_key: String, http_cert: String, http_key: String) -> Vec<GeneratedFile> {
    let cert = |name, pem| GeneratedFile { name, pem, mode: 0o660 };
    let key = |name, pem| GeneratedFile { name, pem, mode: 0o600 };
    vec![
        cert(TRANSPORT_CA_CERT, transport_ca),
        cert(TRANSPORT_CERT, transport_cert),
        key(TRANSPORT_KEY, transport_key),
        cert(HTTP_CA_CERT, http_ca.clone()),
        key(HTTP_CA_KEY, http_ca_key),
        // The chain (leaf, CA): clients pin or trust the CA from it.
        cert(HTTP_CERT, format!("{http_cert}{http_ca}")),
        key(HTTP_KEY, http_key),
    ]
}

fn write_new_file(path: &Path, contents: &[u8], mode: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(mode);
    }
    let mut f = options.open(path)?;
    f.write_all(contents)?;
    f.sync_all()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    }
    #[cfg(not(unix))]
    let _ = mode;
    Ok(())
}

/// Replace `path` with `contents` through a temporary file in the same
/// directory, keeping the replaced file's permissions (ES `fullyWriteFile`).
pub fn replace_file(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!("{name}.{}.tmp", hex(&elasticsearch_engine::security::hash::random_bytes(8))));
    let mode = file_mode(path);
    let result = write_new_file(&tmp, contents, mode).and_then(|_| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
fn file_mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).map(|m| m.permissions().mode() & 0o7777).unwrap_or(0o660)
}

#[cfg(not(unix))]
fn file_mode(_: &Path) -> u32 {
    0o660
}

#[cfg(unix)]
fn owner(path: &Path) -> std::io::Result<u32> {
    use std::os::unix::fs::MetadataExt;
    Ok(std::fs::symlink_metadata(path)?.uid())
}

/// The certificate common name: `node.name`, else `$HOSTNAME`.
fn common_name(settings: &Settings) -> String {
    get_str(settings, "node.name").map(str::to_string).unwrap_or_else(crate::config::hostname)
}

/// Run the auto-configuration (see the module docs) now.
pub fn run(settings: &Settings, conf_dir: &Path) -> Result<Outcome, AutoConfigError> {
    run_at(settings, conf_dir, chrono::Utc::now(), &all_addresses())
}

/// Pre-create the temporary certificate directory (checks that the config
/// dir is writable and that file owners match).
fn create_tmp_dir(conf_dir: &Path, epoch: i64) -> Result<PathBuf, AutoConfigError> {
    let tmp_dir = conf_dir.join(format!("{TLS_GENERATED_CERTS_DIR_NAME}.{epoch}.tmp"));
    let created = std::fs::create_dir(&tmp_dir).and_then(|_| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp_dir, std::fs::Permissions::from_mode(0o750))?;
        }
        Ok(())
    });
    if let Err(e) = created {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return Err(AutoConfigError::new(exit_codes::CANT_CREATE, format!("Could not create auto-configuration directory: {e}")));
    }
    #[cfg(unix)]
    {
        let (new_owner, conf_owner) = (owner(&tmp_dir), owner(conf_dir));
        if let (Ok(new_owner), Ok(conf_owner)) = (new_owner, conf_owner) {
            if new_owner != conf_owner {
                let _ = std::fs::remove_dir_all(&tmp_dir);
                return Err(AutoConfigError::new(exit_codes::CONFIG, format!(
                    "Aborting auto configuration because of config dir ownership mismatch. Config dir is owned by {conf_owner} but auto-configuration directory would be owned by {new_owner}")));
            }
        }
    }
    Ok(tmp_dir)
}

/// Write `files` into `tmp_dir`, move it into place as `certs/` (a previous
/// `certs/` is kept as a backup) and append `block` to `elasticsearch.yml`.
fn install(conf_dir: &Path, tmp_dir: &Path, epoch: i64, files: &[GeneratedFile], block: &str) -> Result<PathBuf, AutoConfigError> {
    for f in files {
        if let Err(e) = write_new_file(&tmp_dir.join(f.name), f.pem.as_bytes(), f.mode) {
            let _ = std::fs::remove_dir_all(tmp_dir);
            return Err(AutoConfigError::new(1, format!("Aborting auto configuration: failed to write [{}]: {e}", f.name)));
        }
    }
    let certs_dir = conf_dir.join(TLS_GENERATED_CERTS_DIR_NAME);
    let backup = conf_dir.join(format!("{TLS_GENERATED_CERTS_DIR_NAME}.{epoch}.orig"));
    let restore = |remove_new: bool| {
        if remove_new {
            let _ = std::fs::remove_dir_all(&certs_dir);
        }
        if backup.exists() {
            let _ = std::fs::rename(&backup, &certs_dir);
        }
        let _ = std::fs::remove_dir_all(tmp_dir);
    };
    if certs_dir.exists() {
        if let Err(e) = std::fs::rename(&certs_dir, &backup) {
            restore(false);
            return Err(AutoConfigError::new(exit_codes::IO_ERROR, format!("Aborting auto configuration: cannot back up [{}]: {e}", certs_dir.display())));
        }
    }
    if let Err(e) = std::fs::rename(tmp_dir, &certs_dir) {
        restore(false);
        return Err(AutoConfigError::new(exit_codes::IO_ERROR, format!("Aborting auto configuration: cannot create [{}]: {e}", certs_dir.display())));
    }
    let yml = conf_dir.join("elasticsearch.yml");
    let written = std::fs::read_to_string(&yml).and_then(|existing| replace_file(&yml, append_block(&existing, block).as_bytes()));
    if let Err(e) = written {
        restore(true);
        return Err(AutoConfigError::new(exit_codes::IO_ERROR, format!("Aborting auto configuration: cannot update [{}]: {e}", yml.display())));
    }
    Ok(certs_dir)
}

fn date_text(now: chrono::DateTime<chrono::Utc>) -> String {
    now.format("%d-%m-%Y %H:%M:%S").to_string()
}

fn subject_alt_names_now(settings: &Settings, addresses: &[IpAddr]) -> Vec<rcgen::SanType> {
    let hostname = std::env::var("HOSTNAME").ok().filter(|h| !h.is_empty()).unwrap_or_else(crate::config::hostname);
    subject_alt_names(settings, addresses, &hostname)
}

/// [`run`] at `now`, with these local addresses (tests).
pub fn run_at(settings: &Settings, conf_dir: &Path, now: chrono::DateTime<chrono::Utc>, addresses: &[IpAddr]) -> Result<Outcome, AutoConfigError> {
    if let Some(s) = preflight(settings, conf_dir) {
        return Ok(Outcome::Skipped(s));
    }
    let epoch = now.timestamp();
    let tmp_dir = create_tmp_dir(conf_dir, epoch)?;
    let generated = match generate_certs(&common_name(settings), subject_alt_names_now(settings, addresses), now) {
        Ok(g) => g,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err(AutoConfigError::new(1, format!("Aborting auto configuration: {e}")));
        }
    };
    let block = config_block(settings, &date_text(now));
    let certs_dir = install(conf_dir, &tmp_dir, epoch, &generated.files, &block)?;
    Ok(Outcome::Configured { certs_dir, http_ca_fingerprint: generated.http_ca_fingerprint })
}

/// `--enrollment-token`: [`enroll_at`] now, on its own runtime (the CLI
/// runs it before the node's runtime exists).
pub fn enroll_blocking(settings: &Settings, conf_dir: &Path, token: &str, verbose: bool) -> Result<Outcome, AutoConfigError> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()
        .map_err(|e| AutoConfigError::new(1, format!("Aborting auto configuration: {e}")))?;
    rt.block_on(enroll_at(settings, conf_dir, token, verbose, chrono::Utc::now(), &all_addresses()))
}

/// ES `AutoConfigureNode` in enrollment mode: see the module docs.
/// `verbose` prints why a token does not parse.
pub async fn enroll_at(settings: &Settings, conf_dir: &Path, token: &str, verbose: bool, now: chrono::DateTime<chrono::Utc>, addresses: &[IpAddr]) -> Result<Outcome, AutoConfigError> {
    use base64::Engine;
    use crate::enrollment::{self, EnrollmentToken};
    // Every reason not to auto-configure fails the start.
    if let Some(s) = preflight_for(settings, conf_dir, true) {
        return Err(AutoConfigError::new(s.exit_code, s.message));
    }
    let epoch = now.timestamp();
    let tmp_dir = create_tmp_dir(conf_dir, epoch)?;
    let fail = |code: i32, message: String| {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        AutoConfigError::new(code, message)
    };
    let token = match EnrollmentToken::decode(token) {
        Ok(t) => t,
        Err(e) => {
            if verbose {
                eprintln!();
                eprintln!("Failed to parse enrollment token : {token} . Error was: {e}");
                eprintln!();
            }
            return Err(fail(exit_codes::DATA_ERROR, "Aborting auto configuration. Invalid enrollment token".into()));
        }
    };
    let client = enrollment::pinned_client(&token.fingerprint).map_err(|e| fail(1, format!("Aborting auto configuration: {e}")))?;
    let authorization = format!("ApiKey {}", base64::engine::general_purpose::STANDARD.encode(token.api_key.as_bytes()));
    // No cluster health wait: a token means the first node is up.
    let mut response: Option<(u16, String)> = None;
    let mut enroll_url = String::new();
    for address in &token.bound_address {
        enroll_url = crate::reset_password::create_url(&format!("https://{address}"), "/_security/enroll/node", "");
        let sent = client.get(&enroll_url).header("Authorization", &authorization).header("Content-Type", "application/json").send().await;
        let received = match sent {
            Ok(r) => {
                let status = r.status().as_u16();
                r.text().await.map(|body| (status, body))
            }
            Err(e) => Err(e),
        };
        match received {
            Ok(r) => {
                response = Some(r);
                break;
            }
            Err(e) => eprintln!("Unable to communicate with the node on {enroll_url}. Error was {e}"),
        }
    }
    let body = match response {
        Some((200, body)) => body,
        _ => {
            return Err(fail(exit_codes::UNAVAILABLE, format!(
                "Aborting enrolling to cluster. Could not communicate with the node on any of the addresses from the enrollment token. All of [{}] were attempted.",
                token.bound_address.join(", "))));
        }
    };
    let Some(Value::Object(map)) = serde_json::from_str::<Value>(&body).ok() else {
        return Err(fail(exit_codes::DATA_ERROR, format!("Aborting enrolling to cluster. Empty response when calling the enroll node API ({enroll_url})")));
    };
    let field = |k: &str| map.get(k).and_then(Value::as_str).filter(|v| !v.is_empty()).map(str::to_string);
    let mut missing = Vec::new();
    for k in ["http_ca_key", "http_ca_cert", "transport_key", "transport_ca_cert", "transport_cert"] {
        if field(k).is_none() {
            missing.push(k);
        }
    }
    let transport_addresses: Vec<String> = map.get("nodes_addresses").and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    if transport_addresses.is_empty() {
        missing.push("nodes_addresses");
    }
    if !missing.is_empty() {
        return Err(fail(exit_codes::DATA_ERROR, format!(
            "Aborting enrolling to cluster. Invalid response when calling the enroll node API ({enroll_url}). The following fields were empty or missing : [{}]", missing.join(", "))));
    }
    let der = |k: &str| base64::engine::general_purpose::STANDARD.decode(field(k).unwrap_or_default().trim()).ok();
    let certificate = |k: &str| -> Result<enrollment::CertInfo, AutoConfigError> {
        der(k).and_then(|d| enrollment::parse_certificate(&d)).ok_or_else(|| {
            if verbose {
                eprintln!();
                eprintln!("Failed to parse Certificate from the response of the Enroll Node API: [{k}] is not a DER certificate");
                eprintln!();
            }
            AutoConfigError::new(exit_codes::DATA_ERROR, "Aborting enrolling to cluster. Failed to parse Certificate from the response of the Enroll Node API")
        })
    };
    let private_key = |k: &str| -> Result<(rcgen::KeyPair, Vec<u8>), AutoConfigError> {
        der(k).and_then(|d| rcgen::KeyPair::try_from(d.as_slice()).ok().map(|kp| (kp, d))).ok_or_else(|| {
            if verbose {
                eprintln!();
                eprintln!("Failed to parse Private Key from the response of the Enroll Node API: [{k}] is not a PKCS#8 key");
                eprintln!();
            }
            AutoConfigError::new(exit_codes::DATA_ERROR, "Aborting enrolling to cluster. Failed to parse Private Key from the response of the Enroll Node API")
        })
    };
    let parsed = (|| Ok::<_, AutoConfigError>((certificate("transport_ca_cert")?, private_key("http_ca_key")?, certificate("http_ca_cert")?, private_key("transport_key")?, certificate("transport_cert")?)))();
    let (transport_ca, (http_ca_key, http_ca_key_der), http_ca, (_, transport_key_der), transport_cert) = match parsed {
        Ok(p) => p,
        Err(e) => return Err(fail(e.exit_code, e.message)),
    };
    // This node's HTTP certificate, issued by the cluster's HTTP CA.
    let issued = enrollment::issuer_certificate(&http_ca, &http_ca_key)
        .and_then(|issuer| http_certificate(&common_name(settings), subject_alt_names_now(settings, addresses), now, &issuer, &http_ca_key).map_err(|e| e.to_string()));
    let (http_cert, http_key) = match issued {
        Ok(c) => c,
        Err(e) => return Err(fail(1, format!("Aborting auto configuration: {e}"))),
    };
    let pem_cert = |c: &enrollment::CertInfo| enrollment::pem_encode("CERTIFICATE", &c.der);
    let files = cert_files(
        pem_cert(&transport_ca),
        pem_cert(&transport_cert),
        enrollment::pem_encode("PRIVATE KEY", &transport_key_der),
        pem_cert(&http_ca),
        enrollment::pem_encode("PRIVATE KEY", &http_ca_key_der),
        http_cert.pem(),
        http_key.serialize_pem(),
    );
    let remote_host = enrollment::any_remote_host_node_address(&transport_addresses, addresses);
    let block = config_block_for(settings, &date_text(now), Some(EnrollmentBlock { transport_addresses: &transport_addresses, remote_host }));
    let certs_dir = install(conf_dir, &tmp_dir, epoch, &files, &block)?;
    let http_ca_fingerprint = hex(&elasticsearch_engine::security::hash::sha256(&http_ca.der));
    Ok(Outcome::Configured { certs_dir, http_ca_fingerprint })
}

// ------------------------------------------------------------ PEM helpers

/// The DER bytes of every `CERTIFICATE` block of a PEM text.
pub fn pem_certificates(text: &str) -> Vec<Vec<u8>> {
    use base64::Engine;
    let mut out = Vec::new();
    let mut body: Option<String> = None;
    for line in text.lines().map(str::trim) {
        if line == "-----BEGIN CERTIFICATE-----" {
            body = Some(String::new());
        } else if line == "-----END CERTIFICATE-----" {
            if let Some(b) = body.take() {
                if let Ok(der) = base64::engine::general_purpose::STANDARD.decode(b.as_bytes()) {
                    out.push(der);
                }
            }
        } else if let Some(b) = body.as_mut() {
            b.push_str(line);
        }
    }
    out
}

/// ES `getHttpsCaFingerprint`: the hex SHA-256 of the CA certificate of the
/// HTTP certificate chain (`xpack.security.http.ssl.certificate`, leaf
/// first). `None` without HTTPS or without a CA in the chain.
pub fn https_ca_fingerprint(settings: &Settings) -> Option<String> {
    if get_str(settings, "xpack.security.http.ssl.enabled") != Some("true") {
        return None;
    }
    let cert = PathBuf::from(get_str(settings, "xpack.security.http.ssl.certificate")?);
    let conf = PathBuf::from(get_str(settings, "path.conf").unwrap_or("config"));
    let path = if cert.is_absolute() { cert } else { conf.join(cert) };
    let chain = pem_certificates(&std::fs::read_to_string(path).ok()?);
    if chain.len() < 2 {
        return None;
    }
    chain.last().map(|ca| hex(&elasticsearch_engine::security::hash::sha256(ca)))
}

// ------------------------------------------------- initial node password

/// ES `CommandUtils.generatePassword`.
pub fn generate_password(len: usize) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789*-_=+";
    random_from(CHARS, len)
}

/// ES `CommandUtils.generateUsername` body characters.
pub fn random_alphanumeric(len: usize) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    random_from(CHARS, len)
}

fn random_from(chars: &[u8], len: usize) -> String {
    // Rejection sampling over the OS RNG: no modulo bias.
    let limit = 256 - (256 % chars.len());
    let mut out = String::with_capacity(len);
    while out.len() < len {
        for b in elasticsearch_engine::security::hash::random_bytes(len * 2) {
            if (b as usize) < limit && out.len() < len {
                out.push(chars[b as usize % chars.len()] as char);
            }
        }
    }
    out
}

/// ES `InitialNodeSecurityAutoConfiguration.outputInformationToConsole`.
/// `password`: `None` = generation failed, `Some("")` = not generated.
/// `kibana_token`: `None` = not generated. `node_token`: `None` = failed,
/// `Some("")` = skipped (only loopback addresses), else the token.
pub fn banner(password: Option<&str>, ca_fingerprint: Option<&str>, kibana_token: Option<&str>, node_token: Option<&str>, unicode: bool, ansi: bool, width: usize) -> String {
    let info = if unicode { "\u{2139}\u{FE0F}" } else { "->" };
    let bullet = if unicode { "\u{2022}" } else { "*" };
    let hyphen = if unicode { "\u{2043}" } else { "-" };
    let error = if unicode { "\u{274C}" } else { "X" };
    let success = if unicode { "\u{2705}" } else { "->" };
    let border = if unicode { "\u{2501}" } else { "-" };
    let (bold_on, bold_off) = if ansi { ("\u{1B}[1m", "\u{1B}[22m") } else { ("", "") };
    let nl = "\n";
    let mut b = String::new();
    b.push_str(&nl.repeat(4));
    b.push_str(&border.repeat(width));
    b.push_str(nl);
    b.push_str(&format!("{success} Elasticsearch security features have been automatically configured!{nl}"));
    b.push_str(&format!("{success} Authentication is enabled and cluster connections are encrypted.{nl}"));
    b.push_str(nl);
    match password {
        None => b.push_str(&format!("{error} Unable to auto-generate the password for the {bold_on}elastic{bold_off} built-in superuser.")),
        Some("") => {}
        Some(p) => {
            b.push_str(&format!("{info}  Password for the {bold_on}elastic{bold_off} user (reset with `bin/elasticsearch-reset-password -u elastic`):{nl}"));
            b.push_str(&format!("  {bold_on}{p}{bold_off}"));
        }
    }
    b.push_str(nl);
    b.push_str(nl);
    if let Some(f) = ca_fingerprint {
        b.push_str(&format!("{info}  HTTP CA certificate SHA-256 fingerprint:{nl}"));
        b.push_str(&format!("  {bold_on}{f}{bold_off}"));
    }
    b.push_str(nl);
    b.push_str(nl);
    match kibana_token {
        Some(t) => {
            b.push_str(&format!("{info}  Configure Kibana to use this cluster:{nl}"));
            b.push_str(&format!("{bullet} Run Kibana and click the configuration link in the terminal when Kibana starts.{nl}"));
            b.push_str(&format!("{bullet} Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):{nl}"));
            b.push_str(&format!("  {bold_on}{t}{bold_off}"));
        }
        None => b.push_str(&format!("{error} Unable to generate an enrollment token for Kibana instances, try invoking `bin/elasticsearch-create-enrollment-token -s kibana`.")),
    }
    b.push_str(nl);
    b.push_str(nl);
    match node_token {
        Some(t) if !t.is_empty() => {
            b.push_str(&format!("{info} Configure other nodes to join this cluster:{nl}"));
            b.push_str(&format!("{bullet} Copy the following enrollment token and start new Elasticsearch nodes with `bin/elasticsearch --enrollment-token <token>` (valid for the next 30 minutes):{nl}"));
            b.push_str(&format!("  {bold_on}{t}{bold_off}{nl}"));
            b.push_str(nl);
            b.push_str(&format!("  If you're running in Docker, copy the enrollment token and run:{nl}"));
            b.push_str(&format!("  `docker run -e \"ENROLLMENT_TOKEN=<token>\" docker.elastic.co/elasticsearch/elasticsearch:{}`", crate::node::ES_VERSION));
        }
        _ => {
            if node_token.is_some() {
                b.push_str(&format!("{info}  Configure other nodes to join this cluster:{nl}"));
                b.push_str(&format!("{bullet} On this node:{nl}"));
                b.push_str(&format!("  {hyphen} Create an enrollment token with `bin/elasticsearch-create-enrollment-token -s node`.{nl}"));
                b.push_str(&format!("  {hyphen} Uncomment the {bold_on}transport.host{bold_off} setting at the end of {bold_on}config/elasticsearch.yml{bold_off}.{nl}"));
            } else {
                b.push_str(&format!("{error} An enrollment token to enroll new nodes wasn't generated. To add nodes and enroll them into this cluster:{nl}"));
                b.push_str(&format!("{bullet} On this node:{nl}"));
                b.push_str(&format!("  {hyphen} Create an enrollment token with `bin/elasticsearch-create-enrollment-token -s node`.{nl}"));
            }
            b.push_str(&format!("  {hyphen} Restart Elasticsearch.{nl}"));
            b.push_str(&format!("{bullet} On other nodes:{nl}"));
            b.push_str(&format!("  {hyphen} Start Elasticsearch with `bin/elasticsearch --enrollment-token <token>`, using the enrollment token that you generated."));
        }
    }
    b.push_str(nl);
    b.push_str(&border.repeat(width));
    b.push_str(&nl.repeat(4));
    b
}

/// `ip:port` with the port of `published` (`host:port`).
fn bound_with_port(ip: IpAddr, published: &str) -> Option<String> {
    let port = published.rsplit(':').next()?.parse::<u16>().ok()?;
    Some(std::net::SocketAddr::new(ip, port).to_string())
}

/// ES `InternalEnrollmentTokenGenerator`: the Kibana token and the node
/// token (`Some("")` when the node binds only loopback addresses for
/// transport or HTTP, `None` when creating it failed).
async fn enrollment_tokens(node: &std::sync::Arc<crate::node::ClusterNode>) -> (Option<String>, Option<String>) {
    use crate::enrollment::{create_internal_token, split_addresses, KIBANA_ENROLLMENT_ACTION, NODE_ENROLLMENT_ACTION};
    let local = &node.local;
    let http: Vec<String> = std::iter::once(local.http_address.clone()).chain(bound_with_port(node.config.http_bind, &local.http_address)).collect();
    let transport: Vec<String> = std::iter::once(local.transport_address.clone()).chain(bound_with_port(node.config.transport_bind, &local.transport_address)).collect();
    let kibana = match create_internal_token(node, KIBANA_ENROLLMENT_ACTION, &http).await {
        Ok(t) => Some(t.encoded()),
        Err(e) => {
            tracing::error!("{e}");
            None
        }
    };
    let node_token = if split_addresses(&transport).1.is_empty() {
        tracing::info!("Will not generate node enrollment token because node is only bound on localhost for transport and cannot connect to nodes from other hosts");
        Some(String::new())
    } else if split_addresses(&http).1.is_empty() {
        tracing::info!("Will not generate node enrollment token because node is only bound on localhost for HTTPS and cannot enroll nodes from other hosts");
        Some(String::new())
    } else {
        match create_internal_token(node, NODE_ENROLLMENT_ACTION, &http).await {
            Ok(t) => Some(t.encoded()),
            Err(e) => {
                tracing::error!("{e}");
                None
            }
        }
    };
    (kibana, node_token)
}

/// The terminal width (ES `console.width()`), 80 when unknown.
fn terminal_width() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills a `winsize` for a terminal fd.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ as _, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as usize;
            }
        }
    }
    80
}

/// Whether the elastic user's password has never been set and no native
/// user exists (ES: the `.security` index does not exist yet).
pub fn security_state_is_new(security: &elasticsearch_engine::security::Security) -> bool {
    let st = security.state.read();
    st.users.is_empty() && st.reserved.values().all(|r| r.password_hash.is_none())
}

/// ES `InitialNodeSecurityAutoConfiguration`: on a node with
/// `xpack.security.enrollment.enabled` (written by the auto-configuration),
/// once it leads a cluster whose security state is new, generate the
/// `elastic` password (unless `bootstrap.password` or
/// `autoconfiguration.password_hash` is set) and print the banner to stdout.
/// Only when stdout is a terminal, as ES.
pub fn spawn_initial_node_password(node: &std::sync::Arc<crate::node::ClusterNode>) {
    let settings = &node.config.settings;
    if get_str(settings, "xpack.security.enrollment.enabled") != Some("true") || !node.engine.security.enabled() {
        return;
    }
    if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        tracing::info!("Auto-configuration will not generate a password for the elastic built-in superuser, as we cannot  determine if there is a terminal attached to the elasticsearch process. You can use the `bin/elasticsearch-reset-password` tool to set the password for the elastic user.");
        return;
    }
    let node = node.clone();
    tokio::spawn(async move {
        // Wait for this node to lead (ES waits for the security index state).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while !node.is_master() {
            if std::time::Instant::now() >= deadline || node.state().master_node_id.as_deref().is_some_and(|m| m != node.local.id) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        if !security_state_is_new(&node.engine.security) {
            return;
        }
        // ES waits a few seconds: nothing depends on this.
        tokio::time::sleep(std::time::Duration::from_secs(9)).await;
        let settings = &node.config.settings;
        let fingerprint = https_ca_fingerprint(settings);
        match &fingerprint {
            Some(f) => tracing::info!("HTTPS has been configured with automatically generated certificates, and the CA's hex-encoded SHA-256 fingerprint is [{f}]"),
            None => tracing::error!("Failed to compute the HTTPS CA fingerprint, probably the certs are not auto-generated"),
        }
        let preset = node.engine.security.cfg.read().bootstrap_password.as_deref().is_some_and(|p| !p.is_empty());
        let password: Option<String> = if !preset && !settings.contains_key("autoconfiguration.password_hash") {
            let password = generate_password(20);
            let hash_input = password.clone();
            let set = node.mutate_engine_metadata(move |engine| {
                let sec = &engine.security;
                if !security_state_is_new(sec) {
                    return Ok(false);
                }
                let hash = sec.cfg.read().password_hashing.hash(&hash_input);
                sec.state.write().reserved.entry("elastic".into()).or_default().password_hash = Some(hash);
                sec.save()?;
                Ok(true)
            }).await;
            match set {
                Ok((true, _)) => Some(password),
                Ok((false, _)) => return,
                Err(e) => {
                    tracing::error!("Failed to generate credentials for the elastic built-in superuser: {}", e.reason);
                    None
                }
            }
        } else {
            if !preset {
                tracing::info!("Auto-configuration will not generate a password for the elastic built-in superuser, you should use the password specified in the node's secure setting [bootstrap.password] in order to authenticate as elastic");
            }
            Some(String::new())
        };
        let (kibana_token, node_token) = enrollment_tokens(&node).await;
        let ansi = std::env::var_os("NO_COLOR").is_none();
        let text = banner(password.as_deref(), fingerprint.as_deref(), kibana_token.as_deref(), node_token.as_deref(), true, ansi, terminal_width());
        println!();
        println!("{text}");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(pairs: &[(&str, &str)]) -> Settings {
        pairs.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect()
    }

    fn conf_with_yml(text: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("elasticsearch.yml"), text).unwrap();
        dir
    }

    fn data(dir: &Path) -> String {
        dir.join("data").to_string_lossy().into_owned()
    }

    #[test]
    fn each_skip_condition() {
        let conf = conf_with_yml("cluster.name: c\n");
        let d = data(conf.path());
        let base = |extra: &[(&str, &str)]| {
            let mut m = s(&[("path.data", d.as_str())]);
            m.extend(s(extra));
            m
        };
        assert_eq!(preflight(&base(&[]), conf.path()), None, "a new, unconfigured node is auto-configured");
        let reason = |extra: &[(&str, &str)]| preflight(&base(extra), conf.path()).map(|s| s.message).unwrap_or_default();
        assert!(reason(&[("xpack.security.enabled", "false")]).contains("security is already configured"));
        assert!(reason(&[("xpack.security.enabled", "true")]).contains("security is already configured"));
        assert!(reason(&[("xpack.security.autoconfiguration.enabled", "false")]).contains("[xpack.security.autoconfiguration.enabled] is false"));
        assert!(reason(&[("xpack.security.transport.ssl.enabled", "false")]).contains("TLS is already configured"));
        assert!(reason(&[("xpack.security.http.ssl.certificate", "x.crt")]).contains("TLS is already configured"));
        assert!(reason(&[("discovery.seed_hosts", "a:9300")]).contains("multi-node cluster"));
        assert!(reason(&[("discovery.seed_providers", "file")]).contains("multi-node cluster"));
        assert!(reason(&[("cluster.initial_master_nodes", "a,b")]).contains("multi-node cluster"));
        assert_eq!(reason(&[("cluster.initial_master_nodes", "me"), ("node.name", "me")]), "", "initial masters = this node");
        assert_eq!(reason(&[("discovery.type", "single-node"), ("discovery.seed_hosts", "a")]), "", "single-node discovery");
        assert!(reason(&[("node.roles", "[data]")]).contains("cannot become master"));
        assert!(reason(&[("node.roles", "[master,voting_only,data]")]).contains("cannot become master"));
        assert!(reason(&[("node.roles", "[master]")]).contains("cannot contain data"));
        assert_eq!(reason(&[("node.roles", "[master,data_hot]")]), "");
        // Every configuration skip is silent (VERBOSE) and a NOOP.
        let sk = preflight(&base(&[("xpack.security.enabled", "false")]), conf.path()).unwrap();
        assert!(sk.verbose && sk.exit_code == exit_codes::NOOP);

        // Existing node data: not a first start.
        std::fs::create_dir_all(conf.path().join("data/_state")).unwrap();
        let sk = preflight(&base(&[]), conf.path()).unwrap();
        assert!(sk.message.contains("not starting up for the first time"), "{}", sk.message);
        // An empty data dir is still a first start.
        std::fs::remove_dir_all(conf.path().join("data/_state")).unwrap();
        assert_eq!(preflight(&base(&[]), conf.path()), None);

        // No elasticsearch.yml: skipped, printed (NORMAL), CONFIG.
        let empty = tempfile::tempdir().unwrap();
        let sk = preflight(&s(&[("path.data", data(empty.path()).as_str())]), empty.path()).unwrap();
        assert!(!sk.verbose && sk.exit_code == exit_codes::CONFIG && sk.message.contains("is missing or is not a regular file"));
        // An unreadable keystore path (a directory).
        std::fs::create_dir(conf.path().join(KEYSTORE_FILENAME)).unwrap();
        let sk = preflight(&base(&[]), conf.path()).unwrap();
        assert!(sk.message.contains("is not a readable regular file"), "{}", sk.message);
    }

    #[test]
    fn block_matches_es_layout() {
        let b = config_block(&s(&[]), "03-11-2026 10:00:00");
        let expected = "#----------------------- BEGIN SECURITY AUTO CONFIGURATION -----------------------\n\
#\n\
# The following settings, TLS certificates, and keys have been automatically      \n\
# generated to configure Elasticsearch security features on 03-11-2026 10:00:00\n\
#\n\
# --------------------------------------------------------------------------------\n\
\n\
# Enable security features\n\
xpack.security.enabled: true\n\
\n\
xpack.security.enrollment.enabled: true\n\
\n\
# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents\n\
xpack.security.http.ssl:\n  enabled: true\n  key: certs/http.key\n  certificate: certs/http.crt\n\
\n\
# Enable encryption and mutual authentication between cluster nodes\n\
xpack.security.transport.ssl:\n  enabled: true\n  verification_mode: certificate\n  key: certs/transport.key\n  certificate: certs/transport.crt\n  certificate_authorities: [ \"certs/transport_ca.crt\" ]\n\
# Create a new cluster with the current node only\n\
# Additional nodes can still join the cluster later\n\
cluster.initial_master_nodes: [\"${HOSTNAME}\"]\n\
\n\
# Allow HTTP API connections from anywhere\n\
# Connections are encrypted and require user authentication\n\
http.host: 0.0.0.0\n\
\n\
# Allow other nodes to join the cluster from anywhere\n\
# Connections are encrypted and mutually authenticated\n\
#transport.host: 0.0.0.0\n\
\n\
#----------------------- END SECURITY AUTO CONFIGURATION -------------------------\n";
        assert_eq!(b, expected);
        // Explicit addresses / node name / single-node / enrollment are respected.
        let b = config_block(&s(&[("network.host", "10.0.0.1"), ("discovery.type", "single-node"), ("xpack.security.enrollment.enabled", "false")]), "d");
        assert!(!b.contains("http.host") && !b.contains("transport.host") && !b.contains("initial_master_nodes") && !b.contains("enrollment"));
        // ES (`AutoConfigureNode`): the last transport line is ended by the
        // newline written before the end marker, no empty line between.
        assert!(b.contains("  certificate_authorities: [ \"certs/transport_ca.crt\" ]\n#-----------------------"), "{b}");
        let b = config_block(&s(&[("node.name", "n1"), ("http.port", "9201")]), "d");
        assert!(b.contains("cluster.initial_master_nodes: [\"n1\"]\n"));
        // The block parses into the settings it is meant to set.
        let parsed = crate::config::parse_yaml(&append_block("cluster.name: c", &config_block(&s(&[]), "d"))).unwrap();
        assert_eq!(get_str(&parsed, "xpack.security.enabled"), Some("true"));
        assert_eq!(get_str(&parsed, "xpack.security.transport.ssl.verification_mode"), Some("certificate"));
        assert_eq!(get_list(&parsed, "xpack.security.transport.ssl.certificate_authorities"), vec!["certs/transport_ca.crt"]);
        assert_eq!(get_str(&parsed, "http.host"), Some("0.0.0.0"));
        assert_eq!(get_str(&parsed, "cluster.name"), Some("c"));
    }

    #[test]
    fn enrollment_block_matches_es_layout() {
        let addrs = vec!["10.0.0.1:9300".to_string(), "10.0.0.2:9300".to_string()];
        let b = config_block_for(&s(&[]), "d", Some(EnrollmentBlock { transport_addresses: &addrs, remote_host: true }));
        assert!(b.contains("  certificate_authorities: [ \"certs/transport_ca.crt\" ]\n# Discover existing nodes in the cluster\ndiscovery.seed_hosts: [\"10.0.0.1:9300\", \"10.0.0.2:9300\"]\n\n# Allow HTTP API connections from anywhere\n"), "{b}");
        assert!(!b.contains("initial_master_nodes"), "{b}");
        assert!(b.ends_with("# Connections are encrypted and mutually authenticated\ntransport.host: 0.0.0.0\n\n#----------------------- END SECURITY AUTO CONFIGURATION -------------------------\n"), "{b}");
        let b = config_block_for(&s(&[]), "d", Some(EnrollmentBlock { transport_addresses: &addrs, remote_host: false }));
        assert!(b.contains("\n#transport.host: 0.0.0.0\n"), "{b}");
        let parsed = crate::config::parse_yaml(&append_block("cluster.name: c", &b)).unwrap();
        assert_eq!(get_list(&parsed, "discovery.seed_hosts"), addrs);
        assert_eq!(get_str(&parsed, "xpack.security.enrollment.enabled"), Some("true"));
    }

    #[test]
    fn enrollment_refusals_fail_the_start() {
        let conf = conf_with_yml("cluster.name: c\n");
        let d = data(conf.path());
        let base = |extra: &[(&str, &str)]| {
            let mut m = s(&[("path.data", d.as_str()), ("path.conf", conf.path().to_str().unwrap())]);
            m.extend(s(extra));
            m
        };
        let token = crate::enrollment::EnrollmentToken::new("id:key".into(), "00".into(), vec!["127.0.0.1:1".into()]).encoded();
        // Configuration conflicts are errors (ES `notifyOfFailure` in enrollment mode).
        let e = enroll_blocking(&base(&[("xpack.security.enabled", "true")]), conf.path(), &token, false).unwrap_err();
        assert_eq!((e.exit_code, e.message.as_str()), (exit_codes::NOOP, "Skipping security auto configuration because it appears that security is already configured."));
        let e = enroll_blocking(&base(&[("discovery.seed_hosts", "a:9300")]), conf.path(), &token, false).unwrap_err();
        assert!(e.message.contains("multi-node cluster"), "{}", e.message);
        // A bad token.
        let e = enroll_blocking(&base(&[]), conf.path(), "bm90IGEgdG9rZW4", false).unwrap_err();
        assert_eq!((e.exit_code, e.message.as_str()), (exit_codes::DATA_ERROR, "Aborting auto configuration. Invalid enrollment token"));
        // Role checks do not apply to an enrolling node: it gets as far as
        // calling the (unreachable) node.
        let e = enroll_blocking(&base(&[("node.roles", "[data]")]), conf.path(), &token, false).unwrap_err();
        assert_eq!(e.exit_code, exit_codes::UNAVAILABLE);
        assert_eq!(e.message, "Aborting enrolling to cluster. Could not communicate with the node on any of the addresses from the enrollment token. All of [127.0.0.1:1] were attempted.");
        // Nothing was left behind.
        let left: Vec<String> = std::fs::read_dir(conf.path()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("certs")).collect();
        assert!(left.is_empty(), "{left:?}");
        assert_eq!(std::fs::read_to_string(conf.path().join("elasticsearch.yml")).unwrap(), "cluster.name: c\n");
        // A node with data is not enrolled.
        std::fs::create_dir_all(conf.path().join("data/_state")).unwrap();
        let e = enroll_blocking(&base(&[]), conf.path(), &token, false).unwrap_err();
        assert!(e.message.contains("not starting up for the first time") && e.exit_code == exit_codes::NOOP, "{}", e.message);
    }

    #[test]
    fn generates_parseable_certificates_and_appends_once() {
        let conf = conf_with_yml("cluster.name: autoconf");
        let settings = s(&[("path.data", data(conf.path()).as_str()), ("node.name", "n0"), ("path.conf", conf.path().to_str().unwrap())]);
        let now = chrono::Utc::now();
        let out = run_at(&settings, conf.path(), now, &[IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)]).unwrap();
        let (certs_dir, http_ca_fingerprint) = match out {
            Outcome::Configured { certs_dir, http_ca_fingerprint } => (certs_dir, http_ca_fingerprint),
            other => panic!("expected auto-configuration: {other:?}"),
        };
        assert_eq!(certs_dir, conf.path().join("certs"));
        for f in [HTTP_CA_CERT, HTTP_CA_KEY, HTTP_CERT, HTTP_KEY, TRANSPORT_CA_CERT, TRANSPORT_CERT, TRANSPORT_KEY] {
            assert!(certs_dir.join(f).is_file(), "{f}");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for f in [HTTP_KEY, HTTP_CA_KEY, TRANSPORT_KEY] {
                assert_eq!(std::fs::metadata(certs_dir.join(f)).unwrap().permissions().mode() & 0o777, 0o600, "{f}");
            }
        }
        // The certificates and keys load as the transport/HTTP layers load them.
        let chain = pem_certificates(&std::fs::read_to_string(certs_dir.join(HTTP_CERT)).unwrap());
        assert_eq!(chain.len(), 2, "leaf and CA");
        assert_eq!(hex(&elasticsearch_engine::security::hash::sha256(&chain[1])), http_ca_fingerprint);
        assert_eq!(http_ca_fingerprint.len(), 64);
        assert!(rcgen::KeyPair::from_pem(&std::fs::read_to_string(certs_dir.join(TRANSPORT_KEY)).unwrap()).is_ok());
        assert!(rcgen::KeyPair::from_pem(&std::fs::read_to_string(certs_dir.join(HTTP_KEY)).unwrap()).is_ok());
        assert_eq!(pem_certificates(&std::fs::read_to_string(certs_dir.join(TRANSPORT_CA_CERT)).unwrap()).len(), 1);
        // The yml gained one block, and the node settings now load.
        let yml = std::fs::read_to_string(conf.path().join("elasticsearch.yml")).unwrap();
        assert!(yml.starts_with("cluster.name: autoconf\n\n#----------------------- BEGIN SECURITY AUTO CONFIGURATION"), "{yml}");
        assert_eq!(yml.matches(AUTO_CONFIGURATION_START_MARKER).count(), 1);
        let mut reloaded = crate::config::parse_yaml(&yml).unwrap();
        reloaded.insert("path.conf".into(), Value::String(conf.path().to_string_lossy().into_owned()));
        assert_eq!(https_ca_fingerprint(&reloaded).as_deref(), Some(http_ca_fingerprint.as_str()));
        let tls = elasticsearch_transport::TlsConfig::from_settings(
            &|k| get_str(&reloaded, k).map(str::to_string), &|k| get_list(&reloaded, k), conf.path()).unwrap().expect("transport TLS on");
        assert_eq!(tls.verification_mode, elasticsearch_transport::VerificationMode::Certificate);
        // A restart with the configured yml (security now set) is skipped.
        let mut restart = reloaded.clone();
        restart.insert("path.data".into(), Value::String(data(conf.path())));
        match run_at(&restart, conf.path(), now, &[]).unwrap() {
            Outcome::Skipped(s) => assert!(s.message.contains("security is already configured")),
            other => panic!("re-ran: {other:?}"),
        }
        // A restart after the node wrote data is skipped too.
        std::fs::create_dir_all(conf.path().join("data/nodes")).unwrap();
        assert!(matches!(run_at(&settings, conf.path(), now, &[]).unwrap(), Outcome::Skipped(_)));
        let yml_after = std::fs::read_to_string(conf.path().join("elasticsearch.yml")).unwrap();
        assert_eq!(yml_after, yml, "the yml is appended once");
    }

    #[test]
    fn passwords_and_banner() {
        let p = generate_password(20);
        assert_eq!(p.len(), 20);
        assert!(p.chars().all(|c| c.is_ascii_alphanumeric() || "*-_=+".contains(c)));
        assert_ne!(p, generate_password(20));
        assert!(random_alphanumeric(8).chars().all(|c| c.is_ascii_alphanumeric()));
        let b = banner(Some("pw123456"), Some("abcd"), Some("kt"), Some(""), false, false, 10);
        assert!(b.contains("-> Elasticsearch security features have been automatically configured!\n"));
        assert!(b.contains("->  Password for the elastic user (reset with `bin/elasticsearch-reset-password -u elastic`):\n  pw123456\n"));
        assert!(b.contains("->  HTTP CA certificate SHA-256 fingerprint:\n  abcd\n"));
        assert!(b.contains("\n----------\n"));
        assert!(b.contains("->  Configure Kibana to use this cluster:\n* Run Kibana and click the configuration link in the terminal when Kibana starts.\n* Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):\n  kt\n\n"), "{b}");
        assert!(b.contains("->  Configure other nodes to join this cluster:\n* On this node:\n  - Create an enrollment token with `bin/elasticsearch-create-enrollment-token -s node`.\n  - Uncomment the transport.host setting at the end of config/elasticsearch.yml.\n  - Restart Elasticsearch.\n"), "{b}");
        let b = banner(Some("p"), None, None, Some("nt"), false, false, 3);
        assert!(b.contains("X Unable to generate an enrollment token for Kibana instances, try invoking `bin/elasticsearch-create-enrollment-token -s kibana`.\n\n"), "{b}");
        assert!(b.contains(&format!("-> Configure other nodes to join this cluster:\n* Copy the following enrollment token and start new Elasticsearch nodes with `bin/elasticsearch --enrollment-token <token>` (valid for the next 30 minutes):\n  nt\n\n  If you're running in Docker, copy the enrollment token and run:\n  `docker run -e \"ENROLLMENT_TOKEN=<token>\" docker.elastic.co/elasticsearch/elasticsearch:{}`\n---\n", crate::node::ES_VERSION)), "{b}");
        let b = banner(Some("p"), None, None, None, false, false, 3);
        assert!(b.contains("X An enrollment token to enroll new nodes wasn't generated. To add nodes and enroll them into this cluster:\n"), "{b}");
        assert!(banner(None, None, None, None, true, true, 5).contains("\u{274C} Unable to auto-generate the password for the \u{1B}[1melastic\u{1B}[22m built-in superuser."));
        assert!(!banner(Some(""), None, None, Some(""), true, false, 5).contains("Password for"));
    }
}
