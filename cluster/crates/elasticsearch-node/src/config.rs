//! Node settings, loaded the way Elasticsearch loads them:
//!
//! 1. `$ES_PATH_CONF/elasticsearch.yml` (default `./config`), nested or flat
//!    YAML, with `${VAR}` / `${VAR:default}` environment substitution;
//! 2. environment variables whose name contains a dot (`cluster.name=x`),
//!    which is how the official Docker image passes settings;
//! 3. `-E key=value` command-line overrides (highest precedence).
//!
//! Everything is flattened to `a.b.c -> string | [string]` like ES's
//! `Settings`. [`NodeConfig`] then interprets the node-scope settings this
//! process needs (identity, roles, network, discovery, paths).

use elasticsearch_cluster::state::Role;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

pub type Settings = BTreeMap<String, Value>;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(msg.into()))
}

/// Node-scoped transport compression policy (ES Compression.Enabled).
/// Validate before opening listeners; never silently ignore a configured policy.
pub fn transport_settings(settings: &Settings) -> Result<elasticsearch_transport::TransportSettings, ConfigError> {
    let mut transport = elasticsearch_transport::TransportSettings::default();
    let value = settings.get("transport.compress").map(scalar).unwrap_or_else(|| "indexing_data".into());
    match value.as_str() {
        "false" => { transport.compress = false; transport.compress_all = false; }
        "true" => { transport.compress = true; transport.compress_all = true; }
        "indexing_data" => { transport.compress = true; transport.compress_all = false; }
        _ => return err(format!("unknown value for [transport.compress]: [{value}], must be one of [true, false, indexing_data]")),
    }
    Ok(transport)
}

/// Replace `${VAR}` and `${VAR:default}` from the environment. ES fails
/// startup on an unset variable without default; so do we.
pub fn substitute_env(input: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<String, ConfigError> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else { return err(format!("unterminated placeholder in [{input}]")) };
        let expr = &after[..end];
        let (name, default) = match expr.split_once(':') {
            Some((n, d)) => (n, Some(d)),
            None => (expr, None),
        };
        match env(name).or_else(|| default.map(str::to_string)) {
            Some(v) => out.push_str(&v),
            None => return err(format!("Could not resolve placeholder '{name}'")),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Flatten nested YAML/JSON into dotted keys. Scalars become strings (ES
/// stores every setting as a string); sequences become string arrays.
pub fn flatten(prefix: &str, v: &Value, out: &mut Settings) -> Result<(), ConfigError> {
    match v {
        Value::Object(m) => {
            for (k, v) in m {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(&key, v, out)?;
            }
        }
        Value::Array(a) => {
            let mut items = Vec::with_capacity(a.len());
            for x in a {
                match x {
                    Value::Object(_) | Value::Array(_) => return err(format!("setting [{prefix}] has a nested structure inside a list")),
                    other => items.push(Value::String(scalar(other))),
                }
            }
            insert(out, prefix, Value::Array(items))?;
        }
        Value::Null => {
            insert(out, prefix, Value::Null)?;
        }
        other => insert(out, prefix, Value::String(scalar(other)))?,
    }
    Ok(())
}

fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn insert(out: &mut Settings, key: &str, v: Value) -> Result<(), ConfigError> {
    if out.contains_key(key) {
        return err(format!("duplicate settings key [{key}] found"));
    }
    out.insert(key.to_string(), v);
    Ok(())
}

/// Parse `elasticsearch.yml` contents (already env-substituted).
pub fn parse_yaml(text: &str) -> Result<Settings, ConfigError> {
    let mut out = Settings::new();
    if text.trim().is_empty() {
        return Ok(out);
    }
    let y: serde_yaml::Value = serde_yaml::from_str(text).map_err(|e| ConfigError(format!("failed to parse elasticsearch.yml: {e}")))?;
    let v: Value = serde_json::to_value(y).map_err(|e| ConfigError(e.to_string()))?;
    if !v.is_null() {
        flatten("", &v, &mut out)?;
    }
    Ok(out)
}

/// Parse a list-valued override such as `a,b` or `["a","b"]` (both forms
/// are accepted by ES on the command line / in env).
fn parse_override_value(raw: &str) -> Value {
    let t = raw.trim();
    if t.starts_with('[') && t.ends_with(']') {
        if let Ok(Value::Array(a)) = serde_json::from_str::<Value>(t) {
            return Value::Array(a.into_iter().map(|x| Value::String(scalar(&x))).collect());
        }
        let inner = &t[1..t.len() - 1];
        return Value::Array(inner.split(',').map(|s| s.trim().trim_matches('"')).filter(|s| !s.is_empty()).map(|s| Value::String(s.into())).collect());
    }
    Value::String(raw.to_string())
}

/// Load settings from all sources. `args` are the process arguments after
/// the binary name; `env` is the process environment.
pub fn load(args: &[String], env: &BTreeMap<String, String>) -> Result<Settings, ConfigError> {
    let lookup = |k: &str| env.get(k).cloned();
    let conf_dir = env.get("ES_PATH_CONF").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("config"));
    let mut settings = Settings::new();
    let yml = conf_dir.join("elasticsearch.yml");
    if yml.exists() {
        let text = std::fs::read_to_string(&yml).map_err(|e| ConfigError(format!("failed to read {}: {e}", yml.display())))?;
        settings = parse_yaml(&substitute_env(&text, &lookup)?)?;
    }
    // Docker-image style: env vars with a dot in the name are settings.
    for (k, v) in env {
        if k.contains('.') && k.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-')) {
            settings.insert(k.clone(), parse_override_value(v));
        }
    }
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let kv = if a == "-E" {
            i += 1;
            args.get(i).cloned().ok_or_else(|| ConfigError("missing argument for -E".into()))?
        } else if let Some(rest) = a.strip_prefix("-E") {
            rest.to_string()
        } else {
            i += 1;
            continue;
        };
        let Some((k, v)) = kv.split_once('=') else { return err(format!("setting [{kv}] must be in the form key=value")) };
        settings.insert(k.to_string(), parse_override_value(&substitute_env(v, &lookup)?));
        i += 1;
    }
    settings.entry("path.conf".into()).or_insert_with(|| Value::String(conf_dir.display().to_string()));
    Ok(settings)
}

pub fn get_str<'a>(s: &'a Settings, key: &str) -> Option<&'a str> {
    s.get(key).and_then(Value::as_str)
}

/// A list setting given as an array, a comma-separated string, or absent.
pub fn get_list(s: &Settings, key: &str) -> Vec<String> {
    match s.get(key) {
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str()).map(str::to_string).collect(),
        // `a,b` or the bracketed `[a, b]` form (`-E node.roles=[master,data]`).
        Some(Value::String(x)) if !x.trim().is_empty() => {
            let t = x.trim();
            let t = t.strip_prefix('[').and_then(|t| t.strip_suffix(']')).unwrap_or(t);
            t.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
        }
        _ => Vec::new(),
    }
}

pub fn get_bool(s: &Settings, key: &str, default: bool) -> Result<bool, ConfigError> {
    match get_str(s, key) {
        None => Ok(default),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(o) => err(format!("Failed to parse value [{o}] as only [true] or [false] are allowed.")),
    }
}

/// `9200`, `9200-9300`: the ports to try binding, in order.
pub fn parse_port_range(spec: &str) -> Result<Vec<u16>, ConfigError> {
    let bad = || ConfigError(format!("Failed to parse port range [{spec}]"));
    match spec.split_once('-') {
        Some((a, b)) => {
            let a: u16 = a.trim().parse().map_err(|_| bad())?;
            let b: u16 = b.trim().parse().map_err(|_| bad())?;
            if a > b {
                return Err(bad());
            }
            Ok((a..=b).collect())
        }
        None => Ok(vec![spec.trim().parse().map_err(|_| bad())?]),
    }
}

/// Resolve an ES network host value (`_local_`, `_site_`, `_global_`,
/// `0.0.0.0`, a literal IP or a hostname) to addresses.
pub fn resolve_host(spec: &str) -> Result<Vec<IpAddr>, ConfigError> {
    match spec {
        "_local_" | "_local:ipv4_" => Ok(vec![IpAddr::V4(Ipv4Addr::LOCALHOST)]),
        "_site_" | "_site:ipv4_" | "_global_" | "_global:ipv4_" => {
            // Without enumerating interfaces, the address the OS would use
            // for outbound traffic is the best site/global candidate.
            Ok(vec![outbound_ip().unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))])
        }
        // `"0"` as Java's `InetAddress.getByName("0")` (ECK writes `network.host: "0"`).
        "0" | "0.0.0.0" | "_any_" => Ok(vec![IpAddr::V4(Ipv4Addr::UNSPECIFIED)]),
        "::" => Ok(vec![IpAddr::V6(std::net::Ipv6Addr::UNSPECIFIED)]),
        other => {
            if let Ok(ip) = other.parse::<IpAddr>() {
                return Ok(vec![ip]);
            }
            use std::net::ToSocketAddrs;
            let addrs: Vec<IpAddr> = (other, 0).to_socket_addrs().map_err(|e| ConfigError(format!("failed to resolve host [{other}]: {e}")))?.map(|a| a.ip()).collect();
            if addrs.is_empty() {
                return err(format!("failed to resolve host [{other}]"));
            }
            Ok(addrs)
        }
    }
}

/// The local IP used for outbound connections (no packets are sent).
fn outbound_ip() -> Option<IpAddr> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    sock.local_addr().ok().map(|a| a.ip())
}

/// The address to advertise when bound to a wildcard address.
fn publish_ip(bind: IpAddr) -> IpAddr {
    if bind.is_unspecified() { outbound_ip().unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST)) } else { bind }
}

#[derive(Clone, Debug)]
pub struct NodeConfig {
    pub settings: Settings,
    pub cluster_name: String,
    /// `node.name`; defaults to the hostname like ES.
    pub node_name: String,
    pub roles: BTreeSet<Role>,
    pub attributes: BTreeMap<String, String>,
    pub data_path: PathBuf,
    pub logs_path: PathBuf,
    pub repo_paths: Vec<PathBuf>,
    pub http_bind: IpAddr,
    pub http_publish: IpAddr,
    pub http_ports: Vec<u16>,
    pub transport_bind: IpAddr,
    pub transport_publish: IpAddr,
    pub transport_ports: Vec<u16>,
    /// `discovery.seed_hosts` entries (`host` or `host:port`; default port 9300).
    pub seed_hosts: Vec<String>,
    pub initial_master_nodes: Vec<String>,
    /// `discovery.type: single-node`.
    pub single_node: bool,
}

pub fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| std::process::Command::new("hostname").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()))
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "localhost".into())
}

impl NodeConfig {
    pub fn from_settings(settings: Settings) -> Result<NodeConfig, ConfigError> {
        transport_settings(&settings)?;
        let cluster_name = get_str(&settings, "cluster.name").unwrap_or("elasticsearch").to_string();
        if cluster_name.contains(':') {
            return err(format!("Illegal cluster.name value [{cluster_name}]: must not contain ':'"));
        }
        let node_name = get_str(&settings, "node.name").map(str::to_string).unwrap_or_else(hostname);

        let roles = if settings.contains_key("node.roles") {
            let mut roles = BTreeSet::new();
            for r in get_list(&settings, "node.roles") {
                match Role::parse(&r) {
                    Some(role) => {
                        roles.insert(role);
                    }
                    None => return err(format!("unknown role [{r}]")),
                }
            }
            if roles.contains(&Role::VotingOnly) && !roles.contains(&Role::Master) {
                return err("voting-only node must be master-eligible");
            }
            roles
        } else if elasticsearch_engine::stateless::enabled_in(&settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect()) {
            Role::stateless_defaults()
        } else {
            Role::defaults()
        };

        let attributes = settings
            .iter()
            .filter_map(|(k, v)| Some((k.strip_prefix("node.attr.")?.to_string(), scalar(v))))
            .collect();

        let single_node = match get_str(&settings, "discovery.type") {
            None | Some("multi-node") => false,
            Some("single-node") => true,
            Some(o) => return err(format!("Unknown discovery type [{o}]")),
        };
        let initial_master_nodes = get_list(&settings, "cluster.initial_master_nodes");
        if single_node && !initial_master_nodes.is_empty() {
            return err("setting [cluster.initial_master_nodes] is not allowed when [discovery.type] is set to [single-node]");
        }

        let host_of = |specific: &str, fallback: &str| -> Result<IpAddr, ConfigError> {
            let spec = get_str(&settings, specific).or_else(|| get_str(&settings, fallback)).or_else(|| get_str(&settings, "network.host")).unwrap_or("_local_");
            Ok(resolve_host(spec)?[0])
        };
        let http_bind = host_of("http.bind_host", "http.host")?;
        let transport_bind = host_of("transport.bind_host", "transport.host")?;
        let http_publish = match get_str(&settings, "http.publish_host").or_else(|| get_str(&settings, "network.publish_host")) {
            Some(h) => resolve_host(h)?[0],
            None => publish_ip(http_bind),
        };
        let transport_publish = match get_str(&settings, "transport.publish_host").or_else(|| get_str(&settings, "network.publish_host")) {
            Some(h) => resolve_host(h)?[0],
            None => publish_ip(transport_bind),
        };
        let http_ports = parse_port_range(get_str(&settings, "http.port").unwrap_or("9200-9300"))?;
        let transport_ports = parse_port_range(get_str(&settings, "transport.port").unwrap_or("9300-9400"))?;

        let data_path = PathBuf::from(get_str(&settings, "path.data").unwrap_or("data"));
        let logs_path = PathBuf::from(get_str(&settings, "path.logs").unwrap_or("logs"));
        let repo_paths = get_list(&settings, "path.repo").into_iter().map(PathBuf::from).collect();

        let mut seed_hosts = get_list(&settings, "discovery.seed_hosts");
        // `discovery.seed_providers: file` reads `$ES_PATH_CONF/unicast_hosts.txt`
        // (ES's `FileBasedSeedHostsProvider`; re-read every discovery round,
        // see `seed_hosts_file`). The settings provider is always on as well.
        for p in get_list(&settings, "discovery.seed_providers") {
            if p != "file" && p != "settings" {
                return err(format!("Unknown seed providers [{p}]"));
            }
        }
        if let Some(file) = seed_hosts_file(&settings) {
            seed_hosts.extend(read_unicast_hosts(&file));
        }

        Ok(NodeConfig {
            cluster_name,
            node_name,
            roles,
            attributes,
            data_path,
            logs_path,
            repo_paths,
            http_bind,
            http_publish,
            http_ports,
            transport_bind,
            transport_publish,
            transport_ports,
            seed_hosts,
            initial_master_nodes,
            single_node,
            settings,
        })
    }

    pub fn is_master_eligible(&self) -> bool {
        self.roles.contains(&Role::Master)
    }

    /// No discovery setting at all (`discovery.seed_hosts`,
    /// `discovery.seed_providers`, `cluster.initial_master_nodes`,
    /// `discovery.type`): ES's development-mode auto-bootstrap applies, so a
    /// bare `elasticsearch` with no configuration forms a one-node cluster.
    /// (ES also folds in peers it finds on the local transport ports after
    /// `discovery.unconfigured_bootstrap_timeout`; this bootstraps alone.)
    pub fn discovery_unconfigured(&self) -> bool {
        self.seed_hosts.is_empty()
            && self.initial_master_nodes.is_empty()
            && !self.single_node
            && ["discovery.seed_providers", "discovery.type"].iter().all(|k| !self.settings.contains_key(*k))
    }

    pub fn is_data(&self) -> bool {
        self.roles.iter().any(|r| r.is_data())
    }

    /// Seed addresses as `SocketAddr` strings, applying the default
    /// transport port (first port of `transport.port`) where none is given.
    pub fn seed_addresses(&self) -> Vec<String> {
        self.with_default_port(&self.seed_hosts)
    }

    /// `$ES_PATH_CONF/unicast_hosts.txt` when `discovery.seed_providers`
    /// includes `file`.
    pub fn seed_hosts_file(&self) -> Option<PathBuf> {
        seed_hosts_file(&self.settings)
    }

    /// The seed addresses as of now: `discovery.seed_hosts` plus the current
    /// content of the seed hosts file. ES's `FileBasedSeedHostsProvider`
    /// reads the file on every discovery round, so hosts written by an
    /// orchestrator after the node started (ECK rewrites the file as pods
    /// come and go) are found without a restart.
    pub fn current_seed_addresses(&self) -> Vec<String> {
        let mut hosts = get_list(&self.settings, "discovery.seed_hosts");
        if let Some(file) = self.seed_hosts_file() {
            hosts.extend(read_unicast_hosts(&file));
        }
        self.with_default_port(&hosts)
    }

    fn with_default_port(&self, hosts: &[String]) -> Vec<String> {
        let default_port = self.transport_ports.first().copied().unwrap_or(9300);
        hosts
            .iter()
            .map(|h| {
                if h.parse::<SocketAddr>().is_ok() || (h.contains(':') && !h.starts_with('[') && h.matches(':').count() == 1) {
                    h.clone()
                } else {
                    format!("{h}:{default_port}")
                }
            })
            .collect()
    }

    /// The settings as ES's `_nodes/settings` renders them (nested JSON),
    /// without secure (keystore) or filtered settings.
    pub fn settings_json(&self) -> Value {
        let mut root = Map::new();
        let filter = elasticsearch_engine::settings_filter::SettingsFilter::for_settings(self.settings.iter());
        for (k, v) in self.settings.iter().filter(|(k, _)| !filter.hides(k)) {
            let mut cur = &mut root;
            let parts: Vec<&str> = k.split('.').collect();
            for p in &parts[..parts.len() - 1] {
                let e = cur.entry(p.to_string()).or_insert_with(|| Value::Object(Map::new()));
                if !e.is_object() {
                    *e = Value::Object(Map::new());
                }
                cur = e.as_object_mut().expect("object");
            }
            cur.insert(parts[parts.len() - 1].to_string(), v.clone());
        }
        Value::Object(root)
    }
}

fn seed_hosts_file(settings: &Settings) -> Option<PathBuf> {
    get_list(settings, "discovery.seed_providers").iter().any(|p| p == "file").then(|| {
        PathBuf::from(get_str(settings, "path.conf").unwrap_or("config")).join("unicast_hosts.txt")
    })
}

/// ES's `FileBasedSeedHostsProvider.getHostsList`: one host per line,
/// trimmed, blank lines and `#` comments skipped; a missing file is no hosts
/// (ES logs a warning).
fn read_unicast_hosts(p: &Path) -> Vec<String> {
    match std::fs::read_to_string(p) {
        Ok(t) => t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string).collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!(file = %p.display(), "seed hosts file does not exist, file path=[{}]", p.display());
            Vec::new()
        }
        Err(e) => {
            tracing::warn!(file = %p.display(), "failed to read file based discovery seed hosts: {e}");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn yaml_nested_and_flat_with_env() {
        let e = env(&[("ZONE", "us-east-1a")]);
        let text = substitute_env("cluster:\n  name: prod\nnode.attr.zone: ${ZONE}\nnode.roles: [master, data]\nhttp.port: ${PORT:9201}\n", &|k| e.get(k).cloned()).unwrap();
        let s = parse_yaml(&text).unwrap();
        assert_eq!(get_str(&s, "cluster.name"), Some("prod"));
        assert_eq!(get_str(&s, "node.attr.zone"), Some("us-east-1a"));
        assert_eq!(get_str(&s, "http.port"), Some("9201"));
        assert_eq!(get_list(&s, "node.roles"), vec!["master", "data"]);
    }

    #[test]
    fn stateless_roles_and_list_forms() {
        let base = |extra: &[(&str, &str)]| -> Settings {
            let mut s = Settings::new();
            s.insert("cluster.name".into(), Value::String("c".into()));
            for (k, v) in extra {
                s.insert(k.to_string(), Value::String(v.to_string()));
            }
            s
        };
        // Comma-separated and bracketed `-E` forms.
        for v in ["master,remote_cluster_client,ingest,index", "[master, remote_cluster_client, ingest, index]"] {
            let c = NodeConfig::from_settings(base(&[("node.roles", v)])).unwrap();
            assert_eq!(c.roles, [Role::Master, Role::RemoteClusterClient, Role::Ingest, Role::Index].into_iter().collect());
            assert!(c.is_data(), "an index node holds shards");
        }
        let c = NodeConfig::from_settings(base(&[("node.roles", "[remote_cluster_client,search]")])).unwrap();
        assert_eq!(c.roles, [Role::RemoteClusterClient, Role::Search].into_iter().collect());
        // Default roles: regular nodes never get the stateless roles; a
        // stateless node gets `index` instead of the data roles.
        let c = NodeConfig::from_settings(base(&[])).unwrap();
        assert!(c.roles.contains(&Role::Data) && !c.roles.contains(&Role::Index) && !c.roles.contains(&Role::Search));
        let c = NodeConfig::from_settings(base(&[("stateless.enabled", "true")])).unwrap();
        assert!(c.roles.contains(&Role::Index) && c.roles.contains(&Role::Master) && !c.roles.contains(&Role::Data) && !c.roles.contains(&Role::Search));
    }

    #[test]
    fn unresolved_placeholder_fails() {
        let e = substitute_env("x: ${NOPE}", &|_| None).unwrap_err();
        assert_eq!(e.0, "Could not resolve placeholder 'NOPE'");
    }

    #[test]
    fn duplicate_keys_rejected() {
        assert!(parse_yaml("cluster.name: a\ncluster:\n  name: b\n").is_err());
    }

    #[test]
    fn precedence_cli_over_env_over_yaml() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("elasticsearch.yml"), "cluster.name: from-yaml\nnode.name: n-yaml\n").unwrap();
        let e = env(&[("ES_PATH_CONF", dir.path().to_str().unwrap()), ("cluster.name", "from-env"), ("discovery.seed_hosts", "a,b:9301")]);
        let s = load(&["-Ecluster.name=from-cli".into(), "-E".into(), "node.roles=[master]".into()], &e).unwrap();
        let c = NodeConfig::from_settings(s).unwrap();
        assert_eq!(c.cluster_name, "from-cli");
        assert_eq!(c.node_name, "n-yaml");
        assert_eq!(c.roles, [Role::Master].into_iter().collect());
        assert_eq!(c.seed_addresses(), vec!["a:9300", "b:9301"]);
    }

    #[test]
    fn file_seed_provider_is_reread() {
        // ECK: `discovery.seed_providers: file`, `discovery.seed_hosts: []`,
        // and a `unicast_hosts.txt` it rewrites as pods come and go.
        let dir = tempfile::tempdir().unwrap();
        let mut s = Settings::new();
        s.insert("path.conf".into(), dir.path().to_str().unwrap().into());
        s.insert("discovery.seed_providers".into(), "file".into());
        s.insert("discovery.seed_hosts".into(), Value::Array(vec![]));
        let c = NodeConfig::from_settings(s.clone()).unwrap();
        assert!(c.seed_addresses().is_empty(), "missing file is no hosts");
        assert!(!c.discovery_unconfigured());
        std::fs::write(dir.path().join("unicast_hosts.txt"), "# comment\n10.0.0.1:9300\n\n  10.0.0.2  \n").unwrap();
        assert_eq!(c.current_seed_addresses(), vec!["10.0.0.1:9300", "10.0.0.2:9300"]);
        std::fs::write(dir.path().join("unicast_hosts.txt"), "10.0.0.3:9300\n").unwrap();
        assert_eq!(c.current_seed_addresses(), vec!["10.0.0.3:9300"]);
        assert_eq!(NodeConfig::from_settings(s).unwrap().seed_addresses(), vec!["10.0.0.3:9300"]);
        let mut bad = Settings::new();
        bad.insert("discovery.seed_providers".into(), "ec2".into());
        assert_eq!(NodeConfig::from_settings(bad).unwrap_err().0, "Unknown seed providers [ec2]");
    }

    #[test]
    fn defaults_and_validation() {
        let c = NodeConfig::from_settings(Settings::new()).unwrap();
        assert_eq!(c.cluster_name, "elasticsearch");
        assert_eq!(c.roles, Role::defaults());
        assert_eq!(c.http_ports.first(), Some(&9200));
        assert_eq!(c.transport_ports.len(), 101);
        let mut s = Settings::new();
        s.insert("discovery.type".into(), "single-node".into());
        s.insert("cluster.initial_master_nodes".into(), "a".into());
        assert!(NodeConfig::from_settings(s).is_err());
        let mut s = Settings::new();
        s.insert("node.roles".into(), Value::Array(vec!["voting_only".into()]));
        assert!(NodeConfig::from_settings(s).is_err());
        assert!(parse_port_range("9300-9200").is_err());
    }

    #[test]
    fn settings_json_nests() {
        let mut s = Settings::new();
        s.insert("cluster.name".into(), "x".into());
        s.insert("node.attr.zone".into(), "a".into());
        let c = NodeConfig::from_settings(s).unwrap();
        assert_eq!(c.settings_json()["node"]["attr"]["zone"], "a");
    }
    #[test]
    fn transport_compression_modes_are_applied_and_invalid_values_rejected() {
        for (value, enabled, all) in [
            (None, true, false),
            (Some(Value::String("indexing_data".into())), true, false),
            (Some(Value::String("true".into())), true, true),
            (Some(Value::Bool(false)), false, false),
        ] {
            let mut settings = Settings::new();
            if let Some(value) = value { settings.insert("transport.compress".into(), value); }
            let config = NodeConfig::from_settings(settings).unwrap();
            let transport = transport_settings(&config.settings).unwrap();
            assert_eq!((transport.compress, transport.compress_all), (enabled, all));
        }
        for value in [Value::String("yes".into()), Value::Array(vec![]), Value::Null] {
            let settings = Settings::from([("transport.compress".into(), value)]);
            assert!(NodeConfig::from_settings(settings).unwrap_err().to_string().contains("transport.compress"));
        }
    }

}
