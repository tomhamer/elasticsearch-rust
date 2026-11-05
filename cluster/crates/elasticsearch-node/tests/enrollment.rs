//! ES's enrollment flow end to end, in process: node 1 auto-configures
//! security on its first start, `elasticsearch-create-enrollment-token -s
//! node` creates a token against it, node 2 enrolls with that token
//! (`--enrollment-token`: pinned `GET /_security/enroll/node`, the cluster's
//! certificates installed) and joins, and the two-node cluster answers over
//! HTTPS with the `elastic` password. Kibana enrollment hands out the HTTP CA
//! and an `elastic/kibana` service account token, and each token's API key
//! may only enroll its own kind.
#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use base64::Engine;
use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::enrollment::EnrollmentToken;
use elasticsearch_node::node::ClusterNode;
use elasticsearch_node::reset_password::Terminal;
use elasticsearch_node::security_autoconfig::{self, Outcome};
use serde_json::Value;
use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const PASSWORD: &str = "enroll-password";

#[derive(Default)]
struct Script {
    out: String,
    err: String,
}

impl Terminal for Script {
    fn print(&mut self, text: &str) {
        self.out.push_str(text);
    }
    fn eprintln(&mut self, text: &str) {
        self.err.push_str(text);
        self.err.push('\n');
    }
    fn read_line(&mut self, _: &str) -> Option<String> {
        None
    }
    fn read_secret(&mut self, _: &str) -> Option<String> {
        None
    }
}

fn env(conf: &Path) -> BTreeMap<String, String> {
    BTreeMap::from([("ES_PATH_CONF".to_string(), conf.to_string_lossy().into_owned())])
}

/// A node's settings as the CLI loads them: its `elasticsearch.yml` and `-E`s.
fn load(dir: &Path, name: &str) -> Settings {
    let conf = dir.join(name).join("config");
    let args: Vec<String> = [
        ("node.name", name.to_string()),
        ("path.data", dir.join(name).join("data").to_string_lossy().into_owned()),
        ("path.logs", dir.join(name).join("logs").to_string_lossy().into_owned()),
        ("network.host", "127.0.0.1".into()),
        ("http.port", "0".into()),
        ("transport.port", "0".into()),
        ("bootstrap.password", PASSWORD.into()),
        ("xpack.security.authc.password_hashing.algorithm", "pbkdf2_1000".into()),
        ("cluster.routing.allocation.disk.watermark.low", "1b".into()),
        ("cluster.routing.allocation.disk.watermark.high", "1b".into()),
        ("cluster.routing.allocation.disk.watermark.flood_stage", "1b".into()),
    ]
    .iter()
    .map(|(k, v)| format!("-E{k}={v}"))
    .collect();
    elasticsearch_node::config::load(&args, &env(&conf)).unwrap()
}

fn new_conf(dir: &Path, name: &str) -> std::path::PathBuf {
    let conf = dir.join(name).join("config");
    std::fs::create_dir_all(&conf).unwrap();
    std::fs::write(conf.join("elasticsearch.yml"), "cluster.name: enrollment-test\n").unwrap();
    std::fs::write(conf.join("users"), "").unwrap();
    std::fs::write(conf.join("users_roles"), "").unwrap();
    conf
}

async fn spawn(settings: Settings) -> Arc<ClusterNode> {
    let (node, http) = ClusterNode::start(NodeConfig::from_settings(settings).unwrap()).await.unwrap();
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let n = node.clone();
    tokio::spawn(async move { elasticsearch_node::front::serve(n, http).await.unwrap() });
    node
}

async fn get(client: &reqwest::Client, url: &str, auth: &str) -> (u16, Value) {
    match client.get(url).header("Authorization", auth).send().await {
        Ok(r) => {
            let status = r.status().as_u16();
            (status, r.json::<Value>().await.unwrap_or(Value::Null))
        }
        Err(e) => (0, Value::String(e.to_string())),
    }
}

fn basic() -> String {
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("elastic:{PASSWORD}")))
}

fn api_key(token: &EnrollmentToken) -> String {
    format!("ApiKey {}", base64::engine::general_purpose::STANDARD.encode(&token.api_key))
}

async fn create_token(conf: &Path, base: &str, scope: &str) -> EnrollmentToken {
    let args: Vec<String> = ["-s", scope, "--url", base, "-E", "xpack.security.authc.password_hashing.algorithm=pbkdf2_1000"].iter().map(|s| s.to_string()).collect();
    let o = elasticsearch_node::create_enrollment_token::parse_args(&args).unwrap();
    let mut term = Script::default();
    elasticsearch_node::create_enrollment_token::run(&o, &env(conf), &mut term).await.unwrap_or_else(|e| panic!("{e}\nout: {}\nerr: {}", term.out, term.err));
    // The token alone, on one line.
    assert_eq!(term.out.lines().count(), 1, "{}", term.out);
    EnrollmentToken::decode(term.out.trim()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn second_node_enrolls_with_a_token_and_joins() {
    let dir = evidence::retained_dir("enrollment");
    let localhost: [IpAddr; 1] = [IpAddr::from([127, 0, 0, 1])];

    // Node 1: first start, auto-configured.
    let conf1 = new_conf(dir.path(), "n1");
    let fingerprint = match security_autoconfig::run_at(&load(dir.path(), "n1"), &conf1, chrono::Utc::now(), &localhost).unwrap() {
        Outcome::Configured { http_ca_fingerprint, .. } => http_ca_fingerprint,
        other => panic!("node 1 was not auto-configured: {other:?}"),
    };
    let node1 = spawn(load(dir.path(), "n1")).await;
    let ca = std::fs::read(conf1.join("certs").join(security_autoconfig::HTTP_CA_CERT)).unwrap();
    let client = reqwest::Client::builder()
        .add_root_certificate(reqwest::Certificate::from_pem(&ca).unwrap())
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let base1 = format!("https://127.0.0.1:{}", ports::http_port(&node1));
    let mut ready = false;
    for _ in 0..300 {
        if get(&client, &format!("{base1}/_cluster/health"), &basic()).await.0 == 200 {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(ready, "node 1 did not answer over HTTPS");

    // The node token: the node's HTTP address, its HTTP CA, a 30-minute key.
    let node_token = create_token(&conf1, &base1, "node").await;
    assert_eq!(node_token.version, "8.14.0");
    assert_eq!(node_token.fingerprint, fingerprint);
    assert_eq!(node_token.bound_address, vec![node1.local.http_address.clone()]);
    let (status, keys) = get(&client, &format!("{base1}/_security/api_key?owner=false"), &basic()).await;
    assert_eq!(status, 200, "{keys}");
    let id = node_token.api_key.split(':').next().unwrap();
    let key = keys["api_keys"].as_array().unwrap().iter().find(|k| k["id"] == id).unwrap_or_else(|| panic!("{keys}"));
    assert!(key["name"].as_str().unwrap().starts_with("enrollment_token_API_key_"), "{key}");
    let lifetime = key["expiration"].as_i64().unwrap() - key["creation"].as_i64().unwrap();
    assert!((29 * 60_000..=31 * 60_000).contains(&lifetime), "{key}");
    // The key enrolls nodes and does nothing else.
    let (status, body) = get(&client, &format!("{base1}/_security/enroll/node"), &api_key(&node_token)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["nodes_addresses"], serde_json::json!([node1.local.transport_address.clone()]));
    assert_eq!(get(&client, &format!("{base1}/_cluster/health"), &api_key(&node_token)).await.0, 403);
    assert_eq!(get(&client, &format!("{base1}/_security/enroll/kibana"), &api_key(&node_token)).await.0, 403);

    // Node 2 enrolls and starts.
    let conf2 = new_conf(dir.path(), "n2");
    let s2 = load(dir.path(), "n2");
    match security_autoconfig::enroll_at(&s2, &conf2, &node_token.encoded(), true, chrono::Utc::now(), &localhost).await {
        Ok(Outcome::Configured { http_ca_fingerprint, .. }) => assert_eq!(http_ca_fingerprint, fingerprint),
        other => panic!("node 2 did not enroll: {other:?}"),
    }
    let yml = std::fs::read_to_string(conf2.join("elasticsearch.yml")).unwrap();
    assert!(yml.contains(&format!("# Discover existing nodes in the cluster\ndiscovery.seed_hosts: [\"{}\"]\n", node1.local.transport_address)), "{yml}");
    assert!(!yml.contains("cluster.initial_master_nodes"), "{yml}");
    // The cluster's transport key and CAs; its own HTTP key.
    for f in [security_autoconfig::TRANSPORT_CA_CERT, security_autoconfig::TRANSPORT_KEY, security_autoconfig::TRANSPORT_CERT, security_autoconfig::HTTP_CA_CERT] {
        let a = elasticsearch_ssl::parse_certificates(&std::fs::read(conf1.join("certs").join(f)).unwrap()).ok();
        let b = elasticsearch_ssl::parse_certificates(&std::fs::read(conf2.join("certs").join(f)).unwrap()).ok();
        assert_eq!(a, b, "{f}");
    }
    assert_ne!(std::fs::read(conf1.join("certs").join(security_autoconfig::HTTP_KEY)).unwrap(), std::fs::read(conf2.join("certs").join(security_autoconfig::HTTP_KEY)).unwrap());
    let node2 = spawn(load(dir.path(), "n2")).await;
    let base2 = format!("https://127.0.0.1:{}", ports::http_port(&node2));

    // Both nodes see a two-node cluster over HTTPS (node 2's certificate is
    // issued by the cluster's HTTP CA).
    let mut formed = false;
    for _ in 0..600 {
        let (s1, h1) = get(&client, &format!("{base1}/_cluster/health"), &basic()).await;
        let (s2, h2) = get(&client, &format!("{base2}/_cluster/health"), &basic()).await;
        if s1 == 200 && s2 == 200 && h1["number_of_nodes"] == 2 && h2["number_of_nodes"] == 2 {
            formed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(formed, "node 2 did not join");
    let (status, nodes) = get(&client, &format!("{base2}/_cat/nodes?format=json&h=name"), &basic()).await;
    assert_eq!(status, 200, "{nodes}");
    let mut names: Vec<&str> = nodes.as_array().unwrap().iter().filter_map(|n| n["name"].as_str()).collect();
    names.sort();
    assert_eq!(names, ["n1", "n2"]);
    // The enroll API now lists both nodes' transport addresses.
    let (status, body) = get(&client, &format!("{base2}/_security/enroll/node"), &basic()).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["nodes_addresses"].as_array().unwrap().len(), 2, "{body}");

    // Kibana: the HTTP CA and an `elastic/kibana` service account token.
    let kibana_token = create_token(&conf1, &base1, "kibana").await;
    assert_eq!(kibana_token.fingerprint, fingerprint);
    assert_eq!(get(&client, &format!("{base2}/_security/enroll/node"), &api_key(&kibana_token)).await.0, 403);
    let (status, body) = get(&client, &format!("{base2}/_security/enroll/kibana"), &api_key(&kibana_token)).await;
    assert_eq!(status, 200, "{body}");
    let keys: Vec<&str> = body.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["token", "http_ca"]);
    assert!(body["token"]["name"].as_str().unwrap().starts_with("enroll-process-token-"), "{body}");
    let ca_der = base64::engine::general_purpose::STANDARD.decode(body["http_ca"].as_str().unwrap()).unwrap();
    assert_eq!(ca_der, elasticsearch_ssl::parse_certificates(&ca).unwrap()[0].to_vec());
    let bearer = format!("Bearer {}", body["token"]["value"].as_str().unwrap());
    let mut authenticated = Value::Null;
    for _ in 0..100 {
        let (status, me) = get(&client, &format!("{base1}/_security/_authenticate"), &bearer).await;
        if status == 200 {
            authenticated = me;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(authenticated["username"], "elastic/kibana", "{authenticated}");

    // A configured node is not enrolled again.
    let e = security_autoconfig::enroll_at(&load(dir.path(), "n2"), &conf2, &node_token.encoded(), false, chrono::Utc::now(), &localhost).await.unwrap_err();
    assert_eq!(e.exit_code, security_autoconfig::exit_codes::NOOP, "{}", e.message);
    // A token pinned to another CA is refused by the TLS layer.
    let wrong = EnrollmentToken::new(node_token.api_key.clone(), "00".repeat(32), node_token.bound_address.clone());
    let conf3 = new_conf(dir.path(), "n3");
    let e = security_autoconfig::enroll_at(&load(dir.path(), "n3"), &conf3, &wrong.encoded(), false, chrono::Utc::now(), &localhost).await.unwrap_err();
    assert_eq!(e.exit_code, security_autoconfig::exit_codes::UNAVAILABLE, "{}", e.message);
    assert!(!conf3.join("certs").exists());
}
