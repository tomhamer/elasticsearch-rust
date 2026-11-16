//! Full cluster restart with transport TLS (the Helm e2e's step 13): every
//! node of a 3-master + 2-data cluster is killed at once and started again
//! from its data directory, half the rounds on new ports (pods come back
//! with new IPs, so the persisted cluster state names stale addresses).
//! A master must be elected and stay elected within 30 s, without the
//! term running away: ES keeps each node's election scheduler, whose
//! randomised delays grow with every attempt, until a state of the current
//! term has been applied, so a master that loses its first publication to
//! a competing election does not restart its own elections at
//! `cluster.election.initial_timeout` (the livelock this guards against:
//! leader in term N deposed ~100 ms later by `joinLeaderInTerm`, term in
//! the thousands after 10 minutes). Indexed documents come back.
//!
//! `FULL_RESTART_ROUNDS` sets the number of restarts (default 3).
mod common;
use common::procs::{start_procs_per_node, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const NODES: usize = 5;
const DOCS: usize = 50;
/// Bounds per restart.
const STABLE_WITHIN: Duration = Duration::from_secs(30);
const MAX_TERM_GROWTH: u64 = 30;
const MAX_ELECTIONS: usize = 15;

/// CA + one node certificate (SANs `localhost`, `127.0.0.1`; server and
/// client auth, as transport TLS needs both), written under `dir`.
fn write_pki(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair, SanType};
    std::fs::create_dir_all(dir).unwrap();
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.distinguished_name.push(DnType::CommonName, "full-restart-test-ca");
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let node_key = KeyPair::generate().unwrap();
    let mut p = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    p.subject_alt_names.push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
    p.distinguished_name.push(DnType::CommonName, "full-restart-node");
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth, ExtendedKeyUsagePurpose::ClientAuth];
    let cert = p.signed_by(&node_key, &ca, &ca_key).unwrap();
    let (ca_path, cert_path, key_path) = (dir.join("ca.crt"), dir.join("node.crt"), dir.join("node.key"));
    std::fs::write(&ca_path, ca.pem()).unwrap();
    std::fs::write(&cert_path, cert.pem()).unwrap();
    std::fs::write(&key_path, node_key.serialize_pem()).unwrap();
    (ca_path, cert_path, key_path)
}

/// What node `i` itself has applied: (master node id, term, node count).
async fn view(p: &Procs, i: usize) -> Option<(String, u64, usize)> {
    let (st, v) = p.call(i, Method::GET, "/_cluster/state/master_node,metadata,nodes?local=true", None).await;
    if st != 200 {
        return None;
    }
    let master = v["master_node"].as_str()?.to_string();
    let term = v["metadata"]["cluster_coordination"]["term"].as_u64()?;
    let nodes = v["nodes"].as_object().map_or(0, |m| m.len());
    Some((master, term, nodes))
}

/// Every node follows the same master in the same term with all nodes
/// joined.
async fn agreed(p: &Procs) -> Option<(String, u64)> {
    let mut seen: Option<(String, u64)> = None;
    for i in 0..NODES {
        let (m, t, n) = view(p, i).await?;
        if n != NODES || seen.as_ref().is_some_and(|s| *s != (m.clone(), t)) {
            return None;
        }
        seen = Some((m, t));
    }
    seen
}

/// `coordinator mode changed` lines since byte `from` of node `i`'s log:
/// (leader elections, highest term).
fn mode_changes(p: &Procs, i: usize, from: u64) -> (usize, u64) {
    let text = std::fs::read(p.log_path(i)).unwrap_or_default();
    let text = String::from_utf8_lossy(&text[(from as usize).min(text.len())..]).into_owned();
    let (mut leaders, mut max_term) = (0, 0);
    for line in text.lines().filter(|l| l.contains("coordinator mode changed")) {
        if line.contains("Leader") {
            leaders += 1;
        }
        let term = line.split("term").nth(1).map(|r| r.trim_start_matches(|c: char| !c.is_ascii_digit()).chars().take_while(char::is_ascii_digit).collect::<String>());
        if let Some(t) = term.and_then(|t| t.parse::<u64>().ok()) {
            max_term = max_term.max(t);
        }
    }
    (leaders, max_term)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_restart_with_transport_tls_elects_a_stable_master() {
    let rounds: usize = std::env::var("FULL_RESTART_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let pki = common::scratch_dir("full-restart-tls-pki");
    let (ca, cert, key) = write_pki(&pki);
    let (ca, cert, key) = (ca.to_string_lossy().into_owned(), cert.to_string_lossy().into_owned(), key.to_string_lossy().into_owned());
    let extra: Vec<(&str, &str)> = vec![
        ("cluster.initial_master_nodes", "n0,n1,n2"),
        ("xpack.security.transport.ssl.enabled", "true"),
        ("xpack.security.transport.ssl.verification_mode", "full"),
        ("xpack.security.transport.ssl.certificate", &cert),
        ("xpack.security.transport.ssl.key", &key),
        ("xpack.security.transport.ssl.certificate_authorities", &ca),
    ];
    let data: &[(&str, &str)] = &[("node.roles", "data")];
    let roles: [&[(&str, &str)]; NODES] = [&[], &[], &[], data, data];
    let mut p = start_procs_per_node(&extra, &roles).await;

    let (st, v) = p.call(0, Method::PUT, "/fr", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    let mut bulk = String::new();
    for d in 0..DOCS {
        bulk.push_str(&format!("{{\"index\":{{\"_index\":\"fr\",\"_id\":\"{d}\"}}}}\n{{\"n\":{d}}}\n"));
    }
    let (st, v) = p.ndjson(0, "/_bulk?refresh=true", bulk).await;
    assert!(st == 200 && v["errors"] == false, "{v}");
    p.wait_health(0, "fr", "green", 120).await;

    let mut report: Vec<String> = Vec::new();
    for round in 0..rounds {
        let (_, term_before) = agreed(&p).await.expect("cluster formed before the restart");
        let offsets: Vec<u64> = (0..NODES).map(|i| std::fs::metadata(p.log_path(i)).map(|m| m.len()).unwrap_or(0)).collect();
        for i in 0..NODES {
            p.kill(i);
        }
        // Odd rounds: every node comes back on new ports (new pod IPs).
        let new_ports = round % 2 == 1;
        if new_ports {
            for i in 0..NODES {
                p.p.http[i] = 0;
                p.p.transport[i] = 0;
            }
        }
        let started = Instant::now();
        for i in 0..NODES {
            p.spawn(i, roles[i]).await;
        }
        let mut stable: Option<(String, u64)> = None;
        while started.elapsed() < STABLE_WITHIN {
            if let Some(a) = agreed(&p).await {
                // Still the same master and term a moment later.
                tokio::time::sleep(Duration::from_secs(2)).await;
                if agreed(&p).await.as_ref() == Some(&a) {
                    stable = Some(a);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let took = started.elapsed();
        let (mut elections, mut max_term) = (0, 0);
        for i in 0..NODES {
            let (l, t) = mode_changes(&p, i, offsets[i]);
            elections += l;
            max_term = max_term.max(t);
        }
        let tails = || (0..NODES).map(|i| format!("--- n{i}\n{}", p.log_tail(i))).collect::<Vec<_>>().join("\n");
        let Some((master, term)) = stable else {
            panic!("round {round} (new ports: {new_ports}): no stable master within {STABLE_WITHIN:?}; term {term_before} -> {max_term}, {elections} elections\n{}", tails());
        };
        let growth = max_term.max(term).saturating_sub(term_before);
        report.push(format!("round {round}: new_ports={new_ports} stable after {:.1}s, master {master}, term {term_before} -> {term} (max seen {max_term}), {elections} leader elections", took.as_secs_f64()));
        assert!(growth <= MAX_TERM_GROWTH, "round {round}: term grew by {growth} ({term_before} -> {max_term})\n{}", tails());
        assert!(elections <= MAX_ELECTIONS, "round {round}: {elections} leader elections before stabilising\n{}", tails());
        p.wait_health(0, "fr", "green", 120).await;
        let (st, v) = p.call(0, Method::GET, "/fr/_count", None).await;
        assert_eq!((st, v["count"].as_u64()), (200, Some(DOCS as u64)), "round {round}: documents lost: {v}");
    }
    let v: Value = Value::Array(report.iter().map(|r| Value::String(r.clone())).collect());
    eprintln!("full restart with transport TLS, {rounds} rounds:\n{}", report.join("\n"));
    let _ = std::fs::write(p.dir.join("full_restart_report.json"), v.to_string());
}
