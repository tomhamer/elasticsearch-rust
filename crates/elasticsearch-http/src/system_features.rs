//! System features and small node admin APIs:
//! `GET /_features`, `POST /_features/_reset`,
//! `GET|POST /_migration/system_features`,
//! `POST /_nodes[/{node_id}]/reload_secure_settings`,
//! `GET /_nodes/{node_id}/_repositories_metering` (+ `DELETE .../{max_archive_version}`)
//! and `GET /_ssl/certificates`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::mapping::wildcard_match;
use elasticsearch_engine::node::{Node, NODE_ID};
use serde_json::{json, Map, Value};
use std::sync::Arc;

use elasticsearch_engine::version::INDEX_RELEASE_VERSION;

/// A system feature: name, description and its system index patterns.
struct Feature {
    name: &'static str,
    description: &'static str,
    indices: &'static [&'static str],
}

const fn f(name: &'static str, description: &'static str, indices: &'static [&'static str]) -> Feature {
    Feature { name, description, indices }
}

/// ES9 system features, in `GET /_features` order.
const FEATURES: &[Feature] = &[
    f("security", "Manages configuration for Security features, such as users and roles", &[".security-*", ".security"]),
    f("searchable_snapshots", "Manages caches and configuration for searchable snapshots", &[".snapshot-blob-cache*"]),
    f("logstash_management", "Enables Logstash Central Management pipeline storage", &[".logstash*"]),
    f("transform", "Manages configuration and state for transforms", &[".transform-internal-*"]),
    f("kibana", "Manages Kibana configuration and reports", &[".kibana*", ".reporting-*", ".apm-agent-configuration*", ".apm-custom-link*"]),
    f("synonyms", "Manages synonyms", &[".synonyms*"]),
    f("async_search", "Manages results of async searches", &[".async-search*"]),
    f("ent_search", "Manages configuration for Enterprise Search features", &[".search-app-*", ".query-rules-*", ".connector-secrets*"]),
    f("machine_learning", "Provides anomaly detection and forecasting functionality", &[".ml-*"]),
    f("geoip", "Manages data related to GeoIP database downloader", &[".geoip_databases*"]),
    f("watcher", "Manages Watch definitions and state", &[".watches*", ".triggered_watches*"]),
    f("fleet", "Manages configuration for Fleet", &[".fleet-*"]),
    f("enrich", "Manages data related to Enrich policies", &[".enrich-*"]),
    f("inference_plugin", "Inference plugin for managing inference services and inference", &[".inference*", ".secrets-inference*"]),
    f("tasks", "Manages task results", &[".tasks*"]),
];

fn sorted_features() -> Vec<&'static Feature> {
    let mut v: Vec<&Feature> = FEATURES.iter().collect();
    v.sort_by_key(|f| f.name);
    v
}

/// Concrete indices (not data stream backing indices) owned by `feat`.
fn feature_indices(node: &Node, feat: &Feature) -> Vec<String> {
    let mut out: Vec<String> = node
        .all_indices()
        .into_iter()
        .map(|s| s.name.clone())
        .filter(|n| !n.starts_with(".ds-") && feat.indices.iter().any(|p| wildcard_match(p, n)))
        .collect();
    out.sort();
    out
}

/// Every feature's concrete system indices on `node` (what
/// `POST /_features/_reset` deletes). A cluster deletes them as a published
/// metadata change before the reset runs on the master's engine.
pub fn feature_system_indices(node: &Node) -> Vec<String> {
    let mut out: Vec<String> = FEATURES.iter().flat_map(|feat| feature_indices(node, feat)).collect();
    out.sort();
    out.dedup();
    out
}

/// `/_features[/_reset]`.
pub async fn handle_features(node: Arc<Node>, req: Req) -> R {
    match (req.seg(1), req.segs.len()) {
        ("", 1) => match req.method {
            Method::GET => Ok(Resp::ok(json!({"features": FEATURES.iter().map(|f| json!({"name": f.name, "description": f.description})).collect::<Vec<_>>()}))),
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("_reset", 2) => match req.method {
            Method::POST => blocking(move || reset_features(&node)).await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        _ => crate::not_found_handler(&req),
    }
}

/// `POST /_features/_reset`: drop every feature's system indices and the
/// state the engine keeps for it outside indices.
fn reset_features(node: &Node) -> R {
    let mut out = Vec::new();
    for feat in sorted_features() {
        let r = reset_feature(node, feat);
        out.push(match r {
            Ok(()) => json!({"feature_name": feat.name, "status": "SUCCESS"}),
            Err(e) => json!({"feature_name": feat.name, "status": "FAILURE", "exception": e.to_json_with_trace(false)["error"].clone()}),
        });
    }
    Ok(Resp::ok(json!({"features": out})))
}

fn reset_feature(node: &Node, feat: &Feature) -> Result<(), EsError> {
    // Stop feature activity before its indices go away.
    match feat.name {
        "transform" => elasticsearch_engine::transform::service_ref(node).reset_all()?,
        "async_search" => node.async_searches.lock().clear(),
        "machine_learning" => elasticsearch_engine::ml::store::service(node).reset(node)?,
        _ => {}
    }
    for idx in feature_indices(node, feat) {
        match node.delete_index(&idx) {
            Ok(()) => {}
            Err(e) if e.status == 404 => {}
            Err(e) => return Err(e),
        }
    }
    match feat.name {
        "security" => node.security.reset()?,
        "ent_search" => {
            let state = elasticsearch_engine::search_ext::ext(node);
            state.rulesets.write().clear();
            state.search_apps.write().clear();
            state.persist()?;
        }
        "inference_plugin" => elasticsearch_engine::inference::service_for(&node.data).reset()?,
        "watcher" => elasticsearch_engine::watcher::service(node).reset(),
        "synonyms" => {
            let st = elasticsearch_engine::search_ext::ext(node);
            let sets: Vec<String> = st.synonyms.read().keys().cloned().collect();
            if !sets.is_empty() {
                st.synonyms.write().clear();
                for s in &sets {
                    st.publish_synonyms(s);
                }
                st.persist()?;
            }
        }
        _ => {}
    }
    node.persist_state()?;
    Ok(())
}

/// `GET|POST /_migration/system_features`.
pub async fn handle_migration(node: Arc<Node>, req: Req) -> R {
    if req.segs.len() != 2 {
        return crate::not_found_handler(&req);
    }
    match req.method {
        Method::GET => blocking(move || {
            let features: Vec<Value> = sorted_features()
                .into_iter()
                .map(|feat| {
                    let indices: Vec<Value> = feature_indices(&node, feat).into_iter().map(|i| json!({"index": i, "version": INDEX_RELEASE_VERSION})).collect();
                    json!({"feature_name": feat.name, "minimum_index_version": INDEX_RELEASE_VERSION, "migration_status": "NO_MIGRATION_NEEDED", "indices": indices})
                })
                .collect();
            Ok(Resp::ok(json!({"features": features, "migration_status": "NO_MIGRATION_NEEDED"})))
        })
        .await,
        Method::POST => Ok(Resp::ok(json!({"accepted": false, "reason": "No system indices require migration"}))),
        _ => method_not_allowed(&req, &["GET", "POST"]),
    }
}

/// Whether `spec` (`_nodes/{node_id}`) selects this node.
fn selects_node(node: &Node, spec: &str) -> bool {
    spec.split(',').map(str::trim).any(|s| match s {
        "" | "_all" | "_local" | "_master" | "*" => true,
        _ => match s.split_once(':') {
            Some((_, v)) => v != "false",
            None => s == NODE_ID || s == node.name || wildcard_match(s, &node.name) || wildcard_match(s, NODE_ID),
        },
    })
}

fn nodes_envelope(node: &Node, nodes: Map<String, Value>) -> Value {
    let n = nodes.len();
    json!({"_nodes": {"total": n, "successful": n, "failed": 0}, "cluster_name": node.cluster_name, "nodes": nodes})
}

/// `_nodes/.../reload_secure_settings` and `_nodes/{id}/_repositories_metering[/{v}]`.
pub async fn handle_nodes(node: Arc<Node>, req: Req) -> R {
    let segs: Vec<&str> = req.segs.iter().map(String::as_str).collect();
    match segs.as_slice() {
        ["_nodes", "reload_secure_settings"] | ["_nodes", _, "reload_secure_settings"] => {
            if req.method != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            let spec = if segs.len() == 3 { segs[1] } else { "_all" };
            reload_secure_settings(&node, &req, spec)
        }
        ["_nodes", spec, "_repositories_metering"] => {
            if req.method != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            Ok(Resp::ok(metering(&node, spec)))
        }
        ["_nodes", spec, "_repositories_metering", version] => {
            if req.method != Method::DELETE {
                return method_not_allowed(&req, &["DELETE"]);
            }
            if version.parse::<i64>().is_err() {
                return Err(EsError::illegal_argument(format!("Failed to parse long parameter [max_archive_version] with value [{version}]")));
            }
            Ok(Resp::ok(metering(&node, spec)))
        }
        _ => crate::not_found_handler(&req),
    }
}

/// Repositories metering: no metered (cloud) repositories on this node.
fn metering(node: &Node, spec: &str) -> Value {
    let mut nodes = Map::new();
    if selects_node(node, spec) {
        nodes.insert(NODE_ID.to_string(), json!([]));
    }
    nodes_envelope(node, nodes)
}

/// The node's keystore has no password (ES's default `elasticsearch.keystore`):
/// an empty password reloads, any other fails like ES's AES-GCM tag check.
fn reload_secure_settings(node: &Node, req: &Req, spec: &str) -> R {
    let body = req.json()?;
    let pw = body.get("secure_settings_password").and_then(Value::as_str).unwrap_or("");
    let mut nodes = Map::new();
    if selects_node(node, spec) {
        let mut n = json!({"name": node.name});
        if !pw.is_empty() {
            n["reload_exception"] = json!({
                "type": "security_exception",
                "reason": "Provided keystore password was incorrect",
                "caused_by": {"type": "i_o_exception", "reason": "javax.crypto.AEADBadTagException: Tag mismatch",
                              "caused_by": {"type": "a_e_a_d_bad_tag_exception", "reason": "Tag mismatch"}}
            });
        }
        nodes.insert(NODE_ID.to_string(), n);
    }
    Ok(Resp::ok(nodes_envelope(node, nodes)))
}

/// `GET /_ssl/certificates` (security enabled): no TLS keystores or PEM
/// files are configured, so no certificates.
pub fn ssl_certificates(node: &Node, req: &Req) -> R {
    if req.method != Method::GET {
        return method_not_allowed(req, &["GET"]);
    }
    let settings = node.node_settings.read().clone();
    let get = |k: &str| settings.get(k).and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_bool().map(|b| b.to_string())));
    let conf = get("path.conf").map(std::path::PathBuf::from).unwrap_or_else(|| node.data.join("config"));
    let mut out: Vec<(String, Value)> = Vec::new();
    for ctx in ["xpack.security.transport.ssl", "xpack.security.http.ssl"] {
        if get(&format!("{ctx}.enabled")).as_deref() != Some("true") {
            continue;
        }
        let Some(cert) = get(&format!("{ctx}.certificate")) else { continue };
        let has_key = get(&format!("{ctx}.key")).is_some();
        let path = conf.join(&cert);
        let Ok(pem) = std::fs::read_to_string(&path) else { continue };
        for der in pem_blocks(&pem, "CERTIFICATE") {
            if let Some(info) = x509_info(&der) {
                let key = format!("{cert}\u{0}{}", info.serial);
                if out.iter().any(|(k, _)| *k == key) {
                    continue;
                }
                out.push((
                    key,
                    json!({
                        "path": cert, "format": "PEM", "alias": null, "subject_dn": info.subject, "serial_number": info.serial,
                        "has_private_key": has_key, "expiry": info.expiry, "issuer": info.issuer
                    }),
                ));
            }
        }
    }
    // ES orders by path, then alias, then serial number.
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Resp::ok(Value::Array(out.into_iter().map(|(_, v)| v).collect())))
}

/// Base64 bodies of the `-----BEGIN {label}-----` blocks of a PEM file.
fn pem_blocks(pem: &str, label: &str) -> Vec<Vec<u8>> {
    use base64::Engine;
    let (begin, end) = (format!("-----BEGIN {label}-----"), format!("-----END {label}-----"));
    let mut out = Vec::new();
    let mut rest = pem;
    while let Some(i) = rest.find(&begin) {
        let after = &rest[i + begin.len()..];
        let Some(j) = after.find(&end) else { break };
        let b64: String = after[..j].chars().filter(|c| !c.is_whitespace()).collect();
        if let Ok(d) = base64::engine::general_purpose::STANDARD.decode(b64) {
            out.push(d);
        }
        rest = &after[j + end.len()..];
    }
    out
}

struct CertInfo {
    serial: String,
    issuer: String,
    subject: String,
    expiry: String,
}

/// One DER TLV: (tag, content, rest).
fn der(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let tag = *b.first()?;
    let l0 = *b.get(1)? as usize;
    let (len, off) = if l0 < 0x80 {
        (l0, 2)
    } else {
        let n = l0 & 0x7f;
        let mut len = 0usize;
        for k in 0..n {
            len = (len << 8) | *b.get(2 + k)? as usize;
        }
        (len, 2 + n)
    };
    let content = b.get(off..off + len)?;
    Some((tag, content, &b[off + len..]))
}

/// Java `X500Principal.toString()`: RDNs most-specific first, `, `-joined.
fn x500_name(seq: &[u8]) -> Option<String> {
    let mut rdns = Vec::new();
    let mut rest = seq;
    while !rest.is_empty() {
        let (_, set, r) = der(rest)?;
        rest = r;
        let mut parts = Vec::new();
        let mut s = set;
        while !s.is_empty() {
            let (_, atv, r2) = der(s)?;
            s = r2;
            let (_, oid, after) = der(atv)?;
            let (_, val, _) = der(after)?;
            let name = match oid {
                [0x55, 0x04, 0x03] => "CN".to_string(),
                [0x55, 0x04, 0x0b] => "OU".to_string(),
                [0x55, 0x04, 0x0a] => "O".to_string(),
                [0x55, 0x04, 0x06] => "C".to_string(),
                [0x55, 0x04, 0x07] => "L".to_string(),
                [0x55, 0x04, 0x08] => "ST".to_string(),
                [0x55, 0x04, 0x09] => "STREET".to_string(),
                [0x09, 0x92, 0x26, 0x89, 0x93, 0xf2, 0x2c, 0x64, 0x01, 0x19] => "DC".to_string(),
                [0x09, 0x92, 0x26, 0x89, 0x93, 0xf2, 0x2c, 0x64, 0x01, 0x01] => "UID".to_string(),
                [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x01] => "EMAILADDRESS".to_string(),
                other => format!("OID.{}", oid_string(other)),
            };
            parts.push(format!("{name}={}", String::from_utf8_lossy(val)));
        }
        rdns.push(parts.join(" + "));
    }
    rdns.reverse();
    Some(rdns.join(", "))
}

fn oid_string(b: &[u8]) -> String {
    let mut out = Vec::new();
    if let Some(f) = b.first() {
        out.push((f / 40) as u64);
        out.push((f % 40) as u64);
    }
    let mut v = 0u64;
    for x in b.iter().skip(1) {
        v = (v << 7) | (x & 0x7f) as u64;
        if x & 0x80 == 0 {
            out.push(v);
            v = 0;
        }
    }
    out.iter().map(u64::to_string).collect::<Vec<_>>().join(".")
}

fn x509_time(tag: u8, v: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(v).ok()?;
    let (year, rest) = if tag == 0x17 {
        let yy: i32 = s.get(0..2)?.parse().ok()?;
        (if yy >= 50 { 1900 + yy } else { 2000 + yy }, s.get(2..)?)
    } else {
        (s.get(0..4)?.parse().ok()?, s.get(4..)?)
    };
    let f = |i: usize| rest.get(i..i + 2);
    Some(format!("{year:04}-{}-{}T{}:{}:{}.000Z", f(0)?, f(2)?, f(4)?, f(6)?, f(8)?))
}

fn x509_info(cert: &[u8]) -> Option<CertInfo> {
    let (_, c, _) = der(cert)?;
    let (_, tbs, _) = der(c)?;
    let mut rest = tbs;
    // optional [0] version
    let (tag, _, r) = der(rest)?;
    if tag == 0xa0 {
        rest = r;
    }
    let (_, serial, r) = der(rest)?;
    rest = r;
    let (_, _sigalg, r) = der(rest)?;
    rest = r;
    let (_, issuer, r) = der(rest)?;
    rest = r;
    let (_, validity, r) = der(rest)?;
    rest = r;
    let (_, subject, _) = der(rest)?;
    let (_, _nb, after) = der(validity)?;
    let (na_tag, na, _) = der(after)?;
    // `BigInteger.toString(16)`.
    let mut hex: String = serial.iter().map(|b| format!("{b:02x}")).collect();
    let neg = serial.first().is_some_and(|b| b & 0x80 != 0);
    if !neg {
        hex = hex.trim_start_matches('0').to_string();
        if hex.is_empty() {
            hex = "0".into();
        }
    }
    Some(CertInfo { serial: hex, issuer: x500_name(issuer)?, subject: x500_name(subject)?, expiry: x509_time(na_tag, na)? })
}
