//! Cross-cluster search between separate rustic clusters, each a set of
//! real `elasticsearch-node` processes: remote cluster connections (sniff
//! and proxy, `_remote/info`, settings updates, reconnects), CCS searches
//! with aggregations under both `ccs_minimize_roundtrips` modes, `_count`,
//! `_msearch`, async search, `_field_caps`, `_resolve/index`,
//! `_resolve/cluster`, `skip_unavailable`, remote failures, and remote-side
//! authorization of the local user (certificate-based model: the remote
//! cluster's role definitions decide).
mod common;
use common::procs::{start_procs, start_procs_secured_env, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const LICENSE: (&str, &str) = ("xpack.license.self_generated.type", "trial");

async fn ok(p: &Procs, node: usize, method: Method, path: &str, body: Option<Value>) -> Value {
    let (st, v) = p.call(node, method.clone(), path, body).await;
    assert!((200..300).contains(&st), "{method} {path}: {st} {v}");
    v
}

/// Index `n` documents `{marker, n, tag}` into `index` and refresh.
async fn load(p: &Procs, index: &str, shards: u32, marker: &str, n: u64) {
    ok(p, 0, Method::PUT, &format!("/{index}"), Some(json!({
        "settings": {"number_of_shards": shards, "number_of_replicas": if p.len() > 1 { 1 } else { 0 }},
        "mappings": {"properties": {"marker": {"type": "keyword"}, "n": {"type": "long"}, "tag": {"type": "keyword"}}}
    }))).await;
    let mut bulk = String::new();
    for i in 0..n {
        bulk.push_str(&format!("{{\"index\":{{\"_index\":\"{index}\",\"_id\":\"{marker}-{i}\"}}}}\n"));
        bulk.push_str(&json!({"marker": marker, "n": i, "tag": format!("t{}", i % 3)}).to_string());
        bulk.push('\n');
    }
    let (st, v) = p.ndjson(0, "/_bulk?refresh=true", bulk).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["errors"], false, "{v}");
    p.wait_health(0, index, "green", 120).await;
}

/// Wait until every node of `p` reports `alias` connected (or, with
/// `connected = false`, disconnected) in `_remote/info`.
async fn wait_remote(p: &Procs, alias: &str, connected: bool, secs: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let mut all = true;
        let mut last = Value::Null;
        for i in p.live() {
            let (st, v) = p.call(i, Method::GET, "/_remote/info", None).await;
            all &= st == 200 && v[alias]["connected"] == json!(connected);
            last = v;
        }
        if all {
            return last;
        }
        assert!(Instant::now() < deadline, "[{alias}] connected != {connected}: {last}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn markers(v: &Value) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = v["hits"]["hits"].as_array().into_iter().flatten()
        .map(|h| (h["_index"].as_str().unwrap_or("").to_string(), h["_source"]["marker"].as_str().unwrap_or("").to_string())).collect();
    out.sort();
    out.dedup();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cross_cluster_search_between_process_clusters() {
    let a = start_procs(2, &[LICENSE]).await;
    let mut b = start_procs(2, &[LICENSE]).await;
    // Distinct clusters.
    let ua = ok(&a, 0, Method::GET, "/", None).await["cluster_uuid"].clone();
    let ub = ok(&b, 0, Method::GET, "/", None).await["cluster_uuid"].clone();
    assert_ne!(ua, ub);
    load(&a, "logs", 2, "a", 30).await;
    load(&b, "logs", 3, "b", 20).await;
    load(&b, "only-b", 1, "ob", 5).await;
    let seed_b = format!("127.0.0.1:{}", b.p.transport[0]);
    // A dead address (a port nothing listens on: b's spare http port).
    let dead = format!("127.0.0.1:{}", common::ports::probe_free_ports(1)[0]);
    a.settings(0, "persistent", json!({
        "cluster.remote.b.seeds": [seed_b.clone()],
        "cluster.remote.b.skip_unavailable": false,
        "cluster.remote.down.seeds": [dead.clone()],
        "cluster.remote.down.skip_unavailable": true,
        "cluster.remote.req.seeds": [dead.clone()],
        "cluster.remote.req.skip_unavailable": false,
    })).await;
    let info = wait_remote(&a, "b", true, 60).await;
    assert_eq!(info["b"]["mode"], "sniff", "{info}");
    assert_eq!(info["b"]["seeds"], json!([seed_b]), "{info}");
    assert_eq!(info["b"]["num_nodes_connected"], 2, "both gateway nodes: {info}");
    assert_eq!(info["b"]["max_connections_per_cluster"], 3, "{info}");
    assert_eq!(info["b"]["skip_unavailable"], false, "{info}");
    assert_eq!(info["down"]["connected"], false, "{info}");
    assert_eq!(info["down"]["num_nodes_connected"], 0, "{info}");

    // Search + aggregations, both reduce modes, from both coordinators.
    for node in [0, 1] {
        for mrt in ["true", "false"] {
            let v = ok(&a, node, Method::POST, &format!("/logs,b:logs/_search?ccs_minimize_roundtrips={mrt}"), Some(json!({
                "size": 100, "track_total_hits": true, "sort": [{"marker": "asc"}, {"n": "asc"}],
                "aggs": {"m": {"terms": {"field": "marker"}}, "avg_n": {"avg": {"field": "n"}}, "card": {"cardinality": {"field": "n"}},
                         "per_tag": {"terms": {"field": "tag"}, "aggs": {"max_n": {"max": {"field": "n"}}}}}
            }))).await;
            assert_eq!(v["hits"]["total"]["value"], 50, "{v}");
            assert_eq!(markers(&v), vec![("b:logs".to_string(), "b".to_string()), ("logs".to_string(), "a".to_string())], "{v}");
            let buckets = &v["aggregations"]["m"]["buckets"];
            assert_eq!(buckets, &json!([{"key": "a", "doc_count": 30}, {"key": "b", "doc_count": 20}]), "{v}");
            // avg over 0..30 and 0..20: (435 + 190) / 50.
            assert!((v["aggregations"]["avg_n"]["value"].as_f64().unwrap() - 12.5).abs() < 1e-9, "{v}");
            assert_eq!(v["aggregations"]["card"]["value"], 30, "{v}");
            assert_eq!(v["_shards"]["total"], 5, "{v}");
            assert_eq!(v["_shards"]["failed"], 0, "{v}");
            let c = &v["_clusters"];
            assert_eq!((c["total"].clone(), c["successful"].clone(), c["skipped"].clone(), c["failed"].clone()), (json!(2), json!(2), json!(0), json!(0)), "{v}");
            assert_eq!(c["details"]["(local)"]["status"], "successful", "{v}");
            assert_eq!(c["details"]["(local)"]["_shards"]["total"], 2, "{v}");
            assert_eq!(c["details"]["b"]["status"], "successful", "{v}");
            assert_eq!(c["details"]["b"]["indices"], "logs", "{v}");
            assert_eq!(c["details"]["b"]["_shards"]["total"], 3, "{v}");
            // Pages across clusters in sort order.
            let page = ok(&a, node, Method::POST, &format!("/logs,b:logs/_search?ccs_minimize_roundtrips={mrt}"), Some(json!({
                "from": 25, "size": 10, "sort": [{"marker": "asc"}, {"n": "asc"}]
            }))).await;
            let ids: Vec<&str> = page["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
            assert_eq!(ids, vec!["a-25", "a-26", "a-27", "a-28", "a-29", "b-0", "b-1", "b-2", "b-3", "b-4"], "{page}");
        }
    }
    // A third cluster: several remotes in one search.
    let c = start_procs(1, &[LICENSE]).await;
    load(&c, "logs", 1, "c", 10).await;
    a.settings(0, "persistent", json!({"cluster.remote.c.seeds": [format!("127.0.0.1:{}", c.p.transport[0])]})).await;
    wait_remote(&a, "c", true, 60).await;
    for mrt in ["true", "false"] {
        let v = ok(&a, 1, Method::POST, &format!("/logs,b:logs,c*:logs/_search?ccs_minimize_roundtrips={mrt}"), Some(json!({
            "size": 0, "aggs": {"m": {"terms": {"field": "marker"}}, "s": {"sum": {"field": "n"}}}
        }))).await;
        assert_eq!(v["hits"]["total"]["value"], 60, "{v}");
        assert_eq!(v["aggregations"]["m"]["buckets"], json!([{"key": "a", "doc_count": 30}, {"key": "b", "doc_count": 20}, {"key": "c", "doc_count": 10}]), "{v}");
        // 435 + 190 + 45
        assert_eq!(v["aggregations"]["s"]["value"], 670.0, "{v}");
        assert_eq!(v["_clusters"]["total"], 3, "{v}");
        assert_eq!(v["_clusters"]["successful"], 3, "{v}");
    }
    // `-c:*` excludes a cluster included before it.
    let v = ok(&a, 0, Method::POST, "/b:logs,c*:logs,-c:*/_search", Some(json!({"size": 0}))).await;
    assert_eq!(v["hits"]["total"]["value"], 20, "{v}");
    a.settings(0, "persistent", json!({"cluster.remote.c.seeds": null})).await;
    drop(c);

    // Remote only, wildcard aliases and index patterns.
    let v = ok(&a, 0, Method::POST, "/b:logs/_search", Some(json!({"size": 50}))).await;
    assert_eq!(v["hits"]["total"]["value"], 20, "{v}");
    assert_eq!(markers(&v), vec![("b:logs".to_string(), "b".to_string())]);
    assert!(v["_clusters"]["details"].get("(local)").is_none(), "{v}");
    let v = ok(&a, 0, Method::POST, "/b*:only-*,b:logs/_search", Some(json!({"size": 50}))).await;
    assert_eq!(v["hits"]["total"]["value"], 25, "{v}");
    // `_count`, `_msearch`, async search.
    let v = ok(&a, 1, Method::POST, "/logs,b:logs/_count", Some(json!({"query": {"term": {"tag": "t0"}}}))).await;
    assert_eq!(v["count"], 10 + 7, "{v}");
    let (st, v) = a.ndjson(0, "/_msearch", "{\"index\":\"b:logs\"}\n{\"size\":0}\n{\"index\":\"logs,b:only-b\"}\n{\"size\":0}\n".into()).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["responses"][0]["hits"]["total"]["value"], 20, "{v}");
    assert_eq!(v["responses"][1]["hits"]["total"]["value"], 35, "{v}");
    let v = ok(&a, 0, Method::POST, "/logs,b:logs/_async_search?wait_for_completion_timeout=30s&keep_on_completion=true", Some(json!({"size": 0}))).await;
    assert_eq!(v["is_running"], false, "{v}");
    assert_eq!(v["response"]["hits"]["total"]["value"], 50, "{v}");
    assert_eq!(v["response"]["_clusters"]["successful"], 2, "{v}");
    // Field caps, resolve index, resolve cluster.
    let v = ok(&a, 0, Method::GET, "/logs,b:only-b/_field_caps?fields=marker,n", None).await;
    let indices: Vec<&str> = v["indices"].as_array().unwrap().iter().map(|i| i.as_str().unwrap()).collect();
    assert!(indices.contains(&"logs") && indices.contains(&"b:only-b"), "{v}");
    assert_eq!(v["fields"]["marker"]["keyword"]["searchable"], true, "{v}");
    let v = ok(&a, 0, Method::GET, "/_resolve/index/logs,b:*", None).await;
    let names: Vec<&str> = v["indices"].as_array().unwrap().iter().map(|i| i["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["b:logs", "b:only-b", "logs"], "{v}");
    let v = ok(&a, 0, Method::GET, "/_resolve/cluster/logs,b:logs,down:logs", None).await;
    assert_eq!(v["b"]["connected"], true, "{v}");
    assert_eq!(v["b"]["matching_indices"], true, "{v}");
    assert_eq!(v["down"]["connected"], false, "{v}");
    assert_eq!(v["(local)"]["matching_indices"], true, "{v}");

    // skip_unavailable: an unreachable optional remote is skipped, a
    // required one fails the search; a missing remote index likewise.
    let v = ok(&a, 0, Method::POST, "/logs,down:logs/_search", Some(json!({"size": 0}))).await;
    assert_eq!(v["hits"]["total"]["value"], 30, "{v}");
    assert_eq!(v["_clusters"]["skipped"], 1, "{v}");
    assert_eq!(v["_clusters"]["details"]["down"]["status"], "skipped", "{v}");
    assert!(v["_clusters"]["details"]["down"]["failures"][0]["reason"]["type"].as_str().is_some(), "{v}");
    let (st, v) = a.call(0, Method::POST, "/logs,req:logs/_search", Some(json!({}))).await;
    assert_eq!(st, 502, "{v}");
    assert_eq!(v["error"]["type"], "connect_transport_exception", "{v}");
    let (st, v) = a.call(0, Method::POST, "/logs,b:missing/_search", Some(json!({}))).await;
    assert_eq!(st, 404, "{v}");
    assert_eq!(v["error"]["type"], "index_not_found_exception", "{v}");
    // skip_unavailable changes in place (no reconnect).
    a.settings(0, "persistent", json!({"cluster.remote.b.skip_unavailable": true})).await;
    let v = ok(&a, 0, Method::POST, "/logs,b:missing/_search", Some(json!({"size": 0}))).await;
    assert_eq!(v["_clusters"]["details"]["b"]["status"], "skipped", "{v}");
    assert_eq!(v["_clusters"]["details"]["b"]["failures"][0]["reason"]["type"], "index_not_found_exception", "{v}");
    let info = wait_remote(&a, "b", true, 5).await;
    assert_eq!(info["b"]["skip_unavailable"], true, "{info}");
    a.settings(0, "persistent", json!({"cluster.remote.b.skip_unavailable": false})).await;

    // Settings validation (ES RemoteClusterSettings).
    let (st, v) = a.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.remote.x.seeds": [seed_b.clone()], "cluster.remote.x.proxy_address": seed_b.clone()}}))).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("cannot be used with the configured \"cluster.remote.x.mode\" [required=PROXY, configured=SNIFF]"), "{v}");
    let (st, v) = a.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"cluster.remote.y.skip_unavailable": true}}))).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("if remote cluster is not enabled"), "{v}");

    // Proxy mode: sockets to one address.
    a.settings(0, "persistent", json!({"cluster.remote.p.mode": "proxy", "cluster.remote.p.proxy_address": seed_b.clone(), "cluster.remote.p.proxy_socket_connections": 3})).await;
    let info = wait_remote(&a, "p", true, 60).await;
    assert_eq!(info["p"]["mode"], "proxy", "{info}");
    assert_eq!(info["p"]["num_proxy_sockets_connected"], 3, "{info}");
    assert_eq!(info["p"]["max_proxy_socket_connections"], 3, "{info}");
    let v = ok(&a, 0, Method::POST, "/p:logs/_search", Some(json!({"size": 0}))).await;
    assert_eq!(v["hits"]["total"]["value"], 20, "{v}");

    // `_cluster/stats?include_remotes`: the remote's own figures.
    let v = ok(&a, 0, Method::GET, "/_cluster/stats?include_remotes=true", None).await;
    assert_eq!(v["ccs"]["clusters"]["b"]["cluster_uuid"], ub, "{v}");
    assert_eq!(v["ccs"]["clusters"]["b"]["nodes_count"], 2, "{v}");
    assert_eq!(v["ccs"]["clusters"]["b"]["mode"], "sniff", "{v}");
    assert!(v["ccs"]["_search"]["total"].as_u64().unwrap() > 0, "{v}");

    // A remote node stops: the other connection serves; both stop: the
    // required remote fails; they come back: reconnected (with backoff).
    b.kill(1);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (st, v) = a.call(0, Method::POST, "/b:only-b/_search", Some(json!({"size": 0}))).await;
        if st == 200 && v["hits"]["total"]["value"] == 5 {
            break;
        }
        assert!(Instant::now() < deadline, "{st} {v}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    b.kill(0);
    wait_remote(&a, "b", false, 60).await;
    let (st, v) = a.call(0, Method::POST, "/logs,b:logs/_search", Some(json!({}))).await;
    assert!(st >= 500, "{st} {v}");
    let v = ok(&a, 0, Method::POST, "/logs,down:logs,p:logs/_search?size=0", None).await;
    assert_eq!(v["_clusters"]["skipped"], 2, "{v}");
    b.spawn(0, &[]).await;
    b.spawn(1, &[]).await;
    b.wait_nodes(0, 2, 120).await;
    b.wait_health(0, "logs", "green", 120).await;
    wait_remote(&a, "b", true, 120).await;
    let v = ok(&a, 0, Method::POST, "/logs,b:logs/_search", Some(json!({"size": 0}))).await;
    assert_eq!(v["hits"]["total"]["value"], 50, "{v}");

    // Removing the alias closes it.
    a.settings(0, "persistent", json!({"cluster.remote.b.seeds": null, "cluster.remote.b.skip_unavailable": null})).await;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (_, info) = a.call(0, Method::GET, "/_remote/info", None).await;
        if info.get("b").is_none() {
            break;
        }
        assert!(Instant::now() < deadline, "{info}");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let (st, v) = a.call(0, Method::POST, "/b:logs/_search", Some(json!({}))).await;
    assert_eq!(st, 404, "{v}");
    assert_eq!(v["error"]["type"], "no_such_remote_cluster_exception", "{v}");
}

const PW: &str = "test-password";

/// A config directory whose keystore holds `entries` (secure settings such
/// as `bootstrap.password` must come from the keystore).
/// A CA and a node certificate for 127.0.0.1 (CA, certificate, key paths).
fn write_pki(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair, SanType};
    std::fs::create_dir_all(dir).unwrap();
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.distinguished_name.push(DnType::CommonName, "rcs-test-ca");
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let node_key = KeyPair::generate().unwrap();
    let mut p = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    p.subject_alt_names.push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
    p.distinguished_name.push(DnType::CommonName, "rcs-node");
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth, ExtendedKeyUsagePurpose::ClientAuth];
    let cert = p.signed_by(&node_key, &ca, &ca_key).unwrap();
    let (ca_path, cert_path, key_path) = (dir.join("ca.crt"), dir.join("node.crt"), dir.join("node.key"));
    std::fs::write(&ca_path, ca.pem()).unwrap();
    std::fs::write(&cert_path, cert.pem()).unwrap();
    std::fs::write(&key_path, node_key.serialize_pem()).unwrap();
    (ca_path, cert_path, key_path)
}

fn keystore_conf(tag: &str, entries: &[(&str, &str)]) -> String {
    let conf = common::scratch_dir(tag);
    std::fs::create_dir_all(&conf).unwrap();
    let mut ks = elasticsearch_node::keystore::KeyStore::create().unwrap();
    for (k, v) in entries {
        ks.set_string(k, v).unwrap();
    }
    ks.save(&conf, "").unwrap();
    conf.to_string_lossy().to_string()
}

async fn call_as(p: &Procs, user: &str, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = p.c.request(method, format!("http://127.0.0.1:{}{path}", p.p.http[0])).basic_auth(user, Some(PW));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.expect("request");
    let st = resp.status().as_u16();
    (st, resp.json().await.unwrap_or(Value::Null))
}

/// The remote cluster authorizes the local user by its own definitions of
/// the user's role names (ES certificate-based remote cluster security).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn remote_cluster_authorizes_the_local_user_with_its_own_roles() {
    let secured = [LICENSE, ("xpack.security.enabled", "true"), ("xpack.security.authc.password_hashing.algorithm", "pbkdf2_1000")];
    let conf_a = keystore_conf("ccs-sec-a", &[("bootstrap.password", PW)]);
    let conf_b = keystore_conf("ccs-sec-b", &[("bootstrap.password", PW)]);
    let a = start_procs_secured_env(1, &secured, &[("ES_PATH_CONF", conf_a.as_str())], "elastic", PW).await;
    let b = start_procs_secured_env(1, &secured, &[("ES_PATH_CONF", conf_b.as_str())], "elastic", PW).await;
    for (index, marker) in [("public-b", "pub"), ("private-b", "secret")] {
        ok(&b, 0, Method::PUT, &format!("/{index}/_doc/1?refresh=true"), Some(json!({"marker": marker}))).await;
    }
    // The role named `ccs_reader` grants `public-b` on the remote cluster;
    // on the local cluster the same name grants nothing on indices.
    ok(&b, 0, Method::PUT, "/_security/role/ccs_reader", Some(json!({"indices": [{"names": ["public-*"], "privileges": ["read", "read_cross_cluster", "view_index_metadata"]}]}))).await;
    ok(&a, 0, Method::PUT, "/_security/role/ccs_reader", Some(json!({"cluster": ["monitor"]}))).await;
    ok(&a, 0, Method::PUT, "/_security/user/alice", Some(json!({"password": PW, "roles": ["ccs_reader"]}))).await;
    ok(&a, 0, Method::PUT, "/_security/role/unknown_there", Some(json!({"cluster": ["monitor"]}))).await;
    ok(&a, 0, Method::PUT, "/_security/user/bob", Some(json!({"password": PW, "roles": ["unknown_there"]}))).await;
    a.settings(0, "persistent", json!({"cluster.remote.b.seeds": [format!("127.0.0.1:{}", b.p.transport[0])], "cluster.remote.b.skip_unavailable": false})).await;
    wait_remote(&a, "b", true, 60).await;

    let (st, v) = call_as(&a, "alice", Method::POST, "/b:public-b/_search", Some(json!({}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_source"]["marker"], "pub", "{v}");
    assert_eq!(v["hits"]["hits"][0]["_index"], "b:public-b", "{v}");
    // Wildcards expand to what the remote cluster lets the user read.
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:*-b/_search", Some(json!({}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
    assert!(!v.to_string().contains("secret"), "{v}");
    // An explicit index the remote role does not grant: the remote's denial.
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:private-b/_search", Some(json!({}))).await;
    assert_eq!(st, 403, "{v}");
    assert_eq!(v["error"]["type"], "security_exception", "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("alice"), "{v}");
    assert!(!v.to_string().contains("\"secret\""), "{v}");
    // A role name the remote cluster does not define grants nothing there.
    let (st, v) = call_as(&a, "bob", Method::POST, "/b:public-b/_search", Some(json!({}))).await;
    assert_eq!(st, 403, "{v}");
    // The superuser role exists on both clusters.
    let (st, v) = call_as(&a, "elastic", Method::POST, "/b:private-b/_search", Some(json!({}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_source"]["marker"], "secret", "{v}");
    // Optional remote: the denial is a skipped cluster, not data.
    a.settings(0, "persistent", json!({"cluster.remote.b.skip_unavailable": true})).await;
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:private-b/_search", Some(json!({}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_clusters"]["details"]["b"]["status"], "skipped", "{v}");
    assert_eq!(v["_clusters"]["details"]["b"]["failures"][0]["reason"]["type"], "security_exception", "{v}");
    assert_eq!(v["hits"]["total"]["value"], 0, "{v}");
}

/// API key based remote cluster security (ES RCS 2.0): the remote cluster
/// serves a remote cluster server port; the querying cluster connects to it
/// with `cluster.remote.<alias>.credentials` (a cross-cluster API key from
/// the keystore) and each request runs as the local user's `remote_indices`
/// grant for that alias, limited by the key's access.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn api_key_remote_cluster_security_limits_by_key_and_user_grants() {
    let rcs_port = common::ports::probe_free_ports(1)[0];
    let port = rcs_port.to_string();
    let secured = [LICENSE, ("xpack.security.enabled", "true"), ("xpack.security.authc.password_hashing.algorithm", "pbkdf2_1000")];
    // The remote cluster server port is TLS by default (ES 9.5.5
    // `xpack.security.remote_cluster_server.ssl.enabled: true`); the
    // querying cluster's client trusts its CA and presents no certificate.
    let (ca, cert, key) = write_pki(&common::scratch_dir("rcs2-pki"));
    let (ca, cert, key) = (ca.to_string_lossy().into_owned(), cert.to_string_lossy().into_owned(), key.to_string_lossy().into_owned());
    let mut fulfilling = secured.to_vec();
    fulfilling.extend([("remote_cluster_server.enabled", "true"), ("remote_cluster.port", port.as_str()),
        ("xpack.security.remote_cluster_server.ssl.certificate", cert.as_str()), ("xpack.security.remote_cluster_server.ssl.key", key.as_str())]);
    let conf_b = keystore_conf("rcs2-b", &[("bootstrap.password", PW)]);
    let b = start_procs_secured_env(1, &fulfilling, &[("ES_PATH_CONF", conf_b.as_str())], "elastic", PW).await;
    for (index, marker) in [("public-b", "pub"), ("other-b", "other"), ("private-b", "secret")] {
        ok(&b, 0, Method::PUT, &format!("/{index}/_doc/1?refresh=true"), Some(json!({"marker": marker}))).await;
    }
    // The key grants `public-*` and `other-*`.
    let key = ok(&b, 0, Method::POST, "/_security/cross_cluster/api_key", Some(json!({"name": "querying", "access": {"search": [{"names": ["public-*", "other-*"]}]}}))).await;
    let encoded = key["encoded"].as_str().expect("encoded key").to_string();
    // The querying cluster's keystore holds the key.
    let conf_text = keystore_conf("rcs2-a", &[("bootstrap.password", PW), ("cluster.remote.b.credentials", encoded.as_str())]);
    let mut querying = secured.to_vec();
    querying.push(("xpack.security.remote_cluster_client.ssl.certificate_authorities", ca.as_str()));
    let a = start_procs_secured_env(1, &querying, &[("ES_PATH_CONF", conf_text.as_str())], "elastic", PW).await;
    // `alice` may read `public-*` and `private-*` on `b` (her role's
    // `remote_indices`); the key limits her to `public-*`.
    ok(&a, 0, Method::PUT, "/_security/role/rcs_reader", Some(json!({"remote_indices": [{"clusters": ["b"], "names": ["public-*", "private-*"], "privileges": ["read", "read_cross_cluster", "view_index_metadata"]}]}))).await;
    ok(&a, 0, Method::PUT, "/_security/user/alice", Some(json!({"password": PW, "roles": ["rcs_reader"]}))).await;
    a.settings(0, "persistent", json!({"cluster.remote.b.seeds": [format!("127.0.0.1:{rcs_port}")], "cluster.remote.b.skip_unavailable": false})).await;
    let info = wait_remote(&a, "b", true, 60).await;
    assert_eq!(info["b"]["cluster_credentials"], "::es_redacted::", "{info}");
    // Sniffed through the seed: the managed connection is to the node's
    // remote cluster server address (TLS), not its transport address.
    assert_eq!(info["b"]["num_nodes_connected"], 1, "{info}");
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:public-b/_search", Some(json!({}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_source"]["marker"], "pub", "{v}");
    // The user's grant does not cover `other-b` (the key would).
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:other-b/_search", Some(json!({}))).await;
    assert_eq!(st, 403, "{v}");
    // The key does not cover `private-b` (the user's grant would).
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:private-b/_search", Some(json!({}))).await;
    assert_eq!(st, 403, "{v}");
    assert!(!v.to_string().contains("\"secret\""), "{v}");
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:*-b/_search", Some(json!({}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");

    // The cross-cluster extras run under the same grants (security on).
    // dfs_query_then_fetch: a DFS round on the remote, then the search.
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:public-b/_search?search_type=dfs_query_then_fetch", Some(json!({"query": {"match": {"marker": "pub"}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_index"], "b:public-b", "{v}");
    assert_eq!(v["_clusters"]["details"]["b"]["status"], "successful", "{v}");
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:private-b/_search?search_type=dfs_query_then_fetch", Some(json!({"query": {"match_all": {}}}))).await;
    assert_eq!(st, 403, "{v}");
    // Retrievers (RRF) across clusters.
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:*-b/_search", Some(json!({"retriever": {"rrf": {"retrievers": [
        {"standard": {"query": {"match": {"marker": "pub"}}}}, {"standard": {"query": {"match_all": {}}}}]}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 1, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_source"]["marker"], "pub", "{v}");
    // Aggregations on a cross-cluster point in time.
    let (st, pit) = call_as(&a, "alice", Method::POST, "/b:public-b/_pit?keep_alive=1m", None).await;
    assert_eq!(st, 200, "{pit}");
    let (st, v) = call_as(&a, "alice", Method::POST, "/_search", Some(json!({"pit": {"id": pit["id"]}, "size": 0,
        "aggs": {"m": {"terms": {"field": "marker.keyword"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["aggregations"]["m"]["buckets"], json!([{"key": "pub", "doc_count": 1}]), "{v}");
    let (st, v) = call_as(&a, "alice", Method::DELETE, "/_pit", Some(json!({"id": pit["id"]}))).await;
    assert_eq!(st, 200, "{v}");
    // collapse with inner_hits across clusters, as the user.
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:*-b/_search", Some(json!({"collapse": {"field": "marker.keyword", "inner_hits": {"name": "m"}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["hits"][0]["inner_hits"]["m"]["hits"]["hits"][0]["_index"], "b:public-b", "{v}");
    // kNN needing DFS across clusters, as the user: the global top k.
    ok(&b, 0, Method::PUT, "/public-vec", Some(json!({"mappings": {"properties": {"v": {"type": "dense_vector", "dims": 2, "index": true, "similarity": "l2_norm"}, "src": {"type": "keyword"}}}}))).await;
    for (id, vec) in [("p1", [1.0, 0.0]), ("p2", [3.0, 3.0]), ("p3", [1.0, 0.2])] {
        ok(&b, 0, Method::PUT, &format!("/public-vec/_doc/{id}?refresh=true"), Some(json!({"v": vec, "src": id}))).await;
    }
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:public-vec/_search", Some(json!({"knn": {"field": "v", "query_vector": [1.0, 0.0], "k": 2, "num_candidates": 10},
        "aggs": {"src": {"terms": {"field": "src"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["hits"]["total"]["value"], 2, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_index"], "b:public-vec", "{v}");
    assert_eq!(v["aggregations"]["src"]["buckets"], json!([{"key": "p1", "doc_count": 1}, {"key": "p3", "doc_count": 1}]), "{v}");
    // ES|QL and SQL read the remote as the user, limited by the key.
    let (st, v) = call_as(&a, "alice", Method::POST, "/_query", Some(json!({"query": "FROM b:public-b METADATA _index | KEEP _index, marker", "include_ccs_metadata": true}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["values"], json!([["b:public-b", "pub"]]), "{v}");
    assert_eq!(v["_clusters"]["details"]["b"]["status"], "successful", "{v}");
    let (st, v) = call_as(&a, "alice", Method::POST, "/_query", Some(json!({"query": "FROM b:private-b | KEEP marker"}))).await;
    assert!(st >= 400, "{st} {v}");
    assert!(!v.to_string().contains("\"secret\""), "{v}");
    let (st, v) = call_as(&a, "alice", Method::POST, "/_sql?format=json", Some(json!({"query": "SELECT marker FROM \"b:public-b\""}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["rows"], json!([["pub"]]), "{v}");
    // The remote cluster server port takes no certificate-only caller: a
    // second alias without credentials pointed at it cannot read.
    a.settings(0, "persistent", json!({"cluster.remote.nokey.seeds": [format!("127.0.0.1:{rcs_port}")], "cluster.remote.nokey.skip_unavailable": false})).await;
    let (st, v) = call_as(&a, "elastic", Method::POST, "/nokey:public-b/_search", Some(json!({}))).await;
    assert!(st >= 400, "{st} {v}");
    // An invalidated key stops working.
    ok(&b, 0, Method::DELETE, "/_security/api_key", Some(json!({"ids": [key["id"].clone()]}))).await;
    let (st, v) = call_as(&a, "alice", Method::POST, "/b:public-b/_search", Some(json!({}))).await;
    assert_eq!(st, 401, "{v}");
}

/// ES's coordinated rounds across clusters: `dfs_query_then_fetch` scores
/// with every cluster's term statistics, retrievers fuse child searches of
/// every cluster, and a cross-cluster point in time aggregates.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn dfs_retrievers_and_pit_aggregations_across_clusters() {
    let a = start_procs(1, &[LICENSE]).await;
    let b = start_procs(1, &[LICENSE]).await;
    // `rare` is common on b and rare on a: global IDF differs from a's own.
    for (p, docs) in [(&a, vec![("a1", "rare word"), ("a2", "other word"), ("a3", "other text")]), (&b, vec![("b1", "rare rare"), ("b2", "rare stuff"), ("b3", "rare thing"), ("b4", "rare word")])] {
        ok(p, 0, Method::PUT, "/docs", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0},
            "mappings": {"properties": {"t": {"type": "text"}, "k": {"type": "keyword"}, "n": {"type": "long"}}}}))).await;
        for (i, (id, t)) in docs.iter().enumerate() {
            ok(p, 0, Method::PUT, &format!("/docs/_doc/{id}?refresh=true"), Some(json!({"t": t, "k": id, "n": i}))).await;
        }
    }
    a.settings(0, "persistent", json!({"cluster.remote.b.seeds": [format!("127.0.0.1:{}", b.p.transport[0])], "cluster.remote.b.skip_unavailable": false})).await;
    wait_remote(&a, "b", true, 60).await;
    // DFS: the same scores as one index holding every document.
    let dfs = ok(&a, 0, Method::POST, "/docs,b:docs/_search?search_type=dfs_query_then_fetch", Some(json!({"query": {"match": {"t": "rare"}}, "size": 10}))).await;
    assert_eq!(dfs["hits"]["total"]["value"], 5, "{dfs}");
    let all = start_procs(1, &[LICENSE]).await;
    ok(&all, 0, Method::PUT, "/docs", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"t": {"type": "text"}}}}))).await;
    for (id, t) in [("a1", "rare word"), ("a2", "other word"), ("a3", "other text"), ("b1", "rare rare"), ("b2", "rare stuff"), ("b3", "rare thing"), ("b4", "rare word")] {
        ok(&all, 0, Method::PUT, &format!("/docs/_doc/{id}?refresh=true"), Some(json!({"t": t}))).await;
    }
    let one = ok(&all, 0, Method::POST, "/docs/_search?search_type=dfs_query_then_fetch", Some(json!({"query": {"match": {"t": "rare"}}, "size": 10}))).await;
    let score = |v: &Value, id: &str| v["hits"]["hits"].as_array().unwrap().iter().find(|h| h["_id"] == id).map(|h| h["_score"].as_f64().unwrap()).unwrap();
    for id in ["a1", "b1", "b4"] {
        assert!((score(&dfs, id) - score(&one, id)).abs() < 1e-4, "{id}: {dfs} vs {one}");
    }
    drop(all);
    // Retrievers: RRF over two child searches spanning both clusters.
    let v = ok(&a, 0, Method::POST, "/docs,b:docs/_search", Some(json!({"size": 10, "retriever": {"rrf": {"retrievers": [
        {"standard": {"query": {"match": {"t": "rare"}}}},
        {"standard": {"query": {"match": {"t": "word"}}}}
    ]}}}))).await;
    let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"a1") && ids.contains(&"b4") && ids.contains(&"b1"), "{v}");
    // a1 and b4 match both children: ranked first.
    assert!(ids[..2].contains(&"a1") && ids[..2].contains(&"b4"), "{v}");
    let idx: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_index"].as_str().unwrap()).collect();
    assert!(idx.contains(&"b:docs") && idx.contains(&"docs"), "{v}");
    // A point in time across clusters with aggregations.
    let pit = ok(&a, 0, Method::POST, "/docs,b:docs/_pit?keep_alive=1m", None).await;
    let id = pit["id"].as_str().unwrap().to_string();
    ok(&a, 0, Method::PUT, "/docs/_doc/a9?refresh=true", Some(json!({"t": "late", "k": "a9", "n": 9}))).await;
    let v = ok(&a, 0, Method::POST, "/_search", Some(json!({"size": 3, "pit": {"id": id, "keep_alive": "1m"}, "sort": [{"k": "asc"}],
        "aggs": {"keys": {"terms": {"field": "k", "size": 20}}, "avg_n": {"avg": {"field": "n"}}, "card": {"cardinality": {"field": "k"}}}}))).await;
    assert_eq!(v["hits"]["total"]["value"], 7, "the point in time does not see a9: {v}");
    assert_eq!(v["aggregations"]["keys"]["buckets"].as_array().unwrap().len(), 7, "{v}");
    assert_eq!(v["aggregations"]["card"]["value"], 7, "{v}");
    // n: a 0,1,2 ; b 0,1,2,3 -> 9 / 7
    assert!((v["aggregations"]["avg_n"]["value"].as_f64().unwrap() - 9.0 / 7.0).abs() < 1e-9, "{v}");
    assert_eq!(v["hits"]["hits"][0]["_id"], "a1", "{v}");
    assert_eq!(v["_clusters"]["total"], 2, "{v}");
    let (st, _) = a.call(0, Method::DELETE, "/_pit", Some(json!({"id": id}))).await;
    assert_eq!(st, 200);

    // `collapse.inner_hits` across clusters (ES's expand phase): each
    // group's inner hits come from every cluster.
    for (p, docs) in [(&a, vec![("a1", "x"), ("a2", "x"), ("a3", "z")]), (&b, vec![("b1", "x"), ("b2", "y")])] {
        ok(p, 0, Method::PUT, "/grp", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0}, "mappings": {"properties": {"g": {"type": "keyword"}, "n": {"type": "long"}}}}))).await;
        for (i, (id, g)) in docs.iter().enumerate() {
            ok(p, 0, Method::PUT, &format!("/grp/_doc/{id}?refresh=true"), Some(json!({"g": g, "n": i}))).await;
        }
    }
    let v = ok(&a, 0, Method::POST, "/grp,b:grp/_search", Some(json!({"sort": [{"g": "asc"}], "collapse": {"field": "g", "inner_hits": {"name": "all", "size": 5, "sort": [{"n": "asc"}]}}}))).await;
    let groups: Vec<(Value, Value)> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["fields"]["g"][0].clone(), h["inner_hits"]["all"]["hits"]["total"]["value"].clone())).collect();
    assert_eq!(groups, vec![(json!("x"), json!(3)), (json!("y"), json!(1)), (json!("z"), json!(1))], "{v}");
    let x = &v["hits"]["hits"][0]["inner_hits"]["all"]["hits"]["hits"];
    let indices: std::collections::BTreeSet<String> = x.as_array().unwrap().iter().map(|h| h["_index"].as_str().unwrap().to_string()).collect();
    assert_eq!(indices, ["b:grp".to_string(), "grp".to_string()].into_iter().collect(), "{v}");

    // kNN with aggregations (ES's DFS kNN phase): the global top k across
    // clusters, not each cluster's own top k.
    for (p, docs) in [(&a, vec![("a1", [1.0, 0.0]), ("a2", [0.0, 1.0])]), (&b, vec![("b1", [1.0, 0.1]), ("b2", [5.0, 5.0])])] {
        ok(p, 0, Method::PUT, "/vec", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0},
            "mappings": {"properties": {"v": {"type": "dense_vector", "dims": 2, "index": true, "similarity": "l2_norm"}, "src": {"type": "keyword"}}}}))).await;
        for (id, vec) in docs {
            ok(p, 0, Method::PUT, &format!("/vec/_doc/{id}?refresh=true"), Some(json!({"v": vec, "src": &id[..1]}))).await;
        }
    }
    let v = ok(&a, 0, Method::POST, "/vec,b:vec/_search", Some(json!({"knn": {"field": "v", "query_vector": [1.0, 0.0], "k": 2, "num_candidates": 10},
        "aggs": {"src": {"terms": {"field": "src"}}}}))).await;
    assert_eq!(v["hits"]["total"]["value"], 2, "{v}");
    let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["a1", "b1"], "{v}");
    assert_eq!(v["aggregations"]["src"]["buckets"], json!([{"key": "a", "doc_count": 1}, {"key": "b", "doc_count": 1}]), "{v}");
    assert_eq!(v["_clusters"]["successful"], 2, "{v}");
}

/// ES|QL `FROM alias:index`: remote clusters' shards are read there (their
/// schemas from their mappings) and combined with this cluster's.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn esql_and_sql_read_remote_clusters() {
    let a = start_procs(1, &[LICENSE]).await;
    let b = start_procs(2, &[LICENSE]).await;
    load(&a, "logs", 2, "a", 30).await;
    load(&b, "logs", 3, "b", 20).await;
    a.settings(0, "persistent", json!({"cluster.remote.b.seeds": [format!("127.0.0.1:{}", b.p.transport[0])], "cluster.remote.b.skip_unavailable": false,
        "cluster.remote.down.seeds": [format!("127.0.0.1:{}", common::ports::probe_free_ports(1)[0])], "cluster.remote.down.skip_unavailable": true})).await;
    wait_remote(&a, "b", true, 60).await;
    let v = ok(&a, 0, Method::POST, "/_query", Some(json!({"query": "FROM logs,b:logs | STATS c = COUNT(*), s = SUM(n) BY marker | SORT marker"}))).await;
    assert_eq!(v["values"], json!([[30, 435, "a"], [20, 190, "b"]]), "{v}");
    let v = ok(&a, 0, Method::POST, "/_query", Some(json!({"query": "FROM b:logs METADATA _index | WHERE n < 2 | KEEP _index, n | SORT n"}))).await;
    assert_eq!(v["values"], json!([["b:logs", 0], ["b:logs", 1]]), "{v}");
    // `_clusters` (ES `include_ccs_metadata`): each cluster's expression and shards.
    let v = ok(&a, 0, Method::POST, "/_query", Some(json!({"query": "FROM logs,b:logs | STATS c = COUNT(*)", "include_ccs_metadata": true}))).await;
    assert_eq!(v["is_partial"], json!(false), "{v}");
    let c = &v["_clusters"];
    assert_eq!((c["total"].clone(), c["successful"].clone(), c["skipped"].clone()), (json!(2), json!(2), json!(0)), "{v}");
    assert_eq!(c["details"]["(local)"]["indices"], json!("logs"), "{v}");
    assert_eq!(c["details"]["(local)"]["_shards"]["total"], json!(2), "{v}");
    assert_eq!(c["details"]["b"]["status"], json!("successful"), "{v}");
    assert_eq!(c["details"]["b"]["_shards"]["total"], json!(3), "{v}");
    // An unavailable optional remote contributes nothing: skipped, partial.
    let v = ok(&a, 0, Method::POST, "/_query", Some(json!({"query": "FROM logs,down:logs | STATS c = COUNT(*)", "include_ccs_metadata": true}))).await;
    assert_eq!(v["values"], json!([[30]]), "{v}");
    assert_eq!(v["is_partial"], json!(true), "{v}");
    assert_eq!(v["_clusters"]["skipped"], json!(1), "{v}");
    assert_eq!(v["_clusters"]["details"]["down"]["status"], json!("skipped"), "{v}");
    assert!(v["_clusters"]["details"]["down"]["failures"][0]["reason"]["type"].is_string(), "{v}");
    // SQL over the same executor.
    let (st, v) = a.call(0, Method::POST, "/_sql?format=json", Some(json!({"query": "SELECT COUNT(*) FROM \"b:logs\""}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["rows"], json!([[20]]), "{v}");
}
