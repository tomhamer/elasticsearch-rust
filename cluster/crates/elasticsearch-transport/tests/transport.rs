//! In-process tests: several `TransportService`s on 127.0.0.1:0.

use bytes::Bytes;
use elasticsearch_cluster::state::{DiscoveryNode, Role, random_id};
use elasticsearch_transport::*;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::broadcast;

fn node(name: &str) -> DiscoveryNode {
    DiscoveryNode {
        id: random_id(),
        name: name.into(),
        ephemeral_id: random_id(),
        transport_address: String::new(),
        http_address: String::new(),
        host: String::new(),
        ip: String::new(),
        roles: Role::defaults(),
        attributes: BTreeMap::new(),
        version: "8.15.0".into(),
    }
}

fn config(cluster: &str, name: &str) -> TransportConfig {
    TransportConfig::new(cluster, node(name), "127.0.0.1:0".parse().unwrap())
}

async fn start(cluster: &str, name: &str) -> TransportService {
    TransportService::bind(config(cluster, name)).await.unwrap()
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Echo {
    msg: String,
    n: u64,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Reply {
    msg: String,
    doubled: u64,
    from: String,
}

const T: Duration = Duration::from_secs(10);

fn register_echo(svc: &TransportService) {
    let name = svc.local_node().name.clone();
    svc.register_handler("test:echo", move |req: Echo| {
        let from = name.clone();
        async move { Ok(Reply { msg: req.msg.to_uppercase(), doubled: req.n * 2, from }) }
    });
}

async fn next_event(rx: &mut broadcast::Receiver<ConnectionEvent>) -> ConnectionEvent {
    tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.expect("event within 5s").unwrap()
}

#[tokio::test]
async fn typed_roundtrip_and_identity() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    register_echo(&b);
    let b_addr = b.bound_address();
    assert_ne!(b_addr.port(), 0);
    assert_eq!(b.local_node().transport_address, b_addr.to_string());

    let r: Reply = a.send_request(b_addr, "test:echo", &Echo { msg: "hi".into(), n: 21 }, T).await.unwrap();
    assert_eq!(r, Reply { msg: "HI".into(), doubled: 42, from: "b".into() });

    // Handshake exposed the remote identity.
    let id = a.remote_identity(b_addr).unwrap();
    assert_eq!(id.cluster_name, "c1");
    assert_eq!(&id.node, b.local_node());
    assert_eq!(a.connect(b_addr).await.unwrap(), id);
    assert_eq!(a.connected_nodes(), vec![(b_addr, id)]);

    let sa = a.stats();
    assert_eq!(sa.total_outbound_connections, 1);
    assert_eq!(sa.outbound_open, 1);
    assert!(sa.tx_count >= 2 && sa.rx_count >= 2, "{sa:?}"); // handshake + request
    let sb = b.stats();
    assert_eq!(sb.server_open, 1);
    assert_eq!(sb.rx_size_in_bytes, sa.tx_size_in_bytes);
    let json = serde_json::to_value(&sb).unwrap();
    for k in ["server_open", "total_outbound_connections", "rx_count", "rx_size_in_bytes", "tx_count", "tx_size_in_bytes"] {
        assert!(json.get(k).is_some(), "missing {k}");
    }
    assert!(json.get("outbound_open").is_none());
}

#[tokio::test]
async fn thousand_concurrent_requests_share_one_connection() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    b.register_handler("test:slow_echo", |req: Echo| async move {
        tokio::time::sleep(Duration::from_millis(req.n % 7)).await;
        Ok(req.n * 2)
    });
    let addr = b.bound_address();
    let futs: Vec<_> = (0..1000u64)
        .map(|n| {
            let a = a.clone();
            async move { a.send_request::<_, u64>(addr, "test:slow_echo", &Echo { msg: String::new(), n }, T).await.map(|r| (n, r)) }
        })
        .collect();
    let results = futures::future::join_all(futs).await;
    for r in results {
        let (n, doubled) = r.unwrap();
        assert_eq!(doubled, n * 2);
    }
    assert_eq!(a.stats().total_outbound_connections, 1);
    assert_eq!(b.stats().server_open, 1);
}

#[tokio::test]
async fn responses_complete_out_of_order() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    // Request i sleeps (3 - i) * 150ms, so the last one sent finishes first.
    b.register_handler("test:sleep", |i: u64| async move {
        tokio::time::sleep(Duration::from_millis((3 - i) * 150)).await;
        Ok(i)
    });
    let addr = b.bound_address();
    a.connect(addr).await.unwrap();
    let mut pending = FuturesUnordered::new();
    for i in 0..4u64 {
        let a = a.clone();
        pending.push(async move { a.send_request::<_, u64>(addr, "test:sleep", &i, T).await.unwrap() });
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut order = Vec::new();
    while let Some(i) = pending.next().await {
        order.push(i);
    }
    assert_eq!(order, vec![3, 2, 1, 0]);
}

#[tokio::test]
async fn request_timeout_then_connection_still_usable() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    register_echo(&b);
    b.register_handler("test:hang", |_: ()| async move {
        tokio::time::sleep(Duration::from_millis(600)).await;
        Ok(())
    });
    let addr = b.bound_address();
    let err = a.send_request::<_, ()>(addr, "test:hang", &(), Duration::from_millis(100)).await.unwrap_err();
    match &err {
        TransportError::ReceiveTimeout { action, timeout_ms, .. } => {
            assert_eq!(action, "test:hang");
            assert_eq!(*timeout_ms, 100);
        }
        e => panic!("unexpected {e:?}"),
    }
    assert!(err.to_string().contains("timed out after [100ms]"), "{err}");
    assert_eq!(err.to_remote_error().error_type, "receive_timeout_transport_exception");
    // The same slow operation completes when callers explicitly opt out of
    // a response deadline (synchronous force-merge uses this mode).
    a.send_request::<_, ()>(addr, "test:hang", &(), None).await.unwrap();
    // The late response is discarded; the connection keeps working.
    tokio::time::sleep(Duration::from_millis(700)).await;
    let r: Reply = a.send_request(addr, "test:echo", &Echo { msg: "x".into(), n: 1 }, T).await.unwrap();
    assert_eq!(r.doubled, 2);
    assert_eq!(a.stats().total_outbound_connections, 1);
}

#[tokio::test]
async fn handler_errors_propagate_with_nested_causes() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    let expected = RemoteError::new("search_phase_execution_exception", "all shards failed", 400).with_cause(
        RemoteError::new("index_not_found_exception", "no such index [logs]", 404)
            .with_metadata("index", "logs")
            .with_metadata("index_uuid", "_na_")
            .with_cause(RemoteError::new("illegal_argument_exception", "root", 400)),
    );
    let e2 = expected.clone();
    b.register_handler("test:fail", move |_: ()| {
        let e = e2.clone();
        async move { Err::<(), _>(e) }
    });
    b.register_handler("test:panic", |_: ()| async move {
        if true {
            panic!("boom");
        }
        Ok(())
    });
    let addr = b.bound_address();
    let err = a.send_request::<_, ()>(addr, "test:fail", &(), T).await.unwrap_err();
    assert_eq!(err, TransportError::Remote(expected.clone()));
    let TransportError::Remote(remote) = err else { unreachable!() };
    assert_eq!(remote.root_cause().reason, "root");
    assert_eq!(remote.status, 400);
    assert_eq!(
        remote.to_json(),
        serde_json::json!({
            "type": "search_phase_execution_exception",
            "reason": "all shards failed",
            "caused_by": {
                "type": "index_not_found_exception",
                "reason": "no such index [logs]",
                "index": "logs",
                "index_uuid": "_na_",
                "caused_by": {"type": "illegal_argument_exception", "reason": "root"}
            }
        })
    );

    // A panicking handler becomes a 500 and does not take the connection down.
    let err = a.send_request::<_, ()>(addr, "test:panic", &(), T).await.unwrap_err();
    match err {
        TransportError::Remote(e) => {
            assert_eq!(e.status, 500);
            assert!(e.reason.contains("boom"));
        }
        e => panic!("unexpected {e:?}"),
    }
    // Wrong request type: deserialization failure on the remote side.
    let err = a.send_request::<_, ()>(addr, "test:fail", &"not a unit but a string", T).await;
    assert!(err.is_err());
    assert!(a.is_connected(addr));
}

#[tokio::test]
async fn unknown_action() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    let err = a.send_request::<_, ()>(b.bound_address(), "indices:data/read/nope", &(), T).await.unwrap_err();
    assert_eq!(err, TransportError::ActionNotFound { action: "indices:data/read/nope".into() });
    assert_eq!(err.to_string(), "No handler for action [indices:data/read/nope]");
    // Same locally.
    let err = a.send_request::<_, ()>(a.bound_address(), "x", &(), T).await.unwrap_err();
    assert_eq!(err.to_string(), "No handler for action [x]");
}

#[tokio::test]
async fn cluster_name_mismatch_is_rejected() {
    let a = start("prod", "a").await;
    let b = start("staging", "b").await;
    register_echo(&b);
    let addr = b.bound_address();
    let mut events = a.subscribe();
    let err = a.connect(addr).await.unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("handshake with [{addr}] failed: remote cluster name [staging] does not match local cluster name [prod]")
    );
    assert!(matches!(err, TransportError::HandshakeFailed { .. }));
    let err = a.send_request::<_, Reply>(addr, "test:echo", &Echo { msg: "x".into(), n: 1 }, T).await.unwrap_err();
    assert!(matches!(err, TransportError::HandshakeFailed { .. }));
    assert!(!a.is_connected(addr));
    assert!(a.connected_nodes().is_empty());
    assert_eq!(a.stats().total_outbound_connections, 0);
    assert!(events.try_recv().is_err(), "no Connected event for a rejected peer");
}

#[tokio::test]
async fn reconnects_after_peer_restart() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    register_echo(&b);
    let addr = b.bound_address();
    let first_ephemeral = b.local_node().ephemeral_id.clone();
    let mut events = a.subscribe();
    a.send_request::<_, Reply>(addr, "test:echo", &Echo { msg: "1".into(), n: 1 }, T).await.unwrap();
    assert!(matches!(next_event(&mut events).await, ConnectionEvent::Connected { .. }));

    b.close().await;
    assert!(matches!(next_event(&mut events).await, ConnectionEvent::Disconnected { .. }));
    let err = a.send_request::<_, Reply>(addr, "test:echo", &Echo { msg: "2".into(), n: 2 }, T).await.unwrap_err();
    assert!(err.is_connection_failure(), "{err:?}");

    // Same port, new process identity.
    let mut cfg = config("c1", "b");
    cfg.bind_addr = addr;
    let b2 = TransportService::bind(cfg).await.unwrap();
    register_echo(&b2);
    let r: Reply = a.send_request(addr, "test:echo", &Echo { msg: "3".into(), n: 3 }, T).await.unwrap();
    assert_eq!(r.doubled, 6);
    let id = a.remote_identity(addr).unwrap();
    assert_ne!(id.node.ephemeral_id, first_ephemeral);
    assert_eq!(a.stats().total_outbound_connections, 2);
    assert_eq!(a.stats().outbound_open, 1);
}

#[tokio::test]
async fn disconnect_notifications() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    let c = start("c1", "c").await;
    let mut events = a.subscribe();

    let b_id = a.connect(b.bound_address()).await.unwrap();
    assert_eq!(next_event(&mut events).await, ConnectionEvent::Connected { addr: b.bound_address(), node: b_id.clone() });

    // Peer closes.
    b.close().await;
    assert_eq!(next_event(&mut events).await, ConnectionEvent::Disconnected { addr: b.bound_address(), node: b_id });
    assert!(!a.is_connected(b.bound_address()));

    // Explicit disconnect.
    let c_id = a.connect(c.bound_address()).await.unwrap();
    assert!(matches!(next_event(&mut events).await, ConnectionEvent::Connected { .. }));
    a.disconnect(c.bound_address()).await;
    assert_eq!(next_event(&mut events).await, ConnectionEvent::Disconnected { addr: c.bound_address(), node: c_id });
    // Lazy reconnect on next use.
    a.connect(c.bound_address()).await.unwrap();
    assert!(a.is_connected(c.bound_address()));
}

#[tokio::test]
async fn in_flight_requests_fail_on_close() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    b.register_handler("test:hang", |_: ()| async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok(())
    });
    let addr = b.bound_address();
    a.connect(addr).await.unwrap();

    // Remote side goes away.
    let a2 = a.clone();
    let req = tokio::spawn(async move { a2.send_request::<_, ()>(addr, "test:hang", &(), None).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    b.close().await;
    let err = tokio::time::timeout(Duration::from_secs(5), req).await.unwrap().unwrap().unwrap_err();
    assert_eq!(err, TransportError::NodeDisconnected { addr, action: "test:hang".into() });

    // Local side closes.
    let b = start("c1", "b2").await;
    b.register_handler("test:hang", |_: ()| async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok(())
    });
    let addr = b.bound_address();
    let a2 = a.clone();
    let req = tokio::spawn(async move { a2.send_request::<_, ()>(addr, "test:hang", &(), None).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    a.close().await;
    let err = tokio::time::timeout(Duration::from_secs(5), req).await.unwrap().unwrap().unwrap_err();
    assert!(matches!(err, TransportError::NodeDisconnected { .. }), "{err:?}");
    assert_eq!(a.send_request::<_, ()>(addr, "test:hang", &(), T).await.unwrap_err(), TransportError::Closed);
    // The listener is gone too.
    let probe = start("c1", "probe").await;
    assert!(probe.connect(a.bound_address()).await.is_err());
}

#[tokio::test]
async fn large_payload_is_compressed() {
    let a = start("c1", "a").await;
    let b = start("c1", "b").await;
    // Echoes the payload back with its first byte bumped: an indexing data
    // action (compressed by default).
    b.register_raw_handler("indices:data/write/test_echo", |payload: Bytes| async move {
        let mut v = payload.to_vec();
        v[0] = v[0].wrapping_add(1);
        Ok(Bytes::from(v))
    });
    let size = 64 << 20;
    let payload: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    let resp = a.send_request_bytes(b.bound_address(), "indices:data/write/test_echo", Bytes::from(payload.clone()), Duration::from_secs(60)).await.unwrap();
    assert_eq!(resp.len(), size);
    assert_eq!(resp[0], 1);
    assert_eq!(&resp[1..], &payload[1..]);
    let s = a.stats();
    assert!(s.tx_size_in_bytes < (size / 10) as u64, "request should be compressed: {s:?}");
    // The response to a compressed request is compressed too.
    assert!(s.rx_size_in_bytes < (size / 10) as u64, "response compressed like its request: {s:?}");

    // Other actions (search traffic) are not compressed by default.
    let before = a.stats();
    let resp = a.send_request_bytes(b.bound_address(), "indices:data/write/test_echo", Bytes::from(vec![7u8; 1 << 20]), Duration::from_secs(60)).await.unwrap();
    assert_eq!(resp.len(), 1 << 20);
    b.register_raw_handler("test:plain_echo", |payload: Bytes| async move { Ok(payload) });
    b.register_handler("test:typed_echo", |v: Vec<u8>| async move { Ok(v) });
    let mid = a.stats();
    let resp = a.send_request_bytes(b.bound_address(), "test:plain_echo", Bytes::from(vec![7u8; 1 << 20]), Duration::from_secs(60)).await.unwrap();
    assert_eq!(resp.len(), 1 << 20);
    let after = a.stats();
    assert!(mid.tx_size_in_bytes - before.tx_size_in_bytes < 64 << 10, "indexing data compressed");
    assert!(after.tx_size_in_bytes - mid.tx_size_in_bytes >= 1 << 20, "other actions sent as they are");
    assert!(after.rx_size_in_bytes - mid.rx_size_in_bytes >= 1 << 20, "and so are their responses");
    // Indexing data under another action (a forwarded write): compressed,
    // with its response.
    let resp: Vec<u8> = a.send_indexing_request(b.bound_address(), "test:typed_echo", &vec![7u8; 1 << 20], Duration::from_secs(60)).await.unwrap();
    assert_eq!(resp.len(), 1 << 20);
    let last = a.stats();
    assert!(last.tx_size_in_bytes - after.tx_size_in_bytes < 64 << 10 && last.rx_size_in_bytes - after.rx_size_in_bytes < 64 << 10, "{last:?}");
    assert!(crate_indexing_data_cases());

    // Typed path with a sizeable struct.
    b.register_handler("test:sum", |v: Vec<u64>| async move { Ok(v.iter().sum::<u64>()) });
    let v: Vec<u64> = (0..1_000_000).collect();
    let sum: u64 = a.send_request(b.bound_address(), "test:sum", &v, Duration::from_secs(60)).await.unwrap();
    assert_eq!(sum, v.iter().sum::<u64>());
}

#[tokio::test]
async fn max_frame_size_is_enforced() {
    let mut cfg = config("c1", "a");
    cfg.settings.max_frame_size = 1 << 20;
    cfg.settings.compress = false;
    let a = TransportService::bind(cfg).await.unwrap();
    let mut cfg = config("c1", "b");
    cfg.settings.max_frame_size = 1 << 20;
    // Every action compressed (`transport.compress: true`), so the blowup
    // reaches `a` as a small frame.
    cfg.settings.compress_all = true;
    let b = TransportService::bind(cfg).await.unwrap();
    b.register_raw_handler("test:blowup", |_p: Bytes| async move { Ok(Bytes::from(vec![0u8; 4 << 20])) });
    b.register_raw_handler("test:ok", |p: Bytes| async move { Ok(p) });
    let addr = b.bound_address();

    let err = a.send_request_bytes(addr, "test:ok", Bytes::from(vec![1u8; 2 << 20]), T).await.unwrap_err();
    assert!(matches!(err, TransportError::FrameTooLarge { max: 1048576, .. }), "{err:?}");
    assert!(err.to_string().contains("exceeds the maximum frame size of [1048576] bytes"));
    // Oversized *response* (b compresses, but 4MB of zeros decompresses past
    // a's limit... b's encoder sees it compressed to a few KB, so the receiver
    // guards against the decompressed size): the connection is dropped and
    // the request fails cleanly.
    let err = a.send_request_bytes(addr, "test:blowup", Bytes::new(), T).await.unwrap_err();
    assert!(err.is_connection_failure() || matches!(err, TransportError::Remote(_)), "{err:?}");
    // Normal traffic still works (reconnecting if needed).
    let r = a.send_request_bytes(addr, "test:ok", Bytes::from_static(b"hello"), T).await.unwrap();
    assert_eq!(&r[..], b"hello");
}

#[tokio::test]
async fn local_requests_short_circuit() {
    let a = start("c1", "a").await;
    register_echo(&a);
    let calls = Arc::new(AtomicUsize::new(0));
    let c2 = calls.clone();
    a.register_handler("test:count", move |_: ()| {
        let c = c2.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });
    let me = a.bound_address();
    assert!(a.is_local_address(me));
    let r: Reply = a.send_request(me, "test:echo", &Echo { msg: "self".into(), n: 5 }, T).await.unwrap();
    assert_eq!(r, Reply { msg: "SELF".into(), doubled: 10, from: "a".into() });
    a.send_request::<_, ()>(me, "test:count", &(), T).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(a.connect(me).await.unwrap(), a.identity().clone());
    let s = a.stats();
    assert_eq!((s.tx_count, s.rx_count, s.server_open, s.total_outbound_connections), (0, 0, 0, 0));

    a.register_handler("test:slow", |_: ()| async move {
        tokio::time::sleep(Duration::from_secs(5)).await;
        Ok(())
    });
    let err = a.send_request::<_, ()>(me, "test:slow", &(), Duration::from_millis(50)).await.unwrap_err();
    assert!(matches!(err, TransportError::ReceiveTimeout { .. }));
}

#[tokio::test]
async fn connect_failures() {
    let a = start("c1", "a").await;
    // Nothing listens on a just-freed port.
    let free: SocketAddr = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let err = a.connect(free).await.unwrap_err();
    assert!(matches!(err, TransportError::ConnectTransport { .. }), "{err:?}");
    assert_eq!(err.to_remote_error().error_type, "connect_transport_exception");

    // A peer that accepts but never handshakes hits connect_timeout.
    let silent = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let silent_addr = silent.local_addr().unwrap();
    let mut cfg = config("c1", "b");
    cfg.settings.connect_timeout = Duration::from_millis(200);
    let b = TransportService::bind(cfg).await.unwrap();
    let err = b.connect(silent_addr).await.unwrap_err();
    assert_eq!(err, TransportError::ConnectTransport { addr: silent_addr, reason: "connect_timeout[200ms]".into() });
    drop(silent);
}

#[tokio::test]
async fn pings_keep_connection_alive() {
    let mut cfg = config("c1", "a");
    cfg.settings.ping_schedule = Some(Duration::from_millis(50));
    let a = TransportService::bind(cfg).await.unwrap();
    let b = start("c1", "b").await;
    a.connect(b.bound_address()).await.unwrap();
    let before = b.stats().rx_count;
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(b.stats().rx_count >= before + 4, "pings received: {:?}", b.stats());
    assert!(a.is_connected(b.bound_address()));
}

// ------------------------------------------------------------------ TLS ----

#[cfg(feature = "tls")]
struct TestPki {
    ca_pem: String,
    cert_pem: String,
    key_pem: String,
}

#[cfg(feature = "tls")]
fn pki() -> (TestPki, TestPki) {
    use rcgen::{BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair, SanType};
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.distinguished_name.push(DnType::CommonName, "transport-test-ca");
    let ca = ca_params.self_signed(&ca_key).unwrap();

    let node_key = KeyPair::generate().unwrap();
    let mut p = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    p.subject_alt_names.push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
    p.distinguished_name.push(DnType::CommonName, "node");
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth, ExtendedKeyUsagePurpose::ClientAuth];
    let cert = p.signed_by(&node_key, &ca, &ca_key).unwrap();
    let good = TestPki { ca_pem: ca.pem(), cert_pem: cert.pem(), key_pem: node_key.serialize_pem() };

    // A second, unrelated CA + node cert.
    let rogue_ca_key = KeyPair::generate().unwrap();
    let mut rp = CertificateParams::new(Vec::<String>::new()).unwrap();
    rp.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let rogue_ca = rp.self_signed(&rogue_ca_key).unwrap();
    let rogue_key = KeyPair::generate().unwrap();
    let mut p = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    p.subject_alt_names.push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
    p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth, ExtendedKeyUsagePurpose::ClientAuth];
    let rogue_cert = p.signed_by(&rogue_key, &rogue_ca, &rogue_ca_key).unwrap();
    let rogue = TestPki { ca_pem: rogue_ca.pem(), cert_pem: rogue_cert.pem(), key_pem: rogue_key.serialize_pem() };
    (good, rogue)
}

#[cfg(feature = "tls")]
fn tls(p: &TestPki, mode: VerificationMode) -> TlsConfig {
    TlsConfig {
        certificate: PemSource::Inline(p.cert_pem.clone()),
        key: PemSource::Inline(p.key_pem.clone()),
        key_passphrase: None,
        certificate_authorities: vec![PemSource::Inline(p.ca_pem.clone())],
        verification_mode: mode,
        client_authentication: ClientAuthentication::Required,
        trust_restrictions: None,
    }
}

/// A CA and two node certificates carrying ECK-style `otherName` CN SANs
/// (type-id 2.5.4.3): one under `*.node.es.local`, one outside it.
#[cfg(feature = "tls")]
fn eck_style_pki() -> (TestPki, TestPki) {
    use rcgen::{BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair, OtherNameValue, SanType};
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let node = |cn: &str| {
        let key = KeyPair::generate().unwrap();
        let mut p = CertificateParams::new(vec![cn.to_string()]).unwrap();
        p.subject_alt_names.insert(0, SanType::OtherName((vec![2, 5, 4, 3], OtherNameValue::Utf8String(cn.to_string()))));
        p.subject_alt_names.push(SanType::IpAddress("127.0.0.1".parse().unwrap()));
        p.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth, ExtendedKeyUsagePurpose::ClientAuth];
        let cert = p.signed_by(&key, &ca, &ca_key).unwrap();
        TestPki { ca_pem: ca.pem(), cert_pem: cert.pem(), key_pem: key.serialize_pem() }
    };
    (node("es-0.node.es.local"), node("es-9.node.other.local"))
}

#[cfg(feature = "tls")]
#[tokio::test]
async fn tls_trust_restrictions_reject_untrusted_names() {
    let (trusted, other) = eck_style_pki();
    let dir = std::env::temp_dir().join(format!("trust-restrictions-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let trust = dir.join("trust.yml");
    std::fs::write(&trust, "trust.subject_name: [\"*.node.es.local\"]\n").unwrap();
    let restricted = |p: &TestPki| {
        let mut c = tls(p, VerificationMode::Certificate);
        c.trust_restrictions = Some(elasticsearch_ssl::TrustRestrictionsConfig { path: trust.clone(), fields: vec![elasticsearch_ssl::X509Field::SanOtherNameCommonName] });
        c
    };
    // Both certificates chain to the same CA; only one has a trusted name.
    let server = start_tls("restricted", Some(restricted(&trusted))).await;
    register_echo(&server);
    let good = start_tls("good", Some(restricted(&trusted))).await;
    let r: Reply = good.send_request(server.bound_address(), "test:echo", &Echo { msg: "ok".into(), n: 2 }, T).await.unwrap();
    assert_eq!(r.doubled, 4);
    // Inbound: an unrestricted client with the untrusted name is refused.
    let intruder = start_tls("intruder", Some(tls(&other, VerificationMode::Certificate))).await;
    assert!(intruder.send_request::<_, Reply>(server.bound_address(), "test:echo", &Echo { msg: "x".into(), n: 1 }, T).await.is_err());
    // Outbound: a restricted client refuses a server with the untrusted name.
    let other_server = start_tls("other-server", Some(tls(&other, VerificationMode::Certificate))).await;
    register_echo(&other_server);
    assert!(good.connect(other_server.bound_address()).await.is_err());
    // Without restrictions the same pair connects.
    let plain_client = start_tls("plain-client", Some(tls(&trusted, VerificationMode::Certificate))).await;
    plain_client.connect(other_server.bound_address()).await.unwrap();
    // trust.yml is one of the watched files: trusting the other name and
    // reloading lets the restricted server accept it (ES rebuilds the
    // restricted trust manager when the file changes).
    std::fs::write(&trust, "trust.subject_name: [\"*.node.es.local\", \"*.node.other.local\"]\n").unwrap();
    assert_eq!(server.ssl_reloader().unwrap().check(), elasticsearch_ssl::ReloadOutcome::Reloaded);
    let r: Reply = intruder.send_request(server.bound_address(), "test:echo", &Echo { msg: "now trusted".into(), n: 3 }, T).await.unwrap();
    assert_eq!(r.doubled, 6);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "tls")]
async fn start_tls(name: &str, tls_cfg: Option<TlsConfig>) -> TransportService {
    let mut cfg = config("c1", name);
    cfg.tls = tls_cfg;
    cfg.settings.connect_timeout = Duration::from_secs(2);
    TransportService::bind(cfg).await.unwrap()
}

#[cfg(feature = "tls")]
#[tokio::test]
async fn tls_mutual_roundtrip() {
    let (good, rogue) = pki();
    for mode in [VerificationMode::Full, VerificationMode::Certificate] {
        let a = start_tls("a", Some(tls(&good, mode))).await;
        let b = start_tls("b", Some(tls(&good, mode))).await;
        register_echo(&b);
        let r: Reply = a.send_request(b.bound_address(), "test:echo", &Echo { msg: "tls".into(), n: 4 }, T).await.unwrap();
        assert_eq!(r.doubled, 8);
        // Large compressed payload over TLS.
        b.register_raw_handler("test:len", |p: Bytes| async move { Ok(Bytes::from((p.len() as u64).to_le_bytes().to_vec())) });
        let resp = a.send_request_bytes(b.bound_address(), "test:len", Bytes::from(vec![9u8; 8 << 20]), T).await.unwrap();
        assert_eq!(u64::from_le_bytes(resp[..8].try_into().unwrap()), 8 << 20);
    }

    let b = start_tls("b", Some(tls(&good, VerificationMode::Certificate))).await;
    register_echo(&b);
    // Plaintext client against a TLS server: fails to handshake.
    let plain = start_tls("plain", None).await;
    assert!(plain.connect(b.bound_address()).await.is_err());
    // Client with a certificate from an untrusted CA: rejected (mutual TLS).
    let mut rogue_cfg = tls(&rogue, VerificationMode::None);
    rogue_cfg.certificate_authorities.clear();
    rogue_cfg.client_authentication = ClientAuthentication::None;
    let r = start_tls("rogue", Some(rogue_cfg)).await;
    assert!(r.connect(b.bound_address()).await.is_err());
    // Client trusting a different CA refuses the server.
    let r2 = start_tls("rogue2", Some(tls(&rogue, VerificationMode::Certificate))).await;
    let err = r2.connect(b.bound_address()).await.unwrap_err();
    assert!(matches!(err, TransportError::ConnectTransport { .. }), "{err:?}");
    // And the good client still works.
    let a = start_tls("a", Some(tls(&good, VerificationMode::Certificate))).await;
    a.connect(b.bound_address()).await.unwrap();
}

#[cfg(feature = "tls")]
#[tokio::test]
async fn tls_config_errors() {
    let (good, _) = pki();
    let mut bad = tls(&good, VerificationMode::Full);
    bad.key = PemSource::Path("/nonexistent/key.pem".into());
    let mut cfg = config("c1", "x");
    cfg.tls = Some(bad);
    assert!(matches!(TransportService::bind(cfg).await.unwrap_err(), TransportError::Config(_)));
}

/// ES's x-pack REST test cluster (`XPackRestIT`): `testnode.crt` with the
/// passphrase-protected `testnode.pem`, no `certificate_authorities`
/// (each node trusts the shared certificate), mutual TLS.
#[cfg(feature = "tls")]
fn testnode_tls(mode: VerificationMode, passphrase: Option<&str>) -> TlsConfig {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    TlsConfig {
        certificate: PemSource::Path(data.join("testnode.crt")),
        key: PemSource::Path(data.join("testnode.pem")),
        key_passphrase: passphrase.map(str::to_string),
        certificate_authorities: vec![],
        verification_mode: mode,
        client_authentication: ClientAuthentication::Required,
        trust_restrictions: None,
    }
}

#[cfg(feature = "tls")]
#[tokio::test]
async fn tls_with_the_es_test_node_certificate_and_encrypted_key() {
    for mode in [VerificationMode::Certificate, VerificationMode::Full] {
        let a = start_tls("a", Some(testnode_tls(mode, Some("testnode")))).await;
        let b = start_tls("b", Some(testnode_tls(mode, Some("testnode")))).await;
        register_echo(&b);
        let r: Reply = a.send_request(b.bound_address(), "test:echo", &Echo { msg: "testnode".into(), n: 21 }, T).await.unwrap();
        assert_eq!(r.doubled, 42, "{mode:?}");
    }
    // A wrong or missing passphrase stops the node from starting.
    for pass in [Some("wrong"), None] {
        let mut cfg = config("c1", "x");
        cfg.tls = Some(testnode_tls(VerificationMode::Certificate, pass));
        match TransportService::bind(cfg).await {
            Err(TransportError::Config(msg)) => assert!(msg.contains("testnode.pem"), "{msg}"),
            other => panic!("expected a config error, got {:?}", other.map(|_| ())),
        }
    }
    // A node with another certificate is not trusted by default.
    let (good, _) = pki();
    let b = start_tls("b", Some(testnode_tls(VerificationMode::Certificate, Some("testnode")))).await;
    let mut other = tls(&good, VerificationMode::Certificate);
    other.certificate_authorities.clear();
    let o = start_tls("other", Some(other)).await;
    assert!(o.connect(b.bound_address()).await.is_err());
}

/// Responses spanning many TLS records (16 KiB each) arrive byte-identical,
/// compressed or not, including many concurrent ones on one connection.
#[cfg(feature = "tls")]
#[tokio::test]
async fn tls_large_responses_are_byte_identical() {
    for compress in [false, true] {
        let mut ca = config("c1", "a");
        ca.tls = Some(testnode_tls(VerificationMode::Certificate, Some("testnode")));
        ca.settings.compress = compress;
        let mut cb = config("c1", "b");
        cb.tls = Some(testnode_tls(VerificationMode::Certificate, Some("testnode")));
        cb.settings.compress = compress;
        let a = Arc::new(TransportService::bind(ca).await.unwrap());
        let b = TransportService::bind(cb).await.unwrap();
        // The handler answers with a deterministic pseudo-random body of the requested size.
        b.register_raw_handler("test:blob", |p: Bytes| async move {
            let n = u64::from_le_bytes(p[..8].try_into().unwrap()) as usize;
            Ok(Bytes::from(blob(n)))
        });
        let addr = b.bound_address();
        let mut tasks = Vec::new();
        for (i, n) in [1usize, 16 << 10, (16 << 10) + 1, 300 << 10, 777_777, 8 << 20].into_iter().cycle().take(24).enumerate() {
            let a = a.clone();
            tasks.push(tokio::spawn(async move {
                let resp = a.send_request_bytes(addr, "test:blob", Bytes::from((n as u64).to_le_bytes().to_vec()), T).await.unwrap();
                assert_eq!(resp.len(), n, "request {i}");
                assert!(resp[..] == blob(n)[..], "request {i}: {n} bytes differ (compress={compress})");
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
    }
}

#[cfg(feature = "tls")]
fn blob(n: usize) -> Vec<u8> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15 ^ n as u64;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8 }).collect()
}

/// Transport throughput with and without TLS (`cargo test -- --ignored
/// --nocapture tls_throughput`): request/response round trips of 64 KiB and
/// 1 MiB payloads over one pooled connection, 16 in flight.
#[cfg(feature = "tls")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn tls_throughput() {
    for (label, tls_cfg) in [("plaintext", None), ("tls", Some(testnode_tls(VerificationMode::Certificate, Some("testnode"))))] {
        let mut ca = config("c1", "a");
        ca.tls = tls_cfg.clone();
        ca.settings.compress = false;
        let mut cb = config("c1", "b");
        cb.tls = tls_cfg;
        cb.settings.compress = false;
        let a = std::sync::Arc::new(TransportService::bind(ca).await.unwrap());
        let b = TransportService::bind(cb).await.unwrap();
        b.register_raw_handler("test:len", |p: Bytes| async move { Ok(Bytes::from((p.len() as u64).to_le_bytes().to_vec())) });
        let addr = b.bound_address();
        a.connect(addr).await.unwrap();
        for size in [64usize << 10, 1 << 20] {
            let payload = Bytes::from(vec![7u8; size]);
            let total: usize = (512usize << 20) / size;
            let start = std::time::Instant::now();
            let mut tasks = Vec::new();
            for w in 0..16 {
                let (a, payload) = (a.clone(), payload.clone());
                tasks.push(tokio::spawn(async move {
                    for _ in (w..total).step_by(16) {
                        a.send_request_bytes(addr, "test:len", payload.clone(), Duration::from_secs(60)).await.unwrap();
                    }
                }));
            }
            for t in tasks {
                t.await.unwrap();
            }
            let secs = start.elapsed().as_secs_f64();
            println!("THROUGHPUT {label} payload={}KiB requests={total} {:.0} MiB/s {:.0} req/s", size >> 10, (total * size) as f64 / secs / (1 << 20) as f64, total as f64 / secs);
        }
    }
}

fn crate_indexing_data_cases() -> bool {
    use elasticsearch_transport::indexing_data;
    indexing_data("indices:data/write/replica")
        && indexing_data("internal:index/seq_no/resync")
        && indexing_data("internal:index/shard/recovery/batch")
        && indexing_data("internal:index/shard/recovery/reuse_batch")
        && !indexing_data("internal:index/shard/recovery/start_recovery")
        && !indexing_data("internal:rest/forward")
}
