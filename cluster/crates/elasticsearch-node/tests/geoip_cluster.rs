//! GeoIP databases on a cluster (finding 446, ES parity): the elected
//! master downloads the default databases into `.geoip_databases`, and
//! every node installs them from there, so documents ingested through any
//! node are enriched.

mod common;

use common::procs::start_procs;
use reqwest::Method;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

fn tar_of(path: &str, data: &[u8]) -> Vec<u8> {
    let mut h = vec![0u8; 512];
    h[..path.len()].copy_from_slice(path.as_bytes());
    h[100..107].copy_from_slice(b"0000644");
    h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
    h[156] = b'0';
    h[257..263].copy_from_slice(b"ustar\0");
    let mut out = h;
    out.extend_from_slice(data);
    out.resize(out.len().div_ceil(512) * 512 + 1024, 0);
    out
}

fn md5_hex(data: &[u8]) -> String {
    elasticsearch_engine::ingest::geoip_download::md5_digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_master_downloads_and_every_node_installs() {
    let mmdb = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../crates/elasticsearch-http/tests/data/geoip/GeoLite2-City.mmdb")).unwrap();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    gz.write_all(&tar_of("GeoLite2-City_20250101/GeoLite2-City.mmdb", &mmdb)).unwrap();
    let archive = gz.finish().unwrap();
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let overview = json!([{"name": "GeoLite2-City.tgz", "md5_hash": md5_hex(&archive), "url": format!("{base}/db/GeoLite2-City.tgz")}]).to_string();
    let downloads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = downloads.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { continue };
            let mut buf = [0u8; 4096];
            let n = c.read(&mut buf).unwrap_or(0);
            let head = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = if head.contains("/db/") {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                archive.clone()
            } else {
                overview.clone().into_bytes()
            };
            let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = c.write_all(&body);
        }
    });
    let endpoint = format!("{base}/v1/database");
    let p = start_procs(2, &[("ingest.geoip.downloader.endpoint", endpoint.as_str())]).await;
    let (st, v) = p.call(0, Method::PUT, "/_ingest/pipeline/g", Some(json!({"processors": [{"geoip": {"field": "ip"}}]}))).await;
    assert_eq!(st, 200, "{v}");
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut done = [false, false];
    let mut i = 0;
    while done.iter().any(|d| !d) {
        for node in 0..2 {
            if done[node] {
                continue;
            }
            i += 1;
            let (st, v) = p.call(node, Method::PUT, &format!("/events/_doc/{i}?pipeline=g&refresh=true"), Some(json!({"ip": "89.160.20.128"}))).await;
            assert!(st == 200 || st == 201, "{v}");
            let (_, d) = p.call(node, Method::GET, &format!("/events/_doc/{i}"), None).await;
            if d["_source"]["geoip"]["city_name"] == "Linköping" {
                done[node] = true;
            }
        }
        assert!(Instant::now() < deadline, "not enriched on every node: {done:?}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    // Downloaded once (by the master), stored as chunks in the system index.
    assert_eq!(downloads.load(std::sync::atomic::Ordering::SeqCst), 1);
    let (st, v): (u16, Value) = p.call(0, Method::POST, "/.geoip_databases/_search?expand_wildcards=all", Some(json!({"size": 0, "query": {"term": {"name": "GeoLite2-City.mmdb"}}}))).await;
    assert_eq!(st, 200, "{v}");
    assert!(v["hits"]["total"]["value"].as_u64().unwrap_or(0) >= 1, "{v}");
}
