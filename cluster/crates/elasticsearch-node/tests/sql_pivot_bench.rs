//! Rally sql `pivot` / `pivot_3pages` through the cluster front:
//! `PIVOT_DOCS` (default 4M) NOAA-like docs over `PIVOT_STATIONS` (40k)
//! stations, `PIVOT_SHARDS` (1) shards on the queried node (`PIVOT_REPLICAS`
//! replicas: on both nodes), force-merged;
//! median of 7 per operation, pages of 1000 rows through the cursor.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::time::Instant;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn sql_pivot_bench() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (docs, stations, shards) = (env("PIVOT_DOCS", 4_000_000), env("PIVOT_STATIONS", 40_000), env("PIVOT_SHARDS", 1));
    let cl = start(&[]).await;
    let base = format!("http://127.0.0.1:{}", cl.p.http[0]);
    // `PIVOT_OFF_MASTER`: the shards on the node that is not the master
    // (where SQL runs), else on n0.
    let off_master = std::env::var_os("PIVOT_OFF_MASTER").is_some();
    // `PIVOT_REPLICAS` (default 0): with 1, both nodes hold every shard, as
    // the Rally cluster does (2 shards, 1 replica, 2 nodes).
    let replicas = env("PIVOT_REPLICAS", 0);
    let mut settings = json!({"number_of_shards": shards, "number_of_replicas": replicas, "refresh_interval": "-1"});
    if replicas > 0 {
        // Copies on both nodes: no allocation filter.
    } else if off_master {
        cl.exclude_master().await;
    } else if std::env::var_os("PIVOT_SPREAD").is_some() {
        // `PIVOT_SPREAD`: the shards over both nodes (no node holds them all).
        settings["index.routing.allocation.total_shards_per_node"] = json!(shards.div_ceil(2));
    } else {
        settings["index.routing.allocation.require._name"] = json!("n0");
    }
    let (st, b) = cl.call(0, Method::PUT, "/w", Some(json!({
        "settings": settings,
        "mappings": {"dynamic": "strict", "properties": {"TMIN": {"type": "float"},
            "station": {"properties": {"name": {"type": "keyword"}, "country_code": {"type": "keyword"}, "elevation": {"type": "float"}}}}}
    }))).await;
    assert_eq!(st, 200, "{b}");
    let (_, h) = cl.call(0, Method::GET, "/_cluster/health/w?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(h["status"], "green", "{h}");
    if off_master {
        assert!(cl.primary_off_master("w").await);
    }
    let load = Instant::now();
    let countries = ["US", "US", "US", "AS", "CA", "MX", "FR"];
    let mut n = 0usize;
    while n < docs {
        let mut bulk = String::with_capacity(12 << 20);
        for _ in 0..50_000.min(docs - n) {
            let s = n.wrapping_mul(2654435761) % stations;
            bulk.push_str("{\"index\":{}}\n");
            let tmin = if n % 5 >= 2 { format!(",\"TMIN\":{}", (n % 61) as f64 - 30.5) } else { String::new() };
            bulk.push_str(&format!("{{\"station\":{{\"name\":\"STATION {s:06} XX\",\"country_code\":\"{}\",\"elevation\":{}}}{tmin}}}\n", countries[s % 7], (s * 37 % 4000) as f64));
            n += 1;
        }
        let r: Value = cl.c.post(format!("{base}/w/_bulk")).header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(r["errors"], false);
    }
    let (st, b) = cl.call(0, Method::POST, "/w/_refresh", None).await;
    assert_eq!(st, 200, "{b}");
    let (st, b) = cl.call(0, Method::POST, "/w/_forcemerge?max_num_segments=1", None).await;
    assert_eq!(st, 200, "{b}");
    eprintln!("loaded {docs} docs in {:.1} s", load.elapsed().as_secs_f64());
    let sql = |body: Value| {
        let cl = &cl;
        let base = base.clone();
        async move {
            let r = cl.c.post(format!("{base}/_sql?format=json")).header("content-type", "application/json").header("accept-encoding", "identity").body(body.to_string()).send().await.unwrap();
            let status = r.status().as_u16();
            let v: Value = r.json().await.unwrap();
            assert_eq!(status, 200, "{v}");
            v
        }
    };
    for (name, query, pages) in [
        ("pivot", "SELECT * FROM (SELECT station.country_code, TMIN FROM w) PIVOT (AVG(TMIN) FOR station.country_code IN ('AS', 'CA'))", 1usize),
        ("pivot_3pages", "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))", 3),
        ("pivot_3pages first page", "SELECT * FROM (SELECT station.name, station.country_code FROM w WHERE station.elevation > 1600) PIVOT (COUNT(*) FOR station.country_code IN ('US', 'AS'))", 1),
    ] {
        let mut times = Vec::new();
        let mut first = Vec::new();
        let mut rows = 0;
        for _ in 0..7 {
            let t = Instant::now();
            let mut v = sql(json!({"query": query, "fetch_size": 1000, "page_timeout": "1000s", "request_timeout": "1000s"})).await;
            first.push(t.elapsed().as_secs_f64() * 1000.0);
            rows = v["rows"].as_array().map_or(0, Vec::len);
            for _ in 1..pages {
                let Some(c) = v.get("cursor").cloned() else { break };
                v = sql(json!({"cursor": c})).await;
                rows += v["rows"].as_array().map_or(0, Vec::len);
            }
            if let Some(c) = v.get("cursor").cloned() {
                sql_close(&cl, &base, c).await;
            }
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        first.sort_by(f64::total_cmp);
        eprintln!("{name:<26} median {:>9.2} ms (first page {:>9.2} ms), {rows} rows", times[3], first[3]);
    }
}

async fn sql_close(cl: &common::Cluster, base: &str, cursor: Value) {
    let _ = cl.c.post(format!("{base}/_sql/close")).header("content-type", "application/json").body(json!({"cursor": cursor}).to_string()).send().await;
}
