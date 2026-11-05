//! Fleet must forward every shard's response without a Value round-trip,
//! including checkpointed msearch's separate coordinator rendering path.
mod common;
use common::procs::start_procs;
use elasticsearch_engine::source_json::{raw_elements, raw_members};
use reqwest::Method;
use serde_json::json;

const NODES: usize = 2;
const SOURCE: &str = r#"{"z":"\ud800","n":1.50,"a":2}"#;

fn field(text: &str, key: &str) -> String {
    raw_members(text).unwrap().into_iter().find(|(k, _)| k == key).unwrap().1.get().to_string()
}
fn check_search(text: &str) {
    let hits = raw_elements(&field(&field(text, "hits"), "hits")).unwrap();
    assert_eq!(hits.len(), 1, "{text}");
    assert_eq!(field(hits[0].get(), "_source"), SOURCE, "source order, lone units and numeric lexemes must survive");
    let shards = field(text, "_shards");
    assert_eq!(field(&shards, "total"), "2");
    assert_eq!(field(&shards, "failed"), "0");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fleet_single_and_multi_search_preserve_raw_sources_on_both_coordinators() {
    let cl = start_procs(NODES, &[]).await;
    let (status, body) = cl.call(0, Method::PUT, "/fleet-raw", Some(json!({
        "settings":{"number_of_shards":2,"number_of_replicas":1},
        "mappings":{"properties":{"z":{"type":"keyword"},"n":{"type":"double"},"a":{"type":"integer"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    cl.wait_health(0, "fleet-raw", "green", 60).await;
    let (status, body) = cl.ndjson(0, "/fleet-raw/_bulk?refresh=true", format!("{{\"index\":{{\"_id\":\"one\"}}}}\n{SOURCE}\n")).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["errors"], false, "{body}");
    for node in 0..2 {
        for suffix in ["", "?wait_for_checkpoints=-1,-1"] {
            let response = cl.c.post(format!("http://127.0.0.1:{}/fleet-raw/_fleet/_fleet_search{suffix}", cl.p.http[node]))
                .json(&json!({"query":{"match_all":{}}})).send().await.unwrap();
            let status = response.status();
            let text = response.text().await.unwrap();
            assert_eq!(status, 200, "{text}");
            check_search(&text);
        }
        for checkpoints in ["", ",\"wait_for_checkpoints\":[-1,-1]"] {
            let body = format!("{{\"index\":\"fleet-raw\"{checkpoints}}}\n{{}}\n{{\"index\":\"fleet-missing\",\"ignore_unavailable\":true,\"allow_no_indices\":true}}\n{{}}\n");
            let response = cl.c.post(format!("http://127.0.0.1:{}/_fleet/_fleet_msearch", cl.p.http[node]))
                .header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
            let status = response.status();
            let text = response.text().await.unwrap();
            assert_eq!(status, 200, "{text}");
            let items = raw_elements(&field(&text, "responses")).unwrap();
            assert_eq!(items.len(), 2, "{text}");
            check_search(items[0].get());
            assert_eq!(field(items[0].get(), "status"), "200");
            assert_eq!(field(items[1].get(), "status"), "200", "header index options must reach coordinator: {text}");
        }
    }
}
