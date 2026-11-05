//! Two real processes; one primary and zero replicas intentionally force the
//! same copy to move away and back. Reader contexts must outlive its writer.
mod common;
use common::procs::{start_procs, Procs};
use reqwest::Method;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

async fn ok(nodes: &Procs, method: Method, path: &str, body: Option<Value>) -> Value {
    let (status, response) = nodes.call(0, method.clone(), path, body).await;
    assert!((200..300).contains(&status), "{method} {path}: {status} {response}");
    response
}

async fn move_to(nodes: &Procs, name: &str) {
    ok(nodes, Method::PUT, "/retained/_settings", Some(json!({"index.routing.allocation.include._name":name}))).await;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        nodes.wait_health(0, "retained", "green", 120).await;
        let rows = ok(nodes, Method::GET, "/_cat/shards/retained?format=json&h=state,node", None).await;
        if rows.as_array().is_some_and(|rows| rows.len() == 1 && rows[0]["state"] == "STARTED" && rows[0]["node"] == name) { break; }
        assert!(Instant::now() < deadline, "copy did not move to {name}: {rows}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    nodes.wait_health(1, "retained", "green", 120).await;
}

#[tokio::test(flavor="multi_thread", worker_threads=8)]
async fn scroll_and_pit_survive_relocation_away_and_back() {
    let nodes = start_procs(2, &[]).await;
    nodes.wait_nodes(0,2,60).await;
    ok(&nodes, Method::PUT, "/retained", Some(json!({"settings":{
        "number_of_shards":1,"number_of_replicas":0,"routing.allocation.include._name":"n0"
    },"mappings":{"properties":{"rank":{"type":"integer"},"tag":{"type":"keyword"}}}}))).await;
    nodes.wait_health(0,"retained","green",60).await;
    let mut bulk = String::new();
    for i in 0..20 { bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{{\"rank\":{i},\"tag\":\"old\"}}\n")); }
    let (status, response) = nodes.ndjson(0,"/retained/_bulk?refresh=true",bulk).await;
    assert_eq!(status,200,"{response}");
    assert_eq!(response["errors"],false,"{response}");
    ok(&nodes,Method::POST,"/retained/_flush",None).await;
    let first = ok(&nodes,Method::POST,"/retained/_search?scroll=5m",Some(json!({"size":5,"sort":[{"rank":"asc"}]}))).await;
    let mut scroll = first["_scroll_id"].as_str().unwrap().to_string();
    let mut hits = first["hits"]["hits"].as_array().unwrap().clone();
    let pit = ok(&nodes,Method::POST,"/retained/_pit?keep_alive=5m",None).await;
    let mut pit_id = pit["id"].as_str().unwrap().to_string();

    move_to(&nodes,"n1").await;
    ok(&nodes,Method::PUT,"/retained/_doc/0?refresh=true",Some(json!({"rank":100,"tag":"new"}))).await;
    ok(&nodes,Method::DELETE,"/retained/_doc/1?refresh=true",None).await;
    ok(&nodes,Method::PUT,"/retained/_doc/replacement?refresh=true",Some(json!({"rank":101,"tag":"new"}))).await;
    ok(&nodes,Method::POST,"/retained/_flush",None).await;
    move_to(&nodes,"n0").await;

    for _ in 0..5 {
        let next = ok(&nodes,Method::POST,"/_search/scroll",Some(json!({"scroll_id":scroll,"scroll":"5m"}))).await;
        assert_eq!(next["_shards"]["failed"],0,"{next}");
        scroll = next["_scroll_id"].as_str().unwrap().to_string();
        let page = next["hits"]["hits"].as_array().unwrap();
        if page.is_empty() { break; }
        hits.extend(page.iter().cloned());
    }
    assert_eq!(hits.len(),20);
    for (i, hit) in hits.iter().enumerate() {
        assert_eq!(hit["_id"], i.to_string());
        assert_eq!(hit["_source"]["tag"],"old");
    }
    let page = ok(&nodes,Method::POST,"/_search",Some(json!({"size":20,"pit":{"id":pit_id,"keep_alive":"5m"},"sort":[{"rank":"asc"}],"docvalue_fields":["rank","tag"]}))).await;
    assert_eq!(page["_shards"]["failed"],0,"{page}");
    if let Some(updated) = page["pit_id"].as_str() { pit_id = updated.to_string(); }
    let pit_hits = page["hits"]["hits"].as_array().unwrap();
    assert_eq!(pit_hits.len(),20);
    for (i, hit) in pit_hits.iter().enumerate() {
        assert_eq!(hit["_id"],i.to_string());
        assert_eq!(hit["fields"]["rank"][0],i);
        assert_eq!(hit["fields"]["tag"][0],"old");
    }
    let live = ok(&nodes,Method::POST,"/retained/_search",Some(json!({"query":{"term":{"tag":"new"}}}))).await;
    assert_eq!(live["hits"]["total"]["value"],2,"{live}");
    let closed = ok(&nodes,Method::DELETE,"/_pit",Some(json!({"id":pit_id}))).await;
    assert_eq!(closed["succeeded"],true,"{closed}");
    ok(&nodes,Method::DELETE,"/_search/scroll",Some(json!({"scroll_id":[scroll]}))).await;
}
