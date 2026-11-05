#[allow(unused_imports)]
use super::{start, Es};
use reqwest::Method;
use serde_json::{json, Value};

async fn seed(es: &Es) {
    let (st, v) = es
        .put(
            "/airline-data",
            json!({"mappings": {"properties": {"time": {"type": "date"}, "airline": {"type": "keyword"}, "responsetime": {"type": "float"}}}}),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    let docs = [
        ("2017-02-18T00:00:00Z", "foo", 1.0),
        ("2017-02-18T00:30:00Z", "foo", 1.0),
        ("2017-02-18T01:00:00Z", "bar", 42.0),
        ("2017-02-18T01:01:00Z", "foo", 42.0),
    ];
    let mut lines = Vec::new();
    for (i, (t, a, r)) in docs.iter().enumerate() {
        lines.push(json!({"index": {"_index": "airline-data", "_id": (i + 1).to_string()}}));
        lines.push(json!({"time": t, "airline": a, "responsetime": r}));
    }
    let (st, v) = es.bulk("/_bulk?refresh=true", &lines).await;
    assert_eq!(st, 200, "{v}");
}

fn pivot_body(dest: &str) -> Value {
    json!({
        "source": {"index": "airline-data"},
        "dest": {"index": dest},
        "pivot": {
            "group_by": {"airline": {"terms": {"field": "airline"}}},
            "aggs": {"avg_response": {"avg": {"field": "responsetime"}}, "n": {"value_count": {"field": "responsetime"}}}
        }
    })
}

async fn wait_stopped(es: &Es, id: &str) -> Value {
    for _ in 0..100 {
        let (_, v) = es.get(&format!("/_transform/{id}/_stats")).await;
        if v["transforms"][0]["state"] == "stopped" && v["transforms"][0]["checkpointing"]["last"]["checkpoint"].as_u64().unwrap_or(0) >= 1 {
            return v;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("transform {id} did not finish");
}

#[tokio::test]
async fn transform_crud_and_validation() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    seed(&es).await;
    let (st, v) = es.put("/_transform/t1", pivot_body("dest1")).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.put("/_transform/t1", pivot_body("dest1")).await;
    assert_eq!(st, 409, "{v}");
    let (st, v) = es.get("/_transform/t1").await;
    assert_eq!(st, 200);
    assert_eq!(v["count"], 1);
    let t = &v["transforms"][0];
    assert_eq!(t["source"]["index"], json!(["airline-data"]));
    assert!(t["source"]["query"]["match_all"].is_object());
    assert_eq!(t["pivot"]["aggregations"]["avg_response"]["avg"]["field"], "responsetime");
    assert!(t["create_time"].is_i64());
    let (_, v) = es.get("/_transform/t1?exclude_generated=true").await;
    assert!(v["transforms"][0].get("create_time").is_none());
    // Validation errors.
    let mut b = pivot_body("d");
    b["frequency"] = json!("999ms");
    let (st, v) = es.put("/_transform/t2", b).await;
    assert_eq!(st, 400);
    assert!(v.to_string().contains("minimum permitted [frequency] is [1s]"), "{v}");
    let (st, v) = es.put("/_transform/BAD!", pivot_body("d")).await;
    assert_eq!(st, 400);
    assert!(v.to_string().contains("can contain lowercase alphanumeric"), "{v}");
    let mut b = pivot_body("airline-data-x");
    b["source"]["index"] = json!("airline-data*");
    let (st, v) = es.put("/_transform/t3", b).await;
    assert_eq!(st, 400);
    assert!(v.to_string().contains("is included in source expression"), "{v}");
    let mut b = pivot_body("d");
    b["pivot"]["aggs"] = json!({"airline": {"avg": {"field": "responsetime"}}});
    let (_, v) = es.put("/_transform/t4", b).await;
    assert!(v.to_string().contains("duplicate field [airline] detected"), "{v}");
    let (st, _) = es.get("/_transform/missing").await;
    assert_eq!(st, 404);
    let (st, v) = es.get("/_transform/_all?size=10000").await;
    assert_eq!(st, 400, "{v}");
    // Update.
    let (st, v) = es.post("/_transform/t1/_update", json!({"description": "d1", "_meta": {"a": 1}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["description"], "d1");
    assert_eq!(v["_meta"]["a"], 1);
    let (st, v) = es.post("/_transform/t1/_update", json!({"sync": {"time": {"field": "time"}}})).await;
    assert_eq!(st, 409, "{v}");
    let (_, v) = es.post_empty("/_transform/_upgrade").await;
    assert_eq!(v["no_action"], 1);
    let (st, _) = es.delete("/_transform/t1").await;
    assert_eq!(st, 200);
    let (st, _) = es.delete("/_transform/t1").await;
    assert_eq!(st, 404);
}

#[tokio::test]
async fn transform_preview_pivot_and_latest() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    seed(&es).await;
    let (st, v) = es
        .post(
            "/_transform/_preview",
            json!({
                "source": {"index": "airline-data"},
                "pivot": {
                    "group_by": {"airline": {"terms": {"field": "airline"}}, "by-hour": {"date_histogram": {"fixed_interval": "1h", "field": "time"}}},
                    "aggs": {"avg_response": {"avg": {"field": "responsetime"}}, "time.max": {"max": {"field": "time"}}}
                }
            }),
        )
        .await;
    assert_eq!(st, 200, "{v}");
    let p = &v["preview"];
    assert_eq!(p.as_array().unwrap().len(), 3);
    assert_eq!(p[0]["airline"], "foo");
    assert_eq!(p[0]["by-hour"], "2017-02-18T00:00:00.000Z");
    assert_eq!(p[0]["avg_response"], 1.0);
    assert_eq!(p[0]["time"]["max"], "2017-02-18T00:30:00.000Z");
    assert_eq!(p[1]["airline"], "bar");
    let props = &v["generated_dest_index"]["mappings"]["properties"];
    assert_eq!(props["airline"]["type"], "keyword");
    assert_eq!(props["by-hour"]["type"], "date");
    assert_eq!(props["avg_response"]["type"], "double");
    assert_eq!(props["time.max"]["type"], "date");
    // latest
    let (st, v) = es.post("/_transform/_preview", json!({"source": {"index": "airline-data"}, "latest": {"unique_key": ["airline"], "sort": "time"}})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["preview"][0]["airline"], "bar");
    assert_eq!(v["preview"][1]["time"], "2017-02-18T01:01:00Z");
    // unsupported agg
    let (st, v) = es
        .post(
            "/_transform/_preview",
            json!({"source": {"index": "airline-data"}, "pivot": {"group_by": {"a": {"terms": {"field": "airline"}}}, "aggs": {"x": {"significant_terms": {"field": "airline"}}}}}),
        )
        .await;
    assert_eq!(st, 400);
    assert!(v.to_string().contains("Unsupported aggregation type [significant_terms]"), "{v}");
}

#[tokio::test]
async fn transform_batch_run_writes_dest() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    seed(&es).await;
    let (st, v) = es.put("/_ingest/pipeline/p1", json!({"processors": [{"set": {"field": "tagged", "value": true}}]})).await;
    assert_eq!(st, 200, "{v}");
    let mut b = pivot_body("by-airline");
    b["dest"]["pipeline"] = json!("p1");
    let (st, v) = es.put("/_transform/batch", b).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post_empty("/_transform/batch/_start").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post_empty("/_transform/batch/_start").await;
    assert_eq!(st, 409, "{v}");
    let (_, m) = es.get("/by-airline/_mapping").await;
    assert_eq!(m["by-airline"]["mappings"]["properties"]["avg_response"]["type"], "double");
    let stats = wait_stopped(&es, "batch").await;
    let s = &stats["transforms"][0]["stats"];
    assert_eq!(s["documents_processed"], 4);
    assert_eq!(s["documents_indexed"], 2);
    let (_, r) = es.post("/by-airline/_search", json!({"sort": [{"airline": "asc"}]})).await;
    let hits = r["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["_source"]["airline"], "bar");
    assert_eq!(hits[0]["_source"]["avg_response"], 42.0);
    assert_eq!(hits[0]["_source"]["tagged"], true);
    assert_eq!(hits[1]["_source"]["n"], 3);
    // _cat/transforms
    let (st, t) = es.text(Method::GET, "/_cat/transforms?v=true&h=id,state,checkpoint,docp,dest_index", None).await;
    assert_eq!(st, 200);
    assert!(t.contains("batch") && t.contains("stopped") && t.contains("by-airline"), "{t}");
    // reset deletes the dest index the transform created
    let (st, v) = es.post_empty("/_transform/batch/_reset").await;
    assert_eq!(st, 200, "{v}");
    let (st, _) = es.call(Method::HEAD, "/by-airline", None).await;
    assert_eq!(st, 404);
}

#[tokio::test]
async fn transform_continuous_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let es = start(dir.path()).await;
    seed(&es).await;
    let mut b = pivot_body("cont-dest");
    b["sync"] = json!({"time": {"field": "time", "delay": "1s"}});
    b["frequency"] = json!("1s");
    let (st, v) = es.put("/_transform/cont", b).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post_empty("/_transform/cont/_start").await;
    assert_eq!(st, 200, "{v}");
    let (_, v) = es.get("/_transform/cont/_stats").await;
    let t = &v["transforms"][0];
    assert!(t["state"] == "started" || t["state"] == "indexing", "{t}");
    assert_eq!(t["checkpointing"]["last"]["checkpoint"], 1);
    assert!(t["node"]["id"].is_string());
    let (st, v) = es.delete("/_transform/cont").await;
    assert_eq!(st, 409, "{v}");
    // New data in a new bucket is picked up by a later checkpoint.
    let now = chrono::Utc::now().to_rfc3339();
    let (st, _) = es.put("/airline-data/_doc/9?refresh=true", json!({"time": now, "airline": "baz", "responsetime": 7.0})).await;
    assert_eq!(st, 201);
    let mut found = false;
    for _ in 0..80 {
        let _ = es.post_empty("/_transform/cont/_schedule_now").await;
        let (_, r) = es.post("/cont-dest/_search", json!({"query": {"term": {"airline": "baz"}}})).await;
        if r["hits"]["total"]["value"] == 1 {
            found = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(found, "continuous checkpoint did not index the new bucket");
    let (st, _) = es.post_empty("/_transform/cont/_stop?wait_for_completion=true").await;
    assert_eq!(st, 200);
    let (_, v) = es.get("/_transform/cont/_stats").await;
    assert_eq!(v["transforms"][0]["state"], "stopped");
    let (st, _) = es.delete("/_transform/cont").await;
    assert_eq!(st, 200);
}
