//! Searches of a multi-shard index of large documents (Rally pmc's shape)
//! run as query_then_fetch, fetching only the merged page's winners, rather
//! than in a single round where every shard renders its own full page: the
//! response is byte for byte the single round's (but `took`) over many hit
//! shapes, and no single round is counted.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::sync::atomic::Ordering::Relaxed;

fn untimed(text: &str) -> String {
    let Some(at) = text.find("\"took\":") else { return text.to_string() };
    let start = at + "\"took\":".len();
    let end = start + text[start..].find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
    format!("{}0{}", &text[..start], &text[end..])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn large_hits_fetch_only_the_winners_with_the_same_response() {
    let cl = start(&[]).await;
    let (status, body) = cl.call(0, Method::PUT, "/big", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
        "mappings": {"properties": {"title": {"type": "text"}, "body": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"},
            "meta": {"properties": {"lang": {"type": "keyword"}}}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(0, Method::GET, "/_cluster/health/big?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{body}");
    let words = ["physician", "newspaper", "coverage", "clinical", "study", "patients", "\"quoted\"", "naïve", "line\nbreak", "tab\t"];
    let mut bulk = String::new();
    for i in 0..80usize {
        let text: String = (0..3600).map(|w| words[(i * 31 + w * 7) % words.len()]).collect::<Vec<_>>().join(" ");
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"d{i}\"}}}}\n"));
        bulk.push_str(&json!({"title": format!("article {i}"), "body": text, "tag": format!("t{}", i % 5), "n": (i * 7919) % 97, "meta": {"lang": (["en", "de"][i % 2])}}).to_string());
        bulk.push('\n');
    }
    let r: Value = cl.c.post(format!("http://127.0.0.1:{}/big/_bulk?refresh=true", cl.p.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["errors"], false, "{r}");
    let search = |body: Value| {
        let cl = &cl;
        async move {
            let r = cl.c.post(format!("http://127.0.0.1:{}/big/_search", cl.p.http[0])).header("content-type", "application/json")
                .body(body.to_string()).send().await.unwrap();
            let status = r.status().as_u16();
            let text = r.text().await.unwrap();
            assert_eq!(status, 200, "{body}: {text}");
            untimed(&text)
        }
    };
    let shapes = [
        json!({"query": {"match_all": {}}}),
        json!({"query": {"term": {"body": "physician"}}}),
        json!({"query": {"match_phrase": {"body": "newspaper coverage"}}}),
        json!({"query": {"match": {"body": "clinical study"}}, "size": 20}),
        json!({"query": {"match": {"body": "patients"}}, "from": 5, "size": 5}),
        json!({"query": {"match_all": {}}, "_source": {"includes": ["title", "meta.*"]}}),
        json!({"query": {"match_all": {}}, "_source": {"excludes": ["body"]}}),
        json!({"query": {"match_all": {}}, "_source": ["title", "n"]}),
        json!({"query": {"match_all": {}}, "fields": ["tag", "meta.lang"], "docvalue_fields": ["n"]}),
        json!({"query": {"match": {"body": "coverage"}}, "highlight": {"fields": {"body": {}}}}),
        json!({"query": {"match_all": {}}, "sort": [{"n": "desc"}, {"tag": "asc"}]}),
        json!({"query": {"match": {"body": "naïve"}}, "sort": [{"n": "asc"}], "track_scores": true}),
        json!({"query": {"match": {"body": "physician"}}, "explain": true}),
        json!({"query": {"bool": {"should": [{"term": {"tag": {"value": "t1", "_name": "one"}}}, {"term": {"tag": {"value": "t2", "_name": "two"}}}]}}}),
        json!({"query": {"match_all": {}}, "stored_fields": ["_none_"]}),
        json!({"query": {"match_all": {}}, "version": true, "seq_no_primary_term": true}),
    ];
    for shape in &shapes {
        elasticsearch_node::front::LARGE_HITS_SINGLE_ROUND.store(true, Relaxed);
        let single = search(shape.clone()).await;
        elasticsearch_node::front::LARGE_HITS_SINGLE_ROUND.store(false, Relaxed);
        let rounds = elasticsearch_node::front::qtf_single_rounds();
        let fetched = search(shape.clone()).await;
        assert_eq!(fetched, single, "{shape}");
        let named = shape.to_string().contains("\"_name\"") && shape.get("sort").is_some();
        if !named {
            assert_eq!(elasticsearch_node::front::qtf_single_rounds(), rounds, "{shape}: large hits answered in a single round");
        }
    }
    // Small documents keep the single round. (A page without `_source` is
    // a light fetch, which never enters query_then_fetch at all: it is
    // answered in one plain fan-out, not counted as a single round.)
    let (status, body) = cl.call(0, Method::PUT, "/small", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
        "mappings": {"properties": {"title": {"type": "text"}, "n": {"type": "long"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(0, Method::GET, "/_cluster/health/small?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{body}");
    let mut small = String::new();
    for i in 0..40 {
        small.push_str(&format!("{{\"index\":{{\"_id\":\"s{i}\"}}}}\n"));
        small.push_str(&json!({"title": format!("note {i}"), "n": i}).to_string());
        small.push('\n');
    }
    let r: Value = cl.c.post(format!("http://127.0.0.1:{}/small/_bulk?refresh=true", cl.p.http[0]))
        .header("content-type", "application/x-ndjson").body(small).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["errors"], false, "{r}");
    let rounds = elasticsearch_node::front::qtf_single_rounds();
    let r = cl.c.post(format!("http://127.0.0.1:{}/small/_search", cl.p.http[0])).header("content-type", "application/json")
        .body(json!({"query": {"match_all": {}}}).to_string()).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(elasticsearch_node::front::qtf_single_rounds(), rounds + 1, "a page of small documents left the single round");
}
