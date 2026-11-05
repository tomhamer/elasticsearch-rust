//! Small searches of a multi-shard index whose every shard is on the
//! coordinating node answer in a single round (ES `query_and_fetch`): the
//! response is byte for byte the query_then_fetch one (but `took`), and the
//! search stats count the same queries and fetches.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
use std::sync::atomic::Ordering::Relaxed;

/// The response text with `took` zeroed.
fn untimed(text: &str) -> String {
    let Some(at) = text.find("\"took\":") else { return text.to_string() };
    let start = at + "\"took\":".len();
    let end = start + text[start..].find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
    format!("{}0{}", &text[..start], &text[end..])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_round_matches_query_then_fetch() {
    let cl = start(&[]).await;
    // Geonames-like: five shards, no replicas, all on n0.
    let (status, body) = cl.call(0, Method::PUT, "/geo", Some(json!({
        "settings": {"number_of_shards": 5, "number_of_replicas": 0, "index.routing.allocation.require._name": "n0"},
        "mappings": {"properties": {"name": {"type": "text"}, "country_code": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "population": {"type": "long"}, "tag": {"type": "keyword"}}}
    }))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(0, Method::GET, "/_cluster/health/geo?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(status, 200, "{body}");
    let names = ["Sankt Georgen im Attergau", "Sankt Peter", "Georgen", "Wien", "Sankt Georgen an der Gusen", "Graz"];
    let countries = ["AT", "DE", "AT", "US", "AT", "CH"];
    let mut bulk = String::new();
    for i in 0..300 {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n"));
        bulk.push_str(&json!({"name": format!("{} {}", names[i % names.len()], i % 11), "country_code": countries[(i / 3) % countries.len()],
            "population": (i * 7919) % 1000, "tag": format!("t{}", i % 4)}).to_string());
        bulk.push('\n');
    }
    let r: Value = cl.c.post(format!("http://127.0.0.1:{}/geo/_bulk?refresh=true", cl.p.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["errors"], false, "{r}");

    let raw = |node: usize, body: Value| {
        let cl = &cl;
        async move {
            let r = cl.c.post(format!("http://127.0.0.1:{}/geo/_search", cl.p.http[node])).header("content-type", "application/json")
                .body(body.to_string()).send().await.unwrap();
            let status = r.status().as_u16();
            let text = r.text().await.unwrap();
            assert_eq!(status, 200, "{body}: {text}");
            untimed(&text)
        }
    };
    let search_stats = || {
        let cl = &cl;
        async move {
            let (status, s) = cl.call(0, Method::GET, "/geo/_stats/search", None).await;
            assert_eq!(status, 200, "{s}");
            let p = &s["_all"]["primaries"]["search"];
            (p["query_total"].as_u64().unwrap_or(0), p["fetch_total"].as_u64().unwrap_or(0))
        }
    };
    let queries = [
        // Rally geonames `default`, `term`, `phrase`.
        json!({"query": {"match_all": {}}}),
        json!({"query": {"term": {"country_code.raw": "AT"}}}),
        json!({"query": {"match_phrase": {"name": "Sankt Georgen"}}}),
        // Scored hits spread over the shards, a later page.
        json!({"query": {"match": {"name": "sankt georgen 3"}}, "from": 4, "size": 12}),
        json!({"query": {"bool": {"should": [{"match": {"name": {"query": "graz", "_name": "g"}}}, {"match": {"name": {"query": "wien", "_name": "w"}}}]}}, "size": 15}),
        // Sorted, with scores, filtered sources and metadata.
        json!({"sort": [{"population": "desc"}, {"tag": "asc"}], "size": 8, "track_scores": true, "_source": ["name", "population"], "version": true, "seq_no_primary_term": true}),
        // (`_source: false` alone is single-phase already: `qtf::eligible`.)
        json!({"sort": [{"population": "asc"}], "search_after": [100], "size": 6, "_source": ["population"]}),
        // Aggregations, totals, post filter, min score.
        json!({"size": 5, "track_total_hits": true, "query": {"match": {"name": "sankt"}}, "aggs": {"c": {"terms": {"field": "country_code.raw"}}, "p": {"avg": {"field": "population"}}}}),
        json!({"size": 10, "track_total_hits": 50, "post_filter": {"term": {"tag": "t1"}}, "query": {"match": {"name": "georgen"}}, "min_score": 0.1}),
        json!({"size": 4, "query": {"match_none": {}}}),
    ];
    let before_rounds = elasticsearch_node::front::qtf_single_rounds();
    for q in &queries {
        // Two phases, from this node and from the node without shards.
        elasticsearch_node::front::QTF_SINGLE_ROUND_DISABLED.store(true, Relaxed);
        let (two_phase, _) = elasticsearch_node::front::qtf_counts();
        let rounds = elasticsearch_node::front::qtf_single_rounds();
        let s0 = search_stats().await;
        let want = raw(0, q.clone()).await;
        let s1 = search_stats().await;
        assert!(elasticsearch_node::front::qtf_counts().0 > two_phase, "{q}: ran as query_then_fetch");
        assert_eq!(elasticsearch_node::front::qtf_single_rounds(), rounds);
        let remote = raw(1, q.clone()).await;
        elasticsearch_node::front::QTF_SINGLE_ROUND_DISABLED.store(false, Relaxed);
        // One round.
        let s2 = search_stats().await;
        let got = raw(0, q.clone()).await;
        let s3 = search_stats().await;
        assert!(elasticsearch_node::front::qtf_single_rounds() > rounds, "{q}: ran in one round");
        assert_eq!(got, want, "{q}: one round vs query_then_fetch");
        assert_eq!(remote, want, "{q}: coordinated from the other node");
        // The same queries and fetches counted (ES counts a fetch on each
        // shard whose hits made the page).
        assert_eq!((s3.0 - s2.0, s3.1 - s2.1), (s1.0 - s0.0, s1.1 - s0.1), "{q}: search stats");
        // From the node without shards it stays two-phase.
        let (two_phase, _) = elasticsearch_node::front::qtf_counts();
        let again = raw(1, q.clone()).await;
        assert_eq!(again, want, "{q}");
        assert!(elasticsearch_node::front::qtf_counts().0 > two_phase, "{q}: remote shards keep two phases");
    }
    assert!(elasticsearch_node::front::qtf_single_rounds() >= before_rounds + queries.len() as u64);
    // Over the single-round window, or with highlighting: two phases.
    for q in [json!({"query": {"match_all": {}}, "size": 21}), json!({"query": {"match": {"name": "graz"}}, "highlight": {"fields": {"name": {}}}})] {
        let rounds = elasticsearch_node::front::qtf_single_rounds();
        let (two_phase, _) = elasticsearch_node::front::qtf_counts();
        raw(0, q.clone()).await;
        assert_eq!(elasticsearch_node::front::qtf_single_rounds(), rounds, "{q}");
        assert!(elasticsearch_node::front::qtf_counts().0 > two_phase, "{q}");
    }
}
