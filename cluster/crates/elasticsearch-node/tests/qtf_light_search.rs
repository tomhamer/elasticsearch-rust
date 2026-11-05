//! A query_then_fetch search whose shards answer light query hits and
//! addressed fetch hits (`qtf::LIGHT_HEADER`, the fetched hits kept as text
//! by the coordinator) answers byte for byte as one whose shards answer
//! full query hits and plain fetch hits (a node of an earlier build), but
//! `took`: from the node holding the shards and from the other, over shards
//! on both nodes.
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
async fn light_query_then_fetch_matches_full_hits() {
    let cl = start(&[]).await;
    let mappings = json!({"properties": {"name": {"type": "text", "fields": {"raw": {"type": "keyword"}}}, "n": {"type": "long"},
        "tag": {"type": "keyword"}, "price": {"type": "double"}, "obj": {"properties": {"k": {"type": "keyword"}}}}});
    for (index, shards, node) in [("a", 3, "n0"), ("b", 2, "n1")] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{index}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.routing.allocation.require._name": node},
            "mappings": mappings.clone()
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(status, 200, "{body}");
        let words = ["alpha beta", "beta gamma", "gamma delta \u{e9}t\u{e9}", "delta \"quoted\" epsilon", "alpha\\path/x", "zeta"];
        let mut bulk = String::new();
        for i in 0..240usize {
            let routing = if i % 5 == 0 { format!(",\"routing\":\"r{}\"", i % 3) } else { String::new() };
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{index}{i}\"{routing}}}}}\n"));
            // Sources as clients send them: key order, spacing and escapes kept.
            bulk.push_str(&format!("{{ \"name\": {}, \"n\": {}, \"tag\": \"t{}\", \"price\": {}.5, \"obj\": {{\"k\": \"\\u0041{}\"}}, \"extra\" : [1, 2.50, null] }}\n",
                serde_json::to_string(&format!("{} {}", words[i % words.len()], i % 13)).unwrap(), (i * 7919) % 97, i % 4, i % 17, i % 3));
        }
        let r: Value = cl.c.post(format!("http://127.0.0.1:{}/{index}/_bulk?refresh=true", cl.p.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(r["errors"], false, "{r}");
    }
    let raw = |node: usize, body: Value| {
        let cl = &cl;
        async move {
            let r = cl.c.post(format!("http://127.0.0.1:{}/a,b/_search", cl.p.http[node])).header("content-type", "application/json")
                .header("accept-encoding", "identity").body(body.to_string()).send().await.unwrap();
            let status = r.status().as_u16();
            let text = r.text().await.unwrap();
            assert_eq!(status, 200, "{body}: {text}");
            untimed(&text)
        }
    };
    let queries = [
        json!({"query": {"match_all": {}}, "size": 50}),
        json!({"query": {"match_all": {}}, "size": 1000}),
        json!({"query": {"match": {"name": "alpha gamma"}}, "from": 10, "size": 40}),
        json!({"query": {"match": {"name": "delta"}}, "size": 30, "sort": [{"n": "desc"}, {"tag": "asc"}], "track_scores": true, "version": true, "seq_no_primary_term": true}),
        json!({"query": {"match_all": {}}, "size": 25, "sort": [{"price": "asc"}, "_score"], "_source": {"includes": ["name", "obj.*"], "excludes": ["obj.k"]}}),
        json!({"query": {"match": {"name": "beta"}}, "size": 30, "highlight": {"fields": {"name": {}}}, "_source": ["n"]}),
        json!({"query": {"match_all": {}}, "size": 30, "docvalue_fields": ["n", "tag"], "fields": ["name.raw", {"field": "price"}]}),
        json!({"query": {"match_all": {}}, "size": 20, "script_fields": {"twice": {"script": {"source": "doc['n'].value * 2"}}}}),
        json!({"query": {"match": {"name": "zeta"}}, "size": 60, "post_filter": {"term": {"tag": "t1"}}, "aggs": {"t": {"terms": {"field": "tag"}}}, "track_total_hits": true}),
        json!({"query": {"match_all": {}}, "size": 40, "indices_boost": [{"b": 2.0}], "min_score": 0.5}),
        json!({"query": {"match_all": {}}, "size": 30, "runtime_mappings": {"half": {"type": "double", "script": {"source": "emit(doc['n'].value / 2.0)"}}}, "fields": ["half"]}),
        json!({"query": {"term": {"tag": "t2"}}, "size": 30, "sort": [{"n": "asc"}], "search_after": [40]}),
    ];
    for q in &queries {
        for node in [0, 1] {
            elasticsearch_node::front::QTF_LIGHT_DISABLED.store(true, Relaxed);
            let (before, _) = elasticsearch_node::front::qtf_counts();
            let want = raw(node, q.clone()).await;
            elasticsearch_node::front::QTF_LIGHT_DISABLED.store(false, Relaxed);
            let got = raw(node, q.clone()).await;
            assert!(elasticsearch_node::front::qtf_counts().0 >= before + 2, "{q}: ran as query_then_fetch");
            assert_eq!(got, want, "{q} from n{node}");
        }
    }
}
