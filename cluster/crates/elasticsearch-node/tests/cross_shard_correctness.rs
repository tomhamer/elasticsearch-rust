//! Real two-node fanout: aggregations and search APIs on a four-shard index
//! agree with the same data in one shard.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};
const DOCS: usize = 30_000;

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_shard_results_match_single_shard() {
    let cl = start(&[]).await;
    let ports = cl.p;
    for (name, shards) in [("whole", 1), ("split", 4)] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": 0},
            "mappings": {"properties": {"k": {"type": "keyword"}, "g": {"type": "keyword"}, "r": {"type": "keyword"}, "n": {"type": "long"},
                "d": {"type": "date"}, "loc": {"type": "geo_point"}, "body": {"type": "text"}, "msg": {"type": "text"}, "c": {"type": "completion"}}}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(status, 200, "{body}");
        for chunk in 0..DOCS / 5000 {
            let mut bulk = String::new();
            for i in chunk * 5000..(chunk + 1) * 5000 {
                // Points either side of the antimeridian.
                let lon = if i % 2 == 0 { 170.0 + (i % 50) as f64 / 10.0 } else { -170.0 - (i % 50) as f64 / 10.0 };
                bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n"));
                // Completion inputs go through the `simple` analyzer, which keeps
                // letters only: the number is spelled in letters (12 -> "bc").
                let spelled: String = i.to_string().bytes().map(|d| (b'a' + d - b'0') as char).collect();
                bulk.push_str(&json!({"c": format!("sug{spelled}"), "k": format!("k{i}"), "g": format!("g{}", i % 3), "r": format!("r{}", i % 10000), "n": i, "d": 1_600_000_000_000i64 + i as i64 * 1000,
                    "loc": {"lat": (i % 80) as f64 - 40.0, "lon": lon}, "body": if i % 10 == 0 { "fresh apple" } else { "plain" },
                    "msg": match i % 3 { 0 => format!("user {i} logged in from host"), 1 => format!("disk {i} failed on storage node"), _ => format!("request {i} took too long to answer") }}).to_string());
                bulk.push('\n');
            }
            let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk", ports.http[0]))
                .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
            assert_eq!(body["errors"], false, "{body}");
        }
        let (status, body) = cl.call(0, Method::POST, &format!("/{name}/_refresh"), None).await;
        assert_eq!(status, 200, "{body}");
    }
    let search = |index: &'static str, node: usize, q: Value| {
        let cl = &cl;
        async move {
            let (status, got) = cl.call(node, Method::POST, &format!("/{index}/_search"), Some(q)).await;
            assert_eq!(status, 200, "{got}");
            assert_eq!(got["_shards"]["failed"], 0, "{got}");
            assert!(!got.to_string().contains("__p"), "{got}");
            got
        }
    };

    // Sketch aggregations merge shard state: no 10k-term helper, no
    // max_buckets trip, at the root and inside buckets.
    let q = json!({"size": 0, "aggs": {
        "card": {"cardinality": {"field": "k"}},
        "pct": {"percentiles": {"field": "n", "percents": [1, 50, 99]}},
        "pct_date": {"percentiles": {"field": "d", "percents": [50]}},
        "hdr": {"percentiles": {"field": "n", "percents": [50], "hdr": {"number_of_significant_value_digits": 3}}},
        "ranks": {"percentile_ranks": {"field": "n", "values": [15000]}},
        "bounds": {"geo_bounds": {"field": "loc"}},
        "per_g": {"terms": {"field": "g"}, "aggs": {"card": {"cardinality": {"field": "k"}}, "pct": {"percentiles": {"field": "n", "percents": [50]}}}}
    }});
    let want = search("whole", 0, q.clone()).await;
    for node in 0..2 {
        let got = search("split", node, json!({"size": 0, "aggs": q["aggs"].clone()})).await;
        let (a, w) = (&got["aggregations"], &want["aggregations"]);
        assert!(close(a["card"]["value"].as_f64().unwrap(), DOCS as f64, 0.05), "{a}");
        for p in ["1.0", "50.0", "99.0"] {
            assert!(close(a["pct"]["values"][p].as_f64().unwrap(), w["pct"]["values"][p].as_f64().unwrap(), 0.02), "{p}: {a} vs {w}");
        }
        assert!(a["pct_date"]["values"]["50.0_as_string"].is_string(), "{a}");
        assert!(close(a["hdr"]["values"]["50.0"].as_f64().unwrap(), w["hdr"]["values"]["50.0"].as_f64().unwrap(), 0.01), "{a}");
        assert!(close(a["ranks"]["values"]["15000.0"].as_f64().unwrap(), 50.0, 0.02), "{a}");
        assert_eq!(a["bounds"], w["bounds"], "the box crosses the antimeridian");
        let buckets = a["per_g"]["buckets"].as_array().unwrap();
        assert_eq!(buckets.len(), 3);
        for b in buckets {
            assert!(close(b["card"]["value"].as_f64().unwrap(), (DOCS / 3) as f64, 0.05), "{b}");
        }
    }

    // Request errors stay the shards' own (they skip rendering sketches).
    for bad in [json!({"percentiles": {"field": "n", "percents": [150]}}), json!({"percentiles": {"field": "n", "hdr": {"number_of_significant_value_digits": 6}}}),
                json!({"percentile_ranks": {"field": "n"}}), json!({"boxplot": {"field": "n", "compression": -1}}), json!({"median_absolute_deviation": {"field": "n", "execution_hint": "x"}})] {
        let q = json!({"size": 0, "aggs": {"m": bad}});
        let (sw, w) = cl.call(0, Method::POST, "/whole/_search", Some(q.clone())).await;
        let (sg, g) = cl.call(1, Method::POST, "/split/_search", Some(q)).await;
        assert!(sw >= 400 && sg >= 400, "{w} / {g}");
        assert_eq!(w["error"]["root_cause"][0]["reason"], g["error"]["root_cause"][0]["reason"], "{w} / {g}");
    }

    // Search templates and rank_eval search every shard.
    let tpl = json!({"source": {"query": {"match": {"body": "{{word}}"}}, "size": 0, "track_total_hits": true}, "params": {"word": "fresh"}});
    for node in 0..2 {
        let (status, got) = cl.call(node, Method::POST, "/split/_search/template", Some(tpl.clone())).await;
        assert_eq!(status, 200, "{got}");
        assert_eq!(got["hits"]["total"]["value"], DOCS / 10, "{got}");
        let mut lines = String::new();
        for (index, word) in [("split", "fresh"), ("whole", "plain"), ("split", "{{")] {
            lines.push_str(&json!({"index": index}).to_string());
            lines.push('\n');
            lines.push_str(&json!({"source": format!("{{\"query\":{{\"match\":{{\"body\":\"{word}\"}}}},\"track_total_hits\":true}}"), "params": {}}).to_string());
            lines.push('\n');
        }
        let got: Value = cl.c.post(format!("http://127.0.0.1:{}/_msearch/template", ports.http[node]))
            .header("content-type", "application/x-ndjson").body(lines).send().await.unwrap().json().await.unwrap();
        let r = got["responses"].as_array().unwrap();
        assert_eq!(r.len(), 3, "{got}");
        assert_eq!(r[0]["hits"]["total"]["value"], DOCS / 10, "{got}");
        assert_eq!(r[0]["status"], 200, "{got}");
        assert_eq!(r[1]["hits"]["total"]["value"], DOCS - DOCS / 10, "{got}");
        assert!(r[2]["error"].is_object(), "{got}");
        let rank = json!({"requests": [{"id": "q", "request": {"query": {"match": {"body": "fresh"}}},
            "ratings": [{"_index": "split", "_id": "0", "rating": 1}, {"_index": "split", "_id": "10", "rating": 1}, {"_index": "split", "_id": "15", "rating": 1}]}],
            "metric": {"recall": {"k": 5000, "relevant_rating_threshold": 1}}});
        let (status, got) = cl.call(node, Method::POST, "/split/_rank_eval", Some(rank)).await;
        assert_eq!(status, 200, "{got}");
        // Docs 0 and 10 match on whichever shard holds them; 15 does not.
        assert!(close(got["metric_score"].as_f64().unwrap(), 2.0 / 3.0, 0.001), "{got}");
    }

    // A point in time over the shards merges the same way, and refuses what
    // it cannot merge.
    let (status, pit) = cl.call(0, Method::POST, "/split/_pit?keep_alive=1m", None).await;
    assert_eq!(status, 200, "{pit}");
    let id = pit["id"].clone();
    let (status, got) = cl.call(1, Method::POST, "/_search", Some(json!({"size": 0, "pit": {"id": id}, "aggs": {"card": {"cardinality": {"field": "k"}}}}))).await;
    assert_eq!(status, 200, "{got}");
    assert!(close(got["aggregations"]["card"]["value"].as_f64().unwrap(), DOCS as f64, 0.05), "{got}");
    let (status, got) = cl.call(1, Method::POST, "/_search", Some(json!({"size": 0, "pit": {"id": id}, "aggs": {"m": {"boxplot": {"field": "n"}}}}))).await;
    assert!(status == 200 || status == 500, "{got}");
    if status == 200 {
        assert!(close(got["aggregations"]["m"]["q2"].as_f64().unwrap(), (DOCS / 2) as f64, 0.02), "{got}");
    }
    // Metrics merged from shard moments / digests agree with one shard.
    let q = json!({"size": 0, "aggs": {
        "box": {"boxplot": {"field": "n"}},
        "mad": {"median_absolute_deviation": {"field": "n"}},
        "ms": {"matrix_stats": {"fields": ["n", "d"]}},
        "tt": {"t_test": {"a": {"field": "n", "filter": {"term": {"g": "g0"}}}, "b": {"field": "n", "filter": {"term": {"g": "g1"}}}, "type": "heteroscedastic"}},
        "rare": {"rare_terms": {"field": "r"}},
        "rare_k": {"rare_terms": {"field": "k"}},
        "mt": {"multi_terms": {"terms": [{"field": "g"}, {"field": "r"}], "size": 1}},
        "comp": {"composite": {"size": 3, "sources": [{"g": {"terms": {"field": "g", "order": "desc", "missing_bucket": true}}}]}},
        "h": {"histogram": {"field": "n", "interval": 10000}, "aggs": {
            "c": {"cardinality": {"field": "k"}}, "cc": {"cumulative_cardinality": {"buckets_path": "c"}},
            "share": {"normalize": {"buckets_path": "_count", "method": "percent_of_sum"}},
            "sd": {"serial_diff": {"buckets_path": "_count"}},
            "mf": {"moving_fn": {"buckets_path": "_count", "window": 2, "script": "MovingFunctions.unweightedAvg(values)"}}}},
        "line": {"geo_line": {"point": {"field": "loc"}, "sort": {"field": "n"}, "size": 5}}
    }});
    let want = search("whole", 0, q.clone()).await;
    let got = search("split", 1, q.clone()).await;
    let (a, w) = (&got["aggregations"], &want["aggregations"]);
    for k in ["min", "max", "q1", "q2", "q3", "lower", "upper"] {
        assert!((a["box"][k].as_f64().unwrap() - w["box"][k].as_f64().unwrap()).abs() <= 0.01 * DOCS as f64, "{k}: {a} vs {w}");
    }
    assert!(close(a["mad"]["value"].as_f64().unwrap(), w["mad"]["value"].as_f64().unwrap(), 0.02), "{a}");
    assert_eq!(a["ms"]["doc_count"], DOCS);
    for (fa, fw) in a["ms"]["fields"].as_array().unwrap().iter().zip(w["ms"]["fields"].as_array().unwrap()) {
        assert_eq!(fa["name"], fw["name"]);
        for k in ["mean", "variance", "skewness", "kurtosis"] {
            assert!(close(fa[k].as_f64().unwrap(), fw[k].as_f64().unwrap(), 1e-6), "{k}: {fa} vs {fw}");
        }
        assert!(close(fa["correlation"]["d"].as_f64().unwrap(), fw["correlation"]["d"].as_f64().unwrap(), 1e-6));
    }
    assert!(close(a["tt"]["value"].as_f64().unwrap(), w["tt"]["value"].as_f64().unwrap(), 1e-6), "{a}");
    // Every r value occurs 3 times: never rare overall, though often once on
    // a shard; where another shard holds it twice only its filter drops it.
    assert_eq!(w["rare"]["buckets"], json!([]));
    assert_eq!(a["rare"]["buckets"], json!([]), "{}", a["rare"]);
    assert_eq!(a["rare_k"]["buckets"].as_array().unwrap().len(), w["rare_k"]["buckets"].as_array().unwrap().len().min(DOCS));
    assert_eq!(a["mt"]["buckets"][0]["doc_count"], w["mt"]["buckets"][0]["doc_count"]);
    assert_eq!(a["comp"]["buckets"], w["comp"]["buckets"]);
    assert_eq!(a["h"]["buckets"].as_array().unwrap().len(), 3);
    assert!(a["h"]["buckets"][2]["sd"]["value"].is_number() && a["h"]["buckets"][2]["mf"]["value"].is_number(), "{a}");
    for (b, wb) in a["h"]["buckets"].as_array().unwrap().iter().zip(w["h"]["buckets"].as_array().unwrap()) {
        assert_eq!(b["cc"], wb["cc"], "{b} vs {wb}");
        assert!(close(b["share"]["value"].as_f64().unwrap(), wb["share"]["value"].as_f64().unwrap(), 1e-9));
        assert_eq!((&b["sd"], &b["mf"]), (&wb["sd"], &wb["mf"]), "{b} vs {wb}");
    }
    assert_eq!(a["line"], w["line"]);

    // Shard categories are re-categorized into the same categories.
    let q = json!({"size": 0, "aggs": {"cat": {"categorize_text": {"field": "msg"}, "aggs": {"c": {"cardinality": {"field": "k"}}}}}});
    let want = search("whole", 0, q.clone()).await;
    let got = search("split", 1, q).await;
    assert_eq!(want["aggregations"]["cat"]["buckets"].as_array().unwrap().len(), 3, "{want}");
    assert_eq!(got["aggregations"], want["aggregations"]);

    // Few terms: significant_terms scores on exact counts, as one shard does.
    let q = json!({"size": 0, "query": {"match": {"body": "fresh"}}, "aggs": {
        "sig": {"significant_terms": {"field": "r", "size": 5}},
        "sig_g": {"significant_terms": {"field": "g", "min_doc_count": 1, "chi_square": {}}}}});
    let want = search("whole", 0, q.clone()).await;
    let got = search("split", 1, q).await;
    assert_eq!(got["aggregations"], want["aggregations"]);

    // The whole reduce counts buckets against search.max_buckets as one shard does.
    let (status, _) = cl.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"search.max_buckets": 100}}))).await;
    assert_eq!(status, 200);
    let q = json!({"size": 0, "aggs": {"h": {"histogram": {"field": "n", "interval": 100}}}});
    for index in ["whole", "split"] {
        let (status, body) = cl.call(1, Method::POST, &format!("/{index}/_search"), Some(q.clone())).await;
        assert!(status >= 400 && body.to_string().contains("too_many_buckets"), "{index}: {status} {body}");
    }
    let (status, _) = cl.call(0, Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"search.max_buckets": null}}))).await;
    assert_eq!(status, 200);

    // query_then_fetch: the fetch phase renders sources, highlights and
    // fields of just the merged page, as one shard does.
    let q = json!({"query": {"match": {"msg": "disk"}}, "sort": [{"n": "desc"}], "from": 3, "size": 12, "track_scores": true,
        "_source": ["k", "msg"], "highlight": {"fields": {"msg": {}}}, "docvalue_fields": ["n"]});
    let want = search("whole", 0, q.clone()).await;
    let (two_phase, fallbacks) = elasticsearch_node::front::qtf_counts();
    for node in [0, 1] {
        let got = search("split", node, q.clone()).await;
        let hits = got["hits"]["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 12, "{got}");
        assert_eq!(got["hits"]["total"], want["hits"]["total"]);
        for (h, w) in hits.iter().zip(want["hits"]["hits"].as_array().unwrap()) {
            for k in ["_id", "sort", "_source", "highlight", "fields"] {
                assert_eq!(h[k], w[k], "{k}: {h} vs {w}");
            }
            assert!(h["_score"].as_f64().is_some(), "{h}");
        }
    }
    // Both searches ran as query_then_fetch, none fell back.
    let (two_phase_after, fallbacks_after) = elasticsearch_node::front::qtf_counts();
    assert!(two_phase_after >= two_phase + 2, "{two_phase} -> {two_phase_after}");
    assert_eq!(fallbacks_after, fallbacks);
    // A completion suggester's options carry their documents' sources: the
    // search stays single-phase and the sources arrive across nodes.
    let mut sq = q.clone();
    sq["suggest"] = json!({"s": {"prefix": "sugbc", "completion": {"field": "c", "size": 5}}});
    let got = search("split", 1, sq).await;
    let options = got["suggest"]["s"][0]["options"].as_array().unwrap();
    assert_eq!(options.len(), 5, "{got}");
    for o in options {
        assert!(o["_source"]["k"].as_str().is_some_and(|k| k.starts_with("k12")), "{o}");
    }
    assert_eq!(elasticsearch_node::front::qtf_counts(), (two_phase_after, fallbacks_after));
    // `top_hits` sources come with the query phase: a lone surrogate and the
    // original number text pass through it as through the fetch phase.
    let (status, body) = cl.call(0, Method::PUT, "/rawtop", Some(json!({"settings": {"number_of_shards": 4, "number_of_replicas": 0},
        "mappings": {"properties": {"n": {"type": "long"}, "p": {"type": "float"}, "t": {"type": "text"}}}}))).await;
    assert_eq!(status, 200, "{body}");
    let mut bulk = String::new();
    for i in 0..40 {
        bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{{\"n\":{i},\"p\":1.50,\"t\":\"x\\ud800\"}}\n"));
    }
    let r: Value = cl.c.post(format!("http://127.0.0.1:{}/rawtop/_bulk?refresh=true", ports.http[0]))
        .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
    assert_eq!(r["errors"], false, "{r}");
    let (two_phase, fallbacks) = elasticsearch_node::front::qtf_counts();
    for node in 0..2 {
        let r = cl.c.post(format!("http://127.0.0.1:{}/rawtop/_search", ports.http[node])).header("content-type", "application/json")
            .body(r#"{"size":3,"sort":[{"n":"asc"}],"aggs":{"top":{"top_hits":{"size":5,"sort":[{"n":"desc"}]}}}}"#).send().await.unwrap();
        let status = r.status();
        let text = r.text().await.unwrap();
        assert_eq!(status, 200, "{text}");
        // 3 hits and 5 top hits, each verbatim.
        assert_eq!(text.matches(r#""p":1.50,"t":"x\ud800""#).count(), 8, "{text}");
        assert!(text.contains(r#""_id":"39""#), "{text}");
    }
    let (two_phase_after, fallbacks_after) = elasticsearch_node::front::qtf_counts();
    assert!(two_phase_after >= two_phase + 2, "{two_phase} -> {two_phase_after}");
    assert_eq!(fallbacks_after, fallbacks);

    // A shard failing on every copy (a script error on the index without
    // the field) is a partial failure of a two-phase search, as in ES; with
    // allow_partial_search_results=false the search fails.
    for (name, node, mapping) in [("pa", "n0", json!({"properties": {"f": {"type": "long"}}})), ("pb", "n1", json!({"properties": {"g": {"type": "long"}}}))] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": node}, "mappings": mapping}))).await;
        assert_eq!(status, 200, "{body}");
        let (status, body) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(status, 200, "{body}");
        for i in 0..5 {
            // Each index gets only its own field: a dynamic `f` on pb would
            // make the script succeed there too.
            let doc = if name == "pa" { json!({"f": i}) } else { json!({"g": i}) };
            let (status, body) = cl.call(0, Method::PUT, &format!("/{name}/_doc/{i}?refresh=true"), Some(doc)).await;
            assert!(status == 200 || status == 201, "{body}");
        }
    }
    let q = json!({"size": 3, "query": {"script_score": {"query": {"match_all": {}}, "script": {"source": "doc['f'].value"}}}});
    let (two_phase, _) = elasticsearch_node::front::qtf_counts();
    let (status, body) = cl.call(1, Method::POST, "/pa,pb/_search", Some(q.clone())).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["_shards"]["total"], 2, "{body}");
    assert_eq!(body["_shards"]["failed"], 1, "{body}");
    assert_eq!(body["_shards"]["failures"][0]["index"], "pb", "{body}");
    assert_eq!(body["_shards"]["failures"][0]["shard"], 0, "{body}");
    assert_eq!(body["hits"]["hits"].as_array().unwrap().len(), 3, "{body}");
    assert!(elasticsearch_node::front::qtf_counts().0 > two_phase);
    let (status, body) = cl.call(1, Method::POST, "/pa,pb/_search?allow_partial_search_results=false", Some(q)).await;
    assert!(status >= 400, "{body}");
    // Fetch-only limits are the query phase's (ES checks them when the
    // shard request creates its context): over the limit on every index,
    // the search fails; on one index, that shard fails, not a silent 200
    // missing its hits.
    let (status, body) = cl.call(0, Method::PUT, "/pb/_settings", Some(json!({"index": {"max_docvalue_fields_search": 1}}))).await;
    assert_eq!(status, 200, "{body}");
    let dv = json!({"size": 10, "query": {"match_all": {}}, "docvalue_fields": ["f", "g"]});
    let (status, body) = cl.call(1, Method::POST, "/pa,pb/_search", Some(dv.clone())).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["_shards"]["failed"], 1, "{body}");
    assert_eq!(body["_shards"]["failures"][0]["index"], "pb", "{body}");
    assert!(body.to_string().contains("too many docvalue_fields"), "{body}");
    assert_eq!(body["hits"]["hits"].as_array().unwrap().len(), 5, "{body}");
    let (status, body) = cl.call(0, Method::PUT, "/pa/_settings", Some(json!({"index": {"max_docvalue_fields_search": 1}}))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(1, Method::POST, "/pa,pb/_search", Some(dv)).await;
    assert_eq!(status, 400, "{body}");
    assert!(body.to_string().contains("all shards failed") && body.to_string().contains("too many docvalue_fields"), "{body}");

    // A stored script in script_fields under query_then_fetch: the query
    // phase checks it in the search's script environment (where the stored
    // script resolves), the fetch phase runs it.
    let (status, body) = cl.call(0, Method::PUT, "/_scripts/dbl", Some(json!({"script": {"lang": "painless", "source": "doc['n'].value * 2"}}))).await;
    assert_eq!(status, 200, "{body}");
    let (two_phase, _) = elasticsearch_node::front::qtf_counts();
    let got = search("split", 1, json!({"size": 3, "sort": [{"n": "asc"}], "script_fields": {"dbl": {"script": {"id": "dbl"}}}})).await;
    let hits = got["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3, "{got}");
    for (i, h) in hits.iter().enumerate() {
        assert_eq!(h["fields"]["dbl"][0].as_i64(), Some(2 * i as i64), "{h}");
    }
    assert!(elasticsearch_node::front::qtf_counts().0 > two_phase);

    // Queries that read another document fetch it through the cluster.
    let (status, body) = cl.call(0, Method::POST, "/split/_search", Some(json!({"size": 0, "track_total_hits": true,
        "query": {"more_like_this": {"fields": ["body"], "like": [{"_index": "split", "_id": "10"}], "min_term_freq": 1, "min_doc_freq": 1}}}))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["hits"]["total"]["value"], DOCS / 10 - 1, "{body}");
    // The liked document is excluded when the item names an alias too.
    let (status, body) = cl.call(0, Method::POST, "/_aliases", Some(json!({"actions": [{"add": {"index": "split", "alias": "split_alias"}}]}))).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cl.call(1, Method::POST, "/split/_search", Some(json!({"size": 0, "track_total_hits": true,
        "query": {"more_like_this": {"fields": ["body"], "like": [{"_index": "split_alias", "_id": "10"}], "min_term_freq": 1, "min_doc_freq": 1}}}))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["hits"]["total"]["value"], DOCS / 10 - 1, "{body}");
}
