//! Differential check of the coordinator's aggregation reduce: the same
//! documents in a one-shard index and in a four-shard index spread over two
//! nodes give the same aggregations, across value multiplicities (absent,
//! one, several values per doc) and nesting.
mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

const DOCS: usize = 600;

/// Equal JSON, numbers within a relative `1e-9` (summation order differs).
fn same(a: &Value, b: &Value, path: &str) -> Result<(), String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0) { Ok(()) } else { Err(format!("{path}: {x} vs {y}")) }
        }
        (Value::Object(x), Value::Object(y)) => {
            let (kx, ky): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (x.keys().collect(), y.keys().collect());
            if kx != ky {
                return Err(format!("{path}: keys {kx:?} vs {ky:?}"));
            }
            x.iter().try_for_each(|(k, v)| same(v, &y[k], &format!("{path}.{k}")))
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Err(format!("{path}: {} vs {} entries", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().try_for_each(|(i, (v, w))| same(v, w, &format!("{path}[{i}]")))
        }
        _ if a == b => Ok(()),
        _ => Err(format!("{path}: {a} vs {b}")),
    }
}

fn doc(i: usize) -> Value {
    let mut d = json!({"g": format!("g{}", i % 3), "n": i, "s": format!("{}", "abcde".repeat(1 + i % 4).chars().skip(i % 5).collect::<String>())});
    // 0, 1, 2 or 3 values of `v`; a weight on all but every ninth doc.
    let v: Vec<f64> = match i % 4 {
        0 => vec![],
        1 => vec![(i % 17) as f64],
        2 => vec![(i % 17) as f64, ((i * 3) % 23) as f64],
        _ => vec![1.5 * (i % 5) as f64, 2.0, (i % 11) as f64],
    };
    if !v.is_empty() {
        d["v"] = json!(v);
    }
    if i % 9 != 0 {
        d["w"] = json!((i % 7) + 1);
    }
    d["k"] = if i % 2 == 0 { json!([format!("k{}", i % 13), format!("k{}x", i % 5)]) } else { json!(format!("k{}", i % 13)) };
    d["t"] = if i % 5 == 0 { json!(["hot", format!("t{}", i % 4)]) } else { json!(format!("t{}", i % 4)) };
    d["d"] = json!(1_600_000_000_000i64 + (i as i64 % 50) * 86_400_000);
    d
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_shard_and_four_shards_reduce_alike() {
    let cl = start(&[]).await;
    for (name, shards) in [("one", 1), ("four", 4)] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.routing.allocation.total_shards_per_node": 2},
            "mappings": {"properties": {"g": {"type": "keyword"}, "k": {"type": "keyword"}, "t": {"type": "keyword"}, "s": {"type": "keyword"},
                "n": {"type": "long"}, "v": {"type": "double"}, "w": {"type": "long"}, "d": {"type": "date"}}}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (_, health) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(health["status"], "green", "{health}");
        let mut bulk = String::new();
        for i in 0..DOCS {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{}\n", doc(i)));
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true", cl.p.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
    }
    // The four shards span both nodes.
    let (_, shards) = cl.call(0, Method::GET, "/_cat/shards/four?format=json&h=node", None).await;
    let nodes: std::collections::BTreeSet<&str> = shards.as_array().unwrap().iter().filter_map(|s| s["node"].as_str()).collect();
    assert_eq!(nodes.len(), 2, "{shards}");

    let wavg = |extra: Value| {
        let mut w = json!({"value": {"field": "v"}, "weight": {"field": "w"}});
        w.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        json!({"weighted_avg": w})
    };
    let shapes: Vec<(&str, Value)> = vec![
        ("weighted_avg", json!({"w": wavg(json!({}))})),
        ("weighted_avg missing", json!({"w": wavg(json!({"value": {"field": "v", "missing": 5}, "weight": {"field": "w", "missing": 2}}))})),
        ("weighted_avg date", json!({"w": wavg(json!({"value": {"field": "d"}, "format": "yyyy-MM-dd"}))})),
        ("weighted_avg in terms", json!({"by_g": {"terms": {"field": "g"}, "aggs": {"w": wavg(json!({})), "a": {"avg": {"field": "v"}}}}})),
        ("weighted_avg in histogram", json!({"h": {"histogram": {"field": "n", "interval": 100}, "aggs": {"w": wavg(json!({}))}}})),
        ("weighted_avg in filters", json!({"f": {"filters": {"filters": {"few": {"range": {"n": {"lt": 50}}}, "rest": {"range": {"n": {"gte": 50}}}}}, "aggs": {"w": wavg(json!({}))}}})),
        ("metrics", json!({"avg": {"avg": {"field": "v"}}, "sum": {"sum": {"field": "v"}}, "min": {"min": {"field": "v"}}, "max": {"max": {"field": "v"}},
            "count": {"value_count": {"field": "v"}}, "stats": {"stats": {"field": "v"}}, "ext": {"extended_stats": {"field": "v"}}})),
        ("metrics in terms", json!({"by_k": {"terms": {"field": "k", "size": 30}, "aggs": {"avg": {"avg": {"field": "v"}}, "stats": {"stats": {"field": "v"}}, "count": {"value_count": {"field": "v"}}}}})),
        ("string_stats", json!({"ss": {"string_stats": {"field": "s"}}, "ssk": {"string_stats": {"field": "k", "show_distribution": true}}})),
        ("cardinality", json!({"c": {"cardinality": {"field": "k"}}, "cv": {"cardinality": {"field": "v"}}})),
        ("rate", json!({"dh": {"date_histogram": {"field": "d", "calendar_interval": "week"}, "aggs": {"r": {"rate": {"field": "v", "unit": "day"}}}}})),
        ("significant_terms", json!({"by_g": {"terms": {"field": "g"}, "aggs": {"sig": {"significant_terms": {"field": "t", "min_doc_count": 1}}}}})),
        ("nested buckets", json!({"by_g": {"terms": {"field": "g"}, "aggs": {"by_t": {"terms": {"field": "t"}, "aggs": {"w": wavg(json!({})), "m": {"max": {"field": "v"}}}}}}})),
    ];
    let search = |index: &'static str, node: usize, aggs: Value| {
        let cl = &cl;
        async move {
            let (status, got) = cl.call(node, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 0, "aggs": aggs}))).await;
            assert_eq!(status, 200, "{index}: {got}");
            assert_eq!(got["_shards"]["failed"], 0, "{got}");
            assert!(!got.to_string().contains("__p") && !got.to_string().contains("__fanout"), "{got}");
            got["aggregations"].clone()
        }
    };
    for (label, aggs) in &shapes {
        let want = search("one", 0, aggs.clone()).await;
        for node in 0..2 {
            let got = search("four", node, aggs.clone()).await;
            if let Err(e) = same(&got, &want, "aggregations") {
                panic!("{label} (via n{node}): {e}\n got {got}\nwant {want}");
            }
        }
    }
    // Parent pipelines: gap policies over empty histogram buckets (avg of
    // no values), missing paths, and chains declared in any order, at two
    // depths. One shard runs them locally; four through the coordinator.
    let query = json!({"bool": {"should": [{"range": {"n": {"lt": 100}}}, {"range": {"n": {"gte": 300, "lt": 420}}}]}});
    let chain = |order: &[usize]| -> Value {
        let defs = [
            ("s", json!({"sum": {"field": "v"}})),
            ("a", json!({"avg": {"field": "v"}})),
            ("d", json!({"derivative": {"buckets_path": "s"}})),
            ("cs", json!({"cumulative_sum": {"buckets_path": "d"}})),
            ("dd", json!({"derivative": {"buckets_path": "cs", "gap_policy": "insert_zeros"}})),
            ("bs", json!({"bucket_script": {"buckets_path": {"x": "cs", "y": "a"}, "script": "params.x + params.y"}})),
            ("bsz", json!({"bucket_script": {"buckets_path": {"y": "a"}, "script": "params.y * 2", "gap_policy": "insert_zeros"}})),
            ("bsk", json!({"bucket_script": {"buckets_path": {"y": "a"}, "script": "params.y * 2", "gap_policy": "keep_values"}})),
        ];
        let mut m = Map::new();
        for i in order {
            m.insert(defs[*i].0.to_string(), defs[*i].1.clone());
        }
        Value::Object(m)
    };
    let orders: [&[usize]; 3] = [&[0, 1, 2, 3, 4, 5, 6, 7], &[7, 6, 5, 4, 3, 2, 1, 0], &[4, 0, 5, 3, 7, 1, 2, 6]];
    let hist = |subs: Value| json!({"histogram": {"field": "n", "interval": 50, "min_doc_count": 0}, "aggs": subs});
    let mut pipeline_shapes: Vec<(String, Value)> = Vec::new();
    for (i, order) in orders.iter().enumerate() {
        pipeline_shapes.push((format!("chain order {i}"), json!({"h": hist(chain(order))})));
        pipeline_shapes.push((format!("nested chain order {i}"), json!({"g": {"terms": {"field": "g"}, "aggs": {"h": hist(chain(order))}}})));
    }
    for policy in ["skip", "insert_zeros", "keep_values"] {
        for script in ["params.x > 0", "params.x <= 0", "params.x != params.x"] {
            pipeline_shapes.push((format!("selector {policy} {script}"), json!({"h": hist(json!({"a": {"avg": {"field": "v"}},
                "sel": {"bucket_selector": {"buckets_path": {"x": "a"}, "script": script, "gap_policy": policy}}}))})));
        }
        // A path to a metric the bucket does not have (`a.max` of an avg
        // is missing everywhere) and to `_count`.
        pipeline_shapes.push((format!("selector missing path {policy}"), json!({"h": hist(json!({"a": {"avg": {"field": "v"}},
            "sel": {"bucket_selector": {"buckets_path": {"x": "a.max", "n": "_count"}, "script": "params.n > 0 || params.x == 0", "gap_policy": policy}}}))})));
    }
    for (label, aggs) in &pipeline_shapes {
        let body = json!({"size": 0, "query": query, "aggs": aggs});
        let (status, want) = cl.call(0, Method::POST, "/one/_search", Some(body.clone())).await;
        assert_eq!(status, 200, "{label}: {want}");
        for node in 0..2 {
            let (status, got) = cl.call(node, Method::POST, "/four/_search", Some(body.clone())).await;
            assert_eq!(status, 200, "{label}: {got}");
            if let Err(e) = same(&got["aggregations"], &want["aggregations"], "aggregations") {
                panic!("{label} (via n{node}): {e}\n got {}\nwant {}", got["aggregations"], want["aggregations"]);
            }
        }
    }
    // Every declaration order gives the same chain.
    let first = cl.call(0, Method::POST, "/one/_search", Some(json!({"size": 0, "query": query, "aggs": pipeline_shapes[0].1}))).await.1;
    let first_buckets = &first["aggregations"]["h"]["buckets"];
    assert!(first_buckets.as_array().unwrap().iter().any(|b| b["doc_count"] == 0), "{first}");
    assert!(first_buckets.as_array().unwrap().iter().any(|b| b.get("cs").is_some_and(|c| c["value"].as_f64().unwrap() != 0.0)), "{first}");
    for (label, aggs) in pipeline_shapes.iter().take(6).step_by(2) {
        let got = cl.call(1, Method::POST, "/four/_search", Some(json!({"size": 0, "query": query, "aggs": aggs}))).await.1;
        if let Err(e) = same(&got["aggregations"]["h"], &first["aggregations"]["h"], "h") {
            panic!("{label}: declaration order changed the result: {e}");
        }
    }
    // Cycles and unresolved paths are rejected, locally and distributed.
    for (aggs, why) in [
        (json!({"h": hist(json!({"x": {"derivative": {"buckets_path": "y"}}, "y": {"cumulative_sum": {"buckets_path": "x"}}}))}), "Cyclical dependency"),
        (json!({"h": hist(json!({"x": {"derivative": {"buckets_path": "nope"}}}))}), "No aggregation found for path [nope]"),
    ] {
        for (index, node) in [("one", 0), ("four", 0), ("four", 1)] {
            let (status, v) = cl.call(node, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 0, "aggs": aggs}))).await;
            assert_eq!(status, 400, "{index} via n{node}: {v}");
            assert!(v.to_string().contains(why), "{index} via n{node}: {v}");
        }
    }

    // The review's example: v:[0,2],w:1 on one shard and v:[10],w:1 on
    // another average to 4.
    let (status, body) = cl.call(0, Method::PUT, "/tiny", Some(json!({
        "settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.routing.allocation.total_shards_per_node": 1},
        "mappings": {"properties": {"v": {"type": "double"}, "w": {"type": "double"}}}}))).await;
    assert_eq!(status, 200, "{body}");
    let (_, health) = cl.call(0, Method::GET, "/_cluster/health/tiny?wait_for_status=green&timeout=60s", None).await;
    assert_eq!(health["status"], "green", "{health}");
    // The two docs on different shards.
    let shard_of = |id: String| {
        let cl = &cl;
        async move {
            for k in 0..2 {
                let (status, v) = cl.call(0, Method::POST, &format!("/tiny/_search?preference=_shards:{k}"), Some(json!({"query": {"ids": {"values": [id]}}}))).await;
                assert_eq!(status, 200, "{v}");
                if v["hits"]["total"]["value"] == 1 { return k; }
            }
            panic!("{id} not found");
        }
    };
    let (status, body) = cl.call(0, Method::PUT, "/tiny/_doc/x?refresh=true", Some(json!({"v": [0.0, 2.0], "w": 1.0}))).await;
    assert_eq!(status, 201, "{body}");
    let first = shard_of("x".into()).await;
    let mut placed = false;
    for i in 0..64 {
        let id = format!("y{i}");
        let (status, body) = cl.call(0, Method::PUT, &format!("/tiny/_doc/{id}?refresh=true"), Some(json!({"v": 10.0, "w": 1.0}))).await;
        assert_eq!(status, 201, "{body}");
        if shard_of(id.clone()).await != first { placed = true; break; }
        let (status, body) = cl.call(0, Method::DELETE, &format!("/tiny/_doc/{id}?refresh=true"), None).await;
        assert_eq!(status, 200, "{body}");
    }
    assert!(placed);
    for node in 0..2 {
        let (status, got) = cl.call(node, Method::POST, "/tiny/_search", Some(json!({"size": 0, "aggs": {"w": wavg(json!({}))}}))).await;
        assert_eq!(status, 200, "{got}");
        assert_eq!(got["aggregations"]["w"]["value"], json!(4.0), "{got}");
    }
}

/// Each `significant_terms` gives the same result alone and next to a
/// same-named one under another parent, on one shard and on four.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn same_named_significant_terms_are_independent() {
    let cl = start(&[]).await;
    for (name, shards) in [("one", 1), ("four", 4)] {
        let (status, body) = cl.call(0, Method::PUT, &format!("/{name}"), Some(json!({
            "settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.routing.allocation.total_shards_per_node": 2},
            "mappings": {"properties": {"g": {"type": "keyword"}, "k": {"type": "keyword"}, "t": {"type": "keyword"}, "s": {"type": "keyword"},
                "n": {"type": "long"}, "v": {"type": "double"}, "w": {"type": "long"}, "d": {"type": "date"}}}
        }))).await;
        assert_eq!(status, 200, "{body}");
        let (_, health) = cl.call(0, Method::GET, &format!("/_cluster/health/{name}?wait_for_status=green&timeout=60s"), None).await;
        assert_eq!(health["status"], "green", "{health}");
        let mut bulk = String::new();
        for i in 0..DOCS {
            bulk.push_str(&format!("{{\"index\":{{\"_id\":\"{i}\"}}}}\n{}\n", doc(i)));
        }
        let body: Value = cl.c.post(format!("http://127.0.0.1:{}/{name}/_bulk?refresh=true", cl.p.http[0]))
            .header("content-type", "application/x-ndjson").body(bulk).send().await.unwrap().json().await.unwrap();
        assert_eq!(body["errors"], false, "{body}");
    }
    let a = json!({"a": {"filter": {"term": {"t": "hot"}}, "aggs": {"sig": {"significant_terms": {"field": "k", "min_doc_count": 1}}}}});
    let b = json!({"b": {"filter": {"term": {"t": "t1"}}, "aggs": {"sig": {"significant_terms": {"field": "s", "min_doc_count": 1, "chi_square": {}}}}}});
    let c = json!({"c": {"terms": {"field": "g"}, "aggs": {"sig": {"significant_terms": {"field": "t", "min_doc_count": 1, "background_filter": {"range": {"n": {"gte": 100}}}}}}}});
    let mut all = Map::new();
    for x in [&a, &b, &c] {
        all.extend(x.as_object().unwrap().clone());
    }
    let all = Value::Object(all);
    let run = |index: &'static str, node: usize, aggs: Value| {
        let cl = &cl;
        async move {
            let (status, got) = cl.call(node, Method::POST, &format!("/{index}/_search"), Some(json!({"size": 0, "aggs": aggs}))).await;
            assert_eq!(status, 200, "{index}: {got}");
            assert_eq!(got["_shards"]["failed"], 0, "{got}");
            got["aggregations"].clone()
        }
    };
    for index in ["one", "four"] {
        for node in 0..2 {
            let together = run(index, node, all.clone()).await;
            for (name, alone) in [("a", &a), ("b", &b), ("c", &c)] {
                let solo = run(index, node, alone.clone()).await;
                if let Err(e) = same(&together[name], &solo[name], name) {
                    panic!("{index} via n{node}: {name} alone differs from {name} beside same-named aggregations: {e}\nalone {}\ntogether {}", solo[name], together[name]);
                }
            }
            for name in ["a", "b"] {
                assert!(!together[name]["sig"]["buckets"].as_array().unwrap().is_empty(), "{name}: {together}");
            }
            // And the four-shard result is the one-shard result.
            if index == "four" {
                let want = run("one", 0, all.clone()).await;
                if let Err(e) = same(&together, &want, "aggregations") {
                    panic!("four via n{node} vs one: {e}\n got {together}\nwant {want}");
                }
            }
        }
    }
}

use serde_json::Map;
