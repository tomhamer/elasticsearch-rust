//! Pre-encoded HTTP requests; separately report wire/body and JSON decoding cost.
use super::*;
use std::collections::BTreeMap;

// Independent predicate for this deterministic vector corpus (tag=t{id%10}).
fn fixture_filter_matches(filter: &Value, doc: usize) -> bool {
    if let Some(term) = filter.get("term") {
        assert_eq!(term.as_object().unwrap().len(), 1);
        return term["tag"].as_str().expect("fixture filter must use tag")
            == format!("t{}", doc % 10);
    }
    let boolean = filter.get("bool").expect("unsupported fixture filter");
    let clauses = |name: &str| -> Vec<&Value> {
        match boolean.get(name) {
            None => vec![],
            Some(Value::Array(v)) => v.iter().collect(),
            Some(v) => vec![v],
        }
    };
    let must = clauses("must");
    let filters = clauses("filter");
    let should = clauses("should");
    let required = boolean
        .get("minimum_should_match")
        .map(|v| v.as_u64().unwrap() as usize)
        .unwrap_or(usize::from(
            must.is_empty() && filters.is_empty() && !should.is_empty(),
        ));
    must.iter()
        .chain(&filters)
        .all(|v| fixture_filter_matches(v, doc))
        && clauses("must_not")
            .iter()
            .all(|v| !fixture_filter_matches(v, doc))
        && should
            .iter()
            .filter(|v| fixture_filter_matches(v, doc))
            .count()
            >= required
}

fn stats(mut xs: Vec<f64>) -> Value {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    json!({"n":n,"p50_us":if n%2==0 {(xs[n/2-1]+xs[n/2])/2.0} else {xs[n/2]},"p95_us":xs[(n as f64*0.95).ceil() as usize-1]})
}
fn validate(a: &Value, b: &Value, request: &Value) {
    assert_eq!(a["hits"]["total"], b["hits"]["total"]);
    let aa = a["hits"]["hits"].as_array().unwrap();
    let bb = b["hits"]["hits"].as_array().unwrap();
    assert_eq!(aa.len(), bb.len());
    let distinct = aa.windows(2).all(|w| w[0]["_score"] != w[1]["_score"]);
    for (x, y) in aa.iter().zip(bb) {
        match (x["_score"].as_f64(), y["_score"].as_f64()) {
            (Some(ax), Some(bx)) => assert!((ax - bx).abs() <= 2e-5 + ax.abs() * 2e-5),
            (None, None) => assert_eq!(x.get("_score"), y.get("_score")),
            _ => panic!("score shape mismatch"),
        }
        if request.get("sort").is_some() {
            assert_eq!(x.get("sort"), y.get("sort"));
        }
        if distinct || request.get("sort").is_some() {
            assert_eq!(x["_id"], y["_id"]);
        }
        assert_eq!(x.get("_source").is_some(), request["_source"] != false);
        assert_eq!(y.get("_source").is_some(), request["_source"] != false);
        if x["_id"] == y["_id"] {
            assert_eq!(x.get("_source"), y.get("_source"));
        }
    }
    if let Some(aggs) = a.get("aggregations") {
        validate_aggregation(aggs, b.get("aggregations").expect("missing aggregations"));
    }
}

// Aggregation counts/keys must agree; tolerate ordinary floating reduction order.
fn validate_aggregation(a: &Value, b: &Value) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in a {
                validate_aggregation(value, b.get(key).expect("missing aggregation key"));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                validate_aggregation(a, b);
            }
        }
        (Value::Number(a), Value::Number(b)) => match (a.as_i64(), b.as_i64()) {
            (Some(a), Some(b)) => assert_eq!(a, b),
            _ => {
                let a = a.as_f64().unwrap();
                let b = b.as_f64().unwrap();
                assert!(
                    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0),
                    "aggregation values {a} != {b}"
                );
            }
        },
        _ => assert_eq!(a, b),
    }
}

fn qdrant_filter(filter: &Value) -> Value {
    if let Some(term) = filter.get("term") {
        let (field, value) = term.as_object().unwrap().iter().next().unwrap();
        return json!({"key":field,"match":{"value":value}});
    }
    let boolean = filter.get("bool").expect("benchmark Qdrant filter subset");
    let clauses = |key: &str| -> Vec<Value> { match boolean.get(key) {
        None => vec![], Some(Value::Array(a)) => a.iter().map(qdrant_filter).collect(), Some(v) => vec![qdrant_filter(v)],
    }};
    let mut must = clauses("must"); must.extend(clauses("filter"));
    let should = clauses("should"); let must_not = clauses("must_not");
    let required = boolean.get("minimum_should_match").map(|v| v.as_u64().unwrap() as usize)
        .unwrap_or(usize::from(must.is_empty() && !should.is_empty()));
    let mut out = json!({});
    if !must.is_empty() { out["must"] = json!(must); }
    if !must_not.is_empty() { out["must_not"] = json!(must_not); }
    if required == 1 { out["should"] = json!(should); }
    else if required > 1 { out["min_should"] = json!({"conditions":should,"min_count":required}); }
    out
}
fn qdrant_request(request: &Value) -> Value {
    assert_eq!(request["_source"], false, "Qdrant comparison supports metadata-only hits");
    let knn = &request["knn"];
    assert!(knn.is_object());
    let mut out = json!({"query":knn["query_vector"],"limit":knn["k"],"with_payload":false,"with_vector":false,
        "params":{"hnsw_ef":knn["num_candidates"]}});
    if let Some(filter) = knn.get("filter") {
        let condition = qdrant_filter(filter);
        out["filter"] = if filter.get("term").is_some() { json!({"must":[condition]}) } else { condition };
    }
    if let Some(oversample) = knn.get("rescore_vector").and_then(|v|v.get("oversample")) {
        out["params"]["quantization"] = json!({"rescore":true,"oversampling":oversample});
    }
    out
}
fn qdrant_response(value: Value) -> Value {
    assert_eq!(value["status"], "ok", "{value}");
    let hits: Vec<Value> = value["result"]["points"].as_array().unwrap().iter().map(|p| {
        json!({"_id":p["id"].as_u64().unwrap().to_string(),"_score":(1.0+p["score"].as_f64().unwrap())/2.0})
    }).collect();
    json!({"hits":{"total":{"value":hits.len(),"relation":"eq"},"hits":hits},
        "_shards":{"total":1,"successful":1,"failed":0}})
}

// The default absolute floor is retained for historical reproducibility.
// `reference=NAME` compares both mean and minimum recall in each workload.
fn recall_gate_failures(recalls: &BTreeMap<String, Value>, gate: &str) -> Vec<String> {
    let reference = gate.strip_prefix("reference=");
    assert!(gate == "absolute95" || reference.is_some_and(|r| !r.is_empty()), "unsupported recall gate {gate}");
    let mut failures = Vec::new();
    for (key, recall) in recalls {
        let (class, system) = key.rsplit_once('/').unwrap();
        if reference == Some(system) { continue; }
        let expected = reference.map(|name| recalls.get(&format!("{class}/{name}")));
        if expected == Some(None) {
            failures.push(format!("{key}: missing reference recall for {class}"));
            continue;
        }
        for metric in if reference.is_some() { &["mean", "min"][..] } else { &["mean"][..] } {
            let floor = expected.flatten().map_or(0.95, |r| r[*metric].as_f64().unwrap());
            let actual = recall[*metric].as_f64().unwrap();
            if actual + 1e-12 < floor {
                failures.push(format!("{key}: {metric} recall {actual} below {floor}"));
            }
        }
    }
    failures
}

pub(super) async fn run(args: &[String]) {
    let entries: Vec<Value> = BufReader::new(std::fs::File::open(&args[2]).unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    let targets: Vec<(String, String)> = args[3]
        .split(',')
        .map(|s| {
            let (name, url) = s.split_once('=').unwrap();
            assert!(url.starts_with("http://127.0.0.1:"));
            (name.into(), url.into())
        })
        .collect();
    let warmup: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(100);
    let rounds: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(100);
    let ground = args.get(7).map(|path| {
        serde_json::from_str::<Value>(&std::fs::read_to_string(path).unwrap()).unwrap()
    });
    let vector_bytes = args.get(8).map(|path| std::fs::read(path).unwrap());
    let vectors: Vec<f32> = vector_bytes
        .as_ref()
        .map(|b| {
            b.chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect()
        })
        .unwrap_or_default();
    let mut recalls = BTreeMap::<String, Vec<f64>>::new();
    let client = reqwest::Client::builder()
        .no_gzip()
        .tcp_nodelay(true)
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();
    let prepared: Vec<Vec<reqwest::Request>> = entries
        .iter()
        .map(|e| {
            targets
                .iter()
                .map(|(name, url)| {
                    let request = if url.contains("/collections/") && url.ends_with("/points/query") {
                        let mut request = qdrant_request(&e["request"]);
                        if let Some(ef) = e.get("qdrant_hnsw_ef").and_then(|v|v.get(name)).and_then(Value::as_u64) {
                            request["params"]["hnsw_ef"] = json!(ef);
                        }
                        request
                    } else {
                        let mut request = e["request"].clone();
                        if let Some(ef) = e.get("target_num_candidates").and_then(|v| v.get(name)).and_then(Value::as_u64) {
                            assert!(request["knn"].is_object(), "candidate override requires KNN");
                            request["knn"]["num_candidates"] = json!(ef);
                        }
                        request
                    };
                    let body = serde_json::to_vec(&request).unwrap();
                    client
                        .post(url)
                        .header("content-type", "application/json")
                        .body(body.clone())
                        .build()
                        .unwrap()
                })
                .collect()
        })
        .collect();
    let mut samples = BTreeMap::<String, Vec<f64>>::new();
    let mut seed = 42u64;
    for r in 0..warmup + rounds {
        for (qi, entry) in entries.iter().enumerate() {
            let mut modes: Vec<usize> = (0..targets.len()).collect();
            for i in (1..modes.len()).rev() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                modes.swap(i, (seed as usize) % (i + 1));
            }
            for mode in modes {
                let start = Instant::now();
                let response = client
                    .execute(prepared[qi][mode].try_clone().unwrap())
                    .await
                    .unwrap();
                if !response.status().is_success() {
                    let status = response.status();
                    panic!("target {} request {}: {status}: {}", targets[mode].0, entry["id"], response.text().await.unwrap());
                }
                let bytes = response.bytes().await.unwrap();
                let wire_us = start.elapsed().as_secs_f64() * 1e6;
                let decode = Instant::now();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                let decode_us = decode.elapsed().as_secs_f64() * 1e6;
                if targets[mode].1.contains("/collections/") {
                    assert_eq!(value["status"], "ok", "{value}");
                } else { assert_eq!(value["_shards"]["failed"], 0); }
                if r >= warmup {
                    let key = format!("{}/{}", entry["class"].as_str().unwrap(), targets[mode].0);
                    samples
                        .entry(format!("{key}/wire_us"))
                        .or_default()
                        .push(wire_us);
                    samples
                        .entry(format!("{key}/decode_us"))
                        .or_default()
                        .push(decode_us);
                    samples
                        .entry(format!("{key}/wire_plus_decode_us"))
                        .or_default()
                        .push(wire_us + decode_us);
                    for stage in ["parse_us", "query_us", "fetch_us"] {
                        if let Some(v) = value["_poc"][stage].as_f64() {
                            samples.entry(format!("{key}/{stage}")).or_default().push(v);
                        }
                    }
                }
            }
        }
    }
    let mut verified = 0;
    for (qi, entry) in entries.iter().enumerate() {
        let mut values = Vec::new();
        for mode in 0..targets.len() {
            let response = client
                .execute(prepared[qi][mode].try_clone().unwrap())
                .await
                .unwrap();
            assert!(response.status().is_success());
            let value = response.json::<Value>().await.unwrap();
            values.push(if targets[mode].1.contains("/collections/") { qdrant_response(value) } else { value });
        }
        if let Some(knn) = entry["request"].get("knn") {
            let ground = ground
                .as_ref()
                .expect("KNN requires truth JSON and vector f32 file arguments");
            let truth = ground["queries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|q| q["id"] == entry["id"])
                .unwrap();
            let expected: std::collections::HashSet<&str> = truth["hits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| h["id"].as_str().unwrap())
                .collect();
            let query: Vec<f64> = knn["query_vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let dim = query.len();
            let qnorm = query.iter().map(|v| v * v).sum::<f64>().sqrt();
            let n = ground["index"]["documents"].as_u64().unwrap() as usize;
            assert!(vectors.len() >= n * dim);
            for (mode, value) in values.iter().enumerate() {
                let hits = value["hits"]["hits"].as_array().unwrap();
                assert_eq!(hits.len(), expected.len());
                assert_eq!(value["hits"]["total"]["value"], expected.len());
                assert_eq!(value["hits"]["total"]["relation"], "eq");
                let mut unique = std::collections::HashSet::new();
                let mut found = 0;
                for hit in hits {
                    let id = hit["_id"].as_str().unwrap();
                    assert!(unique.insert(id));
                    let doc = id.parse::<usize>().unwrap();
                    assert!(doc < n);
                    if let Some(filter) = knn.get("filter") {
                        assert!(
                            fixture_filter_matches(filter, doc),
                            "ineligible document {id}"
                        );
                    }
                    let vector = &vectors[doc * dim..(doc + 1) * dim];
                    let dot = query
                        .iter()
                        .zip(vector)
                        .map(|(q, v)| q * (*v as f64))
                        .sum::<f64>();
                    let vn = vector
                        .iter()
                        .map(|v| (*v as f64) * (*v as f64))
                        .sum::<f64>()
                        .sqrt();
                    let score = (1.0 + dot / (qnorm * vn)) / 2.0;
                    assert!((hit["_score"].as_f64().unwrap() - score).abs() < 2e-5);
                    assert_eq!(
                        hit.get("_source").is_some(),
                        entry["request"]["_source"] != false
                    );
                    if expected.contains(id) {
                        found += 1;
                    }
                }
                recalls
                    .entry(format!(
                        "{}/{}",
                        entry["class"].as_str().unwrap(),
                        targets[mode].0
                    ))
                    .or_default()
                    .push(found as f64 / expected.len() as f64);
            }
        } else {
            for value in &values[1..] {
                validate(&values[0], value, &entry["request"]);
            }
        }
        verified += 1;
    }
    let recall_summary: BTreeMap<_, _> = recalls
        .into_iter()
        .map(|(k, v)| {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (
                k,
                json!({"mean":mean,"min":v.iter().copied().reduce(f64::min).unwrap()}),
            )
        })
        .collect();
    let gate = args.get(9).map(String::as_str).unwrap_or("absolute95");
    if let Some(reference) = gate.strip_prefix("reference=") {
        assert!(ground.is_some(), "reference recall gate requires ground truth");
        assert!(targets.iter().any(|(name, _)| name == reference), "missing reference target {reference}");
    }
    let failures = recall_gate_failures(&recall_summary, gate);
    let out = json!({"requests":args[2],"targets":targets,"warmup":warmup,"rounds":rounds,"schedule":"mixed","verified_requests":verified,"score_parity":true,"recall":recall_summary,"quality_gate":{"mode":gate,"passed":failures.is_empty(),"failures":failures},"ground_truth":args.get(7),"scope":"pre-encoded, persistent HTTP; wire includes response body; JSON decoding reported separately and as wire_plus_decode; no batching","stats":samples.into_iter().map(|(k,v)|(k,stats(v))).collect::<BTreeMap<_,_>>()});
    std::fs::write(&args[4], out.to_string()).unwrap();
    println!("verified {verified}; wrote {}", args[4]);
    // Preserve diagnostics even when an ANN quality gate fails.
    assert!(failures.is_empty(), "recall gate failed: {} (results saved)", failures.join("; "));
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    #[test]
    fn qdrant_adapter_preserves_filter_and_score_conventions() {
        let request = json!({"_source":false,"knn":{"field":"v","query_vector":[1.0,0.0],"k":10,"num_candidates":500,
            "filter":{"bool":{"should":[{"term":{"tag":"t0"}},{"term":{"tag":"t1"}}]}},"rescore_vector":{"oversample":3.0}}});
        let out = qdrant_request(&request);
        assert_eq!(out["filter"]["should"].as_array().unwrap().len(), 2);
        assert_eq!(out["params"]["hnsw_ef"], 500);
        assert_eq!(out["params"]["quantization"]["oversampling"], 3.0);
        let response = qdrant_response(json!({"status":"ok","result":{"points":[{"id":7,"score":-0.5}]}}));
        assert_eq!(response["hits"]["hits"][0]["_id"], "7");
        assert_eq!(response["hits"]["hits"][0]["_score"], 0.25);
    }

    #[test]
    fn reference_recall_gate_checks_each_class_and_tail() {
        let mut recalls = BTreeMap::from([
            ("half/es".into(), json!({"mean":0.815,"min":0.0})),
            ("half/rustic".into(), json!({"mean":0.88,"min":0.4})),
        ]);
        assert!(recall_gate_failures(&recalls, "reference=es").is_empty());
        assert!(!recall_gate_failures(&recalls, "absolute95").is_empty());
        recalls.insert("half/rustic".into(), json!({"mean":0.80,"min":0.4}));
        assert!(!recall_gate_failures(&recalls, "reference=es").is_empty());
        recalls.insert("half/es".into(), json!({"mean":0.80,"min":0.5}));
        assert!(!recall_gate_failures(&recalls, "reference=es").is_empty());
        recalls.insert("ten/rustic".into(), json!({"mean":1.0,"min":1.0}));
        assert!(recall_gate_failures(&recalls, "reference=es").iter().any(|e| e.contains("missing reference")));
    }
}
