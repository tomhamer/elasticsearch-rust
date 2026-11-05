//! Same-index decomposition; transport is deliberately excluded and reported as such.
use super::*;
use std::{collections::BTreeMap, hint::black_box};

fn stats(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    json!({"n":n,"p50_us":if n%2==0 {(values[n/2-1]+values[n/2])/2.0} else {values[n/2]},"p95_us":values[(n as f64*0.95).ceil() as usize-1],"mean_us":values.iter().sum::<f64>()/n as f64})
}

pub(super) fn requests(args: &[String]) {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut output = String::new();
    for line in BufReader::new(std::fs::File::open(&args[2]).unwrap()).lines() {
        let q: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let class = q["class"].as_str().unwrap();
        let count = counts.entry(class.into()).or_default();
        if *count >= 12 {
            continue;
        }
        *count += 1;
        let field = q["field"].as_str().unwrap();
        let query = match q["kind"].as_str().unwrap() {
            "term" => json!({"term":{field:q["term"]}}),
            "and" | "or" => {
                let clause = if q["kind"] == "and" { "must" } else { "should" };
                json!({"bool":{clause:q["terms"].as_array().unwrap().iter().map(|t| json!({"term":{field:t}})).collect::<Vec<_>>()}})
            }
            "phrase" => {
                json!({"match_phrase":{field:q["terms"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect::<Vec<_>>().join(" ")}})
            }
            "range" => json!({"range":{field:{"gte":q["lower"],"lte":q["upper"]}}}),
            _ => panic!("unsupported benchmark query"),
        };
        output.push_str(&json!({"id":q["id"],"class":class,"request":{"query":query,"size":10,"track_total_hits":true,"_source":false}}).to_string());
        output.push('\n');
    }
    std::fs::write(&args[3], output).unwrap();
}

pub(super) fn run(args: &[String]) {
    let app = App {
        index: RwLock::new(Index::open(std::path::Path::new(&args[2])).unwrap()),
        inline: false,
        in_place: false,
    };
    let entries: Vec<Value> = BufReader::new(std::fs::File::open(&args[3]).unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    let warmup: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(200);
    let rounds: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let schedule = args.get(7).map(String::as_str).unwrap_or("grouped");
    assert!(["grouped", "mixed"].contains(&schedule));
    let mut samples = BTreeMap::<String, Vec<f64>>::new();
    let mut per_query = Vec::new();
    let mut seed = 42u64;
    let mut measure = |entry: &Value, measured: bool| {
        let class = entry["class"].as_str().unwrap();
        let req = &entry["request"];
        assert_eq!(req["track_total_hits"], true);
        let bytes = serde_json::to_vec(req).unwrap();
        let query = request_query(req).unwrap();
        let options = SearchOptions {
            top_k: req["size"].as_u64().unwrap_or(10) as usize,
            track_total_hits: TrackTotalHits::True,
            ..Default::default()
        };
        // Shuffle paths deterministically within each iteration.
        let mut modes = [
            "collector",
            "query_build",
            "core",
            "baseline",
            "candidate",
            "mapped-candidate",
            "mapped-topk",
        ];
        for i in (1..modes.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            modes.swap(i, (seed as usize) % (i + 1));
        }
        for mode in modes {
            let start = Instant::now();
            let result = if mode == "collector" || mode == "query_build" {
                let index = app.index.read().unwrap();
                let built;
                let q = if mode == "query_build" {
                    built = request_query(req).unwrap();
                    &built
                } else {
                    &query
                };
                black_box(Searcher::new(index.leaves()).search(q, &options));
                None
            } else {
                Some(black_box(execute(&app, mode, &bytes).unwrap()))
            };
            let us = start.elapsed().as_secs_f64() * 1e6;
            if measured {
                samples
                    .entry(format!("{class}/{mode}/wall"))
                    .or_default()
                    .push(us);
                if let Some(result) = result {
                    let v: Value = serde_json::from_str(&result).unwrap();
                    for stage in ["parse_us", "query_us", "fetch_us"] {
                        samples
                            .entry(format!("{class}/{mode}/{stage}"))
                            .or_default()
                            .push(v["_poc"][stage].as_f64().unwrap());
                    }
                }
            }
        }
    };
    if schedule == "grouped" {
        for entry in &entries {
            for r in 0..warmup + rounds {
                measure(entry, r >= warmup);
            }
        }
    } else {
        for r in 0..warmup + rounds {
            for entry in &entries {
                measure(entry, r >= warmup);
            }
        }
    }
    // Validate after measurement so validation work cannot influence warm-up locality.
    for entry in &entries {
        let req = &entry["request"];
        let bytes = serde_json::to_vec(req).unwrap();
        let query = request_query(req).unwrap();
        let index = app.index.read().unwrap();
        let leaves = index.leaves();
        let top = Searcher::new(leaves.clone()).search(
            &query,
            &SearchOptions {
                top_k: 10,
                track_total_hits: TrackTotalHits::True,
                ..Default::default()
            },
        );
        let baseline: Value =
            serde_json::from_str(&execute(&app, "baseline", &bytes).unwrap()).unwrap();
        let candidate: Value =
            serde_json::from_str(&execute(&app, "candidate", &bytes).unwrap()).unwrap();
        assert_eq!(baseline["hits"], candidate["hits"]);
        let mapped: Value =
            serde_json::from_str(&execute(&app, "mapped-candidate", &bytes).unwrap()).unwrap();
        let trimmed: Value =
            serde_json::from_str(&execute(&app, "mapped-topk", &bytes).unwrap()).unwrap();
        assert_eq!(mapped["hits"], trimmed["hits"]);
        if !req
            .get("knn")
            .is_some_and(|q| q.get("rescore_vector").is_some())
        {
            assert_eq!(candidate["hits"], mapped["hits"]);
        }
        assert_eq!(candidate["hits"]["total"]["value"], top.total_hits);
        let returned = candidate["hits"]["hits"].as_array().unwrap();
        assert_eq!(returned.len(), top.hits.len());
        for (actual, hit) in returned.iter().zip(&top.hits) {
            assert_eq!(
                actual["_id"],
                leaves[hit.leaf].keyword_value(ID, hit.doc).unwrap()
            );
            assert!((actual["_score"].as_f64().unwrap() - hit.score as f64).abs() < 2e-5);
        }
        per_query.push(json!({"id":entry["id"],"class":entry["class"],"hits":top.total_hits,"top_score":top.max_score}));
    }
    let out = json!({"index":args[2],"requests":args[3],"schedule":schedule,"warmup":warmup,"rounds":rounds,"verified":entries.len(),"timing_scope":"in-process; no HTTP or scheduling; collector includes leaf allocation and read lock; query_build adds DSL construction; core includes JSON parsing and envelope but no hits; candidate/baseline include full hit response construction","per_query":per_query,"stats":samples.into_iter().map(|(k,v)|(k,stats(v))).collect::<BTreeMap<_,_>>()});
    std::fs::write(&args[4], out.to_string()).unwrap();
    println!("verified {} requests; wrote {}", entries.len(), args[4]);
}

pub(super) fn truth(args: &[String]) {
    let index = Index::open(std::path::Path::new(&args[2])).unwrap();
    let entries: Vec<Value> = BufReader::new(std::fs::File::open(&args[3]).unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    let leaves = index.leaves();
    let mut result = Vec::new();
    for e in entries {
        let Query::Knn {
            field,
            vector,
            k,
            num_candidates,
            filter,
            ..
        } = request_query(&e["request"]).unwrap()
        else {
            panic!("knn required")
        };
        let req = rustic_core::index::KnnRequest {
            field,
            vector,
            k,
            num_candidates,
            filter: filter.map(|f| *f),
            similarity_threshold: None,
            boost: 1.0,
        };
        let (hits, stats) =
            index.knn_search_with(&req, &rustic_core::vector::store::PlannerConfig::exact()).unwrap();
        result.push(json!({"id":e["id"],"class":e["class"],"vector_ops":stats.vector_ops,"hits":hits.iter().map(|h|json!({"id":leaves[h.leaf].keyword_value(ID,h.doc).unwrap(),"score":h.score})).collect::<Vec<_>>()}));
    }
    std::fs::write(&args[4],json!({"index":index_info(&index),"method":"full-precision exact full scan; independent returned-score checks use f64 cosine","queries":result}).to_string()).unwrap();
}
