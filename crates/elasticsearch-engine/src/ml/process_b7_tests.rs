//! Production review batch 7 closure tests (anomaly detection, datafeeds).

use super::*;
use super::tests::{ndjson, open_job, put_and_open, results, T0};

/// One-minute buckets from `T0`.
fn b(i: i64) -> i64 {
    T0 + i * 60_000
}

fn job_with(analysis: Value) -> Value {
    json!({"job_id": "j", "analysis_config": analysis, "data_description": {"time_field": "time", "time_format": "epoch_ms"}})
}

/// Feed `docs` and finalize every bucket before `until`.
fn feed(p: &mut Proc, docs: Vec<Value>, until: i64) -> Docs {
    let mut out = Vec::new();
    p.ingest(docs, 100, true, &mut out);
    p.flush(&super::super::anomaly::FlushParams { advance_time: Some(until), ..Default::default() }, &mut out);
    out
}

fn records_of(out: &Docs) -> Vec<&Value> {
    out.iter().filter(|(_, d)| d["result_type"] == "record").map(|(_, d)| d).collect()
}

/// #554: information content is the compressed size of the distinct
/// strings, so a varied string carries more than a repetitive one of the
/// same byte length (and the reverse transition is visible too).
#[test]
fn info_content_measures_information_not_bytes() {
    let repetitive: BTreeSet<String> = std::iter::once("a".repeat(512)).collect();
    let mut x = 12345u64;
    let varied: String = (0..512).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1); (b'a' + ((x >> 33) % 26) as u8) as char }).collect();
    let varied: BTreeSet<String> = std::iter::once(varied).collect();
    let (r, v) = (info_content(&repetitive), info_content(&varied));
    assert!(v > 4.0 * r, "varied {v} vs repetitive {r}");
    assert_eq!(info_content(&BTreeSet::new()), 0.0);
    // Through the detector: equal-length strings give different features.
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "high_info_content", "field_name": "q"}]}));
    let mut p = Proc::new(&job);
    let mut docs: Vec<Value> = (0..30).map(|i| json!({"time": b(i), "q": "a".repeat(512)})).collect();
    docs.push(json!({"time": b(30), "q": varied.iter().next().unwrap()}));
    let out = feed(&mut p, docs, b(31));
    assert!(records_of(&out).iter().any(|r| r["timestamp"] == json!(b(30))), "the varied bucket is anomalous");
}

/// #555: `exclude_frequent: by` keeps a frequent entity out of the results
/// and models; `none` analyzes it.
#[test]
fn exclude_frequent_removes_frequent_entities() {
    let run = |exclude: &str| -> (bool, bool) {
        let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "high_count", "by_field_name": "svc", "exclude_frequent": exclude}]}));
        let mut p = Proc::new(&job);
        let mut docs = Vec::new();
        for i in 0..30 {
            // `shared` is in every bucket; `rare` in a few.
            let n = if i == 29 { 200 } else { 10 };
            docs.extend((0..n).map(|k| json!({"time": b(i) + k, "svc": "shared"})));
            if i % 5 == 0 {
                docs.push(json!({"time": b(i) + 500, "svc": "rare"}));
            }
        }
        let out = feed(&mut p, docs, b(30));
        let shared_record = records_of(&out).iter().any(|r| r["by_field_value"] == "shared");
        let shared_model = p.models.keys().any(|k| k.ends_with("|shared"));
        (shared_record, shared_model)
    };
    let (rec, _) = run("none");
    assert!(rec, "without exclusion the spike of `shared` is anomalous");
    let (rec, _) = run("by");
    assert!(!rec, "a frequent by value is excluded from results");
}

/// `exclude_frequent` thresholds as ml-cpp: a person (an individual
/// detector's by value, a population's over value) is frequent above 10% of
/// the (decayed) buckets, a population's by value above 10% of the people;
/// everything is frequent before the first bucket (frequency 0.5); counts
/// are per partition.
#[test]
fn exclude_frequent_uses_ml_cpp_thresholds() {
    let sv = |partition: Option<&str>, by: Option<&str>, over: Option<&str>| SeriesValue {
        partition: partition.map(str::to_string), by: by.map(str::to_string), over: over.map(str::to_string),
        value: 0.0, point: None, extremes: None, influencers: BTreeMap::new(), without: BTreeMap::new(),
    };
    // Individual, by: `steady` in every bucket, `few` in 3 of 20 (15%),
    // `once` in 1 of 20 (5%).
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "svc", "partition_field_name": "dc", "exclude_frequent": "by"}]}));
    let mut p = Proc::new(&job);
    let d = p.detectors[0].clone();
    assert!(p.is_frequent(&d, &sv(Some("a"), Some("anything"), None)), "before any bucket the frequency is 0.5");
    let mut docs = Vec::new();
    for i in 0..20 {
        docs.push(json!({"time": b(i), "svc": "steady", "dc": "a"}));
        if i % 7 == 0 {
            docs.push(json!({"time": b(i) + 1, "svc": "few", "dc": "a"}));
        }
        if i == 0 {
            docs.push(json!({"time": b(i) + 2, "svc": "once", "dc": "a"}));
        }
        // Partition `b`: only `once`, every bucket.
        docs.push(json!({"time": b(i) + 3, "svc": "once", "dc": "b"}));
    }
    feed(&mut p, docs, b(20));
    assert!(p.is_frequent(&d, &sv(Some("a"), Some("steady"), None)));
    assert!(p.is_frequent(&d, &sv(Some("a"), Some("few"), None)), "15% > 10%");
    assert!(!p.is_frequent(&d, &sv(Some("a"), Some("once"), None)), "5% <= 10%");
    assert!(p.is_frequent(&d, &sv(Some("b"), Some("once"), None)), "per partition");

    // Population: over = people, by = attributes shared by > 10% of them.
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "uri", "over_field_name": "ip", "exclude_frequent": "all"}]}));
    let mut p = Proc::new(&job);
    let d = p.detectors[0].clone();
    let mut docs = Vec::new();
    for i in 0..20 {
        // 20 people, each in one bucket (5%); all visit `/home`, one `/admin`.
        let ip = format!("10.0.0.{i}");
        docs.push(json!({"time": b(i), "ip": ip, "uri": "/home"}));
        if i == 3 {
            docs.push(json!({"time": b(i) + 1, "ip": ip, "uri": "/admin"}));
        }
        docs.push(json!({"time": b(i) + 2, "ip": "10.9.9.9", "uri": "/status"}));
    }
    feed(&mut p, docs, b(20));
    assert!(p.is_frequent(&d, &sv(None, Some("/admin"), Some("10.9.9.9"))), "a person in every bucket");
    assert!(!p.is_frequent(&d, &sv(None, Some("/admin"), Some("10.0.0.3"))), "1 of 21 people had /admin, a person in 1 of 20 buckets");
    assert!(p.is_frequent(&d, &sv(None, Some("/home"), Some("10.0.0.3"))), "20 of 21 people had /home");
}

/// #556: a filter update reaches the rules of a running job.
#[test]
fn filter_updates_reach_running_jobs() {
    let data = tempfile::tempdir().unwrap();
    let node = Node::open(data.path()).unwrap();
    super::super::calendar::put_filter(&node, "trusted", &json!({"items": ["host-a"]})).unwrap();
    let rules = json!([{"actions": ["skip_result"], "scope": {"host": {"filter_id": "trusted", "filter_type": "include"}}}]);
    put_and_open(&node, "j", json!({"bucket_span": "60s", "detectors": [{"function": "high_count", "by_field_name": "host", "custom_rules": rules}]}), None);
    let mut docs = Vec::new();
    for i in 0..30i64 {
        let n = if i == 25 { 300 } else { 10 };
        docs.extend((0..n).map(|k| json!({"time": b(i) + k, "host": "host-a"})));
    }
    // Spike at 25 with host-a trusted: skipped.
    let first: Vec<Value> = docs.iter().filter(|d| d["time"].as_i64().unwrap() < b(27)).cloned().collect();
    super::super::anomaly::post_data(&node, "j", &ndjson(&first)).unwrap();
    super::super::anomaly::flush_job(&node, "j", &super::super::anomaly::FlushParams { advance_time: Some(b(27)), ..Default::default() }).unwrap();
    assert!(results(&node, "j", "record").is_empty(), "trusted host-a is skipped");
    // host-a removed from the filter while the job runs: a later spike scores.
    super::super::calendar::update_filter(&node, "trusted", &json!({"remove_items": ["host-a"]})).unwrap();
    let mut later: Vec<Value> = (27..29).flat_map(|i| (0..10).map(move |k| json!({"time": b(i) + k, "host": "host-a"}))).collect();
    later.extend((0..3000).map(|k| json!({"time": b(29) + k, "host": "host-a"})));
    super::super::anomaly::post_data(&node, "j", &ndjson(&later)).unwrap();
    super::super::anomaly::flush_job(&node, "j", &super::super::anomaly::FlushParams { advance_time: Some(b(30)), ..Default::default() }).unwrap();
    let recs = results(&node, "j", "record");
    assert!(recs.iter().any(|r| r["timestamp"] == json!(b(29))), "the update applies: {recs:?}");
}

/// Rule filters (and scheduled events, memory limits) of a live job are
/// kept per node: another node in the same process with a job of the same
/// id, or a filter update there, never changes this job's (the
/// `filter_updates_reach_running_jobs` flake: a concurrent test's job "j"
/// overwrote the shared entry).
#[test]
fn rule_filters_are_scoped_to_the_node() {
    let (da, db) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (a, b2) = (Node::open(da.path()).unwrap(), Node::open(db.path()).unwrap());
    let rules = json!([{"actions": ["skip_result"], "scope": {"host": {"filter_id": "trusted", "filter_type": "include"}}}]);
    super::super::calendar::put_filter(&a, "trusted", &json!({"items": ["host-a"]})).unwrap();
    super::super::calendar::put_filter(&b2, "trusted", &json!({"items": ["host-z"]})).unwrap();
    for node in [&a, &b2] {
        put_and_open(node, "j", json!({"bucket_span": "60s", "detectors": [{"function": "high_count", "by_field_name": "host", "custom_rules": rules}]}), None);
        let docs: Vec<Value> = (0..3).map(|i| json!({"time": b(i), "host": "host-a"})).collect();
        super::super::anomaly::post_data(node, "j", &ndjson(&docs)).unwrap();
    }
    super::super::calendar::update_filter(&b2, "trusted", &json!({"add_items": ["host-y"]})).unwrap();
    let items = |node: &Node| FILTERS.lock().get(&key(node, "j")).and_then(|f| f.get("trusted").cloned()).unwrap_or_default();
    assert_eq!(items(&a), vec!["host-a".to_string()], "node a's job keeps its own filter");
    let mut bi = items(&b2);
    bi.sort();
    assert_eq!(bi, vec!["host-y".to_string(), "host-z".to_string()], "node b's job sees its update");
    for node in [&a, &b2] {
        super::super::anomaly::close_job(node, "j", false, false).unwrap();
    }
}

/// #561: entities of one population bucket are scored against the same
/// model state; renaming them never changes which ones are anomalous.
#[test]
fn population_bucket_scored_before_training() {
    let run = |names: [&str; 2]| -> Vec<String> {
        let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "mean", "field_name": "v", "over_field_name": "host"}]}));
        let mut p = Proc::new(&job);
        let mut docs = Vec::new();
        for i in 0..20 {
            for n in names {
                docs.push(json!({"time": b(i), "host": n, "v": 1.0 + (i % 3) as f64 * 0.01}));
            }
        }
        for n in names {
            docs.push(json!({"time": b(20), "host": n, "v": 1000.0}));
        }
        let out = feed(&mut p, docs, b(21));
        let mut hosts: Vec<String> = records_of(&out).iter().filter(|r| r["timestamp"] == json!(b(20))).filter_map(|r| r["over_field_value"].as_str().map(str::to_string)).collect();
        hosts.sort();
        hosts
    };
    assert_eq!(run(["alpha", "beta"]), vec!["alpha", "beta"], "both identical outliers");
    assert_eq!(run(["zeta", "eta"]), vec!["eta", "zeta"]);
}

/// #562: an influencer value that contributes nothing to an anomalous sum is
/// not published as influencing it; the causal one is.
#[test]
fn zero_contribution_influencers_are_not_blamed() {
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "high_sum", "field_name": "amount"}], "influencers": ["host"]}));
    let mut p = Proc::new(&job);
    let mut docs = Vec::new();
    for i in 0..30 {
        let a = if i == 29 { 1000.0 } else { 10.0 + (i % 3) as f64 };
        docs.push(json!({"time": b(i), "host": "A", "amount": a}));
        docs.push(json!({"time": b(i) + 1, "host": "B", "amount": 0.0}));
    }
    let out = feed(&mut p, docs, b(30));
    let infl: Vec<&str> = out.iter().filter(|(_, d)| d["result_type"] == "influencer" && d["timestamp"] == json!(b(29))).filter_map(|(_, d)| d["influencer_field_value"].as_str()).collect();
    assert_eq!(infl, vec!["A"], "only the causal host");
    let rec = records_of(&out).into_iter().find(|r| r["timestamp"] == json!(b(29))).expect("anomalous bucket");
    assert_eq!(rec["influencers"][0]["influencer_field_values"], json!(["A"]));
}

/// #588: an aligned `skip_time` resumes at that bucket (ceiling), so its
/// records are analyzed; unaligned times round up as before.
#[test]
fn aligned_skip_time_resumes_at_itself() {
    let job = job_with(json!({"bucket_span": "1h", "detectors": [{"function": "count"}]}));
    let mut p = Proc::new(&job);
    let mut out = Vec::new();
    p.ingest(vec![json!({"time": T0})], 10, true, &mut out);
    let skip = T0 + 10 * 3_600_000 - (T0 % 3_600_000);
    p.flush(&super::super::anomaly::FlushParams { skip_time: Some(skip), ..Default::default() }, &mut out);
    p.ingest(vec![json!({"time": skip}), json!({"time": skip + 59 * 60_000})], 10, true, &mut out);
    assert_eq!(p.counts.out_of_order_timestamp_count, 0, "records at the skip time are kept");
    assert_eq!(p.current_bucket, Some(skip));
    p.flush(&super::super::anomaly::FlushParams { skip_time: Some(skip + 3_600_000 + 1), ..Default::default() }, &mut out);
    assert_eq!(p.current_bucket, Some(skip + 2 * 3_600_000), "an unaligned skip rounds up");
}

/// #589: a custom categorization analyzer (here `whitespace` with a
/// `pattern_replace` char filter removing a variable statement) is what
/// categorization tokenizes with.
#[test]
fn categorization_uses_the_configured_analyzer() {
    let analyzer = json!({"char_filter": [{"type": "pattern_replace", "pattern": "\\[statement:.*\\]", "replacement": ""}], "tokenizer": "whitespace"});
    let job = json!({"job_id": "c", "analysis_config": {"bucket_span": "60s", "categorization_field_name": "msg", "detectors": [{"function": "count", "by_field_name": "mlcategory"}],
        "categorization_analyzer": analyzer}, "data_description": {"time_field": "time"}});
    let mut p = Proc::new(&job);
    let a = p.categorize("query failed [statement:select alpha beta gamma delta]", None).unwrap();
    let b = p.categorize("query failed [statement:delete epsilon zeta eta theta]", None).unwrap();
    assert_eq!(a, b, "one category once the statement is removed");
    // The tokenizer is the analyzer's: whitespace keeps `failed,` whole.
    let c = p.categorize("x-1 y_2", None).unwrap();
    assert!(p.categories.iter().any(|cat| cat.id == c && cat.tokens == vec!["x-1".to_string(), "y_2".to_string()]), "{:?}", p.categories.iter().map(|c| &c.tokens).collect::<Vec<_>>());
}

/// #590: buckets whose rare values are all excluded from model updates do
/// not move the rare model's denominator either.
#[test]
fn rare_skip_model_update_freezes_the_total() {
    let rules = json!([{"actions": ["skip_result", "skip_model_update"], "conditions": [{"applies_to": "time", "operator": "gte", "value": (b(25) / 1000) as f64}, {"applies_to": "time", "operator": "lt", "value": (b(225) / 1000) as f64}]}]);
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "rare", "by_field_name": "status", "custom_rules": rules}]}));
    let mut p = Proc::new(&job);
    let docs: Vec<Value> = (0..226).map(|i| json!({"time": b(i), "status": "A"})).collect();
    let out = feed(&mut p, docs, b(227));
    let total = p.frequency_total.get("0|").copied().unwrap_or(0);
    // Buckets 0-24, 225 and the empty 226; not the 200 under the rule.
    assert_eq!(total, 27, "the rule-excluded buckets leave the total alone");
    assert!(!records_of(&out).iter().any(|r| r["timestamp"] == json!(b(225))), "A after the maintenance is ordinary");
}

/// #619: population `rare` measures how few members had the value: a status
/// only one client ever has stays rare however often that client has it.
#[test]
fn population_rare_measures_member_prevalence() {
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "rare", "by_field_name": "status", "over_field_name": "client"}]}));
    let mut p = Proc::new(&job);
    let mut docs = Vec::new();
    for i in 0..30 {
        for c in 0..1500 {
            docs.push(json!({"time": b(i), "client": format!("c{c}"), "status": "A"}));
        }
        docs.push(json!({"time": b(i) + 1, "client": "odd", "status": "B"}));
    }
    let out = feed(&mut p, docs, b(30));
    let late = records_of(&out).into_iter().filter(|r| r["timestamp"].as_i64().unwrap() >= b(25)).filter(|r| r["by_field_value"] == "B").count();
    assert!(late >= 4, "the unique status stays rare in the population: {late}");
    assert!(!records_of(&out).iter().any(|r| r["by_field_value"] == "A" && r["timestamp"].as_i64().unwrap() >= b(25)));
}

/// #622: multivalued dimension values are one sorted, deduplicated value.
#[test]
fn multivalued_dimensions_stay_distinct() {
    let mut m = Map::new();
    flatten(&json!({"tags": ["worker-b", "shared", "worker-b"], "n": [3, 4]}), "", &mut m);
    assert_eq!(m["tags"], json!("shared,worker-b"));
    assert_eq!(m["n"], json!(3), "numbers keep their first value");
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "low_count", "by_field_name": "tags"}]}));
    let mut p = Proc::new(&job);
    let mut docs = Vec::new();
    for i in 0..31 {
        let (a, bb) = if i == 30 { (200, 0) } else { (100, 100) };
        docs.extend((0..a).map(|k| json!({"time": b(i) + k, "tags": ["shared", "worker-a"]})));
        docs.extend((0..bb).map(|k| json!({"time": b(i) + 500 + k, "tags": ["shared", "worker-b"]})));
    }
    let out = feed(&mut p, docs, b(31));
    assert!(p.models.keys().any(|k| k.ends_with("|shared,worker-b")), "{:?}", p.models.keys().collect::<Vec<_>>());
    assert!(records_of(&out).iter().any(|r| r["by_field_value"] == "shared,worker-b" && r["timestamp"] == json!(b(30))), "the failed worker is a low count");
}

/// #595: a request admitted before a close cannot start a replacement
/// process of the closed job; an open lifts the fence.
#[test]
fn closed_jobs_get_no_replacement_process() {
    let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count"}]}), None);
    super::super::anomaly::post_data(&node, "j", &ndjson(&[json!({"time": b(0)})])).unwrap();
    close(&node, "j").unwrap();
    // The delayed caller reaches processing after the close.
    let e = post_docs(&node, "j", vec![json!({"time": b(1)})], 10, true).unwrap_err();
    assert_eq!(e.status, 409, "{e:?}");
    assert!(live(&node, "j").is_none(), "no replacement process");
    open(&node, "j").unwrap();
    assert!(post_docs(&node, "j", vec![json!({"time": b(1)})], 10, true).is_ok());
}

/// #594: interim results of a flush never overwrite the finalized results
/// of a concurrent ingest (calculation and write are ordered per job).
#[test]
fn concurrent_flush_never_leaves_stale_interim_results() {
    let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count"}]}), None);
    let node2 = node.clone();
    let flusher = std::thread::spawn(move || {
        for _ in 0..40 {
            let _ = super::super::anomaly::flush_job(&node2, "j", &super::super::anomaly::FlushParams { calc_interim: true, ..Default::default() });
        }
    });
    for i in 0..40 {
        let docs: Vec<Value> = (0..5).map(|k| json!({"time": b(i) + k})).collect();
        super::super::anomaly::post_data(&node, "j", &ndjson(&docs)).unwrap();
    }
    flusher.join().unwrap();
    super::super::anomaly::flush_job(&node, "j", &super::super::anomaly::FlushParams { advance_time: Some(b(41)), ..Default::default() }).unwrap();
    let stale: Vec<Value> = results(&node, "j", "bucket").into_iter().filter(|b| b["is_interim"] == json!(true)).collect();
    assert!(stale.is_empty(), "finalized buckets stay final: {stale:?}");
}

/// #620: `_delete_expired_data` applies the snapshot retention relative to
/// the newest snapshot: older than `model_snapshot_retention_days` gone,
/// older than `daily_model_snapshot_retention_after_days` thinned to the
/// first of each day, `retain: true` and recent snapshots kept, the state
/// documents of removed snapshots deleted with them.
#[test]
fn expired_model_snapshots_and_states_are_removed() {
    let (_d, node) = open_job(json!({"bucket_span": "60s", "detectors": [{"function": "count"}]}), None);
    let job = service(&node).read().jobs["j"].clone();
    const DAY: i64 = 86_400_000;
    let day = |d: i64, h: i64| 1_700_006_400_000 + d * DAY + h * 3_600_000;
    let snaps = [("old", day(5, 12), false), ("old-retained", day(6, 12), true), ("thin-a", day(15, 8), false), ("thin-b", day(15, 20), false),
        ("recent-1", day(19, 12), false), ("recent-2", day(19, 18), false), ("newest", day(20, 12), false)];
    let docs: Docs = snaps.iter().map(|(sid, ts, retain)| (super::super::results::snapshot_doc_id("j", sid),
        json!({"job_id": "j", "snapshot_id": sid, "timestamp": ts, "snapshot_doc_count": 1, "retain": retain, "min_version": SNAPSHOT_MIN_VERSION}))).collect();
    write_docs(&node, "j", &job, docs).map_err(|(e, _)| e).unwrap();
    ridx::write_state(&node, snaps.iter().map(|(sid, _, _)| (format!("j_model_state_{sid}#1"), json!({STATE_KEY: {}}))).collect()).unwrap();
    super::super::anomaly::delete_expired_data(&node, Some("j")).unwrap();
    let kept: BTreeSet<String> = ridx::search_all(&node, &read_alias("j"), json!({"exists": {"field": "snapshot_doc_count"}}), json!(["_doc"])).unwrap()
        .into_iter().filter_map(|(_, d)| d["snapshot_id"].as_str().map(str::to_string)).collect();
    let want: BTreeSet<String> = ["old-retained", "thin-a", "recent-1", "recent-2", "newest"].iter().map(|s| s.to_string()).collect();
    assert_eq!(kept, want);
    let all: Vec<String> = snaps.iter().map(|(sid, _, _)| format!("j_model_state_{sid}#1")).collect();
    let states: BTreeSet<String> = ridx::search_all(&node, ridx::STATE_PATTERN, json!({"ids": {"values": all}}), json!(["_doc"])).unwrap().into_iter().map(|(id, _)| id).collect();
    for (sid, _, _) in snaps {
        assert_eq!(states.contains(&format!("j_model_state_{sid}#1")), want.contains(sid), "{sid}: {states:?}");
    }
}

/// #621: a lazily opened job waiting for ML memory is assigned once the
/// node's allowance grows (an open retry or the reconciliation), exactly
/// once, and gets its process.
#[test]
fn lazy_open_job_is_assigned_when_capacity_appears() {
    let data = tempfile::tempdir().unwrap();
    let node = Node::open(data.path()).unwrap();
    node.cluster_settings.write().1.insert("xpack.ml.max_machine_memory_percent".into(), json!("1"));
    let max = super::super::anomaly::ml_max_memory(&node);
    let mb = (max >> 20) + 64;
    let job = json!({"analysis_config": {"bucket_span": "60s", "detectors": [{"function": "count"}]}, "analysis_limits": {"model_memory_limit": format!("{mb}mb")},
        "allow_lazy_open": true, "data_description": {"time_field": "time"}});
    super::super::anomaly::put_job(&node, "lazy", &job, &job.to_string(), &Default::default()).unwrap();
    let r = super::super::anomaly::open_job(&node, "lazy").unwrap();
    assert_eq!(r["node"], json!(""), "{r}");
    assert_eq!(service(&node).read().job_state("lazy"), "opening");
    assert!(super::super::anomaly::assign_pending_jobs(&node).is_empty(), "still too small");
    // More ML memory: the explicit open retry assigns it.
    node.cluster_settings.write().1.insert("xpack.ml.max_machine_memory_percent".into(), json!("90"));
    let r = super::super::anomaly::open_job(&node, "lazy").unwrap();
    assert_eq!(r["node"], json!(crate::node::NODE_ID), "{r}");
    assert_eq!(service(&node).read().job_state("lazy"), "opened");
    assert!(live(&node, "lazy").is_some(), "its process runs");
    assert!(super::super::anomaly::assign_pending_jobs(&node).is_empty(), "assigned once");
    let r = super::super::anomaly::open_job(&node, "lazy").unwrap();
    assert_eq!(r["node"], json!(crate::node::NODE_ID), "idempotent");
}

/// ml-cpp's `exclude_frequent` thresholds: a person (the by value of an
/// individual model, the over value of a population) is frequent when its
/// share of the buckets is greater than 0.1; a population attribute (by
/// value) when its share of the people is greater than 0.1.
#[test]
fn exclude_frequent_uses_ml_cpp_frequencies() {
    let job = job_with(json!({"bucket_span": "60s", "detectors": [{"function": "count", "by_field_name": "status", "over_field_name": "host", "exclude_frequent": "all"}]}));
    let mut p = Proc::new(&job);
    let d = p.detectors[0].clone();
    let sv = |by: &str, over: &str| SeriesValue { partition: None, by: Some(by.into()), over: Some(over.into()), value: 1.0, point: None, extremes: None, influencers: BTreeMap::new(), without: BTreeMap::new() };
    // Before any bucket both frequencies are 0.5: excluded, as in ml-cpp.
    assert!(p.is_frequent(&d, &sv("a", "h0")));
    // 20 buckets: host h0 in all of them with status `common`; 30 other
    // hosts once each, all with `common`; host h1 once with `odd`.
    for i in 0..20 {
        let mut vals = vec![sv("common", "h0")];
        if i < 30 { vals.push(sv("common", &format!("x{i}"))); }
        if i == 3 { vals.push(sv("odd", "h1")); }
        p.count_entities(&d, &vals);
    }
    for i in 20..30 { p.count_entities(&d, &[sv("common", &format!("x{i}"))]); }
    assert!(p.is_frequent(&d, &sv("odd", "h0")), "h0 is in 20 of 30 buckets: frequent person");
    assert!(p.is_frequent(&d, &sv("common", "x5")), "`common` is shared by nearly every host: frequent attribute");
    assert!(!p.is_frequent(&d, &sv("odd", "h1")), "h1 (1 of 30 buckets) with `odd` (1 of 32 hosts) is neither");
}

/// The job reset / delete path removes the model states of every snapshot
/// (found through the snapshot descriptors, `_id` takes no prefix query)
/// and the quantiles, not the states of other jobs.
#[test]
fn job_reset_removes_its_model_states() {
    let data = tempfile::tempdir().unwrap();
    let node = Node::open(data.path()).unwrap();
    for id in ["j", "other"] {
        put_and_open(&node, id, json!({"bucket_span": "60s", "detectors": [{"function": "count"}]}), None);
        let docs: Vec<Value> = (0..5).map(|i| json!({"time": b(i)})).collect();
        super::super::anomaly::post_data(&node, id, &ndjson(&docs)).unwrap();
        super::super::anomaly::close_job(&node, id, false, false).unwrap();
    }
    let states = |job: &str| -> Vec<String> {
        let snaps = ridx::search_all(&node, &read_alias(job), json!({"exists": {"field": "snapshot_doc_count"}}), json!(["_doc"])).unwrap_or_default();
        let ids: Vec<String> = snaps.iter().filter_map(|(_, d)| d["snapshot_id"].as_str().map(|s| format!("{job}_model_state_{s}#1"))).chain(std::iter::once(format!("{job}_quantiles"))).collect();
        ridx::search_all(&node, ridx::STATE_PATTERN, json!({"ids": {"values": ids}}), json!(["_doc"])).unwrap().into_iter().map(|(id, _)| id).collect()
    };
    let before = states("j");
    assert!(before.iter().any(|s| s.starts_with("j_model_state_")), "a closed job has a model state: {before:?}");
    let other = states("other");
    let all_j: Vec<String> = before.clone();
    super::super::anomaly::reset_job(&node, "j").unwrap();
    let left = ridx::search_all(&node, ridx::STATE_PATTERN, json!({"ids": {"values": all_j}}), json!(["_doc"])).unwrap();
    assert!(left.is_empty(), "reset removed j's states: {left:?}");
    assert_eq!(states("other"), other, "other jobs keep theirs");
}
