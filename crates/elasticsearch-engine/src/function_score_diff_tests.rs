//! `function_score` execution paths against the generic scorer: the script
//! path (`script_score` functions), the geo-decay path, and the generic core
//! scorer must give the same hits and scores for every `score_mode` and
//! `boost_mode`, with weights, filters, `max_boost`, `min_score` and query
//! boosts (ES `FunctionScoreQuery` semantics: `avg` is Σ(wᵢ·fᵢ)/Σwᵢ over
//! the functions whose filter matches; the boost is folded into the
//! sub-query's score; `min_score` reads the combined score).

use crate::index::{IndexService, WriteReq};
use crate::settings;
use crate::shard::WriteCond;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

struct Ix {
    _dir: tempfile::TempDir,
    svc: Arc<IndexService>,
}

fn index() -> Ix {
    let dir = tempfile::tempdir().unwrap();
    let s = json!({"number_of_shards": 1, "number_of_replicas": 0});
    let mappings = json!({"properties": {
        "txt": {"type": "text"}, "tag": {"type": "keyword"},
        "x": {"type": "double"}, "y": {"type": "double"}, "loc": {"type": "geo_point"}
    }});
    let svc = Arc::new(IndexService::create(dir.path(), "fs", settings::normalize_index_settings(&s), &mappings, BTreeMap::new()).unwrap());
    let docs = [
        ("1", "a", 1.0, 3.0, "foo"),
        ("2", "a", 2.0, 0.5, "foo foo"),
        ("3", "b", 3.0, 1.0, "foo bar"),
        ("4", "b", 0.5, 2.0, "foo foo foo"),
        ("5", "c", 4.0, 4.0, "foo baz baz"),
        ("6", "a", 1.5, 1.5, "foo"),
    ];
    let reqs: Vec<WriteReq> = docs
        .iter()
        .map(|(id, tag, x, y, txt)| WriteReq::Index {
            id: Some(id.to_string()),
            source: json!({"tag": tag, "x": x, "y": y, "txt": txt, "loc": {"lat": 52.0, "lon": 4.0}}).to_string(),
            routing: None,
            cond: WriteCond::default(),
        })
        .collect();
    for r in svc.write(reqs, true) {
        r.unwrap();
    }
    svc.refresh().unwrap();
    Ix { _dir: dir, svc }
}

/// `(id, score)` of every hit, best first (ties by id).
fn hits(ix: &Ix, body: Value) -> Vec<(String, f32)> {
    let req = crate::search::parse_request(&body, &std::collections::HashMap::new()).unwrap();
    let targets = vec![crate::search::Target::new(ix.svc.clone())];
    let opts = crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65536, scroll_id: None, pit_id: None };
    let out = crate::search::execute(&targets, &req, &opts).unwrap();
    let v: Value = serde_json::from_str(&out.json).unwrap();
    let mut h: Vec<(String, f32)> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| (h["_id"].as_str().unwrap().to_string(), h["_score"].as_f64().unwrap() as f32)).collect();
    h.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    h
}

fn same(a: &[(String, f32)], b: &[(String, f32)], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.0, y.0, "{what}: {a:?} vs {b:?}");
        let tol = 1e-5 * x.1.abs().max(1.0);
        assert!((x.1 - y.1).abs() <= tol, "{what}: {a:?} vs {b:?}");
    }
}

const SCORE_MODES: [&str; 6] = ["multiply", "sum", "avg", "first", "max", "min"];
const BOOST_MODES: [&str; 6] = ["multiply", "replace", "sum", "avg", "max", "min"];

fn body(functions: Vec<Value>, sm: &str, bm: &str, extra: &Value) -> Value {
    let mut fs = json!({"query": {"match": {"txt": "foo"}}, "functions": functions, "score_mode": sm, "boost_mode": bm});
    for (k, v) in extra.as_object().unwrap() {
        fs[k] = v.clone();
    }
    json!({"size": 10, "query": {"function_score": fs}})
}

/// The variations every combination is checked under.
fn extras() -> Vec<Value> {
    vec![
        json!({}),
        json!({"max_boost": 2.5}),
        json!({"min_score": 1.2}),
        json!({"boost": 2.0}),
        json!({"boost": 0.5, "min_score": 0.6, "max_boost": 5}),
    ]
}

#[test]
fn script_path_matches_the_generic_scorer() {
    let ix = index();
    // Generic: field_value_factor functions; script: the same values from
    // `script_score` functions, with the same weights and filters.
    let generic = vec![
        json!({"filter": {"term": {"tag": "a"}}, "field_value_factor": {"field": "x"}, "weight": 2}),
        json!({"field_value_factor": {"field": "y"}, "weight": 4}),
        json!({"filter": {"term": {"tag": "b"}}, "weight": 3}),
        json!({"filter": {"term": {"tag": "c"}}, "field_value_factor": {"field": "x", "factor": 0.5}}),
    ];
    let script = vec![
        json!({"filter": {"term": {"tag": "a"}}, "script_score": {"script": "doc['x'].value"}, "weight": 2}),
        json!({"script_score": {"script": "doc['y'].value"}, "weight": 4}),
        json!({"filter": {"term": {"tag": "b"}}, "weight": 3}),
        json!({"filter": {"term": {"tag": "c"}}, "script_score": {"script": "doc['x'].value * 0.5"}}),
    ];
    // A mix of the two in the script path (core functions there too).
    let mixed = vec![generic[0].clone(), script[1].clone(), generic[2].clone(), script[3].clone()];
    for sm in SCORE_MODES {
        for bm in BOOST_MODES {
            for extra in extras() {
                let want = hits(&ix, body(generic.clone(), sm, bm, &extra));
                let what = format!("score_mode {sm} boost_mode {bm} {extra}");
                same(&want, &hits(&ix, body(script.clone(), sm, bm, &extra)), &format!("script: {what}"));
                same(&want, &hits(&ix, body(mixed.clone(), sm, bm, &extra)), &format!("mixed: {what}"));
                // Under an enclosing boost (a bool clause's), too.
                let outer = |b: Value| json!({"size": 10, "query": {"bool": {"must": [b["query"].clone()], "boost": 3.0}}});
                same(&hits(&ix, outer(body(generic.clone(), sm, bm, &extra))), &hits(&ix, outer(body(script.clone(), sm, bm, &extra))), &format!("outer boost: {what}"));
            }
        }
    }
}

#[test]
fn geo_decay_path_matches_the_generic_scorer() {
    let ix = index();
    // Every doc sits at the decay's origin: the decay is 1, so a weighted
    // decay is its weight, and the geo path must equal a `weight` function.
    let decay = |w: f64, filter: Option<Value>| {
        let mut f = json!({"gauss": {"loc": {"origin": {"lat": 52.0, "lon": 4.0}, "scale": "10km"}}, "weight": w});
        if let Some(q) = filter {
            f["filter"] = q;
        }
        f
    };
    let generic = vec![
        json!({"filter": {"term": {"tag": "a"}}, "field_value_factor": {"field": "x"}, "weight": 2}),
        json!({"weight": 4}),
        json!({"filter": {"term": {"tag": "b"}}, "weight": 3}),
    ];
    let geo = vec![generic[0].clone(), decay(4.0, None), decay(3.0, Some(json!({"term": {"tag": "b"}})))];
    for sm in SCORE_MODES {
        for bm in BOOST_MODES {
            for extra in extras() {
                let what = format!("score_mode {sm} boost_mode {bm} {extra}");
                same(&hits(&ix, body(generic.clone(), sm, bm, &extra)), &hits(&ix, body(geo.clone(), sm, bm, &extra)), &format!("geo: {what}"));
            }
        }
    }
}

/// Tom's witness: script functions returning 1 and 3 with weights 2 and 4
/// average to (1·2 + 3·4) / (2 + 4), not (2 + 12) / 2; `min_score` filters
/// on that value.
#[test]
fn weighted_avg_of_script_functions_divides_by_the_weights() {
    let ix = index();
    let q = |min: Option<f64>| {
        let mut fs = json!({"query": {"match_all": {}}, "score_mode": "avg", "boost_mode": "replace", "functions": [
            {"script_score": {"script": "1"}, "weight": 2},
            {"script_score": {"script": "3"}, "weight": 4}
        ]});
        if let Some(m) = min {
            fs["min_score"] = json!(m);
        }
        json!({"size": 10, "query": {"function_score": fs}})
    };
    let h = hits(&ix, q(None));
    assert_eq!(h.len(), 6);
    assert!(h.iter().all(|(_, s)| (s - 14.0 / 6.0).abs() < 1e-5), "{h:?}");
    assert!(hits(&ix, q(Some(3.0))).is_empty(), "7 would pass, 2.33 does not");
    // A function whose filter does not match counts in neither sum.
    let fs = json!({"size": 10, "query": {"function_score": {"query": {"match_all": {}}, "score_mode": "avg", "boost_mode": "replace", "functions": [
        {"script_score": {"script": "1"}, "weight": 2},
        {"filter": {"term": {"tag": "zzz"}}, "script_score": {"script": "3"}, "weight": 4}
    ]}}});
    assert!(hits(&ix, fs).iter().all(|(_, s)| (s - 1.0).abs() < 1e-6));
}

/// `first` uses the first applicable function only: a later script that
/// would fail is never evaluated.
#[test]
fn first_evaluates_no_later_function() {
    let ix = index();
    let fs = json!({"size": 10, "query": {"function_score": {"query": {"match_all": {}}, "score_mode": "first", "boost_mode": "replace", "functions": [
        {"script_score": {"script": "2"}},
        {"script_score": {"script": "doc['nope'].value"}}
    ]}}});
    let h = hits(&ix, fs);
    assert!(h.iter().all(|(_, s)| (s - 2.0).abs() < 1e-6), "{h:?}");
}
