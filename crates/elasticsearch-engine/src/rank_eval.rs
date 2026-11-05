//! `_rank_eval`: run rated search requests and score the rankings with
//! precision@k, recall@k, MRR, DCG/nDCG or ERR.

use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub enum Metric {
    Precision { k: usize, threshold: i64, ignore_unlabeled: bool },
    Recall { k: usize, threshold: i64 },
    Mrr { k: usize, threshold: i64 },
    Dcg { k: usize, normalize: bool, unknown: Option<i64> },
    Err { k: usize, max_rel: i64, unknown: Option<i64> },
}

impl Metric {
    pub fn parse(v: Option<&Value>) -> Result<Metric> {
        let Some(Value::Object(m)) = v else {
            return Ok(Metric::Dcg { k: 10, normalize: false, unknown: None });
        };
        let (name, o) = m.iter().next().ok_or_else(|| EsError::parsing("[metric] cannot be empty"))?;
        let k = o.get("k").and_then(Value::as_u64).unwrap_or(10) as usize;
        let threshold = o.get("relevant_rating_threshold").and_then(Value::as_i64).unwrap_or(1);
        let unknown = o.get("unknown_doc_rating").and_then(Value::as_i64);
        Ok(match name.as_str() {
            "precision" => Metric::Precision { k, threshold, ignore_unlabeled: o.get("ignore_unlabeled").and_then(Value::as_bool).unwrap_or(false) },
            "recall" => Metric::Recall { k, threshold },
            "mean_reciprocal_rank" => Metric::Mrr { k, threshold },
            "dcg" => Metric::Dcg { k, normalize: o.get("normalize").and_then(Value::as_bool).unwrap_or(false), unknown },
            "expected_reciprocal_rank" => Metric::Err {
                k,
                max_rel: o.get("maximum_relevance").and_then(Value::as_i64).ok_or_else(|| EsError::parsing("[expected_reciprocal_rank] requires [maximum_relevance]"))?,
                unknown,
            },
            other => return Err(EsError::parsing(format!("[rank_eval] unknown metric [{other}]"))),
        })
    }

    pub fn k(&self) -> usize {
        match self {
            Metric::Precision { k, .. } | Metric::Recall { k, .. } | Metric::Mrr { k, .. } | Metric::Dcg { k, .. } | Metric::Err { k, .. } => *k,
        }
    }
}

fn dcg(ratings: &[Option<i64>]) -> f64 {
    ratings.iter().enumerate().map(|(i, r)| r.map(|r| (2f64.powi(r as i32) - 1.0) / ((i + 2) as f64).log2()).unwrap_or(0.0)).sum()
}

/// Score one request: `(score, metric_details)`.
pub fn evaluate(metric: &Metric, hits: &[(String, String)], ratings: &HashMap<(String, String), i64>) -> (f64, Value) {
    let rated: Vec<Option<i64>> = hits.iter().map(|h| ratings.get(h).copied()).collect();
    let unrated = rated.iter().filter(|r| r.is_none()).count();
    match metric {
        Metric::Precision { threshold, ignore_unlabeled, .. } => {
            let relevant = rated.iter().filter(|r| r.map(|x| x >= *threshold).unwrap_or(false)).count();
            let retrieved = if *ignore_unlabeled { rated.iter().filter(|r| r.is_some()).count() } else { rated.len() };
            let score = if retrieved == 0 { 0.0 } else { relevant as f64 / retrieved as f64 };
            (score, json!({"precision": {"relevant_docs_retrieved": relevant, "docs_retrieved": retrieved}}))
        }
        Metric::Recall { threshold, .. } => {
            let relevant = rated.iter().filter(|r| r.map(|x| x >= *threshold).unwrap_or(false)).count();
            let total = ratings.values().filter(|r| **r >= *threshold).count();
            let score = if total == 0 { 0.0 } else { relevant as f64 / total as f64 };
            (score, json!({"recall": {"relevant_docs_retrieved": relevant, "relevant_docs": total}}))
        }
        Metric::Mrr { threshold, .. } => {
            let first = rated.iter().position(|r| r.map(|x| x >= *threshold).unwrap_or(false));
            let score = first.map(|p| 1.0 / (p + 1) as f64).unwrap_or(0.0);
            (score, json!({"mean_reciprocal_rank": {"first_relevant": first.map(|p| (p + 1) as i64).unwrap_or(-1)}}))
        }
        Metric::Dcg { k, normalize, unknown } => {
            let r: Vec<Option<i64>> = rated.iter().map(|x| x.or(*unknown)).collect();
            let d = dcg(&r);
            let mut ideal: Vec<i64> = ratings.values().copied().collect();
            ideal.sort_unstable_by(|a, b| b.cmp(a));
            ideal.truncate(*k);
            let idcg = dcg(&ideal.iter().map(|x| Some(*x)).collect::<Vec<_>>());
            let nd = if idcg == 0.0 { 0.0 } else { d / idcg };
            let score = if *normalize { nd } else { d };
            (score, json!({"dcg": {"dcg": d, "ideal_dcg": idcg, "normalized_dcg": nd, "unrated_docs": unrated}}))
        }
        Metric::Err { max_rel, unknown, .. } => {
            let mut p = 1.0;
            let mut err = 0.0;
            for (i, r) in rated.iter().enumerate() {
                let Some(g) = r.or(*unknown) else { continue };
                let prob = (2f64.powi(g as i32) - 1.0) / 2f64.powi(*max_rel as i32);
                err += p * prob / (i + 1) as f64;
                p *= 1.0 - prob;
            }
            (err, json!({"expected_reciprocal_rank": {"unrated_docs": unrated}}))
        }
    }
}

/// Run a `_rank_eval` body. `search(body, request_id)` executes one search
/// request and returns the search response.
pub fn run(body: &Value, search: &dyn Fn(&Value) -> Result<Value>, render: &dyn Fn(&str, &Value) -> Result<Value>) -> Result<Value> {
    let metric = Metric::parse(body.get("metric"))?;
    let k = metric.k();
    let requests = body.get("requests").and_then(Value::as_array).ok_or_else(|| EsError::parsing("[rank_eval] requires [requests]"))?;
    let mut templates: HashMap<String, Value> = HashMap::new();
    if let Some(ts) = body.get("templates").and_then(Value::as_array) {
        for t in ts {
            if let (Some(id), Some(tpl)) = (t.get("id").and_then(Value::as_str), t.get("template")) {
                templates.insert(id.to_string(), tpl.get("source").cloned().unwrap_or_else(|| tpl.clone()));
            }
        }
    }
    let mut details = Map::new();
    let mut failures = Map::new();
    let mut sum = 0.0;
    let mut n = 0usize;
    for r in requests {
        let id = r.get("id").and_then(Value::as_str).ok_or_else(|| EsError::parsing("[request] requires [id]"))?.to_string();
        let ratings_v = r.get("ratings").and_then(Value::as_array).ok_or_else(|| EsError::parsing(format!("[request {id}] requires [ratings]")))?;
        let mut ratings = HashMap::new();
        for rt in ratings_v {
            let ix = rt.get("_index").and_then(Value::as_str).unwrap_or("").to_string();
            let did = rt.get("_id").and_then(Value::as_str).unwrap_or("").to_string();
            let rating = rt.get("rating").and_then(Value::as_i64).unwrap_or(0);
            ratings.insert((ix, did), rating);
        }
        let req_body = match (r.get("request"), r.get("template_id").and_then(Value::as_str)) {
            (Some(b), _) => Ok(b.clone()),
            (None, Some(tid)) => match templates.get(tid) {
                Some(src) => render(&crate::template::source_text(src), r.get("params").unwrap_or(&Value::Null)),
                None => Err(EsError::illegal_argument(format!("Template with id [{tid}] not found"))),
            },
            _ => Err(EsError::parsing(format!("[request {id}] requires [request] or [template_id]"))),
        };
        let res = req_body.and_then(|mut b| {
            if let Some(o) = b.as_object_mut() {
                o.insert("size".into(), json!(k));
                o.remove("from");
            }
            search(&b)
        });
        match res {
            Ok(resp) => {
                let empty = Vec::new();
                let hits_v = resp["hits"]["hits"].as_array().unwrap_or(&empty);
                let hits: Vec<(String, String)> = hits_v
                    .iter()
                    .take(k)
                    .map(|h| (h["_index"].as_str().unwrap_or("").to_string(), h["_id"].as_str().unwrap_or("").to_string()))
                    .collect();
                let (score, md) = evaluate(&metric, &hits, &ratings);
                let unrated: Vec<Value> = hits.iter().filter(|h| !ratings.contains_key(*h)).map(|(i, d)| json!({"_index": i, "_id": d})).collect();
                let hit_list: Vec<Value> = hits_v
                    .iter()
                    .take(k)
                    .map(|h| {
                        let key = (h["_index"].as_str().unwrap_or("").to_string(), h["_id"].as_str().unwrap_or("").to_string());
                        json!({"hit": {"_index": key.0, "_id": key.1, "_score": h["_score"]}, "rating": ratings.get(&key)})
                    })
                    .collect();
                details.insert(id, json!({"metric_score": score, "unrated_docs": unrated, "hits": hit_list, "metric_details": md}));
                sum += score;
                n += 1;
            }
            Err(e) => {
                failures.insert(id, e.error_object(false));
            }
        }
    }
    let total = requests.len().max(1);
    let _ = n;
    Ok(json!({"metric_score": sum / total as f64, "details": details, "failures": failures}))
}
