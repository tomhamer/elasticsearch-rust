//! The in-process `outlier_detection` analytics worker (ES runs the
//! `data_frame_analyzer` native process; this is its counterpart).
//!
//! Phases, as ES reports them: `reindexing` (source documents are copied to
//! the dest index with `ml__incremental_id`, keeping their `_id`),
//! `loading_data` (feature rows of every document that has all analysed
//! fields; the rest count as `skipped_docs_count`), `computing_outliers` and
//! `writing_results` (`<results_field>.outlier_score` and, above
//! `feature_influence_threshold`, `<results_field>.feature_influence`).
//!
//! Scores come from the same ensemble ES uses — `lof`, `ldof`,
//! `distance_kth_nn`, `distance_knn` over exact k nearest neighbours of the
//! (by default standardised) feature vectors — each normalised to [0, 1]
//! so that the `1 - outlier_fraction` quantile scores 0.5, then averaged.
//! Values are not bit-identical to ml-cpp's, which fits its own score
//! distributions. Neighbours are found exhaustively, so a job may analyse at
//! most [`MAX_ROWS`] documents (`_start` rejects larger ones).
//!
//! A stopped job keeps its progress; starting it again re-runs every phase
//! (each write is keyed by the source `_id`, so a rerun overwrites rather
//! than duplicates). A finished job ends `stopped` with every phase at 100.
//! A job that was running when its node restarted is relaunched the same
//! way by `resume_interrupted`, which the node calls every second.

use crate::error::{EsError, Result};
use crate::node::Node;
use parking_lot::{Condvar, Mutex};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Most documents one outlier detection job analyses (exact kNN is
/// quadratic in the row count).
pub const MAX_ROWS: u64 = 200_000;
const PAGE: usize = 1000;

pub(super) struct Worker {
    stop: AtomicBool,
    done: Mutex<bool>,
    completed: Condvar,
}

fn workers() -> &'static Mutex<HashMap<String, Arc<Worker>>> {
    static W: OnceLock<Mutex<HashMap<String, Arc<Worker>>>> = OnceLock::new();
    W.get_or_init(Default::default)
}

fn key(node: &Node, id: &str) -> String {
    format!("{}#{id}", node.data.display())
}

/// Is a worker of this job running?
pub fn running(node: &Node, id: &str) -> bool {
    workers().lock().get(&key(node, id)).is_some_and(|w| !*w.done.lock())
}

/// Ask the job's worker to stop and wait (up to `timeout`) for it to exit.
/// False when it is still running at the deadline.
pub fn stop_and_wait(node: &Node, id: &str, timeout: Duration) -> bool {
    let Some(w) = workers().lock().get(&key(node, id)).cloned() else { return true };
    w.stop.store(true, Ordering::Relaxed);
    let deadline = Instant::now() + timeout;
    let mut done = w.done.lock();
    while !*done {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        w.completed.wait_for(&mut done, left);
    }
    true
}

/// Stop every worker of this node (feature reset).
pub fn stop_all(node: &Node, timeout: Duration) {
    let prefix = format!("{}#", node.data.display());
    let ids: Vec<String> = workers().lock().keys().filter_map(|k| k.strip_prefix(&prefix).map(str::to_string)).collect();
    for id in ids {
        stop_and_wait(node, &id, timeout);
    }
}

/// Start the worker thread of a started job.
pub(super) fn launch(node: Arc<Node>, id: String, cfg: Value, features: Vec<String>) {
    let w = Arc::new(Worker { stop: AtomicBool::new(false), done: Mutex::new(false), completed: Condvar::new() });
    let k = key(&node, &id);
    {
        // At most one worker per job (a `_start` and a restart resume can
        // race to launch it).
        let mut all = workers().lock();
        if all.get(&k).is_some_and(|c| !*c.done.lock()) {
            return;
        }
        // #534: the job is still started (not stopped, failed or deleted
        // since the caller read it). Checked under the worker registry
        // lock: a stop or delete persists its state first and then stops
        // the registered worker, so either this sees it, or it sees this
        // worker.
        {
            let st = super::service(&node).read();
            if !st.configs.contains_key(&id) || st.state_of(&id) != "started" {
                return;
            }
        }
        all.insert(k.clone(), w.clone());
    }
    let coordinator = crate::coord::current();
    let w2 = w.clone();
    let fail_node = node.clone();
    let fail_id = id.clone();
    let spawned = std::thread::Builder::new().name(format!("ml-dfa-{id}")).spawn(move || {
        struct Done(Arc<Worker>, String);
        impl Drop for Done {
            fn drop(&mut self) {
                *self.0.done.lock() = true;
                self.0.completed.notify_all();
                let mut all = workers().lock();
                if all.get(&self.1).is_some_and(|c| Arc::ptr_eq(c, &self.0)) {
                    all.remove(&self.1);
                }
            }
        }
        let _done = Done(w2.clone(), k);
        crate::coord::scoped(coordinator, || {
            let outcome = run(&node, &id, &cfg, &features, &w2);
            finish(&node, &id, outcome, &w2);
        });
    });
    if let Err(e) = spawned {
        *w.done.lock() = true;
        w.completed.notify_all();
        workers().lock().remove(&key(&fail_node, &fail_id));
        finish(&fail_node, &fail_id, Err(EsError::internal(format!("could not start the analytics worker: {e}"))), &w);
    }
}

/// Record the terminal state: `stopped` with every phase at 100 on success,
/// `failed` with the reason on error; nothing when the job was stopped.
fn finish(node: &Node, id: &str, outcome: Result<bool>, w: &Worker) {
    match outcome {
        Ok(true) => {
            let _ = super::report_progress_if_started(node, id, |rt| {
                rt.state = "stopped".into();
                for p in super::phases("outlier_detection") {
                    rt.progress.insert((*p).to_string(), 100);
                }
            });
            audit(node, id, "info", "Finished analysis");
        }
        Ok(false) => {}
        Err(e) if !w.stop.load(Ordering::Relaxed) => {
            let reason = e.reason.clone();
            let _ = super::report_progress_if_started(node, id, |rt| {
                rt.state = "failed".into();
                rt.failure_reason = Some(reason.clone());
            });
            audit(node, id, "error", &format!("Error while running analysis: {reason}"));
        }
        Err(_) => {}
    }
}

fn audit(node: &Node, id: &str, level: &str, message: &str) {
    crate::ml::resultsidx::audit(node, id, "data_frame_analytics", level, message);
}

/// Progress of one phase, written at most every few percent.
struct Phase<'a> {
    node: &'a Node,
    id: &'a str,
    name: &'static str,
    last: u64,
}

impl<'a> Phase<'a> {
    fn new(node: &'a Node, id: &'a str, name: &'static str) -> Self {
        Phase { node, id, name, last: 0 }
    }

    fn set(&mut self, pct: u64) {
        let pct = pct.min(100);
        if pct == self.last || (pct < 100 && pct < self.last + 5) {
            return;
        }
        self.last = pct;
        let name = self.name;
        let _ = super::report_progress_if_started(self.node, self.id, |rt| {
            rt.progress.insert(name.to_string(), pct);
        });
    }
}

fn stopped(w: &Worker) -> bool {
    w.stop.load(Ordering::Relaxed)
}

/// Feature value of a `_source` field as a number (booleans as 0/1).
fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
    .filter(|x| x.is_finite())
}

/// Bytes the worker holds analysing `n` rows of `d` features with `k`
/// nearest neighbours each: the input rows and their standardised copy,
/// the `(distance, row)` neighbour lists (`16 * n * k`, the dominant term),
/// per-row feature influence, scores and per-method score vectors, and the
/// row ids. Saturating, so an absurd job asks for more than any limit.
pub fn estimate_bytes(n: u64, d: u64, k: u64) -> u64 {
    let row = d.saturating_mul(8).saturating_add(24);
    let neighbours = k.saturating_mul(16).saturating_add(24);
    let per_row = row.saturating_mul(3).saturating_add(neighbours).saturating_add(6 * 8 + 48);
    n.saturating_mul(per_row)
}

/// [`estimate_bytes`] for `n` rows under the job's analysis parameters
/// (`n_neighbors`, capped at the other rows, as the computation caps it).
pub fn estimate_for(analysis: &Value, n: u64, d: u64) -> u64 {
    let k = if n < 2 { 0 } else { Params::from(analysis, n as usize).k_max().min(n as usize - 1) as u64 };
    estimate_bytes(n, d, k)
}

/// ml-cpp `bytesToString`: kilobytes below 1mb, else megabytes, rounded up.
fn ml_bytes(v: u64) -> String {
    let kb = v.div_ceil(1024);
    if kb < 1024 { format!("{kb} kb") } else { format!("{} mb", kb.div_ceil(1024)) }
}

/// The job failure of ml-cpp's `CDataFrameAnalysisInstrumentation` when the
/// analysis needs more than `model_memory_limit` (it asks for twice the
/// requirement).
fn memory_limit_exceeded(limit: u64, need: u64) -> EsError {
    EsError::new(
        500,
        "status_exception",
        format!(
            "Input error: memory limit [{}] has been exceeded. Please force stop the job, increase to new estimated limit [{}] and restart.",
            ml_bytes(limit),
            ml_bytes(need.saturating_mul(2))
        ),
    )
}

/// The job's `model_memory_limit` in bytes.
pub fn memory_limit(cfg: &Value) -> u64 {
    let mml = cfg["model_memory_limit"].as_str().unwrap_or(super::config::DEFAULT_MML);
    crate::ilm::parse_size(mml, "model_memory_limit").map(|(b, _)| b).unwrap_or(u64::MAX)
}

/// Admission of `n` loaded rows against the job's memory limit, before
/// more rows are read or the neighbour lists are built.
fn admit(cfg: &Value, n: usize, d: usize) -> Result<()> {
    let limit = memory_limit(cfg);
    let need = estimate_for(&cfg["analysis"]["outlier_detection"], n as u64, d as u64);
    if need > limit {
        return Err(memory_limit_exceeded(limit, need));
    }
    Ok(())
}

/// A feature's value in a hit: its typed `fields` value (ES's extractor
/// reads doc values and runtime fields). A document is skipped when the
/// feature is missing or multi-valued (ES `DataFrameDataExtractor`), or not
/// a finite number.
fn feature_value(hit: &Value, f: &str) -> Option<f64> {
    match hit.get("fields").and_then(|o| o.get(f)) {
        Some(Value::Array(a)) if a.len() == 1 => number(&a[0]),
        Some(_) => None,
        None => None,
    }
}

/// Runs every phase; Ok(false) when stopped part way.
fn run(node: &Node, id: &str, cfg: &Value, features: &[String], w: &Worker) -> Result<bool> {
    audit(node, id, "info", "Started analytics");
    let dest = cfg["dest"]["index"].as_str().unwrap_or("").to_string();
    let results_field = cfg["dest"]["results_field"].as_str().unwrap_or("ml").to_string();
    let sources = cfg["source"]["index"].as_array().map(|a| a.iter().map(super::plain).collect::<Vec<_>>().join(",")).unwrap_or_default();

    // reindexing (+ loading_data: the feature rows are read on the way).
    // The features are read as typed field values (`fields`), as ES's
    // extractor reads doc values / runtime fields: runtime features, runtime
    // overrides of source fields and aliases are what is analysed, not the
    // raw `_source` (which is still copied to the dest index as is).
    let mut body = json!({"size": PAGE, "query": cfg["source"]["query"].clone(), "sort": [{"_doc": "asc"}], "fields": features});
    if let Some(rm) = cfg["source"].get("runtime_mappings").filter(|v| v.is_object()) {
        body["runtime_mappings"] = rm.clone();
    }
    let total = {
        let mut count = json!({"size": 0, "track_total_hits": true, "query": cfg["source"]["query"].clone()});
        if let Some(rm) = body.get("runtime_mappings") {
            count["runtime_mappings"] = rm.clone();
        }
        crate::ml::search(node, &sources, &count)?["hits"]["total"]["value"].as_u64().unwrap_or(0).max(1)
    };
    let source_filter = cfg["source"].get("_source").filter(|v| v.is_object()).map(|f| {
        let list = |k: &str| -> Vec<String> { f[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
        (list("includes"), list("excludes"))
    }).filter(|(inc, exc)| !inc.is_empty() || !exc.is_empty());
    let mut scroll = crate::ml::Scroll::new(&sources, body);
    let mut reindex = Phase::new(node, id, "reindexing");
    // One entry per source `_id`, the last copy read: the dest keeps only the
    // last document copied under an `_id` (several source indices may share
    // one), so only it is analysed (#572). `None`: skipped.
    let mut rows_by_id: Vec<Option<Vec<f64>>> = Vec::new();
    let mut row_ids: Vec<String> = Vec::new();
    let mut slot_of: HashMap<String, usize> = HashMap::new();
    let mut copied = 0u64;
    loop {
        if stopped(w) {
            return Ok(false);
        }
        let hits = scroll.next_page(node)?;
        if hits.is_empty() {
            break;
        }
        let mut docs = Vec::with_capacity(hits.len());
        for h in &hits {
            let Some(doc_id) = h["_id"].as_str() else { continue };
            // The destination gets the source as `source._source` filters it.
            let mut src = match &source_filter {
                Some((inc, exc)) => crate::search::filter_source(&h["_source"], inc, exc),
                None => h["_source"].clone(),
            };
            let vals: Option<Vec<f64>> = features.iter().map(|f| feature_value(h, f)).collect();
            match slot_of.get(doc_id) {
                Some(&at) => rows_by_id[at] = vals,
                None => {
                    slot_of.insert(doc_id.to_string(), rows_by_id.len());
                    rows_by_id.push(vals);
                    row_ids.push(doc_id.to_string());
                }
            }
            if let Some(o) = src.as_object_mut() {
                o.insert("ml__incremental_id".into(), json!(copied));
            }
            copied += 1;
            docs.push((doc_id.to_string(), src));
        }
        crate::ml::resultsidx::write_to(node, &dest, docs)?;
        reindex.set(copied * 100 / total);
        // The rows read so far, with the neighbour lists they will need,
        // must fit the job's memory limit before another page is loaded.
        admit(cfg, rows_by_id.len(), features.len())?;
    }
    drop(scroll);
    drop(slot_of);
    let skipped = rows_by_id.iter().filter(|r| r.is_none()).count() as u64;
    let (row_ids, rows): (Vec<String>, Vec<Vec<f64>>) = row_ids.into_iter().zip(rows_by_id).filter_map(|(id, r)| r.map(|r| (id, r))).unzip();
    // `_start` validated that documents carry every feature: losing all of
    // them in extraction must not finish as a successful empty analysis.
    if rows.is_empty() && copied > 0 {
        return Err(EsError::new(
            400,
            "status_exception",
            format!("no documents in the source indices [{sources}] had a value for every analysed field [{}] at extraction", features.join(", ")),
        ));
    }
    reindex.set(100);
    Phase::new(node, id, "loading_data").set(100);
    #[cfg(test)]
    while hold::held(&key(node, id)) && !stopped(w) {
        std::thread::sleep(Duration::from_millis(5));
    }
    if rows.len() as u64 > MAX_ROWS {
        return Err(EsError::new(400, "status_exception", format!("outlier detection supports at most [{MAX_ROWS}] documents; [{}] have every analysed field", rows.len())));
    }
    let _ = super::report_progress_if_started(node, id, |rt| {
        rt.data_counts.insert("training_docs_count".into(), json!(rows.len()));
        rt.data_counts.insert("test_docs_count".into(), json!(0));
        rt.data_counts.insert("skipped_docs_count".into(), json!(skipped));
    });

    // computing_outliers: admitted before the standardised copy and the
    // N x K neighbour lists are allocated.
    admit(cfg, rows.len(), features.len())?;
    let params = Params::from(&cfg["analysis"]["outlier_detection"], rows.len());
    let mut computing = Phase::new(node, id, "computing_outliers");
    let scored = compute(&rows, &params, || stopped(w), |p| computing.set(p));
    let Some(scored) = scored else { return Ok(false) };
    computing.set(100);
    let peak = estimate_for(&cfg["analysis"]["outlier_detection"], rows.len() as u64, features.len() as u64);
    let _ = super::report_progress_if_started(node, id, |rt| {
        rt.memory_usage.insert("peak_usage_bytes".into(), json!(peak));
        rt.memory_usage.insert("timestamp".into(), json!(crate::date::now_millis()));
    });
    drop(rows);

    // writing_results: read the dest copies back and add the results.
    let by_id: HashMap<&str, usize> = row_ids.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let mut writing = Phase::new(node, id, "writing_results");
    let mut scroll = crate::ml::Scroll::new(&dest, json!({"size": PAGE, "query": {"match_all": {}}, "sort": [{"_doc": "asc"}]}));
    let mut written = 0u64;
    loop {
        if stopped(w) {
            return Ok(false);
        }
        let hits = scroll.next_page(node)?;
        if hits.is_empty() {
            break;
        }
        let mut docs = Vec::new();
        for h in &hits {
            let Some(doc_id) = h["_id"].as_str() else { continue };
            written += 1;
            let Some(&i) = by_id.get(doc_id) else { continue };
            let mut src = h["_source"].clone();
            let mut res = Map::new();
            res.insert("outlier_score".into(), json!(scored.scores[i]));
            if params.compute_feature_influence && scored.scores[i] >= params.feature_influence_threshold {
                let fi: Vec<Value> = features.iter().zip(&scored.influence[i]).map(|(f, v)| json!({"feature_name": f, "influence": v})).collect();
                res.insert("feature_influence".into(), Value::Array(fi));
            }
            if let Some(o) = src.as_object_mut() {
                o.insert(results_field.clone(), Value::Object(res));
            }
            docs.push((doc_id.to_string(), src));
        }
        crate::ml::resultsidx::write_to(node, &dest, docs)?;
        writing.set(written * 100 / copied.max(1));
    }
    if stopped(w) {
        return Ok(false);
    }
    Ok(true)
}

/// `outlier_detection` parameters (ES defaults).
pub struct Params {
    /// `(method, k)` members of the ensemble.
    members: Vec<(Method, usize)>,
    outlier_fraction: f64,
    standardization_enabled: bool,
    pub compute_feature_influence: bool,
    pub feature_influence_threshold: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Method {
    Lof,
    Ldof,
    DistanceKthNn,
    DistanceKnn,
}

impl Params {
    pub fn from(a: &Value, n: usize) -> Params {
        let methods = match a["method"].as_str() {
            Some("lof") => vec![Method::Lof],
            Some("ldof") => vec![Method::Ldof],
            Some("distance_kth_nn") => vec![Method::DistanceKthNn],
            Some("distance_knn") => vec![Method::DistanceKnn],
            _ => vec![Method::Lof, Method::Ldof, Method::DistanceKthNn, Method::DistanceKnn],
        };
        let cap = n.saturating_sub(1).max(1);
        let ks: Vec<usize> = match a["n_neighbors"].as_u64() {
            Some(k) => vec![(k as usize).clamp(1, cap)],
            None => {
                let mut v: Vec<usize> = [5usize, 10, 20].iter().map(|k| (*k).min(cap)).collect();
                v.dedup();
                v
            }
        };
        let members = methods.iter().flat_map(|m| ks.iter().map(move |k| (*m, *k))).collect();
        Params {
            members,
            outlier_fraction: a["outlier_fraction"].as_f64().unwrap_or(0.05).clamp(0.0, 1.0),
            standardization_enabled: a["standardization_enabled"].as_bool().unwrap_or(true),
            compute_feature_influence: a["compute_feature_influence"].as_bool().unwrap_or(true),
            feature_influence_threshold: a["feature_influence_threshold"].as_f64().unwrap_or(0.1),
        }
    }

    fn k_max(&self) -> usize {
        self.members.iter().map(|(_, k)| *k).max().unwrap_or(1)
    }
}

pub struct Scored {
    pub scores: Vec<f64>,
    /// Per row, per feature: its share of the squared distance to the row's
    /// nearest neighbours (sums to 1).
    pub influence: Vec<Vec<f64>>,
}

/// Outlier scores of `rows`; None when `stop` turned true part way.
pub fn compute(rows: &[Vec<f64>], params: &Params, stop: impl Fn() -> bool + Sync, mut progress: impl FnMut(u64)) -> Option<Scored> {
    let n = rows.len();
    let d = rows.first().map(Vec::len).unwrap_or(0);
    if n < 2 {
        return Some(Scored { scores: vec![0.0; n], influence: vec![vec![if d > 0 { 1.0 / d as f64 } else { 0.0 }; d]; n] });
    }
    let x = if params.standardization_enabled { standardize(rows) } else { rows.to_vec() };
    let k_max = params.k_max().min(n - 1);
    // Exact k nearest neighbours, sorted by distance (ties by index).
    let chunk = 256usize;
    let mut knn: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n);
    for (ci, start) in (0..n).step_by(chunk).enumerate() {
        if stop() {
            return None;
        }
        let end = (start + chunk).min(n);
        let part: Vec<Vec<(f64, usize)>> = (start..end)
            .into_par_iter()
            .map(|i| {
                let mut best: Vec<(f64, usize)> = Vec::with_capacity(k_max + 1);
                if stop() {
                    return best;
                }
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let dd = dist(&x[i], &x[j]);
                    if best.len() < k_max || dd < best[best.len() - 1].0 {
                        let pos = best.partition_point(|(b, _)| *b <= dd);
                        best.insert(pos, (dd, j));
                        best.truncate(k_max);
                    }
                }
                best
            })
            .collect();
        if stop() {
            return None;
        }
        knn.extend(part);
        progress(((ci + 1) * chunk).min(n) as u64 * 80 / n as u64);
    }
    let mut total = vec![0.0; n];
    for (m, k) in &params.members {
        let k = (*k).min(k_max);
        let raw = method_scores(*m, k, &x, &knn, &stop)?;
        let norm = normalize(&raw, params.outlier_fraction);
        for (t, s) in total.iter_mut().zip(norm) {
            *t += s;
        }
    }
    let members = params.members.len().max(1) as f64;
    let scores: Vec<f64> = total.into_iter().map(|t| (t / members).clamp(0.0, 1.0)).collect();
    progress(95);
    if stop() {
        return None;
    }
    let influence = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut per = vec![0.0; d];
            for (_, j) in &knn[i] {
                for (f, p) in per.iter_mut().enumerate() {
                    let diff = x[i][f] - x[*j][f];
                    *p += diff * diff;
                }
            }
            let sum: f64 = per.iter().sum();
            if sum > 0.0 {
                per.iter_mut().for_each(|p| *p /= sum);
            } else {
                per.iter_mut().for_each(|p| *p = 1.0 / d.max(1) as f64);
            }
            per
        })
        .collect();
    Some(Scored { scores, influence })
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt()
}

/// Columns to zero mean / unit variance (constant columns to 0).
fn standardize(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rows.len() as f64;
    let d = rows[0].len();
    let mut mean = vec![0.0; d];
    for r in rows {
        for (m, v) in mean.iter_mut().zip(r) {
            *m += v / n;
        }
    }
    let mut var = vec![0.0; d];
    for r in rows {
        for f in 0..d {
            var[f] += (r[f] - mean[f]).powi(2) / n;
        }
    }
    let sd: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
    rows.iter().map(|r| (0..d).map(|f| if sd[f] > 0.0 { (r[f] - mean[f]) / sd[f] } else { 0.0 }).collect()).collect()
}

/// None when `stop` turned true part way (`ldof` is O(n k^2)).
fn method_scores(m: Method, k: usize, x: &[Vec<f64>], knn: &[Vec<(f64, usize)>], stop: &(impl Fn() -> bool + Sync)) -> Option<Vec<f64>> {
    let n = x.len();
    let kdist = |i: usize| knn[i][k - 1].0;
    let scores = match m {
        Method::DistanceKthNn => (0..n).map(kdist).collect(),
        Method::DistanceKnn => (0..n).map(|i| knn[i][..k].iter().map(|(d, _)| d).sum::<f64>() / k as f64).collect(),
        Method::Lof => {
            // Local reachability density, then the neighbours' mean density
            // over the point's own.
            let lrd: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let reach: f64 = knn[i][..k].iter().map(|(d, j)| d.max(kdist(*j))).sum::<f64>() / k as f64;
                    if reach > 0.0 { 1.0 / reach } else { f64::INFINITY }
                })
                .collect();
            (0..n)
                .map(|i| {
                    let mean_nb = knn[i][..k].iter().map(|(_, j)| lrd[*j]).sum::<f64>() / k as f64;
                    match (lrd[i].is_finite(), mean_nb.is_finite()) {
                        (true, true) => mean_nb / lrd[i],
                        (false, _) => 1.0,
                        (true, false) => f64::MAX,
                    }
                })
                .collect()
        }
        Method::Ldof => (0..n)
            .into_par_iter()
            .map(|i| {
                if stop() {
                    return 0.0;
                }
                let nb = &knn[i][..k];
                let mean_d = nb.iter().map(|(d, _)| d).sum::<f64>() / k as f64;
                let mut inner = 0.0;
                let mut pairs = 0usize;
                for a in 0..nb.len() {
                    for b in a + 1..nb.len() {
                        inner += dist(&x[nb[a].1], &x[nb[b].1]);
                        pairs += 1;
                    }
                }
                let inner = if pairs > 0 { inner / pairs as f64 } else { 0.0 };
                if inner > 0.0 { mean_d / inner } else if mean_d > 0.0 { f64::MAX } else { 0.0 }
            })
            .collect(),
    };
    if stop() {
        return None;
    }
    Some(scores)
}

/// Raw scores to [0, 1]: 0 at or below the median, 0.5 at the
/// `1 - outlier_fraction` quantile, approaching 1 beyond.
fn normalize(raw: &[f64], outlier_fraction: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = raw.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![0.0; raw.len()];
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    let median = q(0.5);
    let high = q((1.0 - outlier_fraction).clamp(0.5, 1.0));
    let scale = ((high - median) / std::f64::consts::LN_2).max(1e-9 * median.abs().max(1e-12));
    raw.iter()
        .map(|v| {
            if !v.is_finite() || *v == f64::MAX {
                return 1.0;
            }
            let over = (v - median).max(0.0);
            1.0 - (-over / scale).exp()
        })
        .collect()
}

/// Test gate: a held job pauses after `loading_data`.
#[cfg(test)]
pub(super) mod hold {
    use parking_lot::Mutex;
    static HELD: Mutex<Vec<String>> = Mutex::new(Vec::new());
    pub fn held(k: &str) -> bool {
        HELD.lock().iter().any(|x| x == k)
    }
    pub fn set(node: &crate::node::Node, id: &str, on: bool) {
        let k = super::key(node, id);
        let mut h = HELD.lock();
        h.retain(|x| *x != k);
        if on {
            h.push(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_point_scores_highest_and_influence_names_its_feature() {
        // A tight 2-d grid and one point far out along x.
        let mut rows: Vec<Vec<f64>> = (0..60).map(|i| vec![(i % 10) as f64, (i / 10) as f64]).collect();
        rows.push(vec![60.0, 2.5]);
        let p = Params::from(&json!({}), rows.len());
        let s = compute(&rows, &p, || false, |_| {}).unwrap();
        let top = s.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(top, 60);
        assert!(s.scores[60] > 0.9, "{}", s.scores[60]);
        let inliers = s.scores[..60].iter().filter(|v| **v < 0.5).count();
        assert!(inliers >= 55, "{inliers}");
        assert!(s.influence[60][0] > 0.9, "{:?}", s.influence[60]);
        assert!(s.scores.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn single_methods_and_stop() {
        let rows: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i * 7 % 5) as f64]).chain([vec![500.0, 500.0]]).collect();
        for m in ["lof", "ldof", "distance_kth_nn", "distance_knn"] {
            let p = Params::from(&json!({"method": m, "n_neighbors": 3}), rows.len());
            let s = compute(&rows, &p, || false, |_| {}).unwrap();
            let top = s.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(top, 30, "{m}");
        }
        let p = Params::from(&json!({}), rows.len());
        assert!(compute(&rows, &p, || true, |_| {}).is_none());
    }

    /// #114: the estimate covers the N x K neighbour lists (and is the
    /// figure admitted); the reference example is far over a 1gb limit.
    #[test]
    fn estimate_covers_neighbour_lists_and_rejects_over_limit() {
        let big = estimate_bytes(20_000, 2, 19_999);
        assert!(big >= 16 * 20_000 * 19_999, "{big}");
        assert_eq!(estimate_bytes(u64::MAX, u64::MAX, u64::MAX), u64::MAX, "saturates");
        let cfg = json!({"model_memory_limit": "1kb", "analysis": {"outlier_detection": {"n_neighbors": 50}}});
        let e = admit(&cfg, 100, 2).unwrap_err();
        assert!(e.reason.starts_with("Input error: memory limit [1 kb] has been exceeded"), "{}", e.reason);
        let ok = json!({"model_memory_limit": "1gb", "analysis": {"outlier_detection": {"n_neighbors": 50}}});
        admit(&ok, 100, 2).unwrap();
        // The default limit (1gb) refuses the review's 20,000 x 19,999 case.
        let dflt = json!({"analysis": {"outlier_detection": {"n_neighbors": 19_999}}});
        assert!(admit(&dflt, 20_000, 2).is_err());
        assert_eq!(ml_bytes(1024), "1 kb");
        assert_eq!(ml_bytes(3 << 20), "3 mb");
    }

    /// Stopping is seen inside the kNN chunk and the O(n k^2) `ldof` pass,
    /// not only between chunks.
    #[test]
    fn stop_is_seen_inside_expensive_passes() {
        let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64, (i * 7 % 5) as f64]).collect();
        let p = Params::from(&json!({"method": "ldof", "n_neighbors": 10}), rows.len());
        // Stop after the neighbour search started (the first check passes).
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let stop = || calls.fetch_add(1, Ordering::Relaxed) > 0;
        assert!(compute(&rows, &p, stop, |_| {}).is_none());
        // Stopping only once the ldof pass runs also returns None.
        let knn_checks = 1 + rows.len() + 1;
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let stop = || calls.fetch_add(1, Ordering::Relaxed) >= knn_checks;
        assert!(compute(&rows, &p, stop, |_| {}).is_none());
        assert!(compute(&rows, &p, || false, |_| {}).is_some());
    }
}
