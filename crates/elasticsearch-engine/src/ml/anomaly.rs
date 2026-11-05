//! Anomaly detection job and datafeed APIs: CRUD and the persistent-task
//! state machine (open / close / flush / reset; start / stop).
//!
//! An opened job runs a [`super::process`] (the in-process autodetect) and
//! a started datafeed a [`super::feed`] runner thread; the stats APIs read
//! the live process or, for closed jobs, the stats documents in the results
//! index.

use super::store::{service, DatafeedRt, JobRt, State};
use super::{datafeed, job, process, resultsidx, xc::bad_request};
use crate::error::{EsError, Result};
use crate::node::{Node, NODE_ID};
use serde_json::{json, Map, Value};
use std::sync::{Arc, OnceLock};

const MB: u64 = 1 << 20;

static NODE_INFO: OnceLock<(Value, u64)> = OnceLock::new();

/// Register the node descriptor ES puts in ML stats (`node` object:
/// name, ephemeral_id, transport_address, attributes) and the machine
/// memory. Called by the REST layer; first call wins.
pub fn init_node_info(desc: Value, machine_memory: u64) {
    let _ = NODE_INFO.set((desc, machine_memory));
}

fn node_desc(node: &Node, for_job: bool) -> Value {
    let (d, _) = NODE_INFO.get().cloned().unwrap_or((json!({}), 0));
    let mut attrs = d["attributes"].as_object().cloned().unwrap_or_default();
    attrs.remove("testattr");
    if !for_job {
        attrs.remove("transform.config_version");
        attrs.remove("xpack.installed");
    }
    json!({"id": NODE_ID, "name": node.name, "ephemeral_id": d["ephemeral_id"], "transport_address": d["transport_address"], "attributes": attrs})
}

/// Machine memory (bytes) of this node.
pub fn machine_memory() -> u64 {
    NODE_INFO.get().map(|x| x.1).filter(|m| *m > 0).unwrap_or(16 << 30)
}

/// Native memory ML may use on this node (`xpack.ml.max_machine_memory_percent`, default 30%).
pub fn ml_max_memory(node: &Node) -> u64 {
    let pct = crate::ilm::cluster_setting(node, "xpack.ml.max_machine_memory_percent").and_then(|s| s.parse::<u64>().ok()).unwrap_or(30);
    machine_memory() / 100 * pct
}

/// `xpack.ml.max_model_memory_limit` in MB (`None`/0 = unlimited).
pub fn max_model_memory_mb(node: &Node) -> Option<u64> {
    let s = crate::ilm::cluster_setting(node, "xpack.ml.max_model_memory_limit")?;
    let b = job::parse_bytes(&s, "xpack.ml.max_model_memory_limit").ok()?;
    (b > 0).then_some(b / MB)
}

fn not_found_job(id: &str) -> EsError {
    EsError::resource_not_found(format!("No known job with id '{id}'"))
}

fn not_found_datafeed(id: &str) -> EsError {
    EsError::resource_not_found(format!("No datafeed with id [{id}] exists"))
}

/// Expand a job expression (ids, groups, wildcards) — sorted ids.
pub fn expand_jobs(st: &State, expr: Option<&str>, allow_no_match: bool) -> Result<Vec<String>> {
    let expr = expr.filter(|e| !e.is_empty()).unwrap_or("_all");
    let mut out: Vec<String> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    for tok in expr.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let m = st.jobs_matching(tok);
        if m.is_empty() {
            let wild = tok == "_all" || tok.contains('*');
            if !wild || !allow_no_match {
                missing.push(tok);
            }
        }
        out.extend(m);
    }
    if !missing.is_empty() {
        let only_wild = missing.iter().all(|t| *t == "_all" || t.contains('*'));
        if only_wild && missing.len() == 1 && missing[0] == "_all" {
            // `_all` with no jobs is an empty answer
        } else {
            return Err(not_found_job(&missing.join(",")));
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// Expand a datafeed expression — sorted ids.
pub fn expand_datafeeds(st: &State, expr: Option<&str>, allow_no_match: bool) -> Result<Vec<String>> {
    let expr = expr.filter(|e| !e.is_empty()).unwrap_or("_all");
    let mut out: Vec<String> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    for tok in expr.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let wild = tok == "_all" || tok.contains('*');
        let m: Vec<String> = st.datafeeds.keys().filter(|id| if wild { tok == "_all" || crate::mapping::wildcard_match(tok, id) } else { *id == tok }).cloned().collect();
        if m.is_empty() && (!wild || !allow_no_match) && tok != "_all" {
            missing.push(tok);
        }
        out.extend(m);
    }
    if !missing.is_empty() {
        return Err(not_found_datafeed(&missing.join(",")));
    }
    out.sort();
    out.dedup();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// `PUT _ml/anomaly_detectors/{id}`: (job, deprecation warnings).
pub fn put_job(node: &Node, id: &str, body: &Value, raw: &str, param_opts: &Map<String, Value>) -> Result<(Value, Vec<String>)> {
    let nj = job::parse_put(id, body, raw)?;
    let max = max_model_memory_mb(node);
    if let (Some(m), Some(mx)) = (nj.explicit_mml, max) {
        if m > mx {
            return Err(EsError::illegal_argument(format!(
                "model_memory_limit [{}] must be less than the value of the xpack.ml.max_model_memory_limit setting [{}]",
                job::bytes_to_string(m * MB),
                job::bytes_to_string(mx * MB)
            )));
        }
    }
    let default_mml = max.map(|m| m.min(job::DEFAULT_MML_MB)).unwrap_or(job::DEFAULT_MML_MB);
    let svc = service(node);
    let datafeed_body = nj.datafeed.clone();
    let out = svc.mutate(|st| {
        if st.jobs.contains_key(id) {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("The job cannot be created with the Id '{id}'. The Id is already used.")));
        }
        let groups: Vec<String> = nj.parts.get("groups").and_then(Value::as_array).into_iter().flatten().filter_map(|g| g.as_str().map(str::to_string)).collect();
        check_group_clash(st, id, &groups)?;
        let mut j = job::finish(id, &nj, default_mml, crate::date::now_millis());
        let mut df_out = None;
        if let Some(db) = &datafeed_body {
            let mut o = db.as_object().cloned().unwrap_or_default();
            o.insert("job_id".into(), json!(id));
            let mut df = datafeed::build(id, &o, super::xc::Pos::empty(), param_opts)?;
            datafeed::mark_generated(&mut df, db);
            if st.datafeeds.contains_key(id) {
                return Err(EsError::new(400, "resource_already_exists_exception", format!("A datafeed with id [{id}] already exists")));
            }
            st.datafeeds.insert(id.to_string(), df.clone());
            df_out = Some(datafeed::public(&df, false, node.security.enabled()));
        }
        st.jobs.insert(id.to_string(), j.clone());
        st.job_rt.insert(id.to_string(), JobRt { state: "closed".into(), ..Default::default() });
        if let Some(d) = &df_out {
            j = job::render(&j, Some(d), false);
        }
        Ok(j)
    })?;
    resultsidx::ensure_results_index(node, id, &out);
    Ok((out, nj.warnings))
}

/// `job and group names must be unique` checks for a job's id + groups.
fn check_group_clash(st: &State, id: &str, groups: &[String]) -> Result<()> {
    let clash = |n: &str| EsError::new(400, "resource_already_exists_exception", format!("job and group names must be unique but job [{n}] and group [{n}] have the same name"));
    if st.groups().iter().any(|g| g == id) {
        return Err(clash(id));
    }
    for g in groups {
        if g == id || st.jobs.contains_key(g) {
            return Err(clash(g));
        }
    }
    Ok(())
}

/// `GET _ml/anomaly_detectors[/{expr}]`.
pub fn get_jobs(node: &Node, expr: Option<&str>, allow_no_match: bool, exclude_generated: bool) -> Result<Value> {
    let st = service(node).read();
    let ids = expand_jobs(&st, expr, allow_no_match)?;
    let jobs: Vec<Value> = ids
        .iter()
        .map(|id| {
            let df = st.datafeed_for_job(id).and_then(|d| st.datafeeds.get(&d)).map(|d| datafeed::public(d, false, node.security.enabled()));
            job::render(&st.jobs[id], if exclude_generated { None } else { df.as_ref() }, exclude_generated)
        })
        .collect();
    Ok(json!({"count": jobs.len(), "jobs": jobs}))
}

fn upgrade_explanation() -> &'static str {
    "persistent task cannot be assigned while upgrade mode is enabled."
}

/// One job's `_stats` entry: live process stats for an open job, the
/// stored stats documents otherwise (zeros when the index has none).
fn job_stats_entry(node: &Node, st: &State, id: &str) -> Value {
    let rt = st.job_rt.get(id).cloned().unwrap_or_default();
    let state = st.job_state(id).to_string();
    let now = crate::date::now_millis();
    let open = matches!(state.as_str(), "opened" | "opening" | "closing");
    let live = if open && rt.assigned { process::live_stats(node, id) } else { None };
    let mml_bytes = job::mml_bytes(&st.jobs[id]);
    let (dc, ms, ts) = match live {
        Some(x) => x,
        None => {
            let dc = super::results::stored_data_counts(node, id).map(|d| process::DataCounts::from_json(&d).to_json(id, true)).unwrap_or_else(|| process::DataCounts::default().to_json(id, true));
            let ms = match super::results::stored_model_size_stats(node, id) {
                Some(m) => render_model_size(id, &m, mml_bytes),
                None => {
                    let mut m = Map::new();
                    m.insert("job_id".into(), json!(id));
                    m.insert("result_type".into(), json!("model_size_stats"));
                    m.insert("model_bytes".into(), json!(0));
                    if open && rt.assigned {
                        m.insert("model_bytes_memory_limit".into(), json!(mml_bytes));
                    }
                    for k in ["total_by_field_count", "total_over_field_count", "total_partition_field_count", "bucket_allocation_failures_count"] {
                        m.insert(k.into(), json!(0));
                    }
                    m.insert("memory_status".into(), json!("ok"));
                    if open && rt.assigned {
                        m.insert("assignment_memory_basis".into(), json!("model_memory_limit"));
                    }
                    for k in ["categorized_doc_count", "total_category_count", "frequent_category_count", "rare_category_count", "dead_category_count", "failed_category_count"] {
                        m.insert(k.into(), json!(0));
                    }
                    m.insert("categorization_status".into(), json!("ok"));
                    m.insert("log_time".into(), json!(now));
                    Value::Object(m)
                }
            };
            let ts = super::results::stored_timing_stats(node, id).map(|t| process::TimingStats::from_json(&t)).unwrap_or_default().to_json(id);
            (dc, ms, ts)
        }
    };
    let mut o = Map::new();
    o.insert("job_id".into(), json!(id));
    o.insert("data_counts".into(), dc);
    o.insert("model_size_stats".into(), ms);
    o.insert("forecasts_stats".into(), super::results::forecast_stats(node, id));
    o.insert("state".into(), json!(state));
    if open {
        if st.upgrade_mode {
            o.insert("assignment_explanation".into(), json!(upgrade_explanation()));
        } else if rt.assigned {
            o.insert("node".into(), node_desc(node, true));
            o.insert("assignment_explanation".into(), json!(""));
        } else {
            o.insert("assignment_explanation".into(), json!(lazy_explanation(node, &st.jobs[id])));
        }
        let secs = ((now - rt.open_time.unwrap_or(now)) / 1000).max(0);
        o.insert("open_time".into(), json!(format!("{secs}s")));
    }
    o.insert("timing_stats".into(), ts);
    Value::Object(o)
}

/// A stored `model_size_stats` document in `ModelSizeStats.toXContent`
/// order with ES's defaults.
fn render_model_size(id: &str, m: &Value, mml_bytes: u64) -> Value {
    let mut o = Map::new();
    o.insert("job_id".into(), json!(id));
    o.insert("result_type".into(), json!("model_size_stats"));
    let n = |k: &str| m[k].as_i64().or_else(|| m[k].as_f64().map(|f| f as i64)).unwrap_or(0);
    o.insert("model_bytes".into(), json!(n("model_bytes")));
    if m.get("peak_model_bytes").is_some() {
        o.insert("peak_model_bytes".into(), json!(n("peak_model_bytes")));
    }
    if m.get("model_bytes_exceeded").is_some() {
        o.insert("model_bytes_exceeded".into(), json!(n("model_bytes_exceeded")));
    }
    if m.get("model_bytes_memory_limit").is_some() {
        o.insert("model_bytes_memory_limit".into(), json!(m["model_bytes_memory_limit"].as_i64().unwrap_or(mml_bytes as i64)));
    }
    for k in ["total_by_field_count", "total_over_field_count", "total_partition_field_count", "bucket_allocation_failures_count"] {
        o.insert(k.into(), json!(n(k)));
    }
    o.insert("memory_status".into(), json!(m["memory_status"].as_str().unwrap_or("ok")));
    if let Some(a) = m.get("assignment_memory_basis").filter(|v| v.is_string()) {
        o.insert("assignment_memory_basis".into(), a.clone());
    }
    if m.get("output_memory_allocator_bytes").is_some() {
        o.insert("output_memory_allocator_bytes".into(), json!(n("output_memory_allocator_bytes")));
    }
    for k in ["categorized_doc_count", "total_category_count", "frequent_category_count", "rare_category_count", "dead_category_count", "failed_category_count"] {
        o.insert(k.into(), json!(n(k)));
    }
    o.insert("categorization_status".into(), json!(m["categorization_status"].as_str().unwrap_or("ok")));
    if let Some(t) = m.get("log_time").and_then(resultsidx::ts_millis) {
        o.insert("log_time".into(), json!(t));
    }
    if let Some(t) = m.get("timestamp").and_then(resultsidx::ts_millis) {
        o.insert("timestamp".into(), json!(t));
    }
    Value::Object(o)
}

fn lazy_explanation(node: &Node, job: &Value) -> String {
    let need = job::mml_bytes(job) + 30 * MB;
    format!(
        "Not opening job [{}] on node [{{{}}}{{ml.machine_memory={}}}{{node.ml=true}}], because this node has insufficient available memory. Available memory for ML [{}], memory required by existing jobs [0], estimated memory required for this job [{}]",
        job["job_id"].as_str().unwrap_or(""),
        node.name,
        machine_memory(),
        ml_max_memory(node),
        need
    )
}

/// `GET _ml/anomaly_detectors[/{expr}]/_stats`.
pub fn job_stats(node: &Node, expr: Option<&str>, allow_no_match: bool) -> Result<Value> {
    let st = service(node).read();
    let ids = expand_jobs(&st, expr, allow_no_match)?;
    for id in &ids {
        let index = resultsidx::results_index(&st.jobs[id]);
        if let Some(svc) = node.get_index(&index) {
            if !svc.is_open() {
                return Err(EsError::new(
                    400,
                    "search_phase_execution_exception",
                    "all shards failed",
                )
                .caused_by(EsError::new(403, "cluster_block_exception", format!("index [{index}/{}] blocked by: [FORBIDDEN/4/index closed];", svc.uuid)))
                .with_status_reason(format!("type=cluster_block_exception, reason=index [{index}/{}] blocked by: [FORBIDDEN/4/index closed];", svc.uuid)));
            }
        }
    }
    let jobs: Vec<Value> = ids.iter().map(|id| job_stats_entry(node, &st, id)).collect();
    Ok(json!({"count": jobs.len(), "jobs": jobs}))
}

trait WithReason {
    fn with_status_reason(self, r: String) -> Self;
}
impl WithReason for EsError {
    fn with_status_reason(mut self, r: String) -> Self {
        self.reason = r;
        self
    }
}

/// `DELETE _ml/anomaly_detectors/{id}`.
pub fn delete_job(node: &Node, id: &str, force: bool) -> Result<Value> {
    let svc = service(node);
    let removed = svc.mutate(|st| {
        let job = st.jobs.get(id).cloned().ok_or_else(|| not_found_job(id))?;
        let state = st.job_state(id).to_string();
        if !force && matches!(state.as_str(), "opened" | "opening" | "closing") {
            return Err(EsError::new(409, "status_exception", format!("Cannot delete job [{id}] because the job is {state}")));
        }
        if let Some(df) = st.datafeed_for_job(id) {
            if !force && st.datafeed_state(&df) != "stopped" {
                return Err(EsError::new(409, "status_exception", format!("Cannot delete job [{id}] because datafeed [{df}] refers to it")));
            }
            st.datafeeds.remove(&df);
            st.datafeed_rt.remove(&df);
        }
        st.jobs.remove(id);
        st.job_rt.remove(id);
        for c in st.calendars.values_mut() {
            if let Some(ids) = c.get_mut("job_ids").and_then(Value::as_array_mut) {
                ids.retain(|x| x.as_str() != Some(id));
            }
        }
        Ok(job)
    })?;
    svc.sync_tasks(node);
    super::feed::stop(node, id);
    process::delete_job_data(node, id, &removed);
    resultsidx::remove_aliases(node, id, &removed);
    // A custom results index no other job uses goes with the job.
    let index = resultsidx::results_index(&removed);
    if index != resultsidx::SHARED_RESULTS_INDEX && !svc.read().jobs.values().any(|j| resultsidx::results_index(j) == index) && node.get_index(&index).is_some() {
        let _ = resultsidx::delete_index(node, &index);
    }
    Ok(json!({"acknowledged": true}))
}

/// `POST _ml/anomaly_detectors/{id}/_open`.
pub fn open_job(node: &Node, id: &str) -> Result<Value> {
    let svc = service(node);
    if svc.upgrade_mode() {
        return Err(EsError::new(400, "status_exception", "Cannot perform cluster:admin/xpack/ml/job/open action while upgrade mode is enabled"));
    }
    if crate::ilm::cluster_setting(node, "cluster.persistent_tasks.allocation.enable").as_deref() == Some("none") {
        return Err(EsError::new(
            429,
            "status_exception",
            "Cannot open jobs because persistent task assignment is disabled by the [cluster.persistent_tasks.allocation.enable] setting",
        ));
    }
    let max_mem = ml_max_memory(node);
    let assigned = svc.mutate(|st| {
        let mut j = st.jobs.get(id).cloned().ok_or_else(|| not_found_job(id))?;
        for f in job::referenced_filters(&j) {
            if !st.filters.contains_key(&f) {
                return Err(EsError::resource_not_found(format!("Unable to find filter [{f}]")));
            }
        }
        if let Some(b) = st.blocked.get(id) {
            return Err(EsError::new(409, "status_exception", format!("Cannot open job [{id}] because it is executing [{b}]")));
        }
        let state = st.job_state(id).to_string();
        if matches!(state.as_str(), "opened" | "opening") {
            let rt = st.job_rt.get(id).cloned().unwrap_or_default();
            return Ok(rt.assigned);
        }
        let need = job::mml_bytes(&j) + 30 * MB;
        let fits = need <= max_mem;
        let lazy = j["allow_lazy_open"].as_bool().unwrap_or(false);
        if !fits && !lazy {
            return Err(EsError::new(429, "status_exception", "Could not open job because no ML nodes with sufficient capacity were found").caused_by(EsError::new(500, "illegal_state_exception", lazy_explanation(node, &j))));
        }
        if let Some(o) = j.as_object_mut() {
            o.remove("finished_time");
        }
        st.jobs.insert(id.to_string(), j);
        let rt = st.job_rt.entry(id.to_string()).or_default();
        rt.state = if fits { "opened".into() } else { "opening".into() };
        rt.open_time = Some(crate::date::now_millis());
        rt.assigned = fits;
        Ok(fits)
    })?;
    svc.sync_tasks(node);
    if assigned {
        if let Err(e) = process::open(node, id) {
            let _ = svc.mutate(|st| {
                if let Some(rt) = st.job_rt.get_mut(id) {
                    rt.state = "closed".into();
                    rt.assigned = false;
                }
                Ok(())
            });
            svc.sync_tasks(node);
            return Err(e);
        }
    }
    Ok(json!({"opened": true, "node": if assigned { NODE_ID } else { "" }}))
}

/// `POST _ml/anomaly_detectors/{expr}/_close`.
pub fn close_job(node: &Node, expr: &str, allow_no_match: bool, force: bool) -> Result<Value> {
    let svc = service(node);
    // Validate, then persist the processes (outside the state lock), then
    // record the closed states.
    let closing: Vec<String> = {
        let st = svc.read();
        let ids = expand_jobs(&st, Some(expr), allow_no_match)?;
        let mut out = Vec::new();
        for id in &ids {
            let state = st.job_state(id).to_string();
            if state == "closed" {
                continue;
            }
            if state == "failed" && !force {
                return Err(EsError::new(409, "status_exception", format!("cannot close job [{id}] because it failed, use force close")));
            }
            if let Some(df) = st.datafeed_for_job(id) {
                if st.datafeed_state(&df) != "stopped" && !force {
                    return Err(EsError::new(409, "status_exception", format!("cannot close job datafeed [{df}] hasn't been stopped")));
                }
            }
            out.push(id.clone());
        }
        out
    };
    let mut snapshots: Map<String, Value> = Map::new();
    for id in &closing {
        if force {
            process::drop_process(node, id);
        } else if let Ok(Some(sid)) = process::close(node, id) {
            snapshots.insert(id.clone(), json!(sid));
        }
    }
    svc.mutate(|st| {
        let ids = closing.clone();
        let now = crate::date::now_millis();
        for id in &ids {
            if st.job_state(id) == "closed" {
                continue;
            }
            if let Some(df) = st.datafeed_for_job(id) {
                if st.datafeed_state(&df) != "stopped" {
                    super::feed::stop(node, &df);
                    st.datafeed_rt.insert(df.clone(), DatafeedRt { state: "stopped".into(), ..Default::default() });
                }
            }
            if let Some(o) = st.jobs.get_mut(id).and_then(Value::as_object_mut) {
                // finished_time (and the new model_snapshot_id) sit after create_time
                let mut m = Map::new();
                o.remove("finished_time");
                let snap = snapshots.get(id).cloned().or_else(|| o.remove("model_snapshot_id"));
                for (k, v) in std::mem::take(o) {
                    let after = k == "create_time";
                    m.insert(k, v);
                    if after {
                        m.insert("finished_time".into(), json!(now));
                        if let Some(sv) = &snap {
                            m.insert("model_snapshot_id".into(), sv.clone());
                        }
                    }
                }
                *o = m;
            }
            let rt = st.job_rt.entry(id.clone()).or_default();
            rt.state = "closed".into();
            rt.open_time = None;
            rt.assigned = false;
        }
        Ok(())
    })?;
    svc.sync_tasks(node);
    Ok(json!({"closed": true}))
}

fn require_open(st: &State, id: &str) -> Result<()> {
    if !st.jobs.contains_key(id) {
        return Err(not_found_job(id));
    }
    if st.job_state(id) != "opened" {
        return Err(EsError::new(409, "status_exception", format!("Cannot perform requested action because job [{id}] is not open")));
    }
    Ok(())
}

/// `POST _ml/anomaly_detectors/{id}/_flush`.
pub fn flush_job(node: &Node, id: &str, params: &FlushParams) -> Result<Value> {
    let st = service(node).read();
    require_open(&st, id)?;
    drop(st);
    if let (Some(s), Some(e)) = (params.start, params.end) {
        if e < s {
            return Err(EsError::illegal_argument("Invalid time range: end time must be after start time"));
        }
    }
    process::flush(node, id, params)
}

/// `POST _ml/anomaly_detectors/{id}/_data`.
pub fn post_data(node: &Node, id: &str, body: &[u8]) -> Result<Value> {
    let st = service(node).read();
    require_open(&st, id)?;
    drop(st);
    process::post_data(node, id, body)
}

/// `POST _ml/anomaly_detectors/{id}/_forecast`.
pub fn forecast(node: &Node, id: &str, duration_ms: i64, expires_in_ms: i64) -> Result<Value> {
    let st = service(node).read();
    require_open(&st, id)?;
    drop(st);
    process::forecast(node, id, duration_ms, expires_in_ms)
}

#[derive(Default)]
pub struct FlushParams {
    pub calc_interim: bool,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub advance_time: Option<i64>,
    pub skip_time: Option<i64>,
}

/// `POST _ml/anomaly_detectors/{id}/_reset`.
pub fn reset_job(node: &Node, id: &str) -> Result<Value> {
    let svc = service(node);
    svc.mutate(|st| {
        let j = st.jobs.get_mut(id).ok_or_else(|| not_found_job(id))?;
        let state = st.job_rt.get(id).map(|r| r.state.clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| "closed".into());
        if state != "closed" {
            return Err(EsError::new(409, "status_exception", format!("Can only reset a job when it is closed. Job [{id}] is [{state}]")));
        }
        if let Some(o) = j.as_object_mut() {
            o.remove("finished_time");
            o.remove("model_snapshot_id");
        }
        if let Some(rt) = st.job_rt.get_mut(id) {
            rt.data_counts.clear();
        }
        Ok(j.clone())
    })
    .map(|j| process::delete_job_data(node, id, &j))?;
    Ok(json!({"acknowledged": true}))
}

/// `POST _ml/anomaly_detectors/{id}/_update`.
pub fn update_job(node: &Node, id: &str, body: &Value, raw: &str) -> Result<Value> {
    let svc = service(node);
    {
        let st = svc.read();
        let ids = expand_jobs(&st, Some(id), true)?;
        if ids.len() > 1 || id == "_all" || id.contains('*') {
            return Err(bad_request("Cannot update more than 1 job at a time"));
        }
    }
    let u = job::parse_update(body, raw)?;
    let max = max_model_memory_mb(node);
    svc.mutate(|st| {
        let cur = st.jobs.get(id).cloned().ok_or_else(|| not_found_job(id))?;
        if let Some(g) = u.fields.get("groups").and_then(Value::as_array) {
            let groups: Vec<String> = g.iter().filter_map(|x| x.as_str().map(str::to_string)).collect();
            for gname in &groups {
                if gname == id || st.jobs.contains_key(gname) {
                    return Err(EsError::new(400, "resource_already_exists_exception", format!("job and group names must be unique but job [{gname}] and group [{gname}] have the same name")));
                }
            }
        }
        let open = matches!(st.job_state(id), "opened" | "opening");
        let updated = job::apply_update(&cur, &u, open, max)?;
        st.jobs.insert(id.to_string(), updated.clone());
        Ok(updated)
    })
    .map(|u| {
        process::config_updated(node, id);
        u
    })
}

// ---------------------------------------------------------------------------
// Datafeeds
// ---------------------------------------------------------------------------

/// `PUT _ml/datafeeds/{id}`.
pub fn put_datafeed(node: &Node, id: &str, body: &Value, raw: &str, param_opts: &Map<String, Value>) -> Result<Value> {
    let mut df = datafeed::parse_put(id, body, raw, param_opts)?;
    datafeed::mark_generated(&mut df, body);
    let svc = service(node);
    svc.mutate(|st| {
        if st.datafeeds.contains_key(id) {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("A datafeed with id [{id}] already exists")));
        }
        let job_id = df["job_id"].as_str().unwrap_or("").to_string();
        if !st.jobs.contains_key(&job_id) {
            return Err(not_found_job(&job_id));
        }
        if let Some(other) = st.datafeed_for_job(&job_id) {
            return Err(EsError::new(409, "status_exception", format!("A datafeed [{other}] already exists for job [{job_id}]")));
        }
        st.datafeeds.insert(id.to_string(), df.clone());
        st.datafeed_rt.insert(id.to_string(), DatafeedRt { state: "stopped".into(), ..Default::default() });
        Ok(datafeed::public(&df, false, node.security.enabled()))
    })
}

/// `GET _ml/datafeeds[/{expr}]`.
pub fn get_datafeeds(node: &Node, expr: Option<&str>, allow_no_match: bool, exclude_generated: bool) -> Result<Value> {
    let st = service(node).read();
    let ids = expand_datafeeds(&st, expr, allow_no_match)?;
    let dfs: Vec<Value> = ids.iter().map(|id| datafeed::public(&st.datafeeds[id], exclude_generated, node.security.enabled())).collect();
    Ok(json!({"count": dfs.len(), "datafeeds": dfs}))
}

fn datafeed_stats_entry(node: &Node, st: &State, id: &str) -> Value {
    let rt = st.datafeed_rt.get(id).cloned().unwrap_or_default();
    let state = st.datafeed_state(id).to_string();
    let job_id = st.datafeeds[id]["job_id"].as_str().unwrap_or("").to_string();
    let mut o = Map::new();
    o.insert("datafeed_id".into(), json!(id));
    o.insert("state".into(), json!(state));
    let running = matches!(state.as_str(), "started" | "starting" | "stopping");
    if running {
        if st.upgrade_mode {
            o.insert("assignment_explanation".into(), json!(upgrade_explanation()));
        } else {
            o.insert("node".into(), node_desc(node, false));
            o.insert("assignment_explanation".into(), json!(""));
        }
    }
    o.insert(
        "timing_stats".into(),
        json!({"job_id": job_id, "search_count": rt.search_count, "bucket_count": 0, "total_search_time_ms": rt.total_search_time_ms,
            "exponential_average_search_time_per_hour_ms": rt.total_search_time_ms}),
    );
    if running && !st.upgrade_mode {
        let realtime = rt.end.is_none();
        let mut rs = Map::new();
        rs.insert("real_time_configured".into(), json!(realtime));
        rs.insert("real_time_running".into(), json!(realtime && super::feed::realtime_running(node, id)));
        rs.insert("search_interval".into(), json!({"start_ms": rt.start.unwrap_or(0), "end_ms": rt.end.unwrap_or_else(crate::date::now_millis)}));
        o.insert("running_state".into(), Value::Object(rs));
    }
    Value::Object(o)
}

/// `GET _ml/datafeeds[/{expr}]/_stats`.
pub fn datafeed_stats(node: &Node, expr: Option<&str>, allow_no_match: bool) -> Result<Value> {
    let st = service(node).read();
    let ids = expand_datafeeds(&st, expr, allow_no_match)?;
    let dfs: Vec<Value> = ids.iter().map(|id| datafeed_stats_entry(node, &st, id)).collect();
    Ok(json!({"count": dfs.len(), "datafeeds": dfs}))
}

/// `DELETE _ml/datafeeds/{id}`.
pub fn delete_datafeed(node: &Node, id: &str, force: bool) -> Result<Value> {
    let svc = service(node);
    svc.mutate(|st| {
        if !st.datafeeds.contains_key(id) {
            return Err(not_found_datafeed(id));
        }
        if !force && st.datafeed_state(id) != "stopped" {
            return Err(EsError::new(409, "status_exception", format!("Could not delete datafeed [{id}] because the datafeed is started")));
        }
        st.datafeeds.remove(id);
        st.datafeed_rt.remove(id);
        Ok(())
    })?;
    svc.sync_tasks(node);
    Ok(json!({"acknowledged": true}))
}

/// `POST _ml/datafeeds/{id}/_update`.
pub fn update_datafeed(node: &Node, id: &str, body: &Value, raw: &str, param_opts: &Map<String, Value>) -> Result<Value> {
    let svc = service(node);
    svc.mutate(|st| {
        let cur = st.datafeeds.get(id).cloned().ok_or_else(|| not_found_datafeed(id))?;
        let updated = datafeed::apply_update(&cur, body, raw, param_opts)?;
        if st.datafeed_state(id) != "stopped" {
            return Err(EsError::new(409, "status_exception", format!("Unable to update datafeed [{id}] as it is currently [{}]", st.datafeed_state(id))));
        }
        st.datafeeds.insert(id.to_string(), updated.clone());
        Ok(datafeed::public(&updated, false, node.security.enabled()))
    })
}

/// `TimeUtils.dateStringToEpoch` as the flush / post data time params use
/// it: all digits are epoch seconds when at most 10 of them, millis
/// otherwise; else an ISO date. Unparseable -> `parse_exception`.
pub fn parse_flush_time(s: &str, name: &str) -> Result<i64> {
    let t = s.trim();
    if !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(n) = t.parse::<i64>() {
            return Ok(if t.len() <= 10 { n * 1000 } else { n });
        }
    }
    crate::date::DateFormatter::new("strict_date_optional_time")
        .and_then(|f| f.parse(t))
        .map_err(|_| EsError::new(400, "parse_exception", format!("Query param [{name}] with value [{s}] cannot be parsed as a date or converted to a number (epoch).")))
}

/// Parse a datafeed `start`/`end` (epoch millis, ISO date or `now`).
pub fn parse_datafeed_time(s: &str, name: &str) -> Result<i64> {
    let t = s.trim();
    if t == "now" {
        return Ok(crate::date::now_millis());
    }
    if let Ok(n) = t.parse::<i64>() {
        return Ok(n);
    }
    crate::date::DateFormatter::new("strict_date_optional_time")
        .and_then(|f| f.parse(t))
        .map_err(|_| EsError::illegal_argument(format!("Query param [{name}] with value [{s}] cannot be parsed as a date or converted to a number (epoch).")))
}

/// Index/field checks `_start` performs (`DatafeedNodeSelector`,
/// `DatafeedJobValidator`).
fn check_datafeed_sources(node: &Node, id: &str, df: &Value, job: &Value) -> Result<()> {
    let indices: Vec<String> = df["indices"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
    let mut found_any = false;
    for i in &indices {
        if i.contains(':') {
            found_any = true;
            continue;
        }
        let r = node.resolve(i, crate::node::ResolveOpts { allow_no_indices: true, ..Default::default() });
        match r {
            Ok(t) if !t.is_empty() => found_any = true,
            _ if !i.contains('*') => {
                return Err(EsError::new(400, "status_exception", format!("datafeed [{id}] cannot retrieve data because index [{i}] does not exist")));
            }
            _ => {}
        }
    }
    if !found_any {
        return Err(EsError::new(400, "status_exception", format!("datafeed [{id}] cannot retrieve data because no index matches datafeed's indices [{}]", indices.join(", "))));
    }
    // Every analysis field must be mapped in the source (or be a runtime/script field).
    let mut mapped: std::collections::HashSet<String> = std::collections::HashSet::new();
    for i in &indices {
        if let Ok(targets) = node.resolve(i, crate::node::ResolveOpts { allow_no_indices: true, ..Default::default() }) {
            for t in targets {
                collect_fields(&t.svc.meta().mappings.to_json()["properties"], "", &mut mapped);
            }
        }
    }
    for k in df["runtime_mappings"].as_object().into_iter().flatten().map(|(k, _)| k).chain(df["script_fields"].as_object().into_iter().flatten().map(|(k, _)| k)) {
        mapped.insert(k.clone());
    }
    if df.get("aggregations").is_some() {
        return Ok(());
    }
    let mut needed: Vec<String> = Vec::new();
    for d in job["analysis_config"]["detectors"].as_array().into_iter().flatten() {
        for k in ["field_name", "by_field_name", "over_field_name", "partition_field_name"] {
            if let Some(f) = d[k].as_str() {
                if f != "mlcategory" {
                    needed.push(f.to_string());
                }
            }
        }
    }
    for k in ["summary_count_field_name", "categorization_field_name"] {
        if let Some(f) = job["analysis_config"][k].as_str() {
            needed.push(f.to_string());
        }
    }
    for f in needed {
        if !mapped.contains(&f) {
            return Err(EsError::new(400, "status_exception", format!("[{id}] cannot retrieve field [{f}] because it has no mappings")));
        }
    }
    Ok(())
}

fn collect_fields(props: &Value, prefix: &str, out: &mut std::collections::HashSet<String>) {
    for (k, v) in props.as_object().into_iter().flatten() {
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        out.insert(path.clone());
        collect_fields(&v["properties"], &path, out);
        for (fk, _) in v["fields"].as_object().into_iter().flatten() {
            out.insert(format!("{path}.{fk}"));
        }
    }
}

/// `POST _ml/datafeeds/{id}/_start`.
pub fn start_datafeed(node: &Arc<Node>, id: &str, start: Option<&str>, end: Option<&str>) -> Result<Value> {
    let svc = service(node);
    if svc.upgrade_mode() {
        return Err(EsError::new(400, "status_exception", "Cannot perform cluster:admin/xpack/ml/datafeed/start action while upgrade mode is enabled"));
    }
    let st = svc.read();
    let df = st.datafeeds.get(id).cloned().ok_or_else(|| not_found_datafeed(id))?;
    let start_ms = start.map(|s| parse_datafeed_time(s, "start")).transpose()?;
    let end_ms = end.map(|s| parse_datafeed_time(s, "end")).transpose()?;
    if let (Some(s), Some(e)) = (start_ms, end_ms) {
        if s >= e {
            return Err(EsError::new(400, "status_exception", format!("start [{s}] must be earlier than end [{e}]")).wrap_validation());
        }
    }
    let job_id = df["job_id"].as_str().unwrap_or("").to_string();
    let job = st.jobs.get(&job_id).cloned().ok_or_else(|| not_found_job(&job_id))?;
    let ds = st.datafeed_state(id).to_string();
    if ds != "stopped" {
        return Err(EsError::new(409, "status_exception", format!("cannot start datafeed [{id}] because it has already been started")));
    }
    let js = st.job_state(&job_id).to_string();
    if js != "opened" && js != "opening" {
        return Err(EsError::new(409, "status_exception", format!("cannot start datafeed [{id}] because job [{job_id}] is {js}")));
    }
    drop(st);
    check_datafeed_sources(node, id, &df, &job)?;
    if crate::ilm::cluster_setting(node, "cluster.persistent_tasks.allocation.enable").as_deref() == Some("none") {
        return Err(EsError::new(
            429,
            "status_exception",
            format!("Could not start datafeed [{id}], allocation explanation [no persistent task assignments are allowed due to cluster settings]"),
        ));
    }
    svc.mutate(|st| {
        st.datafeed_rt.insert(id.to_string(), DatafeedRt { state: "started".into(), start: Some(start_ms.unwrap_or(0)), end: end_ms, ..Default::default() });
        Ok(())
    })?;
    svc.sync_tasks(node);
    if js == "opened" {
        let _ = process::open(node, &job_id);
        super::feed::start(node.clone(), id.to_string(), df, job, start_ms, end_ms);
    }
    Ok(json!({"started": true, "node": NODE_ID}))
}

trait WrapValidation {
    fn wrap_validation(self) -> Self;
}
impl WrapValidation for EsError {
    fn wrap_validation(self) -> Self {
        let reason = format!("[{}] {}", "datafeed", self.reason);
        EsError::new(400, "status_exception", self.reason.clone()).caused_by(EsError::illegal_argument(reason))
    }
}

/// `POST _ml/datafeeds/{expr}/_stop`.
pub fn stop_datafeed(node: &Node, expr: &str, allow_no_match: bool, _force: bool) -> Result<Value> {
    let svc = service(node);
    svc.mutate(|st| {
        let ids = expand_datafeeds(st, Some(expr), allow_no_match)?;
        for id in ids {
            super::feed::stop(node, &id);
            if let Some(rt) = st.datafeed_rt.get_mut(&id) {
                rt.state = "stopped".into();
                rt.start = None;
                rt.end = None;
            }
        }
        Ok(())
    })?;
    svc.sync_tasks(node);
    Ok(json!({"stopped": true}))
}

// ---------------------------------------------------------------------------
// Node-level
// ---------------------------------------------------------------------------

/// `POST _ml/set_upgrade_mode?enabled=`.
pub fn set_upgrade_mode(node: &Node, enabled: bool) -> Result<Value> {
    let svc = service(node);
    svc.mutate(|st| {
        st.upgrade_mode = enabled;
        Ok(())
    })?;
    svc.sync_tasks(node);
    Ok(json!({"acknowledged": true}))
}

/// `DELETE _ml/_delete_expired_data[/{job_id}]`. 7b: results / snapshots
/// older than the retention settings would be removed here.
pub fn delete_expired_data(node: &Node, job_expr: Option<&str>) -> Result<Value> {
    let st = service(node).read();
    let ids = expand_jobs(&st, job_expr, true)?;
    let now = crate::date::now_millis();
    for id in ids {
        let job = &st.jobs[id.as_str()];
        if let Some(days) = job["results_retention_days"].as_i64() {
            let cutoff = now - days * 86_400_000;
            resultsidx::delete_by_query(
                node,
                &resultsidx::read_alias(&id),
                json!({"bool": {"filter": [{"terms": {"result_type": ["bucket", "record", "influencer", "bucket_influencer", "model_plot"]}}, {"range": {"timestamp": {"lt": cutoff}}}]}}),
            );
        }
        resultsidx::delete_by_query(node, &resultsidx::read_alias(&id), json!({"bool": {"filter": [{"term": {"result_type": "model_forecast_request_stats"}}, {"range": {"forecast_expiry_timestamp": {"lt": now}}}]}}));
    }
    Ok(json!({"deleted": true}))
}

/// `(anomaly_detectors_in_bytes)` of `_ml/memory/_stats`: open jobs'
/// model memory. ES reports the native process's actual usage; with no
/// process running that is 0.
pub fn anomaly_detector_memory(node: &Node) -> u64 {
    process::live_model_bytes(node)
}

/// Rows for `_cat/ml/anomaly_detectors`: (stats entry, config).
pub fn cat_job_rows(node: &Node, expr: Option<&str>, allow_no_match: bool) -> Result<Vec<(Value, Value)>> {
    let st = service(node).read();
    let ids = expand_jobs(&st, expr, allow_no_match)?;
    Ok(ids.iter().map(|id| (job_stats_entry(node, &st, id), st.jobs[id].clone())).collect())
}

/// Rows for `_cat/ml/datafeeds`.
pub fn cat_datafeed_rows(node: &Node, expr: Option<&str>, allow_no_match: bool) -> Result<Vec<Value>> {
    let st = service(node).read();
    let ids = expand_datafeeds(&st, expr, allow_no_match)?;
    Ok(ids.iter().map(|id| datafeed_stats_entry(node, &st, id)).collect())
}
