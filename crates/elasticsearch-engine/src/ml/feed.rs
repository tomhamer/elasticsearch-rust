//! The datafeed runner: a started datafeed searches its source indices
//! (`DatafeedJob`) and feeds the job's process ([`super::process`]).
//!
//! A lookback (`start`..`end`, or `start`..now minus `query_delay`) is
//! searched in time order through a scroll, `scroll_size` documents at a
//! time, so documents sharing a timestamp are each sent exactly once. A
//! failed source search keeps its window: the runner retries it with
//! exponential backoff (auditing the problem and, after
//! [`MAX_SEARCH_ATTEMPTS`] failures, stopping the datafeed) and never moves
//! past data it could not read. A datafeed
//! with an `end` then stops itself and closes the job, a real-time datafeed
//! keeps searching every `frequency`, flushing interim results after each
//! search.

use super::store::service;
use crate::node::Node;
use parking_lot::{Condvar, Mutex};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

struct Ctl {
    stop: AtomicBool,
    realtime_running: AtomicBool,
    done: Mutex<bool>,
    completed: Condvar,
    /// Held by the worker across each post / flush / close of the job,
    /// which it only makes while not stopped: [`fence`] takes it after
    /// stopping, so no post follows a stop, delete or replacement.
    post: Mutex<()>,
}

impl Ctl {
    fn new() -> Ctl {
        Ctl { stop: AtomicBool::new(false), realtime_running: AtomicBool::new(false), done: Mutex::new(false), completed: Condvar::new(), post: Mutex::new(()) }
    }
}

fn feeds() -> &'static Mutex<HashMap<String, Arc<Ctl>>> {
    static F: OnceLock<Mutex<HashMap<String, Arc<Ctl>>>> = OnceLock::new();
    F.get_or_init(Default::default)
}

fn key(node: &Node, id: &str) -> String {
    format!("{}#{id}", node.data.display())
}

/// Is the datafeed's real-time loop running?
pub fn realtime_running(node: &Node, id: &str) -> bool {
    feeds().lock().get(&key(node, id)).map(|c| c.realtime_running.load(Ordering::Relaxed)).unwrap_or(false)
}

/// `DatafeedConfig.defaultFrequency(bucketSpan)`.
fn default_frequency_ms(bucket_span_ms: i64) -> i64 {
    if bucket_span_ms <= 120_000 {
        60_000
    } else if bucket_span_ms <= 1_200_000 {
        bucket_span_ms
    } else if bucket_span_ms <= 43_200_000 {
        600_000
    } else {
        3_600_000
    }
}

fn time_ms(v: Option<&str>, default: i64) -> i64 {
    v.and_then(|s| super::job::parse_time(s, "t").ok()).map(|t| t.millis()).unwrap_or(default)
}

/// Start the runner thread of a started datafeed.
pub fn start(node: Arc<Node>, id: String, df: Value, job: Value, start: Option<i64>, end: Option<i64>) {
    let ctl = Arc::new(Ctl::new());
    let feed_key = key(&node, &id);
    // A previous worker of this id (a force-deleted datafeed's, still in a
    // search) is fenced before the new one runs: one worker per datafeed.
    let previous = feeds().lock().insert(feed_key.clone(), ctl.clone());
    if let Some(previous) = previous {
        previous.stop.store(true, Ordering::Relaxed);
        drop(previous.post.lock());
    }
    #[cfg(test)]
    tests::STARTED.lock().push((feed_key.clone(), ctl.clone()));
    let completion = ctl.clone();
    let completed_key = feed_key.clone();
    // The runner searches and writes through the request's coordinator.
    let coordinator = crate::coord::current();
    if std::thread::Builder::new().name(format!("ml-datafeed-{id}")).spawn(move || {
        struct Done(Arc<Ctl>, String);
        impl Drop for Done {
            fn drop(&mut self) {
                *self.0.done.lock() = true;
                self.0.completed.notify_all();
                let mut active = feeds().lock();
                if active.get(&self.1).is_some_and(|current| Arc::ptr_eq(current, &self.0)) { active.remove(&self.1); }
            }
        }
        let _done = Done(ctl.clone(), completed_key);
        crate::coord::scoped(coordinator, || run(node, id, df, job, start, end, ctl));
    }).is_err() {
        *completion.done.lock() = true;
        completion.completed.notify_all();
        let mut active = feeds().lock();
        if active.get(&feed_key).is_some_and(|current| Arc::ptr_eq(current, &completion)) { active.remove(&feed_key); }
    }
}

/// Signal the runner to stop (the thread exits after its current search).
pub fn stop(node: &Node, id: &str) {
    if let Some(c) = feeds().lock().get(&key(node, id)) {
        c.stop.store(true, Ordering::Relaxed);
    }
}

/// Is a worker registered for the datafeed?
pub fn running(node: &Node, id: &str) -> bool {
    feeds().lock().contains_key(&key(node, id))
}

/// Stop the datafeed's worker and wait out a post in flight: after this
/// returns the worker sends the job nothing more (it exits after its
/// current search).
pub fn fence(node: &Node, id: &str) {
    let Some(control) = feeds().lock().get(&key(node, id)).cloned() else { return };
    control.stop.store(true, Ordering::Relaxed);
    drop(control.post.lock());
    let mut active = feeds().lock();
    if active.get(&key(node, id)).is_some_and(|current| Arc::ptr_eq(current, &control)) {
        active.remove(&key(node, id));
    }
}

/// Test builds: drop this node's worker registrations (a process restart
/// has no workers).
#[cfg(test)]
pub(crate) fn forget_node(node: &Node) {
    let prefix = format!("{}#", node.data.display());
    let gone: Vec<Arc<Ctl>> = feeds().lock().iter().filter(|(k, _)| k.starts_with(&prefix)).map(|(_, c)| c.clone()).collect();
    for c in &gone {
        c.stop.store(true, Ordering::Relaxed);
        drop(c.post.lock());
    }
    feeds().lock().retain(|k, _| !k.starts_with(&prefix));
}

/// Feature reset must wait for the last search/write before deleting indices.
pub fn stop_and_wait(node: &Node, id: &str, timeout: Duration) -> crate::error::Result<()> {
    let Some(control) = feeds().lock().get(&key(node, id)).cloned() else { return Ok(()); };
    control.stop.store(true, Ordering::Relaxed);
    let deadline = Instant::now() + timeout;
    let mut done = control.done.lock();
    while !*done {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(crate::error::EsError::new(503, "timeout_exception", format!("timed out stopping datafeed [{id}] before feature reset")));
        }
        control.completed.wait_for(&mut done, remaining);
    }
    drop(done);
    let mut active = feeds().lock();
    if active.get(&key(node, id)).is_some_and(|current| Arc::ptr_eq(current, &control)) { active.remove(&key(node, id)); }
    Ok(())
}

fn record_search(node: &Node, id: &str, took_ms: f64) {
    let svc = service(node);
    let _ = svc.mutate(|st| {
        if let Some(rt) = st.datafeed_rt.get_mut(id) {
            rt.search_count += 1;
            rt.total_search_time_ms += took_ms;
        }
        Ok(())
    });
}

/// Attempts at one window's source search before the datafeed stops
/// (the first try plus five retries with exponential backoff).
const MAX_SEARCH_ATTEMPTS: u32 = 6;
/// First retry delay; doubled per failed attempt, capped at [`MAX_BACKOFF_MS`].
static RETRY_BASE_MS: AtomicU64 = AtomicU64::new(1000);
const MAX_BACKOFF_MS: u64 = 60_000;

fn backoff_ms(failures: u32) -> u64 {
    RETRY_BASE_MS.load(Ordering::Relaxed).saturating_mul(1u64 << failures.saturating_sub(1).min(16)).min(MAX_BACKOFF_MS)
}

/// Sleep `ms`, waking early on stop; false when stopped.
fn sleep_unless_stopped(ctl: &Ctl, ms: u64) -> bool {
    let deadline = Instant::now() + Duration::from_millis(ms);
    while Instant::now() < deadline {
        if ctl.stop.load(Ordering::Relaxed) {
            return false;
        }
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(50)));
    }
    !ctl.stop.load(Ordering::Relaxed)
}

fn set_search_problem(node: &Node, id: &str, problem: Option<(u32, String)>) {
    let _ = service(node).mutate(|st| {
        if let Some(rt) = st.datafeed_rt.get_mut(id) {
            match problem {
                Some((n, reason)) => {
                    rt.consecutive_search_failures = n;
                    rt.search_problem = Some(reason);
                }
                None => {
                    rt.consecutive_search_failures = 0;
                    rt.search_problem = None;
                }
            }
        }
        Ok(())
    });
}

fn audit(node: &Node, job_id: &str, level: &str, message: &str) {
    super::resultsidx::audit(node, job_id, "anomaly_detector", level, message);
}

/// One page of the window (test builds can inject a failure here).
fn search_page(node: &Node, id: &str, ex: &mut super::preview::Extractor, job: &Value) -> crate::error::Result<Vec<Value>> {
    #[cfg(test)]
    if let Some(e) = tests::injected_failure(node, id) {
        return Err(e);
    }
    let _ = id;
    ex.next(node, job)
}

fn run(node: Arc<Node>, id: String, df: Value, job: Value, start: Option<i64>, end: Option<i64>, ctl: Arc<Ctl>) {
    let job_id = job["job_id"].as_str().unwrap_or("").to_string();
    let span_ms = time_ms(job["analysis_config"]["bucket_span"].as_str(), 300_000).max(1000);
    let frequency_ms = df["frequency"].as_str().map(|f| time_ms(Some(f), span_ms)).unwrap_or_else(|| default_frequency_ms(span_ms));
    let query_delay_ms = time_ms(df["query_delay"].as_str(), 60_000);
    let scroll_size = df["scroll_size"].as_u64().unwrap_or(1000).clamp(1, 10_000) as usize;
    let chunk_ms = super::datafeed::chunk_span_ms(&df);
    let mut next = start.unwrap_or(0);
    let mut failures = 0u32;
    loop {
        if ctl.stop.load(Ordering::Relaxed) {
            return;
        }
        let upper = match end {
            Some(e) => e,
            None => crate::date::now_millis() - query_delay_ms,
        };
        // Lookback / catch-up: everything in [next, upper), one window. The
        // window only counts as searched once it was read to its end; a
        // failed search retries the same page, never moving past it.
        // Searched one chunk (`chunking_config`) at a time: a chunk counts
        // as searched once read to its end, then the next one starts.
        while next < upper {
            let chunk_end = chunk_ms.map_or(upper, |c| next.saturating_add(c).min(upper));
            let mut ex = super::preview::Extractor::new(&df, &job, next, chunk_end, scroll_size);
            loop {
                if ctl.stop.load(Ordering::Relaxed) {
                    return;
                }
                let t0 = Instant::now();
                let fetched = search_page(&node, &id, &mut ex, &job);
                record_search(&node, &id, t0.elapsed().as_secs_f64() * 1000.0);
                let docs = match fetched {
                    Ok(docs) => docs,
                    Err(e) => {
                        failures += 1;
                        let reason = e.reason.clone();
                        set_search_problem(&node, &id, Some((failures, reason.clone())));
                        if failures == 1 {
                            // ES `DatafeedJob` / `ProblemTracker`.
                            audit(&node, &job_id, "warning", &format!("Datafeed is encountering errors extracting data: {reason}"));
                        }
                        if failures >= MAX_SEARCH_ATTEMPTS {
                            audit(&node, &job_id, "error", &format!(
                                "Datafeed [{id}] stopped after [{failures}] consecutive failed searches of [{next}, {upper}); no data after [{next}] was sent to the job: {reason}"
                            ));
                            feeds().lock().remove(&key(&node, &id));
                            let _ = super::anomaly::stop_datafeed(&node, &id, true, false);
                            audit(&node, &job_id, "info", "Datafeed stopped");
                            return;
                        }
                        if !sleep_unless_stopped(&ctl, backoff_ms(failures)) {
                            return;
                        }
                        continue;
                    }
                };
                if failures > 0 {
                    failures = 0;
                    set_search_problem(&node, &id, None);
                    audit(&node, &job_id, "info", "Datafeed has recovered data extraction and analysis");
                }
                if docs.is_empty() {
                    next = chunk_end;
                    break;
                }
                #[cfg(test)]
                tests::before_post(&node, &id);
                let bytes: usize = docs.iter().map(|d| d.to_string().len() + 1).sum();
                let posted = {
                    let _post = ctl.post.lock();
                    if ctl.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    #[cfg(test)]
                    tests::record_posted(&node, &id, &docs);
                    super::process::post_docs(&node, &job_id, docs, bytes, true)
                };
                if let Err(e) = posted {
                    // Re-searching the window would send its earlier pages
                    // twice: stop instead (ES stops a datafeed whose job can
                    // no longer accept data).
                    audit(&node, &job_id, "error", &format!("Datafeed is encountering errors submitting data for analysis: {}", e.reason));
                    feeds().lock().remove(&key(&node, &id));
                    let _ = super::anomaly::stop_datafeed(&node, &id, true, false);
                    return;
                }
            }
        }
        if ctl.stop.load(Ordering::Relaxed) {
            return;
        }
        if end.is_some() {
            // The lookback is complete (`DatafeedJob.runLookBack`: a final
            // flush with interim results): the datafeed stops and the job closes.
            {
                let _post = ctl.post.lock();
                if ctl.stop.load(Ordering::Relaxed) {
                    return;
                }
                let _ = super::process::flush(&node, &job_id, &super::anomaly::FlushParams { calc_interim: true, ..Default::default() });
            }
            feeds().lock().remove(&key(&node, &id));
            let _ = super::anomaly::stop_datafeed(&node, &id, true, false);
            let _ = super::anomaly::close_job(&node, &job_id, true, false);
            return;
        }
        ctl.realtime_running.store(true, Ordering::Relaxed);
        {
            let _post = ctl.post.lock();
            if ctl.stop.load(Ordering::Relaxed) {
                return;
            }
            // A failed flush keeps its results pending in the job (written
            // by the next post / flush): nothing is skipped.
            let _ = super::process::flush(&node, &job_id, &super::anomaly::FlushParams { calc_interim: true, advance_time: Some(upper), ..Default::default() });
        }
        // Sleep until the next frequency tick, waking early on stop.
        if !sleep_unless_stopped(&ctl, frequency_ms.max(0) as u64) {
            return;
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::index::WriteReq;
    use crate::shard::WriteCond;
    use serde_json::json;

    /// Per feed key: (search calls so far, calls that fail).
    static INJECT: Mutex<Vec<(String, u32, Vec<u32>)>> = Mutex::new(Vec::new());
    /// Every runner started (to wait for it).
    pub(super) static STARTED: Mutex<Vec<(String, Arc<Ctl>)>> = Mutex::new(Vec::new());
    /// Per feed key: every document posted to the job.
    static POSTED: Mutex<Vec<(String, Vec<Value>)>> = Mutex::new(Vec::new());

    pub(super) fn injected_failure(node: &Node, id: &str) -> Option<crate::error::EsError> {
        let k = key(node, id);
        let mut inj = INJECT.lock();
        let e = inj.iter_mut().find(|(x, _, _)| *x == k)?;
        e.1 += 1;
        e.2.contains(&e.1).then(|| crate::error::EsError::new(503, "search_phase_execution_exception", format!("injected failure of search {}", e.1)))
    }

    /// Feed keys whose worker waits before its next post (a source read in
    /// flight), and the keys whose worker reached that point.
    static HOLD: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static REACHED: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub(super) fn before_post(node: &Node, id: &str) {
        let k = key(node, id);
        if !HOLD.lock().contains(&k) {
            return;
        }
        REACHED.lock().push(k.clone());
        while HOLD.lock().contains(&k) {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !ok() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn workers(node: &Node, id: &str) -> usize {
        let k = key(node, id);
        STARTED.lock().iter().filter(|(x, c)| *x == k && !*c.done.lock()).count()
    }

    /// #153: force-deleting a datafeed whose worker holds a source read
    /// fences it: the old worker posts nothing after the delete, and the
    /// recreated datafeed runs exactly one worker that sends every record
    /// once.
    #[test]
    fn force_delete_fences_the_worker() {
        let (_d, node, mut uids) = setup(5, 100);
        let k = key(&node, "df");
        HOLD.lock().push(k.clone());
        super::super::anomaly::start_datafeed(&node, "df", Some(&(T - 10_000).to_string()), Some(&(T + 10_000).to_string())).unwrap();
        wait_for("the worker's source read", || REACHED.lock().contains(&k));
        super::super::anomaly::delete_datafeed(&node, "df", true).unwrap();
        HOLD.lock().retain(|x| *x != k);
        wait_for("the old worker to exit", || workers(&node, "df") == 0);
        assert!(posted(&node, "df").is_empty(), "the deleted datafeed's worker posted after the delete");
        let df = json!({"job_id": "j", "indices": ["src"], "scroll_size": 100});
        super::super::anomaly::put_datafeed(&node, "df", &df, &df.to_string(), &Default::default()).unwrap();
        run_lookback(&node, Vec::new());
        uids.sort();
        assert_eq!(uids_of(&posted(&node, "df")), uids, "the replacement sent every record exactly once");
    }

    /// #152: a datafeed started when the node restarts gets its worker back
    /// (exactly one), from the job's latest record, and new source records
    /// are processed. The restart drops the in-process runtime (workers,
    /// processes), as a process restart does.
    #[test]
    fn started_datafeed_resumes_after_restart() {
        let (_d, node, mut uids) = setup(3, 100);
        super::super::anomaly::start_datafeed(&node, "df", Some(&(T - 10_000).to_string()), None).unwrap();
        uids.sort();
        wait_for("the lookback", || uids_of(&posted(&node, "df")) == uids);
        // Restart: the worker and the job's process are gone; the datafeed
        // and job are still recorded as started / opened.
        super::super::anomaly::forget_runtime(&node);
        wait_for("the old worker to exit", || workers(&node, "df") == 0);
        assert_eq!(service(&node).read().datafeed_state("df"), "started");
        let svc = node.get_index("src").unwrap();
        let doc = WriteReq::Index { id: Some("late".into()), source: json!({"time": T + 2000, "uid": "late"}).to_string(), routing: None, cond: WriteCond::default() };
        assert!(svc.write(vec![doc], false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        assert_eq!(super::super::anomaly::resume_datafeeds(&node), vec!["df".to_string()]);
        assert!(super::super::anomaly::resume_datafeeds(&node).is_empty(), "resumed once");
        assert_eq!(workers(&node, "df"), 1);
        wait_for("the new record", || posted(&node, "df").iter().any(|d| d["uid"] == "late"));
        let late = posted(&node, "df").iter().filter(|d| d["uid"] == "late").count();
        assert_eq!(late, 1);
        let resent = posted(&node, "df").len() - uids.len() - 1;
        assert_eq!(resent, 0, "the resumed worker starts after the job's latest record");
        super::super::anomaly::stop_datafeed(&node, "df", false, false).unwrap();
        wait_for("the worker to stop", || workers(&node, "df") == 0);
    }

    /// A job (count by uid over pre-aggregated rows) and its datafeed
    /// `id` over `index` with `aggregations` (and `chunking`).
    fn aggregated_feed(node: &Arc<Node>, id: &str, index: &str, aggregations: Value, chunking: Option<Value>) {
        let job_id = format!("{id}-job");
        let job = json!({"analysis_config": {"bucket_span": "1h", "summary_count_field_name": "doc_count", "detectors": [{"function": "count", "by_field_name": "uid"}]}, "data_description": {"time_field": "time"}});
        super::super::anomaly::put_job(node, &job_id, &job, &job.to_string(), &Default::default()).unwrap();
        super::super::anomaly::open_job(node, &job_id).unwrap();
        let mut df = json!({"job_id": job_id, "indices": [index], "aggregations": aggregations});
        if let Some(c) = chunking {
            df["chunking_config"] = c;
        }
        super::super::anomaly::put_datafeed(node, id, &df, &df.to_string(), &Default::default()).unwrap();
    }

    fn composite(size: usize) -> Value {
        json!({"buckets": {"composite": {"size": size, "sources": [{"time_bucket": {"date_histogram": {"field": "time", "fixed_interval": "1m"}}}, {"uid": {"terms": {"field": "uid"}}}]},
            "aggregations": {"time": {"max": {"field": "time"}}}}})
    }

    /// #208: a composite aggregation datafeed pages through every
    /// composite page (`after_key`) and keeps each source key (here `uid`)
    /// as a field of its row.
    #[test]
    fn composite_datafeed_pages_and_keeps_keys() {
        let (_d, node, mut uids) = setup(3, 100);
        aggregated_feed(&node, "cf", "src", composite(2), None);
        run_lookback_of(&node, "cf", T - 10_000, T + 10_000);
        uids.sort();
        let rows = posted(&node, "cf");
        assert_eq!(uids_of(&rows), uids, "a row per composite bucket, each with its uid, over {} pages", uids.len().div_ceil(2));
        assert!(rows.iter().all(|r| r["doc_count"] == json!(1) && r["time"].as_i64().is_some()));
    }

    /// #209: a partial aggregation response (one shard failed) is retried:
    /// every bucket reaches the job exactly once.
    #[test]
    fn partial_aggregation_response_is_retried() {
        let (_d, node, mut uids) = setup(3, 100);
        aggregated_feed(&node, "pf", "src", composite(4), None);
        super::super::preview::PARTIAL.lock().push((format!("{}#src", node.data.display()), 2));
        run_lookback_of(&node, "pf", T - 10_000, T + 10_000);
        uids.sort();
        assert_eq!(uids_of(&posted(&node, "pf")), uids, "nothing lost or repeated");
        let a = audits(&node);
        assert!(a.iter().any(|(l, m)| l == "warning" && m.contains("Partial shard failure")), "{a:?}");
        assert_eq!(service(&node).read().job_state("pf-job"), "closed", "the lookback completed");
    }

    /// #210: an aggregated lookback whose histogram would exceed the bucket
    /// limit in one search runs in `chunking_config` time spans; with
    /// chunking off the same lookback cannot run (the control).
    #[test]
    fn manual_chunking_partitions_aggregated_lookback() {
        RETRY_BASE_MS.store(1, Ordering::Relaxed);
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let svc = node.create_index("wide", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"time": {"type": "date"}, "uid": {"type": "keyword"}}}})).unwrap();
        let day = 86_400_000i64;
        let docs: Vec<WriteReq> = [0, 25, 50].iter().map(|d| WriteReq::Index { id: Some(format!("d{d}")), source: json!({"time": T + d * day, "uid": format!("d{d}")}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        assert!(svc.write(docs, false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        let histogram = json!({"buckets": {"date_histogram": {"field": "time", "fixed_interval": "1m"}, "aggregations": {"time": {"max": {"field": "time"}}, "uid": {"terms": {"field": "uid"}}}}});
        aggregated_feed(&node, "chunked", "wide", histogram.clone(), Some(json!({"mode": "manual", "time_span": "1d"})));
        run_lookback_of(&node, "chunked", T - day, T + 51 * day);
        assert_eq!(uids_of(&posted(&node, "chunked")), vec!["d0", "d25", "d50"]);
        assert_eq!(service(&node).read().job_state("chunked-job"), "closed", "the lookback completed");
        // Control: one search over the 50 days exceeds the bucket limit.
        aggregated_feed(&node, "whole", "wide", histogram, Some(json!({"mode": "off"})));
        run_lookback_of(&node, "whole", T - day, T + 51 * day);
        assert!(posted(&node, "whole").is_empty());
        assert_eq!(service(&node).read().job_state("whole-job"), "opened", "the unchunked lookback failed");
    }

    /// #207: upgrade mode halts the datafeed worker and the job process
    /// (persisted as a snapshot) before it is acknowledged: new source data
    /// and posted input reach nothing meanwhile; ending it resumes them
    /// without a gap.
    #[test]
    fn upgrade_mode_halts_and_resumes_workers() {
        let (_d, node, mut uids) = setup(3, 100);
        super::super::anomaly::start_datafeed(&node, "df", Some(&(T - 10_000).to_string()), None).unwrap();
        uids.sort();
        wait_for("the lookback", || uids_of(&posted(&node, "df")) == uids);
        super::super::anomaly::set_upgrade_mode(&node, true).unwrap();
        assert_eq!(workers(&node, "df"), 0, "the worker stopped before the acknowledgement");
        assert!(!super::super::process::is_live(&node, "j"), "the job process was persisted and released");
        assert!(service(&node).read().jobs["j"]["model_snapshot_id"].is_string());
        let svc = node.get_index("src").unwrap();
        let doc = WriteReq::Index { id: Some("during".into()), source: json!({"time": T + 2000, "uid": "during"}).to_string(), routing: None, cond: WriteCond::default() };
        assert!(svc.write(vec![doc], false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        let e = super::super::anomaly::post_data(&node, "j", format!("{}\n", json!({"time": T + 3000, "uid": "x"})).as_bytes()).unwrap_err();
        assert!(e.reason.contains("upgrade mode"), "{e:?}");
        assert!(super::super::anomaly::resume_datafeeds(&node).is_empty(), "nothing resumes during upgrade mode");
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(posted(&node, "df").len(), uids.len(), "nothing posted during upgrade mode");
        super::super::anomaly::set_upgrade_mode(&node, false).unwrap();
        assert_eq!(super::super::anomaly::resume_datafeeds(&node), vec!["df".to_string()]);
        wait_for("the record written during upgrade mode", || posted(&node, "df").iter().any(|d| d["uid"] == "during"));
        assert_eq!(posted(&node, "df").len(), uids.len() + 1, "resumed after the latest record, no gap, no repeat");
        super::super::anomaly::stop_datafeed(&node, "df", false, false).unwrap();
        wait_for("the worker to stop", || workers(&node, "df") == 0);
    }

    pub(super) fn record_posted(node: &Node, id: &str, docs: &[Value]) {
        let k = key(node, id);
        let mut p = POSTED.lock();
        match p.iter_mut().find(|(x, _)| *x == k) {
            Some(e) => e.1.extend(docs.iter().cloned()),
            None => p.push((k, docs.to_vec())),
        }
    }

    fn posted(node: &Node, id: &str) -> Vec<Value> {
        let k = key(node, id);
        POSTED.lock().iter().find(|(x, _)| *x == k).map(|(_, d)| d.clone()).unwrap_or_default()
    }

    const T: i64 = 1_700_000_000_000;

    /// A source index whose docs carry a unique `uid`: `ties` of them at
    /// `T`, a few before and after; a job (count by uid) and a datafeed of
    /// `scroll_size` over it.
    fn setup(ties: usize, scroll_size: usize) -> (tempfile::TempDir, Arc<Node>, Vec<String>) {
        RETRY_BASE_MS.store(1, Ordering::Relaxed);
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let svc = node.create_index("src", &json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0},
            "mappings": {"properties": {"time": {"type": "date"}, "uid": {"type": "keyword"}}}})).unwrap();
        let mut uids = Vec::new();
        let mut reqs = Vec::new();
        let mut add = |uid: String, t: i64| {
            reqs.push(WriteReq::Index { id: Some(uid.clone()), source: json!({"time": t, "uid": uid}).to_string(), routing: None, cond: WriteCond::default() });
            uids.push(uid);
        };
        for i in 0..3 {
            add(format!("before-{i}"), T - 1000 + i);
        }
        for i in 0..ties {
            add(format!("tie-{i}"), T);
        }
        for i in 0..4 {
            add(format!("after-{i}"), T + 1000);
        }
        // Two segments per shard, so ties span segments too.
        let half = reqs.len() / 2;
        let rest = reqs.split_off(half);
        for batch in [reqs, rest] {
            assert!(svc.write(batch, false).iter().all(|r| r.is_ok()));
            svc.refresh().unwrap();
        }
        let job = json!({"analysis_config": {"bucket_span": "1h", "detectors": [{"function": "count", "by_field_name": "uid"}]}, "data_description": {"time_field": "time"}});
        super::super::anomaly::put_job(&node, "j", &job, &job.to_string(), &Default::default()).unwrap();
        super::super::anomaly::open_job(&node, "j").unwrap();
        let df = json!({"job_id": "j", "indices": ["src"], "scroll_size": scroll_size});
        super::super::anomaly::put_datafeed(&node, "df", &df, &df.to_string(), &Default::default()).unwrap();
        (data, node, uids)
    }

    fn run_lookback(node: &Arc<Node>, fail_calls: Vec<u32>) {
        INJECT.lock().push((key(node, "df"), 0, fail_calls));
        run_lookback_of(node, "df", T - 10_000, T + 10_000);
    }

    /// Run datafeed `id` over `[start, end)` to its end.
    fn run_lookback_of(node: &Arc<Node>, id: &str, start: i64, end: i64) {
        super::super::anomaly::start_datafeed(node, id, Some(&start.to_string()), Some(&end.to_string())).unwrap();
        let k = key(node, id);
        let ctl = STARTED.lock().iter().rev().find(|(x, _)| *x == k).map(|(_, c)| c.clone());
        {
            let ctl = ctl.expect("runner started");
            let mut done = ctl.done.lock();
            let deadline = Instant::now() + Duration::from_secs(60);
            while !*done {
                assert!(Instant::now() < deadline, "datafeed did not finish");
                ctl.completed.wait_for(&mut done, Duration::from_millis(100));
            }
        }
    }

    fn uids_of(docs: &[Value]) -> Vec<String> {
        let mut u: Vec<String> = docs.iter().map(|d| d["uid"].as_str().unwrap().to_string()).collect();
        u.sort();
        u
    }

    fn audits(node: &Node) -> Vec<(String, String)> {
        let Ok(r) = super::super::search(node, super::super::resultsidx::NOTIFICATIONS_INDEX, &json!({"size": 100, "sort": [{"timestamp": "asc"}]})) else { return Vec::new() };
        r["hits"]["hits"].as_array().into_iter().flatten().map(|h| (h["_source"]["level"].as_str().unwrap().to_string(), h["_source"]["message"].as_str().unwrap().to_string())).collect()
    }

    /// #49: more than one page of documents share one millisecond; every
    /// document reaches the job exactly once, in time order.
    #[test]
    fn lookback_sends_every_tied_document_exactly_once() {
        let (_d, node, mut uids) = setup(23, 5);
        run_lookback(&node, Vec::new());
        let docs = posted(&node, "df");
        let times: Vec<i64> = docs.iter().map(|d| d["time"].as_i64().unwrap()).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]), "time order: {times:?}");
        uids.sort();
        assert_eq!(uids_of(&docs), uids, "every record exactly once");
        let st = service(&node).read();
        assert_eq!(st.datafeed_state("df"), "stopped");
        assert_eq!(st.job_state("j"), "closed");
    }

    /// The extractor over a window: pages of `size`, ties across pages,
    /// nothing skipped or repeated.
    #[test]
    fn extractor_pages_ties_across_scroll_pages() {
        let (_d, node, mut uids) = setup(12, 4);
        let st = service(&node).read();
        let (df, job) = (st.datafeeds["df"].clone(), st.jobs["j"].clone());
        drop(st);
        let mut ex = super::super::preview::Extractor::new(&df, &job, T - 10_000, T + 10_000, 4);
        let mut all = Vec::new();
        let mut pages = 0;
        loop {
            let page = ex.next(&node, &job).unwrap();
            if page.is_empty() {
                break;
            }
            assert!(page.len() <= 4);
            pages += 1;
            all.extend(page);
        }
        assert!(pages >= 5, "{pages}");
        uids.sort();
        assert_eq!(uids_of(&all), uids);
        // The window bound is exclusive: nothing at T is in [T-10s, T).
        let mut ex = super::super::preview::Extractor::new(&df, &job, T - 10_000, T, 4);
        let mut before = Vec::new();
        loop {
            let page = ex.next(&node, &job).unwrap();
            if page.is_empty() {
                break;
            }
            before.extend(page);
        }
        assert_eq!(uids_of(&before), vec!["before-0", "before-1", "before-2"]);
    }

    /// #598: a restarted datafeed resumes after the job's latest processed
    /// record, whatever earlier start it is given: nothing the job already
    /// has is sent again.
    #[test]
    fn restarted_datafeed_resumes_after_the_latest_record() {
        let (_d, node, mut uids) = setup(3, 4);
        run_lookback_of(&node, "df", T - 10_000, T);
        assert_eq!(uids_of(&posted(&node, "df")), vec!["before-0", "before-1", "before-2"]);
        super::super::anomaly::open_job(&node, "j").unwrap();
        run_lookback_of(&node, "df", T - 10_000, T + 10_000);
        uids.sort();
        assert_eq!(uids_of(&posted(&node, "df")), uids, "every record exactly once across both runs");
    }

    /// #599: of two starts that both found the datafeed stopped, only one
    /// claims it; the other is refused.
    #[test]
    fn concurrent_starts_launch_one_worker() {
        let (_d, node, _) = setup(3, 4);
        let starts: Vec<_> = (0..2).map(|_| {
            let node = node.clone();
            std::thread::spawn(move || super::super::anomaly::start_datafeed(&node, "df", None, None).map(|_| ()).map_err(|e| e.status))
        }).collect();
        let results: Vec<Result<(), u16>> = starts.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1, "{results:?}");
        assert!(results.iter().any(|r| *r == Err(409)), "{results:?}");
        assert!(workers(&node, "df") <= 1);
        super::super::anomaly::stop_datafeed(&node, "df", false, false).unwrap();
        wait_for("the worker to stop", || workers(&node, "df") == 0);
    }

    /// #560: a lost scroll resumes by identity, not by a count: an already
    /// delivered tied document deleted before the loss does not hide its
    /// unread neighbour, and one inserted at the boundary is read once.
    #[test]
    fn lost_scroll_resumes_tied_documents_by_identity() {
        let (_d, node, _) = setup(3, 1);
        let st = service(&node).read();
        let (df, job) = (st.datafeeds["df"].clone(), st.jobs["j"].clone());
        drop(st);
        let mut ex = super::super::preview::Extractor::new(&df, &job, T, T + 1, 1);
        let first = ex.next(&node, &job).unwrap();
        assert_eq!(first.len(), 1);
        let delivered = first[0]["uid"].as_str().unwrap().to_string();
        let svc = node.get_index("src").unwrap();
        assert!(svc.write(vec![WriteReq::Delete { id: delivered.clone(), routing: None, cond: WriteCond::default() },
            WriteReq::Index { id: Some("tie-new".into()), source: json!({"time": T, "uid": "tie-new"}).to_string(), routing: None, cond: WriteCond::default() }], false).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        ex.lose_scroll();
        assert!(ex.next(&node, &job).is_err(), "the lost page fails");
        let mut rest = Vec::new();
        loop {
            let page = ex.next(&node, &job).unwrap();
            if page.is_empty() {
                break;
            }
            rest.extend(page);
        }
        let mut got: Vec<String> = rest.iter().map(|d| d["uid"].as_str().unwrap().to_string()).collect();
        got.sort();
        let mut want: Vec<String> = ["tie-0", "tie-1", "tie-2", "tie-new"].iter().map(|s| s.to_string()).filter(|u| *u != delivered).collect();
        want.sort();
        assert_eq!(got, want, "every unread tied document once (delivered {delivered})");
    }

    /// #50: a failed source search (here the second page of the window) is
    /// retried; the complete window reaches the job before the datafeed
    /// moves on, and the problem and recovery are audited.
    #[test]
    fn failed_search_is_retried_without_losing_the_window() {
        let (_d, node, mut uids) = setup(9, 4);
        run_lookback(&node, vec![2]);
        uids.sort();
        assert_eq!(uids_of(&posted(&node, "df")), uids, "the failed page was neither skipped nor repeated");
        let a = audits(&node);
        assert!(a.iter().any(|(l, m)| l == "warning" && m.starts_with("Datafeed is encountering errors extracting data: injected failure of search 2")), "{a:?}");
        assert!(a.iter().any(|(l, m)| l == "info" && m == "Datafeed has recovered data extraction and analysis"), "{a:?}");
        let st = service(&node).read();
        assert_eq!(st.job_state("j"), "closed", "the lookback completed");
        assert!(st.datafeed_rt["df"].search_problem.is_none());
    }

    /// The very first search failing (twice) still sends the whole window.
    #[test]
    fn failed_first_search_keeps_the_window() {
        let (_d, node, mut uids) = setup(6, 3);
        run_lookback(&node, vec![1, 2]);
        uids.sort();
        assert_eq!(uids_of(&posted(&node, "df")), uids);
    }

    /// Repeated failures stop the datafeed with an error audit; nothing past
    /// the failed page is sent and the job is not closed as if complete.
    #[test]
    fn repeated_failures_stop_the_datafeed_without_advancing() {
        let (_d, node, _uids) = setup(9, 4);
        run_lookback(&node, (2..=1 + MAX_SEARCH_ATTEMPTS).collect());
        let docs = posted(&node, "df");
        assert_eq!(docs.len(), 4, "only the first page was sent");
        let a = audits(&node);
        assert!(a.iter().any(|(l, m)| l == "error" && m.contains("stopped after [6] consecutive failed searches")), "{a:?}");
        let st = service(&node).read();
        assert_eq!(st.datafeed_state("df"), "stopped");
        assert_eq!(st.job_state("j"), "opened", "a failed lookback does not close the job as complete");
    }

    #[test]
    fn reset_waits_for_worker_even_after_stop_was_requested() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let control = Arc::new(Ctl::new());
        control.realtime_running.store(true, Ordering::Relaxed);
        feeds().lock().insert(key(&node, "running-feed"), control.clone());
        stop(&node, "running-feed");
        assert!(control.stop.load(Ordering::Relaxed));
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || send.send(stop_and_wait(&node, "running-feed", Duration::from_secs(5))).unwrap());
        assert!(matches!(receive.recv_timeout(Duration::from_millis(30)), Err(std::sync::mpsc::RecvTimeoutError::Timeout)), "reset acknowledged before the worker finished");
        *control.done.lock() = true;
        control.completed.notify_all();
        receive.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        worker.join().unwrap();
    }
}
