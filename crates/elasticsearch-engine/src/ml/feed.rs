//! The datafeed runner: a started datafeed searches its source indices
//! (`DatafeedJob`) and feeds the job's process ([`super::process`]).
//!
//! A lookback (`start`..`end`, or `start`..now minus `query_delay`) is
//! searched in time order, `scroll_size` documents at a time; a datafeed
//! with an `end` then stops itself and closes the job, a real-time datafeed
//! keeps searching every `frequency`, flushing interim results after each
//! search.

use super::store::service;
use crate::node::Node;
use parking_lot::{Condvar, Mutex};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

struct Ctl {
    stop: AtomicBool,
    realtime_running: AtomicBool,
    done: Mutex<bool>,
    completed: Condvar,
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
    let ctl = Arc::new(Ctl { stop: AtomicBool::new(false), realtime_running: AtomicBool::new(false), done: Mutex::new(false), completed: Condvar::new() });
    let feed_key = key(&node, &id);
    feeds().lock().insert(feed_key.clone(), ctl.clone());
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

fn run(node: Arc<Node>, id: String, df: Value, job: Value, start: Option<i64>, end: Option<i64>, ctl: Arc<Ctl>) {
    let job_id = job["job_id"].as_str().unwrap_or("").to_string();
    let time_field = job["data_description"]["time_field"].as_str().unwrap_or("time").to_string();
    let span_ms = time_ms(job["analysis_config"]["bucket_span"].as_str(), 300_000).max(1000);
    let frequency_ms = df["frequency"].as_str().map(|f| time_ms(Some(f), span_ms)).unwrap_or_else(|| default_frequency_ms(span_ms));
    let query_delay_ms = time_ms(df["query_delay"].as_str(), 60_000);
    let scroll_size = df["scroll_size"].as_u64().unwrap_or(1000).clamp(1, 10_000) as usize;
    let aggregated = df.get("aggregations").map(|a| a.is_object()).unwrap_or(false);
    let mut next = start.unwrap_or(0);
    loop {
        if ctl.stop.load(Ordering::Relaxed) {
            return;
        }
        let upper = match end {
            Some(e) => e,
            None => crate::date::now_millis() - query_delay_ms,
        };
        // Lookback / catch-up: everything in [next, upper).
        while next < upper && !ctl.stop.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            let fetched = super::preview::fetch(&node, &df, &job, Some(next), Some(upper), scroll_size);
            record_search(&node, &id, t0.elapsed().as_secs_f64() * 1000.0);
            let docs = match fetched {
                Ok(Value::Array(a)) => a,
                _ => Vec::new(),
            };
            if docs.is_empty() {
                next = upper;
                break;
            }
            let last = docs.iter().filter_map(|d| d.get(&time_field).and_then(Value::as_i64)).max().unwrap_or(upper);
            let n = docs.len();
            let bytes: usize = docs.iter().map(|d| d.to_string().len() + 1).sum();
            if super::process::post_docs(&node, &job_id, docs, bytes, true).is_err() {
                break;
            }
            if aggregated || n < scroll_size {
                next = upper;
            } else {
                next = last + 1;
            }
        }
        if ctl.stop.load(Ordering::Relaxed) {
            return;
        }
        if end.is_some() {
            // The lookback is complete (`DatafeedJob.runLookBack`: a final
            // flush with interim results): the datafeed stops and the job closes.
            let _ = super::process::flush(&node, &job_id, &super::anomaly::FlushParams { calc_interim: true, ..Default::default() });
            feeds().lock().remove(&key(&node, &id));
            let _ = super::anomaly::stop_datafeed(&node, &id, true, false);
            let _ = super::anomaly::close_job(&node, &job_id, true, false);
            return;
        }
        ctl.realtime_running.store(true, Ordering::Relaxed);
        let _ = super::process::flush(&node, &job_id, &super::anomaly::FlushParams { calc_interim: true, advance_time: Some(upper), ..Default::default() });
        // Sleep until the next frequency tick, waking early on stop.
        let mut waited = 0i64;
        while waited < frequency_ms {
            if ctl.stop.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
            waited += 200;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_waits_for_worker_even_after_stop_was_requested() {
        let data = tempfile::tempdir().unwrap();
        let node = Node::open(data.path()).unwrap();
        let control = Arc::new(Ctl { stop: AtomicBool::new(false), realtime_running: AtomicBool::new(true), done: Mutex::new(false), completed: Condvar::new() });
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
