//! Asynchronous query-language execution (SQL, ES|QL, EQL async APIs): the
//! query runs on a background worker; the submit request waits at most
//! `wait_for_completion_timeout` and then answers with the result, or with
//! a running handle (`is_running: true`) the client retrieves, polls or
//! deletes later (review finding 610). Every saved result carries its
//! `keep_alive` deadline (`__expires`, epoch millis), enforced on reads and
//! by the node's maintenance reaper (finding 611).
//!
//! The per-language maps of `search_ext::ExtState` (`esql_async`,
//! `sql_async`, `eql_async`) hold, under the result's document key, either
//! the stored completed response of that language (as its GET renders it),
//! a stored failure (`__error`, `__status`), or a running placeholder
//! (`__running: true`). Running jobs are also registered here so a GET can
//! wait for them.

use crate::Req;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search_ext::ext;
use std::sync::{Condvar, Mutex};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Wait on `cv` until `deadline`; `true` when it passed.
fn wait_until<'a, T>(cv: &Condvar, g: std::sync::MutexGuard<'a, T>, deadline: Instant) -> (std::sync::MutexGuard<'a, T>, bool) {
    let now = Instant::now();
    if now >= deadline {
        return (g, true);
    }
    match cv.wait_timeout(g, deadline - now) {
        Ok((g, r)) => (g, r.timed_out()),
        Err(p) => {
            let (g, r) = p.into_inner();
            (g, r.timed_out())
        }
    }
}

/// The language whose result map a job uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Lang {
    Esql,
    Sql,
    Eql,
}

/// Run `f` on `node`'s `lang` result map.
fn with_map<R>(node: &Node, lang: Lang, f: impl FnOnce(&mut HashMap<String, Value>) -> R) -> R {
    let state = ext(node);
    let mut m = match lang {
        Lang::Esql => state.esql_async.lock(),
        Lang::Sql => state.sql_async.lock(),
        Lang::Eql => state.eql_async.lock(),
    };
    f(&mut m)
}

/// Async queries running at once on this process: beyond it a submit is
/// rejected (ES runs them on the bounded `search` pool).
const MAX_RUNNING: usize = 1000;
static RUNNING: AtomicUsize = AtomicUsize::new(0);

/// A job's wake-up for waiting GETs.
struct Signal {
    lock: Mutex<()>,
    done: Condvar,
    /// The query's task token: cancelled when its result is deleted (ES
    /// cancels the running async task on DELETE), so the work stops at its
    /// next check (search loops, ES|QL commands and row evaluation).
    cancel: Arc<elasticsearch_engine::rustic_core::cancel::Token>,
}

struct Job<T> {
    /// `Some` once the query ended (taken by the submit).
    result: Mutex<Option<Result<T>>>,
    /// The submit stopped waiting: the worker saves the result itself.
    detached: Mutex<bool>,
    done: Condvar,
    signal: Arc<Signal>,
}

/// Running (detached) jobs by result document key, for GETs that wait.
fn jobs() -> &'static Mutex<HashMap<String, Arc<Signal>>> {
    static JOBS: OnceLock<Mutex<HashMap<String, Arc<Signal>>>> = OnceLock::new();
    JOBS.get_or_init(Default::default)
}

/// What a submit answers.
pub(crate) enum Outcome<T> {
    /// Completed within the wait (saved under `id` only with
    /// `keep_on_completion`).
    Done { id: String, value: T },
    /// Still running: a handle (`id`) to retrieve it with.
    Running { id: String },
}

/// The fields every saved entry carries.
fn stamp(doc: &mut Value, id: &str, owner: &Value, start: i64, expires: i64) {
    if let Some(o) = doc.as_object_mut() {
        o.entry("id").or_insert_with(|| json!(id));
        o.insert("__owner".into(), owner.clone());
        o.insert("__start".into(), json!(start));
        o.insert("__expires".into(), json!(expires));
    }
}

fn failure_doc(e: &EsError) -> Value {
    json!({"__error": e.to_json(), "__status": e.status})
}

/// Run `job` in the background for `node`'s `lang` results. `id` / `key`:
/// the async id and its document key; `owner`: the stored owner; `wait`:
/// how long the submit waits; `keep_on_completion`: save a result that
/// completed within the wait; `keep_alive`: how long a saved result is
/// kept; `to_stored`: the saved form of a result (what that language's
/// GET renders).
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit<T: Send + 'static>(
    node: &Arc<Node>, req: &Req, lang: Lang, id: String, key: String, owner: Value, wait: Duration, keep_on_completion: bool, keep_alive: Duration,
    job: impl FnOnce(&Node, &Req) -> Result<T> + Send + 'static,
    to_stored: impl Fn(&T) -> Value + Send + Sync + 'static,
) -> Result<Outcome<T>> {
    if RUNNING.fetch_add(1, Ordering::SeqCst) >= MAX_RUNNING {
        RUNNING.fetch_sub(1, Ordering::SeqCst);
        return Err(EsError::new(429, "es_rejected_execution_exception", format!("rejected execution of async {lang:?} query: [{MAX_RUNNING}] already running")));
    }
    // The maintenance reaper cancels what it expires (see `cancel`).
    elasticsearch_engine::search_ext::set_async_expiry_cancel(cancel);
    let start = elasticsearch_engine::date::now_millis();
    let expires = start.saturating_add(keep_alive.as_millis() as i64);
    let to_stored = Arc::new(to_stored);
    let state: Arc<Job<T>> = Arc::new(Job { result: Mutex::new(None), detached: Mutex::new(false), done: Condvar::new(), signal: Arc::new(Signal { lock: Mutex::new(()), done: Condvar::new(), cancel: elasticsearch_engine::rustic_core::cancel::Token::new() }) });
    let (worker_node, worker_req, worker_state, worker_store) = (node.clone(), req.detached_clone(), state.clone(), to_stored.clone());
    let (worker_key, worker_id, worker_owner) = (key.clone(), id.clone(), owner.clone());
    let worker_token = state.signal.cancel.clone();
    let coordinator = elasticsearch_engine::coord::current();
    let api_version = elasticsearch_engine::version::current_rest_api_version();
    let inference = elasticsearch_engine::inference::scoped_current();
    let scope = elasticsearch_engine::memory::RequestScope::new("async query worker", elasticsearch_engine::memory::breakers());
    let spawned = std::thread::Builder::new().name("es-async-query".into()).spawn(move || {
        let r = elasticsearch_engine::memory::scoped(Some(scope), || {
            elasticsearch_engine::coord::scoped(coordinator, || {
                elasticsearch_engine::version::with_rest_api_version(api_version, || {
                    elasticsearch_engine::inference::scoped(inference, || elasticsearch_engine::rustic_core::cancel::with_search(Some(worker_token), || job(&worker_node, &worker_req)))
                })
            })
        });
        // Under the job's lock: either the submit takes the result, or it
        // detached and the worker saves it (unless the entry was deleted).
        let detached = {
            let mut result = lock(&worker_state.result);
            let detached = *lock(&worker_state.detached);
            if detached {
                let mut doc = match &r {
                    Ok(v) => worker_store(v),
                    Err(e) => failure_doc(e),
                };
                stamp(&mut doc, &worker_id, &worker_owner, start, expires);
                with_map(&worker_node, lang, |stored| {
                    if stored.get(&worker_key).is_some_and(is_running) {
                        stored.insert(worker_key.clone(), doc);
                    }
                });
            } else {
                *result = Some(r);
            }
            worker_state.done.notify_all();
            detached
        };
        if detached {
            lock(jobs()).remove(&worker_key);
            let _g = lock(&worker_state.signal.lock);
            worker_state.signal.done.notify_all();
        }
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    });
    if let Err(e) = spawned {
        RUNNING.fetch_sub(1, Ordering::SeqCst);
        return Err(EsError::internal(format!("failed to start the async query: {e}")));
    }
    let deadline = Instant::now() + wait;
    let mut result = lock(&state.result);
    while result.is_none() {
        let (g, timed_out) = wait_until(&state.done, result, deadline);
        result = g;
        if timed_out {
            break;
        }
    }
    if let Some(r) = result.take() {
        drop(result);
        let value = r?;
        if keep_on_completion {
            let mut stored = to_stored(&value);
            stamp(&mut stored, &id, &owner, start, expires);
            with_map(node, lang, |m| m.insert(key, stored));
        }
        return Ok(Outcome::Done { id, value });
    }
    // Detach: a running placeholder the worker replaces when it ends.
    let mut placeholder = json!({"__running": true});
    stamp(&mut placeholder, &id, &owner, start, expires);
    with_map(node, lang, |m| m.insert(key.clone(), placeholder));
    lock(jobs()).insert(key, state.signal.clone());
    *lock(&state.detached) = true;
    drop(result);
    Ok(Outcome::Running { id })
}

/// A saved entry of `lang` under `key` (`None`: none, or expired, which is
/// then dropped). A GET with `wait` waits that long for a running one.
pub(crate) fn get(node: &Node, lang: Lang, key: &str, wait: Option<Duration>) -> Option<Value> {
    if let Some(wait) = wait.filter(|w| !w.is_zero()) {
        let signal = lock(jobs()).get(key).cloned();
        if let Some(signal) = signal {
            let deadline = Instant::now() + wait;
            let mut g = lock(&signal.lock);
            while with_map(node, lang, |m| m.get(key).is_some_and(is_running)) {
                let (g2, timed_out) = wait_until(&signal.done, g, deadline);
                g = g2;
                if timed_out {
                    break;
                }
            }
        }
    }
    let now = elasticsearch_engine::date::now_millis();
    let mut expired_running = false;
    let found = with_map(node, lang, |stored| {
        let expired = stored.get(key).is_some_and(|v| v["__expires"].as_i64().is_some_and(|e| e <= now));
        if expired {
            expired_running = stored.remove(key).is_some_and(|v| is_running(&v));
            return None;
        }
        stored.get(key).cloned()
    });
    // ES: an expired async query's task is cancelled, not left running.
    if expired_running {
        cancel(key);
    }
    found
}

/// Cancel the running query whose result is `key` (its result was
/// deleted): ES's async DELETE cancels the task. `false` when none runs.
pub(crate) fn cancel(key: &str) -> bool {
    match lock(jobs()).get(key).cloned() {
        Some(signal) => {
            signal.cancel.cancel("by user request");
            true
        }
        None => false,
    }
}

/// Whether a saved entry is a running placeholder.
pub(crate) fn is_running(v: &Value) -> bool {
    v["__running"] == true
}

/// The stored failure of an entry, as the error response.
pub(crate) fn failure(v: &Value) -> Option<crate::Resp> {
    let e = v.get("__error")?;
    let status = v["__status"].as_u64().unwrap_or(500) as u16;
    Some(crate::Resp::json(status, json!({"error": e, "status": status})))
}

/// The internal fields of a saved entry removed (what a response shows).
pub(crate) fn public(mut v: Value) -> Value {
    if let Some(o) = v.as_object_mut() {
        for k in ["__owner", "__start", "__expires", "__running", "__id"] {
            o.remove(k);
        }
    }
    v
}

/// A duration request option (`wait_for_completion_timeout`, `keep_alive`).
pub(crate) fn duration(v: Option<&str>, default_ms: i64) -> Duration {
    let ms = v.and_then(elasticsearch_engine::date::parse_time_value).unwrap_or(default_ms);
    Duration::from_millis(ms.max(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> Req {
        Req { method: axum::http::Method::POST, path: "/_sql".into(), segs: vec!["_sql".into()], params: Default::default(),
            body: Default::default(), headers: Default::default(), access: None, extensions: axum::http::Extensions::new(), api_version: crate::RestApiVersion::V9 }
    }

    /// Review finding 610: a query outliving `wait_for_completion_timeout`
    /// answers with a running handle at once; a GET that waits gets its
    /// result once it ends; a fast one completes within the wait. Finding
    /// 611: a saved result past its `keep_alive` is gone (reads and the
    /// maintenance reaper).
    #[test]
    fn async_queries_run_in_the_background_and_expire() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Arc::new(Node::open(tmp.path()).unwrap());
        let r = req();
        let started = Instant::now();
        let out = submit(&node, &r, Lang::Sql, "slow-id".into(), "slow".into(), Value::Null, Duration::from_millis(50), false, Duration::from_secs(60),
            |_, _| { std::thread::sleep(Duration::from_millis(600)); Ok(json!({"rows": [[1]]})) }, |v: &Value| v.clone()).unwrap();
        assert!(matches!(out, Outcome::Running { ref id } if id == "slow-id"));
        assert!(started.elapsed() < Duration::from_millis(500), "the submit waited for the query");
        let running = get(&node, Lang::Sql, "slow", None).unwrap();
        assert!(is_running(&running));
        let done = get(&node, Lang::Sql, "slow", Some(Duration::from_secs(5))).unwrap();
        assert!(!is_running(&done), "{done}");
        assert_eq!(done["rows"], json!([[1]]));
        // Fast: within the wait, saved only with keep_on_completion.
        let out = submit(&node, &r, Lang::Sql, "fast-id".into(), "fast".into(), Value::Null, Duration::from_secs(5), false, Duration::from_secs(60), |_, _| Ok(json!({"rows": []})), |v: &Value| v.clone()).unwrap();
        assert!(matches!(out, Outcome::Done { .. }));
        assert!(get(&node, Lang::Sql, "fast", None).is_none());
        // A failure of a detached query is saved as its error.
        submit(&node, &r, Lang::Sql, "bad-id".into(), "bad".into(), Value::Null, Duration::from_millis(10), false, Duration::from_secs(60),
            |_, _| -> Result<Value> { std::thread::sleep(Duration::from_millis(200)); Err(EsError::illegal_argument("boom")) }, |v: &Value| v.clone()).unwrap();
        let failed = get(&node, Lang::Sql, "bad", Some(Duration::from_secs(5))).unwrap();
        assert_eq!(failure(&failed).map(|r| r.status), Some(400));
        // Expiry: on read, and by the maintenance reaper.
        submit(&node, &r, Lang::Esql, "old-id".into(), "old".into(), Value::Null, Duration::from_secs(5), true, Duration::from_millis(100), |_, _| Ok(json!({})), |v: &Value| v.clone()).unwrap();
        submit(&node, &r, Lang::Eql, "old2-id".into(), "old2".into(), Value::Null, Duration::from_secs(5), true, Duration::from_millis(100), |_, _| Ok(json!({})), |v: &Value| v.clone()).unwrap();
        assert!(get(&node, Lang::Esql, "old", None).is_some());
        std::thread::sleep(Duration::from_millis(200));
        assert!(get(&node, Lang::Esql, "old", None).is_none(), "an expired result was served");
        elasticsearch_engine::search_ext::reap_expired_async_results(&node);
        assert!(ext(&node).eql_async.lock().get("old2").is_none(), "the reaper kept an expired result");
    }

    /// ES cancels a running async query when its result is deleted: the
    /// work runs under its task token and stops at its next check; a query
    /// still running whose result was not deleted keeps running.
    #[test]
    fn deleting_a_running_query_cancels_its_work() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Arc::new(Node::open(tmp.path()).unwrap());
        let r = req();
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let iterations = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (s2, i2) = (stopped.clone(), iterations.clone());
        let out = submit(&node, &r, Lang::Esql, "c-id".into(), "c-key".into(), Value::Null, Duration::from_millis(20), false, Duration::from_secs(60),
            move |_, _| -> Result<Value> {
                // A long query: polls its task's cancellation as the search
                // and ES|QL loops do.
                for _ in 0..10_000 {
                    if let Err(e) = elasticsearch_engine::search::check_cancelled() {
                        s2.store(true, std::sync::atomic::Ordering::SeqCst);
                        return Err(e);
                    }
                    i2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok(json!({}))
            }, |v: &Value| v.clone()).unwrap();
        assert!(matches!(out, Outcome::Running { .. }));
        std::thread::sleep(Duration::from_millis(50));
        assert!(!stopped.load(std::sync::atomic::Ordering::SeqCst));
        // The DELETE: the entry goes, the task is cancelled.
        with_map(&node, Lang::Esql, |m| m.remove("c-key"));
        assert!(cancel("c-key"));
        let deadline = Instant::now() + Duration::from_secs(5);
        while !stopped.load(std::sync::atomic::Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the deleted query kept running");
            std::thread::sleep(Duration::from_millis(5));
        }
        let at_stop = iterations.load(std::sync::atomic::Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(iterations.load(std::sync::atomic::Ordering::SeqCst), at_stop, "work went on after the cancel");
        assert!(get(&node, Lang::Esql, "c-key", None).is_none(), "a cancelled result came back");
        assert!(!cancel("never-submitted"));
    }

    /// ES: an expired running async query is cancelled, whether the expiry
    /// is seen by a GET or by the maintenance reaper.
    #[test]
    fn expiring_a_running_query_cancels_its_work() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Arc::new(Node::open(tmp.path()).unwrap());
        let r = req();
        let start = |key: &str| {
            let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let s2 = stopped.clone();
            let out = submit(&node, &r, Lang::Esql, format!("{key}-id"), key.into(), Value::Null, Duration::from_millis(20), false, Duration::from_millis(150),
                move |_, _| -> Result<Value> {
                    for _ in 0..10_000 {
                        if let Err(e) = elasticsearch_engine::search::check_cancelled() {
                            s2.store(true, std::sync::atomic::Ordering::SeqCst);
                            return Err(e);
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Ok(json!({}))
                }, |v: &Value| v.clone()).unwrap();
            assert!(matches!(out, Outcome::Running { .. }));
            stopped
        };
        let wait_stopped = |stopped: &std::sync::atomic::AtomicBool, how: &str| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !stopped.load(std::sync::atomic::Ordering::SeqCst) {
                assert!(Instant::now() < deadline, "the query expired by {how} kept running");
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        let by_get = start("x-get");
        let by_reaper = start("x-reap");
        std::thread::sleep(Duration::from_millis(300));
        assert!(get(&node, Lang::Esql, "x-get", None).is_none());
        wait_stopped(&by_get, "GET");
        elasticsearch_engine::search_ext::reap_expired_async_results(&node);
        wait_stopped(&by_reaper, "the reaper");
    }
}
