//! Async search store (`_async_search`): a search runs on its own thread;
//! submit/get wait up to `wait_for_completion_timeout` for it and otherwise
//! return a partial (empty, single-shard) response with the id. Results are
//! kept until `keep_alive` expires (ES keeps them in the `.async-search`
//! system index; here in memory).

use crate::error::EsError;
use crate::node::Node;
use parking_lot::{Condvar, Mutex};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub enum AsyncState {
    Running,
    /// Final search response (raw JSON) or error.
    Done(std::result::Result<String, EsError>),
}

pub struct AsyncSearch {
    pub id: String,
    pub start_millis: i64,
    pub shards: usize,
    inner: Mutex<Inner>,
    cv: Condvar,
    /// Response warnings the search raised (replayed on every retrieval).
    pub warnings: Mutex<Vec<String>>,
}

struct Inner {
    state: AsyncState,
    retained: Option<crate::memory::Reservation>,
    completion_millis: Option<i64>,
    expiration_millis: i64,
    expires: Instant,
}

/// An `AsyncExecutionId`-encoded id for this node (decodes like ES's).
pub fn new_id() -> String {
    crate::esql::async_id::new_id(crate::node::NODE_ID).0
}

impl AsyncSearch {
    pub fn new(shards: usize, keep_alive: Duration) -> Arc<AsyncSearch> {
        let now = crate::date::now_millis();
        Arc::new(AsyncSearch {
            id: new_id(),
            start_millis: now,
            shards,
            inner: Mutex::new(Inner { state: AsyncState::Running, retained: None, completion_millis: None, expiration_millis: now + keep_alive.as_millis() as i64, expires: Instant::now() + keep_alive }),
            cv: Condvar::new(),
            warnings: Mutex::new(Vec::new()),
        })
    }

    /// Copy a borrowed response only after reserving its retained storage.
    /// Used when the producer owns shared bytes, so transferring a String is
    /// impossible without creating a second allocation.
    pub fn complete_borrowed(&self, text: &str) {
        let prepared = (|| {
            let breakers = crate::memory::breakers();
            let memory = crate::memory::scoped(None, || breakers.reserve_request(text.len() as u64, "async search retained result"))?;
            let mut owned = String::new();
            owned.try_reserve_exact(text.len()).map_err(|error| crate::memory::allocation_refused(error.to_string(),text.len() as u64))?;
            owned.push_str(text);
            Ok::<_, EsError>((owned,memory))
        })();
        let mut g = self.inner.lock();
        // Drop old data before releasing the old retained guard.
        g.state = AsyncState::Running;
        g.retained = None;
        match prepared {
            Ok((text,memory)) => { g.retained = Some(memory); g.state = AsyncState::Done(Ok(text)); }
            Err(error) => { g.state = AsyncState::Done(Err(error)); }
        }
        g.completion_millis = Some(crate::date::now_millis());
        self.cv.notify_all();
    }

    pub fn complete(&self, r: std::result::Result<String, EsError>) {
        let mut g = self.inner.lock();
        // The store outlives the submit request. Admit ownership before
        // retaining the buffer, and keep its guard with the stored result.
        let retained=r.as_ref().ok().map(|text| {
            let breakers=crate::memory::breakers();
            crate::memory::scoped(None,||breakers.reserve_request(text.capacity() as u64,"async search retained result"))
        }).transpose();
        match retained {
            Ok(held)=>{g.retained=held;g.state=AsyncState::Done(r);},
            Err(error)=>{g.retained=None;g.state=AsyncState::Done(Err(error));},
        }
        g.completion_millis = Some(crate::date::now_millis());
        self.cv.notify_all();
    }

    /// Block until done or `timeout` elapsed; returns whether done.
    pub fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut g = self.inner.lock();
        while matches!(g.state, AsyncState::Running) {
            if self.cv.wait_until(&mut g, deadline).timed_out() {
                break;
            }
        }
        !matches!(g.state, AsyncState::Running)
    }

    pub fn is_running(&self) -> bool {
        matches!(self.inner.lock().state, AsyncState::Running)
    }

    pub fn extend(&self, keep_alive: Duration) {
        let mut g = self.inner.lock();
        g.expires = Instant::now() + keep_alive;
        g.expiration_millis = crate::date::now_millis() + keep_alive.as_millis() as i64;
    }

    pub fn expired(&self) -> bool {
        self.inner.lock().expires <= Instant::now()
    }

    /// The `_async_search` response body and HTTP status. `with_id` is
    /// false when a completed search was not stored.
    pub fn response(&self, with_id: bool) -> crate::error::Result<(u16, Value)> {
        let g = self.inner.lock();
        let mut m = Map::new();
        if with_id {
            m.insert("id".into(), json!(self.id));
        }
        let mut status = 200;
        match &g.state {
            AsyncState::Running => {
                m.insert("is_partial".into(), json!(true));
                m.insert("is_running".into(), json!(true));
                m.insert("start_time_in_millis".into(), json!(self.start_millis));
                m.insert("expiration_time_in_millis".into(), json!(g.expiration_millis));
                m.insert(
                    "response".into(),
                    json!({
                        "took": crate::date::now_millis() - self.start_millis, "timed_out": false, "terminated_early": false, "num_reduce_phases": 0,
                        "_shards": {"total": self.shards, "successful": 0, "skipped": 0, "failed": 0},
                        "hits": {"total": {"value": 0, "relation": "gte"}, "max_score": null, "hits": []}
                    }),
                );
            }
            AsyncState::Done(r) => {
                let failed = r.is_err();
                m.insert("is_partial".into(), json!(failed));
                m.insert("is_running".into(), json!(false));
                m.insert("start_time_in_millis".into(), json!(self.start_millis));
                m.insert("expiration_time_in_millis".into(), json!(g.expiration_millis));
                if let Some(c) = g.completion_millis {
                    m.insert("completion_time_in_millis".into(), json!(c));
                }
                match r {
                    Ok(s) => {
                        // Admit the decoded tree at its own size, measured
                        // by a pass that builds nothing (a fixed multiple of
                        // the text made any result over ~1/65 of the limit
                        // unfetchable). The response owns this charge through
                        // HTTP rendering.
                        let guard=crate::memory::breakers().reserve_request(crate::memory::json_text_bytes(s)?,"async search response decode")?;
                        m.insert("response".into(), serde_json::from_str(s).unwrap_or(Value::Null));
                        guard.retain_for_request();
                    }
                    Err(e) => {
                        status = e.status;
                        m.insert(
                            "response".into(),
                            json!({
                                "took": g.completion_millis.unwrap_or(self.start_millis) - self.start_millis, "timed_out": false, "num_reduce_phases": 0,
                                "_shards": {"total": self.shards, "successful": 0, "skipped": 0, "failed": self.shards},
                                "hits": {"total": {"value": 0, "relation": "gte"}, "max_score": null, "hits": []}
                            }),
                        );
                        m.insert("error".into(), e.error_object(false));
                    }
                }
            }
        }
        Ok((status, Value::Object(m)))
    }

    /// `GET /_async_search/status/{id}` body.
    pub fn status(&self) -> Value {
        let g = self.inner.lock();
        let mut m = Map::new();
        m.insert("id".into(), json!(self.id));
        let (running, partial, ok) = match &g.state {
            AsyncState::Running => (true, true, None),
            AsyncState::Done(Ok(_)) => (false, false, Some(200)),
            AsyncState::Done(Err(e)) => (false, true, Some(e.status)),
        };
        m.insert("is_running".into(), json!(running));
        m.insert("is_partial".into(), json!(partial));
        m.insert("start_time_in_millis".into(), json!(self.start_millis));
        m.insert("expiration_time_in_millis".into(), json!(g.expiration_millis));
        if let Some(c) = g.completion_millis {
            m.insert("completion_time_in_millis".into(), json!(c));
        }
        let ok_shards = if ok == Some(200) { self.shards } else { 0 };
        let failed = if ok.map(|s| s != 200).unwrap_or(false) { self.shards } else { 0 };
        m.insert("_shards".into(), json!({"total": self.shards, "successful": ok_shards, "skipped": 0, "failed": failed}));
        if let Some(s) = ok {
            m.insert("completion_status".into(), json!(s));
        }
        Value::Object(m)
    }
}

impl Node {
    /// Look up a live async search (expired ones are dropped).
    pub fn async_search(&self, id: &str) -> Option<Arc<AsyncSearch>> {
        let mut m = self.async_searches.lock();
        m.retain(|_, s| !s.expired());
        m.get(id).cloned()
    }

    /// Drop expired async searches (also done on lookup; run periodically
    /// so results nobody fetches again do not stay in memory).
    pub fn reap_async_searches(&self) {
        self.async_searches.lock().retain(|_, s| !s.expired());
    }

    pub fn async_search_missing(id: &str) -> EsError {
        EsError::resource_not_found(id.to_string())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn retained_results_are_admitted_and_released_with_store_owner() {
        let breakers=crate::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let result=super::AsyncSearch::new(1,std::time::Duration::from_secs(60));
        let worker=crate::memory::RequestScope::new("async worker",breakers);
        crate::memory::scoped(Some(worker),||result.complete(Ok(format!("{{\"value\":\"{}\"}}","x".repeat(2048)))));
        assert!(breakers.request.reserved()>=2048,"stored result outlives worker scope");
        let request=crate::memory::RequestScope::new("async retrieval",breakers);
        let decoded=crate::memory::scoped(Some(request.clone()),||result.response(true));
        let (status,body)=decoded.expect("a stored result that fits is fetchable");
        assert_eq!(status,200);
        assert_eq!(body["response"]["value"].as_str().map(str::len),Some(2048));
        assert!(request.held()>=2048,"the decoded tree is charged for the response");
        drop(request);
        assert!(breakers.request.reserved()>=2048,"stored result stays charged after retrieval");
        // Every retrieval (wait_for_completion included) decodes the same way.
        for _ in 0..3 {
            let request=crate::memory::RequestScope::new("async retrieval again",breakers);
            assert_eq!(crate::memory::scoped(Some(request),||result.response(true)).unwrap().0,200);
        }
        drop(result);
        assert_eq!(breakers.request.reserved(),0);

        let result=super::AsyncSearch::new(1,std::time::Duration::from_secs(60));
        let worker=crate::memory::RequestScope::new("oversized async worker",breakers);
        crate::memory::scoped(Some(worker),||result.complete(Ok("x".repeat(100000))));
        assert_eq!(result.status()["completion_status"],429);
        assert_eq!(breakers.request.reserved(),0,"rejected result buffer is not retained");
    }

    #[test]
    fn borrowed_completion_admits_before_copy_and_retains_exact_storage() {
        use crate::memory::{Breakers,RequestScope,scoped};
        let budget = Breakers::leak_for_tests(1 << 30);
        budget.apply_settings(&|key|match key {
            "indices.breaker.request.limit" => Some("4kb".into()),
            "indices.breaker.total.limit" => Some("-1".into()), _ => None,
        });
        let result = super::AsyncSearch::new(1,std::time::Duration::from_secs(60));
        let source = "x".repeat(8192);
        scoped(Some(RequestScope::new("borrowed refusal",budget)),||result.complete_borrowed(&source));
        assert_eq!(result.status()["completion_status"],429);
        assert_eq!(budget.request.reserved(),0);
        let source = "{}".repeat(512);
        scoped(Some(RequestScope::new("borrowed success",budget)),||result.complete_borrowed(&source));
        assert!(budget.request.reserved() >= source.len() as u64);
        drop(source);
        drop(result);
        assert_eq!(budget.request.reserved(),0);
    }

    #[test]
    fn expired_async_searches_are_reaped_without_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(dir.path()).unwrap();
        let live = super::AsyncSearch::new(1, std::time::Duration::from_secs(60));
        let done = super::AsyncSearch::new(1, std::time::Duration::from_millis(1));
        done.complete(Ok("{}".into()));
        node.async_searches.lock().insert("live".into(), live);
        node.async_searches.lock().insert("done".into(), done);
        std::thread::sleep(std::time::Duration::from_millis(5));
        node.reap_async_searches();
        assert_eq!(node.async_searches.lock().keys().cloned().collect::<Vec<_>>(), vec!["live".to_string()]);
    }
}
