//! The cluster's implementation of the ILM steps that move documents or
//! depend on where shard copies are ([`ClusterLifecycleSteps`]).
//!
//! Lifecycle passes run on the master's metadata copy (`scratch`), which
//! holds no documents: the engine's own shrink would publish an empty
//! target and then delete the source, its force-merge would merge nothing,
//! and its history writes would create the history data stream on the
//! copy. Here those steps go through the ordinary cluster paths, as ES's
//! steps send requests to the cluster: the resize REST path (target created
//! through the master, documents copied from the source copies, see
//! `resize`), `_forcemerge` (every copy), and `_bulk` for the history.
//! Placement waits read the routing table. Long work is started on a
//! step's first call and reported on later passes (the step waits); a
//! source is never deleted before its shrink target holds its documents.

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ShardState;
use elasticsearch_engine::ilm::{ClusterLifecycleSteps, StepProgress};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

/// A finished task's outcome is kept this long (steps read it on their
/// next passes), then forgotten.
const KEEP_DONE: Duration = Duration::from_secs(600);

enum Task {
    Running,
    Done(Instant, Result<(), String>),
}

pub struct ClusterSteps {
    node: Weak<ClusterNode>,
    runtime: tokio::runtime::Handle,
    tasks: Arc<Mutex<HashMap<String, Task>>>,
    /// ILM history documents waiting for the one writer in flight (`None`:
    /// no writer).
    history: Arc<Mutex<Option<Vec<Value>>>>,
}

/// History documents kept while a write is in flight (older ones beyond
/// are dropped: history is best effort, as in ES).
const HISTORY_BACKLOG: usize = 10_000;

fn request(method: &str, uri: std::fmt::Arguments<'_>, body: Option<Value>) -> elasticsearch_engine::error::Result<RestReq> {
    let headers=crate::front::SharedHeaders::try_pairs(&[(crate::front::SYSTEM_HEADER,"true"),("content-type","application/json")])?;
    let mut req=RestReq::try_new_uri(method,uri,&headers,&[],0,Default::default(),None)?;
    if let Some(body)=body { req.set_json(&body)?; } Ok(req)
}
async fn dispatch(node:&Arc<ClusterNode>,req:elasticsearch_engine::error::Result<RestReq>)->RestResp {
    match req { Ok(req)=>crate::front::dispatch(node,req).await, Err(e)=>RestResp::json(e.status,&e.to_json()) }
}

fn encode(s: &str) -> crate::front::UriComponent<'_> { crate::front::UriComponent(s) }

async fn count(node: &Arc<ClusterNode>, index: &str) -> Result<u64, String> {
    // Finish admitted URI construction before suspension: fmt::Arguments is not Send.
    let prepared_request = request("POST", format_args!("/{}/_refresh", encode(index)), None);
    let refreshed = dispatch(node, prepared_request).await;
    if !refreshed.is_success() { return Err(format!("refreshing [{index}] answered {}: {}",refreshed.status,String::from_utf8_lossy(&refreshed.body))); }
    let prepared_request = request("GET", format_args!("/{}/_count?expand_wildcards=all", encode(index)), None);
    let r = dispatch(node, prepared_request).await;
    if !r.is_success() {
        return Err(format!("counting the documents of [{index}] answered {}", r.status));
    }
    let v = r.body_json().ok_or_else(|| format!("counting the documents of [{index}]: no body"))?;
    // Every shard must answer: a missing shard lowers both counts alike.
    let shards = &v["_shards"];
    let (total, successful, failed) = (shards["total"].as_u64(), shards["successful"].as_u64(), shards["failed"].as_u64());
    if failed != Some(0) || total.is_none() || successful != total {
        return Err(format!("counting the documents of [{index}]: not every shard answered ({shards})"));
    }
    v["count"].as_u64().ok_or_else(|| format!("counting the documents of [{index}]: no count"))
}

impl ClusterSteps {
    /// Must be called on the node's runtime.
    pub fn new(node: Weak<ClusterNode>) -> ClusterSteps {
        ClusterSteps { node, runtime: tokio::runtime::Handle::current(), tasks: Default::default(), history: Default::default() }
    }

    /// The progress of task `key`, started with `start` on the first call.
    fn task<F, Fut>(&self, key: String, start: F) -> StepProgress
    where
        F: FnOnce(Arc<ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        let mut tasks = self.tasks.lock();
        tasks.retain(|_, t| !matches!(t, Task::Done(at, _) if at.elapsed() > KEEP_DONE));
        match tasks.get(&key) {
            Some(Task::Running) => return StepProgress::Pending(Some(json!({"message": "waiting for the cluster to complete the step"}))),
            Some(Task::Done(_, Ok(()))) => return StepProgress::Done,
            Some(Task::Done(_, Err(e))) => {
                let e = e.clone();
                // A retry of the step starts the work again.
                tasks.remove(&key);
                return StepProgress::Failed(e);
            }
            None => {}
        }
        let Some(node) = self.node.upgrade() else { return StepProgress::Pending(None) };
        tasks.insert(key.clone(), Task::Running);
        let fut = start(node);
        let registry = self.tasks.clone();
        self.runtime.spawn(async move {
            let result = fut.await;
            registry.lock().insert(key, Task::Done(Instant::now(), result));
        });
        StepProgress::Pending(Some(json!({"message": "waiting for the cluster to complete the step"})))
    }

    /// `index`'s cluster uuid (task keys: a re-created index of the same
    /// name never reads an old result).
    fn uuid(&self, index: &str) -> String {
        self.node.upgrade().and_then(|n| n.state().metadata.indices.get(index).map(|im| im.uuid.clone())).unwrap_or_default()
    }

    fn running(&self, key: &str) -> bool {
        matches!(self.tasks.lock().get(key), Some(Task::Running))
    }
}

impl ClusterLifecycleSteps for ClusterSteps {
    fn single_node(&self, index: &str) -> Option<String> {
        let node = self.node.upgrade()?;
        let st = node.state();
        // The node holding shard 0's started primary (ES picks a node that
        // can hold a copy of every shard; this one already holds one).
        st.routing_table.shard_table(index, 0)?.active_primary()?.node_id.clone()
    }

    fn allocated_on(&self, index: &str, node_id: &str) -> StepProgress {
        let Some(node) = self.node.upgrade() else { return StepProgress::Pending(None) };
        let st = node.state();
        let n = crate::shard_engines::shards_of(&st.metadata, index).max(1);
        let missing = (0..n).filter(|k| {
            !st.routing_table.shard_table(index, *k).is_some_and(|t| t.copies.iter().any(|c| c.node_id.as_deref() == Some(node_id) && c.state == ShardState::Started))
        }).count();
        if missing == 0 {
            StepProgress::Done
        } else {
            StepProgress::Pending(Some(json!({"message": format!("Waiting for node [{node_id}] to contain [{missing}] shards, found [{}], remaining [{missing}]", n as usize - missing),
                "node_id": node_id, "shards_left_to_allocate": missing, "expected_shards": n})))
        }
    }

    fn primaries_active(&self, index: &str) -> StepProgress {
        let Some(node) = self.node.upgrade() else { return StepProgress::Pending(None) };
        let st = node.state();
        if !st.metadata.indices.contains_key(index) {
            return StepProgress::Pending(Some(json!({"message": format!("waiting for index [{index}] to exist")})));
        }
        let n = crate::shard_engines::shards_of(&st.metadata, index).max(1);
        let active = (0..n).filter(|k| st.routing_table.shard_table(index, *k).and_then(|t| t.active_primary()).is_some()).count();
        if active == n as usize {
            StepProgress::Done
        } else {
            StepProgress::Pending(Some(json!({"message": "Waiting for all shard copies to be active", "shards_left_to_allocate": n as usize - active, "all_shards_active": false})))
        }
    }

    fn forcemerge(&self, index: &str, max_num_segments: usize) -> StepProgress {
        let index = index.to_string();
        self.task(format!("forcemerge\u{0}{index}\u{0}{}\u{0}{max_num_segments}", self.uuid(&index)), move |node| async move {
            let prepared_request = request("POST", format_args!("/{}/_forcemerge?max_num_segments={max_num_segments}", encode(&index)), None);
            let r = dispatch(&node, prepared_request).await;
            if r.is_success() {
                Ok(())
            } else {
                Err(format!("force merge of [{index}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
            }
        })
    }

    fn shrink(&self, source: &str, target: &str, body: &Value) -> StepProgress {
        let key = format!("shrink\u{0}{source}\u{0}{}\u{0}{target}", self.uuid(source));
        let body = body.clone();
        let (source, target) = (source.to_string(), target.to_string());
        self.task(key, move |node| async move {
            // A target left by an earlier attempt (a previous master, a
            // failed copy) is kept only if it holds every document; else it
            // is deleted and the shrink redone, as ES's shrink step does.
            if node.state().metadata.indices.contains_key(&target) {
                if let (Ok(s), Ok(t)) = (count(&node, &source).await, count(&node, &target).await) {
                    if t >= s {
                        return Ok(());
                    }
                }
                let prepared_request = request("DELETE", format_args!("/{}", encode(&target)), None);
                let r = dispatch(&node, prepared_request).await;
                if !r.is_success() && r.status != 404 {
                    return Err(format!("deleting the incomplete shrink target [{target}] answered {}", r.status));
                }
            }
            // The cluster resize: creates the target through the master and
            // copies the documents before it answers.
            let prepared_request = request("POST", format_args!("/{}/_shrink/{}", encode(&source), encode(&target)), Some(body));
            let r = dispatch(&node, prepared_request).await;
            // A target that already exists (a retried step) is checked for
            // its documents before the source goes.
            if r.is_success() || node.state().metadata.indices.contains_key(&target) {
                Ok(())
            } else {
                Err(format!("shrink of [{source}] into [{target}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
            }
        })
    }

    fn holds_all_docs(&self, source: &str, target: &str) -> StepProgress {
        if self.running(&format!("shrink\u{0}{source}\u{0}{}\u{0}{target}", self.uuid(source))) {
            return StepProgress::Pending(Some(json!({"message": format!("waiting for the documents of [{source}] to be copied to [{target}]")})));
        }
        let (source, target) = (source.to_string(), target.to_string());
        let key = format!("docs\u{0}{source}\u{0}{}\u{0}{target}\u{0}{}", self.uuid(&source), self.uuid(&target));
        self.task(key, move |node| async move {
            let (s, t) = (count(&node, &source).await?, count(&node, &target).await?);
            if t >= s {
                Ok(())
            } else {
                Err(format!("shrink target [{target}] holds {t} documents, its source [{source}] {s}: the source is kept"))
            }
        })
    }

    fn write_history(&self, items: Vec<Value>) {
        // One writer at a time; documents arriving meanwhile join its next
        // bulk (a pass every second must not pile up writes).
        {
            let mut pending = self.history.lock();
            if let Some(queue) = pending.as_mut() {
                queue.extend(items);
                let over = queue.len().saturating_sub(HISTORY_BACKLOG);
                queue.drain(..over);
                return;
            }
            *pending = Some(Vec::new());
        }
        let Some(node) = self.node.upgrade() else {
            *self.history.lock() = None;
            return;
        };
        let history = self.history.clone();
        self.runtime.spawn(async move {
            let mut items = items;
            loop {
                write_history_bulk(&node, items).await;
                let mut pending = history.lock();
                match pending.as_mut().map(std::mem::take) {
                    Some(next) if !next.is_empty() => items = next,
                    _ => {
                        *pending = None;
                        return;
                    }
                }
            }
        });
    }
}

async fn write_history_bulk(node: &Arc<ClusterNode>, items: Vec<Value>) {
    let ds = elasticsearch_engine::ilm::runner::HISTORY_DATA_STREAM;
    if !node.state().metadata.index_templates.contains_key(ds) {
        let prepared_request = request("PUT", format_args!("/_index_template/{ds}"), Some(elasticsearch_engine::ilm::runner::history_template()));
        let r = dispatch(node, prepared_request).await;
        if !r.is_success() {
            tracing::warn!(status = r.status, "failed to install the ILM history template");
            return;
        }
    }
    let req = request("POST", format_args!("/_bulk"), None).and_then(|req| {
        let mut req=req.try_with_encoded_body(|out| {
            #[derive(serde::Serialize)] struct Meta<'a>{_index:&'a str}
            #[derive(serde::Serialize)] struct Action<'a>{create:Meta<'a>}
            for item in &items {
                serde_json::to_writer(&mut *out,&Action{create:Meta{_index:ds}})?;out.write_all(b"\n")?;
                serde_json::to_writer(&mut *out,item)?;out.write_all(b"\n")?;
            } Ok(())
        })?;
        req.replace_header("content-type","application/x-ndjson")?;Ok(req)
    });
    let r = dispatch(node, req).await;
    if !r.is_success() {
        tracing::warn!(status = r.status, "failed to write ILM history");
    }
}
