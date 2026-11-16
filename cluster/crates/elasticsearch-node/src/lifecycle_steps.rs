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

/// ES `DLMConvertToFrozen.SNAPSHOT_TIMEOUT`: a snapshot under a
/// conversion's name running longer than this is deleted and taken again.
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(12 * 3600);

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

/// What an existing index under a shrink target name is to a source.
enum ShrinkTarget {
    /// The completed resize of this source incarnation.
    Complete,
    /// This source's resize target (that uuid), not completed.
    Incomplete(String),
    /// Not created from this source incarnation: never adopted or deleted.
    Foreign(String),
}

/// The index named `target` judged against source incarnation
/// `source_uuid` (`None`: no such index). A resize target records its
/// source (`index.resize.source.uuid`); it is complete once its resize
/// progress entry is gone (the copy was verified) and it carries no read
/// block (a failed or orphaned resize is left read blocked).
fn target_of(node: &Arc<ClusterNode>, source_uuid: &str, target: &str) -> Option<ShrinkTarget> {
    let st = node.state();
    let im = st.metadata.indices.get(target)?;
    let from = im.settings.get("index.resize.source.uuid").and_then(Value::as_str).unwrap_or_default();
    if source_uuid.is_empty() || from != source_uuid {
        return Some(ShrinkTarget::Foreign(format!(
            "index [{target}] ([{}]) exists but was not created by a resize of this source ([{source_uuid}]; its resize source is [{from}]): it is neither reused nor deleted",
            im.uuid
        )));
    }
    let in_progress = crate::snapshot_api::common::progress_entries(&st, crate::resize::PROGRESS_KIND).iter().any(|e| e["index_uuid"].as_str() == Some(im.uuid.as_str()));
    let read_blocked = matches!(im.settings.get("index.blocks.read"), Some(v) if v == &json!(true) || v == &json!("true"));
    Some(if in_progress || read_blocked { ShrinkTarget::Incomplete(im.uuid.clone()) } else { ShrinkTarget::Complete })
}

/// A shrink's mutations run only under the authority it started with: this
/// node is still the master of term `term`, and the source is still
/// incarnation `source_uuid`.
fn still_authorized(node: &Arc<ClusterNode>, term: u64, source: &str, source_uuid: &str) -> Result<(), String> {
    let st = node.state();
    if !node.is_master() || st.coordination.term != term {
        return Err(format!("the master that started the shrink of [{source}] (term [{term}]) is no longer master: nothing is changed"));
    }
    if !st.metadata.indices.get(source).is_some_and(|m| m.uuid == source_uuid) {
        return Err(format!("shrink source [{source}] is no longer incarnation [{source_uuid}]: nothing is changed"));
    }
    Ok(())
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

    /// A resize of `source` into `target` (`kind`: `shrink` or `clone`,
    /// `body`: the resize request body) through the cluster's resize path.
    fn resize(&self, kind: &'static str, source: &str, target: &str, body: &Value) -> StepProgress {
        let source_uuid = self.uuid(source);
        let key = format!("{kind}\u{0}{source}\u{0}{source_uuid}\u{0}{target}");
        let body = body.clone();
        // The authority the step runs under: this master's term. Work left
        // running past a master change never mutates the target.
        let term = self.node.upgrade().map(|n| n.state().coordination.term).unwrap_or(0);
        let (source, target) = (source.to_string(), target.to_string());
        self.task(key, move |node| async move {
            if source_uuid.is_empty() {
                return Err(format!("{kind} source [{source}] no longer exists"));
            }
            // A target left by an earlier attempt is reused only if this
            // source's resize created it and completed it; one this source's
            // resize created but did not complete is deleted (that
            // incarnation only) and the shrink redone, as ES's
            // `CleanupShrinkIndexStep` does. Any other index under the name
            // is neither adopted nor deleted.
            if let Some(found) = target_of(&node, &source_uuid, &target) {
                match found {
                    ShrinkTarget::Complete => return Ok(()),
                    ShrinkTarget::Foreign(why) => return Err(why),
                    ShrinkTarget::Incomplete(uuid) => {
                        still_authorized(&node, term, &source, &source_uuid)?;
                        // Conditioned on that incarnation: if the name was
                        // re-created meanwhile, the delete is refused.
                        let mut req = request("DELETE", format_args!("/{}", encode(&target)), None).map_err(|e| e.reason)?;
                        req.push_header(elasticsearch_http::IF_INDEX_UUID_HEADER, &uuid).map_err(|e| e.reason)?;
                        let r = dispatch(&node, Ok(req)).await;
                        if !r.is_success() && r.status != 404 {
                            return Err(format!("deleting the incomplete {kind} target [{target}] ([{uuid}]) answered {}: {}", r.status, String::from_utf8_lossy(&r.body)));
                        }
                    }
                }
            }
            still_authorized(&node, term, &source, &source_uuid)?;
            // The cluster resize: creates the target through the master and
            // copies (and verifies) the documents before it answers.
            let prepared_request = request("POST", format_args!("/{}/_{kind}/{}", encode(&source), encode(&target)), Some(body));
            let r = dispatch(&node, prepared_request).await;
            if r.is_success() {
                Ok(())
            } else {
                Err(format!("{kind} of [{source}] into [{target}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
            }
        })
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
        self.resize("shrink", source, target, body)
    }

    fn holds_all_docs(&self, source: &str, target: &str) -> StepProgress {
        let source_uuid = self.uuid(source);
        if self.running(&format!("shrink\u{0}{source}\u{0}{source_uuid}\u{0}{target}")) {
            return StepProgress::Pending(Some(json!({"message": format!("waiting for the documents of [{source}] to be copied to [{target}]")})));
        }
        // Before the source goes: the target must be the completed resize
        // of exactly this source incarnation (a count is no proof of the
        // same documents).
        let Some(node) = self.node.upgrade() else { return StepProgress::Pending(None) };
        match target_of(&node, &source_uuid, target) {
            Some(ShrinkTarget::Complete) => {}
            Some(ShrinkTarget::Foreign(why)) => return StepProgress::Failed(why),
            Some(ShrinkTarget::Incomplete(uuid)) => {
                return StepProgress::Failed(format!("shrink target [{target}] ([{uuid}]) of [{source}] is not a completed resize: the source is kept"));
            }
            None => return StepProgress::Failed(format!("shrink target [{target}] of [{source}] does not exist: the source is kept")),
        }
        let (source, target) = (source.to_string(), target.to_string());
        let key = format!("docs\u{0}{source}\u{0}{source_uuid}\u{0}{target}\u{0}{}", self.uuid(&target));
        self.task(key, move |node| async move {
            let (s, t) = (count(&node, &source).await?, count(&node, &target).await?);
            if t >= s {
                Ok(())
            } else {
                Err(format!("shrink target [{target}] holds {t} documents, its source [{source}] {s}: the source is kept"))
            }
        })
    }

    fn downsample(&self, source: &str, target: &str, body: &Value) -> StepProgress {
        // Keyed by the target incarnation too: a cached success never
        // stands for a target replaced under the name since.
        let key = format!("downsample\u{0}{source}\u{0}{}\u{0}{target}\u{0}{}", self.uuid(source), self.uuid(target));
        let body = body.clone();
        let (source, target) = (source.to_string(), target.to_string());
        self.task(key, move |node| async move {
            // `_downsample` acknowledges a successful target again and
            // rebuilds one an interrupted run left `started`.
            let prepared_request = request("POST", format_args!("/{}/_downsample/{}", encode(&source), encode(&target)), Some(body));
            let r = dispatch(&node, prepared_request).await;
            if r.is_success() {
                Ok(())
            } else {
                Err(format!("downsample of [{source}] into [{target}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
            }
        })
    }

    fn unfollow_step(&self, index: &str, step: &str) -> StepProgress {
        // A not-yet-caught-up check comes back as a failure carrying this
        // prefix and the step info: the step waits and checks again.
        const BEHIND: &str = "\u{0}behind\u{0}";
        let key = format!("unfollow\u{0}{step}\u{0}{index}\u{0}{}", self.uuid(index));
        let (index, step) = (index.to_string(), step.to_string());
        let answer = |r: &RestResp, what: &str| -> Result<(), String> {
            if r.is_success() {
                return Ok(());
            }
            let reason = r.body_json().and_then(|v| v["error"]["reason"].as_str().map(str::to_string)).unwrap_or_default();
            if elasticsearch_engine::ccr::already_unfollowed(&reason) {
                return Ok(());
            }
            Err(format!("{what} answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
        };
        let progress = self.task(key, move |node| async move {
            match step.as_str() {
                // ES `WaitForFollowShardTasksStep`: no follow task left
                // (paused) counts as in sync.
                "wait-for-follow-shard-tasks" => {
                    let prepared_request = request("GET", format_args!("/{}/_ccr/stats", encode(&index)), None);
                    let r = dispatch(&node, prepared_request).await;
                    if !r.is_success() {
                        return if r.status == 404 { Ok(()) } else { answer(&r, "follow stats") };
                    }
                    match r.body_json().as_ref().and_then(elasticsearch_engine::ccr::follow_tasks_in_sync) {
                        None => Ok(()),
                        Some(info) => Err(format!("{BEHIND}{info}")),
                    }
                }
                "pause-follower-index" => {
                    let prepared_request = request("POST", format_args!("/{}/_ccr/pause_follow", encode(&index)), None);
                    let r = dispatch(&node, prepared_request).await;
                    answer(&r, "pause follow")
                }
                "close-follower-index" => {
                    let closed = node.state().metadata.indices.get(&index).is_some_and(|m| m.state == elasticsearch_cluster::state::IndexState::Close);
                    if closed {
                        return Ok(());
                    }
                    let prepared_request = request("POST", format_args!("/{}/_close", encode(&index)), None);
                    let r = dispatch(&node, prepared_request).await;
                    answer(&r, "close")
                }
                "unfollow-follower-index" => {
                    let prepared_request = request("POST", format_args!("/{}/_ccr/unfollow", encode(&index)), None);
                    let r = dispatch(&node, prepared_request).await;
                    answer(&r, "unfollow")
                }
                "open-follower-index" => {
                    let prepared_request = request("POST", format_args!("/{}/_open?wait_for_active_shards=0", encode(&index)), None);
                    let r = dispatch(&node, prepared_request).await;
                    answer(&r, "open")
                }
                other => Err(format!("unknown unfollow step [{other}]")),
            }
        });
        match progress {
            StepProgress::Failed(e) if e.starts_with(BEHIND) => StepProgress::Pending(serde_json::from_str(&e[BEHIND.len()..]).ok()),
            other => other,
        }
    }

    fn snapshot_includes(&self, policy: &str, repository: &str, snapshot: &str, index: &str) -> StepProgress {
        // Keyed by the index's uuid: a re-created index is checked afresh.
        // A failed check is forgotten on reading (`task`), so the next pass
        // looks again (a later SLM run may include the index).
        let key = format!("snapshot-includes\u{0}{repository}\u{0}{snapshot}\u{0}{index}\u{0}{}", self.uuid(index));
        let (policy, repository, snapshot, index) = (policy.to_string(), repository.to_string(), snapshot.to_string(), index.to_string());
        self.task(key, move |node| async move {
            let prepared_request = request("GET", format_args!("/_snapshot/{}/{}?index_names=true", encode(&repository), encode(&snapshot)), None);
            let r = dispatch(&node, prepared_request).await;
            let snapshots = if r.is_success() {
                r.body_json().and_then(|v| v["snapshots"].as_array().cloned()).unwrap_or_default()
            } else if r.status == 404 {
                Vec::new()
            } else {
                return Err(format!("reading snapshot [{snapshot}] of repository [{repository}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)));
            };
            elasticsearch_engine::ilm::verify_snapshot_listing(&snapshots, &policy, &repository, &snapshot, &index)
        })
    }

    fn clone_index(&self, source: &str, target: &str, body: &Value) -> StepProgress {
        let Some(node) = self.node.upgrade() else { return StepProgress::Pending(None) };
        let source_uuid = self.uuid(source);
        let key = format!("clone\u{0}{source}\u{0}{source_uuid}\u{0}{target}");
        if !self.running(&key) {
            match target_of(&node, &source_uuid, target) {
                Some(ShrinkTarget::Complete) => return StepProgress::Done,
                Some(ShrinkTarget::Foreign(why)) => return StepProgress::Failed(why),
                // Being filled (or left incomplete: the resize below
                // deletes that incarnation and clones again).
                Some(ShrinkTarget::Incomplete(_)) | None => {}
            }
            // The clone needs the source's write block, which this
            // lifecycle pass may have just added: wait until it is published.
            let st = node.state();
            let blocked = st.metadata.indices.get(source).is_some_and(|im| matches!(im.settings.get("index.blocks.write"), Some(v) if v == &json!(true) || v == &json!("true")));
            if !blocked {
                return StepProgress::Pending(Some(json!({"message": format!("waiting for the write block of [{source}] to be published")})));
            }
        }
        match self.resize("clone", source, target, body) {
            // The resize answered: done once the target is the completed
            // clone (read again on the next pass).
            StepProgress::Done => match target_of(&node, &source_uuid, target) {
                Some(ShrinkTarget::Complete) => StepProgress::Done,
                Some(ShrinkTarget::Foreign(why)) => StepProgress::Failed(why),
                _ => StepProgress::Pending(Some(json!({"message": format!("waiting for the clone [{target}] of [{source}] to complete")}))),
            },
            other => other,
        }
    }

    fn snapshot_index(&self, repository: &str, snapshot: &str, index: &str, body: &Value) -> StepProgress {
        // Keyed by the index's uuid: a re-created index is snapshotted afresh.
        let key = format!("snapshot-index\u{0}{repository}\u{0}{snapshot}\u{0}{index}\u{0}{}", self.uuid(index));
        let (repository, snapshot, index, body) = (repository.to_string(), snapshot.to_string(), index.to_string(), body.clone());
        self.task(key, move |node| async move {
            // ES `maybeTakeSnapshot`: a snapshot under the name that is
            // running is waited for, one that completed with the index is
            // kept, one that failed or is partial is deleted and retaken.
            let deadline = Instant::now() + SNAPSHOT_TIMEOUT;
            loop {
                let prepared_request = request("GET", format_args!("/_snapshot/{}/{}?ignore_unavailable=true&index_names=true", encode(&repository), encode(&snapshot)), None);
                let r = dispatch(&node, prepared_request).await;
                if r.status == 404 && r.body_json().is_some_and(|v| v["error"]["type"] == "snapshot_missing_exception") {
                    break;
                }
                if !r.is_success() {
                    return Err(format!("DLM failed to start snapshot [{snapshot}] for index [{index}]: reading the repository answered {}: {}", r.status, String::from_utf8_lossy(&r.body)));
                }
                let found = r.body_json().and_then(|v| v["snapshots"].as_array().and_then(|a| a.first().cloned()));
                let Some(info) = found else { break };
                if elasticsearch_engine::ilm::snapshot_steps::snapshot_complete(&info, &index) {
                    return Ok(());
                }
                if elasticsearch_engine::ilm::snapshot_steps::snapshot_in_progress(&info) && Instant::now() < deadline {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
                let prepared_request = request("DELETE", format_args!("/_snapshot/{}/{}", encode(&repository), encode(&snapshot)), None);
                let r = dispatch(&node, prepared_request).await;
                if !r.is_success() && r.status != 404 {
                    return Err(format!("DLM failed to delete incomplete snapshot [{snapshot}] for index [{index}]: {}: {}", r.status, String::from_utf8_lossy(&r.body)));
                }
                break;
            }
            let prepared_request = request("PUT", format_args!("/_snapshot/{}/{}?wait_for_completion=true", encode(&repository), encode(&snapshot)), Some(body));
            let r = dispatch(&node, prepared_request).await;
            if !r.is_success() {
                return Err(format!("DLM failed to start snapshot [{snapshot}] for index [{index}]: {}: {}", r.status, String::from_utf8_lossy(&r.body)));
            }
            let info = r.body_json().map(|v| v["snapshot"].clone());
            elasticsearch_engine::ilm::snapshot_steps::check_snapshot_info(&index, &snapshot, info.as_ref())
        })
    }

    fn mount_snapshot(&self, repository: &str, snapshot: &str, body: &Value, partial: bool) -> StepProgress {
        let mounted = body["renamed_index"].as_str().unwrap_or_default().to_string();
        let key = format!("mount\u{0}{repository}\u{0}{snapshot}\u{0}{mounted}");
        let (repository, snapshot, body) = (repository.to_string(), snapshot.to_string(), body.clone());
        let storage = if partial { "shared_cache" } else { "full_copy" };
        self.task(key, move |node| async move {
            if node.state().metadata.indices.contains_key(&mounted) {
                return Ok(());
            }
            let prepared_request = request("POST", format_args!("/_snapshot/{}/{}/_mount?wait_for_completion=true&storage={storage}", encode(&repository), encode(&snapshot)), Some(body));
            let r = dispatch(&node, prepared_request).await;
            if !r.is_success() {
                return Err(format!("DLM failed while mounting snapshot [{snapshot}]: {}: {}", r.status, String::from_utf8_lossy(&r.body)));
            }
            elasticsearch_engine::ilm::snapshot_steps::check_mount_response(&snapshot, &r.body_json().unwrap_or(Value::Null))
        })
    }

    fn delete_snapshot(&self, repository: &str, snapshot: &str) -> StepProgress {
        let key = format!("delete-snapshot\u{0}{repository}\u{0}{snapshot}");
        let (repository, snapshot) = (repository.to_string(), snapshot.to_string());
        self.task(key, move |node| async move {
            let prepared_request = request("DELETE", format_args!("/_snapshot/{}/{}", encode(&repository), encode(&snapshot)), None);
            let r = dispatch(&node, prepared_request).await;
            if r.is_success() || r.status == 404 {
                Ok(())
            } else {
                Err(format!("deleting snapshot [{snapshot}] of repository [{repository}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
            }
        })
    }

    fn delete_snapshots_matching(&self, repository: &str, pattern: &str, select: elasticsearch_engine::ilm::snapshot_steps::SnapshotSelect) -> StepProgress {
        // One scan at a time per repository and pattern; a finished scan is
        // forgotten on reading so the next cleanup scans again.
        let key = format!("delete-snapshots-matching\u{0}{repository}\u{0}{pattern}");
        let (repository, pattern) = (repository.to_string(), pattern.to_string());
        let progress = self.task(key.clone(), move |node| async move {
            let prepared_request = request("GET", format_args!("/_snapshot/{}/{}?ignore_unavailable=true&index_names=true", encode(&repository), encode(&pattern)), None);
            let r = dispatch(&node, prepared_request).await;
            if r.status == 404 {
                return Ok(());
            }
            if !r.is_success() {
                return Err(format!("listing snapshots [{pattern}] of repository [{repository}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)));
            }
            let infos = r.body_json().and_then(|v| v["snapshots"].as_array().cloned()).unwrap_or_default();
            let names: Vec<String> = infos.iter().filter(|i| select(i)).filter_map(|i| i["snapshot"].as_str().map(str::to_string)).collect();
            if names.is_empty() {
                return Ok(());
            }
            let joined = names.join(",");
            let prepared_request = request("DELETE", format_args!("/_snapshot/{}/{}", encode(&repository), encode(&joined)), None);
            let r = dispatch(&node, prepared_request).await;
            if r.is_success() || r.status == 404 {
                Ok(())
            } else {
                Err(format!("deleting snapshots [{joined}] of repository [{repository}] answered {}: {}", r.status, String::from_utf8_lossy(&r.body)))
            }
        });
        if matches!(progress, StepProgress::Done) {
            self.tasks.lock().remove(&key);
        }
        progress
    }

    fn allocation_routed(&self, index: &str, settings: &std::collections::BTreeMap<String, String>, migrate: bool) -> StepProgress {
        use elasticsearch_cluster::allocation::LifecycleRouting;
        let Some(node) = self.node.upgrade() else { return StepProgress::Pending(None) };
        let st = node.state();
        let Some(meta) = st.metadata.indices.get(index) else {
            return StepProgress::Pending(Some(json!({"message": format!("waiting for index [{index}] to exist")})));
        };
        // The routing is judged against the constraints this pass set.
        if let Some(key) = settings.keys().find(|k| meta.settings.get(k.as_str()).and_then(elasticsearch_engine::ilm::setting_string).as_deref() != Some(settings[*k].as_str())) {
            return StepProgress::Pending(Some(json!({"message": format!("waiting for setting [{key}] of index [{index}] to be published")})));
        }
        let replicas = meta.number_of_replicas;
        match elasticsearch_cluster::allocation::lifecycle_routing(&st, meta, migrate) {
            LifecycleRouting::Routed => StepProgress::Done,
            LifecycleRouting::CopiesInactive => StepProgress::Pending(Some(json!({
                "message": "Waiting for all shard copies to be active", "shards_left_to_allocate": -1, "all_shards_active": false, "number_of_replicas": replicas}))),
            LifecycleRouting::Pending { shards, tier } => {
                let message = if !migrate {
                    format!("Waiting for [{shards}] shards to be allocated to nodes matching the given filters")
                } else {
                    let pref = elasticsearch_cluster::allocation::tier_preference(meta);
                    match tier {
                        Some(tier) => format!("[{index}] lifecycle action [migrate] waiting for [{shards}] shards to be moved to the [{tier}] tier (tier migration preference configuration is [{}])", pref.join(", ")),
                        None => format!("index [{index}] has a preference for tiers [{}], but no nodes for any of those tiers are available in the cluster", pref.join(", ")),
                    }
                };
                StepProgress::Pending(Some(json!({"message": message, "shards_left_to_allocate": shards, "all_shards_active": true, "number_of_replicas": replicas})))
            }
        }
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
