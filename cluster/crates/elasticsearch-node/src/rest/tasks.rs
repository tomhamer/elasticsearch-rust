//! Cluster-wide task management: `_tasks` (list / get / cancel),
//! `_{reindex,update_by_query,delete_by_query}/{id}/_rethrottle` and
//! `_cat/tasks`, answered across every node as ES's `TransportListTasksAction`
//! / `TransportCancelTasksAction` do.
//!
//! Every node runs several engines (engine 0 plus one per shard number `k`,
//! see `shard_engines`), each with its own task manager whose ids carry the
//! engine's fixed node id. The REST layer rewrites them to ES-style ids that
//! name the owning cluster node: `<node id>:<k << 40 | n>`, so `_tasks/{id}`
//! can be routed to the owner and from there to the engine that runs it.
//!
//! Parent/child: a cluster-run task (`front::bg_task`) that fans its work out
//! to shard primaries sends [`PARENT_HEADER`] with each sub-request; the
//! receiving node links the engine task the sub-request starts to that parent
//! (so it lists with `parent_task_id` and is cancelled with it). Cancelling a
//! task by id also broadcasts a ban on its id, as ES's ban-parent does: every
//! node cancels the children it runs for that parent, and children that start
//! within the ban's lifetime are cancelled on registration.

use crate::front::{local_engine, RestReq, RestResp, SharedText, SharedHeaders};
use crate::shared_sequence::SharedSequence;
use crate::node::ClusterNode;
use elasticsearch_cluster::state::{DiscoveryNode,ClusterState};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node as Engine, ES_VERSION, NODE_ID};
use elasticsearch_engine::tasks::{parse_task_id, Task};
use elasticsearch_transport::RemoteError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

/// Header carrying the cluster parent task id on fanned-out sub-requests
/// (internal: stripped from client requests).
pub const PARENT_HEADER: &str = "x-es-cluster-parent-task";

const ACTION_NODE: &str = "internal:cluster/tasks/node";
const LIST_ACTION: &str = "cluster:monitor/tasks/lists";
const BULK_ACTIONS: &[&str] = &["indices:data/write/update/byquery", "indices:data/write/delete/byquery", "indices:data/write/reindex"];
const BAN_TTL: Duration = Duration::from_secs(60);
const ID_BITS: u32 = 40;

/// Engine task (k, n) linked to its cluster parent task id.
#[derive(Default)]
struct ChildRegistry { values:HashMap<(u32,u64),SharedText>,memory:Option<elasticsearch_engine::memory::Reservation> }
impl ChildRegistry {
 fn insert(&mut self,key:(u32,u64),parent:&str)->Result<(),EsError>{
  let parent=elasticsearch_engine::memory::native_control(||SharedText::try_new(parent))?;
  if self.memory.is_none(){self.memory=Some(elasticsearch_engine::memory::breakers().reserve_native(0,"task child registry")?.into_result_owner());}
  if !self.values.contains_key(&key)&&self.values.len()==self.values.capacity(){let n=self.values.len().checked_add(1).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task child capacity overflow"))?;let bytes=n.checked_mul(4*(std::mem::size_of::<((u32,u64),SharedText)>()+1)).and_then(|n|n.checked_add(64)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task child bytes overflow"))?;self.memory.as_mut().unwrap().grow(bytes as u64,"task child table replacement")?;self.values.try_reserve(1).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;}
  self.values.insert(key,parent);Ok(())
 }
}
static CHILDREN:LazyLock<Mutex<ChildRegistry>>=LazyLock::new(Default::default);
/// Cancelled parents: children registering for them are cancelled at once.
#[derive(Default)]
struct BanCache{values:HashMap<SharedText,Instant>,memory:Option<elasticsearch_engine::memory::Reservation>}
impl BanCache{
 fn insert(&mut self,id:&str)->Result<(),EsError>{
  if let Some(time)=self.values.get_mut(id){*time=Instant::now();return Ok(())}
  let id=elasticsearch_engine::memory::native_control(||SharedText::try_new(id))?;
  if self.memory.is_none(){self.memory=Some(elasticsearch_engine::memory::breakers().reserve_native(0,"task ban cache")?.into_result_owner());}
  if self.values.len()==self.values.capacity(){
   let wanted=self.values.len().checked_add(1).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task ban capacity overflow"))?;
   let bytes=wanted.checked_mul(4*(std::mem::size_of::<(SharedText,Instant)>()+1)).and_then(|n|n.checked_add(64)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task ban table overflow"))?;
   self.memory.as_mut().unwrap().grow(bytes as u64,"task ban table replacement")?;
   self.values.try_reserve(1).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
  }
  self.values.insert(id,Instant::now());ANY_BANS.store(true,std::sync::atomic::Ordering::Relaxed);Ok(())
 }
}
static BANS: LazyLock<Mutex<BanCache>> = LazyLock::new(Default::default);
/// Whether [`BANS`] may hold a ban (set under its lock on insert, cleared
/// under it once empty): a search shard request checks for its parent's
/// ban without taking the lock while nothing is banned.
static ANY_BANS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn encode_id(k: u32, n: u64) -> u64 {
    ((k as u64) << ID_BITS) | (n & ((1u64 << ID_BITS) - 1))
}

fn decode_id(id: u64) -> (u32, u64) {
    ((id >> ID_BITS) as u32, id & ((1u64 << ID_BITS) - 1))
}

fn linked_parent(k: u32, n: u64) -> Option<SharedText> {
    CHILDREN.lock().values.get(&(k, n)).cloned()
}

pub(crate) fn banned(parent: &str) -> bool {
    let mut b = BANS.lock();
    b.values.retain(|_, t| t.elapsed() < BAN_TTL);
    if b.values.is_empty() {
        ANY_BANS.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    b.values.contains_key(parent)
}

/// [`banned`], without the lock while no ban is held (every search shard
/// request asks).
pub(crate) fn maybe_banned(parent: &str) -> bool {
    ANY_BANS.load(std::sync::atomic::Ordering::Relaxed) && banned(parent)
}

/// Ban cluster task `parent` on `nodes` in the background (a coordinator that
/// stopped waiting for children it cannot cancel directly): each node cancels
/// the children it runs for it, and children check the ban as they proceed.
pub(crate) fn ban_on(node: &Arc<ClusterNode>, nodes: Vec<String>, parent: &str) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else { return };
    let request = (|| -> Result<NodeReq, EsError> {
        Ok(NodeReq { op: SharedText::try_new("ban")?, ids: SharedSequence::try_collect([SharedText::try_new(parent)?])?, ..Default::default() })
    })();
    let Ok(request) = request else { return };
    let node = node.clone();
    runtime.spawn(async move {
        if let Err(error) = fan_out(&node, nodes, &request).await {
            tracing::debug!(reason = %error.reason, "could not ban the children of a cancelled msearch");
        }
    });
}

/// An engine `TaskInfo` as the cluster shows it: this node's id, the
/// engine-qualified number, the cluster parent (`parent_task_id` stays
/// right before `headers`, where ES prints it).
fn rewrite_info(local: &str, k: u32, info: &Value, linked: Option<&str>) -> Value {
    let Some(o) = info.as_object() else { return info.clone() };
    let parent = linked.map(str::to_string).or_else(|| {
        o.get("parent_task_id").and_then(Value::as_str).map(|p| {
            match p.strip_prefix(NODE_ID).and_then(|r| r.strip_prefix(':')).and_then(|n| n.parse::<u64>().ok()) {
                Some(n) => format!("{local}:{}", encode_id(k, n)),
                None => p.to_string(),
            }
        })
    });
    let mut m = Map::new();
    let mut parent_done = false;
    for (key, v) in o {
        match key.as_str() {
            "node" => {
                m.insert("node".into(), json!(local));
            }
            "id" => {
                m.insert("id".into(), json!(encode_id(k, v.as_u64().unwrap_or(0))));
            }
            "parent_task_id" => {}
            "headers" => {
                if let Some(p) = &parent {
                    m.insert("parent_task_id".into(), json!(p));
                    parent_done = true;
                }
                m.insert("headers".into(), v.clone());
            }
            _ => {
                m.insert(key.clone(), v.clone());
            }
        }
    }
    if let (Some(p), false) = (&parent, parent_done) {
        m.insert("parent_task_id".into(), json!(p));
    }
    Value::Object(m)
}

fn task_error(status:u16,kind:&str,reason:std::fmt::Arguments<'_>)->EsError{
 match SharedText::try_format(reason){Ok(reason)=>EsError::try_owned(status,kind,&reason).unwrap_or_else(|e|e),Err(e)=>e}
}
fn task_key(info: &Value) -> String {
    format!("{}:{}", info["node"].as_str().unwrap_or(""), info["id"])
}

fn err_resp(e: &EsError) -> RestResp {
    RestResp::json(e.status, &e.to_json())
}

/// `?name` as a boolean (`true`, empty, `false`; anything else rejected).
fn flag(params: &HashMap<String, String>, name: &str) -> Result<bool, EsError> {
    match params.get(name).map(String::as_str) {
        None | Some("false") => Ok(false),
        Some("true") | Some("") => Ok(true),
        Some(v) => Err(task_error(400,"illegal_argument_exception",format_args!("Failed to parse value [{v}] as only [true] or [false] are allowed."))),
    }
}

fn timeout_param(params: &HashMap<String, String>) -> Result<Duration, EsError> {
    match params.get("timeout") {
        None => Ok(Duration::from_secs(30)),
        Some(t) => elasticsearch_engine::date::parse_time_value(t)
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .ok_or_else(|| task_error(400,"illegal_argument_exception",format_args!("failed to parse setting [timeout] with value [{t}] as a time value: unit is missing or unrecognized"))),
    }
}

/// `actions=` patterns; the REST list API trims and drops empty entries,
/// `_cat/tasks` uses them as given.
fn parse_actions(s: Option<&String>, trim: bool) -> Result<Option<SharedSequence<SharedText>>,EsError> {
    s.map(|a|SharedSequence::try_collect_results(a.split(',').map(|p| if trim { p.trim() } else { p }).filter(|p| !trim || !p.is_empty()).map(SharedText::try_new))).transpose()
}

fn action_matches(actions: &Option<SharedSequence<SharedText>>, action: &str) -> bool {
    match actions {
        None => true,
        Some(a) => a.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, action)),
    }
}

fn validate_group_by(g: &str) -> Result<(), EsError> {
    match g {
        "nodes" | "parents" | "none" => Ok(()),
        o => Err(task_error(400,"illegal_argument_exception",format_args!("[group_by] must be one of [nodes], [parents] or [none] but was [{o}]"))),
    }
}

fn opaque(req: &RestReq) -> Result<SharedHeaders,EsError> {
    match req.headers.iter().find(|(key,_)|key.eq_ignore_ascii_case("x-opaque-id")) {
        Some((_,value))=>SharedHeaders::try_pairs(&[("X-Opaque-Id",value)]),
        None=>Ok(SharedHeaders::default()),
    }
}

// ---------------------------------------------------------------------------
// Node-level operations (run on the node owning the tasks)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct NodeReq {
    op: SharedText,
    actions: Option<SharedSequence<SharedText>>,
    parent_task_id: Option<SharedText>,
    /// Encoded task number (`get`, `cancel_id`, `rethrottle`).
    task: Option<u64>,
    /// The full task id string the client used.
    task_str: SharedText,
    detailed: bool,
    wait: bool,
    timeout_ms: u64,
    rps: f64,
    /// The coordinating list task (parent of this node's `[n]` task).
    list_task: Option<SharedText>,
    cat: bool,
    opaque: SharedHeaders,
    /// Parent ids to ban (`ban`).
    ids: SharedSequence<SharedText>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NodeResp {
    status: u16,
    /// JSON (a task info array, a get body or an error body); empty for "task not found".
    body: SharedText,
}

impl NodeResp {
    fn ok(v: impl Serialize) -> Result<NodeResp,EsError> {
        Ok(NodeResp { status: 200, body: SharedText::try_json(&v)? })
    }
    fn err(e: &EsError) -> Result<NodeResp,EsError> {
        let response=RestResp::from_es_error(e);
        let text=std::str::from_utf8(&response.body).map_err(|_|EsError::internal("task error response is not UTF-8"))?;
        Ok(NodeResp { status: response.status, body: SharedText::try_new(text)? })
    }
    fn missing() -> Result<NodeResp,EsError> {
        Ok(NodeResp { status: 404, body: SharedText::default() })
    }
    fn json(&self) -> Result<crate::owned_json::OwnedJson,EsError> {
        if self.body.is_empty(){crate::owned_json::OwnedJson::parse("null")}else{crate::owned_json::OwnedJson::parse(&self.body)}
    }

}

pub fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_NODE, move |r: NodeReq| {
        let n = n.clone();
        async move { node_op(&n, r).await.map_err(|e|RemoteError::new(&e.kind,e.reason,e.status)) }
    });
}

struct ListTaskGuard{engine:Arc<Engine>,task:Arc<Task>}
impl std::ops::Deref for ListTaskGuard{type Target=Task;fn deref(&self)->&Task{&self.task}}
impl Drop for ListTaskGuard{fn drop(&mut self){self.engine.tasks.finish(&self.task,Ok(Value::Null));}}
/// Admitted coordination slots; elements drop before their container owner.
struct TaskVec<T>{values:Vec<T>,memory:elasticsearch_engine::memory::Reservation}
impl<T> TaskVec<T>{
 fn new()->Result<Self,EsError>{Ok(Self{values:Vec::new(),memory:elasticsearch_engine::memory::breakers().reserve_request(0,"task coordination slots")?.into_result_owner()})}
 fn push(&mut self,value:T)->Result<(),EsError>{if self.values.len()==self.values.capacity(){let n=self.values.capacity().checked_mul(2).and_then(|n|n.checked_add(1)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task slot overflow"))?;let bytes=n.checked_mul(std::mem::size_of::<T>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task slot bytes overflow"))?;self.memory.grow(bytes as u64,"task slot replacement")?;self.values.try_reserve_exact(n-self.values.len()).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;}self.values.push(value);Ok(())}
 fn clear(&mut self){self.values.clear()}
}
impl<T> std::ops::Deref for TaskVec<T>{type Target=[T];fn deref(&self)->&[T]{&self.values}}
impl<T:Serialize> Serialize for TaskVec<T>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{self.values.serialize(s)}}
struct TaskIntoIter<T>{values:std::vec::IntoIter<T>,_memory:elasticsearch_engine::memory::Reservation}
impl<T> Iterator for TaskIntoIter<T>{type Item=T;fn next(&mut self)->Option<T>{self.values.next()}fn size_hint(&self)->(usize,Option<usize>){self.values.size_hint()}}
impl<T> IntoIterator for TaskVec<T>{type Item=T;type IntoIter=TaskIntoIter<T>;fn into_iter(self)->Self::IntoIter{TaskIntoIter{values:self.values.into_iter(),_memory:self.memory}}}
struct StoredInfo<'a>{local:&'a str,k:u32,info:&'a Value,parent:Option<SharedText>}
impl Serialize for StoredInfo<'_>{
 fn serialize<S:serde::Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{
  use serde::ser::SerializeMap;
  let Some(info)=self.info.as_object()else{return self.info.serialize(serializer)};
  let mut m=serializer.serialize_map(None)?;let mut parent_done=false;
  for (key,value) in info {match key.as_str(){
   "node"=>m.serialize_entry(key,self.local)?,
   "id"=>m.serialize_entry(key,&encode_id(self.k,value.as_u64().unwrap_or(0)))?,
   "parent_task_id"=>{},
   "headers"=>{if let Some(parent)=&self.parent{m.serialize_entry("parent_task_id",parent)?;parent_done=true;}m.serialize_entry(key,value)?;},
   _=>m.serialize_entry(key,value)?,
  }}
  if !parent_done{if let Some(parent)=&self.parent{m.serialize_entry("parent_task_id",parent)?;}}
  m.end()
 }
}
fn projected_parent(local:&str,k:u32,parent:Option<&str>)->Result<Option<SharedText>,EsError>{parent.map(|p|match p.strip_prefix(NODE_ID).and_then(|r|r.strip_prefix(':')).and_then(|n|n.parse::<u64>().ok()){
 Some(n)=>SharedText::try_format(format_args!("{local}:{}",encode_id(k,n))),None=>SharedText::try_new(p)
}).transpose()}
struct StoredReply<'a>{info:StoredInfo<'a>,outcome:&'a Result<Value,Value>}
impl Serialize for StoredReply<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{use serde::ser::SerializeMap;let mut m=s.serialize_map(Some(3))?;m.serialize_entry("completed",&true)?;m.serialize_entry("task",&self.info)?;match self.outcome{Ok(v)=>m.serialize_entry("response",v)?,Err(v)=>m.serialize_entry("error",v)?,}m.end()}}
struct TaskInfo<'a>{task:&'a Task,local:&'a str,k:u32,parent:Option<SharedText>,detailed:bool}
impl Serialize for TaskInfo<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{self.task.serialize_info(self.detailed,self.local,encode_id(self.k,self.task.id),self.parent.as_deref(),s)}}
fn admitted_info(local:&str,k:u32,task:&Task,detailed:bool)->Result<crate::owned_json::OwnedJson,EsError>{
 let linked=linked_parent(k,task.id);
 let parent=match linked{Some(parent)=>Some(parent),None=>projected_parent(local,k,task.parent.as_deref())?};
 let text=SharedText::try_json(&TaskInfo{task,local,k,parent,detailed})?;crate::owned_json::OwnedJson::parse(&text)
}

/// Every running task on this node: (shard engine, engine, task, cluster info).
fn running(node: &ClusterNode, detailed: bool) -> Result<TaskVec<(u32, Arc<Engine>, Arc<Task>, crate::owned_json::OwnedJson)>,EsError> {
    let mut engines=TaskVec::new()?;
    node.try_visit_shard_engines(|k,e|engines.push((k,e.clone())))?;
    let mut tasks=TaskVec::new()?;
    for (k,e) in engines {e.tasks.try_visit_running(|t|tasks.push((k,e.clone(),t.clone())))?;}
    // Do not hold registry locks while reading linked-parent/status locks.
    let mut out=TaskVec::new()?;
    for (k,e,t) in tasks {let info=admitted_info(&node.local.id,k,&t,detailed)?;out.push((k,e,t,info))?;}
    Ok(out)
}

fn parent_of(info: &Value) -> Option<&str> {
    info.get("parent_task_id").and_then(Value::as_str)
}

async fn wait_all(waits: TaskVec<(Arc<Engine>, u64)>, timeout: Duration) -> bool {
    tokio::task::spawn_blocking(elasticsearch_engine::memory::inherit_request_policy(move || {
        let deadline = Instant::now() + timeout;
        waits.iter().all(|(e, n)| e.tasks.wait(*n, deadline.saturating_duration_since(Instant::now())))
    }))
    .await
    .unwrap_or(false)
}

/// Cancel the tasks running here on behalf of cluster task `parent`.
fn cancel_children(node: &ClusterNode, parent: &str) -> Result<TaskVec<(Arc<Engine>, u64)>,EsError> {
    let mut out = TaskVec::new()?;
    for (_, e, t, info) in running(node, false)? {
        if t.cancellable && parent_of(&info) == Some(parent) && e.tasks.cancel(t.id).is_ok() {
            out.push((e, t.id))?;
        }
    }
    Ok(out)
}

async fn node_op(node: &Arc<ClusterNode>, r: NodeReq) -> Result<NodeResp,EsError> {
    let local = node.local.id.as_str();
    let timeout = Duration::from_millis(r.timeout_ms);
    match r.op.as_str() {
        "list" => {
            let child = r.list_task.as_ref().map(|lt| {
                let (desc, cancellable) = if r.cat { (None, false) } else { (Some(""), true) };
                node.engine.tasks.try_register_borrowed("cluster:monitor/tasks/lists[n]", desc, cancellable, false, &r.opaque, Some(lt.as_str())).map(|task|ListTaskGuard{engine:node.engine.clone(),task})
            }).transpose()?;
            let mut infos = TaskVec::new()?;
            let mut waits = TaskVec::new()?;
            for (_, e, t, info) in running(node, r.detailed)? {
                if !action_matches(&r.actions, &t.action) {
                    continue;
                }
                if let Some(p) = &r.parent_task_id {
                    if parent_of(&info) != Some(p.as_str()) {
                        continue;
                    }
                }
                if r.wait && !t.action.starts_with(LIST_ACTION) {
                    waits.push((e.clone(), t.id))?;
                }
                infos.push(info)?;
            }
            if r.wait {
                wait_all(waits, timeout).await;
                infos.clear();
            }
            drop(child);
            NodeResp::ok(&infos)
        }
        "get" => {
            let (k, n) = decode_id(r.task.unwrap_or(0));
            let not_stored = || NodeResp::err(&task_error(404,"resource_not_found_exception",format_args!("task [{}] isn't running and hasn't stored its results", r.task_str)));
            let Some(e) = node.engine_for(k) else { return not_stored() };
            if r.wait && e.tasks.get_running(n).is_some() && !wait_all({let mut waits=TaskVec::new()?;waits.push((e.clone(),n))?;waits}, timeout).await {
                return NodeResp::err(&task_error(429,"timeout_exception",format_args!("Timed out waiting for completion of task [{}]",r.task_str)));
            }
            if let Some(t) = e.tasks.get_running(n) {
                let info=admitted_info(local,k,&t,true)?;
                #[derive(Serialize)] struct Reply<'a>{completed:bool,task:&'a crate::owned_json::OwnedJson}
                return NodeResp::ok(Reply{completed:false,task:&info});
            }
            e.tasks.with_result(n,|result|match result{
                Some(result)=>NodeResp::ok(StoredReply{info:StoredInfo{local,k,info:&result.task,parent:projected_parent(local,k,result.task.get("parent_task_id").and_then(Value::as_str))?},outcome:&result.outcome}),
                None=>not_stored(),
            })
        }

        "cancel_id" => {
            let (k, n) = decode_id(r.task.unwrap_or(0));
            let Some(e) = node.engine_for(k) else { return NodeResp::missing() };
            let t = match e.tasks.cancel(n) {
                Err(err) => {
                    // The engine names its own id; report the cluster one.
                    let reason=SharedText::try_format(format_args!("task [{}] doesn't support cancellation",r.task_str))?;
                    return NodeResp::err(&EsError::try_owned(err.status,&err.kind,&reason)?);
                }
                Ok(None) => return NodeResp::missing(),
                Ok(Some(t)) => t,
            };
            let mut waits = TaskVec::new()?;waits.push((e.clone(),n))?;
            for wait in cancel_children(node,&r.task_str)?{waits.push(wait)?;}
            let mut children=TaskVec::new()?;
            {let registry=CHILDREN.lock();for(key,parent)in &registry.values{if parent==&r.task_str{children.push(*key)?;}}}
            for (ck, cn) in children {
                if let Some(ce) = node.engine_for(ck) {
                    if ce.tasks.cancel(cn).is_ok() {
                        waits.push((ce, cn))?;
                    }
                }
            }
            let info = admitted_info(local,k,&t,false)?;
            if r.wait {
                wait_all(waits, timeout).await;
            }
            NodeResp::ok([&info])
        }
        "cancel" => {
            let mut infos = TaskVec::new()?;
            let mut waits = TaskVec::new()?;
            for (k, e, t, info) in running(node, false)? {
                if !t.cancellable || !action_matches(&r.actions, &t.action) {
                    continue;
                }
                if let Some(p) = &r.parent_task_id {
                    if parent_of(&info) != Some(p.as_str()) {
                        continue;
                    }
                }
                if e.tasks.cancel(t.id).is_ok() {
                    infos.push(admitted_info(local,k,&t,false)?)?;
                    waits.push((e, t.id))?;
                }
            }
            if r.wait {
                wait_all(waits, timeout).await;
            }
            NodeResp::ok(&infos)
        }
        "cancel_all" => {
            for (_, e, t, _) in running(node, false)? {
                if t.cancellable {
                    let _ = e.tasks.cancel(t.id);
                }
            }
            NodeResp::ok(json!([]))
        }
        "ban" => {
            for id in &r.ids {
                BANS.lock().insert(id)?;
                cancel_children(node, id)?;
            }
            NodeResp::ok(json!([]))
        }
        "rethrottle" => {
            let (k, n) = decode_id(r.task.unwrap_or(0));
            let Some(t) = node.engine_for(k).and_then(|e| e.tasks.get_running(n)) else { return NodeResp::missing() };
            t.set_rps(r.rps);
            rethrottle_children(node, &r.task_str, r.rps)?;
            NodeResp::ok([admitted_info(local,k,&t,true)?])
        }
        "rethrottle_children" => {
            rethrottle_children(node, &r.task_str, r.rps)?;
            NodeResp::ok(json!([]))
        }
        o => NodeResp::err(&task_error(400,"illegal_argument_exception",format_args!("unknown tasks op [{o}]"))),
    }
}

fn rethrottle_children(node: &ClusterNode, parent: &str, rps: f64) ->Result<(),EsError>{
    for (_, _, t, info) in running(node, false)? {
        if parent_of(&info) == Some(parent) && BULK_ACTIONS.contains(&t.action.as_str()) {
            t.set_rps(rps);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Coordination
// ---------------------------------------------------------------------------

enum CallFailure { Owned(crate::owned_json::OwnedJson), Admission(EsError) }
struct RemoteCause<'a>(&'a RemoteError);
impl Serialize for RemoteCause<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{use serde::ser::SerializeMap;let mut m=s.serialize_map(None)?;m.serialize_entry("type",&self.0.error_type)?;m.serialize_entry("reason",&self.0.reason)?;for(k,v)in &self.0.metadata{m.serialize_entry(k,v)?;}if let Some(c)=&self.0.caused_by{m.serialize_entry("caused_by",&RemoteCause(c))?;}m.end()}}
fn failed_node(id:&str,cause:impl Serialize)->Result<crate::owned_json::OwnedJson,EsError>{
 #[derive(Serialize)]struct Failure<'a,T>{#[serde(rename="type")]kind:&'static str,reason:SharedText,node_id:&'a str,caused_by:T}
 let text=SharedText::try_json(&Failure{kind:"failed_node_exception",reason:SharedText::try_format(format_args!("Failed node [{id}]"))?,node_id:id,caused_by:cause})?;crate::owned_json::OwnedJson::parse(&text)
}
fn call_failure(result:Result<crate::owned_json::OwnedJson,EsError>)->CallFailure{match result{Ok(v)=>CallFailure::Owned(v),Err(e)=>CallFailure::Admission(e)}}
struct TaskTargets { values:std::collections::BTreeSet<String>,_memory:elasticsearch_engine::memory::Reservation }
struct TaskTargetsIter { values:std::collections::btree_set::IntoIter<String>,_memory:elasticsearch_engine::memory::Reservation }
impl Iterator for TaskTargetsIter{type Item=String;fn next(&mut self)->Option<String>{self.values.next()}}
impl IntoIterator for TaskTargets{type Item=String;type IntoIter=TaskTargetsIter;fn into_iter(self)->Self::IntoIter{TaskTargetsIter{values:self.values.into_iter(),_memory:self._memory}}}
fn task_targets(state:&ClusterState,local:&str,spec:&str)->Result<TaskTargets,EsError>{
 let unit=2*(12*std::mem::size_of::<String>()+16*std::mem::size_of::<usize>());
 let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"task target selection")?;
 // Selector semantics can include local/master even when absent from nodes.
 // Charge each possible owned key and duplicate-insert overlap before resolver
 // allocation; role filters only remove keys, never add arbitrary selector text.
 for id in state.nodes.keys().map(String::as_str).chain([local]).chain(state.master_node_id.as_deref()){
  let bytes=id.len().checked_mul(2).and_then(|n|n.checked_add(unit)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task target key overflow"))?;memory.grow(bytes as u64,"task target keys and nodes")?;
 }
 let values=crate::rest::nodes::resolve_nodes(state,local,spec);Ok(TaskTargets{values,_memory:memory.into_result_owner()})
}
/// Run locally or remotely with owned failure envelopes as well as successes.
async fn call(node:&Arc<ClusterNode>,id:&str,r:&NodeReq)->Result<NodeResp,CallFailure>{
 if id==node.local.id{return node_op(node,r.clone()).await.map_err(|e|call_failure(NodeResp::err(&e).and_then(|r|r.json())));}
 let Some(addr)=node.address_of(id)else{
  #[derive(Serialize)]struct Missing<'a>{#[serde(rename="type")]kind:&'static str,reason:SharedText,node_id:&'a str}
  return Err(call_failure((||{failed_node(id,Missing{kind:"no_such_node_exception",reason:SharedText::try_format(format_args!("No such node [{id}]"))?,node_id:id})})()));
 };
 let to=Duration::from_millis(r.timeout_ms)+Duration::from_secs(30);
 node.transport.send_request::<_,NodeResp>(addr,ACTION_NODE,r,to).await.map_err(|e|call_failure(failed_node(id,RemoteCause(&e.to_remote_error()))))
}
async fn fan_out(node:&Arc<ClusterNode>,ids:impl IntoIterator<Item=impl AsRef<str>>,r:&NodeReq)->Result<TaskVec<(SharedText,Result<NodeResp,CallFailure>)>,EsError>{
 use std::future::Future;
 let mut futures=TaskVec::new()?;let mut results=TaskVec::new()?;
 for id in ids {
  let id=SharedText::try_new(id.as_ref())?;
  let future=async move{let result=call(node,&id,r).await;(id,result)};
  futures.memory.grow(std::mem::size_of_val(&future)as u64,"task fanout future")?;
  futures.push(Some(Box::pin(future)))?;results.push(None)?;
 }
 futures::future::poll_fn(|cx|{
  let mut pending=false;
  for (future,result) in futures.values.iter_mut().zip(results.values.iter_mut()) {
   if let Some(future)=future.as_mut(){match future.as_mut().poll(cx){std::task::Poll::Ready(value)=>{*result=Some(value);},std::task::Poll::Pending=>{pending=true;continue;}}}
   if result.is_some(){*future=None;}
  }
  if pending{std::task::Poll::Pending}else{std::task::Poll::Ready(())}
 }).await;
 let mut out=TaskVec::new()?;for result in results{out.push(result.expect("completed task fanout"))?;}Ok(out)
}
fn single_result(id:&str,result:Result<NodeResp,CallFailure>)->Result<TaskVec<(SharedText,Result<NodeResp,CallFailure>)>,EsError>{let mut out=TaskVec::new()?;out.push((SharedText::try_new(id)?,result))?;Ok(out)}

fn node_header(d: Option<&DiscoveryNode>) -> Map<String, Value> {
    let mut m = Map::new();
    if let Some(d) = d {
        let mut roles: Vec<String> = d.roles.iter().map(|r| r.name().to_string()).collect();
        roles.sort();
        m.insert("name".into(), json!(d.name));
        m.insert("transport_address".into(), json!(d.transport_address));
        m.insert("host".into(), json!(d.host));
        m.insert("ip".into(), json!(d.transport_address));
        m.insert("roles".into(), json!(roles));
        m.insert("attributes".into(), json!(d.attributes));
    }
    m
}

/// ES `ListTasksResponse` rendering for `group_by`.
fn group(entries: Vec<(Map<String, Value>, Vec<Value>)>, failures: Vec<Value>, group_by: &str) -> Result<Value, EsError> {
    validate_group_by(group_by)?;
    let mut out = Map::new();
    if !failures.is_empty() {
        out.insert("node_failures".into(), Value::Array(failures));
    }
    match group_by {
        "nodes" => {
            let mut nodes = Map::new();
            for (header, infos) in entries {
                let Some(first) = infos.first() else { continue };
                let id = first["node"].as_str().unwrap_or("").to_string();
                let mut h = header;
                let tasks: Map<String, Value> = infos.into_iter().map(|i| (task_key(&i), i)).collect();
                h.insert("tasks".into(), Value::Object(tasks));
                nodes.insert(id, Value::Object(h));
            }
            out.insert("nodes".into(), Value::Object(nodes));
        }
        "parents" => {
            let all: Vec<Value> = entries.into_iter().flat_map(|(_, i)| i).collect();
            let keys: HashSet<String> = all.iter().map(task_key).collect();
            let mut children: HashMap<String, Vec<Value>> = HashMap::new();
            let mut roots = Vec::new();
            for i in all {
                match parent_of(&i).filter(|p| keys.contains(*p)).map(str::to_string) {
                    Some(p) => children.entry(p).or_default().push(i),
                    None => roots.push(i),
                }
            }
            fn tree(mut i: Value, children: &mut HashMap<String, Vec<Value>>) -> Value {
                if let Some(cs) = children.remove(&task_key(&i)) {
                    let cs: Vec<Value> = cs.into_iter().map(|c| tree(c, children)).collect();
                    if let Some(o) = i.as_object_mut() {
                        o.insert("children".into(), Value::Array(cs));
                    }
                }
                i
            }
            let tasks: Map<String, Value> = roots.into_iter().map(|r| (task_key(&r), tree(r, &mut children))).collect();
            out.insert("tasks".into(), Value::Object(tasks));
        }
        _ => {
            let all: Vec<Value> = entries.into_iter().flat_map(|(_, i)| i).collect();
            out.insert("tasks".into(), Value::Array(all));
        }
    }
    Ok(Value::Object(out))
}

/// Parsed task trees and every descendant moved into the grouped result keep
/// their admission owners until the result has been serialized and dropped.
struct TaskRender {
    output: Value,
    entries: Vec<(Map<String,Value>,Vec<Value>)>,
    failures: Vec<Value>,
    owners: Vec<crate::owned_json::OwnedJson>,
    memory: elasticsearch_engine::memory::Reservation,
}
impl TaskRender {
    fn new(count:usize,failures:Vec<Value>)->Result<Self,EsError>{
        let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"task response aggregation")?;
        fn slots<T>(memory:&mut elasticsearch_engine::memory::Reservation,n:usize)->Result<Vec<T>,EsError>{
            let bytes=n.checked_mul(std::mem::size_of::<T>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task response capacity overflow"))?;
            memory.grow(bytes as u64,"task response slots")?;
            let mut out=Vec::new();out.try_reserve_exact(n).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;Ok(out)
        }
        let entries=slots(&mut memory,count)?;
        let owners=slots(&mut memory,count.checked_mul(2).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task owner count overflow"))?)?;
        let mut owned_failures=slots(&mut memory,count.checked_add(failures.len()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task failure count overflow"))?)?;
        owned_failures.extend(failures);
        Ok(Self{output:Value::Null,entries,failures:owned_failures,owners,memory})
    }
    fn retain(&mut self,mut parsed:crate::owned_json::OwnedJson)->Result<Value,EsError>{
        let mut value=Value::Null;
        parsed.transform(|v,_|{value=std::mem::take(v);Ok(())})?;
        debug_assert!(self.owners.len()<self.owners.capacity());
        self.owners.push(parsed);Ok(value)
    }
    fn plan_group(&mut self)->Result<(),EsError>{
        // The grouping algorithm moves payload trees. New allocations are
        // exclusively keys and containers: account each task's possible five
        // key copies and map/set/vector slots, including geometric overlap.
        // Map units use the same conservative node-split bound as OwnedJson.
        let map_unit=2*(12*std::mem::size_of::<(String,Value)>()+16*std::mem::size_of::<usize>());
        self.memory.grow((map_unit*4)as u64,"task grouping root maps")?;
        for (_,infos) in &self.entries {
            self.memory.grow((map_unit*3)as u64,"task grouping node maps")?;
            for info in infos {
                let key=info["node"].as_str().unwrap_or("").len().checked_add(21).ok_or_else(||EsError::new(429,"circuit_breaking_exception","task key overflow"))?;
                let parent=parent_of(info).map_or(0,str::len);
                let bytes=key.checked_mul(5).and_then(|n|n.checked_add(parent.saturating_mul(3)))
                    .and_then(|n|n.checked_add(map_unit*5+16*std::mem::size_of::<Value>()+16*std::mem::size_of::<(String,Vec<Value>)>()))
                    .ok_or_else(||EsError::new(429,"circuit_breaking_exception","task grouping overflow"))?;
                self.memory.grow(bytes as u64,"task grouping keys and container replacement")?;
            }
        }
        Ok(())
    }
}
fn admitted_node_header(d:Option<&DiscoveryNode>)->Result<crate::owned_json::OwnedJson,EsError>{
    #[derive(Serialize)]
    struct Header<'a>{name:&'a str,transport_address:&'a str,host:&'a str,ip:&'a str,roles:SharedSequence<&'static str>,attributes:&'a std::collections::BTreeMap<String,String>}
    let Some(d)=d else{return crate::owned_json::OwnedJson::parse("{}")};
    let roles=SharedSequence::try_generate_sorted(|push|{for role in &d.roles{push(role.name())?;}Ok(())})?;
    let text=SharedText::try_json(&Header{name:&d.name,transport_address:&d.transport_address,host:&d.host,ip:&d.transport_address,roles,attributes:&d.attributes})?;
    crate::owned_json::OwnedJson::parse(&text)
}
fn render(node: &ClusterNode, results: TaskVec<(SharedText, Result<NodeResp, CallFailure>)>, extra_failures: Vec<Value>, group_by: &str) -> RestResp {
    let build=(||->Result<TaskRender,EsError>{
        let state=node.state();let mut owned=TaskRender::new(results.len(),extra_failures)?;
        for (id,res) in results {
            match res {
                Ok(r) if r.status==200=>{
                    let header=admitted_node_header(state.nodes.get(id.as_str()))?;
                    let header=match owned.retain(header)?{Value::Object(m)=>m,_=>unreachable!()};
                    let infos=match owned.retain(r.json()?)?{Value::Array(v)=>v,_=>Vec::new()};
                    owned.entries.push((header,infos));
                }
                Ok(r)=>{
                    let parsed=r.json()?;
                    #[derive(Serialize)] struct Failure<'a>{#[serde(rename="type")]kind:&'a str,reason:SharedText,node_id:&'a str,caused_by:&'a Value}
                    let text=SharedText::try_json(&Failure{kind:"failed_node_exception",reason:SharedText::try_format(format_args!("Failed node [{id}]"))?,node_id:&id,caused_by:parsed.get("error").unwrap_or(&Value::Null)})?;
                    let failure=owned.retain(crate::owned_json::OwnedJson::parse(&text)?)?;
                    owned.failures.push(failure);
                }
                Err(CallFailure::Owned(f))=>{let f=owned.retain(f)?;owned.failures.push(f);},
                Err(CallFailure::Admission(e))=>return Err(e),
            }
        }
        owned.plan_group()?;
        owned.output=group(std::mem::take(&mut owned.entries),std::mem::take(&mut owned.failures),group_by)?;
        Ok(owned)
    })();
    match build{Ok(owned)=>RestResp::json(200,&owned.output),Err(e)=>err_resp(&e)}
}

fn node_failure_response(node:&str,kind:&str,reason:&str)->Result<RestResp,EsError>{
 #[derive(Serialize)]struct Cause<'a>{#[serde(rename="type")]kind:&'a str,reason:&'a str}
 let failure=failed_node(node,Cause{kind,reason})?;
 #[derive(Serialize)]struct Reply<'a>{node_failures:[&'a crate::owned_json::OwnedJson;1],nodes:std::collections::BTreeMap<&'static str,u8>}
 let body=SharedText::try_json(&Reply{node_failures:[&failure],nodes:std::collections::BTreeMap::new()})?;
 Ok(RestResp::new(200,vec![],body.as_bytes()))
}
fn missing_task_failure(node_id:&str,id:&str)->Result<RestResp,EsError>{
 let reason=SharedText::try_format(format_args!("task [{id}] is not found"))?;
 node_failure_response(node_id,"resource_not_found_exception",&reason)
}

fn from_node(r: NodeResp) -> RestResp {
    RestResp::new(r.status,vec![("content-type".into(), "application/json".into())],r.body.as_bytes())
}

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let segs: Vec<&str> = segs.iter().map(String::as_str).collect();
    let post = req.method() == axum::http::Method::POST;
    let get = req.method() == axum::http::Method::GET;
    let res = match segs.as_slice() {
        ["_tasks"] if get => list(node, req).await,
        ["_tasks", "_cancel"] if post => cancel(node, req, None).await,
        ["_tasks", id] if get => get_task(node, req, id).await,
        ["_tasks", id, "_cancel"] if post => cancel(node, req, Some(id)).await,
        [.., id, "_rethrottle"] if post && segs.len() >= 3 => rethrottle(node, req, id).await,
        _ => return local_engine(node, req).await,
    };
    res.unwrap_or_else(|e| err_resp(&e))
}

async fn list(node: &Arc<ClusterNode>, req: &RestReq) -> Result<RestResp, EsError> {
    let p = req.params();
    let detailed = flag(&p, "detailed")?;
    let group_by = p.get("group_by").cloned().unwrap_or_else(|| "nodes".into());
    validate_group_by(&group_by)?;
    let wait = flag(&p, "wait_for_completion")?;
    let timeout = if wait { timeout_param(&p)? } else { Duration::ZERO };
    let op = opaque(req)?;
    let me = ListTaskGuard{engine:node.engine.clone(),task:node.engine.tasks.try_register_borrowed(LIST_ACTION, Some(""), true, false, &op, None)?};
    let prepared=(||->Result<NodeReq,EsError>{Ok(NodeReq {
        op: SharedText::try_new("list")?,
        actions: parse_actions(p.get("actions"), true)?,
        parent_task_id: p.get("parent_task_id").map(|s|SharedText::try_new(s)).transpose()?,
        detailed,
        wait,
        timeout_ms: timeout.as_millis() as u64,
        list_task: Some(SharedText::try_format(format_args!("{}:{}", node.local.id, encode_id(0, me.id)))?),
        opaque: op.clone(),
        ..Default::default()
    })})();
    let r=prepared?;
    let targets = task_targets(&node.state(), &node.local.id, p.get("nodes").map(String::as_str).unwrap_or(""))?;
    let results = fan_out(node, targets, &r).await;
    drop(me);
    let results=results?;
    Ok(render(node, results, Vec::new(), &group_by))
}

/// The owning node of task `id`, or `Err` with the parsed number missing.
fn owner(node:&ClusterNode,id:&str)->Result<(SharedText,u64,bool),EsError>{
 let bad=||task_error(400,"illegal_argument_exception",format_args!("malformed task id {id}"));
 let(name,num)=id.split_once(':').ok_or_else(bad)?;
 let num=num.parse::<u64>().map_err(|_|bad())?;
 if name.is_empty(){return Err(bad())}
 let known=node.state().nodes.contains_key(name);Ok((SharedText::try_new(name)?,num,known))
}

async fn get_task(node: &Arc<ClusterNode>, req: &RestReq, id: &str) -> Result<RestResp, EsError> {
    let p = req.params();
    let (n, num, known) = owner(node, id)?;
    if !known {
        return Err(task_error(404,"resource_not_found_exception",format_args!("task [{id}] belongs to the node [{n}] which isn't part of the cluster and there is no record of the task")));
    }
    let wait = flag(&p, "wait_for_completion")?;
    let timeout = if wait { timeout_param(&p)? } else { Duration::ZERO };
    let r = NodeReq { op: SharedText::try_new("get")?, task: Some(num), task_str: SharedText::try_new(id)?, wait, timeout_ms: timeout.as_millis() as u64, ..Default::default() };
    match call(node, &n, &r).await {
        Ok(resp) => Ok(from_node(resp)),
        Err(CallFailure::Admission(e))=>Err(e),
        Err(CallFailure::Owned(f))=>{
            #[derive(Serialize)]struct ErrorReply<'a>{error:&'a crate::owned_json::OwnedJson,status:u16}
            let body=SharedText::try_json(&ErrorReply{error:&f,status:500})?;
            Ok(RestResp::new(500,vec![],body.as_bytes()))
        },
    }
}

async fn cancel(node: &Arc<ClusterNode>, req: &RestReq, id: Option<&str>) -> Result<RestResp, EsError> {
    let p = req.params();
    let group_by = p.get("group_by").cloned().unwrap_or_else(|| "nodes".into());
    let wait = flag(&p, "wait_for_completion")?;
    let timeout_ms = if wait { timeout_param(&p)?.as_millis() as u64 } else { 0 };
    let all_nodes=SharedSequence::try_generate(|push|{for id in node.state().nodes.keys(){push(SharedText::try_new(id)?)?;}Ok(())})?;
    if let Some(id) = id {
        let (n, num, known) = owner(node, id)?;
        if !known {
            return missing_task_failure(&n,id);
        }
        let r = NodeReq { op: SharedText::try_new("cancel_id")?, task: Some(num), task_str: SharedText::try_new(id)?, wait, timeout_ms, ..Default::default() };
        let ban = NodeReq { op: SharedText::try_new("ban")?, ids: SharedSequence::try_collect([SharedText::try_new(id)?])?, ..Default::default() };
        let others=SharedSequence::try_collect(all_nodes.iter().filter(|x|x.as_str()!=n.as_str()).cloned())?;
        let (res, banned) = tokio::join!(call(node, &n, &r), fan_out(node, others, &ban));
        let _=banned?;
        return Ok(match res {
            Ok(resp) if resp.status == 404 && resp.body.is_empty() => missing_task_failure(&n,id)?,
            Ok(resp) if resp.status != 200 => from_node(resp),
            res => render(node, single_result(&n,res)?, Vec::new(), &group_by),
        });
    }
    let actions = parse_actions(p.get("actions"), true)?;
    let parent = p.get("parent_task_id").map(|s|SharedText::try_new(s)).transpose()?;
    let nodes = p.get("nodes").cloned();
    if actions.is_none() && parent.is_none() && nodes.is_none() {
        let r = NodeReq { op: SharedText::try_new("cancel_all")?, ..Default::default() };
        fan_out(node, all_nodes, &r).await?;
        return node_failure_response(&node.local.id,"task_cancelled_exception","task cancelled [by user request]");
    }
    let targets = task_targets(&node.state(), &node.local.id, nodes.as_deref().unwrap_or(""))?;
    let r = NodeReq { op: SharedText::try_new("cancel")?, actions, parent_task_id: parent, wait, timeout_ms, ..Default::default() };
    let results = fan_out(node, targets, &r).await?;
    // Ban the cancelled tasks' children everywhere (ES bans every cancelled parent).
    let ids = SharedSequence::try_generate(|push| {
        for response in results.iter().filter_map(|(_,r)|r.as_ref().ok()) {
            let parsed=response.json()?;
            if let Some(infos)=parsed.as_array(){for i in infos {push(SharedText::try_format(format_args!("{}:{}",i["node"].as_str().unwrap_or(""),i["id"]))?)?;}}
        }
        Ok(())
    })?;
    if !ids.is_empty() {
        fan_out(node, all_nodes, &NodeReq { op: SharedText::try_new("ban")?, ids, ..Default::default() }).await?;
    }
    Ok(render(node, results, Vec::new(), &group_by))
}

async fn rethrottle(node: &Arc<ClusterNode>, req: &RestReq, id: &str) -> Result<RestResp, EsError> {
    let p = req.params();
    let group_by = p.get("group_by").cloned().unwrap_or_else(|| "nodes".into());
    let Some(rps) = p.get("requests_per_second") else {
        return Err(EsError::illegal_argument("requests_per_second is a required parameter"));
    };
    let rps = elasticsearch_engine::reindex::parse_rps(rps)?;
    let (n, num, known) = owner(node, id)?;
    if !known {
        let reason=SharedText::try_format(format_args!("No such node [{n}]"))?;
        // Preserve no_such_node's node_id metadata in the failed-node cause.
        #[derive(Serialize)]struct Missing<'a>{#[serde(rename="type")]kind:&'static str,reason:&'a str,node_id:&'a str}
        let failure=failed_node(&n,Missing{kind:"no_such_node_exception",reason:&reason,node_id:&n})?;
        #[derive(Serialize)]struct Reply<'a>{node_failures:[&'a crate::owned_json::OwnedJson;1],nodes:std::collections::BTreeMap<&'static str,u8>}
        let body=SharedText::try_json(&Reply{node_failures:[&failure],nodes:std::collections::BTreeMap::new()})?;
        return Ok(RestResp::new(200,vec![],body.as_bytes()));
    }
    let r = NodeReq { op: SharedText::try_new("rethrottle")?, task: Some(num), task_str: SharedText::try_new(id)?, rps, ..Default::default() };
    let res = call(node, &n, &r).await;
    if let Ok(resp) = &res {
        if resp.status == 404 && resp.body.is_empty() {
            return missing_task_failure(&n,id);
        }
        if resp.status != 200 {
            return Ok(from_node(resp.clone()));
        }
    }
    let others=SharedSequence::try_generate(|push|{for id in node.state().nodes.keys().filter(|id|id.as_str()!=n.as_str()){push(SharedText::try_new(id)?)?;}Ok(())})?;
    fan_out(node, others, &NodeReq { op: SharedText::try_new("rethrottle_children")?, task_str: SharedText::try_new(id)?, rps, ..Default::default() }).await?;
    Ok(render(node, single_result(&n,res)?, Vec::new(), &group_by))
}

// ---------------------------------------------------------------------------
// Child tasks of cluster tasks, task refs in engine responses
// ---------------------------------------------------------------------------

/// Run a request on shard engine `k`. When it carries [`PARENT_HEADER`], the
/// bulk-by-scroll task it starts there is linked to that cluster parent while
/// it runs (and cancelled at once if the parent is banned).
pub async fn as_child(node: &Arc<ClusterNode>, k: u32, req: &RestReq) -> RestResp {
    let parent=req.headers.iter().find(|(h,_)|h.eq_ignore_ascii_case(PARENT_HEADER)).map(|(_,v)|SharedText::try_new(v)).transpose();
    let parent=match parent{Ok(v)=>v,Err(e)=>return err_resp(&e)};
    let(Some(parent),Some(engine))=(parent,node.engine_for(k))else{return local_engine(node,req).await};
    let mut before=match TaskVec::new(){Ok(v)=>v,Err(e)=>return err_resp(&e)};
    if let Err(e)=engine.tasks.try_visit_running(|t|before.push(t.id)){return err_resp(&e)}
    let mut linked=match TaskVec::new(){Ok(v)=>v,Err(e)=>return err_resp(&e)};
    let link=|linked:&mut TaskVec<u64>|->Result<(),EsError>{
        let mut candidates=TaskVec::new()?;
        let mut failure=None;
        engine.tasks.try_visit_running(|t|{
            if !before.contains(&t.id)&&!linked.contains(&t.id)&&t.parent.is_none()&&BULK_ACTIONS.contains(&t.action.as_str()){
                if failure.is_some(){t.request_cancel();}
                else if let Err(e)=candidates.push(t.clone()){t.request_cancel();failure=Some(e);}
            }
            Ok::<_,EsError>(())
        })?;
        if let Some(error)=failure{for t in candidates{t.request_cancel();}return Err(error)}
        for t in candidates {
            // Admission failure cannot leave this task silently untracked.
            // Cancel it, then allow the dispatched operation to finish before
            // returning the failure and removing already linked children.
            if let Err(e)=linked.push(t.id).and_then(|_|CHILDREN.lock().insert((k,t.id),&parent)){
                let _=engine.tasks.cancel(t.id);return Err(e);
            }
            if banned(&parent){let _=engine.tasks.cancel(t.id);}
        }Ok(())
    };
    let fut=local_engine(node,req);tokio::pin!(fut);
    let mut tick=tokio::time::interval(Duration::from_millis(20));let mut failure=None;
    let response=loop{tokio::select!{response=&mut fut=>break response,_=tick.tick()=>if let Err(e)=link(&mut linked){failure=Some(e)}}};
    let mut registry=CHILDREN.lock();for id in linked{registry.values.remove(&(k,id));}drop(registry);
    match failure{Some(e)=>err_resp(&e),None=>response}
}

/// Rewrite an engine `{"task": "<engine node>:<n>"}` reply from shard engine
/// `k` to the cluster id of that task. Any other reply (a bulk or document
/// write's result) is not parsed. The reply reports work already started or
/// applied, so it is never replaced with an error (the client would run the
/// write again). The reference is a few dozen bytes, formatted and swapped in
/// without breaker admission: a refused rewrite would hand the client the
/// engine-local id, which `_tasks/{id}` and `_cancel` cannot resolve (404).
pub fn rewrite_task_ref(node: &ClusterNode, k: u32, resp: &mut RestResp) {
    rewrite_task_ref_for(&node.local.id, k, resp);
}

/// The largest task reference reply rewritten (an engine reference is
/// `{"task":"<id>:<n>"}`; anything larger is not one).
const TASK_REF_MAX: usize = 512;

fn rewrite_task_ref_for(local_id: &str, k: u32, resp: &mut RestResp) {
    if resp.body.len() > TASK_REF_MAX || !is_task_ref(&resp.body) || resp.headers.iter().any(|(h, v)| h.eq_ignore_ascii_case("content-encoding") && v.contains("gzip")) {
        return;
    }
    let Ok(text) = std::str::from_utf8(&resp.body) else { return };
    let Some(members) = elasticsearch_engine::source_json::raw_members(text) else { return };
    let Some(task) = members.iter().find(|(key, _)| key == "task").and_then(|(_, v)| serde_json::from_str::<String>(v.get()).ok()) else { return };
    let Some(n) = task.strip_prefix(NODE_ID).and_then(|r| r.strip_prefix(':')).and_then(|n| n.parse::<u64>().ok()) else { return };
    let Ok(id) = serde_json::to_string(&format!("{local_id}:{}", encode_id(k, n))) else { return };
    let Some(body) = elasticsearch_engine::source_json::with_member(text, "task", &id, false) else { return };
    if !resp.replace_small_body_unadmitted(body, TASK_REF_MAX + local_id.len() + 32) {
        tracing::debug!("task reference left unrewritten");
    }
}

/// Does `body` open with a `"task"` member (an engine task reference, whose
/// only member it is)? A bulk or document reply opens with another one.
fn is_task_ref(body: &[u8]) -> bool {
    let rest = body.trim_ascii_start();
    let Some(rest) = rest.strip_prefix(b"{") else { return false };
    rest.trim_ascii_start().starts_with(br#""task""#)
}

// ---------------------------------------------------------------------------
// _cat/tasks
// ---------------------------------------------------------------------------

const TASK_COLS: &[crate::rest::cat::ColDef] = &[
    ("id", &[], "id of the task with the node", false, false),
    ("action", &["ac"], "task action", true, false),
    ("task_id", &["ti"], "unique task id", true, false),
    ("parent_task_id", &["pti"], "parent task id", true, false),
    ("type", &["ty"], "task type", true, false),
    ("start_time", &["start"], "start time in ms", true, false),
    ("timestamp", &["ts", "hms", "hhmmss"], "start time in HH:MM:SS", true, false),
    ("running_time_ns", &[], "running time ns", false, false),
    ("running_time", &["time"], "running time", true, false),
    ("node_id", &["ni"], "unique node id", false, false),
    ("ip", &["i"], "ip address", true, false),
    ("port", &["po"], "bound transport port", false, false),
    ("node", &["n"], "node name", true, false),
    ("version", &["v"], "es version", false, false),
    ("x_opaque_id", &["x"], "X-Opaque-ID header", false, false),
    ("description", &["desc"], "task action", false, false),
];

fn time_fmt(ms: u64) -> String {
    let units = [("d", 86_400_000u64), ("h", 3_600_000), ("m", 60_000), ("s", 1000)];
    for (u, n) in units {
        if ms >= n {
            let t = format!("{:.1}", ms as f64 / n as f64);
            return format!("{}{u}", t.strip_suffix(".0").unwrap_or(&t));
        }
    }
    format!("{ms}ms")
}

fn running_time(ns: u64) -> String {
    if ns >= 1_000_000 {
        time_fmt(ns / 1_000_000)
    } else {
        format!("{}micros", (ns / 1000).max(1))
    }
}

fn admit_cat_row(memory:&mut elasticsearch_engine::memory::Reservation,info:&Value,d:Option<&DiscoveryNode>)->Result<(),EsError>{
    // Sixteen fixed cells, bounded decimal/time formatters, and exact lengths
    // of copied wire fields. No encoded-body-size estimate is involved.
    let mut bytes=16*std::mem::size_of::<(&str,Option<String>)>()+512;
    for key in ["action","node","node","parent_task_id","description"]{
        bytes=bytes.checked_add(info.get(key).and_then(Value::as_str).map_or(0,str::len)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","cat task text overflow"))?;
    }
    bytes=bytes.checked_add(info["headers"].get("X-Opaque-Id").and_then(Value::as_str).map_or(0,str::len)).ok_or_else(||EsError::new(429,"circuit_breaking_exception","cat task header overflow"))?;
    if let Some(d)=d {bytes=bytes.checked_add(d.name.len()).and_then(|n|n.checked_add(d.transport_address.len())).ok_or_else(||EsError::new(429,"circuit_breaking_exception","cat node text overflow"))?;}
    memory.grow(bytes as u64,"cat task row storage")
}
fn cat_row(info: &Value, d: Option<&DiscoveryNode>) -> crate::rest::cat::Row {
    let s = |k: &str| info.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let start = info["start_time_in_millis"].as_i64().unwrap_or(0);
    let ns = info["running_time_in_nanos"].as_u64().unwrap_or(0);
    let (ip, port) = d.and_then(|d| d.transport_address.rsplit_once(':')).map(|(h, p)| (h.to_string(), p.to_string())).unwrap_or_default();
    vec![
        ("id", Some(info["id"].to_string())),
        ("action", Some(s("action"))),
        ("task_id", Some(task_key(info))),
        ("parent_task_id", Some(parent_of(info).unwrap_or("-").to_string())),
        ("type", Some("transport".into())),
        ("start_time", Some(start.to_string())),
        ("timestamp", Some(elasticsearch_engine::date::from_millis(start).format("%H:%M:%S").to_string())),
        ("running_time_ns", Some(ns.to_string())),
        ("running_time", Some(running_time(ns))),
        ("node_id", Some(s("node"))),
        ("ip", Some(ip)),
        ("port", Some(port)),
        ("node", Some(d.map(|d| d.name.clone()).unwrap_or_default())),
        ("version", Some(ES_VERSION.into())),
        ("x_opaque_id", Some(info["headers"].get("X-Opaque-Id").and_then(Value::as_str).unwrap_or("").to_string())),
        ("description", Some(s("description"))),
    ]
}

pub async fn cat_tasks(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let p = req.params();
    let detailed = match flag(&p, "detailed") {
        Ok(d) => d,
        Err(e) => return err_resp(&e),
    };
    let op = match opaque(req){Ok(v)=>v,Err(e)=>return err_resp(&e)};
    let parent = match node.engine.tasks.try_register_borrowed(LIST_ACTION,None,false,false,&op,None){Ok(task)=>ListTaskGuard{engine:node.engine.clone(),task},Err(e)=>return err_resp(&e)};
    let prepared=(||->Result<NodeReq,EsError>{Ok(NodeReq {
        op: SharedText::try_new("list")?,
        actions: parse_actions(p.get("actions"), false)?,
        parent_task_id: p.get("parent_task_id").map(|s|SharedText::try_new(s)).transpose()?,
        detailed: true,
        list_task: Some(SharedText::try_format(format_args!("{}:{}", node.local.id, encode_id(0, parent.id)))?),
        cat: true,
        opaque: op.clone(),
        ..Default::default()
    })})();
    let r=match prepared{Ok(r)=>r,Err(e)=>return err_resp(&e)};
    let state = node.state();
    let targets=match task_targets(&state,&node.local.id,p.get("nodes").map(String::as_str).unwrap_or("")){Ok(v)=>v,Err(e)=>return err_resp(&e)};
    let results = fan_out(node, targets, &r).await;
    drop(parent);
    let results=match results{Ok(v)=>v,Err(e)=>return err_resp(&e)};
    let mut row_memory=match elasticsearch_engine::memory::breakers().reserve_request(0,"cat task rows"){Ok(v)=>v,Err(e)=>return err_resp(&e)};
    let mut rows = Vec::new();
    for (id, res) in results {
        let Ok(resp) = res else { continue };
        let parsed=match resp.json(){Ok(v)=>v,Err(e)=>return err_resp(&e)};
        if let Some(infos)=parsed.as_array(){for info in infos {
            if let Err(e)=admit_cat_row(&mut row_memory,info,state.nodes.get(id.as_str())){return err_resp(&e)}
            if rows.len()==rows.capacity(){
                let cap=match rows.capacity().checked_mul(2).and_then(|n|n.checked_add(1)){Some(n)=>n,None=>return err_resp(&EsError::new(429,"circuit_breaking_exception","cat task rows overflow"))};
                let bytes=match cap.checked_mul(std::mem::size_of::<crate::rest::cat::Row>()){Some(n)=>n,None=>return err_resp(&EsError::new(429,"circuit_breaking_exception","cat task row bytes overflow"))};
                if let Err(e)=row_memory.grow(bytes as u64,"cat rows replacement"){return err_resp(&e)}
                if let Err(e)=rows.try_reserve_exact(cap-rows.len()){return err_resp(&EsError::new(429,"circuit_breaking_exception",e.to_string()))}
            }
            rows.push(cat_row(info, state.nodes.get(id.as_str())));
        }}
    }
    let mut cols = crate::rest::cat::cols(TASK_COLS);
    if let Some(c) = cols.iter_mut().find(|c| c.name == "description") {
        c.default = detailed;
    }
    crate::rest::cat::render(req, &cols, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(node: &str, id: u64, parent: Option<&str>) -> Value {
        let mut v = json!({"node": node, "id": id, "type": "transport", "action": "a", "start_time_in_millis": 1, "running_time_in_nanos": 2, "cancellable": false});
        if let Some(p) = parent {
            v["parent_task_id"] = json!(p);
        }
        v["headers"] = json!({});
        v
    }

    #[test]
    fn applied_write_reply_is_never_replaced() {
        let bulk = json!({"took": 3, "errors": false, "items": [{"index": {"_index": "i", "_id": "1", "result": "created", "status": 201, "_seq_no": 0}}]});
        // The reply of an applied and replicated bulk goes back as it is:
        // the client must not retry and apply its operations again.
        let mut resp = RestResp::json(200, &bulk);
        let before = resp.body.clone();
        rewrite_task_ref_for("cluster-node", 2, &mut resp);
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, before);
        // A task reference becomes the cluster id, swapped in without breaker
        // admission (a refused rewrite would leak the engine-local id).
        let task = json!({"task": format!("{NODE_ID}:9")});
        let mut resp = RestResp::json(200, &task);
        rewrite_task_ref_for("cluster-node", 2, &mut resp);
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body_json().unwrap(), json!({"task": format!("cluster-node:{}", encode_id(2, 9))}));
        assert!(!resp.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")));
        // Oversized bodies are not task references and stay untouched.
        let big = json!({"task": format!("{NODE_ID}:9"), "pad": "x".repeat(TASK_REF_MAX)});
        let mut resp = RestResp::json(200, &big);
        let before = resp.body.clone();
        rewrite_task_ref_for("cluster-node", 2, &mut resp);
        assert_eq!(resp.body, before);
    }

    #[test]
    fn only_task_references_are_parsed() {
        assert!(is_task_ref(br#"{"task":"n:1"}"#));
        assert!(is_task_ref(b"{\n  \"task\" : \"n:1\"\n}"));
        assert!(!is_task_ref(br#"{"took":1,"errors":false,"items":[{"task":"x"}]}"#));
        assert!(!is_task_ref(br#"{"_index":"i","result":"created"}"#));
        assert!(!is_task_ref(b""));
    }

    #[test]
    fn id_encoding_roundtrips() {
        assert_eq!(encode_id(0, 7), 7);
        let e = encode_id(3, 12);
        assert_eq!(e, (3u64 << 40) | 12);
        assert_eq!(decode_id(e), (3, 12));
        assert_eq!(decode_id(7), (0, 7));
    }

    #[test]
    fn rewrite_info_names_cluster_node_and_keeps_order() {
        let i = info(NODE_ID, 5, Some(&format!("{NODE_ID}:4")));
        let r = rewrite_info("nodeA", 2, &i, None);
        assert_eq!(r["node"], "nodeA");
        assert_eq!(r["id"], encode_id(2, 5));
        assert_eq!(r["parent_task_id"], format!("nodeA:{}", encode_id(2, 4)));
        let keys: Vec<&str> = r.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["node", "id", "type", "action", "start_time_in_millis", "running_time_in_nanos", "cancellable", "parent_task_id", "headers"]);
        // A linked cluster parent wins; it goes right before `headers`.
        let r = rewrite_info("nodeA", 1, &info(NODE_ID, 9, None), Some("nodeB:3"));
        assert_eq!(r["parent_task_id"], "nodeB:3");
        let keys: Vec<&str> = r.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(&keys[keys.len() - 2..], ["parent_task_id", "headers"]);
        // Cluster parents (already node-qualified) are left alone.
        let r = rewrite_info("nodeA", 0, &info(NODE_ID, 9, Some("nodeC:1")), None);
        assert_eq!(r["parent_task_id"], "nodeC:1");
    }

    #[test]
    fn grouping() {
        let mut h = Map::new();
        h.insert("name".into(), json!("n1"));
        let entries = vec![(h.clone(), vec![info("n1", 1, None), info("n1", 2, Some("n1:1"))]), (Map::new(), vec![])];
        let v = group(entries.clone(), vec![], "nodes").unwrap();
        assert_eq!(v["nodes"]["n1"]["name"], "n1");
        assert_eq!(v["nodes"]["n1"]["tasks"]["n1:2"]["parent_task_id"], "n1:1");
        assert_eq!(v["nodes"].as_object().unwrap().len(), 1);
        let v = group(entries.clone(), vec![], "parents").unwrap();
        assert_eq!(v["tasks"].as_object().unwrap().len(), 1);
        assert_eq!(v["tasks"]["n1:1"]["children"][0]["id"], 2);
        let v = group(entries.clone(), vec![json!({"x": 1})], "none").unwrap();
        assert_eq!(v["tasks"].as_array().unwrap().len(), 2);
        assert_eq!(v.as_object().unwrap().keys().next().unwrap(), "node_failures");
        assert_eq!(group(vec![], vec![], "nodes").unwrap(), json!({"nodes": {}}));
        let e = group(entries, vec![], "foo").unwrap_err();
        assert_eq!(e.reason, "[group_by] must be one of [nodes], [parents] or [none] but was [foo]");
    }

    #[test]
    fn time_and_params() {
        assert_eq!(time_fmt(1500), "1.5s");
        assert_eq!(time_fmt(60_000), "1m");
        assert_eq!(time_fmt(12), "12ms");
        assert_eq!(running_time(500), "1micros");
        assert_eq!(running_time(2_500_000), "2ms");
        let a = parse_actions(Some(&" a*, ,b".to_string()), true).unwrap().unwrap();
        assert_eq!(a.iter().map(|v|v.as_str()).collect::<Vec<_>>(), ["a*", "b"]);
        assert_eq!(parse_actions(Some(&"a, b".to_string()), false).unwrap().unwrap().iter().map(|v|v.as_str()).collect::<Vec<_>>(), ["a", " b"]);
        assert!(action_matches(&Some(a), "abc"));
        let mut p = HashMap::new();
        assert!(!flag(&p, "detailed").unwrap());
        p.insert("detailed".to_string(), String::new());
        assert!(flag(&p, "detailed").unwrap());
        p.insert("detailed".to_string(), "x".into());
        assert_eq!(flag(&p, "detailed").unwrap_err().reason, "Failed to parse value [x] as only [true] or [false] are allowed.");
        p.insert("timeout".to_string(), "5".into());
        assert!(timeout_param(&p).is_err());
    }
}
