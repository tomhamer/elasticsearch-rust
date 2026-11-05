//! `_nodes` info / stats / usage / hot_threads across the cluster.
//!
//! Each node renders its own entry with its engine's single-node
//! implementation (which already produces ES's full section shapes), the
//! coordinating node gathers the entries over the transport and patches in
//! the node's real identity (id, name, addresses, roles, attributes).

use crate::front::{RestReq, RestResp, SharedText};
fn wire_memory(e:elasticsearch_engine::error::EsError)->RemoteError{crate::control_memory::remote_error(&e)}
use crate::node::ClusterNode;
use elasticsearch_cluster::allocation::DiskUsage;
use elasticsearch_cluster::state::{ClusterState, DiscoveryNode, Role};
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
// Borrow the original to_json schema (without root_cause) for counted encoding.
struct RemoteJson<'a>(&'a RemoteError);
impl Serialize for RemoteJson<'_>{
 fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
  use serde::ser::SerializeMap;let e=self.0;let mut out=s.serialize_map(None)?;
  if !e.metadata.contains_key("type"){out.serialize_entry("type",&e.error_type)?;}
  if !e.metadata.contains_key("reason"){out.serialize_entry("reason",&e.reason)?;}
  for(k,v)in &e.metadata{if k!="caused_by"||e.caused_by.is_none(){out.serialize_entry(k,v)?;}}
  if let Some(cause)=&e.caused_by{out.serialize_entry("caused_by",&RemoteJson(cause))?;}out.end()
 }
}


pub const ACTION_LOCAL: &str = "cluster:monitor/nodes/local";
pub const ACTION_GPU_STATS: &str = elasticsearch_engine::gpu::STATS_ACTION;
pub const ACTION_DISK: &str = "internal:monitor/fs/disk";
pub const ACTION_ANALYTICS_USAGE: &str = "cluster:monitor/xpack/analytics/stats";
pub const ACTION_SEARCH_TELEMETRY: &str = "cluster:monitor/stats/search_telemetry";
pub const ACTION_CCS_TELEMETRY: &str = "cluster:monitor/stats/ccs_telemetry";
pub const ACTION_SYNONYMS_RELOAD: &str = "indices:admin/synonyms/reload_analyzers";

#[derive(Serialize, Deserialize)]
struct LocalReq {
    /// Engine path to run locally, e.g. `/_nodes/_local/stats/os`.
    uri: SharedText,
}

const INFO_METRICS: &[&str] = &["settings", "os", "process", "jvm", "thread_pool", "transport", "http", "remote_cluster_server", "plugins", "ingest", "aggregations", "indices", "_all", "_none"];
const STATS_METRICS: &[&str] = &[
    "_all", "_none", "indices", "os", "process", "jvm", "thread_pool", "fs", "transport", "http", "breaker", "script", "discovery", "ingest", "adaptive_selection", "script_cache", "indexing_pressure", "repositories", "allocations",
];

pub fn register(node: &Arc<ClusterNode>) {
    crate::capacity::register(node);
    let n = node.clone();
    node.transport.register_handler(ACTION_LOCAL, move |r: LocalReq| {
        let n = n.clone();
        async move { elasticsearch_engine::memory::request_policy_future(true,async { local_entry(&n, &r.uri).await.map_err(|r| n.transport.admitted_error("exception",std::str::from_utf8(&r.body).unwrap_or("invalid UTF-8 node error"),r.status).unwrap_or_else(|e|e.to_remote_error())).and_then(|v|SharedText::try_json(&*v).map_err(wire_memory)) }).await }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_ANALYTICS_USAGE, move |_: ()| {
        let n = n.clone();
        async move { elasticsearch_engine::memory::native_control(||node_usage_counters(&n).and_then(|v|SharedText::try_json(&*v)).map_err(wire_memory)) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_SYNONYMS_RELOAD, move |set: SharedText| {
        let n = n.clone();
        async move {
            let e = n.engine.clone();
            let r = tokio::task::spawn_blocking(move || elasticsearch_engine::search_ext::reload_dependents(&e, &set)).await.map_err(|e| RemoteError::internal(e.to_string()))?;
            r.map_err(|e| RemoteError::new(&e.kind,e.reason,e.status)).and_then(|v|SharedText::try_json(&v).map_err(wire_memory))
        }
    });
    node.transport.register_handler(ACTION_SEARCH_TELEMETRY, move |_: ()| async move {
        elasticsearch_engine::memory::native_control(||SharedText::try_json(&elasticsearch_http::search_usage_json()).map_err(wire_memory))
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_GPU_STATS, move |_: ()| {
        let n = n.clone();
        async move { elasticsearch_engine::memory::native_control(||{let stats=local_gpu_stats(&n.engine).map_err(wire_memory)?;SharedText::try_json(&*stats).map_err(wire_memory)}) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_CCS_TELEMETRY, move |_: ()| {
        let n = n.clone();
        async move { elasticsearch_engine::memory::native_control(||{let stats=local_ccs_stats(&n).map_err(wire_memory)?;SharedText::try_json(&*stats).map_err(wire_memory)}) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_DISK, move |_: ()| {
        let n = n.clone();
        async move {
            let d = crate::stats::disk_usage(&n.config.data_path);
            Ok::<_, RemoteError>(DiskWire { path: crate::control_memory::formatted(format_args!("{}",n.config.data_path.display())).map_err(|e|RemoteError::new(e.kind,e.reason,e.status))?, total_bytes: d.total_bytes, free_bytes: d.available_bytes })
        }
    });
}

/// Usage counted on the node that ran the work: analytics aggregations
/// (counted by search coordinators) and ES|QL (run on the master). ES merges
/// every ES|QL counter section across nodes (`EsqlUsageTransportAction`).
const NODE_COUNTERS: &[(&str, &str)] = &[
    ("analytics", "stats"),
    ("esql", "features"),
    ("esql", "queries"),
    ("esql", "functions"),
    ("esql", "settings"),
    ("esql", "took"),
];

fn node_usage_counters(node:&ClusterNode)->Result<crate::owned_json::OwnedJson,elasticsearch_engine::error::EsError>{
    let usage=elasticsearch_http::xpack_usage(&node.engine);
    let mut out=crate::owned_json::OwnedJson::parse("{}")?;
    out.transform(|v,m|{
        crate::telemetry_owned::plan_merge(v,&usage,m)?;
        for(section,key)in NODE_COUNTERS{v[*section][*key]=crate::telemetry_owned::copy_json(&usage[*section][*key],m)?;}
        let(docs,bytes)=mode_primary_stats(node,"logsdb")?;
        v["logsdb"]=json!({"num_docs":docs,"size_in_bytes":bytes});
        let(docs,_)=mode_primary_stats(node,"vectordb_document")?;
        v["vectordb_document"]=json!({"num_docs":docs});Ok(())
    })?;Ok(out)
}

/// This node's CCS search counters, over every engine that served a search.
fn local_ccs_stats(node: &ClusterNode) -> Result<crate::telemetry_owned::Typed<elasticsearch_engine::ccs_stats::CcsStats>,elasticsearch_engine::error::EsError> {
    let mut stats=crate::telemetry_owned::Typed::copy(&elasticsearch_engine::ccs_stats::CcsStats::default())?;
    node.try_visit_shard_engines(|_,engine|{let next=crate::telemetry_owned::Typed::copy(&*engine.ccs_stats.lock())?;stats.merge(&next,|a,b|a.merge(b))})?;
    Ok(stats)
}

/// CCS search telemetry merged over every node (ES sums each node's
/// `CCSTelemetrySnapshot` into `_cluster/stats` `ccs._search`).
pub async fn cluster_ccs_stats(node: &Arc<ClusterNode>) -> Result<crate::owned_json::OwnedJson,elasticsearch_engine::error::EsError> {
    let mut stats = local_ccs_stats(node)?;
    let state = node.state();
    for id in state.nodes.keys().filter(|id| **id != node.local.id) {
        let Some(addr) = node.address_of(id) else { continue };
        match node.transport.send_request::<_, SharedText>(addr, ACTION_CCS_TELEMETRY, &(), Duration::from_secs(10)).await {
            Ok(raw)=>match crate::telemetry_owned::Typed::parse(&raw) {
                Ok(remote)=>stats.merge(&remote,|a,b|a.merge(b))?,
                Err(error) if error.status==429=>return Err(error),
                Err(_)=>{},
            },
            Err(error)=>{let error=error.to_remote_error();if error.status==429{return Err(elasticsearch_engine::error::EsError::try_owned(error.status,&error.error_type,&error.reason).unwrap_or_else(|e|e));}}
        }
    }
    stats.render(|value|value.json())
}

/// Search usage belongs to the node that parsed the request. Cluster stats
/// must include every node, including coordinators with no local shard copies.
pub async fn cluster_search_usage(node: &Arc<ClusterNode>, state: &ClusterState) -> Result<(crate::owned_json::OwnedJson,crate::owned_json::OwnedJson),elasticsearch_engine::error::EsError> {
    let mut usage = crate::telemetry_owned::owned_json(&elasticsearch_http::search_usage_json())?;
    let mut failures = crate::owned_json::OwnedJson::parse("[]")?;
    for id in state.nodes.keys().filter(|id| **id != node.local.id) {
        let result = if let Some(addr) = node.address_of(id) {
            node.transport.send_request::<_, SharedText>(addr, ACTION_SEARCH_TELEMETRY, &(), Duration::from_secs(10)).await
                .map_err(|error|{let error=error.to_remote_error();elasticsearch_engine::error::EsError::try_owned(error.status,&error.error_type,&error.reason).unwrap_or_else(|refusal|refusal)})
                .and_then(|raw| crate::owned_json::OwnedJson::parse(&raw))
        } else { Err(elasticsearch_engine::error::EsError::internal("node transport address unavailable")) };
        match result {
            Ok(remote) if remote.is_object() => usage.transform(|value,memory|{crate::telemetry_owned::plan_merge(value,&remote,memory)?;merge_search_counters(value,&remote);Ok(())})?,
            Err(error) if error.status==429=>return Err(error),
            result => {
                #[derive(Serialize)]struct Failure<'a>{node_id:&'a str,reason:Reason<'a>}
                #[derive(Serialize)]struct Reason<'a>{#[serde(rename="type")]kind:&'a str,reason:&'a str}
                let reason=match result.err(){Some(error)=>SharedText::try_format(format_args!("{error}"))?,None=>SharedText::try_new("invalid search telemetry response")?};
                let text=SharedText::try_json(&Failure{node_id:id,reason:Reason{kind:"failed_node_exception",reason:&reason}})?;
                let parsed=crate::owned_json::OwnedJson::parse(&text)?;
                failures.transform(|value,memory|{crate::telemetry_owned::plan_merge(value,&parsed,memory)?;value.as_array_mut().unwrap().push(crate::telemetry_owned::copy_json(&parsed,memory)?);Ok(())})?;
            },
        }
    }
    Ok((usage, failures))
}

fn merge_search_counters(target: &mut Value, source: &Value) {
    if let Some(fields) = source.as_object() {
        if !target.is_object() { *target = json!({}); }
        for (key, value) in fields {
            merge_search_counters(target.as_object_mut().unwrap().entry(key.clone()).or_insert(Value::Null), value);
        }
    } else if let Some(count) = source.as_u64() {
        *target = json!(target.as_u64().unwrap_or(0).saturating_add(count));
    }
}

#[cfg(test)]
mod search_telemetry_tests {
    use super::*;
    #[test]
    fn merges_nested_usage_from_nodes_without_losing_disjoint_fields() {
        let mut local = json!({"total":2,"queries":{"match":2},"extended":{}});
        merge_search_counters(&mut local, &json!({"total":3,"queries":{"match":1,"term":2},
            "extended":{"retrievers":{"text_similarity_reranker":{"chunk_rescorer":1}}}}));
        merge_search_counters(&mut local, &json!({"total":1,
            "extended":{"retrievers":{"text_similarity_reranker":{"chunk_rescorer":1}}}}));
        assert_eq!(local["total"],6);
        assert_eq!(local["queries"]["match"],3);
        assert_eq!(local["queries"]["term"],2);
        assert_eq!(local["extended"]["retrievers"]["text_similarity_reranker"]["chunk_rescorer"],2);
    }
}

/// A synonyms set update's `reload_analyzers_details` over every node: each
/// node reloads the analyzers of its own copies of indices using `set`
/// (ES broadcasts `ReloadAnalyzersRequest` to every shard copy).
pub async fn cluster_synonyms_reload(node: &Arc<ClusterNode>, set: &str) -> Result<crate::owned_json::OwnedJson,elasticsearch_engine::error::EsError> {
    let mut merged = crate::owned_json::OwnedJson::parse(r#"{"_shards":{"total":0,"successful":0,"failed":0},"reload_details":[]}"#)?;
    let state = node.state();
    for id in state.nodes.keys() {
        let raw = if *id == node.local.id {
            let (e, s) = (node.engine.clone(), SharedText::try_new(set)?);
            match tokio::task::spawn_blocking(move || elasticsearch_engine::search_ext::reload_dependents(&e, &s)).await {
                Ok(Ok(v)) => SharedText::try_json(&v)?,
                Ok(Err(e)) if e.status==429=>return Err(e),
                _ => continue,
            }
        } else {
            let Some(addr) = node.address_of(id) else { continue };
            match node.transport.send_request::<_, SharedText>(addr, ACTION_SYNONYMS_RELOAD, &set, Duration::from_secs(30)).await {
                Ok(raw) => raw,
                Err(error) => {let error=error.to_remote_error();if error.status==429{return Err(elasticsearch_engine::error::EsError::try_owned(error.status,&error.error_type,&error.reason).unwrap_or_else(|e|e));}continue},
            }
        };
        match crate::owned_json::OwnedJson::parse(&raw) {
            Ok(v)=>merged.transform(|value,memory|{crate::telemetry_owned::plan_merge(value,&v,memory)?;memory.grow(id.len().checked_mul(v["reload_details"].as_array().map_or(0,Vec::len)).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","synonyms identity overflow"))? as u64,"synonyms node identity")?;merge_reload_details(value,&v,id);Ok(())})?,
            Err(error) if error.status==429=>return Err(error),
            Err(_)=>{},
        }
    }
    Ok(merged)
}

/// Add one node's reload details (`node_id` names its copies) into `into`.
fn merge_reload_details(into: &mut Value, from: &Value, node_id: &str) {
    for k in ["total", "successful", "failed"] {
        let sum = into["_shards"][k].as_u64().unwrap_or(0) + from["_shards"][k].as_u64().unwrap_or(0);
        into["_shards"][k] = json!(sum);
    }
    if let Some(f) = from["_shards"]["failures"].as_array().filter(|f| !f.is_empty()) {
        let mut all = into["_shards"]["failures"].as_array().cloned().unwrap_or_default();
        all.extend(f.iter().cloned());
        into["_shards"]["failures"] = Value::Array(all);
    }
    let mut details = into["reload_details"].as_array().cloned().unwrap_or_default();
    for d in from["reload_details"].as_array().into_iter().flatten() {
        match details.iter_mut().find(|x| x["index"] == d["index"]) {
            Some(x) => {
                let mut ids = x["reloaded_node_ids"].as_array().cloned().unwrap_or_default();
                if !ids.iter().any(|i| i == node_id) {
                    ids.push(json!(node_id));
                }
                x["reloaded_node_ids"] = Value::Array(ids);
            }
            None => {
                let mut d = d.clone();
                d["reloaded_node_ids"] = json!([node_id]);
                details.push(d);
            }
        }
    }
    into["reload_details"] = Value::Array(details);
}

/// Documents and store bytes of the primaries of `index.mode` `mode`
/// indices this node holds (ES `IndexModeStatsActionType`: every data node
/// reports its started primaries, the usage action sums the nodes).
fn mode_primary_stats(node: &ClusterNode, mode: &str) -> Result<(u64, u64),elasticsearch_engine::error::EsError> {
    let (mut docs, mut bytes) = (0u64, 0u64);
    for local in node.shards.admitted_all()?.iter() {
        let (index, shard, matching) = {
            let g = local.lock();
            (SharedText::try_new(&g.index)?, g.shard, g.svc.setting("index.mode").as_deref() == Some(mode))
        };
        if !matching {
            continue;
        }
        if let Some(p) = node.shards.primary(&index, shard) {
            let copy = p.copy();
            docs += copy.doc_count().0;
            bytes += copy.shard.store_size();
        }
    }
    Ok((docs, bytes))
}

/// Add every number under `from` into `into`.
fn add_counters(into: &mut Value, from: &Value) {
    match (into, from) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                match a.get_mut(k) {
                    Some(slot) => add_counters(slot, v),
                    None => { a.insert(k.clone(), v.clone()); }
                }
            }
        }
        (slot @ Value::Number(_), Value::Number(n)) => {
            if let (Some(x), Some(y)) = (slot.as_u64(), n.as_u64()) {
                *slot = json!(x + y);
            }
        }
        _ => {}
    }
}

/// `GET /_xpack/usage`: the local feature usage, with the per-node usage
/// counters summed over every node (`AnalyticsUsageTransportAction`,
/// `EsqlUsageTransportAction`).
/// `_xpack/usage` sections describing resources the master's engine owns.
const MASTER_OWNED_SECTIONS: &[&str] = &["ml", "rollup"];

pub async fn xpack_usage(node:&Arc<ClusterNode>,req:&RestReq)->RestResp{
    let mut resp=crate::front::local_engine(node,req).await;
    if !resp.is_success(){return resp}
    let mut body=match crate::owned_json::OwnedJson::parse(std::str::from_utf8(&resp.body).unwrap_or("null")){Ok(value)=>value,Err(error)=>return RestResp::from_es_error(&error)};
    let mut totals=match node_usage_counters(node){Ok(value)=>value,Err(error)=>return RestResp::from_es_error(&error)};
    let state=node.state();
    for id in state.nodes.keys().filter(|id|**id!=node.local.id){
        let Some(addr)=node.address_of(id)else{continue};
        match node.transport.send_request::<_,SharedText>(addr,ACTION_ANALYTICS_USAGE,&(),Duration::from_secs(10)).await{
            Ok(text)=>match crate::owned_json::OwnedJson::parse(&text){
                Ok(counters)=>if let Err(error)=totals.transform(|v,m|{crate::telemetry_owned::plan_merge(v,&counters,m)?;add_counters(v,&counters);Ok(())}){return RestResp::from_es_error(&error)},
                Err(error)if error.status==429=>return RestResp::from_es_error(&error),Err(_)=>{},
            },
            Err(error)=>{let error=error.to_remote_error();if error.status==429{return RestResp::from_es_error(&elasticsearch_engine::error::EsError::try_owned(error.status,&error.error_type,&error.reason).unwrap_or_else(|e|e));}}
        }
    }
    let updated=body.transform(|v,m|{
        // Existing sections may lack their fixed counter key: cover insertion nodes/keys before indexing.
        let unit=2*(12*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>());
        m.grow((9*(unit+32))as u64,"usage counter projection keys")?;
        for(section,key)in NODE_COUNTERS{if v[*section].is_object()&&!totals[*section][*key].is_null(){v[*section][*key]=crate::telemetry_owned::copy_json(&totals[*section][*key],m)?;}}
        for(section,keys)in[("logsdb",&["num_docs","size_in_bytes"][..]),("vectordb_document",&["num_docs"][..])]{if v[section].is_object(){for key in keys{v[section][*key]=crate::telemetry_owned::copy_json(&totals[section][*key],m)?;}}}Ok(())
    });
    if let Err(error)=updated{return RestResp::from_es_error(&error)}
    if body["gpu_vector_indexing"].is_object(){
        let gpu=match cluster_gpu_usage(node,&state,body["gpu_vector_indexing"]["available"].as_bool().unwrap_or(false)).await{Ok(v)=>v,Err(error)=>return RestResp::from_es_error(&error)};
        if let Err(error)=body.transform(|v,m|{v["gpu_vector_indexing"]=crate::telemetry_owned::copy_json(&gpu,m)?;Ok(())}){return RestResp::from_es_error(&error)}
    }
    if !node.is_master()&&req.hops==0&&MASTER_OWNED_SECTIONS.iter().any(|key|body[*key].is_object()){
        let master=crate::rest::forward_to_master(node,req).await;
        if master.is_success(){
            match crate::owned_json::OwnedJson::parse(std::str::from_utf8(&master.body).unwrap_or("null")){
                Ok(other)=>if let Err(error)=body.transform(|v,m|{for key in MASTER_OWNED_SECTIONS{if v[*key].is_object()&&other[*key].is_object(){v[*key]=crate::telemetry_owned::copy_json(&other[*key],m)?;}}Ok(())}){return RestResp::from_es_error(&error)},
                Err(error)if error.status==429=>return RestResp::from_es_error(&error),Err(_)=>{},
            }
        }
    }
    resp.replace_json(&body);resp
}

#[derive(Serialize, Deserialize)]
struct DiskWire {
    #[serde(deserialize_with="crate::control_memory::decode")]
    path: crate::front::SharedText,
    total_bytes: u64,
    free_bytes: u64,
}

pub async fn fetch_disk(node: &ClusterNode, node_id: &str) -> Result<Option<DiskUsage>, String> {
    let Some(addr) = node.address_of(node_id) else { return Ok(None) };
    let d: DiskWire = node.transport.send_request(addr, ACTION_DISK, &(), Duration::from_secs(10)).await.map_err(|e| e.to_string())?;
    struct PathOwner { text:crate::front::SharedText,_memory:elasticsearch_engine::memory::Reservation }
    impl elasticsearch_cluster::allocation::DiskPathSource for PathOwner{fn as_str(&self)->&str{&self.text}}
    let memory=elasticsearch_engine::memory::breakers().reserve_native((std::mem::size_of::<PathOwner>()+2*std::mem::size_of::<usize>())as u64,"disk path owner").map_err(|e|e.to_string())?;
    let path=elasticsearch_cluster::allocation::DiskPath::from_owner(Arc::new(PathOwner{text:d.path,_memory:memory.into_result_owner()}));
    Ok(Some(DiskUsage { path, total_bytes: d.total_bytes, free_bytes: d.free_bytes }))
}

/// This node's entry for an engine `_nodes` URI, identity patched.
pub(crate) async fn local_entry(node:&ClusterNode,uri:&str)->Result<crate::owned_json::OwnedJson,RestResp>{
    let req=RestReq::try_new("GET",uri,&[],&[],0,Default::default(),None).map_err(|e|RestResp::from_es_error(&e))?;
    let response=crate::front::local_node_monitor(node,&req).await;
    if !response.is_success(){return Err(response)}
    let parsed=crate::owned_json::OwnedJson::parse(std::str::from_utf8(&response.body).unwrap_or("null")).map_err(|e|RestResp::from_es_error(&e))?;
    entry_from_owned(node,uri,parsed).await
}

pub(crate) async fn entry_from(node:&ClusterNode,uri:&str,value:Value)->Result<crate::owned_json::OwnedJson,RestResp>{
    let parsed=crate::telemetry_owned::owned_json(&value).map_err(|e|RestResp::from_es_error(&e))?;
    entry_from_owned(node,uri,parsed).await
}
async fn entry_from_owned(node:&ClusterNode,uri:&str,parsed:crate::owned_json::OwnedJson)->Result<crate::owned_json::OwnedJson,RestResp>{
    let build=(||->Result<_,elasticsearch_engine::error::EsError>{
        let mut memory=elasticsearch_engine::memory::breakers().reserve_request(0,"node telemetry projection")?;
        // The node's entry moves out of the parsed answer (charged here as
        // it leaves the parse's reservation) instead of being deep-copied.
        let mut entry=Value::Null;
        parsed.consume(|mut value,_|{
            if let Some(first)=value.get_mut("nodes").and_then(Value::as_object_mut).and_then(|m|m.values_mut().next()) { entry=std::mem::take(first); }
            Ok(())
        })?;
        memory.grow(elasticsearch_engine::memory::json_bytes(&entry),"node telemetry projection")?;
        if !entry.is_object(){entry=Value::Object(Map::new());}
        let identity=SharedText::try_json(&node.local)?;
        let identity=crate::owned_json::OwnedJson::parse(&identity)?;
        crate::telemetry_owned::plan_merge(&entry,&identity,&mut memory)?;
        let settings=SharedText::try_json(&node.config.settings)?;
        let settings=crate::owned_json::OwnedJson::parse(&settings)?;
        crate::telemetry_owned::plan_merge(&entry,&settings,&mut memory)?;
        let segments=node.config.settings.keys().try_fold(0usize,|sum,key|sum.checked_add(key.bytes().filter(|b|*b==b'.').count()+1)).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","settings projection overflow"))?;
        memory.grow(segments.checked_mul(2*(12*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>())).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","settings projection overflow"))? as u64,"nested node settings maps")?;
        Ok((memory,entry))
    })();
    let (mut memory,mut entry)=build.map_err(|e|RestResp::from_es_error(&e))?;
    if entry.get("adaptive_selection").is_some(){
        // Snapshot serialization is supplied by the existing adaptive counter
        // owner; retained projection receives its own structural admission.
        let adaptive=node.adaptive_selection.snapshot(&node.state());
        crate::telemetry_owned::plan_merge(&entry,&adaptive,&mut memory).map_err(|e|RestResp::from_es_error(&e))?;
        entry["adaptive_selection"]=crate::telemetry_owned::copy_json(&adaptive,&mut memory).map_err(|e|RestResp::from_es_error(&e))?;
    }
    if entry["indices"].is_object()&&uri.contains("/stats"){add_shard_engine_indices(node,uri,&mut entry["indices"],&mut memory).await?;}
    let flat=uri.split_once('?').map(|(_,q)|q.split('&').any(|kv|kv=="flat_settings"||kv=="flat_settings=true")).unwrap_or(false);
    patch_identity(&mut entry,&node.local,node,flat);
    Ok(crate::owned_json::OwnedJson::from_reserved(entry,memory))
}

/// Add the `indices` node stats of every shard engine (shard copies
/// 1.. of multi-shard indices) to engine 0's.
async fn add_shard_engine_indices(node: &ClusterNode, uri: &str, indices: &mut Value, memory:&mut elasticsearch_engine::memory::Reservation) -> Result<(),RestResp> {
    let engines=node.shard_engines.admitted_ids().map_err(|e|RestResp::from_es_error(&e))?;
    let (path,query)=uri.split_once('?').map(|(p,q)|(p,Some(q))).unwrap_or((uri,None));
    let rest=path.split_once("/stats").map(|(_,r)|r.trim_start_matches('/')).unwrap_or("");
    let metrics=rest.split_once('/').map(|(_,m)|m).filter(|m|!m.is_empty());
    let sub_uri=SharedText::try_format(format_args!("/_nodes/_local/stats/indices{}{}{}{}",if metrics.is_some(){"/"}else{""},metrics.unwrap_or(""),if query.is_some(){"?"}else{""},query.unwrap_or(""))).map_err(|e|RestResp::from_es_error(&e))?;
    if engines.is_empty(){return Ok(())}
    // Parsed as the engine route would see the sub-request (path, decoded
    // query parameters).
    let sub=RestReq::try_new("GET",&sub_uri,&[],&[],0,Default::default(),None).map_err(|e|RestResp::from_es_error(&e))?;
    let (path,params)=(sub.path().to_string(),sub.params());
    // Each shard engine's `indices` section in process: engine 0's route
    // already validated this request, so a full in-process HTTP round trip
    // per engine (admission, security, usage counting, an executor hop,
    // serialization and a re-parse) is not repeated. One answer is held at
    // a time, as before.
    // The engines compute concurrently (each a blocking job of its own;
    // one after another they added up to most of a node stats request on
    // a node of many shard engines), merged in engine order.
    let mut jobs=Vec::with_capacity(engines.len());
    for &k in engines.iter(){
        let Some(engine)=node.engine_for(k) else {
            return Err(RestResp::error(503,"shard_not_found_exception",&format!("no engine for shard [{k}] on this node")));
        };
        let (path,params)=(path.clone(),params.clone());
        let job=elasticsearch_engine::memory::inherit_request_policy(move || elasticsearch_http::nodes_stats_in_process(&engine,&path,params));
        jobs.push((k,tokio::task::spawn_blocking(job)));
    }
    let (ks,handles):(Vec<u32>,Vec<_>)=jobs.into_iter().unzip();
    let answers=futures::future::join_all(handles).await;
    for (k,answer) in ks.into_iter().zip(answers){
        let v=answer
            .map_err(|e|RestResp::error(500,"exception",&format!("shard engine [{k}] node stats failed: {e}")))?
            .map_err(|e|RestResp::from_es_error(&e))?;
        let Some(other)=v["nodes"].as_object().and_then(|m|m.values().next()).map(|e|&e["indices"])else{continue};
        crate::telemetry_owned::plan_merge(indices,other,memory).map_err(|e|RestResp::from_es_error(&e))?;
        merge_node_indices(indices,other,k);
    }
    Ok(())
}

/// Add shard engine `k`'s node `indices` stats into `dst`: the summed
/// sections, `level=indices` per index, `level=shards` entries renamed to
/// shard `k`. `mappings` are per index (engine 0 holds every index).
fn merge_node_indices(dst: &mut Value, src: &Value, k: u32) {
    let (Some(d), Some(s)) = (dst.as_object_mut(), src.as_object()) else { return };
    for (key, v) in s {
        match key.as_str() {
            "mappings" => {}
            "indices" => {
                let Some(per) = v.as_object() else { continue };
                let target = d.entry("indices").or_insert_with(|| json!({}));
                for (name, sec) in per {
                    match target.get_mut(name) {
                        Some(cur) => merge_node_indices(cur, sec, k),
                        None => {
                            target[name.as_str()] = sec.clone();
                        }
                    }
                }
            }
            "shards" => {
                let Some(per) = v.as_object() else { continue };
                let target = d.entry("shards").or_insert_with(|| json!({}));
                for (name, list) in per {
                    let renamed: Vec<Value> = list
                        .as_array()
                        .map(|a| a.iter().map(|e| match e.as_object() {
                            Some(o) if o.len() == 1 => json!({ k.to_string(): o.values().next().cloned().unwrap_or(Value::Null) }),
                            _ => e.clone(),
                        }).collect())
                        .unwrap_or_default();
                    match target.get_mut(name).and_then(Value::as_array_mut) {
                        Some(cur) => cur.extend(renamed),
                        None => {
                            target[name.as_str()] = Value::Array(renamed);
                        }
                    }
                }
            }
            _ => match d.get_mut(key) {
                Some(cur) => crate::stats_shards::add_json(cur, v),
                None => {
                    d.insert(key.clone(), v.clone());
                }
            },
        }
    }
}

fn patch_identity(entry: &mut Value, n: &DiscoveryNode, node: &ClusterNode, flat: bool) {
    let Some(o) = entry.as_object_mut() else { return };
    o.insert("name".into(), json!(n.name));
    o.insert("transport_address".into(), json!(n.transport_address));
    o.insert("host".into(), json!(n.host));
    o.insert("ip".into(), json!(n.ip));
    o.insert("roles".into(), json!(n.roles.iter().map(|r| r.name()).collect::<Vec<_>>()));
    o.insert("attributes".into(), json!(n.attributes));
    if o.contains_key("settings") {
        // ES's `SettingsFilter`: no secure (keystore) or filtered setting.
        let filter = elasticsearch_engine::settings_filter::SettingsFilter::for_settings(node.config.settings.iter());
        let filtered: std::collections::BTreeMap<String, Value> = node.config.settings.iter().filter(|(k, _)| !filter.hides(k)).map(|(k, v)| (k.clone(), v.clone())).collect();
        let settings = if flat { Value::Object(filtered.into_iter().collect()) } else { elasticsearch_engine::settings::unflatten(&filtered) };
        o.insert("settings".into(), settings);
    }
    if let Some(http) = o.get_mut("http").and_then(Value::as_object_mut) {
        if http.contains_key("publish_address") {
            http.insert("publish_address".into(), json!(n.http_address));
            http.insert("bound_address".into(), json!([n.http_address]));
        }
    }
    if let Some(t) = o.get_mut("transport").and_then(Value::as_object_mut) {
        if t.contains_key("publish_address") {
            t.insert("publish_address".into(), json!(n.transport_address));
            t.insert("bound_address".into(), json!([n.transport_address]));
        }
    }
}

/// ES node selection (`_all`, `_local`, `_master`, ids, names, wildcard
/// on name/host/ip, `role:true|false`, `attr:value`), applied in order.
pub fn resolve_nodes(state: &ClusterState, local_id: &str, spec: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if spec.is_empty() || spec == "_all" {
        return state.nodes.keys().cloned().collect();
    }
    let wild = |p: &str, s: &str| -> bool {
        match p.split_once('*') {
            None => p == s,
            Some((a, b)) => s.starts_with(a) && s.ends_with(b) && s.len() >= a.len() + b.len(),
        }
    };
    for part in spec.split(',') {
        let part = part.trim();
        match part {
            "_all" | "*" => out.extend(state.nodes.keys().cloned()),
            "_local" => {
                out.insert(local_id.to_string());
            }
            "_master" => out.extend(state.master_node_id.clone()),
            _ => {
                if let Some((k, v)) = part.split_once(':') {
                    let flag = v == "true";
                    let role = match k {
                        "master" => Some(Role::Master),
                        "data" => Some(Role::Data),
                        "ingest" => Some(Role::Ingest),
                        "ml" => Some(Role::Ml),
                        "voting_only" => Some(Role::VotingOnly),
                        "remote_cluster_client" => Some(Role::RemoteClusterClient),
                        "transform" => Some(Role::Transform),
                        other => Role::parse(other),
                    };
                    if let (Some(r), true) = (role, v == "true" || v == "false") {
                        for n in state.nodes.values() {
                            let has = if r == Role::Data { n.is_data() } else { n.roles.contains(&r) };
                            if has == flag {
                                out.insert(n.id.clone());
                            } else {
                                out.remove(&n.id);
                            }
                        }
                    } else {
                        for n in state.nodes.values() {
                            if n.attributes.get(k).map(|x| wild(v, x)).unwrap_or(false) {
                                out.insert(n.id.clone());
                            }
                        }
                    }
                    continue;
                }
                for n in state.nodes.values() {
                    if n.id == part || wild(part, &n.name) || wild(part, &n.host) || wild(part, &n.ip) || n.transport_address == part {
                        out.insert(n.id.clone());
                    }
                }
            }
        }
    }
    out
}

/// Split `/_nodes/...` into (node spec, kind, rest-of-path for the engine).
fn parse_path(segs: &[String]) -> (String, &'static str, String) {
    let rest: Vec<&str> = segs.iter().skip(1).map(String::as_str).collect();
    let kinds = ["stats", "usage", "hot_threads", "hotthreads"];
    let is_metric_list = |s: &str, list: &[&str]| s.split(',').all(|m| list.contains(&m));
    match rest.as_slice() {
        [] => (String::new(), "info", String::new()),
        [k, tail @ ..] if kinds.contains(k) => (String::new(), kind_name(k), tail.join("/")),
        [x] if is_metric_list(x, INFO_METRICS) => (String::new(), "info", x.to_string()),
        [n, k, tail @ ..] if kinds.contains(k) => (n.to_string(), kind_name(k), tail.join("/")),
        [n, tail @ ..] => (n.to_string(), "info", tail.join("/")),
    }
}

fn kind_name(k: &str) -> &'static str {
    match k {
        "stats" => "stats",
        "usage" => "usage",
        _ => "hot_threads",
    }
}

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let (spec, kind, tail) = parse_path(&segs);
    if kind == "stats" {
        if let Some(m) = tail.split('/').next().filter(|m| !m.is_empty()) {
            let bad: Vec<String> = m.split(',').filter(|x| !STATS_METRICS.contains(x)).map(str::to_string).collect();
            if !bad.is_empty() {
                let candidates: Vec<&str> = STATS_METRICS.iter().copied().filter(|c| !c.starts_with('_')).collect();
                return RestResp::error(400, "illegal_argument_exception", &super::util::unrecognized(req.path(), &bad, &candidates, "metric"));
            }
        }
    }
    let state = node.state();
    let ids = resolve_nodes(&state, &node.local.id, &spec);
    let mut engine_uri = match kind {
        "stats" => "/_nodes/_local/stats".to_string(),
        "usage" => "/_nodes/_local/usage".to_string(),
        "hot_threads" => "/_nodes/_local/hot_threads".to_string(),
        _ => "/_nodes/_local".to_string(),
    };
    if !tail.is_empty() {
        engine_uri.push('/');
        engine_uri.push_str(&tail);
    }
    if !req.query().is_empty() {
        engine_uri.push('?');
        engine_uri.push_str(req.query());
    }
    // Response shaping happens once, on the merged response.
    let engine_uri = super::util::strip_params(&engine_uri, &["filter_path", "pretty"]);
    // Validate the request shape on the local engine first (unknown
    // metrics etc. render ES's exact errors).
    let started=std::time::Instant::now();
    let probe_req=match RestReq::try_new("GET",&engine_uri,&req.headers,&[],0,Default::default(),None) { Ok(r)=>r,Err(e)=>return RestResp::json(e.status,&e.to_json()) };
    let probe=crate::front::local_engine(node,&probe_req).await;
    let probe_us=started.elapsed().as_micros();
    if !probe.is_success(){return probe}
    if kind == "hot_threads" {
        let mut out=SharedText::default();
        for id in &ids {
            let text=if *id==node.local.id {
                match std::str::from_utf8(&probe.body).map_err(|_|RestResp::error(500,"transport_serialization_exception","invalid hot threads UTF-8")).and_then(|v|SharedText::try_new(v).map_err(|e|RestResp::from_es_error(&e))){Ok(v)=>v,Err(e)=>return e}
            } else {match fetch_text(node,id,&engine_uri).await{Ok(v)=>v,Err(e)=>return e}};
            out=match SharedText::try_format(format_args!("{out}{text}")){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
        }
        let mut response=RestResp::text(200,String::new());response.replace_body(out.as_bytes());return response;
    }
    let mut memory=match elasticsearch_engine::memory::breakers().reserve_request(0,"node response aggregation"){Ok(m)=>m,Err(e)=>return RestResp::from_es_error(&e)};
    let mut nodes=Map::new();
    let mut failures=Vec::new();
    if let Err(e)=memory.grow(match ids.len().checked_mul(std::mem::size_of::<Value>()){Some(n)=>n as u64,None=>return RestResp::error(429,"circuit_breaking_exception","node failure slots overflow")},"node failure slots"){return RestResp::from_es_error(&e)}
    if let Err(e)=failures.try_reserve_exact(ids.len()){return RestResp::from_es_error(&elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception",e.to_string()))}
    let mut probe_json=match std::str::from_utf8(&probe.body){
        Ok(text)=>match crate::owned_json::OwnedJson::parse(text){Ok(value)=>Some(value),Err(error)if error.status==429=>return RestResp::from_es_error(&error),Err(_)=>None},
        Err(_)=>None,
    };
    // Every node's entry is gathered at once (the local one in process, the
    // others over the transport), then merged in node order: one after
    // another, each remote node's stats waited for the local ones and for
    // the nodes before it.
    enum Fetched { Local(Result<crate::owned_json::OwnedJson,RestResp>), Remote(Result<SharedText,elasticsearch_transport::TransportError>), Skip }
    let mut local_probe=probe_json.take();
    let fetches=ids.iter().map(|id| {
        let local=(*id==node.local.id).then(|| local_probe.take());
        let (state,engine_uri)=(&state,&engine_uri);
        async move {
            if let Some(probe)=local {
                return Fetched::Local(match probe{Some(v)=>entry_from_owned(node,engine_uri,v).await,None=>local_entry(node,engine_uri).await});
            }
            let Some(addr)=state.nodes.get(id).and_then(|n|n.transport_address.parse::<SocketAddr>().ok())else{return Fetched::Skip};
            let uri=match SharedText::try_new(engine_uri){Ok(v)=>v,Err(e)=>return Fetched::Local(Err(RestResp::from_es_error(&e)))};
            Fetched::Remote(node.transport.send_request::<_,SharedText>(addr,ACTION_LOCAL,&LocalReq{uri},Duration::from_secs(30)).await)
        }
    }).collect::<Vec<_>>();
    let parsed_us=started.elapsed().as_micros();
    let fetched=futures::future::join_all(fetches).await;
    let entries_us=started.elapsed().as_micros();
    for (id,got) in ids.iter().zip(fetched) {
        let entry=match got {
            Fetched::Skip=>continue,
            Fetched::Local(entry)=>entry,
            Fetched::Remote(sent)=>match sent {
                Ok(text)=>crate::owned_json::OwnedJson::parse(&text).map_err(|e|RestResp::from_es_error(&e)),
                Err(error)=>{
                    let error=error.to_remote_error();
                    #[derive(Serialize)]struct Failure<'a,T:Serialize>{#[serde(rename="type")]kind:&'static str,reason:SharedText,node_id:&'a str,caused_by:T}
                    let result=(||->Result<Value,elasticsearch_engine::error::EsError>{
                        let text=SharedText::try_json(&Failure{kind:"failed_node_exception",reason:SharedText::try_format(format_args!("Failed node [{id}]"))?,node_id:id,caused_by:RemoteJson(&error)})?;
                        let parsed=crate::owned_json::OwnedJson::parse(&text)?;
                        crate::telemetry_owned::copy_json(&parsed,&mut memory)
                    })();
                    match result{Ok(v)=>failures.push(v),Err(e)=>return RestResp::from_es_error(&e)};
                    continue;
                }
            }
        };
        let entry=match entry{Ok(v)=>v,Err(r)=>return r};
        let result=(||->Result<(),elasticsearch_engine::error::EsError>{
            memory.grow((id.len()+2*(12*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>()))as u64,"node result key and map")?;
            let mut key=String::new();key.try_reserve_exact(id.len()).map_err(|e|elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception",e.to_string()))?;key.push_str(id);
            // Moved, not deep-copied: the entry's tree is charged here as it
            // leaves its own reservation.
            let mut moved=Value::Null;
            entry.consume(|value,_|{moved=value;Ok(())})?;
            memory.grow(elasticsearch_engine::memory::json_bytes(&moved),"node result entry")?;
            nodes.insert(key,moved);Ok(())
        })();
        if let Err(e)=result{return RestResp::from_es_error(&e)}
    }
    #[derive(Serialize)]struct Header<'a>{total:usize,successful:usize,failed:usize,#[serde(skip_serializing_if="Vec::is_empty")]failures:&'a Vec<Value>}
    #[derive(Serialize)]struct Output<'a>{#[serde(rename="_nodes")]header:Header<'a>,cluster_name:&'a str,nodes:&'a Map<String,Value>}
    let merged_us=started.elapsed().as_micros();
    let out=RestResp::json_serialized(200,&Output{header:Header{total:ids.len(),successful:nodes.len(),failed:failures.len(),failures:&failures},cluster_name:&state.cluster_name,nodes:&nodes});
    let total_us=started.elapsed().as_micros();
    // The local engine's whole node stats (`probe`, an in-process route
    // call), its parse, every node's entry (shard engines' indices
    // sections, the remote nodes), their copy into the response and the
    // final render.
    crate::front::stats_phases::record("node_stats", vec![("probe",probe_us),("parse",parsed_us-probe_us),("entries",entries_us-parsed_us),("merge",merged_us-entries_us),("render",total_us-merged_us),("total",total_us)]);
    out
}

async fn fetch_text(node: &ClusterNode, id: &str, uri: &str) -> Result<SharedText,RestResp> {
    let Some(addr) = node.address_of(id) else { return Ok(SharedText::default()) };
    let req=RestReq::try_new("GET",uri,&[],&[],1,Default::default(),None).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
    match node.transport.send_request::<_, crate::front::RestResp>(addr, crate::front::ACTION_FORWARD, &req, Duration::from_secs(30)).await {
        Ok(r) if r.is_success() => std::str::from_utf8(&r.body).map_err(|_|RestResp::error(500,"transport_serialization_exception","invalid hot threads UTF-8")).and_then(|v|SharedText::try_new(v).map_err(|e|RestResp::from_es_error(&e))),
        Ok(r)=>Err(r),
        Err(e)=>{let e=e.to_remote_error();Err(RestResp::error(e.status,&e.error_type,&e.reason))},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_counters_add_up() {
        let mut total = json!({"analytics": {"stats": {"boxplot_usage": 1, "rate_usage": 0}}, "esql": {"features": {"limit": 2}, "queries": {"rest": {"total": 3}}}});
        add_counters(&mut total, &json!({"analytics": {"stats": {"boxplot_usage": 4}}, "esql": {"features": {"limit": 1, "view": 1}, "queries": {"rest": {"total": 2}}}}));
        assert_eq!(total, json!({"analytics": {"stats": {"boxplot_usage": 5, "rate_usage": 0}}, "esql": {"features": {"limit": 3, "view": 1}, "queries": {"rest": {"total": 5}}}}));
    }

    #[test]
    fn synonym_reload_details_merge_per_node() {
        let mut local = json!({"_shards": {"total": 0, "successful": 0, "failed": 0}, "reload_details": []});
        let remote = json!({"_shards": {"total": 1, "successful": 1, "failed": 0},
            "reload_details": [{"index": "my_index", "reloaded_analyzers": ["my_analyzer"], "reloaded_node_ids": ["x"]}]});
        merge_reload_details(&mut local, &remote, "n2");
        merge_reload_details(&mut local, &remote, "n3");
        assert_eq!(local["_shards"], json!({"total": 2, "successful": 2, "failed": 0}));
        assert_eq!(local["reload_details"], json!([{"index": "my_index", "reloaded_analyzers": ["my_analyzer"], "reloaded_node_ids": ["n2", "n3"]}]));
    }

    #[test]
    fn every_esql_counter_section_is_summed() {
        // Usage is read on any node while ES|QL runs on the master: a section
        // left out here reads as the serving node's local counts only.
        for key in ["features", "queries", "functions", "settings", "took"] {
            assert!(NODE_COUNTERS.contains(&("esql", key)), "esql.{key} not summed across nodes");
        }
    }

    #[test]
    fn gpu_nodes_come_from_published_attributes() {
        let mut state = ClusterState::empty("gpu-test");
        for (id, gpu) in [("a", true), ("b", false), ("c", true)] {
            let mut attributes = Default::default();
            crate::internal_capabilities::advertise(&mut attributes);
            advertise_gpu(&mut attributes, gpu);
            state.nodes.insert(id.into(), elasticsearch_cluster::state::DiscoveryNode {
                id: id.into(), name: id.into(), ephemeral_id: format!("{id}-1"), transport_address: String::new(),
                http_address: String::new(), host: String::new(), ip: String::new(), roles: Default::default(),
                attributes, version: String::new(),
            });
        }
        assert_eq!(gpu_supported_nodes(&state).into_iter().collect::<Vec<_>>(), vec!["a", "c"]);
        // A configured attribute in the reserved namespace is not a claim.
        let mut configured: std::collections::BTreeMap<String, String> = [(GPU_SUPPORTED_ATTRIBUTE.to_string(), "1".to_string())].into();
        crate::internal_capabilities::advertise(&mut configured);
        assert!(!configured.contains_key(GPU_SUPPORTED_ATTRIBUTE));
    }

    #[test]
    fn path_parsing() {
        let s = |p: &str| p.split('/').filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>();
        assert_eq!(parse_path(&s("/_nodes")), ("".into(), "info", "".into()));
        assert_eq!(parse_path(&s("/_nodes/stats/jvm")), ("".into(), "stats", "jvm".into()));
        assert_eq!(parse_path(&s("/_nodes/http")), ("".into(), "info", "http".into()));
        assert_eq!(parse_path(&s("/_nodes/_local/stats/os,fs")), ("_local".into(), "stats", "os,fs".into()));
        assert_eq!(parse_path(&s("/_nodes/node-1/jvm")), ("node-1".into(), "info", "jvm".into()));
        assert_eq!(parse_path(&s("/_nodes/n1")), ("n1".into(), "info", "".into()));
    }
}

fn local_gpu_stats(node:&elasticsearch_engine::node::Node)->Result<crate::telemetry_owned::Typed<elasticsearch_engine::gpu::NodeGpuStats>,elasticsearch_engine::error::EsError>{
    let bytes=elasticsearch_engine::rustic_core::vector::gpu::info().map_or(0,|info|info.name.len());
    let memory=elasticsearch_engine::memory::breakers().reserve_request(bytes as u64,"GPU snapshot name")?;
    Ok(crate::telemetry_owned::Typed::from_reserved(elasticsearch_engine::gpu::local_stats(node),memory))
}
/// Node attribute a node with a supported GPU publishes when it joins
/// (`NodeGpuStatsResponse.isGpuSupported`). In the reserved namespace, so it
/// is generated by the process (`internal_capabilities::advertise` drops any
/// configured `rustic.internal.*`), never set through `node.attr.*`.
pub(crate) const GPU_SUPPORTED_ATTRIBUTE: &str = "rustic.internal.gpu_supported";

/// Add [`GPU_SUPPORTED_ATTRIBUTE`] to this node's attributes (after
/// `internal_capabilities::advertise`) when it has a supported GPU.
pub(crate) fn advertise_gpu(attributes: &mut std::collections::BTreeMap<String, String>, supported: bool) {
    if supported { attributes.insert(GPU_SUPPORTED_ATTRIBUTE.into(), "1".into()); }
}

/// The nodes of `state` with a supported GPU: they publish
/// `GPUFeatures.VECTORS_INDEXING_USE_GPU`. Read from the published node
/// attributes, as ES reads node features from the cluster state (carried at
/// join), so every node answers the same and nothing is probed. A node from a
/// build that does not publish the attribute is left out.
pub(crate) fn gpu_supported_nodes(state:&ClusterState)->std::collections::BTreeSet<String>{
    state.nodes.iter().filter(|(_, n)| n.attributes.get(GPU_SUPPORTED_ATTRIBUTE).is_some_and(|v| v == "1")).map(|(id, _)| id.clone()).collect()
}
async fn cluster_gpu_usage(node:&Arc<ClusterNode>,state:&ClusterState,available:bool)->Result<crate::owned_json::OwnedJson,elasticsearch_engine::error::EsError>{
    use elasticsearch_engine::gpu;
    let memory=elasticsearch_engine::memory::breakers().reserve_request(state.nodes.len().checked_mul(std::mem::size_of::<crate::telemetry_owned::Typed<gpu::NodeGpuStats>>()).ok_or_else(||elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception","GPU node array overflow"))? as u64,"GPU node array")?;
    let mut stats=Vec::new();stats.try_reserve_exact(state.nodes.len()).map_err(|e|elasticsearch_engine::error::EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
    for(id,n)in &state.nodes{
        if !n.roles.iter().any(|role|role.name().starts_with("data")){continue}
        if *id==node.local.id{stats.push(local_gpu_stats(&node.engine)?);continue}
        let Some(addr)=node.address_of(id)else{continue};
        match node.transport.send_request::<_,SharedText>(addr,ACTION_GPU_STATS,&(),Duration::from_secs(10)).await{
            Ok(raw)=>match crate::telemetry_owned::Typed::<gpu::NodeGpuStats>::parse(&raw){Ok(value)=>stats.push(value),Err(error)if error.status==429=>return Err(error),Err(_)=>{}},
            Err(error)=>{let error=error.to_remote_error();if error.status==429{return Err(elasticsearch_engine::error::EsError::try_owned(error.status,&error.error_type,&error.reason).unwrap_or_else(|e|e));}}
        }
    }
    struct Nodes<'a>(&'a[crate::telemetry_owned::Typed<gpu::NodeGpuStats>]);
    impl Serialize for Nodes<'_>{fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
        use serde::ser::SerializeSeq;
        #[derive(Serialize)]struct Entry<'a>{#[serde(rename="type")]name:&'a Option<String>,memory_in_bytes:u64,enabled:bool,index_build_count:u64}
        let mut seq=s.serialize_seq(Some(self.0.iter().filter(|n|n.gpu_supported).count()))?;
        for n in self.0.iter().filter(|n|n.gpu_supported){seq.serialize_element(&Entry{name:&n.gpu_name,memory_in_bytes:n.total_gpu_memory_in_bytes,enabled:n.gpu_setting_enabled,index_build_count:n.gpu_usage_count})?;}seq.end()
    }}
    #[derive(Serialize)]struct Output<'a>{available:bool,enabled:bool,index_build_count:u64,nodes_with_gpu:usize,nodes:Nodes<'a>}
    let text=SharedText::try_json(&Output{available,enabled:stats.iter().any(|n|n.gpu_setting_enabled),index_build_count:stats.iter().map(|n|n.gpu_usage_count).sum(),nodes_with_gpu:stats.iter().filter(|n|n.gpu_supported).count(),nodes:Nodes(&stats)})?;
    let output=crate::owned_json::OwnedJson::parse(&text)?;drop(stats);drop(memory);Ok(output)
}
