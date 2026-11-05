//! `GET [/{index}]/_recovery` across the cluster: one entry per assigned
//! shard copy, from the copy's live peer-recovery state when its node has
//! one, otherwise the completed store recovery that created it
//! (`EMPTY_STORE` for new primaries, `EXISTING_STORE` for closed indices,
//! `SNAPSHOT` for a primary being restored, `PEER` from the primary's node
//! for replicas).

use super::{cat_tables, gather_shard_stats};
use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::{ClusterState, IndexState, ShardRouting, ShardState};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use crate::owned_json::OwnedJson;
use elasticsearch_engine::{error::EsError, memory::{breakers, Reservation}};

/// Moved JSON values precede their parser owners and aggregate storage owner.
pub struct RecoveryEntries {
    values: BTreeMap<String, Vec<Value>>,
    parsed: Vec<OwnedJson>,
    memory: Reservation,
}
impl<'a> IntoIterator for &'a RecoveryEntries {
    type Item=(&'a String,&'a Vec<Value>);
    type IntoIter=std::collections::btree_map::Iter<'a,String,Vec<Value>>;
    fn into_iter(self)->Self::IntoIter { self.values.iter() }
}
fn memory_error()->EsError { EsError::new(429,"circuit_breaking_exception","recovery response capacity overflow") }
fn map_storage()->usize { 2*(11*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>()) }
impl RecoveryEntries {
    pub(super) fn prepare_cat_rows(&mut self)->Result<usize,RestResp> {
        fn strings(v:&Value)->Option<usize> { match v {
            Value::String(s)=>Some(s.len()),Value::Array(a)=>a.iter().try_fold(0usize,|n,v|n.checked_add(strings(v)?)),
            Value::Object(m)=>m.values().try_fold(0usize,|n,v|n.checked_add(strings(v)?)),_=>Some(0),
        }}
        let mut bytes=0usize;
        let mut count=0usize;
        for (index,list) in &self.values { for value in list {
            count=count.checked_add(1).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
            // 26 fixed cells, bounded integer/date/percentage formatting;
            // lowercase may expand a Unicode scalar into three scalars.
            let row=32*(std::mem::size_of::<(&str,Option<String>)>()+128)+std::mem::size_of::<Vec<(&str,Option<String>)>>();
            bytes=bytes.checked_add(row).and_then(|n|n.checked_add(index.len())).and_then(|n|strings(value).and_then(|n|n.checked_mul(3)).and_then(|text|n.checked_add(text))).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        }}
        self.charge(bytes)?;
        Ok(count)
    }
    fn charge(&mut self, bytes:usize)->Result<(),RestResp> {
        self.memory.grow(bytes as u64,"recovery aggregate storage").map_err(|e|RestResp::from_es_error(&e))
    }
    fn prepare_copy(&mut self, st:&ClusterState, copy:&ShardRouting, text:Option<&str>)->Result<Option<Value>,RestResp> {
        let mut parsed=match text.map(OwnedJson::parse).transpose() {
            Ok(v)=>v,
            Err(e) if e.status==429=>return Err(RestResp::from_es_error(&e)),
            Err(_)=>None,
        };
        // At most 64 fixed-schema entries (including wrapper and synthesis),
        // two node descriptions, and the temporary source/target IDs.
        let mut bytes=64usize.checked_mul(map_storage()+64).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        for id in [copy.node_id.as_deref(), st.routing_table.shard_table(&copy.index,copy.shard).and_then(|t|t.primary()).and_then(|p|p.node_id.as_deref())].into_iter().flatten(){
            bytes=bytes.checked_add(id.len().checked_mul(4).ok_or_else(||RestResp::from_es_error(&memory_error()))?).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        }
        let max_node=st.nodes.values().try_fold(0usize,|max,n|{
            [n.id.len(),n.host.len(),n.transport_address.len(),n.ip.len(),n.name.len()].into_iter().try_fold(0usize,usize::checked_add).map(|size|max.max(size))
        }).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        bytes=bytes.checked_add(max_node.checked_mul(4).ok_or_else(||RestResp::from_es_error(&memory_error()))?).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        if let Some(v)=parsed.as_ref(){ for side in ["source","target"] {
            bytes=bytes.checked_add(v[side]["id"].as_str().map_or(0,|s|s.len().saturating_mul(4))).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        }}
        self.charge(bytes)?;
        if let Some(mut owner)=parsed.take(){
            if self.parsed.len()==self.parsed.capacity(){
                let wanted=self.parsed.len().checked_add(1).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
                self.charge(wanted.checked_mul(std::mem::size_of::<OwnedJson>()).ok_or_else(||RestResp::from_es_error(&memory_error()))?)?;
                self.parsed.try_reserve_exact(1).map_err(|_|RestResp::from_es_error(&memory_error()))?;
            }
            let mut value=Value::Null;
            owner.transform(|v,_|{value=std::mem::take(v);Ok(())}).map_err(|e|RestResp::from_es_error(&e))?;
            self.parsed.push(owner);
            Ok(Some(value))
        }else{Ok(None)}
    }
}


fn node_json(st: &ClusterState, id_or_name: Option<&str>) -> Value {
    let Some(key) = id_or_name.filter(|s| !s.is_empty()) else { return json!({}) };
    match st.nodes.get(key).or_else(|| st.nodes.values().find(|n| n.name == key)) {
        Some(n) => json!({"id": n.id, "host": n.host, "transport_address": n.transport_address, "ip": n.ip, "name": n.name}),
        None => json!({"id": key, "name": key}),
    }
}

fn synthesized(st: &ClusterState, c: &ShardRouting, created: i64) -> Value {
    let closed = st.metadata.indices.get(&c.index).map(|m| m.state == IndexState::Close).unwrap_or(false);
    let (ty, source) = if let (true, Some(elasticsearch_cluster::state::RecoverySource::Snapshot { repository, snapshot, index })) = (c.primary, c.recovery_source.as_ref()) {
        // A primary still being restored (its node reports the finished
        // restore itself).
        ("SNAPSHOT", json!({"repository": repository, "snapshot": snapshot, "index": index}))
    } else if c.primary {
        (if closed { "EXISTING_STORE" } else { "EMPTY_STORE" }, if closed { json!({"bootstrap_new_history_uuid": false}) } else { json!({}) })
    } else {
        let primary_node = st.routing_table.shard_table(&c.index, c.shard).and_then(|t| t.primary()).and_then(|p| p.node_id.clone());
        ("PEER", node_json(st, primary_node.as_deref()))
    };
    let done = c.state != ShardState::Initializing;
    json!({
        "id": c.shard,
        "type": ty,
        "stage": if done { "DONE" } else { "INIT" },
        "primary": c.primary,
        "start_time_in_millis": created,
        "stop_time_in_millis": created,
        "total_time_in_millis": 0,
        "source": source,
        "target": node_json(st, c.node_id.as_deref()),
        "index": {
            // Stats are those of the recovery that created the copy: a new
            // or closed store copies no files; a peer recovery whose state
            // this node no longer holds is reported without its files.
            "size": {"total_in_bytes": 0, "reused_in_bytes": 0, "recovered_in_bytes": 0, "recovered_from_snapshot_in_bytes": 0, "percent": "0.0%"},
            "files": {"total": 0, "reused": 0, "recovered": 0, "percent": "0.0%"},
            "total_time_in_millis": 0,
            "source_throttle_time_in_millis": 0,
            "target_throttle_time_in_millis": 0
        },
        "translog": {"recovered": 0, "total": 0, "percent": "100.0%", "total_on_start": 0, "total_time_in_millis": 0},
        "verify_index": {"check_index_time_in_millis": 0, "total_time_in_millis": 0}
    })
}

/// Recovery entries per index (index name → shard copy entries).
pub async fn collect(node: &Arc<ClusterNode>, req: &RestReq, expr: Option<&str>) -> Result<RecoveryEntries, RestResp> {
    let opts = cat_tables::cat_resolve_opts(req, true)?;
    let names = cat_tables::resolve(node, expr, opts)?;
    let st = node.state();
    let stats = gather_shard_stats(node, &st).await.map_err(|e|RestResp::from_es_error(&e))?;
    let p = req.params();
    let active_only = matches!(p.get("active_only").map(String::as_str), Some("") | Some("true"));
    let detailed = matches!(p.get("detailed").map(String::as_str), Some("") | Some("true"));
    let mut out = RecoveryEntries{values:BTreeMap::new(),parsed:Vec::new(),memory:breakers().reserve_request(0,"recovery response").map_err(|e|RestResp::from_es_error(&e))?.into_result_owner()};
    for name in names {
        let Some(irt) = st.routing_table.indices.get(&name) else { continue };
        let created = st.metadata.indices.get(&name).map(|m| m.creation_date).unwrap_or(0);
        let count=irt.shards.iter().try_fold(0usize,|sum,t|sum.checked_add(t.copies.iter().filter(|c|c.node_id.is_some()).count())).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        let slots=count.checked_mul(std::mem::size_of::<Value>()).and_then(|n|n.checked_add(name.len()+map_storage())).ok_or_else(||RestResp::from_es_error(&memory_error()))?;
        out.charge(slots)?;
        let mut list = Vec::new();
        list.try_reserve_exact(count).map_err(|_|RestResp::from_es_error(&memory_error()))?;
        for t in &irt.shards {
            for c in t.copies.iter().filter(|c| c.node_id.is_some()) {
                let s = c.allocation_id().and_then(|a| stats.get(a));
                let mut v = match out.prepare_copy(&st,c,s.and_then(|s| s.recovery_json.as_deref()))? {
                    Some(mut v) => {
                        // A state that did not record its nodes (older
                        // peers): the copy's node, recovered from the
                        // primary's (ES always names both).
                        let id = |side: &str| v[side]["id"].as_str().filter(|s| !s.is_empty()).map(str::to_string);
                        let src = id("source").or_else(|| {
                            let primary = st.routing_table.shard_table(&c.index, c.shard).and_then(|t| t.primary()).and_then(|p| p.node_id.clone());
                            primary.filter(|p| Some(p) != c.node_id.as_ref())
                        });
                        let dst = id("target").or_else(|| c.node_id.clone());
                        // A snapshot restore's source is the snapshot.
                        if v["type"] != "SNAPSHOT" {
                            v["source"] = node_json(&st, src.as_deref());
                        }
                        v["target"] = node_json(&st, dst.as_deref());
                        v["primary"] = json!(c.primary);
                        v
                    }
                    None => synthesized(&st, c, created),
                };
                if active_only && v["stage"] == "DONE" {
                    continue;
                }
                // `?detailed`: the file list of a file-based recovery.
                if !detailed {
                    if let Some(files) = v["index"]["files"].as_object_mut() {
                        files.remove("details");
                    }
                } else if v["index"]["files"].get("details").is_none() {
                    v["index"]["files"]["details"] = json!([]);
                }
                list.push(v);
            }
        }
        if !list.is_empty() {
            out.values.insert(name, list);
        }
    }
    Ok(out)
}

pub async fn indices_recovery(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let expr = segs.first().filter(|s| !s.starts_with('_')).cloned();
    match collect(node, req, expr.as_deref()).await {
        Ok(mut m) => {
            let body: Map<String, Value> = std::mem::take(&mut m.values).into_iter().map(|(k, v)| (k, json!({"shards": v}))).collect();
            // Aggregate storage remains in `m` while the root acquires any
            // additional snapshot patch owners through its OwnedJson wrapper.
            let mut root=match OwnedJson::parse("{}"){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
            if let Err(e)=root.transform(|value,_|{*value=Value::Object(body);Ok(())}){return RestResp::from_es_error(&e)}
            if let Err(e)=crate::snapshot_api::patch_recovery_response(&node.state(), &mut root){return RestResp::from_es_error(&e)}
            RestResp::json(200, &*root)
        }
        Err(r) => r,
    }
}
