//! Cluster-level REST APIs (`_cluster/*`, `_nodes/*`, cluster `_cat`s, `/`).
use crate::{front::SharedText,shared_sequence::SharedSequence};
fn wire_memory(e:elasticsearch_engine::error::EsError)->RemoteError{crate::control_memory::remote_error(&e)}

pub mod capabilities;
pub mod cat;
pub mod cat_columns;
pub mod cat_tables;
pub mod cluster_state;
pub mod autoscaling;
pub mod dangling;
pub mod desired_nodes;
pub mod indices_apis;
pub mod internal;
pub mod node_shutdown;
pub mod nodes;
pub mod recovery;
pub mod tasks;
pub mod util;

use crate::front::{self, classify::ClusterApi, local_engine, RestReq, RestResp};
use crate::master::now_millis;
use crate::node::ClusterNode;
use elasticsearch_cluster::allocation::{ExplainOptions, RerouteFlags};
use elasticsearch_cluster::health::{self, ActiveShardCount, HealthInputs, HealthStatus, Level, WaitConditions};
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_transport::RemoteError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub const ACTION_SHARD_STATS: &str = "internal:monitor/shards/local_stats";
const ACTION_SHARD_STATS_V2: &str = "internal:monitor/shards/local_stats_v2";
const ACTION_SHARD_STATS_BASIC_V2: &str = "internal:monitor/shards/local_stats_basic_v2";
const ACTION_SHARD_STATS_BASIC: &str = "internal:monitor/shards/local_stats_basic";
/// V2 plus each segment's bytes and deleted docs (`_cat/segments` size).
const ACTION_SHARD_STATS_V3: &str = "internal:monitor/shards/local_stats_v3";
const ACTION_SHARD_STATS_BASIC_V3: &str = "internal:monitor/shards/local_stats_basic_v3";

/// [`LocalShardStats`] with what its `segment_details` leave off the V2
/// wire: (bytes, deleted docs) per entry, in order. A new action, since the
/// V2 layout is positional.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShardStatsV3 {
    stats: LocalShardStats,
    segment_sizes: SharedSequence<(u64, u64)>,
}
impl ShardStatsV3 {
    fn from_local(stats: LocalShardStats) -> Result<Self, elasticsearch_engine::error::EsError> {
        let segment_sizes = SharedSequence::try_collect(stats.segment_details.iter().flat_map(|d| d.iter()).map(|e| (e.bytes, e.deleted)))?;
        Ok(Self { stats, segment_sizes })
    }
    fn into_local(self) -> Result<LocalShardStats, elasticsearch_engine::error::EsError> {
        let Self { mut stats, segment_sizes } = self;
        if let Some(details) = &stats.segment_details {
            let merged = SharedSequence::try_collect(details.iter().enumerate().map(|(i, e)| {
                let (bytes, deleted) = segment_sizes.get(i).copied().unwrap_or((0, 0));
                elasticsearch_engine::rustic_core::reader_view::SegmentStats { bytes, deleted, ..e.clone() }
            }))?;
            stats.segment_details = Some(merged);
        }
        Ok(stats)
    }
}

/// Per-copy stats a node reports for `_cat/shards` and friends.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalShardStats {
    pub index: SharedText,
    pub shard: u32,
    pub allocation_id: SharedText,
    pub docs: u64,
    /// Deleted (not yet merged away) documents.
    pub deleted: u64,
    pub store_bytes: u64,
    pub recovery_json: Option<SharedText>,
    /// Legacy field: documents per searchable unit only. Pending/commit-only
    /// units belong exclusively in `segment_details` for rolling upgrades.
    #[serde(default)]
    pub segments: SharedSequence<u64>,
    /// Documents with a dense / sparse vector value in this copy.
    #[serde(default)]
    pub dense_vectors: u64,
    #[serde(default)]
    pub sparse_vectors: u64,
    /// None on older peers, whose reported segments were all searchable.
    #[serde(default)]
    pub searchable_segments: Option<usize>,
    /// Rich enumeration is separate: old coordinators must receive only
    /// searchable units in the legacy `segments` field.
    #[serde(default)]
    pub segment_details: Option<SharedSequence<elasticsearch_engine::rustic_core::reader_view::SegmentStats>>,
}

// Positional bincode layout used by pre-v2 peers. Never append fields here:
// a vector of structs has no per-element framing for unknown trailing fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyShardStats {
    index: SharedText,
    shard: u32,
    allocation_id: SharedText,
    docs: u64,
    deleted: u64,
    store_bytes: u64,
    recovery_json: Option<SharedText>,
    segments: SharedSequence<u64>,
    dense_vectors: u64,
    sparse_vectors: u64,
}
impl From<LocalShardStats> for LegacyShardStats {
    fn from(s: LocalShardStats) -> Self {
        Self { index: s.index, shard: s.shard, allocation_id: s.allocation_id, docs: s.docs,
            deleted: s.deleted, store_bytes: s.store_bytes, recovery_json: s.recovery_json,
            segments: s.segments, dense_vectors: s.dense_vectors, sparse_vectors: s.sparse_vectors }
    }
}
impl From<LegacyShardStats> for LocalShardStats {
    fn from(s: LegacyShardStats) -> Self {
        Self { index: s.index, shard: s.shard, allocation_id: s.allocation_id, docs: s.docs,
            deleted: s.deleted, store_bytes: s.store_bytes, recovery_json: s.recovery_json,
            segments: s.segments, dense_vectors: s.dense_vectors, sparse_vectors: s.sparse_vectors,
            searchable_segments: None, segment_details: None }
    }
}

#[derive(Clone)]
struct SegmentDetails<'a>{stats:&'a LocalShardStats,next:usize}
impl SegmentDetails<'_>{fn iter(&self)->Self{self.clone()}}
impl Iterator for SegmentDetails<'_>{
 type Item=elasticsearch_engine::rustic_core::reader_view::SegmentStats;
 fn next(&mut self)->Option<Self::Item>{
  let i=self.next;
  let value=if let Some(details)=&self.stats.segment_details {details.get(i).cloned()}
  else{self.stats.segments.get(i).map(|&docs|Self::Item{id:i as u32,docs,search:i<self.stats.searchable_segments.unwrap_or(self.stats.segments.len()),committed:true,bytes:0,deleted:0})};
  if value.is_some(){self.next+=1;}value
 }
 fn size_hint(&self)->(usize,Option<usize>){let n=self.len();(n,Some(n))}
}
impl ExactSizeIterator for SegmentDetails<'_>{fn len(&self)->usize{self.stats.segment_details.as_ref().map_or(self.stats.segments.len(),|d|d.len()).saturating_sub(self.next)}}
impl<'a> IntoIterator for &SegmentDetails<'a>{type Item=elasticsearch_engine::rustic_core::reader_view::SegmentStats;type IntoIter=SegmentDetails<'a>;fn into_iter(self)->Self::IntoIter{self.clone()}}
impl LocalShardStats {fn segment_details(&self)->SegmentDetails<'_>{SegmentDetails{stats:self,next:0}}}

pub fn register(node: &Arc<ClusterNode>) {
    nodes::register(node);
    crate::keystore::reload::register(node);
    let n = node.clone();
    node.transport.register_handler(cat_tables::ACTION_CAT_FIELDDATA, move |r: cat_tables::FielddataReq| {
        let n = n.clone();
        async move {
            match cat_tables::local_fielddata(&n, &r).await {
                Ok(sizes) => crate::front::SharedText::try_json(&sizes).map_err(|e|RemoteError::new(&e.kind,e.reason,e.status)),
                Err(resp) => Err(n.transport.admitted_error("exception",std::str::from_utf8(&resp.body).unwrap_or("invalid UTF-8 fielddata error"),resp.status).unwrap_or_else(|e|e.to_remote_error())),
            }
        }
    });
    tasks::register(node);
    dangling::register(node);
    autoscaling::register(node);
    let n = node.clone();
    node.transport.register_handler(ACTION_SHARD_STATS, move |_: ()| {
        let n = n.clone();
        async move { local_shard_stats(&n, true).await.and_then(|v| SharedSequence::try_collect(v.into_iter().map(LegacyShardStats::from)).map_err(wire_memory)) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_SHARD_STATS_BASIC, move |_: ()| {
        let n = n.clone();
        async move { local_shard_stats(&n, false).await.and_then(|v| SharedSequence::try_collect(v.into_iter().map(LegacyShardStats::from)).map_err(wire_memory)) }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_SHARD_STATS_V2, move |_: ()| {
        let n = n.clone();
        async move { local_shard_stats(&n, true).await }
    });
    let n = node.clone();
    node.transport.register_handler(ACTION_SHARD_STATS_BASIC_V2, move |_: ()| {
        let n = n.clone();
        async move { local_shard_stats(&n, false).await }
    });
    for (action, vectors) in [(ACTION_SHARD_STATS_V3, true), (ACTION_SHARD_STATS_BASIC_V3, false)] {
        let n = node.clone();
        node.transport.register_handler(action, move |_: ()| {
            let n = n.clone();
            async move { local_shard_stats(&n, vectors).await.and_then(|v| SharedSequence::try_collect_results(v.iter().cloned().map(ShardStatsV3::from_local)).map_err(wire_memory)) }
        });
    }

}

async fn local_shard_stats(node: &ClusterNode, include_vectors: bool) -> Result<SharedSequence<LocalShardStats>, RemoteError> {
    let copies = node.shards.admitted_all().map_err(wire_memory)?;
    // Core locks and filesystem scans must not occupy a transport/runtime
    // worker: a concurrent maintenance operation may keep them waiting.
    tokio::task::spawn_blocking(elasticsearch_engine::memory::inherit_request_policy(move || collect_shard_stats(copies, include_vectors)))
        .await.map_err(|error| RemoteError::internal(error.to_string()))?
}

/// The fixed recovery JSON schema creates a bounded set of maps/keys plus
/// five fields per file. Admit its actual recursive tree workspace before the
/// existing renderer builds it; the encoded SharedText owns independent bytes.
fn recovery_text(state:&crate::replication::recovery::RecoveryState)->Result<SharedText,elasticsearch_engine::error::EsError>{
    use elasticsearch_engine::{error::EsError,memory::breakers};
    fn overflow()->EsError{EsError::new(429,"circuit_breaking_exception","recovery stats workspace overflow")}
    let map_unit=2*(11*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>());
    // At most 64 fixed entries, each literal key <=64bytes, and three percent
    // strings <=64bytes. Two copies each of source/target names are rendered.
    let mut bytes=64usize.checked_mul(map_unit+64).and_then(|n|n.checked_add(3*64)).ok_or_else(overflow)?;
    for text in [&state.source_node,&state.target_node].into_iter().flatten(){bytes=bytes.checked_add(text.len().checked_mul(2).ok_or_else(overflow)?).ok_or_else(overflow)?;}
    for file in state.files.details.iter(){bytes=bytes.checked_add(5*(map_unit+64)+4*std::mem::size_of::<Value>()).and_then(|n|n.checked_add(file.name.len())).ok_or_else(overflow)?;}
    let _workspace=breakers().reserve_request(bytes as u64,"recovery stats JSON tree")?;
    let tree=state.to_recovery_json();SharedText::try_json(&tree)
}

/// [`local_shard_stats`] with each copy's document counts and store size
/// only (index stats reads nothing else): no recovery JSON rendering and no
/// segment enumeration per copy and request.
async fn local_shard_doc_stats(node: &ClusterNode) -> Result<SharedSequence<LocalShardStats>, RemoteError> {
    let copies = node.shards.admitted_all().map_err(wire_memory)?;
    tokio::task::spawn_blocking(elasticsearch_engine::memory::inherit_request_policy(move || collect_shard_doc_stats(copies)))
        .await.map_err(|error| RemoteError::internal(error.to_string()))?
}

/// See [`local_shard_doc_stats`]: the fields [`collect_shard_stats`] fills
/// the same way; recovery, segments and vectors are left empty.
fn collect_shard_doc_stats(copies: SharedSequence<Arc<parking_lot::Mutex<crate::shards::LocalShard>>>) -> Result<SharedSequence<LocalShardStats>, RemoteError> {
    SharedSequence::try_collect_results(copies.iter()
        .map(|s| {
            let (index, shard, allocation_id, svc) = {
                let g = s.lock();
                (SharedText::try_new(&g.index)?, g.shard, SharedText::try_new(&g.allocation_id)?, g.svc.clone())
            };
            let (live, deleted) = svc.doc_count();
            // No open engine copy (closing, or a file recovery between its
            // placeholder and the received copy): nothing stored to report.
            let store_bytes = svc.shard(shard).map_or(0, |copy| copy.shard.store_size());
            Ok(LocalShardStats {
                index,
                shard,
                allocation_id,
                docs: live,
                deleted,
                store_bytes,
                recovery_json: None,
                segments: SharedSequence::default(),
                dense_vectors: 0,
                sparse_vectors: 0,
                searchable_segments: None,
                segment_details: None,
            })
        })).map_err(wire_memory)
}

fn collect_shard_stats(copies: SharedSequence<Arc<parking_lot::Mutex<crate::shards::LocalShard>>>, include_vectors: bool) -> Result<SharedSequence<LocalShardStats>, RemoteError> {
    SharedSequence::try_collect_results(copies.iter()
        .map(|s| {
            // Do not hold the routing/recovery record mutex while waiting on
            // an engine lock. Coordination also needs to access this record.
            let (index, shard, allocation_id, svc, recovery) = {
                let g = s.lock();
                (SharedText::try_new(&g.index)?, g.shard, SharedText::try_new(&g.allocation_id)?, g.svc.clone(), g.recovery.clone())
            };
            let (live, deleted) = svc.doc_count();
            // No open engine copy (closing, or a file recovery between its
            // placeholder and the received copy): no segments, no store yet.
            let copy = svc.shard(shard);
            let (segments, segment_details, searchable_segments, store_bytes) = match &copy {
                Some(copy) => {
                    let view = copy.shard.reader_view();
                    (SharedSequence::try_collect(view.leaves().iter().map(|l| u64::from(l.num_docs())))?,
                        Some(SharedSequence::try_generate(|push|view.try_visit_segment_stats(push))?), Some(view.num_leaves()), copy.shard.store_size())
                }
                None => (SharedSequence::default(), None, None, 0),
            };
            let (dense_vectors, sparse_vectors) = if include_vectors {
                elasticsearch_http::vector_counts_in(&svc, Some(shard))
            } else { (0, 0) };
            Ok(LocalShardStats {
                index,
                shard,
                allocation_id,
                docs: live,
                deleted,
                store_bytes,
                recovery_json: recovery.as_ref().map(|r| r.lock().as_ref().map(recovery_text).transpose()).transpose()?.flatten(),
                segments,
                dense_vectors,
                sparse_vectors,
                searchable_segments,
                segment_details,
            })
        })).map_err(wire_memory)
}

#[derive(Default)]
pub(crate) struct ShardStats {
    values:BTreeMap<SharedText,LocalShardStats>,
    owner:Option<elasticsearch_engine::memory::Reservation>,
}
impl std::ops::Deref for ShardStats {type Target=BTreeMap<SharedText,LocalShardStats>;fn deref(&self)->&Self::Target{&self.values}}
impl ShardStats {
    fn insert(&mut self,value:LocalShardStats)->Result<(),elasticsearch_engine::error::EsError>{
        if self.owner.is_none(){self.owner=Some(elasticsearch_engine::memory::breakers().reserve_request(0,"shard stats map")?.into_result_owner());}
        if !self.values.contains_key(&value.allocation_id){
            self.owner.as_mut().unwrap().grow((2*(12*std::mem::size_of::<(SharedText,LocalShardStats)>()+16*std::mem::size_of::<usize>()))as u64,"shard stats map node split")?;
        }
        self.values.insert(value.allocation_id.clone(),value);Ok(())
    }
}

/// Shard stats from every node, keyed by allocation id.
pub(crate) async fn gather_shard_stats(node: &ClusterNode, state: &ClusterState) -> Result<ShardStats,elasticsearch_engine::error::EsError> {
    gather_shard_stats_selected(node, state, false, false).await
}

/// [`gather_shard_stats`] for callers that read only `docs`, `deleted` and
/// `store_bytes` (index stats): this node's copies come from
/// [`local_shard_doc_stats`]; other nodes answer the basic action as before.
pub(crate) async fn gather_shard_doc_stats(node: &ClusterNode, state: &ClusterState) -> Result<ShardStats,elasticsearch_engine::error::EsError> {
    gather_shard_stats_selected(node, state, false, true).await
}

/// Only cluster stats consumes the per-copy vector counts. Cat, recovery,
/// segments and index-stats routing must not run unused existence searches.
/// `docs_only`: this node's copies skip recovery and segment details.
async fn gather_shard_stats_selected(node: &ClusterNode, state: &ClusterState, include_vectors: bool, docs_only: bool) -> Result<ShardStats,elasticsearch_engine::error::EsError> {
    let mut out = ShardStats::default();
    fn received<T>(result:Result<SharedSequence<T>,elasticsearch_transport::TransportError>)->Result<SharedSequence<T>,elasticsearch_engine::error::EsError>{
        match result {
            Ok(value)=>Ok(value),
            Err(error)=>{
                let error=error.to_remote_error();
                if error.status==429 {Err(crate::control_memory::imported_error(&error))}
                else {Ok(SharedSequence::default())}
            }
        }
    }
    async fn fetch(node:&ClusterNode,n:&elasticsearch_cluster::state::DiscoveryNode,include_vectors:bool,docs_only:bool)->Result<SharedSequence<LocalShardStats>,elasticsearch_engine::error::EsError>{
        if n.id==node.local.id {
            let local=if docs_only&&!include_vectors {local_shard_doc_stats(node).await} else {local_shard_stats(node,include_vectors).await};
            return match local {
                Ok(value)=>Ok(value),Err(error) if error.status==429=>Err(crate::control_memory::imported_error(&error)),Err(_)=>Ok(SharedSequence::default()),
            }
        }
        let Ok(address)=n.transport_address.parse::<SocketAddr>() else{return Ok(SharedSequence::default())};
        let action=if include_vectors{ACTION_SHARD_STATS_V3}else{ACTION_SHARD_STATS_BASIC_V3};
        match node.transport.send_request::<_,SharedSequence<ShardStatsV3>>(address,action,&(),Duration::from_secs(10)).await {
            Err(elasticsearch_transport::TransportError::ActionNotFound{..})=>{}
            result=>return SharedSequence::try_collect_results(received(result)?.iter().cloned().map(ShardStatsV3::into_local)),
        }
        let action=if include_vectors{ACTION_SHARD_STATS_V2}else{ACTION_SHARD_STATS_BASIC_V2};
        match node.transport.send_request::<_,SharedSequence<LocalShardStats>>(address,action,&(),Duration::from_secs(10)).await {
            Err(elasticsearch_transport::TransportError::ActionNotFound{..})=>{
                let action=if include_vectors{ACTION_SHARD_STATS}else{ACTION_SHARD_STATS_BASIC};
                let legacy=match node.transport.send_request::<_,SharedSequence<LegacyShardStats>>(address,action,&(),Duration::from_secs(10)).await {
                    Err(elasticsearch_transport::TransportError::ActionNotFound{..}) if !include_vectors=>node.transport.send_request::<_,SharedSequence<LegacyShardStats>>(address,ACTION_SHARD_STATS,&(),Duration::from_secs(10)).await,
                    result=>result,
                };
                SharedSequence::try_collect(received(legacy)?.into_iter().map(LocalShardStats::from))
            }
            result=>received(result),
        }
    }
    async fn optional(node:&ClusterNode,n:Option<&elasticsearch_cluster::state::DiscoveryNode>,vectors:bool,docs_only:bool)->Option<Result<SharedSequence<LocalShardStats>,elasticsearch_engine::error::EsError>>{match n{Some(n)=>Some(fetch(node,n,vectors,docs_only).await),None=>None}}
    let mut nodes=state.nodes.values();
    loop {
        let first=nodes.next();if first.is_none(){break}
        let results=tokio::join!(optional(node,first,include_vectors,docs_only),optional(node,nodes.next(),include_vectors,docs_only),optional(node,nodes.next(),include_vectors,docs_only),optional(node,nodes.next(),include_vectors,docs_only));
        for list in [results.0,results.1,results.2,results.3].into_iter().flatten(){for stats in list?{out.insert(stats)?;}}
    }
    Ok(out)
}

pub async fn handle(node: &Arc<ClusterNode>, api: ClusterApi, req: &RestReq) -> RestResp {
    // Cat tables, stats and recovery read index names and mappings from the
    // engine mirror, which a peer recovery briefly empties of its index.
    crate::front::await_mirror(node).await;
    let resp = handle_inner(node, api, req).await;
    util::postprocess(req, resp)
}

async fn handle_inner(node: &Arc<ClusterNode>, api: ClusterApi, req: &RestReq) -> RestResp {
    use ClusterApi as C;
    match api {
        C::Root => root(node, req).await,
        C::Health => cluster_health(node, req).await,
        C::State => state_api(node, req).await,
        C::Settings => settings(node, req).await,
        C::PendingTasks => RestResp::json(200, &json!({"tasks": []})),
        C::Reroute => front::retry_on_new_master(node, req, move || reroute(node, req)).await,
        C::AllocationExplain => explain(node, req).await,
        C::VotingConfigExclusions => voting_exclusions(node, req).await,
        C::NodesInfo | C::NodesStats | C::NodesHotThreads | C::NodesUsage => nodes::handle(node, req).await,
        C::NodeShutdown => node_shutdown::handle(node, req).await,
        C::Autoscaling => autoscaling::handle(node, req).await,
        C::Dangling => dangling::handle(node, req).await,
        C::CatNodes => cat_tables::cat_nodes(node, req).await,
        C::CatMaster => cat_tables::cat_master(node, req),
        C::CatHealth => cat_tables::cat_health(node, req),
        C::CatShards => cat_tables::cat_shards(node, req).await,
        C::CatAllocation => cat_tables::cat_allocation(node, req).await,
        C::CatNodeattrs => cat_tables::cat_nodeattrs(node, req),
        C::CatFielddata => cat_tables::cat_fielddata(node, req).await,
        C::CatPendingTasks => cat_tables::cat_pending_tasks(req),
        C::Tasks | C::TaskRethrottle => tasks::handle(node, req).await,
        C::CatTasks => tasks::cat_tasks(node, req).await,
        C::CatIndices => cat_tables::cat_indices(node, req).await,
        C::CatCount => cat_tables::cat_count(node, req).await,
        C::CatRecovery => cat_tables::cat_recovery(node, req).await,
        C::Recovery => recovery::indices_recovery(node, req).await,
        C::Stats => cluster_stats(node, req).await,
        C::CatSegments => indices_apis::cat_segments(node, req).await,
        C::FieldCaps => indices_apis::field_caps(node, req).await,
        C::ShardStores => indices_apis::shard_stores(node, req).await,
        C::Segments => indices_apis::segments(node, req).await,
        C::ClearCache => indices_apis::clear_cache(node, req),
        C::ResolveCluster => indices_apis::resolve_cluster(node, req).await,
        C::ReloadSecureSettings => crate::keystore::reload::handle(node, req).await,
        C::KnnSearch => indices_apis::knn_search(node, req).await,
        C::Snapshot | C::Slm | C::CatSnapshots | C::CatRepositories => crate::snapshot_api::handle(node, req).await,
        C::CatThreadPool | C::CatPlugins => local_engine(node, req).await,
        // Use the configured engine remote aliases after handle() awaited its mirror.
        // Returning an empty object here hid required disconnected remotes.
        C::RemoteInfo => local_engine(node, req).await,
        C::DesiredBalance => internal::desired_balance(node, req).await,
        C::DesiredNodes => desired_nodes::handle(node, req).await,
        C::PrevalidateNodeRemoval => internal::prevalidate_node_removal(node, req),
        C::Capabilities => capabilities::handle(node, req),
    }
}

// -------------------------------------------------------------------- root

async fn root(node: &ClusterNode, req: &RestReq) -> RestResp {
    let mut r = local_engine(node, req).await;
    if let Some(mut v) = r.body_json() {
        let st = node.state();
        v["name"] = json!(node.local.name);
        v["cluster_name"] = json!(st.cluster_name);
        v["cluster_uuid"] = json!(st.metadata.cluster_uuid);
        r.replace_json(&v);
    }
    r
}

// ------------------------------------------------------------------ health

pub(crate) fn parse_timeout(s: &str) -> Option<Duration> {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix("ms") {
        (n, 1)
    } else if let Some(n) = s.strip_suffix('s') {
        (n, 1000)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 60_000)
    } else if let Some(n) = s.strip_suffix('h') {
        (n, 3_600_000)
    } else if s == "-1" {
        return Some(Duration::from_secs(u64::MAX / 4));
    } else {
        (s, 1)
    };
    num.parse::<u64>().ok().map(|n| Duration::from_millis(n * mult))
}

async fn cluster_health(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let p = req.params();
    let level = match p.get("level").map(String::as_str) {
        None => Level::Cluster,
        Some(l) => match Level::parse(l) {
            Some(l) => l,
            None => return RestResp::error(400, "illegal_argument_exception", &format!("level must be one of [cluster, indices, shards] but was [{l}]")),
        },
    };
    let timeout = p.get("timeout").and_then(|t| parse_timeout(t)).unwrap_or(Duration::from_secs(30));
    let local = p.get("local").map(|v| v != "false").unwrap_or(false);
    let mut w = WaitConditions::default();
    if let Some(s) = p.get("wait_for_status") {
        match HealthStatus::parse(s) {
            Some(st) => w.status = Some(st),
            None => return RestResp::error(400, "illegal_argument_exception", &format!("unknown cluster health status [{s}]")),
        }
    }
    w.nodes = p.get("wait_for_nodes").cloned();
    if let Some(a) = p.get("wait_for_active_shards") {
        match ActiveShardCount::parse(a) {
            Ok(c) => w.active_shards = c,
            Err(e) => return RestResp::error(400, "illegal_argument_exception", &e),
        }
    }
    w.no_relocating_shards = p.get("wait_for_no_relocating_shards").map(|v| v != "false").unwrap_or(false);
    w.no_initializing_shards = p.get("wait_for_no_initializing_shards").map(|v| v != "false").unwrap_or(false);
    let segs = req.segments();
    let index_expr = segs.get(2).cloned();
    let deadline = tokio::time::Instant::now() + timeout;
    // `wait_for_events` is a metadata-service barrier, not merely a
    // health predicate. In particular managed registry work must be
    // committed before returning after a template was deleted.
    // A barrier that times out answers with the current health,
    // `timed_out: true` and 408 (ES: `ProcessClusterEventTimeoutException`
    // -> `sendResponse(.., TimeoutState.TIMED_OUT)`), which ECK's
    // `wait_for_events=languid&timeout=0s` probe parses.
    let mut events_timed_out = false;
    if p.contains_key("wait_for_events") && !local {
        if !node.is_master() { return forward_to_master(node, req).await; }
        match tokio::time::timeout_at(deadline, node.publish_engine_metadata()).await {
            Ok(Ok(_)) => {},
            Ok(Err(error)) => return RestResp::error(503, "master_not_discovered_exception", &error),
            Err(_) => events_timed_out = true,
        }
    }
    let mut rx = node.subscribe();
    let waits = w.status.is_some() || w.nodes.is_some() || w.active_shards.is_some() || w.no_relocating_shards || w.no_initializing_shards || p.contains_key("wait_for_events");
    loop {
        let state = rx.borrow().clone();
        let no_master = state.master_node_id.is_none();
        if no_master && !local {
            if tokio::time::Instant::now() >= deadline {
                return RestResp::error(503, "master_not_discovered_exception", "");
            }
        } else {
            let indices = match &index_expr {
                Some(e) => match resolve_health_indices(node, e, &p).await {
                    Ok(v) => Some(v),
                    Err(r) => return r,
                },
                None => None,
            };
            let timed_out = (tokio::time::Instant::now() >= deadline && waits) || events_timed_out;
            let h = health::cluster_health(&state, indices.as_deref(), HealthInputs { timed_out, ..Default::default() });
            let met = match health::wait_conditions_met(&h, &w) {
                Ok(m) => m,
                Err(e) => return RestResp::error(400, "illegal_argument_exception", &e),
            };
            if met || timed_out || !waits {
                let h = if met && !events_timed_out { health::cluster_health(&state, indices.as_deref(), HealthInputs::default()) } else { h };
                return RestResp::json(h.http_status(), &h.to_json(level, p.contains_key("human")));
            }
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            // Loop once more to render the timed-out response.
            continue;
        }
    }
}

/// Health's index expression: ES default options (open, closed, hidden;
/// lenient on missing).
async fn resolve_health_indices(node: &ClusterNode, expr: &str, p: &std::collections::HashMap<String, String>) -> Result<Vec<String>, RestResp> {
    let mut opts = elasticsearch_engine::node::ResolveOpts { ignore_unavailable: false, allow_no_indices: true, expand_open: true, expand_closed: true, expand_hidden: true };
    if let Some(ew) = p.get("expand_wildcards") {
        opts.expand_open = false;
        opts.expand_closed = false;
        opts.expand_hidden = false;
        for w in ew.split(',') {
            match w.trim() {
                "open" => opts.expand_open = true,
                "closed" => opts.expand_closed = true,
                "hidden" => opts.expand_hidden = true,
                "all" => {
                    opts.expand_open = true;
                    opts.expand_closed = true;
                    opts.expand_hidden = true;
                }
                "none" | "" => {}
                other => return Err(RestResp::error(400, "illegal_argument_exception", &format!("No enum constant org.elasticsearch.action.support.IndicesOptions.WildcardStates.{}", other.to_uppercase()))),
            }
        }
    }
    if let Some(v) = p.get("allow_no_indices") {
        opts.allow_no_indices = v != "false";
    }
    // Health treats an unexpanded wildcard as an empty selection. A
    // missing concrete index still produces the native timed-out response.
    if !opts.expand_open && !opts.expand_closed && !opts.expand_hidden && expr.contains(['*', '?']) {
        opts.ignore_unavailable = true;
    }
    match node.engine.resolve(expr, opts) {
        Ok(t) => Ok(t.iter().map(|t| t.svc.name.clone()).collect()),
        Err(e) if e.kind == "index_not_found_exception" => Err(RestResp::json(408, &json!({"cluster_name": node.state().cluster_name, "status": "red", "timed_out": true}))),
        Err(e) => Err(RestResp::json(e.status, &e.to_json_with_trace(false))),
    }
}

// ------------------------------------------------------------------- state

async fn state_api(node: &ClusterNode, req: &RestReq) -> RestResp {
    let segs = req.segments();
    let metrics = match cluster_state::parse_metrics(segs.get(2).map(String::as_str)) {
        Ok(m) => m,
        Err(e) => return RestResp::error(400, "illegal_argument_exception", &e),
    };
    let st = node.state();
    let indices: Option<BTreeSet<String>> = match segs.get(3) {
        Some(expr) => {
            let opts = elasticsearch_engine::node::ResolveOpts::from_params(&req.params());
            match node.engine.resolve(expr, opts) {
                Ok(t) => Some(t.iter().map(|t| t.svc.name.clone()).collect()),
                Err(e) => return RestResp::json(e.status, &e.to_json_with_trace(false)),
            }
        }
        None => None,
    };
    let mut v = cluster_state::render(&st, &metrics, indices.as_ref());
    // Node-specific features of this node, and `vectors.indexing.use_gpu`
    // of every node whose published attributes say it has a supported GPU
    // (the same answer from any node, with no peer probes).
    let local = elasticsearch_http::local_node_features(&node.engine);
    let gpu_nodes = if v["nodes_features"].is_array() { nodes::gpu_supported_nodes(&st) } else { BTreeSet::new() };
    for entry in v["nodes_features"].as_array_mut().into_iter().flatten() {
        let id = entry["node_id"].as_str().unwrap_or_default().to_string();
        let Some(f) = entry["features"].as_array_mut() else { continue };
        if id == node.local.id.as_str() {
            f.extend(local.iter().map(|x| serde_json::json!(x)));
        } else if gpu_nodes.contains(&id) && !f.iter().any(|x| x == elasticsearch_engine::gpu::USE_GPU_FEATURE) {
            f.push(serde_json::json!(elasticsearch_engine::gpu::USE_GPU_FEATURE));
        }
    }
    if metrics.contains("customs") {
        for (k, x) in crate::snapshot_api::render_state_customs(&st) {
            v[k.as_str()] = x;
        }
    }
    if let Some(fp) = req.params().get("filter_path") {
        v = elasticsearch_http::filter_path(&v, fp);
    }
    RestResp::json(200, &v)
}

// ---------------------------------------------------------------- settings

async fn settings(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if req.method() == axum::http::Method::GET {
        // Reads come from the published state; the engine renders them
        // (flat_settings, include_defaults) from its mirror.
        let mut local = req.clone();
        if let Err(e)=local.set_uri(&util::strip_params(&req.uri, &["filter_path", "pretty"])) { return RestResp::json(e.status,&e.to_json()); }
        let mut resp = local_engine(node, &local).await;
        let p = req.params();
        let on = |k: &str| matches!(p.get(k).map(String::as_str), Some("") | Some("true"));
        if resp.is_success() && on("include_defaults") {
            if let Some(mut v) = resp.body_json() {
                // ES reports the node's own settings among the defaults.
                let flat = on("flat_settings");
                let mut defaults = v["defaults"].take();
                if !defaults.is_object() {
                    defaults = json!({});
                }
                let filter = elasticsearch_engine::settings_filter::SettingsFilter::for_settings(node.config.settings.iter());
                for (k, val) in &node.config.settings {
                    if k.starts_with("path.") || k == "discovery.seed_hosts" || k == "cluster.initial_master_nodes" || filter.hides(k) {
                        continue;
                    }
                    if flat {
                        defaults.as_object_mut().map(|m| m.entry(k.clone()).or_insert(val.clone()));
                    } else {
                        let parts: Vec<&str> = k.split('.').collect();
                        let mut cur = &mut defaults;
                        for part in &parts[..parts.len() - 1] {
                            if !cur[*part].is_object() {
                                cur[*part] = json!({});
                            }
                            cur = &mut cur[*part];
                        }
                        if cur.get(parts[parts.len() - 1]).is_none() {
                            cur[parts[parts.len() - 1]] = val.clone();
                        }
                    }
                }
                v["defaults"] = defaults;
                resp.replace_json(&v);
                resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
            }
        }
        return resp;
    }
    front::on_master_route(node, req).await
}

// ----------------------------------------------------------------- reroute

async fn reroute(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if !node.is_master() {
        return forward_to_master(node, req).await;
    }
    let p = req.params();
    let flags = RerouteFlags { dry_run: p.get("dry_run").map(|v| v != "false").unwrap_or(false), explain: p.get("explain").map(|v| v != "false").unwrap_or(false), retry_failed: p.get("retry_failed").map(|v| v != "false").unwrap_or(false) };
    let body = req.json().unwrap_or(json!({}));
    let ctx = node.master.clone();
    let st = node.state();
    let info = ctx.cluster_info.lock().clone();
    let stores = elasticsearch_cluster::allocation::ShardStoreInfo::from_routing(&st);
    let result = match ctx.allocation.reroute_commands(&st, &body, flags, now_millis(), &info, &stores) {
        Ok(r) => r,
        Err((status, kind, reason)) => return RestResp::error(status, &kind, &reason),
    };
    if !flags.dry_run {
        let body2 = body.clone();
        let ctx2 = ctx.clone();
        let task: elasticsearch_cluster::coordination::StateTask = Box::new(move |s: &ClusterState| {
            let info = ctx2.cluster_info.lock().clone();
            let stores = elasticsearch_cluster::allocation::ShardStoreInfo::from_routing(s);
            let r = ctx2.allocation.reroute_commands(s, &body2, flags, now_millis(), &info, &stores).map_err(|(_, _, r)| r)?;
            Ok(ctx2.reroute(&r.state))
        });
        if let Err(e) = node.coord().submit(task).await {
            return front::task_error(&e);
        }
    }
    let mut out = json!({"acknowledged": true});
    if flags.explain {
        out["explanations"] = json!(result.explanations);
    }
    let compat_v8 = req.json_content_type().contains("compatible-with=8")
        || req.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("accept") && v.contains("compatible-with=8"));
    let metric = p.get("metric");
    let mut response = if compat_v8 && metric.map(|m| m.as_str()) != Some("none") {
        let metrics = cluster_state::parse_metrics(Some(metric.map(|m| m.as_str()).unwrap_or("_all"))).unwrap_or_default();
        let live = node.state();
        let mut state = cluster_state::render(if flags.dry_run { &result.state } else { &live }, &metrics, None);
        if metric.is_none() {
            if let Some(object) = state.as_object_mut() { object.remove("metadata"); }
        }
        out["state"] = state;
        RestResp::json(200, &out)
    } else {
        RestResp::json(200, &out)
    };
    let warning = if compat_v8 && out.get("state").is_some() {
        Some("The [state] field in the response to the reroute API is deprecated and will be removed in a future version. Specify ?metric=none to adopt the future behaviour.")
    } else if !compat_v8 && metric.is_some() {
        Some("the [?metric] query parameter to the [POST /_cluster/reroute] API has no effect; its use will be forbidden in a future version")
    } else { None };
    if let Some(warning) = warning {
        response.push_header(("Warning".into(), format!("299 Elasticsearch-{}-{} \"{warning}\"", elasticsearch_engine::node::ES_VERSION, elasticsearch_engine::version::BUILD_HASH)));
    }
    response
}

pub(crate) async fn forward_to_master(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    match node.state().master_node_id.clone() {
        Some(m) if m != node.local.id => front::forward(node, &m, req).await,
        _ => RestResp::error(503, "master_not_discovered_exception", ""),
    }
}

async fn explain(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if !node.is_master() {
        return forward_to_master(node, req).await;
    }
    let p = req.params();
    let opts = ExplainOptions { include_yes_decisions: p.get("include_yes_decisions").map(|v| v == "true").unwrap_or(false), include_disk_info: p.get("include_disk_info").map(|v| v == "true").unwrap_or(false) };
    let body = match allocation_request(req) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let st = node.state();
    let info = node.master.cluster_info.lock().clone();
    let stores = elasticsearch_cluster::allocation::ShardStoreInfo::from_routing(&st);
    match node.master.allocation.explain(&st, &body, opts, now_millis(), &info, &stores) {
        Ok(v) => RestResp::json(200, &v),
        Err((status, kind, reason)) => RestResp::error(status, &kind, &reason),
    }
}

/// ES9 accepts the allocation selector in the body or the URL, never both.
fn allocation_request(req: &RestReq) -> Result<Value, RestResp> {
    const SELECTOR: [&str; 4] = ["current_node", "index", "primary", "shard"];
    let p = req.params();
    let body = if !req.body.is_empty() { Some(req.body.as_ref()) } else { p.get("source").map(|s| s.as_bytes()) };
    if let Some(bytes) = body {
        let given: Vec<_> = SELECTOR.iter().filter(|k| p.contains_key(**k)).collect();
        if !given.is_empty() {
            let list = given.iter().map(|k| format!("[{k}]")).collect::<Vec<_>>().join(", ");
            let noun = if given.len() == 1 { "parameter" } else { "parameters" };
            return Err(RestResp::error(400, "illegal_argument_exception", &format!("request [{}] contains unrecognized {noun}: {list}", req.path())));
        }
        let v: Value = serde_json::from_slice(bytes).map_err(|e| RestResp::error(400, "x_content_parse_exception", &e.to_string()))?;
        if let Some(o) = v.as_object() {
            for k in o.keys() {
                if !SELECTOR.contains(&k.as_str()) {
                    return Err(RestResp::error(400, "x_content_parse_exception", &format!("[1:2] [cluster/allocation/explain] unknown field [{k}]")));
                }
            }
        }
        return Ok(v);
    }
    let mut out = Map::new();
    for key in SELECTOR {
        let Some(value) = p.get(key) else { continue };
        let v = match key {
            "shard" => json!(value.parse::<i32>().map_err(|_| RestResp::error(400, "illegal_argument_exception", &format!("Failed to parse int parameter [shard] with value [{value}]")))?),
            "primary" => json!(match value.as_str() {
                "true" | "" => true,
                "false" => false,
                _ => return Err(RestResp::error(400, "illegal_argument_exception", &format!("Failed to parse value [{value}] as only [true] or [false] are allowed."))),
            }),
            _ => json!(value),
        };
        out.insert(key.to_string(), v);
    }
    Ok(Value::Object(out))
}

// -------------------------------------------------------- voting exclusions

async fn voting_exclusions(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if !node.is_master() {
        return forward_to_master(node, req).await;
    }
    let p = req.params();
    let list = |k: &str| p.get(k).map(|s| s.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>()).unwrap_or_default();
    let r = if req.method() == axum::http::Method::DELETE {
        node.coord().clear_voting_config_exclusions(p.get("wait_for_removal").map(|v| v != "false").unwrap_or(true)).await
    } else {
        let (ids, names) = (list("node_ids"), list("node_names"));
        if ids.is_empty() == names.is_empty() {
            return RestResp::error(400, "illegal_argument_exception", "Please set node identifiers correctly. One and only one of [node_name], [node_names] and [node_ids] has to be set");
        }
        node.coord().add_voting_config_exclusions(ids, names).await
    };
    match r {
        Ok(_) => RestResp::new(200,vec![],vec![]),
        Err(e) => RestResp::error(400, "illegal_argument_exception", &e),
    }
}

// ------------------------------------------------------------ cluster stats

/// `_cluster/stats[/nodes/{ids}]`: the engine renders the full section shapes
/// from its metadata mirror; cluster-wide figures (status, node counts, shard
/// counts, docs and store across every copy) are patched in from the
/// cluster state and per-copy stats of every node.
async fn cluster_stats(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let mut local = req.clone();
    if let Err(e)=local.set_uri(&util::strip_params(&req.uri, &["filter_path", "pretty"])) { return RestResp::json(e.status,&e.to_json()); }
    let mut resp = local_canonical_stats(node, &local).await;
    if !resp.is_success() {
        return resp;
    }
    let text=match std::str::from_utf8(&resp.body){Ok(v)=>v,Err(_)=>return resp};
    let mut owner=match crate::owned_json::OwnedJson::parse(text){Ok(v)=>v,Err(e) if e.status==429=>return RestResp::from_es_error(&e),Err(_)=>return resp};
    let mut v=Value::Null;
    if let Err(e)=owner.transform(|tree,_|{v=std::mem::take(tree);Ok(())}){return RestResp::from_es_error(&e)};
    fn take_owned(owner:&mut crate::owned_json::OwnedJson,mut other:crate::owned_json::OwnedJson)->Result<Value,elasticsearch_engine::error::EsError>{
        owner.prepare_reply_owner(&other)?;
        let mut value=Value::Null;other.transform(|v,_|{value=std::mem::take(v);Ok(())})?;
        owner.retain_reply_owner(other);Ok(value)
    }
    let st = node.state();
    // Fixed patch keys plus version-set entries and existing remote totals.
    let unit=2*(11*std::mem::size_of::<(String,Value)>()+20*std::mem::size_of::<usize>())+64;
    let remotes=v["ccs"]["clusters"].as_object().map_or(0,|m|m.len());
    let bytes=(||->Option<usize>{
        let versions=st.nodes.values().try_fold(0usize,|sum,n|sum.checked_add(n.version.len().checked_mul(2)?))?;
        100usize.checked_add(st.nodes.len())?.checked_add(remotes.checked_mul(6)?)?.checked_mul(unit)?.checked_add(versions)?.checked_add(st.cluster_name.len())?.checked_add(st.metadata.cluster_uuid.len())
    })();
    let bytes=match bytes{Some(v)=>v,None=>return RestResp::error(429,"circuit_breaking_exception","cluster stats projection overflow")};
    if let Err(e)=owner.reserve_extra(bytes,"cluster stats patch storage"){return RestResp::from_es_error(&e)}
    let h = health::cluster_health(&st, None, HealthInputs::default());
    v["status"] = json!(h.status.name());
    v["cluster_name"] = json!(st.cluster_name);
    v["cluster_uuid"] = json!(st.metadata.cluster_uuid);
    // Node counts by role.
    let mut count = Map::new();
    count.insert("total".into(), json!(st.nodes.len()));
    count.insert("coordinating_only".into(), json!(st.nodes.values().filter(|n| n.roles.is_empty()).count()));
    for role in ["data", "data_cold", "data_content", "data_frozen", "data_hot", "data_warm", "index", "ingest", "master", "ml", "remote_cluster_client", "search", "transform", "voting_only"] {
        let c = st.nodes.values().filter(|n| n.roles.iter().any(|r| r.name() == role)).count();
        count.insert(role.into(), json!(c));
    }
    if v["nodes"].is_object() {
        v["nodes"]["count"] = Value::Object(count);
        if let Some(versions) = v["nodes"].get_mut("versions") {
            let set: BTreeSet<String> = st.nodes.values().map(|n| n.version.clone()).collect();
            *versions = json!(set);
        }
    }
    let (search_usage, failures) = match nodes::cluster_search_usage(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let failure_count=failures.as_array().map_or(0,Vec::len);
    v["indices"]["search"] = match take_owned(&mut owner,search_usage){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    v["_nodes"] = json!({"total": st.nodes.len(), "successful": st.nodes.len().saturating_sub(failure_count), "failed": failure_count});
    if failure_count>0 {v["_nodes"]["failures"]=match take_owned(&mut owner,failures){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};}
    // Shards / docs / store over every copy.
    let stats = match gather_shard_stats_selected(node, &st, true, false).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let (mut total, mut primaries, mut docs, mut deleted, mut store) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut dense, mut sparse) = (0u64, 0u64);
    for c in st.routing_table.all_shards().filter(|c| c.active()) {
        total += 1;
        let s = c.allocation_id().and_then(|a| stats.get(a));
        if let Some(s) = s {
            store += s.store_bytes;
        }
        if c.primary {
            primaries += 1;
            if let Some(s) = s {
                docs += s.docs;
                deleted += s.deleted;
                dense += s.dense_vectors;
                sparse += s.sparse_vectors;
            }
        }
    }
    if let Some(ix) = v.get_mut("indices").and_then(Value::as_object_mut) {
        ix.insert("count".into(), json!(st.metadata.indices.len()));
        if let Some(sh) = ix.get_mut("shards").and_then(Value::as_object_mut) {
            if total > 0 {
                sh.insert("total".into(), json!(total));
                sh.insert("primaries".into(), json!(primaries));
                sh.insert("replication".into(), json!(if primaries == 0 { 0.0 } else { (total - primaries) as f64 / primaries as f64 }));
            }
        }
        if let Some(d) = ix.get_mut("docs").and_then(Value::as_object_mut) {
            d.insert("count".into(), json!(docs));
            d.insert("deleted".into(), json!(deleted));
        }
        // Vector counts over every primary, wherever it lives (ES counts the
        // primaries, like docs).
        if let Some(d) = ix.get_mut("dense_vector").and_then(Value::as_object_mut) {
            d.insert("value_count".into(), json!(dense));
        }
        if let Some(d) = ix.get_mut("sparse_vector").and_then(Value::as_object_mut) {
            d.insert("value_count".into(), json!(sparse));
        }
        if let Some(s) = ix.get_mut("store").and_then(Value::as_object_mut) {
            s.insert("size_in_bytes".into(), json!(store));
            if s.contains_key("total_data_set_size_in_bytes") {
                s.insert("total_data_set_size_in_bytes".into(), json!(store));
            }
        }
    }
    if v["ccs"]["_search"].is_object() {
        let ccs=match nodes::cluster_ccs_stats(node).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
        v["ccs"]["_search"] = match take_owned(&mut owner,ccs){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    }
    // Remotes seeded with this cluster's own address describe this cluster:
    // the engine filled them from the serving node's local copies, so use
    // the cluster-wide figures computed above.
    let totals = json!({
        "status": v["status"], "nodes_count": v["nodes"]["count"]["total"], "shards_count": v["indices"]["shards"]["total"],
        "indices_count": v["indices"]["count"], "indices_total_size_in_bytes": v["indices"]["store"]["size_in_bytes"],
    });
    if let Some(clusters) = v["ccs"]["clusters"].as_object_mut() {
        for c in clusters.values_mut() {
            for (k, t) in totals.as_object().into_iter().flatten() {
                if c.get(k).is_some() && !t.is_null() {
                    c[k] = t.clone();
                }
            }
        }
    }
    let snapshot=match crate::snapshot_api::cluster_stats_section(&st,req.params().get("human").is_some_and(|h|h!="false")){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    v["snapshots"]=match take_owned(&mut owner,snapshot){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    resp.replace_json(&v);
    resp.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("content-length"));
    resp
}

async fn local_canonical_stats(node: &ClusterNode, req: &RestReq) -> RestResp {
    let mut r = req.clone();
    if let Err(e)=r.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("accept-encoding")) { return RestResp::json(e.status,&e.to_json()); }
    local_engine(node, &r).await
}

#[cfg(test)]
mod segment_wire_tests {
    use super::*;
    /// Independent byte fixtures for the two positional protocols. Same-type
    /// round trips cannot detect an accidentally appended field in a Vec.
    #[test]
    fn shard_stats_wire_layouts_are_pinned() {
        let value = json!({"index":"i", "shard":7, "allocation_id":"a", "docs":11,
            "deleted":2, "store_bytes":99, "recovery_json":"recovery", "segments":[5,6],
            "dense_vectors":3, "sparse_vectors":4, "searchable_segments":1,
            "segment_details":[{"id":0,"docs":5,"search":true,"committed":false},
                {"id":1,"docs":6,"search":false,"committed":true}]});
        let new: LocalShardStats = serde_json::from_value(value).unwrap();
        let legacy = LegacyShardStats::from(new.clone());
        fn decode_hex(hex: &str) -> Vec<u8> {
            (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i+2], 16).unwrap()).collect()
        }
        // Fixed little-endian bincode: Vec length, then two complete structs.
        let legacy_bytes = decode_hex(LEGACY);
        let current_bytes = decode_hex(CURRENT);
        assert_eq!(bincode::serialize(&vec![legacy.clone(), legacy]).unwrap(), legacy_bytes,
            "legacy layout changed: retain the old action schema and introduce a new version");
        assert_eq!(bincode::serialize(&vec![new.clone(), new]).unwrap(), current_bytes,
            "v2 layout changed: introduce a negotiated new action before changing this pin");
        assert_eq!(bincode::deserialize::<Vec<LegacyShardStats>>(&legacy_bytes).unwrap()[1].store_bytes, 99);
        assert_eq!(bincode::deserialize::<Vec<LocalShardStats>>(&current_bytes).unwrap()[1].segment_details().len(), 2);
        const LEGACY: &str = concat!(
            "0200000000000000010000000000000069070000000100000000000000610b0000000000000002000000000000006300",
            "0000000000000108000000000000007265636f7665727902000000000000000500000000000000060000000000000003",
            "000000000000000400000000000000010000000000000069070000000100000000000000610b00000000000000020000",
            "000000000063000000000000000108000000000000007265636f76657279020000000000000005000000000000000600",
            "00000000000003000000000000000400000000000000",
        );
        const CURRENT: &str = concat!(
            "0200000000000000010000000000000069070000000100000000000000610b0000000000000002000000000000006300",
            "0000000000000108000000000000007265636f7665727902000000000000000500000000000000060000000000000003",
            "000000000000000400000000000000010100000000000000010200000000000000000000000500000000000000010001",
            "00000006000000000000000001010000000000000069070000000100000000000000610b000000000000000200000000",
            "00000063000000000000000108000000000000007265636f766572790200000000000000050000000000000006000000",
            "000000000300000000000000040000000000000001010000000000000001020000000000000000000000050000000000",
            "000001000100000006000000000000000001",
        );
    }

    #[test]
    fn segment_stats_legacy_coordinator_never_receives_pending_units() {
        let value = json!({"index":"i","shard":0,"allocation_id":"a","docs":3,"deleted":0,
            "store_bytes":0,"recovery_json":null,"segments":[3],"searchable_segments":1,
            "segment_details":[{"id":0,"docs":3,"search":true,"committed":false},
                {"id":1,"docs":7,"search":false,"committed":false}]});
        let new: LocalShardStats = serde_json::from_value(value).unwrap();
        let legacy = LegacyShardStats::from(new.clone());
        let bytes = bincode::serialize(&vec![legacy.clone(), legacy]).unwrap();
        let old: Vec<LegacyShardStats> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(old.len(), 2);
        assert!(old.iter().all(|s| s.segments == vec![3]));
        let bytes = bincode::serialize(&vec![new.clone(), new.clone()]).unwrap();
        let decoded: Vec<LocalShardStats> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1].segment_details().len(), 2);
        assert_eq!(new.segment_details().iter().filter(|s| s.search).count(), 1);
        assert_eq!(new.segment_details().len(), 2);
        let old_wire = json!({"index":"i","shard":0,"allocation_id":"a","docs":3,"deleted":0,
            "store_bytes":0,"recovery_json":null,"segments":[3]});
        let new: LocalShardStats = serde_json::from_value(old_wire).unwrap();
        assert_eq!(new.segment_details().len(), 1);
        assert!(new.segment_details().next().unwrap().search);
    }
}
