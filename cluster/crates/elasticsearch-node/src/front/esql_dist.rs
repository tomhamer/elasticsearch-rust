//! ES|QL over shards spread across nodes.
//!
//! `POST /_query` runs on the elected master's engine. [`EsqlShards`] is the
//! engine's [`ShardSource`]: every wanted shard is read from a started local
//! copy when there is one, else from a node holding a started copy (an
//! internal `POST /_query` carrying [`FETCH_HEADER`], served by
//! `elasticsearch_engine::esql::remote::serve` on that node). Shards with no
//! started copy anywhere are reported back (`ESQL_SHARDS_HEADER`) so the
//! query is retried once they start.

use super::{local_canonical, RestReq, RestResp, EXEC_LOCAL};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::esql::remote::{assign, FetchSpec, RemoteShard, ShardResult, ShardSource, Slot, Split};
use elasticsearch_engine::search::Target;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// Internal request header: the [`FetchSpec`] (JSON) of a fetch of this
/// node's shards for a query another node runs.
pub(super) const FETCH_HEADER: &str = "x-rustic-internal-esql-fetch";
/// Internal request header: this run coordinates the query (it may hand
/// the whole query to a node holding every shard).
pub(super) const COORDINATOR_HEADER: &str = "x-rustic-internal-esql-coordinator";
/// Internal response header: run the whole query on this node instead.
pub(super) const FORWARD_HEADER: &str = "x-rustic-internal-esql-forward";
/// The error a coordinating run stops with once it decided to forward.
const FORWARD_ERROR: &str = elasticsearch_engine::esql::remote::FORWARD_ERROR_TYPE;

/// What one `POST /_query` run saw of its shards.
#[derive(Default)]
pub(super) struct Seen {
    /// Shards with no started copy anywhere.
    pub unread: BTreeSet<(String, u32)>,
    /// A node holding started copies of every wanted shard, when some are
    /// not readable here.
    pub forward: Option<String>,
}

/// A node other than `me` holding started copies of every shard in
/// `wanted` (the first by id), if any.
pub(super) fn forward_target(state: &ClusterState, me: &str, wanted: &[(String, u32)]) -> Option<String> {
    if wanted.is_empty() {
        return None;
    }
    let mut nodes: Vec<&String> = state.nodes.keys().filter(|n| n.as_str() != me).collect();
    nodes.sort();
    nodes.into_iter().find(|n| wanted.iter().all(|(index, shard)| holders(state, index, *shard).iter().any(|h| h == *n))).cloned()
}

/// How one wanted shard is read.
#[derive(Debug, PartialEq)]
pub(super) enum ShardRead {
    Local,
    Remote,
    /// No started copy anywhere (yet).
    Unread,
}

/// `index`'s `shard` on `me`: `local_open` says whether this node has the
/// copy open in an engine.
pub(super) fn shard_read(state: &ClusterState, me: &str, index: &str, shard: u32, local_open: bool) -> ShardRead {
    if local_open && super::esql_copy_readable(state, me, index, shard) {
        return ShardRead::Local;
    }
    if !holders(state, index, shard).is_empty() {
        return ShardRead::Remote;
    }
    ShardRead::Unread
}

/// Nodes holding a started copy of `index`'s `shard`, primary first.
pub(super) fn holders(state: &ClusterState, index: &str, shard: u32) -> Vec<String> {
    let Some(table) = state.routing_table.shard_table(index, shard) else { return Vec::new() };
    // Searchable copies (ES|QL reads through `searchShards`).
    let mut copies: Vec<_> = table.searchable_copies(&state.nodes).filter(|c| c.node_id.as_ref().is_some_and(|n| state.nodes.contains_key(n))).collect();
    copies.sort_by_key(|c| !c.primary);
    copies.into_iter().filter_map(|c| c.node_id.clone()).collect()
}

/// The ES|QL shard source of one `POST /_query` on this node.
pub(super) struct EsqlShards {
    pub node: Arc<ClusterNode>,
    pub state: Arc<ClusterState>,
    pub engines: Vec<(u32, Arc<elasticsearch_engine::node::Node>)>,
    /// The `_query` request: fetches carry its credentials and body.
    pub template: RestReq,
    pub runtime: tokio::runtime::Handle,
    pub seen: Arc<Mutex<Seen>>,
    /// The coordinating run: hand the query to a node holding every shard
    /// rather than read some shards remotely.
    pub may_forward: bool,
    /// The node each fold point in time was opened on (`FetchPit` id): its
    /// reads and close go there, whatever the routing says by then.
    pub pit_nodes: Mutex<std::collections::HashMap<String, String>>,
}

impl ShardSource for EsqlShards {
    fn split(&self, targets: Vec<Target>) -> Result<Split> {
        let me = self.node.local.id.as_str();
        let mut out: Vec<Target> = Vec::new();
        let mut slots: Vec<Slot> = Vec::new();
        let mut all_local = true;
        let mut pinned = false;
        let mut wanted_all: Vec<(String, u32)> = Vec::new();
        for target in targets {
            if target.shard.is_some() || target.snapshot.is_some() {
                pinned = true;
                slots.push(Slot::Local(out.len()));
                out.push(target);
                continue;
            }
            let name = target.svc.name.clone();
            let wanted = crate::shard_engines::search_shards(&self.state.metadata, &name, target.routing.as_deref());
            let mut found = 0;
            for shard in &wanted {
                // Only a copy this node holds readable (started, or
                // recovered awaiting its start) is read here; a copy being
                // reset or still recovering is read elsewhere (P0-1).
                let copy = self.engines.iter().filter(|(k, _)| k == shard).find_map(|(_, engine)| {
                    let svc = engine.get_index(&name)?;
                    svc.shard(*shard).map(|copy| (svc, copy))
                }).filter(|(_, copy)| super::copies::readable_copy(&self.node, &name, *shard).is_ok_and(|r| Arc::ptr_eq(&r, copy)));
                wanted_all.push((name.clone(), *shard));
                let mut seen = self.seen.lock().unwrap();
                match (shard_read(&self.state, me, &name, *shard, copy.is_some()), copy) {
                    (ShardRead::Local, Some((svc, copy))) => {
                        found += 1;
                        slots.push(Slot::Local(out.len()));
                        out.push(Target { svc, shard: Some(copy), ..target.clone() });
                    }
                    (ShardRead::Remote, _) => {
                        all_local = false;
                        slots.push(Slot::Remote(RemoteShard { index: name.clone(), shard: *shard, alias_filter: target.alias_filter.clone() }));
                    }
                    _ => {
                        all_local = false;
                        seen.unread.insert((name.clone(), *shard));
                    }
                }
            }
            // Mappings of an index with no readable copy here.
            if found == 0 {
                all_local = false;
                out.push(target);
            }
        }
        if self.may_forward && !pinned && slots.iter().any(|s| matches!(s, Slot::Remote(_))) {
            if let Some(target) = forward_target(&self.state, me, &wanted_all) {
                self.seen.lock().unwrap().forward = Some(target.clone());
                return Err(EsError::new(503, FORWARD_ERROR, format!("ES|QL runs on [{target}], which holds every shard")));
            }
        }
        Ok(Split { targets: out, slots: if all_local { None } else { Some(slots) } })
    }

    fn fetch(&self, spec: &FetchSpec, memory: &mut elasticsearch_engine::memory::Reservation) -> Result<Vec<ShardResult>> {
        let state = self.node.state();
        use elasticsearch_engine::esql::remote::PitPhase;
        let pinned = spec.pit.as_ref().filter(|p| p.phase != PitPhase::Open)
            .and_then(|p| self.pit_nodes.lock().unwrap().get(&p.id).cloned());
        let (by_node, missing) = match pinned {
            Some(node) => (vec![(node, spec.shards.clone())], Vec::new()),
            None => assign(&spec.shards, |index, shard| holders(&state, index, shard)),
        };
        if let Some(pit) = &spec.pit {
            let mut nodes = self.pit_nodes.lock().unwrap();
            match pit.phase {
                PitPhase::Open => if let Some((node, _)) = by_node.first() { nodes.insert(pit.id.clone(), node.clone()); },
                PitPhase::Close => { nodes.remove(&pit.id); }
                PitPhase::Read => {}
            }
        }
        if let Some(s) = missing.first() {
            return Err(EsError::new(503, "no_shard_available_action_exception", format!("[{}][{}] no started copy to read", s.index, s.shard)));
        }
        let mut by_node = by_node;
        let mut out = Vec::new();
        // Nodes whose copies turned out not readable (the receiver checks:
        // `copies::ReadGuard`): their shards go to the next holders.
        let mut skip: BTreeSet<String> = BTreeSet::new();
        // The last copy failure: the answer when no holder is left.
        let mut last_err: Option<EsError> = None;
        loop {
            let requests = by_node.iter().cloned().map(|(target, shards)| {
                let mut sub = self.template.clone();
                let part = FetchSpec { shards, ..spec.clone() };
                let prepared=(|| -> elasticsearch_engine::error::Result<()> {
                    sub.retain_headers(|(name,_)| !name.eq_ignore_ascii_case(FETCH_HEADER) && !name.eq_ignore_ascii_case(EXEC_LOCAL) && !name.eq_ignore_ascii_case(COORDINATOR_HEADER) && !name.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER))?;
                    sub.push_header(FETCH_HEADER,&serde_json::to_string(&part).map_err(|e|EsError::internal(e.to_string()))?)?;
                    // The calling thread waits (`block_on`) for these reads:
                    // run off the executors wherever they go.
                    if elasticsearch_http::pools::sub_requests_nested() && !super::is_nested(&sub.headers) {
                        sub.push_header(super::NESTED_HEADER,"1")?;
                    }
                    Ok(())
                })();
                let node = self.node.clone();
                async move {
                    if let Err(e)=prepared {return RestResp::json(e.status,&e.to_json())}
                    if target == node.local.id {
                        let mut local = sub;
                        if local.hops == 0 {
                            local.hops = 1;
                        }
                        return local_canonical(&node, &local).await;
                    }
                    // Bounded by the node leaving the cluster (a hung node).
                    super::copy_request(&node, &sub, &target, None).await
                }
            });
            let responses = self.runtime.block_on(futures::future::join_all(requests));
            let mut retry: Vec<RemoteShard> = Vec::new();
            for ((target, shards), response) in by_node.iter().zip(responses) {
                let decode_bytes=(response.body.len() as u64).saturating_mul(128).saturating_add(4096);
                let value=decode_page_value(&response.body,memory)?;
                if !response.is_success() {
                    // Any copy failure but a request error goes to the shards'
                    // next holders, as a search's shard requests do.
                    let retry_next = !super::copies::is_request_error(&response);
                    let error = EsError::new(response.status, value["error"]["type"].as_str().unwrap_or("exception"), value["error"]["reason"].as_str().unwrap_or("remote shard read failed"));
                    if retry_next && spec.pit.is_none() && skip.insert(target.clone()) {
                        retry.extend(shards.iter().cloned());
                        last_err = Some(error);
                        drop(value);
                        memory.shrink(decode_bytes);
                        continue;
                    }
                    return Err(error);
                }
                out.extend(serde_json::from_value::<Vec<ShardResult>>(value).map_err(|e| EsError::internal(format!("bad shard read reply: {e}")))?);
            }
            if retry.is_empty() {
                break;
            }
            let (next, missing) = assign(&retry, |index, shard| holders(&state, index, shard).into_iter().filter(|n| !skip.contains(n)).collect());
            if let Some(s) = missing.first() {
                // Every holder failed: their failure, not "no copy".
                return Err(last_err.unwrap_or_else(|| EsError::new(503, "no_shard_available_action_exception", format!("[{}][{}] no started copy to read", s.index, s.shard))));
            }
            by_node = next;
        }
        Ok(out)
    }
}

fn decode_page_value(body:&[u8],memory:&mut elasticsearch_engine::memory::Reservation)->Result<Value> {
    memory.grow((body.len() as u64).saturating_mul(128).saturating_add(4096),"<esql_remote_page_decode>")?;
    serde_json::from_slice(body).map_err(|error|EsError::internal(format!("bad ES|QL shard response: {error}")))
}

/// The fetch spec of an internal shard read (`hops > 0` only: the client
/// edge strips the header).
pub(super) fn fetch_spec(req: &RestReq) -> Option<FetchSpec> {
    if req.hops == 0 {
        return None;
    }
    let (_, v) = req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(FETCH_HEADER))?;
    serde_json::from_str(v).ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn remote_esql_decode_admits_before_parse_and_releases_each_page() {
        let breakers=elasticsearch_engine::memory::Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "indices.breaker.request.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        for _ in 0..128 {
            let mut memory=breakers.reserve_request(0,"remote page").unwrap();
            let value=super::decode_page_value(br#"[{"hits":[{"_source":{"value":"x"}}]}]"#,&mut memory).unwrap();
            assert!(breakers.request.reserved()>0);drop(value);drop(memory);
            assert_eq!(breakers.request.reserved(),0);
        }
        let mut memory=breakers.reserve_request(0,"remote page").unwrap();
        // Malformed JSON would return 500 if parsing ran before admission.
        let error=super::decode_page_value(&vec![b'x';1024],&mut memory).unwrap_err();
        assert_eq!(error.status,429);assert_eq!(breakers.request.reserved(),0);
    }

    use super::*;
    use elasticsearch_cluster::state::{AllocationId, DiscoveryNode, IndexMetadata, IndexRoutingTable, IndexShardRoutingTable, IndexState, Role, ShardRouting, ShardState};
    use serde_json::json;
    use std::collections::BTreeMap;
    use ShardState::{Initializing, Started, Unassigned};

    fn copy(shard: u32, primary: bool, state: ShardState, node: Option<&str>) -> ShardRouting {
        ShardRouting { index: "i".into(), shard, primary, state, node_id: node.map(str::to_string), relocating_node_id: None,
            allocation_id: node.map(|n| AllocationId { id: format!("a-{n}-{shard}"), relocation_id: None }), recovery_source: None,
            unassigned_info: None, expected_shard_size_in_bytes: None }
    }

    fn cluster(shards: Vec<Vec<ShardRouting>>) -> ClusterState {
        let mut s = ClusterState::empty("test");
        for n in ["n1", "n2", "n3"] {
            s.nodes.insert(n.into(), DiscoveryNode { id: n.into(), name: n.into(), ephemeral_id: "e".into(), transport_address: "127.0.0.1:9300".into(),
                http_address: "127.0.0.1:9200".into(), host: "h".into(), ip: "127.0.0.1".into(), roles: Role::defaults(), attributes: BTreeMap::new(), version: "9.5.5".into() });
        }
        let n = shards.len() as u32;
        s.metadata.indices.insert("i".into(), IndexMetadata { name: "i".into(), uuid: "u".into(), version: 1, mapping_version: 1, settings_version: 1,
            aliases_version: 1, state: IndexState::Open, settings: Default::default(), mappings: json!({}).into(), aliases: BTreeMap::new(), number_of_shards: n,
            number_of_replicas: 0, routing_num_shards: 1024, primary_terms: vec![1; n as usize], in_sync_allocation_ids: BTreeMap::new(), creation_date: 0, lifecycle: None });
        s.routing_table.indices.insert("i".into(), IndexRoutingTable { index_uuid: "u".into(),
            shards: shards.into_iter().enumerate().map(|(k, copies)| IndexShardRoutingTable { shard: k as u32, copies }).collect() });
        s
    }

    #[test]
    fn no_replica_split_reads_each_shard_where_it_is_started() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("n1"))], vec![copy(1, true, Started, Some("n2"))]]);
        assert_eq!(shard_read(&s, "n1", "i", 0, true), ShardRead::Local);
        assert_eq!(shard_read(&s, "n1", "i", 1, false), ShardRead::Remote);
        assert_eq!(holders(&s, "i", 1), ["n2"]);
    }

    #[test]
    fn a_recovering_local_copy_is_read_remotely() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("n2")), copy(0, false, Initializing, Some("n1"))]]);
        assert_eq!(shard_read(&s, "n1", "i", 0, true), ShardRead::Remote);
        assert_eq!(holders(&s, "i", 0), ["n2"]);
    }

    #[test]
    fn primaries_are_preferred_and_unstarted_copies_never_read() {
        let s = cluster(vec![vec![copy(0, false, Started, Some("n3")), copy(0, true, Started, Some("n2")), copy(0, false, Initializing, Some("n1"))]]);
        assert_eq!(holders(&s, "i", 0), ["n2", "n3"]);
        assert_eq!(shard_read(&s, "n1", "i", 0, true), ShardRead::Remote);
    }

    #[test]
    fn a_shard_without_a_started_copy_is_unread() {
        let s = cluster(vec![vec![copy(0, true, Initializing, Some("n2"))], vec![copy(1, true, Unassigned, None)]]);
        assert_eq!(shard_read(&s, "n1", "i", 0, false), ShardRead::Unread);
        assert_eq!(shard_read(&s, "n1", "i", 1, false), ShardRead::Unread);
        assert!(holders(&s, "i", 9).is_empty());
    }

    #[test]
    fn copies_on_departed_nodes_are_not_holders() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("gone"))]]);
        assert!(holders(&s, "i", 0).is_empty());
    }

    #[test]
    fn a_node_holding_every_shard_takes_the_query() {
        // 1 replica: n2 holds both shards, n1 only shard 0.
        let s = cluster(vec![
            vec![copy(0, true, Started, Some("n1")), copy(0, false, Started, Some("n2"))],
            vec![copy(1, true, Started, Some("n2")), copy(1, false, Initializing, Some("n1"))],
        ]);
        let wanted = vec![("i".to_string(), 0), ("i".to_string(), 1)];
        assert_eq!(forward_target(&s, "n1", &wanted), Some("n2".into()));
        assert_eq!(forward_target(&s, "n2", &wanted), None);
        assert_eq!(forward_target(&s, "n1", &[]), None);
    }

    #[test]
    fn a_split_without_a_full_holder_is_not_forwarded() {
        let s = cluster(vec![vec![copy(0, true, Started, Some("n1"))], vec![copy(1, true, Started, Some("n2"))], vec![copy(2, true, Started, Some("n3"))]]);
        let wanted: Vec<(String, u32)> = (0..3).map(|k| ("i".to_string(), k)).collect();
        assert_eq!(forward_target(&s, "n1", &wanted), None);
        // Every holder of the subset that is not here.
        assert_eq!(forward_target(&s, "n1", &wanted[1..2]), Some("n2".into()));
    }

    #[test]
    fn fetch_specs_are_internal_only() {
        let spec = FetchSpec { shards: vec![RemoteShard { index: "i".into(), shard: 1, alias_filter: None }], query: json!({"match_all": {}}), body: json!({}), size: None, probes: vec![], agg: None, group_limit: None, pit: None, fast: None };
        let mut req = RestReq::try_new("POST","/_query",&[(FETCH_HEADER.into(),serde_json::to_string(&spec).unwrap())],&[],0,Default::default(),None).unwrap();
        assert_eq!(fetch_spec(&req), None);
        req.hops = 1;
        assert_eq!(fetch_spec(&req), Some(spec));
    }
}
