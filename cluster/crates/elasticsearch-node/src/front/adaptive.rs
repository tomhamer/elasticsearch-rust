//! Adaptive replica selection, as ES's `ResponseCollectorService` and
//! `OperationRouting.rankShardsAndUpdateStats`: per node, EWMAs of the
//! search queue size and service time the executing node reports with each
//! successful query-phase answer and of the response time measured here,
//! ranked with the C3 formula against this coordinator's outstanding shard
//! requests. Explicit preferences continue to bypass this ranking.
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::{Duration, Instant}};
use elasticsearch_cluster::state::ClusterState;

/// Request header marking a query-phase shard request whose answer carries
/// [`STATS_HEADER`]. Only those answers feed the statistics (ES records
/// them for query results only, not fetch, free or failures).
pub(super) const QUERY_HEADER: &str = "x-rustic-internal-ars";
/// Response header: `"{search queue size},{search service time EWMA ns}"`
/// of the executing node (ES's `QuerySearchResult` queue size and service
/// time).
pub(super) const STATS_HEADER: &str = "x-rustic-internal-ars-stats";

/// ES `ResponseCollectorService.ALPHA`.
const ALPHA: f64 = 0.3;
/// Nanoseconds per millisecond (ES `ComputedNodeStats.FACTOR`).
const FACTOR: f64 = 1_000_000.0;

fn ewma(old: f64, value: f64) -> f64 {
    old * (1.0 - ALPHA) + value * ALPHA
}

/// ES `NodeStatistics`: EWMAs of queue size and response time, and the
/// node's own service time EWMA as last reported.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stats { queue: f64, response_ns: f64, service_ns: f64 }

#[derive(Default)]
struct Sample { outstanding: usize, stats: Option<Stats> }
#[derive(Default)]
struct Peer(Mutex<Sample>);

/// ES `ComputedNodeStats`: one node's statistics as read for one ranking.
#[derive(Clone, Copy)]
struct Computed { clients: usize, queue: i64, response_ns: f64, service_ns: f64 }
impl Computed {
    /// ES `ComputedNodeStats.innerRank` (the C3 formula, cubic queue
    /// adjustment).
    fn rank(&self, outstanding: u64) -> f64 {
        let concurrency_compensation = outstanding as f64 * self.clients as f64;
        let q_hat = 1.0 + concurrency_compensation + self.queue as f64;
        let r = self.response_ns / FACTOR;
        let mu = self.service_ns / FACTOR;
        r - 1.0 / mu + q_hat.powi(3) / mu
    }
}

/// The outstanding shard requests per node, captured once per search and
/// raised for each shard's winner while its shards are ranked (ES's
/// `nodeSearchCounts`).
#[derive(Default)]
pub(super) struct Counts(HashMap<String, u64>);

#[derive(Default)]
pub(crate) struct Selection { peers: Mutex<HashMap<(String, String), Arc<Peer>>> }
impl Selection {
    fn peer(&self, state: &ClusterState, id: &str) -> Option<Arc<Peer>> {
        let node = state.nodes.get(id)?;
        let mut peers = self.peers.lock();
        // Process identity matters: a restarted peer must not inherit old load.
        peers.retain(|(id, epoch), _| state.nodes.get(id).is_some_and(|n| &n.ephemeral_id == epoch));
        Some(peers.entry((id.to_string(), node.ephemeral_id.clone())).or_default().clone())
    }
    /// ES `_nodes/stats` `adaptive_selection`.
    pub(crate) fn snapshot(&self, state: &ClusterState) -> serde_json::Value {
        let mut peers = self.peers.lock();
        peers.retain(|(id, epoch), _| state.nodes.get(id).is_some_and(|n| &n.ephemeral_id == epoch));
        let clients = peers.values().filter(|p| p.0.lock().stats.is_some()).count();
        let mut out = serde_json::Map::new();
        for ((id, _), peer) in peers.iter() {
            let sample = peer.0.lock();
            if sample.stats.is_none() && sample.outstanding == 0 {continue;}
            let mut entry=serde_json::json!({"outgoing_searches":sample.outstanding});
            if let Some(s)=sample.stats {
                let computed = Computed { clients, queue: s.queue as i64, response_ns: s.response_ns, service_ns: s.service_ns };
                entry["avg_queue_size"]=serde_json::json!(computed.queue);
                entry["avg_service_time_ns"]=serde_json::json!(s.service_ns as i64);
                entry["avg_response_time_ns"]=serde_json::json!(s.response_ns as i64);
                entry["rank"]=serde_json::json!(format!("{:.1}", computed.rank(sample.outstanding.max(1) as u64)));
            }
            out.insert(id.clone(),entry);
        }
        serde_json::Value::Object(out)
    }
    /// This coordinator's outstanding shard requests per node, captured
    /// once per search (ES `SearchTransportService.getPendingSearchRequests`).
    pub(super) fn counts(&self) -> Counts {
        let peers = self.peers.lock();
        Counts(peers.iter().filter_map(|((id, _), p)| {
            let n = p.0.lock().outstanding;
            (n > 0).then(|| (id.clone(), n as u64))
        }).collect())
    }
    pub(super) fn begin(&self, state: &ClusterState, id: &str) -> Option<Attempt> {
        let peer = self.peer(state, id)?;
        peer.0.lock().outstanding += 1;
        Some(Attempt { peer, start: Instant::now() })
    }
    /// ES `rankShardsAndUpdateStats` for one shard: order `copies` by rank
    /// (nodes without statistics first, ties keep their rotated order), then
    /// move the other nodes' statistics toward the winner's so they get a
    /// chance to receive queries, and count one more outstanding request on
    /// the winner for the rest of this search's shards.
    ///
    /// During a rolling upgrade a node without [`STATS_HEADER`] support
    /// never reports stats, so it would rank first (unranked) for every
    /// shard: while any copy's node lacks the capability the shard keeps
    /// its rotated order.
    pub(super) fn order(&self, state: &ClusterState, copies: &mut [String], counts: &mut Counts) {
        if copies.len() <= 1 {
            return;
        }
        use crate::internal_capabilities::{supports, Capability};
        if !copies.iter().all(|id| state.nodes.get(id).is_some_and(|n| supports(n, Capability::AdaptiveStatsV1))) {
            return;
        }
        let peers: Vec<(String, Option<Arc<Peer>>)> = copies.iter().map(|id| (id.clone(), self.peer(state, id))).collect();
        let stats: HashMap<&str, Option<Stats>> = peers.iter().map(|(id, p)| (id.as_str(), p.as_ref().and_then(|p| p.0.lock().stats))).collect();
        let clients = self.peers.lock().values().filter(|p| p.0.lock().stats.is_some()).count();
        let computed = |s: &Stats| Computed { clients, queue: s.queue as i64, response_ns: s.response_ns, service_ns: s.service_ns };
        // ES `nodeSearchCounts.getOrDefault(nodeId, 1L)`.
        let effective = |counts: &Counts, id: &str| counts.0.get(id).copied().unwrap_or(1);
        let ranks: HashMap<&str, f64> = stats.iter().filter_map(|(id, s)| s.as_ref().map(|s| (*id, computed(s).rank(effective(counts, id))))).collect();
        // ES `NodeRankComparator`: unranked nodes before ranked ones.
        copies.sort_by(|a, b| match (ranks.get(a.as_str()), ranks.get(b.as_str())) {
            (Some(x), Some(y)) => x.total_cmp(y),
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        });
        let winner = copies[0].clone();
        let Some(min) = stats.get(winner.as_str()).copied().flatten() else { return };
        // ES `adjustStats`.
        for (id, peer) in &peers {
            if *id == winner { continue; }
            let Some(peer) = peer else { continue };
            let mut sample = peer.0.lock();
            let Some(s) = sample.stats.as_mut() else { continue };
            let queue = ((min.queue as i64 + s.queue as i64) / 2) as f64;
            let response = ((min.response_ns + s.response_ns) / 2.0).trunc();
            let service = ewma(s.service_ns, (min.service_ns + s.service_ns) / 2.0).trunc();
            s.queue = ewma(s.queue, queue);
            s.response_ns = ewma(s.response_ns, response);
            s.service_ns = service;
        }
        // One more outstanding request on the winner. ES starts this count
        // at 1 for a node it has none for, the same value it ranks such a
        // node with, so an idle winner's first increment changes nothing
        // and every shard of a search on an idle cluster goes to the same
        // node; counting from the value ranked makes each increment count.
        let next = effective(counts, &winner) + 1;
        counts.0.insert(winner, next);
    }
}
pub(super) struct Attempt { peer: Arc<Peer>, start: Instant }
impl Attempt {
    /// A query-phase answer: a successful one carrying the executing
    /// node's queue and service time feeds the statistics (ES
    /// `SearchExecutionStatsCollector.onResponse`); failures and answers
    /// without them only end the outstanding request.
    pub(super) fn finish_query(self, success: bool, reported: Option<(usize, f64)>) {
        if let (true, Some((queue, service_ns))) = (success, reported) {
            self.record(queue, self.start.elapsed(), service_ns);
        }
    }
    /// ES `ResponseCollectorService.addNodeStatistics`.
    fn record(&self, queue: usize, elapsed: Duration, service_ns: f64) {
        if service_ns <= 0.0 {
            return;
        }
        let response_ns = elapsed.as_nanos() as f64;
        let mut s = self.peer.0.lock();
        s.stats = Some(match s.stats {
            None => Stats { queue: queue as f64, response_ns, service_ns },
            Some(old) => Stats { queue: ewma(old.queue, queue as f64), response_ns: ewma(old.response_ns, response_ns), service_ns },
        });
    }
}
impl Drop for Attempt {
    fn drop(&mut self) {
        let mut s = self.peer.0.lock();
        s.outstanding = s.outstanding.saturating_sub(1);
    }
}

/// [`STATS_HEADER`]'s value for this node's search executor.
pub(super) fn local_stats() -> (usize, f64) {
    let search = elasticsearch_http::pools::executor(elasticsearch_http::pools::Pool::Search);
    (search.stats().queue, search.service_time_ewma_ns())
}

pub(super) fn encode_stats((queue, service_ns): (usize, f64)) -> String {
    format!("{queue},{}", service_ns as u64)
}

pub(super) fn decode_stats(value: &str) -> Option<(usize, f64)> {
    let (queue, service) = value.split_once(',')?;
    Some((queue.parse().ok()?, service.parse::<u64>().ok()? as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> ClusterState {
        let mut state = ClusterState::empty("adaptive-test");
        for id in ["a", "b"] {
            state.nodes.insert(id.into(), elasticsearch_cluster::state::DiscoveryNode {
                id:id.into(),name:id.into(),ephemeral_id:format!("{id}-1"),transport_address:String::new(),
                http_address:String::new(),host:String::new(),ip:String::new(),roles:Default::default(),
                attributes:advertised(),version:String::new(),
            });
        }
        state
    }
    fn advertised() -> std::collections::BTreeMap<String, String> {
        let mut attributes = Default::default();
        crate::internal_capabilities::advertise(&mut attributes);
        attributes
    }

    #[test]
    fn copies_on_nodes_without_stats_support_keep_rotation() {
        let mut state = state();
        // `b` runs an older build: it never reports stats.
        state.nodes.get_mut("b").unwrap().attributes = Default::default();
        let selection = Selection::default();
        feed(&selection, &state, "a", 0, Duration::from_millis(1), 1_000_000.0);
        let mut counts = selection.counts();
        for first in ["a", "b", "a", "b"] {
            let mut copies: Vec<String> = if first == "a" { vec!["a".into(), "b".into()] } else { vec!["b".into(), "a".into()] };
            selection.order(&state, &mut copies, &mut counts);
            assert_eq!(copies[0], first, "the old node does not take every shard");
        }
        assert!(counts.0.is_empty());
    }
    fn feed(selection: &Selection, state: &ClusterState, id: &str, queue: usize, response: Duration, service_ns: f64) {
        let attempt = selection.begin(state, id).unwrap();
        attempt.record(queue, response, service_ns);
    }
    /// Plan one search over `shards` shards whose copies are on `a` and `b`,
    /// each starting from the rotation `Plan::new` gives it.
    fn plan(selection: &Selection, state: &ClusterState, shards: usize, rotation: usize) -> Vec<String> {
        let mut counts = selection.counts();
        (0..shards).map(|k| {
            let mut copies: Vec<String> = if (rotation + k) % 2 == 0 { vec!["a".into(), "b".into()] } else { vec!["b".into(), "a".into()] };
            selection.order(state, &mut copies, &mut counts);
            copies[0].clone()
        }).collect()
    }

    #[test]
    fn shards_of_one_search_spread_over_nodes_with_equal_stats() {
        let state = state();
        let selection = Selection::default();
        for id in ["a", "b"] { feed(&selection, &state, id, 0, Duration::from_millis(2), 1_000_000.0); }
        // Whatever copy each shard's rotation starts at, the winner of the
        // first shard ranks worse for the second.
        for rotation in 0..4 {
            for start in [vec!["a", "a"], vec!["b", "b"]] {
                let mut counts = selection.counts();
                let winners: Vec<String> = start.iter().map(|first| {
                    let mut copies: Vec<String> = if *first == "a" { vec!["a".into(), "b".into()] } else { vec!["b".into(), "a".into()] };
                    selection.order(&state, &mut copies, &mut counts);
                    copies[0].clone()
                }).collect();
                assert_ne!(winners[0], winners[1], "rotation {rotation} start {start:?}");
            }
            let mut winners = plan(&selection, &state, 2, rotation);
            winners.sort();
            assert_eq!(winners, vec!["a", "b"]);
        }
    }

    #[test]
    fn coordinator_without_network_time_still_spreads_shards() {
        let state = state();
        let selection = Selection::default();
        // `a` coordinates: its own copy answers in its service time, the
        // peer's adds a network round trip.
        feed(&selection, &state, "a", 0, Duration::from_micros(1_000), 1_000_000.0);
        feed(&selection, &state, "b", 0, Duration::from_micros(1_400), 1_000_000.0);
        let mut winners = plan(&selection, &state, 2, 0);
        winners.sort();
        assert_eq!(winners, vec!["a", "b"]);
    }

    #[test]
    fn unchosen_nodes_move_toward_the_winner_and_get_queries_again() {
        let state = state();
        let selection = Selection::default();
        feed(&selection, &state, "a", 0, Duration::from_millis(1), 1_000_000.0);
        feed(&selection, &state, "b", 0, Duration::from_millis(400), 1_000_000.0);
        // Two searches in flight on `a`: not enough to outweigh `b`'s slow
        // sample at first, but `b` gets no new samples while it loses.
        let _busy: Vec<_> = (0..2).map(|_| selection.begin(&state, "a").unwrap()).collect();
        let rank_once = || {
            let mut copies = vec!["a".to_string(), "b".to_string()];
            selection.order(&state, &mut copies, &mut selection.counts());
            copies[0].clone()
        };
        assert_eq!(rank_once(), "a");
        let rounds = (1..100).find(|_| rank_once() == "b");
        assert!(rounds.is_some_and(|n| n < 20), "a losing node's sample is not frozen: {rounds:?}");
    }

    #[test]
    fn nodes_without_stats_rank_first_and_ties_keep_rotation() {
        let state = state();
        let selection = Selection::default();
        let mut copies = vec!["b".to_string(), "a".to_string()];
        selection.order(&state, &mut copies, &mut Counts::default());
        assert_eq!(copies, vec!["b", "a"]);
        feed(&selection, &state, "b", 0, Duration::from_millis(1), 1_000_000.0);
        selection.order(&state, &mut copies, &mut Counts::default());
        assert_eq!(copies, vec!["a", "b"]);
    }

    #[test]
    fn queue_and_outstanding_raise_rank() {
        let state = state();
        let selection = Selection::default();
        feed(&selection, &state, "a", 20, Duration::from_millis(1), 1_000_000.0);
        feed(&selection, &state, "b", 0, Duration::from_millis(1), 1_000_000.0);
        let mut copies = vec!["a".to_string(), "b".to_string()];
        selection.order(&state, &mut copies, &mut Counts::default());
        assert_eq!(copies, vec!["b", "a"]);
        let busy: Vec<_> = (0..30).map(|_| selection.begin(&state, "b").unwrap()).collect();
        let mut copies = vec!["b".to_string(), "a".to_string()];
        selection.order(&state, &mut copies, &mut selection.counts());
        assert_eq!(copies, vec!["a", "b"]);
        drop(busy);
        assert!(selection.counts().0.is_empty());
    }

    #[test]
    fn failures_and_answers_without_reported_stats_record_nothing() {
        let state = state();
        let selection = Selection::default();
        for (success, reported) in [(false, Some((0, 1e6))), (true, None), (false, None)] {
            selection.begin(&state, "a").unwrap().finish_query(success, reported);
        }
        let peer = selection.peer(&state, "a").unwrap();
        assert!(peer.0.lock().stats.is_none());
        assert_eq!(peer.0.lock().outstanding, 0);
        selection.begin(&state, "a").unwrap().finish_query(true, Some((3, 2e6)));
        let stats = peer.0.lock().stats.unwrap();
        assert_eq!((stats.queue, stats.service_ns), (3.0, 2e6));
    }

    #[test]
    fn restarted_peers_start_over() {
        let mut state = state();
        let selection = Selection::default();
        let stale = selection.begin(&state, "a").unwrap();
        state.nodes.get_mut("a").unwrap().ephemeral_id = "a-2".into();
        stale.finish_query(true, Some((0, 1e6)));
        assert!(selection.peer(&state, "a").unwrap().0.lock().stats.is_none());
        assert_eq!(selection.peers.lock().len(), 1);
    }

    #[test]
    fn telemetry_reports_es_fields() {
        let mut state=state();let selection=Selection::default();
        assert_eq!(selection.snapshot(&state),serde_json::json!({}));
        let attempt=selection.begin(&state,"a").unwrap();
        attempt.record(2, Duration::from_millis(12), 3_000_000.0);
        let stats=selection.snapshot(&state);
        assert_eq!(stats["a"]["outgoing_searches"],1);
        assert_eq!(stats["a"]["avg_response_time_ns"],12_000_000);
        assert_eq!(stats["a"]["avg_queue_size"],2);
        assert_eq!(stats["a"]["avg_service_time_ns"],3_000_000);
        assert!(stats["a"]["rank"].is_string());
        drop(attempt);
        assert_eq!(selection.snapshot(&state)["a"]["outgoing_searches"],0);
        state.nodes.remove("a");
        assert_eq!(selection.snapshot(&state),serde_json::json!({}));
    }

    #[test]
    fn stats_header_round_trips() {
        assert_eq!(decode_stats(&encode_stats((4, 1_500_000.0))), Some((4, 1_500_000.0)));
        assert_eq!(decode_stats("x"), None);
    }
}
