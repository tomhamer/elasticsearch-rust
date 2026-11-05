//! ES's `BalancedShardsAllocator` (the classic, non-desired-balance one) plus
//! the gateway allocators' decisions for existing-store primaries and
//! delayed replicas.
//!
//! Weight of node `n` for index `i` (ES `WeightFunction`):
//! `θ0·(shards(n) − avg) + θ1·(shards(n,i) − avg(i))` with
//! `θ0 = balance.shard/(shard+index)`, `θ1 = balance.index/(shard+index)`;
//! relocating sources are not counted (their target is).
//!
//! Each round: allocate unassigned shards (primaries first, then by index
//! priority / creation date / name, min-weight node among YES/THROTTLE
//! deciders, ES's tie-break), move shards that may not remain, then
//! rebalance per index from the heaviest to the lightest node while the
//! weight delta exceeds `balance.threshold`. Throttled moves are simulated
//! in the model only, as in ES, so one round never over-commits.
//!
//! Data safety: a primary with an `EXISTING_STORE` source is only ever
//! allocated onto a node holding a copy whose allocation id is in the
//! in-sync set (the copy keeps its allocation id); otherwise it stays
//! unassigned with `no_valid_shard_copy`.

use super::context::Allocation;
use super::deciders::{Decision, DecisionType};
use crate::state::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Yes,
    Throttled,
    No,
    /// Existing-store primary with no in-sync copy on any live node.
    NoValidCopy { found_stale: bool },
    AwaitingInfo,
    Delayed { remaining: i64, configured: i64 },
}

impl Outcome {
    /// `UnassignedInfo.AllocationStatus` value recorded on the shard.
    pub fn status(&self) -> &'static str {
        match self {
            Outcome::Yes => "no_attempt",
            Outcome::Throttled => "deciders_throttled",
            Outcome::No => "deciders_no",
            Outcome::NoValidCopy { .. } => "no_valid_shard_copy",
            Outcome::AwaitingInfo => "fetching_shard_data",
            Outcome::Delayed { .. } => "delayed_allocation",
        }
    }
    /// `AllocationDecision` (explain `can_allocate`).
    pub fn allocation_decision(&self) -> &'static str {
        match self {
            Outcome::Yes => "yes",
            Outcome::Throttled => "throttled",
            Outcome::No => "no",
            Outcome::NoValidCopy { .. } => "no_valid_shard_copy",
            Outcome::AwaitingInfo => "awaiting_info",
            Outcome::Delayed { .. } => "allocation_delayed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NodeResult {
    pub node: DiscoveryNode,
    pub decision: Decision,
    /// `yes`, `no`, `throttled`, `worse_balance`.
    pub node_decision: &'static str,
    pub weight_ranking: Option<usize>,
    pub store: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct UnassignedDecision {
    pub outcome: Outcome,
    pub target: Option<String>,
    /// Allocation id to reuse (existing store copy).
    pub allocation_id: Option<String>,
    pub nodes: Vec<NodeResult>,
    /// Decision came from the existing-store primary allocator.
    pub gateway_primary: bool,
}

#[derive(Clone, Debug)]
pub struct MoveDecision {
    pub can_remain: Decision,
    pub target: Option<String>,
    pub best: DecisionType,
    pub nodes: Vec<NodeResult>,
    /// Set when `can_remain` is YES and a rebalance was evaluated.
    pub rebalance: Option<Decision>,
    pub current_ranking: usize,
}

fn less_than(delta: f32, threshold: f32) -> bool {
    delta <= threshold + 0.001
}

impl Allocation<'_> {
    // ---------------------------------------------------------- weights --

    fn thetas(&self) -> (f32, f32) {
        let sum = self.s.balance_shard + self.s.balance_index;
        if sum <= 0.0 { (0.5, 0.5) } else { (self.s.balance_shard / sum, self.s.balance_index / sum) }
    }

    fn index_total_shards(&self, index: &str) -> f32 {
        self.meta(index).map(|m| (m.number_of_shards * (m.number_of_replicas + 1)) as f32).unwrap_or(0.0)
    }

    fn avgs(&self, index: &str) -> (f32, f32) {
        let n = self.state.data_nodes().count().max(1) as f32;
        let total: f32 = self.state.routing_table.indices.keys().map(|i| self.index_total_shards(i)).sum();
        (total / n, self.index_total_shards(index) / n)
    }

    /// Current weight of `node` for `index`, plus `added` hypothetical shards.
    pub fn weight(&self, node: &str, index: &str, added: i64) -> f32 {
        let (t0, t1) = self.thetas();
        let (avg, avg_i) = self.avgs(index);
        let ns = self.stats.node_shards.get(node).copied().unwrap_or(0) as i64 + added;
        let ni = self.stats.node_index_shards.get(&(node.to_string(), index.to_string())).copied().unwrap_or(0) as i64 + added;
        t0 * (ns as f32 - avg) + t1 * (ni as f32 - avg_i)
    }

    /// A missing observation disables the hint for the whole candidate set.
    /// Compare fractions exactly without clipping above-budget measurements.
    fn placement_memory_order(&self, a: &str, b: &str) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        if !self.state.data_nodes().all(|n| self.info.placement_memory.get(&n.id).is_some_and(|m| m.budget_bytes > 0)) {
            return Ordering::Equal;
        }
        let (Some(a), Some(b)) = (self.info.placement_memory.get(a), self.info.placement_memory.get(b)) else { return Ordering::Equal };
        (u128::from(a.measured_bytes) * u128::from(b.budget_bytes))
            .cmp(&(u128::from(b.measured_bytes) * u128::from(a.budget_bytes)))
    }

    // ------------------------------------------------ unassigned decision --

    pub fn decide_unassigned(&self, c: &ShardRouting, explain: bool) -> UnassignedDecision {
        if c.primary && matches!(c.recovery_source, Some(RecoverySource::ExistingStore { bootstrap_new_history_uuid: false })) {
            return self.decide_existing_primary(c, explain);
        }
        if !c.primary && c.unassigned_info.as_ref().is_some_and(|u| u.delayed) {
            let configured = self.configured_delay(c);
            let remaining = self.remaining_delay(c);
            if remaining > 0 {
                // the departed node (or another holding its data) may take it back
                if let Some(d) = self.decide_on_matching_node(c) {
                    if d.outcome != Outcome::No {
                        return d;
                    }
                }
                let mut d = if explain { self.decide_balanced(c, true) } else { UnassignedDecision { outcome: Outcome::No, target: None, allocation_id: None, nodes: vec![], gateway_primary: false } };
                d.outcome = Outcome::Delayed { remaining, configured };
                d.target = None;
                return d;
            }
        }
        if !c.primary {
            if let Some(d) = self.decide_on_matching_node(c) {
                if d.outcome == Outcome::Yes && !explain {
                    return d;
                }
            }
        }
        self.decide_balanced(c, explain)
    }

    /// Replica allocator: a node that already holds data for this shard
    /// (store listing, or the node it was on before leaving).
    fn decide_on_matching_node(&self, c: &ShardRouting) -> Option<UnassignedDecision> {
        let t = self.table(&c.index, c.shard)?;
        let holds: HashSet<&str> = t.copies.iter().filter_map(|x| x.node_id.as_deref()).collect();
        let mut cands: Vec<String> = self.stores.copies(&c.index, c.shard).iter().map(|s| s.node_id.clone()).collect();
        if let Some(n) = c.unassigned_info.as_ref().and_then(|u| u.last_node.clone()) {
            cands.push(n);
        }
        for n in cands {
            if holds.contains(n.as_str()) {
                continue;
            }
            let Some(node) = self.node(&n).filter(|n| n.is_data()) else { continue };
            let d = self.can_allocate(c, node);
            let outcome = match d.kind {
                DecisionType::Yes => Outcome::Yes,
                DecisionType::Throttle => Outcome::Throttled,
                DecisionType::No => Outcome::No,
            };
            return Some(UnassignedDecision { outcome: outcome.clone(), target: (outcome == Outcome::Yes).then(|| n.clone()), allocation_id: None, nodes: vec![], gateway_primary: false });
        }
        None
    }

    fn decide_existing_primary(&self, c: &ShardRouting, explain: bool) -> UnassignedDecision {
        let mut out = UnassignedDecision { outcome: Outcome::No, target: None, allocation_id: None, nodes: vec![], gateway_primary: true };
        if self.stores.is_pending(&c.index, c.shard) {
            out.outcome = Outcome::AwaitingInfo;
            return out;
        }
        // Stateless: the object store holds the shard, so any index node can
        // recover the primary (ES `StatelessExistingShardsAllocator`), with
        // a new allocation id; the node rebuilds the copy from the store.
        let index_nodes: Vec<DiscoveryNode> = self.data_nodes().into_iter().filter(|n| n.roles.contains(&Role::Index)).collect();
        if !index_nodes.is_empty() {
            let mut throttled = false;
            for node in &index_nodes {
                let d = self.can_allocate(c, node);
                match d.kind {
                    DecisionType::Yes if out.target.is_none() => {
                        out.target = Some(node.id.clone());
                        out.outcome = Outcome::Yes;
                    }
                    DecisionType::Throttle => throttled = true,
                    _ => {}
                }
            }
            if out.target.is_none() {
                out.outcome = if throttled { Outcome::Throttled } else { Outcome::No };
            }
            out.gateway_primary = false;
            return out;
        }
        let in_sync = self.meta(&c.index).and_then(|m| m.in_sync_allocation_ids.get(&c.shard).cloned()).unwrap_or_default();
        let listed: Vec<&super::StoreCopy> = self.stores.copies(&c.index, c.shard).iter().filter(|s| self.node(&s.node_id).is_some_and(|n| n.is_data())).collect();
        let copies: BTreeMap<String, Option<String>> = listed.iter().map(|s| (s.node_id.clone(), s.allocation_id.clone())).collect();
        // in-sync copies only; former primaries first, then node id (ES PRIMARY_FIRST_COMPARATOR)
        let mut ranked: Vec<&&super::StoreCopy> = listed.iter().filter(|s| s.allocation_id.as_ref().is_some_and(|a| in_sync.contains(a))).collect();
        ranked.sort_by(|a, b| b.primary.cmp(&a.primary).then(a.node_id.cmp(&b.node_id)));
        let valid: Vec<(String, String)> = ranked.iter().map(|s| (s.node_id.clone(), s.allocation_id.clone().unwrap_or_default())).collect();
        let mut decisions: HashMap<String, Decision> = HashMap::new();
        if valid.is_empty() {
            out.outcome = Outcome::NoValidCopy { found_stale: !copies.is_empty() };
        } else {
            let mut throttled = false;
            for (n, aid) in &valid {
                let Some(node) = self.node(n) else { continue };
                let d = self.can_allocate(c, node);
                let kind = d.kind;
                decisions.insert(n.clone(), d);
                if kind == DecisionType::Yes && out.target.is_none() {
                    out.target = Some(n.clone());
                    out.allocation_id = Some(aid.clone());
                    if !explain {
                        break;
                    }
                }
                throttled |= kind == DecisionType::Throttle;
            }
            if out.target.is_some() {
                out.outcome = Outcome::Yes;
            } else if throttled {
                out.outcome = Outcome::Throttled;
            } else {
                for (n, aid) in &valid {
                    let Some(node) = self.node(n) else { continue };
                    let d = self.can_force_allocate_primary(c, node);
                    if d.kind == DecisionType::Yes {
                        out.outcome = Outcome::Yes;
                        out.target = Some(n.clone());
                        out.allocation_id = Some(aid.clone());
                        break;
                    }
                    throttled |= d.kind == DecisionType::Throttle;
                }
                if out.target.is_none() {
                    out.outcome = if throttled { Outcome::Throttled } else { Outcome::No };
                }
            }
        }
        if explain {
            let mut rows: Vec<(u8, NodeResult)> = Vec::new();
            for node in self.data_nodes() {
                let store = match copies.get(&node.id) {
                    Some(Some(aid)) => json!({"in_sync": in_sync.contains(aid), "allocation_id": aid}),
                    Some(None) => json!({"in_sync": false}),
                    None => json!({"found": false}),
                };
                let (rank, decision, nd) = match decisions.remove(&node.id) {
                    Some(d) => {
                        let nd = d.kind.allocation_decision();
                        (0, d, nd)
                    }
                    None => (if copies.contains_key(&node.id) { 1 } else { 2 }, Decision { kind: DecisionType::No, singles: vec![] }, "no"),
                };
                rows.push((rank, NodeResult { node, decision, node_decision: nd, weight_ranking: None, store: Some(store) }));
            }
            rows.sort_by_key(|(r, _)| *r);
            out.nodes = rows.into_iter().map(|(_, r)| r).collect();
        }
        out
    }

    /// `decideAllocateUnassigned`: min weight-with-shard-added among nodes
    /// whose deciders say YES or THROTTLE.
    pub fn decide_balanced(&self, c: &ShardRouting, explain: bool) -> UnassignedDecision {
        let holds: HashSet<String> = self.table(&c.index, c.shard).map(|t| t.copies.iter().filter_map(|x| x.node_id.clone()).collect()).unwrap_or_default();
        let mut min_weight = f32::INFINITY;
        let mut best: Option<(String, Decision)> = None;
        let mut results: Vec<(f32, NodeResult)> = Vec::new();
        for node in self.data_nodes() {
            if holds.contains(&node.id) && !explain {
                continue;
            }
            let w = self.weight(&node.id, &c.index, 1);
            if w > min_weight && !explain {
                continue;
            }
            let d = self.can_allocate(c, &node);
            if d.kind != DecisionType::No {
                let update = match &best {
                    None => true,
                    Some((bn, bd)) if w == min_weight => {
                        if d.kind == bd.kind {
                            // ES tie-break: prefer the node whose highest primary of
                            // this index makes the shard ids form a ring.
                            let rep = c.shard as i64;
                            let hp = |n: &str| self.stats.highest_primary.get(&(n.to_string(), c.index.clone())).copied().unwrap_or(-1);
                            let (nh, mh) = (hp(&node.id), hp(bn));
                            match self.placement_memory_order(&node.id, bn) {
                                std::cmp::Ordering::Less => true,
                                std::cmp::Ordering::Greater => false,
                                std::cmp::Ordering::Equal => (((nh > rep && mh > rep) || (nh < rep && mh < rep)) && nh < mh) || (nh > rep && mh < rep),
                            }
                        } else {
                            d.kind == DecisionType::Yes
                        }
                    }
                    Some(_) => w < min_weight,
                };
                if update {
                    min_weight = w;
                    best = Some((node.id.clone(), d.clone()));
                }
            }
            if explain {
                let nd = d.kind.allocation_decision();
                results.push((w, NodeResult { node, decision: d, node_decision: nd, weight_ranking: None, store: None }));
            }
        }
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let nodes = results.into_iter().enumerate().map(|(i, (_, mut r))| {
            r.weight_ranking = Some(i + 1);
            r
        }).collect();
        match best {
            Some((n, d)) if d.kind == DecisionType::Yes => UnassignedDecision { outcome: Outcome::Yes, target: Some(n), allocation_id: None, nodes, gateway_primary: false },
            Some(_) => UnassignedDecision { outcome: Outcome::Throttled, target: None, allocation_id: None, nodes, gateway_primary: false },
            None => UnassignedDecision { outcome: Outcome::No, target: None, allocation_id: None, nodes, gateway_primary: false },
        }
    }

    // ------------------------------------------------- allocate unassigned --

    pub fn allocate_unassigned(&mut self) {
        let mut todo: Vec<(bool, i64, i64, String, u32, usize)> = Vec::new();
        for (name, irt) in &self.state.routing_table.indices {
            let meta = self.meta(name);
            let priority = meta.map(|m| super::settings::index_i64(m, "index.priority", 1)).unwrap_or(1);
            let created = meta.map(|m| m.creation_date).unwrap_or(0);
            for t in &irt.shards {
                for (pos, c) in t.copies.iter().enumerate() {
                    if c.state == ShardState::Unassigned {
                        todo.push((c.primary, priority, created, name.clone(), t.shard, pos));
                    }
                }
            }
        }
        todo.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)).then(b.3.cmp(&a.3)).then(a.4.cmp(&b.4)).then(a.5.cmp(&b.5)));
        // a NO for one replica of a shard is the answer for its siblings
        let mut replica_no: HashMap<(String, u32), &'static str> = HashMap::new();
        for (primary, _, _, index, shard, pos) in todo {
            let Some(c) = self.table(&index, shard).and_then(|t| t.copies.get(pos)).filter(|c| c.state == ShardState::Unassigned).cloned() else { continue };
            if !primary {
                if let Some(st) = replica_no.get(&(index.clone(), shard)) {
                    self.set_allocation_status(&index, shard, pos, st);
                    continue;
                }
            }
            let d = self.decide_unassigned(&c, false);
            match (&d.outcome, &d.target) {
                (Outcome::Yes, Some(node)) => {
                    let node = node.clone();
                    self.initialize(&index, shard, pos, &node, d.allocation_id.clone(), None, None);
                }
                (o, _) => {
                    let st = if *o == Outcome::Yes { "deciders_no" } else { o.status() };
                    if !primary && *o == Outcome::No {
                        replica_no.insert((index.clone(), shard), st);
                    }
                    self.set_allocation_status(&index, shard, pos, st);
                }
            }
        }
        self.unassigned_ignored = true;
    }

    // ------------------------------------------------------- move shards --

    pub fn decide_move(&self, c: &ShardRouting, explain: bool) -> MoveDecision {
        let cur = c.node_id.clone().unwrap_or_default();
        let can_remain = self.node(&cur).map(|n| self.can_remain(c, n)).unwrap_or_else(Decision::yes);
        let mut out = MoveDecision { can_remain, target: None, best: DecisionType::No, nodes: vec![], rebalance: None, current_ranking: 0 };
        if out.can_remain.kind != DecisionType::No && !explain {
            return out;
        }
        if out.can_remain.kind != DecisionType::No {
            return self.decide_rebalance(c, out);
        }
        let mut nodes: Vec<(f32, DiscoveryNode)> = self.data_nodes().into_iter().filter(|n| n.id != cur).map(|n| (self.weight(&n.id, &c.index, 0), n)).collect();
        nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (_, node)) in nodes.into_iter().enumerate() {
            let d = self.can_allocate(c, &node);
            if d.kind > out.best || (out.target.is_none() && d.kind == DecisionType::Yes) {
                out.best = d.kind;
                if d.kind == DecisionType::Yes && out.target.is_none() {
                    out.target = Some(node.id.clone());
                    if !explain {
                        break;
                    }
                }
            }
            if explain {
                let nd = d.kind.allocation_decision();
                out.nodes.push(NodeResult { node, decision: d, node_decision: nd, weight_ranking: Some(i + 1), store: None });
            }
        }
        out
    }

    /// `decideRebalance` (explain only): would moving this started shard to
    /// another node improve balance, and may it?
    fn decide_rebalance(&self, c: &ShardRouting, mut out: MoveDecision) -> MoveDecision {
        let cur = c.node_id.clone().unwrap_or_default();
        let rebalance = self.can_rebalance_shard(c);
        let threshold = self.s.balance_threshold;
        let current_weight = self.weight(&cur, &c.index, 0);
        let mut best_kind = DecisionType::No;
        let (mut better, mut same, mut worse) = (vec![], vec![], vec![]);
        for node in self.data_nodes() {
            if node.id == cur {
                continue;
            }
            let mut d = self.can_allocate(c, &node);
            d.merge(self.can_rebalance_to(&node));
            let w = self.weight(&node.id, &c.index, 0);
            let better_weight = w <= current_weight;
            let mut met = false;
            if better_weight {
                let delta = (w - current_weight).abs();
                let above = !less_than(delta, threshold);
                let proposed = 2.0 + w - current_weight;
                met = above && proposed < delta;
                if met && d.kind > best_kind {
                    best_kind = d.kind;
                    out.target = Some(node.id.clone());
                }
            }
            if met {
                better.push((node, d));
            } else if better_weight {
                same.push((node, d));
            } else {
                worse.push((node, d));
            }
        }
        if best_kind == DecisionType::No {
            out.target = None;
        }
        let mut rank = 0;
        for (node, d) in better {
            rank += 1;
            let nd = d.kind.allocation_decision();
            out.nodes.push(NodeResult { node, decision: d, node_decision: nd, weight_ranking: Some(rank), store: None });
        }
        rank += 1;
        out.current_ranking = rank;
        for (node, d) in same {
            let nd = if d.kind == DecisionType::No { "no" } else { "worse_balance" };
            out.nodes.push(NodeResult { node, decision: d, node_decision: nd, weight_ranking: Some(rank), store: None });
        }
        for (node, d) in worse {
            rank += 1;
            let nd = if d.kind == DecisionType::No { "no" } else { "worse_balance" };
            out.nodes.push(NodeResult { node, decision: d, node_decision: nd, weight_ranking: Some(rank), store: None });
        }
        out.best = best_kind;
        out.rebalance = Some(rebalance);
        out
    }

    pub fn move_shards(&mut self) {
        let mut cands: Vec<(String, String, u32, String)> = Vec::new();
        for (name, irt) in &self.state.routing_table.indices {
            for t in &irt.shards {
                for c in &t.copies {
                    if c.state == ShardState::Started {
                        if let (Some(n), Some(a)) = (&c.node_id, c.allocation_id()) {
                            cands.push((n.clone(), name.clone(), t.shard, a.to_string()));
                        }
                    }
                }
            }
        }
        cands.sort();
        for (node, index, shard, aid) in cands {
            let Some(c) = self.pos_by_aid(&index, shard, &aid).and_then(|p| self.table(&index, shard).map(|t| t.copies[p].clone())) else { continue };
            if c.state != ShardState::Started || c.node_id.as_deref() != Some(node.as_str()) {
                continue;
            }
            let Some(n) = self.node(&node) else { continue };
            if self.can_remain(&c, n).kind != DecisionType::No {
                continue;
            }
            let d = self.decide_move(&c, false);
            if let Some(t) = d.target {
                self.relocate(&index, shard, &aid, &t);
            }
        }
    }

    // ---------------------------------------------------------- rebalance --

    pub fn rebalance(&mut self) {
        if self.stores.has_pending() || self.can_rebalance().kind != DecisionType::Yes {
            return;
        }
        let nodes: Vec<DiscoveryNode> = self.data_nodes();
        if nodes.len() < 2 {
            return;
        }
        let (t0, t1) = self.thetas();
        let threshold = self.s.balance_threshold;
        let n = nodes.len() as f32;
        let indices: Vec<String> = self.state.routing_table.indices.keys().cloned().collect();
        let avg = indices.iter().map(|i| self.index_total_shards(i)).sum::<f32>() / n;
        // the model: counts per node / per (node, index), updated for real and simulated moves
        let mut total: HashMap<String, i64> = nodes.iter().map(|x| (x.id.clone(), self.stats.node_shards.get(&x.id).copied().unwrap_or(0) as i64)).collect();
        let mut per_index: HashMap<(String, String), i64> = self.stats.node_index_shards.iter().map(|(k, v)| (k.clone(), *v as i64)).collect();
        let mut simulated: HashSet<String> = HashSet::new();
        let weight = |total: &HashMap<String, i64>, per: &HashMap<(String, String), i64>, node: &str, index: &str, avg_i: f32| -> f32 {
            let ns = total.get(node).copied().unwrap_or(0) as f32;
            let ni = per.get(&(node.to_string(), index.to_string())).copied().unwrap_or(0) as f32;
            t0 * (ns - avg) + t1 * (ni - avg_i)
        };
        // indices with the largest weight spread first
        let mut ordered: Vec<(f32, String)> = indices
            .iter()
            .map(|i| {
                let avg_i = self.index_total_shards(i) / n;
                let ws: Vec<f32> = nodes.iter().map(|x| weight(&total, &per_index, &x.id, i, avg_i)).collect();
                let max = ws.iter().cloned().fold(f32::MIN, f32::max);
                let min = ws.iter().cloned().fold(f32::MAX, f32::min);
                (max - min, i.clone())
            })
            .collect();
        ordered.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        for (_, index) in ordered {
            let Some(meta) = self.meta(&index).cloned() else { continue };
            let avg_i = self.index_total_shards(&index) / n;
            let mut relevant: Vec<String> = nodes
                .iter()
                .filter(|x| per_index.get(&(x.id.clone(), index.clone())).copied().unwrap_or(0) > 0 || self.can_allocate_index(&meta, x).kind != DecisionType::No)
                .map(|x| x.id.clone())
                .collect();
            if relevant.len() < 2 {
                continue;
            }
            let sort = |relevant: &mut Vec<String>, total: &HashMap<String, i64>, per: &HashMap<(String, String), i64>| -> Vec<f32> {
                relevant.sort_by(|a, b| weight(total, per, a, &index, avg_i).partial_cmp(&weight(total, per, b, &index, avg_i)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
                relevant.iter().map(|x| weight(total, per, x, &index, avg_i)).collect()
            };
            let mut weights = sort(&mut relevant, &total, &per_index);
            let (mut low, mut high) = (0usize, relevant.len() - 1);
            let mut guard = 0;
            loop {
                guard += 1;
                if guard > 100_000 {
                    break;
                }
                let (min_node, max_node) = (relevant[low].clone(), relevant[high].clone());
                if per_index.get(&(max_node.clone(), index.clone())).copied().unwrap_or(0) > 0 {
                    let delta = (weights[high] - weights[low]).abs();
                    if less_than(delta, threshold) {
                        if low > 0 && high > 1 && (weights[0] - weights[high - 1]).abs() > threshold {
                            low = 0;
                            high -= 1;
                            continue;
                        }
                        break;
                    }
                    // try_relocate_shard(min_node, max_node, index)
                    let mut cands: Vec<ShardRouting> = self
                        .state
                        .routing_table
                        .indices
                        .get(&index)
                        .map(|irt| irt.shards.iter().flat_map(|t| t.copies.iter()).filter(|c| c.state == ShardState::Started && c.node_id.as_deref() == Some(max_node.as_str()) && !c.allocation_id().is_some_and(|a| simulated.contains(a))).cloned().collect())
                        .unwrap_or_default();
                    cands.sort_by(|a, b| b.shard.cmp(&a.shard));
                    let mut relocated = false;
                    let mut moved = false;
                    let Some(target) = self.node(&min_node).cloned() else { break };
                    if self.can_rebalance_to(&target).kind == DecisionType::No {
                        cands.clear();
                    }
                    for c in cands {
                        let rb = self.can_rebalance_shard(&c);
                        if rb.kind == DecisionType::No {
                            continue;
                        }
                        let al = self.can_allocate(&c, &target);
                        if al.kind == DecisionType::No {
                            continue;
                        }
                        let kind = rb.kind.min(al.kind);
                        let aid = c.allocation_id().unwrap_or_default().to_string();
                        *total.entry(max_node.clone()).or_default() -= 1;
                        *per_index.entry((max_node.clone(), index.clone())).or_default() -= 1;
                        *total.entry(min_node.clone()).or_default() += 1;
                        *per_index.entry((min_node.clone(), index.clone())).or_default() += 1;
                        simulated.insert(aid.clone());
                        moved = true;
                        if kind == DecisionType::Yes {
                            if let Some(tid) = self.relocate(&index, c.shard, &aid, &min_node) {
                                simulated.insert(tid);
                            }
                            relocated = true;
                        }
                        break;
                    }
                    if relocated {
                        weights = sort(&mut relevant, &total, &per_index);
                        low = 0;
                        high = relevant.len() - 1;
                        continue;
                    }
                    let _ = moved;
                }
                if low + 1 < high {
                    low += 1;
                } else if low > 0 {
                    low = 0;
                    high -= 1;
                } else {
                    break;
                }
            }
        }
    }

    /// `removeDelayMarkers`: delayed shards whose delay expired.
    pub fn remove_delay_markers(&mut self) {
        let mut expired = Vec::new();
        for (name, irt) in &self.state.routing_table.indices {
            for t in &irt.shards {
                for (pos, c) in t.copies.iter().enumerate() {
                    if c.state == ShardState::Unassigned && c.unassigned_info.as_ref().is_some_and(|u| u.delayed) && self.remaining_delay(c) == 0 {
                        expired.push((name.clone(), t.shard, pos));
                    }
                }
            }
        }
        for (i, s, p) in expired {
            if let Some(u) = self.state.routing_table.indices.get_mut(&i).and_then(|x| x.shards.get_mut(s as usize)).and_then(|t| t.copies.get_mut(p)).and_then(|c| c.unassigned_info.as_mut()) {
                u.delayed = false;
            }
        }
    }
}
