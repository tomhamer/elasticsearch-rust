//! `_cluster/health` and the `_cat/{health,shards,allocation}` row builders.
//!
//! Status rules are ES's `ClusterStateHealth` / `ClusterIndexHealth` /
//! `ClusterShardHealth`:
//! * a shard is GREEN when its primary is active and every copy is active,
//!   YELLOW when the primary is active but some replica is not;
//! * an inactive primary is YELLOW only while it is brand new (recovering
//!   from `EMPTY_STORE`, `LOCAL_SHARDS` or `SNAPSHOT`, never failed and not
//!   refused by the deciders), otherwise RED;
//! * an index / the cluster takes the worst status of its shards / indices;
//!   a global `SERVICE_UNAVAILABLE` block (state not recovered, no master)
//!   makes the cluster RED;
//! * `active_shards_percent_as_number` is 100 when green, otherwise active
//!   over all shard copies in the routing table (cluster-wide, ES quirk).
//!
//! Closed indices have routing tables (ES 7.2+) and count like open ones.
//! Relocation targets are not separate copies for health or `_cat/shards`
//! (ES keeps them out of the routing table's shard lists).

use crate::allocation::ClusterInfo;
use crate::allocation::settings::{fmt_bytes, fmt_bytes_in, fmt_date_millis, fmt_hms, fmt_time_millis};
use crate::state::*;
use serde_json::{Map, Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HealthStatus {
    Green,
    Yellow,
    Red,
}

impl HealthStatus {
    pub fn name(self) -> &'static str {
        match self {
            HealthStatus::Green => "green",
            HealthStatus::Yellow => "yellow",
            HealthStatus::Red => "red",
        }
    }
    pub fn parse(s: &str) -> Option<HealthStatus> {
        match s.trim().to_ascii_lowercase().as_str() {
            "green" => Some(HealthStatus::Green),
            "yellow" => Some(HealthStatus::Yellow),
            "red" => Some(HealthStatus::Red),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Cluster,
    Indices,
    Shards,
}

impl Level {
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "cluster" => Some(Level::Cluster),
            "indices" => Some(Level::Indices),
            "shards" => Some(Level::Shards),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShardHealth {
    pub shard: u32,
    pub status: HealthStatus,
    pub primary_active: bool,
    pub active_shards: u32,
    pub relocating_shards: u32,
    pub initializing_shards: u32,
    pub unassigned_shards: u32,
    pub unassigned_primary_shards: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexHealth {
    pub index: String,
    pub status: HealthStatus,
    pub number_of_shards: u32,
    pub number_of_replicas: u32,
    pub active_primary_shards: u32,
    pub active_shards: u32,
    pub relocating_shards: u32,
    pub initializing_shards: u32,
    pub unassigned_shards: u32,
    pub unassigned_primary_shards: u32,
    pub shards: Vec<ShardHealth>,
}

/// Inputs that live outside the cluster state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HealthInputs {
    pub timed_out: bool,
    pub number_of_pending_tasks: u64,
    pub number_of_in_flight_fetch: u64,
    pub task_max_waiting_in_queue_millis: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterHealth {
    pub cluster_name: String,
    pub status: HealthStatus,
    pub timed_out: bool,
    pub number_of_nodes: u32,
    pub number_of_data_nodes: u32,
    pub active_primary_shards: u32,
    pub active_shards: u32,
    pub relocating_shards: u32,
    pub initializing_shards: u32,
    pub unassigned_shards: u32,
    pub unassigned_primary_shards: u32,
    pub delayed_unassigned_shards: u32,
    pub number_of_pending_tasks: u64,
    pub number_of_in_flight_fetch: u64,
    pub task_max_waiting_in_queue_millis: u64,
    pub active_shards_percent: f64,
    pub indices: Vec<IndexHealth>,
}

fn is_relocation_target(c: &ShardRouting) -> bool {
    c.state == ShardState::Initializing && c.relocating_node_id.is_some()
}

/// `ClusterShardHealth.getInactivePrimaryHealth`.
fn inactive_primary_health(p: &ShardRouting) -> HealthStatus {
    let Some(u) = &p.unassigned_info else { return HealthStatus::Red };
    let new_source = matches!(p.recovery_source, Some(RecoverySource::EmptyStore) | Some(RecoverySource::LocalShards) | Some(RecoverySource::Snapshot { .. }));
    if u.last_allocation_status != "deciders_no" && u.failed_allocations == 0 && new_source { HealthStatus::Yellow } else { HealthStatus::Red }
}

pub fn shard_health(t: &IndexShardRoutingTable) -> ShardHealth {
    let mut h = ShardHealth { shard: t.shard, status: HealthStatus::Red, primary_active: false, active_shards: 0, relocating_shards: 0, initializing_shards: 0, unassigned_shards: 0, unassigned_primary_shards: 0 };
    let mut total = 0;
    for c in t.copies.iter().filter(|c| !is_relocation_target(c)) {
        total += 1;
        match c.state {
            ShardState::Started => h.active_shards += 1,
            ShardState::Relocating => {
                h.active_shards += 1;
                h.relocating_shards += 1;
            }
            ShardState::Initializing => h.initializing_shards += 1,
            ShardState::Unassigned => {
                h.unassigned_shards += 1;
                if c.primary {
                    h.unassigned_primary_shards += 1;
                }
            }
        }
    }
    let primary = t.copies.iter().find(|c| c.primary && !is_relocation_target(c));
    h.primary_active = primary.is_some_and(|p| p.active());
    h.status = match primary {
        None => HealthStatus::Red,
        Some(p) if p.active() => {
            if h.active_shards == total { HealthStatus::Green } else { HealthStatus::Yellow }
        }
        Some(p) => inactive_primary_health(p),
    };
    h
}

pub fn index_health(meta: &IndexMetadata, irt: &IndexRoutingTable) -> IndexHealth {
    let shards: Vec<ShardHealth> = irt.shards.iter().map(shard_health).collect();
    let mut h = IndexHealth {
        index: meta.name.clone(),
        status: if shards.is_empty() { HealthStatus::Red } else { HealthStatus::Green },
        number_of_shards: meta.number_of_shards,
        number_of_replicas: meta.number_of_replicas,
        active_primary_shards: 0,
        active_shards: 0,
        relocating_shards: 0,
        initializing_shards: 0,
        unassigned_shards: 0,
        unassigned_primary_shards: 0,
        shards: vec![],
    };
    for s in &shards {
        if s.primary_active {
            h.active_primary_shards += 1;
        }
        h.active_shards += s.active_shards;
        h.relocating_shards += s.relocating_shards;
        h.initializing_shards += s.initializing_shards;
        h.unassigned_shards += s.unassigned_shards;
        h.unassigned_primary_shards += s.unassigned_primary_shards;
        h.status = h.status.max(s.status);
    }
    h.shards = shards;
    h
}

/// Computes `_cluster/health` over `indices` (concrete names resolved by the
/// caller; `None` = every index).
pub fn cluster_health(state: &ClusterState, indices: Option<&[String]>, inputs: HealthInputs) -> ClusterHealth {
    let names: Vec<String> = match indices {
        Some(i) => i.to_vec(),
        None => state.metadata.indices.keys().cloned().collect(),
    };
    let mut h = ClusterHealth {
        cluster_name: state.cluster_name.clone(),
        status: HealthStatus::Green,
        timed_out: inputs.timed_out,
        number_of_nodes: state.nodes.len() as u32,
        number_of_data_nodes: state.data_nodes().count() as u32,
        active_primary_shards: 0,
        active_shards: 0,
        relocating_shards: 0,
        initializing_shards: 0,
        unassigned_shards: 0,
        unassigned_primary_shards: 0,
        delayed_unassigned_shards: state.routing_table.all_shards().filter(|c| c.state == ShardState::Unassigned && c.unassigned_info.as_ref().is_some_and(|u| u.delayed)).count() as u32,
        number_of_pending_tasks: inputs.number_of_pending_tasks,
        number_of_in_flight_fetch: inputs.number_of_in_flight_fetch,
        task_max_waiting_in_queue_millis: inputs.task_max_waiting_in_queue_millis,
        active_shards_percent: 100.0,
        indices: vec![],
    };
    for name in names {
        let (Some(meta), Some(irt)) = (state.metadata.indices.get(&name), state.routing_table.indices.get(&name)) else { continue };
        let ih = index_health(meta, irt);
        h.active_primary_shards += ih.active_primary_shards;
        h.active_shards += ih.active_shards;
        h.relocating_shards += ih.relocating_shards;
        h.initializing_shards += ih.initializing_shards;
        h.unassigned_shards += ih.unassigned_shards;
        h.unassigned_primary_shards += ih.unassigned_primary_shards;
        h.status = h.status.max(ih.status);
        h.indices.push(ih);
    }
    // SERVICE_UNAVAILABLE global blocks: state not recovered (1), no master (2)
    if state.blocks.global.iter().any(|b| b.id == 1 || b.id == 2) {
        h.status = HealthStatus::Red;
    }
    if h.status != HealthStatus::Green {
        let (mut active, mut total) = (0u64, 0u64);
        for c in state.routing_table.all_shards().filter(|c| !is_relocation_target(c)) {
            total += 1;
            if c.active() {
                active += 1;
            }
        }
        h.active_shards_percent = if total == 0 { 100.0 } else { active as f64 / total as f64 * 100.0 };
    }
    h
}

fn pct_str(p: f64) -> String {
    format!("{:.1}%", p)
}

impl ClusterHealth {
    /// 408 when `timed_out`, else 200 (ES `ClusterHealthResponse.status`).
    pub fn http_status(&self) -> u16 {
        if self.timed_out { 408 } else { 200 }
    }

    /// The response body in ES's field order. `human` adds the
    /// human-readable twins (`task_max_waiting_in_queue`, `active_shards_percent`).
    pub fn to_json(&self, level: Level, human: bool) -> Value {
        let mut m = Map::new();
        m.insert("cluster_name".into(), json!(self.cluster_name));
        m.insert("status".into(), json!(self.status.name()));
        m.insert("timed_out".into(), json!(self.timed_out));
        m.insert("number_of_nodes".into(), json!(self.number_of_nodes));
        m.insert("number_of_data_nodes".into(), json!(self.number_of_data_nodes));
        m.insert("active_primary_shards".into(), json!(self.active_primary_shards));
        m.insert("active_shards".into(), json!(self.active_shards));
        m.insert("relocating_shards".into(), json!(self.relocating_shards));
        m.insert("initializing_shards".into(), json!(self.initializing_shards));
        m.insert("unassigned_shards".into(), json!(self.unassigned_shards));
        m.insert("unassigned_primary_shards".into(), json!(self.unassigned_primary_shards));
        m.insert("delayed_unassigned_shards".into(), json!(self.delayed_unassigned_shards));
        m.insert("number_of_pending_tasks".into(), json!(self.number_of_pending_tasks));
        m.insert("number_of_in_flight_fetch".into(), json!(self.number_of_in_flight_fetch));
        if human {
            m.insert("task_max_waiting_in_queue".into(), json!(fmt_time_millis(self.task_max_waiting_in_queue_millis as i64)));
        }
        m.insert("task_max_waiting_in_queue_millis".into(), json!(self.task_max_waiting_in_queue_millis));
        if human {
            m.insert("active_shards_percent".into(), json!(pct_str(self.active_shards_percent)));
        }
        m.insert("active_shards_percent_as_number".into(), json!(self.active_shards_percent));
        if level != Level::Cluster {
            let mut idx = Map::new();
            for i in &self.indices {
                let mut im = Map::new();
                im.insert("status".into(), json!(i.status.name()));
                im.insert("number_of_shards".into(), json!(i.number_of_shards));
                im.insert("number_of_replicas".into(), json!(i.number_of_replicas));
                im.insert("active_primary_shards".into(), json!(i.active_primary_shards));
                im.insert("active_shards".into(), json!(i.active_shards));
                im.insert("relocating_shards".into(), json!(i.relocating_shards));
                im.insert("initializing_shards".into(), json!(i.initializing_shards));
                im.insert("unassigned_shards".into(), json!(i.unassigned_shards));
                im.insert("unassigned_primary_shards".into(), json!(i.unassigned_primary_shards));
                if level == Level::Shards {
                    let mut sm = Map::new();
                    for s in &i.shards {
                        sm.insert(
                            s.shard.to_string(),
                            json!({
                                "status": s.status.name(),
                                "primary_active": s.primary_active,
                                "active_shards": s.active_shards,
                                "relocating_shards": s.relocating_shards,
                                "initializing_shards": s.initializing_shards,
                                "unassigned_shards": s.unassigned_shards,
                                "unassigned_primary_shards": s.unassigned_primary_shards,
                            }),
                        );
                    }
                    im.insert("shards".into(), Value::Object(sm));
                }
                idx.insert(i.index.clone(), Value::Object(im));
            }
            m.insert("indices".into(), Value::Object(idx));
        }
        Value::Object(m)
    }
}

// ------------------------------------------------------------ wait_for_* --

/// `wait_for_active_shards` of the health API (`"all"` or a count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveShardCount {
    All,
    Count(u32),
}

impl ActiveShardCount {
    /// `"all"`, `"default"`/`"0"` (= none), or a non-negative number.
    pub fn parse(s: &str) -> Result<Option<ActiveShardCount>, String> {
        match s.trim() {
            "all" => Ok(Some(ActiveShardCount::All)),
            "default" | "0" | "" => Ok(None),
            n => n.parse::<u32>().map(|v| Some(ActiveShardCount::Count(v))).map_err(|_| format!("cannot parse ActiveShardCount[{n}]")),
        }
    }
}

/// The `wait_for_*` conditions of a health request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaitConditions {
    pub status: Option<HealthStatus>,
    /// `">=3"`, `"ge(3)"`, `"<5"`, `"lt(5)"`, `"3"`, ...
    pub nodes: Option<String>,
    pub active_shards: Option<ActiveShardCount>,
    pub no_relocating_shards: bool,
    pub no_initializing_shards: bool,
}

/// ES's `wait_for_nodes` expression check.
pub fn nodes_condition_met(expr: &str, number_of_nodes: u32) -> Result<bool, String> {
    let e = expr.trim();
    let num = |s: &str| s.trim().parse::<i64>().map_err(|_| format!("For input string: \"{}\"", s.trim()));
    let n = number_of_nodes as i64;
    let inner = |s: &str, p: &str| s.strip_prefix(p).and_then(|r| r.strip_suffix(')')).map(String::from);
    Ok(if let Some(v) = e.strip_prefix(">=") {
        n >= num(v)?
    } else if let Some(v) = inner(e, "ge(") {
        n >= num(&v)?
    } else if let Some(v) = e.strip_prefix("<=") {
        n <= num(v)?
    } else if let Some(v) = inner(e, "le(") {
        n <= num(&v)?
    } else if let Some(v) = e.strip_prefix('>') {
        n > num(v)?
    } else if let Some(v) = inner(e, "gt(") {
        n > num(&v)?
    } else if let Some(v) = e.strip_prefix('<') {
        n < num(v)?
    } else if let Some(v) = inner(e, "lt(") {
        n < num(&v)?
    } else {
        n == num(e)?
    })
}

/// Whether every requested `wait_for_*` condition holds (ES
/// `TransportClusterHealthAction.prepareResponse`). The node crate polls this
/// on each applied cluster state until it holds or the timeout elapses.
pub fn wait_conditions_met(h: &ClusterHealth, w: &WaitConditions) -> Result<bool, String> {
    if let Some(s) = w.status {
        if h.status > s {
            return Ok(false);
        }
    }
    if w.no_relocating_shards && h.relocating_shards > 0 {
        return Ok(false);
    }
    if w.no_initializing_shards && h.initializing_shards > 0 {
        return Ok(false);
    }
    match w.active_shards {
        Some(ActiveShardCount::All) if h.unassigned_shards > 0 || h.initializing_shards > 0 => return Ok(false),
        Some(ActiveShardCount::Count(n)) if h.active_shards < n => return Ok(false),
        _ => {}
    }
    if let Some(e) = &w.nodes {
        if !e.trim().is_empty() && !nodes_condition_met(e, h.number_of_nodes)? {
            return Ok(false);
        }
    }
    Ok(true)
}

// -------------------------------------------------------------------- _cat --

/// One `_cat` row: `(column, value)` in column order; `None` renders empty /
/// JSON `null`.
pub type CatRow = Vec<(&'static str, Option<String>)>;

pub const CAT_HEALTH_DEFAULT: &[&str] = &["epoch", "timestamp", "cluster", "status", "node.total", "node.data", "shards", "pri", "relo", "init", "unassign", "unassign.pri", "pending_tasks", "max_task_wait_time", "active_shards_percent"];

pub fn cat_health_row(h: &ClusterHealth, now_millis: i64) -> CatRow {
    let s = |v: u32| Some(v.to_string());
    vec![
        ("epoch", Some((now_millis / 1000).to_string())),
        ("timestamp", Some(fmt_hms(now_millis))),
        ("cluster", Some(h.cluster_name.clone())),
        ("status", Some(h.status.name().into())),
        ("node.total", s(h.number_of_nodes)),
        ("node.data", s(h.number_of_data_nodes)),
        ("shards", s(h.active_shards)),
        ("pri", s(h.active_primary_shards)),
        ("relo", s(h.relocating_shards)),
        ("init", s(h.initializing_shards)),
        ("unassign", s(h.unassigned_shards)),
        ("unassign.pri", s(h.unassigned_primary_shards)),
        ("pending_tasks", Some(h.number_of_pending_tasks.to_string())),
        ("max_task_wait_time", Some(if h.task_max_waiting_in_queue_millis == 0 { "-".into() } else { fmt_time_millis(h.task_max_waiting_in_queue_millis as i64) })),
        ("active_shards_percent", Some(pct_str(h.active_shards_percent))),
    ]
}

/// Per-copy stats `_cat/shards` shows (from the holding node's shard stats).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CatShardStats {
    pub docs: Option<u64>,
    pub store_bytes: Option<u64>,
    pub dataset_bytes: Option<u64>,
    pub max_seq_no: Option<i64>,
    pub local_checkpoint: Option<i64>,
    pub global_checkpoint: Option<i64>,
}

pub const CAT_SHARDS_DEFAULT: &[&str] = &["index", "shard", "prirep", "state", "docs", "store", "dataset", "ip", "node"];

fn bytes_cell(b: Option<u64>, unit: Option<&str>) -> Option<String> {
    b.map(|b| unit.and_then(|u| fmt_bytes_in(b, u)).unwrap_or_else(|| fmt_bytes(b)))
}

fn recovery_type(r: &RecoverySource) -> &'static str {
    match r {
        RecoverySource::EmptyStore => "empty_store",
        RecoverySource::ExistingStore { .. } => "existing_store",
        RecoverySource::Peer => "peer",
        RecoverySource::Snapshot { .. } => "snapshot",
        RecoverySource::LocalShards => "local_shards",
    }
}

/// `_cat/shards` rows for `indices` (`None` = all), index name then shard
/// then primary first. `bytes` is the `bytes=` unit (`None` = human).
pub fn cat_shards_rows(state: &ClusterState, indices: Option<&[String]>, now_millis: i64, stats: &dyn Fn(&ShardRouting) -> Option<CatShardStats>, bytes: Option<&str>) -> Vec<CatRow> {
    let mut rows = Vec::new();
    for (name, irt) in &state.routing_table.indices {
        if indices.is_some_and(|i| !i.contains(name)) {
            continue;
        }
        for t in &irt.shards {
            for c in t.copies.iter().filter(|c| !is_relocation_target(c)) {
                let st = if c.assigned() { stats(c).unwrap_or_default() } else { CatShardStats::default() };
                let node = c.node_id.as_ref().and_then(|n| state.nodes.get(n));
                let node_cell = node.map(|n| {
                    let mut s = n.name.clone();
                    if c.state == ShardState::Relocating {
                        if let Some(r) = c.relocating_node_id.as_ref().and_then(|r| state.nodes.get(r)) {
                            s.push_str(&format!(" -> {} {} {}", r.ip, r.id, r.name));
                        }
                    }
                    s
                });
                let u = c.unassigned_info.as_ref();
                rows.push(vec![
                    ("index", Some(name.clone())),
                    ("shard", Some(t.shard.to_string())),
                    ("prirep", Some(if c.primary { "p" } else { "r" }.into())),
                    ("state", Some(format!("{:?}", c.state).to_uppercase())),
                    ("docs", st.docs.map(|d| d.to_string())),
                    ("store", bytes_cell(st.store_bytes, bytes)),
                    ("dataset", bytes_cell(st.dataset_bytes.or(st.store_bytes), bytes)),
                    ("ip", node.map(|n| n.ip.clone())),
                    ("id", c.node_id.clone()),
                    ("node", node_cell),
                    ("unassigned.reason", u.map(|u| crate::allocation::reason_str(u.reason))),
                    ("unassigned.at", u.map(|u| fmt_date_millis(u.at_millis))),
                    ("unassigned.for", u.map(|u| fmt_time_millis((now_millis - u.at_millis).max(0)))),
                    ("unassigned.details", u.and_then(|u| u.details.clone())),
                    ("recoverysource.type", c.recovery_source.as_ref().map(|r| recovery_type(r).to_string())),
                    ("seq_no.max", st.max_seq_no.map(|v| v.to_string())),
                    ("seq_no.local_checkpoint", st.local_checkpoint.map(|v| v.to_string())),
                    ("seq_no.global_checkpoint", st.global_checkpoint.map(|v| v.to_string())),
                ]);
            }
        }
    }
    rows
}

pub const CAT_ALLOCATION_DEFAULT: &[&str] = &["shards", "disk.indices", "disk.used", "disk.avail", "disk.total", "disk.percent", "host", "ip", "node", "node.role"];

/// `_cat/allocation`: one row per data node plus an `UNASSIGNED` row when
/// some copies are unassigned. Disk figures come from [`ClusterInfo`].
pub fn cat_allocation_rows(state: &ClusterState, info: &ClusterInfo, bytes: Option<&str>) -> Vec<CatRow> {
    let mut counts: std::collections::BTreeMap<String, u32> = Default::default();
    let mut indices_bytes: std::collections::BTreeMap<String, u64> = Default::default();
    let mut unassigned = 0u32;
    for c in state.routing_table.all_shards().filter(|c| !is_relocation_target(c)) {
        match &c.node_id {
            Some(n) => {
                *counts.entry(n.clone()).or_default() += 1;
                *indices_bytes.entry(n.clone()).or_default() += info.shard_size(&c.index, c.shard, c.primary).unwrap_or(0);
            }
            None => unassigned += 1,
        }
    }
    let mut rows = Vec::new();
    for n in state.data_nodes() {
        let usage = info.nodes.get(&n.id);
        let (used, avail, total, pct) = match usage {
            Some(u) if u.total_bytes > 0 => {
                let used = u.total_bytes.saturating_sub(u.free_bytes);
                let denom = used + u.free_bytes;
                (Some(used), Some(u.free_bytes), Some(u.total_bytes), (denom > 0).then(|| (used * 100 / denom).to_string()))
            }
            _ => (None, None, None, None),
        };
        let ib = indices_bytes.get(&n.id).copied().unwrap_or(0);
        rows.push(vec![
            ("shards", Some(counts.get(&n.id).copied().unwrap_or(0).to_string())),
            ("shards.undesired", None),
            ("write_load.forecast", None),
            ("disk.indices.forecast", bytes_cell(Some(ib), bytes)),
            ("disk.indices", bytes_cell(Some(ib), bytes)),
            ("disk.used", bytes_cell(used, bytes)),
            ("disk.avail", bytes_cell(avail, bytes)),
            ("disk.total", bytes_cell(total, bytes)),
            ("disk.percent", pct),
            ("host", Some(n.host.clone())),
            ("ip", Some(n.ip.clone())),
            ("node", Some(n.name.clone())),
            ("node.role", Some(n.role_abbreviations())),
        ]);
    }
    if unassigned > 0 {
        rows.push(vec![
            ("shards", Some(unassigned.to_string())),
            ("shards.undesired", None),
            ("write_load.forecast", None),
            ("disk.indices.forecast", None),
            ("disk.indices", None),
            ("disk.used", None),
            ("disk.avail", None),
            ("disk.total", None),
            ("disk.percent", None),
            ("host", None),
            ("ip", None),
            ("node", Some("UNASSIGNED".into())),
            ("node.role", None),
        ]);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn node(id: &str) -> DiscoveryNode {
        DiscoveryNode { id: id.into(), name: format!("name-{id}"), ephemeral_id: "e".into(), transport_address: "127.0.0.1:9300".into(), http_address: "127.0.0.1:9200".into(), host: "h".into(), ip: "10.0.0.1".into(), roles: Role::defaults(), attributes: BTreeMap::new(), version: "8.15.0".into() }
    }

    fn info(reason: UnassignedReason, status: &str, failed: u32) -> UnassignedInfo {
        UnassignedInfo { reason, at_millis: 0, failed_allocations: failed, delayed: false, details: None, last_allocation_status: status.into(), last_node: None }
    }

    fn started(primary: bool, n: &str) -> ShardRouting {
        ShardRouting { index: "i".into(), shard: 0, primary, state: ShardState::Started, node_id: Some(n.into()), relocating_node_id: None, allocation_id: Some(AllocationId { id: format!("a-{n}"), relocation_id: None }), recovery_source: None, unassigned_info: None, expected_shard_size_in_bytes: None }
    }

    fn state_with(copies: Vec<ShardRouting>) -> ClusterState {
        let mut s = ClusterState::empty("test");
        s.blocks.global.clear();
        s.nodes.insert("n1".into(), node("n1"));
        s.nodes.insert("n2".into(), node("n2"));
        let replicas = copies.iter().filter(|c| !c.primary && !is_relocation_target(c)).count() as u32;
        s.metadata.indices.insert(
            "i".into(),
            IndexMetadata { name: "i".into(), uuid: "u".into(), version: 1, mapping_version: 1, settings_version: 1, aliases_version: 1, state: IndexState::Open, settings: Map::new().into(), mappings: json!({}).into(), aliases: BTreeMap::new(), number_of_shards: 1, number_of_replicas: replicas, routing_num_shards: 1024, primary_terms: vec![1], in_sync_allocation_ids: BTreeMap::from([(0, BTreeSet::new())]), creation_date: 0, lifecycle: None },
        );
        s.routing_table.indices.insert("i".into(), IndexRoutingTable { index_uuid: "u".into(), shards: vec![IndexShardRoutingTable { shard: 0, copies }] });
        s
    }

    #[test]
    fn status_rules() {
        // green: everything active
        let h = cluster_health(&state_with(vec![started(true, "n1"), started(false, "n2")]), None, HealthInputs::default());
        assert_eq!(h.status, HealthStatus::Green);
        assert_eq!(h.active_shards_percent, 100.0);
        // yellow: replica unassigned
        let r = ShardRouting::unassigned("i", 0, false, info(UnassignedReason::ReplicaAdded, "deciders_no", 0), Some(RecoverySource::Peer));
        let h = cluster_health(&state_with(vec![started(true, "n1"), r]), None, HealthInputs::default());
        assert_eq!(h.status, HealthStatus::Yellow);
        assert_eq!(h.active_shards_percent, 50.0);
        assert_eq!(h.unassigned_shards, 1);
        assert_eq!(h.unassigned_primary_shards, 0);
        // new primary unassigned, never tried: yellow
        let p = ShardRouting::unassigned("i", 0, true, info(UnassignedReason::IndexCreated, "no_attempt", 0), Some(RecoverySource::EmptyStore));
        assert_eq!(cluster_health(&state_with(vec![p.clone()]), None, HealthInputs::default()).status, HealthStatus::Yellow);
        // ... refused by deciders: red
        let mut p2 = p.clone();
        p2.unassigned_info = Some(info(UnassignedReason::IndexCreated, "deciders_no", 0));
        assert_eq!(cluster_health(&state_with(vec![p2]), None, HealthInputs::default()).status, HealthStatus::Red);
        // ... failed once: red
        let mut p3 = p.clone();
        p3.unassigned_info = Some(info(UnassignedReason::AllocationFailed, "no_attempt", 1));
        assert_eq!(cluster_health(&state_with(vec![p3]), None, HealthInputs::default()).status, HealthStatus::Red);
        // existing-store primary unassigned: red
        let p4 = ShardRouting::unassigned("i", 0, true, info(UnassignedReason::NodeLeft, "no_attempt", 0), Some(RecoverySource::ExistingStore { bootstrap_new_history_uuid: false }));
        let h = cluster_health(&state_with(vec![p4]), None, HealthInputs::default());
        assert_eq!(h.status, HealthStatus::Red);
        assert_eq!(h.unassigned_primary_shards, 1);
        // no-master block: red even when shards are fine
        let mut s = state_with(vec![started(true, "n1")]);
        s.blocks.global.insert(ClusterBlock::no_master_writes());
        assert_eq!(cluster_health(&s, None, HealthInputs::default()).status, HealthStatus::Red);
    }

    #[test]
    fn relocation_counts_once() {
        let mut src = started(true, "n1");
        src.state = ShardState::Relocating;
        src.relocating_node_id = Some("n2".into());
        let mut tgt = started(true, "n2");
        tgt.state = ShardState::Initializing;
        tgt.relocating_node_id = Some("n1".into());
        let h = cluster_health(&state_with(vec![src, tgt]), None, HealthInputs::default());
        assert_eq!(h.status, HealthStatus::Green);
        assert_eq!((h.active_shards, h.relocating_shards, h.initializing_shards), (1, 1, 0));
    }

    #[test]
    fn json_shape() {
        let h = cluster_health(&state_with(vec![started(true, "n1")]), None, HealthInputs::default());
        let v = h.to_json(Level::Shards, false);
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            vec!["cluster_name", "status", "timed_out", "number_of_nodes", "number_of_data_nodes", "active_primary_shards", "active_shards", "relocating_shards", "initializing_shards", "unassigned_shards", "unassigned_primary_shards", "delayed_unassigned_shards", "number_of_pending_tasks", "number_of_in_flight_fetch", "task_max_waiting_in_queue_millis", "active_shards_percent_as_number", "indices"]
        );
        assert_eq!(v["indices"]["i"]["shards"]["0"]["primary_active"], json!(true));
        assert_eq!(v["active_shards_percent_as_number"], json!(100.0));
        let hv = h.to_json(Level::Cluster, true);
        assert_eq!(hv["active_shards_percent"], json!("100.0%"));
        assert_eq!(hv["task_max_waiting_in_queue"], json!("0s"));
        assert!(hv.get("indices").is_none());
    }

    #[test]
    fn wait_for_predicates() {
        let h = cluster_health(&state_with(vec![started(true, "n1")]), None, HealthInputs::default());
        assert!(nodes_condition_met(">=2", 2).unwrap());
        assert!(nodes_condition_met("ge(2)", 2).unwrap());
        assert!(!nodes_condition_met("<2", 2).unwrap());
        assert!(nodes_condition_met("lt(3)", 2).unwrap());
        assert!(nodes_condition_met("2", 2).unwrap());
        assert!(!nodes_condition_met(">2", 2).unwrap());
        assert!(nodes_condition_met("le(2)", 2).unwrap());
        assert!(nodes_condition_met("x", 2).is_err());
        let w = WaitConditions { status: Some(HealthStatus::Yellow), nodes: Some(">=2".into()), active_shards: Some(ActiveShardCount::All), no_relocating_shards: true, no_initializing_shards: true };
        assert!(wait_conditions_met(&h, &w).unwrap());
        let w = WaitConditions { active_shards: Some(ActiveShardCount::Count(2)), ..Default::default() };
        assert!(!wait_conditions_met(&h, &w).unwrap());
        assert_eq!(ActiveShardCount::parse("all").unwrap(), Some(ActiveShardCount::All));
        assert!(ActiveShardCount::parse("x").is_err());
    }

    #[test]
    fn cat_rows() {
        let r = ShardRouting::unassigned("i", 0, false, info(UnassignedReason::ReplicaAdded, "deciders_no", 0), Some(RecoverySource::Peer));
        let s = state_with(vec![started(true, "n1"), r]);
        let h = cluster_health(&s, None, HealthInputs::default());
        let row = cat_health_row(&h, 1475871424000);
        assert_eq!(row[0], ("epoch", Some("1475871424".into())));
        assert_eq!(row[3], ("status", Some("yellow".into())));
        assert_eq!(row.last().unwrap(), &("active_shards_percent", Some("50.0%".into())));
        let stats = |_: &ShardRouting| Some(CatShardStats { docs: Some(3), store_bytes: Some(2048), ..Default::default() });
        let rows = cat_shards_rows(&s, None, 60_000, &stats, None);
        assert_eq!(rows.len(), 2);
        let p: BTreeMap<_, _> = rows[0].iter().cloned().collect();
        assert_eq!(p["prirep"], Some("p".into()));
        assert_eq!(p["store"], Some("2kb".into()));
        assert_eq!(p["node"], Some("name-n1".into()));
        let r: BTreeMap<_, _> = rows[1].iter().cloned().collect();
        assert_eq!(r["state"], Some("UNASSIGNED".into()));
        assert_eq!(r["unassigned.reason"], Some("REPLICA_ADDED".into()));
        assert_eq!(r["unassigned.for"], Some("1m".into()));
        assert_eq!(r["docs"], None);
        let mut ci = ClusterInfo::default();
        ci.nodes.insert("n1".into(), crate::allocation::DiskUsage { path: "/d".into(), total_bytes: 100 << 30, free_bytes: 40 << 30 });
        let rows = cat_allocation_rows(&s, &ci, Some("b"));
        let n1: BTreeMap<_, _> = rows[0].iter().cloned().collect();
        assert_eq!(n1["shards"], Some("1".into()));
        assert_eq!(n1["disk.percent"], Some("60".into()));
        let un: BTreeMap<_, _> = rows.last().unwrap().iter().cloned().collect();
        assert_eq!(un["node"], Some("UNASSIGNED".into()));
    }
}
