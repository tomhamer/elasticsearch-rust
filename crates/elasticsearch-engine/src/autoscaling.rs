//! Autoscaling decider calculations, ported from ES 9.5.5
//! (`x-pack/plugin/autoscaling` storage / frozen deciders and
//! `x-pack/plugin/ml/.../autoscaling`): the required capacity of a policy's
//! tier from its current nodes, the shard copies it holds or should hold,
//! its data streams and its ML jobs. Pure functions over what the caller
//! gathered: the cluster's master (`_autoscaling/capacity`) and the
//! standalone node feed them the same inputs.
//!
//! Where ES consults its allocation deciders, these use the disk threshold
//! rules alone (the storage deciders only count shards the disk decider
//! refuses); ML node loads are rebuilt from the jobs' memory (first fit,
//! largest first), as this engine runs every job on the master.

use serde_json::{json, Value};

const KB: u64 = 1 << 10;
const MB: u64 = 1 << 20;
const GB: u64 = 1 << 30;

// ------------------------------------------------------------ results

/// A decider's result: the required capacity (`None` per resource: not
/// this decider's; `capacity: None` = it cannot tell now, so the policy has
/// no required capacity, as in ES) and its reason.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decision {
    pub capacity: Option<Capacity>,
    pub summary: String,
    pub details: Value,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Capacity {
    pub total_storage: Option<u64>,
    pub total_memory: Option<u64>,
    pub node_storage: Option<u64>,
    pub node_memory: Option<u64>,
}

/// ES `ByteSizeValue.toString()` (one decimal, no trailing `.0`).
pub fn humanize(b: u64) -> String {
    let units = [("pb", 1u64 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)];
    match units.iter().find(|(_, d)| b >= *d) {
        Some((suffix, div)) => {
            let v = (b as f64 / *div as f64 * 10.0).round() / 10.0;
            if v.fract() == 0.0 { format!("{}{suffix}", v as u64) } else { format!("{v:.1}{suffix}") }
        }
        None => format!("{b}b"),
    }
}

/// ES `TimeValue.getStringRep()` of whole milliseconds.
fn time_rep(ms: u64) -> String {
    for (div, unit) in [(86_400_000, "d"), (3_600_000, "h"), (60_000, "m"), (1_000, "s")] {
        if ms >= div && ms % div == 0 {
            return format!("{}{unit}", ms / div);
        }
    }
    format!("{ms}ms")
}

// ------------------------------------------------------------ disk thresholds

/// A disk watermark (`RelativeByteSizeValue`): a used percentage (ES
/// computes with `getAsPercent()`), or an absolute amount of free space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Watermark {
    Percent(f64),
    Free(u64),
}

impl Watermark {
    /// `RelativeByteSizeValue.parseRelativeByteSizeValue`.
    fn parse(s: &str) -> Option<Watermark> {
        let s = s.trim();
        if !s.ends_with('b') {
            if let Some(p) = s.strip_suffix('%') {
                return p.trim().parse::<f64>().ok().filter(|p| (0.0..=100.0).contains(p)).map(Watermark::Percent);
            }
            if let Ok(r) = s.parse::<f64>() {
                return (0.0..=1.0).contains(&r).then_some(if r == 0.0 { Watermark::Free(0) } else { Watermark::Percent(r * 100.0) });
            }
        }
        crate::ilm::parse_size(s, "watermark").ok().map(|(b, _)| Watermark::Free(b))
    }
}

/// `DiskThresholdSettings`: the low and high watermarks and their max
/// headrooms (`200GB` / `150GB` unless the watermark is set explicitly).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskThresholds {
    pub low: Watermark,
    pub low_headroom: Option<u64>,
    pub high: Watermark,
    pub high_headroom: Option<u64>,
}

impl Default for DiskThresholds {
    fn default() -> Self {
        DiskThresholds { low: Watermark::Percent(85.0), low_headroom: Some(200 * GB), high: Watermark::Percent(90.0), high_headroom: Some(150 * GB) }
    }
}

impl DiskThresholds {
    /// From the cluster settings `cluster.routing.allocation.disk.watermark.*`.
    pub fn from_settings(get: impl Fn(&str) -> Option<String>) -> DiskThresholds {
        let d = DiskThresholds::default();
        let stage = |key: &str, default: Watermark, default_headroom: u64| -> (Watermark, Option<u64>) {
            let set = get(key);
            let mark = set.as_deref().and_then(Watermark::parse).unwrap_or(default);
            let headroom = match get(&format!("{key}.max_headroom")) {
                Some(h) if h.trim() == "-1" => None,
                Some(h) => crate::ilm::parse_size(&h, "max_headroom").ok().map(|(b, _)| b),
                None if set.is_some() => None,
                None => Some(default_headroom),
            };
            (mark, headroom)
        };
        let (low, low_headroom) = stage("cluster.routing.allocation.disk.watermark.low", d.low, 200 * GB);
        let (high, high_headroom) = stage("cluster.routing.allocation.disk.watermark.high", d.high, 150 * GB);
        DiskThresholds { low, low_headroom, high, high_headroom }
    }

    /// `getFreeBytesThreshold`: free bytes the watermark keeps on `total`.
    fn free_threshold(mark: Watermark, headroom: Option<u64>, total: u64) -> u64 {
        match mark {
            Watermark::Free(b) => b,
            Watermark::Percent(p) => {
                // `RelativeByteSizeValue.calculateValue`.
                let ratio_bytes = (total as f64 * p / 100.0).ceil() as u64;
                let used = match headroom { Some(h) => ratio_bytes.max(total.saturating_sub(h)), None => ratio_bytes };
                total.saturating_sub(used)
            }
        }
    }

    fn low_free(&self, total: u64) -> u64 { Self::free_threshold(self.low, self.low_headroom, total) }
    fn high_free(&self, total: u64) -> u64 { Self::free_threshold(self.high, self.high_headroom, total) }

    /// `getMinimumTotalSizeForBelowLowWatermark`.
    fn min_total_below_low(&self, used: u64) -> u64 {
        match self.low {
            Watermark::Free(b) => used + b,
            Watermark::Percent(pct) => {
                if (0.0..100.0).contains(&pct) {
                    let total = ((100 * used) as f64 / pct).ceil() as u64;
                    match self.low_headroom { Some(h) => total.min(used + h), None => total }
                } else {
                    used
                }
            }
        }
    }
}

// ------------------------------------------------------------ storage

/// One current data node of the tier: its data path's disk.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NodeDisk {
    pub total: u64,
    pub free: u64,
}

/// One shard copy relevant to the tier: `[index][n]`, its expected size
/// (an unassigned copy's: its primary's), and the tier node holding it
/// (an index into the tier's nodes; `None`: unassigned).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShardCopy {
    pub id: String,
    pub bytes: u64,
    pub node: Option<usize>,
}

/// `AllocationState.sizeOf`: an unknown size counts as 1kb.
fn size_of(s: &ShardCopy) -> u64 {
    if s.bytes == 0 { KB } else { s.bytes }
}

struct StorageState {
    unassigned: u64,
    unassigned_ids: Vec<String>,
    assigned: u64,
    assigned_ids: Vec<String>,
    max_shard: u64,
}

/// The disk threshold decider's answer for `bytes` on `n`: below the low
/// watermark, and still above the high one with the copy added.
fn disk_allows(th: &DiskThresholds, n: &NodeDisk, bytes: u64) -> bool {
    n.free >= th.low_free(n.total) && n.free.checked_sub(bytes).is_some_and(|after| after >= th.high_free(n.total))
}

/// `ReactiveStorageDeciderService.AllocationState`: unassigned copies the
/// tier's disks refuse (`storagePreventsAllocation`), and copies on nodes
/// past the high watermark that no other tier node can take
/// (`storagePreventsRemainOrMove`, `unmovableSize`).
fn storage_state(nodes: &[NodeDisk], shards: &[ShardCopy], th: &DiskThresholds) -> StorageState {
    let fits_elsewhere = |s: &ShardCopy, except: Option<usize>| nodes.iter().enumerate().any(|(i, n)| Some(i) != except && disk_allows(th, n, size_of(s)));
    let mut st = StorageState { unassigned: 0, unassigned_ids: Vec::new(), assigned: 0, assigned_ids: Vec::new(), max_shard: 0 };
    for s in shards.iter().filter(|s| s.node.is_none()) {
        if nodes.is_empty() || !fits_elsewhere(s, None) {
            st.unassigned += size_of(s);
            st.unassigned_ids.push(s.id.clone());
        }
    }
    for (i, n) in nodes.iter().enumerate() {
        let threshold = th.high_free(n.total);
        if n.free >= threshold {
            continue;
        }
        let unmovable: Vec<&ShardCopy> = shards.iter().filter(|s| s.node == Some(i) && !fits_elsewhere(s, Some(i))).collect();
        if let Some(min) = unmovable.iter().map(|s| size_of(s)).min() {
            st.assigned += (threshold - n.free).max(min);
            st.assigned_ids.extend(unmovable.iter().map(|s| s.id.clone()));
        }
    }
    st.max_shard = shards.iter().filter(|s| s.node.is_some()).map(size_of).max().unwrap_or(0);
    st
}

/// `ReactiveStorageDeciderService.message`.
fn storage_message(unassigned: u64, assigned: u64) -> String {
    if unassigned > 0 || assigned > 0 { format!("not enough storage available, needs {}", humanize(unassigned + assigned)) } else { "storage ok".into() }
}

/// ES `NODE_DISK_OVERHEAD` (ReactiveStorageDeciderService).
const NODE_DISK_OVERHEAD: u64 = 10 * MB;

fn unknown_capacity() -> Decision {
    Decision { capacity: None, summary: "current capacity not available".into(), details: json!({"reason": "current capacity not available", "unassigned": -1, "assigned": -1}) }
}

/// ES `ReactiveStorageDeciderService.scale`: the tier's current storage
/// plus the unassigned copies its disks refuse plus what nodes past the
/// high watermark cannot shed; a node must hold the largest copy below the
/// low watermark, plus 10mb. `current_storage: None`: not known yet.
pub fn reactive_storage(current_storage: Option<u64>, nodes: &[NodeDisk], shards: &[ShardCopy], th: &DiskThresholds) -> Decision {
    let Some(current) = current_storage else { return unknown_capacity() };
    let st = storage_state(nodes, shards, th);
    let required = current + st.unassigned + st.assigned;
    let node = if required > 0 { th.min_total_below_low(st.max_shard) + NODE_DISK_OVERHEAD } else { 0 };
    let summary = storage_message(st.unassigned, st.assigned);
    Decision {
        capacity: Some(Capacity { total_storage: Some(required), node_storage: Some(node), ..Default::default() }),
        details: json!({"reason": summary, "unassigned": st.unassigned, "unassigned_shards": st.unassigned_ids.iter().take(10).collect::<Vec<_>>(),
            "unassigned_shards_count": st.unassigned_ids.len(), "assigned": st.assigned, "assigned_shards": st.assigned_ids.iter().take(10).collect::<Vec<_>>(),
            "assigned_shards_count": st.assigned_ids.len()}),
        summary,
    }
}

/// One backing index of a data stream, for the proactive forecast.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BackingIndex {
    pub name: String,
    pub creation_ms: i64,
    /// The size of every copy (`sizeOf`) summed.
    pub bytes: u64,
    /// Copies (shards x (replicas + 1)) of the index.
    pub copies: u64,
    /// Whether any allocated copy is on a tier node / off the tier.
    pub on_tier: bool,
    pub off_tier: bool,
}

/// A data stream's backing indices, oldest first (the last is the write index).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamIndices {
    pub indices: Vec<BackingIndex>,
}

/// `AllocationState.forecast(ds)`: the new indices a data stream rolls over
/// to within `window` at the rate of its recent indices, as unassigned
/// copies (each new index's size spread over its copies).
fn forecast_stream(ds: &StreamIndices, window: i64, now: i64) -> Vec<ShardCopy> {
    let indices = &ds.indices;
    // `dataStreamAllocatedToNodes`: the newest index with an allocated copy
    // is on the tier only.
    let on_tier = indices.iter().rev().find_map(|i| if i.off_tier { Some(false) } else if i.on_tier { Some(true) } else { None });
    if on_tier != Some(true) {
        return Vec::new();
    }
    let (mut min_creation, mut total, mut count) = (i64::MAX, 0u64, 0u64);
    for idx in indices.iter().rev() {
        count += 1;
        if idx.creation_ms < 0 {
            return Vec::new();
        }
        min_creation = min_creation.min(idx.creation_ms);
        total += idx.bytes;
        if idx.creation_ms <= now - window {
            break;
        }
    }
    if total == 0 {
        return Vec::new();
    }
    let avg = (total - 1) / count + 1;
    let actual = now - min_creation;
    if actual == 0 {
        return Vec::new();
    }
    let (scaled, new_indices) = if actual > window {
        let scaled = (total as u128 * window as u128 / actual as u128) as u64;
        if scaled == 0 {
            return Vec::new();
        }
        (scaled, ((scaled - 1) / avg + 1).min(indices.len() as u64))
    } else {
        (total, count)
    };
    let write = indices.last().expect("a data stream has a write index");
    let copies = write.copies.max(1);
    let mut out = Vec::new();
    for i in 0..new_indices {
        let size = avg.min(scaled - avg * i);
        for c in 0..copies {
            out.push(ShardCopy { id: format!("[{}-forecast-{i}][{c}]", write.name), bytes: size / copies, node: None });
        }
    }
    out
}

/// ES `ProactiveStorageDeciderService.scale`: the reactive state after the
/// data streams' forecast over `forecast_window` (default 30m); a node must
/// hold the largest copy.
pub fn proactive_storage(current_storage: Option<u64>, nodes: &[NodeDisk], shards: &[ShardCopy], streams: &[StreamIndices], window_ms: u64, now_ms: i64, th: &DiskThresholds) -> Decision {
    let Some(current) = current_storage else { return unknown_capacity() };
    let before = storage_state(nodes, shards, th).unassigned;
    let mut forecast = shards.to_vec();
    if window_ms > 0 {
        for ds in streams {
            forecast.extend(forecast_stream(ds, window_ms as i64, now_ms));
        }
    }
    let st = storage_state(nodes, &forecast, th);
    let summary = storage_message(st.unassigned, st.assigned);
    Decision {
        capacity: Some(Capacity { total_storage: Some(current + st.unassigned + st.assigned), node_storage: Some(st.max_shard), ..Default::default() }),
        details: json!({"reason": summary, "unassigned": st.unassigned, "assigned": st.assigned, "forecasted": st.unassigned - before, "forecast_window": time_rep(window_ms)}),
        summary,
    }
}

// ------------------------------------------------------------ frozen

/// ES `FrozenExistenceDeciderService.MINIMUM_FROZEN_MEMORY` / `_STORAGE`.
pub const MINIMUM_FROZEN_MEMORY: u64 = GB;
pub const MINIMUM_FROZEN_STORAGE: u64 = 8 * GB;

/// ES `FrozenExistenceDeciderService.scale`: indices in the ILM frozen
/// phase (up to 10 listed) need a minimal frozen node.
pub fn frozen_existence(frozen_phase_indices: &[String]) -> Decision {
    let listed: Vec<&String> = frozen_phase_indices.iter().take(10).collect();
    let capacity = if listed.is_empty() {
        Capacity { total_storage: Some(0), total_memory: Some(0), ..Default::default() }
    } else {
        Capacity { total_storage: Some(MINIMUM_FROZEN_STORAGE), total_memory: Some(MINIMUM_FROZEN_MEMORY), node_storage: Some(MINIMUM_FROZEN_STORAGE), node_memory: Some(MINIMUM_FROZEN_MEMORY) }
    };
    Decision {
        capacity: Some(capacity),
        summary: format!("indices [{}]", listed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")),
        details: json!({"indices": listed}),
    }
}

/// ES `FrozenShardsDeciderService.DEFAULT_MEMORY_PER_SHARD`: 64gb / 2000.
pub const DEFAULT_MEMORY_PER_SHARD: u64 = (64 * GB) / 2000;

/// ES `FrozenShardsDeciderService.scale`: the frozen indices' total shard
/// count (`getTotalNumberOfShards`) times `memory_per_shard`.
pub fn frozen_shards(shards: u64, memory_per_shard: u64) -> Decision {
    Decision {
        capacity: Some(Capacity { total_memory: Some(shards * memory_per_shard), ..Default::default() }),
        details: json!({"shards": shards}),
        summary: format!("shard count [{shards}]"),
    }
}

/// ES `FrozenStorageDeciderService.DEFAULT_PERCENTAGE`.
pub const DEFAULT_FROZEN_STORAGE_PERCENTAGE: f64 = 5.0;

/// ES `FrozenStorageDeciderService.scale`: `percentage` of the frozen
/// indices' data set size (each shard's size times its copies).
pub fn frozen_storage(data_set_bytes: u64, percentage: f64) -> Decision {
    let storage = ((percentage * data_set_bytes as f64) as i64 / 100).max(0) as u64;
    Decision {
        capacity: Some(Capacity { total_storage: Some(storage), ..Default::default() }),
        details: json!({"total_data_set_size": data_set_bytes}),
        summary: format!("total data set size [{data_set_bytes}]"),
    }
}

/// Whether an index belongs to the frozen tier (`FrozenUtils.isFrozenIndex`):
/// its first tier preference is `data_frozen`.
pub fn is_frozen_index(tier_preference: Option<&str>) -> bool {
    tier_preference.and_then(|p| p.split(',').map(str::trim).find(|t| !t.is_empty())) == Some("data_frozen")
}

// ------------------------------------------------------------ ML

/// ES `MachineLearning.NATIVE_EXECUTABLE_CODE_OVERHEAD`.
const NATIVE_CODE_OVERHEAD: u64 = 30 * MB;
/// ES `Job.PROCESS_MEMORY_OVERHEAD` / `DataFrameAnalyticsConfig.PROCESS_MEMORY_OVERHEAD`.
pub const ANOMALY_PROCESS_OVERHEAD: u64 = 10 * MB;
pub const ANALYTICS_PROCESS_OVERHEAD: u64 = 5 * MB;
/// ES `AnalysisLimits.DEFAULT_MODEL_MEMORY_LIMIT_MB`.
const DEFAULT_MODEL_MEMORY_LIMIT: u64 = 1024 * MB;
// ES `NativeMemoryCalculator`.
const STATIC_JVM_UPPER_THRESHOLD: u64 = 31 * GB;
const MINIMUM_AUTOMATIC_NODE_SIZE: u64 = 512 * MB;
const OS_OVERHEAD: u64 = 200 * MB;
const JVM_SIZE_KNOT_POINT: u64 = 16 * GB;
const BYTES_IN_4MB: u64 = 4 * MB;

fn jvm_from_node_size(node: u64) -> u64 {
    if node <= JVM_SIZE_KNOT_POINT {
        return ((node as f64 * 0.4) as u64 / BYTES_IN_4MB) * BYTES_IN_4MB;
    }
    ((((JVM_SIZE_KNOT_POINT as f64) * 0.4 + (node - JVM_SIZE_KNOT_POINT) as f64 * 0.1) as u64 / BYTES_IN_4MB) * BYTES_IN_4MB).min(STATIC_JVM_UPPER_THRESHOLD)
}

fn jvm_from_native(native: u64) -> u64 {
    let min_jvm = jvm_from_node_size(MINIMUM_AUTOMATIC_NODE_SIZE);
    let with_overhead = native + OS_OVERHEAD;
    let higher = if with_overhead <= JVM_SIZE_KNOT_POINT - jvm_from_node_size(JVM_SIZE_KNOT_POINT) {
        (with_overhead * 2 / 3 / BYTES_IN_4MB) * BYTES_IN_4MB
    } else {
        let above = with_overhead as f64 - JVM_SIZE_KNOT_POINT as f64 * 0.6;
        ((JVM_SIZE_KNOT_POINT as f64 * 0.4 + above / 0.9 * 0.1) as u64 / BYTES_IN_4MB) * BYTES_IN_4MB
    };
    if higher > BYTES_IN_4MB {
        let lower = higher - BYTES_IN_4MB;
        if jvm_from_node_size(with_overhead + lower) == lower {
            return lower.min(STATIC_JVM_UPPER_THRESHOLD).max(min_jvm);
        }
    }
    higher.min(STATIC_JVM_UPPER_THRESHOLD).max(min_jvm)
}

/// The ML settings the decider reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlSettings {
    pub anomaly_queue: usize,
    pub analytics_queue: usize,
    pub down_scale_delay_ms: u64,
    /// `xpack.ml.max_machine_memory_percent` (default 30).
    pub max_machine_memory_percent: u64,
    /// `xpack.ml.use_auto_machine_memory_percent` (default false).
    pub use_auto: bool,
    /// `xpack.ml.max_ml_node_size` (0: unlimited).
    pub max_ml_node_size: u64,
}

impl Default for MlSettings {
    fn default() -> Self {
        MlSettings { anomaly_queue: 0, analytics_queue: 0, down_scale_delay_ms: 3_600_000, max_machine_memory_percent: 30, use_auto: false, max_ml_node_size: 0 }
    }
}

impl MlSettings {
    /// `NativeMemoryCalculator.calculateApproxNecessaryNodeSize`.
    fn node_size_for(&self, native: u64) -> u64 {
        if native == 0 {
            return 0;
        }
        if self.use_auto {
            return (native + jvm_from_native(native) + OS_OVERHEAD).max(MINIMUM_AUTOMATIC_NODE_SIZE);
        }
        ((100.0 / self.max_machine_memory_percent.max(1) as f64) * native as f64).ceil() as u64
    }

    /// `NativeMemoryCalculator.allowedBytesForMl(machineMemory, ...)`.
    fn allowed_for_ml(&self, machine: u64) -> u64 {
        if machine == 0 {
            return 0;
        }
        if self.use_auto {
            let jvm = jvm_from_node_size(machine);
            if machine.saturating_sub(jvm) <= OS_OVERHEAD {
                return machine / 100;
            }
            return (machine - jvm - OS_OVERHEAD).min(machine * 9 / 10);
        }
        machine * self.max_machine_memory_percent / 100
    }

    /// `NativeMemoryCapacity.autoscalingCapacity` (one availability zone):
    /// (node size, tier size) for native memory excluding the per-node
    /// overhead.
    fn capacity_for(&self, tier_native: i64, node_native: u64) -> (u64, u64) {
        if tier_native <= 0 || node_native == 0 {
            return (0, 0);
        }
        let tier_native = tier_native as u64;
        let largest = if self.max_ml_node_size == 0 { u64::MAX / 2 } else { self.allowed_for_ml(self.max_ml_node_size).max(NATIVE_CODE_OVERHEAD + 1) };
        let largest_excl = largest - NATIVE_CODE_OVERHEAD;
        let nodes = tier_native.div_ceil(largest_excl).max(1);
        let per_node = tier_native.div_ceil(nodes);
        let node_size = self.node_size_for(node_native + NATIVE_CODE_OVERHEAD);
        let tier_size = self.node_size_for(per_node + NATIVE_CODE_OVERHEAD) * nodes;
        (node_size, tier_size.max(node_size))
    }
}

/// One ML job: its memory requirement (model memory plus process
/// overhead), whether a node holds it (`false`: waiting), and whether it is
/// a batch job that frees its memory when it finishes (analytics).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlJob {
    pub bytes: u64,
    pub assigned: bool,
    pub analytics: bool,
}

/// The engine's open anomaly detection jobs and running analytics jobs.
pub fn ml_jobs(node: &crate::node::Node) -> Vec<MlJob> {
    let mut out: Vec<MlJob> = crate::ml::anomaly::open_job_memory(node).into_iter()
        .map(|(mml, assigned)| MlJob { bytes: mml + ANOMALY_PROCESS_OVERHEAD, assigned, analytics: false }).collect();
    out.extend(crate::ml::dfa::running_memory(node).into_iter().map(|mml| MlJob { bytes: mml + ANALYTICS_PROCESS_OVERHEAD, assigned: true, analytics: true }));
    out
}

/// ES `ScaleTimer`: when a scale-down was first detected.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScaleTimer {
    pub down_detected_ms: Option<i64>,
}

impl ScaleTimer {
    /// `markDownScaleAndGetMillisLeftFromDelay`.
    fn mark_down(&mut self, now: i64, delay: u64) -> i64 {
        let at = *self.down_detected_ms.get_or_insert(now);
        delay as i64 - (now - at)
    }
    fn reset(&mut self) {
        self.down_detected_ms = None;
    }
}

/// A node's ML load: its ML memory (`allowedBytesForMl`) and what is
/// assigned to it (per-node overhead included once any job is).
#[derive(Clone, Copy, Debug)]
struct Load {
    max: u64,
    assigned: u64,
    jobs: u64,
    batch: u64,
}

impl Load {
    fn free(&self) -> u64 { self.max.saturating_sub(self.assigned) }
    fn free_excluding_overhead(&self) -> u64 { self.max.saturating_sub(self.assigned).saturating_sub(if self.jobs == 0 { NATIVE_CODE_OVERHEAD } else { 0 }) }
    fn assign(&mut self, bytes: u64) {
        if self.jobs == 0 { self.assigned += NATIVE_CODE_OVERHEAD; }
        self.assigned += bytes;
        self.jobs += 1;
    }
}

/// `determineUnassignableJobs`: assign the waiting jobs (largest first) to
/// the node with the most free memory while more than `queue` wait; what
/// cannot be assigned while more than `queue` wait is required capacity
/// (tier: its sum, node: the largest).
fn unassignable(waiting: &[u64], queue: usize, loads: &mut [Load]) -> Option<(u64, u64)> {
    if waiting.is_empty() || waiting.len() < queue {
        return None;
    }
    let mut sizes = waiting.to_vec();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let mut i = 0;
    while sizes.len() > queue && i < sizes.len() {
        let need = sizes[i];
        let best = loads.iter_mut().max_by_key(|l| l.free());
        match best {
            Some(l) if l.free() >= need + if l.jobs == 0 { NATIVE_CODE_OVERHEAD } else { 0 } => { l.assign(need); sizes.remove(i); }
            _ => i += 1,
        }
    }
    let mut left = Vec::new();
    while sizes.len() > queue && !sizes.is_empty() {
        left.push(sizes.remove(0));
    }
    (!left.is_empty()).then(|| (left.iter().sum(), left[0])).or(Some((0, 0)))
}

/// The ML decider's inputs: the jobs, the tier's ML nodes' machine memory,
/// and the policy's current memory (what a pass-through result reports).
pub struct MlInput<'a> {
    pub jobs: &'a [MlJob],
    pub node_memory: &'a [u64],
    pub current_tier_memory: u64,
    pub current_node_memory: u64,
}

fn ml_decision(capacity: Option<(u64, u64)>, reason: String, waiting_anomaly: usize, waiting_analytics: usize) -> Decision {
    Decision {
        capacity: capacity.map(|(node, tier)| Capacity { total_memory: Some(tier), node_memory: Some(node), ..Default::default() }),
        details: json!({"reason": reason, "waiting_anomaly_jobs": waiting_anomaly, "waiting_analytics_jobs": waiting_analytics}),
        summary: reason,
    }
}

/// ES `MlAutoscalingDeciderService.scale` / `MlMemoryAutoscalingDecider.scale`
/// (memory; ES's processor decider is not modelled).
pub fn ml(input: &MlInput, s: &MlSettings, timer: &mut ScaleTimer, now_ms: i64) -> Decision {
    let current = Some((input.current_node_memory, input.current_tier_memory));
    // `NativeMemoryCapacity.currentScale`.
    let allowed: Vec<u64> = input.node_memory.iter().map(|m| s.allowed_for_ml(*m).saturating_sub(NATIVE_CODE_OVERHEAD)).collect();
    let (cur_tier, cur_node) = (allowed.iter().sum::<u64>(), allowed.iter().copied().max().unwrap_or(0));
    let waiting_anomaly: Vec<u64> = input.jobs.iter().filter(|j| !j.assigned && !j.analytics).map(|j| j.bytes).collect();
    let waiting_analytics: Vec<u64> = input.jobs.iter().filter(|j| !j.assigned && j.analytics).map(|j| j.bytes).collect();
    let (wm, wa) = (waiting_anomaly.len(), waiting_analytics.len());
    let delay_message = |left: i64, at: Option<i64>| format!("Passing currently perceived capacity as down scale delay has not been satisfied; configured delay [{}] \
        last detected scale down event [{}]. Will request scale down in approximately [{}]", time_rep(s.down_scale_delay_ms),
        crate::date::human_date(at.unwrap_or(now_ms)), time_rep(left.max(0) as u64));
    // No tasks: `downscaleToZero`.
    if input.jobs.is_empty() {
        if cur_tier == 0 && cur_node == 0 {
            return ml_decision(current, "Passing currently perceived capacity as no scaling changes are necessary".into(), 0, 0);
        }
        let left = timer.mark_down(now_ms, s.down_scale_delay_ms);
        if left > 0 {
            return ml_decision(current, delay_message(left, timer.down_detected_ms), 0, 0);
        }
        return ml_decision(Some((0, 0)), "Requesting scale down as tier and/or node size could be smaller".into(), 0, 0);
    }
    // No ML nodes: `scaleUpFromZero`.
    if input.node_memory.is_empty() && wm + wa > 0 {
        let waiting: Vec<u64> = waiting_anomaly.iter().chain(&waiting_analytics).copied().collect();
        let (mut tier, mut node) = (waiting.iter().sum::<u64>(), waiting.iter().copied().max().unwrap_or(0));
        if node == 0 {
            tier += DEFAULT_MODEL_MEMORY_LIMIT;
            node = DEFAULT_MODEL_MEMORY_LIMIT;
        }
        timer.reset();
        return ml_decision(Some(s.capacity_for(tier as i64, node)),
            "requesting scale up as number of jobs in queues exceeded configured limit and there are no machine learning nodes".into(), wm, wa);
    }
    // Node loads: the assigned jobs, largest first, on the node with the
    // most free memory.
    let mut loads: Vec<Load> = input.node_memory.iter().map(|m| Load { max: s.allowed_for_ml(*m), assigned: 0, jobs: 0, batch: 0 }).collect();
    let mut assigned: Vec<&MlJob> = input.jobs.iter().filter(|j| j.assigned).collect();
    assigned.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    for j in &assigned {
        if let Some(l) = loads.iter_mut().max_by_key(|l| l.free()) {
            l.assign(j.bytes);
            if j.analytics { l.batch += j.bytes; }
        }
    }
    // `checkForScaleUp`.
    if wa > s.analytics_queue || wm > s.anomaly_queue {
        let mut after = loads.clone();
        let anomaly = unassignable(&waiting_anomaly, s.anomaly_queue, &mut after).unwrap_or((0, 0));
        let analytics = unassignable(&waiting_analytics, s.analytics_queue, &mut after).unwrap_or((0, 0));
        if anomaly != (0, 0) || analytics != (0, 0) {
            let max_free = after.iter().map(Load::free_excluding_overhead).max().unwrap_or(0);
            let max_free = if max_free > cur_node || max_free > cur_tier { 0 } else { max_free };
            let tier = cur_tier as i64 - max_free as i64 + anomaly.0 as i64 + analytics.0 as i64;
            let node = cur_node.max(anomaly.1).max(analytics.1);
            timer.reset();
            return ml_decision(Some(s.capacity_for(tier, node)), "requesting scale up as number of jobs in queues exceeded configured limit \
                or there is at least one trained model waiting for assignment and current capacity is not large enough for waiting jobs or models".into(), wm, wa);
        }
    }
    if wm + wa > 0 {
        // Could the waiting jobs ever be assigned? (`calculateFutureAvailableCapacity`:
        // free memory plus the batch jobs' once they finish.)
        let future_tier: u64 = loads.iter().map(|l| l.free_excluding_overhead() + l.batch).sum();
        let waiting_sum: u64 = waiting_anomaly.iter().chain(&waiting_analytics).sum();
        let waiting_max = waiting_anomaly.iter().chain(&waiting_analytics).copied().max().unwrap_or(0);
        let new_tier = waiting_sum as i64 - future_tier as i64;
        let new_node = cur_node.max(waiting_max);
        timer.reset();
        if new_node > cur_node || new_tier > 0 {
            return ml_decision(Some(s.capacity_for(cur_tier as i64 + new_tier.max(0), new_node)),
                "scaling up as adequate space would not automatically become available when running jobs finish".into(), wm, wa);
        }
        return ml_decision(current, format!("Passing currently perceived capacity as there are [0] model snapshot upgrades, [{wa}] analytics and \
            [{wm}] anomaly detection jobs in the queue, [0] trained models not fully-allocated, but the number in the queue is less than the \
            configured maximum allowed or the queued jobs will eventually be assignable at the current size."), wm, wa);
    }
    // `checkForScaleDown`.
    let largest = assigned.first().map(|j| j.bytes).unwrap_or(0);
    let necessary: u64 = assigned.iter().map(|j| j.bytes).sum();
    if necessary < cur_tier || largest < cur_node {
        let (node, tier) = s.capacity_for(necessary.min(cur_tier) as i64, largest.min(cur_node));
        // `ensureScaleDown`: never above the current capacity.
        let down = (node.min(input.current_node_memory), tier.min(input.current_tier_memory));
        if down != (input.current_node_memory, input.current_tier_memory) {
            let left = timer.mark_down(now_ms, s.down_scale_delay_ms);
            if left <= 0 {
                return ml_decision(Some(down), "Requesting scale down as tier and/or node size could be smaller".into(), 0, 0);
            }
            return ml_decision(current, delay_message(left, timer.down_detected_ms), 0, 0);
        }
    }
    timer.reset();
    ml_decision(current, "Passing currently perceived capacity as no scaling changes are necessary".into(), 0, 0)
}

// ------------------------------------------------------------ standalone

/// A standalone node's result for `decider` of a policy: the node is the
/// whole tier when the policy's roles are its own (`in_tier`), else the
/// tier is empty. `timer`: this policy's ML scale-down timer.
pub fn standalone_decision(node: &crate::node::Node, decider: &str, settings: &serde_json::Map<String, Value>, in_tier: bool, disk: NodeDisk,
    machine_memory: u64, timer: &mut ScaleTimer) -> Decision {
    let get = |k: &str| settings.get(k).and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_number().map(|n| n.to_string())));
    let nodes: Vec<NodeDisk> = if in_tier { vec![disk] } else { Vec::new() };
    let current_storage = Some(if in_tier { disk.total } else { 0 });
    let th = DiskThresholds::from_settings(|k| crate::ilm::cluster_setting(node, k));
    let time = |k: &str, default: u64| get(k).and_then(|v| crate::date::parse_time_value(&v)).filter(|v| *v >= 0).map(|v| v as u64).unwrap_or(default);
    let now = crate::date::now_millis();
    // Every local copy (a single node holds all of them) and the indices.
    let indices = node.all_indices();
    fn frozen(svc: &crate::index::IndexService) -> bool {
        is_frozen_index(svc.setting("index.routing.allocation.include._tier_preference").as_deref())
    }
    let mut shards: Vec<ShardCopy> = Vec::new();
    for svc in indices.iter().filter(|_| in_tier) {
        for copy in svc.local_shards() {
            shards.push(ShardCopy { id: format!("[{}][{}]", svc.name, copy.id), bytes: copy.shard.store_size(), node: in_tier.then_some(0) });
        }
    }
    match decider {
        "reactive_storage" => reactive_storage(current_storage, &nodes, &shards, &th),
        "proactive_storage" => {
            let streams: Vec<StreamIndices> = node.data_streams.read().values().map(|ds| StreamIndices {
                indices: ds.indices.iter().filter_map(|name| node.get_index(name)).map(|svc| BackingIndex {
                    name: svc.name.clone(),
                    creation_ms: svc.setting("index.creation_date").and_then(|v| v.parse().ok()).unwrap_or(-1),
                    bytes: svc.local_shards().iter().map(|c| c.shard.store_size().max(KB)).sum(),
                    copies: svc.local_shards().len() as u64,
                    on_tier: in_tier,
                    off_tier: !in_tier,
                }).collect(),
            }).collect();
            proactive_storage(current_storage, &nodes, &shards, &streams, time("forecast_window", 1_800_000), now, &th)
        }
        "frozen_existence" => {
            let names: Vec<String> = indices.iter().filter(|svc| crate::ilm::frozen_phase(node, svc)).map(|svc| svc.name.clone()).collect();
            frozen_existence(&names)
        }
        "frozen_shards" => {
            let per = get("memory_per_shard").and_then(|v| crate::ilm::parse_size(&v, "memory_per_shard").ok()).map(|(b, _)| b).unwrap_or(DEFAULT_MEMORY_PER_SHARD);
            let count: u64 = indices.iter().filter(|svc| frozen(svc)).map(|svc| svc.local_shards().len() as u64).sum();
            frozen_shards(count, per)
        }
        "frozen_storage" => {
            let data_set: u64 = indices.iter().filter(|svc| frozen(svc)).map(|svc| svc.store_size()).sum();
            frozen_storage(data_set, get("percentage").and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_FROZEN_STORAGE_PERCENTAGE))
        }
        "ml" => {
            let queue = |k: &str| get(k).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
            let setting = |k: &str| crate::ilm::cluster_setting(node, k);
            let s = MlSettings {
                anomaly_queue: queue("num_anomaly_jobs_in_queue"),
                analytics_queue: queue("num_analytics_jobs_in_queue"),
                down_scale_delay_ms: time("down_scale_delay", 3_600_000),
                max_machine_memory_percent: setting("xpack.ml.max_machine_memory_percent").and_then(|v| v.trim().parse().ok()).unwrap_or(30),
                use_auto: setting("xpack.ml.use_auto_machine_memory_percent").as_deref() == Some("true"),
                max_ml_node_size: setting("xpack.ml.max_ml_node_size").and_then(|v| crate::ilm::parse_size(&v, "xpack.ml.max_ml_node_size").ok()).map(|(b, _)| b).unwrap_or(0),
            };
            let jobs = if in_tier { ml_jobs(node) } else { Vec::new() };
            let memory: Vec<u64> = if in_tier { vec![machine_memory] } else { Vec::new() };
            let m = if in_tier { machine_memory } else { 0 };
            ml(&MlInput { jobs: &jobs, node_memory: &memory, current_tier_memory: m, current_node_memory: m }, &s, timer, now)
        }
        other => Decision { capacity: None, summary: format!("unknown decider [{other}]"), details: json!({}) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(id: &str, bytes: u64, node: Option<usize>) -> ShardCopy {
        ShardCopy { id: id.into(), bytes, node }
    }

    /// Reactive storage (ES `ReactiveStorageDeciderService.scale`): current
    /// storage plus refused unassigned copies plus what over-watermark
    /// nodes cannot shed; node size = largest copy below the low watermark
    /// plus 10mb.
    #[test]
    fn reactive_storage_follows_es() {
        let th = DiskThresholds::default();
        let nodes = [NodeDisk { total: 100 * GB, free: 50 * GB }, NodeDisk { total: 100 * GB, free: 60 * GB }];
        let ok = reactive_storage(Some(200 * GB), &nodes, &[copy("[a][0]", 9 * GB, Some(0))], &th);
        assert_eq!(ok.summary, "storage ok");
        let c = ok.capacity.unwrap();
        assert_eq!(c.total_storage, Some(200 * GB));
        // 9gb / 0.85, rounded up, plus the 10mb node overhead.
        assert_eq!(c.node_storage, Some(((100 * 9 * GB) as f64 / 85.0).ceil() as u64 + 10 * MB));
        // A 30gb replica fits a node (60gb free, 10gb kept above high).
        let fits = reactive_storage(Some(200 * GB), &nodes, &[copy("[a][0]", 30 * GB, None)], &th);
        assert_eq!(fits.capacity.unwrap().total_storage, Some(200 * GB));
        // A 55gb one fits no node: required grows by it.
        let grow = reactive_storage(Some(200 * GB), &nodes, &[copy("[a][0]", 55 * GB, None)], &th);
        assert_eq!(grow.capacity.unwrap().total_storage, Some(255 * GB));
        assert_eq!(grow.summary, "not enough storage available, needs 55gb");
        assert_eq!(grow.details["unassigned_shards"], json!(["[a][0]"]));
        // A node past the high watermark (5gb free, 10gb kept) whose 20gb copy
        // no other node takes: max(missing 5gb, smallest unmovable copy).
        let full = [NodeDisk { total: 100 * GB, free: 5 * GB }, NodeDisk { total: 100 * GB, free: 25 * GB }];
        let over = reactive_storage(Some(200 * GB), &full, &[copy("[b][0]", 20 * GB, Some(0))], &th);
        assert_eq!(over.capacity.unwrap().total_storage, Some(220 * GB));
        // A tier with no nodes needs every unassigned copy.
        assert_eq!(reactive_storage(Some(0), &[], &[copy("[c][0]", 0, None)], &th).capacity.unwrap().total_storage, Some(KB));
        assert_eq!(reactive_storage(None, &nodes, &[], &th).capacity, None);
        // Max headroom: a 10tb disk keeps 150gb above high, not 1tb.
        assert_eq!(th.high_free(10 << 40), 150 * GB);
        let set = DiskThresholds::from_settings(|k| (k == "cluster.routing.allocation.disk.watermark.high").then(|| "95%".to_string()));
        assert_eq!(set.high_headroom, None, "an explicit watermark has no default headroom");
        assert_eq!(set.high_free(100 * GB), 5 * GB);
    }

    /// Proactive (ES `AllocationState.forecast`): the data stream's recent
    /// indices extrapolated over the window become unassigned copies.
    #[test]
    fn proactive_storage_forecasts_rollovers() {
        let th = DiskThresholds::default();
        let now = 10_000_000;
        let nodes = [NodeDisk { total: 100 * GB, free: 12 * GB }];
        // Two indices, 20gb each (all copies), 2 copies.
        let idx = |name: &str, created: i64| BackingIndex { name: name.into(), creation_ms: created, bytes: 20 * GB, copies: 2, on_tier: true, off_tier: false };
        let ds = StreamIndices { indices: vec![idx(".ds-1", now - 3_600_000), idx(".ds-2", now - 1_800_000)] };
        let d = proactive_storage(Some(100 * GB), &nodes, &[], &[ds.clone()], 1_800_000, now, &th);
        // The newest index already reaches back the 30 minutes: one new
        // 20gb index as two 10gb copies; the node is past the low watermark
        // (12gb free, 15gb kept): both unassigned.
        assert_eq!(d.capacity.unwrap().total_storage, Some(120 * GB));
        assert_eq!(d.details["forecasted"], json!(20 * GB));
        assert_eq!(d.details["forecast_window"], "30m");
        // Off the tier: no forecast.
        let off = StreamIndices { indices: ds.indices.iter().map(|i| BackingIndex { on_tier: false, off_tier: true, ..i.clone() }).collect() };
        assert_eq!(proactive_storage(Some(100 * GB), &nodes, &[], &[off], 1_800_000, now, &th).capacity.unwrap().total_storage, Some(100 * GB));
    }

    #[test]
    fn frozen_deciders_follow_es() {
        let none = frozen_existence(&[]);
        assert_eq!(none.capacity.unwrap(), Capacity { total_storage: Some(0), total_memory: Some(0), ..Default::default() });
        let some = frozen_existence(&["a".into(), "b".into()]);
        assert_eq!(some.capacity.unwrap().node_storage, Some(8 * GB));
        assert_eq!(some.summary, "indices [a, b]");
        assert_eq!(frozen_shards(10, DEFAULT_MEMORY_PER_SHARD).capacity.unwrap().total_memory, Some(10 * DEFAULT_MEMORY_PER_SHARD));
        assert_eq!(frozen_shards(3, DEFAULT_MEMORY_PER_SHARD).summary, "shard count [3]");
        assert_eq!(frozen_storage(1000, 5.0).capacity.unwrap().total_storage, Some(50));
        assert_eq!(frozen_storage(1999, 5.0).capacity.unwrap().total_storage, Some(99), "rounded down as ES's long arithmetic");
        assert!(is_frozen_index(Some("data_frozen")) && !is_frozen_index(Some("data_hot,data_frozen")) && !is_frozen_index(None));
    }

    /// ML (ES `MlMemoryAutoscalingDecider`): node size from native memory
    /// at `max_machine_memory_percent`; scale up past the queue allowance;
    /// scale down only after `down_scale_delay`.
    #[test]
    fn ml_follows_es() {
        let s = MlSettings::default();
        let mut timer = ScaleTimer::default();
        // No ML nodes, one waiting 1gb+10mb job: scale up from zero.
        let job = |bytes, assigned| MlJob { bytes, assigned, analytics: false };
        let waiting = [job(GB + 10 * MB, false)];
        let d = ml(&MlInput { jobs: &waiting, node_memory: &[], current_tier_memory: 0, current_node_memory: 0 }, &s, &mut timer, 0);
        let node = ((100.0 / 30.0) * (GB + 10 * MB + 30 * MB) as f64).ceil() as u64;
        assert_eq!(d.capacity.unwrap().node_memory, Some(node));
        assert_eq!(d.capacity.unwrap().total_memory, Some(node));
        assert!(d.summary.contains("there are no machine learning nodes"), "{}", d.summary);
        // A 4gb ML node (1.2gb ML memory) running a 500mb job, a 1gb job waiting.
        let jobs = [job(500 * MB, true), job(GB, false)];
        let input = MlInput { jobs: &jobs, node_memory: &[4 * GB], current_tier_memory: 4 * GB, current_node_memory: 4 * GB };
        let up = ml(&input, &s, &mut timer, 0);
        assert!(up.summary.starts_with("requesting scale up as number of jobs in queues exceeded"), "{}", up.summary);
        assert!(up.capacity.unwrap().total_memory.unwrap() > 4 * GB);
        // Allowed to queue one: within the queue and assignable later? No:
        // the node can never hold it (no batch job frees memory) -> scale up.
        let queued = ml(&input, &MlSettings { anomaly_queue: 1, ..s }, &mut timer, 0);
        assert!(queued.summary.starts_with("scaling up as adequate space"), "{}", queued.summary);
        // Only a small job on a big node: scale down after the delay.
        let small = [job(100 * MB, true)];
        let input = MlInput { jobs: &small, node_memory: &[16 * GB], current_tier_memory: 16 * GB, current_node_memory: 16 * GB };
        let wait = ml(&input, &s, &mut timer, 1_000);
        assert!(wait.summary.contains("down scale delay has not been satisfied"), "{}", wait.summary);
        assert_eq!(wait.capacity.unwrap().total_memory, Some(16 * GB));
        let down = ml(&input, &s, &mut timer, 1_000 + 3_600_000);
        assert_eq!(down.summary, "Requesting scale down as tier and/or node size could be smaller");
        assert!(down.capacity.unwrap().total_memory.unwrap() < 16 * GB);
        // No jobs and no ML nodes: nothing to change.
        let idle = ml(&MlInput { jobs: &[], node_memory: &[], current_tier_memory: 0, current_node_memory: 0 }, &s, &mut ScaleTimer::default(), 0);
        assert_eq!(idle.summary, "Passing currently perceived capacity as no scaling changes are necessary");
    }
}
