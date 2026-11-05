//! GPU vector indexing (ES `x-pack/plugin/gpu`, `GPUPlugin`).
//!
//! * Node setting `vectors.indexing.use_gpu` (`true` / `false` / `auto`,
//!   default `auto`, static): `auto` uses a supported GPU when there is
//!   one, `true` requires one (bootstrap check), `false` never uses it.
//! * GPU indexing is used for `hnsw` / `int8_hnsw` fields of `float`
//!   vectors when the node allows it and the licence is enterprise (ES
//!   `GPUPlugin.getVectorsFormatProvider`); every other format, and every
//!   build when not allowed, stays on the CPU.
//! * Reporting: `_xpack` info `features.gpu_vector_indexing`
//!   (`GpuInfoTransportAction`), `_xpack/usage` `gpu_vector_indexing`
//!   (`GpuUsageTransportAction` over per-node [`NodeGpuStats`],
//!   `TransportGpuStatsAction`), cluster feature `vectors.indexing.use_gpu`.
//!
//! Detection and the CAGRA builds live in rustic-core `vector::gpu`; without
//! the `gpu` cargo feature no GPU is ever detected and everything here
//! reports GPU indexing as unavailable, as ES does on a node without a GPU.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;
use rustic_core::vector::gpu::{self, GpuGate, GpuInfo};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::node::Node;

/// `GPUPlugin.VECTORS_INDEXING_USE_GPU_NODE_SETTING`.
pub const USE_GPU_SETTING: &str = "vectors.indexing.use_gpu";
/// `GPUFeatures.VECTORS_INDEXING_USE_GPU`: published by a node that indexes
/// vectors on its GPU.
pub const USE_GPU_FEATURE: &str = "vectors.indexing.use_gpu";
/// `GpuStatsAction.NAME`.
pub const STATS_ACTION: &str = "cluster:monitor/xpack/gpu_vector_indexing/stats/dist";

/// `GPUPlugin.GpuMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMode {
    True,
    False,
    Auto,
}

impl GpuMode {
    /// `Setting.enumSetting`: the constant named by the upper-cased value.
    pub fn parse(v: &str) -> Result<GpuMode, String> {
        match v.to_uppercase().as_str() {
            "TRUE" => Ok(GpuMode::True),
            "FALSE" => Ok(GpuMode::False),
            "AUTO" => Ok(GpuMode::Auto),
            other => Err(format!("No enum constant org.elasticsearch.xpack.gpu.GPUPlugin.GpuMode.{other}")),
        }
    }
}

/// The mode in node `settings` (default `auto`).
pub fn mode_of(settings: &Map<String, Value>) -> Result<GpuMode, String> {
    match settings.get(USE_GPU_SETTING) {
        None | Some(Value::Null) => Ok(GpuMode::Auto),
        Some(Value::String(s)) => GpuMode::parse(s),
        Some(Value::Bool(b)) => Ok(if *b { GpuMode::True } else { GpuMode::False }),
        Some(other) => GpuMode::parse(&other.to_string()),
    }
}

/// This node's mode (invalid values fail at startup, see [`bootstrap_check`]).
pub fn mode(node: &Node) -> GpuMode {
    mode_of(&node.node_settings.read()).unwrap_or(GpuMode::Auto)
}

/// Settings validation plus `GPUPlugin.GpuModeBootstrapCheck`: with
/// `use_gpu: true` and no usable GPU the node refuses to start, when
/// bootstrap checks are enforced (ES enforces them on a node bound to a
/// non-loopback address; on loopback the failure is only logged).
pub fn bootstrap_check(settings: &Map<String, Value>, enforce: bool) -> Result<(), String> {
    let mode = mode_of(settings).map_err(|e| format!("illegal value for setting [{USE_GPU_SETTING}]: {e}"))?;
    if mode == GpuMode::True && gpu::info().is_none() {
        let msg = format!("{USE_GPU_SETTING} is set to [true], but GPU resources are not accessible on this node. Check logs for details.");
        if enforce {
            return Err(format!("bootstrap check failure: {msg}"));
        }
        tracing::warn!("bootstrap check failure: {msg}");
    }
    Ok(())
}

/// A supported GPU is present (ES `CuVSGPUSupport.isSupported`).
pub fn supported() -> bool {
    gpu::info().is_some()
}

/// ES `localNodeGpuEnabled` / `gpuSettingEnabled`: a supported GPU and not
/// `use_gpu: false`.
pub fn setting_enabled(mode: GpuMode, supported: bool) -> bool {
    mode != GpuMode::False && supported
}

/// `_xpack` info `features.gpu_vector_indexing.enabled` of this node.
pub fn info_enabled(node: &Node) -> bool {
    setting_enabled(mode(node), gpu::info().is_some())
}

/// `GPUPlugin.GPU_INDEXING_FEATURE.checkWithoutTracking`: enterprise.
pub fn license_available(node: &Node) -> bool {
    crate::license::current(node).effective_mode(crate::date::now_millis()).allows(crate::license::Mode::Enterprise)
}

/// ES `GPUPlugin.getVectorsFormatProvider`: whether this node builds graphs
/// on the GPU now (`true`, or `auto` with a GPU, and an enterprise licence).
pub fn indexing_allowed(node: &Node) -> bool {
    let m = mode(node);
    if gpu::info().is_none() || m == GpuMode::False {
        return false;
    }
    if license_available(node) {
        return true;
    }
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "The current configuration supports GPU indexing, but it is not allowed by the current license. If this is intentional, it is possible to suppress this message by setting [{USE_GPU_SETTING}] to FALSE"
        );
    }
    false
}

/// ES `GPUPlugin.vectorIndexAndElementTypeSupported` (plus the int8 +
/// `max_inner_product` combination ES's GPU format rejects, which stays on
/// the CPU here).
pub fn format_supported(index_type: &str, float_elements: bool, similarity: &str) -> bool {
    float_elements && (index_type == "hnsw" || (index_type == "int8_hnsw" && similarity != "max_inner_product"))
}

/// One node's GPU state (`NodeGpuStatsResponse`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeGpuStats {
    pub gpu_supported: bool,
    pub gpu_setting_enabled: bool,
    pub gpu_usage_count: u64,
    pub total_gpu_memory_in_bytes: u64,
    pub gpu_name: Option<String>,
}

/// `TransportGpuStatsAction.nodeOperation`.
pub fn node_stats(mode: GpuMode, info: Option<&GpuInfo>, usage_count: u64) -> NodeGpuStats {
    NodeGpuStats {
        gpu_supported: info.is_some(),
        gpu_setting_enabled: setting_enabled(mode, info.is_some()),
        gpu_usage_count: usage_count,
        total_gpu_memory_in_bytes: info.map_or(0, |i| i.total_memory),
        gpu_name: info.map(|i| i.name.clone()),
    }
}

/// This node's [`NodeGpuStats`].
pub fn local_stats(node: &Node) -> NodeGpuStats {
    node_stats(mode(node), gpu::info(), gpu::usage_count())
}

/// `GpuUsageTransportAction.buildUsage` rendered as
/// `GpuVectorIndexingFeatureSetUsage`.
pub fn usage_json(available: bool, nodes: &[NodeGpuStats]) -> Value {
    let mut enabled = false;
    let mut count = 0u64;
    let mut with_gpu = Vec::new();
    for n in nodes {
        if n.gpu_supported {
            with_gpu.push(json!({
                "type": n.gpu_name,
                "memory_in_bytes": n.total_gpu_memory_in_bytes,
                "enabled": n.gpu_setting_enabled,
                "index_build_count": n.gpu_usage_count,
            }));
        }
        enabled |= n.gpu_setting_enabled;
        count += n.gpu_usage_count;
    }
    json!({"available": available, "enabled": enabled, "index_build_count": count, "nodes_with_gpu": with_gpu.len(), "nodes": with_gpu})
}

// ---------------------------------------------------------------------------
// Per-node policy for shards (`OptimizerPolicy::gpu`).

/// Nodes of this process by data directory: shards find the node that
/// owns them by path (several in-process nodes can coexist in tests).
static NODES: RwLock<Vec<(PathBuf, Weak<Node>)>> = RwLock::new(Vec::new());

fn gate(node: Weak<Node>) -> GpuGate {
    if gpu::info().is_none() {
        // No GPU: shards keep the default gate and pay nothing.
        return GpuGate::default();
    }
    GpuGate::new(move || node.upgrade().is_some_and(|n| indexing_allowed(&n)))
}

/// The GPU gate of a shard stored under `path` (the default gate when no
/// started node owns it).
pub fn gate_for_path(path: &Path) -> GpuGate {
    let nodes = NODES.read();
    nodes
        .iter()
        .filter(|(dir, w)| path.starts_with(dir) && w.strong_count() > 0)
        .max_by_key(|(dir, _)| dir.as_os_str().len())
        .map_or_else(GpuGate::default, |(_, w)| gate(w.clone()))
}

/// Register a node whose settings are final (called when its REST layer
/// starts): its shards may build on the GPU from now on.
pub fn start(node: &Arc<Node>) {
    {
        let mut nodes = NODES.write();
        nodes.retain(|(dir, w)| w.strong_count() > 0 && dir != &node.data);
        nodes.push((node.data.clone(), Arc::downgrade(node)));
    }
    if gpu::info().is_none() {
        return;
    }
    tracing::info!("GPU vector indexing: mode [{:?}], enabled [{}]", mode(node), info_enabled(node));
    for svc in node.all_indices() {
        for sh in svc.local_shards() {
            sh.shard.core.read().vector_store().set_gpu_gate(gate(Arc::downgrade(node)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(v: Value) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert(USE_GPU_SETTING.into(), v);
        m
    }

    #[test]
    fn mode_setting_parses_like_es_enum_setting() {
        assert_eq!(mode_of(&Map::new()), Ok(GpuMode::Auto));
        assert_eq!(mode_of(&settings(json!("true"))), Ok(GpuMode::True));
        assert_eq!(mode_of(&settings(json!("FALSE"))), Ok(GpuMode::False));
        assert_eq!(mode_of(&settings(json!("Auto"))), Ok(GpuMode::Auto));
        assert_eq!(mode_of(&settings(json!(false))), Ok(GpuMode::False));
        assert_eq!(mode_of(&settings(json!("yes"))), Err("No enum constant org.elasticsearch.xpack.gpu.GPUPlugin.GpuMode.YES".into()));
        assert!(bootstrap_check(&settings(json!("maybe")), false).unwrap_err().contains("illegal value for setting [vectors.indexing.use_gpu]"));
    }

    #[test]
    fn bootstrap_check_refuses_true_without_gpu_when_enforced() {
        if gpu::info().is_some() {
            assert!(bootstrap_check(&settings(json!("true")), true).is_ok());
            return;
        }
        let err = bootstrap_check(&settings(json!("true")), true).unwrap_err();
        assert!(err.ends_with("vectors.indexing.use_gpu is set to [true], but GPU resources are not accessible on this node. Check logs for details."), "{err}");
        // Loopback (development) nodes only log it; other modes always pass.
        assert!(bootstrap_check(&settings(json!("true")), false).is_ok());
        assert!(bootstrap_check(&settings(json!("auto")), true).is_ok());
        assert!(bootstrap_check(&settings(json!("false")), true).is_ok());
    }

    #[test]
    fn only_float_hnsw_and_int8_hnsw_use_the_gpu() {
        assert!(format_supported("hnsw", true, "l2_norm"));
        assert!(format_supported("int8_hnsw", true, "cosine"));
        assert!(format_supported("hnsw", true, "max_inner_product"));
        assert!(!format_supported("int8_hnsw", true, "max_inner_product"));
        for t in ["int4_hnsw", "bbq_hnsw", "flat", "int8_flat", "bbq_disk"] {
            assert!(!format_supported(t, true, "cosine"), "{t}");
        }
        assert!(!format_supported("hnsw", false, "cosine"), "byte / bit / bfloat16 vectors stay on the CPU");
    }

    #[test]
    fn usage_matches_gpu_usage_transport_action() {
        let t4 = GpuInfo { name: "Tesla T4".into(), total_memory: 15_655_829_504, compute_capability: (7, 5), ordinal: 0 };
        // No GPU anywhere (`gpu/not-supported/30_usage.yml`).
        let none = node_stats(GpuMode::Auto, None, 0);
        assert_eq!(none, NodeGpuStats { gpu_supported: false, gpu_setting_enabled: false, gpu_usage_count: 0, total_gpu_memory_in_bytes: 0, gpu_name: None });
        assert_eq!(usage_json(true, &[none.clone(), none]), json!({"available": true, "enabled": false, "index_build_count": 0, "nodes_with_gpu": 0, "nodes": []}));
        // A GPU node with `use_gpu: false` is listed but not enabled.
        let off = node_stats(GpuMode::False, Some(&t4), 0);
        let on = node_stats(GpuMode::True, Some(&t4), 3);
        let u = usage_json(false, &[off, on]);
        assert_eq!(u, json!({"available": false, "enabled": true, "index_build_count": 3, "nodes_with_gpu": 2, "nodes": [
            {"type": "Tesla T4", "memory_in_bytes": 15_655_829_504u64, "enabled": false, "index_build_count": 0},
            {"type": "Tesla T4", "memory_in_bytes": 15_655_829_504u64, "enabled": true, "index_build_count": 3},
        ]}));
        let keys: Vec<&String> = u.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["available", "enabled", "index_build_count", "nodes_with_gpu", "nodes"]);
        assert_eq!(serde_json::from_value::<NodeGpuStats>(serde_json::to_value(node_stats(GpuMode::Auto, Some(&t4), 7)).unwrap()).unwrap().gpu_usage_count, 7);
    }

    #[test]
    fn unowned_paths_get_the_default_gate() {
        assert!(!gate_for_path(Path::new("/nonexistent/rustic-gpu-test")).is_set());
    }
}
