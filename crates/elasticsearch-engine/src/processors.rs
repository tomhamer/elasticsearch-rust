//! The node's allocated processors: `node.processors` when set (ES then
//! starts the JVM with `-XX:ActiveProcessorCount`, so OS-reported and
//! allocated processors are both this value), else the machine's.
//! Thread-pool sizes and defaults such as `index.merge.scheduler.max_thread_count`
//! derive from it, as in ES (`EsExecutors.allocatedProcessors`).

use std::sync::OnceLock;

static PROCESSORS: OnceLock<usize> = OnceLock::new();
static MACHINE: OnceLock<usize> = OnceLock::new();

/// (cached: `available_parallelism` reads cgroup files on every call.)
fn machine() -> usize {
    *MACHINE.get_or_init(|| std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1))
}

/// The allocated processors.
pub fn get() -> usize {
    PROCESSORS.get().copied().unwrap_or_else(machine)
}

/// Apply `node.processors` from node settings (once per process).
pub fn configure(node_settings: &serde_json::Map<String, serde_json::Value>) {
    if let Some(n) = node_settings.get("node.processors").and_then(|v| allocated(v, machine())) {
        let _ = PROCESSORS.set(n);
    }
}

/// `node.processors` rounded up, capped at the machine's processors.
fn allocated(v: &serde_json::Value, machine: usize) -> Option<usize> {
    let n = v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))?;
    (n > 0.0).then(|| (n.ceil() as usize).clamp(1, machine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn node_processors_round_up_and_cap_at_the_machine() {
        assert_eq!(allocated(&json!("2"), 16), Some(2));
        assert_eq!(allocated(&json!(1.5), 16), Some(2));
        assert_eq!(allocated(&json!("64"), 16), Some(16));
        assert_eq!(allocated(&json!("0"), 16), None);
        assert_eq!(allocated(&json!("x"), 16), None);
    }
}
