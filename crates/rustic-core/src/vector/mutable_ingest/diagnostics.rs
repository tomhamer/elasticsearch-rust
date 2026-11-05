//! Opt-in Linux diagnostics, never enabled for authoritative throughput runs.
//! Task CPU is scheduler runtime, not task wall time. Process runtime includes
//! other batches/searches and must not be attributed to this batch alone.
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering}};
use std::time::Instant;
use parking_lot::Mutex;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) struct Batch {
    id: u64,
    field: u16,
    size: usize,
    start: Instant,
    samples: Mutex<Vec<(u64, u64, Option<u64>)>>,
}

fn runtime(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.split_whitespace().next()?.parse().ok()
}
fn process_runtime() -> Option<u64> {
    let mut sum = 0u64;
    for entry in std::fs::read_dir("/proc/self/task").ok()? {
        let path = entry.ok()?.path().join("schedstat");
        sum = sum.checked_add(runtime(path.to_str()?)?)?;
    }
    Some(sum)
}
fn ns(d: std::time::Duration) -> u64 { d.as_nanos().min(u64::MAX as u128) as u64 }

impl Batch {
    pub(super) fn begin(field: u16, size: usize) -> Option<Self> {
        static EVERY: OnceLock<u64> = OnceLock::new();
        let every = *EVERY.get_or_init(|| std::env::var("RUSTIC_HNSW_DIAGNOSTIC_EVERY")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0));
        if every == 0 { return None; }
        let id = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        if id % every != 0 { return None; }
        Some(Self { id, field, size, start: Instant::now(), samples: Mutex::new(Vec::with_capacity(size)) })
    }
    pub(super) fn task<T>(&self, f: impl FnOnce() -> T) -> T {
        let start = ns(self.start.elapsed());
        let cpu = runtime("/proc/thread-self/schedstat");
        let value = f();
        let end_cpu = runtime("/proc/thread-self/schedstat");
        let end = ns(self.start.elapsed());
        self.samples.lock().push((start, end, cpu.zip(end_cpu).and_then(|(a,b)| b.checked_sub(a))));
        value
    }
    pub(super) fn completion(&self) -> (Instant, Option<u64>) {
        (Instant::now(), process_runtime())
    }
    pub(super) fn finish(&self, completion: (Instant, Option<u64>)) {
        let end_cpu = process_runtime();
        let completion_ns = ns(completion.0.elapsed());
        let prepare_ns = ns(completion.0.duration_since(self.start));
        let record = serde_json::json!({
            "diagnostic_only": true, "pid": std::process::id(), "batch": self.id,
            "field": self.field, "size": self.size,
            "prepare_wall_ns": prepare_ns, "complete_wall_ns": completion_ns,
            "complete_process_cpu_ns": completion.1.zip(end_cpu).and_then(|(a,b)| b.checked_sub(a)),
            "tasks_start_end_cpu_ns": &*self.samples.lock(),
        });
        tracing::info!(target: "hnsw_ingest_diagnostic", record = %record, "HNSW batch phase diagnostic");
    }
}

/// Independent low-overhead lock sampling; no /proc reads under the writer.
pub(crate) fn writer_start() -> Option<Instant> {
    static EVERY: OnceLock<u64> = OnceLock::new();
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let every = *EVERY.get_or_init(|| std::env::var("RUSTIC_HNSW_WRITER_DIAGNOSTIC_EVERY")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0));
    if every == 0 || SEQ.fetch_add(1, Ordering::Relaxed) % every != 0 { return None; }
    Some(Instant::now())
}
pub(crate) fn writer_finish(start: Instant, acquired: Instant, finished: Instant) {
    tracing::info!(target: "hnsw_ingest_diagnostic",
        wait_ns = ns(acquired.duration_since(start)),
        held_ns = ns(finished.duration_since(acquired)),
        "HNSW writer diagnostic");
}
