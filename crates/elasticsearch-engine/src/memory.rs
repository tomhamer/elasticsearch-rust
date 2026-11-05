//! Node memory budget and circuit breakers.
//!
//! There is no JVM heap: everything the node allocates is the process's
//! anonymous memory, and segment files are mapped (OS page cache, outside
//! the budget as it is outside ES's heap). Two figures stand in for ES's
//! heap:
//! - the **budget** ([`budget`], internal): the memory available to the
//!   process (the cgroup limit when there is one, else physical RAM). The
//!   breaker limits are fractions of it, and the real-memory parent
//!   compares the process's anonymous memory with it. `-Xmx` never sets
//!   it: the process holds graphs, vector rows and memtables ES keeps off
//!   heap, so a JVM heap size would trip every request;
//! - the **reported heap** ([`es_heap`]): `-Xmx` from `ES_JAVA_OPTS` when
//!   set, else the heap ES would size on this machine (half the available
//!   memory, at most 31gb). Every heap-facing figure (`jvm.mem.heap_max`,
//!   `heap.percent`, `ml.max_jvm_size`, the indexing buffer, indexing
//!   pressure) uses it, so they read as they would on ES.
//!
//! Breakers follow ES's `HierarchyCircuitBreakerService`: child breakers
//! (`request`, `fielddata`, `inflight_requests`, `model_inference`,
//! `eql_sequence`, and ES 7's `accounting`, enforced but never shown as
//! ES 9 has none) reserve bytes before large allocations
//! with a CAS loop (no lock), each against its own limit times its
//! overhead, then the `parent` checks the total: with
//! `indices.breaker.total.use_real_memory` (default) the process's real
//! anonymous memory plus the new bytes, else the children's sum. A trip
//! fails the request with ES's `circuit_breaking_exception` (HTTP 429).
//! Every reservation is a guard released on drop, so errors, cancellation
//! and early termination release what they hold.
//!
//! The `accounting` and `fielddata` breakers also report long-lived memory
//! measured by its owners (`rustic_core::accounting`: closed memtables,
//! reader views, retained captures, mutable vector graphs and rows,
//! uninverted fielddata) plus the memtables being written.

use crate::error::EsError;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

/// The memory budget in bytes: what the breakers are fractions of (see the
/// module docs; not a heap figure, never reported as one).
pub fn budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| {
        if let Some(xmx) = xmx_from_env() {
            tracing::warn!(xmx, "-Xmx in ES_JAVA_OPTS sets the reported heap only; the circuit breakers use the process's memory limit");
        }
        available_memory()
    })
}

/// The heap figure ES reports for this node (`-Xmx` when set, else ES's
/// auto heap): `jvm.mem.heap_max`, the indexing buffer and indexing
/// pressure are this or fractions of it.
pub fn es_heap() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| xmx_from_env().unwrap_or_else(|| auto_budget(available_memory())))
}

/// ES's auto heap for a node: half the available memory, at least 1gb
/// (or all of it below 2gb), at most 31gb.
pub fn auto_budget(available: u64) -> u64 {
    (available / 2).clamp((1u64 << 30).min(available), 31 << 30)
}

/// `-Xmx<size>` (last one wins) from `ES_JAVA_OPTS`.
fn xmx_from_env() -> Option<u64> {
    let opts = std::env::var("ES_JAVA_OPTS").ok()?;
    opts.split_whitespace().filter_map(|o| o.strip_prefix("-Xmx")).filter_map(parse_jvm_size).last()
}

/// A JVM size (`8g`, `512m`, `1048576k`, bytes).
fn parse_jvm_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last()?.to_ascii_lowercase() {
        'k' => (&s[..s.len() - 1], 1u64 << 10),
        'm' => (&s[..s.len() - 1], 1 << 20),
        'g' => (&s[..s.len() - 1], 1 << 30),
        't' => (&s[..s.len() - 1], 1 << 40),
        _ => (s, 1),
    };
    num.parse::<u64>().ok().map(|n| n * mult).filter(|n| *n > 0)
}

/// Memory available to the process: physical RAM, or the cgroup (v2 or
/// v1) limit when lower.
pub fn available_memory() -> u64 {
    let phys = physical_memory();
    cgroup_limit().map_or(phys, |l| l.min(phys))
}

fn physical_memory() -> u64 {
    if let Ok(m) = std::fs::read_to_string("/proc/meminfo") {
        for l in m.lines() {
            if let Some(v) = l.strip_prefix("MemTotal:") {
                if let Ok(kb) = v.trim().trim_end_matches("kB").trim().parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    // macOS (development): `sysctl hw.memsize`.
    if let Ok(o) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
        if let Ok(n) = String::from_utf8_lossy(&o.stdout).trim().parse::<u64>() {
            return n;
        }
    }
    8 << 30
}

fn cgroup_limit() -> Option<u64> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut best: Option<u64> = None;
    let mut take = |v: Option<u64>| {
        if let Some(v) = v.filter(|v| *v > 0 && *v < (1u64 << 60)) {
            best = Some(best.map_or(v, |b| b.min(v)));
        }
    };
    for line in cg.lines() {
        let mut parts = line.splitn(3, ':');
        let (_, ctrl, path) = (parts.next(), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        if ctrl.is_empty() {
            // cgroup v2: the limit of this group and of every ancestor.
            let mut p = std::path::PathBuf::from("/sys/fs/cgroup").join(path.trim_start_matches('/'));
            loop {
                take(std::fs::read_to_string(p.join("memory.max")).ok().and_then(|s| s.trim().parse().ok()));
                if !p.pop() || p == std::path::Path::new("/sys/fs") {
                    break;
                }
            }
        } else if ctrl.split(',').any(|c| c == "memory") {
            // A container without a cgroup namespace sees the host's path,
            // absent inside it: its own group is the mount's root (ES
            // `OsProbe` retries the hierarchy root the same way).
            let p = std::path::PathBuf::from("/sys/fs/cgroup/memory").join(path.trim_start_matches('/'));
            let read = |p: &std::path::Path| std::fs::read_to_string(p.join("memory.limit_in_bytes")).ok().and_then(|s| s.trim().parse().ok());
            take(read(&p).or_else(|| read(std::path::Path::new("/sys/fs/cgroup/memory"))));
        }
    }
    best
}

/// The process's anonymous resident memory (what the budget covers:
/// mapped segment files are page cache), in bytes. `None` where there is
/// no cheap per-kind figure (not Linux).
fn sample_anon_rss() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for l in s.lines() {
            if let Some(v) = l.strip_prefix("RssAnon:") {
                return v.trim().trim_end_matches("kB").trim().parse::<u64>().ok().map(|kb| kb * 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The process's anonymous resident memory (the breakers' own accounting
/// where the OS has no cheap figure).
pub fn anon_rss() -> u64 {
    sample_anon_rss().unwrap_or_else(|| process_breakers().accounted())
}

/// The indexing buffer (`indices.memory.index_buffer_size`, default 10% of
/// the heap ES would size ([`es_heap`]), bounded by `indices.memory.min_index_buffer_size` (48mb)
/// and `indices.memory.max_index_buffer_size` (unbounded) when given as a
/// percentage): in-memory segments past it are written out.
pub fn indexing_buffer() -> u64 {
    match INDEXING_BUFFER.load(Ordering::Relaxed) {
        0 => (es_heap() / 10).max(48 << 20),
        n => n,
    }
}

/// The configured indexing buffer in bytes (`0`: the default).
static INDEXING_BUFFER: AtomicU64 = AtomicU64::new(0);

/// Apply the node settings `indices.memory.index_buffer_size` (a byte
/// size, or a percentage of ES's heap ([`es_heap`]) bounded by
/// `indices.memory.min_index_buffer_size` / `max_index_buffer_size`) at
/// start; an invalid value fails the start, as in ES.
pub fn configure_indexing_buffer(settings: &serde_json::Map<String, serde_json::Value>) -> Result<(), EsError> {
    let text = |key: &str| settings.get(key).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()));
    let size = |key: &str, value: &str| -> Result<u64, EsError> {
        crate::ingest::util::parse_bytes(value.trim()).and_then(|b| u64::try_from(b).ok())
            .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [{key}] with value [{value}] as a size in bytes")))
    };
    // The bounds are validated whatever form the size takes (ES parses
    // every setting at start).
    let min = match text("indices.memory.min_index_buffer_size") { Some(v) => size("indices.memory.min_index_buffer_size", &v)?, None => 48 << 20 };
    let max = match text("indices.memory.max_index_buffer_size") { Some(v) => size("indices.memory.max_index_buffer_size", &v)?, None => u64::MAX };
    let Some(value) = text("indices.memory.index_buffer_size") else { return Ok(()) };
    let bytes = match value.trim().strip_suffix('%') {
        Some(pct) => {
            let pct: f64 = pct.trim().parse().ok().filter(|p| (0.0..=100.0).contains(p))
                .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [indices.memory.index_buffer_size] with value [{value}] as a percentage")))?;
            ((es_heap() as f64 * pct / 100.0) as u64).max(min).min(max)
        }
        None => size("indices.memory.index_buffer_size", &value)?,
    };
    if bytes == 0 {
        return Err(EsError::illegal_argument(format!("failed to parse setting [indices.memory.index_buffer_size] with value [{value}]: the indexing buffer must not be empty")));
    }
    INDEXING_BUFFER.store(bytes, Ordering::Relaxed);
    Ok(())
}

/// Node-wide accounting of the shards' in-memory segment bytes.
pub struct IndexingBuffer {
    total: AtomicU64,
    active: std::sync::atomic::AtomicUsize,
}

impl IndexingBuffer {
    pub fn global() -> &'static IndexingBuffer {
        static G: IndexingBuffer = IndexingBuffer { total: AtomicU64::new(0), active: std::sync::atomic::AtomicUsize::new(0) };
        &G
    }

    /// A shard's bytes went from `old` to `new`: the node-wide total and
    /// the number of shards holding any.
    pub fn update(&self, old: u64, new: u64) -> (u64, usize) {
        let total = if new >= old { self.total.fetch_add(new - old, Ordering::Relaxed) + (new - old) } else { self.total.fetch_sub(old - new, Ordering::Relaxed).saturating_sub(old - new) };
        let active = match (old > 0, new > 0) {
            (false, true) => self.active.fetch_add(1, Ordering::Relaxed) + 1,
            (true, false) => self.active.fetch_sub(1, Ordering::Relaxed).saturating_sub(1),
            _ => self.active.load(Ordering::Relaxed),
        };
        (total, active)
    }

    /// Bytes held in memory by every shard.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Shards holding indexing-buffer bytes now.
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }
}

/// ES `ByteSizeValue.toString()`: the largest unit, one decimal when not
/// whole, truncated (`Strings.format1Decimals`: 5153960755 is `4.7gb`).
pub fn byte_size(n: u64) -> String {
    let units = [("pb", 1u64 << 50), ("tb", 1 << 40), ("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10)];
    for (u, s) in units {
        if n >= s {
            let tenths = (n as u128 * 10 / s as u128) as u64;
            return match tenths % 10 {
                0 => format!("{}{u}", tenths / 10),
                d => format!("{}.{d}{u}", tenths / 10),
            };
        }
    }
    format!("{n}b")
}

/// `N/size` as ES's breaker messages print a byte count.
fn bytes_safe(n: u64) -> String {
    format!("{n}/{}", byte_size(n))
}

/// The reference-docs link ES appends to circuit breaker messages.
pub fn docs_suffix() -> String {
    let minor = crate::node::ES_VERSION.rsplitn(2, '.').last().unwrap_or("9.5");
    format!("; for more information, see https://www.elastic.co/docs/troubleshoot/elasticsearch/circuit-breaker-errors?version={minor}")
}

/// ES `CircuitBreaker.Durability`: whether retrying can succeed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    Transient,
    Permanent,
}

impl Durability {
    pub fn as_str(self) -> &'static str {
        match self {
            Durability::Transient => "TRANSIENT",
            Durability::Permanent => "PERMANENT",
        }
    }
}

/// The child breakers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Request,
    Fielddata,
    InflightRequests,
    ModelInference,
    EqlSequence,
    Accounting,
}

impl Kind {
    /// Every child breaker (what the parent sums).
    pub const ALL: [Kind; 6] = [Kind::ModelInference, Kind::InflightRequests, Kind::Request, Kind::Fielddata, Kind::EqlSequence, Kind::Accounting];
    /// The ones ES 9 has, in the order it lists them (`_nodes/stats`,
    /// parent trip usages). `accounting` (ES 7's) is enforced internally
    /// but never shown.
    pub const VISIBLE: [Kind; 5] = [Kind::ModelInference, Kind::InflightRequests, Kind::Request, Kind::Fielddata, Kind::EqlSequence];
}

/// The `accounting` breaker's limit: `RUSTIC_ACCOUNTING_BREAKER_LIMIT` (a
/// size, a percentage of the budget, or `-1` for none), default 100% as in
/// ES 7. Not a cluster setting: ES 9 has no such breaker. Every charge on
/// it today is made without breaking (long-lived memory already
/// allocated), so it reports and feeds the parent but does not trip; the
/// limit applies to a tripping reservation, should one be added.
fn accounting_limit(budget: u64) -> i64 {
    std::env::var("RUSTIC_ACCOUNTING_BREAKER_LIMIT").ok().and_then(|v| parse_memory(&v, "RUSTIC_ACCOUNTING_BREAKER_LIMIT", budget).ok()).unwrap_or(budget as i64)
}

/// One child circuit breaker (ES `ChildMemoryCircuitBreaker`).
pub struct Breaker {
    pub name: &'static str,
    /// Bytes; `-1`: no limit.
    limit: AtomicI64,
    /// `f64` bits.
    overhead: AtomicU64,
    pub durability: Durability,
    used: AtomicU64,
    tripped: AtomicU64,
    /// Reservations held on this breaker (each in-flight request body is one).
    live: AtomicU64,
    /// Long-lived bytes its owners measure, added to the reserved ones.
    gauge: Option<fn() -> u64>,
}

impl Breaker {
    fn new(name: &'static str, limit: u64, overhead: f64, durability: Durability, gauge: Option<fn() -> u64>) -> Breaker {
        Breaker { name, limit: AtomicI64::new(limit as i64), overhead: AtomicU64::new(overhead.to_bits()), durability, used: AtomicU64::new(0), tripped: AtomicU64::new(0), live: AtomicU64::new(0), gauge }
    }

    /// The limit in bytes (`-1`: none).
    pub fn limit(&self) -> i64 {
        self.limit.load(Ordering::Relaxed)
    }

    pub fn set_limit(&self, bytes: i64) {
        self.limit.store(bytes, Ordering::Relaxed);
    }

    pub fn overhead(&self) -> f64 {
        f64::from_bits(self.overhead.load(Ordering::Relaxed))
    }

    pub fn set_overhead(&self, o: f64) {
        self.overhead.store(o.to_bits(), Ordering::Relaxed);
    }

    /// Bytes reserved on this breaker.
    pub fn reserved(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Bytes this breaker estimates in use: reserved plus measured.
    pub fn used(&self) -> u64 {
        self.reserved() + self.gauge.map_or(0, |g| g())
    }

    /// Times this breaker tripped.
    pub fn tripped(&self) -> u64 {
        self.tripped.load(Ordering::Relaxed)
    }

    /// Reservations currently held on this breaker.
    pub fn live(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }

    /// `used * overhead` (what the parent sums and prints).
    fn used_with_overhead(&self) -> u64 {
        (self.used() as f64 * self.overhead()) as u64
    }

    /// ES `ChildMemoryCircuitBreaker.limit`: add `bytes` unless the new
    /// total times the overhead passes the limit (CAS loop, no lock).
    fn add_and_maybe_break(&self, bytes: u64, label: &str) -> Result<(), EsError> {
        self.add_with_limit(bytes,label,self.limit())
    }
    fn add_with_limit(&self,bytes:u64,label:&str,limit:i64)->Result<(),EsError>{
        let overhead = self.overhead();
        let gauge = self.gauge.map_or(0, |g| g());
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            let new = cur.saturating_add(bytes);
            let wanted = (new.saturating_add(gauge) as f64 * overhead) as u64;
            if limit >= 0 && wanted > limit as u64 {
                self.tripped.fetch_add(1, Ordering::Relaxed);
                let reason = format!(
                    "[{}] Data too large, data for [{label}] would be [{}], which is larger than the limit of [{}]{}",
                    self.name,
                    bytes_safe(wanted),
                    bytes_safe(limit as u64),
                    docs_suffix()
                );
                return Err(breaking(reason, wanted, limit as u64, self.durability));
            }
            match self.used.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Ok(()),
                Err(now) => cur = now,
            }
        }
    }

    fn release(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let before = self.used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |u| Some(u.saturating_sub(bytes))).unwrap_or(0);
        debug_assert!(before >= bytes, "breaker [{}] released {bytes} bytes of {before} reserved", self.name);
    }
}

/// The `parent` breaker (ES `HierarchyCircuitBreakerService` itself).
pub struct Parent {
    limit: AtomicI64,
    /// Whether the limit was set explicitly (else it follows
    /// `use_real_memory`: 95% or 70% of the budget).
    limit_explicit: AtomicBool,
    use_real_memory: AtomicBool,
    tripped: AtomicU64,
    /// Cached real-memory sample: when (ns since `epoch`), the bytes, and
    /// the children's reservations at that time.
    sampled_at: AtomicU64,
    sampled: AtomicU64,
    reserved_at_sample: AtomicU64,
    /// Tests: a fixed real-memory figure instead of the OS's (`u64::MAX`: none).
    real_override: AtomicU64,
    /// Last over-limit attempt to give memory back (ns since `epoch`).
    trimmed_at: AtomicU64,
    /// [`TRIMS`] as of the last sample.
    trims_seen: AtomicU64,
    epoch: std::time::Instant,
}

/// Real memory is sampled at most this often; in between, the children's
/// new reservations are added to the last sample.
const SAMPLE_NANOS: u64 = 10_000_000;

/// The node's breakers (ES 9 defaults, fractions of [`budget`]).
pub struct Breakers {
    budget: u64,
    pub request: Breaker,
    pub fielddata: Breaker,
    pub inflight_requests: Breaker,
    pub model_inference: Breaker,
    pub eql_sequence: Breaker,
    pub accounting: Breaker,
    pub parent: Parent,
}

fn accounting_gauge() -> u64 {
    rustic_core::accounting::accounting_bytes() + IndexingBuffer::global().total()
}

fn fielddata_gauge() -> u64 {
    rustic_core::accounting::bytes(rustic_core::accounting::Category::Fielddata)
}

/// One breaker's `_nodes/stats` figures.
pub struct BreakerStats {
    pub name: &'static str,
    pub limit: i64,
    pub estimated: u64,
    pub overhead: f64,
    pub tripped: u64,
}

impl Breakers {
    fn new(budget: u64, gauges: bool) -> Breakers {
        let pct = |p: u64| (budget as f64 * p as f64 / 100.0) as u64;
        let (acc, fd): (Option<fn() -> u64>, Option<fn() -> u64>) = if gauges { (Some(accounting_gauge), Some(fielddata_gauge)) } else { (None, None) };
        let b = Breakers {
            budget,
            request: Breaker::new("request", pct(60), 1.0, Durability::Transient, None),
            fielddata: Breaker::new("fielddata", pct(40), 1.03, Durability::Permanent, fd),
            inflight_requests: Breaker::new("inflight_requests", budget, 2.0, Durability::Transient, None),
            model_inference: Breaker::new("model_inference", pct(50), 1.0, Durability::Transient, None),
            eql_sequence: Breaker::new("eql_sequence", pct(50), 1.0, Durability::Transient, None),
            accounting: Breaker::new("accounting", 0, 1.0, Durability::Permanent, acc),
            parent: Parent {
                limit: AtomicI64::new(pct(95) as i64),
                limit_explicit: AtomicBool::new(false),
                use_real_memory: AtomicBool::new(true),
                tripped: AtomicU64::new(0),
                sampled_at: AtomicU64::new(0),
                sampled: AtomicU64::new(0),
                reserved_at_sample: AtomicU64::new(0),
                real_override: AtomicU64::new(u64::MAX),
                trimmed_at: AtomicU64::new(0),
                trims_seen: AtomicU64::new(0),
                epoch: std::time::Instant::now(),
            },
        };
        b.accounting.set_limit(accounting_limit(budget));
        b
    }

    pub fn get(&self, k: Kind) -> &Breaker {
        match k {
            Kind::Request => &self.request,
            Kind::Fielddata => &self.fielddata,
            Kind::InflightRequests => &self.inflight_requests,
            Kind::ModelInference => &self.model_inference,
            Kind::EqlSequence => &self.eql_sequence,
            Kind::Accounting => &self.accounting,
        }
    }

    /// Bytes reserved on the child breakers (measured gauges excluded).
    pub fn accounted(&self) -> u64 {
        Kind::ALL.iter().map(|k| self.get(*k).reserved()).sum()
    }

    /// The parent's limit in bytes (`-1`: none).
    pub fn parent_limit(&self) -> i64 {
        self.parent.limit.load(Ordering::Relaxed)
    }

    pub fn parent_tripped(&self) -> u64 {
        self.parent.tripped.load(Ordering::Relaxed)
    }

    pub fn use_real_memory(&self) -> bool {
        self.parent.use_real_memory.load(Ordering::Relaxed)
    }

    /// Reserve `bytes` on the `request` breaker for `label` (ES
    /// `addEstimateBytesAndMaybeBreak`); released when the returned
    /// reservation drops.
    pub fn reserve_request(&'static self, bytes: u64, label: &str) -> Result<Reservation, EsError> {
        self.reserve(Kind::Request, bytes, label)
    }

    /// Reserve `bytes` on breaker `k` (then the parent), released on drop.
    pub fn reserve(&'static self, k: Kind, bytes: u64, label: &str) -> Result<Reservation, EsError> {
        if k==Kind::Request && native_controlled(self) && !is_replaying() {return self.reserve_native(bytes,label)}
        self.add_estimate_bytes_and_maybe_break(k, bytes, label)?;
        let provenance=match provenance::Ledger::new(k,bytes,label,true) {
            Ok(value)=>value,Err(error)=>{self.get(k).release(bytes);return Err(error);}
        };
        self.get(k).live.fetch_add(1, Ordering::Relaxed);
        Ok(Reservation { breakers: self, kind: k, bytes, scope: scope_for(k, bytes), provenance, native:false })
    }

    /// [`Breakers::reserve`] that a [`Replay`] does not force: for memory a
    /// recovery may be refused without losing anything acknowledged (a
    /// store file read into memory: the copy fails and stays on disk).
    pub fn reserve_refusable(&'static self, k: Kind, bytes: u64, label: &str) -> Result<Reservation, EsError> {
        let b = self.get(k);
        b.add_and_maybe_break(bytes, label)?;
        if let Err(e) = self.check_parent_after((bytes as f64 * b.overhead()) as u64, bytes, label) {
            b.release(bytes);
            return Err(e);
        }
        let provenance = match provenance::Ledger::new(k, bytes, label, true) {
            Ok(value) => value,
            Err(error) => {
                b.release(bytes);
                return Err(error);
            }
        };
        b.live.fetch_add(1, Ordering::Relaxed);
        Ok(Reservation { breakers: self, kind: k, bytes, scope: scope_for(k, bytes), provenance, native: false })
    }

    /// Native protocol/control storage has its own finite Accounting admission.
    /// It must remain usable when the RSS parent deliberately rejects query
    /// work (for example settings recovery and emitting the decided error).
    /// This is not unbounded: both the configured native child limit and the
    /// effective node memory budget cap its real retained allocation ownership.
    pub fn reserve_native(&'static self,bytes:u64,label:&str)->Result<Reservation,EsError>{
        self.add_native(bytes,label)?;
        let provenance=match provenance::Ledger::native(bytes,label){Ok(value)=>value,Err(error)=>{self.accounting.release(bytes);return Err(error)}};
        self.accounting.live.fetch_add(1,Ordering::Relaxed);
        Ok(Reservation{breakers:self,kind:Kind::Accounting,bytes,scope:scope_for(Kind::Request,bytes),provenance,native:true})
    }
    fn add_native(&self,bytes:u64,label:&str)->Result<(),EsError>{
        if let Some(replay)=current_replay() {self.force_add(&replay,Kind::Accounting,bytes,label);return Ok(())}
        let configured=self.accounting.limit();
        let hard=self.budget.min(i64::MAX as u64) as i64;
        self.accounting.add_with_limit(bytes,label,if configured<0{hard}else{configured.min(hard)})?;
        let total=Kind::ALL.iter().try_fold(0u64,|n,k|n.checked_add(self.get(*k).reserved())).unwrap_or(u64::MAX);
        if total>self.budget {
            self.accounting.release(bytes);
            // Native reservations are held on `accounting` (not an ES 9
            // breaker of its own): report that breaker, PERMANENT like it.
            self.accounting.tripped.fetch_add(1,Ordering::Relaxed);
            return Err(breaking(format!("[{}] Data too large, data for [{label}] would be [{}], which is larger than the limit of [{}]{}",self.accounting.name,bytes_safe(total),bytes_safe(self.budget),docs_suffix()),total,self.budget,Durability::Permanent));
        }
        Ok(())
    }

    /// Reserve `bytes` on breaker `k` without ever tripping (ES
    /// `addWithoutBreaking`: memory already allocated, or work that must
    /// not fail), released on drop.
    pub fn reserve_without_breaking(&'static self, k: Kind, bytes: u64) -> Reservation {
        self.get(k).used.fetch_add(bytes, Ordering::Relaxed);
        self.get(k).live.fetch_add(1, Ordering::Relaxed);
        Reservation { breakers: self, kind: k, bytes, scope: scope_for(k, bytes), provenance:provenance::Ledger::unchecked(k,bytes), native:false }
    }

    /// ES `addWithoutBreaking` for work of a [`Replay`]: `bytes` are
    /// reserved on breaker `k` whatever the limits (and so count toward the
    /// parent); one that takes the breaker or the parent past its limit is
    /// noted on the replay, which warns once. No trip is counted: nothing
    /// was refused.
    fn force_add(&self, replay: &Replay, k: Kind, bytes: u64, label: &str) {
        let b = self.get(k);
        b.used.fetch_add(bytes, Ordering::Relaxed);
        let limit = b.limit();
        let child_over = limit >= 0 && b.used_with_overhead() > limit as u64;
        let parent_limit = self.parent_limit();
        let parent_over = !child_over && parent_limit >= 0
            && self.parent_total((bytes as f64 * b.overhead()) as u64, bytes).0 > parent_limit as u64;
        if child_over || parent_over {
            replay.note_over(if child_over { b.name } else { "parent" }, label, bytes);
        }
    }

    /// ES `ChildMemoryCircuitBreaker.addEstimateBytesAndMaybeBreak`: the
    /// child's limit, then the parent's with `bytes * overhead` new bytes
    /// (a parent trip takes the child's bytes back). `bytes == 0` checks
    /// both without reserving (ES's `allocated_buckets` check).
    pub fn add_estimate_bytes_and_maybe_break(&self, k: Kind, bytes: u64, label: &str) -> Result<(), EsError> {
        if let Some(replay) = current_replay() {
            self.force_add(&replay, k, bytes, label);
            return Ok(());
        }
        let b = self.get(k);
        b.add_and_maybe_break(bytes, label)?;
        // The child already holds `bytes`: the parent counts them once, as
        // the new bytes (with the overhead), not again among the reservations.
        if let Err(e) = self.check_parent_after((bytes as f64 * b.overhead()) as u64, bytes, label) {
            b.release(bytes);
            return Err(e);
        }
        Ok(())
    }

    /// The real memory figure the parent checks: a sample of the process's
    /// anonymous memory at most [`SAMPLE_NANOS`] old (or taken again after
    /// a background trim), plus what the children reserved since, less
    /// `just_reserved` (bytes the caller's check counts as new).
    fn real_usage(&self, just_reserved: u64) -> u64 {
        let p = &self.parent;
        let fixed = p.real_override.load(Ordering::Relaxed);
        if fixed != u64::MAX {
            return fixed;
        }
        let now = p.epoch.elapsed().as_nanos() as u64;
        let at = p.sampled_at.load(Ordering::Relaxed);
        let trims = TRIMS.load(Ordering::Relaxed);
        let trimmed = p.trims_seen.load(Ordering::Relaxed) != trims;
        if (at == 0 || trimmed || now.saturating_sub(at) >= SAMPLE_NANOS) && p.sampled_at.compare_exchange(at, now.max(1), Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            p.trims_seen.store(trims, Ordering::Relaxed);
            self.sample_now();
        }
        let base = p.sampled.load(Ordering::Relaxed);
        let since = self.accounted().saturating_sub(p.reserved_at_sample.load(Ordering::Relaxed));
        base.saturating_add(since.saturating_sub(just_reserved))
    }

    fn sample_now(&self) {
        let p = &self.parent;
        let reserved = self.accounted();
        let real = sample_anon_rss().unwrap_or(reserved);
        p.reserved_at_sample.store(reserved, Ordering::Relaxed);
        p.sampled.store(real, Ordering::Relaxed);
    }

    /// ES `G1OverLimitStrategy`'s counterpart: when real memory is over
    /// the limit, at most every 5s, hand freed allocator memory back to the
    /// OS ([`set_release_hook`]: mimalloc's `mi_collect` on the cluster
    /// binary, else glibc's `malloc_trim`) on a background thread, never
    /// on the caller's (it may be a transport reader); the next check
    /// measures again. The check that triggered it still fails.
    fn trim_in_background(&self) {
        let p = &self.parent;
        if p.real_override.load(Ordering::Relaxed) != u64::MAX {
            return;
        }
        let now = p.epoch.elapsed().as_nanos() as u64;
        let at = p.trimmed_at.load(Ordering::Relaxed);
        if (at != 0 && now.saturating_sub(at) < 5_000_000_000) || p.trimmed_at.compare_exchange(at, now.max(1), Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        let _ = std::thread::Builder::new().name("breaker-trim".into()).spawn(|| {
            release_free_memory();
            // mimalloc purges what a collect freed after its purge delay
            // (10ms by default): sample again once that has passed.
            std::thread::sleep(std::time::Duration::from_millis(25));
            TRIMS.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// What the parent compares with its limit (ES `memoryUsed`): real
    /// memory plus the new bytes, else the sum of ES 9's children (the
    /// hidden `accounting` is not in it).
    fn parent_total(&self, new_bytes: u64, just_reserved: u64) -> (u64, u64) {
        if self.use_real_memory() {
            let real = self.real_usage(just_reserved);
            (real.saturating_add(new_bytes), real)
        } else {
            let sum = Kind::VISIBLE.iter().map(|k| self.get(*k).used_with_overhead()).fold(0, u64::saturating_add);
            (sum, sum)
        }
    }

    /// The parent's estimate (`_nodes/stats` `parent.estimated_size`).
    pub fn parent_estimated(&self) -> u64 {
        self.parent_total(0, 0).0
    }

    /// ES `HierarchyCircuitBreakerService.checkParentLimit`.
    pub fn check_parent(&self, new_bytes: u64, label: &str) -> Result<(), EsError> {
        self.check_parent_after(new_bytes, 0, label)
    }

    fn check_parent_after(&self, new_bytes: u64, just_reserved: u64, label: &str) -> Result<(), EsError> {
        let limit = self.parent_limit();
        if limit < 0 {
            return Ok(());
        }
        let (total, base) = self.parent_total(new_bytes, just_reserved);
        if total <= limit as u64 {
            if OVER_LIMIT.load(Ordering::Relaxed) && base <= limit as u64 {
                OVER_LIMIT.store(false, Ordering::Relaxed);
            }
            return Ok(());
        }
        if self.use_real_memory() {
            // Memory itself is over the limit (not one oversized request on
            // a node with room): idle workers start giving pages back, and
            // the ones already idle are woken to do it.
            if base > limit as u64 && self.parent.real_override.load(Ordering::Relaxed) == u64::MAX && !OVER_LIMIT.swap(true, Ordering::Relaxed) {
                wake_idle_workers();
            }
            self.trim_in_background();
        }
        self.parent.tripped.fetch_add(1, Ordering::Relaxed);
        let mut reason = format!("[parent] Data too large, data for [{label}] would be [{}], which is larger than the limit of [{}]", bytes_safe(total), bytes_safe(limit as u64));
        if self.use_real_memory() {
            reason.push_str(&format!(", real usage: [{}], new bytes reserved: [{}]", bytes_safe(base), bytes_safe(new_bytes)));
        }
        let usages = Kind::VISIBLE.iter().map(|k| { let b = self.get(*k); format!("{}={}", b.name, bytes_safe(b.used_with_overhead())) }).collect::<Vec<_>>().join(", ");
        reason.push_str(&format!(", usages [{usages}]"));
        reason.push_str(&docs_suffix());
        // ES: TRANSIENT unless the permanent children (fielddata) hold
        // more, over the children ES 9 has (the hidden one does not tip it).
        let (mut transient, mut permanent) = (0u64, 0u64);
        for k in Kind::VISIBLE {
            let b = self.get(k);
            match b.durability {
                Durability::Transient => transient = transient.saturating_add(b.used_with_overhead()),
                Durability::Permanent => permanent = permanent.saturating_add(b.used_with_overhead()),
            }
        }
        let durability = if transient >= permanent { Durability::Transient } else { Durability::Permanent };
        #[cfg(test)]
        if std::ptr::eq(self, process_breakers()) {
            // A trip of the process's breakers in a test: a real shortage or
            // an accounting leak? What the parent read, and what is held.
            eprintln!("process breakers tripped in a test: {}; reserved (live): {}", reason, self.held_summary());
        }
        Err(breaking(reason, total, limit as u64, durability))
    }

    /// Every child breaker's reserved bytes and live reservations
    /// (`request=64/1, ...`), for diagnostics.
    pub fn held_summary(&self) -> String {
        Kind::ALL.iter().map(|k| { let b = self.get(*k); format!("{}={}/{}", b.name, b.reserved(), b.live()) }).collect::<Vec<_>>().join(", ")
    }

    /// Every breaker ES 9 has, its figures in ES 9's order, `parent` last.
    pub fn stats(&self) -> Vec<BreakerStats> {
        let mut v: Vec<BreakerStats> = Kind::VISIBLE
            .iter()
            .map(|k| {
                let b = self.get(*k);
                BreakerStats { name: b.name, limit: b.limit(), estimated: b.used(), overhead: b.overhead(), tripped: b.tripped() }
            })
            .collect();
        v.push(BreakerStats { name: "parent", limit: self.parent_limit(), estimated: self.parent_estimated(), overhead: 1.0, tripped: self.parent_tripped() });
        v
    }

    /// Apply the breaker settings `get` resolves (transient, persistent,
    /// node settings; `None`: the default).
    pub fn apply_settings(&self, get: &dyn Fn(&str) -> Option<String>) {
        let budget = self.budget;
        let mem = |key: &str, default_pct: f64| -> i64 { get(key).and_then(|v| parse_memory(&v, key, budget).ok()).unwrap_or((budget as f64 * default_pct / 100.0) as i64) };
        let over = |key: &str, default: f64| -> f64 { get(key).and_then(|v| v.trim().parse::<f64>().ok()).filter(|o| *o >= 0.0).unwrap_or(default) };
        for (b, lkey, lpct, okey, odef) in [
            (&self.request, "indices.breaker.request.limit", 60.0, "indices.breaker.request.overhead", 1.0),
            (&self.fielddata, "indices.breaker.fielddata.limit", 40.0, "indices.breaker.fielddata.overhead", 1.03),
            (&self.inflight_requests, "network.breaker.inflight_requests.limit", 100.0, "network.breaker.inflight_requests.overhead", 2.0),
            (&self.model_inference, "breaker.model_inference.limit", 50.0, "breaker.model_inference.overhead", 1.0),
            (&self.eql_sequence, "breaker.eql_sequence.limit", 50.0, "breaker.eql_sequence.overhead", 1.0),
        ] {
            b.set_limit(mem(lkey, lpct));
            b.set_overhead(over(okey, odef));
        }
        let real = get("indices.breaker.total.use_real_memory").map(|v| v.trim() != "false").unwrap_or(true);
        self.parent.use_real_memory.store(real, Ordering::Relaxed);
        let explicit = get("indices.breaker.total.limit").and_then(|v| parse_memory(&v, "indices.breaker.total.limit", budget).ok());
        self.parent.limit_explicit.store(explicit.is_some(), Ordering::Relaxed);
        self.parent.limit.store(explicit.unwrap_or((budget as f64 * if real { 0.95 } else { 0.70 }) as i64), Ordering::Relaxed);
    }

    /// Tests: breakers of their own over `budget` (no measured gauges),
    /// for a [`RequestScope`] that keeps a test's limits apart from the
    /// process's.
    #[doc(hidden)]
    pub fn leak_for_tests(budget: u64) -> &'static Breakers {
        Box::leak(Box::new(Breakers::new(budget, false)))
    }

    /// Tests: [`Self::leak_for_tests`] whose parent sees no process memory
    /// (a fixed real-memory figure of 0, so only this test's own bytes): a
    /// unit test's breaker is not tripped by what the rest of the suite,
    /// running in the same process, holds (the parent checks real memory
    /// against 95% of the budget).
    #[cfg(test)]
    pub fn private_for_tests(budget: u64) -> &'static Breakers {
        let b = Self::leak_for_tests(budget);
        b.set_real_memory_for_tests(Some(0));
        b
    }

    /// Tests: the real memory figure the parent sees (`None`: the OS's).
    pub fn set_real_memory_for_tests(&self, bytes: Option<u64>) {
        self.parent.real_override.store(bytes.unwrap_or(u64::MAX), Ordering::Relaxed);
    }
}

/// Background trims completed ([`Breakers::trim_in_background`]): a new
/// one makes the next parent check sample real memory again.
static TRIMS: AtomicU64 = AtomicU64::new(0);

static RELEASE_HOOK: OnceLock<fn(bool)> = OnceLock::new();

/// Set while real memory is over the parent's limit (set by a parent
/// check that finds it so, cleared by one that finds it back under).
static OVER_LIMIT: AtomicBool = AtomicBool::new(false);

static WAKE_HOOKS: std::sync::Mutex<Vec<fn()>> = std::sync::Mutex::new(Vec::new());

/// Register how to wake a set of idle worker threads (an executor's
/// workers, a runtime's), so they run [`idle_collect`] when real memory
/// goes over the parent's limit: a thread already idle would otherwise
/// keep its freed pages until its next job.
pub fn add_wake_hook(hook: fn()) {
    // Callers register once (a `Once` each).
    if let Ok(mut hooks) = WAKE_HOOKS.lock() {
        hooks.push(hook);
    }
}

/// When the idle workers were last woken ([`wake_idle_workers`]).
struct Waking {
    last: Option<std::time::Instant>,
    /// A wake is waiting for the end of the window.
    deferred: bool,
}

impl Waking {
    const WINDOW: std::time::Duration = std::time::Duration::from_secs(1);

    /// A wake asked for at `now`: run it after the returned delay (zero:
    /// now, and the window starts), or not at all (`None`: one is already
    /// deferred to the end of this window).
    fn delay(&mut self, now: std::time::Instant) -> Option<std::time::Duration> {
        match self.last.map(|t| now.saturating_duration_since(t)) {
            Some(since) if since < Self::WINDOW => {
                if self.deferred {
                    return None;
                }
                self.deferred = true;
                Some(Self::WINDOW - since)
            }
            _ => {
                self.last = Some(now);
                Some(std::time::Duration::ZERO)
            }
        }
    }
}

/// Wake every registered set of idle workers, from a thread of its own
/// (the parent check that noticed must not wait), at most once a second:
/// a wake asked for sooner is deferred to the end of that second (one
/// pending wake covers any number asked meanwhile) and runs if memory is
/// still over then. Each worker collects at most once a second anyway.
fn wake_idle_workers() {
    static WAKING: std::sync::Mutex<Waking> = std::sync::Mutex::new(Waking { last: None, deferred: false });
    let delay = WAKING.lock().unwrap_or_else(|e| e.into_inner()).delay(std::time::Instant::now());
    let Some(delay) = delay else { return };
    let hooks: Vec<fn()> = WAKE_HOOKS.lock().map(|h| h.clone()).unwrap_or_default();
    if hooks.is_empty() {
        WAKING.lock().unwrap_or_else(|e| e.into_inner()).deferred = false;
        return;
    }
    let spawned = std::thread::Builder::new().name("breaker-wake".into()).spawn(move || {
        if !delay.is_zero() {
            std::thread::sleep(delay);
            let mut w = WAKING.lock().unwrap_or_else(|e| e.into_inner());
            w.deferred = false;
            // Back under meanwhile: nothing to collect.
            if !OVER_LIMIT.load(Ordering::Relaxed) {
                return;
            }
            w.last = Some(std::time::Instant::now());
        }
        for hook in hooks {
            hook();
        }
    });
    if spawned.is_err() && !delay.is_zero() {
        WAKING.lock().unwrap_or_else(|e| e.into_inner()).deferred = false;
    }
}

/// How the process gives freed allocator memory back to the OS, for the
/// binary's global allocator: `hook(true)` collects everything it can
/// (the background trim), `hook(false)` the calling thread's freed pages
/// only, cheaply (idle workers, [`idle_collect`]). The cluster binary
/// installs mimalloc's `mi_collect`; the default is glibc's `malloc_trim`
/// for the background trim (process-wide: never from idle workers).
pub fn set_release_hook(hook: fn(bool)) {
    let _ = RELEASE_HOOK.set(hook);
}

/// For a worker thread going idle (tokio's `on_thread_park`, an engine
/// pool worker between jobs): while the parent breaker is over its limit,
/// give this thread's freed allocator pages back (at most once a second
/// per thread). mimalloc keeps a thread's freed pages until that thread
/// allocates again; the background trim cannot reach them. One relaxed
/// load otherwise: the allocator's behaviour is unchanged on the happy path.
pub fn idle_collect() {
    if !OVER_LIMIT.load(Ordering::Relaxed) {
        return;
    }
    let Some(hook) = RELEASE_HOOK.get() else { return };
    thread_local! {
        static LAST: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
    }
    let due = LAST.with(|l| {
        let due = l.get().is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(1));
        if due {
            l.set(Some(std::time::Instant::now()));
        }
        due
    });
    if due {
        hook(false);
        // On a quiet node no request checks the parent: see here whether
        // memory is back under the limit (the regular 10ms sampling has
        // passed the purge delay by the next check), so idle workers stop.
        if back_under(process_breakers()) {
            OVER_LIMIT.store(false, Ordering::Relaxed);
        }
    }
}

/// Whether `b`'s parent no longer has real memory over its limit (idle
/// workers then stop collecting).
fn back_under(b: &Breakers) -> bool {
    let limit = b.parent_limit();
    limit < 0 || !b.use_real_memory() || b.parent_total(0, 0).1 <= limit as u64
}

/// Whether idle workers are collecting (real memory over the parent's
/// limit); tests.
#[doc(hidden)]
pub fn collecting() -> bool {
    OVER_LIMIT.load(Ordering::Relaxed)
}

/// Give freed allocator memory back to the OS (the background trim).
fn release_free_memory() {
    if let Some(hook) = RELEASE_HOOK.get() {
        hook(true);
        return;
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: `malloc_trim` only releases free heap pages.
        unsafe {
            malloc_trim(0);
        }
    }
}

/// The breaker settings ES parses as memory sizes.
pub const LIMIT_SETTINGS: &[&str] = &[
    "indices.breaker.total.limit",
    "indices.breaker.request.limit",
    "indices.breaker.fielddata.limit",
    "network.breaker.inflight_requests.limit",
    "breaker.model_inference.limit",
    "breaker.eql_sequence.limit",
];

/// The breaker overhead settings.
pub const OVERHEAD_SETTINGS: &[&str] = &[
    "indices.breaker.request.overhead",
    "indices.breaker.fielddata.overhead",
    "network.breaker.inflight_requests.overhead",
    "breaker.model_inference.overhead",
    "breaker.eql_sequence.overhead",
];

/// ES `MemorySizeValue.parseBytesSizeValueOrHeapRatio`: a percentage of
/// the budget or a byte size (`-1`: no limit), with ES's messages.
pub fn parse_memory(value: &str, key: &str, budget: u64) -> Result<i64, EsError> {
    let v = value.trim();
    if let Some(p) = v.strip_suffix('%') {
        let pct: f64 = p.trim().parse().map_err(|_| EsError::new(400, "parse_exception", format!("failed to parse [{p}] as a double")))?;
        if !(0.0..=100.0).contains(&pct) {
            return Err(EsError::new(400, "parse_exception", format!("percentage should be in [0-100], got [{}]", p.trim())));
        }
        return Ok((budget as f64 * pct / 100.0) as i64);
    }
    if v == "-1" {
        return Ok(-1);
    }
    let lower = v.to_ascii_lowercase();
    let units: [(&str, f64); 7] = [("pb", (1u64 << 50) as f64), ("tb", (1u64 << 40) as f64), ("gb", (1u64 << 30) as f64), ("mb", (1u64 << 20) as f64), ("kb", 1024.0), ("b", 1.0), ("", 0.0)];
    for (suffix, mult) in units {
        let Some(num) = lower.strip_suffix(suffix) else { continue };
        if suffix.is_empty() {
            // ES accepts a bare `0`, nothing else without a unit.
            if num.trim() == "0" {
                return Ok(0);
            }
            break;
        }
        // `kb` before `b`: "1kb" must not be read as "1k" + "b".
        if suffix == "b" && num.ends_with(|c: char| c.is_ascii_alphabetic()) {
            continue;
        }
        if let Ok(n) = num.trim().parse::<f64>() {
            if n >= 0.0 {
                return Ok((n * mult) as i64);
            }
        }
        return Err(EsError::new(400, "parse_exception", format!("failed to parse setting [{key}] with value [{value}] as a size in bytes")));
    }
    Err(EsError::new(400, "parse_exception", format!("failed to parse setting [{key}] with value [{value}] as a size in bytes: unit is missing or unrecognized")))
}

/// Validate a breaker setting's value (cluster settings update).
pub fn validate_setting(key: &str, value: &str) -> Result<(), EsError> {
    if LIMIT_SETTINGS.contains(&key) {
        parse_memory(value, key, budget()).map_err(|e| EsError::illegal_argument(e.reason.clone()).caused_by(e))?;
    } else if OVERHEAD_SETTINGS.contains(&key) {
        match value.trim().parse::<f64>() {
            Ok(o) if o >= 0.0 => {}
            Ok(_) => return Err(EsError::illegal_argument(format!("Failed to parse value [{value}] for setting [{key}] must be >= 0.0"))),
            Err(_) => {
                return Err(EsError::illegal_argument(format!("Failed to parse value [{value}] for setting [{key}]"))
                    .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{value}\""))))
            }
        }
    }
    Ok(())
}

/// Apply `node`'s breaker settings (transient over persistent over node
/// settings) to its breakers: at startup and after every cluster settings
/// change.
pub fn configure(node: &crate::node::Node) {
    rustic_core::query::query::set_max_clause_count(crate::query::limits::clause_limit());
    let cs = node.cluster_settings.read().clone();
    let ns = node.node_settings.read().clone();
    let text = |v: &serde_json::Value| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
    node.breakers.apply_settings(&|k| {
        // ES: a node setting only (`NodeScope`), never a cluster one.
        if k == "indices.breaker.total.use_real_memory" {
            return ns.get(k).map(text);
        }
        cs.1.get(k).or(cs.0.get(k)).or(ns.get(k)).map(text)
    });
}

/// At a node's open: apply its persisted breaker settings, for the first
/// node of the process only. Later ones (a cluster's shard engines, whose
/// persisted settings may be stale) get the cluster's settings when the
/// cluster state is applied to them.
pub fn configure_at_open(node: &crate::node::Node) {
    rustic_core::query::query::set_max_clause_count(crate::query::limits::clause_limit());
    static FIRST: AtomicBool = AtomicBool::new(true);
    if FIRST.swap(false, Ordering::Relaxed) && has_settings(node) {
        configure(node);
    }
}

/// Validate the breaker node settings (`elasticsearch.yml` / `-E`): an
/// invalid value fails the start, as ES fails it.
pub fn validate_node_settings(settings: &serde_json::Map<String, serde_json::Value>) -> Result<(), EsError> {
    for (k, v) in settings {
        if LIMIT_SETTINGS.contains(&k.as_str()) || OVERHEAD_SETTINGS.contains(&k.as_str()) {
            validate_setting(k, &v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))?;
        } else if k == "indices.breaker.total.use_real_memory" && !matches!(v.as_str().map(str::trim).unwrap_or(&v.to_string()), "true" | "false") {
            return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed.")));
        }
    }
    Ok(())
}

/// Whether `node` has any breaker setting of its own.
pub fn has_settings(node: &crate::node::Node) -> bool {
    let is_breaker = |k: &String| LIMIT_SETTINGS.contains(&k.as_str()) || OVERHEAD_SETTINGS.contains(&k.as_str()) || k == "indices.breaker.total.use_real_memory";
    let cs = node.cluster_settings.read();
    cs.0.keys().chain(cs.1.keys()).any(is_breaker) || node.node_settings.read().keys().any(is_breaker)
}

/// ES's `CircuitBreakingException` (HTTP 429).
pub fn breaking(reason: String, wanted: u64, limit: u64, durability: Durability) -> EsError {
    EsError::new(429, "circuit_breaking_exception", reason).with("bytes_wanted", wanted).with("bytes_limit", limit).with("durability", durability.as_str())
}

/// A `circuit_breaking_exception` for a refusal of the core's memory
/// admission (`rustic_core::RusticError::Memory`: "[core] [label] needs N
/// bytes, retained U, limit L"), with ES's fields read from the message:
/// `bytes_wanted` N, `bytes_limit` L and TRANSIENT when the refusal was
/// against the admission limit (memory frees as work completes: clients back
/// off and retry, as from ES's breakers and write-pool rejections); 0 / 0
/// PERMANENT for a refusal that names no limit (a size overflow).
pub fn core_refusal(reason: String) -> EsError {
    let number_after = |key: &str| -> Option<u64> {
        let at = reason.rfind(key)? + key.len();
        let digits: String = reason[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    };
    let wanted = number_after("needs ").unwrap_or(0);
    match number_after("limit ") {
        Some(limit) => breaking(reason, wanted, limit, Durability::Transient),
        None => breaking(reason, wanted, 0, Durability::Permanent),
    }
}

/// A `circuit_breaking_exception` for an allocation refused without a
/// breaker of its own (the allocator, a size overflow): `wanted` bytes when
/// known, no limit, PERMANENT like the node's accounting.
pub fn allocation_refused(reason: impl Into<String>, wanted: u64) -> EsError {
    breaking(reason.into(), wanted, 0, Durability::Permanent)
}

/// Whether `e` is (or was caused by) a circuit breaker trip.
pub fn is_breaking(e: &EsError) -> bool {
    e.kind == "circuit_breaking_exception" || e.caused_by.as_deref().is_some_and(is_breaking)
}

/// Bytes reserved on a breaker; released on drop.
#[path="memory_provenance.rs"]
mod provenance;

pub struct Reservation {
    breakers: &'static Breakers,
    kind: Kind,
    bytes: u64,
    /// The request this reservation was made for (debug accounting).
    scope: Option<Weak<RequestScope>>,
    provenance: provenance::Ledger,
    native: bool,
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {f.debug_struct("Reservation").field("kind",&self.kind).field("bytes",&self.bytes).finish_non_exhaustive()}
}

impl Reservation {
    /// Transfer already-admitted protocol metadata to ordinary request policy
    /// before decoding its body/plan. Both charges overlap during the check.
    pub fn into_query(self,label:&str)->Result<Self,EsError>{
        policy(false,||self.breakers.reserve_request(self.bytes,label))
    }
    /// Grow the reservation by `more` bytes (checking the breakers).
    pub fn grow(&mut self, more: u64, label: &str) -> Result<(), EsError> {
        if self.native {self.breakers.add_native(more,label)?} else {self.breakers.add_estimate_bytes_and_maybe_break(self.kind, more, label)?;}
        if let Err(error)=self.provenance.grow(more,label){self.breakers.get(self.kind).release(more);return Err(error);}
        self.bytes += more;
        scope_add(&self.scope, more as i64);
        Ok(())
    }

    /// Grow the reservation by `more` bytes without tripping.
    pub fn grow_without_breaking(&mut self, more: u64) {
        self.provenance.unchecked_growth(more);
        self.breakers.get(self.kind).used.fetch_add(more, Ordering::Relaxed);
        self.bytes += more;
        scope_add(&self.scope, more as i64);
    }

    /// Give `less` bytes back before the reservation drops.
    pub fn shrink(&mut self, less: u64) {
        let less = less.min(self.bytes);
        self.breakers.get(self.kind).release(less);
        self.bytes -= less;
        self.provenance.shrink(less);
        scope_add(&self.scope, -(less as i64));
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Transfer admission into an explicitly owner-carrying result whose bytes
    /// may outlive executor/request scope (for example an HTTP body stream).
    /// The breaker charge remains held; only scope-local leak tracking ends.
    pub fn into_result_owner(mut self) -> Self {
        scope_add(&self.scope, -(self.bytes as i64));
        self.scope = None;
        self
    }

    /// Transfer a result reservation to its originating request. The JSON
    /// tree has escaped the executor but remains live through response work.
    /// Aggregate guards in-place so retention itself needs no growing registry.
    pub fn retain_for_request(self) {
        let slot=usize::from(self.native);
        self.retain_in(slot)
    }

    /// [`Self::retain_for_request`] for a partial result (a shard's
    /// aggregation state, an ES|QL shard part) that a reduce later consumes.
    /// The reduce releases it ([`consume_retained_partials`],
    /// [`consume_partials_since`]), as ES's `QueryPhaseResultConsumer` frees
    /// reduced partials, so a request's charge does not keep every partial
    /// it has already reduced.
    pub fn retain_partial_for_request(self) {
        if self.native { return self.retain_for_request(); }
        self.retain_in(PARTIALS)
    }

    fn retain_in(mut self, slot: usize) {
        let Some(scope)=self.scope.as_ref().and_then(Weak::upgrade) else { return };
        scope_add(&self.scope,-(self.bytes as i64));
        self.scope=None;
        if slot==PARTIALS { credit_partials(&scope,self.bytes); }
        let mut retained=scope.retained.lock().unwrap_or_else(|e|e.into_inner());
        let retained=&mut retained[slot];
        if let Some(existing)=retained.as_mut() {
            debug_assert!(std::ptr::eq(existing.breakers,self.breakers));
            debug_assert!(existing.kind==self.kind);
            existing.bytes=existing.bytes.checked_add(self.bytes).expect("reserved bytes overflow");
            existing.provenance.append(&mut self.provenance);
            self.bytes=0;
        } else { *retained=Some(self); }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.breakers.get(self.kind).live.fetch_sub(1, Ordering::Relaxed);
        self.breakers.get(self.kind).release(self.bytes);
        scope_add(&self.scope, -(self.bytes as i64));
    }
}

/// The breakers of the work running on this thread: its request scope's
/// ([`scoped`]; the node's), else the process's.
pub fn breakers() -> &'static Breakers {
    SCOPE.with(|s| s.borrow().as_ref().map(|s| s.breakers)).unwrap_or_else(process_breakers)
}

/// `node.store.allow_mmap: false`: a store file read into memory is held
/// on the `accounting` breaker (ES 7's breaker for memory held by Lucene
/// segments past any request, enforced here though ES 9 does not show it)
/// for as long as the buffer lives, so the parent counts it too: with
/// `indices.breaker.total.use_real_memory` (default) the process's real
/// memory plus the file must stay under `indices.breaker.total.limit`. A
/// refusal fails the open (or the merge) instead of the process.
pub fn admit_store_file(bytes: u64, file: &str) -> Result<Box<dyn Send>, rustic_core::codec::directory::StoreFileRefusal> {
    admit_store_file_on(process_breakers(), bytes, file)
}

fn admit_store_file_on(breakers: &'static Breakers, bytes: u64, file: &str) -> Result<Box<dyn Send>, rustic_core::codec::directory::StoreFileRefusal> {
    // Refusable even in a recovery's replay: refusing the file loses
    // nothing (the copy fails and is kept on disk), forcing it would put
    // the whole node over its limit.
    breakers
        .reserve_refusable(Kind::Accounting, bytes, &format!("store file [{file}]"))
        .map(|r| Box::new(r) as Box<dyn Send>)
        .map_err(|e| rustic_core::codec::directory::StoreFileRefusal {
            limit: e.extra.iter().find(|(k, _)| k == "bytes_limit").and_then(|(_, v)| v.as_u64()).unwrap_or(0),
            reason: e.reason,
        })
}

/// Charge store files read into memory to the node's breakers (see
/// [`admit_store_file`]); called at start, before any index opens.
pub fn install_store_file_accounting() {
    rustic_core::codec::directory::set_store_file_admission(admit_store_file);
}

/// The process's breakers: every node of the process shares them (the
/// memory they measure is the process's), cluster shard engines included.
pub fn process_breakers() -> &'static Breakers {
    static B: OnceLock<Breakers> = OnceLock::new();
    B.get_or_init(|| {
        let limit=budget();
        rustic_core::admission::configure_shared(limit/2);
        Breakers::new(limit,true)
    })
}

// ---------------------------------------------------------------- requests

/// The request-breaker accounting of one request's growing structures
/// (aggregation buckets, ES|QL pages, merged hits): bytes are added as the
/// structures grow and charged to the breaker a page (16kb) at a time, as
/// ES `BigArrays` does, so the shared counter is touched rarely; all of it
/// is released when this drops. Single-threaded (`&self` methods): one per
/// request (or per thread of one).
pub struct RequestMemory {
    reservation: std::cell::RefCell<Option<Reservation>>,
    /// Bytes added since the last charge.
    pending: std::cell::Cell<u64>,
    /// Buckets created (ES `MultiBucketConsumer` call count).
    buckets: std::cell::Cell<u64>,
    breakers: &'static Breakers,
}

/// Bytes charged to the breaker at a time ([`RequestMemory::add`]).
pub const PAGE: u64 = 16 << 10;

impl Default for RequestMemory {
    fn default() -> Self {
        RequestMemory::new()
    }
}

impl RequestMemory {
    pub fn new() -> RequestMemory {
        RequestMemory::on(breakers())
    }

    pub fn on(breakers: &'static Breakers) -> RequestMemory {
        RequestMemory { reservation: Default::default(), pending: Default::default(), buckets: Default::default(), breakers }
    }

    /// The structures grew by `bytes` (charged once a page has built up).
    #[inline]
    pub fn add(&self, bytes: u64, label: &str) -> Result<(), EsError> {
        let p = self.pending.get() + bytes;
        if p < PAGE {
            self.pending.set(p);
            return Ok(());
        }
        self.pending.set(0);
        self.charge(p, label)
    }

    /// [`RequestMemory::add`] with a label built only when it charges.
    #[inline]
    pub fn add_with(&self, bytes: u64, label: impl FnOnce() -> String) -> Result<(), EsError> {
        let p = self.pending.get() + bytes;
        if p < PAGE {
            self.pending.set(p);
            return Ok(());
        }
        self.pending.set(0);
        self.charge(p, &label())
    }

    /// Charge `bytes` now (a large single allocation about to be made).
    pub fn charge(&self, bytes: u64, label: &str) -> Result<(), EsError> {
        let mut r = self.reservation.borrow_mut();
        match r.as_mut() {
            Some(res) => res.grow(bytes, label),
            None => {
                *r = Some(self.breakers.reserve(Kind::Request, bytes, label)?);
                Ok(())
            }
        }
    }

    /// The structures shrank by `bytes` (freed before the request ends).
    pub fn release(&self, bytes: u64) {
        let p = self.pending.get();
        if bytes <= p {
            self.pending.set(p - bytes);
            return;
        }
        self.pending.set(0);
        if let Some(r) = self.reservation.borrow_mut().as_mut() {
            r.shrink(bytes - p);
        }
    }

    /// `n` buckets were created: every 1024 the breakers are checked
    /// without reserving (ES `MultiBucketConsumer`'s `allocated_buckets`),
    /// so a runaway aggregation trips on real memory.
    #[inline]
    pub fn buckets(&self, n: u64) -> Result<(), EsError> {
        let before = self.buckets.get();
        let after = before + n;
        self.buckets.set(after);
        if before >> 10 != after >> 10 {
            self.breakers.add_estimate_bytes_and_maybe_break(Kind::Request, 0, "allocated_buckets")?;
        }
        Ok(())
    }

    /// Bytes added so far, charged or pending (a mark for [`Self::release_to`]).
    pub fn level(&self) -> u64 {
        self.charged() + self.pending.get()
    }

    /// Give back everything added since [`Self::level`] returned `mark`:
    /// the structures built since were freed.
    pub fn release_to(&self, mark: u64) {
        let now = self.level();
        if now > mark {
            self.release(now - mark);
        }
    }

    /// Bytes charged to the breaker so far.
    pub fn charged(&self) -> u64 {
        self.reservation.borrow().as_ref().map_or(0, Reservation::bytes)
    }
}

/// Heap bytes of a JSON value (rough: what a parsed response holds).
pub fn json_bytes(v: &serde_json::Value) -> u64 {
    use serde_json::Value;
    match v {
        Value::Null | Value::Bool(_) | Value::Number(_) => 32,
        Value::String(s) => 32 + s.capacity() as u64,
        Value::Array(a) => 32 + a.iter().map(json_bytes).sum::<u64>(),
        Value::Object(o) => 48 + o.iter().map(|(k, v)| 48 + k.capacity() as u64 + json_bytes(v)).sum::<u64>(),
    }
}

/// [`json_bytes`] of the tree `text` would decode to, measured by a
/// streaming pass that builds nothing (unescaped strings are borrowed), so
/// a decode can be admitted at its own size before it allocates.
pub fn json_text_bytes(text: &str) -> Result<u64, EsError> {
    use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
    struct Tree;
    struct Key;
    impl<'de> DeserializeSeed<'de> for Tree {
        type Value = u64;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<u64, D::Error> { d.deserialize_any(Tree) }
    }
    impl<'de> DeserializeSeed<'de> for Key {
        type Value = u64;
        fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<u64, D::Error> { d.deserialize_str(Key) }
    }
    impl<'de> Visitor<'de> for Key {
        type Value = u64;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a JSON object key") }
        fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<u64, E> { Ok(s.len() as u64) }
    }
    impl<'de> Visitor<'de> for Tree {
        type Value = u64;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a JSON value") }
        fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<u64, E> { Ok(32) }
        fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<u64, E> { Ok(32) }
        fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<u64, E> { Ok(32) }
        fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<u64, E> { Ok(32) }
        fn visit_unit<E: serde::de::Error>(self) -> Result<u64, E> { Ok(32) }
        fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<u64, E> { Ok(32 + s.len() as u64) }
        fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<u64, A::Error> {
            let mut n = 32u64;
            while let Some(x) = a.next_element_seed(Tree)? { n = n.saturating_add(x); }
            Ok(n)
        }
        fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<u64, A::Error> {
            let mut n = 48u64;
            while let Some(k) = a.next_key_seed(Key)? { n = n.saturating_add(48 + k).saturating_add(a.next_value_seed(Tree)?); }
            Ok(n)
        }
    }
    let mut d = serde_json::Deserializer::from_str(text);
    let n = Tree.deserialize(&mut d).map_err(|e| EsError::internal(format!("stored JSON: {e}")))?;
    d.end().map_err(|e| EsError::internal(format!("stored JSON: {e}")))?;
    Ok(n)
}

thread_local! {static NATIVE_CONTROL:std::cell::Cell<Option<bool>>=const{std::cell::Cell::new(None)};}
fn native_controlled(breakers:&Breakers)->bool {
    NATIVE_CONTROL.with(|v|v.get()).unwrap_or_else(||current_scope().is_some_and(|s|s.native_control&&std::ptr::eq(s.breakers,breakers)))
}
/// Run only already-classified non-tripping control work or error rendering
/// under finite native admission. Never use this for ordinary query execution.
pub fn native_control<T>(f:impl FnOnce()->T)->T{
    policy(true,f)
}
fn policy<T>(native:bool,f:impl FnOnce()->T)->T{
    struct Restore(Option<bool>);impl Drop for Restore{fn drop(&mut self){NATIVE_CONTROL.with(|v|v.set(self.0));}}
    let _restore=Restore(NATIVE_CONTROL.with(|v|v.replace(Some(native))));f()
}
/// Capture only the current admission policy for a synchronous worker closure.
/// The caller remains responsible for the existing RequestScope/attribution
/// propagation. Ordinary nested dispatch still explicitly overrides this policy.
pub fn inherit_request_policy<R,F:FnOnce()->R>(work:F)->impl FnOnce()->R {
    let native=native_controlled(breakers());
    // Work a replay hands to a worker is part of it (never refused).
    let replay=current_replay();
    move || match replay {Some(replay)=>replay.run(||policy(native,work)),None=>policy(native,work)}
}
/// A request's admission context on this thread (its breaker scope, native
/// policy and replay), captured to run part of the request on a worker
/// thread with the same accounting (per-shard aggregations on the search
/// pool).
#[derive(Clone)]
pub struct RequestContext { scope: Option<Arc<RequestScope>>, native: bool, replay: Option<Replay> }

impl RequestContext {
    /// This thread's context.
    pub fn capture() -> Self {
        Self { scope: current_scope(), native: native_controlled(breakers()), replay: current_replay() }
    }
    /// Run `f` in the captured context (on any thread).
    pub fn run<R>(&self, f: impl FnOnce() -> R) -> R {
        scoped(self.scope.clone(), || match &self.replay {
            Some(replay) => replay.clone().run(|| policy(self.native, f)),
            None => policy(self.native, f),
        })
    }
}

/// The admission policy in effect on this thread (see [`request_policy_future`]).
pub fn current_native_policy()->bool { native_controlled(breakers()) }

/// `future` polled under `scope` (the request's breaker attribution) and
/// the `native` admission policy at every poll: a request's work handed to
/// a task of its own (a spawned shard call) keeps the request's context.
pub async fn request_context_future<F:std::future::Future>(scope:Option<Arc<RequestScope>>,native:bool,future:F)->F::Output {
    let mut future=std::pin::pin!(future);
    std::future::poll_fn(|cx|scoped(scope.clone(),||policy(native,||future.as_mut().poll(cx)))).await
}

/// Re-enter the policy at every poll; no thread-local state spans an await.
pub async fn request_policy_future<F:std::future::Future>(native:bool,future:F)->F::Output {
    let mut future=std::pin::pin!(future);
    std::future::poll_fn(|cx|policy(native,||future.as_mut().poll(cx))).await
}

// ------------------------------------------------------------------ replay

thread_local! {
    static REPLAY: std::cell::RefCell<Option<Replay>> = const { std::cell::RefCell::new(None) };
}

/// One replay of already-acknowledged operations: local translog recovery,
/// the replica WAL, operations a peer recovery ships. ES applies them with
/// no circuit breaker or indexing-pressure check
/// (`InternalEngine.recoverFromTranslog`, origin `LOCAL_TRANSLOG_RECOVERY`;
/// `PEER_RECOVERY`): refusing them frees nothing (the ops must be applied
/// before the copy can serve) and nothing else can recover them, so a
/// replay that waited for memory could wait forever. Work run by
/// [`Replay::run`] still reserves what it uses, so breaker stats stay
/// right, but every reservation on that thread is forced (ES
/// `addWithoutBreaking`), core admission too; the replay keeps its memory
/// bounded by its own batching. The first reservation that goes past a
/// limit logs one warning per replay.
///
/// Two kinds. A *recovery* replay ([`Replay::new`]: WAL / translog replay
/// at open, peer recovery, promotion) is ES's recovery origin: besides the
/// forced reservations, its writes force the copy's background seals for a
/// while (`Shard::replay_seals_forced`) and are never throttled. A *live*
/// replay ([`Replay::live`]) is the completion of a live replicated batch
/// after its WAL append: its memory was admitted before the append (the
/// node's replica indexing pressure, the batch's staged admissions), so
/// what it charges from there is forced as well, as nothing may be refused
/// between the append and marking the ops processed; but it is ordinary
/// replication (ES origin `REPLICA`): it forces no seals and stays under
/// the copy's write throttle and stall.
#[derive(Clone)]
pub struct Replay(Arc<ReplayState>);

struct ReplayState {
    what: String,
    over: AtomicU64,
    /// A live replicated batch's completion, not a recovery.
    live: bool,
}

impl Replay {
    /// A recovery replay described as `what` in its warning.
    pub fn new(what: impl Into<String>) -> Replay {
        Replay(Arc::new(ReplayState { what: what.into(), over: AtomicU64::new(0), live: false }))
    }

    /// The completion of a live replicated batch past its WAL append,
    /// described as `what` in its warning (see the kinds above).
    pub fn live(what: impl Into<String>) -> Replay {
        Replay(Arc::new(ReplayState { what: what.into(), over: AtomicU64::new(0), live: true }))
    }

    /// Whether this is a live replicated batch's completion.
    pub fn is_live(&self) -> bool {
        self.0.live
    }

    /// Run `f` as part of this replay (on this thread; a worker it hands
    /// work to runs that under [`Replay::run`] again).
    pub fn run<R>(&self, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<Replay>);
        impl Drop for Restore {
            fn drop(&mut self) {
                let previous = self.0.take();
                REPLAY.with(|r| *r.borrow_mut() = previous);
            }
        }
        let _restore = Restore(REPLAY.with(|r| r.replace(Some(self.clone()))));
        rustic_core::admission::forced(f)
    }

    /// Reservations of this replay that went past a breaker's limit.
    pub fn over_limit(&self) -> u64 {
        self.0.over.load(Ordering::Relaxed)
    }

    fn note_over(&self, breaker: &str, label: &str, bytes: u64) {
        if self.0.over.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::warn!(replay = %self.0.what, breaker, label, bytes,
                "replaying acknowledged operations past the [{breaker}] circuit breaker limit: the memory is accounted, not refused");
        }
    }
}

fn current_replay() -> Option<Replay> {
    REPLAY.with(|r| r.borrow().clone())
}

/// Whether this thread runs work of a [`Replay`] of either kind (its
/// reservations are forced).
pub fn is_replaying() -> bool {
    REPLAY.with(|r| r.borrow().is_some())
}

/// Whether this thread runs work of a recovery [`Replay`] (WAL / translog
/// replay at open, peer recovery, promotion), not a live replicated batch.
pub fn is_recovery_replay() -> bool {
    REPLAY.with(|r| r.borrow().as_ref().is_some_and(|replay| !replay.is_live()))
}

/// Run `f` as a recovery [`Replay`] described as `what`, or as part of the
/// replay already running on this thread (one warning per outermost
/// replay).
pub fn replaying<R>(what: &str, f: impl FnOnce() -> R) -> R {
    match current_replay() {
        Some(_) => f(),
        None => Replay::new(what).run(f),
    }
}

/// Run `f` as a live [`Replay`] (a live replicated batch past its WAL
/// append) described as `what`, or as part of the replay already running
/// on this thread: a batch a peer recovery applies stays a recovery.
pub fn replaying_live<R>(what: &str, f: impl FnOnce() -> R) -> R {
    match current_replay() {
        Some(_) => f(),
        None => Replay::live(what).run(f),
    }
}

/// Slot of [`RequestScope::retained`] holding retained partial results.
const PARTIALS: usize = 2;

/// One reduce's own count of the partials retained while it runs
/// ([`partials_mark`]): (its request, bytes retained under it).
type ReduceMark = (std::sync::Weak<RequestScope>, std::rc::Rc<std::cell::Cell<u64>>);

thread_local! {
    /// The reduces running on this thread, innermost last. A partial is
    /// credited to the innermost one of its own request only, so a reduce
    /// never releases partials another reduce (another thread, or a nested
    /// reduce that already consumed its own) produced.
    static REDUCE_MARKS: std::cell::RefCell<Vec<ReduceMark>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Credit `bytes` of partials retained for `scope` on this thread to the
/// innermost reduce of that request running here, if any.
fn credit_partials(scope: &Arc<RequestScope>, bytes: u64) {
    REDUCE_MARKS.with(|marks| {
        if let Some((_, count)) = marks.borrow().iter().rev().find(|(s, _)| std::ptr::eq(s.as_ptr(), Arc::as_ptr(scope))) {
            count.set(count.get().saturating_add(bytes));
        }
    });
}

/// The partials one reduce produces ([`partials_mark`]). Dropping it
/// without [`consume_partials_since`] (an error in the reduce) keeps them.
pub struct PartialsMark(Option<std::rc::Rc<std::cell::Cell<u64>>>);

impl Drop for PartialsMark {
    fn drop(&mut self) {
        let Some(count) = self.0.take() else { return };
        REDUCE_MARKS.with(|marks| {
            let mut marks = marks.borrow_mut();
            if let Some(i) = marks.iter().rposition(|(_, c)| std::rc::Rc::ptr_eq(c, &count)) { marks.remove(i); }
        });
    }
}

/// A mark before the partials of one reduce are produced (on this thread,
/// for this thread's request).
pub fn partials_mark() -> PartialsMark {
    let Some(scope) = current_scope() else { return PartialsMark(None) };
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    REDUCE_MARKS.with(|marks| marks.borrow_mut().push((Arc::downgrade(&scope), count.clone())));
    PartialsMark(Some(count))
}

/// The partials retained under `mark` were consumed by a reduce whose own
/// result is charged: release them (and only them).
pub fn consume_partials_since(mark: PartialsMark) {
    let Some(count) = mark.0.clone() else { return };
    // Ends the reduce: later partials go to an enclosing one.
    drop(mark);
    if let Some(scope) = current_scope() { consume_partials_in(&scope, count.get()); }
}

/// A reduce consumed partials of `bytes` (their [`json_bytes`]): release up
/// to that much of this request's retained partials. A reduce that charges
/// its inputs itself would otherwise count them twice.
pub fn consume_retained_partials(bytes: u64) {
    if let Some(scope) = current_scope() { consume_partials_in(&scope, bytes); }
}

fn consume_partials_in(scope: &RequestScope, bytes: u64) {
    let mut retained = scope.retained.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(partials) = retained[PARTIALS].as_mut() {
        partials.shrink(bytes.min(partials.bytes));
    }
}

/// The bytes one request holds on the `request` breaker, across the
/// threads its work runs on (debug builds check it is back to zero when
/// the request and all its work are done: a reservation that outlives its
/// request is a leak).
pub struct RequestScope {
    held: AtomicI64,
    /// Retained query results, native results and partials ([`PARTIALS`]).
    retained: std::sync::Mutex<[Option<Reservation>;3]>,
    pub what: String,
    /// The breakers of the node serving the request.
    pub breakers: &'static Breakers,
    // The diagnostic String may outlive the HTTP request via worker scopes.
    // Keep its producer admission after the String until this scope drops.
    _label_owner: Option<Arc<dyn Send + Sync>>,
    native_control: bool,
}

impl RequestScope {
    pub fn new(what: impl Into<String>, breakers: &'static Breakers) -> Arc<RequestScope> {
        Arc::new(RequestScope { held: AtomicI64::new(0), retained: std::sync::Mutex::new([None,None,None]), what: what.into(), breakers, _label_owner: None, native_control:native_controlled(breakers) })
    }

    pub fn with_label_owner(what: String, breakers: &'static Breakers, owner: Arc<dyn Send + Sync>) -> Arc<RequestScope> {
        Arc::new(RequestScope { held: AtomicI64::new(0), retained: std::sync::Mutex::new([None,None,None]), what, breakers, _label_owner: Some(owner), native_control:native_controlled(breakers) })
    }

    pub fn held(&self) -> i64 {
        self.held.load(Ordering::Relaxed) + self.retained.lock().unwrap_or_else(|e|e.into_inner()).iter().flatten().map(|r|r.bytes as i64).sum::<i64>()
    }
}

impl Drop for RequestScope {
    fn drop(&mut self) {
        for retained in self.retained.get_mut().unwrap_or_else(|e|e.into_inner()){drop(retained.take());}
        let held = self.held.load(Ordering::Relaxed);
        if cfg!(debug_assertions) && held != 0 && !std::thread::panicking() {
            panic!("request [{}] left {held} bytes reserved on the request breaker", self.what);
        }
    }
}

thread_local! {
    static SCOPE: std::cell::RefCell<Option<Arc<RequestScope>>> = const { std::cell::RefCell::new(None) };
}

/// The request scope of work running on this thread.
pub fn current_scope() -> Option<Arc<RequestScope>> {
    SCOPE.with(|s| s.borrow().clone())
}

/// Run `f` as part of the request `scope` (reservations it makes count
/// toward it).
pub fn scoped<R>(scope: Option<Arc<RequestScope>>, f: impl FnOnce() -> R) -> R {
    let prev = SCOPE.with(|s| s.replace(scope));
    struct Restore(Option<Option<Arc<RequestScope>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(p) = self.0.take() {
                SCOPE.with(|s| *s.borrow_mut() = p);
            }
        }
    }
    let _restore = Restore(Some(prev));
    f()
}

fn scope_for(k: Kind, bytes: u64) -> Option<Weak<RequestScope>> {
    if k != Kind::Request {
        return None;
    }
    let s = current_scope()?;
    s.held.fetch_add(bytes as i64, Ordering::Relaxed);
    Some(Arc::downgrade(&s))
}

fn scope_add(scope: &Option<Weak<RequestScope>>, delta: i64) {
    if let Some(s) = scope.as_ref().and_then(Weak::upgrade) {
        s.held.fetch_add(delta, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(budget: u64) -> &'static Breakers {
        let b: &'static Breakers = Box::leak(Box::new(Breakers::new(budget, false)));
        b
    }

    /// Replay of acknowledged operations: every reservation is charged past
    /// the limits (native control included) without counting a trip, noted
    /// on the replay, and released as usual; outside the replay the same
    /// request is refused. A nested replay is part of the outer one.
    #[test]
    fn replay_reservations_are_forced_past_every_limit_and_released() {
        let b = local(1 << 20);
        b.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let limit = b.request.limit() as u64;
        assert!(b.reserve_request(limit + 1, "too big").is_err());
        let tripped = (b.request.tripped(), b.parent_tripped());
        let replay = Replay::new("test replay");
        let (mut held, native) = replay.run(|| {
            assert!(is_replaying());
            let held = b.reserve_request(limit + 1, "acked op").unwrap();
            let native = native_control(|| b.reserve_request(limit, "native control under replay")).unwrap();
            (held, native)
        });
        assert!(!is_replaying(), "the replay ends with the closure");
        assert!(replay.over_limit() >= 2);
        assert_eq!((b.request.tripped(), b.parent_tripped()), tripped, "a forced reservation is not a trip");
        assert_eq!(b.request.reserved(), 2 * limit + 1);
        replay.run(|| held.grow(limit, "acked op growth")).unwrap();
        assert_eq!(b.request.reserved(), 3 * limit + 1);
        assert!(b.reserve_request(1, "outside the replay").is_err());
        drop(held);
        drop(native);
        assert_eq!(b.request.reserved(), 0);
        replaying("outer", || {
            let outer = current_replay().unwrap();
            replaying("inner", || assert!(Arc::ptr_eq(&outer.0, &current_replay().unwrap().0)));
        });
        // A live replicated batch's completion is forced too, but it is not
        // a recovery; nested in a recovery it is part of the recovery.
        let held = replaying_live("live batch", || {
            assert!(is_replaying() && !is_recovery_replay());
            b.reserve_request(limit + 1, "admitted before the append").unwrap()
        });
        assert!(!is_replaying());
        drop(held);
        replaying("peer recovery", || replaying_live("batch", || assert!(is_recovery_replay())));
        assert_eq!(b.request.reserved(), 0);
    }

    /// Review finding 11: a request that retains each shard's partial and
    /// then reduces them holds the partials only until the reduce, and the
    /// final output only after it, not the sum of everything ever retained.
    #[test]
    fn reduced_partials_are_released_and_only_the_output_stays_retained() {
        let b = local(1 << 30);
        b.set_real_memory_for_tests(Some(0));
        let scope = RequestScope::new("multi-shard aggs", b);
        scoped(Some(scope.clone()), || {
            for _ in 0..4 { b.reserve_request(1000, "shard partial").unwrap().retain_partial_for_request(); }
            assert_eq!(b.request.used(), 4000);
            // The reduce consumes them (it charges its inputs itself).
            consume_retained_partials(4000);
            assert_eq!(b.request.used(), 0);
            let mark = partials_mark();
            b.reserve_request(300, "nested reduce output").unwrap().retain_partial_for_request();
            consume_partials_since(mark);
            b.reserve_request(700, "final output").unwrap().retain_for_request();
            // Never more than retained: a large consume cannot free other slots.
            consume_retained_partials(u64::MAX);
        });
        assert_eq!(b.request.used(), 700);
        assert_eq!(scope.held(), 700);
        drop(scope);
        assert_eq!(b.request.used(), 0);
    }

    /// A reduce releases only the partials retained under its own mark: not
    /// another reduce's (another thread of the same request) nor those a
    /// nested reduce already consumed.
    #[test]
    fn a_reduce_releases_only_its_own_partials() {
        let b = local(1 << 30);
        b.set_real_memory_for_tests(Some(0));
        let scope = RequestScope::new("parallel reduces", b);
        scoped(Some(scope.clone()), || {
            let outer = partials_mark();
            b.reserve_request(100, "outer partial").unwrap().retain_partial_for_request();
            let inner = partials_mark();
            b.reserve_request(40, "inner partial").unwrap().retain_partial_for_request();
            consume_partials_since(inner);
            assert_eq!(b.request.used(), 100);
            // Another thread of the request retains a partial meanwhile.
            let other = scope.clone();
            std::thread::spawn(move || scoped(Some(other), || {
                let mark = partials_mark();
                b.reserve_request(500, "other reduce partial").unwrap().retain_partial_for_request();
                drop(mark);
            })).join().unwrap();
            assert_eq!(b.request.used(), 600);
            consume_partials_since(outer);
            assert_eq!(b.request.used(), 500, "the other reduce's partial stays charged");
            // A reduce that fails keeps its partials.
            let failed = partials_mark();
            b.reserve_request(7, "failed reduce partial").unwrap().retain_partial_for_request();
            drop(failed);
            let next = partials_mark();
            consume_partials_since(next);
            assert_eq!(b.request.used(), 507);
        });
        drop(scope);
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn json_text_bytes_measures_the_tree_without_building_it() {
        for text in [r#"{"a":[1,2.5,null,true,"x\"y"],"bb":{"c":"\u00e9t\u00e9","d":[]}}"#, "[]", "\"s\"", "12"] {
            let tree: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(json_text_bytes(text).unwrap(), json_bytes(&tree), "{text}");
        }
        assert!(json_text_bytes("{\"a\":").is_err());
        assert!(json_text_bytes("1 2").is_err());
    }

    #[test]
    fn native_protocol_is_bounded_without_hiding_query_parent_pressure() {
        let b = local(1000);
        b.set_real_memory_for_tests(Some(960));
        let native = b.reserve_native(100, "protocol").unwrap();
        assert_eq!(b.request.used(), 0);
        assert_eq!(b.accounting.used(), 100);
        let error = b.reserve_request(1, "<http_request>").err().unwrap();
        assert_eq!(error.status, 429);
        assert!(error.reason.starts_with("[parent] Data too large, data for [<http_request>]"));
        assert!(b.reserve_native(901, "bounded protocol").is_err());
        assert_eq!(b.accounting.used(), 100);
        drop(native);
        assert_eq!(b.accounting.used(), 0);
    }

    #[test]
    fn native_policy_does_not_escape_into_an_ordinary_child() {
        let b = local(1000);
        b.set_real_memory_for_tests(Some(960));
        native_control(|| {
            let native = b.reserve_request(10, "control output").unwrap();
            assert_eq!(b.accounting.used(), 10);
            assert!(policy(false, || b.reserve_request(1, "query child")).is_err());
            drop(native);
        });
        assert!(b.reserve_request(1, "later query").is_err());
        assert_eq!(b.accounting.used(), 0);
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn retained_native_and_query_outputs_release_their_actual_breakers() {
        let b = local(10_000);
        b.set_real_memory_for_tests(Some(0));
        let scope = RequestScope::new("mixed output", b);
        scoped(Some(scope.clone()), || {
            b.reserve_request(100, "query output").unwrap().retain_for_request();
            b.reserve_native(80, "error output").unwrap().retain_for_request();
        });
        assert_eq!(scope.held(), 180);
        assert_eq!(b.request.used(), 100);
        assert_eq!(b.accounting.used(), 80);
        drop(scope);
        assert_eq!(b.request.used(), 0);
        assert_eq!(b.accounting.used(), 0);
    }

    #[test]
    fn worker_policy_is_captured_and_restored_without_tls_leak() {
        let b=local(1000);b.set_real_memory_for_tests(Some(960));
        let worker=native_control(||inherit_request_policy(move || {
            let held=b.reserve_request(20,"worker control").unwrap();
            assert_eq!(b.accounting.used(),20);drop(held);
            assert!(policy(false,||b.reserve_request(1,"worker query child")).is_err());
        }));
        worker();
        assert!(b.reserve_request(1,"after worker").is_err());
        assert_eq!(b.accounting.used(),0);
    }

    #[test]
    fn ordinary_worker_override_wins_over_inherited_control_scope() {
        let b=local(1000);b.set_real_memory_for_tests(Some(960));
        let scope=native_control(||RequestScope::new("control parent",b));
        let worker=scoped(Some(scope.clone()),||policy(false,||{
            let scope=scope.clone();
            inherit_request_policy(move||scoped(Some(scope),||{
                let error=b.reserve_request(1,"ordinary worker").err().unwrap();
                assert!(error.reason.starts_with("[parent]"));
            }))
        }));
        worker();drop(scope);
        assert_eq!(b.accounting.used(),0);
        assert_eq!(b.request.used(),0);
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_jvm_size("8g"), Some(8 << 30));
        assert_eq!(parse_jvm_size("512m"), Some(512 << 20));
        assert_eq!(parse_jvm_size("1024"), Some(1024));
        assert_eq!(auto_budget(16 << 30), 8 << 30);
        assert_eq!(auto_budget(128 << 30), 31 << 30);
        assert_eq!(auto_budget(1 << 30), 1 << 30);
        assert_eq!(byte_size(123), "123b");
        assert_eq!(byte_size(1536), "1.5kb");
        assert_eq!(byte_size(8 << 30), "8gb");
        // ES truncates (the ES 9 probe's default request limit of an 8gb heap).
        assert_eq!(byte_size(5153960755), "4.7gb");
        assert_eq!(byte_size(3435973836), "3.1gb");
        assert_eq!(byte_size(8160437862), "7.5gb");
    }

    #[test]
    fn the_indexing_buffer_setting_takes_bytes_or_a_bounded_percentage() {
        let settings = |pairs: &[(&str, &str)]| pairs.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect::<serde_json::Map<_, _>>();
        configure_indexing_buffer(&settings(&[("indices.memory.index_buffer_size", "256mb")])).unwrap();
        assert_eq!(indexing_buffer(), 256 << 20);
        configure_indexing_buffer(&settings(&[("indices.memory.index_buffer_size", "0.0001%"), ("indices.memory.min_index_buffer_size", "64mb")])).unwrap();
        assert_eq!(indexing_buffer(), 64 << 20);
        assert!(configure_indexing_buffer(&settings(&[("indices.memory.index_buffer_size", "lots")])).is_err());
        assert!(configure_indexing_buffer(&settings(&[("indices.memory.index_buffer_size", "0b")])).is_err(), "an empty buffer was accepted");
        assert!(configure_indexing_buffer(&settings(&[("indices.memory.index_buffer_size", "256mb"), ("indices.memory.min_index_buffer_size", "bogus")])).is_err(),
            "an invalid bound was ignored next to an absolute size");
        INDEXING_BUFFER.store(0, Ordering::Relaxed);
    }

    #[test]
    fn memory_settings_parse_like_es() {
        assert_eq!(parse_memory("60%", "k", 1000).unwrap(), 600);
        assert_eq!(parse_memory("1kb", "k", 0).unwrap(), 1024);
        assert_eq!(parse_memory("10mb", "k", 0).unwrap(), 10 << 20);
        assert_eq!(parse_memory("100b", "k", 0).unwrap(), 100);
        assert_eq!(parse_memory("-1", "k", 0).unwrap(), -1);
        assert_eq!(parse_memory("0", "k", 0).unwrap(), 0);
        assert_eq!(parse_memory("abc", "indices.breaker.request.limit", 0).unwrap_err().reason, "failed to parse setting [indices.breaker.request.limit] with value [abc] as a size in bytes: unit is missing or unrecognized");
        assert_eq!(parse_memory("x%", "k", 0).unwrap_err().reason, "failed to parse [x] as a double");
        assert!(parse_memory("101%", "k", 0).is_err());
        let e = validate_setting("indices.breaker.request.overhead", "-2").unwrap_err();
        assert_eq!(e.kind, "illegal_argument_exception");
        assert!(validate_setting("indices.breaker.request.limit", "40%").is_ok());
    }

    #[test]
    fn request_breaker_trips_with_es_shape() {
        // The child breaker alone (parent off).
        let b = local(1000);
        b.parent.limit.store(-1, Ordering::Relaxed);
        let r = b.reserve(Kind::Request, 500, "agg [x]").unwrap();
        let e = b.reserve(Kind::Request, 200, "agg [y]").err().unwrap();
        assert_eq!(e.status, 429);
        assert_eq!(e.kind, "circuit_breaking_exception");
        assert_eq!(e.reason, format!("[request] Data too large, data for [agg [y]] would be [700/700b], which is larger than the limit of [600/600b]{}", docs_suffix()));
        let body = e.shard_reason();
        assert_eq!(body["bytes_wanted"], 700);
        assert_eq!(body["bytes_limit"], 600);
        assert_eq!(body["durability"], "TRANSIENT");
        assert_eq!(b.request.used(), 500);
        assert_eq!(b.request.tripped(), 1);
        drop(r);
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn overhead_scales_the_child_check() {
        let b = local(1000);
        b.parent.limit.store(-1, Ordering::Relaxed);
        // inflight_requests: 100% with overhead 2.0, so 501 bytes trip.
        let r = b.reserve(Kind::InflightRequests, 500, "<http_request>").unwrap();
        drop(r);
        let e = b.reserve(Kind::InflightRequests, 501, "<http_request>").err().unwrap();
        assert!(e.reason.starts_with("[inflight_requests] Data too large, data for [<http_request>] would be [1002/1002b]"), "{}", e.reason);
        assert_eq!(b.inflight_requests.used(), 0);
        // fielddata is permanent.
        let e = b.reserve(Kind::Fielddata, 400, "f").err().unwrap();
        assert_eq!(e.shard_reason()["durability"], "PERMANENT");
    }

    #[test]
    fn parent_trips_on_real_memory_and_rolls_the_child_back() {
        let b = local(1000);
        b.set_real_memory_for_tests(Some(900));
        let r = b.reserve(Kind::Request, 50, "a").unwrap();
        let e = b.reserve(Kind::Request, 100, "<agg [big]>").err().unwrap();
        assert_eq!(e.status, 429);
        assert!(
            e.reason.starts_with("[parent] Data too large, data for [<agg [big]>] would be [1000/1000b], which is larger than the limit of [950/950b], real usage: [900/900b], new bytes reserved: [100/100b], usages [model_inference=0/0b, inflight_requests=0/0b, request=150/150b, fielddata=0/0b, eql_sequence=0/0b]"),
            "{}",
            e.reason
        );
        assert_eq!(b.parent_tripped(), 1);
        assert_eq!(b.request.tripped(), 0);
        // The child's bytes were given back.
        assert_eq!(b.request.used(), 50);
        drop(r);
        assert_eq!(b.request.used(), 0);
        // A zero-byte check trips too once real memory is over the limit.
        b.set_real_memory_for_tests(Some(960));
        assert!(b.add_estimate_bytes_and_maybe_break(Kind::Request, 0, "allocated_buckets").is_err());
        assert_eq!(b.parent_estimated(), 960);
    }

    #[test]
    fn parent_sums_children_without_real_memory() {
        let b = local(1000);
        b.apply_settings(&|k| (k == "indices.breaker.total.use_real_memory").then(|| "false".to_string()));
        assert_eq!(b.parent_limit(), 700);
        let _a = b.reserve(Kind::Request, 400, "a").unwrap();
        let _i = b.reserve(Kind::InflightRequests, 100, "b").unwrap();
        // 400 + 100 * 2.0 = 600; another 150 makes 750 > 700.
        let e = b.reserve(Kind::Request, 150, "c").err().unwrap();
        assert!(e.reason.starts_with("[parent] Data too large, data for [c] would be [750/750b], which is larger than the limit of [700/700b], usages ["), "{}", e.reason);
        assert_eq!(b.request.used(), 400);
        assert_eq!(b.parent_estimated(), 600);
    }

    #[test]
    fn settings_are_dynamic() {
        let b = local(1000);
        b.apply_settings(&|k| match k {
            "indices.breaker.request.limit" => Some("10%".into()),
            "indices.breaker.request.overhead" => Some("2".into()),
            "indices.breaker.total.limit" => Some("500b".into()),
            _ => None,
        });
        assert_eq!(b.request.limit(), 100);
        assert_eq!(b.request.overhead(), 2.0);
        assert_eq!(b.parent_limit(), 500);
        b.apply_settings(&|_| None);
        assert_eq!(b.request.limit(), 600);
        assert_eq!(b.parent_limit(), 950);
        let stats = b.stats();
        let names: Vec<_> = stats.iter().map(|s| s.name).collect();
        assert_eq!(names, ["model_inference", "inflight_requests", "request", "fielddata", "eql_sequence", "parent"]);
        // `accounting` is not a setting; its limit is the budget.
        assert_eq!(b.accounting.limit(), 1000);
    }

    #[test]
    fn hidden_accounting_stays_out_of_es9_figures() {
        let b = local(1000);
        b.apply_settings(&|k| (k == "indices.breaker.total.use_real_memory").then(|| "false".to_string()));
        // ES 9's parent sums only its own children: the hidden breaker's
        // bytes neither fill it nor tip a trip's durability.
        let held = b.reserve_without_breaking(Kind::Accounting, 650);
        let r = b.reserve(Kind::Request, 100, "x").unwrap();
        assert_eq!(b.parent_estimated(), 100);
        let e = b.reserve(Kind::Request, 550, "y").err().unwrap();
        assert!(e.reason.starts_with("[request] Data too large"), "{}", e.reason);
        let _f = b.reserve(Kind::Fielddata, 300, "f").unwrap();
        let e = b.reserve(Kind::Request, 350, "z").err().unwrap();
        assert!(e.reason.starts_with("[parent] Data too large, data for [z] would be [759/759b]"), "{}", e.reason);
        assert!(!e.reason.contains("accounting"), "{}", e.reason);
        assert_eq!(e.shard_reason()["durability"], "TRANSIENT");
        drop(r);
        // It still trips on its own limit.
        let e = b.reserve(Kind::Accounting, 400, "<retained>").err().unwrap();
        assert!(e.reason.starts_with("[accounting] Data too large, data for [<retained>]"), "{}", e.reason);
        drop(held);
    }

    /// Real memory over the parent's limit wakes the idle workers once (so
    /// threads already parked give their pages back); an idle collect that
    /// finds the process back under its limit stops the collecting.
    /// Combined review r2: a wake asked for within a second of the last is
    /// deferred to the end of that second, not dropped; any number asked
    /// meanwhile share that one deferred wake.
    #[test]
    fn a_wake_within_the_window_is_deferred_not_dropped() {
        let t0 = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        let mut w = Waking { last: None, deferred: false };
        assert_eq!(w.delay(t0), Some(std::time::Duration::ZERO));
        assert_eq!(w.delay(t0 + ms(300)), Some(ms(700)), "deferred to the window's end");
        assert_eq!(w.delay(t0 + ms(500)), None, "already one pending");
        // The deferred wake ran (the waker restarts the window).
        w.deferred = false;
        w.last = Some(t0 + ms(1000));
        assert_eq!(w.delay(t0 + ms(2500)), Some(std::time::Duration::ZERO));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn going_over_wakes_idle_workers_and_collecting_stops_when_back_under() {
        static WOKEN: AtomicU64 = AtomicU64::new(0);
        fn wake() {
            WOKEN.fetch_add(1, Ordering::Relaxed);
        }
        add_wake_hook(wake);
        // A 1000-byte budget: the test process's own resident memory is far
        // over its parent limit (real memory sampled, no override). The flag
        // is process-wide and other tests trip parents too: going over
        // again, until a wake (ours or theirs) comes through the rate limit.
        let b = local(1000);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while WOKEN.load(Ordering::Relaxed) == 0 && std::time::Instant::now() < deadline {
            OVER_LIMIT.store(false, Ordering::Relaxed);
            assert!(b.reserve(Kind::Request, 1, "x").is_err());
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(WOKEN.load(Ordering::Relaxed) >= 1, "idle workers woken");
        // Whether collecting stops is decided per breakers (the global flag
        // is shared with concurrent tests): the process's own breakers have
        // room, the 1000-byte ones do not.
        assert!(back_under(process_breakers()));
        assert!(!back_under(b));
    }

    #[test]
    fn a_new_reservation_counts_once_against_real_memory() {
        let b = local(1000);
        b.parent.real_override.store(u64::MAX, Ordering::Relaxed);
        // Pin the sample: 100 bytes real, nothing reserved since.
        b.parent.sampled.store(100, Ordering::Relaxed);
        b.parent.reserved_at_sample.store(0, Ordering::Relaxed);
        b.parent.sampled_at.store(u64::MAX / 2, Ordering::Relaxed);
        b.parent.trims_seen.store(TRIMS.load(Ordering::Relaxed), Ordering::Relaxed);
        // 100 real + 500 new = 600 <= 950; counted twice (the child's
        // copy and the new bytes) it would be 1100 and trip.
        let r = b.reserve(Kind::Request, 500, "a");
        assert!(r.is_ok(), "{:?}", r.err());
        let r2 = b.reserve(Kind::Request, 1, "b");
        assert!(r2.is_ok(), "{:?}", r2.err());
    }

    #[test]
    fn request_memory_charges_pages_and_releases_on_drop() {
        let b = local(10 << 20);
        b.parent.limit.store(-1, Ordering::Relaxed);
        let m = RequestMemory::on(b);
        for _ in 0..1000 {
            m.add(100, "<agg [t]>").unwrap();
        }
        // 100_000 bytes added: charged in pages, the remainder pending.
        assert!(m.charged() >= 100_000 - PAGE && m.charged() <= 100_000);
        assert_eq!(b.request.used(), m.charged());
        m.release(50_000);
        assert!(b.request.used() <= 50_000);
        drop(m);
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn request_memory_trips_and_releases_on_error_paths() {
        let b = local(1 << 20);
        b.parent.limit.store(-1, Ordering::Relaxed);
        let run = || -> Result<(), EsError> {
            let m = RequestMemory::on(b);
            loop {
                m.add(1024, "<agg [runaway]>")?;
            }
        };
        let e = run().unwrap_err();
        assert!(e.reason.starts_with("[request] Data too large, data for [<agg [runaway]>]"), "{}", e.reason);
        assert_eq!(b.request.used(), 0, "the error path released everything");
        // A panicking request (cancellation by unwinding) releases too.
        let r = std::panic::catch_unwind(|| {
            let m = RequestMemory::on(b);
            m.charge(1000, "x").unwrap();
            panic!("cancelled");
        });
        assert!(r.is_err());
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn buckets_check_the_parent_every_1024() {
        let b = local(1000);
        let m = RequestMemory::on(b);
        b.set_real_memory_for_tests(Some(990));
        // The first 1023 buckets never check.
        m.buckets(1023).unwrap();
        let e = m.buckets(1).unwrap_err();
        assert!(e.reason.contains("data for [allocated_buckets]"), "{}", e.reason);
        b.set_real_memory_for_tests(Some(0));
        m.buckets(1024).unwrap();
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_limit() {
        let b = local(100_000);
        b.parent.limit.store(-1, Ordering::Relaxed);
        let peak = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..2000 {
                        if let Ok(r) = b.reserve(Kind::Request, 7_000, "c") {
                            peak.fetch_max(b.request.used(), Ordering::Relaxed);
                            drop(r);
                        }
                    }
                });
            }
        });
        assert!(peak.load(Ordering::Relaxed) <= 60_000);
        assert_eq!(b.request.used(), 0);
    }

    #[test]
    fn request_scope_tracks_reservations_across_threads() {
        let b = local(1 << 20);
        b.parent.limit.store(-1, Ordering::Relaxed);
        let scope = RequestScope::new("test", b);
        let r = scoped(Some(scope.clone()), || b.reserve(Kind::Request, 100, "x").unwrap());
        assert_eq!(scope.held(), 100);
        let s2 = scope.clone();
        std::thread::spawn(move || scoped(Some(s2), || {
            let m = RequestMemory::on(b);
            m.charge(10, "y").unwrap();
        }))
        .join()
        .unwrap();
        assert_eq!(scope.held(), 100);
        drop(r);
        assert_eq!(scope.held(), 0);
        assert!(current_scope().is_none());
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;
    use rustic_core::admission::{Category,Owner,Lease,shared};
    fn breakers_for_test()->&'static Breakers {Breakers::private_for_tests(1<<30)}
    #[test]
    fn retained_output_keeps_each_original_shard_owner() {
        let _budget = rustic_core::admission::private_shared(256 << 20);
        let b=breakers_for_test();let scope=RequestScope::new("two shards",b);
        let a=Owner::new();let c=Owner::new();
        scoped(Some(scope.clone()),||{
            {let _a=a.enter();b.reserve_request(2048,"a output").unwrap().retain_for_request();}
            {let _c=c.enter();b.reserve_request(4096,"c output").unwrap().retain_for_request();}
        });
        assert_eq!(scope.held(),6144);
        assert!(a.bytes().current>=2048);assert!(c.bytes().current>=4096);
        assert_eq!(a.bytes(),a.category_bytes(Category::EngineRequest));
        assert_eq!(c.bytes(),c.category_bytes(Category::EngineRequest));
        drop(scope);assert_eq!(a.bytes().current,0);assert_eq!(c.bytes().current,0);
        assert_eq!(b.request.reserved(),0);
    }
    #[test]
    fn engine_buffer_uses_recovery_credit_and_keeps_origin_on_worker() {
        let _budget = rustic_core::admission::private_shared(256 << 20);
        let b=breakers_for_test();let owner=Owner::new();
        let lease=Lease::reserve(shared(),16384,"engine plan").unwrap();
        let mut held={let _owner=owner.enter();let _lease=lease.enter();b.reserve_request(4096,"decode").unwrap()};
        let remaining=lease.available();
        std::thread::scope(|threads|threads.spawn(||held.grow(1024,"worker growth").unwrap()).join().unwrap());
        assert_eq!(lease.available(),remaining-1024);
        lease.expire();
        assert_eq!(held.grow(1,"expired").unwrap_err().status,429);
        assert_eq!(held.bytes(),5120);assert_eq!(b.request.reserved(),5120);
        drop(held);assert_eq!(owner.bytes().current,0);assert_eq!(b.request.reserved(),0);
    }
    #[test]
    fn unchecked_real_allocation_cannot_certify_recovery_plan() {
        let b=breakers_for_test();let owner=Owner::new();
        let lease=Lease::reserve(shared(),4096,"plan").unwrap();
        let held={let _owner=owner.enter();let _lease=lease.enter();b.reserve_without_breaking(Kind::Request,100)};
        assert!(owner.has_unplanned_allocations());assert!(lease.complete().is_err());
        drop(held);drop(lease);
    }
}

#[cfg(test)]
mod store_file_tests {
    use super::*;

    /// A store file read into memory holds exactly its size on the
    /// `accounting` breaker until dropped; over the limit it is refused
    /// with the limit, and nothing stays reserved.
    #[test]
    fn store_files_are_held_on_the_accounting_breaker() {
        let b = Breakers::private_for_tests(1 << 30);
        b.get(Kind::Accounting).set_limit(10_000_000);
        let before = b.get(Kind::Accounting).used();
        let a = admit_store_file_on(b, 4_000_000, "seg_1.rstc").unwrap();
        let c = admit_store_file_on(b, 5_000_000, "seg_2.rstc").unwrap();
        assert_eq!(b.get(Kind::Accounting).used() - before, 9_000_000);
        let refused = admit_store_file_on(b, 2_000_000, "seg_3.rstc").err().expect("refused");
        assert_eq!(refused.limit, 10_000_000);
        assert!(refused.reason.contains("store file [seg_3.rstc]"), "{}", refused.reason);
        assert_eq!(b.get(Kind::Accounting).used() - before, 9_000_000, "a refusal reserves nothing");
        drop(a);
        assert_eq!(b.get(Kind::Accounting).used() - before, 5_000_000);
        let d = admit_store_file_on(b, 2_000_000, "seg_3.rstc").unwrap();
        // A recovery's replay charges past the limits, but not store files.
        let in_replay = replaying("test local recovery", || admit_store_file_on(b, 4_000_000, "seg_4.rstc").err());
        assert!(in_replay.is_some(), "a replay forced a store file past the limit");
        assert_eq!(b.get(Kind::Accounting).used() - before, 7_000_000);
        drop(c);
        drop(d);
        assert_eq!(b.get(Kind::Accounting).used(), before);
    }

    /// Through rustic-core: the refusal is the store file error, which the
    /// node reports as a `circuit_breaking_exception` with the file size
    /// and the limit.
    #[test]
    fn a_refused_store_file_is_a_circuit_breaking_exception() {
        use rustic_core::codec::directory::{Directory, MmapDirectory, StoreFileRefusal};
        fn admit(bytes: u64, file: &str) -> Result<Box<dyn Send>, StoreFileRefusal> {
            static B: OnceLock<&'static Breakers> = OnceLock::new();
            let b = *B.get_or_init(|| {
                let b = Breakers::private_for_tests(1 << 30);
                b.get(Kind::Accounting).set_limit(1_000_000);
                b
            });
            admit_store_file_on(b, bytes, file)
        }
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::open_in_memory(tmp.path(), admit).unwrap();
        dir.write("seg_9.rstc", &vec![7u8; 3_000_000]).unwrap();
        let rustic_core::RusticError::Memory(message) = dir.open_read("seg_9.rstc").unwrap_err() else { panic!("not a memory refusal") };
        let e = core_refusal(message);
        assert_eq!(e.status, 429);
        assert_eq!(e.kind, "circuit_breaking_exception");
        assert!(e.reason.contains("cannot read store file [seg_9.rstc] of [3000000] bytes"), "{}", e.reason);
        assert!(e.reason.contains("[node.store.allow_mmap]") && e.reason.contains("[vm.max_map_count]"), "{}", e.reason);
        let field = |k: &str| e.extra.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
        assert_eq!(field("bytes_wanted"), Some(serde_json::json!(3_000_000u64)));
        assert_eq!(field("bytes_limit"), Some(serde_json::json!(1_000_000u64)));
    }
}
