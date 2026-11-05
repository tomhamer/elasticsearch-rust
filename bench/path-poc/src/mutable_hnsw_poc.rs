//! Standalone f32 topology comparison. Build and execute on EC2 only.
//!
//! Run each mode in a NEW process/output directory; process peak RSS is then
//! meaningful. The same mmap, queries, construction parameters, visibility
//! stages and f64 ground truth are used for all modes. This is a prefix native
//! POC, never a reduced-data Rally result. Quantization, durable ES recovery,
//! historical PIT topology are separate gates. --mixed enables a bounded
//! insert-only simultaneous query/ingest gate, with explicit visibility limits.
#![recursion_limit = "256"]

use memmap2::Mmap;
use rayon::prelude::*;
use rustic_core::vector::{
    hnsw::{self, MutableVectorOwner, MutableVectorRead, MutableHnsw, MutableSearchMode, MutableVectorSource, NodeHandle, VespaFilterEstimate, VespaFilterFirstParams},
    HnswBuilder, HnswGraph, HnswParams, Similarity, VectorSource,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashSet},
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering::{Acquire, Relaxed, Release}},
        Arc, Barrier,
    },
    time::Instant,
};

type AnyResult<T> = Result<T, Box<dyn Error>>;

#[path = "mutable_hnsw_mixed.rs"]
mod mixed;

#[path = "mutable_hnsw_planner.rs"]
mod planner;

// Native default-policy adapter is opt-in; production ES routing is unchanged.
#[path = "mutable_hnsw_vespa_policy.rs"]
mod vespa_policy;

#[path = "mutable_hnsw_checkpoint.rs"]
mod checkpoint;

#[path = "mutable_hnsw_memory.rs"]
mod graph_memory;

struct Args {
    values: BTreeMap<String, String>,
}
impl Args {
    fn parse() -> AnyResult<Self> {
        let mut it = std::env::args().skip(1);
        let mut values = BTreeMap::new();
        while let Some(key) = it.next() {
            if key == "--help" {
                println!("mutable-hnsw-poc --fixture DIR --output NEW_DIR --mode mutable|segmented|single\n\
                    [--limit 100000 --queries 20 --chunks 12 --tail-fraction .12]\n\
                    [--clients 1,2,4,8,16,32 --ef 100,200,400,1000 --k 10,100]\n\
                    [--requests 128 --repeats 1 --filters all --fanout serial|parallel]\n\
                    [--threads N --insert-batch 64 --growth 1000 --churn 1000 --churn-rounds 2]\n\
                    [--mode compare --inputs DIR1,DIR2,DIR3 --output NEW_DIR]\n\
                    [--planner topology|production|exact-diagnostic|vespa; non-topology requires matching query-pool threads]\n\
                    [--exact-query-threads N; exact-diagnostic or vespa, defaults to 1]\n\
                    [--mutable-filter plain|vespa-first; opt-in topology/quiescent mode only]\n\
                    [--mutable-checkpoint COMPLETED_RUN_DIR; read-only topology mode, zero growth/churn]\n\
                    [--checkpoint-growth true; explicitly allow growth-only native profiling from that checkpoint]\n\
                    [--mode correctness --output NEW_DIR]\n\
                    [--mixed true --mixed-seconds 30 --mixed-write-rate 500 --mixed-query-rate 200]\n\
                    [--mixed-clients 8 --mixed-k 10 --mixed-ef 400 --mixed-filter all]\n\
                    [--mixed-batch 64 --mixed-queue-bytes 33554432 --mixed-query-queue 64]\n\
                    Independent stages recompute exact truth; no production ES/Rally claim.");
                std::process::exit(0);
            }
            if !key.starts_with("--") {
                return Err(format!("expected --option, got {key}").into());
            }
            let value = it
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            if values.insert(key[2..].to_owned(), value).is_some() {
                return Err(format!("duplicate {key}").into());
            }
        }
        let known = [
            "fixture",
            "output",
            "mode",
            "limit",
            "queries",
            "chunks",
            "tail-fraction",
            "clients",
            "ef",
            "k",
            "requests",
            "repeats",
            "filters",
            "fanout",
            "threads",
            "insert-batch",
            "growth",
            "churn",
            "churn-rounds",
            "inputs",
            "recall-floor",
            "seed",
            "m",
            "ef-construction",
            "mixed",
            "mixed-seconds",
            "mixed-write-rate",
            "mixed-query-rate",
            "mixed-clients",
            "mixed-k",
            "mixed-ef",
            "mixed-filter",
            "mixed-batch",
            "mixed-queue-bytes",
            "mixed-query-queue",
            "exact-query-threads",
            "planner",
            "mutable-filter",
            "vespa-exact-lower-limit",
            "mutable-checkpoint",
            "checkpoint-growth",
        ];
        if let Some(k) = values.keys().find(|k| !known.contains(&k.as_str())) {
            return Err(format!("unknown --{k}").into());
        }
        Ok(Self { values })
    }
    fn text<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.values.get(key).map(String::as_str).unwrap_or(default)
    }
    fn usize(&self, key: &str, default: usize) -> AnyResult<usize> {
        Ok(self
            .values
            .get(key)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default))
    }
    fn list(&self, key: &str, default: &str) -> AnyResult<Vec<usize>> {
        let v: Vec<usize> = self
            .text(key, default)
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if v.is_empty() || v.contains(&0) {
            return Err(format!("--{key} must contain positive integers").into());
        }
        Ok(v)
    }
}

fn validate_checkpoint_workload(
    checkpoint: bool, allow_growth: bool, mode: &str, supported_planner: bool,
    mixed: bool, growth: usize, churn: usize, rounds: usize,
) -> AnyResult<()> {
    if allow_growth && (!checkpoint || growth == 0) {
        return Err("checkpoint-growth requires a checkpoint and positive growth".into());
    }
    if checkpoint && (mode != "mutable" || !supported_planner || mixed || churn != 0 || rounds != 0
        || (growth != 0 && !allow_growth)) {
        return Err("checkpoint import requires supported mutable quiescent mode, zero churn/rounds, and explicit opt-in for growth".into());
    }
    Ok(())
}

struct Output {
    dir: PathBuf,
    events: BufWriter<File>,
}
impl Output {
    fn new(path: &str) -> AnyResult<Self> {
        if path.is_empty() {
            return Err("--output is required".into());
        }
        let dir = PathBuf::from(path);
        fs::create_dir(&dir)?; // Never overwrite an earlier attempt.
        let events = BufWriter::new(File::create(dir.join("events.jsonl"))?);
        Ok(Self { dir, events })
    }
    fn emit(&mut self, event: Value) -> AnyResult<()> {
        serde_json::to_writer(&mut self.events, &event)?;
        writeln!(&mut self.events)?;
        self.events.flush()?;
        if event["event"] != "query_samples" {
            println!("{event}");
        }
        Ok(())
    }
    fn save(&self, name: &str, value: &Value) -> AnyResult<()> {
        serde_json::to_writer_pretty(BufWriter::new(File::create(self.dir.join(name))?), value)?;
        Ok(())
    }
}

fn cpu_seconds() -> f64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // Whole process, including Rayon workers and every concurrent searcher.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut time) };
    assert_eq!(rc, 0, "process CPU clock unavailable");
    time.tv_sec as f64 + time.tv_nsec as f64 * 1e-9
}
fn memory() -> Value {
    let mut m = serde_json::Map::new();
    for (file, keys) in [
        (
            "/proc/self/status",
            &["VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSize"][..],
        ),
        (
            "/proc/self/smaps_rollup",
            &["Pss", "Private_Dirty", "Shared_Clean"][..],
        ),
    ] {
        for line in fs::read_to_string(file).unwrap_or_default().lines() {
            if let Some((key, value)) = line.split_once(':') {
                if keys.contains(&key) {
                    m.insert(
                        format!("{key}_kib"),
                        json!(value
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse::<u64>().ok())),
                    );
                }
            }
        }
    }
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
        m.insert("minor_faults".into(), json!(usage.ru_minflt));
        m.insert("major_faults".into(), json!(usage.ru_majflt));
        m.insert("voluntary_context_switches".into(), json!(usage.ru_nvcsw));
        m.insert(
            "involuntary_context_switches".into(),
            json!(usage.ru_nivcsw),
        );
    }
    // Resolve the actual v2 process cgroup, rather than reporting root limits.
    if let Some(path) = fs::read_to_string("/proc/self/cgroup").ok().and_then(|s| {
        s.lines()
            .find_map(|l| l.strip_prefix("0::").map(str::to_owned))
    }) {
        let root = Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
        for key in [
            "memory.current",
            "memory.peak",
            "memory.max",
            "memory.events",
            "cpu.max",
            "cpuset.cpus.effective",
        ] {
            if let Ok(v) = fs::read_to_string(root.join(key)) {
                m.insert(format!("cgroup_{key}"), json!(v.trim()));
            }
        }
    }
    Value::Object(m)
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Corpus {
    data: Mmap,
    dims: usize,
    base: usize,
    extras: Vec<Vec<f32>>,
    norms: Vec<f32>,
}
impl Corpus {
    fn load(root: &Path, limit: usize) -> AnyResult<(Self, Value)> {
        if cfg!(target_endian = "big") {
            return Err("fixture requires little-endian host".into());
        }
        let manifest: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        let dims = manifest["dims"].as_u64().ok_or("manifest dims missing")? as usize;
        let docs = manifest["docs"].as_u64().ok_or("manifest docs missing")? as usize;
        if dims == 0 || limit < 100_000 || limit > docs || limit > u32::MAX as usize {
            return Err("require 100K <= limit <= fixture docs and u32 range, nonzero dims".into());
        }
        let file = File::open(root.join("vectors.f32"))?;
        let expected = docs
            .checked_mul(dims)
            .and_then(|n| n.checked_mul(4))
            .ok_or("fixture length overflow")?;
        if file.metadata()?.len() != expected as u64 {
            return Err("vector file length disagrees with manifest".into());
        }
        // The run contract requires an immutable completed fixture. Mapping is
        // read-only; it stays alive for every graph/source borrow.
        let data = unsafe {
            memmap2::MmapOptions::new()
                .len(limit * dims * 4)
                .map(&file)?
        };
        if (data.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
            return Err("unaligned float mapping".into());
        }
        let mut corpus = Self {
            data,
            dims,
            base: limit,
            extras: Vec::new(),
            norms: Vec::new(),
        };
        if corpus.values().iter().any(|v| !v.is_finite()) {
            return Err("fixture has non-finite vectors".into());
        }
        corpus.norms = corpus
            .values()
            .chunks_exact(dims)
            .map(|v| v.iter().map(|x| x * x).sum())
            .collect();
        Ok((corpus, manifest))
    }
    fn values(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast::<f32>(), self.data.len() / 4) }
    }
    fn row(&self, key: u64) -> Option<&[f32]> {
        let key = usize::try_from(key).ok()?;
        if key < self.base {
            Some(&self.values()[key * self.dims..(key + 1) * self.dims])
        } else {
            self.extras.get(key - self.base).map(Vec::as_slice)
        }
    }
    fn len(&self) -> usize {
        self.base + self.extras.len()
    }
    fn memory_stats(&self) -> Value {
        json!({"rows":self.len(),"base_mapped_payload_bytes":self.data.len(),"retained_extra_rows":self.extras.len(),"extra_payload_bytes":self.extras.iter().map(|r|r.len()*std::mem::size_of::<f32>()).sum::<usize>(),"extra_payload_capacity_bytes":self.extras.iter().map(|r|r.capacity()*std::mem::size_of::<f32>()).sum::<usize>(),"extra_vec_header_capacity_bytes":self.extras.capacity()*std::mem::size_of::<Vec<f32>>(),"norm_capacity_bytes":self.norms.capacity()*std::mem::size_of::<f32>(),"scope":"logical payload/capacity excludes allocator overhead; source rows remain retained across churn; graph slot reuse alone does not prove bounded total memory"})
    }
    fn append_variant(&mut self, seed: usize, round: usize) -> u64 {
        let mut row = self.row((seed % self.base) as u64).unwrap().to_vec();
        // Deterministic immutable replacement; original rows stay available to
        // route immutable baseline tombstones and retained graph readers.
        let pos = (seed.wrapping_mul(17) + round.wrapping_mul(31)) % self.dims;
        row[pos] += 0.02 * ((round % 3 + 1) as f32);
        let norm = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut row {
                *v = (*v as f64 / norm) as f32;
            }
        }
        let key = self.len() as u64;
        self.norms.push(row.iter().map(|x| x * x).sum());
        self.extras.push(row);
        key
    }
}
impl MutableVectorSource for Corpus {
    fn get(&self, key: u64) -> Option<&[f32]> {
        self.row(key)
    }
}

struct RangeSource<'a> {
    corpus: &'a Corpus,
    start: usize,
    end: usize,
}
impl VectorSource for RangeSource<'_> {
    fn dim(&self) -> u32 {
        self.corpus.dims as u32
    }
    fn num_docs(&self) -> u32 {
        (self.end - self.start) as u32
    }
    fn get(&self, doc: u32) -> Option<&[f32]> {
        (doc < self.num_docs())
            .then(|| self.corpus.row(self.start as u64 + doc as u64))
            .flatten()
    }
    fn as_contiguous(&self) -> Option<&[f32]> {
        Some(&self.corpus.values()[self.start * self.corpus.dims..self.end * self.corpus.dims])
    }
    fn squared_norms(&self) -> Option<&[f32]> {
        Some(&self.corpus.norms[self.start..self.end])
    }
}
struct Part {
    start: usize,
    end: usize,
    graph: HnswGraph,
}
enum Engine {
    Mutable {
        graph: Arc<MutableHnsw>,
        owner: Option<Arc<MutableVectorOwner>>,
        ordinals: Vec<NodeHandle>,
    },
    Immutable {
        parts: Vec<Part>,
        tail: usize,
        parallel: bool,
    },
}
enum MutableSourceLease<'a> {
    Owned(MutableVectorRead<'a>),
    Corpus(&'a Corpus),
}
impl<'a> MutableSourceLease<'a> {
    fn new(owner: Option<&'a MutableVectorOwner>, corpus: &'a Corpus) -> Result<Self, String> {
        match owner {
            Some(owner) => owner.pin().map(Self::Owned).map_err(|e| e.to_string()),
            None => Ok(Self::Corpus(corpus)),
        }
    }
}
impl MutableVectorSource for MutableSourceLease<'_> {
    #[inline]
    fn prefetch_node(&self, handle: NodeHandle, key: u64) {
        if let Self::Owned(read) = self { read.prefetch_node(handle, key); }
    }
    fn get(&self, key: u64) -> Option<&[f32]> {
        match self { Self::Owned(read) => read.get(key), Self::Corpus(corpus) => corpus.row(key) }
    }
    #[inline]
    fn get_node(&self, handle: NodeHandle, key: u64) -> Option<&[f32]> {
        match self { Self::Owned(read) => read.get_node(handle, key), Self::Corpus(corpus) => corpus.row(key) }
    }
}
impl Engine {
    fn source_read<'a>(&'a self, corpus: &'a Corpus) -> MutableSourceLease<'a> {
        let owner = match self { Self::Mutable { owner, .. } => owner.as_deref(), _ => None };
        MutableSourceLease::new(owner, corpus).expect("mutable source read view")
    }
    fn search(
        &self,
        corpus: &Corpus,
        visible: &[bool],
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Filter,
    ) -> (Vec<(u64, f32)>, u64, u64, u64) {
        self.search_accepted(
            corpus, corpus.len(), query, k, ef, filter,
            |key| visible.get(key as usize).copied().unwrap_or(false),
            None,
        )
    }

    fn search_accepted(
        &self,
        corpus: &Corpus,
        point_limit: usize,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Filter,
        visible: impl Fn(u64) -> bool + Sync,
        filter_estimate: Option<VespaFilterEstimate>,
    ) -> (Vec<(u64, f32)>, u64, u64, u64) {
        let accept = |key: u64| key < point_limit as u64 && visible(key) && filter.accept(key);
        match self {
            Self::Mutable { graph, .. } => {
                let source = self.source_read(corpus);
                let result = if let Some(estimate) = filter_estimate {
                    graph
                        .search_with_filter_estimate(query, k, ef, &source, accept, estimate)
                        .expect("precomputed filter metadata matches captured visibility")
                } else {
                    graph.search(query, k, ef, &source, accept)
                };
                (
                    result.hits,
                    result.vector_ops,
                    result.entry_fallback_scanned,
                    result.missing_source_lookups,
                )
            }
            Self::Immutable {
                parts,
                tail,
                parallel,
            } => {
                let search = |part: &Part| {
                    let source = RangeSource {
                        corpus,
                        start: part.start,
                        end: part.end,
                    };
                    let before = hnsw::poc_vector_ops();
                    let hits = part
                        .graph
                        .search(query, k, ef, &source, Similarity::Cosine, |d| {
                            accept(part.start as u64 + d as u64)
                        });
                    let ops = hnsw::poc_vector_ops().wrapping_sub(before);
                    (
                        hits.into_iter()
                            .map(|(d, s)| (part.start as u64 + d as u64, s))
                            .collect::<Vec<_>>(),
                        ops,
                    )
                };
                let scan_tail = || {
                    let ords: Vec<u32> = (*tail..point_limit)
                        .filter(|&key| accept(key as u64))
                        .map(|key| key as u32)
                        .collect();
                    let scan = |chunk: &[u32]| {
                        rustic_core::vector::flat::poc_exact_top_k(
                            query,
                            Similarity::Cosine,
                            k,
                            chunk
                                .iter()
                                .map(|&key| (key, corpus.row(key as u64).unwrap())),
                            Some(&corpus.norms),
                        )
                    };
                    if !*parallel || ords.len() < 4096 {
                        scan(&ords)
                    } else {
                        // Same accepted-ordinal threshold/chunking as production
                        // VectorSegment::search_exact. Share the graph fanout's
                        // Rayon pool, rather than creating a pool per request.
                        let parts: Vec<_> = ords.par_chunks(1024).map(scan).collect();
                        let mut hits = Vec::with_capacity(parts.len() * k);
                        let mut scanned = 0;
                        for (part, count) in parts {
                            hits.extend(part);
                            scanned += count;
                        }
                        hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                        hits.truncate(k);
                        (hits, scanned)
                    }
                };
                let (chunks, (tail_hits, scanned)): (Vec<_>, _) = if *parallel {
                    rayon::join(|| parts.par_iter().map(search).collect(), scan_tail)
                } else {
                    (parts.iter().map(search).collect(), scan_tail())
                };
                let mut hits = Vec::with_capacity(parts.len() * k + k);
                let mut ops = 0;
                for (h, n) in chunks {
                    hits.extend(h);
                    ops += n;
                }
                // Call the production SIMD/cached-norm/prefetch scanner. The
                // ids here are global ordinals, matching corpus.norms exactly.
                ops += scanned;
                hits.extend(
                    tail_hits
                        .into_iter()
                        .map(|(key, score)| (key as u64, score)),
                );
                hits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                hits.truncate(k);
                (hits, ops, 0, 0)
            }
        }
    }
    fn stats(&self, corpus: &Corpus) -> Value {
        let mut stats = match self {
            Self::Mutable { graph, .. } => {
                let s = graph.stats();
                json!({"graphs":1,"allocated_nodes":s.allocated_slots,"active_nodes":s.active_nodes,"unindexed_tail":0,"pending_reservations":s.pending_reservations,"reusable_slots":s.reusable_slots,"retired_slots":s.retired_slots,"exhausted_slots":s.exhausted_slots,"visibility_retained":s.visibility_retained,"issued_key_watermark":s.issued_key_watermark,"live_adjacency_bytes":s.live_adjacency_bytes,"retired_adjacency_bytes":s.retired_adjacency_bytes,"neighbor_size":s.neighbor_size,"storage":graph_memory::report(graph)})
            }
            Self::Immutable {
                parts,
                tail,
                parallel,
            } => {
                json!({"graphs":parts.len(),"graph_nodes":parts.iter().map(|p|p.graph.num_nodes()).sum::<usize>(),"unindexed_tail":corpus.len()-tail,"parallel_fanout":parallel})
            }
        };
        stats["source_storage"] = corpus.memory_stats();
        if let Self::Mutable { owner: Some(owner), .. } = self {
            let (allocated, held) = owner.memory_bytes();
            stats["mutable_source_storage"] = json!({"allocated_bytes":allocated,"held_bytes":held,"scope":"owned vectors in addition to shared fixture/truth corpus"});
        }
        stats
    }
}

#[derive(Clone, Copy)]
struct Ranked {
    key: u64,
    score: f64,
}
impl PartialEq for Ranked {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.score.to_bits() == o.score.to_bits()
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
// Heap root is the worst accepted result (smaller score, larger tie id).
impl Ord for Ranked {
    fn cmp(&self, o: &Self) -> Ordering {
        o.score.total_cmp(&self.score).then(self.key.cmp(&o.key))
    }
}
fn keep(heap: &mut BinaryHeap<Ranked>, row: Ranked, k: usize) {
    if heap.len() < k {
        heap.push(row);
    } else if row < *heap.peek().unwrap() {
        heap.pop();
        heap.push(row);
    }
}
#[derive(Clone, Copy)]
enum Filter {
    One,
    All,
    Ten,
    Half,
}
impl Filter {
    fn parse(s: &str) -> AnyResult<Self> {
        match s {
            "one" => Ok(Self::One),
            "all" => Ok(Self::All),
            "ten" => Ok(Self::Ten),
            "half" => Ok(Self::Half),
            _ => Err(format!("unknown filter {s}").into()),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::One => "one",
            Self::All => "all",
            Self::Ten => "ten",
            Self::Half => "half",
        }
    }
    fn accept(self, key: u64) -> bool {
        match self {
            Self::One => key % 100 == 0,
            Self::All => true,
            Self::Ten => key % 10 == 0,
            Self::Half => key % 10 < 5,
        }
    }
}
fn exact_cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut an, mut bn) = (0., 0., 0.);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        an += a * a;
        bn += b * b;
    }
    (1. + if an > 0. && bn > 0. {
        dot / (an * bn).sqrt()
    } else {
        0.
    }) / 2.
}
fn truth(
    corpus: &Corpus,
    visible: &[bool],
    queries: &[Vec<f32>],
    k: usize,
    filter: Filter,
) -> Vec<Vec<(u64, f64)>> {
    queries
        .par_iter()
        .map(|q| {
            let mut heap = BinaryHeap::new();
            for (key, &live) in visible.iter().enumerate() {
                if live && filter.accept(key as u64) {
                    keep(
                        &mut heap,
                        Ranked {
                            key: key as u64,
                            score: exact_cosine(q, corpus.row(key as u64).unwrap()),
                        },
                        k,
                    );
                }
            }
            let mut out: Vec<_> = heap.into_iter().map(|r| (r.key, r.score)).collect();
            out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            out
        })
        .collect()
}
fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        0.
    } else {
        v[((v.len() - 1) as f64 * p).ceil() as usize]
    }
}

fn drain_graph_reclamation(graph: &MutableHnsw, owner: Option<&MutableVectorOwner>, stage: &str, out: &mut Output) -> AnyResult<()> {
    let started = Instant::now();
    let mut reclaimed = 0;
    loop {
        if let Some(owner) = owner {
            reclaimed += owner.reclaim()?;
            owner.commit()?;
        } else {
            let tokens = graph.collect_reclaimed();
            reclaimed += tokens.len();
            drop(tokens);
        }
        let stats = graph.stats();
        if stats.retired_slots == 0 && stats.retired_adjacency_bytes == 0 {
            out.emit(json!({"event":"quiescent_graph_reclamation","stage":stage,"reclaimed_slots":reclaimed,"elapsed_seconds":started.elapsed().as_secs_f64(),"allocated_slots":stats.allocated_slots,"reusable_slots":stats.reusable_slots,"retired_slots":stats.retired_slots,"retired_adjacency_bytes":stats.retired_adjacency_bytes,"scope":if owner.is_some() { "no concurrent readers; owned source references released before graph-slot reuse; truth fixture remains retained" } else { "no concurrent readers; graph reuse only; checkpoint fixture source remains retained" }}))?;
            return Ok(());
        }
        if started.elapsed().as_secs() >= 10 {
            return Err(format!("quiescent graph reclamation timed out in {stage}: {} slots / {} adjacency bytes retained",stats.retired_slots,stats.retired_adjacency_bytes).into());
        }
        std::thread::yield_now();
    }
}

fn insert_keys(
    graph: &MutableHnsw,
    owner: Option<&MutableVectorOwner>,
    corpus: &Corpus,
    keys: &[u64],
    batch: usize,
    ordinals: &mut Vec<NodeHandle>,
    out: &mut Output,
    stage: &str,
) -> AnyResult<()> {
    let start = Instant::now();
    let cpu = cpu_seconds();
    // One preparing batch may overlap the sequenced completion of the previous
    // batch. A rendezvous channel bounds proposals to two batches, including
    // the batch held by the writer. Dropping the receiver cancels the producer
    // on any completion/output error; scope exit joins it before returning.
    std::thread::scope(|scope| -> AnyResult<()> {
        let (send, receive) = std::sync::mpsc::sync_channel(0);
        let preparer = scope.spawn(move || -> Result<(), String> {
            for chunk in keys.chunks(batch) {
                let begin = Instant::now();
                let pcpu = cpu_seconds();
                let reservations = chunk.iter().map(|&key| graph.reserve_insert(key))
                    .collect::<rustic_core::error::Result<Vec<_>>>()
                    .map_err(|e| e.to_string())?;
                if let Some(owner) = owner {
                    for reservation in &reservations {
                        owner.install(reservation, corpus.row(reservation.key()).ok_or("missing insertion row")?)
                            .map_err(|e| e.to_string())?;
                    }
                }
                let source = MutableSourceLease::new(owner, corpus)?;
                let prepared = reservations.into_par_iter()
                    .map(|reservation| graph.prepare(reservation, &source))
                    .collect::<rustic_core::error::Result<Vec<_>>>()
                    .map_err(|e| e.to_string())?;
                let prepare_ms = begin.elapsed().as_secs_f64() * 1000.;
                let prepare_cpu_ms = (cpu_seconds() - pcpu) * 1000.;
                if send.send((chunk[0], chunk.len(), begin, prepare_ms, prepare_cpu_ms, prepared)).is_err() {
                    return Ok(()); // writer failed; its original error wins
                }
            }
            Ok(())
        });
        let result = (|| -> AnyResult<()> {
            for (first_key, rows, begin, prepare_ms, prepare_cpu_ms, prepared) in receive.iter() {
                let bootstrap = prepared.iter().filter(|p| p.is_bootstrap()).count();
                let commit = Instant::now();
                let ccpu = cpu_seconds();
                {
                    let _commit = graph.commit_batch()?;
                    let source = MutableSourceLease::new(owner, corpus)?;
                    for proposal in prepared {
                        ordinals.push(graph.complete(proposal, &source)?);
                    }
                }
                if let Some(owner) = owner { owner.commit()?; }
                out.emit(json!({"event":"insert_batch","stage":stage,"first_key":first_key,"rows":rows,"bootstrap_markers":bootstrap,"two_phase_proposals":rows-bootstrap,"bootstrap_decision":"captured_by_graph_at_prepare","prepare_ms":prepare_ms,"prepare_cpu_ms":prepare_cpu_ms,"complete_ms":commit.elapsed().as_secs_f64()*1000.,"complete_cpu_ms":(cpu_seconds()-ccpu)*1000.,"bounded_prepared_operations":batch.saturating_mul(2),"elapsed_ms":begin.elapsed().as_secs_f64()*1000.,"pipeline":"overlapped_prepare_sequenced_complete","cpu_accounting":"process CPU intervals overlap; use insert_stage for total CPU"}))?;
            }
            Ok(())
        })();
        drop(receive);
        let producer_result = preparer.join();
        result?;
        producer_result.map_err(|_| "insertion preparer panicked")?
            .map_err(|error| -> Box<dyn Error> { error.into() })?;
        Ok(())
    })?;
    let compact_started = Instant::now();
    let compacted_buffers = graph.compact_storage()?;
    let source_compacted_buffers = if let Some(owner) = owner { owner.compact_storage()? } else { 0 };
    out.emit(json!({"event":"graph_storage_compaction","stage":stage,"buffers":compacted_buffers,"source_buffers":source_compacted_buffers,"elapsed_seconds":compact_started.elapsed().as_secs_f64(),"scope":"stage-boundary array relocation; no graph rebuild"}))?;
    out.emit(json!({"event":"insert_stage","stage":stage,"rows":keys.len(),"elapsed_seconds":start.elapsed().as_secs_f64(),"process_cpu_seconds":cpu_seconds()-cpu,"pipeline":"overlapped_prepare_sequenced_complete","bounded_prepared_operations":batch.saturating_mul(2),"memory":memory()}))
}

#[derive(Clone)]
struct Quality {
    ef: usize,
    k: usize,
    filter: Filter,
    mean: f64,
    min: f64,
    fingerprints: Vec<[u8; 32]>,
    plan: Option<Arc<planner::PreparedFilter>>,
    mutable_filter: Option<VespaFilterFirstParams>,
    plain_filter_estimate: Option<VespaFilterEstimate>,
}

impl Quality {
    fn search(&self, engine: &Engine, corpus: &Corpus, visible: &[bool], query: &[f32]) -> planner::Outcome {
        quiescent_search(engine,corpus,visible,query,self.k,self.ef,self.filter,self.plan.as_deref(),self.mutable_filter,self.plain_filter_estimate)
    }
}

#[allow(clippy::too_many_arguments)]
fn quiescent_search(engine: &Engine, corpus: &Corpus, visible: &[bool], query: &[f32], k: usize, ef: usize, filter: Filter,
    plan: Option<&planner::PreparedFilter>, mutable_filter: Option<VespaFilterFirstParams>,
    plain_filter_estimate: Option<VespaFilterEstimate>) -> planner::Outcome {
    if let (Engine::Mutable { graph, .. }, Some(params)) = (engine, mutable_filter) {
        let source = engine.source_read(corpus);
        let result = graph.search_mode(query,k,ef,&source,
            |key| visible.get(key as usize).copied().unwrap_or(false) && filter.accept(key),
            MutableSearchMode::VespaFilterFirst(params)).expect("validated precomputed filter-first configuration");
        return planner::Outcome { hits:result.hits, vector_ops:result.vector_ops,
            entry_fallback:result.entry_fallback_scanned, missing_source:result.missing_source_lookups,
            stats:planner::QueryStats::default() };
    }
    if plan.is_none() {
        if let (Engine::Mutable { graph, .. }, Some(estimate)) = (engine, plain_filter_estimate) {
            let source = engine.source_read(corpus);
            let result = graph.search_with_filter_estimate(query,k,ef,&source,
                |key| visible.get(key as usize).copied().unwrap_or(false) && filter.accept(key), estimate)
                .expect("precomputed Plain filter metadata");
            return planner::Outcome { hits:result.hits, vector_ops:result.vector_ops,
                entry_fallback:result.entry_fallback_scanned, missing_source:result.missing_source_lookups,
                stats:planner::QueryStats::default() };
        }
    }
    planner::search(engine,corpus,visible,query,k,ef,filter,plan)
}

// Source keys are document IDs in this fixture. Their universe includes hidden
// old incarnations and is deliberately independent of reused graph node slots.
fn prepared_filter_estimate(visible: &[bool], filter: Filter) -> AnyResult<Option<VespaFilterEstimate>> {
    if matches!(filter, Filter::All) {
        return Ok(None);
    }
    let size = u32::try_from(visible.len())?;
    let count = u32::try_from(
        visible
            .iter()
            .enumerate()
            .filter(|(key, live)| **live && filter.accept(*key as u64))
            .count(),
    )?;
    Ok(Some(VespaFilterEstimate { size, count }))
}

fn prepared_mutable_filter(visible: &[bool], filter: Filter, enabled: bool) -> AnyResult<Option<VespaFilterFirstParams>> {
    if !enabled || matches!(filter, Filter::All) {
        return Ok(None);
    }
    Ok(Some(VespaFilterFirstParams {
        filter_estimate: prepared_filter_estimate(visible, filter)?,
        ..Default::default()
    }))
}

fn hit_fingerprint(hits: &[(u64, f32)]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update((hits.len() as u64).to_le_bytes());
    for &(key, score) in hits {
        digest.update(key.to_le_bytes());
        digest.update(score.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

fn validate_quiescent_hits(
    hits: &[(u64, f32)],
    wanted: usize,
    accept: impl Fn(u64) -> bool,
    expected_score: impl Fn(u64) -> Result<f64, String>,
) -> Result<(), String> {
    if hits.len() != wanted {
        return Err(format!("incomplete hits {} != {wanted}", hits.len()));
    }
    let mut seen = HashSet::new();
    let mut previous = f32::INFINITY;
    for &(key, score) in hits {
        if !seen.insert(key) || !accept(key) || !score.is_finite() || score > previous {
            return Err("duplicate, invisible, rejected, nonfinite or unordered hit".into());
        }
        previous = score;
        let expected = expected_score(key)?;
        if !expected.is_finite() || (score as f64 - expected).abs() > 2e-5 {
            return Err(format!("score mismatch {score} != {expected}"));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct TimedSample {
    sequence: usize,
    query: usize,
    latency_ms: f64,
    vector_ops: u64,
    fallback: u64,
    missing_source: u64,
    fingerprint: [u8; 32],
    planner_stats: planner::QueryStats,
}

fn validate_timed_samples(
    samples: &[TimedSample],
    requests: usize,
    fingerprints: &[[u8; 32]],
) -> Result<(), String> {
    if fingerprints.is_empty() || samples.len() != requests {
        return Err("timed query result coverage differs from requested work".into());
    }
    for (sequence, sample) in samples.iter().enumerate() {
        let query = sequence % fingerprints.len();
        if sample.sequence != sequence || sample.query != query {
            return Err(format!("timed query sequence/query coverage mismatch at {sequence}"));
        }
        if !sample.latency_ms.is_finite() || sample.latency_ms < 0. {
            return Err(format!("invalid timed query latency at {sequence}"));
        }
        if sample.fallback != 0 {
            return Err(format!("unexpected timed entry scan at {sequence}: {} nodes", sample.fallback));
        }
        if sample.missing_source != 0 {
            return Err(format!("missing timed source lookup at {sequence}: {} lookups", sample.missing_source));
        }
        if sample.fingerprint != fingerprints[query] {
            return Err(format!("timed query {sequence} differs from its validated ordered IDs/scores"));
        }
    }
    Ok(())
}

fn quiescent_worker<T>(
    ready: &Barrier,
    start: &Barrier,
    cancelled: &AtomicBool,
    warmup: impl FnOnce() -> Result<(), String>,
    measured: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let warmed = catch_unwind(AssertUnwindSafe(warmup))
        .unwrap_or_else(|_| Err("quiescent worker warmup panicked".into()));
    if warmed.is_err() {
        cancelled.store(true, Release);
    }
    // A failed warmup must still rendezvous; returning early strands the
    // coordinator and other workers before they can observe cancellation.
    ready.wait();
    start.wait();
    warmed?;
    if cancelled.load(Acquire) {
        return Err("quiescent measurement cancelled during startup".into());
    }
    let result = catch_unwind(AssertUnwindSafe(measured))
        .unwrap_or_else(|_| Err("quiescent measured worker panicked".into()));
    if result.is_err() {
        cancelled.store(true, Release);
    }
    result
}

fn quality_stage(
    engine: &Engine,
    corpus: &Corpus,
    visible: &[bool],
    queries: &[Vec<f32>],
    ks: &[usize],
    efs: &[usize],
    filters: &[Filter],
    stage: &str,
    planner_settings: Option<&planner::Settings>,
    vespa_filter_first: bool,
    out: &mut Output,
) -> AnyResult<Vec<Quality>> {
    let mut results = Vec::new();
    let max_k = *ks.iter().max().unwrap();
    for &filter in filters {
        let preparation = Instant::now();
        let plan = planner_settings.map(|settings| Arc::new(planner::PreparedFilter::prepare(engine,corpus,visible,filter,settings)));
        if let Some(plan) = &plan {
            let elapsed = preparation.elapsed().as_secs_f64();
            out.emit(json!({"event":"prepared_filter_plan","stage":stage,"filter":filter.name(),"preparation_seconds":elapsed,"excluded_from_query_timing":true,"plan":plan.report()}))?;
        }
        let preparation = Instant::now();
        let mutable_filter = prepared_mutable_filter(visible,filter,vespa_filter_first)?;
        let plain_filter_estimate = prepared_filter_estimate(visible, filter)?;
        if let Engine::Mutable { graph, .. } = engine {
            out.emit(json!({"event":"prepared_visited_filter_metadata","stage":stage,"filter":filter.name(),
                "document_filter_size":plain_filter_estimate.map(|f| f.size),
                "accepted_documents":plain_filter_estimate.map(|f| f.count),
                "graph_nodeid_limit":graph.snapshot().num_nodes().saturating_add(1),
                "excluded_from_query_timing":true,
                "scope":"native All has no GlobalFilter metadata; filtered document-key counts include visibility and never substitute for reusable graph node bounds"}))?;
        }
        if vespa_filter_first {
            let estimate = mutable_filter.and_then(|params| params.filter_estimate);
            out.emit(json!({"event":"prepared_mutable_filter","stage":stage,"filter":filter.name(),
                "applied":matches!(engine,Engine::Mutable { .. }) && mutable_filter.is_some(),
                "document_limit":estimate.map(|e|e.size),"accepted_documents":estimate.map(|e|e.count),
                "preparation_seconds":preparation.elapsed().as_secs_f64(),"excluded_from_query_timing":true,
                "scope":"single-vector documents with monotonic incarnation IDs; visibility and filter predicate checks remain inside traversal; All uses Plain; immutable controls retain existing behavior"}))?;
        }
        let start = Instant::now();
        let exact = truth(corpus, visible, queries, max_k, filter);
        out.save(&format!("truth-{stage}-{}.json",filter.name()),&json!({"stage":stage,"filter":filter.name(),"accumulation":"f64 original stored f32","queries":queries,"hits":exact}))?;
        out.emit(json!({"event":"ground_truth","stage":stage,"filter":filter.name(),"queries":queries.len(),"k":max_k,"elapsed_seconds":start.elapsed().as_secs_f64()}))?;
        for &k in ks {
            for &ef in efs {
                let mut recalls = Vec::new();
                let mut details = Vec::new();
                let mut fingerprints = Vec::with_capacity(queries.len());
                for (qi, q) in queries.iter().enumerate() {
                    let result = quiescent_search(engine,corpus,visible,q,k,ef,filter,plan.as_deref(),mutable_filter,plain_filter_estimate);
                    let (hits, ops, fallback, missing_source) = (result.hits,result.vector_ops,result.entry_fallback,result.missing_source);
                    if missing_source != 0 {
                        return Err(format!("missing source lookup in quiescent {stage}: {missing_source}").into());
                    }
                    if fallback != 0 {
                        return Err(format!(
                            "unexpected entry scan in quiescent {stage}: {fallback} nodes"
                        )
                        .into());
                    }
                    let wanted = exact[qi].len().min(k);
                    validate_quiescent_hits(
                        &hits,
                        wanted,
                        |key| visible.get(key as usize).copied().unwrap_or(false) && filter.accept(key),
                        |key| corpus.row(key).map(|row| exact_cosine(q, row)).ok_or_else(|| "missing vector".into()),
                    )?;
                    fingerprints.push(hit_fingerprint(&hits));
                    let recall = if wanted == 0 {
                        1.
                    } else {
                        hits.iter()
                            .filter(|h| exact[qi][..wanted].iter().any(|t| t.0 == h.0))
                            .count() as f64
                            / wanted as f64
                    };
                    recalls.push(recall);
                    let mut detail = json!({"query":qi,"recall":recall,"hits":hits,"vector_ops":ops,"entry_fallback_scanned":fallback,"missing_source_lookups":missing_source});
                    if plan.is_some() { detail["planner_stats"] = result.stats.json(); }
                    details.push(detail);
                }
                let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let min = recalls.iter().copied().fold(1., f64::min);
                out.emit(json!({"event":"quality","stage":stage,"filter":filter.name(),"k":k,"ef":ef,"mean_recall":mean,"min_recall":min,"per_query":details,"scores_verified":true}))?;
                results.push(Quality {
                    ef,
                    k,
                    filter,
                    mean,
                    min,
                    fingerprints,
                    plan: plan.clone(),
                    mutable_filter,
                    plain_filter_estimate,
                });
            }
        }
    }
    Ok(results)
}

fn measure(
    engine: &Engine,
    corpus: &Corpus,
    visible: &[bool],
    queries: &[Vec<f32>],
    q: &Quality,
    clients: usize,
    requests: usize,
    repeat: usize,
    stage: &str,
    out: &mut Output,
) -> AnyResult<()> {
    let requested_requests = requests;
    let requests = requests.max(clients * 32).div_ceil(queries.len()) * queries.len();
    let worker_warmups = queries.len().min(4);
    let next = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let ready = Barrier::new(clients + 1);
    let barrier = Barrier::new(clients + 1);
    let (mut samples, wall, cpu) = std::thread::scope(|scope| -> AnyResult<_> {
        let workers: Vec<_> = (0..clients)
            .map(|_| {
                scope.spawn(|| {
                    quiescent_worker(
                        &ready,
                        &barrier,
                        &cancelled,
                        || {
                            // Scratch/visited buffers are thread-local. Warm
                            // the actual query worker outside both clocks.
                            for query in &queries[..worker_warmups] {
                                std::hint::black_box(q.search(engine,corpus,visible,query));
                            }
                            Ok(())
                        },
                        || {
                            let mut samples = Vec::new();
                            while !cancelled.load(Acquire) {
                                let seq = next.fetch_add(1, Relaxed);
                                if seq >= requests {
                                    break;
                                }
                                let qi = seq % queries.len();
                                let t = Instant::now();
                                let result = q.search(engine,corpus,visible,&queries[qi]);
                                let latency_ms = t.elapsed().as_secs_f64() * 1000.;
                                // Hash ordered IDs and exact score bits after
                                // search latency, but inside aggregate CPU/wall
                                // timing. Retaining a compact digest keeps long
                                // profile runs from retaining every top-k list.
                                let fingerprint = hit_fingerprint(&result.hits);
                                samples.push(TimedSample {
                                    sequence: seq,
                                    query: qi,
                                    latency_ms,
                                    vector_ops: result.vector_ops,
                                    fallback: result.entry_fallback,
                                    missing_source: result.missing_source,
                                    fingerprint,
                                    planner_stats: result.stats,
                                });
                            }
                            Ok(samples)
                        },
                    )
                })
            })
            .collect();
        ready.wait();
        // Profilers can wait for this marker instead of attaching during
        // f64 quality checks or first-use worker scratch allocation. Logging
        // precedes both clocks and the release of measured requests.
        let start_marker = out.emit(json!({"event":"query_measurement_start","stage":stage,"filter":q.filter.name(),"k":q.k,"ef":q.ef,"clients":clients,"repeat":repeat,"requests":requests,"worker_warmups_complete":true}));
        let before = cpu_seconds();
        let t = Instant::now();
        barrier.wait();
        let mut samples = Vec::new();
        let mut failure = None;
        // Join everyone even after failure, so no worker can outlive a failed
        // measurement or contaminate the next mode's benchmark process.
        for worker in workers {
            let result = worker.join().unwrap_or_else(|_| {
                cancelled.store(true, Release);
                Err("quiescent worker panicked outside its guarded body".into())
            });
            match result {
                Ok(mut worker_samples) => samples.append(&mut worker_samples),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        let wall = t.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - before;
        // Even a logging error must release/join waiting workers first.
        start_marker?;
        if let Some(error) = failure {
            return Err(error.into());
        }
        Ok((samples, wall, cpu))
    })?;
    out.emit(json!({"event":"query_measurement_end","stage":stage,"filter":q.filter.name(),"k":q.k,"ef":q.ef,"clients":clients,"repeat":repeat,"requests":requests,"wall_seconds":wall,"process_cpu_seconds":cpu}))?;
    samples.sort_by_key(|s| s.sequence);
    validate_timed_samples(&samples, requests, &q.fingerprints)?;
    let mut latencies: Vec<_> = samples.iter().map(|s| s.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);
    let ops: u64 = samples.iter().map(|s| s.vector_ops).sum();
    let fallback: u64 = samples.iter().map(|s| s.fallback).sum();
    let missing_source: u64 = samples.iter().map(|s| s.missing_source).sum();
    if let Some(plan) = &q.plan {
        out.emit(json!({"event":"planner_measurement","stage":stage,"repeat":repeat,"clients":clients,"requests":requests,"k":q.k,"ef":q.ef,"filter":q.filter.name(),"plan":plan.report(),"stats_total":planner::QueryStats::total(samples.iter().map(|s|s.planner_stats)).json(),"intentional_short_hit_exact_fallbacks_allowed":true}))?;
    }
    out.emit(json!({"event":"measurement","stage":stage,"repeat":repeat,"clients":clients,"requests":requests,"requested_requests":requested_requests,"worker_warmup_queries":worker_warmups,"target_mean_requests_per_client":32,"k":q.k,"ef":q.ef,"filter":q.filter.name(),"mean_recall":q.mean,"min_recall":q.min,"wall_seconds":wall,"process_cpu_seconds":cpu,"cpu_ms_per_query":cpu*1000./requests as f64,"qps":requests as f64/wall,"p50_ms":percentile(&latencies,0.5),"p95_ms":percentile(&latencies,0.95),"p99_ms":percentile(&latencies,0.99),"vector_ops_per_query":ops as f64/requests as f64,"entry_fallback_scanned":fallback,"missing_source_lookups":missing_source,"timed_results_verified":true,"result_verification":"every timed ordered ID/score-bit SHA256 matches its validated quiescent reference; exact sequence/query coverage, zero fallback and zero missing source lookups checked after timing","timing_instrumentation":"fingerprinting included in aggregate CPU/wall, excluded from per-request search latency","memory":memory(),"graph":engine.stats(corpus),"load_model":"fixed equal request count, closed-loop clients, no pacing"}))?;
    let records: Vec<_> = samples.iter().map(|s| (s.sequence, s.query, s.latency_ms, s.vector_ops, s.fallback, s.fingerprint, s.missing_source)).collect();
    out.emit(json!({"event":"query_samples","stage":stage,"repeat":repeat,"clients":clients,"k":q.k,"ef":q.ef,"filter":q.filter.name(),"columns":["sequence","query","latency_ms","vector_ops","entry_fallback_scanned","ordered_hits_sha256_bytes","missing_source_lookups"],"samples":records}))
}

fn correctness(out: &mut Output) -> AnyResult<()> {
    struct Rows(Vec<Vec<f32>>);
    impl MutableVectorSource for Rows {
        fn get(&self, k: u64) -> Option<&[f32]> {
            self.0.get(k as usize).map(Vec::as_slice)
        }
    }
    let rows = Rows((0..513).map(|i| vec![i as f32, (i % 7) as f32]).collect());
    let graph = MutableHnsw::new(
        2,
        Similarity::L2,
        HnswParams {
            m: 8,
            ef_construction: 64,
            threads: 1,
            ..Default::default()
        },
    );
    let mut ordinals = Vec::new();
    for key in 0..512 {
        ordinals.push(graph.insert(key, &rows)?);
    }
    let check = |g: &MutableHnsw, visible: &[bool]| -> AnyResult<()> {
        for id in [0usize, 7, 255, 511] {
            let q = &rows.0[id];
            let result = g.search(q, 10, 1024, &rows, |k| {
                visible.get(k as usize).copied().unwrap_or(false) && k % 2 == 0
            });
            if result.missing_source_lookups != 0 {
                return Err("numeric correctness encountered a missing source lookup".into());
            }
            let mut exact: Vec<_> = visible
                .iter()
                .enumerate()
                .filter(|(i, v)| **v && i % 2 == 0)
                .map(|(i, _)| (i as u64, Similarity::L2.score(q, &rows.0[i])))
                .collect();
            exact.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            exact.truncate(10);
            let mut actual = result.hits;
            actual.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            if actual != exact {
                return Err(format!("numeric correctness mismatch query {id}").into());
            }
        }
        Ok(())
    };
    let mut visible = vec![true; 513];
    visible[512] = false;
    check(&graph, &visible)?;
    let inserted = graph.insert(512, &rows)?; // Graph-ready but not yet visible.
    check(&graph, &visible)?;
    visible[512] = true;
    graph.remove(ordinals[256], &rows)?;
    visible[256] = false;
    check(&graph, &visible)?;
    graph.remove(inserted, &rows)?;
    visible[512] = false;
    check(&graph, &visible)?;
    let bytes = graph.encode();
    let reopened = MutableHnsw::open(&bytes)?;
    check(&reopened, &visible)?;
    out.emit(json!({"event":"numeric_correctness","passed":true,"rows":513,"metric":"l2","checks":["exact filtered top10","hidden ready node","delete","quiescent encode/open"],"scope":"small correctness fixture only; no performance claim or PIT-repeatability proof"}))
}

fn compare(args: &Args, out: &mut Output) -> AnyResult<()> {
    let floor = args.text("recall-floor", "0.95").parse::<f64>()?;
    if !(0.0..=1.0).contains(&floor) {
        return Err("invalid recall floor".into());
    }
    let mut runs = Vec::new();
    let mut identities = HashSet::new();
    for path in args.text("inputs", "").split(',').filter(|s| !s.is_empty()) {
        let root = Path::new(path);
        let mut run: Value = serde_json::from_slice(&fs::read(root.join("run.json"))?)?;
        run["comparison_input_dir"] = json!(root);
        let identity = if run["mode"] == "mutable" {
            "mutable".to_owned()
        } else {
            format!("{}/{}", run["mode"], run["fanout"])
        };
        if !identities.insert(identity) {
            return Err(
                "duplicate mode/fanout input (use separate comparisons for repeats)".into(),
            );
        }
        if !root.join("complete.json").exists() {
            return Err(format!("incomplete run {path}").into());
        }
        let events: Vec<Value> = fs::read_to_string(root.join("events.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        runs.push((run, events));
    }
    if runs.len() < 2 {
        return Err("need mutable and at least one baseline run".into());
    }
    if runs
        .iter()
        .any(|run| run.0["mutable_construction"] != runs[0].0["mutable_construction"])
    {
        return Err("incompatible runs: mutable_construction".into());
    }
    if runs.iter().any(|r| !r.0["mixed"].is_null()) {
        return mixed::compare(&runs, floor, out);
    }
    let mutable = runs
        .iter()
        .find(|r| r.0["mode"] == "mutable")
        .ok_or("mutable run missing")?;
    let mut comparisons = Vec::new();
    for (run, events) in &runs {
        if run["mode"] == "mutable" {
            continue;
        }
        for key in [
            "planner",
            "planner_configuration",
            "mutable_filter",
            "mutable_filter_configuration",
            "fixture_sha256",
            "query_sha256",
            "limit",
            "dims",
            "seed",
            "threads",
            "insert_batch",
            "queries",
            "cpu_affinity",
            "m",
            "ef_construction",
            "growth",
            "churn",
            "churn_rounds",
            "requests",
            "repeats",
            "filters",
            "clients",
            "k",
            "ef",
        ] {
            if run[key] != mutable.0[key] {
                return Err(format!("incompatible runs: {key}").into());
            }
        }
        let mut groups: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
        for e in events.iter().filter(|e| e["event"] == "measurement") {
            let key = format!(
                "{}/{}/{}/{}/{}",
                e["stage"], e["filter"], e["k"], e["clients"], e["repeat"]
            );
            groups.entry(key).or_default().push(e);
        }
        for (key, rows) in groups {
            let baseline = rows
                .iter()
                .filter(|e| e["mean_recall"].as_f64().unwrap_or(0.) >= floor)
                .max_by(|a, b| {
                    a["qps"]
                        .as_f64()
                        .partial_cmp(&b["qps"].as_f64())
                        .unwrap_or(Ordering::Equal)
                });
            let Some(&b) = baseline else {
                comparisons.push(json!({"group":key,"baseline":run["mode"],"fanout":run["fanout"],"tail_scheduling":run["tail_scheduling"],"tail_chunk_rows":run["tail_chunk_rows"],"accepted":false,"reason":"baseline below recall floor"}));
                continue;
            };
            let target = floor.max(b["mean_recall"].as_f64().unwrap());
            let candidate = mutable
                .1
                .iter()
                .filter(|e| {
                    e["event"] == "measurement"
                        && ["stage", "filter", "k", "clients", "repeat"]
                            .iter()
                            .all(|f| e[*f] == b[*f])
                        && e["mean_recall"].as_f64().unwrap_or(0.) + 1e-12 >= target
                })
                .max_by(|a, b| {
                    a["qps"]
                        .as_f64()
                        .partial_cmp(&b["qps"].as_f64())
                        .unwrap_or(Ordering::Equal)
                });
            if let Some(c) = candidate {
                comparisons.push(json!({"group":key,"baseline":run["mode"],"fanout":run["fanout"],"tail_scheduling":run["tail_scheduling"],"tail_chunk_rows":run["tail_chunk_rows"],"matched_quality":true,"target_recall":target,"baseline_ef":b["ef"],"mutable_ef":c["ef"],"baseline_recall":b["mean_recall"],"mutable_recall":c["mean_recall"],"qps_speedup":c["qps"].as_f64().unwrap()/b["qps"].as_f64().unwrap(),"cpu_reduction_factor":b["cpu_ms_per_query"].as_f64().unwrap()/c["cpu_ms_per_query"].as_f64().unwrap(),"p99_speedup":b["p99_ms"].as_f64().unwrap()/c["p99_ms"].as_f64().unwrap()}));
            } else {
                comparisons.push(json!({"group":key,"baseline":run["mode"],"fanout":run["fanout"],"tail_scheduling":run["tail_scheduling"],"tail_chunk_rows":run["tail_chunk_rows"],"matched_quality":false,"target_recall":target,"reason":"no candidate ef meets baseline recall; no speedup accepted"}));
            }
        }
    }
    let planner_provenance = json!({
        "planner": mutable.0["planner"],
        "scope": mutable.0["planner_scope"],
        "configuration": mutable.0["planner_configuration"],
        "candidate_filter_first_capability": if mutable.0["mutable_filter"] == "vespa-first" {
            "pinned Vespa filter-first traversal enabled for filtered quiescent cases; All remains Plain; no Vespa planner selection or ES integration"
        } else { "Vespa filter-first is opt-in and disabled for this comparison; candidate uses exact-or-Plain in production mode and Plain in topology mode; candidate TwoHop remains absent" },
        "mutable_filter": mutable.0["mutable_filter"],
        "mutable_filter_configuration": mutable.0["mutable_filter_configuration"],
        "inputs": runs.iter().map(|(run, _)| json!({
            "run_json": Path::new(run["comparison_input_dir"].as_str().unwrap()).join("run.json"),
            "source_revision": run["source_revision"],
            "mode": run["mode"],
            "fanout": run["fanout"],
            "planner": run["planner"],
            "scope": run["planner_scope"],
            "configuration": run["planner_configuration"],
            "mutable_filter": run["mutable_filter"],
            "mutable_filter_configuration": run["mutable_filter_configuration"],
            "not_covered": run["not_covered"]
        })).collect::<Vec<_>>()
    });
    out.save("comparison.json",&json!({"recall_floor":floor,"selection":"fastest measured baseline meeting floor, then fastest mutable configuration meeting that baseline recall","planner_provenance":planner_provenance,"comparisons":comparisons,"caveat":"diagnostic matched-quality envelope; repeat close results, not statistical significance"}))?;
    out.emit(json!({"event":"comparison","groups":comparisons.len(),"path":out.dir.join("comparison.json")}))
}

fn run(args: &Args, out: &mut Output) -> AnyResult<()> {
    let mode = args.text("mode", "mutable");
    if mode == "compare" {
        return compare(args, out);
    }
    let vespa_planner = args.text("planner", "topology") == "vespa";
    let exact_diagnostic = args.text("planner", "topology") == "exact-diagnostic";
    let production_planner = match args.text("planner", "topology") {
        "topology" => false,
        "production" | "exact-diagnostic" | "vespa" => true,
        _ => return Err("--planner expects topology, production, exact-diagnostic or vespa".into()),
    };
    let vespa_filter_first = match args.text("mutable-filter", "plain") {
        "plain" => false,
        "vespa-first" => true,
        _ => return Err("--mutable-filter expects plain or vespa-first".into()),
    };
    if vespa_filter_first && (production_planner || args.text("mixed", "false") != "false") {
        return Err("--mutable-filter vespa-first supports only topology/quiescent mode; it is not a Vespa planner or mixed-load implementation".into());
    }
    if production_planner && args.text("mixed", "false") != "false" {
        return Err("production planner mode currently supports quiescent stages only; mixed publication-aware prepared filters are not implemented".into());
    }
    if (exact_diagnostic || vespa_planner) && mode != "mutable" {
        return Err("exact-diagnostic and vespa require mutable mode".into());
    }
    if !exact_diagnostic && !vespa_planner && args.values.contains_key("exact-query-threads") {
        return Err("exact-query-threads requires exact-diagnostic or vespa planner".into());
    }
    if args.values.contains_key("vespa-exact-lower-limit") && !vespa_planner {
        return Err("vespa-exact-lower-limit requires --planner vespa".into());
    }
    correctness(out)?;
    if mode == "correctness" {
        return Ok(());
    }
    if !["mutable", "segmented", "single"].contains(&mode) {
        return Err("invalid mode".into());
    }
    let root = Path::new(args.text("fixture", ""));
    let limit = args.usize("limit", 100_000)?;
    let (mut corpus, manifest) = Corpus::load(root, limit)?;
    let mut queries: Vec<Vec<f32>> = serde_json::from_slice(&fs::read(root.join("queries.json"))?)?;
    queries.truncate(args.usize("queries", 20)?);
    if queries.is_empty()
        || queries
            .iter()
            .any(|q| q.len() != corpus.dims || q.iter().any(|v| !v.is_finite()))
    {
        return Err("invalid/nonempty query dimensions required".into());
    }
    let clients = args.list("clients", "1,2,4,8,16,32")?;
    let efs = args.list("ef", "100,200,400,1000")?;
    let ks = args.list("k", "10,100")?;
    if vespa_planner && ks.iter().chain(&efs).any(|&n| n == 0 || n > u32::MAX as usize) {
        return Err("vespa k and ef must fit positive u32 budgets".into());
    }
    let requests = args.usize("requests", 128)?;
    let repeats = args.usize("repeats", 1)?;
    if requests < queries.len() || requests < *clients.iter().max().unwrap() || repeats == 0 {
        return Err("requests must cover every query/client and repeats must be positive".into());
    }
    let filters: Vec<_> = args
        .text("filters", "all")
        .split(',')
        .map(Filter::parse)
        .collect::<AnyResult<_>>()?;
    let chunks = args.usize("chunks", 12)?;
    let tail_fraction = args.text("tail-fraction", "0.12").parse::<f64>()?;
    let batch = args.usize("insert-batch", 64)?;
    let growth = args.usize("growth", 1000)?;
    let churn = args.usize("churn", 1000)?;
    let rounds = args.usize("churn-rounds", 2)?;
    let checkpoint_path = args.values.get("mutable-checkpoint");
    let checkpoint_growth = match args.text("checkpoint-growth", "false") {
        "true" => true, "false" => false,
        _ => return Err("checkpoint-growth expects true or false".into()),
    };
    validate_checkpoint_workload(checkpoint_path.is_some(), checkpoint_growth, mode,
        !production_planner || exact_diagnostic || vespa_planner,
        args.text("mixed", "false") != "false", growth, churn, rounds)?;
    let max_keys = limit
        .checked_add(growth)
        .and_then(|n| churn.checked_mul(rounds).and_then(|c| n.checked_add(c)));
    if max_keys.is_none_or(|n| n > u32::MAX as usize) {
        return Err("key count exceeds baseline u32 ordinals".into());
    }
    if chunks == 0 || batch == 0 || !(0.0..1.0).contains(&tail_fraction) || churn > limit {
        return Err("invalid chunks/batch/tail/churn".into());
    }
    let threads = args.usize("threads", std::thread::available_parallelism()?.get())?;
    let params = HnswParams {
        m: args.usize("m", 16)?,
        ef_construction: args.usize("ef-construction", 100)?,
        seed: args.text("seed", "11400714819323198485").parse()?,
        threads,
    };
    if params.m == 0 || params.ef_construction == 0 || threads == 0 {
        return Err("positive threads/m/construction required".into());
    }
    let fanout = args.text("fanout", if production_planner { "parallel" } else { "serial" });
    if !["serial", "parallel"].contains(&fanout) {
        return Err("invalid fanout".into());
    }
    if production_planner && fanout != "parallel" {
        return Err("production planner mode requires parallel dedicated-pool scheduling; use topology mode for the serial control".into());
    }
    let planner_settings = if vespa_planner { Some(planner::Settings::vespa_defaults(threads, args.usize("exact-query-threads", 1)?)?.with_exact_lower_limit(args.text("vespa-exact-lower-limit", "0.02").parse()?)?) } else if exact_diagnostic { Some(planner::Settings::exact_diagnostic(threads, args.usize("exact-query-threads", 1)?)?) } else if production_planner { Some(planner::Settings::initialize(threads)?) } else { None };
    let mut provenance = json!({"planner":"plain_hnsw_topology_control","planner_scope":"immutable graph search uses Plain strategy; bypasses production exact/TwoHop PlannerConfig thresholds; mutable also has no selective exact fallback; topology diagnostic only, not production-planner speedup evidence","mode":mode,"fixture":root,"manifest":manifest,"fixture_sha256":hash(&corpus.data),"query_sha256":hash(&serde_json::to_vec(&queries)?),"limit":limit,"dims":corpus.dims,"queries":queries.len(),"seed":params.seed,"m":params.m,"ef_construction":params.ef_construction,"threads":threads,"clients":clients,"ef":efs,"k":ks,"requests":requests,"repeats":repeats,"filters":args.text("filters","all"),"chunks":chunks,"tail_fraction":tail_fraction,"fanout":fanout,"tail_scheduling":if mode=="mutable"{"none"}else if fanout=="parallel"{"parallel_accepted_ordinals_shared_rayon_pool"}else{"serial_exact_scan"},"tail_chunk_rows":1024,"tail_parallel_threshold_accepted_rows":4096,"insert_batch":batch,"growth":growth,"churn":churn,"churn_rounds":rounds,"argv":std::env::args().collect::<Vec<_>>(),"source_revision":option_env!("RUSTIC_POC_SOURCE_REVISION"),"cpu_affinity":fs::read_to_string("/proc/self/status").unwrap_or_default().lines().find(|s|s.starts_with("Cpus_allowed_list:")).unwrap_or("").to_owned(),"memory_before":memory(),"semantics":"original f32 cosine, exact f64 truth; one process per mode; prefix native POC; query concurrency with quiescent mutation stages","not_covered":["quantized scoring","ES refresh or PIT","durable concurrent checkpoint/WAL","sustained simultaneous ingest and query"]});
    provenance["mutable_construction"] = json!({
        "reference_revision":"0521b00640ef934b359d2fc356f97d84499dd17d",
        "level_stream":"shared MT19937_64, fixed seed 0x1234deadbeef5678; observed libstdc++ uniform conversion; inverse-log and cap29",
        "bootstrap":"prepare-time marker below 10000 active nodes; one serialized preparation at completion; no all-stale retry",
        "harness_bootstrap_override":false,
        "seed_argument":"used by immutable controls; retained as mutable legacy codec metadata, not used by mutable level stream",
        "reopen":"RMHG4 canonical codec; explicit validated RMHG3 migration; topology and legacy metadata retained, newly constructed generator starts its fixed stream again; no continued RNG-state claim",
        "candidate_ordering":"pinned numeric distance-only comparator, including equal signed zero; Rust heap equal-distance order need not match C++",
        "proposal_ordering":"retain valid prepared neighbors in proposal order; no completion ID sort or dedup",
        "reconnect_ordering":"numeric pair-distance-only comparator, including equal signed zero; tied Rust sort order need not match C++",
        "plain_cutoff":"layer bound starts at maximum finite distance and tightens only after result-heap overflow; strict admission",
        "storage":"packed 32-bit level/link refs, dense node table, shared graph/preparation generation domain; acquired entry and candidate references",
        "remaining_fidelity_gaps":["Rust visited bitmap/hash storage layout differs from Vespa AllocatedBitVector/hash_set","ES compaction scheduling pending; native stage-boundary compaction wired","angular arithmetic opt-in; architecture-specific SIMD reduction","ordinary queries use current topology; retained capture has separate validation gate"]
    });
    provenance["mutable_visited"] = json!({
        "policy":"pinned Vespa layer-local adaptive bitmap/hash estimate and bitmap/hash threshold",
        "construction_filter":"none; each searched layer uses its own degree limit",
        "query_filter":"actual document-key universe and visible accepted count prepared outside clocks for Ten/Half; native All has no GlobalFilter metadata",
        "node_universe":"captured graph node-ID bound, separate from document keys under slot reuse",
        "compatibility_api":"closure-only core API has no filter statistics; it uses the unscaled visit estimate",
        "scope":"visited allocation/mark storage only; traversal and stopping rules unchanged"
    });
    provenance["mutable_filter"] = json!(if vespa_filter_first { "vespa-first" } else { "plain" });
    if vespa_filter_first {
        provenance["mutable_filter_configuration"] = json!({
            "reference_revision":"0521b00640ef934b359d2fc356f97d84499dd17d",
            "policy":"explicit traversal opt-in; no Vespa planner thresholds",
            "applies_to":"mutable filtered queries only; All and immutable controls retain Plain topology behavior",
            "filter_statistics":"actual document ID limit and visible accepted count precomputed once per stage/filter outside query clocks",
            "exploration":0.01,"exploration_slack":0.0,"prefetch_tensors":false,"deadline":null,
            "visited":"Vespa visit estimate and adaptive bitmap/hash policy; Rust allocation/hash layout differs",
            "scope":"single-vector original-f32 native topology experiment; not full ES, Vespa metric/construction equivalence, planner selection or sustainable mixed-load acceptance"
        });
        provenance["planner_scope"] = json!("native topology control: immutable Plain ANN bypasses production exact/TwoHop thresholds; mutable filtered queries explicitly use pinned Vespa filter-first, All remains Plain; no Vespa planner thresholds or production speedup claim");
    }
    if let Some(settings) = &planner_settings {
        provenance["planner"] = json!("production_thresholds_precomputed_ordinals");
        provenance["planner_scope"] = json!("production ordinal-filter routing and dedicated query pool; baseline exact/Plain/TwoHop, candidate exact/Plain with missing TwoHop explicit; filter evaluation/translation excluded; not full ES or sustainable concurrency acceptance");
        provenance["planner_configuration"] = settings.manifest();
        if exact_diagnostic {
            provenance["planner"] = json!("explicit_exact_static_docid_partition_diagnostic");
            provenance["planner_scope"] = json!("explicit shard-wide exact route with precomputed accepted rows; static document-range partitioning and bounded tasks per query; shared Rayon pool; no Vespa adaptive scheduler, thread-bundle admission, automatic policy selection or ES integration claim");
        }
        if vespa_planner {
            provenance["planner"] = json!("native_vespa_default_policy");
            provenance["planner_scope"] = json!("pinned Vespa default routing in quiescent native harness; fixture nominal estimates; prepared live filter counts; static exact partitions or single-thread ANN; no exact-fill fallback, ES integration or thread-bundle admission claim");
        }
        provenance["tail_scheduling"] = json!("production_exact_kernel_dedicated_query_pool");
        provenance["not_covered"] = json!(["candidate TwoHop traversal","quantized scoring","payload overlays","filter evaluation and owner-doc translation","ES refresh or PIT","durable concurrent checkpoint/WAL","sustained simultaneous ingest and query"]);
    }
    let imported = if let Some(path) = checkpoint_path {
        let start = Instant::now();
        let (graph, mut report) = checkpoint::load(Path::new(path), &provenance)?;
        report["load_and_verify_seconds"] = json!(start.elapsed().as_secs_f64());
        report["row_restore_source"] = json!(if Path::new(path).join("mutable-rows.bin").try_exists()? { "saved_owned_rows" } else { "legacy_fixture_rows" });
        provenance["mutable_checkpoint"] = report;
        provenance["checkpoint_growth"] = json!({"enabled":checkpoint_growth,
            "scope":"import never modifies source files; opt-in grows a private in-memory graph with deterministic synthetic variants, no churn, fresh level RNG after reopen; native insertion profiling only"});
        Some(graph)
    } else { None };
    provenance["angular_arithmetic"] = json!(if cfg!(feature = "mutable-hnsw-vespa-angular") {
        "experimental f32 SIMD products with f64 normalization; ES scores unchanged"
    } else { "legacy f32 normalization control; ES scores unchanged" });
    provenance["mutable_source"] = json!(if mode == "mutable" {
        "generation_owned_rows_atomic_current_directory"
    } else { "immutable_fixture_control" });
    out.save("run.json", &provenance)?;
    out.emit(json!({"event":"configuration","run":provenance}))?;
    let mut visible = vec![true; limit];
    let started = Instant::now();
    let cpu = cpu_seconds();
    let mut engine = if let Some(graph) = imported {
        let started = Instant::now();
        let cpu = cpu_seconds();
        let (graph, owner, ordinals) = checkpoint::restore_owned_run(Path::new(checkpoint_path.unwrap()), graph, &corpus, limit)?;
        out.emit(json!({"event":"owned_source_restore","rows":limit,
            "elapsed_seconds":started.elapsed().as_secs_f64(),"process_cpu_seconds":cpu_seconds()-cpu,
            "scope":"restore owned dense rows and source-key handle map without graph reinsertion; not ES durability/recovery"}))?;
        Engine::Mutable { graph, owner: Some(Arc::new(owner)), ordinals }
    } else if mode == "mutable" {
        let graph = Arc::new(MutableHnsw::new(corpus.dims as u32, Similarity::Cosine, params));
        let owner = MutableVectorOwner::new(graph.clone())?;
        let mut ordinals = Vec::new();
        insert_keys(
            &graph,
            Some(&owner),
            &corpus,
            &(0..limit as u64).collect::<Vec<_>>(),
            batch,
            &mut ordinals,
            out,
            "initial",
        )?;
        Engine::Mutable { graph, owner: Some(Arc::new(owner)), ordinals }
    } else {
        let graph_rows = if mode == "single" {
            limit
        } else {
            limit - ((limit as f64 * tail_fraction).round() as usize)
        };
        let count = if mode == "single" {
            1
        } else {
            chunks.min(graph_rows)
        };
        let step = graph_rows.div_ceil(count);
        let mut parts = Vec::new();
        for start in (0..graph_rows).step_by(step) {
            let end = (start + step).min(graph_rows);
            let t = Instant::now();
            let source = RangeSource {
                corpus: &corpus,
                start,
                end,
            };
            let graph = HnswBuilder::build(
                corpus.dims as u32,
                &source,
                (end - start) as u32,
                Similarity::Cosine,
                params,
            );
            out.emit(json!({"event":"immutable_build_part","start":start,"end":end,"elapsed_seconds":t.elapsed().as_secs_f64(),"nodes":graph.num_nodes(),"layers":graph.num_layers()}))?;
            parts.push(Part { start, end, graph });
        }
        Engine::Immutable {
            parts,
            tail: graph_rows,
            parallel: fanout == "parallel",
        }
    };
    if let Engine::Mutable { graph, owner, .. } = &engine {
        drain_graph_reclamation(graph, owner.as_deref(), "initial", out)?;
    }
    out.emit(json!({"event":"build","kind":if checkpoint_path.is_some(){"read_only_checkpoint_already_loaded"}else{"constructed"},"elapsed_seconds":started.elapsed().as_secs_f64(),"process_cpu_seconds":cpu_seconds()-cpu,"graph":engine.stats(&corpus),"memory":memory()}))?;
    match args.text("mixed", "false") {
        "true" => return mixed::run(args, &engine, &mut corpus, &queries, out),
        "false" => {}
        _ => return Err("--mixed expects true or false".into()),
    }
    let quality = quality_stage(
        &engine, &corpus, &visible, &queries, &ks, &efs, &filters, "initial", planner_settings.as_ref(), vespa_filter_first, out,
    )?;
    for repeat in 0..repeats {
        for q in &quality {
            for &c in &clients {
                measure(
                    &engine, &corpus, &visible, &queries, q, c, requests, repeat, "initial", out,
                )?;
            }
        }
    }
    let mut current: Vec<u64> = (0..limit as u64).collect();
    for stage_no in 0..=rounds {
        let stage = if stage_no == 0 {
            "growth".to_owned()
        } else {
            format!("churn-{stage_no}")
        };
        let count = if stage_no == 0 { growth } else { churn };
        if count == 0 {
            continue;
        }
        let t = Instant::now();
        let cpu = cpu_seconds();
        let mut added = Vec::new();
        let mut removed = Vec::new();
        for i in 0..count {
            if stage_no > 0 {
                let slot = (i + stage_no * churn) % limit;
                let old = current[slot];
                visible[old as usize] = false;
                removed.push(old);
                let key = corpus.append_variant(slot, stage_no);
                current[slot] = key;
                added.push(key);
            } else {
                added.push(corpus.append_variant(i * 13, stage_no));
            }
            visible.push(true);
        }
        if let Engine::Mutable { graph, owner, ordinals } = &mut engine {
            for &key in &removed {
                let source = MutableSourceLease::new(owner.as_deref(), &corpus)?;
                if !graph.remove(ordinals[key as usize], &source)? {
                    return Err("live node removal rejected".into());
                }
            }
            drain_graph_reclamation(graph, owner.as_deref(), &format!("{stage}-after-removals"), out)?;
            insert_keys(graph, owner.as_deref(), &corpus, &added, batch, ordinals, out, &stage)?;
            drain_graph_reclamation(graph, owner.as_deref(), &format!("{stage}-after-inserts"), out)?;
        }
        out.emit(json!({"event":"mutation_stage","stage":stage,"inserted":added.len(),"removed":removed.len(),"live":visible.iter().filter(|v|**v).count(),"elapsed_seconds":t.elapsed().as_secs_f64(),"process_cpu_seconds":cpu_seconds()-cpu,"graph":engine.stats(&corpus),"memory":memory(),"baseline_policy":"immutable graph updates masked, incoming replacements exact-scanned"}))?;
        let quality = quality_stage(
            &engine, &corpus, &visible, &queries, &ks, &efs, &filters, &stage, planner_settings.as_ref(), vespa_filter_first, out,
        )?;
        // Lifecycle stages use the same query set but a shorter latency probe.
        // Full concurrent matrix remains above; no mixed-write claim is made.
        for q in quality
            .iter()
            .filter(|q| q.ef == *efs.iter().max().unwrap())
        {
            for &c in clients.iter().filter(|&&c| c == 1 || c == 8) {
                measure(
                    &engine, &corpus, &visible, &queries, q, c, requests, 0, &stage, out,
                )?;
            }
        }
    }
    if let Engine::Mutable { graph, owner, .. } = &engine {
        let t = Instant::now();
        let checkpoint = owner.as_ref().map(|owner| owner.capture_checkpoint()).transpose()?;
        let capture_seconds = t.elapsed().as_secs_f64();
        let legacy = checkpoint.is_none().then(|| graph.encode());
        let bytes = checkpoint.as_ref().map(|c| c.graph_bytes()).unwrap_or_else(|| legacy.as_ref().unwrap());
        fs::write(out.dir.join("mutable-graph.bin"), bytes)?;
        let source_bytes = if let Some(checkpoint) = &checkpoint {
            checkpoint.write_file(&rustic_core::codec::directory::MmapDirectory::open(&out.dir)?, "mutable-checkpoint.bin")?;
            let path = out.dir.join("mutable-rows.bin");
            let mut writer = BufWriter::new(File::create(&path)?);
            checkpoint.write_rows(&mut writer)?;
            writer.flush()?;
            Some(fs::metadata(path)?.len())
        } else { None };
        out.emit(json!({"event":"quiescent_checkpoint","bytes":bytes.len(),"sha256":hash(bytes),
            "owned_source_bytes":source_bytes,"capture_seconds":capture_seconds,
            "elapsed_seconds":t.elapsed().as_secs_f64(),"memory":memory(),
            "scope":"coherent graph and streamed source capture; not an ES document/WAL manifest commit"}))?;
    }
    Ok(())
}

fn main() -> AnyResult<()> {
    let args = Args::parse()?;
    let mut output = Output::new(args.text("output", ""))?;
    let threads = args.usize("threads", std::thread::available_parallelism()?.get())?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    match run(&args, &mut output) {
        Ok(()) => {
            output.save(
                "complete.json",
                &json!({"completed":true,"performance_gate":"Completion means experiment finished; matched-quality acceptance is in comparison.json","memory":memory()}),
            )?;
            Ok(())
        }
        Err(error) => {
            output.emit(json!({"event":"failure","error":error.to_string(),"memory":memory()}))?;
            Err(error)
        }
    }
}

#[cfg(test)]
mod quiescent_tests {
    use super::*;
    #[test]
    fn checkpoint_growth_requires_opt_in_and_rejects_unsupported_mutations() {
        assert!(validate_checkpoint_workload(true, false, "mutable", true, false, 0, 0, 0).is_ok());
        assert!(validate_checkpoint_workload(true, true, "mutable", true, false, 50000, 0, 0).is_ok());
        for (checkpoint, enabled, mode, planner, mixed, growth, churn, rounds) in [
            (true, false, "mutable", true, false, 1, 0, 0),
            (false, true, "mutable", true, false, 1, 0, 0),
            (true, true, "mutable", true, false, 0, 0, 0),
            (true, true, "segmented", true, false, 1, 0, 0),
            (true, true, "mutable", false, false, 1, 0, 0),
            (true, true, "mutable", true, true, 1, 0, 0),
            (true, true, "mutable", true, false, 1, 1, 0),
            (true, true, "mutable", true, false, 1, 0, 1),
        ] {
            assert!(validate_checkpoint_workload(checkpoint, enabled, mode, planner, mixed, growth, churn, rounds).is_err());
        }
    }

    #[test]
    fn prepared_filter_first_counts_follow_visibility_growth_and_churn() {
        let mut visible = vec![true; 30];
        assert!(prepared_mutable_filter(&visible, Filter::All, true).unwrap().is_none());
        assert!(prepared_mutable_filter(&visible, Filter::Ten, false).unwrap().is_none());
        let initial = prepared_mutable_filter(&visible, Filter::Ten, true).unwrap().unwrap();
        assert_eq!(initial.filter_estimate.unwrap().size, 30);
        assert_eq!(initial.filter_estimate.unwrap().count, 3);
        visible[10] = false;
        let deleted = prepared_mutable_filter(&visible, Filter::Ten, true).unwrap().unwrap();
        assert_eq!(deleted.filter_estimate.unwrap().count, 2);
        visible.push(true); // New incarnation 30; old incarnation 10 stays hidden.
        let grown = prepared_mutable_filter(&visible, Filter::Ten, true).unwrap().unwrap();
        assert_eq!(grown.filter_estimate.unwrap().size, 31);
        assert_eq!(grown.filter_estimate.unwrap().count, 3);
        assert_eq!(initial.filter_estimate.unwrap().size, 30);
        let half = prepared_mutable_filter(&visible, Filter::Half, true).unwrap().unwrap();
        assert_eq!(half.filter_estimate.unwrap().count, 15);
        visible.fill(false);
        assert_eq!(prepared_mutable_filter(&visible, Filter::Ten, true).unwrap().unwrap().filter_estimate.unwrap().count, 0);
    }

    #[test]
    fn plain_filter_metadata_uses_document_keys_and_actual_visibility() {
        let mut visible = vec![false; 101];
        for key in [0, 10, 19, 20, 98, 100] {
            visible[key] = true;
        }
        let estimate = prepared_filter_estimate(&visible, Filter::Ten)
            .unwrap()
            .unwrap();
        assert_eq!(estimate.size, 101); // Six live vectors can occupy six reused slots.
        assert_eq!(estimate.count, 4);
        let half = prepared_filter_estimate(&visible, Filter::Half)
            .unwrap()
            .unwrap();
        assert_eq!(half.count, 4);
        visible[100] = false;
        assert_eq!(
            prepared_filter_estimate(&visible, Filter::Ten)
                .unwrap()
                .unwrap()
                .count,
            3
        );
        assert_eq!(estimate.count, 4, "prior stage metadata stays immutable");
        assert!(
            prepared_filter_estimate(&visible, Filter::All)
                .unwrap()
                .is_none()
        );
        visible.fill(false);
        assert_eq!(
            prepared_filter_estimate(&visible, Filter::Ten)
                .unwrap()
                .unwrap()
                .count,
            0
        );
    }

    #[test]
    fn quiescent_hit_validation_rejects_nonfinite_unordered_and_invalid_hits() {
        let valid = [(7, 0.75), (9, 0.5)];
        let validate = |hits: &[(u64, f32)]| {
            validate_quiescent_hits(hits, 2, |key| key == 7 || key == 9, |key| {
                Ok(if key == 7 { 0.75 } else { 0.5 })
            })
        };
        assert!(validate(&valid).is_ok());
        for invalid in [
            vec![(7, f32::NAN), (9, 0.5)],
            vec![(7, f32::INFINITY), (9, 0.5)],
            vec![(9, 0.5), (7, 0.75)],
            vec![(7, 0.75), (7, 0.75)],
            vec![(7, 0.75), (11, 0.5)],
            vec![(7, 0.75)],
            vec![(7, 0.7), (9, 0.5)],
        ] {
            assert!(validate(&invalid).is_err(), "accepted {invalid:?}");
        }
        assert!(validate_quiescent_hits(&valid, 2, |_| true, |_| Ok(f64::NAN)).is_err());
        assert_ne!(hit_fingerprint(&valid), hit_fingerprint(&[(9, 0.5), (7, 0.75)]));
        assert_ne!(hit_fingerprint(&valid), hit_fingerprint(&[(7, 0.75), (9, 0.50001)]));
    }

    #[test]
    fn quiescent_timed_validation_rejects_fallback_coverage_and_changed_hits() {
        let references = [hit_fingerprint(&[(7, 0.75)]), hit_fingerprint(&[(9, 0.5)])];
        let samples: Vec<_> = (0..4).map(|sequence| TimedSample {
            sequence,
            query: sequence % 2,
            latency_ms: 1.,
            vector_ops: 5,
            fallback: 0,
            missing_source: 0,
            fingerprint: references[sequence % 2],
            planner_stats: planner::QueryStats::default(),
        }).collect();
        assert!(validate_timed_samples(&samples, 4, &references).is_ok());
        assert!(validate_timed_samples(&samples[..3], 4, &references).is_err());
        assert!(validate_timed_samples(&samples, 4, &[]).is_err());
        let mut invalid = samples.clone();
        invalid[2].sequence = 1;
        assert!(validate_timed_samples(&invalid, 4, &references).is_err());
        invalid = samples.clone();
        invalid[2].query = 1;
        assert!(validate_timed_samples(&invalid, 4, &references).is_err());
        invalid = samples.clone();
        invalid[2].fallback = 1;
        assert!(validate_timed_samples(&invalid, 4, &references).is_err());
        invalid = samples.clone();
        invalid[2].missing_source = 1;
        assert!(validate_timed_samples(&invalid, 4, &references).is_err());
        invalid = samples.clone();
        invalid[2].fingerprint = hit_fingerprint(&[(7, 0.7)]);
        assert!(validate_timed_samples(&invalid, 4, &references).is_err());
        for latency in [f64::NAN, f64::INFINITY, -1.] {
            invalid = samples.clone();
            invalid[2].latency_ms = latency;
            assert!(validate_timed_samples(&invalid, 4, &references).is_err());
        }
    }

    #[test]
    fn quiescent_workers_cancel_warmup_and_measured_panics() {
        // The coordinator lives in a detached thread so a future rendezvous
        // regression fails by deadline instead of hanging the whole test run.
        let (sent, received) = std::sync::mpsc::channel();
        let coordinator = std::thread::spawn(move || {
            for fail_during_warmup in [true, false] {
                let ready = Barrier::new(3);
                let start = Barrier::new(3);
                let cancelled = AtomicBool::new(false);
                let outcomes = std::thread::scope(|scope| {
                    let workers: Vec<_> = (0..2).map(|id| {
                        let ready = &ready;
                        let start = &start;
                        let cancelled = &cancelled;
                        scope.spawn(move || quiescent_worker(
                            ready, start, cancelled,
                            || {
                                if id == 0 && fail_during_warmup { panic!("synthetic warmup failure"); }
                                Ok(())
                            },
                            || {
                                if id == 0 && !fail_during_warmup { panic!("synthetic measured failure"); }
                                Ok(())
                            },
                        ))
                    }).collect();
                    ready.wait();
                    start.wait();
                    workers.into_iter().map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
                });
                assert!(cancelled.load(Acquire));
                assert!(outcomes.iter().any(Result::is_err));
                if fail_during_warmup {
                    assert!(outcomes.iter().all(Result::is_err));
                }
            }
            sent.send(()).unwrap();
        });
        received.recv_timeout(std::time::Duration::from_secs(10)).expect("quiescent worker cancellation stranded a rendezvous");
        coordinator.join().unwrap();
    }
}
