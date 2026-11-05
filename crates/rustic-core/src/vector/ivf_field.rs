//! The DiskBBQ index of one `bbq_disk` field of a shard: **one** set of
//! centroids and posting lists shared by every vector segment of the field
//! (as the field's mutable HNSW graph is shared), where ES keeps one per
//! Lucene segment. Clustering, spill, codes and the visit budget are ES's
//! (see [`super::ivf`]).
//!
//! **Entries are point ids.** A posting list entry names a stable
//! [`super::store::PointId`] (bit 63: the vector's spilled SOAR copy), so
//! payload compactions, vacuums and text merges, which relocate rows but
//! keep point ids, never touch the index, and merges never re-cluster.
//! Liveness is the store's point table: a deleted (or replaced) document's
//! point is dead there, and its entries are skipped and later compacted
//! away; retained views read their captured table.
//!
//! **Files** (`ivf_FFFF_NNNNNNNNNNNNNNNN.*`, `F` the field, `N` a per-field
//! counter):
//!
//! * `.rivc`, the centroids: read into the heap at open (charged to
//!   `vector_graphs`) with each centroid's generation (high bit: dropped,
//!   its slot reused by a later split with a newer generation);
//! * `.rivp`, an immutable posting batch: runs of entries for
//!   `(centroid, generation)`, each its point ids, OSQ codes relative to
//!   that centroid and corrective terms. Batches stay in the mapping; only
//!   their run table (24 bytes a run) is resident.
//!
//! A centroid's posting list is its runs of its current generation across
//! the batches. The commit point lists the centroid file and batches of
//! each field ([`IvfFieldManifest`], in `meta.json`); file-based recovery
//! ships them like the graph checkpoints (`transfer.rs`).
//!
//! **Writes are incremental** ([`super::store::VectorStore`] maintenance,
//! on the merge pool):
//!
//! * a field below its `flat_index_threshold` has no index and is searched
//!   flat; the first pass above it trains centroids (hierarchical k-means
//!   with SOAR) on the field's rows;
//! * every later pass assigns the rows not yet indexed (fresh chunks, rows
//!   found at open past the checkpoint) to their nearest centroid with a
//!   SOAR spill and appends them as one batch;
//! * a cluster with more than twice `cluster_size` live vectors is split
//!   (`HierarchicalKMeans.clusterAndSplit` on its vectors, new
//!   generations); one with fewer than `cluster_size / 16` is dropped, its
//!   vectors moved to their nearest other centroids;
//! * batches are compacted (dead entries and superseded runs dropped, codes
//!   copied verbatim) when there are more than [`MAX_BATCHES`] or 20% of
//!   their entries are garbage.
//!
//! Each step writes new files and installs a new immutable [`IvfState`];
//! searches and retained views (PIT, scroll) keep the state they captured,
//! whose mapped files stay readable after a commit deletes them.
//!
//! Layout (little-endian; `.rivc` and `.rivp` end with `u32 crc32, magic`):
//!
//! ```text
//! .rivc: "RIVC" u32 version, u16 field, u8 bits, u8 query_bits, u8 sim (0 l2, 1 dot, 2 mip),
//!        u8 flags (1 unit, 2 preconditioned), u16 0, u32 dim, u32 k, u64 preconditioner_len,
//!        [u32 generation] x k, pad8, [f32 x dim] x k, preconditioner
//! .rivp: "RIVP" u32 version, u16 field, u8 bits, u8 query_bits, u32 dim, u32 runs, u64 entries, u32 0,
//!        (u32 centroid, u32 generation, u64 offset, u32 len, u32 0) x runs, pad8,
//!        runs: [u64 entry] x len, [code] x len, [f32 lower, f32 upper, f32 additional, i32 sum] x len
//! ```

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::distance::{dot, l2_sq};
use super::ivf::{self, IvfOptions, Points, Preconditioner, Scratch, CORRECTION_BYTES, NO_SPILL};
use super::osq::{OsqCorrections, OsqQuantizer, OsqSim};
use super::Similarity;
use crate::accounting::{Category, Charge};
use crate::codec::directory::{Directory, FileSlice};
use crate::error::{Result, RusticError};

/// Posting entry flag: the vector's spilled (SOAR) copy.
pub(crate) const SPILL: u64 = 1 << 63;
/// Centroid generation flag: the slot is dropped (its last generation kept
/// below, so a reused slot gets a newer one than any run written for it).
const DROPPED: u32 = 1 << 31;

/// Whether a centroid generation is in use (runs carry only these).
fn live_gen(g: u32) -> bool {
    g != 0 && g & DROPPED == 0
}
const CENTROIDS_MAGIC: &[u8; 4] = b"RIVC";
const BATCH_MAGIC: &[u8; 4] = b"RIVP";
/// Version of the `.rivc` / `.rivp` files.
pub const IVF_FILES_VERSION: u32 = 1;
const FLAG_UNIT: u8 = 1;
const FLAG_PRECONDITIONED: u8 = 2;
const CENTROIDS_HEADER: usize = 32;
const BATCH_HEADER: usize = 32;
const RUN_BYTES: usize = 24;
/// Posting batches above which maintenance compacts.
pub const MAX_BATCHES: usize = 8;
/// Dead or superseded entry fraction of the batches that triggers a
/// compaction.
const GARBAGE_RATIO: f32 = 0.2;
/// A cluster splits above this many times `cluster_size` live vectors.
const SPLIT_FACTOR: usize = 2;
/// A cluster is dropped below `cluster_size / DROP_DIVISOR` live vectors.
const DROP_DIVISOR: usize = 16;
/// Lists a wave takes when the accept rate is not known yet (no entry
/// accepted so far).
const WAVE_LISTS: usize = 8;
/// Entries a wave past the visit budget takes at most (the accept rate it
/// is sized by is still changing).
const WAVE_ENTRIES: u64 = 65_536;
/// Entries per worker above which a wave is searched in parallel.
const PARALLEL_ENTRIES: u64 = 4096;
/// Workers of one wave at most.
const MAX_WORKERS: usize = 8;
/// Posting bytes one compaction pass rewrites at most.
const COMPACT_BUDGET: usize = 512 << 20;

/// `.rivc` file of `field`'s centroids number `id`.
pub fn centroid_file_name(field: u16, id: u64) -> String {
    format!("ivf_{field:04x}_{id:016x}.rivc")
}

/// `.rivp` posting batch file of `field` number `id`.
pub fn batch_file_name(field: u16, id: u64) -> String {
    format!("ivf_{field:04x}_{id:016x}.rivp")
}

/// Whether `name` is a shared-IVF file.
pub fn is_ivf_file(name: &str) -> bool {
    name.starts_with("ivf_") && (name.ends_with(".rivc") || name.ends_with(".rivp"))
}

/// `(field, number)` of a shared-IVF file name.
pub fn parse_file_name(name: &str) -> Option<(u16, u64)> {
    let stem = name.strip_prefix("ivf_")?.strip_suffix(".rivc").or_else(|| name.strip_prefix("ivf_")?.strip_suffix(".rivp"))?;
    let (field, id) = stem.split_once('_')?;
    Some((u16::from_str_radix(field, 16).ok()?, u64::from_str_radix(id, 16).ok()?))
}

/// A field's index files at a commit point (`meta.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfFieldManifest {
    pub field: u16,
    /// The `.rivc` file number.
    pub centroids: u64,
    /// The `.rivp` file numbers.
    pub batches: Vec<u64>,
}

impl IvfFieldManifest {
    /// Every file this manifest references.
    pub fn files(&self) -> Vec<String> {
        std::iter::once(centroid_file_name(self.field, self.centroids)).chain(self.batches.iter().map(|&b| batch_file_name(self.field, b))).collect()
    }
}

/// What a field's codes were built with: a mapping change of any of these
/// rebuilds the index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BuildParams {
    pub dim: usize,
    pub sim: OsqSim,
    pub unit: bool,
    pub bits: u8,
    pub query_bits: u8,
    pub precondition: bool,
}

impl BuildParams {
    pub(crate) fn of(similarity: Similarity, dim: usize, options: &IvfOptions) -> Self {
        BuildParams {
            dim,
            sim: OsqSim::of(similarity),
            unit: similarity == Similarity::Cosine,
            bits: options.doc_bits(),
            query_bits: options.query_bits(),
            precondition: options.precondition,
        }
    }

    fn flags(&self) -> u8 {
        (if self.unit { FLAG_UNIT } else { 0 }) | (if self.precondition { FLAG_PRECONDITIONED } else { 0 })
    }
}

fn rd_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn rd_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn rd_f32(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn bad(what: &str) -> RusticError {
    RusticError::Corrupted(format!("IVF index: {what}"))
}

fn seal(mut out: Vec<u8>, magic: &[u8; 4]) -> Vec<u8> {
    let crc = crate::codec::crc32(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(magic);
    out
}

// ---------------------------------------------------------------------------
// Centroids.

/// A field's centroids (resident, charged to `vector_graphs`).
pub struct Centroids {
    /// The `.rivc` file number.
    pub(crate) file: u64,
    /// The `.rivc` file length (ES `cenivf`), once written or read.
    file_len: usize,
    pub(crate) params: BuildParams,
    pre: Option<Preconditioner>,
    /// Per centroid its generation; `0`: dropped.
    gens: Vec<u32>,
    data: Vec<f32>,
    _charge: Charge,
    _memory: crate::admission::Reservation,
}

impl Centroids {
    fn new(file: u64, params: BuildParams, pre: Option<Preconditioner>, mut gens: Vec<u32>, mut data: Vec<f32>) -> Result<Self> {
        let bytes = data.capacity() * 4 + gens.capacity() * 4 + pre.as_ref().map_or(0, Preconditioner::heap_bytes);
        let memory = crate::admission::shared().reserve(bytes as u64, "IVF centroids")?;
        Ok(Centroids { file, file_len: 0, params, pre, gens, data, _charge: Charge::new(Category::VectorGraphs, bytes as u64), _memory: memory })
    }

    /// Centroid slots (dropped ones included).
    pub fn len(&self) -> usize {
        self.gens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gens.is_empty()
    }

    /// Centroids in use.
    pub fn active(&self) -> usize {
        self.gens.iter().filter(|&&g| live_gen(g)).count()
    }

    fn vector(&self, c: usize) -> &[f32] {
        ivf::centroid(&self.data, self.params.dim, c)
    }

    /// Heap bytes held (charged).
    pub fn heap_bytes(&self) -> usize {
        self.data.len() * 4 + self.gens.len() * 4 + self.pre.as_ref().map_or(0, Preconditioner::heap_bytes)
    }

    fn encode(&self, field: u16) -> Vec<u8> {
        let p = &self.params;
        let mut pre = Vec::new();
        if let Some(x) = &self.pre {
            x.encode(&mut pre);
        }
        let mut out = Vec::with_capacity(CENTROIDS_HEADER + self.gens.len() * 4 + self.data.len() * 4 + pre.len() + 16);
        out.extend_from_slice(CENTROIDS_MAGIC);
        out.extend_from_slice(&IVF_FILES_VERSION.to_le_bytes());
        out.extend_from_slice(&field.to_le_bytes());
        out.extend_from_slice(&[p.bits, p.query_bits, ivf::sim_tag(p.sim), p.flags(), 0, 0]);
        out.extend_from_slice(&(p.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.gens.len() as u32).to_le_bytes());
        out.extend_from_slice(&(pre.len() as u64).to_le_bytes());
        debug_assert_eq!(out.len(), CENTROIDS_HEADER);
        for g in &self.gens {
            out.extend_from_slice(&g.to_le_bytes());
        }
        ivf::pad8(&mut out);
        for x in &self.data {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&pre);
        seal(out, CENTROIDS_MAGIC)
    }

    /// `None`: a version this build does not know (the index is rebuilt).
    fn decode(field: u16, file: u64, b: &[u8]) -> Result<Option<Self>> {
        if b.len() < CENTROIDS_HEADER + 8 || &b[..4] != CENTROIDS_MAGIC || &b[b.len() - 4..] != CENTROIDS_MAGIC {
            return Err(bad("centroid file magic"));
        }
        if rd_u32(b, 4) != IVF_FILES_VERSION {
            return Ok(None);
        }
        let body = b.len() - 8;
        if crate::codec::crc32(&b[..body]) != rd_u32(b, body) {
            return Err(bad("centroid file checksum"));
        }
        if rd_u16(b, 8) != field {
            return Err(bad("centroid file field"));
        }
        let (bits, query_bits, sim, flags) = (b[10], b[11], ivf::sim_from_tag(b[12]).ok_or_else(|| bad("similarity"))?, b[13]);
        let dim = rd_u32(b, 16) as usize;
        let k = rd_u32(b, 20) as usize;
        let pre_len = usize::try_from(rd_u64(b, 24)).map_err(|_| bad("centroid file size"))?;
        if !matches!(bits, 1 | 2 | 4 | 7) || query_bits != (if bits == 7 { 7 } else { 4 }) || dim == 0 {
            return Err(bad("centroid parameters"));
        }
        let gens_at = CENTROIDS_HEADER;
        let data_at = k.checked_mul(4).and_then(|n| gens_at.checked_add(n))
            .and_then(|n| n.checked_add(7)).map(|n| n & !7).ok_or_else(|| bad("centroid file size"))?;
        let values = k.checked_mul(dim).ok_or_else(|| bad("centroid file size"))?;
        let pre_at = values.checked_mul(4).and_then(|n| data_at.checked_add(n)).ok_or_else(|| bad("centroid file size"))?;
        if pre_at.checked_add(pre_len).is_none_or(|end| end != body) {
            return Err(bad("centroid file size"));
        }
        let _decode_memory = crate::admission::shared().reserve((body as u64).saturating_mul(4), "IVF centroid decode")?;
        let gens = (0..k).map(|c| rd_u32(b, gens_at + c * 4)).collect();
        let data: Vec<f32> = (0..values).map(|i| rd_f32(b, data_at + i * 4)).collect();
        if data.iter().any(|v| !v.is_finite()) || flags & !(FLAG_UNIT | FLAG_PRECONDITIONED) != 0
            || (flags & FLAG_PRECONDITIONED == 0 && pre_len != 0) {
            return Err(bad("centroid values or flags"));
        }
        let params = BuildParams { dim, sim, unit: flags & FLAG_UNIT != 0, bits, query_bits, precondition: flags & FLAG_PRECONDITIONED != 0 };
        let pre = if params.precondition { Some(Preconditioner::decode(&b[pre_at..pre_at + pre_len], dim)?) } else { None };
        let mut c = Centroids::new(file, params, pre, gens, data)?;
        c.file_len = b.len();
        Ok(Some(c))
    }
}

// ---------------------------------------------------------------------------
// Posting batches.

#[derive(Clone, Copy, Debug)]
struct Run {
    centroid: u32,
    generation: u32,
    /// Byte offset of the run in the batch file.
    at: usize,
    len: u32,
}

/// One immutable `.rivp` file: posting runs in the mapping, the run table
/// on the heap (charged).
pub struct PostingBatch {
    pub(crate) id: u64,
    data: FileSlice,
    runs: Vec<Run>,
    entries: u64,
    _charge: Charge,
    _memory: crate::admission::Reservation,
}

impl PostingBatch {
    /// Entries of every run (superseded ones included).
    pub fn entries(&self) -> u64 {
        self.entries
    }

    /// Bytes of the file.
    pub fn file_bytes(&self) -> usize {
        self.data.len()
    }

    fn open(field: u16, id: u64, data: FileSlice, params: &BuildParams) -> Result<Self> {
        let b = data.as_slice();
        if b.len() < BATCH_HEADER + 8 || &b[..4] != BATCH_MAGIC || &b[b.len() - 4..] != BATCH_MAGIC {
            return Err(bad("batch magic"));
        }
        if rd_u32(b, 4) != IVF_FILES_VERSION {
            return Err(bad("batch version"));
        }
        if rd_u16(b, 8) != field || b[10] != params.bits || b[11] != params.query_bits || rd_u32(b, 12) as usize != params.dim {
            return Err(bad("batch parameters"));
        }
        let n = rd_u32(b, 16) as usize;
        let entries = rd_u64(b, 20);
        let body = b.len() - 8;
        if crate::codec::crc32(&b[..body]) != rd_u32(b, body) {
            return Err(bad("batch checksum"));
        }
        let table_end = n.checked_mul(RUN_BYTES).and_then(|n| BATCH_HEADER.checked_add(n))
            .ok_or_else(|| bad("batch run table"))?;
        if table_end > body {
            return Err(bad("batch run table"));
        }
        let memory = crate::admission::shared().reserve((n as u64).saturating_mul(std::mem::size_of::<Run>() as u64), "IVF posting run table")?;
        let mut runs = Vec::with_capacity(n);
        let mut total = 0u64;
        let mut previous_end = table_end;
        for r in 0..n {
            let at = BATCH_HEADER + r * RUN_BYTES;
            let run = Run { centroid: rd_u32(b, at), generation: rd_u32(b, at + 4), at: usize::try_from(rd_u64(b, at + 8)).map_err(|_| bad("batch run range"))?, len: rd_u32(b, at + 16) };
            let bytes_per_entry = 8usize.checked_add(ivf::code_bytes(params.bits, params.dim))
                .and_then(|n| n.checked_add(CORRECTION_BYTES)).ok_or_else(|| bad("batch run range"))?;
            let end = (run.len as usize).checked_mul(bytes_per_entry).and_then(|n| run.at.checked_add(n))
                .ok_or_else(|| bad("batch run range"))?;
            if run.at % 8 != 0 || run.at < previous_end || end > body {
                return Err(bad("batch run range"));
            }
            previous_end = end;
            total += u64::from(run.len);
            runs.push(run);
        }
        if total != entries {
            return Err(bad("batch entry count"));
        }
        let charge = Charge::new(Category::VectorGraphs, (runs.capacity() * std::mem::size_of::<Run>()) as u64);
        Ok(PostingBatch { id, data, runs, entries, _charge: charge, _memory: memory })
    }

    fn entry(&self, run: &Run, j: usize) -> u64 {
        rd_u64(self.data.as_slice(), run.at + j * 8)
    }

    /// The raw bytes of entry `j` of `run`: `(entry, code, corrections)`.
    fn parts(&self, run: &Run, j: usize, bits: u8, dim: usize) -> (u64, &[u8], &[u8]) {
        let b = self.data.as_slice();
        let len = run.len as usize;
        let cb = ivf::code_bytes(bits, dim);
        let codes_at = run.at + len * 8;
        let corr_at = codes_at + len * cb;
        (rd_u64(b, run.at + j * 8), &b[codes_at + j * cb..codes_at + (j + 1) * cb], &b[corr_at + j * CORRECTION_BYTES..corr_at + (j + 1) * CORRECTION_BYTES])
    }
}

/// One run being written: `(centroid, generation)` and its entries, codes
/// and corrections (`len` of each).
struct RunOut {
    centroid: u32,
    generation: u32,
    len: u32,
    entries: Vec<u8>,
    codes: Vec<u8>,
    corrections: Vec<u8>,
}

impl RunOut {
    fn new(centroid: u32, generation: u32) -> Self {
        RunOut { centroid, generation, len: 0, entries: Vec::new(), codes: Vec::new(), corrections: Vec::new() }
    }

    /// Sized for `n` entries of `code` bytes (no regrowth while encoding).
    fn with_capacity(centroid: u32, generation: u32, n: usize, code: usize) -> Self {
        RunOut { centroid, generation, len: 0, entries: Vec::with_capacity(n * 8), codes: Vec::with_capacity(n * code), corrections: Vec::with_capacity(n * CORRECTION_BYTES) }
    }

    fn push_raw(&mut self, entry: u64, code: &[u8], corrections: &[u8]) {
        self.entries.extend_from_slice(&entry.to_le_bytes());
        self.codes.extend_from_slice(code);
        self.corrections.extend_from_slice(corrections);
        self.len += 1;
    }
}

/// Quantize `rows` (`(entry, vector as indexed)`) relative to centroid `c`.
fn encode_run(centroids: &Centroids, c: u32, rows: &[(u64, &[f32])]) -> RunOut {
    let p = &centroids.params;
    let quantizer = OsqQuantizer::from_centroid(p.sim, centroids.vector(c as usize).to_vec(), p.bits, p.query_bits);
    let mut out = RunOut::with_capacity(c, centroids.gens[c as usize], rows.len(), ivf::code_bytes(p.bits, p.dim));
    let mut s = Scratch::default();
    let mut codes = Vec::with_capacity(p.dim);
    let mut packed = Vec::with_capacity(ivf::code_bytes(p.bits, p.dim));
    for &(entry, row) in rows {
        let v = ivf::prepare(row, p.unit, centroids.pre.as_ref(), &mut s);
        let corr = quantizer.quantize_doc(v, &mut codes);
        packed.clear();
        ivf::pack_codes(p.bits, &codes, p.dim, &mut packed);
        let mut c16 = [0u8; CORRECTION_BYTES];
        c16[..4].copy_from_slice(&corr.lower.to_le_bytes());
        c16[4..8].copy_from_slice(&corr.upper.to_le_bytes());
        c16[8..12].copy_from_slice(&corr.additional.to_le_bytes());
        c16[12..].copy_from_slice(&corr.component_sum.to_le_bytes());
        out.push_raw(entry, &packed, &c16);
    }
    out
}

fn encode_batch(field: u16, params: &BuildParams, runs: &[RunOut]) -> Vec<u8> {
    let entries: u64 = runs.iter().map(|r| u64::from(r.len)).sum();
    let mut out = Vec::with_capacity(BATCH_HEADER + runs.len() * RUN_BYTES + runs.iter().map(|r| r.entries.len() + r.codes.len() + r.corrections.len()).sum::<usize>() + 16);
    out.extend_from_slice(BATCH_MAGIC);
    out.extend_from_slice(&IVF_FILES_VERSION.to_le_bytes());
    out.extend_from_slice(&field.to_le_bytes());
    out.push(params.bits);
    out.push(params.query_bits);
    out.extend_from_slice(&(params.dim as u32).to_le_bytes());
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    out.extend_from_slice(&entries.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), BATCH_HEADER);
    let table_at = out.len();
    out.resize(table_at + runs.len() * RUN_BYTES, 0);
    ivf::pad8(&mut out);
    for (i, r) in runs.iter().enumerate() {
        ivf::pad8(&mut out);
        let at = out.len();
        out.extend_from_slice(&r.entries);
        out.extend_from_slice(&r.codes);
        out.extend_from_slice(&r.corrections);
        let t = table_at + i * RUN_BYTES;
        out[t..t + 4].copy_from_slice(&r.centroid.to_le_bytes());
        out[t + 4..t + 8].copy_from_slice(&r.generation.to_le_bytes());
        out[t + 8..t + 16].copy_from_slice(&(at as u64).to_le_bytes());
        out[t + 16..t + 20].copy_from_slice(&r.len.to_le_bytes());
    }
    ivf::pad8(&mut out);
    seal(out, BATCH_MAGIC)
}

// ---------------------------------------------------------------------------
// Indexed points.

/// Bytes [`Indexed`] accounts for each 64-point word it holds: a
/// `BTreeMap<u64, u64>` entry is 16 bytes of key and value in leaves of up
/// to 11 entries that ascending inserts leave about half full, plus node
/// headers, parent links and internal nodes (about 40 bytes in all); 64
/// rounds that up. About one byte a point for dense point ids.
const INDEXED_WORD_BYTES: u64 = 64;

/// The points the index holds entries for (one bit a point, charged):
/// what an insert pass must still add.
pub struct Indexed {
    words: BTreeMap<u64, u64>,
    count: u64,
    _charge: Charge,
    _memory: crate::admission::Reservation,
}

impl Indexed {
    fn empty() -> Result<Self> {
        Ok(Self { words: BTreeMap::new(), count: 0, _charge: Charge::new(Category::VectorGraphs, 0),
            _memory: crate::admission::shared().reserve(0, "IVF indexed points")? })
    }

    fn insert(&mut self, p: u64) -> Result<()> {
        let p = p & !SPILL;
        if !self.words.contains_key(&(p >> 6)) {
            self._memory.grow(INDEXED_WORD_BYTES, "IVF indexed points")?;
        }
        *self.words.entry(p >> 6).or_default() |= 1 << (p & 63);
        Ok(())
    }

    fn finish(&mut self) {
        self.count = self.words.values().map(|w| u64::from(w.count_ones())).sum();
        self._charge = Charge::new(Category::VectorGraphs, (self.words.len() as u64).saturating_mul(INDEXED_WORD_BYTES));
    }

    pub fn contains(&self, p: u64) -> bool {
        let p = p & !SPILL;
        self.words.get(&(p >> 6)).is_some_and(|w| w >> (p & 63) & 1 == 1)
    }

    /// Points indexed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Bytes admitted for the set.
    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> u64 {
        self._memory.bytes()
    }

    pub(crate) fn with(&self, points: impl IntoIterator<Item = u64>) -> Result<Self> {
        let memory = crate::admission::shared().reserve((self.words.len() as u64).saturating_mul(INDEXED_WORD_BYTES), "IVF indexed copy")?;
        let mut next = Self { words: self.words.clone(), count: self.count,
            _charge: Charge::new(Category::VectorGraphs, 0), _memory: memory };
        for p in points { next.insert(p)?; }
        next.finish();
        Ok(next)
    }

}

// ---------------------------------------------------------------------------
// The state.

/// One immutable version of a field's index (see the module docs).
pub struct IvfState {
    pub(crate) field: u16,
    pub(crate) centroids: Arc<Centroids>,
    pub(crate) batches: Vec<Arc<PostingBatch>>,
    /// Per centroid its current runs, `(batch, run)`.
    lists: Vec<Vec<(u32, u32)>>,
    pub(crate) indexed: Arc<Indexed>,
    /// Vector segments all of whose points are indexed (searched only
    /// through the index).
    pub(crate) covered: HashSet<u32>,
    /// Next file number of the field. The one mutable part of a state:
    /// a voided build advances it past the numbers it used (see
    /// [`IvfState::reserve_files_through`]); only maintenance, under the
    /// store's optimizer lock, reads or advances it.
    next_file: std::sync::atomic::AtomicU64,
    /// `(centroid, generation)` of clusters a split could not divide
    /// (identical vectors): not retried until they change generation.
    /// In memory only (a reopened index retries each once).
    unsplittable: HashSet<(u32, u32)>,
    /// Live rows the centroids were last trained for (in memory; a
    /// reopened index counts its indexed points). An insert that would
    /// take the index past twice this retrains on every row instead of
    /// stretching these centroids (see [`IvfState::wants_retrain`]).
    pub(crate) trained: usize,
    _charge: Charge,
    _memory: crate::admission::Reservation,
}

impl std::fmt::Debug for IvfState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IvfState").field("field", &self.field).field("centroids", &self.centroids.active()).field("batches", &self.batches.len()).finish()
    }
}

/// What one [`IvfState::search`] did (`IVFVectorsReader` counters).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IvfStats {
    pub centroids_scored: u64,
    pub postings_visited: u64,
    /// Entries of the visited lists (`expectedDocs`).
    pub entries_visited: u64,
    /// Accepted entries scored (`actualDocs`).
    pub entries_scored: u64,
}

/// Larger is worse: `-score` in a total order.
fn worse_key(score: f32) -> u32 {
    let b = (-score).to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

fn score_of_key(k: u32) -> f32 {
    -f32::from_bits(if k & 0x8000_0000 != 0 { k & 0x7fff_ffff } else { !k })
}

/// The ES score of a collector rank: the estimate (an inner product), or
/// the negated squared distance for euclidean (see [`IvfState::visit`]).
fn rank_to_score(sim: OsqSim, rank: f32) -> f32 {
    super::osq::score_of_estimate(sim, if sim == OsqSim::Euclidean { -rank } else { rank })
}

/// The collector budget counts distinct stable points, including when SOAR
/// places one point in several lists with different estimates (ES's
/// collector lets the copies take slots; its budget's 2x factor is for
/// them).
#[derive(Default)]
struct PointCollector {
    order: BTreeSet<(u32, u64)>,
    points: BTreeMap<u64, u32>,
    /// `order`'s last (worst) entry once the collector is full.
    worst: Option<(u32, u64)>,
}
impl PointCollector {
    fn len(&self) -> usize { self.points.len() }
    #[inline]
    fn offer(&mut self, point: u64, key: u32, budget: usize) {
        if budget == 0 { return; }
        // Not better than the worst kept entry of a full collector: a copy
        // of the point already kept has a key at most that worst one.
        if self.worst.is_some_and(|worst| (key, point) >= worst) {
            return;
        }
        if let Some(&previous) = self.points.get(&point) {
            if key >= previous { return; }
            self.order.remove(&(previous, point));
        } else if self.len() == budget {
            let worst = *self.order.last().expect("nonempty bounded collector");
            self.order.remove(&worst);
            self.points.remove(&worst.1);
        }
        self.order.insert((key, point));
        self.points.insert(point, key);
        if self.len() == budget {
            self.worst = self.order.last().copied();
        }
    }

    /// `(key, point)` best first.
    fn into_sorted(self) -> Vec<(u32, u64)> {
        self.order.into_iter().collect()
    }
}

impl IvfState {
    pub(crate) fn assemble(field: u16, centroids: Arc<Centroids>, batches: Vec<Arc<PostingBatch>>, indexed: Arc<Indexed>, covered: HashSet<u32>, next_file: u64) -> Result<Self> {
        let bytes = centroids.len().saturating_mul(24).saturating_add(batches.iter().map(|b| b.runs.len().saturating_mul(32)).sum::<usize>()).saturating_add(covered.capacity().saturating_mul(16)).saturating_add(batches.capacity().saturating_mul(8)).saturating_add(256);
        let memory = crate::admission::shared().reserve(bytes as u64, "IVF state registry")?;
        let mut lists = vec![Vec::new(); centroids.len()];
        for (bi, b) in batches.iter().enumerate() {
            for (ri, r) in b.runs.iter().enumerate() {
                let c = r.centroid as usize;
                if c < centroids.len() && live_gen(r.generation) && centroids.gens[c] == r.generation {
                    lists[c].push((bi as u32, ri as u32));
                }
            }
        }
        let bytes = lists.iter().map(|l| l.capacity() * 8 + 24).sum::<usize>() + covered.len() * 8;
        let trained = indexed.count() as usize;
        Ok(IvfState { field, centroids, batches, lists, indexed, covered, next_file: std::sync::atomic::AtomicU64::new(next_file), unsplittable: HashSet::new(), trained, _charge: Charge::new(Category::VectorGraphs, bytes as u64), _memory: memory })
    }

    /// The same index with `covered` replaced.
    pub(crate) fn with_covered(&self, covered: HashSet<u32>) -> Result<Self> {
        let _copies = crate::admission::shared().reserve((self.batches.len() as u64).saturating_mul(8).saturating_add((self.unsplittable.capacity() as u64).saturating_mul(16)).saturating_add(256), "IVF state copy producers")?;
        let mut out = Self::assemble(self.field, self.centroids.clone(), self.batches.clone(), self.indexed.clone(), covered, self.next_file())?.trained_on(self.trained);
        out._memory.grow((self.unsplittable.capacity() * 16) as u64, "IVF unsplittable registry")?;
        out.unsplittable = self.unsplittable.clone();
        Ok(out)
    }

    /// The same index, its centroids trained for `rows` live rows.
    pub(crate) fn trained_on(mut self, rows: usize) -> Self {
        self.trained = rows;
        self
    }

    /// Whether adding `adding` rows should retrain the centroids on every
    /// row rather than assign them: the index would grow past twice the
    /// rows its centroids were trained for, so most clusters would end up
    /// split one by one (each a step) from centroids fitted to a fraction
    /// of the data. Retraining at each doubling (ES clusters every
    /// segment it flushes or merges from scratch) costs at most about
    /// twice the final row count in all.
    pub(crate) fn wants_retrain(&self, adding: usize) -> bool {
        (self.indexed.count() as usize).saturating_add(adding) > self.trained.max(1).saturating_mul(2)
    }

    /// The same index with these unsplittable clusters.
    pub(crate) fn with_unsplittable(mut self, marks: HashSet<(u32, u32)>) -> Result<Self> {
        self._memory.grow((marks.capacity() * 16) as u64, "IVF unsplittable registry")?;
        self.unsplittable = marks;
        Ok(self)
    }

    /// The same index with cluster `c` (its current generation) marked
    /// unsplittable.
    pub(crate) fn marked_unsplittable(&self, c: usize) -> Result<Self> {
        let _covered = crate::admission::shared().reserve((self.covered.capacity() as u64).saturating_mul(16).saturating_add(256), "IVF covered copy producer")?;
        let mut out = self.with_covered(self.covered.clone())?;
        out._memory.grow((out.unsplittable.capacity() as u64).saturating_mul(16).saturating_add(64), "IVF unsplittable cluster")?;
        out.unsplittable.insert((c as u32, self.centroids.gens[c]));
        Ok(out)
    }

    /// Next file number of the field.
    pub(crate) fn next_file(&self) -> u64 {
        self.next_file.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Number the field's next files from `next` on (if past the current
    /// next number): a voided build's files await deletion at the next
    /// commit, and a later step writing one of their names again would
    /// have the commit delete a live file.
    pub(crate) fn reserve_files_through(&self, next: u64) {
        self.next_file.fetch_max(next, std::sync::atomic::Ordering::AcqRel);
    }

    pub(crate) fn unsplittable(&self) -> &HashSet<(u32, u32)> {
        &self.unsplittable
    }

    /// Centroid `c`'s current generation (`None`: dropped or absent).
    pub(crate) fn generation(&self, c: usize) -> Option<u32> {
        self.centroids.gens.get(c).copied().filter(|&g| live_gen(g))
    }

    /// The points this index holds.
    pub fn indexed(&self) -> &Indexed {
        &self.indexed
    }

    /// Whether every point of vector segment `segment` is indexed.
    pub fn covers(&self, segment: u32) -> bool {
        self.covered.contains(&segment)
    }

    /// Active centroids.
    pub fn num_centroids(&self) -> usize {
        self.centroids.active()
    }

    /// Posting batches.
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    pub(crate) fn params(&self) -> BuildParams {
        self.centroids.params
    }

    /// The commit-point record.
    pub fn manifest(&self) -> IvfFieldManifest {
        IvfFieldManifest { field: self.field, centroids: self.centroids.file, batches: self.batches.iter().map(|b| b.id).collect() }
    }

    /// Heap bytes this state holds (centroids, run tables, list index,
    /// indexed bits): the field's resident index memory.
    pub fn heap_bytes(&self) -> usize {
        self.centroids.heap_bytes()
            + self.batches.iter().map(|b| b.runs.capacity() * std::mem::size_of::<Run>()).sum::<usize>()
            + self.lists.iter().map(|l| l.capacity() * 8 + 24).sum::<usize>()
            + self.indexed.words.len() * INDEXED_WORD_BYTES as usize
            + self.covered.len() * 8
    }

    /// Bytes of the centroid file (ES `cenivf`).
    pub fn centroid_bytes(&self) -> usize {
        self.centroids.file_len
    }

    /// Bytes of the mapped posting files (ES `clivf`).
    pub fn posting_bytes(&self) -> usize {
        self.batches.iter().map(|b| b.file_bytes()).sum()
    }

    /// Entries of the current posting lists.
    pub fn entries(&self) -> u64 {
        self.lists.iter().flatten().map(|&(bi, ri)| u64::from(self.batches[bi as usize].runs[ri as usize].len)).sum()
    }

    /// Entries of centroid `c`'s current list.
    pub fn list_len(&self, c: usize) -> usize {
        self.lists[c].iter().map(|&(bi, ri)| self.batches[bi as usize].runs[ri as usize].len as usize).sum()
    }

    /// The entries (point ids, [`SPILL`] flagged) of centroid `c`'s list.
    pub(crate) fn list_entries(&self, c: usize) -> impl Iterator<Item = u64> + '_ {
        self.lists[c].iter().flat_map(move |&(bi, ri)| {
            let batch = &self.batches[bi as usize];
            let run = &batch.runs[ri as usize];
            (0..run.len as usize).map(move |j| batch.entry(run, j))
        })
    }

    /// Per active centroid, its live own (unspilled) vectors (`kept`: the
    /// point table), and the garbage fraction of the batches (dead entries
    /// and superseded runs over all entries).
    pub(crate) fn census(&self, kept: &(dyn Fn(u64) -> bool + Sync)) -> Result<(crate::admission::Allocation<Vec<usize>>, f32)> {
        let memory = crate::admission::shared().reserve((self.centroids.len() as u64).saturating_mul(8), "IVF centroid census")?;
        let live: Vec<usize> = (0..self.centroids.len()).into_par_iter().map(|c| {
            if !live_gen(self.centroids.gens[c]) { return 0; }
            let mut n = 0;
            for &(bi, ri) in &self.lists[c] {
                let b = &self.batches[bi as usize];
                let run = b.runs[ri as usize];
                n += (0..run.len as usize).filter(|&j| { let e = b.entry(&run, j); e & SPILL == 0 && kept(e) }).count();
            }
            n
        }).collect();
        let total: u64 = self.batches.iter().map(|b| b.entries).sum();
        let live_entries: usize = (0..self.centroids.len()).into_par_iter().map(|c| {
            let mut n = 0;
            for &(bi, ri) in &self.lists[c] {
                let b = &self.batches[bi as usize];
                let run = b.runs[ri as usize];
                n += (0..run.len as usize).filter(|&j| kept(b.entry(&run, j) & !SPILL)).count();
            }
            n
        }).sum();
        let garbage = total.saturating_sub(live_entries as u64);
        Ok((crate::admission::Allocation::new(live, memory), if total == 0 { 0.0 } else { garbage as f32 / total as f32 }))
    }

    /// Top `budget` entries by estimate (see [`super::ivf`]): `(point, ES
    /// score estimate)`, best first, each point once. `visit_ratio` `0`:
    /// ES's dynamic ratio for `k` / `num_candidates` (ES's adjusted count:
    /// at least `ceil(k * oversample)` when rescoring). `population`: the
    /// field's live vectors, the `n` of the visit budget (`None`: the
    /// points indexed, dead ones included). `matching`: accepted vectors
    /// when filtered (the filter's cost).
    #[allow(clippy::too_many_arguments)]
    pub fn search(&self, query: &[f32], k: usize, num_candidates: usize, visit_ratio: f32, budget: usize, population: Option<usize>, matching: Option<usize>, refine_query: bool, accept: impl Fn(u64) -> bool + Sync) -> Result<(crate::admission::Allocation<Vec<(u64, f32)>>, IvfStats)> {
        let bytes = (self.centroids.len() as u64).saturating_mul(16)
            .saturating_add((budget as u64).saturating_mul(1024))
            .saturating_add((self.params().dim as u64).saturating_mul(64)).saturating_add(4096);
        let _workspace = crate::admission::shared().reserve(bytes, "IVF search workspace")?;
        let mut stats = IvfStats::default();
        let n = population.unwrap_or(self.indexed.count() as usize).max(1);
        let p = self.centroids.params;
        if self.indexed.count() == 0 || budget == 0 || self.centroids.active() == 0 {
            return Ok((crate::admission::Allocation::new(Vec::new(), crate::admission::shared().reserve(0, "IVF search result")?), stats));
        }
        let mut s = Scratch::default();
        let q = ivf::prepare(query, p.unit, self.centroids.pre.as_ref(), &mut s).to_vec();
        let ratio = if visit_ratio > 0.0 { visit_ratio.min(1.0) } else { ivf::dynamic_visit_ratio(num_candidates, k).min(ivf::segment_size_cap(n)) };
        let max_visit = (2.0 * f64::from(ratio) * n as f64) as u64;
        let mut order: Vec<(f32, u32)> = (0..self.centroids.len()).filter(|&c| live_gen(self.centroids.gens[c])).map(|c| {
            let v = self.centroids.vector(c);
            let score = match p.sim {
                OsqSim::Euclidean => -l2_sq(&q, v),
                OsqSim::Dot | OsqSim::MaxInnerProduct => dot(&q, v),
            };
            (score, c as u32)
        }).collect();
        stats.centroids_scored = order.len() as u64;
        order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut heap = PointCollector::default();
        let mut qs = QueryScratch { refine: refine_query, ..QueryScratch::default() };
        let mut next = 0;
        // Postings are visited in waves: the lists the conditions below need
        // at the current accept rate (exactly the prefix a one-list-at-a-time
        // scan would stop after, for the visit budget), searched in parallel
        // when large (ES searches its segments concurrently). Each wave's
        // collectors merge into one: the top `budget` of all entries offered,
        // as a sequential scan keeps.
        let accept_rate = |st: &IvfStats| if st.entries_visited == 0 { 1.0 } else { st.entries_scored as f64 / st.entries_visited as f64 };
        // ES: while under the visit budget or the collector is not full.
        while next < order.len() && (max_visit > stats.entries_visited || heap.len() < budget) {
            let want = if max_visit > stats.entries_visited {
                max_visit - stats.entries_visited
            } else {
                let rate = accept_rate(&stats);
                if rate > 0.0 { (((budget - heap.len()) as f64 / rate).ceil() as u64).min(WAVE_ENTRIES) } else { 0 }
            };
            next = self.wave(&order, next, want, &q, &accept, &mut heap, budget, &mut qs, &mut stats);
        }
        if let Some(matching) = matching {
            // Keep visiting until about as many accepted vectors were scored
            // as an unfiltered search would have (`expectedScored`).
            let filtered = (matching.min(n) as f32).ceil();
            let unfiltered_ratio = stats.entries_visited as f32 / n as f32;
            let expected = (2.0 * filtered * unfiltered_ratio).min(stats.entries_visited as f32 / 2.0);
            while next < order.len() && ((stats.entries_scored as f32) < expected || (stats.entries_scored as usize) < budget) {
                let need = (f64::from(expected) - stats.entries_scored as f64).max(budget as f64 - stats.entries_scored as f64).max(1.0);
                let rate = accept_rate(&stats);
                // The accept rate grows as the scan reaches the matches:
                // waves stay small enough not to overshoot far.
                let want = if rate > 0.0 { ((need / rate).ceil() as u64).min(WAVE_ENTRIES) } else { 0 };
                next = self.wave(&order, next, want, &q, &accept, &mut heap, budget, &mut qs, &mut stats);
            }
        }
        let memory = crate::admission::shared().reserve((heap.len() as u64).saturating_mul(std::mem::size_of::<(u64, f32)>() as u64 + 16), "IVF search result")?;
        let hits = heap.into_sorted().into_iter().map(|(key, point)| (point, rank_to_score(p.sim, score_of_key(key)))).collect();
        Ok((crate::admission::Allocation::new(hits, memory), stats))
    }

    /// Visit the lists of `order[next..]` whose entries first reach `want`
    /// (at least one list; `want` 0: [`WAVE_LISTS`] lists, the accept rate
    /// being unknown); the next position.
    #[allow(clippy::too_many_arguments)]
    fn wave<F: Fn(u64) -> bool + Sync>(&self, order: &[(f32, u32)], next: usize, want: u64, q: &[f32], accept: &F, heap: &mut PointCollector, budget: usize, s: &mut QueryScratch, stats: &mut IvfStats) -> usize {
        let mut end = next;
        let mut entries = 0u64;
        while end < order.len() && (end == next || (want == 0 && end - next < WAVE_LISTS) || (want > 0 && entries < want)) {
            entries += self.list_len(order[end].1 as usize) as u64;
            end += 1;
        }
        let lists = &order[next..end];
        let workers = ((entries / PARALLEL_ENTRIES) as usize).clamp(1, MAX_WORKERS).min(lists.len());
        if workers <= 1 || rayon::current_thread_index().is_some() {
            for &(_, c) in lists {
                self.visit(c as usize, q, accept, heap, budget, s, stats);
            }
            return end;
        }
        // Contiguous groups of about equal entries.
        let per = entries.div_ceil(workers as u64);
        let mut groups: Vec<&[(f32, u32)]> = Vec::with_capacity(workers);
        let (mut start, mut acc) = (0, 0u64);
        for (i, &(_, c)) in lists.iter().enumerate() {
            acc += self.list_len(c as usize) as u64;
            if acc >= per && groups.len() + 1 < workers {
                groups.push(&lists[start..=i]);
                start = i + 1;
                acc = 0;
            }
        }
        if start < lists.len() {
            groups.push(&lists[start..]);
        }
        let parts: Vec<(PointCollector, IvfStats)> = crate::query::searcher::in_search_pool(|| groups.par_iter().map(|g| {
            let (mut h, mut st, mut qs) = (PointCollector::default(), IvfStats::default(), QueryScratch { refine: s.refine, ..QueryScratch::default() });
            for &(_, c) in g.iter() {
                self.visit(c as usize, q, accept, &mut h, budget, &mut qs, &mut st);
            }
            (h, st)
        }).collect());
        for (h, st) in parts {
            for (key, point) in h.into_sorted() {
                heap.offer(point, key, budget);
            }
            stats.postings_visited += st.postings_visited;
            stats.entries_visited += st.entries_visited;
            stats.entries_scored += st.entries_scored;
        }
        end
    }

    /// Score centroid `c`'s accepted entries into `heap` (at most `budget`,
    /// the worst on top).
    #[allow(clippy::too_many_arguments)]
    fn visit<F: Fn(u64) -> bool + Sync>(&self, c: usize, q: &[f32], accept: &F, heap: &mut PointCollector, budget: usize, s: &mut QueryScratch, stats: &mut IvfStats) {
        stats.postings_visited += 1;
        if self.lists[c].is_empty() {
            return;
        }
        let p = self.centroids.params;
        let cb = ivf::code_bytes(p.bits, p.dim);
        // The query is quantized against this centroid only once an
        // accepted entry needs scoring (ES `quantizeQueryIfNecessary`): a
        // filtered scan passes many lists without one.
        let mut quantized: Option<(OsqQuantizer, OsqCorrections)> = None;
        let mut scored = 0u64;
        for &(bi, ri) in &self.lists[c] {
            let b = &self.batches[bi as usize];
            let run = b.runs[ri as usize];
            let len = run.len as usize;
            stats.entries_visited += len as u64;
            let data = b.data.as_slice();
            let ids = &data[run.at..run.at + len * 8];
            let codes = &data[run.at + len * 8..run.at + len * (8 + cb)];
            let corrs = &data[run.at + len * (8 + cb)..run.at + len * (8 + cb + CORRECTION_BYTES)];
            // Blocks of entries (ES `BULK_SIZE`): the accepted ones' code dot
            // products first, independent of each other, then their ranks,
            // so the per-entry latency chains overlap.
            const BLOCK: usize = 16;
            let euclidean = p.sim == OsqSim::Euclidean;
            let mut at = 0;
            while at < len {
                let end = (at + BLOCK).min(len);
                let mut picked = [0usize; BLOCK];
                let mut points = [0u64; BLOCK];
                let mut m = 0;
                for j in at..end {
                    let point = u64::from_le_bytes(ids[j * 8..j * 8 + 8].try_into().unwrap()) & !SPILL;
                    if accept(point) {
                        picked[m] = j;
                        points[m] = point;
                        m += 1;
                    }
                }
                at = end;
                if m == 0 {
                    continue;
                }
                let (quantizer, qc) = quantized.get_or_insert_with(|| {
                    let quantizer = OsqQuantizer::from_centroid(p.sim, self.centroids.vector(c).to_vec(), p.bits, p.query_bits);
                    let qc = if s.refine { quantizer.quantize_query(q, &mut s.codes) } else { quantizer.quantize_query_fast(q, &mut s.codes) };
                    if p.bits != 7 {
                        ivf::query_planes(&s.codes, p.dim, &mut s.planes);
                    }
                    (quantizer, qc)
                });
                scored += m as u64;
                let mut raws = [0u32; BLOCK];
                for i in 0..m {
                    let j = picked[i];
                    raws[i] = ivf::raw_dot(p.bits, &codes[j * cb..(j + 1) * cb], &s.planes, &s.codes, p.dim);
                }
                for i in 0..m {
                    let corr = &corrs[picked[i] * CORRECTION_BYTES..(picked[i] + 1) * CORRECTION_BYTES];
                    let dc = OsqCorrections { lower: rd_f32(corr, 0), upper: rd_f32(corr, 4), additional: rd_f32(corr, 8), component_sum: rd_u32(corr, 12) as i32 };
                    // Ranked by the estimate itself, which every ES score
                    // transform is monotone in (no division per entry).
                    let e = quantizer.estimate(raws[i] as f32, qc, &dc);
                    let rank = if euclidean { -e } else { e };
                    if rank.is_nan() {
                        continue;
                    }
                    heap.offer(points[i], worse_key(rank), budget);
                }
            }
        }
        stats.entries_scored += scored;
        super::hnsw::add_vector_ops(scored);
    }

    /// The ES estimates of every entry of centroid `c`'s list against
    /// `query` (`(entry, estimate)`), for tests and diagnostics.
    pub fn score_list(&self, query: &[f32], c: usize) -> Vec<(u64, f32)> {
        let p = self.centroids.params;
        let mut s = Scratch::default();
        let q = ivf::prepare(query, p.unit, self.centroids.pre.as_ref(), &mut s).to_vec();
        let quantizer = OsqQuantizer::from_centroid(p.sim, self.centroids.vector(c).to_vec(), p.bits, p.query_bits);
        let mut qs = QueryScratch::default();
        let qc = quantizer.quantize_query(&q, &mut qs.codes);
        if p.bits != 7 {
            ivf::query_planes(&qs.codes, p.dim, &mut qs.planes);
        }
        let mut out = Vec::new();
        for &(bi, ri) in &self.lists[c] {
            let b = &self.batches[bi as usize];
            let run = b.runs[ri as usize];
            for j in 0..run.len as usize {
                let (entry, code, corr) = b.parts(&run, j, p.bits, p.dim);
                let raw = ivf::raw_dot(p.bits, code, &qs.planes, &qs.codes, p.dim);
                let dc = OsqCorrections { lower: rd_f32(corr, 0), upper: rd_f32(corr, 4), additional: rd_f32(corr, 8), component_sum: rd_u32(corr, 12) as i32 };
                out.push((entry, quantizer.score_raw(raw as f32, &qc, &dc)));
            }
        }
        out
    }
}

/// Per-search buffers of [`IvfState::visit`].
#[derive(Default)]
struct QueryScratch {
    codes: Vec<u8>,
    planes: Vec<u64>,
    /// Quantize the query with ES's full interval refinement: the
    /// estimates are the returned scores (nothing is rescored).
    refine: bool,
}

// ---------------------------------------------------------------------------
// Building (maintenance). Each step writes new files and returns the new
// state; the store installs it and retires the replaced files.

/// Where a maintenance step writes and how it is paced.
pub(crate) struct Writer<'a> {
    pub dir: &'a dyn Directory,
    pub field: u16,
    /// Through the merge throttle.
    pub throttled: bool,
    /// Called after every pass over that many vectors (the throttle).
    pub work: &'a mut dyn FnMut(usize),
    /// Files written so far (removed again if the step fails).
    pub written: Vec<String>,
}

impl Writer<'_> {
    fn write(&mut self, name: String, bytes: &[u8]) -> Result<FileSlice> {
        let out = self.dir.create_output(&name)?;
        let mut out = if self.throttled { crate::merge::throttled_output(out) } else { out };
        self.written.push(name.clone());
        out.write_all(bytes)?;
        out.finish()?;
        self.dir.open_read(&name)
    }

    /// Delete what this step wrote (it failed).
    pub(crate) fn discard(&mut self) {
        for name in self.written.drain(..) {
            let _ = self.dir.delete(&name);
        }
    }
}

/// The result of a maintenance step: the new state and the files it
/// replaces.
pub(crate) struct Step {
    pub state: IvfState,
    pub obsolete: Vec<String>,
}

fn write_centroids(w: &mut Writer<'_>, mut c: Centroids) -> Result<Arc<Centroids>> {
    let bytes = c.encode(w.field);
    w.write(centroid_file_name(w.field, c.file), &bytes)?;
    c.file_len = bytes.len();
    Ok(Arc::new(c))
}

fn write_batch(w: &mut Writer<'_>, id: u64, params: &BuildParams, mut runs: Vec<RunOut>) -> Result<Arc<PostingBatch>> {
    runs.retain(|r| r.len > 0);
    runs.sort_by_key(|r| (r.centroid, r.generation));
    let bytes = encode_batch(w.field, params, &runs);
    let data = w.write(batch_file_name(w.field, id), &bytes)?;
    Ok(Arc::new(PostingBatch::open(w.field, id, data, params)?))
}

/// Group `entries` (`(centroid, entry, row)`) into runs and quantize them.
fn runs_of(centroids: &Centroids, mut entries: Vec<(u32, u64, &[f32])>) -> Vec<RunOut> {
    entries.sort_by_key(|e| (e.0, e.1 & !SPILL));
    let mut groups: Vec<(u32, Vec<(u64, &[f32])>)> = Vec::new();
    for (c, e, row) in entries {
        match groups.last_mut() {
            Some((gc, g)) if *gc == c => g.push((e, row)),
            _ => groups.push((c, vec![(e, row)])),
        }
    }
    groups.par_iter().map(|(c, rows)| encode_run(centroids, *c, rows)).collect()
}

/// Transient bytes a maintenance step allocates per row it trains on,
/// inserts or moves (vectors stay in the mapped segments; only references
/// and codes are built):
///
/// * 16, the row slice the clusterer reads ([`Points`]);
/// * 96, clustering: ids, assignments, spill, `update_centroids` members,
///   sample positions (a `usize` a row) and `cluster_and_split` position
///   lists, a few words a row along one recursion path;
/// * `dim / 128`, the per-chunk partial means (`dim` f64 a 1024 rows);
/// * 64, up to two `(centroid, entry, row)` entries (primary and SOAR spill);
/// * 48, the same entries grouped as `(entry, row)` for encoding;
/// * `4 x (24 + code)`: up to two entries (8-byte entry, packed code,
///   16 bytes of corrections) held twice, as runs and as the encoded file.
pub(crate) fn step_row_bytes(dim: usize, bits: u8) -> u64 {
    let entry = (8 + CORRECTION_BYTES + ivf::code_bytes(bits, dim)) as u64;
    (16 + 96 + 64 + 48 + dim as u64 / 128).saturating_add(entry.saturating_mul(4))
}

/// Upper bound on the centroids a clustering of `rows` makes: about one
/// per `target` rows, with slack for unbalanced splits (never more than
/// the rows).
fn cluster_bound(rows: usize, target: usize) -> usize {
    rows.min(rows.saturating_mul(2) / target.max(1) + ivf::CLUSTERS_PER_NEIGHBORHOOD)
}

/// Admission for one step's transient memory: [`step_row_bytes`] for each
/// of `rows`; `clusters` new centroids (`dim` f32 and a neighbourhood
/// list each); per worker thread its scratch vectors and the distances to
/// `scan` centroids ([`ivf::assign`], the neighbourhood search); and
/// `extra` bytes.
fn workspace(rows: usize, params: &BuildParams, clusters: usize, scan: usize, extra: usize) -> Result<crate::admission::Reservation> {
    crate::admission::shared().reserve(workspace_bytes(rows, params, clusters, scan, extra, rayon::current_num_threads()), "IVF maintenance workspace")
}

/// [`workspace`]'s bytes with `threads` workers.
fn workspace_bytes(rows: usize, params: &BuildParams, clusters: usize, scan: usize, extra: usize, threads: usize) -> u64 {
    let dim = params.dim as u64;
    let per_cluster = dim.saturating_mul(4).saturating_add(ivf::CLUSTERS_PER_NEIGHBORHOOD as u64 * 4 + 32);
    let per_thread = dim.saturating_mul(128).saturating_add((scan as u64).saturating_mul(8));
    (rows as u64).saturating_mul(step_row_bytes(params.dim, params.bits))
        .saturating_add((clusters as u64).saturating_mul(per_cluster))
        .saturating_add((threads as u64).saturating_mul(per_thread))
        .saturating_add(extra as u64).saturating_add(4096)
}

/// Most rows one train or insert step takes so that its workspace stays
/// within a quarter of a `budget`-byte admission limit (at least one
/// cluster's worth); the rest are left for the next steps.
pub(crate) fn max_step_rows(params: &BuildParams, budget: u64) -> usize {
    usize::try_from(budget / 4 / step_row_bytes(params.dim, params.bits).max(1)).unwrap_or(usize::MAX).max(ivf::MAX_VECTORS_PER_CLUSTER as usize)
}

/// Train a field's first centroids on `rows` (`(point, vector)`, every
/// row of the field) and index them all.
pub(crate) fn train(w: &mut Writer<'_>, params: BuildParams, options: &IvfOptions, rows: &[(u64, &[f32])], next_file: u64) -> Result<(Arc<Centroids>, Arc<PostingBatch>, u64)> {
    next_file.checked_add(2).ok_or_else(|| bad("file sequence exhausted"))?;
    let clusters = cluster_bound(rows.len(), options.target_cluster_size());
    let _workspace = workspace(rows.len(), &params, clusters, clusters, 0)?;
    let pre = params.precondition.then(|| Preconditioner::new(params.dim));
    let slices: Vec<&[f32]> = rows.iter().map(|r| r.1).collect();
    let points = Points::new(&slices, params.unit, pre.as_ref(), params.dim);
    let clusters = ivf::cluster(&points, options.target_cluster_size(), w.work);
    let k = clusters.len();
    let centroids = Centroids::new(next_file, params, pre, vec![1; k], clusters.centroids)?;
    let mut entries = Vec::with_capacity(rows.len() * 2);
    for (i, &(p, row)) in rows.iter().enumerate() {
        entries.push((clusters.assign[i], p, row));
        if clusters.spill[i] != NO_SPILL {
            entries.push((clusters.spill[i], p | SPILL, row));
        }
    }
    let runs = runs_of(&centroids, entries);
    (w.work)(rows.len());
    let centroids = write_centroids(w, centroids)?;
    let batch = write_batch(w, next_file + 1, &params, runs)?;
    Ok((centroids, batch, next_file + 2))
}

/// Index `rows` not yet indexed: nearest centroid plus SOAR spill, one new
/// batch. `dead`: points found unindexed but no longer live, marked
/// indexed with no entries (not inserted, and their segments are covered).
pub(crate) fn insert(w: &mut Writer<'_>, state: &IvfState, rows: &[(u64, &[f32])], dead: &[u64]) -> Result<Step> {
    state.next_file().checked_add(1).ok_or_else(|| bad("file sequence exhausted"))?;
    // The indexed copy reserves itself (`Indexed::with`).
    let _workspace = workspace(rows.len(), &state.params(), 0, state.centroids.len(), dead.len().saturating_mul(8))?;
    if rows.is_empty() {
        let indexed = Arc::new(state.indexed.with(dead.iter().copied())?);
        return Ok(Step { state: IvfState::assemble(state.field, state.centroids.clone(), state.batches.clone(), indexed, state.covered.clone(), state.next_file())?.trained_on(state.trained), obsolete: Vec::new() });
    }
    let c = &state.centroids;
    let p = c.params;
    let slices: Vec<&[f32]> = rows.iter().map(|r| r.1).collect();
    let points = Points::new(&slices, p.unit, c.pre.as_ref(), p.dim);
    let active: Vec<bool> = c.gens.iter().map(|&g| live_gen(g)).collect();
    let assigned = ivf::assign(&points, &c.data, &active);
    (w.work)(rows.len());
    let mut entries = Vec::with_capacity(rows.len() * 2);
    for (&(point, row), &(primary, spill)) in rows.iter().zip(&assigned) {
        if primary == u32::MAX {
            continue;
        }
        entries.push((primary, point, row));
        if spill != NO_SPILL {
            entries.push((spill, point | SPILL, row));
        }
    }
    let runs = runs_of(c, entries);
    let batch = write_batch(w, state.next_file(), &p, runs)?;
    let mut batches = state.batches.clone();
    batches.push(batch);
    let indexed = Arc::new(state.indexed.with(rows.iter().map(|r| r.0).chain(dead.iter().copied()))?);
    Ok(Step { state: IvfState::assemble(state.field, c.clone(), batches, indexed, state.covered.clone(), state.next_file() + 1)?.trained_on(state.trained), obsolete: Vec::new() })
}

/// The live entries of one cluster and their rows (`(entry, row)`).
pub(crate) type Members<'a> = Vec<(u64, &'a [f32])>;

/// Split centroid `c` (`members`: its list's live entries and their rows):
/// [`split_many`] of one cluster. `None`: it could not be divided.
pub(crate) fn split(w: &mut Writer<'_>, state: &IvfState, c: usize, members: &[(u64, &[f32])], options: &IvfOptions) -> Result<Option<Step>> {
    let step = split_many(w, state, &[(c, members.to_vec())], options)?;
    Ok((!step.obsolete.is_empty()).then_some(step))
}

/// Split each cluster of `groups` (`(centroid, its live entries and their
/// rows)`) with `HierarchicalKMeans.clusterAndSplit`, all in one step (one
/// centroid file, one batch): the centroid takes a new generation, its
/// other parts dropped slots or new ones; its own vectors move to their
/// nearest part, the copies spilled into it to theirs (their other copies
/// stay). A cluster that cannot be divided (identical vectors) is marked
/// unsplittable for its generation; when none could be, the step only
/// records the marks (no files, `obsolete` empty).
pub(crate) fn split_many(w: &mut Writer<'_>, state: &IvfState, groups: &[(usize, Members<'_>)], options: &IvfOptions) -> Result<Step> {
    state.next_file().checked_add(2).ok_or_else(|| bad("file sequence exhausted"))?;
    let total: usize = groups.iter().map(|g| g.1.len()).sum();
    // The new centroid table reserves itself (`Centroids::new`).
    let _workspace = workspace(total, &state.params(), cluster_bound(total, options.target_cluster_size()), 0, 0)?;
    let old = &state.centroids;
    let p = old.params;
    let mut gens = old.gens.clone();
    let mut data = old.data.clone();
    let mut free: std::collections::VecDeque<u32> = (0..gens.len()).filter(|&s| gens[s] & DROPPED != 0).map(|s| s as u32).collect();
    let mut entries: Vec<(u32, u64, &[f32])> = Vec::with_capacity(total);
    let mut marks: Vec<(u32, u32)> = Vec::new();
    let mut s = Scratch::default();
    for (c, members) in groups {
        let c = *c;
        let own: Vec<&(u64, &[f32])> = members.iter().filter(|m| m.0 & SPILL == 0).collect();
        let slices: Vec<&[f32]> = own.iter().map(|m| m.1).collect();
        let points = Points::new(&slices, p.unit, old.pre.as_ref(), p.dim);
        let (parts, part_of) = ivf::split(&points, options.target_cluster_size(), w.work);
        let k = parts.len() / p.dim;
        if k <= 1 {
            marks.push((c as u32, old.gens[c]));
            continue;
        }
        // `c` first (its next generation), then dropped slots, then new ones.
        let mut slots: Vec<u32> = vec![c as u32];
        while slots.len() < k {
            match free.pop_front() {
                Some(slot) => slots.push(slot),
                None => {
                    slots.push(gens.len() as u32);
                    gens.push(0);
                    data.extend(std::iter::repeat_n(0f32, p.dim));
                }
            }
        }
        for (part, &slot) in slots.iter().enumerate() {
            let slot = slot as usize;
            gens[slot] = (gens[slot] & !DROPPED).checked_add(1).filter(|&g| g & DROPPED == 0).ok_or_else(|| bad("centroid generation exhausted"))?;
            data[slot * p.dim..(slot + 1) * p.dim].copy_from_slice(ivf::centroid(&parts, p.dim, part));
        }
        entries.extend(own.iter().zip(&part_of).map(|(m, &a)| (slots[a as usize], m.0, m.1)));
        // Copies spilled into `c`: to the nearest part.
        for m in members.iter().filter(|m| m.0 & SPILL != 0) {
            let v = ivf::prepare(m.1, p.unit, old.pre.as_ref(), &mut s);
            let best = (0..k).min_by(|&a, &b| l2_sq(v, ivf::centroid(&parts, p.dim, a)).total_cmp(&l2_sq(v, ivf::centroid(&parts, p.dim, b)))).unwrap_or(0);
            entries.push((slots[best], m.0, m.1));
        }
    }
    let mut unsplittable = state.unsplittable.clone();
    unsplittable.extend(marks.iter().copied());
    if entries.is_empty() {
        return Ok(Step { state: state.with_covered(state.covered.clone())?.with_unsplittable(unsplittable)?, obsolete: Vec::new() });
    }
    let centroids = Centroids::new(state.next_file(), p, old.pre.clone(), gens, data)?;
    (w.work)(total);
    let runs = runs_of(&centroids, entries);
    let centroids = write_centroids(w, centroids)?;
    let mut batches = state.batches.clone();
    batches.push(write_batch(w, state.next_file() + 1, &p, runs)?);
    let obsolete = vec![centroid_file_name(state.field, old.file)];
    let next = IvfState::assemble(state.field, centroids, batches, state.indexed.clone(), state.covered.clone(), state.next_file() + 2)?.trained_on(state.trained);
    Ok(Step { state: next.with_unsplittable(unsplittable)?, obsolete })
}

/// Drop centroid `c` (`members` as for [`split`]): [`drop_many`] of one.
pub(crate) fn drop_centroid(w: &mut Writer<'_>, state: &IvfState, c: usize, members: &[(u64, &[f32])]) -> Result<Option<Step>> {
    drop_many(w, state, &[(c, members.to_vec())])
}

/// Drop every centroid of `groups` in one step: their entries move to
/// their nearest remaining centroid. `None`: no centroid would remain.
pub(crate) fn drop_many(w: &mut Writer<'_>, state: &IvfState, groups: &[(usize, Members<'_>)]) -> Result<Option<Step>> {
    state.next_file().checked_add(2).ok_or_else(|| bad("file sequence exhausted"))?;
    let total: usize = groups.iter().map(|g| g.1.len()).sum();
    let _workspace = workspace(total, &state.params(), 0, state.centroids.len(), 0)?;
    let old = &state.centroids;
    let p = old.params;
    let mut gens = old.gens.clone();
    for (c, _) in groups {
        gens[*c] |= DROPPED;
    }
    if !gens.iter().any(|&g| live_gen(g)) {
        return Ok(None);
    }
    let centroids = Centroids::new(state.next_file(), p, old.pre.clone(), gens, old.data.clone())?;
    let active: Vec<bool> = centroids.gens.iter().map(|&g| live_gen(g)).collect();
    let members: Vec<(u64, &[f32])> = groups.iter().flat_map(|g| g.1.iter().copied()).collect();
    let slices: Vec<&[f32]> = members.iter().map(|m| m.1).collect();
    let points = Points::new(&slices, p.unit, old.pre.as_ref(), p.dim);
    let assigned = ivf::assign(&points, &centroids.data, &active);
    (w.work)(members.len());
    let entries: Vec<(u32, u64, &[f32])> = members.iter().zip(&assigned).filter(|(_, a)| a.0 != u32::MAX).map(|(m, a)| (a.0, m.0, m.1)).collect();
    let runs = runs_of(&centroids, entries);
    let centroids = write_centroids(w, centroids)?;
    let mut batches = state.batches.clone();
    batches.push(write_batch(w, state.next_file() + 1, &p, runs)?);
    let obsolete = vec![centroid_file_name(state.field, old.file)];
    Ok(Some(Step { state: IvfState::assemble(state.field, centroids, batches, state.indexed.clone(), state.covered.clone(), state.next_file() + 2)?.trained_on(state.trained), obsolete }))
}

/// Rewrite batches `which` (indexes) as one: their current runs without
/// the entries of points no longer `kept`, codes copied verbatim.
pub(crate) fn compact(w: &mut Writer<'_>, state: &IvfState, which: &[usize], kept: &(dyn Fn(u64) -> bool + Sync)) -> Result<Step> {
    state.next_file().checked_add(1).ok_or_else(|| bad("file sequence exhausted"))?;
    let bytes: usize = which.iter().map(|&i| state.batches[i].file_bytes()).sum();
    // Runs as copied (grown by doubling), merged, then encoded.
    let _workspace = workspace(0, &state.params(), 0, 0, bytes.saturating_mul(4))?;
    let c = &state.centroids;
    let p = c.params;
    let chosen: HashSet<usize> = which.iter().copied().collect();
    let mut runs: Vec<RunOut> = Vec::new();
    let mut copied = 0usize;
    for &bi in which {
        let b = &state.batches[bi];
        for r in &b.runs {
            let cc = r.centroid as usize;
            if cc >= c.len() || !live_gen(c.gens[cc]) || c.gens[cc] != r.generation {
                continue;
            }
            let mut out = RunOut::new(r.centroid, r.generation);
            for j in 0..r.len as usize {
                let (entry, code, corr) = b.parts(r, j, p.bits, p.dim);
                if kept(entry & !SPILL) {
                    out.push_raw(entry, code, corr);
                }
            }
            copied += r.len as usize;
            runs.push(out);
        }
    }
    (w.work)(copied);
    // One run per (centroid, generation).
    runs.sort_by_key(|r| (r.centroid, r.generation));
    let mut merged: Vec<RunOut> = Vec::new();
    for r in runs {
        match merged.last_mut() {
            Some(m) if m.centroid == r.centroid && m.generation == r.generation => {
                m.entries.extend_from_slice(&r.entries);
                m.codes.extend_from_slice(&r.codes);
                m.corrections.extend_from_slice(&r.corrections);
                m.len += r.len;
            }
            _ => merged.push(r),
        }
    }
    let batch = write_batch(w, state.next_file(), &p, merged)?;
    let obsolete = which.iter().map(|&bi| batch_file_name(state.field, state.batches[bi].id)).collect();
    let mut batches: Vec<Arc<PostingBatch>> = state.batches.iter().enumerate().filter(|(i, _)| !chosen.contains(i)).map(|(_, b)| b.clone()).collect();
    batches.push(batch);
    Ok(Step { state: IvfState::assemble(state.field, c.clone(), batches, state.indexed.clone(), state.covered.clone(), state.next_file() + 1)?.trained_on(state.trained), obsolete })
}

/// The next maintenance action a state needs (`live`, `garbage`: its
/// [`IvfState::census`]), in priority order: split, drop, compact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    /// Split these clusters (largest first).
    Split(Vec<usize>),
    /// Drop these clusters.
    Drop(Vec<usize>),
    /// Compact these batches.
    Compact(Vec<usize>),
}

/// Live own vectors one batched split or drop step takes at most (about
/// twice as many entries with their spilled copies); the rest wait for
/// the next step.
const STEP_MEMBERS: usize = 1 << 18;

pub(crate) fn next_action(state: &IvfState, options: &IvfOptions, live: &[usize], garbage: f32, force: bool) -> Option<Action> {
    let target = options.target_cluster_size();
    let splittable = |c: usize| state.generation(c).is_some_and(|g| !state.unsplittable.contains(&(c as u32, g)));
    let mut over: Vec<(usize, usize)> = live.iter().enumerate().filter(|&(c, &n)| n > SPLIT_FACTOR * target && splittable(c)).map(|(c, &n)| (c, n)).collect();
    if !over.is_empty() {
        over.sort_by_key(|&(c, n)| (std::cmp::Reverse(n), c));
        let mut taken = 0usize;
        let chosen: Vec<usize> = over.into_iter().take_while(|&(_, n)| { let first = taken == 0; taken += n; first || taken <= STEP_MEMBERS }).map(|(c, _)| c).collect();
        return Some(Action::Split(chosen));
    }
    let active = state.centroids.active();
    if active > 1 {
        let min = (target / DROP_DIVISOR).max(1);
        let mut taken = 0usize;
        let chosen: Vec<usize> = live.iter().enumerate().filter(|&(c, &n)| live_gen(state.centroids.gens[c]) && n < min)
            .take(active - 1).take_while(|&(_, &n)| { let first = taken == 0; taken += n.max(1); first || taken <= STEP_MEMBERS }).map(|(c, _)| c).collect();
        if !chosen.is_empty() {
            return Some(Action::Drop(chosen));
        }
    }
    let garbage_high = garbage > GARBAGE_RATIO;
    let batches = state.batches.len();
    if batches > MAX_BATCHES || (batches > 0 && garbage_high) || (force && (batches > 1 || garbage > 0.0)) {
        // Smallest batches first, within the pass budget (a force merge
        // takes them all).
        let mut order: Vec<usize> = (0..batches).collect();
        order.sort_by_key(|&i| (state.batches[i].file_bytes(), i));
        let mut chosen = Vec::new();
        let mut bytes = 0;
        for i in order {
            if !force && chosen.len() >= 2 && bytes + state.batches[i].file_bytes() > COMPACT_BUDGET {
                break;
            }
            bytes += state.batches[i].file_bytes();
            chosen.push(i);
        }
        if chosen.len() >= 2 || garbage > 0.0 {
            return Some(Action::Compact(chosen));
        }
    }
    None
}

/// Open a field's committed index. `Ok(None)`: files of a version this
/// build does not know (the index is rebuilt by maintenance).
pub(crate) fn open(dir: &dyn Directory, manifest: &IvfFieldManifest) -> Result<Option<(Arc<Centroids>, Vec<Arc<PostingBatch>>, Arc<Indexed>, u64, crate::admission::Reservation)>> {
    let workspace = crate::admission::shared().reserve((manifest.batches.len() as u64).saturating_mul(128).saturating_add(4096), "IVF open registry")?;
    let field = manifest.field;
    let bytes = dir.open_read(&centroid_file_name(field, manifest.centroids))?;
    let Some(centroids) = Centroids::decode(field, manifest.centroids, bytes.as_slice())? else { return Ok(None) };
    let params = centroids.params;
    let mut batches = Vec::with_capacity(manifest.batches.len());
    for &id in &manifest.batches {
        batches.push(Arc::new(PostingBatch::open(field, id, dir.open_read(&batch_file_name(field, id))?, &params)?));
    }
    // The indexed points: every entry's point (reads only the entry
    // arrays of the mapping).
    let mut indexed = Indexed::empty()?;
    for b in &batches {
        // A long open is cancellable between batches.
        crate::cancel::check()?;
        for r in &b.runs {
            for j in 0..r.len as usize {
                let p = b.entry(r, j) & !SPILL;
                indexed.insert(p)?;
            }
        }
    }
    indexed.finish();
    let next = manifest.batches.iter().copied().chain(std::iter::once(manifest.centroids)).max().unwrap_or(0).checked_add(1).ok_or_else(|| bad("file sequence exhausted"))?;
    Ok(Some((Arc::new(centroids), batches, Arc::new(indexed), next, workspace)))
}

/// An empty indexed set (a state assembled from scratch).
pub(crate) fn indexed_of(points: impl IntoIterator<Item = u64>) -> Result<Arc<Indexed>> {
    Ok(Arc::new(Indexed::empty()?.with(points)?))
}

#[cfg(test)]
mod tests;
