//! Immutable durable vector payloads, indexed by stable point IDs.
//!
//! New chunks and compacted payloads contain no HNSW graph. Mutable field
//! owners maintain ANN topology independently. Explicit flat fields may
//! retain a quantized scoring column. Historical file decoding remains
//! available, but there is no immutable graph construction or merge writer.
//!
//! Liveness belongs to the vector store's point table, not these files.

use std::cmp::Ordering;
use std::sync::{Arc, OnceLock};

use crate::codec::directory::{Directory, FileSlice};
use crate::codec::vector_segment_file_name;
use crate::codec::vectors::{open_vector_segment, write_vector_segment_streamed, VectorSegmentParts};
use crate::error::Result;
use crate::types::DocId;

use super::hnsw::{FilterMode, HnswGraph, VectorSource};
use super::int4::{self, Int4Vectors, UniformQuantizer};
use super::quantize::{Int8Vectors, Quantization, QuantizedVectors, ScalarQuantizer};
use super::Similarity;

/// What a sealed segment carries (ES `index_options` family).
#[derive(Clone, Debug)]
pub enum SealKind {
    /// Appendable chunk: no graph, no quantized copy.
    Chunk,
    /// Sealed without a graph (ES `*_flat`): brute force.
    Flat,
}

/// How to seal (or keep appendable) a field's rows.
#[derive(Clone, Debug)]
pub struct SealSpec {
    pub kind: SealKind,
    /// Which quantized copy to store (`None` = `f32` only).
    pub quantize: Quantization,
    /// Lucene confidence interval for the quantizer fit (`0` = dynamic).
    pub confidence_interval: Option<f32>,
}

impl SealSpec {
    /// Appendable chunk: no graph, no quantized copy.
    pub fn chunk() -> Self {
        Self { kind: SealKind::Chunk, quantize: Quantization::None, confidence_interval: None }
    }
}

/// Transient payload-encoding bound of [`VectorSegment::write_with`]: the
/// flattened f32 column (4 bytes per cell), plus for a quantized copy its
/// per-row codes and encoded column (int8: 2 bytes per cell; int4: a
/// normalized f32 copy and packed codes, 6), per-row metadata and writer
/// buffers. Sources remain owned by the caller and are not released to
/// satisfy this reservation. (It used to charge 32 bytes per cell whatever
/// was encoded: a 67k-row 1536-d vector segment asked for 3.3 GB.)
fn payload_workspace(budget: &Arc<crate::admission::Budget>, rows: usize, dim: u32, quantize: Quantization) -> Result<crate::admission::Reservation> {
    let per_cell = 4 + match quantize { Quantization::Int8 => 2, Quantization::Int4 => 6, _ => 0 };
    let bytes = (rows as u64).saturating_mul((dim as u64).saturating_mul(per_cell).saturating_add(128)).saturating_add(64 << 10);
    budget.reserve(bytes, "vector payload encoding")
}

/// Where the `f32` column lives.
enum Floats {
    /// Zero-copy view into the mapped file (4-byte aligned).
    Mapped(FileSlice),
    /// Copied out because the mapping was misaligned.
    Owned(Vec<f32>),
}

struct CachedFloats {
    values: Vec<f32>,
    _reservation: super::memory::Reservation<'static>,
}

/// An immutable, file-backed vector segment.
pub struct VectorSegment {
    pub id: u32,
    pub field: u16,
    pub similarity: Similarity,
    pub dim: u32,
    points: Vec<u64>,
    floats: Floats,
    /// Budget-admitted RAM-resident copy of a mapped `f32` column, made before the first
    /// graph traversal that scores with `f32` (see
    /// [`make_resident`](Self::make_resident)). Once set, every read of the
    /// column goes through it, so the layer-0 beam never takes a page fault.
    resident: OnceLock<Option<CachedFloats>>,
    squared_norms: OnceLock<Vec<f32>>,
    original_norms: OnceLock<Option<CachedFloats>>,
    normalized: OnceLock<Option<CachedFloats>>,
    graph_bytes: Option<FileSlice>,
    graph: OnceLock<Arc<HnswGraph>>,
    int8: Option<FileSlice>,
    int4: Option<FileSlice>,
    flat: bool,
    q_int8: OnceLock<Arc<QuantizedVectors>>,
    q_int4: OnceLock<Arc<QuantizedVectors>>,
    q_binary: OnceLock<Arc<QuantizedVectors>>,
    /// ES-scoring quantizers fitted on this segment's rows (see
    /// [`super::es_quant`]), by [`super::es_quant::EsQuant::key`].
    es_quant: parking_lot::Mutex<Vec<((u8, u32), Arc<super::es_quant::EsCodes>)>>,
    /// The field's `confidence_interval` for an int4 copy fitted in memory
    /// (a segment without a stored int4 column), noted by the store before
    /// the first int4 use.
    int4_ci: OnceLock<Option<f32>>,
}

impl VectorSegment {
    /// Open vector segment `id` from `dir`.
    pub fn open(dir: &dyn Directory, id: u32) -> Result<Self> {
        let f = open_vector_segment(dir.open_read(&vector_segment_file_name(id))?, false)?;
        let bytes = f.vectors.as_slice();
        let floats = if cfg!(target_endian = "little") && bytes.as_ptr().align_offset(4) == 0 {
            Floats::Mapped(f.vectors.clone())
        } else {
            Floats::Owned(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };
        Ok(Self {
            id,
            field: f.field,
            similarity: f.similarity,
            dim: f.dim,
            points: f.points,
            floats,
            resident: OnceLock::new(),
            squared_norms: OnceLock::new(),
            original_norms: OnceLock::new(),
            normalized: OnceLock::new(),
            graph_bytes: f.hnsw,
            graph: OnceLock::new(),
            int8: f.int8,
            int4: f.int4,
            flat: f.flat,
            q_int8: OnceLock::new(),
            q_int4: OnceLock::new(),
            q_binary: OnceLock::new(),
            es_quant: parking_lot::Mutex::new(Vec::new()),
            int4_ci: OnceLock::new(),
        })
    }

    /// Write durable vector payloads. ANN topology belongs exclusively to
    /// the mutable field owner and is never constructed by this writer.
    pub fn write_with(
        dir: &dyn Directory, id: u32, field: u16, similarity: Similarity,
        dim: u32, rows: &[(u64, &[f32])], spec: SealSpec,
    ) -> Result<Self> {
        // Payload encoding keeps the source rows, flattened f32 column and
        // encoded output alive together. Admit that peak before copying.
        let quantize = if matches!(spec.kind, SealKind::Chunk) { Quantization::None } else { spec.quantize };
        let _workspace = payload_workspace(&crate::admission::shared(), rows.len(), dim, quantize)?;
        let points: Vec<u64> = rows.iter().map(|(p, _)| *p).collect();
        let mut vectors = Vec::with_capacity(rows.len() * dim as usize);
        for (_, v) in rows {
            assert_eq!(v.len(), dim as usize, "vector dim mismatch");
            vectors.extend_from_slice(v);
        }
        let hnsw: Option<Vec<u8>> = None;
        let n = rows.len();
        let row = |i: usize| rows[i].1;
        let int8 = (quantize == Quantization::Int8 && n > 0).then(|| {
            // Lucene's interval (7-bit fit, fixed or dynamic) on today's
            // [-127, 127] encoding, so the stored column stays readable by
            // every node.
            let fit = match spec.confidence_interval {
                Some(ci) if ci == int4::DYNAMIC_CONFIDENCE_INTERVAL => UniformQuantizer::fit_auto(n, row, similarity, 7),
                Some(ci) => UniformQuantizer::fit(n, row, ci, 7),
                None => UniformQuantizer::fit(n, row, int4::default_confidence_interval(dim as usize), 7),
            };
            let q = ScalarQuantizer { min: fit.min, max: fit.max };
            let qrows: Vec<Vec<i8>> = rows.iter().map(|(_, v)| q.quantize(v)).collect();
            Int8Vectors::encode(dim, q, &qrows)
        });
        let int4 = (quantize == Quantization::Int4 && n > 0).then(|| {
            let (q, normalised, qrows, offsets) = int4::build_int4(n, row, similarity, spec.confidence_interval);
            Int4Vectors::encode(dim, q, normalised, &qrows, &offsets)
        });
        let parts = VectorSegmentParts {
            field,
            similarity,
            dim,
            points: &points,
            vectors: &vectors,
            hnsw: hnsw.as_deref(),
            int8: int8.as_deref(),
            int4: int4.as_deref(),
            flat: matches!(spec.kind, SealKind::Flat),
        };
        // Keep the historical allocation only for controlled POC comparisons.
        #[cfg(feature = "payload-hnsw-poc")]
        if std::env::var_os("RUSTIC_POC_BUFFERED_VECTOR_SEGMENT").is_some() {
            let bytes = crate::codec::vectors::encode_vector_segment(&parts);
            dir.write(&vector_segment_file_name(id), &bytes)?;
            return Self::open(dir, id);
        }
        write_vector_segment_streamed(dir, &vector_segment_file_name(id), &parts)?;
        Self::open(dir, id)
    }

    /// Number of ordinals (live or not).
    /// The quantizer ES would have fitted on this segment's vectors for
    /// `spec` (cached).
    pub fn es_quantizer(&self, spec: &super::es_quant::EsQuant) -> Result<super::es_quant::EsQuantizer> {
        Ok(self.es_codes(spec)?.q.clone())
    }

    /// One admitted build per segment. Hold the cache lock across construction
    /// so competing misses cannot allocate duplicate full code columns.
    pub fn es_codes(&self, spec: &super::es_quant::EsQuant) -> Result<Arc<super::es_quant::EsCodes>> {
        self.es_codes_admitted(spec, &crate::admission::shared())
    }
    fn es_codes_admitted(&self, spec: &super::es_quant::EsQuant, budget: &Arc<crate::admission::Budget>) -> Result<Arc<super::es_quant::EsCodes>> {
        let key = spec.key();
        let mut cache=self.es_quant.lock();
        if let Some((_, codes))=cache.iter().find(|(k,_)| *k==key) { return Ok(codes.clone()); }
        let n=self.len() as usize;
        let dims=self.dim as usize;
        let retained=(n as u64).saturating_mul((dims as u64).saturating_add(64)).saturating_add((dims as u64).saturating_mul(64)).saturating_add(4096);
        let memory=budget.reserve(retained,"vector quantized codes")?;
        // Scalar fit and normalization may hold multiple sampled float rows.
        let _scratch=budget.reserve((n as u64).saturating_mul((dims as u64).saturating_mul(16).saturating_add(64)).saturating_add(65536),"vector quantizer fit")?;
        let q=spec.fit(self.similarity,n,|i|self.vector(i as u32));
        let mut codes=super::es_quant::EsCodes::build(q,self.similarity,n,dims,|i|self.vector(i as u32));
        codes.retain_admission(memory);
        let codes=Arc::new(codes);
        cache.push((key,codes.clone()));
        Ok(codes)
    }

    /// Number of ordinals (live or not).
    pub fn len(&self) -> u32 {
        self.points.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Whether this segment is sealed (immutable: has an HNSW graph, or is
    /// a flat sealed segment). Appendable chunks are not.
    pub fn is_sealed(&self) -> bool {
        self.graph_bytes.is_some() || self.flat
    }

    /// Whether this segment has an HNSW graph.
    pub fn has_graph(&self) -> bool {
        self.graph_bytes.is_some()
    }

    /// Whether this is a flat sealed segment (no graph, brute force).
    pub fn is_flat(&self) -> bool {
        self.flat && self.graph_bytes.is_none()
    }

    /// Point id of each ordinal.
    pub fn points(&self) -> &[u64] {
        &self.points
    }

    /// The vector at `ord`.
    pub fn vector(&self, ord: u32) -> &[f32] {
        let dim = self.dim as usize;
        let start = ord as usize * dim;
        if let Some(v) = self.resident.get().and_then(Option::as_ref) {
            return &v.values[start..start + dim];
        }
        match &self.floats {
            Floats::Mapped(s) => {
                let b = &s.as_slice()[start * 4..(start + dim) * 4];
                // SAFETY: checked 4-byte aligned at open; any bit pattern is
                // a valid f32; the slice lives as long as `self`.
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, dim) }
            }
            Floats::Owned(v) => &v[start..start + dim],
        }
    }

    /// Original-float magnitudes for script scoring. Use the exact same norm
    /// reduction as the interpreter, not the ANN kernel's squared norms (its
    /// accumulation order can differ). Optional storage is budget admitted.
    pub(super) fn original_norm(&self, ord: u32) -> f32 {
        let cached = self.original_norms.get_or_init(|| {
            let bytes = self.points.len().checked_mul(std::mem::size_of::<f32>())?;
            let reservation = super::memory::column_budget().reserve(bytes)?;
            let values = (0..self.len()).map(|i| super::distance::norm(self.vector(i))).collect();
            Some(CachedFloats { values, _reservation: reservation })
        });
        cached.as_ref().map_or_else(|| super::distance::norm(self.vector(ord)), |v| v.values[ord as usize])
    }

    /// Copy a memory-mapped `f32` column into the heap (once). A filtered
    /// graph search visits ~`1 / selectivity` times more nodes than an
    /// unfiltered one, so a page fault (or any random mmap read) in the
    /// traversal loop would dominate; with the column resident the loop only
    /// touches heap memory, like the graph, the ordinal filter and the
    /// quantized codes. A column that is already owned is left alone.
    /// Returns the bytes held by the copy (0 when nothing was copied).
    pub fn make_resident(&self) -> usize {
        if !matches!(self.floats, Floats::Mapped(_)) {
            return 0;
        }
        let len = self.points.len() * self.dim as usize;
        if len == 0 {
            return 0;
        }
        self.resident.get_or_init(|| {
            let reservation = super::memory::column_budget().reserve(len.checked_mul(4)?)?;
            let values = match &self.floats {
                // SAFETY: checked 4-byte aligned at open; `len` floats long.
                Floats::Mapped(s) => unsafe { std::slice::from_raw_parts(s.as_slice().as_ptr() as *const f32, len) }.to_vec(),
                Floats::Owned(v) => v.clone(),
            };
            Some(CachedFloats { values, _reservation: reservation })
        }).as_ref().map_or(0, |v| v.values.len() * 4)
    }

    /// Heap bytes of resident original/normalized columns and script magnitudes.
    pub fn resident_bytes(&self) -> usize {
        self.resident.get().and_then(Option::as_ref).map_or(0, |v| v.values.len() * 4)
            + self.normalized.get().and_then(Option::as_ref).map_or(0, |v| v.values.len() * 4)
            + self.original_norms.get().and_then(Option::as_ref).map_or(0, |v| v.values.len() * 4)
    }

    /// Load everything a search touches into RAM: the decoded graph, the
    /// quantized scoring index of `quant` (built from the stored column),
    /// and, when a graph is traversed in `f32`, the `f32` column.
    /// Idempotent; a no-op on appendable chunks.
    pub fn warm(&self, quant: Quantization) {
        if !self.is_sealed() {
            return;
        }
        let q = self.quantized(quant);
        if self.graph().is_some() && q.is_none() {
            self.make_resident();
            if self.similarity == Similarity::Cosine {
                let _ = self.squared_norms();
                let _ = self.normalized_contiguous();
            }
        }
    }

    /// The int8 quantized copy, if this segment was written with one.
    pub fn int8(&self) -> Option<Int8Vectors<'_>> {
        Int8Vectors::open(self.int8.as_ref()?.as_slice()).ok()
    }

    /// The packed int4 quantized copy, if this segment was written with one.
    pub fn int4(&self) -> Option<Int4Vectors<'_>> {
        Int4Vectors::open(self.int4.as_ref()?.as_slice()).ok()
    }

    /// Whether a stored int4 copy is present (ES `int4_*`).
    pub fn has_int4(&self) -> bool {
        self.int4.is_some()
    }

    /// The quantization this segment stores codes for (`None` when it only
    /// holds `f32`).
    pub fn stored_quantization(&self) -> Quantization {
        if self.int4.is_some() {
            Quantization::Int4
        } else if self.int8.is_some() {
            Quantization::Int8
        } else {
            Quantization::None
        }
    }

    /// The HNSW graph (decoded once, then cached). `None` for appendable chunks.
    pub fn graph(&self) -> Option<Arc<HnswGraph>> {
        let bytes = self.graph_bytes.as_ref()?;
        if let Some(g) = self.graph.get() {
            return Some(g.clone());
        }
        let g = Arc::new(HnswGraph::open(bytes.as_slice()).ok()?);
        Some(self.graph.get_or_init(|| g).clone())
    }

    /// Note the field's `confidence_interval` for an int4 copy fitted in
    /// memory (first call wins; the copy is built once).
    pub(crate) fn note_int4_confidence_interval(&self, ci: Option<f32>) {
        let _ = self.int4_ci.set(ci);
    }

    /// Whether a stored int8 copy is present (ES `int8_hnsw`).
    pub fn has_int8(&self) -> bool {
        self.int8.is_some()
    }

    /// The in-memory quantized scoring index of `kind` (built once, then
    /// cached). `Auto` resolves to int4 when the segment stores an int4
    /// copy, else int8 when it stores an int8 copy; `None` (and `Auto`
    /// without either) returns `None`.
    pub fn quantized(&self, kind: Quantization) -> Option<Arc<QuantizedVectors>> {
        match self.resolve(kind) {
            Quantization::Int8 => Some(
                self.q_int8
                    .get_or_init(|| {
                        Arc::new(match self.int8() {
                            Some(col) if col.num_docs() == self.len() && col.dim() == self.dim => {
                                QuantizedVectors::from_int8_column(&col, self, self.similarity)
                            }
                            _ => QuantizedVectors::int8(self, self.similarity),
                        })
                    })
                    .clone(),
            ),
            Quantization::Int4 => Some(
                self.q_int4
                    .get_or_init(|| {
                        Arc::new(match self.int4() {
                            Some(col) if col.num_docs() == self.len() && col.dim() == self.dim => {
                                QuantizedVectors::from_int4_column(&col, self, self.similarity)
                            }
                            // No stored column (an appendable chunk, or written
                            // before int4 columns): fit int4 in memory, never a
                            // silently different (int8) scoring.
                            _ => QuantizedVectors::int4(self, self.similarity, self.int4_ci.get().copied().flatten()),
                        })
                    })
                    .clone(),
            ),
            Quantization::Binary => {
                Some(self.q_binary.get_or_init(|| Arc::new(QuantizedVectors::binary(self, self.similarity))).clone())
            }
            Quantization::Auto | Quantization::None => None,
        }
    }

    fn resolve(&self, kind: Quantization) -> Quantization {
        match kind {
            Quantization::Auto if self.int4.is_some() => Quantization::Int4,
            Quantization::Auto if self.int8.is_some() => Quantization::Int8,
            Quantization::Auto => Quantization::None,
            k => k,
        }
    }

    /// Exact top-`k` over the given ordinals. Returns `(ord, score)`,
    /// score descending, ties by ordinal. SIMD brute force with a bounded
    /// heap (no per-candidate allocation).
    pub fn search_exact(&self, query: &[f32], k: usize, ords: impl Iterator<Item = u32>) -> Vec<(u32, f32)> {
        let n = self.len();
        if k == 0 { return Vec::new(); }
        let ords: Vec<u32> = ords.filter(|&o| o < n).collect();
        // POC only: avoid scanning the entire column to populate norms before
        // the first exact query, especially when ords is a selective subset.
        #[cfg(feature = "payload-hnsw-poc")]
        let lazy_norms = std::env::var_os("RUSTIC_POC_LAZY_EXACT_NORMS").is_some();
        #[cfg(not(feature = "payload-hnsw-poc"))]
        let lazy_norms = false;
        let norms = if self.similarity == Similarity::Cosine {
            if lazy_norms { self.squared_norms.get().map(Vec::as_slice) } else { self.squared_norms() }
        } else { None };
        // Never join the indexing pool while the caller holds a shard or
        // point-table read lock. Indexing-pool callers stay sequential;
        // nested work in the dedicated query pool can share its workers.
        if ords.len() < 4096 || (rayon::current_thread_index().is_some()
            && !crate::query::searcher::on_search_pool_worker()) {
            let (hits, scanned) = super::flat::exact_top_k_uncounted(query, self.similarity, k,
                ords.into_iter().map(|o| (o, self.vector(o))), norms);
            super::hnsw::add_vector_ops(scanned);
            return hits;
        }
        use rayon::prelude::*;
        let parts: Vec<_> = crate::query::searcher::in_search_pool(|| ords.par_chunks(1024).map(|chunk| {
            super::flat::exact_top_k_uncounted(query, self.similarity, k,
                chunk.iter().map(|&o| (o, self.vector(o))), norms)
        }).collect());
        let mut hits = Vec::new();
        let mut scanned = 0;
        for (part, operations) in parts {
            hits.extend(part);
            scanned += operations;
        }
        super::hnsw::add_vector_ops(scanned);
        top_k(&mut hits, k);
        hits
    }

    /// HNSW top-`k` with `accept` applied during traversal (rejected nodes
    /// are still expanded for connectivity). Falls back to nothing on an
    /// appendable chunk; callers check [`is_sealed`](Self::is_sealed).
    pub fn search_graph(&self, query: &[f32], k: usize, ef: usize, accept: impl Fn(u32) -> bool) -> Vec<(u32, f32)> {
        let Some(g) = self.graph() else {
            return Vec::new();
        };
        self.make_resident();
        g.search(query, k, ef, self, self.similarity, accept)
    }

    /// [`search_graph`](Self::search_graph) scoring the traversal with the
    /// `quant` index and rescoring the best `k * oversample` with `f32`.
    pub fn search_graph_with(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        quant: Quantization,
        oversample: f32,
        accept: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        self.search_graph_mode(query, k, ef, quant, oversample, accept, FilterMode::Plain)
    }

    /// [`VectorSegment::search_graph_with`] for a selective filter: the
    /// traversal scores only accepted nodes, reaching them through rejected
    /// neighbours (two-hop expansion, Lucene's filtered HNSW searcher).
    #[allow(clippy::too_many_arguments)]
    pub fn search_graph_mode(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        quant: Quantization,
        oversample: f32,
        accept: impl Fn(u32) -> bool,
        mode: FilterMode,
    ) -> Vec<(u32, f32)> {
        let Some(g) = self.graph() else {
            return Vec::new();
        };
        match self.quantized(quant) {
            // Traversal scores the in-memory codes; `f32` is read only for
            // the `k * oversample` rescore.
            Some(q) => g.search_quantized_mode(query, k, ef, oversample, self, self.similarity, &q, accept, mode),
            None => {
                self.make_resident();
                g.search_mode(query, k, ef, self, self.similarity, accept, mode)
            }
        }
    }

    /// Brute force over `ords` scoring with the `quant` index (`None`, or
    /// `Auto` without a stored copy, scores exact `f32`): the best
    /// `ceil(k * oversample)` by quantized distance are rescored with `f32`
    /// and the top `k` returned; with `oversample <= 0` the quantized
    /// estimates are the scores (ES `int8_flat` / `int4_flat` / `bbq_flat`
    /// without `rescore_vector`). Used for flat sealed segments and for the
    /// planner's exact-search path on sealed segments. `(ord, score)`,
    /// score descending, ties by ordinal.
    pub fn search_flat(&self, query: &[f32], k: usize, quant: Quantization, oversample: f32, ords: impl Iterator<Item = u32>) -> Vec<(u32, f32)> {
        let Some(q) = self.quantized(quant) else {
            return self.search_exact(query, k, ords);
        };
        if k == 0 {
            return Vec::new();
        }
        let n = self.len();
        let pq = q.prepare(query);
        // ES `OVERSAMPLE_LIMIT`: at most 10,000 rescored candidates (never below `k`).
        let keep = if oversample <= 0.0 { k } else { ((k as f64 * f64::from(oversample.max(1.0))).ceil().min(10_000.0) as usize).max(k) };
        // Bounded max-heap of packed (distance, ord) keys: the worst kept
        // candidate on top.
        let mut heap: std::collections::BinaryHeap<u64> = std::collections::BinaryHeap::with_capacity(keep + 1);
        let mut scanned = 0u64;
        // The ordinals of a filter are scattered over the segment: each row
        // (codes and correction terms) is a set of cache misses. Issue them
        // `AHEAD` rows early, so they overlap instead of serializing.
        const AHEAD: usize = 12;
        let ords: Vec<u32> = ords.filter(|&o| o < n).collect();
        for &o in ords.iter().take(AHEAD) { q.prefetch(o); }
        for (i, &o) in ords.iter().enumerate() {
            if let Some(&next) = ords.get(i + AHEAD) { q.prefetch(next); }
            scanned += 1;
            let d = q.distance(&pq, o);
            if d.is_nan() {
                continue;
            }
            let key = super::hnsw::pack(d, o);
            if heap.len() < keep {
                heap.push(key);
            } else if key < *heap.peek().unwrap() {
                heap.pop();
                heap.push(key);
            }
        }
        let mut cands: Vec<u64> = heap.into_vec();
        cands.sort_unstable();
        // Brute force over the codes, plus the `f32` rescoring below.
        super::hnsw::add_vector_ops(scanned + if oversample <= 0.0 { 0 } else { cands.len() as u64 });
        let mut out: Vec<(u32, f32)> = if oversample <= 0.0 {
            cands
                .iter()
                .map(|&c| {
                    let (d, o) = super::hnsw::unpack(c);
                    (o, self.similarity.score_of_distance(d).max(0.0))
                })
                .collect()
        } else {
            cands
                .iter()
                .map(|&c| {
                    let o = super::hnsw::unpack(c).1;
                    (o, self.similarity.score(query, self.vector(o)))
                })
                .collect()
        };
        top_k(&mut out, k);
        out
    }
}

impl VectorSource for VectorSegment {
    fn normalized_contiguous(&self) -> Option<&[f32]> {
        if self.similarity != Similarity::Cosine || self.dim == 0 { return None }
        self.normalized.get_or_init(|| {
            let data = self.as_contiguous()?;
            let reservation = super::memory::column_budget().reserve(data.len().checked_mul(4)?)?;
            let norms: Vec<f32> = data.chunks_exact(self.dim as usize).map(super::distance::norm).collect();
            // Preserve the existing cosine kernel for non-finite norms.
            if norms.iter().any(|n| !n.is_finite()) { return None }
            let mut normalized = Vec::with_capacity(data.len());
            for (row, n) in data.chunks_exact(self.dim as usize).zip(norms) {
                normalized.extend(row.iter().map(|v| if n > 0.0 { *v / n } else { 0.0 }));
            }
            Some(CachedFloats { values: normalized, _reservation: reservation })
        }).as_ref().map(|cached| cached.values.as_slice())
    }
    fn squared_norms(&self) -> Option<&[f32]> {
        Some(self.squared_norms.get_or_init(|| (0..self.len()).map(|o| {
            let vector = self.vector(o);
            super::distance::dot_and_sq_norm(vector, vector).1
        }).collect()))
    }
    fn dim(&self) -> u32 {
        self.dim
    }
    fn num_docs(&self) -> u32 {
        self.len()
    }
    fn get(&self, doc: DocId) -> Option<&[f32]> {
        ((doc as usize) < self.points.len()).then(|| self.vector(doc))
    }
    fn as_contiguous(&self) -> Option<&[f32]> {
        let len = self.points.len() * self.dim as usize;
        if let Some(v) = self.resident.get().and_then(Option::as_ref) {
            return Some(&v.values[..len]);
        }
        Some(match &self.floats {
            // SAFETY: checked 4-byte aligned at open; `len` floats long.
            Floats::Mapped(s) => unsafe { std::slice::from_raw_parts(s.as_slice().as_ptr() as *const f32, len) },
            Floats::Owned(v) => &v[..len],
        })
    }
}

/// Keep the best `k` of `(id, score)` sorted by score desc, then id asc.
pub(crate) fn top_k<T: Ord + Copy>(v: &mut Vec<(T, f32)>, k: usize) {
    let cmp = |a: &(T, f32), b: &(T, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if v.len() > k && k > 0 {
        v.select_nth_unstable_by(k - 1, cmp);
        v.truncate(k);
    }
    if k == 0 {
        v.clear();
    }
    v.sort_by(cmp);
}

/// Ordering helper shared by search code: score desc, then key asc.
pub(crate) fn by_score<T: Ord>(a: &(T, f32), b: &(T, f32)) -> Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::directory::RamDirectory;


    #[test]
    fn durable_payload_write_and_reopen_never_constructs_a_graph() {
        let dir = RamDirectory::new();
        let rows = [(10, [1.0f32, 2.0]), (20, [3.0, 4.0])];
        let refs: Vec<_> = rows.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        for (id, spec) in [(1, SealSpec::chunk()), (2, SealSpec { kind: SealKind::Flat,
            quantize: Quantization::None, confidence_interval: None })] {
            let before = super::super::hnsw::HnswBuilder::builds_on_this_thread();
            let segment = VectorSegment::write_with(&dir, id, 4, Similarity::L2, 2, &refs, spec).unwrap();
            assert!(!segment.has_graph());
            assert_eq!(super::super::hnsw::HnswBuilder::builds_on_this_thread(), before);
            let reopened = VectorSegment::open(&dir, id).unwrap();
            assert_eq!(reopened.points(), &[10, 20]);
            assert_eq!(reopened.vector(1), &[3.0, 4.0]);
            assert_eq!(reopened.search_exact(&[3.0, 4.0], 1, 0..2)[0].0, 1);
        }
    }

    #[test]
    fn script_magnitudes_preserve_original_norm_bits_and_account_for_cache() {
        let dir = RamDirectory::new();
        for dim in [3, 17, 128] {
            let rows: Vec<(u64, Vec<f32>)> = (0..8).map(|i| (i, (0..dim).map(|j| {
                if i == 0 { 0.0 } else { (((i as usize * 37 + j * 13) % 101) as f32 - 50.0) / 17.0 }
            }).collect())).collect();
            let refs: Vec<_> = rows.iter().map(|(p, v)| (*p, v.as_slice())).collect();
            let seg = VectorSegment::write_with(&dir, dim as u32, 0, Similarity::Cosine,
                dim as u32, &refs, SealSpec::chunk()).unwrap();
            for _ in 0..2 {
                for (ord, (_, row)) in rows.iter().enumerate() {
                    assert_eq!(seg.original_norm(ord as u32).to_bits(), super::super::distance::norm(row).to_bits());
                }
            }
            let expected = if seg.original_norms.get().unwrap().is_some() { rows.len() * 4 } else { 0 };
            assert_eq!(seg.resident_bytes(), expected);
        }
    }

    /// `bit` / `byte` rows are stored as their bytes (`dims / 8` / `dims`
    /// per row, here 1024 bits and 768 bytes), keep every bit pattern across
    /// a write and reopen, and score exactly as their unpacked values.
    #[test]
    fn packed_bit_and_byte_rows_store_their_bytes_and_score_as_unpacked() {
        use super::super::packed;
        let dir = RamDirectory::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for (id, similarity, bytes) in [(1u32, Similarity::BitHamming, 1024 / 8), (2, Similarity::ByteL2, 768),
            (3, Similarity::ByteCosine, 768), (4, Similarity::ByteDotProduct, 768), (5, Similarity::ByteMaxInnerProduct, 768)] {
            let raw: Vec<Vec<u8>> = (0..64).map(|_| (0..bytes).map(|_| next() as u8).collect()).collect();
            let rows: Vec<(u64, Vec<f32>)> = raw.iter().enumerate().map(|(i, r)| (i as u64, packed::pack(r))).collect();
            let refs: Vec<_> = rows.iter().map(|(p, v)| (*p, v.as_slice())).collect();
            VectorSegment::write_with(&dir, id, 0, similarity, packed::words(bytes) as u32, &refs, SealSpec::chunk()).unwrap();
            let file = crate::codec::vectors::open_vector_segment(
                dir.open_read(&crate::codec::vector_segment_file_name(id)).unwrap(), true).unwrap();
            assert_eq!(file.vectors.len(), rows.len() * bytes, "{similarity:?}: one byte per value, or per 8 bits");
            let segment = VectorSegment::open(&dir, id).unwrap();
            for (ord, (_, row)) in rows.iter().enumerate() {
                assert!(packed::same(segment.vector(ord as u32), row), "{similarity:?} row {ord} changed");
                assert_eq!(packed::as_bytes(segment.vector(ord as u32)), raw[ord].as_slice());
            }
            let unpack = |r: &[f32]| if similarity == Similarity::BitHamming { packed::unpack_bits(r, bytes * 8) } else { packed::unpack_i8(r, bytes) };
            for q in [0usize, 5, 63] {
                let query = &rows[q].1;
                let got = segment.search_exact(query, 10, 0..rows.len() as u32);
                let mut want: Vec<(u32, f32)> = rows.iter().enumerate()
                    .map(|(o, (_, r))| (o as u32, similarity.unpacked().score(&unpack(query), &unpack(r)))).collect();
                want.sort_by(by_score);
                want.truncate(10);
                assert_eq!(got, want, "{similarity:?} query {q}");
            }
        }
    }








}

#[cfg(test)]
mod workspace_admission_tests {
    use super::*;
    #[test]
    fn quantized_cache_admits_once_and_follows_last_reader() {
        let dir=crate::codec::directory::RamDirectory::new();
        let values=[[1.0f32,2.0],[3.0,4.0]];
        let rows:Vec<_>=values.iter().enumerate().map(|(i,v)|(i as u64,v.as_slice())).collect();
        let segment=VectorSegment::write_with(&dir,1,0,Similarity::L2,2,&rows,SealSpec::chunk()).unwrap();
        let spec=super::super::es_quant::EsQuant {bits:7,confidence_interval:None,osq:false};
        let tiny=crate::admission::Budget::new(1);
        assert!(segment.es_codes_admitted(&spec,&tiny).is_err());
        assert_eq!(tiny.used(),0);
        assert!(segment.es_quant.lock().is_empty());
        let budget=crate::admission::Budget::new(1<<20);
        let readers=std::thread::scope(|scope| {
            let handles:Vec<_>=(0..8).map(|_| {let segment=&segment;let budget=&budget;let spec=&spec;
                scope.spawn(move||segment.es_codes_admitted(spec,budget).unwrap())}).collect();
            handles.into_iter().map(|handle|handle.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(readers.iter().all(|value|Arc::ptr_eq(value,&readers[0])));
        assert_eq!(segment.es_quant.lock().len(),1);
        let held=budget.used();assert!(held>0);
        drop(segment);assert_eq!(budget.used(),held);
        drop(readers);assert_eq!(budget.used(),0);
    }

    #[test]
    fn payload_workspace_rejects_before_copy_and_shares_retained_capacity() {
        let budget=crate::admission::Budget::new(1<<20);
        assert!(payload_workspace(&budget,10000,96,Quantization::Int4).is_err());
        assert_eq!(budget.used(),0);
        let retained=budget.reserve(900000,"retained query cache").unwrap();
        assert!(payload_workspace(&budget,100,96,Quantization::Int4).is_err());
        drop(retained);
        let encoding=payload_workspace(&budget,100,96,Quantization::Int4).unwrap();
        assert!(budget.used()>0);drop(encoding);assert_eq!(budget.used(),0);
    }
}
