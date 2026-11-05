//! Immutable calibration epochs. Calibration is bounded write-side work;
//! queries cache each encountered epoch once and never hold a writer lock.
use super::*;
use crate::vector::es_quant::{es_unit, EsCodes, EsQuant, EsQuantScorer, EsQuantizer};
use crate::vector::osq::{OsqQuantizer, OsqSim};
use crate::vector::Similarity;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Admissions between refreshes of the cached score quantizer's centroid.
const SCORE_REFRESH_ROWS: u64 = 256;

pub(super) struct Epoch {
    pub id: u64,
    pub q: EsQuantizer,
    live: AtomicUsize,
    _memory: crate::admission::Reservation,
}

pub(crate) struct EncodedRow {
    pub(super) epoch: Arc<Epoch>,
    pub(super) words: Vec<u32>,
    pub(super) _memory: Option<Arc<crate::admission::Reservation>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Calibration {
    spec: EsQuant,
    current: u64,
    seen: u64,
    since_fit: usize,
    next_warmup: usize,
    samples: Vec<Vec<f32>>,
    /// Sum (cosine: of the ES unit vectors) and count of every admitted row:
    /// the centroid returned scores are estimated with (B5). Absent in
    /// checkpoints written before it existed (then the current epoch scores).
    #[serde(default)]
    sum: Vec<f64>,
    #[serde(default)]
    count: u64,
    /// Admissions when the current epoch was fitted (absent in older
    /// checkpoints: 0, the previous fixed schedule until the next fit).
    #[serde(default)]
    fitted_seen: u64,
}

/// Fewest admissions between refits after warmup.
const REFIT_MIN: usize = 4096;
/// After warmup, refit only once the admissions since the last fit reach
/// this multiple of the admissions at that fit. Every row is encoded with
/// the epoch current at its admission, and a query is quantized once per
/// epoch it meets, so a fixed refit interval made query cost grow with the
/// field (hundreds of epochs at millions of rows). A geometric schedule keeps
/// the epoch count logarithmic; the fit's 256-row reservoir sample does not
/// improve by refitting more often.
const REFIT_GROWTH: u64 = 8;
/// Relative centroid drift ([`QuantizedOwner::centroid_drift`]) below which
/// a scheduled refit is skipped.
const SETTLED: f64 = 1e-3;
/// Relative centroid drift that refits before the geometric schedule.
const DRIFT_REFIT: f64 = 1e-2;
/// Whether a drifting mean refits early (`RUSTIC_OSQ_DRIFT_REFIT=0` turns
/// it off).
fn drift_refits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUSTIC_OSQ_DRIFT_REFIT").map_or(true, |v| v != "0"))
}
/// Live epochs up to which drift refits early.
const DRIFT_EPOCHS: usize = 16;

pub(super) struct QuantizedOwner {
    budget: Arc<crate::admission::Budget>,
    calibration: parking_lot::Mutex<Calibration>,
    epochs: parking_lot::RwLock<BTreeMap<u64, Arc<Epoch>>>,
    /// The score quantizer as of the last calibration refit (see
    /// [`QuantizedOwner::score_quantizer`]): queries read it without the
    /// calibration mutex, allocation or any row.
    score: parking_lot::RwLock<Option<EsQuantizer>>,
    /// Admissions counted in the cached score quantizer's centroid.
    score_count: std::sync::atomic::AtomicU64,
    /// Admissions at the last early drift check (see `prepare`).
    drift_checked: std::sync::atomic::AtomicU64,
    dim: usize,
    similarity: Similarity,
    // Rust drops fields in declaration order: release calibration/score
    // storage before returning its capacity to concurrent admissions.
    _memory: crate::admission::Reservation,
}

impl QuantizedOwner {
    /// Test seam: a scorer of `query` for codes of epoch `id`.
    #[cfg(test)]
    pub(crate) fn scorer_for_test(&self, id: u64, query: &[f32]) -> Option<crate::vector::es_quant::EsQuantScorer> {
        Some(crate::vector::es_quant::EsQuantScorer::new(self.get(id)?.q.clone(), self.similarity, query))
    }
    pub fn new(spec: EsQuant, dim: usize, similarity: Similarity) -> Result<Self> {
        Self::with_budget(spec,dim,similarity,crate::admission::shared())
    }
    fn with_budget(spec:EsQuant,dim:usize,similarity:Similarity,budget:Arc<crate::admission::Budget>)->Result<Self> {
        let memory=budget.reserve((dim as u64).saturating_mul(256*16).saturating_add(65536),"mutable quantizer calibration")?;
        Ok(Self { budget, _memory:memory, calibration: parking_lot::Mutex::new(Calibration { spec, current: 0, seen: 0,
            since_fit: 0, next_warmup: 32, samples: Vec::new(), sum: vec![0.0; dim], count: 0, fitted_seen: 0 }),
            epochs: parking_lot::RwLock::new(BTreeMap::new()), score: parking_lot::RwLock::new(None), score_count: Default::default(), drift_checked: Default::default(), dim, similarity })
    }
    pub fn spec(&self) -> EsQuant { self.calibration.lock().spec }

    // At most 256 sampled rows and one immutable fitted epoch per 4096
    // admissions, apart from bounded bootstrap at 1,32,128,256 rows. Batch
    // ingestion supplies its sample before any graph identity is published.
    pub fn prepare(&self, rows: &[&[f32]]) -> Result<Vec<EncodedRow>> {
        if rows.iter().any(|v| v.len() != self.dim || v.iter().any(|x| !x.is_finite())) {
            return Err(failure("invalid quantized source row"));
        }
        if rows.is_empty() { return Ok(Vec::new()); }
        // Admit before calibration changes or code/word vectors are built.
        // One owner follows every prepared row until installation/cancellation.
        let prepared=Arc::new(self.budget.reserve((rows.len() as u64).saturating_mul((self.dim as u64).saturating_mul(32).saturating_add(256)).saturating_add(65536),"mutable quantizer prepared rows")?);
        let mut next_epoch=Some(self.budget.reserve((self.dim as u64).saturating_mul(64).saturating_add(4096),"mutable quantizer epoch")?);
        let epoch = {
            let mut state = self.calibration.lock();
            // A pre-centroid checkpoint has no sum: start one only from an
            // empty field, never from a partial history.
            let centroid = state.sum.len() == self.dim;
            for row in rows {
                if centroid {
                    let unit = if self.similarity == Similarity::Cosine { es_unit(row) } else { std::borrow::Cow::Borrowed(*row) };
                    for (a, x) in state.sum.iter_mut().zip(unit.iter()) { *a += f64::from(*x); }
                    state.count += 1;
                }
                state.seen = state.seen.checked_add(1).ok_or_else(|| failure("calibration admission overflow"))?;
                if state.samples.len() < 256 { state.samples.push(row.to_vec()); }
                else {
                    // Deterministic reservoir sampling; no random global state.
                    let mut hash = state.seen.wrapping_mul(0x9e3779b97f4a7c15);
                    hash ^= hash >> 30; hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
                    hash ^= hash >> 27;
                    let slot = hash % state.seen;
                    if slot < 256 { state.samples[slot as usize] = row.to_vec(); }
                }
            }
            state.since_fit = state.since_fit.saturating_add(rows.len());
            let refit_after = REFIT_MIN.max(usize::try_from(state.fitted_seen.saturating_mul(REFIT_GROWTH)).unwrap_or(usize::MAX));
            // Between scheduled refits, an OSQ field whose mean moved (data
            // indexed in topic order) refits every REFIT_MIN admissions, up
            // to DRIFT_EPOCHS live epochs (a query is quantized once per
            // epoch it meets).
            // Checked once per REFIT_MIN admissions: it reads the whole sample
            // under the calibration mutex every batch would serialize.
            let drifted = drift_refits() && state.since_fit >= REFIT_MIN && state.since_fit < refit_after && state.current != 0
                && state.seen >= self.drift_checked.load(Ordering::Relaxed) + REFIT_MIN as u64
                && self.epochs.read().len() < DRIFT_EPOCHS && {
                    self.drift_checked.store(state.seen, Ordering::Relaxed);
                    self.centroid_drift(&state).is_some_and(|d| d > DRIFT_REFIT)
                };
            let due = state.current == 0 || state.since_fit >= refit_after || state.samples.len() >= state.next_warmup || drifted;
            if due && state.current != 0 && self.centroid_settled(&state) {
                // Nothing to gain from a new epoch: keep the current one (a
                // query is quantized once per epoch it meets) and restart the
                // geometric schedule from here.
                state.since_fit = 0;
                state.fitted_seen = state.seen;
                state.next_warmup = match state.samples.len() { 0..=31 => 32, 32..=127 => 128, 128..=255 => 256, _ => usize::MAX };
            }
            if state.current == 0 || state.since_fit >= refit_after || state.samples.len() >= state.next_warmup || drifted {
                // OSQ quantizes around a centroid (ES: the mean of every vector
                // of the segment). The running mean of every admitted row is
                // that mean for one merged segment; a 256-row sample's mean is
                // noisy in high dimensions, which costs 1-bit codes the most
                // (their bits are the signs of the centered vector).
                let q = if state.spec.osq && state.count > 0 && state.sum.len() == self.dim {
                    let centroid = state.sum.iter().map(|s| (*s / state.count as f64) as f32).collect();
                    let query_bits = if state.spec.bits <= 4 { 4 } else { state.spec.bits };
                    EsQuantizer::Osq(Arc::new(OsqQuantizer::from_centroid(OsqSim::of(self.similarity), centroid, state.spec.bits, query_bits)))
                } else {
                    state.spec.fit(self.similarity, state.samples.len(), |i| &state.samples[i])
                };
                let id = state.current.checked_add(1).ok_or_else(|| failure("calibration epoch overflow"))?;
                let epoch = Arc::new(Epoch { id, q, live: AtomicUsize::new(0), _memory:next_epoch.take().expect("new epoch admission") });
                self.epochs.write().insert(id, epoch.clone());
                state.current = id;
                state.since_fit = 0;
                state.fitted_seen = state.seen;
                state.next_warmup = match state.samples.len() { 0..=31 => 32, 32..=127 => 128, 128..=255 => 256, _ => usize::MAX };
                self.refresh_score_quantizer(&state, &epoch.q);
                epoch
            } else {
                let epoch = self.epochs.read().get(&state.current).cloned().ok_or_else(|| failure("current calibration absent"))?;
                // The centroid follows admissions (O(dim) per 256 rows).
                if state.count >= self.score_count.load(Ordering::Relaxed) + SCORE_REFRESH_ROWS {
                    self.refresh_score_quantizer(&state, &epoch.q);
                }
                epoch
            }
        };
        // Encoding never holds the graph writer, epoch registry or calibration
        // mutex. Prepared rows retain the epoch through cancellation/admission.
        let codes = EsCodes::build(epoch.q.clone(), self.similarity, rows.len(), self.dim, |i| rows[i]);
        Ok((0..rows.len()).map(|i| EncodedRow { epoch: epoch.clone(), words: codes.row_words(i), _memory:Some(prepared.clone()) }).collect())
    }

    /// Whether the running mean of the admitted rows is still the current
    /// OSQ epoch's centroid, to within a negligible fraction of the rows'
    /// spread around it (the sampled rows' mean squared distance to it):
    /// a refit would then encode rows the same way. Lucene 9 quantiles and
    /// pre-centroid checkpoints always refit.
    fn centroid_settled(&self, state: &Calibration) -> bool {
        self.centroid_drift(state).is_some_and(|d| d <= SETTLED)
    }

    /// The squared distance between the running mean and the current OSQ
    /// epoch's centroid, relative to the sampled rows' mean squared distance
    /// to the running mean (`None`: not an OSQ field with a running mean).
    fn centroid_drift(&self, state: &Calibration) -> Option<f64> {
        if !state.spec.osq || state.count == 0 || state.sum.len() != self.dim || state.samples.is_empty() { return None; }
        let epoch = self.epochs.read().get(&state.current).cloned()?;
        let EsQuantizer::Osq(current) = &epoch.q else { return None };
        if current.centroid.len() != self.dim { return None; }
        let mean: Vec<f64> = state.sum.iter().map(|s| *s / state.count as f64).collect();
        let drift: f64 = mean.iter().zip(&current.centroid).map(|(m, c)| (m - f64::from(*c)).powi(2)).sum();
        let spread: f64 = state.samples.iter().map(|row| {
            let unit = if self.similarity == Similarity::Cosine { es_unit(row) } else { std::borrow::Cow::Borrowed(row.as_slice()) };
            unit.iter().zip(&mean).map(|(x, m)| (f64::from(*x) - m).powi(2)).sum::<f64>()
        }).sum::<f64>() / state.samples.len() as f64;
        (spread > 0.0).then(|| drift / spread)
    }

    /// The quantizer returned (unrescored) scores are estimated with. ES
    /// quantizes each Lucene segment around the centroid of all its vectors
    /// (`ES94ScalarQuantizedVectorsFormat` / `ES93BinaryQuantizedVectorsFormat`
    /// OSQ), so the estimate uses the running centroid of every row this
    /// field admitted: what one merged ES segment would hold. Calibration
    /// epochs (fitted on a bounded sample) drive graph traversal only.
    /// Lucene 9 scalar quantiles, and checkpoints without a centroid, keep
    /// the current epoch's quantizer.
    ///
    /// Refitted with each calibration epoch (every 4096 admissions, sooner
    /// while the field is small) and cached: the query path takes neither
    /// the calibration mutex (which admission holds while it sums rows) nor
    /// allocates. Deleted rows stay in the centroid, as a Lucene segment's
    /// centroid keeps its deleted documents until a merge.
    pub fn score_quantizer(&self) -> Option<EsQuantizer> {
        self.score.read().clone()
    }

    fn refresh_score_quantizer(&self, state: &Calibration, current: &EsQuantizer) {
        *self.score.write() = self.fit_score_quantizer(state, current);
        self.score_count.store(state.count, Ordering::Relaxed);
    }

    fn fit_score_quantizer(&self, state: &Calibration, current: &EsQuantizer) -> Option<EsQuantizer> {
        if !state.spec.osq || state.count == 0 || state.sum.len() != self.dim {
            return Some(current.clone());
        }
        let centroid = state.sum.iter().map(|s| (*s / state.count as f64) as f32).collect();
        // Queries: 4 bits against 1/2/4-bit codes, else the codes' own width
        // (as `EsQuant::fit`).
        let query_bits = if state.spec.bits <= 4 { 4 } else { state.spec.bits };
        Some(EsQuantizer::Osq(Arc::new(OsqQuantizer::from_centroid(OsqSim::of(self.similarity), centroid, state.spec.bits, query_bits))))
    }

    pub fn installed(&self, epoch: &Arc<Epoch>) { epoch.live.fetch_add(1, Ordering::Relaxed); }
    pub fn released(&self, id: u64) -> Result<()> {
        let epochs = self.epochs.read();
        let epoch = epochs.get(&id).ok_or_else(|| failure("reclaimed calibration absent"))?;
        epoch.live.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .map_err(|_| failure("calibration ownership underflow"))?;
        Ok(())
    }
    pub fn collect(&self) {
        let current = self.calibration.lock().current;
        self.epochs.write().retain(|id, epoch| *id == current || epoch.live.load(Ordering::Acquire) != 0 || Arc::strong_count(epoch) > 1);
    }
    pub fn memory_bytes(&self) -> usize {
        let state = self.calibration.lock();
        let samples = state.samples.capacity() * std::mem::size_of::<Vec<f32>>()
            + state.samples.iter().map(|v| v.capacity() * 4).sum::<usize>() + state.sum.capacity() * 8;
        samples + self.epochs.read().values().map(|epoch| std::mem::size_of::<Epoch>() + match &epoch.q {
            EsQuantizer::Lucene99(_) => 0,
            EsQuantizer::Osq(q) => std::mem::size_of_val(q.as_ref()) + q.centroid.capacity() * 4,
        }).sum::<usize>()
    }
    pub fn retain_epochs(&self) -> Result<crate::admission::Allocation<Vec<Arc<Epoch>>>> {
        let epochs=self.epochs.read();
        let bytes=epochs.len().checked_mul(std::mem::size_of::<Arc<Epoch>>()).ok_or_else(||failure("retained epoch slots overflow"))?;
        let memory=self.budget.reserve(bytes as u64,"checkpoint retained quantizer epochs")?;
        let mut values=Vec::new();values.try_reserve_exact(epochs.len()).map_err(|_|failure("retained epochs allocation"))?;
        values.extend(epochs.values().cloned());
        Ok(crate::admission::Allocation::new(values,memory))
    }
    pub fn get(&self, id: u64) -> Option<Arc<Epoch>> { self.epochs.read().get(&id).cloned() }

    // Capture is called while the source/graph publication fence is held.
    // The immutable epochs remain retained until the checkpoint has streamed.
    pub fn checkpoint(&self) -> Result<crate::admission::Allocation<Vec<u8>>> {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self,bytes:&[u8])->std::io::Result<usize>{self.0=self.0.checked_add(bytes.len()).ok_or_else(||std::io::Error::other("calibration length overflow"))?;Ok(bytes.len())}
            fn flush(&mut self)->std::io::Result<()>{Ok(())}
        }
        let calibration = self.calibration.lock();
        let mut count=Count(0);
        serde_json::to_writer(&mut count,&*calibration).map_err(crate::vector::hnsw::mutable::failure_error)?;
        let epochs = self.epochs.read();
        let mut length=count.0.checked_add(16).ok_or_else(||failure("calibration output length"))?;
        for epoch in epochs.values() {
            let codes=EsCodes::build(epoch.q.clone(),self.similarity,0,self.dim,|_|&[]);
            let encoded=codes.checkpoint_len()?;
            length=length.checked_add(16).and_then(|n|n.checked_add(encoded)).ok_or_else(||failure("calibration output length"))?;
        }
        let memory=self.budget.reserve(length as u64,"serialized calibration checkpoint")?;
        let mut out=Vec::new();out.try_reserve_exact(length).map_err(|_|failure("calibration checkpoint allocation"))?;
        put64(&mut out,count.0 as u64);out.resize(8+count.0,0);
        let mut destination=&mut out[8..];
        serde_json::to_writer(&mut destination,&*calibration).map_err(crate::vector::hnsw::mutable::failure_error)?;
        if !destination.is_empty(){return Err(failure("calibration serialization length changed"))}
        put64(&mut out,epochs.len() as u64);
        for (&id,epoch) in epochs.iter() {
            let codes=EsCodes::build(epoch.q.clone(),self.similarity,0,self.dim,|_|&[]);
            put64(&mut out,id);put64(&mut out,codes.checkpoint_len()? as u64);
            codes.append_checkpoint(&mut out)?;
        }
        debug_assert_eq!(out.len(),length);
        Ok(crate::admission::Allocation::new(out,memory))
    }
    pub fn restore(bytes: &[u8], dim: usize, similarity: Similarity) -> Result<Self> {
        let memory=crate::admission::shared().reserve((dim as u64).saturating_mul(256*16).saturating_add(65536),"mutable calibration restore")?;
        let _decode=crate::admission::shared().reserve((bytes.len() as u64).saturating_mul(64).saturating_add(65536),"mutable calibration decode")?;
        let mut input = Decoder { bytes, offset: 0 };
        let length = usize::try_from(input.u64()?).map_err(|_| corrupt("calibration length"))?;
        let calibration: Calibration = serde_json::from_slice(input.take(length)?).map_err(|_| corrupt("calibration state"))?;
        if calibration.samples.len() > 256 || calibration.samples.iter().any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))
            || !(calibration.sum.is_empty() || calibration.sum.len() == dim) || calibration.sum.iter().any(|x| !x.is_finite())
            || !matches!(calibration.spec.bits, 1 | 4 | 7) || (calibration.spec.bits == 1 && !calibration.spec.osq) {
            return Err(corrupt("calibration shape"));
        }
        let count = usize::try_from(input.u64()?).map_err(|_| corrupt("calibration count"))?;
        if count > input.remaining() / 16 { return Err(corrupt("calibration count exceeds input")); }
        let mut epochs = BTreeMap::new();
        for _ in 0..count {
            let id = input.u64()?;
            let length = usize::try_from(input.u64()?).map_err(|_| corrupt("calibration code length"))?;
            let epoch_memory=crate::admission::shared().reserve((dim as u64).saturating_mul(64).saturating_add(4096),"mutable restored epoch")?;
            let codes = EsCodes::from_checkpoint(input.take(length)?)?;
            if !codes.is_calibration_for(calibration.spec, dim, similarity) { return Err(corrupt("calibration format/dimension mismatch")); }
            if id == 0 || id > calibration.current || epochs.insert(id, Arc::new(Epoch { id, q: codes.q, live: AtomicUsize::new(0), _memory:epoch_memory })).is_some() {
                return Err(corrupt("calibration identity"));
            }
        }
        if input.remaining() != 0 || (calibration.current != 0 && !epochs.contains_key(&calibration.current)) {
            return Err(corrupt("calibration current/trailing bytes"));
        }
        let owner = Self { budget:crate::admission::shared(), _memory:memory, calibration: parking_lot::Mutex::new(calibration), epochs: parking_lot::RwLock::new(epochs), score: parking_lot::RwLock::new(None),
            score_count: Default::default(), drift_checked: Default::default(), dim, similarity };
        {
            let state = owner.calibration.lock();
            let current = owner.epochs.read().get(&state.current).map(|epoch| epoch.q.clone());
            if let Some(q) = current {
                owner.refresh_score_quantizer(&state, &q);
            }
        }
        Ok(owner)
    }
}

pub(super) struct BoundCodes<'a, F: Fn(NodeHandle, u64) -> Option<&'a [f32]>> {
    pub owner: &'a QuantizedOwner,
    pub query: Vec<f32>,
    pub row: F,
    /// One scorer per quantizer epoch a query meets: a handful at most, so a
    /// linear scan beats hashing on every scored row.
    pub cache: std::cell::RefCell<Vec<(u64, EsQuantScorer)>>,
    /// The query's quantizations shared with the other workers of one
    /// search (parallel exact scan or beam): an epoch is quantized once per
    /// query, by the first worker that meets it.
    pub shared: Option<&'a SharedQuantized>,
}
/// Per-query quantizations of the query, by calibration epoch, shared by a
/// search's workers.
pub(crate) type SharedQuantized = parking_lot::Mutex<Vec<(u64, EsQuantScorer)>>;
impl<'a, F: Fn(NodeHandle, u64) -> Option<&'a [f32]>> BoundCodes<'a, F> {
    /// The query quantized for epoch `id` (from the shared set when there
    /// is one, quantizing and publishing it there otherwise).
    fn quantized(&self, id: u64) -> Option<EsQuantScorer> {
        let make = || self.owner.get(id).map(|epoch| EsQuantScorer::new(epoch.q.clone(), self.owner.similarity, &self.query));
        let Some(shared) = self.shared else { return make() };
        let mut set = shared.lock();
        if let Some((_, scorer)) = set.iter().find(|(epoch, _)| *epoch == id) { return Some(scorer.clone()); }
        let scorer = make()?;
        set.push((id, scorer.clone()));
        Some(scorer)
    }
}
impl<'a, F: Fn(NodeHandle, u64) -> Option<&'a [f32]>> MutableQueryScorer for BoundCodes<'a, F> {
    fn score(&self, handle: NodeHandle, key: u64) -> Option<f32> {
        let row = (self.row)(handle, key)?;
        let id = u64::from(row[3].to_bits()) | (u64::from(row[4].to_bits()) << 32);
        let mut cache = self.cache.borrow_mut();
        let at = match cache.iter().position(|(epoch, _)| *epoch == id) {
            Some(at) => at,
            None => {
                cache.push((id, self.quantized(id)?));
                cache.len() - 1
            }
        };
        let start = super::codes_start(row, self.owner.dim);
        cache[at].1.score_words(row.get(start..)?, self.owner.dim)
    }
    fn quantized_state(&self) -> Option<Vec<(u64, EsQuantScorer)>> { Some(self.cache.borrow().clone()) }
}

/// [`BoundCodes`] with the owner's [`CodePlane`](super::code_plane::CodePlane):
/// traversal scores by node id (one computed address, four rows per dot
/// kernel); every other caller scores rows through `inner`.
pub(super) struct PlaneCodes<'a, F: Fn(NodeHandle, u64) -> Option<&'a [f32]>> {
    pub inner: BoundCodes<'a, F>,
    pub plane: &'a super::code_plane::CodePlane,
    pub stride: usize,
}
impl<'a, F: Fn(NodeHandle, u64) -> Option<&'a [f32]>> MutableQueryScorer for PlaneCodes<'a, F> {
    fn score(&self, handle: NodeHandle, key: u64) -> Option<f32> { self.inner.score(handle, key) }
    fn quantized_state(&self) -> Option<Vec<(u64, EsQuantScorer)>> { self.inner.quantized_state() }
    #[inline]
    fn prefetch_id(&self, id: u32) { self.plane.prefetch(id, self.stride); }
    fn score_entries(&self, entries: &[usize], out: &mut [f32]) -> bool {
        // SAFETY: the caller's contract (see the trait): each nonzero
        // address is a plane entry of `stride` words of an acquired node.
        self.score_with(entries.len(), out, |i| (entries[i] != 0)
            .then(|| unsafe { std::slice::from_raw_parts(entries[i] as *const f32, self.stride) }));
        true
    }
    fn score_ids(&self, ids: &[u32], out: &mut [f32]) -> bool {
        self.score_with(ids.len(), out, |i| self.plane.entry(ids[i], self.stride));
        true
    }
}
impl<'a, F: Fn(NodeHandle, u64) -> Option<&'a [f32]>> PlaneCodes<'a, F> {
    /// Scores of `count` plane entries (`entry_of(i)`, `None`: none) into
    /// `out` (NaN: no entry, score that node from its row).
    #[inline]
    fn score_with<'e>(&self, count: usize, out: &mut [f32], entry_of: impl Fn(usize) -> Option<&'e [f32]>) {
        let dim = self.inner.owner.dim;
        let mut cache = self.inner.cache.borrow_mut();
        // The epoch scorer of an entry: resolved once per epoch a call meets
        // (one, almost always), not per row.
        let scorer_of = |epoch: u64, cache: &mut Vec<(u64, EsQuantScorer)>| -> Option<usize> {
            if let Some(at) = cache.iter().position(|(e, _)| *e == epoch) { return Some(at); }
            cache.push((epoch, self.inner.quantized(epoch)?));
            Some(cache.len() - 1)
        };
        let entry = |i: usize| -> Option<(u64, &'e [f32])> {
            let e = entry_of(i)?;
            let epoch = u64::from(e[0].to_bits()) | (u64::from(e[1].to_bits()) << 32);
            (epoch != 0).then_some((epoch, &e[2..]))
        };
        let mut i = 0;
        while i < count {
            let len = (count - i).min(4);
            let mut entries: [Option<(u64, &[f32])>; 4] = [None; 4];
            for (j, slot) in entries.iter_mut().take(len).enumerate() { *slot = entry(i + j); }
            let first = entries[0].map(|e| e.0);
            let same = len == 4 && entries.iter().all(|e| e.is_some_and(|(epoch, _)| Some(epoch) == first));
            if same {
                let (epoch, _) = entries[0].unwrap();
                match scorer_of(epoch, &mut cache) {
                    Some(at) => {
                        let len = cache[at].1.words_len(dim);
                        let rows = [0, 1, 2, 3].map(|j| { let words = entries[j].unwrap().1; &words[..len.min(words.len())] });
                        for (o, s) in out[i..i + 4].iter_mut().zip(cache[at].1.score_words_four(rows, dim)) { *o = s.unwrap_or(f32::NAN); }
                    }
                    None => out[i..i + 4].fill(f32::NAN),
                }
            } else {
                for (j, e) in entries.iter().take(len).enumerate() {
                    out[i + j] = e.and_then(|(epoch, words)| {
                        let at = scorer_of(epoch, &mut cache)?;
                        let len = cache[at].1.words_len(dim);
                        cache[at].1.score_words(&words[..len.min(words.len())], dim)
                    }).unwrap_or(f32::NAN);
                }
            }
            i += len;
        }
    }
}

#[cfg(test)]
mod score_quantizer_tests {
    use super::*;
    #[test]
    fn calibration_checkpoint_refusal_and_escaped_output_keep_exact_owner() {
        let budget=crate::admission::Budget::new(1<<20);
        let owner=QuantizedOwner::with_budget(EsQuant{bits:7,osq:false,confidence_interval:None},8,Similarity::DotProduct,budget.clone()).unwrap();
        let retained=budget.used();budget.set_limit(retained);
        assert!(owner.checkpoint().is_err());assert_eq!(budget.used(),retained);
        budget.set_limit(1<<20);
        let checkpoint=owner.checkpoint().unwrap();
        assert_eq!(budget.used(),retained+checkpoint.len() as u64);
        drop(owner);assert_eq!(budget.used(),checkpoint.len() as u64);
        drop(checkpoint);assert_eq!(budget.used(),0);
    }

    fn rows(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| (0..dim).map(|d| ((i * 31 + d * 17) % 97) as f32 / 40.0 - 1.1).collect()).collect()
    }

    #[test]
    fn prepared_code_refusal_precedes_calibration_and_rows_keep_admission() {
        let spec=EsQuant {bits:7,confidence_interval:None,osq:false};
        let budget=crate::admission::Budget::new(1<<20);
        let owner=QuantizedOwner::with_budget(spec,4,Similarity::L2,budget.clone()).unwrap();
        let row=[1.0f32;4];
        let before=budget.used();
        assert!(owner.prepare(&vec![row.as_slice();10000]).is_err());
        assert_eq!(owner.calibration.lock().seen,0);
        assert_eq!(budget.used(),before);
        let mut prepared=owner.prepare(&[&row,&row]).unwrap();
        let last=prepared.pop().unwrap();drop(prepared);
        let held=budget.used();assert!(held>before);
        drop(owner);assert!(budget.used()>0,"prepared row owns code and epoch admission");
        drop(last);assert_eq!(budget.used(),0);
    }

    #[test]
    fn score_centroid_is_the_mean_of_every_admitted_row_and_survives_restore() {
        for similarity in [Similarity::L2, Similarity::Cosine, Similarity::MaxInnerProduct] {
            let spec = EsQuant { bits: 7, confidence_interval: None, osq: true };
            let owner = QuantizedOwner::new(spec, 8, similarity).unwrap();
            let data = rows(900, 8);
            // Batches of 97, then one that ends past a refresh of the
            // cached quantizer (at most every 256 admissions).
            for chunk in data[..582].chunks(97) {
                let refs: Vec<&[f32]> = chunk.iter().map(Vec::as_slice).collect();
                owner.prepare(&refs).unwrap();
            }
            let refs: Vec<&[f32]> = data[582..].iter().map(Vec::as_slice).collect();
            owner.prepare(&refs).unwrap();
            let unit = |v: &[f32]| if similarity == Similarity::Cosine { es_unit(v).into_owned() } else { v.to_vec() };
            let mean: Vec<f32> = (0..8).map(|d| (data.iter().map(|v| f64::from(unit(v)[d])).sum::<f64>() / data.len() as f64) as f32).collect();
            let Some(EsQuantizer::Osq(q)) = owner.score_quantizer() else { panic!("osq score quantizer") };
            for (a, b) in q.centroid.iter().zip(&mean) { assert!((a - b).abs() < 1e-5, "{similarity:?}: {a} vs {b}"); }
            // The traversal epoch is fitted on a bounded sample, not the whole field.
            let restored = QuantizedOwner::restore(&owner.checkpoint().unwrap(), 8, similarity).unwrap();
            let Some(EsQuantizer::Osq(r)) = restored.score_quantizer() else { panic!("restored") };
            assert_eq!(r.centroid, q.centroid);
        }
    }

    #[test]
    fn a_drifting_mean_refits_before_the_geometric_schedule() {
        let spec = EsQuant { bits: 7, confidence_interval: None, osq: true };
        let owner = QuantizedOwner::new(spec, 8, Similarity::L2).unwrap();
        // One fit over 5000 rows: the next scheduled refit is at 8 x 5000.
        let first = rows(5000, 8);
        owner.prepare(&first.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        assert_eq!(owner.calibration.lock().current, 1);
        // A shifted distribution (topic-ordered data) moves the mean.
        let shifted: Vec<Vec<f32>> = rows(6000, 8).into_iter().map(|v| v.into_iter().map(|x| x + 3.0).collect()).collect();
        for chunk in shifted.chunks(512) {
            owner.prepare(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        }
        let state = owner.calibration.lock();
        assert!(state.current > 1, "drift refit after {} admissions (scheduled at 40000)", state.seen);
        assert!(state.seen < 40_000);
    }

    #[test]
    fn a_settled_mean_keeps_its_epoch() {
        let spec = EsQuant { bits: 7, confidence_interval: None, osq: true };
        let owner = QuantizedOwner::new(spec, 8, Similarity::L2).unwrap();
        let data = rows(11000, 8);
        owner.prepare(&data[..5000].iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        for chunk in data[5000..].chunks(512) {
            owner.prepare(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        }
        assert_eq!(owner.calibration.lock().current, 1, "same distribution: no drift refit");
    }

    #[test]
    fn lucene99_specs_and_empty_owners_score_with_the_current_epoch() {
        let owner = QuantizedOwner::new(EsQuant { bits: 4, confidence_interval: None, osq: false }, 4, Similarity::L2).unwrap();
        assert!(owner.score_quantizer().is_none(), "no epoch yet");
        let data = rows(40, 4);
        let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
        owner.prepare(&refs).unwrap();
        assert!(matches!(owner.score_quantizer(), Some(EsQuantizer::Lucene99(_))));
    }
}
