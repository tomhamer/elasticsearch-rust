//! On-disk postings format (segment format v4).
//!
//! One list per term, little endian:
//!
//! ```text
//! vint  doc_freq
//! vint  total_term_freq
//! u8    flags      bit0 HAS_POSITIONS, bit1 HAS_FREQS (some tf > 1),
//!                  bit2 HAS_NORMS (min_norm is exact), bit3 DOC_GAPS,
//!                  bit4 IMPACTS (skip entries end with 4 impact bytes)
//! vint  num_full_blocks                     (= doc_freq / 128)
//! [positions] u32 pos_off                   (positions region, from blocks start)
//!             u32 tail_pos_ptr              (tail's positions, from pos_off)
//! skip table: num_full_blocks entries of 12 bytes (16 with positions)
//!     u32 last_doc      last doc id of the block
//!     u32 block_end     end of the block, from blocks start
//!     u16 max_tf        max tf in the block (0xFFFF = unbounded)
//!     u8  min_norm      min norm byte in the block (1 for no-norms fields,
//!                       0 when the writer did not know the norms)
//!     u8  reserved
//!     [u32 pos_ptr]     block's positions, from pos_off
//!     [u8 n1, u8 t1, u8 n2, u8 t2]   with IMPACTS: docs with norm < n1 have
//!                       tf <= t1, docs with n1 <= norm < n2 have tf <= t2
//!                       (255: up to max_tf); n1 = 0: no refinement
//! blocks (128 docs each):
//!     u8 doc_bits, u8 tf_bits
//!     16*doc_bits  BitPacker4x doc payload (see below)
//!     16*tf_bits   BitPacker4x `tf - 1` (== compress_sorted(0, C) with
//!                  C[i] = sum_{j<=i} (tf_j - 1), the cumulative extra tf)
//! tail (doc_freq % 128 docs, if any): a short bit-packed block
//!     u8 doc_bits, u8 tf_bits
//!     doc gaps (vs the previous doc, the first vs the last full block's
//!     last doc), sequentially packed at doc_bits, then `tf - 1` packed at
//!     tf_bits (each sequence starts on a byte boundary)
//! positions region (only with positions; never read by doc-only paths):
//!     per doc block (full blocks, then the tail): u8 bits, then the ABSOLUTE
//!     positions of every occurrence in the block (doc order, then position
//!     order), sequentially bit-packed (random access)
//!     8 zero bytes (`SEQ_PAD`)
//! ```
//!
//! Doc payload: frame of reference against the previous block's last doc
//! (`doc - prev_last`, BitPacker4x plain), or, with `DOC_GAPS`, d-gaps
//! decoded by BitPacker4x's fused SIMD prefix sum. FOR keeps O(1) random
//! access to any doc of a packed block ([`get4x`]), which the block
//! conjunction and the 2-clause disjunction probe without decoding; gaps are
//! smaller. The writer uses FOR, chosen on the bench harness (200k docs,
//! 1 segment, two interleaved rounds, FOR vs gaps p50 µs: term 6.4–6.6 vs
//! 7.0–7.3, and2 35–51 vs 36–38, or2 33–44 vs 34–39, all within noise on a
//! loaded machine; index 207 MB vs 199 MB). With latency a wash, FOR's
//! random access, which the AND/OR probes rely on, decides it. The reader
//! handles both.
//!
//! The tf payload stores `tf - 1`, which is byte-identical to the
//! sorted/delta packing of the cumulative `C[i]`. Random access gives one
//! doc's tf in O(1); one SIMD `decompress_sorted` gives `C`, from which doc
//! `i`'s positions start at index `i + C[i-1]` of the block's packed
//! positions, so a phrase decodes O(tf) positions per candidate. (Storing
//! `C` itself for O(1) random access was measured on the bench harness: no
//! phrase p50 gain over this layout once the seek path is lean, a worse
//! phrase mean, and slightly slower AND/OR tf reads, so it was not kept.)
//!
//! The skip table is fixed width, read lazily (nothing is parsed at open),
//! and carries block-max impacts (`max_tf`, `min_norm`) so top-k scorers can
//! skip blocks that cannot compete while still counting exactly. Alone the
//! two are loose wherever the block's highest tf is in a long doc and its
//! shortest doc has tf 1 (any long-text field): `score(max_tf, min_norm)`
//! then nears the term's saturation weight, 20-50% above any real score,
//! which in a sum of many such bounds keeps common terms essential. Lucene keeps each block's whole
//! competitive (freq, norm) frontier; IMPACTS keeps a 3-step staircase over
//! it ([`block_impacts`]), so the bound is the best of three corners.

use super::bitpack::{
    packed_len, bits_for, get4x, len4x, num_bits4x, num_bits_sorted4x, pack4x, pack_seq, pack_sorted4x, unpack4x,
    unpack_seq_at, unpack_sorted4x, SEQ_PAD,
};
use super::directory::FileSlice;
use super::varint::{read_vint, write_vint};
use crate::error::{Result, RusticError};
use crate::types::DocId;

pub const BLOCK_DOCS: usize = 128;
const FLAG_POSITIONS: u8 = 1;
const FLAG_FREQS: u8 = 2;
const FLAG_NORMS: u8 = 4;
const FLAG_DOC_GAPS: u8 = 8;
const FLAG_IMPACTS: u8 = 16;
/// Impact bytes at the end of a skip entry (with `FLAG_IMPACTS`).
const IMPACT_BYTES: usize = 4;
const SKIP_ENTRY: usize = 12;
const SKIP_ENTRY_POS: usize = 16;
/// Sentinel `last_doc` for the tail block.
const TAIL_LAST: u32 = u32::MAX - 1;
const NO_BLOCK: u32 = u32::MAX;

/// Decode `out.len()` values packed sequentially at `bits` from the start
/// of `data`, which may end right after the packed bytes (no `SEQ_PAD`
/// slack: the tail block can be the last thing in the postings section).
#[inline]
fn unpack_short(data: &[u8], bits: u8, out: &mut [u32]) {
    let need = packed_len(out.len(), bits as usize);
    if data.len() >= need + SEQ_PAD - 1 {
        unpack_seq_at(data, bits, 0, out);
        return;
    }
    let mut buf = [0u8; BLOCK_DOCS * 4 + SEQ_PAD];
    let m = need.min(data.len());
    buf[..m].copy_from_slice(&data[..m]);
    unpack_seq_at(&buf, bits, 0, out);
}

/// Whether the writer uses d-gaps (true) or frame-of-reference (false)
/// for every list. FOR stays the default (see the module docs); a dense
/// list switches to d-gaps per [`dense_doc_gaps`].
fn use_doc_gaps() -> bool {
    false
}

/// Whether `doc_ids` (ascending) packs into at most half the bits as d-gaps
/// than as frame-of-reference offsets over its full blocks: a term in most
/// docs (a keyword shared by a metrics stream, an ip every doc lists) takes
/// 8 bits a doc as offsets from the previous block's last doc but 1 bit as
/// gaps, as Lucene's dense postings blocks do. Readers decode either layout
/// (the list's `DOC_GAPS` flag, read since format v4); sparse lists, where
/// the two differ by a bit or two, keep FOR.
fn dense_doc_gaps(doc_ids: &[DocId]) -> bool {
    let nfull = doc_ids.len() / BLOCK_DOCS;
    if nfull == 0 {
        return false;
    }
    let (mut for_bits, mut gap_bits) = (0usize, 0usize);
    let mut prev = 0u32;
    let mut vals = [0u32; BLOCK_DOCS];
    for b in 0..nfull {
        let docs = &doc_ids[b * BLOCK_DOCS..(b + 1) * BLOCK_DOCS];
        for (v, &d) in vals.iter_mut().zip(docs) {
            *v = d - prev;
        }
        for_bits += num_bits4x(&vals) as usize;
        gap_bits += num_bits_sorted4x(prev, docs) as usize;
        prev = docs[BLOCK_DOCS - 1];
    }
    gap_bits * 2 <= for_bits
}

/// What the writer knows about the field's norms (for `min_norm`).
#[derive(Clone, Copy, Debug)]
pub enum BlockNorms<'a> {
    /// Unknown: `min_norm = 0`, a safe lower bound.
    Unknown,
    /// The field has no norms (every doc scores with norm byte 1).
    Absent,
    /// Per-doc norm bytes, indexed by doc id.
    Bytes(&'a [u8]),
}

/// Encode one postings list without positions. `doc_ids` must be ascending.
pub fn write_postings(doc_ids: &[DocId], term_freqs: &[u32], out: &mut Vec<u8>) {
    write_postings_full(doc_ids, term_freqs, None, out)
}

/// Encode one postings list. `positions`, if given, is the flattened
/// position list: doc `i`'s positions are the next `term_freqs[i]` values
/// (ascending within a doc).
pub fn write_postings_full(doc_ids: &[DocId], term_freqs: &[u32], positions: Option<&[u32]>, out: &mut Vec<u8>) {
    write_postings_with_norms(doc_ids, term_freqs, positions, BlockNorms::Unknown, out)
}

/// [`write_postings_full`] with the field's norms, so the skip table gets
/// exact per-block `min_norm` impacts. Reads the live
/// [`super::write_version`] per call: fine for a list encoded on its own
/// (a memtable's search view), never for a list that is part of a file,
/// whose writer passes its captured version to [`write_postings_impl`].
pub fn write_postings_with_norms(
    doc_ids: &[DocId],
    term_freqs: &[u32],
    positions: Option<&[u32]>,
    norms: BlockNorms<'_>,
    out: &mut Vec<u8>,
) {
    write_postings_impl(doc_ids, term_freqs, positions, norms, super::write_version() >= super::IMPACTS_SINCE, out)
}

/// [`write_postings_with_norms`] with impact staircases allowed or not (the
/// segment write version decides; older builds cannot read them).
pub(crate) fn write_postings_impl(
    doc_ids: &[DocId],
    term_freqs: &[u32],
    positions: Option<&[u32]>,
    norms: BlockNorms<'_>,
    allow_impacts: bool,
    out: &mut Vec<u8>,
) {
    write_postings_scratch(doc_ids, term_freqs, positions, norms, allow_impacts, true, out)
}

/// Merge workers own their scratch admission; do not retain position buffers
/// in thread locals after that ownership ends.
pub(crate) fn write_postings_uncached(doc_ids: &[DocId], term_freqs: &[u32], positions: Option<&[u32]>, norms: BlockNorms<'_>, allow_impacts: bool, out: &mut Vec<u8>) {
    write_postings_scratch(doc_ids, term_freqs, positions, norms, allow_impacts, false, out)
}

fn write_postings_scratch(doc_ids: &[DocId], term_freqs: &[u32], positions: Option<&[u32]>, norms: BlockNorms<'_>, allow_impacts: bool, retain_scratch: bool, out: &mut Vec<u8>) {
    assert_eq!(doc_ids.len(), term_freqs.len());
    if let Some(p) = positions {
        debug_assert_eq!(p.len() as u64, term_freqs.iter().map(|&t| t as u64).sum::<u64>());
    }
    let df = doc_ids.len();
    let ttf: u64 = term_freqs.iter().map(|&t| t as u64).sum();
    let gaps = use_doc_gaps() || dense_doc_gaps(doc_ids);
    let mut flags = 0u8;
    if positions.is_some() {
        flags |= FLAG_POSITIONS;
    }
    if ttf != df as u64 {
        flags |= FLAG_FREQS;
    }
    if matches!(norms, BlockNorms::Bytes(_)) {
        flags |= FLAG_NORMS;
    }
    if gaps {
        flags |= FLAG_DOC_GAPS;
    }
    let impacts = matches!(norms, BlockNorms::Bytes(_)) && df >= BLOCK_DOCS && allow_impacts;
    if impacts {
        flags |= FLAG_IMPACTS;
    }
    write_vint(df as u64, out);
    write_vint(ttf, out);
    out.push(flags);
    let nfull = df / BLOCK_DOCS;
    write_vint(nfull as u64, out);
    let pos_hdr = out.len();
    if positions.is_some() {
        out.extend_from_slice(&[0u8; 8]);
    }
    let base_entry = if positions.is_some() { SKIP_ENTRY_POS } else { SKIP_ENTRY };
    let entry = base_entry + if impacts { IMPACT_BYTES } else { 0 };
    let skip_start = out.len();
    out.resize(skip_start + nfull * entry, 0);
    let blocks_start = out.len();

    // The positions region is built aside, then appended: a per-thread
    // scratch buffer (not one allocation per term of a text field).
    // Without a retained scratch (merges), size it for the worst case once:
    // a merge admits that bound, and no doubling overshoots it.
    let mut pos_region: Vec<u8> = match positions {
        Some(_) if retain_scratch => POS_REGION.with(|r| std::mem::take(&mut *r.borrow_mut())),
        Some(p) => Vec::with_capacity(positions_region_bound(p.len() as u64, df as u64) as usize),
        None => Vec::new(),
    };
    pos_region.clear();
    let mut pos_cursor = 0usize;
    let mut write_block_positions = |tfs: &[u32], region: &mut Vec<u8>| -> u32 {
        let ptr = region.len() as u32;
        let all = positions.unwrap();
        let cnt: usize = tfs.iter().map(|&t| t as usize).sum();
        let vals = &all[pos_cursor..pos_cursor + cnt];
        pos_cursor += cnt;
        let bits = bits_for(vals.iter().copied().max().unwrap_or(0));
        region.push(bits);
        pack_seq(vals, bits, region);
        ptr
    };

    let mut prev = 0u32;
    let mut vals = [0u32; BLOCK_DOCS];
    for b in 0..nfull {
        let docs = &doc_ids[b * BLOCK_DOCS..(b + 1) * BLOCK_DOCS];
        let tfs = &term_freqs[b * BLOCK_DOCS..(b + 1) * BLOCK_DOCS];
        let doc_bits;
        let at = out.len();
        out.extend_from_slice(&[0, 0]);
        if gaps {
            doc_bits = num_bits_sorted4x(prev, docs);
            pack_sorted4x(prev, docs, doc_bits, out);
        } else {
            for (v, &d) in vals.iter_mut().zip(docs) {
                *v = d - prev;
            }
            doc_bits = num_bits4x(&vals);
            pack4x(&vals, doc_bits, out);
        }
        for (v, &t) in vals.iter_mut().zip(tfs) {
            *v = t.max(1) - 1;
        }
        let tf_bits = num_bits4x(&vals);
        pack4x(&vals, tf_bits, out);
        out[at] = doc_bits;
        out[at + 1] = tf_bits;

        let max_tf = tfs.iter().map(|&t| t.max(1)).max().unwrap_or(1);
        let min_norm = match norms {
            BlockNorms::Unknown => 0,
            BlockNorms::Absent => 1,
            BlockNorms::Bytes(n) => docs.iter().map(|&d| n.get(d as usize).copied().unwrap_or(0)).min().unwrap_or(0),
        };
        let e = skip_start + b * entry;
        let block_end = (out.len() - blocks_start) as u32;
        out[e..e + 4].copy_from_slice(&docs[BLOCK_DOCS - 1].to_le_bytes());
        out[e + 4..e + 8].copy_from_slice(&block_end.to_le_bytes());
        let mt: u16 = if max_tf >= 0xFFFF { 0xFFFF } else { max_tf as u16 };
        out[e + 8..e + 10].copy_from_slice(&mt.to_le_bytes());
        out[e + 10] = min_norm;
        if positions.is_some() {
            let ptr = write_block_positions(tfs, &mut pos_region);
            out[e + 12..e + 16].copy_from_slice(&ptr.to_le_bytes());
        }
        if let (true, BlockNorms::Bytes(n)) = (impacts, norms) {
            out[e + base_entry..e + entry].copy_from_slice(&block_impacts(docs, tfs, n, max_tf, min_norm));
        }
        prev = docs[BLOCK_DOCS - 1];
    }
    // Tail: a short bit-packed block (gaps, then tf - 1).
    let tail = nfull * BLOCK_DOCS..df;
    if !tail.is_empty() {
        let n = tail.len();
        let (mut gaps, mut extra) = ([0u32; BLOCK_DOCS], [0u32; BLOCK_DOCS]);
        for (k, i) in tail.enumerate() {
            gaps[k] = doc_ids[i] - prev;
            prev = doc_ids[i];
            extra[k] = term_freqs[i].max(1) - 1;
        }
        let db = bits_for(gaps[..n].iter().copied().max().unwrap_or(0));
        let tb = bits_for(extra[..n].iter().copied().max().unwrap_or(0));
        out.push(db);
        out.push(tb);
        pack_seq(&gaps[..n], db, out);
        pack_seq(&extra[..n], tb, out);
    }
    if positions.is_some() {
        let tail_ptr = if df > nfull * BLOCK_DOCS {
            write_block_positions(&term_freqs[nfull * BLOCK_DOCS..], &mut pos_region)
        } else {
            pos_region.len() as u32
        };
        let pos_off = (out.len() - blocks_start) as u32;
        out[pos_hdr..pos_hdr + 4].copy_from_slice(&pos_off.to_le_bytes());
        out[pos_hdr + 4..pos_hdr + 8].copy_from_slice(&tail_ptr.to_le_bytes());
        out.extend_from_slice(&pos_region);
        out.extend_from_slice(&[0u8; SEQ_PAD]);
        // Kept for the thread's next list unless one huge list grew it.
        if retain_scratch && pos_region.capacity() <= POS_REGION_KEEP {
            POS_REGION.with(|r| *r.borrow_mut() = pos_region);
        }
    }
}

/// Largest positions region of a list of `df` docs and `ttf` positions:
/// per doc block a bit-width byte, then at most 32 bits per position, plus
/// the 8 bytes `pack_seq` writes past its end before truncating.
pub(crate) fn positions_region_bound(ttf: u64, df: u64) -> u64 {
    ttf.saturating_mul(4).saturating_add(df / BLOCK_DOCS as u64 + 1).saturating_add(8)
}

/// Largest encoded list of `df` docs with total term frequency `ttf` in a
/// segment of `num_docs` docs: header, skip table, doc/tf blocks and, with
/// `positions`, the positions region (one position per unit of `ttf`).
/// A doc payload value (offset or gap from an earlier doc) is below
/// `num_docs`, a `tf - 1` at most `ttf - df`; positions take up to 32 bits.
/// A list without positions (a `rank_features` / sparse-vector field, whose
/// frequency carries a weight of up to `0x7F7F`) has no positions region:
/// its `ttf` is a sum of weights, not a position count.
pub(crate) fn encoded_postings_bound(df: u64, ttf: u64, num_docs: u64, positions: bool) -> u64 {
    let width = |max: u64| (64 - max.leading_zeros()) as u64;
    let (db, tb) = (width(num_docs), width(ttf.saturating_sub(df)));
    let (full, tail) = (df / BLOCK_DOCS as u64, df % BLOCK_DOCS as u64);
    // Header vints/flags/positions header, the 8-byte positions pad and
    // `pack_seq`'s 8-byte slack; per full block 2 width bytes, 16 words per
    // payload width bit and a skip entry of at most 20 bytes; the tail packs
    // its gaps and `tf - 1` sequentially after 2 width bytes.
    let blocks = full.saturating_mul(22u64.saturating_add(16 * (db + tb)));
    let tail = 2 + (tail * db).div_ceil(8) + (tail * tb).div_ceil(8);
    let region = if positions { positions_region_bound(ttf, df) } else { 0 };
    64u64.saturating_add(blocks).saturating_add(tail).saturating_add(region)
}

/// Largest positions scratch a thread keeps between lists.
const POS_REGION_KEEP: usize = 1 << 20;

thread_local! {
    /// [`write_postings_with_norms`]'s positions region scratch.
    static POS_REGION: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A block's impact staircase `[n1, t1, n2, t2]` (see the format notes):
/// of the frontier steps where the best tf so far grows (norms ascending),
/// the two that minimize the bound's worst corner under BM25 (k1 1.2,
/// b 0.75, the block's mean length standing in for the field's: the choice
/// only affects tightness, every corner is a true upper bound).
fn block_impacts(docs: &[DocId], tfs: &[u32], norms: &[u8], max_tf: u32, min_norm: u8) -> [u8; IMPACT_BYTES] {
    let lens = crate::query::bm25::norm_table();
    let mut best = [0u32; 256];
    let mut sum_len = 0f64;
    for (&d, &t) in docs.iter().zip(tfs) {
        let nb = norms.get(d as usize).copied().unwrap_or(0);
        // As the encoder writes it (a tf is at least 1).
        best[nb as usize] = best[nb as usize].max(t.max(1));
        sum_len += lens[nb as usize] as f64;
    }
    // Steps (norm, best tf below it) where the running best tf grows.
    let mut steps: Vec<(u8, u32)> = Vec::new();
    let mut run = 0u32;
    for nb in min_norm as usize..256 {
        if best[nb] > run {
            if nb > min_norm as usize {
                steps.push((nb as u8, run));
            }
            run = best[nb];
        }
    }
    if steps.is_empty() {
        // The shortest doc holds the highest tf: the corner is exact.
        return [0; IMPACT_BYTES];
    }
    let avg = (sum_len / docs.len() as f64).max(1.0);
    let f = |tf: u32, nb: u8| {
        let k = 1.2 * (0.25 + 0.75 * lens[nb as usize] as f64 / avg);
        tf as f64 / (tf as f64 + k)
    };
    let byte = |t: u32| if t >= 255 { 255 } else { t as u8 };
    let (mut pick, mut cost) = ((0, 0), f64::INFINITY);
    for a in 0..steps.len() {
        for b in a..steps.len() {
            // Below steps[a]: tf <= its run; [a, b): tf <= steps[b]'s run; from b: max_tf.
            let c = f(steps[a].1, min_norm).max(f(steps[b].1, steps[a].0)).max(f(max_tf, steps[b].0));
            if c < cost {
                (pick, cost) = ((a, b), c);
            }
        }
    }
    let (a, b) = (steps[pick.0], steps[pick.1]);
    [a.0, byte(a.1), b.0, byte(b.1)]
}

/// One skip-table entry (block-level metadata).
#[derive(Clone, Copy, Debug)]
pub struct SkipEntry {
    /// Last doc id of the block.
    pub last_doc: DocId,
    /// End offset of the block (from the blocks area start).
    pub block_end: u32,
    /// Max tf in the block (`u32::MAX` = unbounded).
    pub max_tf: u32,
    /// Min norm byte in the block.
    pub min_norm: u8,
    /// Impact staircase `[n1, t1, n2, t2]` (`n1 == 0`: none; see the
    /// format notes and `Bm25Weight::block_bound`).
    pub impacts: [u8; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Unstarted,
    InBlock,
    Exhausted,
}

fn corrupted(msg: &str) -> RusticError {
    RusticError::Corrupted(format!("postings: {msg}"))
}

/// A postings list read directly from a segment file slice.
///
/// Allocation-free on the doc path: the skip table is read lazily, a block is
/// decoded into fixed arrays, tfs are read by random access or decoded on
/// first need, and positions live in a separate region touched only by
/// [`positions`](Self::positions) and the phrase helpers.
pub struct PostingsReader {
    data: FileSlice,
    /// `data.as_slice()` cached (the slice's backing `Arc` lives in `data`).
    ptr: *const u8,
    len: usize,
    doc_freq: u32,
    total_term_freq: u64,
    flags: u8,
    num_full: u32,
    entry: u32,
    skip_off: u32,
    blocks_off: u32,
    tail_off: u32,
    pos_off: u32,
    tail_pos_ptr: u32,
    // Cursor.
    state: State,
    cur: u32,
    n: u32,
    idx: u32,
    cum_ready: bool,
    /// The cursor's decode buffers, allocated on the first block load: the
    /// bulk scorers never use them, and keeping 1 KiB of arrays out of the
    /// struct makes `open` and every move of a reader (into a scorer, into
    /// a `Box`) cheap. Invariant: `Some` whenever `state == InBlock`.
    bufs: Option<Box<CursorBufs>>,
    pos_buf: Vec<u32>,
}

/// Decode buffers of a [`PostingsReader`] cursor.
struct CursorBufs {
    docs: [u32; BLOCK_DOCS],
    /// Cumulative extra tf `C[i]` of the current block (valid if `cum_ready`).
    cum: [u32; BLOCK_DOCS],
}

// SAFETY: `ptr`/`len` point into memory owned by `data` (an `Arc` of
// `Send + Sync` bytes), which lives exactly as long as the reader.
unsafe impl Send for PostingsReader {}
unsafe impl Sync for PostingsReader {}

impl Clone for PostingsReader {
    fn clone(&self) -> Self {
        let mut r = Self::open(self.data.clone()).expect("reopen");
        r.state = State::Unstarted;
        r
    }
}

impl PostingsReader {
    /// Open a postings list at the start of `data` (trailing bytes ignored).
    pub fn open(data: FileSlice) -> Result<Self> {
        Self::open_inner(data, None)
    }

    /// [`PostingsReader::open`] for a list whose `doc_freq` and
    /// `total_term_freq` the term dictionary entry already holds (Lucene
    /// reads the term's metadata from its dictionary block): no varint is
    /// decoded, their encoded lengths are skipped, and the block count
    /// follows from `doc_freq`. Searches open a list per term, field and
    /// segment; this keeps that O(1) in bytes touched.
    pub fn open_known(data: FileSlice, doc_freq: u32, total_term_freq: u64) -> Result<Self> {
        Self::open_inner(data, Some((doc_freq, total_term_freq)))
    }

    fn open_inner(data: FileSlice, known: Option<(u32, u64)>) -> Result<Self> {
        let bytes = data.as_slice();
        let (ptr, len) = (bytes.as_ptr(), bytes.len());
        let mut pos = 0;
        let (doc_freq, total_term_freq, flags, num_full) = match known {
            Some((doc_freq, total_term_freq)) => {
                // LEB128 lengths of the header's varints, from their values.
                let vlen = |v: u64| ((64 - v.leading_zeros()).max(1) as usize).div_ceil(7);
                pos = vlen(doc_freq as u64) + vlen(total_term_freq);
                let flags = *bytes.get(pos).ok_or_else(|| corrupted("truncated"))?;
                pos += 1;
                let num_full = doc_freq / BLOCK_DOCS as u32;
                pos += vlen(num_full as u64);
                #[cfg(debug_assertions)]
                {
                    let mut p = 0;
                    debug_assert_eq!(read_vint(bytes, &mut p).ok(), Some(doc_freq as u64), "postings header doc_freq");
                    debug_assert_eq!(read_vint(bytes, &mut p).ok(), Some(total_term_freq), "postings header total_term_freq");
                    p += 1;
                    debug_assert_eq!(read_vint(bytes, &mut p).ok(), Some(num_full as u64), "postings header block count");
                    debug_assert_eq!(p, pos);
                }
                (doc_freq, total_term_freq, flags, num_full)
            }
            None => {
                let doc_freq = read_vint(bytes, &mut pos)? as u32;
                let total_term_freq = read_vint(bytes, &mut pos)?;
                let flags = *bytes.get(pos).ok_or_else(|| corrupted("truncated"))?;
                pos += 1;
                let num_full = read_vint(bytes, &mut pos)? as u32;
                (doc_freq, total_term_freq, flags, num_full)
            }
        };
        if num_full as usize > doc_freq as usize / BLOCK_DOCS {
            return Err(corrupted("bad block count"));
        }
        let has_pos = flags & FLAG_POSITIONS != 0;
        let (mut pos_off, mut tail_pos_ptr) = (0u32, 0u32);
        if has_pos {
            let h = bytes.get(pos..pos + 8).ok_or_else(|| corrupted("truncated"))?;
            pos_off = u32::from_le_bytes(h[0..4].try_into().unwrap());
            tail_pos_ptr = u32::from_le_bytes(h[4..8].try_into().unwrap());
            pos += 8;
        }
        let entry = if has_pos { SKIP_ENTRY_POS } else { SKIP_ENTRY } + if flags & FLAG_IMPACTS != 0 { IMPACT_BYTES } else { 0 };
        let skip_off = pos;
        let blocks_off = skip_off + num_full as usize * entry;
        if blocks_off > bytes.len() {
            return Err(corrupted("skip table truncated"));
        }
        let entry = entry as u32;
        let skip_off = skip_off as u32;
        let blocks_off = blocks_off as u32;
        let mut tail_off = blocks_off;
        if num_full > 0 {
            // `block_end` of the last full block (skip entry bytes 4..8).
            let o = skip_off as usize + (num_full as usize - 1) * entry as usize + 4;
            let e = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            let end = blocks_off as usize + e as usize;
            if end > bytes.len() {
                return Err(corrupted("blocks truncated"));
            }
            tail_off = end as u32;
        }
        if has_pos {
            let p = blocks_off as usize + pos_off as usize;
            if p > bytes.len() {
                return Err(corrupted("positions truncated"));
            }
            pos_off = p as u32;
        }
        Ok(Self {
            data,
            ptr,
            len,
            doc_freq,
            total_term_freq,
            flags,
            num_full,
            entry,
            skip_off,
            blocks_off,
            tail_off,
            pos_off,
            tail_pos_ptr,
            state: State::Unstarted,
            cur: NO_BLOCK,
            n: 0,
            idx: 0,
            cum_ready: false,
            bufs: None,
            pos_buf: Vec::new(),
        })
    }

    /// Convenience for tests and in-memory use.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::open(FileSlice::from_vec(bytes))
    }

    #[inline(always)]
    fn bytes(&self) -> &[u8] {
        // SAFETY: see the `Send` impl; `ptr`/`len` describe `data`'s bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The cursor buffers (only valid while `state == InBlock`).
    #[inline(always)]
    fn bufs(&self) -> &CursorBufs {
        debug_assert!(self.bufs.is_some());
        // SAFETY: `bufs` is set by `load`, before `state` becomes `InBlock`,
        // and never cleared.
        unsafe { self.bufs.as_deref().unwrap_unchecked() }
    }

    #[inline(always)]
    fn bufs_mut(&mut self) -> &mut CursorBufs {
        debug_assert!(self.bufs.is_some());
        // SAFETY: as in `bufs`.
        unsafe { self.bufs.as_deref_mut().unwrap_unchecked() }
    }

    pub fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    pub fn total_term_freq(&self) -> u64 {
        self.total_term_freq
    }

    pub fn has_positions(&self) -> bool {
        self.flags & FLAG_POSITIONS != 0
    }

    /// True when some doc has tf > 1.
    pub fn has_freqs(&self) -> bool {
        self.flags & FLAG_FREQS != 0
    }

    /// True when the skip table's `min_norm` values come from real norms.
    pub fn has_norm_impacts(&self) -> bool {
        self.flags & FLAG_NORMS != 0
    }

    /// True when the list carries per-block impact staircases (written
    /// under segment format v12 and later only).
    pub fn has_impact_staircases(&self) -> bool {
        self.flags & FLAG_IMPACTS != 0
    }

    // ---- block-level API (bulk scorers) ----------------------------------

    /// Number of full 128-doc blocks.
    #[inline(always)]
    pub fn num_full_blocks(&self) -> usize {
        self.num_full as usize
    }

    /// Number of docs in the tail block.
    #[inline(always)]
    pub fn tail_len(&self) -> usize {
        self.doc_freq as usize - self.num_full as usize * BLOCK_DOCS
    }

    /// Full blocks plus the tail (if non-empty).
    #[inline(always)]
    pub fn num_blocks(&self) -> usize {
        self.num_full as usize + (self.tail_len() > 0) as usize
    }

    /// Skip entry of full block `b`.
    #[inline(always)]
    pub fn skip(&self, b: usize) -> SkipEntry {
        let o = self.skip_off as usize + b * self.entry as usize;
        let e = &self.bytes()[o..o + self.entry as usize];
        let mt = u16::from_le_bytes([e[8], e[9]]);
        let impacts = if self.flags & FLAG_IMPACTS != 0 { e[e.len() - IMPACT_BYTES..].try_into().unwrap() } else { [0; 4] };
        SkipEntry {
            last_doc: u32::from_le_bytes([e[0], e[1], e[2], e[3]]),
            block_end: u32::from_le_bytes([e[4], e[5], e[6], e[7]]),
            max_tf: if mt == 0xFFFF { u32::MAX } else { mt as u32 },
            min_norm: e[10],
            impacts,
        }
    }

    /// Last doc of block `b` (full block), or a sentinel for the tail.
    #[inline(always)]
    pub fn block_last_doc(&self, b: usize) -> DocId {
        if b < self.num_full as usize {
            let o = self.skip_off as usize + b * self.entry as usize;
            let e = &self.bytes()[o..o + 4];
            u32::from_le_bytes([e[0], e[1], e[2], e[3]])
        } else {
            TAIL_LAST
        }
    }

    #[inline(always)]
    fn block_end(&self, b: usize) -> usize {
        let o = self.skip_off as usize + b * self.entry as usize + 4;
        let e = &self.bytes()[o..o + 4];
        u32::from_le_bytes([e[0], e[1], e[2], e[3]]) as usize
    }

    #[inline(always)]
    fn block_start(&self, b: usize) -> usize {
        self.blocks_off as usize + if b == 0 { 0 } else { self.block_end(b - 1) }
    }

    #[inline(always)]
    fn prev_last(&self, b: usize) -> u32 {
        if b == 0 { 0 } else { self.block_last_doc(b - 1) }
    }

    /// `(doc_bits, tf_bits, doc payload offset)` of full block `b`.
    #[inline(always)]
    fn block_header(&self, b: usize) -> (u8, u8, usize) {
        let s = self.block_start(b);
        let h = &self.bytes()[s..s + 2];
        (h[0], h[1], s + 2)
    }

    /// Decode only the doc ids of full block `b`.
    #[inline]
    pub fn decode_docs(&self, b: usize, out: &mut [u32; BLOCK_DOCS]) {
        let (db, _, off) = self.block_header(b);
        let base = self.prev_last(b);
        let payload = &self.bytes()[off..];
        if self.flags & FLAG_DOC_GAPS != 0 {
            unpack_sorted4x(base, payload, db, out);
        } else {
            unpack4x(payload, db, out);
            for d in out.iter_mut() {
                *d += base;
            }
        }
    }

    /// Decode the tfs of full block `b`. Returns true (and leaves `out`
    /// untouched) when every tf in the block is 1.
    #[inline]
    pub fn decode_tfs(&self, b: usize, out: &mut [u32; BLOCK_DOCS]) -> bool {
        let (db, tb, off) = self.block_header(b);
        if tb == 0 {
            return true;
        }
        unpack4x(&self.bytes()[off + len4x(db)..], tb, out);
        for t in out.iter_mut() {
            *t += 1;
        }
        false
    }

    /// Decode doc ids and tfs of full block `b` (see [`decode_tfs`](Self::decode_tfs)).
    #[inline]
    pub fn decode_block(&self, b: usize, docs: &mut [u32; BLOCK_DOCS], tfs: &mut [u32; BLOCK_DOCS]) -> bool {
        self.decode_docs(b, docs);
        self.decode_tfs(b, tfs)
    }

    /// Random access: doc `i` of full block `b` (frame-of-reference format
    /// only; d-gap lists decode the block).
    #[inline]
    pub fn doc_at(&self, b: usize, i: usize) -> DocId {
        if self.flags & FLAG_DOC_GAPS != 0 {
            let mut d = [0u32; BLOCK_DOCS];
            self.decode_docs(b, &mut d);
            return d[i];
        }
        let (db, _, off) = self.block_header(b);
        self.prev_last(b) + get4x(&self.bytes()[off..], db, i)
    }

    /// Random access: tf of doc `i` of full block `b`.
    #[inline]
    pub fn tf_at(&self, b: usize, i: usize) -> u32 {
        let (db, tb, off) = self.block_header(b);
        if tb == 0 {
            return 1;
        }
        1 + get4x(&self.bytes()[off + len4x(db)..], tb, i)
    }

    /// Decode the tail block (docs and tfs); returns its length.
    pub fn decode_tail(&self, docs: &mut [u32; BLOCK_DOCS], tfs: &mut [u32; BLOCK_DOCS]) -> usize {
        let n = self.tail_len();
        if n == 0 {
            return 0;
        }
        let bytes = self.bytes();
        let at = self.tail_off as usize;
        let (db, tb) = (bytes[at], bytes[at + 1]);
        let d = at + 2;
        let dl = packed_len(n, db as usize);
        unpack_short(&bytes[d..], db, &mut docs[..n]);
        let mut prev = if self.num_full == 0 { 0 } else { self.block_last_doc(self.num_full as usize - 1) };
        for x in docs[..n].iter_mut() {
            prev += *x;
            *x = prev;
        }
        if tb == 0 {
            tfs[..n].fill(1);
        } else {
            unpack_short(&bytes[d + dl..], tb, &mut tfs[..n]);
            for t in tfs[..n].iter_mut() {
                *t += 1;
            }
        }
        n
    }

    // ---- cursor ----------------------------------------------------------

    /// Load block `b` (full or tail) into the cursor buffer.
    fn load(&mut self, b: usize) {
        if self.bufs.is_none() {
            self.bufs = Some(Box::new(CursorBufs { docs: [0; BLOCK_DOCS], cum: [0; BLOCK_DOCS] }));
        }
        if b < self.num_full as usize {
            // Decode straight into the cursor buffer (no 512-byte copy).
            let (db, _, off) = self.block_header(b);
            let base = self.prev_last(b);
            assert!(off + len4x(db) <= self.len, "postings: block truncated");
            // SAFETY: in bounds (asserted); `ptr`/`len` describe `data`'s
            // bytes (see `Send`) and do not alias the buffers.
            let payload = unsafe { std::slice::from_raw_parts(self.ptr.add(off), self.len - off) };
            let gaps = self.flags & FLAG_DOC_GAPS != 0;
            let docs = &mut self.bufs_mut().docs;
            if gaps {
                unpack_sorted4x(base, payload, db, docs);
            } else {
                unpack4x(payload, db, docs);
                for d in docs.iter_mut() {
                    *d += base;
                }
            }
            self.n = BLOCK_DOCS as u32;
            self.cum_ready = false;
        } else {
            let mut tfs = [0u32; BLOCK_DOCS];
            let mut docs = [0u32; BLOCK_DOCS];
            let n = self.decode_tail(&mut docs, &mut tfs);
            let bufs = self.bufs_mut();
            bufs.docs[..n].copy_from_slice(&docs[..n]);
            let mut c = 0u32;
            for i in 0..n {
                c += tfs[i].max(1) - 1;
                bufs.cum[i] = c;
            }
            self.n = n as u32;
            self.cum_ready = true;
        }
        self.cur = b as u32;
        self.idx = 0;
        self.state = State::InBlock;
    }

    #[inline]
    fn ensure_cum(&mut self) {
        if self.cum_ready {
            return;
        }
        let (db, tb, off) = self.block_header(self.cur as usize);
        assert!(off + len4x(db) + len4x(tb) <= self.len, "postings: block truncated");
        // SAFETY: in bounds (asserted); the bytes do not alias `self.cum`.
        let p = unsafe { std::slice::from_raw_parts(self.ptr.add(off + len4x(db)), self.len - off - len4x(db)) };
        unpack_sorted4x(0, p, tb, &mut self.bufs_mut().cum);
        self.cum_ready = true;
    }

    /// Advance to the first doc >= target. Returns the doc, or None.
    ///
    /// Forward seeks resume from the current doc (a target at or before the
    /// current doc returns it); a backward seek re-positions from the block
    /// table.
    pub fn advance(&mut self, target: DocId) -> Option<DocId> {
        let nb = self.num_blocks();
        if nb == 0 || self.state == State::Exhausted {
            self.state = State::Exhausted;
            return None;
        }
        let mut lo_block = 0usize;
        if self.state == State::InBlock {
            let cur = self.cur as usize;
            let n = self.n as usize;
            let cd = self.bufs().docs[self.idx as usize];
            let block_first_ok = cur == 0 || target > self.block_last_doc(cur - 1);
            if target <= cd && (self.idx == 0 || self.bufs().docs[self.idx as usize - 1] < target) && block_first_ok {
                return Some(cd);
            }
            let last = self.bufs().docs[n - 1];
            if target <= last && block_first_ok {
                let from = if target > cd { self.idx as usize } else { 0 };
                let i = from + self.bufs().docs[from..n].partition_point(|&d| d < target);
                self.idx = i as u32;
                return Some(self.bufs().docs[i]);
            }
            if target > last {
                lo_block = cur + 1;
            }
        }
        // First full block whose last doc is >= target.
        let (mut lo, mut hi) = (lo_block.min(self.num_full as usize), self.num_full as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.block_last_doc(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= nb {
            self.state = State::Exhausted;
            return None;
        }
        if lo != self.cur as usize || self.state != State::InBlock {
            self.load(lo);
        }
        let n = self.n as usize;
        let i = self.bufs().docs[..n].partition_point(|&d| d < target);
        if i >= n {
            self.state = State::Exhausted;
            return None;
        }
        self.idx = i as u32;
        self.state = State::InBlock;
        Some(self.bufs().docs[i])
    }

    /// Forward-only seek: first doc >= `target`, never moving backwards (a
    /// target at or before the current doc returns the current doc). The
    /// hot path of conjunction/phrase approximations: a linear scan inside
    /// the current block (candidate gaps are short), else the next block or
    /// a binary search of the skip table.
    #[inline]
    pub fn seek_forward(&mut self, target: DocId) -> Option<DocId> {
        let mut b = match self.state {
            State::InBlock => {
                let mut i = self.idx as usize;
                if self.bufs().docs[i] >= target {
                    return Some(self.bufs().docs[i]);
                }
                let n = self.n as usize;
                if self.bufs().docs[n - 1] >= target {
                    // Terminates: docs[n-1] >= target.
                    i += 1;
                    while self.bufs().docs[i] < target {
                        i += 1;
                    }
                    self.idx = i as u32;
                    return Some(self.bufs().docs[i]);
                }
                self.cur as usize + 1
            }
            State::Unstarted => 0,
            State::Exhausted => return None,
        };
        let nf = self.num_full as usize;
        if b < nf && self.block_last_doc(b) < target {
            let (mut lo, mut hi) = (b + 1, nf);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if self.block_last_doc(mid) < target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            b = lo;
        }
        if b >= self.num_blocks() {
            self.state = State::Exhausted;
            return None;
        }
        self.load(b);
        let n = self.n as usize;
        let mut i = 0;
        while i < n && self.bufs().docs[i] < target {
            i += 1;
        }
        if i >= n {
            self.state = State::Exhausted;
            return None;
        }
        self.idx = i as u32;
        Some(self.bufs().docs[i])
    }

    /// Next doc in iteration order (from the current position).
    #[inline]
    pub fn next(&mut self) -> Option<DocId> {
        match self.state {
            State::InBlock => {
                if self.idx + 1 < self.n {
                    self.idx += 1;
                    return Some(self.bufs().docs[self.idx as usize]);
                }
                let b = self.cur as usize + 1;
                if b >= self.num_blocks() {
                    self.state = State::Exhausted;
                    return None;
                }
                self.load(b);
                Some(self.bufs().docs[0])
            }
            State::Unstarted => {
                if self.num_blocks() == 0 {
                    self.state = State::Exhausted;
                    return None;
                }
                self.load(0);
                Some(self.bufs().docs[0])
            }
            State::Exhausted => None,
        }
    }

    /// Current doc (the last one returned by `next`/`advance`).
    #[inline]
    pub fn doc(&self) -> Option<DocId> {
        if self.state == State::InBlock { Some(self.bufs().docs[self.idx as usize]) } else { None }
    }

    /// Term frequency of the current doc.
    #[inline]
    pub fn term_freq(&self) -> Option<u32> {
        if self.state != State::InBlock {
            return None;
        }
        let i = self.idx as usize;
        if self.cum_ready {
            let lo = if i == 0 { 0 } else { self.bufs().cum[i - 1] };
            return Some(self.bufs().cum[i] - lo + 1);
        }
        Some(self.tf_at(self.cur as usize, i))
    }

    /// `(tf, index of the doc's first position in the block's positions)`.
    #[inline]
    fn tf_start(&mut self) -> (usize, usize) {
        self.ensure_cum();
        let i = self.idx as usize;
        let lo = if i == 0 { 0 } else { self.bufs().cum[i - 1] };
        (1 + (self.bufs().cum[i] - lo) as usize, i + lo as usize)
    }

    /// Packed positions of the current block: `(bits, data from payload)`.
    #[inline]
    fn block_positions(&self) -> (u8, usize) {
        self.block_positions_of(self.cur as usize)
    }

    /// `(bits, data offset)` of the packed positions of block `b` (full
    /// block, or `num_full_blocks()` for the tail).
    #[inline]
    fn block_positions_of(&self, b: usize) -> (u8, usize) {
        let ptr = if b < self.num_full as usize {
            let o = self.skip_off as usize + b * self.entry as usize + 12;
            let e = &self.bytes()[o..o + 4];
            u32::from_le_bytes([e[0], e[1], e[2], e[3]])
        } else {
            self.tail_pos_ptr
        };
        let at = self.pos_off as usize + ptr as usize;
        (self.bytes()[at], at + 1)
    }

    /// Decode every position of block `b` (a full block, or
    /// `num_full_blocks()` for the tail) into `out`, in doc order: `count`
    /// is the block's total tf. Empty without positions.
    pub fn decode_block_positions(&self, b: usize, count: usize, out: &mut Vec<u32>) {
        out.clear();
        if !self.has_positions() || count == 0 {
            return;
        }
        let (bits, at) = self.block_positions_of(b);
        out.resize(count, 0);
        unpack_seq_at(&self.bytes()[at..], bits, 0, out);
    }

    /// [`Self::decode_block_positions`] appended to `out` (a merge copying
    /// every position of the block needs no per-block scratch).
    pub fn append_block_positions(&self, b: usize, count: usize, out: &mut Vec<u32>) {
        if !self.has_positions() || count == 0 {
            return;
        }
        let (bits, at) = self.block_positions_of(b);
        let start = out.len();
        out.resize(start + count, 0);
        unpack_seq_at(&self.bytes()[at..], bits, 0, &mut out[start..]);
    }

    /// Decode the current doc's positions into `out` (cleared first), each
    /// plus `shift`.
    pub fn positions_into(&mut self, out: &mut Vec<u32>, shift: u32) {
        out.clear();
        if !self.has_positions() || self.state != State::InBlock {
            return;
        }
        let (tf, start) = self.tf_start();
        let (bits, at) = self.block_positions();
        out.resize(tf, 0);
        unpack_seq_at(&self.bytes()[at..], bits, start, out);
        if shift != 0 {
            for p in out.iter_mut() {
                *p += shift;
            }
        }
    }

    /// Number of the current doc's positions `p` with `p + shift` in the
    /// sorted `targets` (phrase confirmation without materializing them).
    pub fn count_shifted_matches(&mut self, targets: &[u32], shift: u32) -> u32 {
        if !self.has_positions() || self.state != State::InBlock || targets.is_empty() {
            return 0;
        }
        let (tf, start) = self.tf_start();
        let (bits, at) = self.block_positions();
        let data = &self.bytes()[at..];
        let b = bits as usize;
        let end_byte = ((start + tf) * b + 7) / 8;
        assert!(data.len() >= end_byte + SEQ_PAD - 1);
        let mask: u64 = if b == 32 { u32::MAX as u64 } else if b == 0 { 0 } else { (1u64 << b) - 1 };
        let p = data.as_ptr();
        let get = |k: usize| -> u32 {
            let bp = k * b;
            // SAFETY: k < start + tf, so the 8-byte load ends within the
            // asserted slack.
            let w = u64::from_le(unsafe { (p.add(bp >> 3) as *const u64).read_unaligned() });
            ((w >> (bp & 7)) & mask) as u32 + shift
        };
        let (mut k, end) = (start, start + tf);
        let mut v = get(k);
        let mut c = 0;
        for &t in targets {
            while v < t {
                k += 1;
                if k == end {
                    return c;
                }
                v = get(k);
            }
            c += (v == t) as u32;
        }
        c
    }

    /// Current doc's tf, decoding the block's cumulative tfs if needed
    /// (cheaper than [`term_freq`](Self::term_freq) when positions follow).
    pub fn freq(&mut self) -> u32 {
        if self.state != State::InBlock {
            return 0;
        }
        self.tf_start().0 as u32
    }

    /// Positions of the current doc (empty if not indexed with positions).
    pub fn positions(&mut self) -> &[u32] {
        let mut buf = std::mem::take(&mut self.pos_buf);
        self.positions_into(&mut buf, 0);
        self.pos_buf = buf;
        &self.pos_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(doc_ids: &[DocId], tfs: &[u32]) -> PostingsReader {
        let mut buf = Vec::new();
        write_postings(doc_ids, tfs, &mut buf);
        PostingsReader::from_bytes(buf).unwrap()
    }

    #[test]
    fn sequential_iteration() {
        let docs: Vec<DocId> = (0..300u32).collect();
        let tfs: Vec<u32> = (0..300).map(|i| (i % 5) + 1).collect();
        let mut r = roundtrip(&docs, &tfs);
        assert_eq!(r.doc_freq(), 300);
        assert_eq!(r.total_term_freq(), tfs.iter().sum::<u32>() as u64);
        assert!(!r.has_positions());
        for (i, expected) in docs.iter().enumerate() {
            assert_eq!(r.next(), Some(*expected), "doc {i}");
            assert_eq!(r.term_freq(), Some(tfs[i]));
            assert!(r.positions().is_empty());
        }
        assert_eq!(r.next(), None);
    }

    #[test]
    fn advance_matches_bruteforce() {
        let docs: Vec<DocId> = (0..100_000u32).step_by(7).collect();
        let tfs: Vec<u32> = vec![1; docs.len()];
        let mut r = roundtrip(&docs, &tfs);
        for target in (0..100_010u32).step_by(3) {
            let expected = {
                let idx = docs.partition_point(|&d| d < target);
                docs.get(idx).copied()
            };
            assert_eq!(r.advance(target), expected, "target {target}");
        }
    }

    #[test]
    fn advance_then_continue() {
        let docs: Vec<DocId> = (0..500u32).step_by(3).collect();
        let tfs: Vec<u32> = vec![2; docs.len()];
        let mut r = roundtrip(&docs, &tfs);
        assert_eq!(r.advance(100), Some(102));
        assert_eq!(r.term_freq(), Some(2));
        assert_eq!(r.next(), Some(105));
        assert_eq!(r.advance(400), Some(402));
        assert_eq!(r.next(), Some(405));
        assert_eq!(r.advance(9999), None);
        assert_eq!(r.next(), None);
    }

    #[test]
    fn backward_advance_repositions() {
        let docs: Vec<DocId> = (0..2000u32).step_by(2).collect();
        let tfs: Vec<u32> = (0..docs.len() as u32).map(|i| i % 3 + 1).collect();
        let mut r = roundtrip(&docs, &tfs);
        assert_eq!(r.advance(1500), Some(1500));
        assert_eq!(r.advance(11), Some(12));
        assert_eq!(r.term_freq(), Some(tfs[6]));
        assert_eq!(r.advance(1999), None);
    }

    #[test]
    fn empty_list() {
        let mut buf = Vec::new();
        write_postings(&[], &[], &mut buf);
        let mut r = PostingsReader::from_bytes(buf).unwrap();
        assert_eq!(r.doc_freq(), 0);
        assert_eq!(r.next(), None);
        assert_eq!(r.advance(0), None);
    }

    #[test]
    fn single_doc() {
        let mut r = roundtrip(&[42], &[3]);
        assert_eq!(r.next(), Some(42));
        assert_eq!(r.term_freq(), Some(3));
        assert_eq!(r.advance(42), Some(42));
        assert_eq!(r.advance(43), None);
    }

    /// The impact staircase bounds every doc of its block under any field
    /// average and boost, and is far tighter than `(max_tf, min_norm)` when
    /// the highest tf sits in a long doc.
    #[test]
    fn impact_staircase_bounds_every_doc_and_tightens() {
        use crate::query::bm25::{int_to_byte4, Bm25Params, Bm25Weight};
        let mut seed = 99u64;
        let mut rnd = |m: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        for case in 0..40 {
            let n = 128 * (1 + case % 3) + case;
            let docs: Vec<DocId> = (0..n as u32).map(|i| i * 2).collect();
            // Length grows with tf (long articles repeat words), plus noise.
            let tfs: Vec<u32> = (0..n).map(|_| 1 + (rnd(10) * rnd(10)) as u32 / (1 + case as u32 % 4)).collect();
            let mut norms = vec![0u8; 2 * n + 2];
            for (i, &d) in docs.iter().enumerate() {
                norms[d as usize] = int_to_byte4(5 + tfs[i] * (20 + rnd(200) as u32) + rnd(50) as u32);
            }
            let mut buf = Vec::new();
            write_postings_impl(&docs, &tfs, None, BlockNorms::Bytes(&norms), true, &mut buf);
            let r = PostingsReader::from_bytes(buf).unwrap();
            for avg in [3.0f32, 40.0, 300.0, 5000.0] {
                for boost in [0.5f32, 1.0, 4.0] {
                    let w = Bm25Weight::new(n as u64, 10 * n as u64, Bm25Params::default(), avg).with_boost(boost);
                    for b in 0..r.num_full_blocks() {
                        let s = r.skip(b);
                        let bound = w.block_bound(&s);
                        assert!(bound <= w.bound(s.max_tf, s.min_norm));
                        for i in b * 128..(b + 1) * 128 {
                            let sc = w.score_norm(tfs[i], norms[docs[i] as usize]);
                            assert!(sc <= bound, "case {case} avg {avg} block {b}: doc {i} {sc} > {bound} ({s:?})");
                        }
                    }
                }
            }
        }
        // One long doc with tf 30, every other doc short with tf 1.
        let docs: Vec<DocId> = (0..128).collect();
        let tfs: Vec<u32> = (0..128).map(|i| if i == 7 { 30 } else { 1 }).collect();
        let norms: Vec<u8> = (0..128).map(|i| int_to_byte4(if i == 7 { 3000 } else { 10 })).collect();
        let mut buf = Vec::new();
        write_postings_impl(&docs, &tfs, None, BlockNorms::Bytes(&norms), true, &mut buf);
        let r = PostingsReader::from_bytes(buf).unwrap();
        let w = Bm25Weight::new(128, 100_000, Bm25Params::default(), 400.0);
        let s = r.skip(0);
        let real = (0..128).map(|i| w.score_norm(tfs[i], norms[i])).fold(0f32, f32::max);
        assert!(w.block_bound(&s) <= real * 1.0001, "{} vs real {real} (old {})", w.block_bound(&s), w.bound(s.max_tf, s.min_norm));
        assert!(w.bound(s.max_tf, s.min_norm) > 1.2 * real);
        // Without impacts (a v11 segment) the list has no staircase and the
        // old bound, and reads the same docs.
        let mut old = Vec::new();
        write_postings_impl(&docs, &tfs, None, BlockNorms::Bytes(&norms), false, &mut old);
        let r11 = PostingsReader::from_bytes(old).unwrap();
        assert_eq!(r11.skip(0).impacts, [0; 4]);
        assert_eq!(w.block_bound(&r11.skip(0)), w.bound(s.max_tf, s.min_norm));
    }

    #[test]
    fn dense_lists_use_doc_gaps_and_read_back() {
        // A term in every doc, one in every third doc, and a sparse one.
        for (step, dense) in [(1u32, true), (3, true), (997, false)] {
            let docs: Vec<DocId> = (0..5000u32).map(|i| i * step + 2).collect();
            let tfs: Vec<u32> = (0..5000u32).map(|i| if i % 5 == 0 { 2 } else { 1 }).collect();
            let mut buf = Vec::new();
            write_postings(&docs, &tfs, &mut buf);
            let mut plain = Vec::new();
            let r = PostingsReader::from_bytes(buf.clone()).unwrap();
            assert_eq!(r.flags & FLAG_DOC_GAPS != 0, dense, "step {step}");
            if dense {
                // Far smaller than frame-of-reference offsets would be.
                for b in 0..docs.len() / BLOCK_DOCS {
                    let bd = &docs[b * BLOCK_DOCS..(b + 1) * BLOCK_DOCS];
                    let prev = if b == 0 { 0 } else { docs[b * BLOCK_DOCS - 1] };
                    plain.push(num_bits4x(&std::array::from_fn::<u32, BLOCK_DOCS, _>(|i| bd[i] - prev)));
                }
                assert!(plain.iter().all(|&bits| bits >= 7), "step {step}: {plain:?}");
            }
            let (mut d, mut t) = ([0u32; BLOCK_DOCS], [0u32; BLOCK_DOCS]);
            for b in 0..r.num_full_blocks() as usize {
                r.decode_block(b, &mut d, &mut t);
                assert_eq!(&d[..], &docs[b * BLOCK_DOCS..(b + 1) * BLOCK_DOCS], "step {step} block {b}");
                assert_eq!(r.doc_at(b, 77), docs[b * BLOCK_DOCS + 77]);
                assert_eq!(r.tf_at(b, 5), tfs[b * BLOCK_DOCS + 5]);
            }
            let n = r.decode_tail(&mut d, &mut t);
            assert_eq!(&d[..n], &docs[docs.len() - n..]);
            // The cursor walks and seeks the same docs.
            let mut c = PostingsReader::from_bytes(buf).unwrap();
            let mut seen = Vec::new();
            while let Some(x) = c.next() {
                seen.push(x);
            }
            assert_eq!(seen, docs, "step {step}");
            let mut c = PostingsReader::from_bytes({ let mut b = Vec::new(); write_postings(&docs, &tfs, &mut b); b }).unwrap();
            for target in [0u32, docs[300] - 1, docs[300], docs[1000] + 1, docs[4999]] {
                let want = docs.iter().copied().find(|&x| x >= target);
                assert_eq!(c.advance(target), want, "step {step} target {target}");
            }
        }
    }

    #[test]
    fn block_api_and_impacts() {
        let docs: Vec<DocId> = (0..1000u32).map(|i| i * 3 + 1).collect();
        let tfs: Vec<u32> = (0..1000u32).map(|i| (i * 13) % 7 + 1).collect();
        let norms: Vec<u8> = (0..3100u32).map(|d| (d % 200) as u8 + 5).collect();
        let mut buf = Vec::new();
        write_postings_with_norms(&docs, &tfs, None, BlockNorms::Bytes(&norms), &mut buf);
        let r = PostingsReader::from_bytes(buf).unwrap();
        assert_eq!(r.num_full_blocks(), 7);
        assert_eq!(r.tail_len(), 1000 - 7 * 128);
        let (mut d, mut t) = ([0u32; 128], [0u32; 128]);
        for b in 0..7 {
            let s = r.skip(b);
            let bd = &docs[b * 128..(b + 1) * 128];
            let bt = &tfs[b * 128..(b + 1) * 128];
            assert_eq!(s.last_doc, bd[127]);
            assert_eq!(s.max_tf, *bt.iter().max().unwrap());
            assert_eq!(s.min_norm, bd.iter().map(|&x| norms[x as usize]).min().unwrap());
            assert!(!r.decode_block(b, &mut d, &mut t));
            assert_eq!(&d[..], bd);
            assert_eq!(&t[..], bt);
            for i in [0usize, 1, 63, 127] {
                assert_eq!(r.doc_at(b, i), bd[i]);
                assert_eq!(r.tf_at(b, i), bt[i]);
            }
        }
        let n = r.decode_tail(&mut d, &mut t);
        assert_eq!(&d[..n], &docs[896..]);
        assert_eq!(&t[..n], &tfs[896..]);
    }

    #[test]
    fn reused_positions_scratch_encodes_like_a_fresh_one() {
        let list = |n: u32, step: u32| {
            let docs: Vec<DocId> = (0..n).map(|i| i * step).collect();
            let tfs: Vec<u32> = (0..n).map(|i| (i % 4) + 1).collect();
            let positions: Vec<u32> = (0..n).flat_map(|i| (0..(i % 4) + 1).map(move |k| 1000 * i + k)).collect();
            (docs, tfs, positions)
        };
        let encode = |(docs, tfs, positions): &(Vec<DocId>, Vec<u32>, Vec<u32>)| {
            let mut buf = Vec::new();
            write_postings_full(docs, tfs, Some(positions), &mut buf);
            buf
        };
        let (big, small) = (list(5000, 3), list(70, 5));
        // Fresh thread: an empty scratch.
        let fresh = std::thread::spawn({ let small = small.clone(); move || encode(&small) }).join().unwrap();
        encode(&big);
        assert_eq!(encode(&small), fresh, "a larger list left nothing behind");
        assert_eq!(encode(&small), fresh);
    }

    #[test]
    fn positions_roundtrip() {
        // 300 docs, doc i has (i % 3) + 1 positions at 10*i + k*7.
        let docs: Vec<DocId> = (0..300u32).map(|i| i * 2).collect();
        let tfs: Vec<u32> = (0..300u32).map(|i| (i % 3) + 1).collect();
        let mut positions = Vec::new();
        for i in 0..300u32 {
            for k in 0..tfs[i as usize] {
                positions.push(10 * i + k * 7);
            }
        }
        let mut buf = Vec::new();
        write_postings_full(&docs, &tfs, Some(&positions), &mut buf);
        let mut r = PostingsReader::from_bytes(buf).unwrap();
        assert!(r.has_positions());
        let mut cursor = 0;
        for i in 0..300u32 {
            assert_eq!(r.next(), Some(i * 2));
            let n = tfs[i as usize] as usize;
            assert_eq!(r.term_freq(), Some(n as u32));
            assert_eq!(r.positions(), &positions[cursor..cursor + n], "doc {i}");
            cursor += n;
        }
        // advance lands on the right positions too.
        assert_eq!(r.advance(257 * 2), Some(514));
        let i = 257usize;
        let start: usize = tfs[..i].iter().map(|&t| t as usize).sum();
        assert_eq!(r.positions(), &positions[start..start + tfs[i] as usize]);
        // Random access in a full block, via a fresh reader and advance.
        let mut r2 = r.clone();
        assert_eq!(r2.advance(100), Some(100));
        let start: usize = tfs[..50].iter().map(|&t| t as usize).sum();
        let want = &positions[start..start + tfs[50] as usize];
        assert_eq!(r2.positions(), want);
        let targets: Vec<u32> = want.iter().map(|&p| p + 3).collect();
        assert_eq!(r2.count_shifted_matches(&targets, 3), want.len() as u32);
        assert_eq!(r2.count_shifted_matches(&targets, 4), 0);
        // Lists with positions: block decode and random access give tfs.
        let (mut d, mut t) = ([0u32; 128], [0u32; 128]);
        for b in 0..2 {
            assert!(!r2.decode_block(b, &mut d, &mut t));
            assert_eq!(&t[..], &tfs[b * 128..(b + 1) * 128]);
            for i in [0usize, 1, 77, 127] {
                assert_eq!(r2.tf_at(b, i), tfs[b * 128 + i]);
            }
        }
        // Forward-only seek.
        let mut r3 = r2.clone();
        assert_eq!(r3.seek_forward(3), Some(4));
        assert_eq!(r3.seek_forward(2), Some(4));
        assert_eq!(r3.seek_forward(300), Some(300));
        assert_eq!(r3.term_freq(), Some(tfs[150]));
        assert_eq!(r3.positions(), &positions[tfs[..150].iter().sum::<u32>() as usize..][..tfs[150] as usize]);
        assert_eq!(r3.seek_forward(597), Some(598));
        assert_eq!(r3.seek_forward(599), None);
    }
}
