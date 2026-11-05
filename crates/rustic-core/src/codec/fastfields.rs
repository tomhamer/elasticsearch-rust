//! On-disk columnar fast fields (Lucene doc values).
//!
//! ```text
//! directory: vint n, per column: u16 field, u8 tag, vint off, vint len
//! numeric body (tags 1=i64, 2=u64, 3=f64):
//!     u32 num_docs, u32 num_blocks, [min; num_blocks] 8B LE, [max] 8B LE,
//!     [value; num_docs] 8B LE   (missing docs are 0 / 0.0),
//!     [u32 doc; num_docs] LE    (segment v7+: the 1-D points index, every
//!                                doc sorted by (value, doc); NaN last)
//! sparse numeric v10 (COMPACT_SPARSE block-count flag): same header/min/max,
//!     u32 present_count, [u32 prefix_rank; ceil(num_docs/64)],
//!     [u64 presence; ceil(num_docs/64)], [value; present_count],
//!     optional [u32 doc; present_count] sorted by (value, doc).
//! v11 packed numeric (PACKED block-count flag): after the max table,
//!     u8 value_width, u8 point_width, u64 base; each value (dense or
//!     compact) is `bits(value) - base` in `value_width` LE bytes (0: every
//!     value is `base`), each points-index doc `point_width` LE bytes;
//!     7 zero bytes follow 1-7 byte values and 1 follows 3-byte docs, so
//!     every read is one 8- or 4-byte load and a mask.
//! v14 table packing (`value_width` byte with its top bit set, [`TABLE`]):
//!     `base` is the number of distinct values `k`, and `[u64 bits; k]`
//!     (sorted by the column's total order) follows the packing header;
//!     each value is its index in that table in `value_width & 0x7f` bytes
//!     (Lucene's table-compressed numeric doc values: a float column with a
//!     few thousand distinct values takes 2 bytes a value, not 8).
//! bool body (tag 4): u32 num_docs, [u64 word] LE
//! norms body (tag 6): u32 num_docs, u64 total_len (exact token count),
//!     [u8 norm; num_docs] (Lucene `SmallFloat.intToByte4` of the length)
//! constant norms body (tag 12, v27+): u32 num_docs, u64 total_len, u8 norm:
//!     every doc's norm byte (Lucene's norms of a field whose docs all have
//!     the same length take no space either)
//! keyword body (tag 5): vint num_terms, per term: vint len, bytes (sorted),
//!     u32 num_docs, [u32 ordinal; num_docs] (u32::MAX = missing)
//!     v11: the top two bits of `num_docs` give the ordinal width (0: 4
//!     bytes as above, 1-3: that many LE bytes, all-ones = missing), the
//!     narrowest that holds `num_terms` ordinals plus the missing marker;
//!     `4 - width` zero bytes follow narrow ordinals (one 4-byte load each)
//! sparse keyword body (on-disk tag 11, segment v14+; read as a keyword
//!     column, [`ColumnEntry::sparse`]): the keyword term table and
//!     ordinal header word as above, then u32 present_count,
//!     [u32 prefix_rank; ceil(num_docs/64)], [u64 presence; ceil(num_docs/64)],
//!     [ordinal; present_count] (`width` bytes each, padded as above): the
//!     ordinals of the docs that have a value only (Lucene SORTED doc values
//!     behind an `IndexedDISI`)
//! multi numeric body (tags 7=i64, 8=u64, 9=f64):
//!     u32 num_docs, u32 total_values, [u32 offsets; num_docs+1],
//!     [min; num_blocks] 8B LE, [max; num_blocks] 8B LE
//!     (blocks of 256 flattened values), [value 8B LE; total_values]
//! multi keyword body (tag 10): sorted unique term table (vint n, per term
//!     vint len + bytes), u32 num_docs, u32 total, [u32 offsets; num_docs+1],
//!     [u32 ordinal; total] with per-doc ordinals sorted ascending
//!     (Lucene SORTED_SET: deduped within a doc)
//!     v26 packed (top bit of `num_docs` set, [`PACKED_MULTI`]): u32 total,
//!     u8 offset_bits, u8 ordinal_bits, then the `num_docs + 1` offsets
//!     bit-packed LSB-first at `offset_bits` (`bits(total)`), the ordinals
//!     bit-packed at `ordinal_bits` (`bits(num_terms - 1)`), each part
//!     starting on a byte boundary, and 8 zero bytes so every read is one
//!     unaligned 8-byte load (Lucene stores SORTED_SET ordinals in
//!     `DirectWriter.unsignedBitsRequired(maxOrd)` bits: a doc's 100 values
//!     of a 128-term field take 100 bytes, not 400)
//! ```
//!
//! Numeric columns are read in place from the mapped file (no decode
//! step). Range queries on single-valued columns binary-search the points
//! index (Lucene's 1-D BKD equivalent) and set the matching docs straight
//! into a bitset, so their cost is `O(log n + min(matches, n - matches))`
//! whatever the value/doc correlation; columns without it (older segments,
//! multi-valued columns) skip or fully accept 256-value blocks via the
//! min/max table.

use super::varint::{read_bytes, read_u32_le, read_vint, write_vint};
use crate::error::{Result, RusticError};
use crate::fast_fields::FastFields;
use crate::types::DocId;

pub const BLOCK_SIZE: usize = 256;

const TAG_I64: u8 = 1;
const TAG_U64: u8 = 2;
const TAG_F64: u8 = 3;
const TAG_BOOL: u8 = 4;
pub(crate) const TAG_KEYWORD: u8 = 5;
/// Norms: 1-byte SmallFloat field length per doc + u64 total; see [`NormsReader`].
const TAG_NORMS: u8 = 6;
/// Norms whose every doc has the same byte (v27+, [`crate::codec::CONSTANT_NORMS_SINCE`]).
const TAG_CONST_NORMS: u8 = 12;
const TAG_MULTI_I64: u8 = 7;
const TAG_MULTI_U64: u8 = 8;
const TAG_MULTI_F64: u8 = 9;
pub(crate) const TAG_MULTI_KEYWORD: u8 = 10;
/// On-disk tag of a single-valued keyword column holding only its present
/// docs' ordinals (v14+). The directory parser reports it as
/// [`TAG_KEYWORD`] with [`ColumnEntry::sparse`] set, so every reader and
/// merge sees one keyword column kind.
pub(crate) const TAG_SPARSE_KEYWORD: u8 = 11;
/// On-disk tag of a dense single-valued keyword column whose ordinals are
/// packed per block (v28+, [`crate::codec::BLOCK_PACKED_SINCE`]): read as
/// [`TAG_KEYWORD`] with [`ColumnEntry::blocked`] set.
pub(crate) const TAG_BLOCK_KEYWORD: u8 = 13;
/// A multi-valued keyword column of feature values (`token\0` + the hex
/// bits of a [`crate::fieldtypes::features::quantize`]d weight: what
/// `sparse_vector` / `rank_features` doc values hold), stored as a token
/// table, a weight table and per value a packed (token, weight) pair
/// instead of one table entry per distinct pair (v29+, see
/// [`FeaturesReader`]). Reads back the same strings.
pub(crate) const TAG_FEATURES: u8 = 14;
/// A boolean column that knows which docs have a value (v37+,
/// [`crate::codec::BOOL_PRESENCE_SINCE`]): an i64 numeric column of 0 / 1
/// without points (so presence runs and block packing apply: a TSDB flag
/// present on one metricset's series, constant along each). Read through
/// [`BoolReader`] like [`TAG_BOOL`].
pub(crate) const TAG_BOOL_NUMERIC: u8 = 15;

/// How a single-valued keyword column lays out its ordinals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrdLayout {
    /// One `width`-byte ordinal per doc.
    Dense,
    /// Present docs' ordinals, with a presence bitmap ([`TAG_SPARSE_KEYWORD`]).
    Sparse,
    /// Per 256-doc block: the block's smallest ordinal and the others as
    /// bit-packed offsets from it ([`TAG_BLOCK_KEYWORD`]).
    Blocked,
}

impl OrdLayout {
    /// The layout of a single-valued keyword column written with `tag`.
    pub fn from_tag(tag: u8) -> Self {
        match tag {
            TAG_SPARSE_KEYWORD => OrdLayout::Sparse,
            TAG_BLOCK_KEYWORD => OrdLayout::Blocked,
            _ => OrdLayout::Dense,
        }
    }
}

impl From<bool> for OrdLayout {
    /// `sparse`.
    fn from(sparse: bool) -> Self {
        if sparse { OrdLayout::Sparse } else { OrdLayout::Dense }
    }
}

/// Bytes of a blocked column's block entry: `offset << 8 | width` (`u64`,
/// offset into the ordinals area) and the block's smallest ordinal (`u32`).
const ORD_BLOCK_ENTRY: usize = 12;

/// Ordinal `i` of a blocked keyword column's ordinals area.
#[inline(always)]
fn blocked_ord_at(ordinals: &[u8], i: usize) -> u32 {
    let b = (i / BLOCK_SIZE) * ORD_BLOCK_ENTRY;
    let e = u64::from_le_bytes(ordinals[b..b + 8].try_into().unwrap());
    let min = u32::from_le_bytes(ordinals[b + 8..b + 12].try_into().unwrap());
    let w = ((e & ORD_BLOCK_WIDTH) as usize).min(32);
    // A block with docs without a value (v36+, [`ORD_BLOCK_MISSING`]):
    // the all-ones offset is a missing doc; width 0 is all missing.
    let missing = e & ORD_BLOCK_MISSING != 0;
    if w == 0 {
        return if missing { MISSING_ORD } else { min };
    }
    let bit = (i % BLOCK_SIZE) * w;
    let at = (e >> 8) as usize + bit / 8;
    // A corrupt entry (columns open unvalidated, see `read_blocked_ords`)
    // reads as missing, never past the column.
    let Some(bytes) = ordinals.get(at..at + 8) else { return MISSING_ORD };
    let mask = u64::MAX >> (64 - w);
    let raw = (u64::from_le_bytes(bytes.try_into().unwrap()) >> (bit % 8)) & mask;
    if missing && raw == mask {
        return MISSING_ORD;
    }
    min.saturating_add(raw as u32)
}

/// Width bits of a blocked keyword column's block entry.
const ORD_BLOCK_WIDTH: u64 = 0x3F;
/// Flag of a blocked keyword column's block entry (format
/// [`crate::codec::BLOCKED_MISSING_KEYWORD_SINCE`]+): some docs of the
/// block have no value; their offset is all ones (`width` bits), and a
/// width-0 block has no value at all. A TSDB keyword is constant along a
/// series (docs sorted by `_tsid`) but absent from other metricsets'
/// series: per block it takes no bits either way.
const ORD_BLOCK_MISSING: u64 = 0x80;
/// Block-count flag of a numeric column without a points index (Lucene's
/// doc-values skipper only): `values` are followed directly by the presence
/// bitmap of a sparse column.
const NO_POINTS: u32 = 1 << 31;
/// Block-count flag (format [`crate::codec::VALUE_LIST_POINTS_SINCE`]+): a
/// dense column of at most [`VALUE_LISTS_MAX`] distinct values keeps its
/// points index as per-value doc lists ([`ValueLists`]) instead of every
/// doc's id sorted by value (3 bytes a doc): `u32 k`, the `k` values
/// sorted (8 bytes each), `k + 1` cumulative counts (`u32`), per value its
/// sample stride (`u32`) and `k + 1` offsets into the samples (`u32`),
/// then the samples: each value's every `stride`-th doc (`u32`). A
/// position's value is read off the counts; its doc from the nearest
/// sample and a scan of the column's values (strides keep a scan to about
/// 64 docs; a rare value lists every doc).
const VALUE_LISTS: u32 = 1 << 28;
/// Most distinct values a [`VALUE_LISTS`] column holds.
const VALUE_LISTS_MAX: usize = 64;
/// v10: rank table + bitmap + present values + optional present-doc points.
const COMPACT_SPARSE: u32 = 1 << 30;
/// Block-count flag (with [`COMPACT_SPARSE`], format
/// [`crate::codec::COMPACT_RUNS_SINCE`]+): the present docs are given as
/// runs instead of the rank table and bitmap: `u32 r`, `r` × (`u32`
/// start, `u32` end, `u32` rank of the start, `u32` bounds-table index of
/// the start's block), then per [`RUN_DIR_DOCS`] docs the index of the
/// first run ending past them (`u32`). The block min / max table then
/// holds only the blocks with a value (the block count field counts
/// those). A TSDB metric is present on whole series (docs sorted by
/// `_tsid`): a few runs per thousand docs, not 1.5 bits a doc and 16 bytes
/// a block.
const COMPACT_RUNS: u32 = 1 << 27;
/// Docs per entry of a [`COMPACT_RUNS`] column's run directory.
const RUN_DIR_DOCS: usize = 512;

/// Whether some run of `runs` (ascending, disjoint `[start, end)`) has a
/// doc in block `b`.
fn runs_cover(runs: &[(u32, u32)], b: usize) -> bool {
    let (lo, hi) = ((b * BLOCK_SIZE) as u32, ((b + 1) * BLOCK_SIZE) as u32);
    let k = runs.partition_point(|r| r.1 <= lo);
    k < runs.len() && runs[k].0 < hi
}
/// v11: values stored as narrow offsets from a base, points-index docs in
/// fewer than 4 bytes; see [`Packing`].
const PACKED: u32 = 1 << 29;

/// Top bit of the packing header's `value_width` byte: values are indexes
/// into a table of distinct values (v14+).
const TABLE: u8 = 0x80;
/// Bit of the packing header's `value_width` byte: values are packed per
/// block (v28+, see [`Packing::blocked`]).
const BLOCKED: u8 = 0x40;
/// Widest offset a block packs (one 8-byte load reads any value).
const BLOCKED_MAX_BITS: u32 = 56;
/// Bit of the packing header's `value_width` byte (with [`BLOCKED`], v30+):
/// a float column's values are integers over `10^e` (`e` and each block's
/// integer minimum follow the header), packed per block like an integer
/// column. Every value round-trips exactly (checked when written).
const DECIMAL: u8 = 0x20;
/// Bit of the packing header's `value_width` byte (with [`BLOCKED`], not
/// [`DECIMAL`]; format [`crate::codec::DELTA_BLOCKS_SINCE`]+): blocks may
/// store their values as deltas. Per block [`DELTA_HEAD`] bytes follow the
/// header: the first value's offset from the block minimum (`u32`, in
/// units of the GCD), the smallest delta (`i32`; `i32::MIN`: the block
/// holds offsets), and the sums of entries `[0, 64)`, `[0, 128)` and
/// `[0, 192)` (`u16` each). A delta block's `i`-th entry is
/// `delta_i - min delta` (entry 0 is 0) in 1, 2, 4 or 8 bits, so value `j`
/// is the first offset plus `j * min delta` plus the sum of entries
/// `0..=j`: a checkpoint and a popcount per bit plane of at most `w`
/// words. An index-sorted timestamp (deltas 0 or one second) takes a bit
/// or two a doc instead of the bits of its block's span.
const DELTA: u8 = 0x10;
/// Bytes of a [`DELTA`] block head.
const DELTA_HEAD: usize = 16;
/// Largest decimal exponent a float column takes.
const DECIMAL_MAX_EXP: usize = 9;
const POW10: [f64; DECIMAL_MAX_EXP + 1] = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9];

/// Most distinct values a table-packed column holds (2-byte indexes).
const TABLE_MAX: usize = 1 << 16;

/// How a numeric column's values and points-index docs are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Packing {
    /// Bytes per value (0-8); 8 is the plain value.
    value_width: usize,
    /// Bytes per points-index doc (3 or 4).
    point_width: usize,
    /// Bits every stored value is offset from; with `table`, the number of
    /// table entries.
    base: u64,
    /// Values are indexes into a table of distinct values.
    table: bool,
    /// v28 block packing ([`BLOCKED`]): `values` holds the blocks' running
    /// bit widths (`u32`, one more than the blocks), then each block's
    /// values as `width`-bit offsets from its minimum in units of `base`
    /// (the column's GCD), 32 bytes per bit of width.
    blocked: bool,
}

/// The distinct value bits of a column, if there are at most [`TABLE_MAX`]:
/// an open-addressing set with linear probing (one pass, no allocation per
/// value), turned into the sorted table and an index lookup.
struct DistinctBits {
    keys: Vec<u64>,
    /// Per slot: [`Self::EMPTY`], or (after [`Self::into_table`]) the
    /// key's table index.
    idx: Vec<u32>,
    mask: usize,
    shift: u32,
    count: usize,
}

impl DistinctBits {
    const EMPTY: u32 = u32::MAX;
    /// A set for up to `values` values (at most [`TABLE_MAX`] distinct):
    /// slots for twice the distinct count it can reach.
    fn new(values: usize) -> Self {
        let slots = (values.min(TABLE_MAX) * 2).next_power_of_two().max(16);
        Self { keys: vec![0; slots], idx: vec![Self::EMPTY; slots], mask: slots - 1, shift: 64 - slots.trailing_zeros(), count: 0 }
    }
    #[inline]
    fn slot(&self, bits: u64) -> usize {
        (bits.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }
    /// Add `bits`; false once there would be more than [`TABLE_MAX`] distinct.
    fn insert(&mut self, bits: u64) -> bool {
        let mut i = self.slot(bits);
        loop {
            if self.idx[i] == Self::EMPTY {
                if self.count == TABLE_MAX || self.count * 2 >= self.keys.len() {
                    return false;
                }
                self.keys[i] = bits;
                self.idx[i] = 0;
                self.count += 1;
                return true;
            }
            if self.keys[i] == bits {
                return true;
            }
            i = (i + 1) & self.mask;
        }
    }
    /// The distinct bits sorted by `T`'s total order (ties, i.e. NaN
    /// payloads, by bits); afterwards [`Self::index`] gives each one's
    /// position.
    fn into_table<T: FastValue>(&mut self) -> Vec<u64> {
        let mut table: Vec<u64> = (0..self.keys.len()).filter(|&i| self.idx[i] != Self::EMPTY).map(|i| self.keys[i]).collect();
        table.sort_unstable_by(|a, b| T::from_le(a.to_le_bytes()).sort_cmp(&T::from_le(b.to_le_bytes())).then(a.cmp(b)));
        for (k, &bits) in table.iter().enumerate() {
            let mut i = self.slot(bits);
            while self.keys[i] != bits || self.idx[i] == Self::EMPTY {
                i = (i + 1) & self.mask;
            }
            self.idx[i] = k as u32;
        }
        table
    }
    /// Table index of `bits` (which was inserted).
    #[inline]
    fn index(&self, bits: u64) -> u32 {
        let mut i = self.slot(bits);
        while self.keys[i] != bits || self.idx[i] == Self::EMPTY {
            i = (i + 1) & self.mask;
        }
        self.idx[i]
    }
}

impl Packing {
    const PLAIN: Packing = Packing { value_width: 8, point_width: 4, base: 0, table: false, blocked: false };

    /// The narrowest layout for the `values` that will be written, in a
    /// column of `num_docs` docs. Integers are offsets from their minimum;
    /// floats pack only when every value has the same bits. With `tables`
    /// (format v14+), a column of `count` values with at most
    /// [`TABLE_MAX`] distinct ones takes table indexes instead when that is
    /// smaller (floats with a few thousand distinct values, dates at day
    /// granularity); the table and its lookup come back with the packing.
    fn choose<T: FastValue>(values: impl Iterator<Item = T>, num_docs: u32, count: usize, tables: bool) -> (Packing, Option<(Vec<u64>, DistinctBits)>) {
        let bits = |v: T| u64::from_le_bytes(v.to_le());
        let point_width = if num_docs <= 1 << 24 { 3 } else { 4 };
        let mut range: Option<(T, T)> = None;
        let mut same = true;
        let mut distinct = tables.then(|| DistinctBits::new(count));
        for v in values {
            if let Some(d) = &mut distinct {
                if !d.insert(bits(v)) {
                    distinct = None;
                }
            }
            range = Some(match range {
                None => (v, v),
                Some((lo, hi)) => {
                    same &= bits(v) == bits(lo);
                    (if v < lo { v } else { lo }, if v > hi { v } else { hi })
                }
            });
        }
        let Some((lo, hi)) = range else { return (Packing { value_width: 0, point_width, base: 0, table: false, blocked: false }, None) };
        let value_width = if same {
            0
        } else if T::TOTAL_ORDER {
            (64 - bits(hi).wrapping_sub(bits(lo)).leading_zeros() as usize).div_ceil(8)
        } else {
            8
        };
        let offset = Packing { value_width, point_width, base: if value_width == 8 { 0 } else { bits(lo) }, table: false, blocked: false };
        if let Some(mut d) = distinct.filter(|_| value_width > 1) {
            let k = d.count;
            let width = if k <= 1 << 8 { 1 } else { 2 };
            if count * width + k * 8 < count * value_width {
                let table = d.into_table::<T>();
                return (Packing { value_width: width, point_width, base: k as u64, table: true, blocked: false }, Some((table, d)));
            }
        }
        (offset, None)
    }

    fn write_value<T: FastValue>(&self, v: T, out: &mut Vec<u8>) {
        let delta = u64::from_le_bytes(v.to_le()).wrapping_sub(self.base);
        out.extend_from_slice(&delta.to_le_bytes()[..self.value_width]);
    }

    /// The bytes written for `v`: its offset from `base`, or its index in
    /// the `table` lookup of a table-packed column.
    fn write_packed<T: FastValue>(&self, v: T, table: Option<&DistinctBits>, out: &mut Vec<u8>) {
        match table {
            Some(t) if self.table => out.extend_from_slice(&(t.index(u64::from_le_bytes(v.to_le())) as u64).to_le_bytes()[..self.value_width]),
            _ => self.write_value(v, out),
        }
    }

    fn write_doc(&self, d: u32, out: &mut Vec<u8>) {
        out.extend_from_slice(&d.to_le_bytes()[..self.point_width]);
    }

    /// Zero bytes after the values that let one 8-byte load read any value.
    fn value_pad(&self) -> usize {
        if (1..8).contains(&self.value_width) { 7 } else { 0 }
    }

    /// Zero bytes after the points docs that let one 4-byte load read any doc.
    fn point_pad(&self) -> usize {
        4 - self.point_width
    }
}

/// A fixed-width 8-byte value readable from a column.
pub trait FastValue: Copy + PartialOrd + Send + Sync + 'static {
    /// Whether `PartialOrd` is total (no NaN): a block's min/max then bound
    /// every value in it, so a block inside a range is accepted unscanned.
    const TOTAL_ORDER: bool;
    /// Total order used by the points index: consistent with `PartialOrd`
    /// on comparable values, NaNs (which match no range) last.
    fn sort_cmp(&self, other: &Self) -> std::cmp::Ordering;
    /// Optional cached key with exactly the same ordering/equality as
    /// `sort_cmp`. Float columns avoid repeated NaN tests and indirect loads.
    fn points_sort_key(self) -> Option<u64> { None }
    /// Whether this value is NaN (only meaningful when `!TOTAL_ORDER`).
    fn is_nan(&self) -> bool;
    fn from_le(bytes: [u8; 8]) -> Self;
    fn to_le(self) -> [u8; 8];
    fn zero() -> Self;
    /// ES's doc value of a `scaled_float` stored as `self` (the source
    /// value rounded to `long / scale`): `round(self * scale) * inv`, where
    /// `inv` is `1 / scale` (ES `ScaledFloatLeafFieldData`: the scaled long
    /// times the inverse factor). Identity for integer columns.
    fn es_scaled(self, _scale: f64, _inv: f64) -> Self {
        self
    }
    /// Wrapping addition (for multi-value sums).
    fn add(self, other: Self) -> Self;
    /// Lossy conversion to f64 (for multi-value averages).
    fn to_f64(self) -> f64;
}

impl FastValue for i64 {
    const TOTAL_ORDER: bool = true;
    fn sort_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
    fn is_nan(&self) -> bool {
        false
    }
    fn from_le(b: [u8; 8]) -> Self {
        i64::from_le_bytes(b)
    }
    fn to_le(self) -> [u8; 8] {
        self.to_le_bytes()
    }
    fn zero() -> Self {
        0
    }
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FastValue for u64 {
    const TOTAL_ORDER: bool = true;
    fn sort_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
    fn is_nan(&self) -> bool {
        false
    }
    fn from_le(b: [u8; 8]) -> Self {
        u64::from_le_bytes(b)
    }
    fn to_le(self) -> [u8; 8] {
        self.to_le_bytes()
    }
    fn zero() -> Self {
        0
    }
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FastValue for f64 {
    const TOTAL_ORDER: bool = false;
    fn points_sort_key(self) -> Option<u64> {
        if self.is_nan() { return Some(u64::MAX); }
        // Both signed zeros compare equal, as do all NaN payloads/signs.
        let bits = if self == 0.0 { 0 } else { self.to_bits() };
        Some(if bits >> 63 != 0 { !bits } else { bits ^ (1 << 63) })
    }
    fn sort_cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Total order consistent with `PartialOrd` (so `-0.0 == 0.0`), with
        // NaNs last. `total_cmp` would order `-0.0 < 0.0`, which disagrees
        // with the `>=`/`<=` range predicate and misplaces the range.
        // NaNs compare Equal to each other (the caller breaks ties by doc id),
        // so the points index is sorted with NaNs last.
        match (self.is_nan(), other.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self.partial_cmp(other).unwrap(),
        }
    }
    fn is_nan(&self) -> bool {
        <f64>::is_nan(*self)
    }
    #[inline(always)]
    fn es_scaled(self, scale: f64, inv: f64) -> Self {
        // Infinities and NaN pass through unchanged (no branch: the loops
        // over a block vectorize).
        (self * scale).round() * inv
    }
    fn from_le(b: [u8; 8]) -> Self {
        f64::from_le_bytes(b)
    }
    fn to_le(self) -> [u8; 8] {
        self.to_le_bytes()
    }
    fn zero() -> Self {
        0.0
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// `present`: the builder column's presence bits (`None`: every doc has a
/// value). When some doc in `0..num_docs` has no value, the column body ends
/// with a presence bitmap (`ceil(n / 64)` LE words) after the points index, so
/// a missing doc is not read back as the padding `0` (range hits, `exists`,
/// `get`). Block min/max then cover present values only.
///
/// `points: false` leaves the points index out; the column then flags this in
/// the high bit of its block count (see [`NO_POINTS`]).
fn encode_numeric<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, out: &mut Vec<u8>) {
    encode_numeric_v(values, present, num_docs, points, false, out)
}

/// [`encode_numeric`]; `tables`: format v14+, values may be table-packed.
fn encode_numeric_v<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, out: &mut Vec<u8>) {
    encode_numeric_vb(values, present, num_docs, points, tables, false, out)
}

/// The per-block layout of a dense integer column (`mins`: each block's
/// minimum): (GCD of the offsets, running bit widths), when every block
/// fits [`BLOCKED_MAX_BITS`].
fn block_packing<T: FastValue>(n: usize, value_at: &dyn Fn(usize) -> T, mins: &[T]) -> Option<(u64, Vec<u32>)> {
    block_packing_tv(n, value_at, mins, false)
}

/// [`block_packing`]; `two_value` (format
/// [`crate::codec::TWO_VALUE_BLOCKS_SINCE`]+): a block of exactly two
/// distinct values takes 1 bit a value (its minimum or maximum, see
/// [`ColumnReader::blocked_at`]) whatever their distance: a TSDB column
/// constant along each series (`_ts_routing_hash`) changes value at most
/// once or twice in a block.
fn block_packing_tv<T: FastValue>(n: usize, value_at: &dyn Fn(usize) -> T, mins: &[T], two_value: bool) -> Option<(u64, Vec<u32>)> {
    let bits = |v: T| u64::from_le_bytes(v.to_le());
    let mut gcd = 0u64;
    for i in 0..n {
        let d = bits(value_at(i)).wrapping_sub(bits(mins[i / BLOCK_SIZE]));
        if d != 0 {
            gcd = gcd_u64(gcd, d);
            if gcd == 1 {
                break;
            }
        }
    }
    let gcd = gcd.max(1);
    let mut cum = Vec::with_capacity(mins.len() + 1);
    cum.push(0u32);
    for (b, &m) in mins.iter().enumerate() {
        let start = b * BLOCK_SIZE;
        let max = (start..(start + BLOCK_SIZE).min(n)).map(|i| bits(value_at(i)).wrapping_sub(bits(m)) / gcd).max().unwrap_or(0);
        let mut w = 64 - max.leading_zeros();
        if two_value && w > 1 {
            let top = bits(m).wrapping_add(max.wrapping_mul(gcd));
            if (start..(start + BLOCK_SIZE).min(n)).all(|i| { let v = bits(value_at(i)); v == bits(m) || v == top }) {
                w = 1;
            }
        }
        if w > BLOCKED_MAX_BITS {
            return None;
        }
        cum.push(cum.last().copied().unwrap_or(0).checked_add(w)?);
    }
    Some((gcd, cum))
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// [`encode_numeric_v`]; `blocked`: format v28+, a dense integer column
/// may pack its values per block when that is at least a tenth smaller.
fn encode_numeric_vb<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, blocked: bool, out: &mut Vec<u8>) {
    encode_numeric_vbd(values, present, num_docs, points, tables, blocked, false, out)
}

/// A float column's values as integers over the smallest `10^e` that
/// round-trips every one exactly (`(k as f64) / 10^e` has the value's
/// bits), packed per block: `(e, integers, block minimums, GCD, running
/// widths)`. `None` when no exponent fits or a block is too wide.
#[allow(clippy::type_complexity)]
fn decimal_packing(n: usize, value: &dyn Fn(usize) -> f64) -> Option<(u8, Vec<i64>, Vec<i64>, u64, Vec<u32>)> {
    'exp: for (e, &scale) in POW10.iter().enumerate() {
        let mut ints = Vec::with_capacity(n);
        for i in 0..n {
            let v = value(i);
            let k = (v * scale).round();
            // Exact integers only (and no NaN / infinity).
            if !(k.abs() < 9.0e15) {
                continue 'exp;
            }
            let k = k as i64;
            if ((k as f64) / scale).to_bits() != v.to_bits() {
                continue 'exp;
            }
            ints.push(k);
        }
        let mins: Vec<i64> = ints.chunks(BLOCK_SIZE).map(|c| c.iter().copied().min().unwrap_or(0)).collect();
        let (gcd, cum) = block_packing::<i64>(n, &|i| ints[i], &mins)?;
        return Some((e as u8, ints, mins, gcd, cum));
    }
    None
}

/// [`encode_numeric_vb`]; `decimal`: format v30+, a dense float column may
/// store its values as integers over `10^e` packed per block ([`DECIMAL`]).
#[allow(clippy::too_many_arguments)]
fn encode_numeric_vbd<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, blocked: bool, decimal: bool, out: &mut Vec<u8>) {
    encode_numeric_vbdl(values, present, num_docs, points, tables, blocked, decimal, false, out)
}

/// The sorted distinct values of a dense column eligible for a
/// [`VALUE_LISTS`] points index (at most [`VALUE_LISTS_MAX`], no NaN).
fn value_list_values<T: FastValue>(values: &[T], n: usize) -> Option<Vec<T>> {
    let mut distinct: Vec<T> = Vec::new();
    let mut seen: std::collections::HashSet<[u8; 8]> = std::collections::HashSet::new();
    for i in 0..n {
        let v = values.get(i).copied().unwrap_or_else(T::zero);
        if v.is_nan() {
            return None;
        }
        if seen.insert(v.to_le()) {
            if distinct.len() == VALUE_LISTS_MAX {
                return None;
            }
            distinct.push(v);
        }
    }
    distinct.sort_by(|a, b| a.sort_cmp(b));
    // Values equal under the order but of other bits (-0.0 / 0.0) would
    // share a position run: keep the general index for them.
    if distinct.windows(2).any(|w| w[0].sort_cmp(&w[1]) == std::cmp::Ordering::Equal) {
        return None;
    }
    Some(distinct)
}

/// Write the points index of a column whose block word is at
/// `col_start + 4`: per-value doc lists when `lists` holds the distinct
/// values ([`VALUE_LISTS`], flagged in the block word), else every doc
/// sorted by value.
fn write_points<T: FastValue>(values: &[T], num_docs: u32, pack: Packing, lists: Option<&[T]>, col_start: usize, out: &mut Vec<u8>) {
    let Some(distinct) = lists else {
        encode_points_order(values, num_docs, pack, out);
        out.resize(out.len() + pack.point_pad(), 0);
        return;
    };
    let n = num_docs as usize;
    let k = distinct.len();
    let index = |v: T| distinct.binary_search_by(|x| x.sort_cmp(&v)).unwrap_or(0);
    let mut docs: Vec<Vec<u32>> = vec![Vec::new(); k];
    for d in 0..n {
        docs[index(values.get(d).copied().unwrap_or_else(T::zero))].push(d as u32);
    }
    let w = u32::from_le_bytes(out[col_start + 4..col_start + 8].try_into().unwrap()) | VALUE_LISTS;
    out[col_start + 4..col_start + 8].copy_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&(k as u32).to_le_bytes());
    for v in distinct {
        out.extend_from_slice(&v.to_le());
    }
    let mut start = 0u32;
    out.extend_from_slice(&start.to_le_bytes());
    for list in &docs {
        start += list.len() as u32;
        out.extend_from_slice(&start.to_le_bytes());
    }
    // A stride keeps a scan between samples to about 64 docs.
    let strides: Vec<usize> = docs.iter().map(|l| ((64 * l.len()) / n.max(1)).max(1)).collect();
    for s in &strides {
        out.extend_from_slice(&(*s as u32).to_le_bytes());
    }
    let mut off = 0u32;
    out.extend_from_slice(&off.to_le_bytes());
    for (list, s) in docs.iter().zip(&strides) {
        off += list.len().div_ceil(*s) as u32;
        out.extend_from_slice(&off.to_le_bytes());
    }
    for (list, s) in docs.iter().zip(&strides) {
        for d in list.iter().step_by(*s) {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
}

/// [`encode_numeric_vbd`]; `lists`: format v34+, a dense column of few
/// distinct values keeps per-value doc lists as its points index.
#[allow(clippy::too_many_arguments)]
fn encode_numeric_vbdl<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, blocked: bool, decimal: bool, lists: bool, out: &mut Vec<u8>) {
    encode_numeric_vbdld(values, present, num_docs, points, tables, blocked, decimal, lists, false, out)
}

/// The per-block delta layout of a block-packed integer column ([`DELTA`]):
/// per block its head (first offset, smallest delta or `i32::MIN`) and the
/// running widths, when deltas make the column at least a twentieth
/// smaller than [`block_packing`]'s `cum`.
fn delta_packing<T: FastValue>(n: usize, value_at: &dyn Fn(usize) -> T, mins: &[T], gcd: u64, cum: &[u32]) -> Option<(Vec<(u32, i32, [u16; 3])>, Vec<u32>)> {
    let bits = |v: T| u64::from_le_bytes(v.to_le());
    let mut heads = Vec::with_capacity(mins.len());
    let mut dcum = Vec::with_capacity(cum.len());
    dcum.push(0u32);
    for (b, &m) in mins.iter().enumerate() {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n);
        let plain = cum[b + 1] - cum[b];
        let off = |i: usize| (bits(value_at(i)).wrapping_sub(bits(m)) / gcd) as i64;
        let first = off(start);
        let mut dmin = i64::MAX;
        let mut dmax = i64::MIN;
        for i in start + 1..end {
            let d = off(i) - off(i - 1);
            dmin = dmin.min(d);
            dmax = dmax.max(d);
        }
        let plain_head = (0, i32::MIN, [0u16; 3]);
        let (head, w) = if end - start > 1 && first <= u32::MAX as i64 && dmin > i32::MIN as i64 && dmin <= i32::MAX as i64 {
            // Widths dividing 64 (a popcount per bit plane), at most 8 (u16
            // checkpoints hold 192 entries).
            let w = match 64 - ((dmax - dmin) as u64).leading_zeros() { 0 => 0, 1 => 1, 2 => 2, 3 | 4 => 4, 5..=8 => 8, _ => u32::MAX };
            if w < plain {
                let mut cps = [0u16; 3];
                let mut sum = 0u64;
                for i in start + 1..end {
                    if (i - start) % 64 == 0 {
                        cps[(i - start) / 64 - 1] = sum as u16;
                    }
                    sum += (off(i) - off(i - 1) - dmin) as u64;
                }
                for q in (end - start).div_ceil(64).max(1)..4 {
                    cps[q - 1] = sum as u16;
                }
                ((first as u32, dmin as i32, cps), w)
            } else {
                (plain_head, plain)
            }
        } else {
            (plain_head, plain)
        };
        heads.push(head);
        dcum.push(dcum.last().copied().unwrap_or(0) + w);
    }
    let size = |c: &[u32]| *c.last().unwrap_or(&0) as usize * 32;
    (size(&dcum) + heads.len() * DELTA_HEAD < size(cum) * 19 / 20).then_some((heads, dcum))
}

/// Bit plane `b` of `w`-bit entries in a word (`w` in 1, 2, 4, 8).
const fn plane_masks() -> [[u64; 8]; 9] {
    let mut m = [[0u64; 8]; 9];
    let mut w = 1;
    while w <= 8 {
        let mut b = 0;
        while b < w {
            let mut p = b;
            while p < 64 {
                m[w][b] |= 1 << p;
                p += w;
            }
            b += 1;
        }
        w *= 2;
    }
    m
}
const PLANE_MASKS: [[u64; 8]; 9] = plane_masks();

/// The sum of the first `count` `w`-bit entries of `bytes` (LSB first;
/// `bytes` holds at least `count * w` bits plus 8 bytes): a popcount per
/// bit plane when `w` is 1, 2, 4 or 8.
#[inline]
fn packed_prefix_sum(bytes: &[u8], w: usize, count: usize) -> u64 {
    let total = count * w;
    let word = |k: usize| bytes.get(k * 8..k * 8 + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
    let mut sum = 0u64;
    if matches!(w, 1 | 2 | 4 | 8) {
        let (full, rem) = (total / 64, total % 64);
        let tail = if rem > 0 { word(full) & ((1u64 << rem) - 1) } else { 0 };
        for (b, &m) in PLANE_MASKS[w][..w].iter().enumerate() {
            let mut c = (tail & m).count_ones() as u64;
            for k in 0..full {
                c += (word(k) & m).count_ones() as u64;
            }
            sum += c << b;
        }
        return sum;
    }
    let mask = if w == 0 { 0 } else { u64::MAX >> (64 - w) };
    for i in 0..count {
        let bit = i * w;
        let raw = bytes.get(bit / 8..bit / 8 + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()));
        sum += (raw >> (bit % 8)) & mask;
    }
    sum
}

/// [`encode_numeric_vbdl`]; `delta`: format v34+, a block-packed integer
/// column's blocks may hold deltas ([`DELTA`]).
#[allow(clippy::too_many_arguments)]
fn encode_numeric_vbdld<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, blocked: bool, decimal: bool, lists: bool, delta: bool, out: &mut Vec<u8>) {
    encode_numeric_vbdldr(values, present, num_docs, points, tables, blocked, decimal, lists, delta, false, out)
}

/// [`encode_numeric_vbdld`]; `compact_v36`: format v36+, a compact sparse
/// column may give its presence as runs ([`COMPACT_RUNS`]) and pack its
/// (integer) values per block of 256 present values.
#[allow(clippy::too_many_arguments)]
fn encode_numeric_vbdldr<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, blocked: bool, decimal: bool, lists: bool, delta: bool, compact_v36: bool, out: &mut Vec<u8>) {
    encode_numeric_v37(values, present, num_docs, points, tables, blocked, decimal, lists, delta, compact_v36, false, out)
}

/// [`encode_numeric_vbdldr`]; `two_value`: format v37+, a block-packed
/// integer block of two values takes 1 bit ([`block_packing_tv`]).
#[allow(clippy::too_many_arguments)]
fn encode_numeric_v37<T: FastValue>(values: &[T], present: Option<&[u64]>, num_docs: u32, points: bool, tables: bool, blocked: bool, decimal: bool, lists: bool, delta: bool, compact_v36: bool, two_value: bool, out: &mut Vec<u8>) {
    let col_start = out.len();
    let n = num_docs as usize;
    let num_blocks = n.div_ceil(BLOCK_SIZE);
    let nwords = n.div_ceil(64);
    let word = |w: usize| present.map_or(u64::MAX, |p| p.get(w).copied().unwrap_or(0));
    // Mask of the valid bits of word `w`.
    let valid = |w: usize| if (w + 1) * 64 <= n { u64::MAX } else { u64::MAX >> (64 - (n - w * 64)) };
    let sparse = present.is_some() && (0..nwords).any(|w| word(w) & valid(w) != valid(w));
    let is_p = |i: usize| !sparse || (word(i / 64) >> (i % 64)) & 1 == 1;
    let present_count: usize = (0..nwords).map(|w| (word(w) & valid(w)).count_ones() as usize).sum();
    let list_values = if lists && points && !sparse && n > 1 { value_list_values(values, n) } else { None };
    // A dense column of one value (an `event.ingested` every doc shares)
    // needs no points index: a range holds every doc or none, which the
    // block min/max answer per block (Lucene's BKD stores such a column in
    // a few bytes; the docs-by-value array cost 3 bytes a doc).
    let points = points && !(!sparse && n > 1 && {
        let first = values.first().copied().unwrap_or_else(T::zero);
        !first.is_nan() && (0..n).all(|i| values.get(i).copied().unwrap_or_else(T::zero).to_le() == first.to_le())
    });
    let width = if points { 12 } else { 8 };
    let compact = sparse && 4 + nwords * 4 + present_count * width < n * width;
    let value_at = |i: usize| values.get(i).copied().unwrap_or_else(T::zero);
    // The values written: present ones for a compact column, every doc's
    // (padding included) otherwise.
    let (pack, table) = if compact {
        Packing::choose((0..n).filter(|&i| is_p(i)).map(value_at), num_docs, present_count, tables)
    } else {
        Packing::choose((0..n).map(value_at), num_docs, n, tables)
    };
    let packed = pack.value_width < 8 || (points && pack.point_width < 4);
    let pack = if packed { pack } else { Packing::PLAIN };
    let lookup = table.as_ref().map(|(_, d)| d);
    out.extend_from_slice(&num_docs.to_le_bytes());
    out.extend_from_slice(&(num_blocks as u32 | if points { 0 } else { NO_POINTS }
        | if compact { COMPACT_SPARSE } else { 0 } | if packed { PACKED } else { 0 }).to_le_bytes());
    let mut compact_docs = Vec::new();
    let (mins, maxs) = if compact {
        compact_docs.reserve(present_count);
        let mut mins = vec![T::zero(); num_blocks];
        let mut maxs = vec![T::zero(); num_blocks];
        let mut seen = vec![false; num_blocks];
        for w in 0..nwords {
            let mut bits = word(w) & valid(w);
            while bits != 0 {
                let d = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                compact_docs.push(d as u32);
                let b = d / BLOCK_SIZE;
                let v = value_at(d);
                if !seen[b] {
                    mins[b] = v;
                    maxs[b] = v;
                    seen[b] = true;
                } else {
                    if v < mins[b] { mins[b] = v; }
                    if v > maxs[b] { maxs[b] = v; }
                }
            }
        }
        (mins, maxs)
    } else {
    let mut mins = Vec::with_capacity(num_blocks);
    let mut maxs = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n);
        let mut it = (start..end).filter(|&i| is_p(i)).map(value_at);
        let (mut min, mut max) = match it.next() {
            Some(v) => (v, v),
            None => (T::zero(), T::zero()),
        };
        for v in it {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        mins.push(min);
        maxs.push(max);
    }
        (mins, maxs)
    };
    // v36: a compact column's present docs as runs, when smaller than the
    // rank table and bitmap.
    let runs: Option<Vec<(u32, u32)>> = (compact_v36 && compact).then(|| {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for &d in &compact_docs {
            match runs.last_mut() {
                Some(r) if r.1 == d => r.1 = d + 1,
                _ => runs.push((d, d + 1)),
            }
        }
        runs
    }).filter(|runs| {
        let filled = (0..num_blocks).filter(|&b| runs_cover(runs, b)).count();
        4 + runs.len() * 16 + n.div_ceil(RUN_DIR_DOCS) * 4 + filled * 16 < nwords * 12 + num_blocks * 16
    });
    // v36: a compact integer column's present values packed per block of
    // 256 of them (by rank), when smaller.
    let compact_values: Vec<T> = if compact_v36 && compact && T::TOTAL_ORDER && blocked && present_count > 0 {
        compact_docs.iter().map(|&d| value_at(d as usize)).collect()
    } else {
        Vec::new()
    };
    let compact_layout = (!compact_values.is_empty()).then(|| {
        let vmins: Vec<T> = compact_values.chunks(BLOCK_SIZE).map(|c| c.iter().copied().fold(c[0], |m, v| if v < m { v } else { m })).collect();
        block_packing(present_count, &|r| compact_values[r], &vmins).map(|(gcd, cum)| (vmins, gcd, cum))
    }).flatten().filter(|(vmins, _, cum)| {
        let size = vmins.len() * 8 + cum.len() * 4 + *cum.last().unwrap() as usize * 32 + 8;
        let current = present_count * pack.value_width + pack.value_pad() + table.as_ref().map_or(0, |(t, _)| t.len() * 8);
        size * 10 < current * 9
    });
    // With runs, the bounds of the blocks with a value only, and the block
    // count field counts those.
    let filled: Option<Vec<usize>> = runs.as_ref().map(|runs| (0..num_blocks).filter(|&b| runs_cover(runs, b)).collect());
    if runs.is_some() || compact_layout.is_some() {
        let at = col_start + 4;
        let mut w = u32::from_le_bytes(out[at..at + 4].try_into().unwrap()) | if runs.is_some() { COMPACT_RUNS } else { 0 } | if compact_layout.is_some() { PACKED } else { 0 };
        if let Some(filled) = &filled {
            w = (w & (NO_POINTS | COMPACT_SPARSE | PACKED | VALUE_LISTS | COMPACT_RUNS)) | filled.len() as u32;
        }
        out[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }
    // v28: a dense integer column's values packed per block, when smaller;
    // v36: a sparse one without points too (a doc without a value holds
    // its block's minimum, so it takes no bits; the presence bitmap
    // follows the values).
    let sparse_blocks = compact_v36 && sparse && !points;
    let filled_at = |i: usize| if sparse_blocks && !is_p(i) { mins[i / BLOCK_SIZE] } else { value_at(i) };
    let block_layout = if blocked && (!sparse || sparse_blocks) && !compact && T::TOTAL_ORDER && n > 0 {
        block_packing_tv(n, &filled_at, &mins, two_value).filter(|(_, cum)| {
            let size = cum.len() * 4 + *cum.last().unwrap() as usize * 32 + 8;
            let current = n * pack.value_width + pack.value_pad() + table.as_ref().map_or(0, |(t, _)| t.len() * 8);
            size * 10 < current * 9
        })
    } else {
        None
    };
    // v30: a dense float column of exact decimals, packed per block.
    let decimal_layout = if decimal && blocked && block_layout.is_none() && !sparse && !compact && !T::TOTAL_ORDER && n > 0 {
        decimal_packing(n, &|i| value_at(i).to_f64()).filter(|(_, _, mins, _, cum)| {
            let size = 1 + mins.len() * 8 + cum.len() * 4 + *cum.last().unwrap() as usize * 32 + 8;
            let current = n * pack.value_width + pack.value_pad() + table.as_ref().map_or(0, |(t, _)| t.len() * 8);
            size * 10 < current * 9
        })
    } else {
        None
    };
    if (block_layout.is_some() || decimal_layout.is_some()) && !packed {
        // The packing header carries the block flag: rewrite the block word.
        let at = out.len() - 4;
        let w = u32::from_le_bytes(out[at..].try_into().unwrap()) | PACKED;
        out[at..].copy_from_slice(&w.to_le_bytes());
    }
    match &filled {
        Some(filled) => {
            for &b in filled {
                out.extend_from_slice(&mins[b].to_le());
            }
            for &b in filled {
                out.extend_from_slice(&maxs[b].to_le());
            }
        }
        None => {
            for m in &mins {
                out.extend_from_slice(&m.to_le());
            }
            for m in &maxs {
                out.extend_from_slice(&m.to_le());
            }
        }
    }
    if let Some((exp, ints, int_mins, gcd, cum)) = &decimal_layout {
        let point_width = if packed { pack.point_width } else { Packing::PLAIN.point_width };
        out.push(BLOCKED | DECIMAL);
        out.push(point_width as u8);
        out.extend_from_slice(&gcd.to_le_bytes());
        out.push(*exp);
        for m in int_mins {
            out.extend_from_slice(&m.to_le_bytes());
        }
        for c in cum {
            out.extend_from_slice(&c.to_le_bytes());
        }
        let mut acc = 0u128;
        let mut nbits = 0u32;
        for (b, &m) in int_mins.iter().enumerate() {
            let w = cum[b + 1] - cum[b];
            if w == 0 {
                continue;
            }
            let start = b * BLOCK_SIZE;
            for i in start..start + BLOCK_SIZE {
                let d = if i < n { (ints[i] as u64).wrapping_sub(m as u64) / gcd } else { 0 };
                acc |= (d as u128) << nbits;
                nbits += w;
                while nbits >= 8 {
                    out.push(acc as u8);
                    acc >>= 8;
                    nbits -= 8;
                }
            }
        }
        debug_assert_eq!(nbits, 0, "a block's bits fill whole bytes");
        out.extend_from_slice(&[0u8; 8]);
        let pack = Packing { value_width: 0, point_width, base: *gcd, table: false, blocked: true };
        if !points {
            return;
        }
        write_points(values, num_docs, pack, list_values.as_deref(), col_start, out);
        return;
    }
    if let Some((gcd, cum)) = &block_layout {
        // Points docs keep the chosen width; the values are the blocks'.
        let point_width = if packed { pack.point_width } else { Packing::PLAIN.point_width };
        let deltas = if delta { delta_packing(n, &filled_at, &mins, *gcd, cum) } else { None };
        let cum = deltas.as_ref().map_or(cum, |(_, c)| c);
        out.push(BLOCKED | if deltas.is_some() { DELTA } else { 0 });
        out.push(point_width as u8);
        out.extend_from_slice(&gcd.to_le_bytes());
        if let Some((heads, _)) = &deltas {
            for &(first, dmin, cps) in heads {
                out.extend_from_slice(&first.to_le_bytes());
                out.extend_from_slice(&dmin.to_le_bytes());
                for c in cps {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                out.extend_from_slice(&[0, 0]);
            }
        }
        for c in cum {
            out.extend_from_slice(&c.to_le_bytes());
        }
        let bits = |v: T| u64::from_le_bytes(v.to_le());
        let mut acc = 0u128;
        let mut nbits = 0u32;
        for (b, &m) in mins.iter().enumerate() {
            let w = cum[b + 1] - cum[b];
            if w == 0 {
                continue;
            }
            let start = b * BLOCK_SIZE;
            let head = deltas.as_ref().map(|(h, _)| (h[b].0, h[b].1)).filter(|h| h.1 != i32::MIN);
            let off = |i: usize| bits(filled_at(i)).wrapping_sub(bits(m)) / gcd;
            for i in start..start + BLOCK_SIZE {
                let d = match head {
                    // A delta block: entry 0 is 0, then delta - min delta.
                    Some((_, dmin)) => if i > start && i < n { (off(i) as i64 - off(i - 1) as i64 - dmin as i64) as u64 } else { 0 },
                    // A 1-bit block: whether the value is not the minimum
                    // (the maximum: a two-value block, or min + GCD).
                    None if w == 1 => (i < n && bits(filled_at(i)) != bits(m)) as u64,
                    None => if i < n { off(i) } else { 0 },
                };
                acc |= (d as u128) << nbits;
                nbits += w;
                while nbits >= 8 {
                    out.push(acc as u8);
                    acc >>= 8;
                    nbits -= 8;
                }
            }
        }
        debug_assert_eq!(nbits, 0, "a block's bits fill whole bytes");
        out.extend_from_slice(&[0u8; 8]);
        let pack = Packing { value_width: 0, point_width, base: *gcd, table: false, blocked: true };
        if !points {
            if sparse {
                for w in 0..nwords {
                    out.extend_from_slice(&(word(w) & valid(w)).to_le_bytes());
                }
            }
            return;
        }
        write_points(values, num_docs, pack, list_values.as_deref(), col_start, out);
        return;
    }
    // A compact column packed per block: the points docs keep the chosen
    // width; the values are the blocks'.
    let pack = match &compact_layout {
        Some((_, gcd, _)) => Packing { value_width: 0, point_width: if packed { pack.point_width } else { Packing::PLAIN.point_width }, base: *gcd, table: false, blocked: true },
        None => pack,
    };
    if compact_layout.is_some() {
        out.push(BLOCKED);
        out.push(pack.point_width as u8);
        out.extend_from_slice(&pack.base.to_le_bytes());
    } else if packed {
        out.push(pack.value_width as u8 | if pack.table { TABLE } else { 0 });
        out.push(pack.point_width as u8);
        out.extend_from_slice(&pack.base.to_le_bytes());
        if let Some((table, _)) = &table {
            for b in table {
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
    }
    if compact {
        out.extend_from_slice(&(present_count as u32).to_le_bytes());
        if let Some(runs) = &runs {
            out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
            let filled = filled.as_ref().expect("runs come with their filled blocks");
            let mut rank = 0u32;
            for &(a, b) in runs {
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
                out.extend_from_slice(&rank.to_le_bytes());
                // The bounds-table index of the run's first block.
                let at = filled.partition_point(|&blk| blk < a as usize / BLOCK_SIZE);
                out.extend_from_slice(&(at as u32).to_le_bytes());
                rank += b - a;
            }
            let mut k = 0usize;
            for e in 0..n.div_ceil(RUN_DIR_DOCS) {
                while k < runs.len() && runs[k].1 as usize <= e * RUN_DIR_DOCS {
                    k += 1;
                }
                out.extend_from_slice(&(k as u32).to_le_bytes());
            }
        } else {
            let mut rank = 0u32;
            for w in 0..nwords {
                out.extend_from_slice(&rank.to_le_bytes());
                rank += (word(w) & valid(w)).count_ones();
            }
            for w in 0..nwords { out.extend_from_slice(&(word(w) & valid(w)).to_le_bytes()); }
        }
        let mut docs = compact_docs;
        if let Some((vmins, gcd, cum)) = &compact_layout {
            for m in vmins {
                out.extend_from_slice(&m.to_le());
            }
            for c in cum {
                out.extend_from_slice(&c.to_le_bytes());
            }
            let bits = |v: T| u64::from_le_bytes(v.to_le());
            let mut acc = 0u128;
            let mut nbits = 0u32;
            for (b, &m) in vmins.iter().enumerate() {
                let w = cum[b + 1] - cum[b];
                if w == 0 {
                    continue;
                }
                for r in b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE {
                    let d = if r < present_count { bits(compact_values[r]).wrapping_sub(bits(m)) / gcd } else { 0 };
                    acc |= (d as u128) << nbits;
                    nbits += w;
                    while nbits >= 8 {
                        out.push(acc as u8);
                        acc >>= 8;
                        nbits -= 8;
                    }
                }
            }
            debug_assert_eq!(nbits, 0, "a block's bits fill whole bytes");
            out.extend_from_slice(&[0u8; 8]);
        } else {
            for &d in &docs { pack.write_packed(value_at(d as usize), lookup, out); }
            out.resize(out.len() + pack.value_pad(), 0);
        }
        if points {
            docs.sort_unstable_by(|&a, &b| value_at(a as usize).sort_cmp(&value_at(b as usize)).then(a.cmp(&b)));
            for d in docs { pack.write_doc(d, out); }
            out.resize(out.len() + pack.point_pad(), 0);
        }
        return;
    }
    for i in 0..n {
        pack.write_packed(value_at(i), lookup, out);
    }
    out.resize(out.len() + pack.value_pad(), 0);
    // Points index: doc ids sorted by (value, doc).
    if !points {
        if sparse {
            for w in 0..nwords {
                out.extend_from_slice(&(word(w) & valid(w)).to_le_bytes());
            }
        }
        return;
    }
    write_points(values, num_docs, pack, list_values.as_deref(), col_start, out);
    if sparse {
        for w in 0..nwords {
            out.extend_from_slice(&(word(w) & valid(w)).to_le_bytes());
        }
    }
}

fn encode_points_order<T: FastValue>(values: &[T], num_docs: u32, pack: Packing, out: &mut Vec<u8>) {
    let value_at = |i: u32| values.get(i as usize).copied().unwrap_or_else(T::zero);
    if T::zero().points_sort_key().is_some() {
        // Preserve the cheap monotonic/constant case without a tuple buffer.
        if (1..num_docs).all(|i| !value_at(i - 1).sort_cmp(&value_at(i)).is_gt()) {
            for d in 0..num_docs { pack.write_doc(d, out); }
        } else {
            let mut pairs: Vec<_> = (0..num_docs)
                .map(|d| (value_at(d).points_sort_key().expect("consistent numeric key support"), d)).collect();
            pairs.sort_unstable();
            for (_, d) in pairs { pack.write_doc(d, out); }
        }
    } else {
        let mut order: Vec<u32> = (0..num_docs).collect();
        order.sort_unstable_by(|&a, &b| value_at(a).sort_cmp(&value_at(b)).then(a.cmp(&b)));
        for d in order { pack.write_doc(d, out); }
    }
}

/// Set bits `[from, to)` of `words`.
#[inline]
pub fn set_bit_range(words: &mut [u64], from: usize, to: usize) {
    if from >= to {
        return;
    }
    let (fw, tw) = (from / 64, (to - 1) / 64);
    let first = u64::MAX << (from % 64);
    let last = u64::MAX >> (63 - (to - 1) % 64);
    if fw == tw {
        words[fw] |= first & last;
        return;
    }
    words[fw] |= first;
    for w in &mut words[fw + 1..tw] {
        *w = u64::MAX;
    }
    words[tw] |= last;
}

/// A [`VALUE_LISTS`] points index.
#[derive(Clone, Copy, Debug)]
struct ValueLists<'a> {
    k: usize,
    values: &'a [u8],
    starts: &'a [u8],
    strides: &'a [u8],
    offsets: &'a [u8],
    samples: &'a [u8],
}

#[inline(always)]
fn u32_at(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as usize
}

impl<'a> ValueLists<'a> {
    /// The bytes of the region ([`VALUE_LISTS`]).
    fn byte_len(&self) -> usize {
        4 + self.values.len() + self.starts.len() + self.strides.len() + self.offsets.len() + self.samples.len()
    }

    fn parse(data: &'a [u8], pos: &mut usize, num_docs: usize) -> Result<Self> {
        let bad = || RusticError::Corrupted("numeric value lists".into());
        let k = read_u32_le(data, pos)? as usize;
        if k == 0 || k > VALUE_LISTS_MAX {
            return Err(bad());
        }
        let values = read_bytes(data, pos, k * 8)?;
        let starts = read_bytes(data, pos, (k + 1) * 4)?;
        let strides = read_bytes(data, pos, k * 4)?;
        let offsets = read_bytes(data, pos, (k + 1) * 4)?;
        let total = u32_at(offsets, k);
        let samples = read_bytes(data, pos, total * 4)?;
        let v = ValueLists { k, values, starts, strides, offsets, samples };
        if v.start(0) != 0 || v.start(k) > num_docs || (0..k).any(|t| v.start(t) > v.start(t + 1) || v.stride(t) == 0 || v.offset(t) > v.offset(t + 1)
            || v.offset(t + 1) - v.offset(t) != (v.start(t + 1) - v.start(t)).div_ceil(v.stride(t))) {
            return Err(bad());
        }
        Ok(v)
    }
    #[inline(always)]
    fn start(&self, t: usize) -> usize { u32_at(self.starts, t) }
    #[inline(always)]
    fn stride(&self, t: usize) -> usize { u32_at(self.strides, t) }
    #[inline(always)]
    fn offset(&self, t: usize) -> usize { u32_at(self.offsets, t) }
    #[inline(always)]
    fn value_bits(&self, t: usize) -> [u8; 8] { self.values[t * 8..t * 8 + 8].try_into().unwrap() }
    /// The value index of position `i`.
    #[inline]
    fn value_of(&self, i: usize) -> usize {
        let (mut lo, mut hi) = (0usize, self.k);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.start(mid) <= i { lo = mid } else { hi = mid }
        }
        lo
    }
}

// Column readers are shared across threads (ES|QL and aggregation leaf
// splits): keep them `Sync + Send`.
const _: () = {
    const fn sync_send<T: Sync + Send>() {}
    sync_send::<ColumnReader<'static, i64>>();
    sync_send::<ColumnReader<'static, u64>>();
    sync_send::<ColumnReader<'static, f64>>();
    sync_send::<KeywordReader<'static>>();
    sync_send::<MultiColumnReader<'static, i64>>();
};

/// Ids of [`DELTA`] column readers (never reused: a cache slot of a
/// dropped reader can never answer for another).
static NEXT_DELTA_READER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A thread's decoded [`DELTA`] blocks, a few readers at a time (readers
/// are shared between threads, so the cache is the thread's): per slot the
/// reader id, the last block touched and the block decoded (plus one; 0:
/// none), and that block's value bits.
#[derive(Default)]
struct DeltaSlot {
    reader: u64,
    last: usize,
    decoded: usize,
    vals: Vec<u64>,
}

thread_local! {
    static DELTA_SLOTS: std::cell::RefCell<([DeltaSlot; 4], usize)> = std::cell::RefCell::new(Default::default());
}

/// Zero-copy numeric column reader.
pub struct ColumnReader<'a, T: FastValue> {
    num_docs: u32,
    mins: &'a [u8],
    maxs: &'a [u8],
    values: &'a [u8],
    /// Points index (`u32` docs sorted by value), empty before segment v7.
    sorted: &'a [u8],
    /// The points index as per-value doc lists ([`VALUE_LISTS`]).
    vlists: Option<ValueLists<'a>>,
    /// Presence bitmap (LE `u64` words) of a sparse column; empty when every
    /// doc has a value.
    present: &'a [u8],
    /// Encoded without a points index on purpose ([`NO_POINTS`]).
    skipper: bool,
    ranks: &'a [u8],
    /// A [`COMPACT_RUNS`] column's runs (`r` × start, end, rank) and run
    /// directory; empty otherwise.
    runs: &'a [u8],
    run_dir: &'a [u8],
    /// A compact column packed per block of present values: each block's
    /// minimum (the blocks' base); empty otherwise.
    vmins: &'a [u8],
    compact: bool,
    pack: Packing,
    /// A decimal-packed float column ([`DECIMAL`]): `10^e` (0: not decimal)
    /// and each block's integer minimum (`i64` LE), the blocks' base.
    decimal: f64,
    int_mins: &'a [u8],
    /// Per-block delta heads of a [`DELTA`] column ([`DELTA_HEAD`] bytes a
    /// block).
    deltas: &'a [u8],
    /// This reader's id in the thread's decoded-block cache ([`DeltaSlot`];
    /// 0: not a [`DELTA`] column): a block touched twice in a row (a scan)
    /// is decoded once, a lone read takes a checkpoint and a popcount.
    delta_id: u64,
    /// A table-packed column's distinct values (`u64` LE bits, sorted).
    table: &'a [u8],
    /// `scaled_float` doc values as ES reads them ([`FastValue::es_scaled`]):
    /// the factor and its inverse (`0`: values as stored). Applied by the
    /// value accessors ([`Self::get`], [`Self::block_values`], min / max),
    /// never by the range and points functions, which work on the stored
    /// values.
    es_scale: f64,
    es_inv: f64,
    _t: std::marker::PhantomData<T>,
}

#[inline]
fn read_at<T: FastValue>(bytes: &[u8], i: usize) -> T {
    T::from_le(bytes[i * 8..i * 8 + 8].try_into().unwrap())
}

/// The docs of a single-valued numeric column with a value in `[lower,
/// upper]`, ascending, found as they are asked for
/// ([`ColumnReader::lazy_range`]): exactly the docs of
/// [`ColumnReader::range_bits`], without building its bitset. Blocks outside
/// the range are skipped by their bounds and blocks inside it taken whole
/// (Lucene's doc-values skipper), so a search that stops after its first
/// matches (`size: 10`, `track_total_hits` 10,000) reads only those.
pub struct RangeDocs<'a, T: FastValue> {
    col: ColumnReader<'a, T>,
    lower: T,
    upper: T,
    /// The current doc ([`TERMINATED`](crate::query::docset::TERMINATED)
    /// before the first advance and once exhausted).
    cur: DocId,
    /// The next doc to test (`usize::MAX`: exhausted).
    next: usize,
    /// The block [`RangeDocs::state`] describes, plus one (0: none).
    block: usize,
    /// 0: no doc of the block can match, 1: every doc with a value
    /// matches, 2: each doc's value is tested.
    state: u8,
    /// The docs it holds: exact from the points index when the column has
    /// one ([`ColumnReader::counts_from_points`]), else every doc.
    hint: u32,
}

impl<'a, T: FastValue> RangeDocs<'a, T> {
    fn next_from(&mut self, mut d: usize) -> DocId {
        let n = self.col.num_docs as usize;
        while d < n {
            let b = d / BLOCK_SIZE;
            if self.block != b + 1 {
                self.block = b + 1;
                let (bmin, bmax) = (read_at::<T>(self.col.mins, b), read_at::<T>(self.col.maxs, b));
                let (any, _) = self.col.block_presence(b);
                self.state = if !any || bmax < self.lower || bmin > self.upper {
                    0
                } else if T::TOTAL_ORDER && bmin >= self.lower && bmax <= self.upper {
                    1
                } else {
                    2
                };
            }
            match self.state {
                0 => d = (b + 1) * BLOCK_SIZE,
                1 => {
                    if self.col.has_value(d as DocId) {
                        return d as DocId;
                    }
                    d += 1;
                }
                _ => {
                    if self.col.has_value(d as DocId) {
                        let v = self.col.raw_value(d as DocId);
                        if v >= self.lower && v <= self.upper {
                            return d as DocId;
                        }
                    }
                    d += 1;
                }
            }
        }
        crate::query::docset::TERMINATED
    }
}

impl<T: FastValue> crate::query::docset::DocSet for RangeDocs<'_, T> {
    fn advance(&mut self) -> DocId {
        if self.next >= self.col.num_docs as usize {
            self.cur = crate::query::docset::TERMINATED;
            self.next = usize::MAX;
            return self.cur;
        }
        self.cur = self.next_from(self.next);
        self.next = if self.cur == crate::query::docset::TERMINATED { usize::MAX } else { self.cur as usize + 1 };
        self.cur
    }
    fn seek(&mut self, target: DocId) -> DocId {
        if self.cur != crate::query::docset::TERMINATED && self.cur >= target {
            return self.cur;
        }
        if self.next != usize::MAX {
            self.next = self.next.max(target as usize);
        }
        self.advance()
    }
    fn doc(&self) -> DocId {
        self.cur
    }
    fn size_hint(&self) -> u32 {
        self.hint
    }
    /// A doc's membership is its value (no scan): a conjunction led by a
    /// smaller clause tests its candidates here.
    fn has_fast_exact_seek(&self) -> bool {
        true
    }
    fn exact_seek_reads_value(&self) -> bool {
        true
    }
    fn seek_exact(&mut self, target: DocId) -> bool {
        let n = self.col.num_docs as usize;
        let t = target as usize;
        if target == crate::query::docset::TERMINATED || t >= n {
            self.cur = crate::query::docset::TERMINATED;
            self.next = usize::MAX;
            return false;
        }
        if self.next == usize::MAX || (self.cur != crate::query::docset::TERMINATED && self.cur > target) {
            return false;
        }
        // The block's bounds first (cached per block, as `next_from`
        // walks them): a value is read only in a block the range cuts.
        let b = t / BLOCK_SIZE;
        if self.block != b + 1 {
            self.block = b + 1;
            let (bmin, bmax) = (read_at::<T>(self.col.mins, b), read_at::<T>(self.col.maxs, b));
            let (any, _) = self.col.block_presence(b);
            self.state = if !any || bmax < self.lower || bmin > self.upper {
                0
            } else if T::TOTAL_ORDER && bmin >= self.lower && bmax <= self.upper {
                1
            } else {
                2
            };
        }
        let hit = match self.state {
            0 => false,
            1 => self.col.has_value(target),
            _ => self.col.has_value(target) && {
                let v = self.col.raw_value(target);
                v >= self.lower && v <= self.upper
            },
        };
        // Forward from here, whether or not it matched.
        self.next = t + 1;
        self.cur = if hit { target } else { crate::query::docset::TERMINATED };
        hit
    }
}

impl<'a, T: FastValue> ColumnReader<'a, T> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        Self::open_checked(data, true)
    }

    /// [`Self::open`]; `check_ranks: false` skips re-validating a compact
    /// column's presence ranks (a scan of every rank word: ~0.5 ms over 33M
    /// docs), for column bytes this segment already opened with the check
    /// (immutable, so they still pass). Every bound check stays.
    pub fn open_checked(data: &'a [u8], check_ranks: bool) -> Result<Self> {
        let mut pos = 0;
        let num_docs = read_u32_le(data, &mut pos)? as usize;
        let raw_blocks = read_u32_le(data, &mut pos)?;
        let skipper = raw_blocks & NO_POINTS != 0;
        let compact = raw_blocks & COMPACT_SPARSE != 0;
        let has_runs = raw_blocks & COMPACT_RUNS != 0;
        if has_runs && !compact {
            return Err(RusticError::Corrupted("presence runs without a compact column".into()));
        }
        let num_blocks = (raw_blocks & !(NO_POINTS | COMPACT_SPARSE | PACKED | VALUE_LISTS | COMPACT_RUNS)) as usize;
        let mins = read_bytes(data, &mut pos, num_blocks * 8)?;
        let maxs = read_bytes(data, &mut pos, num_blocks * 8)?;
        let mut table: &'a [u8] = &[];
        let mut decimal = 0.0f64;
        let mut int_mins: &'a [u8] = &[];
        let mut deltas: &'a [u8] = &[];
        let pack = if raw_blocks & PACKED != 0 {
            let h = read_bytes(data, &mut pos, 10)?;
            let is_table = h[0] & TABLE != 0;
            let blocked = h[0] & BLOCKED != 0;
            let is_decimal = h[0] & DECIMAL != 0;
            let is_delta = h[0] & DELTA != 0;
            let pack = Packing { value_width: (h[0] & !(TABLE | BLOCKED | DECIMAL | DELTA)) as usize, point_width: h[1] as usize, base: u64::from_le_bytes(h[2..10].try_into().unwrap()), table: is_table, blocked };
            if blocked && (is_table || pack.value_width != 0 || pack.base == 0) {
                return Err(RusticError::Corrupted("invalid block packing".into()));
            }
            if is_delta {
                if !blocked || is_decimal || compact {
                    return Err(RusticError::Corrupted("invalid delta packing".into()));
                }
                deltas = read_bytes(data, &mut pos, num_blocks * DELTA_HEAD)?;
            }
            if is_decimal {
                // (Size accounting opens any numeric column as integers;
                // only float readers decode decimal values.)
                if !blocked || compact {
                    return Err(RusticError::Corrupted("invalid decimal packing".into()));
                }
                let e = read_bytes(data, &mut pos, 1)?[0] as usize;
                if e > DECIMAL_MAX_EXP {
                    return Err(RusticError::Corrupted("invalid decimal exponent".into()));
                }
                decimal = POW10[e];
                int_mins = read_bytes(data, &mut pos, num_blocks * 8)?;
            }
            if pack.value_width > 8 || !(3..=4).contains(&pack.point_width) {
                return Err(RusticError::Corrupted("invalid numeric packing".into()));
            }
            if is_table {
                let k = pack.base as usize;
                if k == 0 || k > TABLE_MAX || !(1..=2).contains(&pack.value_width) {
                    return Err(RusticError::Corrupted("invalid numeric table packing".into()));
                }
                table = read_bytes(data, &mut pos, k * 8)?;
            }
            pack
        } else {
            Packing::PLAIN
        };
        let (vw, pw) = (pack.value_width, pack.point_width);
        let mut vlists: Option<ValueLists<'a>> = None;
        let (mut runs, mut run_dir, mut vmins): (&'a [u8], &'a [u8], &'a [u8]) = (&[], &[], &[]);
        let (values, sorted, present, ranks) = if compact {
            let count = read_u32_le(data, &mut pos)? as usize;
            if count > num_docs { return Err(RusticError::Corrupted("numeric presence count exceeds docs".into())); }
            let words = num_docs.div_ceil(64);
            let (ranks, present) = if has_runs {
                let r = read_u32_le(data, &mut pos)? as usize;
                runs = read_bytes(data, &mut pos, r.checked_mul(16).ok_or_else(|| RusticError::Corrupted("presence runs".into()))?)?;
                run_dir = read_bytes(data, &mut pos, num_docs.div_ceil(RUN_DIR_DOCS) * 4)?;
                if check_ranks {
                    // Ascending, disjoint, inside the docs, ranks adding up;
                    // the directory in range and ascending.
                    // The bounds table: one entry per block with a value.
                    let (mut rank, mut prev_end, mut filled, mut last_block) = (0usize, 0usize, 0usize, usize::MAX);
                    for k in 0..r {
                        let (a, b, rk, bi) = (u32_at(runs, 4 * k), u32_at(runs, 4 * k + 1), u32_at(runs, 4 * k + 2), u32_at(runs, 4 * k + 3));
                        if a < prev_end || b <= a || b > num_docs || rk != rank {
                            return Err(RusticError::Corrupted("invalid presence runs".into()));
                        }
                        let (first, last) = (a / BLOCK_SIZE, (b - 1) / BLOCK_SIZE);
                        let at = if first == last_block { filled - 1 } else { filled };
                        if bi != at {
                            return Err(RusticError::Corrupted("invalid presence runs".into()));
                        }
                        filled = at + 1 + (last - first);
                        last_block = last;
                        rank += b - a;
                        prev_end = b;
                    }
                    if filled != num_blocks {
                        return Err(RusticError::Corrupted("invalid presence runs".into()));
                    }
                    if rank != count || (0..num_docs.div_ceil(RUN_DIR_DOCS)).any(|e| u32_at(run_dir, e) > r || (e > 0 && u32_at(run_dir, e) < u32_at(run_dir, e - 1))) {
                        return Err(RusticError::Corrupted("invalid presence runs".into()));
                    }
                }
                (&[][..], &[][..])
            } else {
                (read_bytes(data, &mut pos, words * 4)?, read_bytes(data, &mut pos, words * 8)?)
            };
            if check_ranks && !has_runs {
                let mut rank = 0u32;
                for w in 0..words {
                    let stored = u32::from_le_bytes(ranks[w*4..w*4+4].try_into().unwrap());
                    let bits = u64::from_le_bytes(present[w*8..w*8+8].try_into().unwrap());
                    if stored != rank || (w + 1 == words && num_docs % 64 != 0 && bits >> (num_docs % 64) != 0) {
                        return Err(RusticError::Corrupted("invalid numeric presence rank".into()));
                    }
                    rank += bits.count_ones();
                }
                if rank as usize != count { return Err(RusticError::Corrupted("numeric presence count mismatch".into())); }
            }
            let values = if pack.blocked {
                // Per block of present values its minimum, the running
                // widths, the blocks' bits and an 8-byte pad.
                let vb = count.div_ceil(BLOCK_SIZE);
                vmins = read_bytes(data, &mut pos, vb * 8)?;
                let start = pos;
                let cum = read_bytes(data, &mut pos, (vb + 1) * 4)?;
                let total = u32::from_le_bytes(cum[vb * 4..].try_into().unwrap());
                read_bytes(data, &mut pos, total as usize * 32 + 8)?;
                &data[start..pos]
            } else {
                read_bytes(data, &mut pos, count * vw + pack.value_pad())?
            };
            let sorted = if skipper { &[][..] } else { read_bytes(data, &mut pos, count * pw + pack.point_pad())? };
            (values, sorted, present, ranks)
        } else {
            let values = if pack.blocked {
                // Running widths, then the blocks' bits and an 8-byte pad.
                // Only the total is checked here (columns open per fetched
                // hit): a corrupt width reads as an out-of-range value, which
                // `blocked_at` bounds, never past the column.
                let start = pos;
                let cum = read_bytes(data, &mut pos, (num_blocks + 1) * 4)?;
                let total = u32::from_le_bytes(cum[num_blocks * 4..].try_into().unwrap());
                read_bytes(data, &mut pos, total as usize * 32 + 8)?;
                &data[start..pos]
            } else {
                read_bytes(data, &mut pos, num_docs * vw + pack.value_pad())?
            };
            let plen = num_docs * pw + pack.point_pad();
            let sorted = if raw_blocks & VALUE_LISTS != 0 {
                vlists = Some(ValueLists::parse(data, &mut pos, num_docs)?);
                &[]
            } else if !skipper && data.len() - pos >= plen { read_bytes(data, &mut pos, plen)? } else { &[] };
            let pbytes = num_docs.div_ceil(64) * 8;
            let present = if (skipper || !sorted.is_empty()) && num_docs > 0 && data.len() - pos >= pbytes { read_bytes(data, &mut pos, pbytes)? } else { &[] };
            (values, sorted, present, &[][..])
        };
        Ok(Self {
            num_docs: num_docs as u32,
            mins,
            maxs,
            values,
            sorted,
            vlists,
            present,
            skipper,
            ranks,
            runs,
            run_dir,
            vmins,
            compact,
            pack,
            decimal,
            int_mins,
            deltas,
            delta_id: if deltas.is_empty() { 0 } else { NEXT_DELTA_READER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) },
            table,
            es_scale: 0.0,
            es_inv: 0.0,
            _t: std::marker::PhantomData,
        })
    }

    /// Whether the column carries the points index.
    pub fn has_points(&self) -> bool {
        !self.skipper && (self.compact || !self.sorted.is_empty() || self.vlists.is_some() || self.num_docs == 0)
    }

    /// The docs of occurrences `[j0, j1)` of value `t` of a [`VALUE_LISTS`]
    /// index, in doc order: from the nearest sample, scanning the values.
    fn vl_docs(&self, v: &ValueLists<'_>, t: usize, j0: usize, j1: usize, f: &mut dyn FnMut(usize)) {
        let (stride, base) = (v.stride(t), v.offset(t));
        // Samples are bounds-checked here, not on open (a reader opens per
        // query): a corrupt sample past the docs is skipped.
        let n = self.num_docs as usize;
        if stride == 1 {
            for j in j0..j1 {
                let d = u32_at(v.samples, base + j);
                if d < n {
                    f(d);
                }
            }
            return;
        }
        let want = v.value_bits(t);
        let q = j0 / stride;
        let mut count = q * stride;
        let mut d = u32_at(v.samples, base + q);
        while count < j1 && d < n {
            if self.value_at(d).to_le() == want {
                if count >= j0 {
                    f(d);
                }
                count += 1;
            }
            d += 1;
        }
    }

    /// Visit the docs at points-index positions `[lo, hi)` in order.
    fn for_points(&self, lo: usize, hi: usize, f: &mut dyn FnMut(usize)) {
        let Some(v) = &self.vlists else {
            for i in lo..hi {
                f(self.sorted_doc(i));
            }
            return;
        };
        if lo >= hi {
            return;
        }
        let mut t = v.value_of(lo);
        let mut at = lo;
        while at < hi && t < v.k {
            let (s, e) = (v.start(t), v.start(t + 1));
            let end = hi.min(e);
            if at < end {
                self.vl_docs(v, t, at - s, end - s, f);
            }
            at = end.max(at);
            t += 1;
        }
    }

    /// The value at points-index position `i` (the value of
    /// [`Self::point_doc`]`(i)`).
    #[inline]
    pub fn point_value(&self, i: usize) -> T {
        match &self.vlists {
            Some(v) => T::from_le(v.value_bits(v.value_of(i))),
            None => self.raw_value(self.sorted_doc(i) as DocId),
        }
    }

    /// Whether the column was written without a points index on purpose
    /// (doc-values skipper only); a merge keeps it that way.
    pub fn skips_points(&self) -> bool {
        self.skipper
    }

    /// Whether every doc holds one same stored value (dense, two docs or
    /// more): the columns the writer leaves without a points index for
    /// their shape alone, not for a doc-values-only mapping.
    pub fn written_constant(&self) -> bool {
        if self.is_sparse() || self.num_docs < 2 {
            return false;
        }
        let mut bounds = (0..self.num_blocks()).flat_map(|b| { let (lo, hi) = self.bound(b); [lo, hi] });
        let Some(first) = bounds.next() else { return false };
        !first.is_nan() && bounds.all(|v| v.to_le() == first.to_le())
    }

    #[inline(always)]
    fn sorted_doc(&self, i: usize) -> usize {
        if let Some(v) = &self.vlists {
            let t = v.value_of(i);
            let mut doc = 0usize;
            self.vl_docs(v, t, i - v.start(t), i - v.start(t) + 1, &mut |d| doc = d);
            return doc;
        }
        let w = self.pack.point_width;
        (u32::from_le_bytes(self.sorted[i * w..i * w + 4].try_into().unwrap()) & (u32::MAX >> (32 - 8 * w as u32))) as usize
    }

    /// Stored value `j` (doc `j`, or the `j`-th present doc of a compact column).
    #[inline(always)]
    fn value_at(&self, j: usize) -> T {
        if self.pack.blocked {
            return self.blocked_at(j);
        }
        let w = self.pack.value_width;
        if w == 0 {
            return T::from_le(self.pack.base.to_le_bytes());
        }
        let raw = u64::from_le_bytes(self.values[j * w..j * w + 8].try_into().unwrap()) & (u64::MAX >> (64 - 8 * w as u32));
        if self.pack.table {
            // An index past the table (corrupt bytes) reads as zero.
            let at = raw as usize * 8;
            return match self.table.get(at..at + 8) {
                Some(b) => T::from_le(b.try_into().unwrap()),
                None => T::zero(),
            };
        }
        T::from_le(self.pack.base.wrapping_add(raw).to_le_bytes())
    }

    /// Positions `[lo, hi)` of the points index whose values are in
    /// `[lower, upper]` (two binary searches over the sorted docs).
    fn points_range(&self, lower: T, upper: T) -> (usize, usize) {
        if let Some(v) = &self.vlists {
            // The distinct values are sorted (no NaN): a search over them.
            let value = |t: usize| T::from_le(v.value_bits(t));
            let part = |pred: &dyn Fn(T) -> bool| {
                let (mut lo, mut hi) = (0usize, v.k);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if pred(value(mid)) { lo = mid + 1 } else { hi = mid }
                }
                lo
            };
            let lo = part(&|x| x.sort_cmp(&lower) == std::cmp::Ordering::Less);
            let hi = part(&|x| x.sort_cmp(&upper) != std::cmp::Ordering::Greater).max(lo);
            return (v.start(lo), v.start(hi));
        }
        let n = self.points_len();
        let value = |i: usize| self.raw_value(self.sorted_doc(i) as DocId);
        let part = |pred: &dyn Fn(T) -> bool| {
            let (mut plo, mut phi) = (0usize, n);
            while plo < phi {
                let mid = (plo + phi) / 2;
                if pred(value(mid)) {
                    plo = mid + 1;
                } else {
                    phi = mid;
                }
            }
            plo
        };
        // Use the total order (`sort_cmp`), not raw `<`/`<=`: with a NaN
        // bound, raw comparisons are always false and the binary search
        // would misplace the range (NaNs sort last in the points index).
        let lo = part(&|v| v.sort_cmp(&lower) == std::cmp::Ordering::Less);
        let hi_raw = part(&|v| v.sort_cmp(&upper) != std::cmp::Ordering::Greater);
        let mut hi = hi_raw.max(lo);
        // NaNs sort last but match no range (`v >= lower && v <= upper` is
        // false for NaN), so clamp the range to the first NaN.
        let nan_start = if !T::TOTAL_ORDER {
            // `part` counts the leading values satisfying the predicate, so ask for the non-NaN prefix.
            part(&|v| !v.is_nan())
        } else { n };
        if hi > nan_start {
            hi = nan_start;
        }
        (lo.min(hi), hi)
    }

    /// Number of docs whose value is in `[lower, upper]`; `O(log n)` with
    /// the points index.
    pub fn count_range(&self, lower: T, upper: T) -> u64 {
        if !(lower <= upper) { return 0; }
        if self.is_sparse() && !self.compact {
            let mut w = vec![0u64; (self.num_docs as usize).div_ceil(64)];
            self.range_bits(lower, upper, &mut w);
            return w.iter().map(|x| x.count_ones() as u64).sum();
        }
        if self.has_points() {
            let (lo, hi) = self.points_range(lower, upper);
            return (hi - lo) as u64;
        }
        self.count_in_range(lower, upper) as u64
    }

    /// Whether [`Self::count_range`] answers from the points index alone
    /// (two binary searches, no per-doc read): the index exists and holds
    /// exactly the docs that have a value (dense, or a compact sparse
    /// column; a legacy sparse column's index also holds padding docs).
    #[inline]
    pub fn counts_from_points(&self) -> bool {
        self.has_points() && (self.compact || !self.is_sparse())
    }

    /// Smallest and largest stored value over the docs that have one, read
    /// from the ends of the points index; `None` without such an index
    /// ([`Self::counts_from_points`]), for an empty column, or for a type
    /// without a total order (a float column's index ends with its NaNs).
    /// [`Self::points_min_max`] of the values in `[lower, upper]`.
    pub fn points_min_max_in(&self, lower: T, upper: T) -> Option<(T, T)> {
        if !T::TOTAL_ORDER || !self.counts_from_points() || !(lower <= upper) {
            return None;
        }
        let (a, b) = self.points_range(lower, upper);
        (a < b).then(|| (self.point_value(a), self.point_value(b - 1)))
    }

    /// Visit the docs that have a value from the largest value down (equal
    /// values in points order) until `f` returns false. `false`, visiting
    /// nothing, without a points index holding exactly the docs with a value
    /// ([`Self::counts_from_points`]) or for a type without a total order.
    pub fn visit_desc(&self, mut f: impl FnMut(DocId, T) -> bool) -> bool {
        if !T::TOTAL_ORDER || !self.counts_from_points() {
            return false;
        }
        if let Some(v) = &self.vlists {
            // Values from the largest, each value's docs from the last: one
            // sample window (about 64 docs) decoded at a time.
            let mut window = Vec::new();
            for t in (0..v.k).rev() {
                let count = v.start(t + 1) - v.start(t);
                let stride = v.stride(t);
                let x = T::from_le(v.value_bits(t));
                let mut hi = count;
                while hi > 0 {
                    let lo = (hi - 1) / stride * stride;
                    window.clear();
                    self.vl_docs(v, t, lo, hi, &mut |d| window.push(d));
                    for &d in window.iter().rev() {
                        if !f(d as DocId, x) {
                            return true;
                        }
                    }
                    hi = lo;
                }
            }
            return true;
        }
        for i in (0..self.points_len()).rev() {
            let d = self.sorted_doc(i) as DocId;
            if !f(d, self.raw_value(d)) {
                break;
            }
        }
        true
    }

    pub fn points_min_max(&self) -> Option<(T, T)> {
        if !T::TOTAL_ORDER || !self.counts_from_points() {
            return None;
        }
        let n = self.points_len();
        if n == 0 {
            return None;
        }
        Some((self.point_value(0), self.point_value(n - 1)))
    }

    /// OR the docs whose value is in `[lower, upper]` into `words` (bit `d`
    /// = doc `d`; `words` must cover `num_docs`). With the points index this
    /// touches only `min(matches, num_docs - matches)` docs; otherwise
    /// 256-value blocks are skipped or fully accepted via their min/max.
    pub fn range_bits(&self, lower: T, upper: T, words: &mut [u64]) {
        if !(lower <= upper) { return; }
        if self.compact && self.has_points() {
            let (lo, hi) = self.points_range(lower, upper);
            for i in lo..hi {
                let d = self.sorted_doc(i);
                words[d / 64] |= 1u64 << (d % 64);
            }
            return;
        }
        if !self.is_sparse() {
            return self.range_bits_raw(lower, upper, words);
        }
        // Sparse column: missing docs hold the padding value, so keep only
        // the docs that have a value.
        let mut tmp = vec![0u64; words.len()];
        self.range_bits_raw(lower, upper, &mut tmp);
        for (i, (w, t)) in words.iter_mut().zip(tmp).enumerate() {
            *w |= t & self.present_word(i);
        }
    }

    /// OR the docs whose value is in `[lower, upper]` into `words`, keeping
    /// every other bit ([`Self::range_bits`] may write whole words: its wide
    /// ranges set everything, then clear the complement).
    pub fn or_range_bits(&self, lower: T, upper: T, words: &mut [u64]) {
        if !(lower <= upper) {
            return;
        }
        // A points index holding exactly the docs with a value.
        if self.has_points() && (self.compact || !self.is_sparse()) {
            let (lo, hi) = self.points_range(lower, upper);
            self.for_points(lo, hi, &mut |d| words[d / 64] |= 1u64 << (d % 64));
            return;
        }
        let mut tmp = vec![0u64; words.len()];
        self.range_bits(lower, upper, &mut tmp);
        for (w, t) in words.iter_mut().zip(tmp) {
            *w |= t;
        }
    }

    /// Whether some doc has no value (the column carries a presence bitmap
    /// or runs).
    #[inline]
    pub fn is_sparse(&self) -> bool {
        !self.no_presence()
    }

    /// No presence index: every doc has a value.
    #[inline(always)]
    fn no_presence(&self) -> bool {
        self.present.is_empty() && self.run_dir.is_empty()
    }

    #[inline]
    #[inline(always)]
    fn present_word(&self, w: usize) -> u64 {
        if !self.run_dir.is_empty() {
            return self.run_word(w);
        }
        match self.present.get(w * 8..w * 8 + 8) {
            Some(b) => u64::from_le_bytes(b.try_into().unwrap()),
            None => 0,
        }
    }

    /// Number of presence runs ([`COMPACT_RUNS`]).
    #[inline(always)]
    fn run_count(&self) -> usize {
        self.runs.len() / 16
    }

    /// Block `b`'s (min, max) as stored (zeros for a block without a
    /// value): a [`COMPACT_RUNS`] column keeps only its filled blocks'.
    #[inline(always)]
    fn bound(&self, b: usize) -> (T, T) {
        if self.run_dir.is_empty() {
            return (read_at::<T>(self.mins, b), read_at::<T>(self.maxs, b));
        }
        let k = self.run_find(b * BLOCK_SIZE);
        if k >= self.run_count() || u32_at(self.runs, 4 * k) >= (b + 1) * BLOCK_SIZE {
            return (T::zero(), T::zero());
        }
        let at = u32_at(self.runs, 4 * k + 3) + b.saturating_sub(u32_at(self.runs, 4 * k) / BLOCK_SIZE);
        match (self.mins.get(at * 8..at * 8 + 8), self.maxs.get(at * 8..at * 8 + 8)) {
            (Some(a), Some(z)) => (T::from_le(a.try_into().unwrap()), T::from_le(z.try_into().unwrap())),
            _ => (T::zero(), T::zero()),
        }
    }

    /// The first run ending past doc `i` (the run count if none).
    #[inline(always)]
    fn run_find(&self, i: usize) -> usize {
        let r = self.run_count();
        let Some(e) = self.run_dir.get((i / RUN_DIR_DOCS) * 4..(i / RUN_DIR_DOCS) * 4 + 4) else { return r };
        let mut k = (u32::from_le_bytes(e.try_into().unwrap()) as usize).min(r);
        while k < r && u32_at(self.runs, 4 * k + 1) <= i {
            k += 1;
        }
        k
    }

    /// Presence word `w` from the runs.
    fn run_word(&self, w: usize) -> u64 {
        let (lo, hi) = (w * 64, w * 64 + 64);
        let r = self.run_count();
        let mut k = self.run_find(lo);
        let mut word = 0u64;
        while k < r {
            let a = u32_at(self.runs, 4 * k);
            if a >= hi {
                break;
            }
            let (from, to) = (a.max(lo) - lo, u32_at(self.runs, 4 * k + 1).min(hi) - lo);
            word |= (u64::MAX >> (64 - (to - from))) << from;
            k += 1;
        }
        word
    }

    /// Rank of present doc `i` among the present docs (its value's index).
    #[inline(always)]
    fn present_rank(&self, i: usize) -> usize {
        if !self.run_dir.is_empty() {
            let k = self.run_find(i);
            return u32_at(self.runs, 4 * k + 2) + (i - u32_at(self.runs, 4 * k));
        }
        let w = i / 64;
        let rank = u32::from_le_bytes(self.ranks[w * 4..w * 4 + 4].try_into().unwrap()) as usize;
        rank + (self.present_word(w) & ((1u64 << (i % 64)) - 1)).count_ones() as usize
    }

    /// The docs with a value as bitset words over the column's docs (bit
    /// `d` = doc `d`, the same answers as [`Self::has_value`]); `None` when
    /// every doc has one (no presence bitmap).
    pub fn presence_words(&self) -> Option<Vec<u64>> {
        if self.no_presence() {
            return None;
        }
        let n = self.num_docs as usize;
        let mut words: Vec<u64> = (0..n.div_ceil(64)).map(|w| self.present_word(w)).collect();
        if n % 64 != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << (n % 64)) - 1;
            }
        }
        Some(words)
    }

    /// Whether `doc` has a value (false past the last doc).
    /// [`RangeDocs`] over this column for `[lower, upper]` (stored values,
    /// as [`Self::range_bits`] compares them); `None` for a compact sparse
    /// column (its block bounds are over its values, not its docs) or an
    /// empty range.
    pub fn lazy_range(self, lower: T, upper: T) -> Option<RangeDocs<'a, T>> {
        if self.compact || !(lower <= upper) {
            return None;
        }
        // Its size, so a conjunction leads with its cheapest clause (a rare
        // term in a day of logs, not the day): two binary searches.
        let hint = if self.counts_from_points() {
            let (a, b) = self.points_range(lower, upper);
            u32::try_from(b - a).unwrap_or(u32::MAX)
        } else {
            self.num_docs
        };
        Some(RangeDocs { col: self, lower, upper, cur: crate::query::docset::TERMINATED, next: 0, block: 0, state: 0, hint })
    }

    #[inline]
    #[inline(always)]
    pub fn has_value(&self, doc: DocId) -> bool {
        let i = doc as usize;
        if i >= self.num_docs as usize {
            return false;
        }
        if !self.run_dir.is_empty() {
            let k = self.run_find(i);
            return k < self.run_count() && u32_at(self.runs, 4 * k) <= i;
        }
        self.present.is_empty() || (self.present_word(i / 64) >> (i % 64)) & 1 == 1
    }

    /// Document IDs with a value, in ascending order. Sparse columns walk
    /// set presence bits, including legacy dense-value columns with a bitmap.
    pub fn present_docs(&self) -> impl Iterator<Item = DocId> + '_ {
        let mut dense_doc = 0;
        let mut word = 0usize;
        let mut bits = 0u64;
        std::iter::from_fn(move || {
            if self.no_presence() {
                if dense_doc >= self.num_docs { return None; }
                let doc = dense_doc;
                dense_doc += 1;
                return Some(doc);
            }
            loop {
                if bits != 0 {
                    let doc = (word - 1) * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    return (doc < self.num_docs as usize).then_some(doc as DocId);
                }
                if word >= (self.num_docs as usize).div_ceil(64) { return None; }
                bits = self.present_word(word);
                word += 1;
            }
        })
    }

    fn range_bits_raw(&self, lower: T, upper: T, words: &mut [u64]) {
        let n = self.num_docs as usize;
        if n == 0 {
            return;
        }
        if !self.sorted.is_empty() || self.vlists.is_some() {
            let (lo, hi) = self.points_range(lower, upper);
            if hi - lo <= n / 2 {
                self.for_points(lo, hi, &mut |d| words[d >> 6] |= 1u64 << (d & 63));
            } else {
                // Wide range: set everything, clear the complement.
                set_bit_range(words, 0, n);
                let mut clear = |d: usize| words[d >> 6] &= !(1u64 << (d & 63));
                self.for_points(0, lo, &mut clear);
                self.for_points(hi, n, &mut clear);
            }
            return;
        }
        for b in 0..self.num_blocks() {
            let (bmin, bmax) = self.bound(b);
            if bmax < lower || bmin > upper {
                continue;
            }
            let start = b * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(n);
            if T::TOTAL_ORDER && bmin >= lower && bmax <= upper {
                set_bit_range(words, start, end);
                continue;
            }
            for i in start..end {
                let v = self.raw_value(i as DocId);
                if v >= lower && v <= upper {
                    words[i >> 6] |= 1u64 << (i & 63);
                }
            }
        }
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Docs with a value (deleted docs included): every doc of a dense
    /// column, the presence bitmap's popcount over the column's docs for a
    /// sparse one (compact or legacy). Read-only.
    pub fn value_doc_count(&self) -> u64 {
        if self.no_presence() {
            return self.num_docs as u64;
        }
        if !self.run_dir.is_empty() {
            return (0..self.run_count()).map(|k| (u32_at(self.runs, 4 * k + 1) - u32_at(self.runs, 4 * k)) as u64).sum();
        }
        let n = self.num_docs as usize;
        let full = n / 64;
        let mut c: u64 = (0..full).map(|w| self.present_word(w).count_ones() as u64).sum();
        if n % 64 != 0 {
            c += (self.present_word(full) & ((1u64 << (n % 64)) - 1)).count_ones() as u64;
        }
        c
    }

    #[inline(always)]
    pub fn get(&self, doc: DocId) -> Option<T> {
        self.has_value(doc).then(|| self.es(self.raw_value(doc)))
    }

    /// [`Self::get`] as stored (no `scaled_float` conversion).
    #[inline(always)]
    pub fn get_stored(&self, doc: DocId) -> Option<T> {
        self.has_value(doc).then(|| self.raw_value(doc))
    }

    /// This reader with its values read as ES reads a `scaled_float`'s doc
    /// values with `factor` ([`FastValue::es_scaled`]).
    pub fn with_es_scaled(mut self, factor: f64) -> Self {
        if factor.is_finite() && factor > 0.0 {
            self.es_scale = factor;
            self.es_inv = 1.0 / factor;
        }
        self
    }

    /// Whether the value accessors convert stored values ([`Self::with_es_scaled`]).
    #[inline]
    pub fn is_es_scaled(&self) -> bool {
        self.es_scale != 0.0
    }

    /// A stored value as the accessors return it.
    #[inline(always)]
    pub fn es(&self, v: T) -> T {
        if self.es_scale != 0.0 { v.es_scaled(self.es_scale, self.es_inv) } else { v }
    }

    fn num_blocks(&self) -> usize {
        if !self.run_dir.is_empty() {
            return (self.num_docs as usize).div_ceil(BLOCK_SIZE);
        }
        self.mins.len() / 8
    }

    /// [`Self::value_at`] of a block-packed column.
    #[inline(always)]
    fn blocked_at(&self, j: usize) -> T {
        if !self.deltas.is_empty() {
            return self.delta_column_at(j);
        }
        let b = j / BLOCK_SIZE;
        // A decimal column's blocks are based on their integer minimum, a
        // compact column's on its blocks of present values' minimum.
        let bases = if self.decimal != 0.0 { self.int_mins } else if self.compact { self.vmins } else { self.mins };
        let Some(min) = bases.get(b * 8..b * 8 + 8) else { return T::zero() };
        let min = u64::from_le_bytes(min.try_into().unwrap());
        let c0 = u32::from_le_bytes(self.values[b * 4..b * 4 + 4].try_into().unwrap()) as usize;
        let c1 = u32::from_le_bytes(self.values[b * 4 + 4..b * 4 + 8].try_into().unwrap()) as usize;
        let w = c1.saturating_sub(c0).min(BLOCKED_MAX_BITS as usize);
        if w == 0 {
            return self.from_block_bits(min);
        }
        let blocks = bases.len() / 8;
        let bit = (j % BLOCK_SIZE) * w;
        let at = (blocks + 1) * 4 + c0 * 32 + bit / 8;
        let Some(bytes) = self.values.get(at..at + 8) else { return self.from_block_bits(min) };
        let raw = u64::from_le_bytes(bytes.try_into().unwrap()) >> (bit % 8);
        let d = raw & (u64::MAX >> (64 - w));
        if w == 1 && self.two_value_blocks() {
            return if d == 0 { self.from_block_bits(min) } else { self.block_max_value(b) };
        }
        self.from_block_bits(min.wrapping_add(d.wrapping_mul(self.pack.base)))
    }

    /// Whether this column's 1-bit blocks read as their minimum or maximum
    /// (any block-packed doc-block integer column: a v37 two-value block,
    /// or an older one whose maximum is its minimum plus the GCD; not a
    /// compact or decimal column, whose blocks have no such maximum).
    #[inline(always)]
    fn two_value_blocks(&self) -> bool {
        !self.compact && self.decimal == 0.0
    }

    /// Block `b`'s maximum as stored.
    #[inline(always)]
    fn block_max_value(&self, b: usize) -> T {
        self.bound(b).1
    }

    /// [`Self::blocked_at`] of a [`DELTA`] column (out of line: the other
    /// columns' reads keep their code). A block read twice in a row (a
    /// scan) is decoded once into the cache; a lone read takes a checkpoint
    /// and a popcount.
    #[inline(never)]
    fn delta_column_at(&self, j: usize) -> T {
        let b = j / BLOCK_SIZE;
        let i = j % BLOCK_SIZE;
        DELTA_SLOTS.with(|slots| {
            let mut guard = slots.borrow_mut();
            let (slots, next) = &mut *guard;
            let at = match slots.iter().position(|s| s.reader == self.delta_id) {
                Some(at) => at,
                None => {
                    let at = *next % slots.len();
                    *next += 1;
                    slots[at] = DeltaSlot { reader: self.delta_id, last: 0, decoded: 0, vals: std::mem::take(&mut slots[at].vals) };
                    at
                }
            };
            let slot = &mut slots[at];
            if slot.decoded == b + 1 {
                if let Some(&v) = slot.vals.get(i) {
                    return T::from_le(v.to_le_bytes());
                }
            }
            self.delta_block_at(b, i, slot)
        })
    }

    /// [`Self::delta_column_at`] past the decoded block: decode block `b`
    /// into `slot` on a second read in a row, else a checkpoint and a
    /// popcount.
    fn delta_block_at(&self, b: usize, i: usize, slot: &mut DeltaSlot) -> T {
        let min = u64::from_le_bytes(self.mins[b * 8..b * 8 + 8].try_into().unwrap());
        let c0 = u32::from_le_bytes(self.values[b * 4..b * 4 + 4].try_into().unwrap()) as usize;
        let c1 = u32::from_le_bytes(self.values[b * 4 + 4..b * 4 + 8].try_into().unwrap()) as usize;
        let w = c1.saturating_sub(c0).min(BLOCKED_MAX_BITS as usize);
        let blocks = self.mins.len() / 8;
        let start = (blocks + 1) * 4 + c0 * 32;
        let Some((first, dmin)) = self.delta_head(b) else {
            // A block of offsets.
            if w == 0 {
                return self.from_block_bits(min);
            }
            let bit = i * w;
            let at = start + bit / 8;
            let Some(bytes) = self.values.get(at..at + 8) else { return self.from_block_bits(min) };
            let d = (u64::from_le_bytes(bytes.try_into().unwrap()) >> (bit % 8)) & (u64::MAX >> (64 - w));
            if w == 1 && self.two_value_blocks() {
                return if d == 0 { self.from_block_bits(min) } else { self.block_max_value(b) };
            }
            return self.from_block_bits(min.wrapping_add(d.wrapping_mul(self.pack.base)));
        };
        // (Width 0: every delta is the smallest.)
        if w > 0 {
            if slot.last == b + 1 {
                // A second read in this block: decode it.
                let mask = u64::MAX >> (64 - w);
                let n = (self.num_docs as usize).saturating_sub(b * BLOCK_SIZE).min(BLOCK_SIZE);
                let vals = &mut slot.vals;
                vals.clear();
                let mut off = first as i64;
                for k in 0..n {
                    if k > 0 {
                        let bit = k * w;
                        let at = start + bit / 8;
                        let e = self.values.get(at..at + 8).map_or(0, |x| (u64::from_le_bytes(x.try_into().unwrap()) >> (bit % 8)) & mask);
                        off = off.wrapping_add(dmin as i64).wrapping_add(e as i64);
                    }
                    vals.push(u64::from_le_bytes(self.from_block_bits(min.wrapping_add((off as u64).wrapping_mul(self.pack.base))).to_le()));
                }
                slot.decoded = b + 1;
                if let Some(&v) = vals.get(i) {
                    return T::from_le(v.to_le_bytes());
                }
            }
            slot.last = b + 1;
        }
        let sum = if w == 0 {
            0
        } else {
            // The checkpoint before entry `i`'s 64, then the rest.
            let q = i / 64;
            let cp = if q == 0 { 0 } else {
                let at = b * DELTA_HEAD + 8 + (q - 1) * 2;
                self.deltas.get(at..at + 2).map_or(0, |c| u16::from_le_bytes([c[0], c[1]]) as u64)
            };
            cp + packed_prefix_sum(self.values.get(start + q * w * 8..).unwrap_or(&[]), w, i % 64 + 1)
        };
        let off = (first as i64).wrapping_add(i as i64 * dmin as i64).wrapping_add(sum as i64) as u64;
        self.from_block_bits(min.wrapping_add(off.wrapping_mul(self.pack.base)))
    }

    /// Block `b`'s delta head ([`DELTA`]): first offset and smallest delta,
    /// or `None` for a block of offsets.
    #[inline(always)]
    fn delta_head(&self, b: usize) -> Option<(u32, i32)> {
        let h = self.deltas.get(b * DELTA_HEAD..b * DELTA_HEAD + 8)?;
        let dmin = i32::from_le_bytes(h[4..8].try_into().unwrap());
        (dmin != i32::MIN).then(|| (u32::from_le_bytes(h[..4].try_into().unwrap()), dmin))
    }

    /// A block-packed value from its bits: the value's own bits, or a
    /// decimal column's integer over `10^e`.
    #[inline(always)]
    fn from_block_bits(&self, bits: u64) -> T {
        if self.decimal != 0.0 {
            return T::from_le(((bits as i64) as f64 / self.decimal).to_le_bytes());
        }
        T::from_le(bits.to_le_bytes())
    }

    /// Entries sorted by (value, doc). Compact columns include present docs
    /// only; legacy columns also include padding docs. Zero without points.
    #[inline]
    pub fn points_len(&self) -> usize {
        if let Some(v) = &self.vlists {
            return v.start(v.k);
        }
        // No points index: `sorted` is empty, without its padding byte.
        self.sorted.len().saturating_sub(self.pack.point_pad()) / self.pack.point_width
    }

    /// Doc at position `i` of the points index.
    #[inline]
    pub fn point_doc(&self, i: usize) -> DocId {
        self.sorted_doc(i) as DocId
    }

    /// The stored value of `doc` whether or not it has one (padding for a
    /// doc without a value).
    #[inline(always)]
    pub fn raw_value(&self, doc: DocId) -> T {
        let i = doc as usize;
        if !self.compact { return self.value_at(i); }
        if !self.has_value(doc) { return T::zero(); }
        self.value_at(self.present_rank(i))
    }

    /// Number of [`BLOCK_SIZE`]-doc blocks in the min/max table.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.num_blocks()
    }

    /// Whether every doc has a value (no presence words).
    #[inline]
    pub fn all_present(&self) -> bool {
        self.no_presence()
    }

    /// The stored values of docs `[b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)`
    /// (fewer for the last block) into `out`, in doc order: what
    /// [`Self::raw_value`] returns for each (padding for a doc without a
    /// value: check [`Self::has_value`] unless [`Self::all_present`]), with
    /// the block's header read once instead of per doc. Returns how many
    /// were written; `None` for a compact column (values by rank), whose
    /// callers read doc by doc.
    pub fn block_values(&self, b: usize, out: &mut [T]) -> Option<usize> {
        let n = self.block_values_stored(b, out)?;
        if self.es_scale != 0.0 {
            for v in &mut out[..n] {
                *v = v.es_scaled(self.es_scale, self.es_inv);
            }
        }
        Some(n)
    }

    /// [`Self::block_values`] as stored (no `scaled_float` conversion).
    pub fn block_values_stored(&self, b: usize, out: &mut [T]) -> Option<usize> {
        if self.compact {
            return None;
        }
        let start = b.checked_mul(BLOCK_SIZE)?;
        let n = (self.num_docs as usize).saturating_sub(start).min(BLOCK_SIZE).min(out.len());
        if !self.pack.blocked {
            for (i, o) in out[..n].iter_mut().enumerate() {
                *o = self.value_at(start + i);
            }
            return Some(n);
        }
        let bases = if self.decimal != 0.0 { self.int_mins } else { self.mins };
        let min = u64::from_le_bytes(bases.get(b * 8..b * 8 + 8)?.try_into().ok()?);
        let c0 = u32::from_le_bytes(self.values.get(b * 4..b * 4 + 4)?.try_into().ok()?) as usize;
        let c1 = u32::from_le_bytes(self.values.get(b * 4 + 4..b * 4 + 8)?.try_into().ok()?) as usize;
        let w = c1.saturating_sub(c0).min(BLOCKED_MAX_BITS as usize);
        let blocks = self.mins.len() / 8;
        let first = (blocks + 1) * 4 + c0 * 32;
        let base = self.pack.base;
        let delta = self.delta_head(b);
        if w == 0 && delta.is_none() {
            out[..n].fill(self.from_block_bits(min));
            return Some(n);
        }
        let mask = if w == 0 { 0 } else { u64::MAX >> (64 - w) };
        if let Some((head, dmin)) = delta {
            // Running sum of the entries.
            let mut off = head as i64;
            for (i, o) in out[..n].iter_mut().enumerate() {
                let bit = i * w;
                let at = first + bit / 8;
                let e = self.values.get(at..at + 8).map_or(0, |bytes| (u64::from_le_bytes(bytes.try_into().unwrap()) >> (bit % 8)) & mask);
                if i > 0 {
                    off = off.wrapping_add(dmin as i64).wrapping_add(e as i64);
                }
                *o = self.from_block_bits(min.wrapping_add((off as u64).wrapping_mul(base)));
            }
            return Some(n);
        }
        let top = (w == 1 && self.two_value_blocks()).then(|| self.block_max_value(b));
        for (i, o) in out[..n].iter_mut().enumerate() {
            let bit = i * w;
            let at = first + bit / 8;
            *o = match self.values.get(at..at + 8) {
                Some(bytes) => {
                    let d = (u64::from_le_bytes(bytes.try_into().unwrap()) >> (bit % 8)) & mask;
                    match top {
                        Some(top) if d != 0 => top,
                        _ => self.from_block_bits(min.wrapping_add(d.wrapping_mul(base))),
                    }
                }
                None => self.from_block_bits(min),
            };
        }
        Some(n)
    }

    /// Min and max of the present values of docs `[b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)`
    /// (meaningless when the block has no present value, see
    /// [`ColumnReader::block_presence`]).
    #[inline]
    pub fn block_min_max(&self, b: usize) -> (T, T) {
        let (lo, hi) = self.bound(b);
        (self.es(lo), self.es(hi))
    }

    /// Whether block `b` has (some present value, some doc without a value).
    #[inline]
    pub fn block_presence(&self, b: usize) -> (bool, bool) {
        if self.no_presence() {
            return (true, false);
        }
        let n = self.num_docs as usize;
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n);
        let (mut any, mut all) = (false, true);
        let mut w = start / 64;
        while w * 64 < end {
            let lo = w * 64;
            let bits = (end - lo).min(64);
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            let x = self.present_word(w) & mask;
            any |= x != 0;
            all &= x == mask;
            w += 1;
        }
        (any, !all)
    }

    /// Visit every doc whose value is in `[lower, upper]`.
    pub fn for_each_in_range(&self, lower: T, upper: T, mut f: impl FnMut(DocId)) {
        let n = self.num_docs as usize;
        for b in 0..self.num_blocks() {
            let (bmin, bmax) = self.bound(b);
            if bmax < lower || bmin > upper {
                continue;
            }
            let start = b * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(n);
            for i in start..end {
                let v = self.raw_value(i as DocId);
                if v >= lower && v <= upper && self.has_value(i as DocId) {
                    f(i as DocId);
                }
            }
        }
    }

    pub fn range_query(&self, lower: T, upper: T) -> Vec<DocId> {
        let mut out = Vec::new();
        self.for_each_in_range(lower, upper, |d| out.push(d));
        out
    }

    pub fn count_in_range(&self, lower: T, upper: T) -> u32 {
        let mut c = 0;
        self.for_each_in_range(lower, upper, |_| c += 1);
        c
    }

    pub fn min(&self) -> Option<T> {
        if self.is_sparse() {
            return (0..self.num_docs).filter_map(|d| self.get(d)).reduce(|a, b| if b < a { b } else { a });
        }
        (0..self.num_blocks())
            .map(|b| self.bound(b).0)
            .reduce(|a, b| if b < a { b } else { a })
            .map(|v| self.es(v))
    }

    pub fn max(&self) -> Option<T> {
        if self.is_sparse() {
            return (0..self.num_docs).filter_map(|d| self.get(d)).reduce(|a, b| if b > a { b } else { a });
        }
        (0..self.num_blocks())
            .map(|b| self.bound(b).1)
            .reduce(|a, b| if b > a { b } else { a })
            .map(|v| self.es(v))
    }
}

fn encode_multi_numeric<T: FastValue>(values: &[T], offsets: &[u32], num_docs: u32, out: &mut Vec<u8>) {
    let n = num_docs as usize;
    let total = values.len();
    let num_blocks = total.div_ceil(BLOCK_SIZE);
    out.extend_from_slice(&num_docs.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    // The builder only tracks offsets up to the last doc that received a
    // value; trailing docs without values get `total`.
    for i in 0..=n {
        let o = offsets.get(i).copied().unwrap_or(total as u32);
        out.extend_from_slice(&o.to_le_bytes());
    }
    let mut mins = Vec::with_capacity(num_blocks);
    let mut maxs = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(total);
        let mut min = values[start];
        let mut max = min;
        for i in start + 1..end {
            let v = values[i];
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        mins.push(min);
        maxs.push(max);
    }
    for m in &mins {
        out.extend_from_slice(&m.to_le());
    }
    for m in &maxs {
        out.extend_from_slice(&m.to_le());
    }
    for v in values {
        out.extend_from_slice(&v.to_le());
    }
}

/// Zero-copy multi-valued numeric column reader (Lucene
/// `SortedNumericDocValues`).
pub struct MultiColumnReader<'a, T: FastValue> {
    num_docs: u32,
    total: usize,
    offsets: &'a [u8],
    mins: &'a [u8],
    maxs: &'a [u8],
    values: &'a [u8],
    _t: std::marker::PhantomData<T>,
}

impl<'a, T: FastValue> MultiColumnReader<'a, T> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let mut pos = 0;
        let num_docs = read_u32_le(data, &mut pos)? as usize;
        let total = read_u32_le(data, &mut pos)? as usize;
        let offsets = read_bytes(data, &mut pos, (num_docs + 1) * 4)?;
        let num_blocks = total.div_ceil(BLOCK_SIZE);
        let mins = read_bytes(data, &mut pos, num_blocks * 8)?;
        let maxs = read_bytes(data, &mut pos, num_blocks * 8)?;
        let values = read_bytes(data, &mut pos, total * 8)?;
        Ok(Self {
            num_docs: num_docs as u32,
            total,
            offsets,
            mins,
            maxs,
            values,
            _t: std::marker::PhantomData,
        })
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Total number of values in the column.
    pub fn total_values(&self) -> usize {
        self.total
    }

    fn off(&self, i: usize) -> usize {
        u32::from_le_bytes(self.offsets[i * 4..i * 4 + 4].try_into().unwrap()) as usize
    }

    /// Number of values of `doc`.
    #[inline]
    pub fn count(&self, doc: DocId) -> u32 {
        let i = doc as usize;
        if i >= self.num_docs as usize {
            return 0;
        }
        (self.off(i + 1) - self.off(i)) as u32
    }

    /// The values of `doc` (empty if the doc has none).
    pub fn values(&self, doc: DocId) -> impl Iterator<Item = T> + '_ {
        let i = doc as usize;
        let (a, b) = if i < self.num_docs as usize {
            (self.off(i), self.off(i + 1))
        } else {
            (0, 0)
        };
        (a..b).map(move |j| read_at::<T>(self.values, j))
    }

    /// The values of `doc` as a `Vec`.
    pub fn values_vec(&self, doc: DocId) -> Vec<T> {
        self.values(doc).collect()
    }

    /// Minimum value of `doc` (None if the doc has no values).
    pub fn min_of(&self, doc: DocId) -> Option<T> {
        self.values(doc).reduce(|a, b| if b < a { b } else { a })
    }

    /// Maximum value of `doc` (None if the doc has no values).
    pub fn max_of(&self, doc: DocId) -> Option<T> {
        self.values(doc).reduce(|a, b| if b > a { b } else { a })
    }

    /// Sum of the values of `doc` (0 if the doc has no values).
    pub fn sum_of(&self, doc: DocId) -> T {
        self.values(doc).fold(T::zero(), |s, v| s.add(v))
    }

    /// Average of the values of `doc` (None if the doc has no values).
    pub fn avg_of(&self, doc: DocId) -> Option<f64> {
        let n = self.count(doc);
        if n == 0 {
            None
        } else {
            Some(self.values(doc).fold(0.0, |s, v| s + v.to_f64()) / n as f64)
        }
    }

    fn num_blocks(&self) -> usize {
        self.mins.len() / 8
    }

    /// Visit every doc that has at least one value in `[lower, upper]`,
    /// each doc at most once, in ascending doc order.
    pub fn for_each_doc_in_range(&self, lower: T, upper: T, mut f: impl FnMut(DocId)) {
        let n = self.num_docs as usize;
        if n == 0 {
            return;
        }
        // Doc cursor: a value at flat index j belongs to the doc d with
        // offsets[d] <= j < offsets[d+1]. Blocks are visited in order so the
        // cursor only ever moves forward.
        let mut d = 0usize;
        // A doc whose values straddle a block boundary must not be emitted
        // twice.
        let mut last: Option<usize> = None;
        for b in 0..self.num_blocks() {
            if read_at::<T>(self.maxs, b) < lower || read_at::<T>(self.mins, b) > upper {
                continue;
            }
            let start = b * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(self.total);
            while d + 1 < n && self.off(d + 1) <= start {
                d += 1;
            }
            let mut j = start;
            while j < end {
                let v = read_at::<T>(self.values, j);
                if v >= lower && v <= upper {
                    while d + 1 < n && self.off(d + 1) <= j {
                        d += 1;
                    }
                    if last != Some(d) {
                        f(d as DocId);
                        last = Some(d);
                    }
                    // Skip the rest of this doc's values in this block.
                    j = self.off(d + 1).min(end);
                } else {
                    j += 1;
                }
            }
        }
    }

    /// Docs with at least one value in `[lower, upper]`, ascending, deduped.
    pub fn range_query(&self, lower: T, upper: T) -> Vec<DocId> {
        let mut out = Vec::new();
        self.for_each_doc_in_range(lower, upper, |d| out.push(d));
        out
    }

    /// OR the docs with at least one value in `[lower, upper]` into `words`.
    pub fn range_bits(&self, lower: T, upper: T, words: &mut [u64]) {
        self.for_each_doc_in_range(lower, upper, |d| words[d as usize >> 6] |= 1u64 << (d & 63));
    }

    /// Number of docs with at least one value in `[lower, upper]`.
    pub fn count_in_range(&self, lower: T, upper: T) -> u32 {
        let mut c = 0;
        self.for_each_doc_in_range(lower, upper, |_| c += 1);
        c
    }
}

/// Bit-per-doc boolean column.
pub struct BoolReader<'a> {
    num_docs: u32,
    words: &'a [u8],
    /// A [`TAG_BOOL_NUMERIC`] column (`words` empty): values and presence.
    numeric: Option<ColumnReader<'a, i64>>,
}

impl<'a> BoolReader<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let mut pos = 0;
        let num_docs = read_u32_le(data, &mut pos)?;
        let nwords = (num_docs as usize).div_ceil(64);
        let words = read_bytes(data, &mut pos, nwords * 8)?;
        Ok(Self { num_docs, words, numeric: None })
    }

    /// A [`TAG_BOOL_NUMERIC`] column.
    pub fn open_numeric(data: &'a [u8]) -> Result<Self> {
        Ok(Self::from_numeric(ColumnReader::<i64>::open(data)?))
    }

    /// A [`TAG_BOOL_NUMERIC`] column already opened.
    pub fn from_numeric(c: ColumnReader<'a, i64>) -> Self {
        Self { num_docs: c.num_docs(), words: &[], numeric: Some(c) }
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Whether the column knows which docs have a value (a doc without
    /// one otherwise reads `false`, as present).
    pub fn presence_known(&self) -> bool {
        self.numeric.is_some()
    }

    /// Whether `doc` has a value (every doc of a column that does not know).
    pub fn has_value(&self, doc: DocId) -> bool {
        match &self.numeric {
            Some(c) => c.has_value(doc),
            None => doc < self.num_docs,
        }
    }

    /// The docs with a value as words, when the column knows them.
    pub fn presence_words(&self) -> Option<Vec<u64>> {
        let c = self.numeric.as_ref()?;
        Some(c.presence_words().unwrap_or_else(|| {
            let n = self.num_docs as usize;
            let mut words = vec![u64::MAX; n.div_ceil(64)];
            if n % 64 != 0 {
                if let Some(last) = words.last_mut() {
                    *last = (1u64 << (n % 64)) - 1;
                }
            }
            words
        }))
    }

    #[inline]
    fn word(&self, i: usize) -> u64 {
        if let Some(c) = &self.numeric {
            let mut w = 0u64;
            let start = i * 64;
            for b in 0..64usize.min((self.num_docs as usize).saturating_sub(start)) {
                if c.get((start + b) as DocId).is_some_and(|v| v != 0) {
                    w |= 1 << b;
                }
            }
            return w;
        }
        u64::from_le_bytes(self.words[i * 8..i * 8 + 8].try_into().unwrap())
    }

    fn num_words(&self) -> usize {
        (self.num_docs as usize).div_ceil(64)
    }

    pub fn get(&self, doc: DocId) -> bool {
        let i = doc as usize;
        if i >= self.num_docs as usize {
            return false;
        }
        if let Some(c) = &self.numeric {
            return c.get(doc).is_some_and(|v| v != 0);
        }
        (self.word(i / 64) >> (i % 64)) & 1 == 1
    }

    pub fn count_true(&self) -> u32 {
        (0..self.num_words()).map(|i| self.word(i).count_ones()).sum()
    }

    pub fn true_docs(&self) -> Vec<DocId> {
        let mut out = Vec::new();
        for w in 0..self.num_words() {
            let mut word = self.word(w);
            while word != 0 {
                let bit = word.trailing_zeros();
                out.push((w * 64) as DocId + bit);
                word &= word - 1;
            }
        }
        out
    }

    /// Raw words (for building a BitSet).
    pub fn words(&self) -> Vec<u64> {
        (0..self.num_words()).map(|i| self.word(i)).collect()
    }
}

/// Per-doc field lengths for BM25 (Lucene norms: one `SmallFloat` byte per
/// doc), plus the exact field total so the average length is O(1).
pub struct NormsReader<'a> {
    num_docs: u32,
    total: u64,
    bytes: &'a [u8],
    /// [`TAG_CONST_NORMS`]: `bytes`, a run of [`constant_bytes`].
    run: Option<&'static [u8]>,
}

impl<'a> NormsReader<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let mut pos = 0;
        let num_docs = read_u32_le(data, &mut pos)?;
        let total = super::varint::read_u64_le(data, &mut pos)?;
        let bytes = read_bytes(data, &mut pos, num_docs as usize)?;
        Ok(Self { num_docs, total, bytes, run: None })
    }

    /// A constant norms column ([`TAG_CONST_NORMS`]): its bytes are a view
    /// of a process-wide run of that byte.
    pub fn open_constant(data: &[u8]) -> Result<NormsReader<'static>> {
        let mut pos = 0;
        let num_docs = read_u32_le(data, &mut pos)?;
        let total = super::varint::read_u64_le(data, &mut pos)?;
        let value = read_bytes(data, &mut pos, 1)?[0];
        let run = constant_bytes(value, num_docs as usize);
        Ok(NormsReader { num_docs, total, bytes: run, run: Some(run) })
    }

    /// A constant column's bytes: a process-wide run, not the segment's
    /// mapping.
    pub fn constant_run(&self) -> Option<&'static [u8]> {
        self.run
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Norm byte of a doc; 0 if missing.
    #[inline]
    pub fn get_byte(&self, doc: DocId) -> u8 {
        self.bytes.get(doc as usize).copied().unwrap_or(0)
    }

    /// Field length of a doc as scoring sees it (the decoded norm byte,
    /// exact up to 39 tokens); 0 if missing.
    #[inline]
    pub fn get(&self, doc: DocId) -> u32 {
        crate::query::bm25::byte4_to_int(self.get_byte(doc))
    }

    /// Exact sum of the field lengths.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Average field length over the segment (>= 1.0).
    pub fn avg(&self) -> f32 {
        if self.num_docs == 0 {
            return 1.0;
        }
        (self.total as f64 / self.num_docs as f64).max(1.0) as f32
    }

    /// Decoded lengths of all docs.
    pub fn lengths(&self) -> Vec<u32> {
        (0..self.num_docs).map(|d| self.get(d)).collect()
    }

    /// The raw norm bytes, one per doc (zero-copy norms).
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Alias of [`bytes`](Self::bytes).
    pub fn lengths_bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// The process-wide runs of one byte that constant norms read through
/// (`len` bytes of `value`). A run grows to the next power of two of the
/// largest segment asking for it; the smaller one it replaces stays alive
/// (readers may still hold it), so at most twice the largest run is kept per
/// byte value.
fn constant_bytes(value: u8, len: usize) -> &'static [u8] {
    static RUNS: std::sync::Mutex<Vec<&'static [u8]>> = std::sync::Mutex::new(Vec::new());
    let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
    if runs.is_empty() {
        runs.resize(256, &[]);
    }
    let slot = &mut runs[value as usize];
    if slot.len() < len {
        *slot = Box::leak(vec![value; len.next_power_of_two()].into_boxed_slice());
    }
    let run: &'static [u8] = slot;
    &run[..len]
}

/// Keyword column: sorted term table + per-doc ordinal (Lucene SORTED doc values).
pub struct KeywordReader<'a> {
    /// (offset, len) into `term_bytes` per ordinal, in sorted order.
    terms: std::sync::Arc<KeywordOffsets>,
    term_bytes: &'a [u8],
    num_docs: u32,
    ordinals: &'a [u8],
    /// Bytes per ordinal (1-4, see [`ord_width`]).
    width: usize,
    /// Sparse layout ([`TAG_SPARSE_KEYWORD`]): presence words and the rank
    /// before each word; `ordinals` then holds present docs only. Empty for
    /// a dense column.
    present: &'a [u8],
    ranks: &'a [u8],
}

/// Parsed offsets of an immutable keyword dictionary. The strings and
/// per-document ordinals stay in the mapped segment; only this small index
/// is retained, so opening a column per fetched hit does not scan its terms.
#[derive(Debug)]
pub struct KeywordTable {
    terms: std::sync::Arc<KeywordOffsets>,
    table_start: usize,
    table_end: usize,
    num_docs: u32,
    ordinals_start: usize,
    ordinals_end: usize,
    width: usize,
    /// Sparse layout: (ranks start, presence start, words).
    sparse: Option<(usize, usize, usize)>,
}

/// The docs of a keyword column grouped by ordinal (doc values inverted:
/// docs of ordinal `o` are `docs[starts[o]..starts[o + 1]]`, ascending).
/// Docs without a value are left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrdIndex {
    pub docs: Vec<u32>,
    pub starts: Vec<u32>,
}

/// Ordinal bounds of one [`BLOCK_SIZE`]-doc block of a keyword column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdBlock {
    /// Smallest ordinal (`u32::MAX` when no doc of the block has a value).
    pub min: u32,
    /// Largest ordinal.
    pub max: u32,
    /// Some doc of the block has no value.
    pub missing: bool,
}

/// Parse a keyword column's term table: `(offset, len)` per ordinal
/// relative to the table start, the table start and the position after it.
#[derive(Debug)]
pub struct KeywordOffsets {
    /// Plain table: (offset, length) per term. Front-coded table: (offset of
    /// each [`FC_BLOCK`]-term block, 0).
    indexed: Option<crate::admission::Allocation<Vec<(usize,usize)>>>,
    count: usize,
    encoded_len: usize,
    /// The table is front-coded ([`FC_MARK`]): only [`KeywordOffsets::term_at`]
    /// and [`KeywordOffsets::lower_bound`] read it.
    fc: bool,
    /// Its blocks are packed ([`FCP_MARK`]).
    packed: bool,
    /// A packed table's block heads, unpacked on its first by-value
    /// lookup ([`KeywordOffsets::lower_bound`]; `None`: not admitted).
    heads: std::sync::OnceLock<Option<crate::admission::Allocation<Vec<Box<[u8]>>>>>,
}

/// First varint of a front-coded term table (v30+, in place of the term
/// count, which follows): no plain table holds that many terms. Each block of
/// [`FC_BLOCK`] terms starts with a whole term (`vint len, bytes`); the others
/// are `vint shared prefix, vint suffix len, suffix` against the term before
/// (Lucene's terms dictionary blocks). High-cardinality keywords (a `_tsid`,
/// ips, urls) share long prefixes once sorted.
const FC_MARK: u64 = u64::MAX >> 2;
/// Terms per front-coded block.
const FC_BLOCK: usize = 16;
/// First varint of a packed front-coded term table (v33+,
/// [`crate::codec::PACKED_KEYWORD_TABLES_SINCE`]): each block opens with a
/// header byte (`terms` dictionary blocks' alphabet and length flags): its
/// head `vint len` then the head packed, then per other term its lengths
/// (one byte when the block's are all under 16) and its suffix packed. A
/// unique hex or base64 keyword's table (big5's event.id) takes 4 or 6
/// bits a byte.
const FCP_MARK: u64 = u64::MAX >> 3;

/// The term count a term table starts with (plain or front-coded).
pub(crate) fn term_count(data: &[u8]) -> Option<u64> {
    let mut pos = 0;
    let n = read_vint(data, &mut pos).ok()?;
    if n == FC_MARK || n == FCP_MARK { read_vint(data, &mut pos).ok() } else { Some(n) }
}

/// Streaming writer of a front-coded term table (packed or not): the table
/// header, then the terms in order; packed blocks are buffered to choose
/// their alphabet.
pub(crate) struct FcTableWriter {
    packed: bool,
    i: usize,
    prev: Vec<u8>,
    block: Vec<Vec<u8>>,
}

impl FcTableWriter {
    pub(crate) fn new(packed: bool) -> Self {
        Self { packed, i: 0, prev: Vec::new(), block: Vec::with_capacity(if packed { FC_BLOCK } else { 0 }) }
    }
    /// The table's marker and term count.
    pub(crate) fn header(&self, out: &mut Vec<u8>, num_terms: usize) {
        write_vint(if self.packed { FCP_MARK } else { FC_MARK }, out);
        write_vint(num_terms as u64, out);
    }
    pub(crate) fn push(&mut self, out: &mut Vec<u8>, t: &[u8]) {
        if !self.packed {
            write_front_coded(out, self.i, &self.prev, t);
            self.prev.clear();
            self.prev.extend_from_slice(t);
            self.i += 1;
            return;
        }
        self.block.push(t.to_vec());
        self.i += 1;
        if self.block.len() == FC_BLOCK {
            self.flush(out);
        }
    }
    pub(crate) fn finish(&mut self, out: &mut Vec<u8>) {
        if self.packed {
            self.flush(out);
        }
    }
    fn flush(&mut self, out: &mut Vec<u8>) {
        use crate::codec::terms::{choose_alpha, pack_suffix, NIBBLE_LENS};
        if self.block.is_empty() {
            return;
        }
        // (shared, suffix) per term: the head shares nothing.
        let mut parts: Vec<(usize, usize)> = Vec::with_capacity(self.block.len());
        let mut all: Vec<u8> = Vec::new();
        for (k, t) in self.block.iter().enumerate() {
            let shared = if k == 0 { 0 } else { t.iter().zip(&self.block[k - 1]).take_while(|(a, b)| a == b).count() };
            parts.push((shared, t.len() - shared));
            all.extend_from_slice(&t[shared..]);
        }
        let alpha = choose_alpha(&all);
        let nibble = parts[1..].iter().all(|&(sh, len)| sh < 16 && len < 16);
        out.push(alpha | if nibble { NIBBLE_LENS } else { 0 });
        for (k, t) in self.block.iter().enumerate() {
            let (shared, len) = parts[k];
            if k == 0 {
                write_vint(len as u64, out);
            } else if nibble {
                out.push(((shared << 4) | len) as u8);
            } else {
                write_vint(shared as u64, out);
                write_vint(len as u64, out);
            }
            pack_suffix(alpha, &t[shared..], out);
        }
        self.block.clear();
    }
}

/// Decode the terms of one front-coded block at `pos` (packed or not), at
/// most `n`, calling `f(term)` until it returns false; `pos` ends after the
/// last term read. `None`: malformed.
fn fc_block_terms(bytes: &[u8], pos: &mut usize, packed: bool, n: usize, mut f: impl FnMut(&[u8]) -> bool) -> Option<()> {
    use crate::codec::terms::{packed_len, unpack_suffix, ALPHA_RAW, NIBBLE_LENS};
    let head = if packed { let h = *bytes.get(*pos)?; *pos += 1; h } else { 0 };
    let alpha = head & 3;
    let mut term: Vec<u8> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    for k in 0..n {
        let (shared, len) = if k == 0 {
            (0, read_vint(bytes, pos).ok()? as usize)
        } else if head & NIBBLE_LENS != 0 {
            let b = *bytes.get(*pos)?;
            *pos += 1;
            ((b >> 4) as usize, (b & 15) as usize)
        } else {
            let s = read_vint(bytes, pos).ok()? as usize;
            (s, read_vint(bytes, pos).ok()? as usize)
        };
        if shared > term.len() {
            return None;
        }
        let stored = read_bytes(bytes, pos, packed_len(alpha, len)).ok()?;
        term.truncate(shared);
        if alpha == ALPHA_RAW {
            term.extend_from_slice(stored);
        } else {
            unpack_suffix(alpha, stored, len, &mut scratch);
            term.extend_from_slice(&scratch);
        }
        if !f(&term) {
            break;
        }
    }
    Some(())
}

/// Within the packed block at `pos` of `n` terms: the index of its first
/// term not before `key` (`> key` when `inclusive`, else `>= key`), or
/// `n`. Terms are compared as they are read, without unpacking: a term
/// sharing more of its predecessor than the predecessor shares with the
/// key orders as the predecessor; one sharing less is past the key (the
/// terms ascend); only one sharing exactly that much is compared, byte by
/// byte from there. `None`: malformed.
fn packed_block_lower_bound(bytes: &[u8], mut pos: usize, n: usize, key: &[u8], inclusive: bool) -> Option<usize> {
    use crate::codec::terms::{packed_char, packed_len, ALPHA_RAW, NIBBLE_LENS};
    let head = *bytes.get(pos)?;
    pos += 1;
    let alpha = head & 3;
    let at = |stored: &[u8], i: usize| if alpha == ALPHA_RAW { stored[i] } else { packed_char(alpha, stored, i) };
    // Common prefix of the previous term with `key`.
    let mut matched = 0usize;
    for k in 0..n {
        let (shared, len) = if k == 0 {
            (0, read_vint(bytes, &mut pos).ok()? as usize)
        } else if head & NIBBLE_LENS != 0 {
            let b = *bytes.get(pos)?;
            pos += 1;
            ((b >> 4) as usize, (b & 15) as usize)
        } else {
            let s = read_vint(bytes, &mut pos).ok()? as usize;
            (s, read_vint(bytes, &mut pos).ok()? as usize)
        };
        let stored = read_bytes(bytes, &mut pos, packed_len(alpha, len)).ok()?;
        if k > 0 && shared < matched {
            return Some(k);
        }
        if k > 0 && shared > matched {
            continue;
        }
        // Compare from `shared` on: the term is key[..shared] ++ suffix.
        let total = shared + len;
        let mut m = shared;
        while m < total && m < key.len() && at(stored, m - shared) == key[m] {
            m += 1;
        }
        let before = if m == total && m == key.len() {
            inclusive
        } else if m == total {
            true
        } else if m == key.len() {
            false
        } else {
            at(stored, m - shared) < key[m]
        };
        if !before {
            return Some(k);
        }
        matched = m;
    }
    Some(n)
}

/// The first term of the packed front-coded block at `pos` into `out`.
fn packed_block_head(bytes: &[u8], mut pos: usize, out: &mut Vec<u8>) -> Option<()> {
    use crate::codec::terms::{packed_len, unpack_suffix, ALPHA_RAW};
    let alpha = *bytes.get(pos)? & 3;
    pos += 1;
    let len = read_vint(bytes, &mut pos).ok()? as usize;
    let stored = read_bytes(bytes, &mut pos, packed_len(alpha, len)).ok()?;
    if alpha == ALPHA_RAW {
        out.clear();
        out.extend_from_slice(stored);
    } else {
        unpack_suffix(alpha, stored, len, out);
    }
    Some(())
}

/// Write a sorted term table, front-coded and packed ([`FCP_MARK`]).
pub(crate) fn write_term_table_packed<'t>(out: &mut Vec<u8>, num_terms: usize, terms: impl Iterator<Item = &'t [u8]>) {
    let mut w = FcTableWriter::new(true);
    w.header(out, num_terms);
    for t in terms {
        w.push(out, t);
    }
    w.finish(out);
}

/// Write a sorted term table (front-coded when `fc`).
pub(crate) fn write_term_table<'t>(out: &mut Vec<u8>, num_terms: usize, terms: impl Iterator<Item = &'t [u8]>, fc: bool) {
    if !fc {
        write_vint(num_terms as u64, out);
        for t in terms {
            write_vint(t.len() as u64, out);
            out.extend_from_slice(t);
        }
        return;
    }
    write_vint(FC_MARK, out);
    write_vint(num_terms as u64, out);
    let mut prev: &[u8] = &[];
    for (i, t) in terms.enumerate() {
        write_front_coded(out, i, prev, t);
        prev = t;
    }
}

/// Term `i` of a front-coded table after `prev` (see [`FC_MARK`]).
pub(crate) fn write_front_coded(out: &mut Vec<u8>, i: usize, prev: &[u8], t: &[u8]) {
    if i % FC_BLOCK == 0 {
        write_vint(t.len() as u64, out);
        out.extend_from_slice(t);
    } else {
        let shared = t.iter().zip(prev).take_while(|(a, b)| a == b).count();
        write_vint(shared as u64, out);
        write_vint((t.len() - shared) as u64, out);
        out.extend_from_slice(&t[shared..]);
    }
}

impl KeywordOffsets {
    fn len(&self)->usize {self.count}
    /// Every term of the table `bytes` in ordinal order, each block decoded
    /// once (where [`Self::term_at`] decodes a block per term): `f(ordinal,
    /// term)` returning false stops. `None` when the table is unreadable.
    fn for_each_term(&self, bytes: &[u8], mut f: impl FnMut(usize, &[u8]) -> bool) -> Option<()> {
        if !self.fc {
            for ordinal in 0..self.count {
                let (off, len) = self.get(bytes, ordinal)?;
                if !f(ordinal, bytes.get(off..off + len)?) { return Some(()); }
            }
            return Some(());
        }
        let mut ordinal = 0usize;
        for block in 0..self.count.div_ceil(FC_BLOCK) {
            let n = FC_BLOCK.min(self.count - block * FC_BLOCK);
            let mut pos = self.block_start(bytes, block)?;
            let mut go = true;
            if self.packed {
                fc_block_terms(bytes, &mut pos, true, n, |t| {
                    go = f(ordinal, t);
                    ordinal += 1;
                    go
                })?;
            } else {
                let len = read_vint(bytes, &mut pos).ok()? as usize;
                let mut term = read_bytes(bytes, &mut pos, len).ok()?.to_vec();
                go = f(ordinal, &term);
                ordinal += 1;
                for _ in 1..n {
                    if !go { break; }
                    let shared = read_vint(bytes, &mut pos).ok()? as usize;
                    let suffix = read_vint(bytes, &mut pos).ok()? as usize;
                    let suffix = read_bytes(bytes, &mut pos, suffix).ok()?;
                    if shared > term.len() { return None; }
                    term.truncate(shared);
                    term.extend_from_slice(suffix);
                    go = f(ordinal, &term);
                    ordinal += 1;
                }
            }
            if !go { return Some(()); }
        }
        Some(())
    }
    /// Term `ordinal` of the table `bytes`: borrowed from a plain table,
    /// decoded from its block in a front-coded one.
    fn term_at<'b>(&self, bytes: &'b [u8], ordinal: usize) -> Option<std::borrow::Cow<'b, [u8]>> {
        if !self.fc {
            let (off, len) = self.get(bytes, ordinal)?;
            return bytes.get(off..off + len).map(std::borrow::Cow::Borrowed);
        }
        if ordinal >= self.count {
            return None;
        }
        let mut pos = self.block_start(bytes, ordinal / FC_BLOCK)?;
        if self.packed {
            let mut out = None;
            let mut k = 0;
            fc_block_terms(bytes, &mut pos, true, ordinal % FC_BLOCK + 1, |t| {
                if k == ordinal % FC_BLOCK { out = Some(t.to_vec()); }
                k += 1;
                true
            })?;
            return out.map(std::borrow::Cow::Owned);
        }
        let len = read_vint(bytes, &mut pos).ok()? as usize;
        let head = read_bytes(bytes, &mut pos, len).ok()?;
        // A block head is stored whole: borrowed.
        if ordinal % FC_BLOCK == 0 {
            return Some(std::borrow::Cow::Borrowed(head));
        }
        let mut term = head.to_vec();
        for _ in 0..ordinal % FC_BLOCK {
            let shared = read_vint(bytes, &mut pos).ok()? as usize;
            let suffix = read_vint(bytes, &mut pos).ok()? as usize;
            let suffix = read_bytes(bytes, &mut pos, suffix).ok()?;
            if shared > term.len() {
                return None;
            }
            term.truncate(shared);
            term.extend_from_slice(suffix);
        }
        Some(std::borrow::Cow::Owned(term))
    }
    /// Every block's first term of a packed table, unpacked, charged to the
    /// shared budget (`None` when refused or malformed).
    fn unpacked_heads(&self, bytes: &[u8]) -> Option<crate::admission::Allocation<Vec<Box<[u8]>>>> {
        // Without the block index, finding each block decodes the ones before.
        self.indexed.as_ref()?;
        let blocks = self.count.div_ceil(FC_BLOCK);
        let mut heads = Vec::with_capacity(blocks);
        let mut head = Vec::new();
        let mut size = blocks * std::mem::size_of::<Box<[u8]>>();
        for b in 0..blocks {
            packed_block_head(bytes, self.block_start(bytes, b)?, &mut head)?;
            size += head.len();
            heads.push(head.as_slice().into());
        }
        let memory = crate::admission::shared().reserve(size as u64, "keyword table block heads").ok()?;
        Some(crate::admission::Allocation::new(heads, memory))
    }
    /// Offset of front-coded block `block` (indexed, or found by decoding
    /// the blocks before it when the index was not admitted).
    fn block_start(&self, bytes: &[u8], block: usize) -> Option<usize> {
        if let Some(index) = &self.indexed {
            return index.get(block).map(|&(off, _)| off);
        }
        let mut pos = 0;
        if self.packed {
            for b in 0..block {
                let n = (self.count - b * FC_BLOCK).min(FC_BLOCK);
                fc_block_terms(bytes, &mut pos, true, n, |_| true)?;
            }
            return Some(pos);
        }
        for i in 0..block * FC_BLOCK {
            if i % FC_BLOCK == 0 {
                let len = read_vint(bytes, &mut pos).ok()? as usize;
                read_bytes(bytes, &mut pos, len).ok()?;
            } else {
                read_vint(bytes, &mut pos).ok()?;
                let suffix = read_vint(bytes, &mut pos).ok()? as usize;
                read_bytes(bytes, &mut pos, suffix).ok()?;
            }
        }
        Some(pos)
    }
    /// The first ordinal whose term is `> key` (`inclusive`) or `>= key`.
    fn lower_bound(&self, bytes: &[u8], key: &[u8], inclusive: bool) -> usize {
        let before = |t: &[u8]| if inclusive { t <= key } else { t < key };
        if !self.fc {
            return self.partition_point(bytes, |&(off, len)| before(&bytes[off..off + len]));
        }
        // Blocks whose first term is before the key; then within the last.
        let blocks = self.count.div_ceil(FC_BLOCK);
        let (mut lo, mut hi) = (0usize, blocks);
        // A packed table's heads, unpacked once (a write's synthetic id
        // lookup probes the `_tsid` table of every segment: unpacking the
        // probed heads every time cost TSDB indexing ~15%); a reused buffer
        // when they were not admitted.
        let heads = if self.packed { self.heads.get_or_init(|| self.unpacked_heads(bytes)).as_ref() } else { None };
        if let Some(heads) = heads {
            lo = heads.partition_point(|h| before(h));
            hi = lo;
        }
        let mut head = Vec::new();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let is_before = if self.packed {
                self.block_start(bytes, mid).and_then(|pos| packed_block_head(bytes, pos, &mut head)).is_some_and(|()| before(&head))
            } else {
                self.term_at(bytes, mid * FC_BLOCK).is_some_and(|t| before(&t))
            };
            if is_before { lo = mid + 1 } else { hi = mid }
        }
        if lo == 0 {
            return 0;
        }
        // Within the last such block: its terms decoded in order, once.
        let block = lo - 1;
        let end = ((block + 1) * FC_BLOCK).min(self.count);
        let Some(mut pos) = self.block_start(bytes, block) else { return end };
        if self.packed {
            if let Some(i) = packed_block_lower_bound(bytes, pos, end - block * FC_BLOCK, key, inclusive) {
                return block * FC_BLOCK + i;
            }
            let mut o = block * FC_BLOCK;
            let mut found = None;
            let ok = fc_block_terms(bytes, &mut pos, true, end - o, |t| {
                if !before(t) { found = Some(o); return false; }
                o += 1;
                true
            });
            return match (ok, found) { (_, Some(o)) => o, (Some(()), None) => end, (None, None) => o };
        }
        let mut term: Vec<u8> = Vec::new();
        for o in block * FC_BLOCK..end {
            let ok = (|| -> Option<()> {
                if o % FC_BLOCK == 0 {
                    let len = read_vint(bytes, &mut pos).ok()? as usize;
                    term.clear();
                    term.extend_from_slice(read_bytes(bytes, &mut pos, len).ok()?);
                } else {
                    let shared = read_vint(bytes, &mut pos).ok()? as usize;
                    let suffix = read_vint(bytes, &mut pos).ok()? as usize;
                    let suffix = read_bytes(bytes, &mut pos, suffix).ok()?;
                    if shared > term.len() { return None; }
                    term.truncate(shared);
                    term.extend_from_slice(suffix);
                }
                Some(())
            })();
            if ok.is_none() || !before(&term) {
                return o;
            }
        }
        end
    }
    fn get(&self, bytes:&[u8], ordinal:usize)->Option<(usize,usize)> {
        if ordinal>=self.count{return None;}
        if let Some(index)=&self.indexed {return index.get(ordinal).copied();}
        // Low-memory fallback reads only immutable encoded bytes. No offset
        // vector is allocated and a budget refusal never hides a field.
        let mut pos=0;
        for i in 0..=ordinal {
            let len=read_vint(bytes,&mut pos).ok()? as usize;
            let off=pos;read_bytes(bytes,&mut pos,len).ok()?;
            if i==ordinal{return Some((off,len));}
        }
        None
    }
    fn partition_point(&self,bytes:&[u8],mut pred:impl FnMut(&(usize,usize))->bool)->usize {
        let(mut lo,mut hi)=(0,self.count);
        while lo<hi {let mid=lo+(hi-lo)/2;let pair=self.get(bytes,mid).expect("validated keyword table");if pred(&pair){lo=mid+1}else{hi=mid}}
        lo
    }
    fn binary_search_by(&self,bytes:&[u8],mut cmp:impl FnMut(&(usize,usize))->std::cmp::Ordering)->std::result::Result<usize,usize> {
        let p=self.partition_point(bytes,|pair|cmp(pair)==std::cmp::Ordering::Less);
        if self.get(bytes,p).is_some_and(|pair|cmp(&pair)==std::cmp::Ordering::Equal){Ok(p)}else{Err(p)}
    }
    #[cfg(test)]
    fn as_ptr(&self)->*const (usize,usize) {self.indexed.as_ref().map_or(std::ptr::null(),|v|v.as_ptr())}
}
fn parse_term_table(data: &[u8]) -> Result<(KeywordOffsets, usize, usize)> {
    parse_term_table_admitted(data,&crate::admission::shared())
}
fn parse_term_table_admitted(data:&[u8],budget:&std::sync::Arc<crate::admission::Budget>)->Result<(KeywordOffsets,usize,usize)> {
    let mut pos=0;
    let first=read_vint(data,&mut pos)?;
    if first==FCP_MARK {
        let num_terms=read_vint(data,&mut pos)? as usize;
        let table_start=pos;
        let blocks=num_terms.div_ceil(FC_BLOCK);
        let memory=budget.reserve((blocks as u64).saturating_mul(std::mem::size_of::<(usize,usize)>() as u64).saturating_add(256),"keyword block offset cache").ok();
        let mut indexed=memory.map(|memory|crate::admission::Allocation::new(Vec::with_capacity(blocks),memory));
        for b in 0..blocks {
            if let Some(index)=&mut indexed {index.push((pos-table_start,0));}
            let n=(num_terms-b*FC_BLOCK).min(FC_BLOCK);
            let mut local=pos-table_start;
            fc_block_terms(&data[table_start..],&mut local,true,n,|_|true).ok_or_else(||RusticError::Corrupted("packed keyword table".into()))?;
            pos=table_start+local;
        }
        return Ok((KeywordOffsets{indexed,count:num_terms,encoded_len:pos-table_start,fc:true,packed:true,heads:std::sync::OnceLock::new()},table_start,pos));
    }
    if first==FC_MARK {
        let num_terms=read_vint(data,&mut pos)? as usize;
        let table_start=pos;
        let blocks=num_terms.div_ceil(FC_BLOCK);
        let memory=budget.reserve((blocks as u64).saturating_mul(std::mem::size_of::<(usize,usize)>() as u64).saturating_add(256),"keyword block offset cache").ok();
        let mut indexed=memory.map(|memory|crate::admission::Allocation::new(Vec::with_capacity(blocks),memory));
        let mut prev=0usize;
        for i in 0..num_terms {
            if i%FC_BLOCK==0 {
                if let Some(index)=&mut indexed {index.push((pos-table_start,0));}
                let len=read_vint(data,&mut pos)? as usize;
                read_bytes(data,&mut pos,len)?;
                prev=len;
            } else {
                let shared=read_vint(data,&mut pos)? as usize;
                let suffix=read_vint(data,&mut pos)? as usize;
                if shared>prev {return Err(RusticError::Corrupted("front-coded keyword table".into()));}
                read_bytes(data,&mut pos,suffix)?;
                prev=shared+suffix;
            }
        }
        return Ok((KeywordOffsets{indexed,count:num_terms,encoded_len:pos-table_start,fc:true,packed:false,heads:std::sync::OnceLock::new()},table_start,pos));
    }
    let num_terms=first as usize;
    let table_start=pos;
    let memory=budget.reserve((num_terms as u64).saturating_mul(std::mem::size_of::<(usize,usize)>() as u64).saturating_add(256),"keyword offset cache").ok();
    let mut indexed=memory.map(|memory|crate::admission::Allocation::new(Vec::with_capacity(num_terms),memory));
    for _ in 0..num_terms {
        let len=read_vint(data,&mut pos)? as usize;
        if let Some(index)=&mut indexed {index.push((pos-table_start,len));}
        read_bytes(data,&mut pos,len)?;
    }
    Ok((KeywordOffsets{indexed,count:num_terms,encoded_len:pos-table_start,fc:false,packed:false,heads:std::sync::OnceLock::new()},table_start,pos))
}

/// Locate the term table of a keyword column whose parsed table is known:
/// (table start, position after it).
fn term_table_bounds(data: &[u8], terms: &KeywordOffsets) -> Result<(usize, usize)> {
    let mut pos = 0;
    let mut n = read_vint(data, &mut pos)?;
    if n == FC_MARK || n == FCP_MARK {
        n = read_vint(data, &mut pos)?;
    }
    let n = n as usize;
    if n != terms.len() {
        return Err(RusticError::Corrupted("keyword term table mismatch".into()));
    }
    let end = terms.encoded_len;
    Ok((pos, pos + end))
}

/// The parsed term table of a keyword column (single- or multi-valued), to
/// share between readers of the same column.
pub fn keyword_term_table(data: &[u8]) -> Result<std::sync::Arc<KeywordOffsets>> {
    Ok(std::sync::Arc::new(parse_term_table(data)?.0))
}


impl KeywordTable {
    /// Parse a dense keyword column ([`TAG_KEYWORD`] on disk).
    pub fn parse(data: &[u8]) -> Result<Self> {
        Self::parse_layout(data, false)
    }

    /// Parse a keyword column of either layout (`sparse`: on-disk
    /// [`TAG_SPARSE_KEYWORD`], see [`ColumnEntry::sparse`]).
    pub fn parse_layout(data: &[u8], layout: impl Into<OrdLayout>) -> Result<Self> {
        let layout = layout.into();
        let (terms,table_start,mut pos)=parse_term_table(data)?;
        let table_end=pos;
        let (num_docs, width) = read_ord_header(data, &mut pos)?;
        if layout == OrdLayout::Blocked {
            let ordinals_start = pos;
            read_blocked_ords(data, &mut pos, num_docs, terms.len(), width)?;
            return Ok(Self { terms: std::sync::Arc::new(terms), table_start, table_end, num_docs, ordinals_start, ordinals_end: pos, width: 0, sparse: None });
        }
        let sparse = layout == OrdLayout::Sparse;
        let (layout, count) = if sparse {
            let (ranks, present, words, count) = read_sparse_ords(data, &mut pos, num_docs, true)?;
            (Some((ranks, present, words)), count)
        } else {
            (None, num_docs as usize)
        };
        let ordinals_start = pos;
        read_bytes(data, &mut pos, count * width + ord_pad(width))?;
        Ok(Self { terms: std::sync::Arc::new(terms), table_start, table_end, num_docs, ordinals_start, ordinals_end: pos, width, sparse: layout })
    }
}

/// Read a sparse keyword column's presence header at `pos` (after the
/// ordinal header word): `(ranks start, presence start, words, present
/// count)`, leaving `pos` at its ordinals. `validate` checks every rank
/// against the bitmap (O(words): done once when a segment parses the
/// column, not per reader).
fn read_sparse_ords(data: &[u8], pos: &mut usize, num_docs: u32, validate: bool) -> Result<(usize, usize, usize, usize)> {
    let count = read_u32_le(data, pos)? as usize;
    if count > num_docs as usize {
        return Err(RusticError::Corrupted("keyword presence count exceeds docs".into()));
    }
    let words = (num_docs as usize).div_ceil(64);
    let ranks_start = *pos;
    let ranks = read_bytes(data, pos, words * 4)?;
    let present_start = *pos;
    let present = read_bytes(data, pos, words * 8)?;
    if validate {
        let mut rank = 0u32;
        for w in 0..words {
            let stored = u32::from_le_bytes(ranks[w * 4..w * 4 + 4].try_into().unwrap());
            let bits = u64::from_le_bytes(present[w * 8..w * 8 + 8].try_into().unwrap());
            if stored != rank || (w + 1 == words && num_docs % 64 != 0 && bits >> (num_docs % 64) != 0) {
                return Err(RusticError::Corrupted("invalid keyword presence rank".into()));
            }
            rank += bits.count_ones();
        }
        if rank as usize != count {
            return Err(RusticError::Corrupted("keyword presence count mismatch".into()));
        }
    }
    Ok((ranks_start, present_start, words, count))
}

/// Ordinal of doc `i` of a sparse column: the doc's presence bit, then its
/// rank among present docs (`ranks` before its word, plus the set bits
/// below it in the word). A rank past the ordinals (a corrupt bitmap the
/// unvalidated open did not check) reads as missing.
#[inline(always)]
fn sparse_ord_at(present: &[u8], ranks: &[u8], ordinals: &[u8], width: usize, i: usize) -> u32 {
    let w = i / 64;
    let bits = u64::from_le_bytes(present[w * 8..w * 8 + 8].try_into().unwrap());
    let b = i % 64;
    if (bits >> b) & 1 == 0 {
        return MISSING_ORD;
    }
    let rank = u32::from_le_bytes(ranks[w * 4..w * 4 + 4].try_into().unwrap()) as usize + (bits & ((1u64 << b) - 1)).count_ones() as usize;
    if (rank + 1) * width + ord_pad(width) > ordinals.len() {
        return MISSING_ORD;
    }
    ord_at(ordinals, width, rank)
}

/// Check a blocked column's ordinals area at `pos` (block entries, then
/// the packed bits and an 8-byte pad) and move `pos` past it.
/// Ord-header width code of a blocked keyword column of one term every
/// doc has (v37+, [`crate::codec::CONSTANT_KEYWORD_COLUMN_SINCE`]): no
/// block table and no ordinals follow (every ordinal is 0). Blocked
/// columns otherwise write the 4-byte code.
const CONSTANT_ORDS_WIDTH: usize = 1;

fn read_blocked_ords(data: &[u8], pos: &mut usize, num_docs: u32, num_terms: usize, width: usize) -> Result<()> {
    if width == CONSTANT_ORDS_WIDTH {
        return if num_terms == 1 { Ok(()) } else { Err(RusticError::Corrupted("constant keyword column of several terms".into())) };
    }
    // Only the extent is checked (columns open per fetched hit): the last
    // block's entry gives the end; `blocked_ord_at` bounds every read.
    let _ = num_terms;
    let blocks = (num_docs as usize).div_ceil(BLOCK_SIZE);
    let table = read_bytes(data, pos, blocks * ORD_BLOCK_ENTRY)?;
    let mut end = blocks * ORD_BLOCK_ENTRY;
    if let Some(e) = table.chunks_exact(ORD_BLOCK_ENTRY).last() {
        let x = u64::from_le_bytes(e[..8].try_into().unwrap());
        // Every entry holds its block's offset, a block of one value too.
        let (off, w) = ((x >> 8) as usize, (x & ORD_BLOCK_WIDTH) as usize);
        end = off.saturating_add(32 * w.min(32));
    }
    let data_start = *pos - blocks * ORD_BLOCK_ENTRY;
    let end = end.max(blocks * ORD_BLOCK_ENTRY);
    let need = end.checked_sub(*pos - data_start).ok_or_else(|| RusticError::Corrupted("invalid keyword ordinal blocks".into()))?;
    read_bytes(data, pos, need.saturating_add(8))?;
    Ok(())
}

/// Write a blocked keyword column's ordinals area ([`OrdLayout::Blocked`])
/// when it is at least a tenth smaller than `width`-byte ordinals (and,
/// with `vs_sparse`, smaller than the sparse layout). Without
/// `missing_ok` every doc must have a value; with it (format
/// [`crate::codec::BLOCKED_MISSING_KEYWORD_SINCE`]+) a block with missing
/// docs is flagged ([`ORD_BLOCK_MISSING`]). Returns whether it wrote.
fn write_blocked_ords(body: &mut Vec<u8>, num_docs: u32, width: usize, missing_ok: bool, vs_sparse: bool, ords: &mut dyn FnMut(&mut dyn FnMut(u32))) -> bool {
    let n = num_docs as usize;
    let blocks = n.div_ceil(BLOCK_SIZE);
    let mut mins = vec![u32::MAX; blocks];
    let mut maxs = vec![0u32; blocks];
    let mut gaps = vec![false; blocks];
    let mut d = 0usize;
    let mut missing = false;
    let mut present = 0usize;
    ords(&mut |o| {
        if d < n {
            let b = d / BLOCK_SIZE;
            if o == MISSING_ORD {
                missing = true;
                gaps[b] = true;
            } else {
                present += 1;
                mins[b] = mins[b].min(o);
                maxs[b] = maxs[b].max(o);
            }
        }
        d += 1;
    });
    if (missing && !missing_ok) || d != n || n == 0 {
        return false;
    }
    // A block with gaps keeps the all-ones offset for them: one more code.
    let widths: Vec<usize> = (0..blocks)
        .map(|b| match (gaps[b], mins[b] <= maxs[b]) {
            (_, false) => 0,
            (false, true) => 32 - (maxs[b] - mins[b]).leading_zeros() as usize,
            (true, true) => 32 - (maxs[b] - mins[b] + 1).leading_zeros() as usize,
        })
        .collect();
    let size = blocks * ORD_BLOCK_ENTRY + widths.iter().map(|w| 32 * w).sum::<usize>() + 8;
    // The sparse layout: its presence index plus present ordinals.
    let sparse_bytes = 4 + n.div_ceil(64) * 12 + present * width;
    if size * 10 >= (n * width + ord_pad(width)) * 9 || (vs_sparse && missing && size >= sparse_bytes) {
        return false;
    }
    write_ord_header(num_docs, 4, body);
    let mut off = blocks * ORD_BLOCK_ENTRY;
    for b in 0..blocks {
        let flag = if gaps[b] { ORD_BLOCK_MISSING } else { 0 };
        body.extend_from_slice(&((off as u64) << 8 | flag | widths[b] as u64).to_le_bytes());
        body.extend_from_slice(&(if mins[b] <= maxs[b] { mins[b] } else { 0 }).to_le_bytes());
        off += 32 * widths[b];
    }
    let mut acc = 0u64;
    let mut nbits = 0usize;
    let mut d = 0usize;
    ords(&mut |o| {
        if d < n {
            let b = d / BLOCK_SIZE;
            let w = widths[b];
            if w > 0 {
                let code = if o == MISSING_ORD { (1u64 << w) - 1 } else { (o - mins[b]) as u64 };
                acc |= code << nbits;
                nbits += w;
                while nbits >= 8 {
                    body.push(acc as u8);
                    acc >>= 8;
                    nbits -= 8;
                }
            }
        }
        d += 1;
    });
    // The last block's unused slots: zero bits.
    let last = blocks - 1;
    let pad_slots = blocks * BLOCK_SIZE - n;
    for _ in 0..pad_slots {
        nbits += widths[last];
        while nbits >= 8 {
            body.push(acc as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    body.extend_from_slice(&[0u8; 8]);
    true
}

/// Write a single-valued keyword column's ordinal header and ordinals
/// after its term table, returning the column's on-disk tag. `ords` calls
/// its argument with every doc's ordinal in doc order ([`MISSING_ORD`] for
/// none); it is called twice when `sparse_ok`. With `sparse_ok` (format
/// [`crate::codec::SPARSE_KEYWORD_SINCE`]+) a column whose present docs'
/// ordinals plus presence index are smaller than one ordinal per doc, and
/// which at most half the docs have, is written sparse
/// ([`TAG_SPARSE_KEYWORD`]); otherwise dense ([`TAG_KEYWORD`]).
pub(crate) fn write_keyword_ords(body: &mut Vec<u8>, num_terms: usize, num_docs: u32, sparse_ok: bool, ords: &mut dyn FnMut(&mut dyn FnMut(u32))) -> u8 {
    write_keyword_ords_v(body, num_terms, num_docs, sparse_ok, false, ords)
}

/// [`write_keyword_ords`]; `blocked_ok`: format v28+, a column every doc
/// has a value in may pack its ordinals per block ([`TAG_BLOCK_KEYWORD`]).
pub(crate) fn write_keyword_ords_v(body: &mut Vec<u8>, num_terms: usize, num_docs: u32, sparse_ok: bool, blocked_ok: bool, ords: &mut dyn FnMut(&mut dyn FnMut(u32))) -> u8 {
    write_keyword_ords_vm(body, num_terms, num_docs, sparse_ok, blocked_ok, false, ords)
}

/// [`write_keyword_ords_v`]; `blocked_missing_ok`: format v36+, a column
/// with docs without a value may pack its ordinals per block too
/// ([`ORD_BLOCK_MISSING`]) when that is smaller than both other layouts.
pub(crate) fn write_keyword_ords_vm(body: &mut Vec<u8>, num_terms: usize, num_docs: u32, sparse_ok: bool, blocked_ok: bool, blocked_missing_ok: bool, ords: &mut dyn FnMut(&mut dyn FnMut(u32))) -> u8 {
    write_keyword_ords_vmc(body, num_terms, num_docs, sparse_ok, blocked_ok, blocked_missing_ok, false, ords)
}

/// [`write_keyword_ords_vm`]; `constant_ok`: format v37+, a column of one
/// term every doc has stores no ordinals ([`CONSTANT_ORDS_WIDTH`]: a
/// TSDB `ecs.version` or `agent.type`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_keyword_ords_vmc(body: &mut Vec<u8>, num_terms: usize, num_docs: u32, sparse_ok: bool, blocked_ok: bool, blocked_missing_ok: bool, constant_ok: bool, ords: &mut dyn FnMut(&mut dyn FnMut(u32))) -> u8 {
    let width = ord_width(num_terms, num_docs);
    if blocked_ok && constant_ok && num_terms == 1 && num_docs > 0 {
        let (mut all, mut d) = (true, 0usize);
        ords(&mut |o| {
            if d < num_docs as usize && o != 0 {
                all = false;
            }
            d += 1;
        });
        if all && d == num_docs as usize {
            write_ord_header(num_docs, CONSTANT_ORDS_WIDTH, body);
            return TAG_BLOCK_KEYWORD;
        }
    }
    if blocked_ok {
        let mark = body.len();
        if write_blocked_ords(body, num_docs, width, blocked_missing_ok, sparse_ok, ords) {
            return TAG_BLOCK_KEYWORD;
        }
        body.truncate(mark);
    }
    let n = num_docs as usize;
    let nwords = n.div_ceil(64);
    let mut words: Vec<u64> = Vec::new();
    let mut present = 0usize;
    if sparse_ok {
        words = vec![0u64; nwords];
        let mut d = 0usize;
        ords(&mut |o| {
            if o != MISSING_ORD && d < n {
                words[d / 64] |= 1u64 << (d % 64);
                present += 1;
            }
            d += 1;
        });
    }
    // Sparse only for columns at most half present: a mostly present
    // column would pay a rank lookup per doc (terms aggregations, sorts)
    // to save a few percent.
    let sparse = sparse_ok && present * 2 <= n && 4 + nwords * 12 + present * width < n * width;
    write_ord_header(num_docs, width, body);
    if !sparse {
        ords(&mut |o| push_ord(o, width, body));
        finish_ords(width, body);
        return TAG_KEYWORD;
    }
    body.extend_from_slice(&(present as u32).to_le_bytes());
    let mut rank = 0u32;
    for w in &words {
        body.extend_from_slice(&rank.to_le_bytes());
        rank += w.count_ones();
    }
    for w in &words {
        body.extend_from_slice(&w.to_le_bytes());
    }
    ords(&mut |o| if o != MISSING_ORD { push_ord(o, width, body) });
    finish_ords(width, body);
    TAG_SPARSE_KEYWORD
}

/// Shared immutable keyword bytes for lazy scorers. Derived from the owned
/// column approach in perf/percolate; reuses this branch's checked term table.
pub struct OwnedKeywordColumn {
    data: super::directory::FileSlice,
    terms: std::sync::Arc<KeywordOffsets>,
    term_start: usize,
    ordinals_start: usize,
    ordinals_end: usize,
    num_docs: u32,
    width: usize,
    /// Sparse layout: (ranks start, presence start), see [`KeywordReader`].
    sparse: Option<(usize, usize)>,
}

impl OwnedKeywordColumn {
    /// Open a dense keyword column ([`TAG_KEYWORD`] on disk).
    pub fn open(data: super::directory::FileSlice, terms: Option<std::sync::Arc<KeywordOffsets>>) -> Result<Self> {
        Self::open_layout(data, terms, false)
    }
    /// Open a keyword column of either layout ([`ColumnEntry::sparse`]).
    pub fn open_layout(data: super::directory::FileSlice, terms: Option<std::sync::Arc<KeywordOffsets>>, layout: impl Into<OrdLayout>) -> Result<Self> {
        let reader = KeywordReader::open_with_layout(data.as_slice(), terms, layout)?;
        let base = data.as_slice().as_ptr() as usize;
        let term_start = reader.term_bytes.as_ptr() as usize - base;
        let ordinals_start = reader.ordinals.as_ptr() as usize - base;
        let ordinals_end = ordinals_start + reader.ordinals.len();
        let sparse = (!reader.present.is_empty()).then(|| (reader.ranks.as_ptr() as usize - base, reader.present.as_ptr() as usize - base));
        let terms = reader.terms.clone();
        let (num_docs, width) = (reader.num_docs, reader.width);
        Ok(Self { data, terms, term_start, ordinals_start, ordinals_end, num_docs, width, sparse })
    }
    pub fn num_terms(&self) -> usize { self.terms.len() }
    /// The first doc at or after `from` that may have a value, and its
    /// ordinal ([`MISSING_ORD`] for a dense column's missing doc); `None`
    /// past the last doc. A sparse column skips docs without a value a
    /// presence word at a time, so a scan visits only its present docs.
    #[inline]
    pub fn next_value(&self, from: DocId) -> Option<(DocId, u32)> {
        if from >= self.num_docs { return None; }
        let Some((ranks, present)) = self.sparse else { return Some((from, self.ord(from))) };
        let bytes = self.data.as_slice();
        let words = (self.num_docs as usize).div_ceil(64);
        let mut w = from as usize / 64;
        let word = |w: usize| u64::from_le_bytes(bytes[present + w * 8..present + w * 8 + 8].try_into().unwrap());
        let mut bits = word(w) & (u64::MAX << (from % 64));
        while bits == 0 {
            w += 1;
            if w >= words { return None; }
            bits = word(w);
        }
        let b = bits.trailing_zeros() as usize;
        let doc = w * 64 + b;
        if doc >= self.num_docs as usize { return None; }
        let all = word(w);
        let rank = u32::from_le_bytes(bytes[ranks + w * 4..ranks + w * 4 + 4].try_into().unwrap()) as usize + (all & ((1u64 << b) - 1)).count_ones() as usize;
        let ordinals = &bytes[self.ordinals_start..self.ordinals_end];
        let ord = if (rank + 1) * self.width + ord_pad(self.width) > ordinals.len() { MISSING_ORD } else { ord_at(ordinals, self.width, rank) };
        Some((doc as DocId, ord))
    }
    #[inline]
    pub fn ord(&self, doc: DocId) -> u32 {
        if doc >= self.num_docs { return MISSING_ORD; }
        let bytes = self.data.as_slice();
        let ordinals = &bytes[self.ordinals_start..self.ordinals_end];
        match self.sparse {
            None => ord_at(ordinals, self.width, doc as usize),
            Some((ranks, present)) => {
                let words = (self.num_docs as usize).div_ceil(64);
                sparse_ord_at(&bytes[present..present + words * 8], &bytes[ranks..ranks + words * 4], ordinals, self.width, doc as usize)
            }
        }
    }
    pub fn term(&self, ord: u32) -> Option<std::borrow::Cow<'_, str>> {
        cow_str(self.terms.term_at(&self.data.as_slice()[self.term_start..self.term_start + self.terms.encoded_len], ord as usize)?)
    }
}

/// A term's bytes as UTF-8, keeping a borrowed slice borrowed.
fn cow_str(b: std::borrow::Cow<'_, [u8]>) -> Option<std::borrow::Cow<'_, str>> {
    match b {
        std::borrow::Cow::Borrowed(b) => std::str::from_utf8(b).ok().map(std::borrow::Cow::Borrowed),
        std::borrow::Cow::Owned(v) => String::from_utf8(v).ok().map(std::borrow::Cow::Owned),
    }
}

pub const MISSING_ORD: u32 = u32::MAX;

/// Top bits of a keyword column's `num_docs` word holding its ordinal width.
const ORD_WIDTH_SHIFT: u32 = 30;

/// Byte width of the ordinals of a keyword column with `num_terms` terms
/// and `num_docs` docs: the narrowest that also leaves the all-ones value
/// free for [`MISSING_ORD`]. Columns too large for the width bits keep 4.
pub(crate) fn ord_width(num_terms: usize, num_docs: u32) -> usize {
    if num_docs >= 1 << ORD_WIDTH_SHIFT {
        return 4;
    }
    match num_terms {
        n if n < 0xFF => 1,
        n if n < 0xFFFF => 2,
        n if n < 0xFF_FFFF => 3,
        _ => 4,
    }
}

/// Write a keyword column's `num_docs` word for ordinals `width` bytes wide.
pub(crate) fn write_ord_header(num_docs: u32, width: usize, out: &mut Vec<u8>) {
    let code = if width == 4 { 0 } else { width as u32 };
    out.extend_from_slice(&(num_docs | code << ORD_WIDTH_SHIFT).to_le_bytes());
}

/// Append one ordinal `width` bytes wide.
#[inline]
pub(crate) fn push_ord(ord: u32, width: usize, out: &mut Vec<u8>) {
    out.extend_from_slice(&ord.to_le_bytes()[..width]);
}

/// Bytes after `width`-byte ordinals that let [`ord_at`] load 4 at a time.
fn ord_pad(width: usize) -> usize {
    4 - width
}

/// Close a keyword column's ordinals written with [`push_ord`].
pub(crate) fn finish_ords(width: usize, out: &mut Vec<u8>) {
    out.resize(out.len() + ord_pad(width), 0);
}

/// Read a keyword column's `num_docs` word: `(num_docs, ordinal width)`.
fn read_ord_header(data: &[u8], pos: &mut usize) -> Result<(u32, usize)> {
    let raw = read_u32_le(data, pos)?;
    let code = raw >> ORD_WIDTH_SHIFT;
    Ok((raw & ((1 << ORD_WIDTH_SHIFT) - 1), if code == 0 { 4 } else { code as usize }))
}

/// Ordinal `i` of `width`-byte ordinals (followed by [`ord_pad`] bytes):
/// one 4-byte load and a mask; the all-ones value is [`MISSING_ORD`].
#[inline(always)]
fn ord_at(ordinals: &[u8], width: usize, i: usize) -> u32 {
    // Width 0: a blocked column ([`OrdLayout::Blocked`]); without
    // ordinals, one term every doc has ([`CONSTANT_ORDS_WIDTH`]).
    if width == 0 {
        if ordinals.is_empty() {
            return 0;
        }
        return blocked_ord_at(ordinals, i);
    }
    let mask = u32::MAX >> (32 - 8 * width as u32);
    let o = u32::from_le_bytes(ordinals[i * width..i * width + 4].try_into().unwrap()) & mask;
    if o == mask { MISSING_ORD } else { o }
}

impl<'a> KeywordReader<'a> {
    /// Share offsets without reparsing or copying the immutable dictionary.
    pub(crate) fn shared_terms(&self) -> std::sync::Arc<KeywordOffsets> {
        self.terms.clone()
    }

    /// Open a dense keyword column ([`TAG_KEYWORD`] on disk).
    pub fn open_with(data: &'a [u8], table: Option<std::sync::Arc<KeywordOffsets>>) -> Result<Self> {
        Self::open_with_layout(data, table, false)
    }

    /// [`KeywordReader::open_with`] for either layout ([`ColumnEntry::sparse`]).
    pub fn open_with_layout(data: &'a [u8], table: Option<std::sync::Arc<KeywordOffsets>>, layout: impl Into<OrdLayout>) -> Result<Self> {
        let layout = layout.into();
        let sparse = layout == OrdLayout::Sparse;
        if let Some(terms) = table {
            let (start, mut pos) = term_table_bounds(data, &terms)?;
            let term_bytes = data.get(start..pos).ok_or_else(|| RusticError::Corrupted("keyword term table".into()))?;
            let (num_docs, width) = read_ord_header(data, &mut pos)?;
            if layout == OrdLayout::Blocked {
                let at = pos;
                read_blocked_ords(data, &mut pos, num_docs, terms.len(), width)?;
                return Ok(Self { terms, term_bytes, num_docs, ordinals: &data[at..pos], width: 0, present: &[], ranks: &[] });
            }
            let (present, ranks, count): (&'a [u8], &'a [u8], usize) = if sparse {
                // Bounds only: ranks are validated when the segment parses
                // the column (and reads past them come back missing).
                let (r, p, words, count) = read_sparse_ords(data, &mut pos, num_docs, false)?;
                (&data[p..p + words * 8], &data[r..r + words * 4], count)
            } else {
                (&[], &[], num_docs as usize)
            };
            let ordinals = read_bytes(data, &mut pos, count * width + ord_pad(width))?;
            Ok(Self { terms, term_bytes, num_docs, ordinals, width, present, ranks })
        } else { Self::open_layout(data, layout) }
    }

    /// Open a dense keyword column ([`TAG_KEYWORD`] on disk).
    pub fn open(data: &'a [u8]) -> Result<Self> {
        Self::open_layout(data, false)
    }

    /// [`KeywordReader::open`] for either layout ([`ColumnEntry::sparse`]).
    pub fn open_layout(data: &'a [u8], layout: impl Into<OrdLayout>) -> Result<Self> {
        let t = KeywordTable::parse_layout(data, layout)?;
        let (present, ranks): (&'a [u8], &'a [u8]) = match t.sparse {
            Some((r, p, words)) => (&data[p..p + words * 8], &data[r..r + words * 4]),
            None => (&[], &[]),
        };
        Ok(Self {
            terms: t.terms,
            term_bytes: &data[t.table_start..t.table_end],
            num_docs: t.num_docs,
            ordinals: &data[t.ordinals_start..t.ordinals_end],
            width: t.width,
            present,
            ranks,
        })
    }

    /// Open against a table parsed from these same immutable column bytes.
    pub fn with_table(data: &'a [u8], table: &'a KeywordTable) -> Self {
        let (present, ranks): (&'a [u8], &'a [u8]) = match table.sparse {
            Some((r, p, words)) => (&data[p..p + words * 8], &data[r..r + words * 4]),
            None => (&[], &[]),
        };
        Self {
            terms: table.terms.clone(),
            term_bytes: &data[table.table_start..table.table_end],
            num_docs: table.num_docs,
            ordinals: &data[table.ordinals_start..table.ordinals_end],
            width: table.width,
            present,
            ranks,
        }
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Whether the term table is front-coded (a term decodes by replaying
    /// its block; see `Segment::keyword_terms_decoded`).
    pub fn terms_front_coded(&self) -> bool {
        self.terms.fc
    }

    /// Bytes of the encoded term table.
    pub fn term_table_len(&self) -> usize {
        self.term_bytes.len()
    }

    /// Term for an ordinal (borrowed from a plain table, decoded from a
    /// front-coded one).
    pub fn term(&self, ord: u32) -> Option<std::borrow::Cow<'a, str>> {
        cow_str(self.terms.term_at(self.term_bytes, ord as usize)?)
    }

    /// The UTF-8 bytes of a term, unchecked (callers that only hash them).
    #[inline]
    pub fn term_bytes(&self, ord: u32) -> Option<std::borrow::Cow<'a, [u8]>> {
        self.terms.term_at(self.term_bytes, ord as usize)
    }

    /// Every term in ordinal order, each front-coded block decoded once:
    /// `f(ord, term)` returning false stops. `None`: an unreadable table.
    pub fn for_each_term(&self, mut f: impl FnMut(u32, &[u8]) -> bool) -> Option<()> {
        self.terms.for_each_term(self.term_bytes, |ordinal, term| f(ordinal as u32, term))
    }

    /// Ordinal of a doc, or `MISSING_ORD`.
    #[inline]
    pub fn ord(&self, doc: DocId) -> u32 {
        let i = doc as usize;
        if i >= self.num_docs as usize {
            return MISSING_ORD;
        }
        if self.present.is_empty() {
            ord_at(self.ordinals, self.width, i)
        } else {
            sparse_ord_at(self.present, self.ranks, self.ordinals, self.width, i)
        }
    }

    pub fn get(&self, doc: DocId) -> Option<std::borrow::Cow<'a, str>> {
        let ord = self.ord(doc);
        if ord == MISSING_ORD {
            None
        } else {
            self.term(ord)
        }
    }

    /// Docs with a value (an upper bound for a dense column: its docs).
    pub fn present_count(&self) -> u32 {
        if self.present.is_empty() {
            return self.num_docs;
        }
        let n = self.num_docs as usize;
        let mut c = 0u32;
        for w in 0..n.div_ceil(64) {
            let Some(b) = self.present.get(w * 8..w * 8 + 8) else { break };
            let mut bits = u64::from_le_bytes(b.try_into().unwrap());
            if (w + 1) * 64 > n {
                bits &= (1u64 << (n % 64)) - 1;
            }
            c += bits.count_ones();
        }
        c
    }

    /// `f(doc, ordinal)` for every doc with a value, ascending: a sparse
    /// column walks its presence words and reads its ordinals in order (no
    /// per-doc rank lookup), a dense one checks every doc.
    pub fn for_each_present(&self, mut f: impl FnMut(DocId, u32)) {
        if self.present.is_empty() {
            for d in 0..self.num_docs {
                let o = self.ord(d);
                if o != MISSING_ORD {
                    f(d, o);
                }
            }
            return;
        }
        let n = self.num_docs as usize;
        let mut rank = 0usize;
        for w in 0..n.div_ceil(64) {
            let Some(b) = self.present.get(w * 8..w * 8 + 8) else { return };
            let mut bits = u64::from_le_bytes(b.try_into().unwrap());
            if (w + 1) * 64 > n {
                bits &= (1u64 << (n % 64)) - 1;
            }
            while bits != 0 {
                let d = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                // A corrupt column's ordinals end early: the rest reads as missing.
                if (rank + 1) * self.width + ord_pad(self.width) > self.ordinals.len() {
                    return;
                }
                let o = ord_at(self.ordinals, self.width, rank);
                rank += 1;
                if o != MISSING_ORD {
                    f(d as DocId, o);
                }
            }
        }
    }

    /// The docs with a value, as presence words (bit `d` = doc `d`): a
    /// sparse column's stored bitmap, else one ordinal check per doc. No
    /// term is decoded (an `exists` filter needs presence only).
    pub fn presence_words(&self) -> Vec<u64> {
        let n = self.num_docs as usize;
        let nwords = n.div_ceil(64);
        if !self.present.is_empty() {
            let mut words: Vec<u64> = (0..nwords)
                .map(|w| self.present.get(w * 8..w * 8 + 8).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap())))
                .collect();
            if n % 64 != 0 {
                if let Some(last) = words.last_mut() { *last &= (1u64 << (n % 64)) - 1; }
            }
            return words;
        }
        let mut words = vec![0u64; nwords];
        for i in 0..n {
            if ord_at(self.ordinals, self.width, i) != MISSING_ORD { words[i / 64] |= 1u64 << (i % 64); }
        }
        words
    }

    /// Ordinal of `term`, if present.
    pub fn lookup_ord(&self, term: &str) -> Option<u32> {
        let p = self.terms.lower_bound(self.term_bytes, term.as_bytes(), false);
        self.terms.term_at(self.term_bytes, p).is_some_and(|t| &*t == term.as_bytes()).then_some(p as u32)
    }

    /// Docs whose keyword is in `[lower, upper]` (inclusive), ascending.
    pub fn range_query(&self, lower: &str, upper: &str) -> Vec<DocId> {
        let lo = self.terms.lower_bound(self.term_bytes, lower.as_bytes(), false) as u32;
        let hi = self.terms.lower_bound(self.term_bytes, upper.as_bytes(), true) as u32;
        if lo >= hi {
            return Vec::new();
        }
        (0..self.num_docs)
            .filter(|&d| {
                let o = self.ord(d);
                o != MISSING_ORD && o >= lo && o < hi
            })
            .collect()
    }

    pub fn exact_match(&self, term: &str) -> Vec<DocId> {
        match self.lookup_ord(term) {
            Some(ord) => (0..self.num_docs).filter(|&d| self.ord(d) == ord).collect(),
            None => Vec::new(),
        }
    }

    /// Lower bound of `term` in the term table: `(first ordinal whose term
    /// is >= term, whether that term equals it)`.
    pub fn seek_ord(&self, term: &str) -> (u32, bool) {
        let p = self.terms.lower_bound(self.term_bytes, term.as_bytes(), false);
        let exact = self.terms.term_at(self.term_bytes, p).is_some_and(|t| &*t == term.as_bytes());
        (p as u32, exact)
    }

    /// Ordinal bounds of every [`BLOCK_SIZE`]-doc block (the sort skipper of
    /// a SORTED column, Lucene's `DocValuesSkipper`).
    pub fn ord_blocks(&self) -> Vec<OrdBlock> {
        let n = self.num_docs as usize;
        let mut out = Vec::with_capacity(n.div_ceil(BLOCK_SIZE));
        for start in (0..n).step_by(BLOCK_SIZE) {
            let mut b = OrdBlock { min: u32::MAX, max: 0, missing: false };
            for i in start..(start + BLOCK_SIZE).min(n) {
                let o = self.ord(i as DocId);
                if o == MISSING_ORD {
                    b.missing = true;
                } else {
                    b.min = b.min.min(o);
                    b.max = b.max.max(o);
                }
            }
            out.push(b);
        }
        out
    }

    /// The docs grouped by ordinal (a counting sort of the column).
    pub fn ord_index(&self) -> OrdIndex {
        let nt = self.terms.len();
        let mut starts = vec![0u32; nt + 1];
        for d in 0..self.num_docs {
            let o = self.ord(d);
            if o != MISSING_ORD && (o as usize) < nt {
                starts[o as usize + 1] += 1;
            }
        }
        for i in 0..nt {
            starts[i + 1] += starts[i];
        }
        let mut fill = starts.clone();
        let mut docs = vec![0u32; starts[nt] as usize];
        for d in 0..self.num_docs {
            let o = self.ord(d);
            if o != MISSING_ORD && (o as usize) < nt {
                docs[fill[o as usize] as usize] = d;
                fill[o as usize] += 1;
            }
        }
        OrdIndex { docs, starts }
    }

    /// Count of docs per ordinal (terms aggregation primitive).
    pub fn ordinal_counts(&self, accept: impl Fn(DocId) -> bool) -> Vec<u64> {
        let mut counts = vec![0u64; self.terms.len()];
        for d in 0..self.num_docs {
            let o = self.ord(d);
            if o != MISSING_ORD && accept(d) {
                counts[o as usize] += 1;
            }
        }
        counts
    }
}

/// Top bit of a multi-valued keyword column's `num_docs` word: its offsets
/// and ordinals are bit-packed (format
/// [`crate::codec::PACKED_MULTI_KEYWORD_SINCE`]+).
const PACKED_MULTI: u32 = 1 << 31;

/// Bits needed for values up to `max` (0 for 0).
#[inline]
fn width_for(max: u32) -> u8 {
    (32 - max.leading_zeros()) as u8
}

/// Bytes of `n` values packed at `bits`.
#[inline]
fn packed_bytes(n: usize, bits: u8) -> usize {
    (n * bits as usize).div_ceil(8)
}

/// LSB-first bit packer for [`PACKED_MULTI`] columns (values up to 32 bits).
struct BitPacker<'o> {
    out: &'o mut Vec<u8>,
    acc: u64,
    n: u32,
    bits: u8,
}

impl<'o> BitPacker<'o> {
    fn new(out: &'o mut Vec<u8>, bits: u8) -> Self {
        Self { out, acc: 0, n: 0, bits }
    }
    #[inline]
    fn push(&mut self, v: u32) {
        if self.bits == 0 {
            return;
        }
        self.acc |= (v as u64) << self.n;
        self.n += self.bits as u32;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// Flush the last partial byte (the next part starts byte-aligned).
    fn finish(self) {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
    }
}

/// Value `i` of values packed at `bits` from the start of `data`, which
/// extends at least 7 bytes past the packed bytes (the column's padding).
#[inline]
fn packed_at(data: &[u8], bits: u8, i: usize) -> u32 {
    if bits == 0 {
        return 0;
    }
    let bit = i * bits as usize;
    let at = bit / 8;
    let word = u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
    ((word >> (bit % 8)) & ((1u64 << bits) - 1)) as u32
}

/// Write a multi-valued keyword column's part after its term table:
/// `num_docs` docs holding `total` values of `num_terms` terms. `counts`
/// calls its argument with every doc's value count in doc order, `ords`
/// with every value's ordinal (per doc ascending) in doc order; each is
/// called once. `packed` (format [`crate::codec::PACKED_MULTI_KEYWORD_SINCE`]+)
/// writes the [`PACKED_MULTI`] layout, else four bytes per offset and ordinal.
pub(crate) fn write_multi_keyword_ords(
    body: &mut Vec<u8>,
    num_terms: usize,
    num_docs: u32,
    total: u32,
    packed: bool,
    counts: &mut dyn FnMut(&mut dyn FnMut(u32)),
    ords: &mut dyn FnMut(&mut dyn FnMut(u32)),
) {
    if !packed || num_docs >= PACKED_MULTI {
        body.extend_from_slice(&num_docs.to_le_bytes());
        body.extend_from_slice(&total.to_le_bytes());
        let mut offset = 0u32;
        body.extend_from_slice(&offset.to_le_bytes());
        counts(&mut |n| {
            offset += n;
            body.extend_from_slice(&offset.to_le_bytes());
        });
        ords(&mut |o| body.extend_from_slice(&o.to_le_bytes()));
        return;
    }
    let off_bits = width_for(total);
    let ord_bits = width_for((num_terms as u32).saturating_sub(1));
    body.extend_from_slice(&(num_docs | PACKED_MULTI).to_le_bytes());
    body.extend_from_slice(&total.to_le_bytes());
    body.push(off_bits);
    body.push(ord_bits);
    body.reserve(packed_bytes(num_docs as usize + 1, off_bits) + packed_bytes(total as usize, ord_bits) + 8);
    let mut w = BitPacker::new(body, off_bits);
    let mut offset = 0u32;
    w.push(0);
    counts(&mut |n| {
        offset += n;
        w.push(offset);
    });
    w.finish();
    let mut w = BitPacker::new(body, ord_bits);
    ords(&mut |o| w.push(o));
    w.finish();
    body.extend_from_slice(&[0u8; 8]);
}

/// A feature value (`token\0` + 8 lowercase hex digits of a weight whose
/// low 15 bits are 0) as `(token, weight bits >> 15)`, when it is exactly
/// what [`crate::fieldtypes::features::encode`] writes for them.
fn feature_parts(v: &str) -> Option<(&str, u32)> {
    let (t, hex) = v.split_once('\u{0}')?;
    if hex.len() != 8 || t.contains('\u{0}') || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return None;
    }
    let bits = u32::from_str_radix(hex, 16).ok()?;
    (bits & 0x7fff == 0).then_some((t, bits >> 15))
}

/// The [`TAG_FEATURES`] body of a multi-valued keyword column whose every
/// value is a feature value ([`feature_parts`]), else `None` (the column is
/// written as a plain multi-valued keyword column).
///
/// ```text
/// u32 tokens, [u32 end; tokens], token bytes   (sorted unique tokens)
/// u32 weights, [u32 bits >> 15; weights]       (sorted unique weights)
/// u32 num_docs, u32 total, u8 offset bits, u8 token bits, u8 weight bits,
/// offsets (num_docs + 1), token ordinals (total), weight indexes (total),
///   each bit-packed LSB-first from a byte boundary, then 8 zero bytes
/// ```
/// A doc's values are its (token, weight) pairs in value order (the order
/// of the strings they stand for).
fn encode_features(col: &crate::fast_fields::MultiKeywordColumn, num_docs: u32) -> Option<Vec<u8>> {
    let all = col.total_values() as u32;
    let off = |i: usize| col.offsets().get(i).copied().unwrap_or(all);
    let total = off(num_docs as usize) as usize;
    if num_docs == 0 || total == 0 {
        return None;
    }
    // Values are stored once in the column's string table: parse each
    // distinct one once, then give every value its distinct value's index.
    let (table, starts, lens) = (col.string_table(), col.value_offsets(), col.value_lengths());
    let mut distinct: std::collections::HashMap<u32, u32, ahash::RandomState> = Default::default();
    let mut parts: Vec<(&str, u32)> = Vec::new();
    let mut value_of: Vec<u32> = Vec::with_capacity(total);
    for k in 0..total {
        let start = starts[k];
        let id = match distinct.get(&start) {
            Some(&id) => id,
            None => {
                let v = std::str::from_utf8(table.get(start as usize..start as usize + lens[k] as usize)?).ok()?;
                parts.push(feature_parts(v)?);
                let id = parts.len() as u32 - 1;
                distinct.insert(start, id);
                id
            }
        };
        value_of.push(id);
    }
    let mut tokens: Vec<&str> = parts.iter().map(|p| p.0).collect();
    tokens.sort_unstable();
    tokens.dedup();
    let mut weights: Vec<u32> = parts.iter().map(|p| p.1).collect();
    weights.sort_unstable();
    weights.dedup();
    let tok_of: std::collections::HashMap<&str, u32, ahash::RandomState> = tokens.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
    let pair: Vec<(u32, u32)> = parts.iter().map(|&(t, w)| (tok_of[t], weights.binary_search(&w).unwrap_or(0) as u32)).collect();
    let mut body = Vec::with_capacity(tokens.len() * 12 + weights.len() * 4 + total * 4 + num_docs as usize * 4 + 64);
    write_features_tables(&mut body, &tokens, &weights)?;
    let (off_bits, tok_bits, w_bits) = (width_for(total as u32), width_for(tokens.len() as u32 - 1), width_for(weights.len() as u32 - 1));
    body.extend_from_slice(&num_docs.to_le_bytes());
    body.extend_from_slice(&(total as u32).to_le_bytes());
    body.push(off_bits);
    body.push(tok_bits);
    body.push(w_bits);
    let mut p = BitPacker::new(&mut body, off_bits);
    for i in 0..=num_docs as usize {
        p.push(off(i));
    }
    p.finish();
    let mut p = BitPacker::new(&mut body, tok_bits);
    for &v in &value_of {
        p.push(pair[v as usize].0);
    }
    p.finish();
    let mut p = BitPacker::new(&mut body, w_bits);
    for &v in &value_of {
        p.push(pair[v as usize].1);
    }
    p.finish();
    body.extend_from_slice(&[0u8; 8]);
    Some(body)
}

/// The token and weight tables of a [`TAG_FEATURES`] body.
fn write_features_tables(body: &mut Vec<u8>, tokens: &[&str], weights: &[u32]) -> Option<()> {
    body.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    let mut end = 0u32;
    for t in tokens {
        end = end.checked_add(t.len() as u32)?;
        body.extend_from_slice(&end.to_le_bytes());
    }
    for t in tokens {
        body.extend_from_slice(t.as_bytes());
    }
    body.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for w in weights {
        body.extend_from_slice(&w.to_le_bytes());
    }
    Some(())
}

/// Merge [`TAG_FEATURES`] columns: `inputs` (one per input segment, `None`
/// when it has no column) and, per input, the new id of each old doc
/// (`None`: deleted), docs taken in input order (the merged segment's doc
/// order). Tables are the union of the inputs' (both sorted, so a doc's
/// pairs keep their order). `None` when there is nothing to write.
pub fn merge_features(inputs: &[Option<FeaturesReader<'_>>], new_id: &dyn Fn(usize, DocId) -> Option<DocId>, num_docs: u32) -> Option<Vec<u8>> {
    let mut tokens: Vec<&str> = Vec::new();
    let mut weights: Vec<u32> = Vec::new();
    for r in inputs.iter().flatten() {
        for i in 0..r.num_tokens {
            tokens.push(r.token(i)?);
        }
        for i in 0..r.num_weights {
            weights.push(u32::from_le_bytes(r.weights[i * 4..i * 4 + 4].try_into().unwrap()));
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    weights.sort_unstable();
    weights.dedup();
    if tokens.is_empty() || weights.is_empty() {
        return None;
    }
    // Each input's ordinals in the merged tables (both monotonic).
    let maps: Vec<Option<(Vec<u32>, Vec<u32>)>> = inputs
        .iter()
        .map(|r| {
            r.as_ref().map(|r| {
                let t = (0..r.num_tokens).map(|i| r.token(i).and_then(|x| tokens.binary_search(&x).ok()).unwrap_or(0) as u32).collect();
                let w = (0..r.num_weights).map(|i| weights.binary_search(&u32::from_le_bytes(r.weights[i * 4..i * 4 + 4].try_into().unwrap())).unwrap_or(0) as u32).collect();
                (t, w)
            })
        })
        .collect();
    // The live docs in merged order, with their value ranges.
    let mut docs: Vec<(DocId, usize, std::ops::Range<usize>)> = Vec::new();
    for (i, r) in inputs.iter().enumerate() {
        let Some(r) = r else { continue };
        for old in 0..r.num_docs {
            if let Some(new) = new_id(i, old) {
                let range = r.range(old);
                if !range.is_empty() {
                    docs.push((new, i, range));
                }
            }
        }
    }
    docs.sort_unstable_by_key(|d| d.0);
    let total: usize = docs.iter().map(|d| d.2.len()).sum();
    if total == 0 {
        return None;
    }
    let mut body = Vec::with_capacity(tokens.len() * 12 + weights.len() * 4 + total * 4 + num_docs as usize * 4 + 64);
    write_features_tables(&mut body, &tokens, &weights)?;
    let (off_bits, tok_bits, w_bits) = (width_for(total as u32), width_for(tokens.len() as u32 - 1), width_for(weights.len() as u32 - 1));
    body.extend_from_slice(&num_docs.to_le_bytes());
    body.extend_from_slice(&(total as u32).to_le_bytes());
    body.push(off_bits);
    body.push(tok_bits);
    body.push(w_bits);
    let mut p = BitPacker::new(&mut body, off_bits);
    let mut at = 0u32;
    let mut next = docs.iter().peekable();
    for d in 0..=num_docs {
        p.push(at);
        while let Some(x) = next.next_if(|x| x.0 == d) {
            at += x.2.len() as u32;
        }
    }
    p.finish();
    for which in 0..2 {
        let bits = if which == 0 { tok_bits } else { w_bits };
        let mut p = BitPacker::new(&mut body, bits);
        for (_, i, range) in &docs {
            let (r, (tm, wm)) = (inputs[*i].as_ref()?, maps[*i].as_ref()?);
            for k in range.clone() {
                p.push(if which == 0 { tm[packed_at(r.ords, r.bits.1, k) as usize] } else { wm[packed_at(r.widx, r.bits.2, k) as usize] });
            }
        }
        p.finish();
    }
    body.extend_from_slice(&[0u8; 8]);
    Some(body)
}

/// A [`TAG_FEATURES`] column, read in place (opening checks its extent
/// only). [`FeaturesReader::values`] gives a doc's values as the strings
/// the plain multi-valued keyword column would hold.
pub struct FeaturesReader<'a> {
    ends: &'a [u8],
    tokens: &'a [u8],
    weights: &'a [u8],
    num_tokens: usize,
    num_weights: usize,
    num_docs: u32,
    total: usize,
    bits: (u8, u8, u8),
    offsets: &'a [u8],
    ords: &'a [u8],
    widx: &'a [u8],
    tables_len: usize,
}

impl<'a> FeaturesReader<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        let bad = || RusticError::Corrupted("feature column".into());
        let mut pos = 0;
        let num_tokens = read_u32_le(data, &mut pos)? as usize;
        let ends = read_bytes(data, &mut pos, num_tokens.checked_mul(4).ok_or_else(bad)?)?;
        let token_len = if num_tokens == 0 { 0 } else { u32::from_le_bytes(ends[ends.len() - 4..].try_into().unwrap()) as usize };
        let tokens = read_bytes(data, &mut pos, token_len)?;
        let num_weights = read_u32_le(data, &mut pos)? as usize;
        let weights = read_bytes(data, &mut pos, num_weights.checked_mul(4).ok_or_else(bad)?)?;
        let tables_len = pos;
        let num_docs = read_u32_le(data, &mut pos)?;
        let total = read_u32_le(data, &mut pos)? as usize;
        let b = read_bytes(data, &mut pos, 3)?;
        let bits = (b[0], b[1], b[2]);
        if bits.0 > 32 || bits.1 > 32 || bits.2 > 32 || num_tokens == 0 || num_weights == 0 {
            return Err(bad());
        }
        let offsets = &data[pos..];
        read_bytes(data, &mut pos, packed_bytes(num_docs as usize + 1, bits.0))?;
        let ords = &data[pos..];
        read_bytes(data, &mut pos, packed_bytes(total, bits.1))?;
        let widx = &data[pos..];
        read_bytes(data, &mut pos, packed_bytes(total, bits.2))?;
        // The padding every packed read relies on.
        read_bytes(data, &mut pos, 8)?;
        Ok(Self { ends, tokens, weights, num_tokens, num_weights, num_docs, total, bits, offsets, ords, widx, tables_len })
    }

    /// Bytes of the token and weight tables (disk usage).
    pub fn tables_len(&self) -> usize {
        self.tables_len
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    fn end(&self, i: usize) -> usize {
        u32::from_le_bytes(self.ends[i * 4..i * 4 + 4].try_into().unwrap()) as usize
    }

    fn token(&self, i: usize) -> Option<&'a str> {
        if i >= self.num_tokens {
            return None;
        }
        let start = if i == 0 { 0 } else { self.end(i - 1) };
        std::str::from_utf8(self.tokens.get(start..self.end(i))?).ok()
    }

    fn weight(&self, i: usize) -> Option<f32> {
        if i >= self.num_weights {
            return None;
        }
        Some(f32::from_bits(u32::from_le_bytes(self.weights[i * 4..i * 4 + 4].try_into().unwrap()) << 15))
    }

    /// The value range of `doc` (empty past the last doc).
    fn range(&self, doc: DocId) -> std::ops::Range<usize> {
        if doc >= self.num_docs {
            return 0..0;
        }
        let a = packed_at(self.offsets, self.bits.0, doc as usize) as usize;
        let b = packed_at(self.offsets, self.bits.0, doc as usize + 1) as usize;
        a.min(self.total)..b.min(self.total).max(a.min(self.total))
    }

    /// Number of values of `doc`.
    pub fn count(&self, doc: DocId) -> u32 {
        self.range(doc).len() as u32
    }

    /// The values of `doc`, as the strings they stand for, in order.
    pub fn values(&self, doc: DocId) -> Vec<String> {
        self.range(doc)
            .filter_map(|k| {
                let t = self.token(packed_at(self.ords, self.bits.1, k) as usize)?;
                let w = self.weight(packed_at(self.widx, self.bits.2, k) as usize)?;
                Some(crate::fieldtypes::features::encode(t, w))
            })
            .collect()
    }
}

/// Multi-valued keyword column (Lucene `SortedSetDocValues`): sorted unique
/// term table + per-doc sorted, deduped ordinals.
pub struct MultiKeywordReader<'a> {
    /// (offset, len) into `term_bytes` per ordinal, in sorted order
    /// (shareable: segments cache it, see [`KeywordReader::open_with`]).
    terms: std::sync::Arc<KeywordOffsets>,
    term_bytes: &'a [u8],
    num_docs: u32,
    total: usize,
    /// The offsets (packed: through the column's padding).
    offsets: &'a [u8],
    /// The ordinals (packed: through the column's padding).
    ordinals: &'a [u8],
    /// `(offset bits, ordinal bits)` of a [`PACKED_MULTI`] column.
    packed: Option<(u8, u8)>,
}

impl<'a> MultiKeywordReader<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        Self::open_with(data, None)
    }

    /// [`MultiKeywordReader::open`] reusing an already parsed term table.
    pub fn open_with(data: &'a [u8], table: Option<std::sync::Arc<KeywordOffsets>>) -> Result<Self> {
        let (terms, table_start, mut pos) = match table {
            Some(t) => {
                let (a, b) = term_table_bounds(data, &t)?;
                (t, a, b)
            }
            None => {
                let (t, a, b) = parse_term_table(data)?;
                (std::sync::Arc::new(t), a, b)
            }
        };
        if terms.fc {
            return Err(RusticError::Corrupted("front-coded multi-valued keyword table".into()));
        }
        let term_bytes = data.get(table_start..pos).ok_or_else(|| RusticError::Corrupted("keyword term table".into()))?;
        let word = read_u32_le(data, &mut pos)?;
        let total = read_u32_le(data, &mut pos)? as usize;
        if word & PACKED_MULTI != 0 {
            let num_docs = word & !PACKED_MULTI;
            let bad = || RusticError::Corrupted("packed multi keyword column".into());
            let widths = read_bytes(data, &mut pos, 2)?;
            let (off_bits, ord_bits) = (widths[0], widths[1]);
            if off_bits > 32 || ord_bits > 32 || (total > 0 && (off_bits as u32) < width_for(total as u32) as u32) {
                return Err(bad());
            }
            let off_start = pos;
            read_bytes(data, &mut pos, packed_bytes(num_docs as usize + 1, off_bits))?;
            let ord_start = pos;
            read_bytes(data, &mut pos, packed_bytes(total, ord_bits))?;
            read_bytes(data, &mut pos, 8)?;
            let reader = Self {
                terms,
                term_bytes,
                num_docs,
                total,
                offsets: &data[off_start..pos],
                ordinals: &data[ord_start..pos],
                packed: Some((off_bits, ord_bits)),
            };
            // Offsets are monotonic from 0 to `total` (O(docs): once per
            // column open, as the plain layout's slices are bounds-checked).
            if reader.off(0) != 0 || reader.off(num_docs as usize) != total {
                return Err(bad());
            }
            return Ok(reader);
        }
        let num_docs = word;
        let offsets = read_bytes(data, &mut pos, (num_docs as usize + 1) * 4)?;
        let ordinals = read_bytes(data, &mut pos, total * 4)?;
        Ok(Self {
            terms,
            term_bytes,
            num_docs,
            total,
            offsets,
            ordinals,
            packed: None,
        })
    }

    /// Ordinal number `j` of the column's flattened ordinals.
    #[inline]
    fn ord_at(&self, j: usize) -> u32 {
        match self.packed {
            Some((_, bits)) => packed_at(self.ordinals, bits, j),
            None => u32::from_le_bytes(self.ordinals[j * 4..j * 4 + 4].try_into().unwrap()),
        }
    }

    /// Whether the column uses the bit-packed layout ([`PACKED_MULTI`]).
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Total number of (doc, value) pairs.
    pub fn total_values(&self) -> usize {
        self.total
    }

    fn off(&self, i: usize) -> usize {
        match self.packed {
            Some((bits, _)) => packed_at(self.offsets, bits, i) as usize,
            None => u32::from_le_bytes(self.offsets[i * 4..i * 4 + 4].try_into().unwrap()) as usize,
        }
    }

    /// Number of distinct values of `doc`.
    #[inline]
    pub fn count(&self, doc: DocId) -> u32 {
        let i = doc as usize;
        if i >= self.num_docs as usize {
            return 0;
        }
        (self.off(i + 1) - self.off(i)) as u32
    }

    /// The ordinals of `doc`, sorted ascending (empty if the doc has none).
    pub fn ords(&self, doc: DocId) -> impl Iterator<Item = u32> + '_ {
        let i = doc as usize;
        let (a, b) = if i < self.num_docs as usize {
            (self.off(i), self.off(i + 1))
        } else {
            (0, 0)
        };
        // A corrupt packed column's offsets may run backwards or past the
        // ordinals: read none rather than out of bounds.
        let b = if a <= b && b <= self.total { b } else { a };
        (a..b).map(move |j| self.ord_at(j))
    }

    /// The values of `doc` (deduped, ascending).
    pub fn values(&self, doc: DocId) -> Vec<&'a str> {
        self.ords(doc)
            .filter_map(|o| self.term(o))
            .collect()
    }

    /// Term for an ordinal.
    pub fn term(&self, ord: u32) -> Option<&'a str> {
        let (off, len) = self.terms.get(self.term_bytes,ord as usize)?;
        std::str::from_utf8(&self.term_bytes[off..off + len]).ok()
    }

    /// Ordinal of `term`, if present.
    pub fn lookup_ord(&self, term: &str) -> Option<u32> {
        self.terms
            .binary_search_by(self.term_bytes,|&(off, len)| self.term_bytes[off..off + len].cmp(term.as_bytes()))
            .ok()
            .map(|i| i as u32)
    }

    /// Lower bound of `term` in the term table (see [`KeywordReader::seek_ord`]).
    pub fn seek_ord(&self, term: &str) -> (u32, bool) {
        let p = self.terms.partition_point(self.term_bytes,|&(off, len)| &self.term_bytes[off..off + len] < term.as_bytes());
        let exact = self.terms.get(self.term_bytes,p).is_some_and(|(off, len)| &self.term_bytes[off..off + len] == term.as_bytes());
        (p as u32, exact)
    }

    /// Docs containing `term` at least once, ascending.
    pub fn exact_match(&self, term: &str) -> Vec<DocId> {
        let Some(ord) = self.lookup_ord(term) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for d in 0..self.num_docs {
            if self.ords(d).any(|o| o == ord) {
                out.push(d);
            }
        }
        out
    }

    /// Docs with at least one value in `[lower, upper]` (inclusive),
    /// ascending, deduped.
    pub fn range_query(&self, lower: &str, upper: &str) -> Vec<DocId> {
        let lo = self
            .terms
            .partition_point(self.term_bytes,|&(off, len)| self.term_bytes[off..off + len] < *lower.as_bytes())
            as u32;
        let hi = self
            .terms
            .partition_point(self.term_bytes,|&(off, len)| self.term_bytes[off..off + len] <= *upper.as_bytes())
            as u32;
        if lo >= hi {
            return Vec::new();
        }
        let mut out = Vec::new();
        for d in 0..self.num_docs {
            if self.ords(d).any(|o| o >= lo && o < hi) {
                out.push(d);
            }
        }
        out
    }

    /// Count of docs per ordinal (terms aggregation semantics: each doc
    /// counts once per distinct ordinal it has).
    pub fn ordinal_counts(&self, accept: impl Fn(DocId) -> bool) -> Vec<u64> {
        let mut counts = vec![0u64; self.terms.len()];
        for d in 0..self.num_docs {
            if !accept(d) {
                continue;
            }
            for o in self.ords(d) {
                if let Some(c) = counts.get_mut(o as usize) {
                    *c += 1;
                }
            }
        }
        counts
    }
}

/// Encode all columns of an in-memory `FastFields` builder.
pub fn encode_fast_fields(ff: &FastFields) -> Vec<u8> {
    encode_fast_fields_n(ff, ff.num_docs())
}

/// [`encode_fast_fields`] for `num_docs` docs (a memtable's columns, whose
/// own count is not maintained while they fill).
pub fn encode_fast_fields_n(ff: &FastFields, num_docs: u32) -> Vec<u8> {
    encode_fast_fields_versioned(ff, num_docs, crate::codec::write_version())
}

/// [`encode_fast_fields_n`] as segment format `version` (the writer's
/// snapshot of [`crate::codec::write_version`], as its file header).
pub fn encode_fast_fields_versioned(ff: &FastFields, num_docs: u32, version: u32) -> Vec<u8> {
    encode_fast_fields_with_version(ff, num_docs, Vec::new(), version)
}

/// [`encode_fast_fields_n`] plus columns already encoded by the caller
/// (`(field, tag, body)`; a merge writes keyword columns by ordinal).
pub fn encode_fast_fields_with(ff: &FastFields, num_docs: u32, encoded: Vec<(u16, u8, Vec<u8>)>) -> Vec<u8> {
    encode_fast_fields_with_version(ff, num_docs, encoded, crate::codec::write_version())
}

fn encode_fast_fields_with_version(ff: &FastFields, num_docs: u32, encoded: Vec<(u16, u8, Vec<u8>)>, version: u32) -> Vec<u8> {
    let mut columns: Vec<(u16, u8, Vec<u8>)> = encoded;

    columns.extend(encode_fast_field_columns_versioned(ff, num_docs, version));
    columns.sort_by_key(|(f, tag, _)| (*f, *tag));

    // Directory first (fixed-size entries would need two passes; vints are
    // fine because we compute offsets relative to the end of the directory).
    let mut dir = Vec::new();
    write_vint(columns.len() as u64, &mut dir);
    let mut off = 0usize;
    for (field, tag, body) in &columns {
        dir.extend_from_slice(&field.to_le_bytes());
        dir.push(*tag);
        write_vint(off as u64, &mut dir);
        write_vint(body.len() as u64, &mut dir);
        off += body.len();
    }
    let mut out = Vec::with_capacity(dir.len() + off);
    out.extend_from_slice(&dir);
    for (_, _, body) in columns {
        out.extend_from_slice(&body);
    }
    out
}

/// Encode column bodies without assembling the section. Merge callers can
/// release each mutable column before constructing the next one.
pub fn encode_fast_field_columns(ff: &FastFields, num_docs: u32) -> Vec<(u16, u8, Vec<u8>)> {
    encode_fast_field_columns_versioned(ff, num_docs, crate::codec::write_version())
}

/// [`encode_fast_field_columns`] as segment format `version`.
pub fn encode_fast_field_columns_versioned(ff: &FastFields, num_docs: u32, version: u32) -> Vec<(u16, u8, Vec<u8>)> {
    let sparse_keywords = version >= crate::codec::SPARSE_KEYWORD_SINCE;
    let numeric_tables = version >= crate::codec::NUMERIC_TABLES_SINCE;
    let blocked = version >= crate::codec::BLOCK_PACKED_SINCE;
    let lists = version >= crate::codec::VALUE_LIST_POINTS_SINCE;
    let deltas = version >= crate::codec::DELTA_BLOCKS_SINCE;
    let blocked_missing = version >= crate::codec::BLOCKED_MISSING_KEYWORD_SINCE;
    let compact_v36 = version >= crate::codec::COMPACT_RUNS_SINCE;
    let two_value = version >= crate::codec::TWO_VALUE_BLOCKS_SINCE;
    let constant_columns = version >= crate::codec::CONSTANT_KEYWORD_COLUMN_SINCE;
    // Columns encode independently: in parallel (on the caller's pool, the
    // encode pool for seals and merges), in no particular order (callers
    // sort the directory).
    use rayon::prelude::*;
    let mut columns: Vec<(u16, u8, Vec<u8>)> = Vec::new();
    columns.par_extend(ff.i64_columns().par_iter().map(|(&field, col)| {
        let mut body = Vec::new();
        encode_numeric_v37(col.values(), Some(col.present_words()), num_docs, ff.has_points(field), numeric_tables, blocked, false, lists, deltas, compact_v36, two_value, &mut body);
        (field, TAG_I64, body)
    }));
    columns.par_extend(ff.u64_columns().par_iter().map(|(&field, col)| {
        let mut body = Vec::new();
        encode_numeric_v37(col.values(), Some(col.present_words()), num_docs, ff.has_points(field), numeric_tables, blocked, false, lists, deltas, compact_v36, two_value, &mut body);
        (field, TAG_U64, body)
    }));
    let decimal = version >= crate::codec::DECIMAL_FLOAT_SINCE;
    let front_coded = version >= crate::codec::FRONT_CODED_TERMS_SINCE;
    let packed_tables = version >= crate::codec::PACKED_KEYWORD_TABLES_SINCE;
    columns.par_extend(ff.f64_columns().par_iter().map(|(&field, col)| {
        let mut body = Vec::new();
        encode_numeric_v37(col.values(), Some(col.present_words()), num_docs, ff.has_points(field), numeric_tables, blocked, decimal, lists, deltas, compact_v36, two_value, &mut body);
        (field, TAG_F64, body)
    }));
    let bool_presence = version >= crate::codec::BOOL_PRESENCE_SINCE;
    for (&field, col) in ff.bool_columns() {
        let mut body = Vec::new();
        // v37: a column that knows its docs with a value, as 0 / 1 numerics.
        if let Some(present) = col.present_words().filter(|_| bool_presence) {
            let values: Vec<i64> = (0..num_docs).map(|d| col.get(d) as i64).collect();
            let words: Vec<u64> = (0..(num_docs as usize).div_ceil(64)).map(|w| present.get(w).copied().unwrap_or(0)).collect();
            encode_numeric_v37(&values, Some(&words), num_docs, false, numeric_tables, blocked, false, lists, deltas, compact_v36, two_value, &mut body);
            columns.push((field, TAG_BOOL_NUMERIC, body));
            continue;
        }
        body.extend_from_slice(&num_docs.to_le_bytes());
        let nwords = (num_docs as usize).div_ceil(64);
        let words = col.words();
        for i in 0..nwords {
            body.extend_from_slice(&words.get(i).copied().unwrap_or(0).to_le_bytes());
        }
        columns.push((field, TAG_BOOL, body));
    }
    columns.par_extend(ff.keyword_columns().par_iter().filter_map(|(&field, col)| {
        if version >= crate::codec::KEYWORD_FROM_POSTINGS_SINCE && ff.is_from_postings(field) {
            return None;
        }
        let mut body = Vec::new();
        // The distinct terms from the column's term index (sorting distinct
        // terms, not every doc's value: a field of few values over many
        // docs, the usual metrics shape, sorts a handful), then each doc's
        // ordinal by a hash lookup instead of a binary search.
        let mut terms: Vec<&str> = col.terms_with_docs().filter(|(_, docs)| docs.first().is_some_and(|&d| d < num_docs)).map(|(t, _)| t).collect();
        terms.sort_unstable();
        let ord_of: std::collections::HashMap<&str, u32, ahash::RandomState> = terms.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
        if packed_tables {
            write_term_table_packed(&mut body, terms.len(), terms.iter().map(|t| t.as_bytes()));
        } else {
            write_term_table(&mut body, terms.len(), terms.iter().map(|t| t.as_bytes()), front_coded);
        }
        let tag = write_keyword_ords_vmc(&mut body, terms.len(), num_docs, sparse_keywords, blocked, blocked_missing, constant_columns, &mut |f: &mut dyn FnMut(u32)| {
            for d in 0..num_docs {
                f(match col.get(d) {
                    Some(t) => ord_of.get(t).copied().unwrap_or(MISSING_ORD),
                    None => MISSING_ORD,
                });
            }
        });
        Some((field, tag, body))
    }));
    for (&field, col) in ff.multi_i64_columns() {
        let mut body = Vec::new();
        encode_multi_numeric(col.flat(), col.offsets(), num_docs, &mut body);
        columns.push((field, TAG_MULTI_I64, body));
    }
    for (&field, col) in ff.multi_u64_columns() {
        let mut body = Vec::new();
        encode_multi_numeric(col.flat(), col.offsets(), num_docs, &mut body);
        columns.push((field, TAG_MULTI_U64, body));
    }
    for (&field, col) in ff.multi_f64_columns() {
        let mut body = Vec::new();
        encode_multi_numeric(col.flat(), col.offsets(), num_docs, &mut body);
        columns.push((field, TAG_MULTI_F64, body));
    }
    let packed_multi = version >= crate::codec::PACKED_MULTI_KEYWORD_SINCE;
    let features = version >= crate::codec::FEATURES_COLUMN_SINCE;
    columns.par_extend(ff.multi_keyword_columns().par_iter().map(|(&field, col)| {
        if features {
            if let Some(body) = encode_features(col, num_docs) {
                return (field, TAG_FEATURES, body);
            }
        }
        let mut body = Vec::new();
        // The column's distinct values (each stored once), else every doc's.
        let mut terms: Vec<&str> = match col.distinct_values(num_docs) {
            Some(d) => d.collect(),
            None => (0..num_docs).flat_map(|d| col.values(d)).collect(),
        };
        terms.sort_unstable();
        terms.dedup();
        write_vint(terms.len() as u64, &mut body);
        for t in &terms {
            write_vint(t.len() as u64, &mut body);
            body.extend_from_slice(t.as_bytes());
        }
        let all = col.total_values() as u32;
        // Offsets as the column holds them (`all` past its last doc); the
        // values of docs `0..num_docs` (all of them, but for a memtable
        // snapshot taken while a doc was being added).
        let off = |i: usize| col.offsets().get(i).copied().unwrap_or(all);
        let total = off(num_docs as usize);
        // A doc's ordinal by a hash lookup (a metrics array of a hundred
        // values per doc over a few hundred terms: no binary search each).
        let ord_of: std::collections::HashMap<&str, u32, ahash::RandomState> = terms.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
        let num_terms = terms.len();
        write_multi_keyword_ords(
            &mut body,
            num_terms,
            num_docs,
            total,
            packed_multi,
            &mut |f: &mut dyn FnMut(u32)| {
                for i in 0..num_docs as usize {
                    f(off(i + 1) - off(i));
                }
            },
            &mut |f: &mut dyn FnMut(u32)| {
                for d in 0..num_docs {
                    for v in col.values(d) {
                        f(ord_of.get(v).copied().unwrap_or(MISSING_ORD));
                    }
                }
            },
        );
        (field, TAG_MULTI_KEYWORD, body)
    }));
    let constant_norms = version >= crate::codec::CONSTANT_NORMS_SINCE;
    for (&field, col) in ff.norms() {
        let mut body = Vec::new();
        body.extend_from_slice(&num_docs.to_le_bytes());
        body.extend_from_slice(&col.total().to_le_bytes());
        let first = col.get_byte(0);
        if constant_norms && num_docs > 0 && (1..num_docs).all(|d| col.get_byte(d) == first) {
            body.push(first);
            columns.push((field, TAG_CONST_NORMS, body));
            continue;
        }
        for d in 0..num_docs {
            body.push(col.get_byte(d));
        }
        columns.push((field, TAG_NORMS, body));
    }
    columns
}

/// One entry of the fast-field column directory: `(field, tag)` and the
/// byte range of the column body inside the fast-field section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnEntry {
    pub field: u16,
    /// The column kind; an on-disk [`TAG_SPARSE_KEYWORD`] reads as
    /// [`TAG_KEYWORD`] with `sparse` set.
    pub tag: u8,
    pub start: usize,
    pub end: usize,
    /// A keyword column in the sparse layout.
    pub sparse: bool,
    /// A keyword column whose ordinals are packed per block.
    pub blocked: bool,
}

impl ColumnEntry {
    /// The ordinal layout of a keyword column.
    pub fn layout(&self) -> OrdLayout {
        if self.blocked { OrdLayout::Blocked } else { OrdLayout::from(self.sparse) }
    }
}

/// Parse the column directory of a fast-field section (what
/// [`FastFieldsReader::open`] does on every call; segments parse it once
/// and reuse it through [`FastFieldsReader::with_directory`]).
pub fn parse_column_directory(data: &[u8]) -> Result<Vec<ColumnEntry>> {
    let mut pos = 0;
    let n = read_vint(data, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let field = u16::from_le_bytes(read_bytes(data, &mut pos, 2)?.try_into().unwrap());
        let tag = read_bytes(data, &mut pos, 1)?[0];
        let off = read_vint(data, &mut pos)? as usize;
        let len = read_vint(data, &mut pos)? as usize;
        entries.push((field, tag, off, len));
    }
    let base = pos;
    let mut out = Vec::with_capacity(entries.len());
    for (field, tag, off, len) in entries {
        let start = base + off;
        let end = start + len;
        if end > data.len() {
            return Err(RusticError::Corrupted("fast field column out of range".into()));
        }
        let sparse = tag == TAG_SPARSE_KEYWORD;
        let blocked = tag == TAG_BLOCK_KEYWORD;
        out.push(ColumnEntry { field, tag: if sparse || blocked { TAG_KEYWORD } else { tag }, start, end, sparse, blocked });
    }
    // By field, so a column is found by binary search ([`FastFieldsReader::entry`]):
    // a synthetic source probes each of a segment's hundred-odd columns,
    // several kinds each, per document.
    out.sort_by_key(|e| (e.field, e.tag));
    Ok(out)
}

/// Parts of one encoded column, for disk-usage reporting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColumnParts {
    /// A single-valued numeric column's points index (docs sorted by value).
    pub points: u64,
    /// A keyword column's sorted term table.
    pub term_table: u64,
    /// A norms column.
    pub norms: u64,
}

/// Split one column (`tag`, its bytes) into [`ColumnParts`]; the rest of
/// the column is its values / ordinals / presence data.
pub fn column_parts(tag: u8, data: &[u8]) -> Result<ColumnParts> {
    let mut parts = ColumnParts::default();
    match tag {
        TAG_I64 | TAG_U64 | TAG_F64 => {
            // Docs sorted by value, or per-value doc lists ([`VALUE_LISTS`]).
            let c = ColumnReader::<u64>::open(data)?;
            parts.points = (c.sorted.len() + c.vlists.map_or(0, |v| v.byte_len())) as u64;
        }
        TAG_KEYWORD | TAG_MULTI_KEYWORD => {
            let (_, start, end) = parse_term_table(data)?;
            parts.term_table = (end - start) as u64 + start as u64;
        }
        TAG_NORMS | TAG_CONST_NORMS => parts.norms = data.len() as u64,
        TAG_FEATURES => parts.term_table = FeaturesReader::open(data)?.tables_len() as u64,
        _ => {}
    }
    Ok(parts)
}

/// All fast-field columns of a segment, resolved lazily from the mapped bytes.
pub struct FastFieldsReader<'a> {
    data: &'a [u8],
    columns: std::borrow::Cow<'a, [ColumnEntry]>,
    keyword_tables: Option<&'a std::collections::HashMap<u16, std::sync::OnceLock<Option<KeywordTable>>>>,
    /// Segment-owned: the single-valued numeric columns already opened with
    /// their presence ranks validated ([`ColumnReader::open_checked`]).
    numeric_checks: Option<&'a NumericChecks>,
}

/// Per segment, one cell per single-valued numeric column `(field, tag)`,
/// set once the column's presence ranks were validated.
pub type NumericChecks = std::collections::HashMap<(u16, u8), std::sync::OnceLock<()>>;

/// The [`NumericChecks`] cells of a parsed column directory.
pub fn numeric_checks(columns: &[ColumnEntry]) -> NumericChecks {
    columns.iter().filter(|c| matches!(c.tag, TAG_I64 | TAG_U64 | TAG_F64 | TAG_BOOL_NUMERIC)).map(|c| ((c.field, c.tag), std::sync::OnceLock::new())).collect()
}

impl<'a> FastFieldsReader<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self> {
        Ok(Self { data, columns: std::borrow::Cow::Owned(parse_column_directory(data)?), keyword_tables: None, numeric_checks: None })
    }

    /// A reader over `data` with an already parsed directory (no parsing,
    /// no allocation).
    pub fn with_directory(data: &'a [u8], columns: &'a [ColumnEntry]) -> Self {
        Self { data, columns: std::borrow::Cow::Borrowed(columns), keyword_tables: None, numeric_checks: None }
    }

    /// Segment-owned parsed keyword tables, initialized lazily once per column.
    pub fn with_keyword_tables(data: &'a [u8], columns: &'a [ColumnEntry], keyword_tables: &'a std::collections::HashMap<u16, std::sync::OnceLock<Option<KeywordTable>>>) -> Self {
        Self { data, columns: std::borrow::Cow::Borrowed(columns), keyword_tables: Some(keyword_tables), numeric_checks: None }
    }

    /// [`Self::with_keyword_tables`] that also validates each numeric
    /// column's presence ranks once per segment (`numeric_checks`) instead of
    /// on every open: SQL and ES|QL column reads open a column per batch.
    pub fn with_segment_tables(data: &'a [u8], columns: &'a [ColumnEntry], keyword_tables: &'a std::collections::HashMap<u16, std::sync::OnceLock<Option<KeywordTable>>>, numeric_checks: &'a NumericChecks) -> Self {
        Self { data, columns: std::borrow::Cow::Borrowed(columns), keyword_tables: Some(keyword_tables), numeric_checks: Some(numeric_checks) }
    }

    pub fn empty() -> Self {
        Self { data: &[], columns: std::borrow::Cow::Borrowed(&[]), keyword_tables: None, numeric_checks: None }
    }

    #[inline]
    /// The directory entry of `field`'s `tag` column (every directory comes
    /// sorted by field from [`parse_column_directory`]).
    fn entry(&self, field: u16, tag: u8) -> Option<&ColumnEntry> {
        let cols: &[ColumnEntry] = &self.columns;
        let at = cols.partition_point(|e| e.field < field);
        cols[at..].iter().take_while(|e| e.field == field).find(|e| e.tag == tag)
    }

    fn column(&self, field: u16, tag: u8) -> Option<&'a [u8]> {
        let data: &'a [u8] = self.data;
        self.entry(field, tag).map(|e| &data[e.start..e.end])
    }

    /// A single-valued numeric column, its presence ranks validated on the
    /// segment's first open only (see [`Self::with_segment_tables`]).
    fn numeric<T: FastValue>(&self, field: u16, tag: u8) -> Option<ColumnReader<'a, T>> {
        let bytes = self.column(field, tag)?;
        match self.numeric_checks.and_then(|checks| checks.get(&(field, tag))) {
            Some(cell) if cell.get().is_some() => ColumnReader::<T>::open_checked(bytes, false).ok(),
            Some(cell) => {
                let reader = ColumnReader::<T>::open(bytes).ok()?;
                let _ = cell.set(());
                Some(reader)
            }
            None => ColumnReader::<T>::open(bytes).ok(),
        }
    }

    pub fn get_i64(&self, field: u16) -> Option<ColumnReader<'a, i64>> {
        self.numeric(field, TAG_I64)
    }

    pub fn get_u64(&self, field: u16) -> Option<ColumnReader<'a, u64>> {
        self.numeric(field, TAG_U64)
    }

    pub fn get_f64(&self, field: u16) -> Option<ColumnReader<'a, f64>> {
        self.numeric(field, TAG_F64)
    }

    pub fn get_bool(&self, field: u16) -> Option<BoolReader<'a>> {
        match self.column(field, TAG_BOOL) {
            Some(b) => BoolReader::open(b).ok(),
            // Validated once per segment, as numeric columns are (a hit's
            // synthetic source opens its booleans per document).
            None => self.numeric::<i64>(field, TAG_BOOL_NUMERIC).map(BoolReader::from_numeric),
        }
    }

    /// Raw bytes of the single-valued keyword column of `field`.
    pub fn keyword_column_bytes(&self, field: u16) -> Option<&'a [u8]> {
        self.column(field, TAG_KEYWORD)
    }

    /// Raw bytes of the multi-valued keyword column of `field`.
    pub fn multi_keyword_column_bytes(&self, field: u16) -> Option<&'a [u8]> {
        self.column(field, TAG_MULTI_KEYWORD)
    }

    pub fn get_keyword(&self, field: u16) -> Option<KeywordReader<'a>> {
        let (bytes, sparse) = self.keyword_column(field)?;
        if let Some(cell) = self.keyword_tables.and_then(|tables| tables.get(&field)) {
            return cell.get_or_init(|| KeywordTable::parse_layout(bytes, sparse).ok()).as_ref().map(|table| KeywordReader::with_table(bytes, table));
        }
        KeywordReader::open_layout(bytes, sparse).ok()
    }

    /// Raw bytes of the single-valued keyword column of `field` and whether
    /// it is in the sparse layout.
    pub fn keyword_column(&self, field: u16) -> Option<(&'a [u8], OrdLayout)> {
        let data: &'a [u8] = self.data;
        self.entry(field, TAG_KEYWORD).map(|e| (&data[e.start..e.end], e.layout()))
    }

    pub fn get_norms(&self, field: u16) -> Option<NormsReader<'a>> {
        match self.column(field, TAG_NORMS) {
            Some(b) => NormsReader::open(b).ok(),
            None => self.column(field, TAG_CONST_NORMS).and_then(|b| NormsReader::open_constant(b).ok()),
        }
    }

    pub fn get_multi_i64(&self, field: u16) -> Option<MultiColumnReader<'a, i64>> {
        self.column(field, TAG_MULTI_I64).and_then(|b| MultiColumnReader::open(b).ok())
    }

    pub fn get_multi_u64(&self, field: u16) -> Option<MultiColumnReader<'a, u64>> {
        self.column(field, TAG_MULTI_U64).and_then(|b| MultiColumnReader::open(b).ok())
    }

    pub fn get_multi_f64(&self, field: u16) -> Option<MultiColumnReader<'a, f64>> {
        self.column(field, TAG_MULTI_F64).and_then(|b| MultiColumnReader::open(b).ok())
    }

    pub fn get_multi_keyword(&self, field: u16) -> Option<MultiKeywordReader<'a>> {
        self.column(field, TAG_MULTI_KEYWORD).and_then(|b| MultiKeywordReader::open(b).ok())
    }

    /// The feature-values column of `field` ([`TAG_FEATURES`]).
    pub fn get_features(&self, field: u16) -> Option<FeaturesReader<'a>> {
        self.column(field, TAG_FEATURES).and_then(|b| FeaturesReader::open(b).ok())
    }

    /// Number of encoded columns, without allocating their listing.
    pub fn field_count(&self) -> usize { self.columns.len() }

    /// Field ids present, with their tag.
    /// Every column's field and kind (a [`TAG_BOOL_NUMERIC`] column is a
    /// [`TAG_BOOL`] one: [`Self::get_bool`] reads either), sorted.
    pub fn fields(&self) -> Vec<(u16, u8)> {
        let mut v: Vec<_> = self.columns.iter().map(|e| (e.field, if e.tag == TAG_BOOL_NUMERIC { TAG_BOOL } else { e.tag })).collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    /// [`RangeDocs`] yields exactly [`ColumnReader::range_bits`]'s docs,
    /// by `advance` and by `seek`: dense and sparse columns, ascending
    /// (whole blocks in range) and random values, empty and full ranges.
    #[test]
    fn lazy_range_docs_match_range_bits() {
        use crate::query::docset::{DocSet, TERMINATED};
        let n = 5000usize;
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        for shape in 0..4 {
            let vals: Vec<i64> = (0..n).map(|i| match shape { 0 | 2 => (i as i64) * 3 - 700, 3 => 42, _ => (rnd() % 2000) as i64 - 1000 }).collect();
            let present: Vec<u64> = if shape == 2 || shape == 1 {
                let mut p = vec![0u64; n.div_ceil(64)];
                for i in 0..n { if rnd() % 5 != 0 { p[i / 64] |= 1 << (i % 64); } }
                p
            } else { vec![u64::MAX; n.div_ceil(64)] };
            for points in [false, true] {
                let mut b = Vec::new();
                encode_numeric_vbdld(&vals, Some(&present), n as u32, points, true, true, false, false, true, &mut b);
                for (lo, hi) in [(-1000, 1000), (-50, 60), (0, 0), (42, 42), (5000, 6000), (i64::MIN, i64::MAX), (100, 99), (-700, 2000)] {
                    let c = ColumnReader::<i64>::open(&b).unwrap();
                    let mut words = vec![0u64; n.div_ceil(64)];
                    c.range_bits(lo, hi, &mut words);
                    let want: Vec<DocId> = (0..n as DocId).filter(|&d| words[d as usize / 64] >> (d % 64) & 1 == 1).collect();
                    let Some(mut docs) = ColumnReader::<i64>::open(&b).unwrap().lazy_range(lo, hi) else {
                        assert!(lo > hi || ColumnReader::<i64>::open(&b).unwrap().compact, "shape {shape} {lo}..{hi}");
                        continue;
                    };
                    assert_eq!(docs.doc(), TERMINATED);
                    let mut got = Vec::new();
                    loop { let d = docs.advance(); if d == TERMINATED { break; } got.push(d); }
                    assert_eq!(got, want, "shape {shape} points {points} {lo}..{hi}");
                    assert_eq!(docs.advance(), TERMINATED);
                    // Seeks: every 7th target.
                    let mut docs = ColumnReader::<i64>::open(&b).unwrap().lazy_range(lo, hi).unwrap();
                    for t in (0..n as DocId + 3).step_by(7) {
                        let expect = want.iter().copied().find(|&d| d >= t.max(if docs.doc() == TERMINATED { 0 } else { docs.doc() })).unwrap_or(TERMINATED);
                        let d = docs.seek(t);
                        assert_eq!(d, expect, "shape {shape} seek {t} {lo}..{hi}");
                        if d == TERMINATED { break; }
                    }
                    // Exact seeks of ascending candidates, then the rest by
                    // advance; and its size: exact from a points index that
                    // holds exactly the docs with a value, else a bound.
                    let mut docs = ColumnReader::<i64>::open(&b).unwrap().lazy_range(lo, hi).unwrap();
                    assert!(docs.has_fast_exact_seek());
                    if ColumnReader::<i64>::open(&b).unwrap().counts_from_points() {
                        assert_eq!(docs.size_hint() as usize, want.len(), "shape {shape} {lo}..{hi}");
                    } else {
                        assert!(docs.size_hint() as usize >= want.len(), "shape {shape} {lo}..{hi}");
                    }
                    let mut last = 0;
                    for t in (0..n as DocId).step_by(5) {
                        assert_eq!(docs.seek_exact(t), want.binary_search(&t).is_ok(), "shape {shape} exact {t} {lo}..{hi}");
                        last = t;
                    }
                    let rest: Vec<DocId> = std::iter::from_fn(|| Some(docs.advance()).filter(|&d| d != TERMINATED)).collect();
                    assert_eq!(rest, want.iter().copied().filter(|&d| d > last).collect::<Vec<_>>(), "shape {shape} after exact {lo}..{hi}");
                }
            }
        }
    }

    use super::*;

    #[test]
    fn keyword_ordinals_use_the_narrowest_width() {
        // (distinct terms, expected width): the all-ones value stays free for
        // missing docs, so 255 terms already need 2 bytes.
        for (terms, width) in [(1usize, 1usize), (254, 1), (255, 2), (65534, 2), (65535, 3)] {
            let n = (terms + terms / 3 + 1) as u32;
            let mut fields = FastFields::new(n);
            let term = |doc: u32| format!("t{:06}", (doc - doc / 4) as usize % terms);
            for doc in 0..n {
                if doc % 4 != 3 {
                    fields.keyword_column_mut(1).set(doc, &term(doc));
                }
            }
            fields.build_all();
            let bytes = encode_fast_fields(&fields);
            let r = FastFieldsReader::open(&bytes).unwrap();
            let body = r.column(1, TAG_KEYWORD).unwrap();
            let k = r.get_keyword(1).unwrap();
            assert_eq!(k.num_terms(), terms);
            assert_eq!(k.width, width, "{terms} terms");
            assert!(body.len() < n as usize * width + 16 + terms * 8);
            let blocks = k.ord_blocks();
            for doc in 0..n {
                let expected = (doc % 4 != 3).then(|| term(doc));
                assert_eq!(k.get(doc).map(|v| v.into_owned()), expected, "{terms} terms doc {doc}");
                let b = blocks[doc as usize / BLOCK_SIZE];
                match k.ord(doc) {
                    MISSING_ORD => assert!(b.missing),
                    o => assert!(b.min <= o && o <= b.max),
                }
            }
            assert_eq!(k.ord(n), MISSING_ORD);
        }
    }

    fn check_packed<T: FastValue + std::fmt::Debug>(values: &[Option<T>], points: bool, widths: (usize, usize), probes: &[(T, T)]) {
        check_packing(values, points, false, widths, probes);
    }

    /// [`check_packed`] with v14 table packing allowed; returns whether the
    /// column was table-packed.
    fn check_packing<T: FastValue + std::fmt::Debug>(values: &[Option<T>], points: bool, tables: bool, widths: (usize, usize), probes: &[(T, T)]) -> bool {
        let n = values.len();
        let mut present = vec![0u64; n.div_ceil(64)];
        let dense: Vec<T> = values.iter().enumerate().map(|(i, v)| {
            if v.is_some() { present[i / 64] |= 1 << (i % 64); }
            v.unwrap_or_else(T::zero)
        }).collect();
        let mut bytes = Vec::new();
        encode_numeric_v(&dense, Some(&present), n as u32, points, tables, &mut bytes);
        let c = ColumnReader::<T>::open(&bytes).unwrap();
        assert_eq!((c.pack.value_width, c.pack.point_width), widths, "{values:?}");
        for (d, v) in values.iter().enumerate() {
            assert_eq!(c.get(d as DocId).map(|x| x.to_le()), v.map(|x| x.to_le()), "doc {d}");
        }
        for &(lo, hi) in probes {
            let expected: Vec<usize> = (0..n).filter(|&d| values[d].is_some_and(|v| v >= lo && v <= hi)).collect();
            let mut words = vec![0u64; n.div_ceil(64)];
            c.range_bits(lo, hi, &mut words);
            let got: Vec<usize> = (0..n).filter(|&d| words[d / 64] >> (d % 64) & 1 == 1).collect();
            assert_eq!(got, expected, "range {lo:?}..={hi:?}");
            assert_eq!(c.count_range(lo, hi), expected.len() as u64);
        }
        c.pack.table
    }

    /// v14 table packing: low-cardinality floats and day-granular dates take
    /// table indexes and read back bit for bit (signed zeros, NaN, sparse
    /// columns, ranges through the points index and block bounds); a column
    /// with more distinct values than the table holds keeps its packing.
    #[test]
    fn numeric_columns_pack_through_a_table_of_distinct_values() {
        // Temperatures in tenths: ~1200 distinct floats -> 2-byte indexes.
        let t: Vec<Option<f64>> = (0..20_000).map(|i| Some(((i * 37 % 1201) as f64 - 600.0) / 10.0)).collect();
        assert!(check_packing(&t, true, true, (2, 3), &[(-5.0, 5.0), (-60.0, -59.9), (59.9, 61.0), (100.0, 200.0)]));
        assert!(!check_packing(&t, true, false, (8, 3), &[(-5.0, 5.0)]), "before v14 floats stay plain");
        // Sparse with signed zeros and a NaN: 1-byte indexes, exact bits.
        let z: Vec<Option<f64>> = (0..5000).map(|i| match i % 50 { 0 => Some(-0.0), 1 => Some(0.0), 2 => Some(f64::NAN), 3 => Some(1.5), _ => None }).collect();
        assert!(check_packing(&z, true, true, (1, 3), &[(0.0, 0.0), (-1.0, 2.0), (1.5, 1.5)]));
        let col = { let mut present = vec![0u64; 5000usize.div_ceil(64)]; let dense: Vec<f64> = z.iter().enumerate().map(|(i, v)| { if v.is_some() { present[i / 64] |= 1 << (i % 64); } v.unwrap_or(0.0) }).collect(); let mut b = Vec::new(); encode_numeric_v(&dense, Some(&present), 5000, true, true, &mut b); b };
        let c = ColumnReader::<f64>::open(&col).unwrap();
        assert_eq!(c.get(0).map(f64::to_bits), Some((-0.0f64).to_bits()));
        assert_eq!(c.get(1).map(f64::to_bits), Some(0.0f64.to_bits()));
        assert!(c.get(2).is_some_and(f64::is_nan));
        // Day-granular millis over ~3 years: 5-byte offsets -> 2-byte indexes.
        let d: Vec<Option<i64>> = (0..30_000).map(|i| Some(1_451_606_400_000 + (i * 7919 % 1100) as i64 * 86_400_000)).collect();
        assert!(check_packing(&d, true, true, (2, 3), &[(1_451_606_400_000, 1_451_606_400_000 + 86_400_000 * 10), (0, 1)]));
        // Dense small-range integers keep their 1-2 byte offsets.
        let s: Vec<Option<i64>> = (0..3000).map(|i| Some(i * 7 % 2001 - 1000)).collect();
        assert!(!check_packing(&s, true, true, (2, 3), &[(-5, 5)]));
        // More distinct values than a table holds: plain 8-byte floats.
        let f: Vec<Option<f64>> = (0..70_000).map(|i| Some((i as f64).sin())).collect();
        assert!(!check_packing(&f, true, true, (8, 3), &[(-0.5, 0.5)]));
    }

    #[test]
    fn numeric_columns_pack_to_the_value_range() {
        // Signed range crossing zero: 2 bytes; dense with points.
        let v: Vec<Option<i64>> = (0..3000).map(|i| Some(i * 7 % 2001 - 1000)).collect();
        check_packed(&v, true, (2, 3), &[(-1000, 1000), (-5, 5), (999, 2000), (-3000, -1001)]);
        // Constant columns take no value bytes, floats included.
        check_packed(&vec![Some(1i64); 700], true, (0, 3), &[(1, 1), (0, 0), (2, 9)]);
        check_packed(&vec![Some(2.5f64); 700], false, (0, 3), &[(2.5, 2.5), (0.0, 1.0)]);
        // Varied floats keep 8-byte values; docs still narrow.
        let f: Vec<Option<f64>> = (0..900).map(|i| Some((i as f64).sin())).collect();
        check_packed(&f, true, (8, 3), &[(-0.5, 0.5), (0.99, 1.0)]);
        // Full u64 range: plain values.
        check_packed(&[Some(0u64), Some(u64::MAX), Some(7)], true, (8, 3), &[(0, 7), (8, u64::MAX)]);
        // Mostly-present sparse column (padding zeros written): the range
        // covers the padding, so 0 must stay representable.
        let s: Vec<Option<u64>> = (0..1000).map(|i| (i != 3).then_some(5000 + i)).collect();
        check_packed(&s, true, (2, 3), &[(5000, 5500), (0, 10), (5999, 6000)]);
        // Very sparse (compact) column: only present values are packed.
        let c: Vec<Option<i64>> = (0..5000).map(|i| (i % 100 == 0).then_some(-(i as i64))).collect();
        check_packed(&c, true, (2, 3), &[(-4900, -100), (-50, 0), (1, 5)]);
        let b: Vec<Option<i64>> = (0..5000).map(|i| (i % 100 == 0).then_some(i as i64 % 200)).collect();
        check_packed(&b, true, (1, 3), &[(0, 0), (100, 100), (1, 99)]);
    }

    /// From v14 a keyword column that is cheaper with a presence index is
    /// written sparse; it reads exactly as the dense column does (values,
    /// missing docs, block bounds, ordinal index, owned column), and is
    /// written dense before v14 or when most docs have a value.
    /// v28 block packing: dense integer columns (runs, a GCD, negatives,
    /// constant blocks) read back exactly, with and without points, and
    /// range counts match a scan; columns it would not shrink, sparse and
    /// float columns keep their layout.
    /// v30 front-coded keyword term tables: every ordinal, lookup, seek,
    /// range query and doc value reads as from the plain table, across block
    /// boundaries (an empty term, shared prefixes, a lone last block), with
    /// and without the block index; the table shrinks; v29 writes plain.
    #[test]
    fn front_coded_keyword_tables_read_like_plain_ones() {
        for n in [1usize, 15, 16, 17, 33, 1000] {
            let mut terms: Vec<String> = (0..n).map(|i| if i == 0 { String::new() } else { format!("https://10.{}.{}.{}", i % 7, i / 7 % 250, i % 251) }).collect();
            terms.sort();
            terms.dedup();
            let mut plain = Vec::new();
            write_term_table(&mut plain, terms.len(), terms.iter().map(|t| t.as_bytes()), false);
            let mut fc = Vec::new();
            write_term_table(&mut fc, terms.len(), terms.iter().map(|t| t.as_bytes()), true);
            if terms.len() > 16 {
                assert!(fc.len() * 10 < plain.len() * 8, "n={n}: {} vs {}", fc.len(), plain.len());
            }
            let (p_off, p_start, _) = parse_term_table(&plain).unwrap();
            let (f_off, f_start, f_end) = parse_term_table(&fc).unwrap();
            assert!(f_off.fc && !p_off.fc);
            let fbytes = &fc[f_start..f_end];
            let pbytes = &plain[p_start..];
            let unindexed = KeywordOffsets { indexed: None, count: f_off.count, encoded_len: f_off.encoded_len, fc: true, packed: false, heads: std::sync::OnceLock::new() };
            for (o, t) in terms.iter().enumerate() {
                assert_eq!(f_off.term_at(fbytes, o).as_deref(), Some(t.as_bytes()), "n={n} ord {o}");
                assert_eq!(unindexed.term_at(fbytes, o).as_deref(), Some(t.as_bytes()), "unindexed n={n} ord {o}");
            }
            assert!(f_off.term_at(fbytes, terms.len()).is_none());
            let mut keys: Vec<String> = terms.clone();
            keys.extend(["", "a", "https://10.3", "https://10.3.5.", "zzz", "https://10.6.249.250~"].map(String::from));
            for k in &keys {
                for inclusive in [false, true] {
                    assert_eq!(f_off.lower_bound(fbytes, k.as_bytes(), inclusive), p_off.lower_bound(pbytes, k.as_bytes(), inclusive), "n={n} {k:?} {inclusive}");
                }
            }
        }
        // A column: v30 front-coded, v29 plain, the same values and lookups.
        let n = 3000u32;
        let mut ff = crate::fast_fields::FastFields::new(n);
        for d in 0..n {
            if d % 11 != 0 {
                ff.keyword_column_mut(5).set(d, &format!("pod-ip-10.{}.{}.{}", d % 3, d / 3 % 200, d % 97));
            }
        }
        let body = |version| encode_fast_field_columns_versioned(&ff, n, version).into_iter().find(|(f, _, _)| *f == 5).unwrap();
        let (_, tag30, b30) = body(30);
        let (_, tag29, b29) = body(29);
        assert!(b30.len() < b29.len(), "{} vs {}", b30.len(), b29.len());
        let r30 = KeywordReader::open_layout(&b30, OrdLayout::from_tag(tag30)).unwrap();
        let r29 = KeywordReader::open_layout(&b29, OrdLayout::from_tag(tag29)).unwrap();
        for d in 0..n {
            assert_eq!(r30.get(d), r29.get(d), "doc {d}");
        }
        for t in ["pod-ip-10.1.7.22", "pod-ip-10.2.199.50", "nope", ""] {
            assert_eq!(r30.lookup_ord(t), r29.lookup_ord(t), "{t}");
            assert_eq!(r30.seek_ord(t), r29.seek_ord(t), "{t}");
        }
        assert_eq!(r30.range_query("pod-ip-10.1", "pod-ip-10.2"), r29.range_query("pod-ip-10.1", "pod-ip-10.2"));
    }

    /// v34 delta blocks: index-sorted timestamps (descending seconds in
    /// milliseconds), a regular counter (constant deltas, zero-width
    /// blocks), runs that only some blocks delta-pack, large offsets and a
    /// partial last block read back exactly through `get`, block values,
    /// ranges and min / max; timestamps take far fewer bytes, random
    /// values none more.
    #[test]
    fn delta_blocks_read_back_exactly() {
        fn check<T: FastValue + std::fmt::Debug>(name: &str, vals: &[T], probes: &[(T, T)], expect_delta: bool) -> (usize, usize) {
            let n = vals.len();
            let present = vec![u64::MAX; n.div_ceil(64)];
            let (mut a, mut b) = (Vec::new(), Vec::new());
            encode_numeric_vbdld(vals, Some(&present), n as u32, false, true, true, false, false, false, &mut a);
            encode_numeric_vbdld(vals, Some(&present), n as u32, false, true, true, false, false, true, &mut b);
            let (x, y) = (ColumnReader::<T>::open(&a).unwrap(), ColumnReader::<T>::open(&b).unwrap());
            assert_eq!(!y.deltas.is_empty(), expect_delta, "{name}");
            for d in 0..n as u32 {
                assert_eq!(y.get(d).map(|v| v.to_le()), Some(vals[d as usize].to_le()), "{name} doc {d}");
            }
            let mut buf = vec![T::zero(); BLOCK_SIZE];
            for blk in 0..n.div_ceil(BLOCK_SIZE) {
                let m = y.block_values(blk, &mut buf).unwrap();
                for i in 0..m {
                    assert_eq!(buf[i].to_le(), vals[blk * BLOCK_SIZE + i].to_le(), "{name} block {blk} entry {i}");
                }
            }
            for &(lo, hi) in probes {
                assert_eq!(x.count_range(lo, hi), y.count_range(lo, hi), "{name} [{lo:?}, {hi:?}]");
                let (mut ra, mut rb) = (vec![0u64; n.div_ceil(64)], vec![0u64; n.div_ceil(64)]);
                x.range_bits(lo, hi, &mut ra);
                y.range_bits(lo, hi, &mut rb);
                assert_eq!(ra, rb, "{name} range [{lo:?}, {hi:?}]");
            }
            (a.len(), b.len())
        }
        let n = 70_000usize;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m };
        let mut ts = 894_366_188_000i64;
        let stamps: Vec<i64> = (0..n).map(|_| { if rnd(4) == 0 { ts -= 1000 * (1 + rnd(3) as i64 * (rnd(50) == 0) as i64); } ts }).collect();
        let (plain, delta) = check("timestamps", &stamps, &[(stamps[n - 1], stamps[0]), (stamps[500], stamps[400]), (stamps[n / 2] - 1, stamps[n / 2] + 1), (0, 1)], true);
        assert!(delta * 2 < plain, "timestamps: {delta} vs {plain}");
        let counter: Vec<i64> = (0..n as i64 - 77).map(|i| -5_000 + 3 * i).collect();
        check("counter", &counter, &[(-5_000, 0), (100, 100), (101, 101), (i64::MIN, i64::MAX)], true);
        // Runs: half the blocks ascending by small steps, half random.
        let mixed: Vec<i64> = (0..n).map(|i| if (i / BLOCK_SIZE) % 2 == 0 { 1_000_000 + (i as i64) * 2 + rnd(2) as i64 } else { rnd(1 << 20) as i64 }).collect();
        check("mixed", &mixed, &[(0, 1 << 19), (1_000_000, 1_050_000)], true);
        let big: Vec<u64> = (0..n).map(|i| (1u64 << 50) + (i as u64 / 3) * 7).collect();
        check("u64", &big, &[(1 << 50, (1 << 50) + 1000), (0, u64::MAX)], true);
        let random: Vec<i64> = (0..n).map(|_| rnd(1 << 30) as i64).collect();
        let (p, d) = check("random", &random, &[(0, 1 << 29)], false);
        assert_eq!(p, d);
        // A first offset past u32 keeps a block plain (here every block).
        let wide: Vec<i64> = (0..n).map(|i| if i % BLOCK_SIZE == 0 { 1i64 << 40 } else { (i as i64 % BLOCK_SIZE as i64) * 5 }).collect();
        check("wide", &wide, &[(0, 100), (1 << 40, 1 << 40)], false);
        // Only some blocks: wide blocks plain, the others deltas.
        let some: Vec<i64> = (0..n).map(|i| if (i / BLOCK_SIZE) % 3 == 0 && i % BLOCK_SIZE == 0 { 1i64 << 40 } else { 7 * i as i64 }).collect();
        check("some", &some, &[(0, 100), (1 << 40, 1 << 40), (7 * 300, 7 * 9000)], true);
        // The thread's block cache: six readers read interleaved (more than
        // its slots), in doc order and in random order, and from threads.
        let cols: Vec<Vec<i64>> = (0..6).map(|k| (0..n).map(|i| stamps[i] + k * 7_000).collect()).collect();
        let bytes: Vec<Vec<u8>> = cols.iter().map(|v| {
            let mut b = Vec::new();
            encode_numeric_vbdld(v, Some(&vec![u64::MAX; n.div_ceil(64)]), n as u32, false, true, true, false, false, true, &mut b);
            b
        }).collect();
        let readers: Vec<ColumnReader<'_, i64>> = bytes.iter().map(|b| ColumnReader::open(b).unwrap()).collect();
        for d in 0..n {
            for k in [0usize, 3, 1, 5, 2, 4].into_iter().take(if d % 2 == 0 { 6 } else { 2 }) {
                assert_eq!(readers[k].get(d as u32), Some(cols[k][d]), "reader {k} doc {d}");
            }
        }
        for _ in 0..20_000 {
            let (k, d) = (rnd(6) as usize, rnd(n as u64) as usize);
            assert_eq!(readers[k].get(d as u32), Some(cols[k][d]));
        }
        std::thread::scope(|sc| {
            for t in 0..4usize {
                let (readers, cols) = (&readers, &cols);
                sc.spawn(move || for d in (t..n).step_by(3) { assert_eq!(readers[t].get(d as u32), Some(cols[t][d])); });
            }
        });
        // Prefix sums agree with a plain loop for every width.
        let bytes: Vec<u8> = (0..400).map(|_| rnd(256) as u8).collect();
        for w in 1..=40usize {
            for count in [0usize, 1, 7, 63, 64, 65, 100, 255] {
                let count = count.min(390 * 8 / w - 1);
                let mask = u64::MAX >> (64 - w);
                let want: u64 = (0..count).map(|i| { let bit = i * w; (u64::from_le_bytes(bytes[bit / 8..bit / 8 + 8].try_into().unwrap()) >> (bit % 8)) & mask }).sum();
                assert_eq!(packed_prefix_sum(&bytes, w, count), want, "w {w} count {count}");
            }
        }
    }

    /// v34 value-list points: a dense column of few distinct values answers
    /// every points question (counts, range bits, min / max, positions and
    /// their docs and values, the descending visit) as the docs-sorted
    /// index does, for i64, u64 and f64; it is far smaller; sparse, NaN,
    /// many-valued and constant columns keep their layouts.
    #[test]
    fn value_list_points_answer_like_sorted_docs() {
        fn check<T: FastValue + std::fmt::Debug>(name: &str, vals: &[T], probes: &[(T, T)]) {
            let n = vals.len();
            let present = vec![u64::MAX; n.div_ceil(64)];
            let (mut a, mut b) = (Vec::new(), Vec::new());
            encode_numeric_vbdl(vals, Some(&present), n as u32, true, true, true, false, false, &mut a);
            encode_numeric_vbdl(vals, Some(&present), n as u32, true, true, true, false, true, &mut b);
            let (x, y) = (ColumnReader::<T>::open(&a).unwrap(), ColumnReader::<T>::open(&b).unwrap());
            assert!(y.vlists.is_some() && x.vlists.is_none(), "{name}");
            // `_disk_usage` counts the doc lists as the points index.
            let tag = if T::TOTAL_ORDER { TAG_I64 } else { TAG_F64 };
            let pts = column_parts(tag, &b).unwrap().points;
            assert!(pts > 0 && (pts as usize) < b.len(), "{name}: points {pts} of {}", b.len());
            assert!(b.len() * 2 < a.len(), "{name}: {} vs {}", b.len(), a.len());
            assert!(y.has_points() && y.counts_from_points());
            assert_eq!(x.points_len(), y.points_len());
            for i in 0..n {
                assert_eq!(x.point_doc(i), y.point_doc(i), "{name} pos {i}");
                assert_eq!(x.point_value(i).to_le(), y.point_value(i).to_le(), "{name} pos {i}");
            }
            for d in 0..n as u32 {
                assert_eq!(x.get(d).map(|v| v.to_le()), y.get(d).map(|v| v.to_le()));
            }
            assert_eq!(x.points_min_max().map(|(p, q)| (p.to_le(), q.to_le())), y.points_min_max().map(|(p, q)| (p.to_le(), q.to_le())));
            for &(lo, hi) in probes {
                assert_eq!(x.count_range(lo, hi), y.count_range(lo, hi), "{name} [{lo:?}, {hi:?}]");
                let (mut wa, mut wb) = (vec![0u64; n.div_ceil(64)], vec![0u64; n.div_ceil(64)]);
                x.or_range_bits(lo, hi, &mut wa);
                y.or_range_bits(lo, hi, &mut wb);
                assert_eq!(wa, wb, "{name} or [{lo:?}, {hi:?}]");
                let (mut ra, mut rb) = (vec![0u64; n.div_ceil(64)], vec![0u64; n.div_ceil(64)]);
                x.range_bits(lo, hi, &mut ra);
                y.range_bits(lo, hi, &mut rb);
                assert_eq!(ra, rb, "{name} range [{lo:?}, {hi:?}]");
                assert_eq!(x.points_min_max_in(lo, hi).map(|(p, q)| (p.to_le(), q.to_le())), y.points_min_max_in(lo, hi).map(|(p, q)| (p.to_le(), q.to_le())));
            }
            let (mut da, mut db) = (Vec::new(), Vec::new());
            x.visit_desc(|d, _| { da.push(d); da.len() < 3000 });
            y.visit_desc(|d, _| { db.push(d); db.len() < 3000 });
            assert_eq!(da, db, "{name} desc");
        }
        let n = 20_000usize;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m };
        // An http status: mostly 200, a few others, one rare.
        let codes = [200i64, 304, 404, 500, 206, 302, 403, 400];
        let status: Vec<i64> = (0..n).map(|i| if i == 7777 { 501 } else { let r = rnd(1000); if r < 850 { 200 } else { codes[1 + (r % 7) as usize] } }).collect();
        check("status", &status, &[(200, 200), (300, 404), (i64::MIN, i64::MAX), (501, 501), (100, 199), (305, 403), (500, 600), (404, 200)]);
        let small: Vec<u64> = (0..n).map(|_| rnd(5) * 7).collect();
        check("u64", &small, &[(0, 0), (7, 21), (0, u64::MAX), (1, 6), (28, 28)]);
        let rates: Vec<f64> = (0..n).map(|_| [0.5, 1.25, -3.0, 1e9][rnd(4) as usize]).collect();
        check("f64", &rates, &[(0.5, 0.5), (-3.0, 1.25), (f64::NEG_INFINITY, f64::INFINITY), (2.0, 3.0), (1e9, 1e9)]);
        // Kept layouts.
        let present = vec![u64::MAX; n.div_ceil(64)];
        let layout = |vals: &[f64], present: &[u64]| {
            let mut b = Vec::new();
            encode_numeric_vbdl(vals, Some(present), n as u32, true, true, true, false, true, &mut b);
            ColumnReader::<f64>::open(&b).unwrap().vlists.is_some()
        };
        let mut nan = rates.clone();
        nan[3] = f64::NAN;
        assert!(!layout(&nan, &present));
        let many: Vec<f64> = (0..n).map(|i| (i % 100) as f64).collect();
        assert!(!layout(&many, &present));
        let mut sparse = present.clone();
        sparse[0] = 0;
        assert!(!layout(&rates, &sparse));
        assert!(!layout(&vec![1.5; n], &present));
        let zeros: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }).collect();
        assert!(!layout(&zeros, &present));
    }

    /// v33 packed keyword term tables: every ordinal, lookup bound and doc
    /// value reads as from the plain table, for hex, base64, mixed and long
    /// terms across block boundaries, with and without the block index;
    /// hex and base64 tables take far fewer bytes; a v33 column reads as a
    /// v32 one and is smaller.
    #[test]
    fn packed_keyword_tables_read_like_plain_ones() {
        let term_of = |kind: usize, i: u64| -> String {
            let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            match kind {
                0 => format!("{:016x}", h),
                1 => format!("AaG{:016x}", h).replace('a', "_").replace('3', "Q").replace('f', "-"),
                2 => format!("{:08x}-{:04x}", h >> 32, h & 0xffff),
                _ => if i % 7 == 0 { String::new() } else { format!("https://10.{}.{}.{}/p {}", i % 7, i / 7 % 250, i % 251, "x".repeat((i % 30) as usize)) },
            }
        };
        for kind in 0..4 {
            for n in [1usize, 15, 16, 17, 33, 2000] {
                let mut terms: Vec<String> = (0..n as u64).map(|i| term_of(kind, i)).collect();
                terms.sort();
                terms.dedup();
                let mut plain = Vec::new();
                write_term_table(&mut plain, terms.len(), terms.iter().map(|t| t.as_bytes()), false);
                let mut packed = Vec::new();
                write_term_table_packed(&mut packed, terms.len(), terms.iter().map(|t| t.as_bytes()));
                let mut fc = Vec::new();
                write_term_table(&mut fc, terms.len(), terms.iter().map(|t| t.as_bytes()), true);
                if terms.len() > 100 && kind < 3 {
                    assert!(packed.len() * 10 < fc.len() * 8, "kind {kind} n={n}: packed {} vs front-coded {}", packed.len(), fc.len());
                }
                assert_eq!(term_count(&packed), Some(terms.len() as u64));
                let (p_off, p_start, _) = parse_term_table(&plain).unwrap();
                let (k_off, k_start, k_end) = parse_term_table(&packed).unwrap();
                assert!(k_off.fc && k_off.packed);
                assert_eq!(k_end, packed.len());
                let kbytes = &packed[k_start..k_end];
                let pbytes = &plain[p_start..];
                let unindexed = KeywordOffsets { indexed: None, count: k_off.count, encoded_len: k_off.encoded_len, fc: true, packed: true, heads: std::sync::OnceLock::new() };
                for (o, t) in terms.iter().enumerate() {
                    assert_eq!(k_off.term_at(kbytes, o).as_deref(), Some(t.as_bytes()), "kind {kind} n={n} ord {o}");
                    assert_eq!(unindexed.term_at(kbytes, o).as_deref(), Some(t.as_bytes()), "unindexed kind {kind} n={n} ord {o}");
                }
                assert!(k_off.term_at(kbytes, terms.len()).is_none());
                let mut keys: Vec<String> = terms.iter().step_by(3).cloned().collect();
                keys.extend(["", "0", "a", "_", "~", "AaG", "https://10.3"].map(String::from));
                keys.extend(terms.iter().step_by(5).map(|t| format!("{t}0")));
                // Proper prefixes, and keys just below terms.
                keys.extend(terms.iter().step_by(4).filter(|t| t.len() > 1).map(|t| t[..t.len() / 2].to_string()));
                keys.extend(terms.iter().step_by(4).filter(|t| t.len() > 1).map(|t| { let mut b = t.clone().into_bytes(); let l = b.len() - 1; b[l] -= 1; String::from_utf8_lossy(&b).into_owned() }));
                for k in &keys {
                    for inclusive in [false, true] {
                        assert_eq!(k_off.lower_bound(kbytes, k.as_bytes(), inclusive), p_off.lower_bound(pbytes, k.as_bytes(), inclusive), "kind {kind} n={n} {k:?} {inclusive}");
                        assert_eq!(unindexed.lower_bound(kbytes, k.as_bytes(), inclusive), p_off.lower_bound(pbytes, k.as_bytes(), inclusive));
                    }
                }
            }
        }
        // A column: v33 packed, v32 front-coded, the same values and lookups.
        let n = 3000u32;
        let mut ff = crate::fast_fields::FastFields::new(n);
        for d in 0..n {
            if d % 11 != 0 {
                ff.keyword_column_mut(5).set(d, &term_of(0, d as u64));
            }
        }
        let body = |version| encode_fast_field_columns_versioned(&ff, n, version).into_iter().find(|(f, _, _)| *f == 5).unwrap();
        let (_, tag33, b33) = body(33);
        let (_, tag32, b32) = body(32);
        assert!(b33.len() * 10 < b32.len() * 8, "{} vs {}", b33.len(), b32.len());
        let r33 = KeywordReader::open_layout(&b33, OrdLayout::from_tag(tag33)).unwrap();
        let r32 = KeywordReader::open_layout(&b32, OrdLayout::from_tag(tag32)).unwrap();
        for d in 0..n {
            assert_eq!(r33.get(d), r32.get(d), "doc {d}");
        }
        for t in [term_of(0, 7), term_of(0, 2999), "nope".to_string(), String::new(), "8".to_string()] {
            assert_eq!(r33.lookup_ord(&t), r32.lookup_ord(&t), "{t}");
            assert_eq!(r33.seek_ord(&t), r32.seek_ord(&t), "{t}");
        }
        assert_eq!(r33.range_query("3", "a"), r32.range_query("3", "a"));
    }

    /// v30 decimal floats: dense float columns of exact decimals
    /// (`scaled_float` gauges, fixed-precision values, negatives, constant
    /// blocks) read back bit-exactly, with and without points, range counts
    /// match a scan, and they shrink; columns of inexact floats, -0.0 or NaN
    /// keep their layout, and v29 writes none.
    #[test]
    fn decimal_float_columns_read_back() {
        let n = 5_000usize;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m };
        let pct: Vec<f64> = (0..n).map(|i| ((i / 300) as f64 * 37.0 + rnd(200) as f64) / 1000.0).collect();
        let money: Vec<f64> = (0..n).map(|i| (rnd(2_000_000) as i64 - 1_000_000 + (i / 256) as i64 * 100) as f64 / 100.0).collect();
        let constant = vec![0.25f64; n];
        let whole: Vec<f64> = (0..n).map(|i| (1_000_000 + i as u64 * 3 + rnd(3)) as f64).collect();
        let present = vec![u64::MAX; n.div_ceil(64)];
        for points in [true, false] {
            for (name, values) in [("pct", &pct), ("money", &money), ("constant", &constant), ("whole", &whole)] {
                let mut plain = Vec::new();
                encode_numeric_vbd(values, Some(&present), n as u32, points, true, true, false, &mut plain);
                let mut bytes = Vec::new();
                encode_numeric_vbd(values, Some(&present), n as u32, points, true, true, true, &mut bytes);
                let c = ColumnReader::<f64>::open(&bytes).unwrap();
                assert!(c.decimal != 0.0 || name == "constant", "{name}: not decimal");
                if name != "constant" {
                    assert!(bytes.len() * 10 < plain.len() * 9, "{name}: {} vs {}", bytes.len(), plain.len());
                }
                for (d, v) in values.iter().enumerate() {
                    assert_eq!(c.get(d as DocId).map(f64::to_bits), Some(v.to_bits()), "{name} doc {d}");
                }
                // A constant column needs no points index.
                assert_eq!(c.has_points(), points && name != "constant");
                let (lo, hi) = (values[n / 3].min(values[n / 2]), values[n / 3].max(values[n / 2]));
                assert_eq!(c.count_range(lo, hi), values.iter().filter(|v| **v >= lo && **v <= hi).count() as u64, "{name}");
            }
            // Not exact decimals: plain layout, same values.
            let sines: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let mut signed_zero = pct.clone();
            signed_zero[17] = -0.0;
            let mut nan = pct.clone();
            nan[9] = f64::NAN;
            for (name, values) in [("sin", &sines), ("-0.0", &signed_zero), ("nan", &nan)] {
                let mut bytes = Vec::new();
                encode_numeric_vbd(values, Some(&present), n as u32, points, true, true, true, &mut bytes);
                let c = ColumnReader::<f64>::open(&bytes).unwrap();
                assert_eq!(c.decimal, 0.0, "{name}");
                for (d, v) in values.iter().enumerate() {
                    assert_eq!(c.get(d as DocId).map(f64::to_bits), Some(v.to_bits()), "{name} doc {d}");
                }
            }
        }
        // Through the column encoder: v30 writes decimals, v29 does not.
        let mut ff = crate::fast_fields::FastFields::new(n as u32);
        for (d, v) in pct.iter().enumerate() {
            ff.f64_column_mut(3).set(d as DocId, *v);
        }
        for (version, want) in [(30, true), (29, false)] {
            let cols = encode_fast_field_columns_versioned(&ff, n as u32, version);
            let (_, _, body) = cols.iter().find(|(f, t, _)| *f == 3 && *t == TAG_F64).unwrap();
            let c = ColumnReader::<f64>::open(body).unwrap();
            assert_eq!(c.decimal != 0.0, want, "v{version}");
            assert_eq!(c.get(1234).map(f64::to_bits), Some(pct[1234].to_bits()));
        }
    }

    #[test]
    fn block_values_are_the_raw_values_of_each_block() {
        fn check<T: FastValue + Copy + std::fmt::Debug>(c: &ColumnReader<'_, T>, n: usize, name: &str) -> usize {
            let mut buf = vec![T::zero(); BLOCK_SIZE];
            let mut seen = 0;
            for b in 0..n.div_ceil(BLOCK_SIZE) {
                let Some(k) = c.block_values(b, &mut buf) else { return 0 };
                assert_eq!(k, (n - b * BLOCK_SIZE).min(BLOCK_SIZE), "{name} block {b}");
                for (i, v) in buf[..k].iter().enumerate() {
                    let d = (b * BLOCK_SIZE + i) as DocId;
                    assert_eq!(v.to_le(), c.raw_value(d).to_le(), "{name} doc {d}");
                    seen += 1;
                }
            }
            seen
        }
        let n = 5_000usize;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m };
        let ts: Vec<i64> = (0..n).map(|i| 1_672_531_200_000 + (i as i64) * 1000 + rnd(3) as i64 * 1000).collect();
        let constant: Vec<i64> = vec![-42; n];
        // Two-decimal amounts (a scaled_float's values), and arbitrary doubles.
        let cents: Vec<f64> = (0..n).map(|_| rnd(100_000) as f64 / 100.0).collect();
        let raw: Vec<f64> = (0..n).map(|_| f64::from_bits(rnd(u64::MAX >> 2) | (1023u64 << 52))).collect();
        let present = vec![u64::MAX; n.div_ceil(64)];
        let mut sparse = present.clone();
        sparse[5] = 0;
        let mut checked = 0;
        for blocked in [false, true] {
            for (name, values) in [("ts", &ts), ("constant", &constant)] {
                let mut bytes = Vec::new();
                encode_numeric_vb(values, Some(&present), n as u32, true, true, blocked, &mut bytes);
                checked += check(&ColumnReader::<i64>::open(&bytes).unwrap(), n, name);
            }
            for (name, values) in [("cents", &cents), ("raw", &raw)] {
                let mut bytes = Vec::new();
                encode_numeric_vb(values, Some(&present), n as u32, true, true, blocked, &mut bytes);
                checked += check(&ColumnReader::<f64>::open(&bytes).unwrap(), n, name);
            }
            let mut bytes = Vec::new();
            encode_numeric_vb(&ts, Some(&sparse), n as u32, true, true, blocked, &mut bytes);
            let c = ColumnReader::<i64>::open(&bytes).unwrap();
            assert!(!c.all_present());
            checked += check(&c, n, "sparse");
        }
        assert!(checked >= 8 * n, "{checked} values checked");
    }

    #[test]
    fn block_packed_numeric_columns_read_back() {
        let n = 5_000usize;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move |m: u64| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % m };
        // Timestamps in ms at second granularity, ascending in runs.
        let ts: Vec<i64> = (0..n).map(|i| 1_672_531_200_000 + ((i % 700) as i64) * 1000 + if i % 700 == 0 { 0 } else { rnd(3) as i64 * 1000 }).collect();
        let neg: Vec<i64> = (0..n).map(|i| -5_000_000 + (i / 256) as i64 * 7 - rnd(5) as i64).collect();
        let constant: Vec<i64> = vec![-42; n];
        let big: Vec<u64> = (0..n).map(|i| u64::MAX - (i / 100) as u64 * 3).collect();
        for points in [true, false] {
            for (name, values) in [("ts", &ts), ("neg", &neg), ("constant", &constant)] {
                let present = vec![u64::MAX; n.div_ceil(64)];
                let mut plain = Vec::new();
                encode_numeric_vb(values, Some(&present), n as u32, points, true, false, &mut plain);
                let mut bytes = Vec::new();
                encode_numeric_vb(values, Some(&present), n as u32, points, true, true, &mut bytes);
                let c = ColumnReader::<i64>::open(&bytes).unwrap();
                assert!(c.pack.blocked || name == "constant", "{name}: not block-packed");
                assert!(bytes.len() <= plain.len(), "{name}: {} vs {}", bytes.len(), plain.len());
                for (d, v) in values.iter().enumerate() {
                    assert_eq!(c.get(d as DocId), Some(*v), "{name} doc {d}");
                }
                // A constant column needs no points index.
                assert_eq!(c.has_points(), points && name != "constant");
                let (lo, hi) = (values[n / 3].min(values[n / 2]), values[n / 3].max(values[n / 2]));
                assert_eq!(c.count_range(lo, hi), values.iter().filter(|v| (lo..=hi).contains(*v)).count() as u64, "{name}");
            }
            let present = vec![u64::MAX; n.div_ceil(64)];
            let mut bytes = Vec::new();
            encode_numeric_vb(&big, Some(&present), n as u32, points, true, true, &mut bytes);
            let c = ColumnReader::<u64>::open(&bytes).unwrap();
            for (d, v) in big.iter().enumerate() {
                assert_eq!(c.get(d as DocId), Some(*v), "u64 doc {d}");
            }
            // A sparse column keeps its layout.
            let mut sparse = vec![u64::MAX; n.div_ceil(64)];
            sparse[3] = 0;
            let mut bytes = Vec::new();
            encode_numeric_vb(&ts, Some(&sparse), n as u32, points, true, true, &mut bytes);
            let c = ColumnReader::<i64>::open(&bytes).unwrap();
            assert!(!c.pack.blocked);
            for d in 0..n {
                assert_eq!(c.get(d as DocId), if d / 64 == 3 { None } else { Some(ts[d]) });
            }
        }
    }

    /// v37: a boolean column that knows its docs with a value (0 / 1
    /// numerics), a keyword column of one term every doc has (no
    /// ordinals) and 1-bit blocks of two distant values read like their
    /// v36 forms, and v36 booleans still read every doc as present.
    #[test]
    fn v37_bools_constant_keywords_and_two_value_blocks_read_like_v36() {
        let n = 5_000u32;
        let mut fields = FastFields::new(n);
        fields.skip_points(3);
        for d in 0..n {
            // Field 1: a flag of every other 700-doc series.
            if (d / 700) % 2 == 0 { fields.bool_column_mut(1).set(d, (d / 700) % 4 == 0); }
            // Field 2: one term, every doc.
            fields.keyword_column_mut(2).set(d, "7.6.2");
            // Field 3: a routing hash constant along 300-doc series.
            fields.i64_column_mut(3).set(d, ((d / 300) as i64 * 0x5bd1_e995) % 2_000_000_011 - 1_000_000_000);
            // Field 4: one term, some docs.
            if d % 3 == 0 { fields.keyword_column_mut(4).set(d, "x"); }
        }
        fields.build_all();
        let encode = |version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let bytes = encode_fast_fields(&fields);
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            bytes
        };
        let (old, new) = (encode(36), encode(37));
        let (ro, rn) = (FastFieldsReader::open(&old).unwrap(), FastFieldsReader::open(&new).unwrap());
        let size = |d: &[ColumnEntry], f: u16| d.iter().find(|e| e.field == f).map(|e| e.end - e.start).unwrap();
        let (dir_old, dir_new) = (parse_column_directory(&old).unwrap(), parse_column_directory(&new).unwrap());
        // (Presence costs little: runs of series; values pack per block.)
        assert!(size(&dir_new, 1) <= size(&dir_old, 1), "bool {} vs {}", size(&dir_new, 1), size(&dir_old, 1));
        assert!(size(&dir_new, 2) < 32, "constant keyword {} bytes", size(&dir_new, 2));
        assert!(size(&dir_new, 3) * 3 < size(&dir_old, 3), "two-value {} vs {}", size(&dir_new, 3), size(&dir_old, 3));
        let (bo, bn) = (ro.get_bool(1).unwrap(), rn.get_bool(1).unwrap());
        assert!(!bo.presence_known() && bn.presence_known());
        assert_eq!(bo.words(), bn.words());
        assert_eq!(bo.count_true(), bn.count_true());
        assert_eq!(bo.true_docs(), bn.true_docs());
        for d in 0..n {
            assert_eq!(bo.get(d), bn.get(d), "bool doc {d}");
            assert!(bo.has_value(d));
            assert_eq!(bn.has_value(d), (d / 700) % 2 == 0, "bool doc {d}");
        }
        let present = bn.presence_words().unwrap();
        assert_eq!(present.iter().map(|w| w.count_ones()).sum::<u32>(), (0..n).filter(|d| (d / 700) % 2 == 0).count() as u32);
        assert!(bo.presence_words().is_none());
        for f in [2u16, 4] {
            let (a, b) = (ro.get_keyword(f).unwrap(), rn.get_keyword(f).unwrap());
            for d in 0..=n {
                assert_eq!(a.ord(d), b.ord(d), "keyword {f} doc {d}");
                assert_eq!(a.get(d), b.get(d));
            }
            assert_eq!(a.presence_words(), b.presence_words());
            assert_eq!(a.ordinal_counts(|_| true), b.ordinal_counts(|_| true));
            let (bytes, layout) = rn.keyword_column(f).unwrap();
            let table = KeywordTable::parse_layout(bytes, layout).unwrap();
            let shared = KeywordReader::with_table(bytes, &table);
            for d in 0..n {
                assert_eq!(shared.ord(d), a.ord(d));
            }
        }
        let (a, b) = (ro.get_i64(3).unwrap(), rn.get_i64(3).unwrap());
        assert!(b.pack.blocked);
        let mut buf = [0i64; BLOCK_SIZE];
        let mut buf2 = [0i64; BLOCK_SIZE];
        for blk in 0..a.block_count() {
            assert_eq!(a.block_values(blk, &mut buf), b.block_values(blk, &mut buf2));
            assert_eq!(buf, buf2, "block {blk}");
            assert_eq!(a.block_min_max(blk), b.block_min_max(blk));
        }
        for d in 0..n {
            assert_eq!(a.get(d), b.get(d), "i64 doc {d}");
        }
        for (lo, hi) in [(i64::MIN, i64::MAX), (-500_000_000, 0), (1, 900_000_000)] {
            assert_eq!(a.range_query(lo, hi), b.range_query(lo, hi));
        }
    }

    /// v36 block-packed keyword ordinals with missing docs: a TSDB-shaped
    /// column (runs of one value, whole runs absent, scattered gaps) packs
    /// per block and reads like its v35 form through every reader.
    #[test]
    fn block_packed_keyword_columns_with_missing_docs_read_like_v35() {
        let n = 20_000u32;
        let mut fields = FastFields::new(n);
        for d in 0..n {
            // Field 1: series of 700 docs, every third series without a value.
            if (d / 700) % 3 != 2 { fields.keyword_column_mut(1).set(d, &format!("pod-{:04}", d / 700)); }
            // Field 2: runs with a gap every 17 docs.
            if d % 17 != 0 { fields.keyword_column_mut(2).set(d, &format!("r{}", d / 3000)); }
            // Field 3: mostly missing (sparse wins).
            if d % 97 == 0 { fields.keyword_column_mut(3).set(d, &format!("s{}", d % 5)); }
            // Field 4: one term over the first half only (whole blocks of
            // none, then of one value).
            if d < n / 2 { fields.keyword_column_mut(4).set(d, "only"); }
        }
        fields.build_all();
        let encode = |version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let bytes = encode_fast_fields(&fields);
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            bytes
        };
        let (old, new) = (encode(35), encode(36));
        let (dir_old, dir_new) = (parse_column_directory(&old).unwrap(), parse_column_directory(&new).unwrap());
        let entry = |d: &[ColumnEntry], f: u16| d.iter().find(|e| e.field == f).copied().unwrap();
        for f in [1u16, 2, 4] {
            let e = entry(&dir_new, f);
            assert!(e.blocked && e.tag == TAG_KEYWORD, "field {f}");
            assert!(e.end - e.start < (entry(&dir_old, f).end - entry(&dir_old, f).start) / 4, "field {f}: {} bytes", e.end - e.start);
        }
        assert!(!entry(&dir_old, 1).blocked && !entry(&dir_new, 3).blocked);
        let (ro, rn) = (FastFieldsReader::open(&old).unwrap(), FastFieldsReader::open(&new).unwrap());
        for f in [1u16, 2, 3, 4] {
            let (a, b) = (ro.get_keyword(f).unwrap(), rn.get_keyword(f).unwrap());
            for d in 0..=n {
                assert_eq!(a.ord(d), b.ord(d), "field {f} doc {d}");
                assert_eq!(a.get(d), b.get(d));
            }
            assert_eq!(a.ord_blocks(), b.ord_blocks());
            assert_eq!(a.ord_index(), b.ord_index());
            assert_eq!(a.ordinal_counts(|_| true), b.ordinal_counts(|_| true));
            assert_eq!(a.presence_words(), b.presence_words());
            let (mut pa, mut pb) = (Vec::new(), Vec::new());
            a.for_each_present(|d, o| pa.push((d, o)));
            b.for_each_present(|d, o| pb.push((d, o)));
            assert_eq!(pa, pb, "field {f}");
            let (bytes, layout) = rn.keyword_column(f).unwrap();
            let table = KeywordTable::parse_layout(bytes, layout).unwrap();
            let shared = KeywordReader::with_table(bytes, &table);
            for d in 0..n {
                assert_eq!(shared.ord(d), a.ord(d), "field {f} doc {d}");
            }
        }
    }

    /// v28 block-packed keyword ordinals: a column every doc has a value in
    /// whose ordinals run in blocks reads like its dense v27 form through
    /// every reader.
    #[test]
    fn block_packed_keyword_columns_read_like_dense_ones() {
        let n = 10_000u32;
        let mut fields = FastFields::new(n);
        for d in 0..n {
            // Field 1: runs of one value (an index sort's first key); field 2:
            // values spread over 60,000 terms (no block shrinks: stays dense);
            // field 3: runs with gaps (some docs missing: stays dense).
            fields.keyword_column_mut(1).set(d, &format!("host-{:03}", d / 1000));
            fields.keyword_column_mut(2).set(d, &format!("v{}", (d as u64 * 7919) % 60_000));
            if d % 17 != 0 { fields.keyword_column_mut(3).set(d, &format!("r{}", d / 3000)); }
        }
        fields.build_all();
        let encode = |version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let bytes = encode_fast_fields(&fields);
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            bytes
        };
        let (old, new) = (encode(27), encode(28));
        let dir_new = parse_column_directory(&new).unwrap();
        let entry = |d: &[ColumnEntry], f: u16| d.iter().find(|e| e.field == f).copied().unwrap();
        assert!(entry(&dir_new, 1).blocked && entry(&dir_new, 1).tag == TAG_KEYWORD);
        assert!(!entry(&dir_new, 2).blocked && !entry(&dir_new, 3).blocked);
        let e = entry(&dir_new, 1);
        assert!(e.end - e.start < 1000, "{} bytes", e.end - e.start);
        let (ro, rn) = (FastFieldsReader::open(&old).unwrap(), FastFieldsReader::open(&new).unwrap());
        for f in [1u16, 2, 3] {
            let (a, b) = (ro.get_keyword(f).unwrap(), rn.get_keyword(f).unwrap());
            for d in 0..=n {
                assert_eq!(a.ord(d), b.ord(d), "field {f} doc {d}");
                assert_eq!(a.get(d), b.get(d));
            }
            assert_eq!(a.ord_blocks(), b.ord_blocks());
            assert_eq!(a.ord_index(), b.ord_index());
            assert_eq!(a.ordinal_counts(|_| true), b.ordinal_counts(|_| true));
            assert_eq!(a.presence_words(), b.presence_words());
            assert_eq!(a.range_query("host-002", "host-005"), b.range_query("host-002", "host-005"));
            let (bytes, layout) = rn.keyword_column(f).unwrap();
            let owned = OwnedKeywordColumn::open_layout(super::super::directory::FileSlice::from_vec(bytes.to_vec()), None, layout).unwrap();
            let table = KeywordTable::parse_layout(bytes, layout).unwrap();
            let shared = KeywordReader::with_table(bytes, &table);
            for d in 0..n {
                assert_eq!(owned.ord(d), a.ord(d), "owned field {f} doc {d}");
                assert_eq!(owned.next_value(d).map(|x| x.1), Some(a.ord(d)));
                assert_eq!(shared.ord(d), a.ord(d));
            }
        }
    }

    #[test]
    fn sparse_keyword_columns_read_like_dense_ones() {
        let n = 10_000u32;
        let mut fields = FastFields::new(n);
        // Field 1: 3% of docs, 3 terms; field 2: 90% of docs.
        let sparse_term = |d: u32| ["ab", "cd", "ef"][(d / 33 % 3) as usize];
        for d in 0..n {
            if d % 33 == 7 { fields.keyword_column_mut(1).set(d, sparse_term(d)); }
            if d % 10 != 0 { fields.keyword_column_mut(2).set(d, &format!("v{}", d % 300)); }
        }
        fields.build_all();
        let encode = |version: u32| {
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(version));
            let bytes = encode_fast_fields(&fields);
            crate::codec::TEST_WRITE_VERSION.with(|v| v.set(0));
            bytes
        };
        let (old, new) = (encode(13), encode(14));
        let dir_old = parse_column_directory(&old).unwrap();
        let dir_new = parse_column_directory(&new).unwrap();
        let entry = |d: &[ColumnEntry], f: u16| d.iter().find(|e| e.field == f).copied().unwrap();
        assert!(!entry(&dir_old, 1).sparse && entry(&dir_new, 1).sparse);
        assert_eq!(entry(&dir_new, 1).tag, TAG_KEYWORD, "read as a keyword column");
        assert!(!entry(&dir_new, 2).sparse, "a mostly present column stays dense");
        let (e_old, e_new) = (entry(&dir_old, 1), entry(&dir_new, 1));
        assert!(e_new.end - e_new.start < (e_old.end - e_old.start) / 2, "sparse {} vs dense {}", e_new.end - e_new.start, e_old.end - e_old.start);
        let (ro, rn) = (FastFieldsReader::open(&old).unwrap(), FastFieldsReader::open(&new).unwrap());
        for f in [1u16, 2] {
            let (a, b) = (ro.get_keyword(f).unwrap(), rn.get_keyword(f).unwrap());
            assert_eq!(a.num_terms(), b.num_terms());
            for d in 0..=n {
                assert_eq!(a.ord(d), b.ord(d), "field {f} doc {d}");
                assert_eq!(a.get(d), b.get(d));
            }
            assert_eq!(a.ord_blocks(), b.ord_blocks());
            assert_eq!(a.ord_index(), b.ord_index());
            assert_eq!(a.ordinal_counts(|_| true), b.ordinal_counts(|_| true));
            assert_eq!(a.range_query("a", "d"), b.range_query("a", "d"));
            let (bytes, sparse) = rn.keyword_column(f).unwrap();
            let owned = OwnedKeywordColumn::open_layout(super::super::directory::FileSlice::from_vec(bytes.to_vec()), None, sparse).unwrap();
            let table = KeywordTable::parse_layout(bytes, sparse).unwrap();
            let shared = KeywordReader::with_table(bytes, &table);
            for d in 0..n {
                assert_eq!(owned.ord(d), a.ord(d));
                assert_eq!(shared.ord(d), a.ord(d));
            }
            // `next_value` visits every doc with a value, in order, with its
            // ordinal (a sparse column skips the rest).
            let mut seen = Vec::new();
            let mut from = 0;
            while let Some((d, o)) = owned.next_value(from) {
                if o != MISSING_ORD { seen.push((d, o)); }
                from = d + 1;
            }
            let want: Vec<_> = (0..n).filter(|&d| a.ord(d) != MISSING_ORD).map(|d| (d, a.ord(d))).collect();
            assert_eq!(seen, want, "field {f}");
            assert_eq!(owned.next_value(n), None);
        }
        // A rank table that disagrees with the bitmap is refused on parse.
        let e = entry(&dir_new, 1);
        let mut body = new[e.start..e.end].to_vec();
        // After the term table: the ordinal header word, the present count,
        // then the ranks (the first is always 0).
        let (_, _, end) = parse_term_table(&body).unwrap();
        body[end + 8] ^= 1;
        assert!(KeywordTable::parse_layout(&body, true).is_err());
    }

    #[test]
    fn four_byte_keyword_ordinals_still_read() {
        // A v10 body: u32 num_docs without width bits, 4-byte ordinals.
        let mut body = Vec::new();
        write_vint(2, &mut body);
        for t in ["a", "b"] {
            write_vint(1, &mut body);
            body.extend_from_slice(t.as_bytes());
        }
        body.extend_from_slice(&3u32.to_le_bytes());
        for o in [1u32, MISSING_ORD, 0] {
            body.extend_from_slice(&o.to_le_bytes());
        }
        let k = KeywordReader::open(&body).unwrap();
        assert_eq!((k.num_docs(), k.width), (3, 4));
        assert_eq!((k.get(0).as_deref(), k.get(1).as_deref(), k.get(2).as_deref()), (Some("b"), None, Some("a")));
        let t = KeywordTable::parse(&body).unwrap();
        assert_eq!(KeywordReader::with_table(&body, &t).get(0).as_deref(), Some("b"));
    }

    /// v36 compact columns: presence as runs and values packed per block of
    /// present values read exactly like the v35 rank table and packed
    /// values, through every reader, with and without points; a
    /// TSDB-shaped column (runs of present docs) shrinks.
    #[test]
    fn compact_runs_and_rank_blocks_read_like_v35() {
        let n = 30_000usize;
        let shapes: Vec<(&str, Box<dyn Fn(usize) -> bool>)> = vec![
            ("series", Box::new(|d: usize| (d / 850) % 4 == 1)),
            ("scattered", Box::new(|d: usize| d % 7 == 3)),
            ("tail", Box::new(|d: usize| d >= n - 300)),
            ("one", Box::new(|d: usize| d == 12_345)),
            // Nearly every doc: sparse but not compact (bitmap after values).
            ("almost", Box::new(|d: usize| d % 500 != 7)),
        ];
        for (name, has) in &shapes {
            // A slowly moving gauge per series, a counter, a negative range.
            let gauge: Vec<i64> = (0..n).map(|d| 1_000_000_000 + ((d / 850) as i64) * 7_777 + (d as i64 % 13) * 4096).collect();
            let counter: Vec<i64> = (0..n).map(|d| d as i64 * 10).collect();
            let neg: Vec<i64> = (0..n).map(|d| -((d as i64 * 7919) % 100_000)).collect();
            let mut presence = vec![0u64; n.div_ceil(64)];
            for d in 0..n {
                if has(d) { presence[d / 64] |= 1 << (d % 64); }
            }
            for values in [&gauge, &counter, &neg] {
                for points in [false, true] {
                    let (mut old, mut new) = (Vec::new(), Vec::new());
                    encode_numeric_vbdldr(values, Some(&presence), n as u32, points, true, true, false, true, true, false, &mut old);
                    encode_numeric_vbdldr(values, Some(&presence), n as u32, points, true, true, false, true, true, true, &mut new);
                    let (a, b) = (ColumnReader::<i64>::open_checked(&old, true).unwrap(), ColumnReader::<i64>::open_checked(&new, true).unwrap());
                    assert_eq!(a.compact, b.compact, "{name}");
                    assert!(b.compact || *name == "almost", "{name}");
                    // A sparse, not compact counter without points packs per
                    // block (a doc without a value takes its block's minimum).
                    if *name == "almost" && !points && std::ptr::eq(values, &counter) {
                        assert!(b.pack.blocked && !a.pack.blocked && b.is_sparse(), "{name}");
                    }
                    if *name == "series" {
                        assert!(!b.run_dir.is_empty() && new.len() < old.len(), "{name}: {} vs {}", new.len(), old.len());
                        // A counter packs per block of present values (the
                        // gauge's few distinct values stay table-packed).
                        if std::ptr::eq(values, &counter) {
                            assert!(!b.vmins.is_empty(), "{name}");
                            // (With points, both keep 3 bytes a present doc.)
                            assert!(points || new.len() * 2 < old.len(), "{name}: {} vs {}", new.len(), old.len());
                        }
                    }
                    assert_eq!(a.has_points(), b.has_points());
                    assert_eq!(a.value_doc_count(), b.value_doc_count(), "{name}");
                    assert_eq!(a.presence_words(), b.presence_words(), "{name}");
                    assert_eq!(a.present_docs().collect::<Vec<_>>(), b.present_docs().collect::<Vec<_>>());
                    for d in 0..=n as u32 {
                        assert_eq!(a.get(d), b.get(d), "{name} doc {d}");
                        assert_eq!(a.has_value(d), b.has_value(d));
                    }
                    assert_eq!(a.block_count(), b.block_count(), "{name}");
                    for blk in 0..a.block_count() {
                        assert_eq!(a.block_presence(blk), b.block_presence(blk));
                        assert_eq!(a.block_min_max(blk), b.block_min_max(blk), "{name} block {blk}");
                    }
                    for (lo, hi) in [(i64::MIN, i64::MAX), (1_000_050_000, 1_000_200_000), (100, 5_000), (-50_000, -10)] {
                        assert_eq!(a.range_query(lo, hi), b.range_query(lo, hi), "{name} {lo}..{hi}");
                        assert_eq!(a.count_range(lo, hi), b.count_range(lo, hi));
                        let (mut wa, mut wb) = (vec![0u64; n.div_ceil(64)], vec![0u64; n.div_ceil(64)]);
                        a.range_bits(lo, hi, &mut wa);
                        b.range_bits(lo, hi, &mut wb);
                        assert_eq!(wa, wb);
                    }
                    assert_eq!((a.min(), a.max()), (b.min(), b.max()));
                    if points {
                        for i in 0..a.points_len() {
                            assert_eq!((a.point_doc(i), a.point_value(i)), (b.point_doc(i), b.point_value(i)));
                        }
                    }
                    // A corrupt run fails validation.
                    if !b.run_dir.is_empty() {
                        let at = b.runs.as_ptr() as usize - new.as_ptr() as usize;
                        let mut bad = new.clone();
                        bad[at + 4] ^= 0x7f;
                        assert!(ColumnReader::<i64>::open_checked(&bad, true).is_err(), "{name}");
                    }
                }
            }
        }
    }

    fn check_compact<T: FastValue>(values: Vec<T>, bounds: &[(T, T)]) {
        let n = values.len();
        for points in [false, true] {
            for stride in [7, 73, n + 1] {
                let mut presence = vec![0u64; n.div_ceil(64)];
                for d in 0..n {
                    if d % stride == 1 { presence[d / 64] |= 1 << (d % 64); }
                }
                let mut bytes = Vec::new();
                encode_numeric(&values, Some(&presence), n as u32, points, &mut bytes);
                let c = ColumnReader::<T>::open(&bytes).unwrap();
                assert!(c.compact);
                assert_eq!(c.has_points(), points);
                let present_docs: Vec<_> = (0..n as u32).filter(|&d| c.has_value(d)).collect();
                assert_eq!(c.present_docs().collect::<Vec<_>>(), present_docs);
                for d in 0..n {
                    let present = presence[d / 64] >> (d % 64) & 1 != 0;
                    assert_eq!(c.get(d as u32).map(T::to_le), present.then(|| values[d].to_le()));
                    assert_eq!(c.raw_value(d as u32).to_le(), if present { values[d] } else { T::zero() }.to_le());
                }
                for &(lo, hi) in bounds {
                    let expected: Vec<_> = (0..n).filter(|&d| presence[d/64] >> (d%64) & 1 != 0 && values[d] >= lo && values[d] <= hi).map(|d| d as u32).collect();
                    assert_eq!(c.range_query(lo, hi), expected);
                    assert_eq!(c.count_range(lo, hi), expected.len() as u64);
                    let mut bits = vec![0u64; n.div_ceil(64)];
                    bits[0] = 1; // OR must preserve an existing missing-doc hit.
                    c.range_bits(lo, hi, &mut bits);
                    for d in 0..n { assert_eq!(bits[d/64] >> (d%64) & 1 != 0, d == 0 || expected.contains(&(d as u32))); }
                }
                // Old dense-values + presence layout remains readable.
                let mut old = Vec::new();
                encode_numeric(&values, None, n as u32, points, &mut old);
                assert_eq!(ColumnReader::<T>::open(&old).unwrap().present_docs().collect::<Vec<_>>(), (0..n as u32).collect::<Vec<_>>());
                for word in &presence { old.extend_from_slice(&word.to_le_bytes()); }
                let legacy = ColumnReader::<T>::open(&old).unwrap();
                assert!(!legacy.compact);
                assert_eq!(legacy.present_docs().collect::<Vec<_>>(), present_docs);
                for d in 0..n { assert_eq!(legacy.get(d as u32).map(T::to_le), c.get(d as u32).map(T::to_le)); }
            }
        }
    }

    #[test]
    fn compact_sparse_numeric_preserves_values_ranges_and_legacy_reads() {
        for n in [63, 64, 65, 257, 1025] {
            check_compact((0..n).map(|d| [i64::MIN, -1, 0, 1, i64::MAX][d%5]).collect(),
                &[(i64::MIN, i64::MAX), (-1, 1), (0, 0), (1, -1)]);
            check_compact((0..n).map(|d| [0, 1, u64::MAX][d%3]).collect(),
                &[(0, u64::MAX), (0, 0), (1, 0)]);
            check_compact((0..n).map(|d| [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN, -1.0, 1.0][d%7]).collect(),
                &[(f64::NEG_INFINITY, f64::INFINITY), (-0.0, 0.0), (-1.0, 1.0), (f64::NAN, 1.0), (1.0, f64::NAN), (1.0, -1.0)]);
        }
    }

    #[test]
    fn compact_empty_presence_and_corrupt_rank() {
        for points in [false, true] {
            let mut bytes = Vec::new();
            encode_numeric(&vec![0i64; 65], Some(&[0, 0]), 65, points, &mut bytes);
            let c = ColumnReader::<i64>::open(&bytes).unwrap();
            assert!(c.compact);
            assert_eq!(c.points_len(), 0);
            assert_eq!(c.count_range(i64::MIN, i64::MAX), 0);
            assert_eq!(c.get(64), None);
            let mut bits = [0, 1];
            c.range_bits(i64::MIN, i64::MAX, &mut bits);
            assert_eq!(bits, [0, 1]);
            // Header + one block min/max (+ packing) + present count, then first rank.
            let rank = if c.pack == Packing::PLAIN { 28 } else { 38 };
            bytes[rank] = 1;
            assert!(ColumnReader::<i64>::open(&bytes).is_err());
        }
        for n in [0, 1, 65] {
            let mut bytes = Vec::new();
            encode_numeric(&vec![42u64; n], None, n as u32, true, &mut bytes);
            let c = ColumnReader::<u64>::open(&bytes).unwrap();
            assert!(!c.compact);
            assert_eq!(c.count_range(42, 42), n as u64);
        }
    }

    #[test]
    fn cached_float_points_equal_comparator_order() {
        let mut seed = 123456789u64;
        let mut random = Vec::new();
        for _ in 0..10000 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            random.push(f64::from_bits(seed));
        }
        let special = vec![f64::NAN, -0.0, 0.0, f64::NEG_INFINITY,
            f64::INFINITY, -f64::NAN, f64::from_bits(1), -f64::from_bits(1),
            f64::from_bits(0x7ff0000000000001), 1.0, -1.0];
        for values in [vec![], vec![0.0; 1000], (0..1000).map(|i| i as f64).collect(), special, random] {
            for padding in [0, 17] {
                let n = values.len() as u32 + padding;
                let value = |d: u32| values.get(d as usize).copied().unwrap_or(0.0);
                let mut expected: Vec<u32> = (0..n).collect();
                expected.sort_unstable_by(|&a, &b| value(a).sort_cmp(&value(b)).then(a.cmp(&b)));
                let expected: Vec<u8> = expected.into_iter().flat_map(u32::to_le_bytes).collect();
                let mut actual = Vec::new();
                encode_points_order(&values, n, Packing::PLAIN, &mut actual);
                assert_eq!(actual, expected, "length {}, padding {padding}", values.len());
            }
        }
    }

    #[test]
    fn keyword_offset_refusal_uses_mapped_bytes_without_losing_matches() {
        let mut fields=FastFields::new(100);
        for doc in 0..100 {
            if doc%7!=0 {fields.keyword_column_mut(3).set(doc,&format!("term-{doc:03}"));}
            fields.multi_keyword_column_mut(4).push(doc,&format!("term-{doc:03}"));
            fields.multi_keyword_column_mut(4).push(doc,"common");
        }
        fields.build_all();let bytes=encode_fast_fields(&fields);
        let ff=FastFieldsReader::open(&bytes).unwrap();
        let small=crate::admission::Budget::new(1);
        let single=ff.keyword_column_bytes(3).unwrap();
        let (offsets,_,_)=parse_term_table_admitted(single,&small).unwrap();
        assert!(offsets.indexed.is_none());assert_eq!(small.used(),0);
        let fallback=KeywordReader::open_with(single,Some(std::sync::Arc::new(offsets))).unwrap();
        let normal=KeywordReader::open(single).unwrap();
        for doc in 0..=100 {assert_eq!(fallback.get(doc),normal.get(doc));}
        for term in ["", "term-010", "term-050", "zzz"] {
            assert_eq!(fallback.lookup_ord(term),normal.lookup_ord(term));
            assert_eq!(fallback.seek_ord(term),normal.seek_ord(term));
        }
        assert_eq!(fallback.range_query("term-020","term-080"),normal.range_query("term-020","term-080"));
        let multi=ff.multi_keyword_column_bytes(4).unwrap();
        let (offsets,_,_)=parse_term_table_admitted(multi,&small).unwrap();
        let fallback=MultiKeywordReader::open_with(multi,Some(std::sync::Arc::new(offsets))).unwrap();
        let normal=MultiKeywordReader::open(multi).unwrap();
        for doc in 0..100 {assert_eq!(fallback.values(doc),normal.values(doc));}
        assert_eq!(fallback.lookup_ord("common"),normal.lookup_ord("common"));
        assert_eq!(fallback.range_query("term-020","term-080"),normal.range_query("term-020","term-080"));
        let budget=crate::admission::Budget::new(1<<20);
        let (offsets,_,_)=parse_term_table_admitted(single,&budget).unwrap();
        assert!(offsets.indexed.is_some());let offsets=std::sync::Arc::new(offsets);
        let reader=KeywordReader::open_with(single,Some(offsets.clone())).unwrap();
        drop(offsets);assert!(budget.used()>0);drop(reader);assert_eq!(budget.used(),0);
    }

    #[test]
    fn cached_keyword_dictionary_preserves_missing_values_and_shared_offsets() {
        let mut fields = FastFields::new(100);
        for doc in 0..100 {
            if doc % 7 != 0 {
                fields.keyword_column_mut(3).set(doc, &format!("id-{doc:04}-é"));
            }
        }
        fields.keyword_column_mut(3).set(1, "");
        fields.build_all();
        let bytes = encode_fast_fields(&fields);
        let directory = parse_column_directory(&bytes).unwrap();
        let tables = std::collections::HashMap::from([(3, std::sync::OnceLock::new())]);
        let direct = FastFieldsReader::open(&bytes).unwrap().get_keyword(3).unwrap();
        let body = FastFieldsReader::open(&bytes).unwrap().column(3, TAG_KEYWORD).unwrap();
        for end in 0..body.len() { assert!(KeywordTable::parse(&body[..end]).is_err()); }
        let cached = FastFieldsReader::with_keyword_tables(&bytes, &directory, &tables).get_keyword(3).unwrap();
        let reopened = FastFieldsReader::with_keyword_tables(&bytes, &directory, &tables).get_keyword(3).unwrap();
        assert_eq!(cached.terms.as_ptr(), reopened.terms.as_ptr());
        assert_eq!(cached.get(1).as_deref(), Some(""));
        for doc in 0..=100 {
            assert_eq!(cached.get(doc), direct.get(doc));
            assert_eq!(cached.ord(doc), direct.ord(doc));
        }
        assert_eq!(cached.range_query("id-0020", "id-0080"), direct.range_query("id-0020", "id-0080"));
        for ord in 0..direct.num_terms() as u32 {
            let term = direct.term(ord).unwrap();
            assert_eq!(cached.lookup_ord(&term), Some(ord));
            assert_eq!(cached.exact_match(&term), direct.exact_match(&term));
        }
        // Readers on different request threads borrow the same immutable table.
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let reader = FastFieldsReader::with_keyword_tables(&bytes, &directory, &tables).get_keyword(3).unwrap();
                    for doc in 0..100 { assert_eq!(reader.get(doc), direct.get(doc)); }
                });
            }
        });
    }

    #[test]
    fn value_doc_count_counts_present_docs() {
        for (n, every) in [(5000usize, 1usize), (5000, 3), (130, 7), (64, 2)] {
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            for points in [false, true] {
                let mut bytes = Vec::new();
                if every == 1 {
                    encode_numeric(&values, None, n as u32, points, &mut bytes);
                } else {
                    let mut presence = vec![0u64; n.div_ceil(64)];
                    for d in (0..n).filter(|d| d % every == 0) {
                        presence[d / 64] |= 1 << (d % 64);
                    }
                    encode_numeric(&values, Some(&presence), n as u32, points, &mut bytes);
                }
                let c = ColumnReader::<f64>::open(&bytes).unwrap();
                let want = (0..n as DocId).filter(|&d| c.has_value(d)).count() as u64;
                assert_eq!(c.value_doc_count(), want, "n={n} every={every} points={points}");
                assert_eq!(want, (0..n).filter(|d| d % every == 0).count() as u64);
            }
        }
        // A missing column: no reader at all.
        let dir: Vec<ColumnEntry> = Vec::new();
        assert!(FastFieldsReader::with_directory(&[], &dir).get_f64(3).is_none());
    }

    #[test]
    fn segment_numeric_checks_validate_presence_ranks_once_and_read_the_same_values() {
        let n = 5000usize;
        let values: Vec<f64> = (0..n).map(|i| i as f64 / 4.0 - 300.0).collect();
        let mut presence = vec![0u64; n.div_ceil(64)];
        for d in (0..n).filter(|d| d % 3 == 1) { presence[d / 64] |= 1 << (d % 64); }
        let mut bytes = Vec::new();
        encode_numeric(&values, Some(&presence), n as u32, true, &mut bytes);
        let reference = ColumnReader::<f64>::open(&bytes).unwrap();
        assert!(reference.compact);
        let dir = vec![ColumnEntry { field: 7, tag: TAG_F64, start: 0, end: bytes.len(), sparse: false, blocked: false }];
        let tables = std::collections::HashMap::new();
        let checks = numeric_checks(&dir);
        assert_eq!(checks.len(), 1);
        assert!(checks[&(7, TAG_F64)].get().is_none());
        for _ in 0..3 {
            let reader = FastFieldsReader::with_segment_tables(&bytes, &dir, &tables, &checks).get_f64(7).unwrap();
            assert!(checks[&(7, TAG_F64)].get().is_some());
            for d in 0..n as DocId {
                assert_eq!(reader.get(d).map(f64::to_bits), reference.get(d).map(f64::to_bits), "doc {d}");
            }
        }
        assert!(FastFieldsReader::with_segment_tables(&bytes, &dir, &tables, &checks).get_f64(8).is_none());
        assert!(FastFieldsReader::with_segment_tables(&bytes, &dir, &tables, &checks).get_i64(7).is_none());
        // The skipped check is the only difference: a column whose ranks do
        // not add up still fails a checked open.
        let mut corrupt = bytes.clone();
        let at = (0..corrupt.len() - 4).find(|&i| {
            let mut c = corrupt.clone();
            c[i] ^= 0x01;
            ColumnReader::<f64>::open(&c).is_err() && ColumnReader::<f64>::open_checked(&c, false).is_ok()
        });
        if let Some(i) = at {
            corrupt[i] ^= 0x01;
            let fresh = numeric_checks(&dir);
            assert!(FastFieldsReader::with_segment_tables(&corrupt, &dir, &tables, &fresh).get_f64(7).is_none());
            assert!(fresh[&(7, TAG_F64)].get().is_none());
        }
    }

    #[test]
    fn segment_keyword_tables_are_reused_and_isolated_between_segments() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = crate::Index::create(dir.path(), "keyword-cache", vec![crate::Field::new(0, "id", crate::FieldType::Keyword)], crate::IndexSettings::default()).unwrap();
        for prefix in ["a", "b"] {
            for doc in 0..2 {
                let mut d = crate::Document::new();
                d.add_keyword(0, &format!("{prefix}{doc}"));
                index.add_document(&d).unwrap();
            }
            index.refresh().unwrap();
        }
        assert_eq!(index.segments().len(), 2);
        let a = index.segments()[0].fast_fields().get_keyword(0).unwrap();
        let again = index.segments()[0].fast_fields().get_keyword(0).unwrap();
        let b = index.segments()[1].fast_fields().get_keyword(0).unwrap();
        assert_eq!(a.terms.as_ptr(), again.terms.as_ptr());
        let segment = &index.segments()[0];
        let shape_reader = segment.keyword_reader(0).unwrap();
        let owned = segment.owned_keyword_column(0).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a.terms, &shape_reader.terms));
        assert!(std::sync::Arc::ptr_eq(&a.terms, &owned.terms));
        for doc in 0..=2 {
            assert_eq!(a.ord(doc), owned.ord(doc));
            assert_eq!(a.get(doc), owned.term(owned.ord(doc)));
        }
        assert_ne!(a.terms.as_ptr(), b.terms.as_ptr());
        assert_eq!(a.get(0).as_deref(), Some("a0"));
        assert_eq!(b.get(0).as_deref(), Some("b0"));
    }

    #[test]
    fn concurrent_geo_keyword_readers_share_one_dictionary() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = crate::Index::create(dir.path(), "shape-cache", vec![crate::Field::new(0, "shape", crate::FieldType::Keyword)], crate::IndexSettings::default()).unwrap();
        for doc in 0..8192 {
            let mut d = crate::Document::new();
            d.add_keyword(0, &format!("unique-shape-{doc:08}"));
            index.add_document(&d).unwrap();
        }
        index.refresh().unwrap();
        let segment = &index.segments()[0];
        let barrier = std::sync::Barrier::new(12);
        // Race the first presence check, per-document shape lookup, and
        // owned scorer. All must use one allocation even on a cold segment.
        let tables = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..12).map(|i| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    match i % 3 {
                        0 => segment.fast_fields().get_keyword(0).unwrap().terms,
                        1 => segment.keyword_reader(0).unwrap().terms,
                        _ => segment.owned_keyword_column(0).unwrap().terms,
                    }
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
        });
        assert_eq!(tables[0].len(), 8192);
        assert!(tables.iter().all(|t| std::sync::Arc::ptr_eq(t, &tables[0])));
    }

    fn brute_bits<T: FastValue>(vals: &[T], lo: T, hi: T) -> Vec<u64> {
        let mut w = vec![0u64; vals.len().div_ceil(64)];
        for (i, v) in vals.iter().enumerate() {
            if *v >= lo && *v <= hi {
                w[i / 64] |= 1 << (i % 64);
            }
        }
        w
    }

    #[test]
    /// A dense column of one value is written without a points index (a
    /// range holds every doc or none); its ranges, counts and sorted reads
    /// answer as before. A sparse or NaN column keeps its index.
    fn constant_columns_skip_the_points_index() {
        let n = 3000usize;
        let range_docs = |c: &ColumnReader<i64>, lo: i64, hi: i64| -> Vec<usize> {
            let mut w = vec![0u64; n.div_ceil(64)];
            c.or_range_bits(lo, hi, &mut w);
            (0..n).filter(|d| w[d / 64] >> (d % 64) & 1 == 1).collect()
        };
        for v in [1_689_826_834_181i64, 0, -7] {
            let vals = vec![v; n];
            let mut body = Vec::new();
            encode_numeric(&vals, None, n as u32, true, &mut body);
            let c = ColumnReader::<i64>::open(&body).unwrap();
            assert!(!c.has_points(), "constant {v}");
            for (lo, hi, all) in [(v, v, true), (v - 1, v + 1, true), (i64::MIN, i64::MAX, true), (v + 1, v + 9, false), (v - 9, v - 1, false)] {
                assert_eq!(c.count_range(lo, hi), if all { n as u64 } else { 0 }, "{v} [{lo}, {hi}]");
                assert_eq!(range_docs(&c, lo, hi).len(), if all { n } else { 0 });
            }
            assert!((0..n).all(|d| c.get(d as u32) == Some(v)));
            // Smaller than the column with the index.
            let mut with = Vec::new();
            encode_numeric(&[vals.clone(), vec![v + 1]].concat(), None, n as u32 + 1, true, &mut with);
            assert!(body.len() * 2 < with.len(), "{} vs {}", body.len(), with.len());
        }
        let mut f = Vec::new();
        encode_numeric(&vec![f64::NAN; 100], None, 100, true, &mut f);
        assert!(ColumnReader::<f64>::open(&f).unwrap().has_points());
        let mut one = Vec::new();
        encode_numeric(&[5i64], None, 1, true, &mut one);
        assert!(ColumnReader::<i64>::open(&one).unwrap().has_points());
    }

    #[test]
    fn points_index_range_bits_match_brute_force() {
        // Random values (uncorrelated with doc ids: block min/max is useless),
        // sorted runs (block skipping/full acceptance), duplicates, -0.0,
        // infinities and NaN.
        let n = 5000usize;
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let f: Vec<f64> = (0..n)
            .map(|i| match i % 97 {
                0 => -0.0,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => f64::NAN,
                _ if i < 1500 => i as f64,
                _ => (rnd() % 1000) as f64 / 10.0 - 20.0,
            })
            .collect();
        let iv: Vec<i64> = (0..n).map(|i| if i < 2000 { i as i64 } else { (rnd() % 400) as i64 - 200 }).collect();
        let mut fbody = Vec::new();
        encode_numeric(&f, None, n as u32, true, &mut fbody);
        let mut ibody = Vec::new();
        encode_numeric(&iv, None, n as u32, true, &mut ibody);
        // The v6 layout (no points index) must give the same answers.
        let points = |b: &[u8]| { let p = ColumnReader::<i64>::open(b).unwrap().pack; n * p.point_width + p.point_pad() };
        let fold = &fbody[..fbody.len() - points(&fbody)];
        let iold = &ibody[..ibody.len() - points(&ibody)];
        let fcols = [ColumnReader::<f64>::open(&fbody).unwrap(), ColumnReader::<f64>::open(fold).unwrap()];
        let icols = [ColumnReader::<i64>::open(&ibody).unwrap(), ColumnReader::<i64>::open(iold).unwrap()];
        assert!(fcols[0].has_points() && !fcols[1].has_points());
        let franges = [
            (f64::NEG_INFINITY, f64::INFINITY),
            (-5.0, 5.0),
            (0.0, 0.0),
            (-0.0, -0.0),
            (100.0, 1200.0),
            (-20.0, 79.9),
            (3.0, 2.0),
            (f64::INFINITY, f64::INFINITY),
            (f64::MIN, f64::MAX),
            (f64::NAN, 1.0),
        ];
        for (lo, hi) in franges {
            let want = brute_bits(&f, lo, hi);
            for c in &fcols {
                let mut got = vec![0u64; n.div_ceil(64)];
                c.range_bits(lo, hi, &mut got);
                assert_eq!(got, want, "f64 [{lo}, {hi}] points={}", c.has_points());
                assert_eq!(c.count_range(lo, hi), want.iter().map(|w| w.count_ones() as u64).sum::<u64>());
            }
        }
        for (lo, hi) in [(i64::MIN, i64::MAX), (-10, 10), (0, 1999), (500, 700), (-200, -150), (5, 4), (i64::MAX, i64::MAX)] {
            let want = brute_bits(&iv, lo, hi);
            for c in &icols {
                let mut got = vec![0u64; n.div_ceil(64)];
                c.range_bits(lo, hi, &mut got);
                assert_eq!(got, want, "i64 [{lo}, {hi}] points={}", c.has_points());
            }
        }
    }

    /// A column written without a points index (skipper only) answers
    /// ranges, counts and presence exactly like one with it, dense or sparse.
    #[test]
    fn skipper_columns_match_points_columns() {
        for sparse in [false, true] {
            let n = 1000u32;
            let present = |d: u32| !sparse || d % 4 != 1;
            let mut with = FastFields::new(0);
            let mut without = FastFields::new(0);
            without.skip_points(2);
            for ff in [&mut with, &mut without] {
                for d in 0..n {
                    if present(d) {
                        // Index-sorted descending timestamps with repeats.
                        ff.i64_column_mut(2).set(d, 50_000 - (d as i64 / 3) * 7);
                    }
                }
                ff.set_num_docs(n);
                ff.build_all();
            }
            let (a, b) = (encode_fast_fields(&with), encode_fast_fields(&without));
            assert!(b.len() < a.len());
            let (ra, rb) = (FastFieldsReader::open(&a).unwrap(), FastFieldsReader::open(&b).unwrap());
            let (ca, cb) = (ra.get_i64(2).unwrap(), rb.get_i64(2).unwrap());
            assert!(!ca.skips_points() && cb.skips_points() && !cb.has_points());
            assert_eq!(ca.is_sparse(), sparse);
            assert_eq!(cb.is_sparse(), sparse);
            for (lo, hi) in [(i64::MIN, i64::MAX), (48_000, 49_000), (0, 10), (50_000, 50_000), (47_669, 47_676)] {
                let words = (n as usize).div_ceil(64);
                let (mut wa, mut wb) = (vec![0u64; words], vec![0u64; words]);
                ca.range_bits(lo, hi, &mut wa);
                cb.range_bits(lo, hi, &mut wb);
                assert_eq!(wa, wb, "[{lo}, {hi}] sparse={sparse}");
                assert_eq!(ca.count_range(lo, hi), cb.count_range(lo, hi));
            }
            for d in 0..n {
                assert_eq!(cb.get(d), ca.get(d));
            }
        }
    }

    /// A single-valued column with missing docs keeps its presence through
    /// encode/decode: `get`, range hits (a missing doc is not the padding 0),
    /// counts and min/max see only docs that have a value.
    #[test]
    fn sparse_columns_keep_presence() {
        let n = 700u32; // > 2 blocks, > 10 presence words
        let mut ff = FastFields::new(0);
        let present = |d: u32| d % 3 == 0 || d == 699;
        for d in 0..n {
            if present(d) {
                ff.f64_column_mut(1).set(d, if d == 0 { -0.5 } else { d as f64 });
                ff.i64_column_mut(2).set(d, d as i64 - 300);
                ff.u64_column_mut(3).set(d, d as u64);
            }
        }
        ff.set_num_docs(n);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();
        let f = r.get_f64(1).unwrap();
        let i = r.get_i64(2).unwrap();
        let u = r.get_u64(3).unwrap();
        assert!(f.is_sparse() && i.is_sparse() && u.is_sparse());
        for d in 0..n {
            assert_eq!(f.has_value(d), present(d), "doc {d}");
            assert_eq!(f.get(d).is_some(), present(d), "doc {d}");
            assert_eq!(i.get(d).is_some(), present(d));
            assert_eq!(u.get(d).is_some(), present(d));
        }
        assert!(!f.has_value(n));
        // Ranges around 0 must not pick up the missing docs (padding value 0).
        let want_f = |lo: f64, hi: f64| -> Vec<u32> { (0..n).filter(|&d| present(d) && { let v = if d == 0 { -0.5 } else { d as f64 }; v >= lo && v <= hi }).collect() };
        for (lo, hi) in [(-1.0, 1.0), (0.0, 0.0), (-1e300, 1e300), (10.0, 100.0), (f64::NEG_INFINITY, f64::INFINITY), (-0.5, -0.5)] {
            let mut w = vec![0u64; (n as usize).div_ceil(64)];
            f.range_bits(lo, hi, &mut w);
            let got: Vec<u32> = (0..n).filter(|&d| w[d as usize / 64] >> (d % 64) & 1 == 1).collect();
            assert_eq!(got, want_f(lo, hi), "f64 [{lo}, {hi}]");
            assert_eq!(f.count_range(lo, hi), got.len() as u64);
            assert_eq!(f.range_query(lo, hi), got);
        }
        let mut w = vec![0u64; (n as usize).div_ceil(64)];
        i.range_bits(-5, 5, &mut w);
        let got: Vec<u32> = (0..n).filter(|&d| w[d as usize / 64] >> (d % 64) & 1 == 1).collect();
        assert_eq!(got, (0..n).filter(|&d| present(d) && (d as i64 - 300).abs() <= 5).collect::<Vec<_>>());
        assert_eq!(f.min(), Some(-0.5));
        assert_eq!(f.max(), Some(699.0));
        assert_eq!(u.min(), Some(0));
        // A dense column carries no presence bitmap.
        let mut dense = FastFields::new(0);
        for d in 0..10 {
            dense.f64_column_mut(1).set(d, d as f64);
        }
        dense.set_num_docs(10);
        dense.build_all();
        let b = encode_fast_fields(&dense);
        let c = FastFieldsReader::open(&b).unwrap().get_f64(1).unwrap();
        assert!(!c.is_sparse() && (0..10).all(|d| c.has_value(d)));
    }

    #[test]
    fn set_bit_range_edges() {
        for (a, b) in [(0, 0), (0, 1), (3, 64), (0, 64), (63, 65), (10, 200), (64, 128), (5, 191)] {
            let mut w = vec![0u64; 4];
            set_bit_range(&mut w, a, b);
            for i in 0..256 {
                assert_eq!((w[i / 64] >> (i % 64)) & 1 == 1, i >= a && i < b, "[{a}, {b}) bit {i}");
            }
        }
    }

    #[test]
    fn numeric_roundtrip_and_range() {
        let mut ff = FastFields::new(0);
        for i in 0..1000u32 {
            ff.i64_column_mut(0).set(i, i as i64 * 3 - 500);
            ff.f64_column_mut(1).set(i, i as f64 / 2.0);
            ff.u64_column_mut(2).set(i, i as u64);
        }
        ff.set_num_docs(1000);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();

        let c = r.get_i64(0).unwrap();
        assert_eq!(c.num_docs(), 1000);
        assert_eq!(c.get(0), Some(-500));
        assert_eq!(c.get(999), Some(999 * 3 - 500));
        assert_eq!(c.get(1000), None);
        assert_eq!(c.min(), Some(-500));
        assert_eq!(c.max(), Some(2497));
        let expected: Vec<DocId> = (0..1000u32).filter(|&i| {
            let v = i as i64 * 3 - 500;
            (0..=100).contains(&v)
        }).collect();
        assert_eq!(c.range_query(0, 100), expected);
        assert_eq!(c.count_in_range(0, 100), expected.len() as u32);
        assert_eq!(c.count_in_range(100_000, 200_000), 0);

        let f = r.get_f64(1).unwrap();
        assert_eq!(f.get(7), Some(3.5));
        assert_eq!(f.count_in_range(10.0, 20.0), 21);

        let u = r.get_u64(2).unwrap();
        assert_eq!(u.range_query(990, 2000).len(), 10);
        assert!(r.get_i64(1).is_none());
    }

    #[test]
    fn owned_keyword_column_survives_source_drop_and_handles_missing_values() {
        let mut ff = FastFields::new(0);
        ff.keyword_column_mut(3).set(0, "alpha");
        ff.keyword_column_mut(3).set(2, "βeta");
        ff.set_num_docs(3);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let reader = FastFieldsReader::open(&bytes).unwrap();
        let data = reader.keyword_column_bytes(3).unwrap();
        let table = keyword_term_table(data).unwrap();
        let owned = OwnedKeywordColumn::open(super::super::directory::FileSlice::from_vec(data.to_vec()), Some(table)).unwrap();
        drop(bytes);
        assert_eq!(owned.term(owned.ord(0)).as_deref(), Some("alpha"));
        assert_eq!(owned.term(owned.ord(2)).as_deref(), Some("βeta"));
        assert_eq!(owned.ord(1), MISSING_ORD);
        assert_eq!(owned.ord(99), MISSING_ORD);
        assert_eq!(owned.term(MISSING_ORD), None);
        assert!(OwnedKeywordColumn::open(super::super::directory::FileSlice::from_vec(vec![255]), None).is_err());
    }

    #[test]
    fn bool_and_keyword_roundtrip() {
        let mut ff = FastFields::new(0);
        let fruits = ["apple", "banana", "cherry"];
        for i in 0..200u32 {
            ff.bool_column_mut(5).set(i, i % 3 == 0);
            if i % 10 != 9 {
                ff.keyword_column_mut(6).set(i, fruits[(i % 3) as usize]);
            }
        }
        ff.set_num_docs(200);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();

        let b = r.get_bool(5).unwrap();
        assert_eq!(b.count_true(), 67);
        assert!(b.get(0) && b.get(3) && !b.get(1) && !b.get(500));
        assert_eq!(b.true_docs()[..3], [0, 3, 6]);

        // A reader over a shared, already parsed term table behaves the same.
        let col = r.keyword_column_bytes(6).unwrap();
        let shared = KeywordReader::open_with(col, Some(keyword_term_table(col).unwrap())).unwrap();
        let k0 = r.get_keyword(6).unwrap();
        assert!((0..210).all(|d| shared.get(d) == k0.get(d) && shared.ord(d) == k0.ord(d)));
        assert_eq!(shared.seek_ord("b"), (1, false));
        assert_eq!(shared.seek_ord("banana"), (1, true));
        assert_eq!(shared.seek_ord("zzz"), (3, false));
        let blocks = shared.ord_blocks();
        assert_eq!(blocks.len(), 1);
        assert_eq!((blocks[0].min, blocks[0].max, blocks[0].missing), (0, 2, true));
        assert!(KeywordReader::open_with(col, Some(std::sync::Arc::new(KeywordOffsets { indexed: None, count: 1, encoded_len: 1, fc: false, packed: false, heads: std::sync::OnceLock::new() }))).is_err());
        let k = r.get_keyword(6).unwrap();
        assert_eq!(k.num_terms(), 3);
        assert_eq!(k.get(0).as_deref(), Some("apple"));
        assert_eq!(k.get(1).as_deref(), Some("banana"));
        assert_eq!(k.get(9), None);
        assert_eq!(k.lookup_ord("cherry"), Some(2));
        assert_eq!(k.lookup_ord("durian"), None);
        assert_eq!(k.exact_match("banana").len(), 60);
        assert_eq!(k.range_query("banana", "cherry").len(), 120);
        assert_eq!(k.range_query("a", "apple").len(), 60);
        assert_eq!(k.range_query("x", "z").len(), 0);
        let counts = k.ordinal_counts(|_| true);
        assert_eq!(counts, vec![60, 60, 60]);
        assert_eq!(r.fields(), vec![(5, TAG_BOOL), (6, TAG_KEYWORD)]);
    }

    #[test]
    fn empty_fast_fields() {
        let ff = FastFields::new(0);
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();
        assert!(r.get_i64(0).is_none());
        assert!(r.fields().is_empty());
    }

    /// A norms column whose docs all have the same length is one byte from
    /// v27 (Lucene's constant norms) and reads as the per-doc column it
    /// replaces; before v27, and for any other column, it stays per doc.
    #[test]
    fn constant_norms_roundtrip() {
        for (lengths, docs) in [(vec![3u32], 500u32), (vec![1, 2], 300), (vec![7], 1)] {
            let mut ff = FastFields::new(0);
            for i in 0..docs {
                ff.norms_mut(4).set(i, lengths[i as usize % lengths.len()]);
            }
            ff.set_num_docs(docs);
            let constant = lengths.len() == 1;
            for version in [crate::codec::CONSTANT_NORMS_SINCE - 1, crate::codec::CONSTANT_NORMS_SINCE] {
                let bytes = encode_fast_fields_versioned(&ff, docs, version);
                let r = FastFieldsReader::open(&bytes).unwrap();
                let n = r.get_norms(4).unwrap();
                let small = version >= crate::codec::CONSTANT_NORMS_SINCE && constant;
                assert_eq!(n.constant_run().is_some(), small, "{lengths:?} v{version}");
                if small {
                    assert!(bytes.len() < 64, "{} bytes", bytes.len());
                }
                assert_eq!(n.num_docs(), docs);
                assert_eq!(n.bytes().len(), docs as usize);
                for d in 0..docs {
                    assert_eq!(n.get(d), lengths[d as usize % lengths.len()]);
                }
                assert_eq!(n.get(docs + 5), 0);
                assert_eq!(n.total(), (0..docs).map(|d| lengths[d as usize % lengths.len()] as u64).sum::<u64>());
                let parts = column_parts(if small { TAG_CONST_NORMS } else { TAG_NORMS }, &r.data[r.columns[0].start..r.columns[0].end]).unwrap();
                assert!(parts.norms > 0);
            }
        }
        // A run longer than a smaller one already handed out keeps both right.
        let mut ff = FastFields::new(0);
        for i in 0..10_000u32 { ff.norms_mut(1).set(i, 2); }
        ff.set_num_docs(10_000);
        let bytes = encode_fast_fields_versioned(&ff, 10_000, crate::codec::CONSTANT_NORMS_SINCE);
        let r = FastFieldsReader::open(&bytes).unwrap();
        assert!(r.get_norms(1).unwrap().bytes().iter().all(|&b| b == crate::query::bm25::int_to_byte4(2)));
    }

    #[test]
    fn norms_roundtrip() {
        let mut ff = FastFields::new(0);
        for i in 0..300u32 {
            ff.norms_mut(7).set(i, i % 13 + 1);
        }
        ff.set_num_docs(300);
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();
        let n = r.get_norms(7).unwrap();
        assert_eq!(n.num_docs(), 300);
        assert_eq!(n.get(0), 1);
        assert_eq!(n.get(12), 13);
        assert_eq!(n.get(13), 1);
        assert_eq!(n.get(999), 0);
        assert_eq!(n.get_byte(12), crate::query::bm25::int_to_byte4(13));
        assert_eq!(n.bytes().len(), 300);
        let total: u64 = (0..300u64).map(|i| i % 13 + 1).sum();
        assert_eq!(n.total(), total);
        assert!((n.avg() - total as f32 / 300.0).abs() < 1e-3);
        assert!(r.get_norms(8).is_none());
        assert!(r.get_i64(7).is_none());
    }

    #[test]
    fn multi_numeric_roundtrip_and_range() {
        // 2000 docs, 0-3 values each; deterministic pseudo-random.
        let mut ff = FastFields::new(0);
        let mut expected: Vec<Vec<i64>> = Vec::with_capacity(2000);
        let mut seed = 0x243F6A88870D6E1Du64;
        for d in 0..2000u32 {
            let n = (d % 4) as usize; // 0..=3 values
            let mut vals = Vec::with_capacity(n);
            for _ in 0..n {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                vals.push((seed as i64 % 1000) - 500);
            }
            for v in &vals {
                ff.multi_i64_column_mut(0).push(d, *v);
            }
            expected.push(vals);
        }
        // f64 and u64 with a few docs having no values.
        for d in 0..2000u32 {
            if d % 5 != 0 {
                ff.multi_f64_column_mut(1).push(d, d as f64 * 0.25);
                if d % 7 != 0 {
                    ff.multi_f64_column_mut(1).push(d, d as f64 * 0.25 + 1.0);
                }
            }
            if d % 3 != 0 {
                ff.multi_u64_column_mut(2).push(d, d as u64 * 10);
            }
        }
        ff.set_num_docs(2000);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();

        let c = r.get_multi_i64(0).unwrap();
        assert_eq!(c.num_docs(), 2000);
        assert_eq!(c.total_values(), expected.iter().map(|v| v.len()).sum::<usize>());
        for d in (0..2000u32).step_by(37) {
            assert_eq!(c.values_vec(d), expected[d as usize], "doc {d}");
            assert_eq!(c.count(d) as usize, expected[d as usize].len());
        }
        assert_eq!(c.values_vec(0), Vec::<i64>::new()); // doc 0 has 0 values
        // Sort modes.
        let d = 100u32;
        let vals = &expected[d as usize];
        if !vals.is_empty() {
            assert_eq!(c.min_of(d), vals.iter().min().copied());
            assert_eq!(c.max_of(d), vals.iter().max().copied());
            assert_eq!(c.sum_of(d), vals.iter().sum::<i64>());
            assert!((c.avg_of(d).unwrap() - vals.iter().sum::<i64>() as f64 / vals.len() as f64).abs() < 1e-9);
        }
        assert_eq!(c.min_of(0), None);
        assert_eq!(c.avg_of(0), None);

        // Any-value range semantics against brute force.
        for (lo, hi) in [(-500i64, 500), (-100, 100), (499, 500), (1000, 2000), (-1000, -999)] {
            let expected_docs: Vec<DocId> = (0..2000u32)
                .filter(|&d| expected[d as usize].iter().any(|v| *v >= lo && *v <= hi))
                .collect();
            assert_eq!(c.range_query(lo, hi), expected_docs, "range [{lo},{hi}]");
            assert_eq!(c.count_in_range(lo, hi), expected_docs.len() as u32);
        }

        let f = r.get_multi_f64(1).unwrap();
        assert_eq!(f.values_vec(0), Vec::<f64>::new());
        assert_eq!(f.values_vec(1), vec![0.25, 1.25]);
        assert_eq!(f.values_vec(7), vec![1.75]); // 7 % 7 == 0 → second value skipped
        let expected_f: Vec<DocId> = (0..2000u32)
            .filter(|&d| {
                let v1 = d as f64 * 0.25;
                let v2 = v1 + 1.0;
                (d % 5 != 0) && ((v1 >= 10.0 && v1 <= 20.0) || (d % 7 != 0 && v2 >= 10.0 && v2 <= 20.0))
            })
            .collect();
        assert_eq!(f.range_query(10.0, 20.0), expected_f);

        let u = r.get_multi_u64(2).unwrap();
        assert_eq!(u.values_vec(0), Vec::<u64>::new());
        assert_eq!(u.values_vec(3), Vec::<u64>::new()); // 3 % 3 == 0 → no value
        assert_eq!(u.values_vec(4), vec![40]);
        let expected_u: Vec<DocId> = (0..2000u32).filter(|&d| d % 3 != 0 && d * 10 >= 900 && d * 10 <= 1000).collect();
        assert_eq!(u.range_query(900, 1000), expected_u);

        assert!(r.get_i64(0).is_none());
        assert!(r.get_multi_i64(1).is_none());
        assert!(r.fields().contains(&(0, TAG_MULTI_I64)));
        assert!(r.fields().contains(&(1, TAG_MULTI_F64)));
        assert!(r.fields().contains(&(2, TAG_MULTI_U64)));
    }

    #[test]
    fn feature_columns_read_back_the_same_values_and_are_smaller() {
        use crate::fieldtypes::features::{encode, quantize};
        // sparse_vector doc values: ~80 of ~3,000 tokens a doc, quantized
        // weights; field 1 is an ordinary keyword column next to it.
        let mut ff = FastFields::new(0);
        let n = 2000u32;
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        for d in 0..n {
            if d % 97 == 5 {
                continue;
            }
            for _ in 0..80 {
                let tok = format!("{}", rnd() % 3000);
                let w = quantize(0.01 + (rnd() % 100_000) as f32 / 30_000.0);
                ff.multi_keyword_column_mut(0).push(d, &encode(&tok, w));
            }
            ff.multi_keyword_column_mut(1).push(d, &format!("v{}", d % 13));
        }
        ff.set_num_docs(n);
        ff.build_all();
        let old = encode_fast_fields_versioned(&ff, n, crate::codec::FEATURES_COLUMN_SINCE - 1);
        let new = encode_fast_fields_versioned(&ff, n, crate::codec::FEATURES_COLUMN_SINCE);
        assert!(new.len() * 3 < old.len() * 2, "features {} vs plain {}", new.len(), old.len());
        let (or, nr) = (FastFieldsReader::open(&old).unwrap(), FastFieldsReader::open(&new).unwrap());
        assert!(nr.get_multi_keyword(0).is_none() && or.get_features(0).is_none());
        assert!(nr.get_multi_keyword(1).is_some() && nr.get_features(1).is_none(), "a non-feature column stays plain");
        let (plain, packed) = (or.get_multi_keyword(0).unwrap(), nr.get_features(0).unwrap());
        for d in 0..n + 2 {
            let want: Vec<String> = plain.values(d).into_iter().map(str::to_string).collect();
            assert_eq!(packed.values(d), want, "doc {d}");
            assert_eq!(packed.count(d), want.len() as u32);
        }
        // A weight with low bits set is not a feature value: plain column.
        let mut ff = FastFields::new(0);
        ff.multi_keyword_column_mut(0).push(0, &encode("a", 0.123_456_7));
        ff.set_num_docs(1);
        ff.build_all();
        let r = encode_fast_fields_versioned(&ff, 1, crate::codec::FEATURES_COLUMN_SINCE);
        assert!(FastFieldsReader::open(&r).unwrap().get_features(0).is_none());
    }

    #[test]
    fn feature_columns_merge_like_their_values() {
        use crate::fieldtypes::features::{encode, quantize};
        let col = |seed: u64, n: u32, toks: u64| {
            let mut ff = FastFields::new(0);
            let mut x = seed;
            for d in 0..n {
                for _ in 0..(d % 7) * 5 {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    ff.multi_keyword_column_mut(0).push(d, &encode(&format!("t{}", x % toks), quantize(0.01 + (x % 9999) as f32 / 777.0)));
                }
            }
            ff.set_num_docs(n);
            ff.build_all();
            encode_fast_fields_versioned(&ff, n, crate::codec::FEATURES_COLUMN_SINCE)
        };
        let (a, b) = (col(7, 300, 500), col(11, 200, 900));
        let (ra, rb) = (FastFieldsReader::open(&a).unwrap(), FastFieldsReader::open(&b).unwrap());
        let inputs = vec![ra.get_features(0), None, rb.get_features(0)];
        // Every third doc of `a` deleted; `b`'s docs after `a`'s live ones.
        let a_live: Vec<DocId> = (0..300).filter(|d| d % 3 != 0).collect();
        let new_id = |i: usize, old: DocId| -> Option<DocId> {
            match i {
                0 => a_live.iter().position(|&d| d == old).map(|p| p as DocId),
                2 => Some(a_live.len() as DocId + old),
                _ => None,
            }
        };
        let n = a_live.len() as u32 + 200;
        let merged = merge_features(&inputs, &new_id, n).unwrap();
        let m = FeaturesReader::open(&merged).unwrap();
        for (new, &old) in a_live.iter().enumerate() {
            assert_eq!(m.values(new as DocId), inputs[0].as_ref().unwrap().values(old), "a doc {old}");
        }
        for old in 0..200 {
            assert_eq!(m.values(a_live.len() as DocId + old), inputs[2].as_ref().unwrap().values(old), "b doc {old}");
        }
    }

    #[test]
    fn packed_multi_keyword_reads_like_plain_and_is_smaller() {
        // A metrics-shaped column: every doc lists the same ~100 values (a
        // node's addresses), some docs a few more, some none.
        let mut ff = FastFields::new(0);
        let shared: Vec<String> = (0..109).map(|i| format!("10.{}.{}.1", i / 10, i % 10)).collect();
        let n = 3000u32;
        for d in 0..n {
            if d % 101 == 7 {
                continue;
            }
            for v in &shared {
                ff.multi_keyword_column_mut(0).push(d, v);
            }
            for k in 0..(d % 4) {
                ff.multi_keyword_column_mut(0).push(d, &format!("extra-{}", (d + k) % 37));
            }
        }
        ff.set_num_docs(n);
        ff.build_all();
        let plain_bytes = encode_fast_fields_versioned(&ff, n, crate::codec::PACKED_MULTI_KEYWORD_SINCE - 1);
        let packed_bytes = encode_fast_fields_versioned(&ff, n, crate::codec::PACKED_MULTI_KEYWORD_SINCE);
        assert!(packed_bytes.len() * 3 < plain_bytes.len(), "packed {} vs plain {}", packed_bytes.len(), plain_bytes.len());
        let pr = FastFieldsReader::open(&plain_bytes).unwrap();
        let kr = FastFieldsReader::open(&packed_bytes).unwrap();
        let (plain, packed) = (pr.get_multi_keyword(0).unwrap(), kr.get_multi_keyword(0).unwrap());
        assert!(!plain.is_packed() && packed.is_packed());
        assert_eq!((packed.num_docs(), packed.num_terms(), packed.total_values()), (plain.num_docs(), plain.num_terms(), plain.total_values()));
        for d in 0..n + 2 {
            assert_eq!(packed.count(d), plain.count(d), "doc {d}");
            assert_eq!(packed.ords(d).collect::<Vec<_>>(), plain.ords(d).collect::<Vec<_>>(), "doc {d}");
            assert_eq!(packed.values(d), plain.values(d), "doc {d}");
        }
        for t in ["10.3.4.1", "extra-5", "nope"] {
            assert_eq!(packed.exact_match(t), plain.exact_match(t), "{t}");
        }
        assert_eq!(packed.range_query("10.2", "10.5"), plain.range_query("10.2", "10.5"));
        assert_eq!(packed.ordinal_counts(|d| d % 3 == 0), plain.ordinal_counts(|d| d % 3 == 0));
        // A single-term column packs its ordinals in zero bits.
        let mut one = FastFields::new(0);
        for d in 0..100u32 {
            one.multi_keyword_column_mut(1).push(d, "only");
            if d % 2 == 0 {
                one.multi_keyword_column_mut(1).push(d, "only");
            }
        }
        one.set_num_docs(100);
        one.build_all();
        let bytes = encode_fast_fields_versioned(&one, 100, crate::codec::VERSION);
        let r = FastFieldsReader::open(&bytes).unwrap();
        let k = r.get_multi_keyword(1).unwrap();
        assert!(k.is_packed());
        assert!((0..100).all(|d| k.values(d) == vec!["only"]));
        // Truncated or inconsistent packed columns are refused.
        let body = kr.multi_keyword_column_bytes(0).unwrap();
        assert!(MultiKeywordReader::open(&body[..body.len() - 9]).is_err());
    }

    #[test]
    fn multi_keyword_roundtrip_sorted_set() {
        let mut ff = FastFields::new(0);
        let words = ["banana", "apple", "cherry", "date"];
        // Ordinals follow sorted term order, not `words` order.
        let mut sorted = words;
        sorted.sort_unstable();
        for d in 0..100u32 {
            let n = (d % 3) as usize; // 0..=2 values
            for k in 0..n {
                // Deliberately unsorted + duplicated pushes.
                ff.multi_keyword_column_mut(0).push(d, words[(d as usize + k) % 4]);
                if k == 0 {
                    ff.multi_keyword_column_mut(0).push(d, words[(d as usize + k) % 4]); // dup
                }
            }
        }
        ff.set_num_docs(100);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();
        let k = r.get_multi_keyword(0).unwrap();
        assert_eq!(k.num_docs(), 100);
        assert_eq!(k.num_terms(), 4);
        assert_eq!(k.lookup_ord("apple"), Some(0));
        assert_eq!(k.lookup_ord("date"), Some(3));
        assert_eq!(k.lookup_ord("fig"), None);
        assert_eq!(k.term(1), Some("banana"));

        for d in 0..100u32 {
            let n = (d % 3) as usize;
            let mut expected: Vec<&str> = (0..n).map(|k| words[(d as usize + k) % 4]).collect();
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(k.values(d), expected, "doc {d}");
            assert_eq!(k.count(d) as usize, expected.len());
            // Per-doc ordinals sorted ascending.
            let ords: Vec<u32> = k.ords(d).collect();
            assert!(ords.windows(2).all(|w| w[0] < w[1]), "doc {d}");
        }
        assert!(k.values(0).is_empty());
        assert_eq!(k.count(0), 0);

        // exact_match / range_query: any-value semantics, deduped docs.
        for d in 0..100u32 {
            let n = (d % 3) as usize;
            let has: Vec<&str> = (0..n).map(|k| words[(d as usize + k) % 4]).collect();
            let has = |w: &str| has.iter().any(|x| *x == w);
            for w in words {
                if has(w) {
                    assert!(k.exact_match(w).contains(&d), "{w} doc {d}");
                } else {
                    assert!(!k.exact_match(w).contains(&d), "{w} doc {d}");
                }
            }
        }
        let in_range: Vec<DocId> = (0..100u32)
            .filter(|&d| {
                let n = (d % 3) as usize;
                (0..n).any(|k| {
                    let w = words[(d as usize + k) % 4];
                    w >= "apple" && w <= "banana"
                })
            })
            .collect();
        assert_eq!(k.range_query("apple", "banana"), in_range);
        assert!(k.range_query("x", "z").is_empty());

        // ordinal_counts: each doc counts once per distinct ord.
        let counts = k.ordinal_counts(|_| true);
        let mut expected_counts = [0u64; 4];
        for d in 0..100u32 {
            let n = (d % 3) as usize;
            let mut set: Vec<&str> = (0..n).map(|k| words[(d as usize + k) % 4]).collect();
            set.sort_unstable();
            set.dedup();
            for w in set {
                expected_counts[sorted.iter().position(|x| *x == w).unwrap()] += 1;
            }
        }
        assert_eq!(counts, expected_counts.to_vec());
        // With an accept filter.
        let counts = k.ordinal_counts(|d| d % 2 == 0);
        let mut expected_counts = [0u64; 4];
        for d in (0..100u32).step_by(2) {
            let n = (d % 3) as usize;
            let mut set: Vec<&str> = (0..n).map(|k| words[(d as usize + k) % 4]).collect();
            set.sort_unstable();
            set.dedup();
            for w in set {
                expected_counts[sorted.iter().position(|x| *x == w).unwrap()] += 1;
            }
        }
        assert_eq!(counts, expected_counts.to_vec());
    }

    #[test]
    fn multi_numeric_over_256_values() {
        // > 256 total values forces multiple min/max blocks.
        let mut ff = FastFields::new(0);
        for d in 0..100u32 {
            for k in 0..5 {
                ff.multi_i64_column_mut(0).push(d, d as i64 * 10 + k as i64);
            }
        }
        ff.set_num_docs(100);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();
        let c = r.get_multi_i64(0).unwrap();
        assert_eq!(c.total_values(), 500);
        assert_eq!(c.values_vec(0), (0..5).collect::<Vec<i64>>());
        assert_eq!(c.values_vec(99), (990..995).collect::<Vec<i64>>());
        // Range spanning several blocks.
        let expected: Vec<DocId> = (0..100u32).filter(|&d| (d * 10..d * 10 + 5).any(|v| v >= 490 && v <= 510)).collect();
        assert_eq!(c.range_query(490, 510), expected);
        assert_eq!(c.count_in_range(490, 510), expected.len() as u32);
    }

    #[test]
    fn sparse_column_padded_with_zero() {
        let mut ff = FastFields::new(0);
        ff.i64_column_mut(0).set(2, 7);
        ff.set_num_docs(5);
        ff.build_all();
        let bytes = encode_fast_fields(&ff);
        let r = FastFieldsReader::open(&bytes).unwrap();
        let c = r.get_i64(0).unwrap();
        assert_eq!(c.num_docs(), 5);
        assert_eq!(c.get(2), Some(7));
        // Docs without a value are absent, not the padding `0`.
        assert_eq!((c.get(0), c.get(4)), (None, None));
        assert!(c.has_value(2) && !c.has_value(4));
    }
}
