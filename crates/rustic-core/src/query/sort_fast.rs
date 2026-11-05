//! Top-k by one field (Lucene's `TopFieldCollector` with a
//! `NumericComparator` / `TermOrdValComparator` and their competitive
//! iterators).
//!
//! Per leaf, each doc's sort value maps to an order-preserving `u128` key
//! with the direction and the missing-value placement folded in, so
//! collection compares integers: no per-doc `SortValue`, and keywords
//! compare by segment ordinal instead of by string.
//!
//! Once the queue is full and total-hit counting is over (the
//! `track_total_hits` bound is reached, or hits are not tracked), whole
//! [`BLOCK_SIZE`]-doc blocks whose value range cannot beat the queue's
//! bottom, or lies entirely at or before `search_after`, are skipped by
//! seeking the query past them. Block bounds come from the numeric column's
//! min/max table, or for a keyword column from per-block ordinal bounds
//! built once per segment (Lucene's `DocValuesSkipper`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::codec::fastfields::{
    BLOCK_SIZE, ColumnReader, KeywordReader, MISSING_ORD, MultiColumnReader, MultiKeywordReader,
    OrdBlock,
};
use crate::query::collector::SortValue;
use crate::query::docset::TERMINATED;
use crate::query::reader::LeafReader;
use crate::query::scorer::Scorer;
use crate::query::searcher::{SortField, SortKind, SortMissing, SortMode};
use crate::types::DocId;

const SIGN: u64 = 1 << 63;
/// Key class of a doc with a value (missing-first docs are class 0,
/// missing-last class 2).
const VALUE_CLASS: u128 = 1 << 64;
const LAST_CLASS: u128 = 2 << 64;

/// Order-preserving map of an i64 onto u64.
#[inline]
fn o_i64(v: i64) -> u64 {
    (v as u64) ^ SIGN
}

/// Order-preserving map of an f64 onto u64 (`-0.0` folded onto `0.0`, as
/// the sort comparator treats them as equal).
#[inline]
fn o_f64(v: f64) -> u64 {
    let v = if v == 0.0 { 0.0 } else { v };
    let b = v.to_bits();
    if b & SIGN != 0 { !b } else { b | SIGN }
}

/// A leaf's sort column.
enum Src<'a> {
    I64(ColumnReader<'a, i64>),
    /// `unsigned_long`: sorts as the signed long shifted by `Long.MIN_VALUE`,
    /// i.e. by the raw u64.
    U64(ColumnReader<'a, u64>),
    F64(ColumnReader<'a, f64>),
    MI64(MultiColumnReader<'a, i64>),
    MU64(MultiColumnReader<'a, u64>),
    MF64(MultiColumnReader<'a, f64>),
    /// Keyword column, its block bounds, and the segment / field (for the
    /// ordinal index, built on demand).
    Kw(KeywordReader<'a>, Arc<crate::admission::Allocation<Vec<OrdBlock>>>, &'a crate::segment::Segment, u16),
    MKw(MultiKeywordReader<'a>),
    /// The leaf has no column for the field: every doc is missing.
    Absent,
    /// Distance from `(lat, lon)` to the doc's points (ES `_geo_distance`;
    /// docs without a point are at +inf). `cursor` walks `points` along
    /// the ascending docs of collection.
    Geo {
        points: &'a crate::geo::GeoValues,
        cursor: std::cell::Cell<usize>,
        lat: f64,
        lon: f64,
        plane: bool,
        blocks: Arc<crate::admission::Allocation<Vec<crate::geo::GeoBlock>>>,
    },
}

/// How one leaf evaluates a single-field sort.
pub(crate) struct LeafSort<'a> {
    src: Src<'a>,
    desc: bool,
    /// Multi-valued columns: sort by the largest value (else the smallest).
    pick_max: bool,
    /// Key of a doc without a value.
    missing: u128,
    /// Keyword `missing` value that is not a term of the leaf: its lower
    /// bound and the string.
    missing_gap: Option<(u32, String)>,
    num_docs: u32,
    /// The segment and field of a numeric column (its cached block
    /// ranking, [`crate::segment::Segment::sort_block_ranking`]).
    seg: Option<(&'a crate::segment::Segment, u16)>,
}

impl<'a> LeafSort<'a> {
    /// The plan for `sf` on `leaf`, or `None` when the generic collector
    /// must run (memtable leaves, sum/avg/median over multi-valued columns,
    /// boolean columns, a `missing` value of another type).
    pub(crate) fn new(leaf: &LeafReader<'a>, sf: &SortField) -> Option<Self> {
        if let SortField::GeoDistance { field, lat, lon, descending, mode, plane } = sf {
            let LeafReader::Segment(seg) = *leaf else {
                return None;
            };
            let mode = mode.unwrap_or(if *descending { SortMode::Max } else { SortMode::Min });
            if !matches!(mode, SortMode::Min | SortMode::Max) {
                return None;
            }
            let src = match seg.geo_index(*field) {
                Some(g) => Src::Geo {
                    points: g.values(),
                    cursor: std::cell::Cell::new(0),
                    lat: *lat,
                    lon: *lon,
                    plane: *plane,
                    blocks: seg.geo_blocks(*field, BLOCK_SIZE)?,
                },
                None => Src::Absent,
            };
            let mut ls = LeafSort { src, desc: *descending, pick_max: mode == SortMode::Max, missing: 0, missing_gap: None, num_docs: seg.num_docs, seg: None };
            ls.missing = ls.value_key(o_f64(f64::INFINITY));
            return Some(ls);
        }
        let SortField::Field {
            field,
            kind,
            descending,
            missing,
            mode,
        } = sf
        else {
            return None;
        };
        let LeafReader::Segment(seg) = *leaf else {
            return None;
        };
        let field = *field;
        let mode = mode.unwrap_or(if *descending {
            SortMode::Max
        } else {
            SortMode::Min
        });
        let multi_ok = matches!(mode, SortMode::Min | SortMode::Max);
        let ff = seg.fast_fields();
        let src = match kind {
            SortKind::Fielddata => return None,
            SortKind::I64 => {
                if let Some(c) = ff.get_i64(field) {
                    Src::I64(c)
                } else if let Some(c) = ff.get_multi_i64(field) {
                    if !multi_ok {
                        return None;
                    }
                    Src::MI64(c)
                } else if ff.get_bool(field).is_some() {
                    return None;
                } else if let Some(c) = ff.get_u64(field) {
                    Src::U64(c)
                } else if let Some(c) = ff.get_multi_u64(field) {
                    if !multi_ok {
                        return None;
                    }
                    Src::MU64(c)
                } else {
                    Src::Absent
                }
            }
            SortKind::F64 => {
                if let Some(c) = ff.get_f64(field) {
                    Src::F64(c)
                } else if let Some(c) = ff.get_multi_f64(field) {
                    if !multi_ok {
                        return None;
                    }
                    Src::MF64(c)
                } else {
                    Src::Absent
                }
            }
            SortKind::Keyword => {
                if let Some(c) = seg.keyword_reader(field) {
                    // Optional acceleration only: returning None selects the normal
                    // collector, preserving complete matching and sort results.
                    let blocks = seg.keyword_ord_blocks(field).ok().flatten()?;
                    Src::Kw(c, blocks, seg, field)
                } else if let Some(c) = seg.multi_keyword_reader(field) {
                    Src::MKw(c)
                } else {
                    Src::Absent
                }
            }
        };
        let missing_gap = match (missing, &src) {
            (SortMissing::Value(SortValue::Str(ms)), Src::Kw(c, ..)) => Some((c.seek_ord(ms), ms)),
            (SortMissing::Value(SortValue::Str(ms)), Src::MKw(c)) => Some((c.seek_ord(ms), ms)),
            _ => None,
        }
        .and_then(|((p, exact), ms)| (!exact).then(|| (p, ms.clone())));
        let mut ls = LeafSort {
            src,
            desc: *descending,
            pick_max: mode == SortMode::Max,
            missing: 0,
            missing_gap,
            num_docs: seg.num_docs,
            seg: Some((seg, field)),
        };
        // A numeric column's missing value sorts as its sentinel (ES
        // `missingObject`: Long.MIN / MAX, -/+Infinity at the end it goes
        // to), so a real value equal to it ties with it and the doc order
        // decides, as in the generic collector (`SortValue::cmp_asc`).
        // Only a leaf holding such a value needs it: else the missing class
        // orders the same (`after_key` maps the sentinel onto it) and keeps
        // the point walks.
        let low = matches!(missing, SortMissing::First) != *descending;
        fn holds<T: crate::codec::fastfields::FastValue>(c: &ColumnReader<'_, T>, at: T, low: bool) -> bool {
            // The points index's ends answer without a pass over the blocks.
            if c.counts_from_points() {
                return c.points_min_max().is_some_and(|(mn, mx)| if low { mn == at } else { mx == at });
            }
            (0..c.block_count()).any(|b| {
                let (mn, mx) = c.block_min_max(b);
                (if low { mn == at } else { mx == at }) && c.block_presence(b).0
            })
        }
        let sentinel = match &ls.src {
            Src::I64(c) => {
                let at = if low { i64::MIN } else { i64::MAX };
                let held = match ls.dense_i64_bounds() {
                    Some((mn, mx)) => if low { mn == at } else { mx == at },
                    None => holds(c, at, low),
                };
                held.then_some(if low { 0 } else { u64::MAX })
            }
            Src::U64(c) => holds(c, if low { 0 } else { u64::MAX }, low).then_some(if low { 0 } else { u64::MAX }),
            Src::F64(c) => holds(c, if low { f64::NEG_INFINITY } else { f64::INFINITY }, low).then(|| o_f64(if low { f64::NEG_INFINITY } else { f64::INFINITY })),
            Src::MI64(_) | Src::MU64(_) => Some(if low { 0 } else { u64::MAX }),
            Src::MF64(_) => Some(o_f64(if low { f64::NEG_INFINITY } else { f64::INFINITY })),
            _ => None,
        };
        ls.missing = match missing {
            SortMissing::First | SortMissing::Last if sentinel.is_some() => ls.value_key(sentinel.unwrap_or(0)),
            SortMissing::First => 0,
            SortMissing::Last => LAST_CLASS,
            SortMissing::Value(v) => {
                if matches!(ls.src, Src::Absent) {
                    return None;
                }
                ls.value_key(ls.value_o(v)?)
            }
        };
        Some(ls)
    }

    /// `v` in this column's order-preserving u64 domain (`None` when it
    /// cannot be placed exactly).
    fn value_o(&self, v: &SortValue) -> Option<u64> {
        match (&self.src, v) {
            (Src::I64(_) | Src::MI64(_), SortValue::I64(x)) => Some(o_i64(*x)),
            (Src::I64(_) | Src::MI64(_), SortValue::F64(x)) => {
                (x.fract() == 0.0 && *x >= -9.2e18 && *x <= 9.2e18).then(|| o_i64(*x as i64))
            }
            // Shifted-long sort values back to the raw u64 order.
            (Src::U64(_) | Src::MU64(_), SortValue::I64(x)) => Some(o_i64(*x)),
            (Src::F64(_) | Src::MF64(_) | Src::Geo { .. }, SortValue::F64(x)) => Some(o_f64(*x)),
            (Src::F64(_) | Src::MF64(_) | Src::Geo { .. }, SortValue::I64(x)) => Some(o_f64(*x as f64)),
            (Src::Kw(c, ..), SortValue::Str(s)) => Some(self.str_o(c.seek_ord(s), s)),
            (Src::MKw(c), SortValue::Str(s)) => Some(self.str_o(c.seek_ord(s), s)),
            // No doc has a value: any value key sorts consistently.
            (Src::Absent, SortValue::I64(_) | SortValue::F64(_) | SortValue::Str(_)) => Some(0),
            _ => None,
        }
    }

    /// A string's slot given its lower bound `(p, exact)` in the term
    /// table: term `p` is `4p + 3`; a string between terms `p - 1` and `p`
    /// takes `4p + 1` (or `4p` / `4p + 2` when it sorts before / after the
    /// absent `missing` string of the same gap), so the `missing` value and
    /// a `search_after` string keep their order without being terms.
    fn str_o(&self, (p, exact): (u32, bool), s: &str) -> u64 {
        let base = 4 * p as u64;
        if exact {
            return base + 3;
        }
        match &self.missing_gap {
            Some((mp, ms)) if *mp == p => base + (s.cmp(ms.as_str()) as i64 + 1) as u64,
            _ => base + 1,
        }
    }

    /// Key of a present value.
    #[inline(always)]
    fn value_key(&self, o: u64) -> u128 {
        VALUE_CLASS | (if self.desc { !o } else { o }) as u128
    }

    /// Key of a `search_after` value, or `None` when it cannot be placed
    /// exactly (the generic collector then runs).
    pub(crate) fn after_key(&self, v: &SortValue) -> Option<u128> {
        match v {
            SortValue::Missing => {
                (self.missing == 0 || self.missing == LAST_CLASS).then_some(self.missing)
            }
            // A class-keyed missing (no doc holds its sentinel) ties with
            // the sentinel, as in `SortValue::cmp_asc`.
            v if self.class_low().is_some_and(|low| v.is_missing_sentinel(low)) => Some(self.missing),
            v => self.value_o(v).map(|o| self.value_key(o)),
        }
    }

    /// Whether a missing class sorts at the low end of the values.
    fn class_low(&self) -> Option<bool> {
        match self.missing {
            0 => Some(!self.desc),
            LAST_CLASS => Some(self.desc),
            _ => None,
        }
    }

    /// The doc's value in the u64 domain.
    #[inline(always)]
    fn value(&self, doc: DocId) -> Option<u64> {
        match &self.src {
            Src::I64(c) => c.get(doc).map(o_i64),
            Src::U64(c) => c.get(doc),
            Src::F64(c) => c.get(doc).map(o_f64),
            Src::MI64(c) => if self.pick_max {
                c.max_of(doc)
            } else {
                c.min_of(doc)
            }
            .map(o_i64),
            Src::MU64(c) => {
                if self.pick_max {
                    c.max_of(doc)
                } else {
                    c.min_of(doc)
                }
            }
            Src::MF64(c) => {
                let mut it = c.values(doc);
                let first = it.next()?;
                Some(o_f64(if self.pick_max {
                    it.fold(first, f64::max)
                } else {
                    it.fold(first, f64::min)
                }))
            }
            Src::Kw(c, ..) => {
                let o = c.ord(doc);
                (o != MISSING_ORD).then_some(4 * o as u64 + 3)
            }
            Src::MKw(c) => {
                let mut it = c.ords(doc);
                if self.pick_max { it.last() } else { it.next() }.map(|o| 4 * o as u64 + 3)
            }
            Src::Absent => None,
            Src::Geo { points, cursor, lat, lon, plane, .. } => {
                let r = points.doc_range_hint(doc, cursor.get());
                let end = r.end;
                let mut best: Option<f64> = None;
                for p in points.iter_range(r) {
                    let (a, b) = (crate::geo::decode_lat(p.lat), crate::geo::decode_lon(p.lon));
                    let d = if *plane { crate::geo::plane_meters(*lat, *lon, a, b) } else { crate::sloppy_math::haversin_meters(*lat, *lon, a, b) };
                    best = Some(match best {
                        None => d,
                        Some(x) if self.pick_max => x.max(d),
                        Some(x) => x.min(d),
                    });
                }
                cursor.set(end);
                best.map(o_f64)
            }
        }
    }

    /// Sort key of `doc`: smaller sorts first.
    #[inline(always)]
    pub(crate) fn key(&self, doc: DocId) -> u128 {
        match self.value(doc) {
            Some(o) => self.value_key(o),
            None => self.missing,
        }
    }

    /// Keyword sorts: the ordinals whose docs have keys below `bottom`, as
    /// a range `[a, b)` (keys are monotonic in the ordinal).
    fn competitive_ords(&self, bottom: u128) -> Option<(u32, u32)> {
        let Src::Kw(c, ..) = &self.src else {
            return None;
        };
        let nt = c.num_terms() as u32;
        let key = |o: u32| self.value_key(4 * o as u64 + 3);
        let part = |pred: &dyn Fn(u32) -> bool| {
            let (mut lo, mut hi) = (0u32, nt);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if pred(mid) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        Some(if self.desc { (part(&|o| key(o) >= bottom), nt) } else { (0, part(&|o| key(o) < bottom)) })
    }

    /// The best (smallest) key any doc of the leaf can have, from the block
    /// bounds, when keys compare across leaves: numeric columns (keys are
    /// the values) and absent ones. `None` for keyword ordinals, multi-valued
    /// and geo sources.
    pub(crate) fn best_key(&self) -> Option<u128> {
        if !matches!(self.src, Src::I64(_) | Src::U64(_) | Src::F64(_) | Src::Absent) || self.num_docs == 0 {
            return None;
        }
        if let Some((best, _)) = self.points_key_range() {
            return Some(best);
        }
        Some((0..(self.num_docs as usize).div_ceil(BLOCK_SIZE)).map(|b| self.block_range(b).0).min().unwrap_or(0))
    }

    /// The worst (largest) key any doc of the leaf can have, from the block
    /// bounds; `None` as for [`Self::best_key`].
    pub(crate) fn worst_key(&self) -> Option<u128> {
        if !matches!(self.src, Src::I64(_) | Src::U64(_) | Src::F64(_) | Src::Absent) || self.num_docs == 0 {
            return None;
        }
        if let Some((_, worst)) = self.points_key_range() {
            return Some(worst);
        }
        Some((0..(self.num_docs as usize).div_ceil(BLOCK_SIZE)).map(|b| self.block_range(b).1).max().unwrap_or(u128::MAX))
    }

    /// `(best, worst)` keys of a long column's docs from the two ends of its
    /// points index, or a skipper column's kept bounds
    /// ([`Self::dense_i64_bounds`]) (`O(1)`, where the block bounds take a pass over every
    /// block of the leaf: thousands on a large segment, for every leaf of
    /// every sorted search), the missing key included when some doc has no
    /// value. The same range the blocks give (both hold deleted docs).
    fn points_key_range(&self) -> Option<(u128, u128)> {
        fn ends<T: crate::codec::fastfields::FastValue>(ls: &LeafSort<'_>, c: &ColumnReader<'_, T>, o: impl Fn(T) -> u64) -> Option<(u128, u128)> {
            if c.num_docs() != ls.num_docs {
                return None;
            }
            let (mn, mx) = c.points_min_max()?;
            let (a, z) = (ls.value_key(o(mn)), ls.value_key(o(mx)));
            let (mut lo, mut hi) = if ls.desc { (z, a) } else { (a, z) };
            if c.value_doc_count() < ls.num_docs as u64 {
                lo = lo.min(ls.missing);
                hi = hi.max(ls.missing);
            }
            Some((lo, hi))
        }
        match &self.src {
            Src::I64(c) => ends(self, c, o_i64).or_else(|| {
                let (mn, mx) = self.dense_i64_bounds()?;
                let (a, z) = (self.value_key(o_i64(mn)), self.value_key(o_i64(mx)));
                Some(if self.desc { (z, a) } else { (a, z) })
            }),
            Src::U64(c) => ends(self, c, |v| v),
            _ => None,
        }
    }

    /// The smallest and largest value of a long column with a value in
    /// every doc and no points index (a doc-values skipper column, e.g.
    /// logsdb's `@timestamp`), kept with the segment: what a pass over
    /// every block's bounds gives, without the pass (millions of blocks on
    /// a billion-doc leaf, for every leaf of every sorted search).
    fn dense_i64_bounds(&self) -> Option<(i64, i64)> {
        let (Src::I64(c), Some((seg, field))) = (&self.src, self.seg) else { return None };
        if c.is_sparse() || c.counts_from_points() || c.num_docs() != self.num_docs || c.block_count() == 0 {
            return None;
        }
        seg.min_max_i64(field)
    }

    /// Whether blocks can be skipped (single-valued columns).
    fn has_blocks(&self) -> bool {
        matches!(
            self.src,
            Src::I64(_) | Src::U64(_) | Src::F64(_) | Src::Kw(..) | Src::Absent
        ) || (matches!(self.src, Src::Geo { .. }) && !self.desc && !self.pick_max)
    }

    /// `[lo, hi]` key range of block `b` (every doc of the block has a key
    /// in it).
    #[inline]
    fn block_range(&self, b: usize) -> (u128, u128) {
        fn num<T: crate::codec::fastfields::FastValue>(
            ls: &LeafSort<'_>,
            c: &ColumnReader<'_, T>,
            b: usize,
            o: impl Fn(T) -> Option<u64>,
        ) -> (u128, u128) {
            if b >= c.block_count() {
                return (ls.missing, ls.missing);
            }
            let (any, miss) = c.block_presence(b);
            let (mut lo, mut hi) = (u128::MAX, 0u128);
            if any {
                let (mn, mx) = c.block_min_max(b);
                match (o(mn), o(mx)) {
                    (Some(a), Some(z)) => {
                        let (a, z) = (ls.value_key(a), ls.value_key(z));
                        (lo, hi) = if ls.desc { (z, a) } else { (a, z) };
                    }
                    // NaN bounds: no usable range.
                    _ => return (0, u128::MAX),
                }
            }
            if miss {
                lo = lo.min(ls.missing);
                hi = hi.max(ls.missing);
            }
            (lo, hi)
        }
        match &self.src {
            Src::I64(c) => num(self, c, b, |v| Some(o_i64(v))),
            Src::U64(c) => num(self, c, b, Some),
            Src::F64(c) => num(self, c, b, |v: f64| (!v.is_nan()).then(|| o_f64(v))),
            Src::Kw(_, blocks, ..) => match blocks.get(b) {
                None => (self.missing, self.missing),
                Some(ob) => {
                    let (mut lo, mut hi) = (u128::MAX, 0u128);
                    if ob.min != u32::MAX {
                        let (a, z) = (
                            self.value_key(4 * ob.min as u64 + 3),
                            self.value_key(4 * ob.max as u64 + 3),
                        );
                        (lo, hi) = if self.desc { (z, a) } else { (a, z) };
                    }
                    if ob.missing {
                        lo = lo.min(self.missing);
                        hi = hi.max(self.missing);
                    }
                    (lo, hi)
                }
            },
            Src::Absent => (self.missing, self.missing),
            // Ascending min distance: only a lower bound is known.
            Src::Geo { lat, lon, plane, blocks, .. } if !self.desc && !self.pick_max => match blocks.get(b) {
                Some(gb) if gb.count > 0 => (self.value_key(o_f64(crate::geo::block_min_meters(*lat, *lon, gb, *plane))), u128::MAX),
                _ => (self.missing, self.missing),
            },
            _ => (0, u128::MAX),
        }
    }
}

/// A queued hit; the max-heap keeps the worst (largest key, then doc) on top.
struct Entry {
    key: u128,
    doc: DocId,
    score: f32,
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.doc == o.doc
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key.cmp(&o.key).then(self.doc.cmp(&o.doc))
    }
}

/// Blocks skipped (tests check skipping happens).
#[cfg(test)]
pub(crate) static SKIPPED_BLOCKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Keyword candidate lists built (tests check they are used).
#[cfg(test)]
pub(crate) static CANDIDATE_LISTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Leaves collected off the points index (tests check it runs).
#[cfg(test)]
pub(crate) static POINT_WALKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Leaves whose queue was seeded from their best blocks (tests check it
/// runs).
#[cfg(test)]
pub(crate) static SEEDED_LEAVES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Leaves collected best block first (tests check it runs).
#[cfg(test)]
pub(crate) static BLOCK_WALKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Leaves [`collect_best_blocks`] collected (tests).
#[cfg(test)]
pub(crate) static BEST_BLOCK_WALKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Walks [`collect_best_blocks`] gave up on, past their rewinds (tests).
#[cfg(test)]
pub(crate) static BEST_BLOCK_BAILS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
thread_local! {
    /// Blocks [`collect_all_blocks`] scanned on this thread (tests).
    pub(crate) static WALKED_BLOCKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// One leaf's collection.
pub(crate) struct Collected {
    /// Hits in sort order with their scores (0 when scores are not needed).
    pub hits: Vec<(DocId, f32)>,
    /// Matching live docs counted, up to the count limit.
    pub total: u64,
    /// More docs matched than the count limit.
    pub saturated: bool,
}

/// Collection parameters of one leaf.
pub(crate) struct Params<'b> {
    /// Hits to keep (`from + size`).
    pub need: usize,
    /// Count matching docs up to this many (`u64::MAX`: exactly).
    pub count_limit: u64,
    /// The leaf's exact live match count when known up front (counting is
    /// then skipped).
    pub precounted: Option<u64>,
    /// `search_after` as (key, leaf-local doc; -1 = every tie is after,
    /// `i64::MAX` = no tie is after).
    pub after: Option<(u128, i64)>,
    /// Compute scores of the collected hits.
    pub scores: bool,
    /// Docs to skip (deleted / nested children), bit per doc.
    pub deleted: Option<&'b [u64]>,
    /// Matches the shard's leaves counted so far (published in batches):
    /// the count limit is shard-wide, as Lucene's shared hits threshold,
    /// so a leaf stops counting once the leaves together reach it.
    pub shard_counted: Option<&'b std::sync::atomic::AtomicU64>,
}

/// Fewest blocks a leaf needs before [`collect`] seeds its queue from
/// the best blocks first.
const SEED_MIN_BLOCKS: usize = 64;

/// The blocks [`collect`] seeds its queue from: the blocks with the best
/// bounds, at least `need * 32` docs (two blocks at least) between them,
/// in doc order. `None` when seeding cannot pay off.
fn seed_blocks(ls: &LeafSort<'_>, need: usize, num_blocks: usize) -> Option<Vec<usize>> {
    let want = (need.saturating_mul(32).div_ceil(BLOCK_SIZE)).max(2);
    if num_blocks < SEED_MIN_BLOCKS || want * 8 > num_blocks {
        return None;
    }
    let mut by_best: Vec<(u128, usize)> = (0..num_blocks).map(|b| (ls.block_range(b).0, b)).collect();
    by_best.select_nth_unstable(want - 1);
    by_best.truncate(want);
    // Every block as good as every other (a constant column): nothing to gain.
    if by_best.iter().all(|e| e.0 == by_best[0].0) && by_best[0].0 == ls.block_range(0).0 {
        return None;
    }
    let mut blocks: Vec<usize> = by_best.into_iter().map(|e| e.1).collect();
    blocks.sort_unstable();
    Some(blocks)
}

/// Blocks [`block_ranking`] keeps per leaf and sort: a page lives in a few
/// blocks, so a walk rarely reads past them, and the cache stays small on
/// a billion-doc leaf (millions of blocks, 32 bytes each).
const RANKED_BLOCKS: usize = 4096;

/// The leaf's best [`RANKED_BLOCKS`] blocks, best key first (ties by
/// block), from the segment's cache when it can keep them (`None`: not a
/// single-valued numeric or keyword sort). A keyword column's keys are its
/// segment ordinals' (Lucene's keyword sort skips by competitive terms
/// alike). Shorter than the leaf's blocks: a walk that
/// reaches its end without its page takes another path.
fn block_ranking(ls: &LeafSort<'_>, num_blocks: usize) -> Option<crate::segment::SortRanking> {
    let (seg, field) = ls.seg?;
    if !matches!(ls.src, Src::I64(_) | Src::U64(_) | Src::F64(_) | Src::Kw(..)) || num_blocks < SEED_MIN_BLOCKS {
        return None;
    }
    let keep = num_blocks.min(RANKED_BLOCKS);
    seg.sort_block_ranking((field, ls.desc, ls.pick_max, ls.missing), keep, || {
        let mut v: Vec<(u128, u32)> = (0..num_blocks).map(|b| (ls.block_range(b).0, b as u32)).collect();
        if keep < v.len() {
            v.select_nth_unstable(keep);
            v.truncate(keep);
            v.shrink_to_fit();
        }
        v.sort_unstable();
        v
    })
}

/// The top `p.need` docs of the leaf by `ls` when nothing is left to count
/// (`track_total_hits: false`, or a count known before): the blocks best
/// key first, read in batches ([`BEST_BLOCK_BATCH`], doubling) whose blocks
/// one fresh scorer of the query visits in doc order, until the next
/// block's best key cannot beat the queue. Visits the few blocks a top page
/// lives in, not every competitive block in doc order behind a weak first
/// bottom; a fresh scorer rebuilds the query's filters (a dense filter
/// conjunction reads its bitsets), so the walk builds one per batch, not
/// one per step back in doc order (bulk-interleaved timestamps overlap in
/// most blocks: EQL's stage searches built thousands). `None` when the leaf
/// has no cached ranking or no scorer, when the walk runs past the ranked
/// blocks ([`RANKED_BLOCKS`]), or when [`BEST_BLOCK_BATCHES`]
/// batches leave the queue short, or would at the rate the batches read so
/// far matched (the query matches little in the best blocks): the pass in
/// doc order then runs.
fn collect_best_blocks(ls: &LeafSort<'_>, p: &Params<'_>, fresh: &dyn Fn() -> Option<Box<dyn Scorer>>, num_blocks: usize, counting: bool) -> Option<(Vec<(DocId, f32)>, bool)> {
    let rank = block_ranking(ls, num_blocks)?;
    let need = p.need;
    let live = |d: DocId| p.deleted.is_none_or(|w| w.get(d as usize >> 6).is_none_or(|x| (x >> (d & 63)) & 1 == 0));
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(need.min(1 << 16) + 1);
    // Whether a block whose best key is `best` (first doc `lo`) can no
    // longer beat a full queue.
    let beaten = |heap: &BinaryHeap<Entry>, best: u128, lo: DocId| heap.len() >= need && heap.peek().is_some_and(|e| best > e.key || (best == e.key && lo > e.doc));
    let mut any = false;
    // The ranked blocks in batches of doubling size, each batch's blocks
    // in doc order under one fresh scorer: a few scorers per leaf, however
    // many blocks overlap in key (bulk-interleaved timestamps), instead of
    // one per step back in doc order.
    let mut pos = 0usize;
    let mut batch = BEST_BLOCK_BATCH;
    let mut batches = 0usize;
    let mut blocks: Vec<u32> = Vec::new();
    // Blocks visited and matches queued while the queue was short.
    let (mut visited, mut matched) = (0usize, 0usize);
    while pos < rank.len() {
        let (best, b) = rank[pos];
        if beaten(&heap, best, (b as usize * BLOCK_SIZE) as DocId) {
            break;
        }
        // The queue still not full, and at the rate the batches so far
        // matched it would not fill within [`BEST_BLOCK_BATCHES`]: the query
        // matches little in the best blocks (an EQL stage search); the pass
        // in doc order runs, without building more scorers first.
        if heap.len() < need && batches > 0 {
            let reach = BEST_BLOCK_BATCH * ((1 << BEST_BLOCK_BATCHES) - 1);
            if batches == BEST_BLOCK_BATCHES || matched * reach < need * visited {
                #[cfg(test)]
                BEST_BLOCK_BAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
        }
        let end = (pos + batch).min(rank.len());
        blocks.clear();
        blocks.extend(rank[pos..end].iter().map(|&(_, b)| b));
        blocks.sort_unstable();
        let mut s = fresh()?;
        // The scorer's position (`None`: not started).
        let mut at: Option<DocId> = None;
        for &b in &blocks {
            let lo = (b as usize * BLOCK_SIZE) as DocId;
            // A block of this batch the queue has since outgrown.
            if beaten(&heap, ls.block_range(b as usize).0, lo) {
                continue;
            }
            let hi = ((b as usize + 1) * BLOCK_SIZE).min(ls.num_docs as usize) as DocId;
            if heap.len() < need {
                visited += 1;
            }
            let mut d = match at {
                Some(c) if c >= lo => c,
                _ => s.seek(lo),
            };
            while d < hi {
                if live(d) {
                    any = true;
                    let e = Entry { key: ls.key(d), doc: d, score: if p.scores { s.score() } else { 0.0 } };
                    if heap.len() < need {
                        heap.push(e);
                        matched += 1;
                    } else if let Some(mut top) = heap.peek_mut() {
                        if e < *top {
                            *top = e;
                        }
                    }
                }
                d = s.advance();
            }
            at = Some(d);
            if d == TERMINATED {
                break;
            }
        }
        pos = end;
        batch *= 2;
        if heap.len() < need {
            batches += 1;
        }
    }
    // Past the ranked blocks with the page still open: the blocks after
    // them are not ranked, so the pass in doc order runs.
    if pos >= rank.len() && rank.len() < num_blocks {
        #[cfg(test)]
        BEST_BLOCK_BAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    #[cfg(test)]
    BEST_BLOCK_WALKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some((heap.into_sorted_vec().into_iter().map(|e| (e.doc, e.score)).collect(), counting && any))
}

/// Matches a leaf counts between publishing them to the shard
/// ([`Params::shard_counted`]).
const SHARD_COUNT_BATCH: u64 = 1024;

/// Blocks in [`collect_best_blocks`]'s first batch (later ones double).
const BEST_BLOCK_BATCH: usize = 8;
/// Batches [`collect_best_blocks`] reads before a full queue: past them the
/// query matches too little in the best blocks for the walk to pay.
const BEST_BLOCK_BATCHES: usize = 4;

/// Collect the top `p.need` docs of `scorer` by `ls`. `fresh` builds
/// another scorer of the same query over the leaf: with it, a leaf whose
/// matches need no counting first queues the matches of its best blocks
/// (by the column's block bounds), so the pass in doc order then skips
/// every block that cannot beat them, as Lucene's competitive iterators
/// skip on points (desc sorts over docs indexed in ascending order would
/// otherwise replace the queue's bottom at nearly every doc).
pub(crate) fn collect(ls: &LeafSort<'_>, scorer: &mut dyn Scorer, p: &Params<'_>, fresh: Option<&dyn Fn() -> Option<Box<dyn Scorer>>>) -> Collected {
    let need = p.need;
    let counting = p.precounted.is_none();
    let (mut total, mut saturated) = match p.precounted {
        Some(n) => (n.min(p.count_limit), n > p.count_limit),
        None => (0u64, false),
    };
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(need.min(1 << 16) + 1);
    let skippable = ls.has_blocks();
    let num_blocks = (ls.num_docs as usize).div_ceil(BLOCK_SIZE);
    let after_key = p.after.map(|a| a.0);
    let live = |d: DocId| {
        p.deleted.is_none_or(|w| {
            w.get(d as usize >> 6)
                .is_none_or(|x| (x >> (d & 63)) & 1 == 0)
        })
    };
    // Seed the queue from the best blocks (see above). Only when nothing is
    // left to count (each match is then visited once, in either pass).
    let mut seeded: Option<Vec<usize>> = None;
    // Nothing to count: the best blocks only, from the segment's ranking.
    if let Some(fresh) = fresh.filter(|_| need > 0 && skippable && p.after.is_none() && (!counting || p.count_limit == 0)) {
        if let Some((hits, found)) = collect_best_blocks(ls, p, fresh, num_blocks, counting) {
            return Collected { hits, total, saturated: saturated || found };
        }
    }
    if let Some(fresh) = fresh.filter(|_| need > 0 && skippable && p.after.is_none() && (!counting || p.count_limit == 0) && matches!(ls.src, Src::I64(_) | Src::U64(_) | Src::F64(_))) {
        if let Some(blocks) = seed_blocks(ls, need, num_blocks) {
            if let Some(mut s) = fresh() {
                #[cfg(test)]
                SEEDED_LEAVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // The scorer's position (`None`: not started).
                let mut at: Option<DocId> = None;
                for &b in &blocks {
                    let (lo, hi) = ((b * BLOCK_SIZE) as DocId, ((b + 1) * BLOCK_SIZE).min(ls.num_docs as usize) as DocId);
                    let mut d = match at {
                        Some(c) if c >= lo => c,
                        _ => s.seek(lo),
                    };
                    while d < hi {
                        if live(d) {
                            if counting {
                                saturated = true;
                            }
                            let key = ls.key(d);
                            if heap.len() < need {
                                let score = if p.scores { s.score() } else { 0.0 };
                                heap.push(Entry { key, doc: d, score });
                            } else if let Some(mut top) = heap.peek_mut() {
                                if key < top.key {
                                    let score = if p.scores { s.score() } else { 0.0 };
                                    *top = Entry { key, doc: d, score };
                                }
                            }
                        }
                        d = s.advance();
                    }
                    at = Some(d);
                    if d == TERMINATED {
                        break;
                    }
                }
                seeded = Some(blocks);
            }
        }
    }
    let mut checked_block = usize::MAX;
    // Keyword sorts: once the bottom is known, the docs of the competitive
    // ordinals (from the column's ordinal index) when they are few: the
    // query is then only advanced to them (Lucene's TermOrdValComparator
    // competitive iterator).
    let kw = match &ls.src {
        Src::Kw(_, _, seg, field) => Some((*seg, *field)),
        _ => None,
    };
    let mut cand: Option<Vec<DocId>> = None;
    let mut ci = 0usize;
    let mut cand_bottom = u128::MAX;
    // Counts published to the shard, and whether the shard reached the
    // limit (this leaf's count then stops; the total is a lower bound).
    let mut published = 0u64;
    let mut shard_done = false;
    let mut d = scorer.advance();
    'docs: while d != TERMINATED {
        let counted = !counting || total >= p.count_limit || shard_done;
        if let Some(cs) = &cand {
            while ci < cs.len() && cs[ci] < d {
                ci += 1;
            }
            let Some(&c) = cs.get(ci) else { break };
            if c > d {
                d = scorer.seek(c);
                continue;
            }
        } else if counted {
            if need == 0 {
                break;
            }
            let full = heap.len() >= need;
            if let (Some((seg, field)), true) = (kw, full) {
                let bottom = heap.peek().map_or(u128::MAX, |e| e.key);
                if bottom != cand_bottom && ls.missing >= bottom {
                    cand_bottom = bottom;
                    if let (Some((a, b)), Some(ix)) = (ls.competitive_ords(bottom), seg.keyword_ord_index(field).ok().flatten()) {
                        let (x, y) = (ix.starts[a as usize] as usize, ix.starts[b as usize] as usize);
                        if (y - x) as u64 * 8 <= (ls.num_docs - d) as u64 {
                            let mut v: Vec<DocId> = ix.docs[x..y].iter().copied().filter(|&c| c >= d).collect();
                            v.sort_unstable();
                            #[cfg(test)]
                            CANDIDATE_LISTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            cand = Some(v);
                            ci = 0;
                            continue;
                        }
                    }
                }
            }
            // No column (the field holds no value in this leaf, e.g. big5's
            // `meta.file`): every doc ties on the missing key, so a full
            // queue already holds the leaf's first docs and nothing later
            // can win the doc tie-break (no walk over the remaining blocks).
            if full && after_key.is_none() && seeded.is_none() && matches!(ls.src, Src::Absent) {
                break;
            }
            if skippable && (full || after_key.is_some() || seeded.is_some()) {
                let b = d as usize / BLOCK_SIZE;
                if b != checked_block {
                    let (bottom, bottom_doc) = if full {
                        heap.peek().map_or((u128::MAX, 0), |e| (e.key, e.doc))
                    } else {
                        (u128::MAX, 0)
                    };
                    let done = |b: usize| seeded.as_ref().is_some_and(|s| s.binary_search(&b).is_ok());
                    // A seeded queue holds docs past this block: an equal
                    // key still wins on an earlier doc.
                    let competitive = |b: usize| {
                        let (lo, hi) = ls.block_range(b);
                        (lo < bottom || (lo == bottom && ((b * BLOCK_SIZE) as DocId) < bottom_doc))
                            && after_key.is_none_or(|ak| hi >= ak)
                            && !done(b)
                    };
                    let mut nb = b;
                    while !competitive(nb) {
                        nb += 1;
                        if nb >= num_blocks {
                            break 'docs;
                        }
                    }
                    if nb != b {
                        #[cfg(test)]
                        SKIPPED_BLOCKS
                            .fetch_add((nb - b) as u64, std::sync::atomic::Ordering::Relaxed);
                        d = scorer.seek((nb * BLOCK_SIZE) as DocId);
                        continue;
                    }
                    checked_block = b;
                }
            }
        }
        if live(d) {
            if counting && !shard_done {
                if total < p.count_limit {
                    total += 1;
                    if let Some(c) = p.shard_counted {
                        if total - published >= SHARD_COUNT_BATCH {
                            let delta = total - published;
                            let before = c.fetch_add(delta, std::sync::atomic::Ordering::Relaxed);
                            published = total;
                            if before + delta >= p.count_limit {
                                shard_done = true;
                                saturated = true;
                            }
                        }
                    }
                } else {
                    saturated = true;
                }
            }
            if need > 0 {
                let key = ls.key(d);
                let after_ok = p
                    .after
                    .is_none_or(|(ak, ad)| key > ak || (key == ak && d as i64 > ad));
                if after_ok {
                    if heap.len() < need {
                        let score = if p.scores { scorer.score() } else { 0.0 };
                        heap.push(Entry { key, doc: d, score });
                    } else if let Some(mut top) = heap.peek_mut() {
                        // An equal key wins only on an earlier doc (seeded
                        // docs can come later than this one).
                        if key < top.key || (key == top.key && d < top.doc) {
                            let score = if p.scores { scorer.score() } else { 0.0 };
                            *top = Entry { key, doc: d, score };
                        }
                    }
                }
            }
        }
        d = scorer.advance();
    }
    if let (Some(c), true) = (p.shard_counted, counting && total > published) {
        c.fetch_add(total - published, std::sync::atomic::Ordering::Relaxed);
    }
    let hits = heap
        .into_sorted_vec()
        .into_iter()
        .map(|e| (e.doc, e.score))
        .collect();
    Collected {
        hits,
        total,
        saturated,
    }
}

/// Top hits of a leaf all of whose live docs match (`match_all`), read off
/// the column's points index (docs sorted by value) in sort order: `O(need
/// + log n)` instead of a pass over the leaf. `None` when the column has no
/// points index (or holds NaNs, or a sparse column sorts missing docs as a
/// value); the caller then collects normally. Counting is the caller's.
pub(crate) fn collect_all_docs(ls: &LeafSort<'_>, p: &Params<'_>, score: f32) -> Option<Vec<(DocId, f32)>> {
    match &ls.src {
        Src::I64(c) => walk_points(ls, c, p, score, o_i64),
        Src::U64(c) => walk_points(ls, c, p, score, |v| v),
        Src::F64(c) => {
            let n = c.points_len();
            if n > 0 && c.point_value(n - 1).is_nan() {
                return None;
            }
            walk_points(ls, c, p, score, o_f64)
        }
        Src::Kw(c, blocks, seg, field) => walk_ords(ls, c, blocks, &*seg.keyword_ord_index(*field).ok().flatten()?, p, score),
        _ => None,
    }
}

/// Top hits of a leaf all of whose live docs match (`match_all`) when the
/// column has no points index (a doc-values skipper column, e.g. logsdb's
/// `@timestamp`): its blocks are visited best key first, each from its best
/// key that `search_after` still admits, until the next block's best key
/// cannot beat the queue (Lucene's skipper-driven competitive iterator,
/// without its doc-order constraint). `None` when the sort has no blocks.
pub(crate) fn collect_all_blocks(ls: &LeafSort<'_>, p: &Params<'_>, score: f32) -> Option<Vec<(DocId, f32)>> {
    if !ls.has_blocks() || p.need == 0 {
        return None;
    }
    #[cfg(test)]
    BLOCK_WALKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let need = p.need;
    // No column (an unmapped sort field): every doc ties, so the first
    // live docs in doc order are the page.
    if matches!(ls.src, Src::Absent) && p.after.is_none() {
        let live = |d: DocId| p.deleted.is_none_or(|w| w.get(d as usize >> 6).is_none_or(|x| (x >> (d & 63)) & 1 == 0));
        return Some((0..ls.num_docs).filter(|&d| live(d)).take(need).map(|d| (d, score)).collect());
    }
    let num_blocks = (ls.num_docs as usize).div_ceil(BLOCK_SIZE);
    let after_key = p.after.map(|a| a.0);
    let live = |d: DocId| p.deleted.is_none_or(|w| w.get(d as usize >> 6).is_none_or(|x| (x >> (d & 63)) & 1 == 0));
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(need.min(1 << 16) + 1);
    // Blocks best first, until the next one's best key cannot beat the
    // queue's worst (an equal key may still win the doc tie-break).
    // Blocks come in (best key, block) order: once the queue is full, a
    // block whose best key is worse than its worst, or ties it from a
    // later doc on (ties go to the earlier doc), ends the walk (a constant
    // or unmapped sort key ties every block).
    let scan = |blocks: &[(u128, u32)], heap: &mut BinaryHeap<Entry>| {
        for &(best, b) in blocks {
            if heap.len() >= need && heap.peek().is_some_and(|e| best > e.key || (best == e.key && (b as usize * BLOCK_SIZE) as DocId > e.doc)) {
                return true;
            }
            #[cfg(test)]
            WALKED_BLOCKS.with(|w| w.set(w.get() + 1));
            let start = b as usize * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(ls.num_docs as usize);
            for d in start as DocId..end as DocId {
                if !live(d) {
                    continue;
                }
                let key = ls.key(d);
                if !p.after.is_none_or(|(ak, ad)| key > ak || (key == ak && d as i64 > ad)) {
                    continue;
                }
                let e = Entry { key, doc: d, score };
                if heap.len() < need {
                    heap.push(e);
                } else if let Some(mut top) = heap.peek_mut() {
                    // Blocks come out of doc order: compare with the doc tie-break.
                    if e < *top {
                        *top = e;
                    }
                }
            }
        }
        false
    };
    // No cursor: the segment's cached ranking of the best blocks (the
    // order below), so a search reads the few blocks a page lives in, not
    // every block's bounds (a billion-doc leaf has millions). A page past
    // the ranked blocks takes the pass below.
    if let Some(rank) = (p.after.is_none()).then(|| block_ranking(ls, num_blocks)).flatten() {
        if scan(&rank[..], &mut heap) || rank.len() == num_blocks {
            return Some(heap.into_sorted_vec().into_iter().map(|e| (e.doc, e.score)).collect());
        }
        heap.clear();
    }
    // (best admissible key, block); blocks whose keys all sort before the
    // cursor hold nothing.
    let mut order: Vec<(u128, u32)> = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let (lo, hi) = ls.block_range(b);
        if after_key.is_some_and(|ak| hi < ak) {
            continue;
        }
        order.push((after_key.map_or(lo, |ak| lo.max(ak)), b as u32));
    }
    // The best few blocks first (a partial selection, not a sort of every
    // block); then only the blocks that can still beat the queue, sorted.
    let first = order.len().min(64.max(need.div_ceil(BLOCK_SIZE / 4)));
    if first < order.len() {
        order.select_nth_unstable(first);
    }
    let (head, _) = order.split_at_mut(first);
    head.sort_unstable();
    if !scan(head, &mut heap) && first < order.len() {
        let bottom = if heap.len() >= need { heap.peek().map_or(u128::MAX, |e| e.key) } else { u128::MAX };
        let bottom_doc = if heap.len() >= need { heap.peek().map_or(DocId::MAX, |e| e.doc) } else { DocId::MAX };
        let mut rest: Vec<(u128, u32)> = order[first..].iter().copied().filter(|&(best, b)| best < bottom || (best == bottom && ((b as usize * BLOCK_SIZE) as DocId) < bottom_doc)).collect();
        rest.sort_unstable();
        scan(&rest, &mut heap);
    }
    Some(heap.into_sorted_vec().into_iter().map(|e| (e.doc, e.score)).collect())
}

/// [`walk_points`] for a keyword column over its ordinal index.
fn walk_ords(ls: &LeafSort<'_>, c: &KeywordReader<'_>, blocks: &[OrdBlock], ix: &crate::codec::fastfields::OrdIndex, p: &Params<'_>, score: f32) -> Option<Vec<(DocId, f32)>> {
    let (missing_first, missing_last) = (ls.missing == 0, ls.missing == LAST_CLASS);
    let any_missing = blocks.iter().any(|b| b.missing);
    if any_missing && !missing_first && !missing_last {
        return None;
    }
    let need = p.need;
    let nt = c.num_terms() as u32;
    let live = |d: DocId| p.deleted.is_none_or(|w| w.get(d as usize >> 6).is_none_or(|x| (x >> (d & 63)) & 1 == 0));
    let after_ok = |key: u128, d: DocId| p.after.is_none_or(|(ak, ad)| key > ak || (key == ak && d as i64 > ad));
    let mut hits: Vec<(DocId, f32)> = Vec::with_capacity(need.min(ls.num_docs as usize));
    let push_missing = |hits: &mut Vec<(DocId, f32)>| {
        for (b, ob) in blocks.iter().enumerate() {
            if !ob.missing {
                continue;
            }
            let end = ((b + 1) * BLOCK_SIZE).min(ls.num_docs as usize);
            for d in (b * BLOCK_SIZE) as DocId..end as DocId {
                if hits.len() >= need {
                    return;
                }
                if c.ord(d) == MISSING_ORD && live(d) && after_ok(ls.missing, d) {
                    hits.push((d, score));
                }
            }
        }
    };
    if missing_first {
        push_missing(&mut hits);
    }
    // Ordinals in key order; the after key bounds where to start.
    let key = |o: u32| ls.value_key(4 * o as u64 + 3);
    let ords: Box<dyn Iterator<Item = u32>> = match p.after.map(|(ak, _)| ak) {
        Some(ak) if !ls.desc => {
            let (mut lo, mut hi) = (0u32, nt);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if key(mid) < ak { lo = mid + 1 } else { hi = mid }
            }
            Box::new(lo..nt)
        }
        Some(ak) => {
            let (mut lo, mut hi) = (0u32, nt);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if key(mid) >= ak { lo = mid + 1 } else { hi = mid }
            }
            // Ords < lo have keys >= ak (desc keys fall with the ordinal).
            Box::new((0..lo).rev())
        }
        None if ls.desc => Box::new((0..nt).rev()),
        None => Box::new(0..nt),
    };
    for o in ords {
        if hits.len() >= need {
            break;
        }
        let k = key(o);
        for &d in &ix.docs[ix.starts[o as usize] as usize..ix.starts[o as usize + 1] as usize] {
            if hits.len() >= need {
                break;
            }
            if live(d) && after_ok(k, d) {
                hits.push((d, score));
            }
        }
    }
    if missing_last && hits.len() < need {
        push_missing(&mut hits);
    }
    Some(hits)
}

fn walk_points<T: crate::codec::fastfields::FastValue>(ls: &LeafSort<'_>, c: &ColumnReader<'_, T>, p: &Params<'_>, score: f32, o: impl Fn(T) -> u64) -> Option<Vec<(DocId, f32)>> {
    let n = c.points_len();
    if n == 0 || c.num_docs() != ls.num_docs {
        return None;
    }
    #[cfg(test)]
    POINT_WALKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sparse = c.is_sparse();
    let (missing_first, missing_last) = (ls.missing == 0, ls.missing == LAST_CLASS);
    if sparse && !missing_first && !missing_last {
        return None;
    }
    let need = p.need;
    let live = |d: DocId| p.deleted.is_none_or(|w| w.get(d as usize >> 6).is_none_or(|x| (x >> (d & 63)) & 1 == 0));
    let after_ok = |key: u128, d: DocId| p.after.is_none_or(|(ak, ad)| key > ak || (key == ak && d as i64 > ad));
    let mut hits: Vec<(DocId, f32)> = Vec::with_capacity(need.min(n));
    let push_missing = |hits: &mut Vec<(DocId, f32)>| {
        if !sparse {
            return;
        }
        for d in 0..ls.num_docs {
            if hits.len() >= need {
                break;
            }
            if !c.has_value(d) && live(d) && after_ok(ls.missing, d) {
                hits.push((d, score));
            }
        }
    };
    if missing_first {
        push_missing(&mut hits);
    }
    // Position values come off the index (no doc read with value lists).
    let val = |i: usize| o(c.point_value(i));
    // First position in [0, end) whose value fails `pred` (values ascend).
    let part = |end: usize, pred: &dyn Fn(u64) -> bool| {
        let (mut lo, mut hi) = (0usize, end);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if pred(val(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    };
    // The after value in the column's u64 domain (class 1 keys only).
    let after_o = p.after.and_then(|(ak, _)| (ak >> 64 == 1).then(|| if ls.desc { !(ak as u64) } else { ak as u64 }));
    let after_class = p.after.map(|(ak, _)| ak >> 64);
    let emit = |i: usize, hits: &mut Vec<(DocId, f32)>| {
        let d = c.point_doc(i);
        if c.has_value(d) && live(d) && after_ok(ls.value_key(val(i)), d) {
            hits.push((d, score));
        }
    };
    if hits.len() < need {
        if !ls.desc {
            let start = match (after_class, after_o) {
                (Some(2), _) => n,
                (Some(1), Some(av)) => part(n, &|v| v < av),
                _ => 0,
            };
            for i in start..n {
                if hits.len() >= need {
                    break;
                }
                emit(i, &mut hits);
            }
        } else {
            // Value descending, ties by doc ascending: walk equal-value runs
            // from the top, each run forwards.
            let mut end = match (after_class, after_o) {
                (Some(2), _) => 0,
                (Some(1), Some(av)) => part(n, &|v| v <= av),
                _ => n,
            };
            while end > 0 && hits.len() < need {
                let v = val(end - 1);
                let lo = part(end, &|x| x < v);
                for i in lo..end {
                    if hits.len() >= need {
                        break;
                    }
                    emit(i, &mut hits);
                }
                end = lo;
            }
        }
    }
    if missing_last && hits.len() < need {
        push_missing(&mut hits);
    }
    Some(hits)
}

/// Number of docs of a leaf not marked in `deleted` (bit per doc).
pub(crate) fn live_count(num_docs: u32, deleted: Option<&[u64]>) -> u64 {
    let n = num_docs as usize;
    let Some(w) = deleted else {
        return n as u64;
    };
    let mut dead = 0u64;
    for (i, x) in w.iter().enumerate().take(n.div_ceil(64)) {
        let lo = i * 64;
        let bits = (n - lo).min(64);
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        dead += (x & mask).count_ones() as u64;
    }
    n as u64 - dead
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long column's best and worst keys from its points index equal the
    /// ones its blocks give, ascending and descending, with and without
    /// docs missing the value (missing first or last) and deletions.
    #[test]
    fn points_key_range_matches_the_block_bounds() {
        use crate::codec::directory::RamDirectory;
        use crate::query::searcher::{SortField, SortKind, SortMissing};
        use crate::segment::SegmentWriter;
        use crate::types::{Document, Field, FieldType};
        let schema = vec![Field::new(0, "population", FieldType::I64)];
        let dir = RamDirectory::new();
        for (id, sparse) in [(1u32, false), (2, true)] {
            let mut w = SegmentWriter::new(id);
            for doc in 0..20_000u32 {
                let mut d = Document::new();
                if !sparse || doc % 9 != 0 { d.add_i64(0, ((doc as i64 * 7919) % 100_003) - 500); }
                w.add_document(&d, &schema);
            }
            let seg = w.write_to(&dir).unwrap();
            for d in (0..20_000).step_by(31) { seg.delete_doc(d).unwrap(); }
            let leaf = LeafReader::Segment(&seg);
            for descending in [false, true] {
                for missing in [SortMissing::First, SortMissing::Last] {
                    let sf = SortField::Field { field: 0, kind: SortKind::I64, descending, missing: missing.clone(), mode: None };
                    let ls = LeafSort::new(&leaf, &sf).unwrap();
                    let blocks = (0..(ls.num_docs as usize).div_ceil(BLOCK_SIZE)).map(|b| ls.block_range(b));
                    let (best, worst) = blocks.fold((u128::MAX, 0), |(lo, hi), (a, z)| (lo.min(a), hi.max(z)));
                    assert_eq!(ls.points_key_range(), Some((best, worst)), "sparse={sparse} desc={descending} {missing:?}");
                    assert_eq!((ls.best_key(), ls.worst_key()), (Some(best), Some(worst)));
                }
            }
        }
    }

    #[test]
    fn numeric_maps_preserve_order() {
        let xs = [i64::MIN, -5, -1, 0, 1, 7, i64::MAX];
        for w in xs.windows(2) {
            assert!(o_i64(w[0]) < o_i64(w[1]));
        }
        let fs = [
            f64::NEG_INFINITY,
            -1e300,
            -2.5,
            -0.0,
            0.0,
            1e-300,
            3.0,
            f64::INFINITY,
        ];
        for w in fs.windows(2) {
            assert!(o_f64(w[0]) <= o_f64(w[1]), "{} {}", w[0], w[1]);
        }
        assert_eq!(o_f64(-0.0), o_f64(0.0));
    }

    #[test]
    fn live_count_masks_tail() {
        assert_eq!(live_count(70, None), 70);
        assert_eq!(live_count(70, Some(&[1, u64::MAX])), 70 - 1 - 6);
    }
}
