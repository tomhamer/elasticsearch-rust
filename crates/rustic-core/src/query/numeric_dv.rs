//! Per-leaf numeric and geo-point doc-value accessors opened once per leaf
//! (Lucene `NumericDocValues` / `SortedNumericDocValues` /
//! `MultiGeoPointValues`), so per-doc reads are a bounds check and a load
//! instead of a column lookup plus a `Vec` per call.
//!
//! [`NumericDv`] reads exactly what [`LeafReader::numeric_values`] returns
//! (f64 columns, then i64 incl. booleans as 0/1, then u64, single- before
//! multi-valued), as `f64`. [`GeoDv`] reads the doc-sorted points of a
//! geo-point field ([`LeafReader::geo_values`]) with a forward cursor.

use crate::codec::fastfields::{BoolReader, ColumnReader, MultiColumnReader};
use crate::fast_fields::{BoolColumn, F64Column, I64Column, MultiF64Column, MultiI64Column, MultiU64Column, U64Column};
use crate::geo::{decode_lat, decode_lon, GeoEntries, GeoValues};
use crate::query::reader::LeafReader;
use crate::types::DocId;
use std::cell::Cell;

/// The numeric doc values of one field on one leaf.
pub enum NumericDv<'a> {
    /// The leaf has no numeric column for the field.
    Empty,
    SegF64(ColumnReader<'a, f64>),
    SegI64(ColumnReader<'a, i64>),
    SegU64(ColumnReader<'a, u64>),
    SegBool(BoolReader<'a>),
    SegMultiF64(MultiColumnReader<'a, f64>),
    SegMultiI64(MultiColumnReader<'a, i64>),
    SegMultiU64(MultiColumnReader<'a, u64>),
    MemF64(&'a F64Column),
    MemI64(&'a I64Column),
    MemU64(&'a U64Column),
    MemBool(&'a BoolColumn),
    MemMultiF64(&'a MultiF64Column),
    MemMultiI64(&'a MultiI64Column),
    MemMultiU64(&'a MultiU64Column),
}

impl<'a> NumericDv<'a> {
    /// Open `field` on `leaf`, in the lookup order of
    /// [`LeafReader::numeric_values`].
    pub fn open(leaf: &LeafReader<'a>, field: u16) -> NumericDv<'a> {
        match *leaf {
            LeafReader::Segment(s) => {
                let ff = s.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    NumericDv::SegF64(c)
                } else if let Some(c) = ff.get_multi_f64(field) {
                    NumericDv::SegMultiF64(c)
                } else if let Some(c) = ff.get_i64(field) {
                    NumericDv::SegI64(c)
                } else if let Some(c) = ff.get_multi_i64(field) {
                    NumericDv::SegMultiI64(c)
                } else if let Some(c) = ff.get_bool(field) {
                    NumericDv::SegBool(c)
                } else if let Some(c) = ff.get_u64(field) {
                    NumericDv::SegU64(c)
                } else if let Some(c) = ff.get_multi_u64(field) {
                    NumericDv::SegMultiU64(c)
                } else {
                    NumericDv::Empty
                }
            }
            LeafReader::Memtable(w, ..) => {
                let ff = w.fast_fields();
                if let Some(c) = ff.get_f64(field) {
                    NumericDv::MemF64(c)
                } else if let Some(c) = ff.get_multi_f64(field) {
                    NumericDv::MemMultiF64(c)
                } else if let Some(c) = ff.get_i64(field) {
                    NumericDv::MemI64(c)
                } else if let Some(c) = ff.get_multi_i64(field) {
                    NumericDv::MemMultiI64(c)
                } else if let Some(c) = ff.get_bool(field) {
                    NumericDv::MemBool(c)
                } else if let Some(c) = ff.get_u64(field) {
                    NumericDv::MemU64(c)
                } else if let Some(c) = ff.get_multi_u64(field) {
                    NumericDv::MemMultiU64(c)
                } else {
                    NumericDv::Empty
                }
            }
        }
    }

    /// The first value of `doc`, if it has one.
    #[inline]
    pub fn first(&self, doc: DocId) -> Option<f64> {
        match self {
            NumericDv::Empty => None,
            NumericDv::SegF64(c) => c.get(doc),
            NumericDv::SegI64(c) => c.get(doc).map(|v| v as f64),
            NumericDv::SegU64(c) => c.get(doc).map(|v| v as f64),
            NumericDv::SegBool(c) => c.has_value(doc).then(|| c.get(doc) as i64 as f64),
            NumericDv::SegMultiF64(c) => c.values(doc).next(),
            NumericDv::SegMultiI64(c) => c.values(doc).next().map(|v| v as f64),
            NumericDv::SegMultiU64(c) => c.values(doc).next().map(|v| v as f64),
            NumericDv::MemF64(c) => c.get(doc),
            NumericDv::MemI64(c) => c.get(doc).map(|v| v as f64),
            NumericDv::MemU64(c) => c.get(doc).map(|v| v as f64),
            NumericDv::MemBool(c) => c.has_value(doc).then(|| c.get(doc) as i64 as f64),
            NumericDv::MemMultiF64(c) => c.values(doc).first().copied(),
            NumericDv::MemMultiI64(c) => c.values(doc).first().map(|v| *v as f64),
            NumericDv::MemMultiU64(c) => c.values(doc).first().map(|v| *v as f64),
        }
    }

    /// Visit every value of `doc` (stored order); returns how many.
    #[inline]
    pub fn for_each(&self, doc: DocId, mut f: impl FnMut(f64)) -> usize {
        match self {
            NumericDv::SegMultiF64(c) => c.values(doc).map(|v| f(v)).count(),
            NumericDv::SegMultiI64(c) => c.values(doc).map(|v| f(v as f64)).count(),
            NumericDv::SegMultiU64(c) => c.values(doc).map(|v| f(v as f64)).count(),
            NumericDv::MemMultiF64(c) => c.values(doc).iter().map(|v| f(*v)).count(),
            NumericDv::MemMultiI64(c) => c.values(doc).iter().map(|v| f(*v as f64)).count(),
            NumericDv::MemMultiU64(c) => c.values(doc).iter().map(|v| f(*v as f64)).count(),
            single => match single.first(doc) {
                Some(v) => {
                    f(v);
                    1
                }
                None => 0,
            },
        }
    }

    /// Whether the column holds at most one value per doc.
    pub fn is_single_valued(&self) -> bool {
        !matches!(
            self,
            NumericDv::SegMultiF64(_) | NumericDv::SegMultiI64(_) | NumericDv::SegMultiU64(_) | NumericDv::MemMultiF64(_) | NumericDv::MemMultiI64(_) | NumericDv::MemMultiU64(_)
        )
    }
}

/// The geo-point doc values of one field on one leaf: a cursor over the
/// doc-sorted points, cheap for the ascending doc order of scorers
/// (galloping forward, binary search when moving backwards).
pub struct GeoDv<'a> {
    points: Option<&'a GeoValues>,
    pos: Cell<usize>,
}

impl<'a> GeoDv<'a> {
    /// Open `field` on `leaf` (no points when the leaf has none).
    pub fn open(leaf: &LeafReader<'a>, field: u16) -> GeoDv<'a> {
        let points: Option<&'a GeoValues> = match *leaf {
            LeafReader::Segment(s) => s.geo_index(field).map(|g| g.values()),
            LeafReader::Memtable(w, ..) => w.geo_values(field),
        };
        GeoDv { points, pos: Cell::new(0) }
    }

    /// Every encoded point of the leaf, sorted by doc (`None`: none).
    pub fn values(&self) -> Option<&'a GeoValues> {
        self.points
    }

    /// Every point of the leaf, in doc order (by value).
    pub fn iter(&self) -> impl Iterator<Item = crate::geo::GeoPointEntry> + 'a {
        self.points.into_iter().flat_map(|v| v.iter())
    }

    /// Number of points of the leaf.
    pub fn len(&self) -> usize {
        self.points.map_or(0, GeoValues::len)
    }

    /// Whether the leaf has any point of the field.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Encoded points of `doc` (a cursor makes ascending docs cheap).
    #[inline]
    pub fn entries(&self, doc: DocId) -> GeoEntries<'a> {
        let Some(p) = self.points else { return GeoEntries::EMPTY };
        let r = p.doc_range_hint(doc, self.pos.get());
        self.pos.set(r.end);
        p.entries_of(r)
    }

    /// Visit the decoded `(lat, lon)` points of `doc`; returns how many.
    #[inline]
    pub fn for_each(&self, doc: DocId, mut f: impl FnMut(f64, f64)) -> usize {
        let e = self.entries(doc);
        for p in e {
            f(decode_lat(p.lat), decode_lon(p.lon));
        }
        e.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{encode_lat, encode_lon, GeoPointEntry};

    #[test]
    fn geo_cursor_any_order() {
        let mut values = GeoValues::default();
        let mut pts = Vec::new();
        for d in [0u32, 2, 2, 5, 9, 9, 9, 40, 41] {
            values.push(d, d as f64 / 2.0, -(d as f64));
            pts.push(GeoPointEntry { doc: d, lat: encode_lat(d as f64 / 2.0), lon: encode_lon(-(d as f64)) });
        }
        let g = GeoDv { points: Some(&values), pos: Cell::new(0) };
        let expect = |d: u32| pts.iter().filter(|p| p.doc == d).count();
        for d in (0..45).chain([3, 9, 0, 41, 2, 40, 1]) {
            assert_eq!(g.entries(d).len(), expect(d), "doc {d}");
            assert!(g.entries(d).iter().all(|p| p.doc == d));
        }
    }
}
