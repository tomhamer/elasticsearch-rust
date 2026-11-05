//! A `geo_shape` field's index structures in a memtable and in a
//! segment: ES's two (`geo_shape` with doc values), fed by one
//! tessellation per value:
//!
//! - the triangle tree ([`super::shape_bkd`]) the queries walk
//!   (`LatLonShape` points), when the field is indexed;
//! - the per-doc value ([`super::shape_dv`]: centroid, extent, triangle
//!   tree) the aggregations, scripts and ES|QL read, when it has doc
//!   values.
//!
//! A memtable keeps every doc's triangles and value as they arrive;
//! queries relate its docs triangle by triangle with the tree's leaf
//! tests ([`super::shape_bkd::Matcher::doc`]).

use super::shape::CentroidCalc;
use super::shape_bkd::{Entry, ShapeBkd};
use super::shape_dv::{ShapeDvColumn, ShapeValue};
use crate::types::DocId;

/// Triangles per chunk of [`ShapeBuf`].
const TRI_CHUNK: usize = 1 << 16;
/// Value bytes per chunk of [`ShapeBuf`].
const DV_CHUNK: usize = 1 << 20;

/// One `geo_shape` field of a memtable: per doc (ascending) its triangles
/// and its doc value, in chunks (a doc's never split), so the buffer never
/// reallocates what it holds.
#[derive(Clone, Debug, Default)]
pub struct ShapeBuf {
    /// (doc, chunk, offset, len) of the triangles.
    docs: Vec<(DocId, u32, u32, u32)>,
    tris: Vec<Vec<[i32; 7]>>,
    /// (doc, chunk, offset, len) of the values.
    values: Vec<(DocId, u32, u32, u32)>,
    dv: Vec<Vec<u8>>,
}

impl ShapeBuf {
    /// Add a doc's triangles (all its values) and, when the field has doc
    /// values, its value (docs arrive ascending, each once).
    pub fn push(&mut self, doc: DocId, tris: &[[i32; 7]], value: Option<&[u8]>) {
        if !tris.is_empty() {
            let n = tris.len();
            if self.tris.last().is_none_or(|c| c.capacity() - c.len() < n) {
                self.tris.push(Vec::with_capacity(TRI_CHUNK.max(n)));
            }
            let ci = self.tris.len() - 1;
            let c = &mut self.tris[ci];
            let off = c.len() as u32;
            c.extend_from_slice(tris);
            self.docs.push((doc, ci as u32, off, n as u32));
        }
        if let Some(v) = value {
            let n = v.len();
            if self.dv.last().is_none_or(|c| c.capacity() - c.len() < n) {
                self.dv.push(Vec::with_capacity(DV_CHUNK.max(n)));
            }
            let ci = self.dv.len() - 1;
            let c = &mut self.dv[ci];
            let off = c.len() as u32;
            c.extend_from_slice(v);
            self.values.push((doc, ci as u32, off, n as u32));
        }
    }

    pub fn heap_bytes(&self) -> usize {
        self.docs.capacity() * 16 + self.values.capacity() * 16 + self.tris.iter().map(|c| c.capacity() * 28).sum::<usize>() + self.dv.iter().map(Vec::capacity).sum::<usize>()
    }

    /// Docs with triangles, ascending, with them.
    pub fn for_each_doc(&self, limit: u32, f: &mut dyn FnMut(DocId, &[[i32; 7]])) {
        for &(d, c, o, n) in &self.docs {
            if d >= limit {
                break;
            }
            f(d, &self.tris[c as usize][o as usize..(o + n) as usize]);
        }
    }

    /// The doc's triangles, if any.
    pub fn triangles(&self, doc: DocId) -> Option<&[[i32; 7]]> {
        let i = self.docs.binary_search_by_key(&doc, |e| e.0).ok()?;
        let (_, c, o, n) = self.docs[i];
        Some(&self.tris[c as usize][o as usize..(o + n) as usize])
    }

    /// The doc's value, if any.
    pub fn value(&self, doc: DocId) -> Option<ShapeValue<'_>> {
        let i = self.values.binary_search_by_key(&doc, |e| e.0).ok()?;
        let (_, c, o, n) = self.values[i];
        Some(ShapeValue::new(&self.dv[c as usize][o as usize..(o + n) as usize]))
    }

    /// The tree parts over the first `num_docs` docs.
    pub fn encode_tree(&self, num_docs: u32) -> Vec<Vec<u8>> {
        let mut entries: Vec<Entry> = Vec::with_capacity(self.docs.iter().map(|e| e.3 as usize).sum());
        self.for_each_doc(num_docs, &mut |d, ts| entries.extend(ts.iter().map(|t| Entry { tri: *t, doc: d })));
        super::shape_bkd::build_parts(&mut entries)
    }

    /// The doc-value column parts over the first `num_docs` docs.
    pub fn encode_values(&self, num_docs: u32) -> Vec<Vec<u8>> {
        if self.values.is_empty() {
            return Vec::new();
        }
        let mut i = 0usize;
        let vals = &self.values;
        let dv = &self.dv;
        // Docs come in ascending order: walk the values alongside.
        super::shape_dv::encode_column(num_docs, |doc| {
            while i < vals.len() && vals[i].0 < doc {
                i += 1;
            }
            match vals.get(i) {
                Some(&(d, c, o, n)) if d == doc => Some(ShapeValue::new(&dv[c as usize][o as usize..(o + n) as usize])),
                _ => None,
            }
        })
    }
}

/// One doc's shape value, written as its triangles and centroid arrive.
pub fn doc_value(tris: &[[i32; 7]], centroid: &CentroidCalc) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(32 + tris.len() * 12);
    super::shape_dv::write(tris, centroid, &mut out)?;
    Ok(out)
}

/// A `geo_shape` field of a segment: its tree parts and doc values.
#[derive(Clone, Default)]
pub struct ShapeFieldData {
    pub tree: Vec<ShapeBkd>,
    pub values: Option<ShapeDvColumn>,
}

impl ShapeFieldData {
    /// Every `(doc, triangle)` of the segment's tree (merges).
    pub fn for_each_triangle(&self, f: &mut dyn FnMut(DocId, &[i32; 7])) {
        for p in &self.tree {
            p.for_each(f);
        }
    }

    pub fn tree_bytes(&self) -> usize {
        self.tree.iter().map(ShapeBkd::bytes_len).sum()
    }

    pub fn value_bytes(&self) -> usize {
        self.values.as_ref().map_or(0, ShapeDvColumn::bytes_len)
    }
}
