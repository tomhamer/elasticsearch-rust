//! A segment's `geo_shape` triangle tree: Lucene's `ShapeField` points
//! (seven encoded ints per triangle, the first four its box `minY, minX,
//! maxY, maxX` as the index dimensions) in a BKD tree, and the
//! `SpatialQuery` walks over it (Lucene 10.5.1): each node carries the
//! lower and upper bounds of the four box dims, queries relate nodes with
//! the query's `relate` (`CELL_INSIDE_QUERY` / `CELL_OUTSIDE_QUERY` /
//! `CELL_CROSSES_QUERY`) and leaf triangles with its per-relation
//! predicate, and docs combine as `ShapeQuery` combines them:
//!
//! - INTERSECTS: a doc matches when any triangle intersects (inside nodes
//!   take their docs whole);
//! - WITHIN / DISJOINT: when every triangle passes (a doc with a failing
//!   triangle, or with triangles in a node outside, is excluded);
//! - CONTAINS: when some triangle is a `CANDIDATE` and none `NOTWITHIN`
//!   (docs of inside nodes are excluded).
//!
//! Layout of a part (little endian): `u32 version (1, bit 16: one
//! triangle a doc), u32 leaves, u32
//! nodes, u64 triangles, u32 docs` (distinct docs), the nodes (each
//! `8 x i32` bounds then `u32 left, u32 right, u32 triangles`; a leaf has
//! `left = LEAF | leaf`), the leaf offsets (`(leaves + 1) x u64`), then the
//! leaves. A leaf: `u16 count, u16 docs`, the distinct doc ids (a
//! column), each triangle's index into them (a column) and the seven
//! triangle columns; a column is `i32 base, u8 bits`, then `n` values of
//! `value - base` packed `bits` bits each, little endian (triangle
//! columns 2..6 relative to their triangle: height, width, the third
//! vertex from the box corner, the bits).

use super::geo2d::{LeafQuery, QueryRelation, Relation, Tri, WithinRelation};
use crate::codec::directory::FileSlice;
use crate::error::{Result, RusticError};
use crate::types::DocId;
use rayon::slice::ParallelSliceMut;

/// Triangles per leaf (Lucene `BKDConfig.DEFAULT_MAX_POINTS_IN_LEAF_NODE`).
pub const LEAF_TRIANGLES: usize = 512;
const FORMAT: u32 = 1;
/// Header flag: every doc of the part has exactly one triangle, and no
/// triangle in another part.
const SINGLE: u32 = 1 << 16;
const LEAF: u32 = 1 << 31;
const NODE_BYTES: usize = 44;
/// Triangles per part (an `[extra]` entry holds at most `u32::MAX` bytes).
const PART_TRIANGLES: usize = 64 << 20;

/// One triangle and its doc, as the writer sorts them.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub tri: [i32; 7],
    pub doc: DocId,
}

// ---------------------------------------------------------------------------
// Writing

fn bounds(es: &[Entry]) -> ([i32; 4], [i32; 4]) {
    let mut lo = [i32::MAX; 4];
    let mut hi = [i32::MIN; 4];
    for e in es {
        for d in 0..4 {
            lo[d] = lo[d].min(e.tri[d]);
            hi[d] = hi[d].max(e.tri[d]);
        }
    }
    (lo, hi)
}

/// Bits holding `span`.
fn width(span: u32) -> u8 {
    (32 - span.leading_zeros()) as u8
}

fn put_column(vals: &mut dyn Iterator<Item = i32>, n: usize, out: &mut Vec<u8>, scratch: &mut Vec<i32>) {
    scratch.clear();
    scratch.extend(vals.take(n));
    let base = scratch.iter().copied().min().unwrap_or(0);
    let span = scratch.iter().map(|&v| (v as i64 - base as i64) as u32).max().unwrap_or(0);
    let w = width(span);
    out.extend_from_slice(&base.to_le_bytes());
    out.push(w);
    if w > 0 {
        // Little-endian bit packing, `w` bits a value.
        let start = out.len();
        out.resize(start + (n * w as usize).div_ceil(8), 0);
        let data = &mut out[start..];
        for (i, &v) in scratch.iter().enumerate() {
            let d = (v as i64 - base as i64) as u32 as u64;
            let bit = i * w as usize;
            let mut x = d << (bit & 7);
            let mut at = bit >> 3;
            while x != 0 {
                data[at] |= x as u8;
                x >>= 8;
                at += 1;
            }
        }
    }
}

fn write_leaf(es: &[Entry], out: &mut Vec<u8>, scratch: &mut Vec<i32>) {
    let n = es.len();
    // The leaf's distinct docs (its triangles come sorted by doc), then
    // each triangle's index into them.
    let mut docs: Vec<i32> = Vec::new();
    let mut idx: Vec<u16> = Vec::with_capacity(n);
    for e in es {
        if docs.last() != Some(&(e.doc as i32)) {
            docs.push(e.doc as i32);
        }
        idx.push(docs.len() as u16 - 1);
    }
    out.extend_from_slice(&(n as u16).to_le_bytes());
    out.extend_from_slice(&(docs.len() as u16).to_le_bytes());
    put_column(&mut docs.iter().copied(), docs.len(), out, scratch);
    put_column(&mut idx.iter().map(|&i| i as i32), n, out, scratch);
    put_column(&mut es.iter().map(|e| e.tri[0]), n, out, scratch);
    put_column(&mut es.iter().map(|e| e.tri[1]), n, out, scratch);
    put_column(&mut es.iter().map(|e| (e.tri[2] as i64 - e.tri[0] as i64) as i32), n, out, scratch);
    put_column(&mut es.iter().map(|e| (e.tri[3] as i64 - e.tri[1] as i64) as i32), n, out, scratch);
    put_column(&mut es.iter().map(|e| (e.tri[4] as i64 - e.tri[0] as i64) as i32), n, out, scratch);
    put_column(&mut es.iter().map(|e| (e.tri[5] as i64 - e.tri[1] as i64) as i32), n, out, scratch);
    put_column(&mut es.iter().map(|e| e.tri[6]), n, out, scratch);
}

struct Builder {
    nodes: Vec<u8>,
    n_nodes: u32,
    leaves: Vec<u8>,
    leaf_offsets: Vec<u64>,
    scratch: Vec<i32>,
}

impl Builder {
    /// Build the subtree over `es`; returns its node id.
    fn build(&mut self, es: &mut [Entry]) -> u32 {
        let (lo, hi) = bounds(es);
        let me = self.n_nodes;
        self.n_nodes += 1;
        let at = self.nodes.len();
        self.nodes.resize(at + NODE_BYTES, 0);
        for (k, v) in lo.iter().chain(hi.iter()).enumerate() {
            self.nodes[at + 4 * k..at + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
        }
        let (left, right) = if es.len() <= LEAF_TRIANGLES {
            let leaf = self.leaf_offsets.len() as u32 - 1;
            // Leaf triangles grouped by doc (consecutive doc ids compress).
            es.sort_unstable_by_key(|e| e.doc);
            write_leaf(es, &mut self.leaves, &mut self.scratch);
            self.leaf_offsets.push(self.leaves.len() as u64);
            (LEAF | leaf, 0)
        } else {
            // Split on the box dim with the largest spread, at a leaf
            // boundary near the middle.
            let dim = (0..4).max_by_key(|&d| hi[d] as i64 - lo[d] as i64).unwrap_or(0);
            let leaves = es.len().div_ceil(LEAF_TRIANGLES);
            let mid = ((leaves / 2) * LEAF_TRIANGLES).clamp(1, es.len() - 1);
            es.select_nth_unstable_by_key(mid, |e| e.tri[dim]);
            let (l, r) = es.split_at_mut(mid);
            let left = self.build(l);
            let right = self.build(r);
            (left, right)
        };
        self.nodes[at + 32..at + 36].copy_from_slice(&left.to_le_bytes());
        self.nodes[at + 36..at + 40].copy_from_slice(&right.to_le_bytes());
        self.nodes[at + 40..at + 44].copy_from_slice(&(es.len() as u32).to_le_bytes());
        me
    }
}

/// Levels split in parallel before subtrees build sequentially.
const PAR_DEPTH: u32 = 6;

/// Triangles each child of a node needs for a query walk to split the two
/// over the pool (smaller subtrees walk on one thread: a join costs
/// microseconds, a walk over this many triangles a hundred or more).
const PAR_WALK_MIN: u64 = 1 << 15;

/// The largest crossing subtree a single-point count tests as one task.
const FRONTIER_TRIANGLES: u64 = 1 << 16;

/// The top of a part's tree: split nodes over subtrees built alone.
enum Plan {
    Node { lo: [i32; 4], hi: [i32; 4], count: u32, left: Box<Plan>, right: Box<Plan> },
    Sub(Builder),
}

fn plan(es: &mut [Entry], depth: u32) -> Plan {
    if depth >= PAR_DEPTH || es.len() <= LEAF_TRIANGLES * 8 {
        let mut b = Builder { nodes: Vec::new(), n_nodes: 0, leaves: Vec::with_capacity(es.len() * 20), leaf_offsets: vec![0], scratch: Vec::new() };
        b.build(es);
        return Plan::Sub(b);
    }
    // The split `Builder::build` takes.
    let (lo, hi) = bounds(es);
    let dim = (0..4).max_by_key(|&d| hi[d] as i64 - lo[d] as i64).unwrap_or(0);
    let leaves = es.len().div_ceil(LEAF_TRIANGLES);
    let mid = ((leaves / 2) * LEAF_TRIANGLES).clamp(1, es.len() - 1);
    es.select_nth_unstable_by_key(mid, |e| e.tri[dim]);
    let (l, r) = es.split_at_mut(mid);
    let (left, right) = rayon::join(|| plan(l, depth + 1), || plan(r, depth + 1));
    Plan::Node { lo, hi, count: es.len() as u32, left: Box::new(left), right: Box::new(right) }
}

/// Lay `p` out into `out` as the sequential build would; returns its root.
fn stitch(p: Plan, out: &mut Builder) -> u32 {
    match p {
        Plan::Node { lo, hi, count, left, right } => {
            let me = out.n_nodes;
            out.n_nodes += 1;
            let at = out.nodes.len();
            out.nodes.resize(at + NODE_BYTES, 0);
            for (k, v) in lo.iter().chain(hi.iter()).enumerate() {
                out.nodes[at + 4 * k..at + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
            }
            let l = stitch(*left, out);
            let r = stitch(*right, out);
            out.nodes[at + 32..at + 36].copy_from_slice(&l.to_le_bytes());
            out.nodes[at + 36..at + 40].copy_from_slice(&r.to_le_bytes());
            out.nodes[at + 40..at + 44].copy_from_slice(&count.to_le_bytes());
            me
        }
        Plan::Sub(b) => {
            let base_node = out.n_nodes;
            let base_leaf = out.leaf_offsets.len() as u32 - 1;
            let base_bytes = out.leaves.len() as u64;
            for i in 0..b.n_nodes as usize {
                let at = i * NODE_BYTES;
                let n = &b.nodes[at..at + NODE_BYTES];
                out.nodes.extend_from_slice(&n[..32]);
                let left = u32::from_le_bytes(n[32..36].try_into().unwrap());
                let right = u32::from_le_bytes(n[36..40].try_into().unwrap());
                let (left, right) = if left & LEAF != 0 { (LEAF | ((left & !LEAF) + base_leaf), right) } else { (left + base_node, right + base_node) };
                out.nodes.extend_from_slice(&left.to_le_bytes());
                out.nodes.extend_from_slice(&right.to_le_bytes());
                out.nodes.extend_from_slice(&n[40..44]);
            }
            out.n_nodes += b.n_nodes;
            out.leaves.extend_from_slice(&b.leaves);
            out.leaf_offsets.extend(b.leaf_offsets[1..].iter().map(|o| o + base_bytes));
            base_node
        }
    }
}

/// Encode a field's triangles as BKD parts (one unless there are more
/// than a part holds).
pub fn build_parts(entries: &mut [Entry]) -> Vec<Vec<u8>> {
    build_parts_with(entries, true)
}

/// [`build_parts`]; `whole_docs`: every doc's triangles are all in
/// `entries` (a part then records when each doc has exactly one, see
/// [`ShapeBkd::one_triangle_per_doc`]).
pub fn build_parts_with(entries: &mut [Entry], whole_docs: bool) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if entries.is_empty() {
        return out;
    }
    if entries.len() <= PART_TRIANGLES {
        return vec![build_part(entries, whole_docs)];
    }
    // Parts over doc ranges (a doc's triangles stay in one part).
    entries.par_sort_unstable_by_key(|e| e.doc);
    let mut start = 0;
    while start < entries.len() {
        let mut end = (start + PART_TRIANGLES).min(entries.len());
        while end < entries.len() && entries[end].doc == entries[end - 1].doc {
            end += 1;
        }
        out.push(build_part(&mut entries[start..end], whole_docs));
        start = end;
    }
    out
}

fn build_part(es: &mut [Entry], whole_docs: bool) -> Vec<u8> {
    let docs = {
        let max = es.iter().map(|e| e.doc).max().unwrap_or(0) as usize;
        let mut seen = vec![0u64; max / 64 + 1];
        let mut n = 0u32;
        for e in es.iter() {
            let (w, b) = (e.doc as usize / 64, e.doc % 64);
            if seen[w] >> b & 1 == 0 {
                seen[w] |= 1 << b;
                n += 1;
            }
        }
        n
    };
    let tris = es.len() as u64;
    // The top levels split in parallel; their subtrees build on their own
    // and are stitched in the order the sequential build lays them out
    // (preorder nodes, leaves left to right): the same bytes.
    let plan = plan(es, 0);
    let mut b = Builder { nodes: Vec::new(), n_nodes: 0, leaves: Vec::with_capacity(tris as usize * 20), leaf_offsets: vec![0], scratch: Vec::new() };
    stitch(plan, &mut b);
    let n_leaves = b.leaf_offsets.len() as u32 - 1;
    let mut out = Vec::with_capacity(24 + b.nodes.len() + b.leaf_offsets.len() * 8 + b.leaves.len());
    let single = whole_docs && tris == docs as u64;
    out.extend_from_slice(&(FORMAT | if single { SINGLE } else { 0 }).to_le_bytes());
    out.extend_from_slice(&n_leaves.to_le_bytes());
    out.extend_from_slice(&b.n_nodes.to_le_bytes());
    out.extend_from_slice(&tris.to_le_bytes());
    out.extend_from_slice(&docs.to_le_bytes());
    out.extend_from_slice(&b.nodes);
    for o in &b.leaf_offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    out.extend_from_slice(&b.leaves);
    out
}

// ---------------------------------------------------------------------------
// Reading

/// One part of a field's triangle tree, read in place.
#[derive(Clone)]
pub struct ShapeBkd {
    data: FileSlice,
    n_leaves: u32,
    n_nodes: u32,
    tris: u64,
    docs: u32,
    single: bool,
    nodes_at: usize,
    offsets_at: usize,
    leaves_at: usize,
}

impl std::fmt::Debug for ShapeBkd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeBkd").field("leaves", &self.n_leaves).field("triangles", &self.tris).field("docs", &self.docs).finish()
    }
}

/// A decoded leaf: doc ids (per triangle, and distinct) and triangles.
#[derive(Default)]
pub struct LeafBuf {
    pub docs: Vec<DocId>,
    pub distinct: Vec<DocId>,
    pub tris: Vec<[i32; 7]>,
    cols: Vec<i32>,
}

/// Step over a column of `n` values.
fn skip_column(b: &[u8], pos: &mut usize, n: usize) -> Option<()> {
    let w = *b.get(*pos + 4)? as usize;
    *pos += 5 + (n * w).div_ceil(8);
    (*pos <= b.len()).then_some(())
}

fn read_column(b: &[u8], pos: &mut usize, n: usize, out: &mut Vec<i32>) -> Option<()> {
    let base = i32::from_le_bytes(b.get(*pos..*pos + 4)?.try_into().ok()?);
    let w = *b.get(*pos + 4)? as usize;
    *pos += 5;
    out.clear();
    if w == 0 {
        out.resize(n, base);
        return Some(());
    }
    if w > 32 {
        return None;
    }
    let len = (n * w).div_ceil(8);
    let bytes = b.get(*pos..*pos + len)?;
    *pos += len;
    out.reserve(n);
    let mask = if w == 32 { u32::MAX as u64 } else { (1u64 << w) - 1 };
    // Every value is one unaligned 8-byte read: straight from the column
    // while 8 bytes remain, the last few from a zero-padded copy of the
    // tail.
    let direct = if len >= 8 { (((len - 8) * 8 + 7) / w + 1).min(n) } else { 0 };
    let ptr = bytes.as_ptr();
    out.extend((0..direct).map(|i| {
        let bit = i * w;
        // SAFETY: `i < direct`, so `bit / 8 + 8 <= len`.
        let x = u64::from_le(unsafe { std::ptr::read_unaligned(ptr.add(bit >> 3) as *const u64) });
        base.wrapping_add(((x >> (bit & 7)) & mask) as u32 as i32)
    }));
    if direct < n {
        let from = (direct * w) >> 3;
        let mut tail = [0u8; 48];
        let rest = &bytes[from..];
        tail[..rest.len()].copy_from_slice(rest);
        out.extend((direct..n).map(|i| {
            let bit = i * w - from * 8;
            let x = u64::from_le_bytes(tail[bit >> 3..(bit >> 3) + 8].try_into().unwrap());
            base.wrapping_add(((x >> (bit & 7)) & mask) as u32 as i32)
        }));
    }
    Some(())
}

impl ShapeBkd {
    pub fn open(data: FileSlice) -> Result<ShapeBkd> {
        let b = data.as_slice();
        let bad = |m: &str| RusticError::Corrupted(format!("shape tree {m}"));
        if b.len() < 24 {
            return Err(bad("truncated"));
        }
        let u32_at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        if u32_at(0) & 0xffff != FORMAT {
            return Err(bad("format"));
        }
        let single = u32_at(0) & SINGLE != 0;
        let n_leaves = u32_at(4);
        let n_nodes = u32_at(8);
        let tris = u64::from_le_bytes(b[12..20].try_into().unwrap());
        let docs = u32_at(20);
        let nodes_at = 24;
        let offsets_at = nodes_at + n_nodes as usize * NODE_BYTES;
        let leaves_at = offsets_at + (n_leaves as usize + 1) * 8;
        if b.len() < leaves_at {
            return Err(bad("truncated nodes"));
        }
        let last = u64::from_le_bytes(b[leaves_at - 8..leaves_at].try_into().unwrap()) as usize;
        if b.len() < leaves_at + last {
            return Err(bad("truncated leaves"));
        }
        Ok(ShapeBkd { data, n_leaves, n_nodes, tris, docs, single, nodes_at, offsets_at, leaves_at })
    }

    /// Triangles in the part.
    pub fn size(&self) -> u64 {
        self.tris
    }

    /// Leaves in the part.
    pub fn leaf_count(&self) -> u32 {
        self.n_leaves
    }

    /// Distinct docs in the part.
    pub fn doc_count(&self) -> u32 {
        self.docs
    }

    pub fn bytes_len(&self) -> usize {
        self.data.as_slice().len()
    }

    #[inline]
    fn node(&self, i: u32) -> ([i32; 4], [i32; 4], u32, u32) {
        let b = self.data.as_slice();
        let at = self.nodes_at + i as usize * NODE_BYTES;
        let v = |k: usize| i32::from_le_bytes(b[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
        let u = |k: usize| u32::from_le_bytes(b[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
        ([v(0), v(1), v(2), v(3)], [v(4), v(5), v(6), v(7)], u(8), u(9))
    }

    /// The root's bounds (`getMinPackedValue` / `getMaxPackedValue`).
    pub fn root_bounds(&self) -> ([i32; 4], [i32; 4]) {
        let (lo, hi, _, _) = self.node(0);
        (lo, hi)
    }

    fn leaf_bytes(&self, leaf: u32) -> &[u8] {
        let b = self.data.as_slice();
        let o = |k: usize| u64::from_le_bytes(b[self.offsets_at + 8 * k..self.offsets_at + 8 * k + 8].try_into().unwrap()) as usize;
        let (s, e) = (o(leaf as usize), o(leaf as usize + 1));
        &b[self.leaves_at + s..self.leaves_at + e]
    }

    /// The distinct docs of a leaf, ascending (into `out.distinct`);
    /// returns the position after them.
    fn read_distinct(&self, b: &[u8], out: &mut LeafBuf) -> Option<usize> {
        let nd = u16::from_le_bytes([*b.get(2)?, *b.get(3)?]) as usize;
        let mut pos = 4;
        read_column(b, &mut pos, nd, &mut out.cols)?;
        out.distinct.clear();
        out.distinct.extend(out.cols.iter().map(|&d| d as u32));
        Some(pos)
    }

    /// The distinct doc ids of a leaf (`out.distinct`).
    pub fn leaf_docs(&self, leaf: u32, out: &mut LeafBuf) {
        let b = self.leaf_bytes(leaf);
        if self.read_distinct(b, out).is_none() {
            out.distinct.clear();
        }
    }

    /// A leaf's docs (per triangle) and triangles.
    pub fn read_leaf(&self, leaf: u32, out: &mut LeafBuf) {
        let b = self.leaf_bytes(leaf);
        let n = u16::from_le_bytes([b[0], b[1]]) as usize;
        out.docs.clear();
        out.tris.clear();
        let Some(mut pos) = self.read_distinct(b, out) else { return };
        let mut col = std::mem::take(&mut out.cols);
        if read_column(b, &mut pos, n, &mut col).is_none() {
            out.cols = col;
            return;
        }
        out.docs.extend(col.iter().map(|&i| out.distinct.get(i as usize).copied().unwrap_or(0)));
        out.cols = col;
        self.read_tris(b, pos, n, out);
    }

    /// A leaf's triangles alone (no doc ids: counting single-point parts).
    fn read_leaf_tris(&self, leaf: u32, out: &mut LeafBuf) {
        let b = self.leaf_bytes(leaf);
        let n = u16::from_le_bytes([b[0], b[1]]) as usize;
        let nd = u16::from_le_bytes([b[2], b[3]]) as usize;
        out.tris.clear();
        let mut pos = 4;
        if skip_column(b, &mut pos, nd).is_none() || skip_column(b, &mut pos, n).is_none() {
            return;
        }
        self.read_tris(b, pos, n, out);
    }

    /// The seven triangle columns from `pos` into `out.tris`.
    fn read_tris(&self, b: &[u8], mut pos: usize, n: usize, out: &mut LeafBuf) {
        let mut col = std::mem::take(&mut out.cols);
        out.tris.clear();
        out.tris.resize(n, [0; 7]);
        for k in 0..7 {
            if read_column(b, &mut pos, n, &mut col).is_none() {
                out.docs.clear();
                out.tris.clear();
                break;
            }
            for (t, &v) in out.tris.iter_mut().zip(col.iter()) {
                t[k] = match k {
                    2 => (t[0] as i64 + v as i64) as i32,
                    3 => (t[1] as i64 + v as i64) as i32,
                    4 => (t[0] as i64 + v as i64) as i32,
                    5 => (t[1] as i64 + v as i64) as i32,
                    _ => v,
                };
            }
        }
        out.cols = col;
    }

    /// Triangles under node `i`.
    fn node_count(&self, i: u32) -> u64 {
        let b = self.data.as_slice();
        let at = self.nodes_at + i as usize * NODE_BYTES + 40;
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as u64
    }

    /// The part's triangles.
    pub fn triangles(&self) -> u64 {
        if self.n_nodes == 0 {
            0
        } else {
            self.node_count(0)
        }
    }

    /// Whether every doc of the part has one triangle (single points):
    /// then nodes count their docs.
    pub fn one_triangle_per_doc(&self) -> bool {
        self.single
    }

    /// The docs matching `q` when every doc has one triangle
    /// ([`Self::one_triangle_per_doc`]): nodes the query takes whole count
    /// their triangles, crossing leaves test theirs (as [`Matcher`] decides
    /// each doc). `None`: a CONTAINS relation Lucene refuses.
    pub fn count_single(&self, q: &(dyn LeafQuery + Sync)) -> Option<u64> {
        self.count_single_at_least(q, u64::MAX)
    }

    /// [`Self::count_single`] capped at `limit` (`min(count, limit)`,
    /// exact): the nodes the query takes whole count first, from the node
    /// tree alone; only when they fall short of `limit` are the crossing
    /// leaves' triangles tested, the crossing subtrees in parallel.
    pub fn count_single_at_least(&self, q: &(dyn LeafQuery + Sync), limit: u64) -> Option<u64> {
        use rayon::prelude::*;
        if self.n_nodes == 0 {
            return Some(0);
        }
        let mut inside = 0u64;
        let mut crossing: Vec<u32> = Vec::new();
        self.frontier(0, q, &mut inside, &mut crossing);
        if inside >= limit {
            return Some(limit);
        }
        let work: u64 = crossing.iter().map(|&i| self.node_count(i)).sum();
        let count = |&i: &u32| -> Option<u64> {
            let (mut n, mut err) = (0u64, false);
            self.count_walk(i, q, &mut LeafBuf::default(), &mut n, &mut err);
            (!err).then_some(n)
        };
        let rest: Option<u64> = if work < 4 * FRONTIER_TRIANGLES || crossing.len() < 2 {
            crossing.iter().map(count).sum()
        } else {
            crossing.par_iter().map(count).sum()
        };
        Some((inside + rest?).min(limit))
    }

    /// The nodes under `i` the query takes whole (their triangles added to
    /// `inside`; none under CONTAINS, whose inside nodes never match) and
    /// the crossing subtrees of at most [`FRONTIER_TRIANGLES`] triangles (or
    /// leaves) whose triangles need testing.
    fn frontier(&self, i: u32, q: &dyn LeafQuery, inside: &mut u64, crossing: &mut Vec<u32>) {
        let (lo, hi, left, right) = self.node(i);
        match q.relate_node(&lo, &hi) {
            Relation::Inside => {
                if q.relation() != QueryRelation::Contains {
                    *inside += self.node_count(i);
                }
            }
            Relation::Outside => {}
            Relation::Crosses => {
                if left & LEAF != 0 || self.node_count(i) <= FRONTIER_TRIANGLES {
                    crossing.push(i);
                } else {
                    self.frontier(left, q, inside, crossing);
                    self.frontier(right, q, inside, crossing);
                }
            }
        }
    }

    fn count_walk(&self, i: u32, q: &dyn LeafQuery, buf: &mut LeafBuf, n: &mut u64, err: &mut bool) {
        let (lo, hi, left, right) = self.node(i);
        let rel = q.relation();
        match q.relate_node(&lo, &hi) {
            // CONTAINS: inside excludes; outside never matches.
            Relation::Inside => {
                if rel != QueryRelation::Contains {
                    *n += self.node_count(i);
                }
            }
            Relation::Outside => {}
            Relation::Crosses => {
                if left & LEAF != 0 {
                    self.read_leaf_tris(left & !LEAF, buf);
                    for t in &buf.tris {
                        let t = Tri::decode(t);
                        let hit = match rel {
                            QueryRelation::Contains => match q.leaf_within(&t) {
                                Ok(w) => w == WithinRelation::Candidate,
                                Err(_) => {
                                    *err = true;
                                    false
                                }
                            },
                            _ => q.leaf_matches(&t),
                        };
                        *n += u64::from(hit);
                    }
                } else {
                    self.count_walk(left, q, buf, n, err);
                    self.count_walk(right, q, buf, n, err);
                }
            }
        }
    }

    /// Every leaf under node `i`.
    /// The first and last leaf under node `i`.
    fn leaf_range(&self, i: u32) -> (u32, u32) {
        let mut a = i;
        loop {
            let (_, _, left, _) = self.node(a);
            if left & LEAF != 0 {
                break;
            }
            a = left;
        }
        let mut b = i;
        loop {
            let (_, _, left, right) = self.node(b);
            if left & LEAF != 0 {
                break;
            }
            b = right;
        }
        (self.node(a).2 & !LEAF, self.node(b).2 & !LEAF)
    }

    fn leaves_under(&self, i: u32, f: &mut dyn FnMut(u32)) {
        let (_, _, left, right) = self.node(i);
        if left & LEAF != 0 {
            f(left & !LEAF);
        } else {
            self.leaves_under(left, f);
            self.leaves_under(right, f);
        }
    }

    /// The walk of node `i` with walkers from `mk`, the top levels' crossing
    /// subtrees in parallel (each its own walker; their errors into
    /// `error`) when both children have work and enough of it. A crossing
    /// node with one child the walk skips goes on down the other (the split
    /// may come lower, as for a small query over many triangles); a node
    /// with little work under it walks on this thread.
    fn walk_par<'w>(&self, i: u32, depth: u32, mk: &(dyn Fn() -> SharedMatcher<'w> + Sync), error: &std::sync::Mutex<Option<String>>) {
        let (lo, hi, left, right) = self.node(i);
        if depth < 5 && left & LEAF == 0 && self.node_count(i) >= 2 * PAR_WALK_MIN {
            let mut w = mk();
            if w.compare(&lo, &hi) == Relation::Crosses {
                match (self.child_work(left, &mut w), self.child_work(right, &mut w)) {
                    (Some(a), Some(b)) if a >= PAR_WALK_MIN && b >= PAR_WALK_MIN => {
                        rayon::join(|| self.walk_par(left, depth + 1, mk, error), || self.walk_par(right, depth + 1, mk, error));
                        return;
                    }
                    (Some(_), None) => return self.walk_par(left, depth, mk, error),
                    (None, Some(_)) => return self.walk_par(right, depth, mk, error),
                    (None, None) => return,
                    _ => {}
                }
            }
        }
        let mut w = mk();
        self.walk(i, &mut w, &mut LeafBuf::default());
        if let Some(e) = w.error {
            error.lock().unwrap().get_or_insert(e);
        }
    }

    /// The triangles the walk reads under child `c` (`None`: it skips the
    /// child, outside or inside with those docs unwanted).
    fn child_work(&self, c: u32, w: &mut dyn TreeWalk) -> Option<u64> {
        let (lo, hi, _, _) = self.node(c);
        let reads = match w.compare(&lo, &hi) {
            Relation::Outside => w.wants_outside(),
            Relation::Inside => w.wants_inside(),
            Relation::Crosses => true,
        };
        reads.then(|| self.node_count(c))
    }

    /// `PointValues.intersect` with a visitor: `compare` relates nodes,
    /// `inside` takes every doc of a node, `leaf` gets the docs and
    /// triangles of a crossing leaf.
    pub fn intersect(&self, v: &mut dyn TreeWalk, buf: &mut LeafBuf) {
        if self.n_nodes > 0 {
            self.walk(0, v, buf);
        }
    }

    /// [`ShapeBkd::intersect`] with the nodes the walk takes whole first:
    /// the tree's nodes are related as the walk relates them, inside and
    /// outside nodes handled at once, crossing leaves only after every
    /// node is (each as `intersect` would handle it), so a walk that stops
    /// at a count gets there from whole nodes before testing triangles.
    /// The same calls as `intersect`, in another order.
    pub fn intersect_whole_first(&self, v: &mut dyn TreeWalk, buf: &mut LeafBuf) {
        let crossing = self.intersect_whole_nodes(v, buf);
        for leaf in crossing {
            if v.done() {
                return;
            }
            self.visit_crossing_leaf(leaf, v, buf);
        }
    }

    /// A crossing leaf as `walk` hands it over, skipped when the walk
    /// has already decided all of its docs.
    fn visit_crossing_leaf(&self, leaf: u32, v: &mut dyn TreeWalk, buf: &mut LeafBuf) {
        if !v.wants_crossing_leaves() {
            return;
        }
        if v.wants_leaf_triangles() {
            self.leaf_docs(leaf, buf);
            if v.skip_leaf(&buf.distinct) {
                return;
            }
            self.read_leaf(leaf, buf);
            v.leaf(&buf.docs, &buf.tris);
        } else {
            self.leaf_docs(leaf, buf);
            v.leaf_docs(&buf.distinct);
        }
    }

    /// The first half of [`ShapeBkd::intersect_whole_first`]: every node
    /// related and the ones the walk takes whole (inside or outside)
    /// handled; the crossing leaves left, returned.
    pub fn intersect_whole_nodes(&self, v: &mut dyn TreeWalk, buf: &mut LeafBuf) -> Vec<u32> {
        let mut crossing: Vec<u32> = Vec::new();
        if self.n_nodes > 0 {
            self.walk_deferring(0, v, buf, &mut crossing);
        }
        crossing
    }

    fn walk_deferring(&self, i: u32, v: &mut dyn TreeWalk, buf: &mut LeafBuf, crossing: &mut Vec<u32>) {
        if v.done() {
            return;
        }
        let (lo, hi, left, right) = self.node(i);
        match v.compare(&lo, &hi) {
            Relation::Crosses => {
                if left & LEAF != 0 {
                    crossing.push(left & !LEAF);
                } else {
                    self.walk_deferring(left, v, buf, crossing);
                    self.walk_deferring(right, v, buf, crossing);
                }
            }
            // Inside or outside: as `walk` handles the node.
            _ => self.walk(i, v, buf),
        }
    }

    fn walk(&self, i: u32, v: &mut dyn TreeWalk, buf: &mut LeafBuf) {
        if v.done() {
            return;
        }
        let (lo, hi, left, right) = self.node(i);
        match v.compare(&lo, &hi) {
            Relation::Outside => {
                if v.wants_outside() {
                    self.each_doc(i, buf, &mut |d| v.outside_doc(d));
                }
            }
            Relation::Inside => {
                if v.wants_inside() {
                    // A node's leaves are a run of leaf ids (built depth
                    // first); a walk that is done stops between leaves.
                    let (a, b) = self.leaf_range(i);
                    for leaf in a..=b {
                        if v.done() {
                            break;
                        }
                        self.leaf_docs(leaf, buf);
                        for &d in &buf.distinct {
                            v.inside_doc(d);
                        }
                    }
                }
            }
            Relation::Crosses => {
                if left & LEAF != 0 {
                    self.visit_crossing_leaf(left & !LEAF, v, buf);
                } else {
                    self.walk(left, v, buf);
                    self.walk(right, v, buf);
                }
            }
        }
    }

    /// Every doc of node `i`'s triangles (once per leaf holding it).
    fn each_doc(&self, i: u32, buf: &mut LeafBuf, f: &mut dyn FnMut(DocId)) {
        // A node's leaves are a run of leaf ids (built depth first).
        let (a, b) = self.leaf_range(i);
        for leaf in a..=b {
            self.leaf_docs(leaf, buf);
            for &d in &buf.distinct {
                f(d);
            }
        }
    }

    /// Every `(doc, triangle)` of the part (merges).
    pub fn for_each(&self, f: &mut dyn FnMut(DocId, &[i32; 7])) {
        let mut buf = LeafBuf::default();
        for leaf in 0..self.n_leaves {
            self.read_leaf(leaf, &mut buf);
            for (d, t) in buf.docs.iter().zip(&buf.tris) {
                f(*d, t);
            }
        }
    }
}

/// A walk over the tree (`IntersectVisitor`).
pub trait TreeWalk {
    fn compare(&mut self, min: &[i32; 4], max: &[i32; 4]) -> Relation;
    fn inside_doc(&mut self, doc: DocId);
    /// The walk has what it needs: no more nodes or leaves are visited.
    fn done(&self) -> bool {
        false
    }
    /// Whether the docs of nodes inside are wanted ([`TreeWalk::inside_doc`]).
    fn wants_inside(&self) -> bool {
        true
    }
    /// Whether the docs of nodes outside are wanted ([`TreeWalk::outside_doc`]).
    fn wants_outside(&self) -> bool {
        false
    }
    fn outside_doc(&mut self, _doc: DocId) {}
    fn leaf(&mut self, docs: &[DocId], tris: &[[i32; 7]]);
    /// Whether crossing leaves are wanted with their triangles
    /// ([`TreeWalk::leaf`]) or only their distinct docs
    /// ([`TreeWalk::leaf_docs`]).
    fn wants_leaf_triangles(&self) -> bool {
        true
    }
    fn leaf_docs(&mut self, _docs: &[DocId]) {}
    /// Whether crossing leaves are wanted at all (a pass over the nodes
    /// outside only, Lucene's shallow inverse visitor, has nothing to do in
    /// them and does not read them).
    fn wants_crossing_leaves(&self) -> bool {
        true
    }
    /// Whether a crossing leaf with these distinct docs can be skipped
    /// (every doc already decided): its triangles are then not decoded.
    fn skip_leaf(&self, _distinct: &[DocId]) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Queries

/// Docs as bits (`bit d % 64` of word `d / 64`).
fn set(bits: &mut [u64], d: DocId) {
    if let Some(w) = bits.get_mut(d as usize / 64) {
        *w |= 1 << (d % 64);
    }
}
fn get(bits: &[u64], d: DocId) -> bool {
    bits.get(d as usize / 64).is_some_and(|w| w >> (d % 64) & 1 == 1)
}

/// The docs of a segment's tree parts matching `q` (`num_docs` docs):
/// parts walk in parallel, each with its own doc state, combined as one
/// walk over them all would (a doc matches somewhere and is excluded
/// nowhere).
pub fn match_parts(q: &(dyn LeafQuery + Sync), parts: &[ShapeBkd], num_docs: u32) -> std::result::Result<Vec<u64>, String> {
    match_parts_as(q, parts, num_docs, false)
}

/// CONTAINS candidates: the docs with a triangle that is a `CANDIDATE`,
/// ignoring every exclusion (nodes inside, `NOTWITHIN` triangles), so no
/// node inside is read. A superset of the matches; callers decide each
/// candidate exactly (its doc value).
pub fn contains_candidates(q: &(dyn LeafQuery + Sync), parts: &[ShapeBkd], num_docs: u32) -> std::result::Result<Vec<u64>, String> {
    match_parts_as(q, parts, num_docs, true)
}

fn match_parts_as(q: &(dyn LeafQuery + Sync), parts: &[ShapeBkd], num_docs: u32, candidates: bool) -> std::result::Result<Vec<u64>, String> {
    use rayon::prelude::*;
    use std::sync::atomic::AtomicU64;
    let words = (num_docs as usize).div_ceil(64);
    let rel = q.relation();
    // Zeroed allocations (the allocator's fresh zero pages: only the pages
    // of set bits are ever written), shared as atomics by the walks.
    let mut out = vec![0u64; words];
    // Exclusions only where a doc may have triangles in several places.
    let excludes = !candidates && rel != QueryRelation::Intersects && !parts.iter().all(ShapeBkd::one_triangle_per_doc);
    let mut excl: Vec<u64> = if excludes { vec![0; words] } else { Vec::new() };
    // SAFETY: `AtomicU64` has `u64`'s size and alignment; the vectors are
    // only reached through these views until the walks are done.
    let result: &[AtomicU64] = unsafe { std::slice::from_raw_parts(out.as_mut_ptr() as *const AtomicU64, out.len()) };
    let excluded: &[AtomicU64] = unsafe { std::slice::from_raw_parts(excl.as_mut_ptr() as *const AtomicU64, excl.len()) };
    let error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
    // `outside_pass`: the second pass of a two-pass walk, which may only
    // start once every part's first pass is done (a doc's triangles may be
    // in several parts: its match may come from another part).
    let walk_part = |p: &ShapeBkd, outside_pass: bool| {
        if p.n_nodes == 0 {
            return;
        }
        let single = p.one_triangle_per_doc();
        let two_pass = excludes && !single && matches!(rel, QueryRelation::Within | QueryRelation::Disjoint);
        let inverse = !candidates && !single && rel == QueryRelation::Disjoint && parts.len() == 1 && p.doc_count() == num_docs;
        if outside_pass {
            if two_pass && !inverse {
                let mk = || SharedMatcher { q, result, excluded, single, candidates, error: None, phase: Phase::Outside };
                p.walk_par(0, 0, &mk, &error);
            }
            return;
        }
        // `SpatialQuery.getScorerSupplier`: the root relation first.
        let (lo, hi) = p.root_bounds();
        if rel == QueryRelation::Contains && q.relate_node(&lo, &hi) == Relation::Inside {
            // Every triangle inside: no doc of this part contains the query.
            if excludes && !single {
                p.each_doc(0, &mut LeafBuf::default(), &mut |d| set_atomic(excluded, d));
            }
            return;
        }
        // WITHIN / DISJOINT over docs of several triangles: Lucene's
        // `getDenseScorer`: the matches from the nodes the query takes and
        // the crossing leaves first (nodes outside not read), then a pass
        // over the nodes outside only (`getShallowInverseDenseVisitor`)
        // drops the matched docs found there. Setting an exclusion bit for
        // every doc outside (millions of contended atomic writes) cost more
        // than the walks.
        // DISJOINT where every doc of the segment has triangles in this one
        // part (Lucene's `getDenseScorer` when `docCount == maxDoc`, with
        // `getInverseDenseVisitor`): every doc starts matched and one walk
        // clears the docs of the nodes the query covers and of crossing
        // triangles that intersect it; nodes away from the query are not
        // read.
        if inverse {
            for (i, w) in result.iter().enumerate() {
                let bits = if (i + 1) * 64 <= num_docs as usize { u64::MAX } else { (1u64 << (num_docs as usize - i * 64)) - 1 };
                w.store(bits, std::sync::atomic::Ordering::Relaxed);
            }
            let mk = || SharedMatcher { q, result, excluded, single, candidates, error: None, phase: Phase::Inverse };
            p.walk_par(0, 0, &mk, &error);
            return;
        }
        if two_pass {
            let mk = || SharedMatcher { q, result, excluded, single, candidates, error: None, phase: Phase::Matches };
            p.walk_par(0, 0, &mk, &error);
            return;
        }
        let mk = || SharedMatcher { q, result, excluded, single, candidates, error: None, phase: Phase::One };
        p.walk_par(0, 0, &mk, &error);
    };
    // One part (a merged segment's usual case) walks on this thread, split
    // only where its subtrees have work.
    for outside_pass in [false, true] {
        if parts.len() == 1 {
            walk_part(&parts[0], outside_pass);
        } else {
            parts.par_iter().for_each(|p| walk_part(p, outside_pass));
        }
    }
    if let Some(e) = error.into_inner().unwrap_or(None) {
        return Err(e);
    }
    if !excl.is_empty() {
        for (a, b) in out.iter_mut().zip(&excl) {
            *a &= !b;
        }
    }
    Ok(out)
}

#[inline]
fn set_atomic(bits: &[std::sync::atomic::AtomicU64], d: DocId) {
    if let Some(w) = bits.get(d as usize / 64) {
        w.fetch_or(1 << (d % 64), std::sync::atomic::Ordering::Relaxed);
    }
}

#[inline]
fn clear_atomic(bits: &[std::sync::atomic::AtomicU64], d: DocId) {
    if let Some(w) = bits.get(d as usize / 64) {
        // A plain load first: most clears in a pass hit cleared bits.
        let m = 1u64 << (d % 64);
        if w.load(std::sync::atomic::Ordering::Relaxed) & m != 0 {
            w.fetch_and(!m, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[inline]
fn get_atomic(bits: &[std::sync::atomic::AtomicU64], d: DocId) -> bool {
    bits.get(d as usize / 64).is_some_and(|w| w.load(std::sync::atomic::Ordering::Relaxed) >> (d % 64) & 1 == 1)
}

/// [`Matcher`] over doc bits shared by the walks of a segment's subtrees
/// running in parallel: a doc's result and exclusion bits are set in any
/// order (a doc matches when set in `result` and not in `excluded`, as one
/// walk decides it); the reads of `excluded` only skip work.
struct SharedMatcher<'a> {
    q: &'a (dyn LeafQuery + Sync),
    result: &'a [std::sync::atomic::AtomicU64],
    excluded: &'a [std::sync::atomic::AtomicU64],
    single: bool,
    candidates: bool,
    error: Option<String>,
    phase: Phase,
}

/// Which part of the WITHIN / DISJOINT work a [`SharedMatcher`] walk does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Everything in one walk.
    One,
    /// Nodes inside and crossing leaves; nodes outside not read.
    Matches,
    /// Nodes outside only: their docs already matched are excluded.
    Outside,
    /// Every doc starts matched: docs of nodes outside and of failing
    /// crossing triangles are cleared from the result.
    Inverse,
}

impl SharedMatcher<'_> {
    #[inline]
    fn exclude(&self, d: DocId) {
        set_atomic(self.excluded, d);
    }

    #[inline]
    fn is_excluded(&self, d: DocId) -> bool {
        !self.excluded.is_empty() && get_atomic(self.excluded, d)
    }
}

impl TreeWalk for SharedMatcher<'_> {
    fn compare(&mut self, min: &[i32; 4], max: &[i32; 4]) -> Relation {
        self.q.relate_node(min, max)
    }

    fn inside_doc(&mut self, doc: DocId) {
        match self.q.relation() {
            QueryRelation::Contains => {
                if !self.single && !self.candidates {
                    self.exclude(doc);
                }
            }
            _ => set_atomic(self.result, doc),
        }
    }

    fn wants_inside(&self) -> bool {
        !matches!(self.phase, Phase::Outside | Phase::Inverse) && !((self.single || self.candidates) && self.q.relation() == QueryRelation::Contains)
    }

    fn wants_outside(&self) -> bool {
        self.phase != Phase::Matches && !self.single && matches!(self.q.relation(), QueryRelation::Within | QueryRelation::Disjoint)
    }

    fn wants_crossing_leaves(&self) -> bool {
        self.phase != Phase::Outside
    }

    fn outside_doc(&mut self, doc: DocId) {
        if self.phase == Phase::Inverse {
            clear_atomic(self.result, doc);
        } else if self.phase == Phase::Outside {
            // Only a doc that matched needs dropping (plain loads for the
            // rest, no write).
            if get_atomic(self.result, doc) && !self.is_excluded(doc) {
                self.exclude(doc);
            }
        } else {
            self.exclude(doc);
        }
    }

    fn leaf(&mut self, docs: &[DocId], tris: &[[i32; 7]]) {
        match self.q.relation() {
            QueryRelation::Intersects => {
                for (&d, t) in docs.iter().zip(tris) {
                    if !get_atomic(self.result, d) && self.q.leaf_matches(&Tri::decode(t)) {
                        set_atomic(self.result, d);
                    }
                }
            }
            QueryRelation::Within | QueryRelation::Disjoint if self.phase == Phase::Inverse => {
                for (&d, t) in docs.iter().zip(tris) {
                    if get_atomic(self.result, d) && !self.q.leaf_matches(&Tri::decode(t)) {
                        clear_atomic(self.result, d);
                    }
                }
            }
            QueryRelation::Within | QueryRelation::Disjoint => {
                for (&d, t) in docs.iter().zip(tris) {
                    if !self.is_excluded(d) {
                        if self.q.leaf_matches(&Tri::decode(t)) {
                            set_atomic(self.result, d);
                        } else if !self.single {
                            self.exclude(d);
                        }
                    }
                }
            }
            QueryRelation::Contains => {
                for (&d, t) in docs.iter().zip(tris) {
                    if !self.is_excluded(d) {
                        match self.q.leaf_within(&Tri::decode(t)) {
                            Ok(WithinRelation::Candidate) => set_atomic(self.result, d),
                            Ok(WithinRelation::NotWithin) => {
                                if !self.single && !self.candidates {
                                    self.exclude(d);
                                }
                            }
                            Ok(WithinRelation::Disjoint) => {}
                            Err(e) => {
                                self.error.get_or_insert(e);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The doc-level state of one relation across a segment's parts (and the
/// docs of a memtable, related triangle by triangle).
pub struct Matcher<'q> {
    q: &'q dyn LeafQuery,
    pub result: Vec<u64>,
    /// Docs a relation excludes; allocated on first use.
    pub excluded: Vec<u64>,
    pub error: Option<String>,
    words: usize,
    /// The part being walked holds one triangle per doc (single points):
    /// no doc has triangles elsewhere, so nothing needs excluding (a doc's
    /// one triangle decides it) and WITHIN / DISJOINT / CONTAINS keep no
    /// exclusion bits.
    single: bool,
    /// CONTAINS candidates only ([`contains_candidates`]).
    candidates: bool,
}

impl<'q> Matcher<'q> {
    pub fn new(q: &'q dyn LeafQuery, num_docs: u32) -> Matcher<'q> {
        let words = (num_docs as usize).div_ceil(64);
        Matcher { q, result: vec![0; words], excluded: Vec::new(), error: None, words, single: false, candidates: false }
    }

    #[inline]
    fn exclude(&mut self, d: DocId) {
        if self.excluded.is_empty() {
            self.excluded = vec![0; self.words];
        }
        set(&mut self.excluded, d);
    }

    #[inline]
    fn is_excluded(&self, d: DocId) -> bool {
        !self.excluded.is_empty() && get(&self.excluded, d)
    }

    /// The matching docs.
    pub fn finish(mut self) -> std::result::Result<Vec<u64>, String> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        if !self.excluded.is_empty() {
            for (r, x) in self.result.iter_mut().zip(&self.excluded) {
                *r &= !x;
            }
        }
        Ok(self.result)
    }

    /// Relate the triangles of one part.
    pub fn part(&mut self, bkd: &ShapeBkd, buf: &mut LeafBuf) {
        self.single = bkd.one_triangle_per_doc();
        // `SpatialQuery.getScorerSupplier`: the root relation first.
        let (lo, hi) = bkd.root_bounds();
        let r = self.q.relate_node(&lo, &hi);
        if self.q.relation() == QueryRelation::Contains && r == Relation::Inside {
            // Every triangle inside: no doc of this part contains the query
            // (with one triangle a doc, none is anywhere else either).
            if !self.single && !self.candidates {
                bkd.each_doc(0, buf, &mut |d| self.exclude(d));
            }
            return;
        }
        bkd.intersect(self, buf);
        self.single = false;
    }

    /// Relate one doc's triangles (a memtable doc; the same per-triangle
    /// tests as the tree's leaves).
    pub fn doc(&mut self, doc: DocId, tris: &[[i32; 7]]) {
        self.single = false;
        let docs = vec![doc; tris.len()];
        TreeWalk::leaf(self, &docs, tris);
    }
}

impl TreeWalk for Matcher<'_> {
    fn compare(&mut self, min: &[i32; 4], max: &[i32; 4]) -> Relation {
        self.q.relate_node(min, max)
    }

    fn inside_doc(&mut self, doc: DocId) {
        match self.q.relation() {
            // Every triangle of the node fails `contains` (its points are in
            // the query): `getContainsDenseVisitor.visit(docID)` excludes.
            QueryRelation::Contains => {
                if !self.single {
                    self.exclude(doc);
                }
            }
            _ => set(&mut self.result, doc),
        }
    }

    fn wants_inside(&self) -> bool {
        // CONTAINS over one triangle a doc: inside docs only get excluded,
        // and nothing else could make them match.
        !((self.single || self.candidates) && self.q.relation() == QueryRelation::Contains)
    }

    fn wants_outside(&self) -> bool {
        // WITHIN / DISJOINT: docs with triangles in a node outside (after
        // the transposed relation, a node that fails as a whole) are
        // excluded (`getShallowInverseDenseVisitor`); a single-triangle doc
        // there just never matches.
        !self.single && matches!(self.q.relation(), QueryRelation::Within | QueryRelation::Disjoint)
    }

    fn outside_doc(&mut self, doc: DocId) {
        self.exclude(doc);
    }

    fn leaf(&mut self, docs: &[DocId], tris: &[[i32; 7]]) {
        match self.q.relation() {
            QueryRelation::Intersects => {
                for (&d, t) in docs.iter().zip(tris) {
                    if !get(&self.result, d) && self.q.leaf_matches(&Tri::decode(t)) {
                        set(&mut self.result, d);
                    }
                }
            }
            QueryRelation::Within | QueryRelation::Disjoint if self.single => {
                for (&d, t) in docs.iter().zip(tris) {
                    if self.q.leaf_matches(&Tri::decode(t)) {
                        set(&mut self.result, d);
                    }
                }
            }
            QueryRelation::Within | QueryRelation::Disjoint => {
                for (&d, t) in docs.iter().zip(tris) {
                    if !self.is_excluded(d) {
                        if self.q.leaf_matches(&Tri::decode(t)) {
                            set(&mut self.result, d);
                        } else {
                            self.exclude(d);
                        }
                    }
                }
            }
            QueryRelation::Contains => {
                for (&d, t) in docs.iter().zip(tris) {
                    if !self.is_excluded(d) {
                        match self.q.leaf_within(&Tri::decode(t)) {
                            Ok(WithinRelation::Candidate) => set(&mut self.result, d),
                            Ok(WithinRelation::NotWithin) => {
                                if !self.single && !self.candidates {
                                    self.exclude(d);
                                }
                            }
                            Ok(WithinRelation::Disjoint) => {}
                            Err(e) => {
                                self.error.get_or_insert(e);
                                self.exclude(d);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fieldtypes::geo2d::{ComponentLeaf, LatLonGeometry, RectLeaf, ShapeQuery};
    use crate::fieldtypes::shape::{parse_wkt, ParseOpts};
    use crate::fieldtypes::tessellator::geometry_triangles;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn corpus(n: u32) -> (Vec<Vec<[i32; 7]>>, Vec<Entry>) {
        let mut r = Lcg(9);
        let mut docs = Vec::new();
        let mut es = Vec::new();
        for d in 0..n {
            let (x, y) = (-20.0 + 40.0 * r.next(), -20.0 + 40.0 * r.next());
            let s = 0.05 + 2.0 * r.next();
            let w = match d % 4 {
                0 => format!("POINT ({x} {y})"),
                1 => format!("LINESTRING ({x} {y}, {} {}, {} {})", x + s, y + s / 2.0, x + 2.0 * s, y),
                2 => format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + s, x + s, y + s, y + s),
                _ => format!("MULTIPOLYGON ((({x} {y}, {} {y}, {} {}, {x} {y})), (({} {}, {} {}, {} {}, {} {})))", x + s, x + s, y + s, x + 5.0, y + 5.0, x + 5.0 + s, y + 5.0, x + 5.0, y + 5.0 + s, x + 5.0, y + 5.0),
            };
            let g = parse_wkt(&w, &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
            let mut t = Vec::new();
            geometry_triangles(&g, &mut t).unwrap();
            for tri in &t {
                es.push(Entry { tri: *tri, doc: d });
            }
            docs.push(t);
        }
        (docs, es)
    }

    fn brute(q: &dyn LeafQuery, docs: &[Vec<[i32; 7]>]) -> Vec<u64> {
        let mut m = Matcher::new(q, docs.len() as u32);
        for (d, ts) in docs.iter().enumerate() {
            m.doc(d as u32, ts);
        }
        m.finish().unwrap()
    }

    /// `match_parts` (the parallel walks: WITHIN / DISJOINT in two passes
    /// over several parts, DISJOINT's inverse walk over one part holding
    /// every doc) against the relation decided doc by doc.
    #[test]
    fn match_parts_matches_brute_force() {
        let (docs, es) = corpus(6000);
        // Every third doc without a shape (no inverse walk then).
        let sparse_docs: Vec<Vec<[i32; 7]>> = docs.iter().enumerate().map(|(d, t)| if d % 3 == 1 { Vec::new() } else { t.clone() }).collect();
        let open = |parts: Vec<Vec<u8>>| -> Vec<ShapeBkd> { parts.into_iter().map(|p| ShapeBkd::open(FileSlice::from_vec(p)).unwrap()).collect() };
        // One part holding every doc.
        let one = open(build_parts(&mut es.clone()));
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].doc_count(), docs.len() as u32);
        // Three parts, a doc's triangles spread over them (every third
        // triangle in turn).
        let mut chunks: [Vec<Entry>; 3] = Default::default();
        for (i, e) in es.iter().enumerate() {
            chunks[i % 3].push(*e);
        }
        // (Not whole docs: built as a merge builds parts.)
        let three = open(chunks.into_iter().flat_map(|mut c| build_parts_with(&mut c, false)).collect());
        // One part, some docs without a shape.
        let mut sparse: Vec<Entry> = es.iter().copied().filter(|e| e.doc % 3 != 1).collect();
        let one_sparse = open(build_parts(&mut sparse));
        let geoms = [
            vec![LatLonGeometry::Polygon(vec![vec![(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)]])],
            vec![LatLonGeometry::Rectangle { min_lat: -3.0, max_lat: 7.0, min_lon: -10.0, max_lon: 2.0 }],
            vec![LatLonGeometry::Polygon(vec![vec![(-12.0, -12.0), (12.0, -12.0), (0.0, 14.0), (-12.0, -12.0)], vec![(-2.0, -2.0), (2.0, -2.0), (0.0, 2.0), (-2.0, -2.0)]])],
        ];
        for g in &geoms {
            for rel in [QueryRelation::Intersects, QueryRelation::Within, QueryRelation::Disjoint] {
                let q = ShapeQuery::new(g, rel);
                let leaf: Box<dyn LeafQuery + Sync> = match &q {
                    ShapeQuery::Component(c, r) => Box::new(ComponentLeaf { component: c, rel: *r }),
                    ShapeQuery::Rect(e, _, r) => Box::new(RectLeaf { rect: e, rel: *r }),
                    _ => continue,
                };
                let want = brute(leaf.as_ref(), &docs);
                for (name, parts) in [("one part", &one), ("three parts", &three)] {
                    let got = match_parts(leaf.as_ref(), parts, docs.len() as u32).unwrap();
                    let n: u32 = got.iter().map(|w| w.count_ones()).sum();
                    assert_eq!(got, want, "{name} {g:?} {rel:?} ({n} docs)");
                }
                // Docs without a shape are not in the tree: never matched.
                let want = {
                    let mut m = Matcher::new(leaf.as_ref(), sparse_docs.len() as u32);
                    for (d, ts) in sparse_docs.iter().enumerate().filter(|(_, ts)| !ts.is_empty()) {
                        m.doc(d as u32, ts);
                    }
                    m.finish().unwrap()
                };
                let got = match_parts(leaf.as_ref(), &one_sparse, docs.len() as u32).unwrap();
                assert_eq!(got, want, "sparse {g:?} {rel:?}");
            }
        }
    }

    #[test]
    fn tree_walks_match_triangle_by_triangle() {
        let (docs, mut es) = corpus(6000);
        let parts = build_parts(&mut es);
        let bkds: Vec<ShapeBkd> = parts.into_iter().map(|p| ShapeBkd::open(FileSlice::from_vec(p)).unwrap()).collect();
        let total: u64 = bkds.iter().map(ShapeBkd::size).sum();
        assert_eq!(total, docs.iter().map(|t| t.len() as u64).sum::<u64>());
        // Every triangle back.
        let mut back = 0u64;
        for b in &bkds {
            b.for_each(&mut |d, t| {
                assert!(docs[d as usize].contains(t));
                back += 1;
            });
        }
        assert_eq!(back, total);
        let geoms = [
            vec![LatLonGeometry::Polygon(vec![vec![(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)]])],
            vec![LatLonGeometry::Rectangle { min_lat: -3.0, max_lat: 7.0, min_lon: -10.0, max_lon: 2.0 }],
            vec![LatLonGeometry::Line(vec![(-15.0, -15.0), (15.0, 10.0)])],
            vec![LatLonGeometry::Point { lat: 1.0, lon: 1.0 }],
            vec![LatLonGeometry::Polygon(vec![vec![(-5.0, -5.0), (5.0, -5.0), (0.0, 8.0), (-5.0, -5.0)]]), LatLonGeometry::Point { lat: 10.0, lon: 10.0 }],
        ];
        for g in &geoms {
            for rel in [QueryRelation::Intersects, QueryRelation::Within, QueryRelation::Disjoint, QueryRelation::Contains] {
                if rel == QueryRelation::Within && matches!(g[0], LatLonGeometry::Line(_)) {
                    continue;
                }
                let mut queries = vec![];
                match ShapeQuery::new(g, rel) {
                    ShapeQuery::And(v) => queries.extend(v),
                    q => queries.push(q),
                }
                for q in &queries {
                    let leaf: Box<dyn LeafQuery> = match q {
                        ShapeQuery::Component(c, r) => Box::new(ComponentLeaf { component: c, rel: *r }),
                        ShapeQuery::Rect(e, _, r) => Box::new(RectLeaf { rect: e, rel: *r }),
                        _ => unreachable!(),
                    };
                    let want = brute(leaf.as_ref(), &docs);
                    let mut m = Matcher::new(leaf.as_ref(), docs.len() as u32);
                    let mut buf = LeafBuf::default();
                    for b in &bkds {
                        m.part(b, &mut buf);
                    }
                    let got = m.finish().unwrap();
                    let n: u32 = got.iter().map(|w| w.count_ones()).sum();
                    assert_eq!(got, want, "{g:?} {rel:?} ({n} docs)");
                }
            }
        }
    }
}
