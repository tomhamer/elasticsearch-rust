//! ES's `geo_shape` doc value (`BinaryShapeDocValuesField` /
//! `GeometryDocValueWriter`, 9.5.5): per doc, the centroid of all its
//! values (encoded x / y, the highest dimension, the summed weight), the
//! `Extent` of its triangles, and the triangles themselves in a tree
//! (`TriangleTreeWriter`) that relations walk with pruning
//! (`TriangleTreeReader` + a `TriangleTreeVisitor`). Same layout as ES:
//!
//! `int centroidX, int centroidY` (big endian), `byte dimension`,
//! `vlong doubleToLongBits(sumWeight)`, the compressed extent, then the
//! root node: `byte metadata`, the component's coordinates as vlongs below
//! the node's max x / y, then the children (each `vlong dMaxX, vlong
//! dMaxY, vint size`, the right one prefixed by its size below the root).
//!
//! The visitors are ES's: [`Component2DVisitor`] (the four query
//! relations), [`RelationVisitor`] (`Component2DRelationVisitor`, a
//! [`GeoRelation`] to a component), [`TileVisitor`] (`Tile2DVisitor`, a
//! relation to an encoded box), and the label position.
//!
//! [`ShapeDvColumn`] holds a field's doc values in a segment: parts over
//! consecutive docs, each with the docs' headers decoded into fixed-width
//! slots ([`Header`]: aggregations read the centroid and extent without
//! decoding anything) and the trees after them ([`DvPart`]).

use super::geo2d::{Component, QueryRelation, Relation, WithinRelation};
use super::shape::{CentroidCalc, ShapeDim, ShapeExtent};
use crate::codec::varint::{read_vint, write_vint};
use crate::geo::{decode_lat, decode_lon};

const LEFT: u8 = 1;
const RIGHT: u8 = 1 << 1;
const POINT: u8 = 1 << 2;
const LINE: u8 = 1 << 3;
const AB_FROM_TRIANGLE: u8 = 1 << 4;
const BC_FROM_TRIANGLE: u8 = 1 << 5;
const CA_FROM_TRIANGLE: u8 = 1 << 6;

/// ES `GeoRelation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoRelation {
    QueryCrosses,
    QueryInside,
    QueryDisjoint,
    QueryContains,
}

// ---------------------------------------------------------------------------
// Writing

#[derive(Clone, Copy)]
struct Node {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    t: super::tessellator::Decoded,
    left: u32,
    right: u32,
}

const NONE: u32 = u32::MAX;

fn node_of(v: &[i32; 7]) -> Node {
    let t = super::tessellator::decode_triangle(v);
    Node { min_x: t.ax.min(t.bx).min(t.cx), max_x: t.ax.max(t.bx).max(t.cx), min_y: t.ay.min(t.by).min(t.cy), max_y: t.ay.max(t.by).max(t.cy), t, left: NONE, right: NONE }
}

fn vlong_len(v: u64) -> usize {
    let mut n = 1;
    let mut v = v >> 7;
    while v != 0 {
        n += 1;
        v >>= 7;
    }
    n
}

fn put_be_i32(v: i32, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// ES `Extent.writeCompressed`.
pub fn write_extent(e: &ShapeExtent, out: &mut Vec<u8>) {
    put_be_i32(e.top, out);
    write_vint((e.top as i64 - e.bottom as i64) as u64, out);
    let (neg, pos) = (e.has_neg(), e.has_pos());
    let ty = if !neg {
        if !pos {
            0
        } else {
            1
        }
    } else if !pos {
        2
    } else if e.pos_left == 0 && e.neg_right == 0 {
        3
    } else {
        4
    };
    out.push(ty);
    match ty {
        1 => {
            write_vint(e.pos_left as u32 as u64, out);
            write_vint((e.pos_right as i64 - e.pos_left as i64) as u64, out);
        }
        2 => {
            write_vint((-(e.neg_right as i64)) as u32 as u64, out);
            write_vint((e.neg_right as i64 - e.neg_left as i64) as u64, out);
        }
        3 => {
            write_vint(e.pos_right as u32 as u64, out);
            write_vint((-(e.neg_left as i64)) as u32 as u64, out);
        }
        4 => {
            write_vint(e.pos_left as u32 as u64, out);
            write_vint((e.pos_right as i64 - e.pos_left as i64) as u64, out);
            write_vint((-(e.neg_right as i64)) as u32 as u64, out);
            write_vint((e.neg_right as i64 - e.neg_left as i64) as u64, out);
        }
        _ => {}
    }
}

/// ES `Extent.readFromCompressed`.
fn read_extent(b: &[u8], pos: &mut usize) -> Option<ShapeExtent> {
    let top = i32::from_be_bytes(b.get(*pos..*pos + 4)?.try_into().ok()?);
    *pos += 4;
    let bottom = (top as i64 - read_vint(b, pos).ok()? as i64) as i32;
    let ty = *b.get(*pos)?;
    *pos += 1;
    let mut e = ShapeExtent { top, bottom, ..ShapeExtent::default() };
    let mut rd = || read_vint(b, pos).ok();
    match ty {
        0 => {}
        1 => {
            e.pos_left = rd()? as u32 as i32;
            e.pos_right = (rd()? as i64 + e.pos_left as i64) as i32;
        }
        2 => {
            e.neg_right = -(rd()? as u32 as i32);
            e.neg_left = (e.neg_right as i64 - rd()? as i64) as i32;
        }
        3 => {
            e.pos_right = rd()? as u32 as i32;
            e.neg_left = -(rd()? as u32 as i32);
            e.pos_left = 0;
            e.neg_right = 0;
        }
        4 => {
            e.pos_left = rd()? as u32 as i32;
            e.pos_right = (rd()? as i64 + e.pos_left as i64) as i32;
            e.neg_right = -(rd()? as u32 as i32);
            e.neg_left = (e.neg_right as i64 - rd()? as i64) as i32;
        }
        _ => return None,
    }
    Some(e)
}

/// The header's centroid as ES writes it (`encodeX(normalizeX(x))`):
/// `None` when the doc has no finite centroid (ES rejects the doc).
pub fn header_centroid(c: &CentroidCalc) -> Option<(i32, i32)> {
    let s = super::shape::ShapeCentroid::of(c)?;
    Some((s.lon, s.lat))
}

/// Append one doc's value (`GeometryDocValueWriter.write`) of its
/// triangles (all values, in order) and centroid. `Err` without a finite
/// centroid or without triangles.
pub fn write(tris: &[[i32; 7]], centroid: &CentroidCalc, out: &mut Vec<u8>) -> Result<(), String> {
    let (cx, cy) = header_centroid(centroid).ok_or_else(|| "shape has no finite centroid".to_string())?;
    if tris.is_empty() {
        return Err("shape has no triangles".into());
    }
    put_be_i32(cx, out);
    put_be_i32(cy, out);
    out.push(centroid.dim().ordinal());
    write_vint(centroid.sum_weight().to_bits(), out);
    // `TriangleTreeWriter.writeTo`: the extent of the triangles, then the tree.
    let mut nodes: Vec<Node> = tris.iter().map(node_of).collect();
    let mut extent = ShapeExtent::default();
    for n in &nodes {
        extent.add_rect(n.min_x, n.min_y, n.max_x, n.max_y);
    }
    write_extent(&extent, out);
    let n = nodes.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let root = build(&mut nodes, &mut order, true);
    let mut sizes = vec![0u64; n];
    subtree_size(&nodes, root, &mut sizes);
    write_root(&nodes, root, &sizes, out);
    Ok(())
}

/// `createTree`: the median by `(minX, maxX)` / `(minY, maxY)` at each
/// level, children below; parents' max boxes cover their children.
fn build(nodes: &mut [Node], order: &mut [u32], split_x: bool) -> u32 {
    if order.is_empty() {
        return NONE;
    }
    let mid = (order.len() - 1) / 2;
    if order.len() > 1 {
        if split_x {
            order.select_nth_unstable_by_key(mid, |&i| (nodes[i as usize].min_x, nodes[i as usize].max_x));
        } else {
            order.select_nth_unstable_by_key(mid, |&i| (nodes[i as usize].min_y, nodes[i as usize].max_y));
        }
    }
    let me = order[mid];
    let (l, r) = order.split_at_mut(mid);
    let left = build(nodes, l, !split_x);
    let right = build(nodes, &mut r[1..], !split_x);
    let m = me as usize;
    nodes[m].left = left;
    nodes[m].right = right;
    if left != NONE {
        nodes[m].max_x = nodes[m].max_x.max(nodes[left as usize].max_x);
        nodes[m].max_y = nodes[m].max_y.max(nodes[left as usize].max_y);
    }
    if right != NONE {
        nodes[m].max_x = nodes[m].max_x.max(nodes[right as usize].max_x);
        nodes[m].max_y = nodes[m].max_y.max(nodes[right as usize].max_y);
    }
    me
}

fn component_size(n: &Node) -> usize {
    use super::tessellator::TriKind;
    let d = |a: i32, b: i32| vlong_len((a as i64 - b as i64) as u64);
    let t = &n.t;
    match t.kind {
        TriKind::Point => d(n.max_x, t.ax) + d(n.max_y, t.ay),
        TriKind::Line => d(n.max_x, t.ax) + d(n.max_y, t.ay) + d(n.max_x, t.bx) + d(n.max_y, t.by),
        TriKind::Triangle => d(n.max_x, t.ax) + d(n.max_y, t.ay) + d(n.max_x, t.bx) + d(n.max_y, t.by) + d(n.max_x, t.cx) + d(n.max_y, t.cy),
    }
}

/// `nodeSize(false, ..)` of every node: metadata, component and children
/// (each with its box, the right one with its jump size too).
fn subtree_size(nodes: &[Node], i: u32, sizes: &mut [u64]) -> u64 {
    let n = &nodes[i as usize];
    let mut size = 1 + component_size(n) as u64;
    if n.left != NONE {
        size += with_box(nodes, n.left, n, sizes);
    }
    if n.right != NONE {
        let r = with_box(nodes, n.right, n, sizes);
        size += vlong_len(r) as u64 + r;
    }
    sizes[i as usize] = size;
    size
}

/// `nodeSize(true, parentMaxX, parentMaxY, ..)`.
fn with_box(nodes: &[Node], i: u32, parent: &Node, sizes: &mut [u64]) -> u64 {
    let inner = subtree_size(nodes, i, sizes);
    let n = &nodes[i as usize];
    inner + vlong_len((parent.max_x as i64 - n.max_x as i64) as u64) as u64 + vlong_len((parent.max_y as i64 - n.max_y as i64) as u64) as u64 + vlong_len(inner) as u64
}

fn write_meta(n: &Node, out: &mut Vec<u8>) {
    use super::tessellator::TriKind;
    let mut m = 0u8;
    if n.left != NONE {
        m |= LEFT;
    }
    if n.right != NONE {
        m |= RIGHT;
    }
    match n.t.kind {
        TriKind::Point => m |= POINT,
        TriKind::Line => {
            m |= LINE;
            if n.t.ab {
                m |= AB_FROM_TRIANGLE;
            }
        }
        TriKind::Triangle => {
            if n.t.ab {
                m |= AB_FROM_TRIANGLE;
            }
            if n.t.bc {
                m |= BC_FROM_TRIANGLE;
            }
            if n.t.ca {
                m |= CA_FROM_TRIANGLE;
            }
        }
    }
    out.push(m);
}

fn write_component(n: &Node, out: &mut Vec<u8>) {
    use super::tessellator::TriKind;
    let mut w = |a: i32, b: i32| write_vint((a as i64 - b as i64) as u64, out);
    let t = &n.t;
    w(n.max_x, t.ax);
    w(n.max_y, t.ay);
    if t.kind == TriKind::Point {
        return;
    }
    w(n.max_x, t.bx);
    w(n.max_y, t.by);
    if t.kind == TriKind::Line {
        return;
    }
    w(n.max_x, t.cx);
    w(n.max_y, t.cy);
}

fn write_root(nodes: &[Node], root: u32, sizes: &[u64], out: &mut Vec<u8>) {
    let n = &nodes[root as usize];
    write_meta(n, out);
    write_component(n, out);
    if n.left != NONE {
        write_node(nodes, n.left, n, sizes, out);
    }
    if n.right != NONE {
        write_node(nodes, n.right, n, sizes, out);
    }
}

fn write_node(nodes: &[Node], i: u32, parent: &Node, sizes: &[u64], out: &mut Vec<u8>) {
    let n = &nodes[i as usize];
    write_vint((parent.max_x as i64 - n.max_x as i64) as u64, out);
    write_vint((parent.max_y as i64 - n.max_y as i64) as u64, out);
    write_vint(sizes[i as usize], out);
    write_meta(n, out);
    write_component(n, out);
    if n.left != NONE {
        write_node(nodes, n.left, n, sizes, out);
    }
    if n.right != NONE {
        let r = &nodes[n.right as usize];
        let right_size = sizes[n.right as usize] + vlong_len((n.max_x as i64 - r.max_x as i64) as u64) as u64 + vlong_len((n.max_y as i64 - r.max_y as i64) as u64) as u64 + vlong_len(sizes[n.right as usize]) as u64;
        write_vint(right_size, out);
        write_node(nodes, n.right, n, sizes, out);
    }
}

// ---------------------------------------------------------------------------
// Reading

/// A doc value's header: what ES writes before the triangle tree
/// (centroid, dimension, summed weight, extent), decoded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Header {
    pub cx: i32,
    pub cy: i32,
    pub dim: u8,
    pub weight: f64,
    /// `None`: an unreadable extent (no tree is visited).
    pub extent: Option<ShapeExtent>,
}

/// Bytes of a header in a column part ([`DvPart`]): `i32 cx, i32 cy, u8
/// meta` (bits 0-1 the dimension, bits 2-4 ES's extent type), `f64
/// weight, i32 top, i32 bottom, i32 a, i32 b` (little endian): `a..b` the
/// extent's longitudes (east part, west part or both joined at the prime
/// meridian, by type); type 4 (separate parts) keeps the west part as the
/// first 8 bytes of the doc's tree bytes.
pub const HEADER_BYTES: usize = 33;

/// ES `Extent` type of `e` (`writeCompressed`).
fn extent_type(e: &ShapeExtent) -> u8 {
    match (e.has_neg(), e.has_pos()) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) if e.pos_left == 0 && e.neg_right == 0 => 3,
        _ => 4,
    }
}

impl Header {
    /// The fixed slot, and the west part ahead of the tree for type 4.
    fn write_fixed(&self, out: &mut Vec<u8>, tree_prefix: &mut Vec<u8>) {
        let e = self.extent.unwrap_or_default();
        let ty = extent_type(&e);
        out.extend_from_slice(&self.cx.to_le_bytes());
        out.extend_from_slice(&self.cy.to_le_bytes());
        out.push((self.dim & 3) | ty << 2);
        out.extend_from_slice(&self.weight.to_bits().to_le_bytes());
        let (a, b) = match ty {
            1 | 4 => (e.pos_left, e.pos_right),
            2 => (e.neg_left, e.neg_right),
            3 => (e.neg_left, e.pos_right),
            _ => (0, 0),
        };
        for v in [e.top, e.bottom, a, b] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if ty == 4 {
            tree_prefix.extend_from_slice(&e.neg_left.to_le_bytes());
            tree_prefix.extend_from_slice(&e.neg_right.to_le_bytes());
        }
    }

    /// The header of slot `b` and the tree after any extent prefix in
    /// `body`.
    #[inline]
    fn read_fixed<'a>(b: &[u8], body: &'a [u8]) -> (Header, &'a [u8]) {
        let i = |s: &[u8], at: usize| i32::from_le_bytes(s[at..at + 4].try_into().unwrap());
        let meta = b[8];
        let mut e = ShapeExtent { top: i(b, 17), bottom: i(b, 21), ..ShapeExtent::default() };
        let (a, z) = (i(b, 25), i(b, 29));
        let mut tree = body;
        match meta >> 2 & 7 {
            1 => (e.pos_left, e.pos_right) = (a, z),
            2 => (e.neg_left, e.neg_right) = (a, z),
            3 => (e.neg_left, e.neg_right, e.pos_left, e.pos_right) = (a, 0, 0, z),
            4 if body.len() >= 8 => {
                (e.pos_left, e.pos_right) = (a, z);
                (e.neg_left, e.neg_right) = (i(body, 0), i(body, 4));
                tree = &body[8..];
            }
            _ => {}
        }
        let h = Header { cx: i(b, 0), cy: i(b, 4), dim: meta & 3, weight: f64::from_bits(u64::from_le_bytes(b[9..17].try_into().unwrap())), extent: (e.top >= e.bottom).then_some(e) };
        (h, tree)
    }
}

/// One doc's value (`GeometryDocValueReader`): its header and its triangle
/// tree's bytes.
#[derive(Clone, Copy, Debug)]
pub struct ShapeValue<'a> {
    h: Header,
    tree: &'a [u8],
}

/// ES `TriangleTreeVisitor` (encoded ints).
pub trait TreeVisitor {
    fn visit_point(&mut self, x: i32, y: i32);
    fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, metadata: u8);
    #[allow(clippy::too_many_arguments)]
    fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, metadata: u8);
    fn push(&mut self) -> bool;
    fn push_x(&mut self, min_x: i32) -> bool;
    fn push_y(&mut self, min_y: i32) -> bool;
    fn push_max(&mut self, max_x: i32, max_y: i32) -> bool;
    fn push_box(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> bool;
}

#[inline]
pub fn ab_from_triangle(m: u8) -> bool {
    m & AB_FROM_TRIANGLE != 0
}
#[inline]
pub fn bc_from_triangle(m: u8) -> bool {
    m & BC_FROM_TRIANGLE != 0
}
#[inline]
pub fn ca_from_triangle(m: u8) -> bool {
    m & CA_FROM_TRIANGLE != 0
}

impl<'a> ShapeValue<'a> {
    /// A value in ES's byte layout ([`write`]).
    pub fn new(bytes: &'a [u8]) -> ShapeValue<'a> {
        let be = |at: usize| bytes.get(at..at + 4).map_or(0, |b| i32::from_be_bytes(b.try_into().unwrap()));
        let mut pos = 9;
        let weight = read_vint(bytes, &mut pos).map(f64::from_bits).unwrap_or(0.0);
        let extent = if pos <= bytes.len() { read_extent(bytes, &mut pos) } else { None };
        let tree = if extent.is_some() { &bytes[pos.min(bytes.len())..] } else { &[] };
        ShapeValue { h: Header { cx: be(0), cy: be(4), dim: bytes.get(8).copied().unwrap_or(0), weight, extent }, tree }
    }

    /// A value from its header and tree bytes (a column part's).
    pub fn from_parts(h: Header, tree: &'a [u8]) -> ShapeValue<'a> {
        ShapeValue { h, tree }
    }

    pub fn header(&self) -> &Header {
        &self.h
    }

    pub fn tree_bytes(&self) -> &'a [u8] {
        self.tree
    }

    /// The value in ES's byte layout (header, compressed extent, tree).
    pub fn to_es_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.tree.len());
        put_be_i32(self.h.cx, &mut out);
        put_be_i32(self.h.cy, &mut out);
        out.push(self.h.dim);
        write_vint(self.h.weight.to_bits(), &mut out);
        if let Some(e) = &self.h.extent {
            write_extent(e, &mut out);
        }
        out.extend_from_slice(self.tree);
        out
    }

    /// Encoded centroid x (longitude).
    pub fn centroid_x(&self) -> i32 {
        self.h.cx
    }

    /// Encoded centroid y (latitude).
    pub fn centroid_y(&self) -> i32 {
        self.h.cy
    }

    /// Decoded centroid `(x, y)` in degrees.
    pub fn centroid(&self) -> (f64, f64) {
        (decode_lon(self.centroid_x()), decode_lat(self.centroid_y()))
    }

    pub fn dimension(&self) -> ShapeDim {
        ShapeDim::from_ordinal(self.h.dim).unwrap_or_default()
    }

    /// `getSumCentroidWeight`.
    pub fn weight(&self) -> f64 {
        self.h.weight
    }

    /// `getExtent`.
    pub fn extent(&self) -> Option<ShapeExtent> {
        self.h.extent
    }

    /// ES `BoundingBox` of the extent: `(top, bottom, neg_left, neg_right,
    /// pos_left, pos_right)` in degrees, unset sides as infinities.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        self.extent().map(|e| BoundingBox::of(&e))
    }

    /// `GeometryDocValueReader.visit`.
    pub fn visit(&self, v: &mut dyn TreeVisitor) {
        let Some(e) = self.h.extent else { return };
        if self.tree.is_empty() {
            return;
        }
        let mut pos = 0;
        let (max_x, min_x, max_y, min_y) = (e.neg_right.max(e.pos_right), e.neg_left.min(e.pos_left), e.top, e.bottom);
        if v.push_box(min_x, min_y, max_x, max_y) {
            let _ = visit_node(self.tree, &mut pos, v, true, max_x, max_y, true);
        }
    }

    /// Every triangle of the doc, in tree order (no pruning).
    pub fn for_each_triangle(&self, f: &mut dyn FnMut(&super::geo2d::Tri)) {
        self.try_each_triangle(&mut |t| {
            f(t);
            true
        });
    }

    /// The doc's triangles in tree order (no pruning) while `f` returns
    /// true: the walk stops after the first triangle it returns false for.
    pub fn try_each_triangle(&self, f: &mut dyn FnMut(&super::geo2d::Tri) -> bool) {
        struct All<'f> {
            f: &'f mut dyn FnMut(&super::geo2d::Tri) -> bool,
            go: bool,
        }
        impl TreeVisitor for All<'_> {
            fn visit_point(&mut self, x: i32, y: i32) {
                self.go = (self.f)(&super::geo2d::Tri { kind: super::tessellator::TriKind::Point, ax: x, ay: y, bx: x, by: y, cx: x, cy: y, ab: true, bc: true, ca: true });
            }
            fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, m: u8) {
                self.go = (self.f)(&super::geo2d::Tri { kind: super::tessellator::TriKind::Line, ax, ay, bx, by, cx: ax, cy: ay, ab: ab_from_triangle(m), bc: ab_from_triangle(m), ca: ab_from_triangle(m) });
            }
            fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, m: u8) {
                self.go = (self.f)(&super::geo2d::Tri { kind: super::tessellator::TriKind::Triangle, ax, ay, bx, by, cx, cy, ab: ab_from_triangle(m), bc: bc_from_triangle(m), ca: ca_from_triangle(m) });
            }
            fn push(&mut self) -> bool {
                self.go
            }
            fn push_x(&mut self, _: i32) -> bool {
                true
            }
            fn push_y(&mut self, _: i32) -> bool {
                true
            }
            fn push_max(&mut self, _: i32, _: i32) -> bool {
                true
            }
            fn push_box(&mut self, _: i32, _: i32, _: i32, _: i32) -> bool {
                true
            }
        }
        self.visit(&mut All { f, go: true });
    }

    /// The doc's relation to a query (`Component2DVisitor`).
    pub fn relate_query(&self, c: &Component, rel: QueryRelation) -> Result<bool, String> {
        let mut v = Component2DVisitor::new(c, rel);
        self.visit(&mut v);
        if let Some(e) = v.error {
            return Err(e);
        }
        Ok(v.matches())
    }

    /// `ShapeValue.relate(Component2D)` (`Component2DRelationVisitor`).
    pub fn relate(&self, c: &Component) -> GeoRelation {
        let mut v = RelationVisitor { c, relation: None };
        self.visit(&mut v);
        v.relation.unwrap_or(GeoRelation::QueryDisjoint)
    }

    /// `GeoShapeValue.relate(minX, maxX, minY, maxY)` (`Tile2DVisitor`).
    pub fn relate_tile(&self, min_x: i32, max_x: i32, min_y: i32, max_y: i32) -> GeoRelation {
        let mut v = TileVisitor { min_x, max_x, min_y, max_y, relation: GeoRelation::QueryDisjoint };
        self.visit(&mut v);
        v.relation
    }

    /// `ShapeValue.labelPosition()`: the centroid when a polygon contains
    /// it, else the first component of the tree (its centre).
    pub fn label_position(&self) -> Option<(f64, f64)> {
        let (x, y) = self.centroid();
        if self.dimension() == ShapeDim::Polygon {
            let p = super::geo2d::LatLonGeometry::Point { lat: y, lon: x }.to_component();
            if self.relate_query(&p, QueryRelation::Intersects).unwrap_or(false) {
                return Some((x, y));
            }
        }
        let mut v = LabelVisitor { label: None };
        self.visit(&mut v);
        v.label
    }
}

/// `TriangleTreeReader.visit`: `false` when the visitor stopped the walk.
#[allow(clippy::too_many_arguments)]
fn visit_node(b: &[u8], pos: &mut usize, v: &mut dyn TreeVisitor, split_x: bool, this_max_x: i32, this_max_y: i32, is_root: bool) -> Option<bool> {
    let metadata = *b.get(*pos)?;
    *pos += 1;
    let mut rd = |base: i32| -> Option<i32> { read_vint(b, pos).ok().map(|d| (base as i64 - d as i64) as i32) };
    let (this_min_x, this_min_y);
    if metadata & POINT != 0 {
        let x = rd(this_max_x)?;
        let y = rd(this_max_y)?;
        v.visit_point(x, y);
        if !v.push() {
            return Some(false);
        }
        this_min_x = x;
        this_min_y = y;
    } else if metadata & LINE != 0 {
        let ax = rd(this_max_x)?;
        let ay = rd(this_max_y)?;
        let bx = rd(this_max_x)?;
        let by = rd(this_max_y)?;
        v.visit_line(ax, ay, bx, by, metadata);
        if !v.push() {
            return Some(false);
        }
        this_min_x = ax;
        this_min_y = ay.min(by);
    } else {
        let ax = rd(this_max_x)?;
        let ay = rd(this_max_y)?;
        let bx = rd(this_max_x)?;
        let by = rd(this_max_y)?;
        let cx = rd(this_max_x)?;
        let cy = rd(this_max_y)?;
        v.visit_triangle(ax, ay, bx, by, cx, cy, metadata);
        if !v.push() {
            return Some(false);
        }
        this_min_x = ax;
        this_min_y = ay.min(by).min(cy);
    }
    if metadata & LEFT != 0 && !push_left(b, pos, v, this_max_x, this_max_y, split_x)? {
        return Some(false);
    }
    if metadata & RIGHT != 0 {
        let right_size = if is_root { 0 } else { read_vint(b, pos).ok()? as usize };
        if !push_right(b, pos, v, this_max_x, this_max_y, this_min_x, this_min_y, split_x, right_size)? {
            return Some(false);
        }
    }
    Some(v.push())
}

fn push_left(b: &[u8], pos: &mut usize, v: &mut dyn TreeVisitor, this_max_x: i32, this_max_y: i32, split_x: bool) -> Option<bool> {
    let next_max_x = (this_max_x as i64 - read_vint(b, pos).ok()? as i64) as i32;
    let next_max_y = (this_max_y as i64 - read_vint(b, pos).ok()? as i64) as i32;
    let size = read_vint(b, pos).ok()? as usize;
    if v.push_max(next_max_x, next_max_y) {
        visit_node(b, pos, v, !split_x, next_max_x, next_max_y, false)
    } else {
        *pos += size;
        Some(v.push())
    }
}

#[allow(clippy::too_many_arguments)]
fn push_right(b: &[u8], pos: &mut usize, v: &mut dyn TreeVisitor, this_max_x: i32, this_max_y: i32, this_min_x: i32, this_min_y: i32, split_x: bool, right_size: usize) -> Option<bool> {
    if (!split_x && v.push_y(this_min_y)) || (split_x && v.push_x(this_min_x)) {
        let next_max_x = (this_max_x as i64 - read_vint(b, pos).ok()? as i64) as i32;
        let next_max_y = (this_max_y as i64 - read_vint(b, pos).ok()? as i64) as i32;
        let size = read_vint(b, pos).ok()? as usize;
        if v.push_max(next_max_x, next_max_y) {
            return visit_node(b, pos, v, !split_x, next_max_x, next_max_y, false);
        } else {
            *pos += size;
        }
    } else {
        *pos += right_size;
    }
    Some(v.push())
}

/// ES `ShapeValues.BoundingBox` (degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingBox {
    pub top: f64,
    pub bottom: f64,
    pub neg_left: f64,
    pub neg_right: f64,
    pub pos_left: f64,
    pub pos_right: f64,
}

impl BoundingBox {
    pub fn of(e: &ShapeExtent) -> BoundingBox {
        let (neg_left, neg_right) = if !e.has_neg() { (f64::INFINITY, f64::NEG_INFINITY) } else { (decode_lon(e.neg_left), decode_lon(e.neg_right)) };
        let (pos_left, pos_right) = if !e.has_pos() { (f64::INFINITY, f64::NEG_INFINITY) } else { (decode_lon(e.pos_left), decode_lon(e.pos_right)) };
        BoundingBox { top: decode_lat(e.top), bottom: decode_lat(e.bottom), neg_left, neg_right, pos_left, pos_right }
    }
    pub fn min_y(&self) -> f64 {
        self.bottom
    }
    pub fn max_y(&self) -> f64 {
        self.top
    }
    pub fn min_x(&self) -> f64 {
        self.neg_left.min(self.pos_left)
    }
    pub fn max_x(&self) -> f64 {
        self.neg_right.max(self.pos_right)
    }
}

// ---------------------------------------------------------------------------
// Visitors

/// ES `Component2DVisitor` (the doc-value query of each relation).
pub struct Component2DVisitor<'c> {
    c: &'c Component,
    rel: QueryRelation,
    /// INTERSECTS: intersects; DISJOINT: disjoint; WITHIN: within.
    flag: bool,
    /// CONTAINS.
    answer: WithinRelation,
    pub error: Option<String>,
}

impl<'c> Component2DVisitor<'c> {
    pub fn new(c: &'c Component, rel: QueryRelation) -> Self {
        let flag = matches!(rel, QueryRelation::Disjoint | QueryRelation::Within);
        Component2DVisitor { c, rel, flag, answer: WithinRelation::Disjoint, error: None }
    }

    pub fn matches(&self) -> bool {
        match self.rel {
            QueryRelation::Contains => self.answer == WithinRelation::Candidate,
            _ => self.flag,
        }
    }

    fn contains_rel(&mut self, r: Result<WithinRelation, &'static str>) {
        match r {
            Ok(WithinRelation::Disjoint) => {}
            Ok(r) => self.answer = r,
            Err(e) => {
                self.error = Some(e.to_string());
                self.answer = WithinRelation::NotWithin;
            }
        }
    }

    fn push_rel(&mut self, r: Relation) -> bool {
        match self.rel {
            QueryRelation::Intersects => match r {
                Relation::Outside => false,
                Relation::Inside => {
                    self.flag = true;
                    false
                }
                Relation::Crosses => true,
            },
            QueryRelation::Disjoint => match r {
                Relation::Outside => false,
                Relation::Inside => {
                    self.flag = false;
                    false
                }
                Relation::Crosses => true,
            },
            QueryRelation::Within => {
                if r == Relation::Outside {
                    self.flag = false;
                }
                self.flag
            }
            QueryRelation::Contains => r == Relation::Crosses,
        }
    }
}

impl TreeVisitor for Component2DVisitor<'_> {
    fn visit_point(&mut self, x: i32, y: i32) {
        let (x, y) = (decode_lon(x), decode_lat(y));
        match self.rel {
            QueryRelation::Intersects | QueryRelation::Within => self.flag = self.c.contains(x, y),
            QueryRelation::Disjoint => self.flag = !self.c.contains(x, y),
            QueryRelation::Contains => {
                let r = self.c.within_point(x, y);
                self.contains_rel(r)
            }
        }
    }

    fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, m: u8) {
        let (ax, ay, bx, by) = (decode_lon(ax), decode_lat(ay), decode_lon(bx), decode_lat(by));
        match self.rel {
            QueryRelation::Intersects => self.flag = self.c.intersects_line(ax, ay, bx, by),
            QueryRelation::Disjoint => self.flag = !self.c.intersects_line(ax, ay, bx, by),
            QueryRelation::Within => self.flag = self.c.contains_line(ax, ay, bx, by),
            QueryRelation::Contains => {
                let r = self.c.within_line(ax, ay, ab_from_triangle(m), bx, by);
                self.contains_rel(r)
            }
        }
    }

    fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, m: u8) {
        let (ax, ay, bx, by, cx, cy) = (decode_lon(ax), decode_lat(ay), decode_lon(bx), decode_lat(by), decode_lon(cx), decode_lat(cy));
        match self.rel {
            QueryRelation::Intersects => self.flag = self.c.intersects_triangle(ax, ay, bx, by, cx, cy),
            QueryRelation::Disjoint => self.flag = !self.c.intersects_triangle(ax, ay, bx, by, cx, cy),
            QueryRelation::Within => self.flag = self.c.contains_triangle(ax, ay, bx, by, cx, cy),
            QueryRelation::Contains => {
                let r = self.c.within_triangle(ax, ay, ab_from_triangle(m), bx, by, bc_from_triangle(m), cx, cy, ca_from_triangle(m));
                self.contains_rel(r)
            }
        }
    }

    fn push(&mut self) -> bool {
        match self.rel {
            QueryRelation::Intersects => !self.flag,
            QueryRelation::Disjoint | QueryRelation::Within => self.flag,
            QueryRelation::Contains => self.answer != WithinRelation::NotWithin,
        }
    }

    fn push_x(&mut self, min_x: i32) -> bool {
        let r = self.c.max_x() >= decode_lon(min_x);
        if self.rel == QueryRelation::Within {
            self.flag = r;
        }
        r
    }

    fn push_y(&mut self, min_y: i32) -> bool {
        let r = self.c.max_y() >= decode_lat(min_y);
        if self.rel == QueryRelation::Within {
            self.flag = r;
        }
        r
    }

    fn push_max(&mut self, max_x: i32, max_y: i32) -> bool {
        let r = self.c.min_x() <= decode_lon(max_x) && self.c.min_y() <= decode_lat(max_y);
        if self.rel == QueryRelation::Within {
            self.flag = r;
        }
        r
    }

    fn push_box(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> bool {
        let r = self.c.relate(decode_lon(min_x), decode_lon(max_x), decode_lat(min_y), decode_lat(max_y));
        self.push_rel(r)
    }
}

/// ES `Component2DRelationVisitor`.
struct RelationVisitor<'c> {
    c: &'c Component,
    relation: Option<GeoRelation>,
}

impl RelationVisitor<'_> {
    fn not_intersecting(&mut self) {
        match self.relation {
            None => self.relation = Some(GeoRelation::QueryDisjoint),
            Some(GeoRelation::QueryContains) => self.relation = Some(GeoRelation::QueryCrosses),
            _ => {}
        }
    }
    fn can_be_contained(&self) -> bool {
        matches!(self.relation, None | Some(GeoRelation::QueryContains))
    }
    fn can_be_inside(&self) -> bool {
        self.relation != Some(GeoRelation::QueryContains)
    }
}

impl TreeVisitor for RelationVisitor<'_> {
    fn visit_point(&mut self, x: i32, y: i32) {
        let (x, y) = (decode_lon(x), decode_lat(y));
        if self.c.contains(x, y) {
            if self.can_be_inside() && self.c.within_point(x, y) == Ok(WithinRelation::Candidate) {
                self.relation = Some(GeoRelation::QueryInside);
            } else if self.can_be_contained() {
                self.relation = Some(GeoRelation::QueryContains);
            } else {
                self.relation = Some(GeoRelation::QueryCrosses);
            }
        } else {
            self.not_intersecting();
        }
    }

    fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, m: u8) {
        let (ax, ay, bx, by) = (decode_lon(ax), decode_lat(ay), decode_lon(bx), decode_lat(by));
        if self.c.intersects_line(ax, ay, bx, by) {
            if self.can_be_inside() && self.c.within_line(ax, ay, ab_from_triangle(m), bx, by) == Ok(WithinRelation::Candidate) {
                self.relation = Some(GeoRelation::QueryInside);
            } else if self.can_be_contained() && self.c.contains_line(ax, ay, bx, by) {
                self.relation = Some(GeoRelation::QueryContains);
            } else {
                self.relation = Some(GeoRelation::QueryCrosses);
            }
        } else {
            self.not_intersecting();
        }
    }

    fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, m: u8) {
        let (ax, ay, bx, by, cx, cy) = (decode_lon(ax), decode_lat(ay), decode_lon(bx), decode_lat(by), decode_lon(cx), decode_lat(cy));
        if self.c.intersects_triangle(ax, ay, bx, by, cx, cy) {
            if self.can_be_inside() && self.c.within_triangle(ax, ay, ab_from_triangle(m), bx, by, bc_from_triangle(m), cx, cy, ca_from_triangle(m)) == Ok(WithinRelation::Candidate) {
                self.relation = Some(GeoRelation::QueryInside);
            } else if self.can_be_contained() && self.c.contains_triangle(ax, ay, bx, by, cx, cy) {
                self.relation = Some(GeoRelation::QueryContains);
            } else {
                self.relation = Some(GeoRelation::QueryCrosses);
            }
        } else {
            self.not_intersecting();
        }
    }

    fn push(&mut self) -> bool {
        self.relation != Some(GeoRelation::QueryCrosses)
    }

    fn push_x(&mut self, min_x: i32) -> bool {
        if self.c.max_x() >= decode_lon(min_x) {
            return true;
        }
        self.not_intersecting();
        false
    }

    fn push_y(&mut self, min_y: i32) -> bool {
        if self.c.max_y() >= decode_lat(min_y) {
            return true;
        }
        self.not_intersecting();
        false
    }

    fn push_max(&mut self, max_x: i32, max_y: i32) -> bool {
        if self.c.min_y() <= decode_lat(max_y) && self.c.min_x() <= decode_lon(max_x) {
            return true;
        }
        self.not_intersecting();
        false
    }

    fn push_box(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> bool {
        if self.c.relate(decode_lon(min_x), decode_lon(max_x), decode_lat(min_y), decode_lat(max_y)) == Relation::Outside {
            self.relation = Some(GeoRelation::QueryDisjoint);
            return false;
        }
        true
    }
}

/// ES `Tile2DVisitor` (encoded ints throughout).
pub struct TileVisitor {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    relation: GeoRelation,
}

impl TileVisitor {
    fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    fn intersects_line(&self, ax: i32, ay: i32, bx: i32, by: i32) -> bool {
        if self.contains(ax, ay) || self.contains(bx, by) {
            return true;
        }
        let (t_min_x, t_max_x, t_min_y, t_max_y) = (ax.min(bx), ax.max(bx), ay.min(by), ay.max(by));
        if t_max_x < self.min_x || t_min_x > self.max_x || t_min_y > self.max_y || t_max_y < self.min_y {
            return false;
        }
        self.edge_intersects_query(ax, ay, bx, by)
    }

    #[allow(clippy::too_many_arguments)]
    fn relate_triangle(&self, ax: i32, ay: i32, ab: bool, bx: i32, by: i32, bc: bool, cx: i32, cy: i32, ca: bool) -> GeoRelation {
        let t_min_x = ax.min(bx).min(cx);
        let t_max_x = ax.max(bx).max(cx);
        let t_min_y = ay.min(by).min(cy);
        let t_max_y = ay.max(by).max(cy);
        if t_max_x < self.min_x || t_min_x > self.max_x || t_min_y > self.max_y || t_max_y < self.min_y {
            return GeoRelation::QueryDisjoint;
        }
        if self.contains(ax, ay) || self.contains(bx, by) || self.contains(cx, cy) {
            return GeoRelation::QueryCrosses;
        }
        let mut within = false;
        for (x1, y1, x2, y2, from_poly) in [(ax, ay, bx, by, ab), (bx, by, cx, cy, bc), (cx, cy, ax, ay, ca)] {
            if self.edge_intersects_query(x1, y1, x2, y2) {
                if from_poly {
                    return GeoRelation::QueryCrosses;
                }
                within = true;
            }
        }
        let f = |v: i32| v as f64;
        if within || super::geo2d::point_in_triangle(f(t_min_x), f(t_max_x), f(t_min_y), f(t_max_y), f(self.min_x), f(self.min_y), f(ax), f(ay), f(bx), f(by), f(cx), f(cy)) {
            return GeoRelation::QueryInside;
        }
        GeoRelation::QueryDisjoint
    }

    fn edge_intersects_query(&self, ax: i32, ay: i32, bx: i32, by: i32) -> bool {
        let (min_x, max_x, min_y, max_y) = (self.min_x, self.max_x, self.min_y, self.max_y);
        if ax.max(bx) < min_x || ax.min(bx) > max_x || ay.max(by) < min_y || ay.min(by) > max_y {
            return false;
        }
        let o = |ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32| super::geo2d::orient(ax as f64, ay as f64, bx as f64, by as f64, cx as f64, cy as f64);
        (o(ax, ay, bx, by, min_x, max_y) * o(ax, ay, bx, by, max_x, max_y) <= 0 && o(min_x, max_y, max_x, max_y, ax, ay) * o(min_x, max_y, max_x, max_y, bx, by) <= 0)
            || (o(ax, ay, bx, by, max_x, max_y) * o(ax, ay, bx, by, max_x, min_y) <= 0 && o(max_x, max_y, max_x, min_y, ax, ay) * o(max_x, max_y, max_x, min_y, bx, by) <= 0)
            || (o(ax, ay, bx, by, max_x, min_y) * o(ax, ay, bx, by, min_x, min_y) <= 0 && o(max_x, min_y, min_x, min_y, ax, ay) * o(max_x, min_y, min_x, min_y, bx, by) <= 0)
            || (o(ax, ay, bx, by, min_x, min_y) * o(ax, ay, bx, by, min_x, max_y) <= 0 && o(min_x, min_y, min_x, max_y, ax, ay) * o(min_x, min_y, min_x, max_y, bx, by) <= 0)
    }
}

impl TreeVisitor for TileVisitor {
    fn visit_point(&mut self, x: i32, y: i32) {
        if self.contains(x, y) {
            self.relation = GeoRelation::QueryCrosses;
        }
    }
    fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, _m: u8) {
        if self.intersects_line(ax, ay, bx, by) {
            self.relation = GeoRelation::QueryCrosses;
        }
    }
    fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, m: u8) {
        let r = self.relate_triangle(ax, ay, ab_from_triangle(m), bx, by, bc_from_triangle(m), cx, cy, ca_from_triangle(m));
        if r != GeoRelation::QueryDisjoint {
            self.relation = r;
        }
    }
    fn push(&mut self) -> bool {
        self.relation != GeoRelation::QueryCrosses
    }
    fn push_x(&mut self, min_x: i32) -> bool {
        self.max_x >= min_x
    }
    fn push_y(&mut self, min_y: i32) -> bool {
        self.max_y >= min_y
    }
    fn push_max(&mut self, max_x: i32, max_y: i32) -> bool {
        self.min_y <= max_y && self.min_x <= max_x
    }
    fn push_box(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> bool {
        if min_x > self.max_x || max_x < self.min_x || min_y > self.max_y || max_y < self.min_y {
            self.relation = GeoRelation::QueryDisjoint;
            return false;
        }
        if self.min_x <= min_x && self.max_x >= max_x && self.min_y <= min_y && self.max_y >= max_y {
            self.relation = GeoRelation::QueryContains;
            return false;
        }
        true
    }
}

/// ES `LabelPositionVisitor`: the first component's centre `(x, y)`.
struct LabelVisitor {
    label: Option<(f64, f64)>,
}

impl TreeVisitor for LabelVisitor {
    fn visit_point(&mut self, x: i32, y: i32) {
        self.label = Some((decode_lon(x), decode_lat(y)));
    }
    fn visit_line(&mut self, ax: i32, ay: i32, bx: i32, by: i32, _m: u8) {
        self.label = Some(((decode_lon(ax) + decode_lon(bx)) / 2.0, (decode_lat(ay) + decode_lat(by)) / 2.0));
    }
    fn visit_triangle(&mut self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32, _m: u8) {
        self.label = Some(((decode_lon(ax) + decode_lon(bx) + decode_lon(cx)) / 3.0, (decode_lat(ay) + decode_lat(by) + decode_lat(cy)) / 3.0));
    }
    fn push(&mut self) -> bool {
        self.label.is_none()
    }
    fn push_x(&mut self, _: i32) -> bool {
        self.label.is_none()
    }
    fn push_y(&mut self, _: i32) -> bool {
        self.label.is_none()
    }
    fn push_max(&mut self, _: i32, _: i32) -> bool {
        self.label.is_none()
    }
    fn push_box(&mut self, _: i32, _: i32, _: i32, _: i32) -> bool {
        self.label.is_none()
    }
}

// ---------------------------------------------------------------------------
// The segment column

use crate::codec::directory::FileSlice;
use crate::error::RusticError;
use crate::types::DocId;

/// Most value bytes per part (an `[extra]` entry holds at most `u32::MAX`).
const PART_BYTES: usize = 1 << 31;

/// One part of a field's doc-value column: `u32 first_doc, u32 docs`,
/// each doc's fixed-width [`Header`] (`HEADER_BYTES`), `(docs + 1) x u32`
/// offsets of the trees, then the trees (a doc without a value has an
/// empty range; a type-4 extent's west part leads its tree bytes).
#[derive(Clone)]
pub struct DvPart {
    first: DocId,
    n: u32,
    data: FileSlice,
    /// Offsets of the headers, the tree offsets and the trees in `data`.
    headers: usize,
    offsets: usize,
    base: usize,
}

impl DvPart {
    pub fn open(data: FileSlice) -> crate::error::Result<DvPart> {
        let b = data.as_slice();
        let bad = |m: &str| RusticError::Corrupted(format!("shape doc values {m}"));
        if b.len() < 8 {
            return Err(bad("truncated"));
        }
        let first = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let n = u32::from_le_bytes(b[4..8].try_into().unwrap());
        let headers = 8;
        let offsets = headers + n as usize * HEADER_BYTES;
        let base = offsets + (n as usize + 1) * 4;
        if b.len() < base {
            return Err(bad("truncated offsets"));
        }
        let last = u32::from_le_bytes(b[base - 4..base].try_into().unwrap()) as usize;
        if b.len() < base + last {
            return Err(bad("truncated values"));
        }
        Ok(DvPart { first, n, data, headers, offsets, base })
    }

    #[inline]
    fn get(&self, doc: DocId) -> Option<ShapeValue<'_>> {
        let i = doc.checked_sub(self.first)?;
        if i >= self.n {
            return None;
        }
        let b = self.data.as_slice();
        let at = self.offsets + i as usize * 4;
        let s = u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
        let e = u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap()) as usize;
        if e <= s {
            return None;
        }
        let h = self.headers + i as usize * HEADER_BYTES;
        let (header, tree) = Header::read_fixed(&b[h..h + HEADER_BYTES], &b[self.base + s..self.base + e]);
        Some(ShapeValue::from_parts(header, tree))
    }

    pub fn bytes_len(&self) -> usize {
        self.data.as_slice().len()
    }

    /// The doc's centroid, dimension and weight (header only).
    #[inline]
    fn centroid(&self, doc: DocId) -> Option<(i32, i32, u8, f64)> {
        let i = doc.checked_sub(self.first)?;
        if i >= self.n {
            return None;
        }
        let b = self.data.as_slice();
        let at = self.offsets + i as usize * 4;
        if u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap()) <= u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) {
            return None;
        }
        let h = &b[self.headers + i as usize * HEADER_BYTES..];
        Some((i32::from_le_bytes(h[0..4].try_into().unwrap()), i32::from_le_bytes(h[4..8].try_into().unwrap()), h[8] & 3, f64::from_bits(u64::from_le_bytes(h[9..17].try_into().unwrap()))))
    }
}

/// A field's doc values in a segment (its parts, ascending by doc).
#[derive(Clone, Default)]
pub struct ShapeDvColumn {
    parts: Vec<DvPart>,
    /// The union of every value's extent, computed on first use.
    union: std::sync::Arc<std::sync::OnceLock<Option<ShapeExtent>>>,
}

impl ShapeDvColumn {
    pub fn push_part(&mut self, p: DvPart) {
        self.union = Default::default();
        self.parts.push(p);
        self.parts.sort_by_key(|p| p.first);
    }

    /// The doc's value, if any.
    pub fn get(&self, doc: DocId) -> Option<ShapeValue<'_>> {
        let i = self.parts.partition_point(|p| p.first <= doc);
        let p = self.parts.get(i.checked_sub(1)?)?;
        p.get(doc)
    }

    pub fn has_value(&self, doc: DocId) -> bool {
        self.get(doc).is_some()
    }

    /// The doc's encoded centroid `(x, y)`, dimension and summed weight,
    /// read from its fixed-width header alone.
    #[inline]
    pub fn centroid(&self, doc: DocId) -> Option<(i32, i32, ShapeDim, f64)> {
        let p = match self.parts.len() {
            1 => &self.parts[0],
            _ => {
                let i = self.parts.partition_point(|p| p.first <= doc);
                self.parts.get(i.checked_sub(1)?)?
            }
        };
        p.centroid(doc).map(|(x, y, d, w)| (x, y, ShapeDim::from_ordinal(d).unwrap_or_default(), w))
    }

    /// The union of the extents of every doc's value (`None`: no value).
    /// Equal to adding each doc's extent in turn (unions by min / max).
    pub fn union_extent(&self) -> Option<ShapeExtent> {
        *self.union.get_or_init(|| {
            let mut e = ShapeExtent::default();
            let mut any = false;
            for p in &self.parts {
                for i in 0..p.n {
                    if let Some(x) = p.get(p.first + i).and_then(|v| v.extent()) {
                        e.merge(&x);
                        any = true;
                    }
                }
            }
            any.then_some(e)
        })
    }

    pub fn bytes_len(&self) -> usize {
        self.parts.iter().map(DvPart::bytes_len).sum()
    }
}

/// Encode a field's values (`values(doc)` for docs `0..num_docs`) as
/// column parts.
pub fn encode_column<'v>(num_docs: u32, mut values: impl FnMut(DocId) -> Option<ShapeValue<'v>>) -> Vec<Vec<u8>> {
    let mut parts = Vec::new();
    let mut doc = 0u32;
    while doc < num_docs {
        let first = doc;
        let mut offsets: Vec<u32> = vec![0];
        let mut headers: Vec<u8> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        while doc < num_docs && (bytes.len() + headers.len() < PART_BYTES || doc == first) {
            match values(doc) {
                Some(v) => {
                    v.h.write_fixed(&mut headers, &mut bytes);
                    bytes.extend_from_slice(v.tree);
                }
                None => headers.resize(headers.len() + HEADER_BYTES, 0),
            }
            offsets.push(bytes.len() as u32);
            doc += 1;
        }
        let n = offsets.len() as u32 - 1;
        let mut out = Vec::with_capacity(8 + headers.len() + offsets.len() * 4 + bytes.len());
        out.extend_from_slice(&first.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&headers);
        for o in &offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out.extend_from_slice(&bytes);
        if bytes.is_empty() && !parts.is_empty() {
            continue;
        }
        parts.push(out);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fieldtypes::geo2d::LatLonGeometry;
    use crate::fieldtypes::shape::{parse_wkt, Geometry, ParseOpts};
    use crate::fieldtypes::tessellator::geometry_triangles;

    fn value(wkt: &str) -> Vec<u8> {
        let g = parse_wkt(wkt, &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
        let mut tris = Vec::new();
        geometry_triangles(&g, &mut tris).unwrap();
        let mut c = CentroidCalc::default();
        c.add(&g);
        let mut out = Vec::new();
        write(&tris, &c, &mut out).unwrap();
        out
    }

    fn tris_of(v: &ShapeValue<'_>) -> Vec<[i32; 6]> {
        let mut out = Vec::new();
        v.for_each_triangle(&mut |t| out.push([t.ax, t.ay, t.bx, t.by, t.cx, t.cy]));
        out.sort_unstable();
        out
    }

    #[test]
    fn round_trips_header_and_triangles() {
        let wkt = "POLYGON ((0 0, 10 0, 10 10, 5 15, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))";
        let b = value(wkt);
        let v = ShapeValue::new(&b);
        let g = parse_wkt(wkt, &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
        let mut c = CentroidCalc::default();
        c.add(&g);
        assert_eq!((v.centroid_x(), v.centroid_y()), header_centroid(&c).unwrap());
        assert_eq!(v.dimension(), ShapeDim::Polygon);
        assert_eq!(v.weight(), c.sum_weight());
        let mut tris = Vec::new();
        geometry_triangles(&g, &mut tris).unwrap();
        let mut want: Vec<[i32; 6]> = tris
            .iter()
            .map(|t| {
                let d = crate::fieldtypes::tessellator::decode_triangle(t);
                [d.ax, d.ay, d.bx, d.by, d.cx, d.cy]
            })
            .collect();
        want.sort_unstable();
        assert_eq!(tris_of(&v), want);
        let e = v.extent().unwrap();
        assert_eq!((e.bottom, e.top), (crate::geo::encode_lat(0.0), crate::geo::encode_lat(15.0)));
        // Points and lines too, and a value crossing the prime meridian.
        for w in ["POINT (1 2)", "LINESTRING (-1 0, 1 1, 2 3)", "MULTIPOINT ((-5 -5), (6 7))", "POLYGON ((-3 -3, 3 -3, 3 3, -3 3, -3 -3))"] {
            let b = value(w);
            let v = ShapeValue::new(&b);
            let g = parse_wkt(w, &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
            let mut tris = Vec::new();
            geometry_triangles(&g, &mut tris).unwrap();
            assert_eq!(tris_of(&v).len(), tris.len(), "{w}");
            assert!(v.extent().is_some());
        }
    }

    #[test]
    fn relations_and_tiles() {
        let b = value("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))");
        let v = ShapeValue::new(&b);
        let inner = LatLonGeometry::Polygon(vec![vec![(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0), (2.0, 2.0)]]).to_component();
        let outer = LatLonGeometry::Polygon(vec![vec![(-1.0, -1.0), (11.0, -1.0), (11.0, 11.0), (-1.0, 11.0), (-1.0, -1.0)]]).to_component();
        let far = LatLonGeometry::Point { lat: 50.0, lon: 50.0 }.to_component();
        assert!(v.relate_query(&inner, QueryRelation::Intersects).unwrap());
        assert!(v.relate_query(&inner, QueryRelation::Contains).unwrap());
        assert!(!v.relate_query(&inner, QueryRelation::Within).unwrap());
        assert!(v.relate_query(&outer, QueryRelation::Within).unwrap());
        assert!(v.relate_query(&far, QueryRelation::Disjoint).unwrap());
        assert_eq!(v.relate(&inner), GeoRelation::QueryInside);
        assert_eq!(v.relate(&outer), GeoRelation::QueryContains);
        assert_eq!(v.relate(&far), GeoRelation::QueryDisjoint);
        let e = |x: f64, y: f64| (crate::geo::encode_lon(x), crate::geo::encode_lat(y));
        let (a, b2) = (e(2.0, 2.0), e(3.0, 3.0));
        assert_eq!(v.relate_tile(a.0, b2.0, a.1, b2.1), GeoRelation::QueryInside);
        let (a, b2) = (e(-20.0, -20.0), e(20.0, 20.0));
        assert_eq!(v.relate_tile(a.0, b2.0, a.1, b2.1), GeoRelation::QueryContains);
        let (a, b2) = (e(30.0, 30.0), e(40.0, 40.0));
        assert_eq!(v.relate_tile(a.0, b2.0, a.1, b2.1), GeoRelation::QueryDisjoint);
        assert_eq!(v.label_position(), Some(v.centroid()));
    }

    #[test]
    fn column_parts() {
        // Every extent type: east only, west only, joined at the prime
        // meridian, and two separate parts (its west part ahead of the tree).
        let wkts = ["POINT (1 2)", "LINESTRING (0 0, 1 1, 2 0)", "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))", "POINT (-3 4)", "LINESTRING (-2 0, 3 1)", "MULTIPOLYGON (((170 0, 175 0, 175 5, 170 0)), ((-175 0, -170 0, -170 5, -175 0)))"];
        let vals: Vec<Option<Vec<u8>>> = (0..12u32).map(|d| (d % 7 != 1).then(|| value(wkts[d as usize % wkts.len()]))).collect();
        let parts = encode_column(12, |d| vals[d as usize].as_deref().map(ShapeValue::new));
        let mut col = ShapeDvColumn::default();
        for p in parts {
            col.push_part(DvPart::open(FileSlice::from_vec(p)).unwrap());
        }
        for d in 0..12u32 {
            assert_eq!(col.get(d).map(|v| v.to_es_bytes()), vals[d as usize].clone(), "{d}");
            assert_eq!(col.get(d).map(|v| v.extent()), vals[d as usize].as_deref().map(|b| ShapeValue::new(b).extent()), "{d}");
        }
        assert!(col.get(12).is_none());
        let _ = Geometry::Point((0.0, 0.0));
    }
}
