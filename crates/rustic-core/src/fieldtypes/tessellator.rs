//! Lucene's `Tessellator` (10.5.1, the version ES 9.5.5 ships), ported
//! function for function: an ear-clipping triangulation of a polygon with
//! holes, with z-order (Morton) acceleration above 80 vertices, local
//! intersection curing and polygon splitting, and the same failures
//! (self-intersections when asked, illegal holes, shapes it cannot
//! tessellate). Plus `ShapeField`'s triangle encoding (seven sortable ints:
//! the box, the third vertex and a code for the layout and edge flags) and
//! `LatLonShape`'s decomposition of a geometry into triangles (lines into
//! degenerate `(a, b, a)` triangles, points into `(a, a, a)`).
//!
//! Lucene works on linked nodes; here they live in an arena and link by
//! index. Removed nodes keep their own links, as Lucene's do. The ear test,
//! curing and splitting compare the raw `f64` coordinates (Lucene's
//! `getX()` / `getY()`), triangles carry the encoded ints.

use crate::geo::{encode_lat, encode_lon};

/// Lucene `Tessellator.VERTEX_THRESHOLD`.
const VERTEX_THRESHOLD: i64 = 80;

const UNABLE: &str = "Unable to Tessellate shape. Possible malformed shape detected.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Winding {
    Cw,
    Ccw,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Init,
    Cure,
    Split,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    /// Index of the vertex over the polygon and its holes (Lucene `idx`).
    idx: i32,
    x: f64,
    y: f64,
    /// Encoded longitude / latitude.
    xe: i32,
    ye: i32,
    morton: u64,
    prev: u32,
    next: u32,
    /// Z-order links ([`NIL`]: none).
    prev_z: u32,
    next_z: u32,
    /// Whether the edge to `next` is an edge of the input polygon.
    efp: bool,
}

/// A triangle of the tessellation (encoded vertices `a`, `b`, `c` and
/// whether each of its edges `ab`, `bc`, `ca` is an edge of the polygon).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Triangle {
    pub ax: i32,
    pub ay: i32,
    pub bx: i32,
    pub by: i32,
    pub cx: i32,
    pub cy: i32,
    pub ab: bool,
    pub bc: bool,
    pub ca: bool,
}

/// A ring of `(lon, lat)` vertices, closed (first == last), as Lucene's
/// `Polygon` holds it, with its box and winding order.
struct Ring<'a> {
    pts: &'a [(f64, f64)],
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    winding: Winding,
}

/// Lucene `Polygon`'s checks and winding order.
fn ring(pts: &[(f64, f64)]) -> Result<Ring<'_>, String> {
    if pts.len() < 4 {
        return Err("at least 4 polygon points required".into());
    }
    let (first, last) = (pts[0], pts[pts.len() - 1]);
    if first.1 != last.1 || first.0 != last.0 {
        return Err("first and last points of the polygon must be the same (it must close itself)".into());
    }
    for p in pts {
        if !(-90.0..=90.0).contains(&p.1) || p.1.is_nan() {
            return Err(format!("invalid latitude {}; must be between -90.0 and 90.0", p.1));
        }
        if !(-180.0..=180.0).contains(&p.0) || p.0.is_nan() {
            return Err(format!("invalid longitude {}; must be between -180.0 and 180.0", p.0));
        }
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (pts[0].0, pts[0].0, pts[0].1, pts[0].1);
    let mut winding_sum = 0.0;
    let n = pts.len() - 1;
    let mut j = 0;
    for i in 1..n {
        min_y = min_y.min(pts[i].1);
        max_y = max_y.max(pts[i].1);
        min_x = min_x.min(pts[i].0);
        max_x = max_x.max(pts[i].0);
        winding_sum += (pts[j].0 - pts[n].0) * (pts[i].1 - pts[n].1) - (pts[j].1 - pts[n].1) * (pts[i].0 - pts[n].0);
        j = i;
    }
    Ok(Ring { pts, min_x, max_x, min_y, max_y, winding: if winding_sum < 0.0 { Winding::Ccw } else { Winding::Cw } })
}

/// Lucene `BitUtil.interleave`: `even`'s bits at even positions, `odd`'s at
/// odd positions.
pub fn interleave(even: i32, odd: i32) -> u64 {
    fn spread(v: u64) -> u64 {
        let mut v = v & 0x0000_0000_FFFF_FFFF;
        v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
        v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        v = (v | (v << 1)) & 0x5555_5555_5555_5555;
        v
    }
    (spread(odd as u32 as u64) << 1) | spread(even as u32 as u64)
}

/// Tessellate a polygon: `outer` and `holes` are closed `(lon, lat)` rings.
/// `check_self_intersections`: fail on crossing or overlapping edges
/// (ES's `GeoShapeIndexer` asks for it).
pub fn tessellate(outer: &[(f64, f64)], holes: &[Vec<(f64, f64)>], check_self_intersections: bool) -> Result<Vec<Triangle>, String> {
    let poly = ring(outer)?;
    let hole_rings: Vec<Ring<'_>> = holes.iter().map(|h| ring(h)).collect::<Result<_, _>>()?;
    // The node arena, edge boxes and raw triangles are reused per thread
    // (ingest tessellates every polygon; growing fresh vectors each time
    // showed as allocator time in the c28w ingest profile).
    with_scratch(outer.len() + holes.iter().map(Vec::len).sum::<usize>(), |t, out| -> Result<Vec<Triangle>, String> {
        tessellate_in(t, out, &poly, &hole_rings, check_self_intersections)?;
        Ok(out.iter().map(|&(a, ab, b, bc, c, ca)| Triangle { ax: t.n[a].xe, ay: t.n[a].ye, bx: t.n[b].xe, by: t.n[b].ye, cx: t.n[c].xe, cy: t.n[c].ye, ab, bc, ca }).collect())
    })
}

/// [`tessellate`] appending the encoded triangles to `out` (reserved for
/// them), with no intermediate list.
pub fn tessellate_encoded(outer: &[(f64, f64)], holes: &[Vec<(f64, f64)>], check_self_intersections: bool, out: &mut Vec<[i32; 7]>) -> Result<(), String> {
    let poly = ring(outer)?;
    let hole_rings: Vec<Ring<'_>> = holes.iter().map(|h| ring(h)).collect::<Result<_, _>>()?;
    with_scratch(outer.len() + holes.iter().map(Vec::len).sum::<usize>(), |t, raw| -> Result<(), String> {
        tessellate_in(t, raw, &poly, &hole_rings, check_self_intersections)?;
        out.reserve(raw.len());
        for &(a, ab, b, bc, c, ca) in raw.iter() {
            out.push(encode_triangle(t.n[a].ye, t.n[a].xe, ab, t.n[b].ye, t.n[b].xe, bc, t.n[c].ye, t.n[c].xe, ca)?);
        }
        Ok(())
    })
}

/// Run `f` on this thread's scratch arena and raw triangle list (cleared,
/// the arena reserved for `vertices`).
fn with_scratch<R>(vertices: usize, f: impl FnOnce(&mut Tess, &mut Vec<RawTri>) -> R) -> R {
    let (mut t, mut out) = SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    t.n.clear();
    t.boxes.clear();
    out.clear();
    t.n.reserve(vertices + 16);
    let r = f(&mut t, &mut out);
    if t.n.capacity() <= SCRATCH_KEEP_NODES {
        SCRATCH.with(|c| *c.borrow_mut() = (t, out));
    }
    r
}

/// Most nodes a thread keeps its scratch arena for.
const SCRATCH_KEEP_NODES: usize = 1 << 16;

thread_local! {
    static SCRATCH: std::cell::RefCell<(Tess, Vec<RawTri>)> = std::cell::RefCell::new((Tess::default(), Vec::new()));
}

fn tessellate_in(t: &mut Tess, out: &mut Vec<RawTri>, poly: &Ring<'_>, hole_rings: &[Ring<'_>], check_self_intersections: bool) -> Result<(), String> {
    let outer_node = t.create_list(poly, 0, Winding::Cw).ok_or_else(|| "Malformed shape detected in Tessellator!".to_string())?;
    let mut outer_node = outer_node;
    if outer_node == t.next(outer_node) || outer_node == t.next(t.next(outer_node)) {
        return Err("at least three non-collinear points required".into());
    }
    if !hole_rings.is_empty() {
        outer_node = t.eliminate_holes(poly, hole_rings, outer_node)?;
    }
    let mut threshold = VERTEX_THRESHOLD - poly.pts.len() as i64;
    let mut i = 0;
    while threshold >= 0 && i < hole_rings.len() {
        threshold -= hole_rings[i].pts.len() as i64;
        i += 1;
    }
    let morton = threshold < 0;
    if morton {
        t.sort_by_morton(outer_node);
    }
    if check_self_intersections {
        t.fill_edge_boxes();
        t.check_intersection(outer_node, morton)?;
    }
    t.earcut(Some(outer_node), out, State::Init, morton, 0)?;
    if out.is_empty() {
        return Err(UNABLE.into());
    }
    Ok(())
}

/// `(a, ab, b, bc, c, ca)` node indices and edge flags.
type RawTri = (usize, bool, usize, bool, usize, bool);

/// No z-order link.
const NIL: u32 = u32::MAX;

/// A z-order link of a node index.
#[inline]
fn z(o: Option<usize>) -> u32 {
    o.map_or(NIL, |i| i as u32)
}

/// The node of a z-order link.
#[inline]
fn unz(l: u32) -> Option<usize> {
    (l != NIL).then_some(l as usize)
}

#[derive(Default)]
struct Tess {
    n: Vec<Node>,
    /// Per node, the box `[min_y, max_y, min_x, max_x]` of its edge to
    /// `next` (filled for the self-intersection check, when no node moves).
    boxes: Vec<[f64; 4]>,
    /// The self-intersection check's visiting order as arrays: node ids
    /// (ring order, or z order) with their edge boxes and Morton codes.
    order: Vec<u32>,
    order_boxes: Vec<[f64; 4]>,
    order_morton: Vec<u64>,
    /// Per node, its index in `order` (`NIL`: not in it).
    order_pos: Vec<u32>,
}

/// Lucene `Tessellator.area` (twice the signed area, negative for CCW in
/// its convention).
#[inline]
fn area(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
    (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn point_in_ear(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
    (cx - x) * (ay - y) - (ax - x) * (cy - y) >= 0.0 && (ax - x) * (by - y) - (bx - x) * (ay - y) >= 0.0 && (bx - x) * (cy - y) - (cx - x) * (by - y) >= 0.0
}

/// Lucene `GeoUtils.orient`.
#[inline]
fn orient(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> i32 {
    let v1 = (bx - ax) * (cy - ay);
    let v2 = (cx - ax) * (by - ay);
    if v1 > v2 {
        1
    } else if v1 < v2 {
        -1
    } else {
        0
    }
}

#[allow(clippy::too_many_arguments)]
fn line_crosses_line(a1x: f64, a1y: f64, b1x: f64, b1y: f64, a2x: f64, a2y: f64, b2x: f64, b2y: f64) -> bool {
    orient(a2x, a2y, b2x, b2y, a1x, a1y) * orient(a2x, a2y, b2x, b2y, b1x, b1y) < 0 && orient(a1x, a1y, b1x, b1y, a2x, a2y) * orient(a1x, a1y, b1x, b1y, b2x, b2y) < 0
}

#[allow(clippy::too_many_arguments)]
fn line_overlap_line(a1x: f64, a1y: f64, b1x: f64, b1y: f64, a2x: f64, a2y: f64, b2x: f64, b2y: f64) -> bool {
    orient(a2x, a2y, b2x, b2y, a1x, a1y) == 0 && orient(a2x, a2y, b2x, b2y, b1x, b1y) == 0 && orient(a1x, a1y, b1x, b1y, a2x, a2y) == 0 && orient(a1x, a1y, b1x, b1y, b2x, b2y) == 0
}

/// Lucene `Tessellator.linesIntersect`.
#[allow(clippy::too_many_arguments)]
fn lines_intersect(a_x0: f64, a_y0: f64, a_x1: f64, a_y1: f64, b_x0: f64, b_y0: f64, b_x1: f64, b_y1: f64) -> bool {
    ((area(a_x0, a_y0, a_x1, a_y1, b_x0, b_y0) > 0.0) != (area(a_x0, a_y0, a_x1, a_y1, b_x1, b_y1) > 0.0))
        && ((area(b_x0, b_y0, b_x1, b_y1, a_x0, a_y0) > 0.0) != (area(b_x0, b_y0, b_x1, b_y1, a_x1, a_y1) > 0.0))
}

impl Tess {
    #[inline]
    fn x(&self, i: usize) -> f64 {
        self.n[i].x
    }
    #[inline]
    fn y(&self, i: usize) -> f64 {
        self.n[i].y
    }
    #[inline]
    fn next(&self, i: usize) -> usize {
        self.n[i].next as usize
    }
    #[inline]
    fn prev(&self, i: usize) -> usize {
        self.n[i].prev as usize
    }

    fn area3(&self, a: usize, b: usize, c: usize) -> f64 {
        area(self.x(a), self.y(a), self.x(b), self.y(b), self.x(c), self.y(c))
    }

    fn create_list(&mut self, r: &Ring<'_>, mut start_index: i32, winding: Winding) -> Option<usize> {
        let mut last: Option<usize> = None;
        if winding == r.winding {
            for i in 0..r.pts.len() {
                last = Some(self.insert_node(r.pts[i].0, r.pts[i].1, start_index, last));
                start_index += 1;
            }
        } else {
            for i in (0..r.pts.len()).rev() {
                last = Some(self.insert_node(r.pts[i].0, r.pts[i].1, start_index, last));
                start_index += 1;
            }
        }
        if let Some(l) = last {
            if self.vertex_eq(l, self.next(l)) {
                self.remove_node(l, true);
                last = Some(self.next(l));
            }
        }
        self.filter_points(last, None)
    }

    fn eliminate_holes(&mut self, poly: &Ring<'_>, holes: &[Ring<'_>], outer: usize) -> Result<usize, String> {
        let mut list: Vec<(usize, usize)> = Vec::with_capacity(holes.len());
        let mut node_index = poly.pts.len() as i32;
        for (i, h) in holes.iter().enumerate() {
            let l = self.create_list(h, node_index, Winding::Ccw);
            let Some(l) = l.filter(|&l| l != self.next(l)) else {
                return Err(format!("Points are all coplanar in hole: {}", geojson(h.pts)));
            };
            let left = self.fetch_leftmost(l);
            list.push((left, i));
            node_index += h.pts.len() as i32;
        }
        list.sort_by(|&(a, _), &(b, _)| {
            let mut diff = self.x(a) - self.x(b);
            if diff == 0.0 {
                diff = self.y(a) - self.y(b);
                if diff == 0.0 {
                    let da = self.y(self.prev(a)).min(self.y(self.next(a)));
                    let db = self.y(self.prev(b)).min(self.y(self.next(b)));
                    diff = da - db;
                }
            }
            if diff < 0.0 {
                std::cmp::Ordering::Less
            } else if diff > 0.0 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let mut outer = outer;
        for &(hole_node, hi) in &list {
            let h = &holes[hi];
            match self.eliminate_hole(hole_node, outer, h.min_x, h.max_x, h.min_y, h.max_y) {
                Some(r) => {
                    let nr = self.next(r);
                    outer = self.filter_points(Some(r), Some(nr)).unwrap_or(r);
                }
                None => {
                    let mut p = geojson(h.pts);
                    if p.len() > 100 {
                        p.truncate(100);
                        p.push_str("...");
                    }
                    return Err(format!("Illegal hole detected: {p}"));
                }
            }
        }
        Ok(self.filter_points(Some(outer), None).unwrap_or(outer))
    }

    fn eliminate_hole(&mut self, hole: usize, outer: usize, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Option<usize> {
        if let Some(m) = self.maybe_merge_shared(hole, outer, min_x, max_x, min_y, max_y) {
            return Some(m);
        }
        let bridge = self.fetch_hole_bridge(hole, outer)?;
        let from_polygon = self.point_in_line_n(bridge, self.next(bridge), hole)
            || self.point_in_line_n(bridge, self.prev(bridge), hole)
            || self.point_in_line_n(hole, self.next(hole), bridge)
            || self.point_in_line_n(hole, self.prev(hole), bridge);
        self.split_polygon(bridge, hole, from_polygon);
        Some(outer)
    }

    fn maybe_merge_shared(&mut self, hole: usize, outer: usize, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Option<usize> {
        let mut shared: Option<usize> = None;
        let mut shared_conn: Option<usize> = None;
        let mut leftmost_conn: Option<usize> = None;
        let mut next = outer;
        loop {
            let (nx, ny) = (self.x(next), self.y(next));
            if ny >= min_y && ny <= max_y && nx >= min_x && nx <= max_x {
                if let Some(new_shared) = self.shared_vertex(hole, next) {
                    if self.vertex_eq(new_shared, hole) && leftmost_conn.is_none() {
                        leftmost_conn = Some(next);
                    }
                    match shared {
                        None => {
                            shared = Some(new_shared);
                            shared_conn = Some(next);
                        }
                        Some(s) if s == new_shared => {
                            shared_conn = Some(self.shared_inside_vertex(s, shared_conn.unwrap_or(next), next));
                        }
                        Some(_) => {}
                    }
                }
            }
            next = self.next(next);
            if next == outer {
                break;
            }
        }
        if let (Some(l), Some(sc)) = (leftmost_conn, shared_conn) {
            if self.n[l].idx >= self.n[sc].idx {
                self.split_polygon(l, hole, true);
                if l != sc {
                    return Some(l);
                }
                return Some(outer);
            }
        }
        if let (Some(s), Some(sc)) = (shared, shared_conn) {
            self.split_polygon(sc, s, true);
            return Some(outer);
        }
        None
    }

    fn shared_vertex(&self, polygon: usize, vertex: usize) -> Option<usize> {
        let mut next = polygon;
        loop {
            if self.vertex_eq(next, vertex) {
                return Some(next);
            }
            next = self.next(next);
            if next == polygon {
                return None;
            }
        }
    }

    fn shared_inside_vertex(&self, hole_vertex: usize, a: usize, b: usize) -> usize {
        let hn = self.next(hole_vertex);
        let a1 = area(self.x(self.prev(a)), self.y(self.prev(a)), self.x(hole_vertex), self.y(hole_vertex), self.x(hn), self.y(hn));
        let a2 = area(self.x(self.prev(b)), self.y(self.prev(b)), self.x(hole_vertex), self.y(hole_vertex), self.x(hn), self.y(hn));
        if (a1 < 0.0) != (a2 < 0.0) {
            if a1 < a2 {
                a
            } else {
                b
            }
        } else {
            let angle1 = self.angle(self.prev(a), a, hn);
            let angle2 = self.angle(self.prev(b), b, hn);
            if angle1 < angle2 {
                a
            } else {
                b
            }
        }
    }

    fn angle(&self, a: usize, b: usize, c: usize) -> f64 {
        let (ax, ay) = (self.x(a) - self.x(b), self.y(a) - self.y(b));
        let (cx, cy) = (self.x(c) - self.x(b), self.y(c) - self.y(b));
        let dot = ax * cx + ay * cy;
        let al = (ax * ax + ay * ay).sqrt();
        let bl = (cx * cx + cy * cy).sqrt();
        (dot / (al * bl)).acos()
    }

    fn fetch_hole_bridge(&self, hole: usize, outer: usize) -> Option<usize> {
        let mut p = outer;
        let mut qx = f64::NEG_INFINITY;
        let (hx, hy) = (self.x(hole), self.y(hole));
        let mut connection: Option<usize> = None;
        loop {
            let pn = self.next(p);
            if hy <= self.y(p) && hy >= self.y(pn) && self.y(pn) != self.y(p) {
                let x = self.x(p) + (hy - self.y(p)) * (self.x(pn) - self.x(p)) / (self.y(pn) - self.y(p));
                if x <= hx && x > qx {
                    qx = x;
                    if x == hx {
                        if hy == self.y(p) {
                            return Some(p);
                        }
                        if hy == self.y(pn) {
                            return Some(pn);
                        }
                    }
                    connection = Some(if self.x(p) < self.x(pn) { p } else { pn });
                }
            }
            p = pn;
            if p == outer {
                break;
            }
        }
        let mut connection = connection?;
        if hx == qx {
            return Some(self.prev(connection));
        }
        let stop = connection;
        let (mx, my) = (self.x(connection), self.y(connection));
        let mut tan_min = f64::INFINITY;
        p = connection;
        loop {
            let (px, py) = (self.x(p), self.y(p));
            if hx >= px
                && px >= mx
                && hx != px
                && point_in_ear(px, py, if hy < my { hx } else { qx }, hy, mx, my, if hy < my { qx } else { hx }, hy)
            {
                let tan = (hy - py).abs() / (hx - px);
                if (tan < tan_min || (tan == tan_min && px > self.x(connection))) && self.locally_inside(p, hole) {
                    connection = p;
                    tan_min = tan;
                }
            }
            p = self.next(p);
            if p == stop {
                break;
            }
        }
        Some(connection)
    }

    fn fetch_leftmost(&self, start: usize) -> usize {
        let mut node = start;
        let mut left = start;
        loop {
            if self.x(node) < self.x(left) || (self.x(node) == self.x(left) && self.y(node) < self.y(left)) {
                left = node;
            }
            node = self.next(node);
            if node == start {
                return left;
            }
        }
    }

    fn earcut(&mut self, curr: Option<usize>, out: &mut Vec<RawTri>, mut state: State, morton: bool, depth: u32) -> Result<(), String> {
        let mut curr_ear = curr;
        'earcut: loop {
            let Some(mut ear) = curr_ear else { return Ok(()) };
            if self.prev(ear) == self.next(ear) {
                return Ok(());
            }
            let mut stop = ear;
            loop {
                let prev = self.prev(ear);
                let next = self.next(ear);
                let reflex = self.area3(prev, ear, next) >= 0.0;
                if !reflex && self.is_ear(ear, morton) {
                    let ab = self.n[prev].efp;
                    let bc = self.n[ear].efp;
                    let ca = self.is_edge_from_polygon(prev, next, morton);
                    out.push((prev, ab, ear, bc, next, ca));
                    self.remove_node(ear, ca);
                    ear = self.next(next);
                    stop = self.next(next);
                    if self.prev(ear) == self.next(ear) {
                        break;
                    }
                    continue;
                }
                ear = next;
                if ear == stop {
                    match state {
                        State::Init => {
                            curr_ear = self.filter_points(Some(ear), None);
                            state = State::Cure;
                            continue 'earcut;
                        }
                        State::Cure => {
                            curr_ear = Some(self.cure_local_intersections(ear, out, morton));
                            state = State::Split;
                            continue 'earcut;
                        }
                        State::Split => {
                            if !self.split_earcut(ear, out, morton, depth + 1)? {
                                return Err(UNABLE.into());
                            }
                        }
                    }
                    break;
                }
                if self.prev(ear) == self.next(ear) {
                    break;
                }
            }
            break;
        }
        Ok(())
    }

    fn is_ear(&self, ear: usize, morton: bool) -> bool {
        if morton {
            return self.morton_is_ear(ear);
        }
        let (p, n) = (self.prev(ear), self.next(ear));
        let mut node = self.next(n);
        while node != p {
            if point_in_ear(self.x(node), self.y(node), self.x(p), self.y(p), self.x(ear), self.y(ear), self.x(n), self.y(n)) && self.area3(self.prev(node), node, self.next(node)) >= 0.0 {
                return false;
            }
            node = self.next(node);
        }
        true
    }

    /// Whether node `q` blocks ear `ear` (Lucene `mortonIsEar`'s test).
    fn blocks_ear(&self, q: usize, ear: usize) -> bool {
        let (p, n) = (self.prev(ear), self.next(ear));
        self.n[q].idx != self.n[p].idx
            && self.n[q].idx != self.n[n].idx
            && point_in_ear(self.x(q), self.y(q), self.x(p), self.y(p), self.x(ear), self.y(ear), self.x(n), self.y(n))
            && self.area3(self.prev(q), q, self.next(q)) >= 0.0
    }

    fn morton_is_ear(&self, ear: usize) -> bool {
        let (pe, ne) = (self.prev(ear), self.next(ear));
        let min_tx = self.n[pe].xe.min(self.n[ear].xe).min(self.n[ne].xe) ^ i32::MIN;
        let min_ty = self.n[pe].ye.min(self.n[ear].ye).min(self.n[ne].ye) ^ i32::MIN;
        let max_tx = self.n[pe].xe.max(self.n[ear].xe).max(self.n[ne].xe) ^ i32::MIN;
        let max_ty = self.n[pe].ye.max(self.n[ear].ye).max(self.n[ne].ye) ^ i32::MIN;
        let min_z = interleave(min_tx, min_ty);
        let max_z = interleave(max_tx, max_ty);
        let mut p = unz(self.n[ear].prev_z);
        let mut n = unz(self.n[ear].next_z);
        while let (Some(pp), Some(nn)) = (p, n) {
            if !(self.n[pp].morton >= min_z && self.n[nn].morton <= max_z) {
                break;
            }
            if self.blocks_ear(pp, ear) {
                return false;
            }
            p = unz(self.n[pp].prev_z);
            if self.blocks_ear(nn, ear) {
                return false;
            }
            n = unz(self.n[nn].next_z);
        }
        while let Some(pp) = p {
            if self.n[pp].morton < min_z {
                break;
            }
            if self.blocks_ear(pp, ear) {
                return false;
            }
            p = unz(self.n[pp].prev_z);
        }
        while let Some(nn) = n {
            if self.n[nn].morton > max_z {
                break;
            }
            if self.blocks_ear(nn, ear) {
                return false;
            }
            n = unz(self.n[nn].next_z);
        }
        true
    }

    fn cure_local_intersections(&mut self, start: usize, out: &mut Vec<RawTri>, morton: bool) -> usize {
        let mut start = start;
        let mut node = start;
        loop {
            let next = self.next(node);
            let a = self.prev(node);
            let b = self.next(next);
            if !self.vertex_eq(a, b)
                && lines_intersect(self.x(a), self.y(a), self.x(node), self.y(node), self.x(next), self.y(next), self.x(b), self.y(b))
                && self.locally_inside(a, b)
                && self.locally_inside(b, a)
                && !self.intersecting_polygon(a, self.x(a), self.y(a), self.x(b), self.y(b))
            {
                let ab = if self.next(a) == node { self.n[a].efp } else { self.is_edge_from_polygon(a, node, morton) };
                let bc = if self.next(node) == b { self.n[node].efp } else { self.is_edge_from_polygon(node, b, morton) };
                let ca = if self.next(b) == a { self.n[b].efp } else { self.is_edge_from_polygon(a, b, morton) };
                // Lucene adds this triangle twice.
                out.push((a, ab, node, bc, b, ca));
                out.push((a, ab, node, bc, b, ca));
                self.remove_node(node, ca);
                let nn = self.next(node);
                self.remove_node(nn, ca);
                node = b;
                start = b;
            }
            node = self.next(node);
            if node == start {
                return node;
            }
        }
    }

    fn split_earcut(&mut self, start: usize, out: &mut Vec<RawTri>, morton: bool, depth: u32) -> Result<bool, String> {
        let mut search = start;
        loop {
            let next = self.next(search);
            let mut diagonal = self.next(next);
            while diagonal != self.prev(search) {
                if self.n[search].idx != self.n[diagonal].idx && self.is_valid_diagonal(search, diagonal) {
                    let efp = self.is_edge_from_polygon(search, diagonal, morton);
                    let split = self.split_polygon(search, diagonal, efp);
                    let sn = self.next(search);
                    let search2 = self.filter_points(Some(search), Some(sn)).unwrap_or(search);
                    let spn = self.next(split);
                    let split2 = self.filter_points(Some(split), Some(spn)).unwrap_or(split);
                    if morton {
                        self.sort_by_morton_with_reset(search2);
                        self.sort_by_morton_with_reset(split2);
                    }
                    self.earcut(Some(search2), out, State::Init, morton, depth)?;
                    self.earcut(Some(split2), out, State::Init, morton, depth)?;
                    return Ok(true);
                }
                diagonal = self.next(diagonal);
            }
            search = self.next(search);
            if search == start {
                break;
            }
        }
        Ok(self.signed_area(start, start) == 0.0)
    }

    /// Lucene's `checkIntersection` over arrays: the same pairs in the same
    /// order (so the same first error), the ring or z-order list read from
    /// contiguous ids and edge boxes instead of chasing links. Falls back
    /// to [`Tess::check_intersection_linked`] when the lists are not as
    /// expected.
    fn check_intersection(&mut self, a: usize, morton: bool) -> Result<(), String> {
        if !self.fill_order(a, morton) {
            return self.check_intersection_linked(a, morton);
        }
        let m = self.order.len();
        // Disjoint edge boxes: Lucene's rejection, from the arrays.
        let clear = |p: &[f64; 4], q: &[f64; 4]| p[1] <= q[0] || p[0] >= q[1] || p[3] <= q[2] || p[2] >= q[3];
        if !morton {
            // Ring positions: order[0] = a; the outer edge runs over 1..m-1
            // (until `a.previous`), the inner from i + 1 around to i - 1.
            for i in 1..m - 1 {
                let bi = self.order_boxes[i];
                let ni = self.order[i] as usize;
                let stop = (i + m - 1) % m;
                let mut j = (i + 1) % m;
                loop {
                    if !clear(&bi, &self.order_boxes[j]) {
                        self.check_intersection_point(ni, self.order[j] as usize)?;
                    }
                    j = (j + 1) % m;
                    if j == stop {
                        break;
                    }
                }
            }
            return Ok(());
        }
        // Z order: for each edge (Lucene's ring walk from `a.next` to before
        // `a.previous`), the nodes around its next node within the edge's
        // Morton range, alternating down and up, then the rest of each side.
        let mut next = self.next(a);
        loop {
            let b = self.next(next);
            let pb = self.order_pos[b];
            if pb == NIL {
                self.morton_check_intersection(next, b)?;
            } else {
                let an = self.next(next);
                let min_tx = self.n[next].xe.min(self.n[an].xe) ^ i32::MIN;
                let min_ty = self.n[next].ye.min(self.n[an].ye) ^ i32::MIN;
                let max_tx = self.n[next].xe.max(self.n[an].xe) ^ i32::MIN;
                let max_ty = self.n[next].ye.max(self.n[an].ye) ^ i32::MIN;
                let (min_z, max_z) = (interleave(min_tx, min_ty), interleave(max_tx, max_ty));
                let ba = self.boxes[next];
                let pb = pb as usize;
                let (mut i, mut j) = (pb.checked_sub(1), pb + 1);
                let test = |t: &Tess, k: usize| -> Result<(), String> {
                    if clear(&t.order_boxes[k], &ba) {
                        return Ok(());
                    }
                    t.check_intersection_point(t.order[k] as usize, next)
                };
                while let Some(ii) = i {
                    if !(self.order_morton[ii] >= min_z && j < m && self.order_morton[j] <= max_z) {
                        break;
                    }
                    test(self, ii)?;
                    i = ii.checked_sub(1);
                    test(self, j)?;
                    j += 1;
                }
                while let Some(ii) = i {
                    if self.order_morton[ii] < min_z {
                        break;
                    }
                    test(self, ii)?;
                    i = ii.checked_sub(1);
                }
                while j < m && self.order_morton[j] <= max_z {
                    test(self, j)?;
                    j += 1;
                }
            }
            next = self.next(next);
            if next == self.prev(a) {
                return Ok(());
            }
        }
    }

    /// Fill the visiting order of [`Tess::check_intersection`]: the ring
    /// from `a` (not Morton) or the z-order list from its head (Morton).
    /// `false` when a list does not close as expected.
    fn fill_order(&mut self, a: usize, morton: bool) -> bool {
        let len = self.n.len();
        self.order.clear();
        self.order_boxes.clear();
        self.order_morton.clear();
        self.order_pos.clear();
        self.order_pos.resize(len, NIL);
        if !morton {
            let mut q = a;
            loop {
                if self.order.len() >= len || self.order_pos[q] != NIL {
                    return false;
                }
                self.order_pos[q] = self.order.len() as u32;
                self.order.push(q as u32);
                q = self.next(q);
                if q == a {
                    break;
                }
            }
        } else {
            // The head of the sorted z list (no previous), then forward.
            let mut head = a;
            let mut steps = 0;
            while let Some(p) = unz(self.n[head].prev_z) {
                head = p;
                steps += 1;
                if steps > len {
                    return false;
                }
            }
            let mut q = Some(head);
            while let Some(i) = q {
                if self.order.len() >= len || self.order_pos[i] != NIL {
                    return false;
                }
                self.order_pos[i] = self.order.len() as u32;
                self.order.push(i as u32);
                q = unz(self.n[i].next_z);
            }
        }
        for k in 0..self.order.len() {
            let i = self.order[k] as usize;
            self.order_boxes.push(self.boxes[i]);
            self.order_morton.push(self.n[i].morton);
        }
        self.order.len() >= 3 || morton
    }

    fn check_intersection_linked(&self, a: usize, morton: bool) -> Result<(), String> {
        let mut next = self.next(a);
        loop {
            let mut inner = self.next(next);
            if morton {
                self.morton_check_intersection(next, inner)?;
            } else {
                loop {
                    self.check_intersection_point(next, inner)?;
                    inner = self.next(inner);
                    if inner == self.prev(next) {
                        break;
                    }
                }
            }
            next = self.next(next);
            if next == self.prev(a) {
                return Ok(());
            }
        }
    }

    fn morton_check_intersection(&self, a: usize, b: usize) -> Result<(), String> {
        let an = self.next(a);
        let min_tx = self.n[a].xe.min(self.n[an].xe) ^ i32::MIN;
        let min_ty = self.n[a].ye.min(self.n[an].ye) ^ i32::MIN;
        let max_tx = self.n[a].xe.max(self.n[an].xe) ^ i32::MIN;
        let max_ty = self.n[a].ye.max(self.n[an].ye) ^ i32::MIN;
        let min_z = interleave(min_tx, min_ty);
        let max_z = interleave(max_tx, max_ty);
        let mut p = unz(self.n[b].prev_z);
        let mut n = unz(self.n[b].next_z);
        while let (Some(pp), Some(nn)) = (p, n) {
            if !(self.n[pp].morton >= min_z && self.n[nn].morton <= max_z) {
                break;
            }
            self.check_intersection_point(pp, a)?;
            p = unz(self.n[pp].prev_z);
            self.check_intersection_point(nn, a)?;
            n = unz(self.n[nn].next_z);
        }
        while let Some(pp) = p {
            if self.n[pp].morton < min_z {
                break;
            }
            self.check_intersection_point(pp, a)?;
            p = unz(self.n[pp].prev_z);
        }
        while let Some(nn) = n {
            if self.n[nn].morton > max_z {
                break;
            }
            self.check_intersection_point(nn, a)?;
            n = unz(self.n[nn].next_z);
        }
        Ok(())
    }

    /// Fill [`Tess::boxes`] for every node from its current `next`.
    fn fill_edge_boxes(&mut self) {
        self.boxes.clear();
        self.boxes.reserve(self.n.len());
        for i in 0..self.n.len() {
            let j = self.n[i].next as usize;
            let (ax, ay, bx, by) = (self.n[i].x, self.n[i].y, self.n[j].x, self.n[j].y);
            self.boxes.push([ay.min(by), ay.max(by), ax.min(bx), ax.max(bx)]);
        }
    }

    fn check_intersection_point(&self, a: usize, b: usize) -> Result<(), String> {
        if a == b {
            return Ok(());
        }
        // Lucene's box rejection, from the edge boxes (the same min / max
        // of the same coordinates, read from one contiguous array).
        let (ea, eb) = (&self.boxes[a], &self.boxes[b]);
        if ea[1] <= eb[0] || ea[0] >= eb[1] || ea[3] <= eb[2] || ea[2] >= eb[3] {
            return Ok(());
        }
        let (an, bn) = (self.next(a), self.next(b));
        let (ax, ay, anx, any) = (self.x(a), self.y(a), self.x(an), self.y(an));
        let (bx, by, bnx, bny) = (self.x(b), self.y(b), self.x(bn), self.y(bn));
        if line_crosses_line(ax, ay, anx, any, bx, by, bnx, bny) {
            let a1 = any - ay;
            let b1 = ax - anx;
            let c1 = a1 * ax + b1 * ay;
            let a2 = bny - by;
            let b2 = bx - bnx;
            let c2 = a2 * bx + b2 * by;
            let det = a1 * b2 - a2 * b1;
            let x = (b2 * c1 - b1 * c2) / det;
            let y = (a1 * c2 - a2 * c1) / det;
            return Err(format!("Polygon self-intersection at lat={} lon={}", java_double(y), java_double(x)));
        }
        if self.n[a].efp && self.n[b].efp && line_overlap_line(ax, ay, anx, any, bx, by, bnx, bny) {
            return Err(format!("Polygon ring self-intersection at lat={} lon={}", java_double(ay), java_double(ax)));
        }
        Ok(())
    }

    fn is_edge_from_polygon(&self, a: usize, b: usize, morton: bool) -> bool {
        if morton {
            return self.morton_is_edge_from_polygon(a, b);
        }
        let mut next = a;
        loop {
            if let Some(v) = self.edge_check(next, a, b) {
                return v;
            }
            next = self.next(next);
            if next == a {
                return false;
            }
        }
    }

    /// Lucene's two `isPointInLine` checks of `isEdgeFromPolygon` at node `q`.
    fn edge_check(&self, q: usize, a: usize, b: usize) -> Option<bool> {
        let (qn, qp) = (self.next(q), self.prev(q));
        if self.point_in_line_n(q, qn, a) && self.point_in_line_n(q, qn, b) {
            return Some(self.n[q].efp);
        }
        if self.point_in_line_n(q, qp, a) && self.point_in_line_n(q, qp, b) {
            return Some(self.n[qp].efp);
        }
        None
    }

    fn morton_is_edge_from_polygon(&self, a: usize, b: usize) -> bool {
        let min_tx = self.n[a].xe.min(self.n[b].xe) ^ i32::MIN;
        let min_ty = self.n[a].ye.min(self.n[b].ye) ^ i32::MIN;
        let max_tx = self.n[a].xe.max(self.n[b].xe) ^ i32::MIN;
        let max_ty = self.n[a].ye.max(self.n[b].ye) ^ i32::MIN;
        let min_z = interleave(min_tx, min_ty);
        let max_z = interleave(max_tx, max_ty);
        let mut p = unz(self.n[a].prev_z);
        let mut n = unz(self.n[a].next_z);
        while let (Some(pp), Some(nn)) = (p, n) {
            if !(self.n[pp].morton >= min_z && self.n[nn].morton <= max_z) {
                break;
            }
            if let Some(v) = self.edge_check(pp, a, b) {
                return v;
            }
            p = unz(self.n[pp].prev_z);
            if let Some(v) = self.edge_check(nn, a, b) {
                return v;
            }
            n = unz(self.n[nn].next_z);
        }
        while let Some(pp) = p {
            if self.n[pp].morton < min_z {
                break;
            }
            if let Some(v) = self.edge_check(pp, a, b) {
                return v;
            }
            p = unz(self.n[pp].prev_z);
        }
        while let Some(nn) = n {
            if self.n[nn].morton > max_z {
                break;
            }
            if let Some(v) = self.edge_check(nn, a, b) {
                return v;
            }
            n = unz(self.n[nn].next_z);
        }
        false
    }

    fn point_in_line_n(&self, a: usize, b: usize, p: usize) -> bool {
        self.point_in_line(a, b, self.x(p), self.y(p))
    }

    fn point_in_line(&self, a: usize, b: usize, lon: f64, lat: f64) -> bool {
        let (ax, ay, bx, by) = (self.x(a), self.y(a), self.x(b), self.y(b));
        let dxc = lon - ax;
        let dyc = lat - ay;
        let dxl = bx - ax;
        let dyl = by - ay;
        if dxc * dyl - dyc * dxl == 0.0 {
            if dxl.abs() >= dyl.abs() {
                return if dxl > 0.0 { ax <= lon && lon <= bx } else { bx <= lon && lon <= ax };
            } else {
                return if dyl > 0.0 { ay <= lat && lat <= by } else { by <= lat && lat <= ay };
            }
        }
        false
    }

    fn split_polygon(&mut self, a: usize, b: usize, efp: bool) -> usize {
        let a2 = self.n.len();
        self.n.push(self.n[a]);
        let b2 = self.n.len();
        self.n.push(self.n[b]);
        let an = self.next(a);
        let bp = self.prev(b);
        self.n[a].next = b as u32;
        self.n[a].efp = efp;
        self.n[a].next_z = b as u32;
        self.n[b].prev = a as u32;
        self.n[b].prev_z = a as u32;
        self.n[a2].next = an as u32;
        self.n[a2].next_z = an as u32;
        self.n[an].prev = a2 as u32;
        self.n[an].prev_z = a2 as u32;
        self.n[b2].next = a2 as u32;
        self.n[b2].efp = efp;
        self.n[b2].next_z = a2 as u32;
        self.n[a2].prev = b2 as u32;
        self.n[a2].prev_z = b2 as u32;
        self.n[bp].next = b2 as u32;
        self.n[bp].next_z = b2 as u32;
        b2
    }

    fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
        if self.n[self.next(a)].idx == self.n[b].idx
            || self.n[self.prev(a)].idx == self.n[b].idx
            || !self.locally_inside(self.prev(a), b)
            || !self.locally_inside(self.next(b), a)
            || !self.is_cw_polygon(a, b)
            || !self.is_cw_polygon(b, a)
        {
            return false;
        }
        if self.vertex_eq(a, b) {
            return true;
        }
        let (ax, ay, bx, by) = (self.x(a), self.y(a), self.x(b), self.y(b));
        self.locally_inside(a, b)
            && self.locally_inside(b, a)
            && self.middle_insert(a, ax, ay, bx, by)
            && area(self.x(self.prev(a)), self.y(self.prev(a)), ax, ay, bx, by) != 0.0
            && area(ax, ay, bx, by, self.x(self.next(b)), self.y(self.next(b))) != 0.0
            && area(self.x(self.next(a)), self.y(self.next(a)), ax, ay, bx, by) != 0.0
            && area(ax, ay, bx, by, self.x(self.prev(b)), self.y(self.prev(b))) != 0.0
            && !self.intersecting_polygon(a, ax, ay, bx, by)
    }

    fn is_cw_polygon(&self, start: usize, end: usize) -> bool {
        self.signed_area(start, end) < 0.0
    }

    fn signed_area(&self, start: usize, end: usize) -> f64 {
        let mut next = start;
        let mut sum = 0.0;
        loop {
            let nn = self.next(next);
            sum += area(self.x(next), self.y(next), self.x(nn), self.y(nn), self.x(end), self.y(end));
            next = nn;
            if self.next(next) == end {
                return sum;
            }
        }
    }

    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (p, n) = (self.prev(a), self.next(a));
        let ar = self.area3(p, a, n);
        if ar == 0.0 {
            false
        } else if ar < 0.0 {
            self.area3(a, b, n) >= 0.0 && self.area3(a, p, b) >= 0.0
        } else {
            self.area3(a, b, p) <= 0.0 || self.area3(a, n, b) <= 0.0
        }
    }

    fn middle_insert(&self, start: usize, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
        let mut node = start;
        let mut inside = false;
        let ldx = (x0 + x1) / 2.0;
        let ldy = (y0 + y1) / 2.0;
        loop {
            let next = self.next(node);
            if (self.y(node) > ldy) != (self.y(next) > ldy) && ldx < (self.x(next) - self.x(node)) * (ldy - self.y(node)) / (self.y(next) - self.y(node)) + self.x(node) {
                inside = !inside;
            }
            node = next;
            if node == start {
                return inside;
            }
        }
    }

    fn intersecting_polygon(&self, start: usize, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
        let mut node = start;
        loop {
            let next = self.next(node);
            if !self.vertex_eq_xy(node, x0, y0) && !self.vertex_eq_xy(node, x1, y1) && lines_intersect(self.x(node), self.y(node), self.x(next), self.y(next), x0, y0, x1, y1) {
                return true;
            }
            node = next;
            if node == start {
                return false;
            }
        }
    }

    fn sort_by_morton_with_reset(&mut self, start: usize) {
        let mut next = start;
        loop {
            self.n[next].prev_z = self.n[next].prev;
            self.n[next].next_z = self.n[next].next;
            next = self.next(next);
            if next == start {
                break;
            }
        }
        self.sort_by_morton(start);
    }

    fn sort_by_morton(&mut self, start: usize) {
        if let Some(pz) = unz(self.n[start].prev_z) {
            self.n[pz].next_z = NIL;
        }
        self.n[start].prev_z = NIL;
        self.tatham_sort(Some(start));
    }

    /// Simon Tatham's linked-list merge sort on the z links, by Morton code.
    fn tatham_sort(&mut self, list: Option<usize>) {
        let mut list = list;
        if list.is_none() {
            return;
        }
        let mut in_size = 1usize;
        loop {
            let mut p = list;
            list = None;
            let mut tail: Option<usize> = None;
            let mut num_merges = 0;
            while let Some(pp) = p {
                num_merges += 1;
                let mut q = Some(pp);
                let mut p_size = 0usize;
                for _ in 0..in_size {
                    match q {
                        Some(qq) => {
                            p_size += 1;
                            q = unz(self.n[qq].next_z);
                        }
                        None => break,
                    }
                }
                let mut q_size = in_size;
                let mut p_cur = Some(pp);
                while p_size > 0 || (q_size > 0 && q.is_some()) {
                    let e;
                    let take_p = p_size != 0 && (q_size == 0 || q.is_none() || self.n[p_cur.unwrap()].morton <= self.n[q.unwrap()].morton);
                    if take_p {
                        e = p_cur.unwrap();
                        p_cur = unz(self.n[e].next_z);
                        p_size -= 1;
                    } else {
                        e = q.unwrap();
                        q = unz(self.n[e].next_z);
                        q_size -= 1;
                    }
                    match tail {
                        Some(t) => self.n[t].next_z = e as u32,
                        None => list = Some(e),
                    }
                    self.n[e].prev_z = z(tail);
                    tail = Some(e);
                }
                p = q;
            }
            if let Some(t) = tail {
                self.n[t].next_z = NIL;
            }
            in_size *= 2;
            if num_merges <= 1 {
                break;
            }
        }
    }

    fn filter_points(&mut self, start: Option<usize>, end: Option<usize>) -> Option<usize> {
        let start = start?;
        let mut end = end.unwrap_or(start);
        let mut node = start;
        loop {
            let mut cont = false;
            let next = self.next(node);
            let prev = self.prev(node);
            if self.vertex_eq(node, next)
                || self.vertex_eq(prev, next)
                || ((self.n[prev].efp == self.n[node].efp || self.point_in_line(prev, node, self.x(next), self.y(next))) && self.area3(prev, node, next) == 0.0)
            {
                let efp = self.n[prev].efp;
                self.remove_node(node, efp);
                node = prev;
                end = prev;
                if node == next {
                    break;
                }
                cont = true;
            } else {
                node = next;
            }
            if !(cont || node != end) {
                break;
            }
        }
        Some(end)
    }

    fn insert_node(&mut self, x: f64, y: f64, idx: i32, last: Option<usize>) -> usize {
        let id = self.n.len();
        let (xe, ye) = (encode_lon(x), encode_lat(y));
        let morton = interleave(xe ^ i32::MIN, ye ^ i32::MIN);
        self.n.push(Node { idx, x, y, xe, ye, morton, prev: id as u32, next: id as u32, prev_z: id as u32, next_z: id as u32, efp: true });
        if let Some(l) = last {
            let ln = self.n[l].next as usize;
            let lnz = unz(self.n[l].next_z);
            self.n[id].next = ln as u32;
            self.n[id].next_z = z(lnz);
            self.n[id].prev = l as u32;
            self.n[id].prev_z = l as u32;
            self.n[ln].prev = id as u32;
            if let Some(zn) = lnz {
                self.n[zn].prev_z = id as u32;
            }
            self.n[l].next = id as u32;
            self.n[l].next_z = id as u32;
        }
        id
    }

    fn remove_node(&mut self, node: usize, efp: bool) {
        let (p, n) = (self.n[node].prev as usize, self.n[node].next as usize);
        self.n[n].prev = p as u32;
        self.n[p].next = n as u32;
        self.n[p].efp = efp;
        let (pz, nz) = (unz(self.n[node].prev_z), unz(self.n[node].next_z));
        if let Some(pz) = pz {
            self.n[pz].next_z = z(nz);
        }
        if let Some(nz) = nz {
            self.n[nz].prev_z = z(pz);
        }
    }

    #[inline]
    fn vertex_eq(&self, a: usize, b: usize) -> bool {
        self.x(a) == self.x(b) && self.y(a) == self.y(b)
    }

    #[inline]
    fn vertex_eq_xy(&self, a: usize, x: f64, y: f64) -> bool {
        self.x(a) == x && self.y(a) == y
    }
}

/// Java's `Double.toString` for the coordinates of error messages (close
/// enough: plain decimal with at least one fractional digit).
fn java_double(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e7 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

/// Lucene `Polygon.verticesToGeoJSON` of a ring.
fn geojson(pts: &[(f64, f64)]) -> String {
    let parts: Vec<String> = pts.iter().map(|p| format!("[{}, {}]", java_double(p.0), java_double(p.1))).collect();
    format!("[{}]", parts.join(", "))
}

// ---------------------------------------------------------------------------
// `ShapeField` triangle encoding.

const MINY_MINX_MAXY_MAXX_Y_X: i32 = 0;
const MINY_MINX_Y_X_MAXY_MAXX: i32 = 1;
const MAXY_MINX_Y_X_MINY_MAXX: i32 = 2;
const MAXY_MINX_MINY_MAXX_Y_X: i32 = 3;
const Y_MINX_MINY_X_MAXY_MAXX: i32 = 4;
const Y_MINX_MINY_MAXX_MAXY_X: i32 = 5;
const MAXY_MINX_MINY_X_Y_MAXX: i32 = 6;
const MINY_MINX_Y_MAXX_MAXY_X: i32 = 7;

/// A triangle as `ShapeField.encodeTriangle` stores it: `[min_y, min_x,
/// max_y, max_x, y, x, bits]` (the low 3 bits of `bits` say which corner
/// of the box each vertex is; bits 3-5 are the `ab`, `bc`, `ca` flags).
/// `Err` for a triangle Lucene cannot encode.
#[allow(clippy::too_many_arguments)]
pub fn encode_triangle(a_y: i32, a_x: i32, ab: bool, b_y: i32, b_x: i32, bc: bool, c_y: i32, c_x: i32, ca: bool) -> Result<[i32; 7], String> {
    let (mut ax, mut ay, mut ab) = (a_x, a_y, ab);
    let (mut bx, mut by, mut bc) = (b_x, b_y, bc);
    let (mut cx, mut cy, mut ca) = (c_x, c_y, ca);
    if bx < ax || cx < ax {
        let (tx, ty, tb) = (ax, ay, ab);
        if bx < cx {
            ax = bx;
            ay = by;
            ab = bc;
            bx = cx;
            by = cy;
            bc = ca;
            cx = tx;
            cy = ty;
            ca = tb;
        } else {
            ax = cx;
            ay = cy;
            ab = ca;
            cx = bx;
            cy = by;
            ca = bc;
            bx = tx;
            by = ty;
            bc = tb;
        }
    } else if ax == bx && ax == cx && (by < ay || cy < ay) {
        let (tx, ty, tb) = (ax, ay, ab);
        if by < cy {
            ax = bx;
            ay = by;
            ab = bc;
            bx = cx;
            by = cy;
            bc = ca;
            cx = tx;
            cy = ty;
            ca = tb;
        } else {
            ax = cx;
            ay = cy;
            ab = ca;
            cx = bx;
            cy = by;
            ca = bc;
            bx = tx;
            by = ty;
            bc = tb;
        }
    }
    if orient(ax as f64, ay as f64, bx as f64, by as f64, cx as f64, cy as f64) == -1 {
        // As Lucene: `bc` keeps its value.
        let (tx, ty, tb) = (bx, by, ab);
        ab = bc;
        bx = cx;
        by = cy;
        cx = tx;
        cy = ty;
        ca = tb;
    }
    let min_x = ax;
    let min_y = ay.min(by.min(cy));
    let max_x = ax.max(bx.max(cx));
    let max_y = ay.max(by.max(cy));
    let (mut bits, x, y);
    if min_y == ay {
        if max_y == by && max_x == bx {
            y = cy;
            x = cx;
            bits = MINY_MINX_MAXY_MAXX_Y_X;
        } else if max_y == cy && max_x == cx {
            y = by;
            x = bx;
            bits = MINY_MINX_Y_X_MAXY_MAXX;
        } else {
            y = by;
            x = cx;
            bits = MINY_MINX_Y_MAXX_MAXY_X;
        }
    } else if max_y == ay {
        if min_y == by && max_x == bx {
            y = cy;
            x = cx;
            bits = MAXY_MINX_MINY_MAXX_Y_X;
        } else if min_y == cy && max_x == cx {
            y = by;
            x = bx;
            bits = MAXY_MINX_Y_X_MINY_MAXX;
        } else {
            y = cy;
            x = bx;
            bits = MAXY_MINX_MINY_X_Y_MAXX;
        }
    } else if max_x == bx && min_y == by {
        y = ay;
        x = cx;
        bits = Y_MINX_MINY_MAXX_MAXY_X;
    } else if max_x == cx && max_y == cy {
        y = ay;
        x = bx;
        bits = Y_MINX_MINY_X_MAXY_MAXX;
    } else {
        return Err("Could not encode the provided triangle".into());
    }
    if ab {
        bits |= 1 << 3;
    }
    if bc {
        bits |= 1 << 4;
    }
    if ca {
        bits |= 1 << 5;
    }
    Ok([min_y, min_x, max_y, max_x, y, x, bits])
}

/// Kind of a decoded triangle (`ShapeField.DecodedTriangle.TYPE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriKind {
    Point,
    Line,
    Triangle,
}

/// `ShapeField.DecodedTriangle`: CCW (or degenerate) vertices in encoded
/// ints, the edge flags and the kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub ax: i32,
    pub ay: i32,
    pub bx: i32,
    pub by: i32,
    pub cx: i32,
    pub cy: i32,
    pub ab: bool,
    pub bc: bool,
    pub ca: bool,
    pub kind: TriKind,
}

/// `ShapeField.decodeTriangle` (with `resolveTriangleType`).
pub fn decode_triangle(t: &[i32; 7]) -> Decoded {
    let bits = t[6];
    let (ay, ax, by, bx, cy, cx) = match bits & 7 {
        MINY_MINX_MAXY_MAXX_Y_X => (t[0], t[1], t[2], t[3], t[4], t[5]),
        MINY_MINX_Y_X_MAXY_MAXX => (t[0], t[1], t[4], t[5], t[2], t[3]),
        MAXY_MINX_Y_X_MINY_MAXX => (t[2], t[1], t[4], t[5], t[0], t[3]),
        MAXY_MINX_MINY_MAXX_Y_X => (t[2], t[1], t[0], t[3], t[4], t[5]),
        Y_MINX_MINY_X_MAXY_MAXX => (t[4], t[1], t[0], t[5], t[2], t[3]),
        Y_MINX_MINY_MAXX_MAXY_X => (t[4], t[1], t[0], t[3], t[2], t[5]),
        MAXY_MINX_MINY_X_Y_MAXX => (t[2], t[1], t[0], t[5], t[4], t[3]),
        _ => (t[0], t[1], t[4], t[3], t[2], t[5]),
    };
    let mut d = Decoded { ax, ay, bx, by, cx, cy, ab: bits & (1 << 3) != 0, bc: bits & (1 << 4) != 0, ca: bits & (1 << 5) != 0, kind: TriKind::Triangle };
    if d.ax == d.bx && d.ay == d.by {
        if d.ax == d.cx && d.ay == d.cy {
            d.kind = TriKind::Point;
        } else {
            d.ab = d.bc | d.ca;
            d.bx = d.cx;
            d.by = d.cy;
            d.cx = d.ax;
            d.cy = d.ay;
            d.kind = TriKind::Line;
        }
    } else if d.ax == d.cx && d.ay == d.cy {
        d.ab |= d.bc;
        d.kind = TriKind::Line;
    } else if d.bx == d.cx && d.by == d.cy {
        d.ab |= d.ca;
        d.cx = d.ax;
        d.cy = d.ay;
        d.kind = TriKind::Line;
    }
    d
}

// ---------------------------------------------------------------------------
// Geometries into triangles (ES `GeoShapeIndexer` over Lucene `LatLonShape`).

/// The encoded triangles ES indexes for a `geo_shape` value, from its
/// dateline-normalized geometry ([`crate::fieldtypes::shape::normalize_geo`]):
/// a point as `(p, p, p)`, each line segment as `(a, b, a)`, each polygon
/// tessellated (self-intersections checked, as ES asks), envelopes as ES
/// splits and quantizes them. `Err`: a polygon Lucene refuses.
pub fn geometry_triangles(g: &crate::fieldtypes::shape::Geometry, out: &mut Vec<[i32; 7]>) -> Result<(), String> {
    use crate::fieldtypes::shape::Geometry;
    match g {
        Geometry::Point(p) => point(*p, out),
        Geometry::MultiPoint(ps) => {
            for p in ps {
                point(*p, out)?;
            }
            Ok(())
        }
        Geometry::LineString(l) => line(l, out),
        Geometry::MultiLineString(ls) => {
            for l in ls {
                line(l, out)?;
            }
            Ok(())
        }
        Geometry::Polygon(rings) => polygon(rings, out),
        Geometry::MultiPolygon(ps) => {
            for p in ps {
                polygon(p, out)?;
            }
            Ok(())
        }
        Geometry::Collection(gs) => {
            for g in gs {
                geometry_triangles(g, out)?;
            }
            Ok(())
        }
        Geometry::Envelope { min_x, max_x, max_y, min_y } => rectangle(*min_x, *max_x, *max_y, *min_y, out),
    }
}

fn check(p: (f64, f64)) -> Result<(), String> {
    if !(-90.0..=90.0).contains(&p.1) || !(-180.0..=180.0).contains(&p.0) || p.0.is_nan() || p.1.is_nan() {
        return Err(format!("invalid coordinates [{}, {}]", p.0, p.1));
    }
    Ok(())
}

fn push(out: &mut Vec<[i32; 7]>, a: (i32, i32), ab: bool, b: (i32, i32), bc: bool, c: (i32, i32), ca: bool) -> Result<(), String> {
    out.push(encode_triangle(a.1, a.0, ab, b.1, b.0, bc, c.1, c.0, ca)?);
    Ok(())
}

fn enc(p: (f64, f64)) -> (i32, i32) {
    (encode_lon(p.0), encode_lat(p.1))
}

fn point(p: (f64, f64), out: &mut Vec<[i32; 7]>) -> Result<(), String> {
    check(p)?;
    let e = enc(p);
    push(out, e, true, e, true, e, true)
}

fn line(l: &[(f64, f64)], out: &mut Vec<[i32; 7]>) -> Result<(), String> {
    for p in l {
        check(*p)?;
    }
    out.reserve(l.len().saturating_sub(1));
    for w in l.windows(2) {
        let (a, b) = (enc(w[0]), enc(w[1]));
        push(out, a, true, b, true, a, true)?;
    }
    Ok(())
}

fn polygon(rings: &[Vec<(f64, f64)>], out: &mut Vec<[i32; 7]>) -> Result<(), String> {
    let Some((outer, holes)) = rings.split_first() else { return Ok(()) };
    tessellate_encoded(outer, holes, true, out)
}

/// ES `GeoShapeIndexer.LuceneGeometryIndexer.visit(Rectangle)`.
fn rectangle(min_x: f64, max_x: f64, max_y: f64, min_y: f64, out: &mut Vec<[i32; 7]>) -> Result<(), String> {
    let (min_lat, max_lat) = (encode_lat(min_y), encode_lat(max_y));
    let (min_lon, max_lon) = (encode_lon(min_x), encode_lon(max_x));
    if min_x > max_x {
        if min_lon == i32::MAX {
            line(&[(180.0, max_y), (180.0, min_y)], out)?;
        } else {
            rectangle(min_x, 180.0, max_y, min_y, out)?;
        }
        if max_lon == i32::MIN {
            line(&[(-180.0, max_y), (-180.0, min_y)], out)?;
        } else {
            rectangle(-180.0, max_x, max_y, min_y, out)?;
        }
        Ok(())
    } else if min_lon == max_lon {
        if min_lat == max_lat {
            point((min_x, min_y), out)
        } else {
            line(&[(min_x, max_y), (max_x, min_y)], out)
        }
    } else if min_lat == max_lat {
        line(&[(min_x, max_y), (max_x, min_y)], out)
    } else {
        use crate::geo::{decode_lat, decode_lon};
        let (x0, x1, y1, y0) = (decode_lon(min_lon), decode_lon(max_lon), decode_lat(max_lat), decode_lat(min_lat));
        let ring = vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)];
        for t in tessellate(&ring, &[], false)? {
            push(out, (t.ax, t.ay), t.ab, (t.bx, t.by), t.bc, (t.cx, t.cy), t.ca)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{decode_lat, decode_lon};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A star-shaped ring around `(cx, cy)` with `n` vertices, closed.
    fn star(r: &mut Lcg, cx: f64, cy: f64, rad: f64, n: usize, ccw: bool) -> Vec<(f64, f64)> {
        let mut pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                let rr = rad * (0.4 + 0.6 * r.next());
                (cx + rr * a.cos(), cy + rr * a.sin())
            })
            .collect();
        if !ccw {
            pts.reverse();
        }
        pts.push(pts[0]);
        pts
    }

    fn ring_area(r: &[(f64, f64)]) -> f64 {
        r.windows(2).map(|w| w[0].0 * w[1].1 - w[1].0 * w[0].1).sum::<f64>().abs() / 2.0
    }

    fn tri_area(t: &Triangle) -> f64 {
        let (ax, ay, bx, by, cx, cy) = (decode_lon(t.ax), decode_lat(t.ay), decode_lon(t.bx), decode_lat(t.by), decode_lon(t.cx), decode_lat(t.cy));
        ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs() / 2.0
    }

    fn in_ring(p: (f64, f64), r: &[(f64, f64)]) -> bool {
        let mut inside = false;
        for w in r.windows(2) {
            let (a, b) = (w[0], w[1]);
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
        }
        inside
    }

    /// Random star polygons (small and above the Morton threshold, either
    /// winding) with and without a hole: the triangles cover the polygon's
    /// area (within the encoding's rounding) and each lies inside it.
    #[test]
    fn stars_tessellate_to_their_area() {
        let mut r = Lcg(3);
        for case in 0..400 {
            let n = if case % 3 == 0 { 120 } else { 5 + case % 40 };
            let (cx, cy) = (-170.0 + 340.0 * r.next(), -80.0 + 160.0 * r.next());
            let rad = 0.01 + 5.0 * r.next();
            let outer = star(&mut r, cx, cy, rad, n, case % 2 == 0);
            let holes: Vec<Vec<(f64, f64)>> = if case % 4 == 1 { vec![star(&mut r, cx, cy, rad * 0.2, 6 + case % 5, case % 3 == 0)] } else { vec![] };
            let tris = tessellate(&outer, &holes, true).unwrap_or_else(|e| panic!("case {case}: {e}"));
            let want = ring_area(&outer) - holes.iter().map(|h| ring_area(h)).sum::<f64>();
            let got: f64 = tris.iter().map(tri_area).sum();
            assert!((got - want).abs() <= want * 1e-3 + 1e-9, "case {case}: area {got} vs {want}");
            for t in &tris {
                let c = ((decode_lon(t.ax) + decode_lon(t.bx) + decode_lon(t.cx)) / 3.0, (decode_lat(t.ay) + decode_lat(t.by) + decode_lat(t.cy)) / 3.0);
                if tri_area(t) > want * 1e-3 {
                    assert!(in_ring(c, &outer) && !holes.iter().any(|h| in_ring(c, h)), "case {case}: triangle {t:?} outside");
                }
            }
        }
    }

    /// `middleInsert` casts the ray at the diagonal's midpoint height:
    /// `(x1 - x0) * (y - y0) / (y1 - y0) + x0`. Without the `(y - y0)`
    /// factor a midpoint outside a slanted edge read as inside, so split
    /// diagonals could leave the polygon (or valid ones be refused).
    #[test]
    fn middle_insert_casts_the_ray_at_the_midpoint() {
        let pts = vec![(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (0.0, 0.0)];
        let r = ring(&pts).unwrap();
        let mut t = Tess::default();
        let start = t.create_list(&r, 0, Winding::Cw).unwrap();
        // Midpoint (1, 1.5): inside.
        assert!(t.middle_insert(start, 1.0, 1.0, 1.0, 2.0));
        // Midpoint (2.5, 2): beyond the hypotenuse x + y = 4.
        assert!(!t.middle_insert(start, 2.0, 2.0, 3.0, 2.0));
        // Midpoint (0.5, 3): inside, near the apex.
        assert!(t.middle_insert(start, 0.25, 3.0, 0.75, 3.0));
        // Midpoint (1.5, 3): outside, near the apex.
        assert!(!t.middle_insert(start, 1.0, 3.0, 2.0, 3.0));
    }

    /// The per-thread scratch arena carries nothing over: the same polygon
    /// tessellates the same after other polygons (and failures) on the
    /// thread, and a failure reads the same after successes.
    #[test]
    fn scratch_reuse_is_invisible() {
        let mut r = Lcg(5);
        let big = star(&mut r, 10.0, 10.0, 2.0, 300, true);
        let small = star(&mut r, -20.0, 5.0, 1.0, 12, false);
        let bow = vec![(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];
        let (a, e) = (tessellate(&small, &[], true).unwrap(), tessellate(&bow, &[], true).unwrap_err());
        for _ in 0..3 {
            let _ = tessellate(&big, &[], true).unwrap();
            assert_eq!(tessellate(&small, &[], true).unwrap(), a);
            assert_eq!(tessellate(&bow, &[], true).unwrap_err(), e);
        }
    }

    /// The array-based self-intersection check visits Lucene's pairs in
    /// Lucene's order: the same result and the same first error as the
    /// linked walk, for rings below and above the Morton threshold, with
    /// and without holes, valid and crossing.
    #[test]
    fn array_intersection_check_matches_linked_walk() {
        let mut r = Lcg(17);
        let mut errors = 0;
        for case in 0..300 {
            let n = if case % 2 == 0 { 6 + case % 50 } else { 90 + case % 200 };
            let (cx, cy) = (-100.0 + 200.0 * r.next(), -60.0 + 120.0 * r.next());
            let rad = 1.0 + r.next();
            let mut outer = star(&mut r, cx, cy, rad, n, case % 3 == 0);
            if case % 4 == 1 {
                // Swap two vertices: the ring crosses itself.
                let (i, j) = (1 + (r.next() * (n - 2) as f64) as usize, 1 + (r.next() * (n - 2) as f64) as usize);
                outer.swap(i, j);
            }
            let holes: Vec<Vec<(f64, f64)>> = if case % 5 == 2 { vec![star(&mut r, cx, cy, 0.2, 7, true)] } else { vec![] };
            let poly = ring(&outer).unwrap();
            let hole_rings: Vec<Ring<'_>> = holes.iter().map(|h| ring(h).unwrap()).collect();
            let mut t = Tess::default();
            let Some(mut node) = t.create_list(&poly, 0, Winding::Cw) else { continue };
            if node == t.next(node) || node == t.next(t.next(node)) {
                continue;
            }
            if !hole_rings.is_empty() {
                match t.eliminate_holes(&poly, &hole_rings, node) {
                    Ok(o) => node = o,
                    Err(_) => continue,
                }
            }
            let morton = poly.pts.len() + hole_rings.iter().map(|h| h.pts.len()).sum::<usize>() > VERTEX_THRESHOLD as usize;
            if morton {
                t.sort_by_morton(node);
            }
            t.fill_edge_boxes();
            let linked = t.check_intersection_linked(node, morton);
            let arrays = t.check_intersection(node, morton);
            assert_eq!(arrays, linked, "case {case} morton {morton}");
            errors += linked.is_err() as usize;
        }
        assert!(errors > 10, "{errors}");
    }

    #[test]
    fn lucene_failures() {
        // A bow tie crosses itself.
        let bow = vec![(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];
        assert!(tessellate(&bow, &[], true).unwrap_err().contains("self-intersection"));
        let flat = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0)];
        assert!(tessellate(&flat, &[], true).is_err());
        assert!(tessellate(&[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)], &[], true).unwrap_err().contains("at least 4"));
        assert!(tessellate(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], &[], true).unwrap_err().contains("close itself"));
        // A square with a hole: two triangles at least per side.
        let sq = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)];
        let hole = vec![(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0), (2.0, 2.0)];
        let t = tessellate(&sq, &[hole], true).unwrap();
        assert_eq!(t.len(), 8);
    }

    /// Encoding round trip: the decoded triangle has the same vertex set
    /// and flagged edges, counter-clockwise, and its box is the stored one.
    #[test]
    fn encoding_round_trips() {
        let mut r = Lcg(11);
        for i in 0..20_000 {
            let mut v = |k: u32| -> i32 { if i % 7 == 0 && k == 2 { 5 } else { ((r.next() - 0.5) * 2e9) as i32 } };
            let p = [(v(0), v(1)), (v(2), v(3)), (v(4), v(5))];
            let p = if i % 11 == 0 { [p[0], p[1], p[0]] } else if i % 13 == 0 { [p[0], p[0], p[0]] } else { p };
            let flags = [i % 2 == 0, i % 3 == 0, i % 5 == 0];
            let Ok(e) = encode_triangle(p[0].1, p[0].0, flags[0], p[1].1, p[1].0, flags[1], p[2].1, p[2].0, flags[2]) else { continue };
            let d = decode_triangle(&e);
            let mut want: Vec<(i32, i32)> = p.to_vec();
            want.sort_unstable();
            want.dedup();
            let mut got = vec![(d.ax, d.ay), (d.bx, d.by), (d.cx, d.cy)];
            got.sort_unstable();
            got.dedup();
            assert_eq!(got, want, "{p:?}");
            assert!(orient(d.ax as f64, d.ay as f64, d.bx as f64, d.by as f64, d.cx as f64, d.cy as f64) >= 0);
            let xs = [d.ax, d.bx, d.cx];
            let ys = [d.ay, d.by, d.cy];
            assert_eq!(e[0], *ys.iter().min().unwrap());
            assert_eq!(e[2], *ys.iter().max().unwrap());
            assert_eq!(e[1], *xs.iter().min().unwrap());
            assert_eq!(e[3], *xs.iter().max().unwrap());
            let kind = match want.len() {
                1 => TriKind::Point,
                2 => TriKind::Line,
                _ => if d.kind == TriKind::Line { TriKind::Line } else { TriKind::Triangle },
            };
            assert_eq!(d.kind, kind, "{p:?}");
        }
    }

    #[test]
    fn interleave_matches_lucene() {
        assert_eq!(interleave(0, 0), 0);
        assert_eq!(interleave(1, 0), 1);
        assert_eq!(interleave(0, 1), 2);
        assert_eq!(interleave(-1, 0), 0x5555_5555_5555_5555);
        assert_eq!(interleave(0, -1), 0xAAAA_AAAA_AAAA_AAAA);
        assert_eq!(interleave(3, 5), 0b100111);
    }

    /// Geometries as ES indexes them: points, line segments, envelopes
    /// (crossing the dateline too) and polygons.
    #[test]
    fn geometries_to_triangles() {
        use crate::fieldtypes::shape::Geometry;
        let mut out = Vec::new();
        geometry_triangles(&Geometry::Point((1.0, 2.0)), &mut out).unwrap();
        assert_eq!(decode_triangle(&out[0]).kind, TriKind::Point);
        out.clear();
        geometry_triangles(&Geometry::LineString(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]), &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| decode_triangle(t).kind == TriKind::Line));
        out.clear();
        geometry_triangles(&Geometry::Envelope { min_x: 170.0, max_x: -170.0, max_y: 10.0, min_y: -10.0 }, &mut out).unwrap();
        assert_eq!(out.len(), 4);
        out.clear();
        geometry_triangles(&Geometry::Polygon(vec![vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)]]), &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|t| decode_triangle(t).kind == TriKind::Triangle));
    }
}
