//! H3 (`geohex`) cells for `geohex_grid` and the `geo_grid` query, built on
//! [`crate::h3`], the port of ES's own H3 library (cells and boundaries
//! ES's to the bit).
//!
//! Cell polygons are made planar in `(lon, lat)`: dateline-crossing cells
//! are unwrapped and tested in both copies, cells around a pole are closed
//! through the pole.

use super::geo_shape::BBox;
use crate::h3;

/// An H3 cell index.
pub(crate) type CellIndex = u64;
use rustic_core::fieldtypes::shape::Pt;

/// The H3 cell (as ES prints it) of a point at `res`.
pub(crate) fn point_cell(lat: f64, lon: f64, res: u32) -> Option<String> {
    point_cell_id(lat, lon, res).map(h3::to_string)
}

/// [`point_cell`] as the cell's 64-bit index.
pub(crate) fn point_cell_id(lat: f64, lon: f64, res: u32) -> Option<u64> {
    if !lat.is_finite() || !lon.is_finite() {
        return None;
    }
    h3::geo_to_h3(lat, lon, i32::try_from(res).ok()?)
}

/// The address of a cell index, as [`point_cell`] prints it.
pub(crate) fn cell_address(id: u64) -> Option<String> {
    h3::is_valid(id).then(|| h3::to_string(id))
}

/// Parse an H3 address the way ES does (`Long.parseUnsignedLong(s, 16)`).
pub(crate) fn parse_cell(s: &str) -> Option<CellIndex> {
    let raw = u64::from_str_radix(s, 16).ok()?;
    h3::is_valid(raw).then_some(raw)
}

/// The cell's boundary vertices as `(lon, lat)` degrees.
pub(crate) fn boundary_deg(c: CellIndex) -> Vec<Pt> {
    h3::boundary(c).iter().map(|l| (l.lon_deg(), l.lat_deg())).collect()
}

fn res_of(c: CellIndex) -> u32 {
    h3::resolution(c) as u32
}

/// The 64-bit value of a cell address (bucket ordering).
pub(super) fn cell_long(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap_or(0)
}

fn contains_pole(c: CellIndex) -> Option<f64> {
    let r = h3::resolution(c);
    if h3::north_polar_h3(r) == Some(c) {
        Some(90.0)
    } else if h3::south_polar_h3(r) == Some(c) {
        Some(-90.0)
    } else {
        None
    }
}

/// The cell boundary unwrapped to continuous longitudes, closed; pole cells
/// are closed through the pole. Second value: the longitude span.
fn unwrapped_ring(c: CellIndex) -> Vec<Pt> {
    let b: Vec<Pt> = boundary_deg(c);
    if b.is_empty() {
        return Vec::new();
    }
    let unwrap = |prev: f64, mut x: f64| {
        while x - prev > 180.0 {
            x -= 360.0;
        }
        while x - prev < -180.0 {
            x += 360.0;
        }
        x
    };
    let mut u: Vec<Pt> = Vec::with_capacity(b.len() + 4);
    u.push(b[0]);
    for p in &b[1..] {
        let x = unwrap(u.last().unwrap().0, p.0);
        u.push((x, p.1));
    }
    if let Some(plat) = contains_pole(c) {
        let close = unwrap(u.last().unwrap().0, b[0].0);
        u.push((close, b[0].1));
        u.push((close, plat));
        u.push((b[0].0, plat));
    }
    u.push(u[0]);
    u
}

/// Planar copies of the cell polygon covering its place in `[-180, 180]`.
pub(crate) fn cell_polys(c: CellIndex) -> Vec<Vec<Pt>> {
    let u = unwrapped_ring(c);
    copies(u)
}

/// The cell as ES's `geo_grid` geohex query relates shapes to it
/// (`H3CartesianUtil.getLatLonGeometry`): the boundary on the Lucene grid,
/// closed (polar cells through the pole, `computePoints`); a cell across
/// the dateline, whose longitudes ES normalizes to `[0, 360)`, as its two
/// pieces either side of the dateline.
pub(crate) fn cartesian_polys(c: CellIndex) -> Vec<Vec<Pt>> {
    let (xs, ys) = super::shape_tiler::compute_points(c);
    let ring: Vec<Pt> = xs.iter().copied().zip(ys.iter().copied()).collect();
    if ring.len() < 4 {
        return Vec::new();
    }
    let (min_x, max_x) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &x| (a.0.min(x), a.1.max(x)));
    if contains_pole(c).is_some() || max_x - min_x <= 180.0 {
        return vec![ring];
    }
    let pos: Vec<Pt> = ring.iter().map(|&(x, y)| (if x < 0.0 { x + 360.0 } else { x }, y)).collect();
    let east = clip_x(&pos, 180.0, true);
    let west: Vec<Pt> = clip_x(&pos, 180.0, false).into_iter().map(|(x, y)| (x - 360.0, y)).collect();
    [east, west].into_iter().filter(|r| r.len() >= 4).map(dateline_first).collect()
}

/// The closed ring started, as ES's decomposer starts a piece, at its
/// edge along the dateline (two consecutive vertices at longitude 180 or
/// -180), closed.
fn dateline_first(r: Vec<Pt>) -> Vec<Pt> {
    let open = &r[..r.len() - 1];
    let n = open.len();
    let on = |p: Pt| p.0.abs() == 180.0;
    let Some(i) = (0..n).find(|&i| on(open[i]) && on(open[(i + 1) % n])) else { return r };
    let mut out: Vec<Pt> = (0..n).map(|k| open[(i + k) % n]).collect();
    out.push(out[0]);
    out
}

/// The closed ring `r` clipped to `x <= at` (`keep_below`) or `x >= at`
/// (Sutherland-Hodgman against one vertical line), closed.
fn clip_x(r: &[Pt], at: f64, keep_below: bool) -> Vec<Pt> {
    let inside = |p: Pt| if keep_below { p.0 <= at } else { p.0 >= at };
    let mut out: Vec<Pt> = Vec::new();
    for w in r.windows(2) {
        let (a, b) = (w[0], w[1]);
        // ES's `GeoPolygonDecomposer` intersection: the edge's fraction
        // first.
        let cross = || {
            let t = (at - a.0) / (b.0 - a.0);
            (at, a.1 + t * (b.1 - a.1))
        };
        match (inside(a), inside(b)) {
            (true, true) => out.push(b),
            (true, false) => out.push(cross()),
            (false, true) => {
                out.push(cross());
                out.push(b);
            }
            (false, false) => {}
        }
    }
    if let Some(&f) = out.first() {
        if out.last() != Some(&f) {
            out.push(f);
        }
    }
    out
}

fn copies(u: Vec<Pt>) -> Vec<Vec<Pt>> {
    if u.is_empty() {
        return Vec::new();
    }
    let (min_x, max_x) = u.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let shift = |dx: f64| u.iter().map(|p| (p.0 + dx, p.1)).collect::<Vec<Pt>>();
    let mut out = vec![u.clone()];
    if max_x > 180.0 {
        out.push(shift(-360.0));
    }
    if min_x < -180.0 {
        out.push(shift(360.0));
    }
    out
}

/// `(top, left, bottom, right)` of a cell; `left > right` when it crosses
/// the dateline, the whole longitude range around a pole.
pub(crate) fn hex_bbox(c: CellIndex) -> BBox {
    let b: Vec<Pt> = boundary_deg(c);
    let (mut top, mut bottom) = (f64::NEG_INFINITY, f64::INFINITY);
    for p in &b {
        top = top.max(p.1);
        bottom = bottom.min(p.1);
    }
    if let Some(plat) = contains_pole(c) {
        if plat > 0.0 {
            top = 90.0;
        } else {
            bottom = -90.0;
        }
        return (top, -180.0, bottom, 180.0);
    }
    let (min_x, max_x) = b.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    if max_x - min_x > 180.0 {
        let left = b.iter().filter(|p| p.0 >= 0.0).map(|p| p.0).fold(f64::INFINITY, f64::min);
        let right = b.iter().filter(|p| p.0 < 0.0).map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        return (top, left, bottom, right);
    }
    (top, min_x, bottom, max_x)
}

pub(super) fn hex_bbox_of(key: &str) -> Option<BBox> {
    parse_cell(key).map(hex_bbox)
}

fn hex_margin() -> f64 {
    (1e-4f64).to_radians().sin()
}

/// Unit vector of a `(lat, lon)` in degrees.
fn unit(lat: f64, lon: f64) -> [f64; 3] {
    let (p, l) = (lat.to_radians(), lon.to_radians());
    [p.cos() * l.cos(), p.cos() * l.sin(), p.sin()]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The smallest `n . p` over the points `p` of a lat/lon box (degrees,
/// `min_lon <= max_lon`), for a unit `n`. `n . p = A cos(lat) cos(lon -
/// lon_n) + c sin(lat)`: its minimum over the box lies at a corner, at the
/// minimum along an edge (a parallel: `lon = lon_n + pi`; a meridian: where
/// `R cos(lat - psi)` bottoms out) or at `-n` itself.
pub(crate) fn box_min_dot(n: [f64; 3], min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> f64 {
    use std::f64::consts::PI;
    let (p0, p1, l0, l1) = (min_lat.to_radians(), max_lat.to_radians(), min_lon.to_radians(), max_lon.to_radians());
    let f = |p: f64, l: f64| n[0] * p.cos() * l.cos() + n[1] * p.cos() * l.sin() + n[2] * p.sin();
    let mut m = f(p0, l0).min(f(p0, l1)).min(f(p1, l0)).min(f(p1, l1));
    let ln = n[1].atan2(n[0]);
    // Parallels: the far side of the normal's meridian.
    for k in [-1.0, 0.0, 1.0] {
        let l = ln + PI + 2.0 * PI * k;
        if l >= l0 && l <= l1 {
            m = m.min(f(p0, l)).min(f(p1, l));
        }
    }
    // Meridians.
    for l in [l0, l1] {
        let kk = (n[0] * n[0] + n[1] * n[1]).sqrt() * (l - ln).cos();
        let psi = n[2].atan2(kk);
        for k in [-1.0, 0.0, 1.0] {
            let p = psi + PI + 2.0 * PI * k;
            if p >= p0 && p <= p1 {
                m = m.min(f(p, l));
            }
        }
    }
    // The antipode of `n`, the global minimum.
    let (ap, al) = ((-n[2]).clamp(-1.0, 1.0).asin(), (-n[1]).atan2(-n[0]));
    if ap >= p0 && ap <= p1 && al >= l0 && al <= l1 {
        m = m.min(-1.0);
    }
    m
}

/// Points whose H3 cell at the cell's resolution is that cell (ES's
/// `geo_grid` geohex query on `geo_point`: `H3SphericalGeometry`, whose
/// `contains` is `geoToH3(lat, lon, res) == h3`).
///
/// Leaf boxes are related to the cell for the KD walk: outside the cell's
/// candidate box (its box grown by a tenth of its height), or wholly
/// inside the cell by more than ES's `1e-4` degree buffer, measured
/// against the great circles through consecutive boundary vertices (H3
/// edges are straight in each face's gnomonic projection, i.e. great
/// circles). Only points of the other boxes compute their cell. A box is
/// claimed inside only when every edge's half-space contains it, which
/// implies it is inside the cell even for cells bent across an icosahedron
/// edge; cells around a pole are never claimed.
#[derive(Debug)]
pub(crate) struct HexCellPred {
    cell: u64,
    res: u32,
    /// The candidate box (`left > right`: crosses the dateline).
    bbox: rustic_core::geo::GeoShape,
    /// Inward unit normals of the edges' great circles (empty: never inside).
    normals: Vec<[f64; 3]>,
    margin: f64,
    /// Inward unit normals of the edges of the vertices' convex hull
    /// (empty: never claimed outside). The cell lies inside the hull
    /// (convex or not, e.g. bent across an icosahedron edge), so a box
    /// beyond one hull edge is outside the cell.
    hull: Vec<[f64; 3]>,
}

/// The latitude range of the cell's boundary: its vertices and, along
/// each great-circle edge, the highest / lowest point when it lies inside
/// the edge (edges bulge poleward of their end points).
fn lat_range(verts: &[[f64; 3]]) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (i, a) in verts.iter().enumerate() {
        let b = verts[(i + 1) % verts.len()];
        lo = lo.min(a[2].clamp(-1.0, 1.0).asin());
        hi = hi.max(a[2].clamp(-1.0, 1.0).asin());
        let n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let len = dot(n, n).sqrt();
        if len < 1e-15 {
            continue;
        }
        let n = [n[0] / len, n[1] / len, n[2] / len];
        for sign in [1.0, -1.0] {
            // The great circle's point nearest the pole `sign * z`.
            let p = [-n[2] * n[0] * sign, -n[2] * n[1] * sign, sign * (1.0 - n[2] * n[2])];
            let pl = dot(p, p).sqrt();
            if pl < 1e-15 {
                continue;
            }
            let p = [p[0] / pl, p[1] / pl, p[2] / pl];
            let cross = |u: [f64; 3], v: [f64; 3]| [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            if dot(cross(*a, p), n) >= 0.0 && dot(cross(p, b), n) >= 0.0 {
                let lat = p[2].clamp(-1.0, 1.0).asin();
                lo = lo.min(lat);
                hi = hi.max(lat);
            }
        }
    }
    (lo.to_degrees(), hi.to_degrees())
}

/// A box holding every point of the cell, grown by a tenth of its height:
/// the latitudes of [`lat_range`] (to the pole for a cell around it), and
/// every longitude for cells reaching past 75 degrees of latitude (near a
/// pole a cell's vertices may span more than half the longitudes, which
/// the dateline test of [`hex_bbox`] would misread); otherwise the
/// vertices' longitudes: along a great-circle arc that passes over no pole
/// the longitude is monotonic, so the edges stay between their vertices'.
fn candidate_box(c: CellIndex, verts: &[[f64; 3]]) -> rustic_core::geo::GeoShape {
    let (top, left, bottom, right) = hex_bbox(c);
    let (lo, hi) = if verts.is_empty() { (bottom, top) } else { lat_range(verts) };
    let (mut bottom, mut top) = (bottom.min(lo), top.max(hi));
    match contains_pole(c) {
        Some(p) if p > 0.0 => top = 90.0,
        Some(_) => bottom = -90.0,
        None => {}
    }
    let m = ((top - bottom) * 0.1).max(1e-6);
    let crosses = left > right;
    let full = contains_pole(c).is_some() || top > 75.0 || bottom < -75.0 || (!crosses && right - left >= 359.0);
    let (l, r) = if full { (-180.0, 180.0) } else { (left - m, right + m) };
    let l = if l < -180.0 { l + 360.0 } else { l };
    let r = if r > 180.0 { r - 360.0 } else { r };
    // Grown past a full turn: every longitude.
    let (l, r) = if !full && (right - left + 2.0 * m >= 360.0 || (crosses && 360.0 - left + right + 2.0 * m >= 360.0)) { (-180.0, 180.0) } else { (l, r) };
    rustic_core::geo::GeoShape::BoundingBox { top: (top + m).min(90.0), left: l, bottom: (bottom - m).max(-90.0), right: r }
}

/// Half-diagonal (radians, about 1.1 degrees) below which a box that fails
/// the cheap inside bound is not given the exact test (grid cells of the
/// point predicate: thousands per query).
const SMALL_BOX: f64 = 0.02;

impl HexCellPred {
    pub(crate) fn new(c: CellIndex) -> HexCellPred {
        let verts: Vec<[f64; 3]> = h3::boundary(c).iter().map(|v| unit(v.lat_deg(), v.lon_deg())).collect();
        let bbox = candidate_box(c, &verts);
        let mut normals = Vec::new();
        // Leaves are claimed inside only for cells away from the poles and
        // finer than the base cells (anything else is tested point by
        // point: never a member dropped, only time spent).
        let (lat_lo, lat_hi) = lat_range(&verts);
        if contains_pole(c).is_none() && verts.len() >= 3 && res_of(c) > 0 && lat_hi < 80.0 && lat_lo > -80.0 {
            let mut center = [0.0; 3];
            for v in &verts {
                center = [center[0] + v[0], center[1] + v[1], center[2] + v[2]];
            }
            for i in 0..verts.len() {
                let (a, b) = (verts[i], verts[(i + 1) % verts.len()]);
                let n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                let len = dot(n, n).sqrt();
                if len > 0.0 {
                    normals.push([n[0] / len, n[1] / len, n[2] / len]);
                }
            }
            // One winding for every edge (a reflex edge of a bent cell
            // keeps its own side).
            let s: f64 = normals.iter().map(|n| dot(*n, center)).sum();
            if s < 0.0 {
                normals.iter_mut().for_each(|n| *n = [-n[0], -n[1], -n[2]]);
            }
            if normals.len() < 3 {
                normals.clear();
            }
        }
        // Hull edges: great circles through two vertices with every other
        // vertex on one side.
        let mut hull = Vec::new();
        if !normals.is_empty() {
            for i in 0..verts.len() {
                for j in i + 1..verts.len() {
                    let (a, b) = (verts[i], verts[j]);
                    let n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                    let len = dot(n, n).sqrt();
                    if len < 1e-9 {
                        continue;
                    }
                    let n = [n[0] / len, n[1] / len, n[2] / len];
                    let others = || verts.iter().enumerate().filter(|(k, _)| *k != i && *k != j).map(|(_, v)| dot(n, *v));
                    if others().all(|d| d >= -1e-12) {
                        hull.push(n);
                    } else if others().all(|d| d <= 1e-12) {
                        hull.push([-n[0], -n[1], -n[2]]);
                    }
                }
            }
        }
        HexCellPred { cell: c, res: res_of(c), bbox, normals, margin: hex_margin(), hull }
    }

    /// Whether the point may be in the cell (inside its candidate box,
    /// which holds every point of the cell).
    pub(crate) fn may_contain(&self, lat: f64, lon: f64) -> bool {
        self.bbox.contains(lat, lon)
    }

    /// The candidate box as `(min_lat, max_lat, min_lon, max_lon)` when it
    /// does not cross the dateline.
    pub(crate) fn plain_bbox(&self) -> Option<(f64, f64, f64, f64)> {
        self.bbox.plain_bbox().map(|b| (b.0.max(-90.0), b.1.min(90.0), b.2.max(-180.0), b.3.min(180.0)))
    }

    /// Whether every point of the box is in the cell (conservative).
    ///
    /// First a cheap bound: every point of the box lies within the angle
    /// `rho = sqrt(dlat^2 + dlon^2) / 2` of its centre `c` (the metric is
    /// `dphi^2 + cos^2(phi) dlambda^2 <= dphi^2 + dlambda^2`), and `n . p`
    /// moves by at most the chord, at most that angle, so `n . c > margin +
    /// rho` for every edge proves the box inside. A small box that fails it
    /// is called crossing (its points are then tested one by one); only
    /// larger boxes (KD blocks) pay the exact minimum over the box.
    pub(crate) fn box_inside(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> bool {
        if self.normals.is_empty() || min_lon > max_lon {
            return false;
        }
        let (dlat, dlon) = ((max_lat - min_lat).to_radians() / 2.0, (max_lon - min_lon).to_radians() / 2.0);
        let rho = (dlat * dlat + dlon * dlon).sqrt();
        let c = unit((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0);
        // Rounding of the unit vector and dot product: far below the margin.
        let slack = self.margin + rho + 1e-12;
        if self.normals.iter().all(|n| dot(*n, c) > slack) {
            return true;
        }
        if rho < SMALL_BOX {
            return false;
        }
        self.normals.iter().all(|n| box_min_dot(*n, min_lat, max_lat, min_lon, max_lon) > self.margin)
    }
}

impl HexCellPred {
    /// Whether the point is in the cell by more than the margin (inside
    /// every edge's great circle: [`HexCellPred::box_inside`] of the point
    /// alone), so its H3 cell is this one; `false` says nothing.
    #[inline]
    pub(crate) fn point_inside(&self, lat: f64, lon: f64) -> bool {
        !self.normals.is_empty() && self.unit_inside(unit(lat, lon))
    }

    /// [`HexCellPred::point_inside`] of the point with unit vector `c`
    /// ([`unit_vector`]).
    #[inline]
    pub(crate) fn unit_inside(&self, c: [f64; 3]) -> bool {
        let slack = self.margin + 1e-12;
        !self.normals.is_empty() && self.normals.iter().all(|n| dot(*n, c) > slack)
    }
}

/// The unit vector of a `(lat, lon)` in degrees, as [`HexCellPred`]
/// tests it.
#[inline]
pub(crate) fn unit_vector(lat: f64, lon: f64) -> [f64; 3] {
    unit(lat, lon)
}

impl HexCellPred {
}

impl HexCellPred {
    /// Whether every point of the box is outside the cell by more than the
    /// margin: beyond one edge of the vertices' convex hull. Like
    /// [`HexCellPred::box_inside`], the cheap center bound first, the
    /// exact maximum only for larger boxes.
    pub(crate) fn box_outside(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> bool {
        if self.hull.is_empty() || min_lon > max_lon {
            return false;
        }
        let (dlat, dlon) = ((max_lat - min_lat).to_radians() / 2.0, (max_lon - min_lon).to_radians() / 2.0);
        let rho = (dlat * dlat + dlon * dlon).sqrt();
        let c = unit((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0);
        let slack = self.margin + rho + 1e-12;
        if self.hull.iter().any(|n| dot(*n, c) < -slack) {
            return true;
        }
        if rho < SMALL_BOX {
            return false;
        }
        // max over the box of n·p = -(min of (-n)·p).
        self.hull.iter().any(|n| -box_min_dot([-n[0], -n[1], -n[2]], min_lat, max_lat, min_lon, max_lon) < -self.margin)
    }
}

impl rustic_core::geo::PointPredicate for HexCellPred {
    fn classify(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> rustic_core::geo::Relation {
        use rustic_core::geo::Relation;
        // Boxes in the candidate box but outside the hexagon used to be
        // "crosses": every point in them ran the exact H3 test.
        if self.bbox.relate(min_lat, max_lat, min_lon, max_lon) == Relation::Outside || self.box_outside(min_lat, max_lat, min_lon, max_lon) {
            Relation::Outside
        } else if self.box_inside(min_lat, max_lat, min_lon, max_lon) {
            Relation::Inside
        } else {
            Relation::Crosses
        }
    }

    // The exact membership alone: the candidate box only prunes leaves.
    fn test(&self, lat: f64, lon: f64) -> bool {
        #[cfg(test)]
        H3_TESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        point_cell_id(lat, lon, self.res) == Some(self.cell)
    }
}

/// Exact H3 membership tests run (test builds; the microbench reports it).
#[cfg(test)]
pub(crate) static H3_TESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
mod hex_pred_tests {
    use super::*;
    use rustic_core::geo::{PointPredicate, Relation};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `box_min_dot` is the minimum over the box: never above a sampled
    /// point's value, and reached (within rounding) by dense sampling.
    #[test]
    fn box_min_dot_is_the_minimum() {
        let mut r = Lcg(5);
        for _ in 0..3000 {
            let n = unit(-90.0 + 180.0 * r.next(), -180.0 + 360.0 * r.next());
            let (a, b) = (-90.0 + 180.0 * r.next(), -90.0 + 180.0 * r.next());
            let (c, d) = (-180.0 + 360.0 * r.next(), -180.0 + 360.0 * r.next());
            let scale = if r.next() < 0.5 { 1.0 } else { 0.01 };
            let (lat0, lat1) = (a.min(b), a.min(b) + (a - b).abs() * scale);
            let (lon0, lon1) = (c.min(d), c.min(d) + (c - d).abs() * scale);
            let m = box_min_dot(n, lat0, lat1, lon0, lon1);
            let mut sampled = f64::INFINITY;
            for i in 0..=40 {
                for j in 0..=40 {
                    let p = unit(lat0 + (lat1 - lat0) * i as f64 / 40.0, lon0 + (lon1 - lon0) * j as f64 / 40.0);
                    sampled = sampled.min(dot(n, p));
                }
            }
            assert!(m <= sampled + 1e-12, "{n:?} {lat0} {lat1} {lon0} {lon1}: {m} > {sampled}");
            // Tight up to the sampling step (a wide box can hold the exact
            // minimum, e.g. the antipode, between samples).
            if scale < 1.0 {
                assert!(sampled - m < 2e-3, "{n:?} {lat0} {lat1} {lon0} {lon1}: {m} far below {sampled}");
            } else {
                let mut fine = f64::INFINITY;
                for i in 0..=200 {
                    for j in 0..=200 {
                        let p = unit(lat0 + (lat1 - lat0) * i as f64 / 200.0, lon0 + (lon1 - lon0) * j as f64 / 200.0);
                        fine = fine.min(dot(n, p));
                    }
                }
                assert!(m <= fine + 1e-12 && fine - m < 2e-3, "{n:?} {lat0} {lat1} {lon0} {lon1}: {m} vs {fine}");
            }
        }
    }

    /// Cells at every resolution 0-8 (random points, pentagons, cells on
    /// the dateline, at the poles and on icosahedron edges): a box the
    /// predicate calls inside has all its (sampled) points in the cell, an
    /// outside box none, and the per-point test is the cell membership.
    #[test]
    fn inside_boxes_hold_only_cell_points() {
        let mut r = Lcg(9);
        let mut cells: Vec<CellIndex> = Vec::new();
        for res in 0..=8u8 {
            for _ in 0..12 {
                let (lat, lon) = (-90.0 + 180.0 * r.next(), -180.0 + 360.0 * r.next());
                cells.push(point_cell_id(lat, lon, res as u32).unwrap());
            }
            for (lat, lon) in [(0.0, 180.0), (0.0, -179.99), (89.99, 0.0), (-89.99, 10.0), (55.0, 7.0), (48.85, 2.3), (10.536199, 64.7)] {
                cells.push(point_cell_id(lat, lon, res as u32).unwrap());
            }
            cells.extend(h3::res0_cells().filter(|c| h3::is_pentagon(*c)).filter_map(|c| h3::center_child(c, res as i32)));
        }
        cells.push(parse_cell("811fbffffffffff").unwrap());
        for c in cells {
            let pred = HexCellPred::new(c);
            let res = res_of(c);
            let (top, left, bottom, right) = hex_bbox(c);
            let span_lat = (top - bottom).max(1e-6);
            let span_lon = if left <= right { (right - left).max(1e-6) } else { 360.0 - left + right };
            let mut inside = 0;
            for _ in 0..300 {
                let lat0 = (bottom - 0.2 * span_lat + 1.4 * span_lat * r.next()).clamp(-90.0, 90.0);
                let lon0 = left - 0.2 * span_lon + 1.4 * span_lon * r.next();
                let lon0 = (if lon0 > 180.0 { lon0 - 360.0 } else { lon0 }).max(-180.0);
                let h = span_lat * r.next() * 0.3;
                let w = span_lon * r.next() * 0.3;
                let (lat1, lon1) = ((lat0 + h).min(90.0), (lon0 + w).min(180.0));
                let rel = pred.classify(lat0, lat1, lon0, lon1);
                if rel == Relation::Inside {
                    inside += 1;
                }
                for i in 0..=8 {
                    for j in 0..=8 {
                        let (lat, lon) = (lat0 + (lat1 - lat0) * i as f64 / 8.0, lon0 + (lon1 - lon0) * j as f64 / 8.0);
                        let (lat, lon) = rustic_core::geo::quantize(lat, lon);
                        let in_cell = point_cell_id(lat, lon, res) == Some(c);
                        assert_eq!(pred.test(lat, lon), in_cell, "{c:x} {lat} {lon}");
                        match rel {
                            Relation::Inside => assert!(in_cell, "{c:x}: box {lat0} {lat1} {lon0} {lon1} called inside, {lat} {lon} is not in the cell"),
                            Relation::Outside => assert!(!in_cell, "{c:x}: box {lat0} {lat1} {lon0} {lon1} called outside, {lat} {lon} is in the cell"),
                            Relation::Crosses => {}
                        }
                    }
                }
            }
            // Small boxes around the cell's center are inside (the walk
            // then takes whole leaves), except for base cells and cells
            // reaching 80 degrees of latitude (always tested per point).
            let verts: Vec<[f64; 3]> = h3::boundary(c).iter().map(|v| unit(v.lat_deg(), v.lon_deg())).collect();
            let (lo, hi) = lat_range(&verts);
            if contains_pole(c).is_none() && res_of(c) > 0 && hi < 80.0 && lo > -80.0 {
                let ll = h3::to_lat_lng(c);
                let e = span_lat * 0.01;
                assert_eq!(pred.classify(ll.lat_deg() - e, ll.lat_deg() + e, ll.lon_deg() - e, ll.lon_deg() + e), Relation::Inside, "{c:x}");
            }
            let _ = inside;
        }
    }

    /// The cells of earlier failures: every point of a fine
    /// grid over the whole globe that is in the cell passes the test and
    /// lies in the candidate box; the pole-near and southern base cells.
    #[test]
    fn candidate_box_holds_every_cell_point() {
        for addr in ["8003fffffffffff", "80f1fffffffffff", "8001fffffffffff", "811fbffffffffff", "80effffffffffff", "8073fffffffffff"] {
            let c = parse_cell(addr).unwrap();
            let pred = HexCellPred::new(c);
            let res = res_of(c);
            let mut members = 0;
            for i in 0..=360 {
                for j in 0..=720 {
                    let (lat, lon) = rustic_core::geo::quantize(-90.0 + i as f64 * 0.5, -180.0 + j as f64 * 0.5);
                    if point_cell_id(lat, lon, res) == Some(c) {
                        members += 1;
                        assert!(pred.test(lat, lon), "{addr} {lat} {lon}");
                        assert!(pred.bbox.contains(lat, lon), "{addr} {lat} {lon} outside the candidate box");
                        assert_ne!(pred.classify(lat, lat, lon, lon), Relation::Outside, "{addr} {lat} {lon}");
                    }
                }
            }
            assert!(members > 0, "{addr}");
        }
        assert!(box_min_dot([0.2328, -0.9490, -0.2129], -2.929, 52.03, -174.20, 168.03) <= -0.9999);
        // The Rally cell (western Europe) keeps a box of its own longitudes,
        // and leaves inside it are taken whole.
        let rally = HexCellPred::new(parse_cell("811fbffffffffff").unwrap());
        let b = rally.plain_bbox().expect("no dateline");
        assert!(b.3 - b.2 < 60.0, "{b:?}");
        assert!(!rally.normals.is_empty());
    }

    /// Boxes the cheap bound calls inside hold only cell points: random
    /// boxes from a hundredth to a fifth of a degree (the predicate grid's
    /// cell sizes) near the Rally cell's edges and centre, sampled.
    #[test]
    fn cheap_inside_bound_is_conservative() {
        let mut r = Lcg(77);
        let c = parse_cell("811fbffffffffff").unwrap();
        let pred = HexCellPred::new(c);
        let mut inside = 0;
        for _ in 0..20_000 {
            let (lat0, lon0) = (43.5 + 9.0 * r.next(), 0.8 + 12.5 * r.next());
            let s = 0.01 + 0.19 * r.next();
            let (lat1, lon1) = (lat0 + s * r.next(), lon0 + s * r.next());
            if !pred.box_inside(lat0, lat1, lon0, lon1) {
                continue;
            }
            inside += 1;
            for i in 0..=6 {
                for j in 0..=6 {
                    let (lat, lon) = rustic_core::geo::quantize(lat0 + (lat1 - lat0) * i as f64 / 6.0, lon0 + (lon1 - lon0) * j as f64 / 6.0);
                    assert_eq!(point_cell_id(lat, lon, 1), Some(c), "{lat0} {lat1} {lon0} {lon1}: {lat} {lon}");
                }
            }
        }
        assert!(inside > 5000, "{inside}");
    }

    /// Boxes claimed outside a cell hold no point of it: the Rally cell
    /// and cells of several resolutions and places, boxes of all sizes
    /// around them (corners, edges and inner samples tested with H3).
    #[test]
    fn outside_claims_are_conservative() {
        let mut r = Lcg(31);
        let mut cells = vec![parse_cell("811fbffffffffff").unwrap()];
        for (lat, lon, res) in [(48.0, 7.0, 2u32), (10.0, -60.0, 3), (-33.0, 151.0, 1), (60.0, 100.0, 4), (0.5, 0.5, 2), (40.0, -100.0, 5)] {
            cells.push(point_cell_id(lat, lon, res).unwrap());
        }
        let (mut outside, mut convex) = (0, 0u32);
        for c in cells {
            let pred = HexCellPred::new(c);
            convex += u32::from(!pred.hull.is_empty());
            let Some(b) = pred.plain_bbox() else { continue };
            let (h, w) = (b.1 - b.0, b.3 - b.2);
            for _ in 0..6000 {
                let (lat0, lon0) = (b.0 - 0.2 * h + 1.4 * h * r.next(), b.2 - 0.2 * w + 1.4 * w * r.next());
                let s = r.next() * r.next() * 0.5;
                let (lat1, lon1) = ((lat0 + s * h).min(90.0), (lon0 + s * w).min(180.0));
                if !pred.box_outside(lat0, lat1, lon0, lon1) {
                    continue;
                }
                outside += 1;
                assert!(!pred.box_inside(lat0, lat1, lon0, lon1));
                for i in 0..=6 {
                    for j in 0..=6 {
                        let (lat, lon) = rustic_core::geo::quantize(lat0 + (lat1 - lat0) * i as f64 / 6.0, lon0 + (lon1 - lon0) * j as f64 / 6.0);
                        assert_ne!(point_cell_id(lat, lon, pred.res), Some(c), "{c:x} {lat0} {lat1} {lon0} {lon1}: {lat} {lon}");
                    }
                }
            }
        }
        assert!(convex >= 4 && outside > 3000, "{convex} {outside}");
    }

}
