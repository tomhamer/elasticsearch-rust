//! The `fields` format `mvt(z/x/y[@extent][:buffer])` of `geo_shape`
//! fields: each hit's geometry as Mapbox vector tile features of that tile
//! (ES x-pack `vector-tile` `SpatialGeometryFormatterExtension` and
//! `FeatureFactory`, encoded as `PatchedJtsAdapter` does), each feature's
//! protobuf bytes rendered base64 as ES renders a `byte[]`.
//!
//! The geometry is projected to spherical Mercator, dropped when it misses
//! the tile grown by `buffer` pixels, clipped to that box when it is not
//! inside it, simplified with a tolerance of one pixel, snapped to the
//! tile's `extent` grid (y down) and encoded with ES's rules: per feature
//! the cursor starts at 0, repeated points (as ints) are skipped,
//! exterior rings are made clockwise and holes counter-clockwise in tile
//! space, rings of zero area are dropped, and a polygon with a hole at
//! least as large as its shell is dropped. Collections give one feature
//! per member (in ES's order, the members of a collection reversed); a
//! multi-geometry is one feature.
//!
//! Lines are clipped segment by segment (Liang-Barsky), polygon rings with
//! Sutherland-Hodgman against the box, and lines and rings are simplified
//! with Douglas-Peucker (rings never below four points).

use rustic_core::fieldtypes::shape::Geometry;

type P = (f64, f64);

/// `z/x/y`, extent and buffer pixels of an `mvt(...)` format (ES
/// `SimpleVectorTileFormatter.parse`); `None` when it does not parse.
pub fn parse_spec(format: &str) -> Option<(u32, u32, u32, u32, u32)> {
    let inner = format.strip_prefix("mvt(")?.strip_suffix(')')?;
    let (zxy, rest) = match inner.find(['@', ':']) {
        Some(i) => (&inner[..i], &inner[i..]),
        None => (inner, ""),
    };
    let mut it = zxy.split('/');
    let num = |s: Option<&str>| -> Option<u32> { s.filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))?.parse().ok() };
    let (z, x, y) = (num(it.next())?, num(it.next())?, num(it.next())?);
    if it.next().is_some() || z > 29 {
        return None;
    }
    let tiles = 1u64 << z;
    if x as u64 >= tiles || y as u64 >= tiles {
        return None;
    }
    let (mut extent, mut buffer) = (4096u32, 5u32);
    let (ext, buf) = match rest.split_once(':') {
        Some((a, b)) => (a, Some(b)),
        None => (rest, None),
    };
    if let Some(e) = ext.strip_prefix('@') {
        extent = num(Some(e))?;
    } else if !ext.is_empty() {
        return None;
    }
    if let Some(b) = buf {
        buffer = num(Some(b))?;
    }
    if extent == 0 {
        return None;
    }
    Some((z, x, y, extent, buffer))
}

/// A geometry in Mercator meters, then in tile pixels.
#[derive(Clone, Debug)]
enum G {
    Point(P),
    MultiPoint(Vec<P>),
    Line(Vec<P>),
    MultiLine(Vec<Vec<P>>),
    Polygon(Vec<Vec<P>>),
    MultiPolygon(Vec<Vec<Vec<P>>>),
    Collection(Vec<G>),
}

/// The tile's projection, clip box and pixel grid.
struct Tile {
    /// Clip box `(min_x, min_y, max_x, max_y)` in Mercator meters.
    clip: (f64, f64, f64, f64),
    pixel: f64,
    x_scale: f64,
    y_scale: f64,
    x_translate: f64,
    y_translate: f64,
    extent: i64,
}

impl Tile {
    fn new(z: u32, x: u32, y: u32, extent: u32, buffer: u32) -> Tile {
        use crate::aggs::geo_shape::{lat_mercator, lon_mercator};
        const MERCATOR_BOUNDS: f64 = 20037508.34;
        let pixel = 2.0 * MERCATOR_BOUNDS / (((1u64 << z) as f64) * extent as f64);
        let (top, left, bottom, right) = super::geo::tile_bbox(z, x, y);
        let (min_x, max_x, max_y, min_y) = (lon_mercator(left), lon_mercator(right), lat_mercator(top), lat_mercator(bottom));
        let pad = buffer as f64 * pixel;
        let x_scale = extent as f64 / (max_x - min_x);
        let y_scale = -(extent as f64) / (max_y - min_y);
        Tile { clip: (min_x - pad, min_y - pad, max_x + pad, max_y + pad), pixel, x_scale, y_scale, x_translate: -x_scale * min_x, y_translate: -y_scale * min_y, extent: extent as i64 }
    }

    /// Java `(int) Math.round(v)`.
    fn round(v: f64) -> f64 {
        (v + 0.5).floor()
    }

    fn snap(&self, p: P) -> P {
        (Self::round(self.x_scale * p.0 + self.x_translate), Self::round(self.y_scale * p.1 + self.y_translate) + self.extent as f64)
    }
}

/// The features of a doc's geometry (one value, or a collection of its
/// values) as base64 strings; empty when nothing of it is in the tile.
pub fn features(g: &Geometry, spec: (u32, u32, u32, u32, u32)) -> Vec<String> {
    use base64::Engine;
    let t = Tile::new(spec.0, spec.1, spec.2, spec.3, spec.4);
    let Some(m) = build(&t, g) else { return Vec::new() };
    let mut flat = Vec::new();
    flatten(m, &mut flat);
    flat.iter().filter_map(encode_feature).map(|b| base64::engine::general_purpose::STANDARD.encode(b)).collect()
}

fn merc(p: P) -> P {
    use crate::aggs::geo_shape::{lat_mercator, lon_mercator};
    (lon_mercator(p.0), lat_mercator(p.1))
}

/// ES `MVTGeometryBuilder`: each simple geometry projected, clipped,
/// simplified and snapped; multi-geometries and collections built from
/// their members' results as JTS `buildGeometry` builds them.
fn build(t: &Tile, g: &Geometry) -> Option<G> {
    match g {
        Geometry::Point(p) => finish(t, G::Point(merc(*p))),
        Geometry::MultiPoint(ps) => {
            let mut v: Vec<P> = ps.iter().map(|p| merc(*p)).collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            finish(t, G::MultiPoint(v))
        }
        Geometry::LineString(l) => finish(t, G::Line(l.iter().map(|p| merc(*p)).collect())),
        Geometry::Polygon(r) => finish(t, G::Polygon(r.iter().map(|ring| ring.iter().map(|p| merc(*p)).collect()).collect())),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => {
            let (x0, x1, y0, y1) = (merc((*min_x, *min_y)).0, merc((*max_x, *max_y)).0, merc((*min_x, *min_y)).1, merc((*max_x, *max_y)).1);
            let rect = |a: f64, b: f64| G::Polygon(vec![vec![(a, y0), (a, y1), (b, y1), (b, y0), (a, y0)]]);
            if min_x > max_x {
                const MB: f64 = 20037508.34;
                let parts: Vec<G> = [rect(-MB, x1), rect(x0, MB)].into_iter().filter_map(|r| finish(t, r)).collect();
                collect(parts)
            } else {
                finish(t, rect(x0, x1))
            }
        }
        Geometry::MultiLineString(ls) => collect(ls.iter().filter_map(|l| build(t, &Geometry::LineString(l.clone()))).collect()),
        Geometry::MultiPolygon(ps) => collect(ps.iter().filter_map(|p| build(t, &Geometry::Polygon(p.clone()))).collect()),
        Geometry::Collection(gs) => collect(gs.iter().filter_map(|x| build(t, x)).collect()),
    }
}

/// JTS `GeometryFactory.buildGeometry` over the members of `parts`: one
/// geometry as is, members of one kind as its multi-geometry, else a
/// collection; `None` for nothing.
fn collect(parts: Vec<G>) -> Option<G> {
    let mut members: Vec<G> = Vec::new();
    for p in parts {
        match p {
            G::MultiPoint(v) => members.extend(v.into_iter().map(G::Point)),
            G::MultiLine(v) => members.extend(v.into_iter().map(G::Line)),
            G::MultiPolygon(v) => members.extend(v.into_iter().map(G::Polygon)),
            other => members.push(other),
        }
    }
    if members.is_empty() {
        return Some(G::Collection(Vec::new()));
    }
    if members.len() == 1 {
        return members.pop();
    }
    if members.iter().all(|m| matches!(m, G::Point(_))) {
        return Some(G::MultiPoint(members.into_iter().map(|m| if let G::Point(p) = m { p } else { unreachable!() }).collect()));
    }
    if members.iter().all(|m| matches!(m, G::Line(_))) {
        return Some(G::MultiLine(members.into_iter().map(|m| if let G::Line(l) = m { l } else { unreachable!() }).collect()));
    }
    if members.iter().all(|m| matches!(m, G::Polygon(_))) {
        return Some(G::MultiPolygon(members.into_iter().map(|m| if let G::Polygon(p) = m { p } else { unreachable!() }).collect()));
    }
    Some(G::Collection(members))
}

/// ES `toMVTGeometry`: clip, simplify, snap.
fn finish(t: &Tile, g: G) -> Option<G> {
    let g = clip(t, g)?;
    let g = simplify(g, t.pixel);
    Some(map_points(g, &|p| t.snap(p)))
}

fn env_of(g: &G) -> Option<(f64, f64, f64, f64)> {
    let mut e: Option<(f64, f64, f64, f64)> = None;
    let mut add = |p: &P| {
        e = Some(match e {
            None => (p.0, p.1, p.0, p.1),
            Some(b) => (b.0.min(p.0), b.1.min(p.1), b.2.max(p.0), b.3.max(p.1)),
        })
    };
    each_point(g, &mut add);
    e
}

fn each_point(g: &G, f: &mut dyn FnMut(&P)) {
    match g {
        G::Point(p) => f(p),
        G::MultiPoint(v) | G::Line(v) => v.iter().for_each(|p| f(p)),
        G::MultiLine(v) | G::Polygon(v) => v.iter().flatten().for_each(|p| f(p)),
        G::MultiPolygon(v) => v.iter().flatten().flatten().for_each(|p| f(p)),
        G::Collection(v) => v.iter().for_each(|x| each_point(x, f)),
    }
}

fn map_points(g: G, f: &dyn Fn(P) -> P) -> G {
    let line = |v: Vec<P>| v.into_iter().map(f).collect::<Vec<P>>();
    match g {
        G::Point(p) => G::Point(f(p)),
        G::MultiPoint(v) => G::MultiPoint(line(v)),
        G::Line(v) => G::Line(line(v)),
        G::MultiLine(v) => G::MultiLine(v.into_iter().map(line).collect()),
        G::Polygon(v) => G::Polygon(v.into_iter().map(line).collect()),
        G::MultiPolygon(v) => G::MultiPolygon(v.into_iter().map(|p| p.into_iter().map(line).collect()).collect()),
        G::Collection(v) => G::Collection(v.into_iter().map(|x| map_points(x, f)).collect()),
    }
}

/// ES `clipGeometry` against the buffered tile box.
fn clip(t: &Tile, g: G) -> Option<G> {
    let (x0, y0, x1, y1) = t.clip;
    let e = env_of(&g)?;
    if e.2 < x0 || e.0 > x1 || e.3 < y0 || e.1 > y1 {
        return None;
    }
    if e.0 >= x0 && e.2 <= x1 && e.1 >= y0 && e.3 <= y1 {
        return Some(g);
    }
    let inside = |p: &P| p.0 >= x0 && p.0 <= x1 && p.1 >= y0 && p.1 <= y1;
    match g {
        G::Point(p) => inside(&p).then_some(G::Point(p)),
        G::MultiPoint(v) => {
            let v: Vec<P> = v.into_iter().filter(inside).collect();
            match v.len() {
                0 => None,
                1 => Some(G::Point(v[0])),
                _ => Some(G::MultiPoint(v)),
            }
        }
        G::Line(l) => {
            let pieces = clip_line(&l, t.clip);
            match pieces.len() {
                0 => None,
                1 => pieces.into_iter().next().map(G::Line),
                _ => Some(G::MultiLine(pieces)),
            }
        }
        G::Polygon(rings) => {
            let mut out: Vec<Vec<P>> = Vec::new();
            for (i, r) in rings.iter().enumerate() {
                let c = clip_ring(r, t.clip);
                if c.len() < 4 || ring_area(&c) == 0.0 {
                    if i == 0 {
                        return None;
                    }
                    continue;
                }
                out.push(c);
            }
            Some(G::Polygon(out))
        }
        other => Some(other),
    }
}

/// Liang-Barsky clipping of each segment; consecutive kept pieces join.
fn clip_line(l: &[P], b: (f64, f64, f64, f64)) -> Vec<Vec<P>> {
    let mut out: Vec<Vec<P>> = Vec::new();
    let mut cur: Vec<P> = Vec::new();
    for w in l.windows(2) {
        match clip_segment(w[0], w[1], b) {
            Some((a, c)) => {
                if cur.last() != Some(&a) {
                    if cur.len() >= 2 {
                        out.push(std::mem::take(&mut cur));
                    }
                    cur.clear();
                    cur.push(a);
                }
                cur.push(c);
            }
            None => {
                if cur.len() >= 2 {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

fn clip_segment(a: P, c: P, b: (f64, f64, f64, f64)) -> Option<(P, P)> {
    let (dx, dy) = (c.0 - a.0, c.1 - a.1);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [(-dx, a.0 - b.0), (dx, b.2 - a.0), (-dy, a.1 - b.1), (dy, b.3 - a.1)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t1 {
                    return None;
                }
                t0 = t0.max(r);
            } else {
                if r < t0 {
                    return None;
                }
                t1 = t1.min(r);
            }
        }
    }
    let at = |t: f64| if t == 0.0 { a } else if t == 1.0 { c } else { (a.0 + t * dx, a.1 + t * dy) };
    Some((at(t0), at(t1)))
}

/// Sutherland-Hodgman clipping of a closed ring to the box; closed result.
fn clip_ring(r: &[P], b: (f64, f64, f64, f64)) -> Vec<P> {
    let mut pts: Vec<P> = r.to_vec();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    // (inside test, intersection) per box edge.
    let edges: [(&dyn Fn(&P) -> bool, &dyn Fn(&P, &P) -> P); 4] = [
        (&|p: &P| p.0 >= b.0, &|p: &P, q: &P| (b.0, p.1 + (q.1 - p.1) * (b.0 - p.0) / (q.0 - p.0))),
        (&|p: &P| p.0 <= b.2, &|p: &P, q: &P| (b.2, p.1 + (q.1 - p.1) * (b.2 - p.0) / (q.0 - p.0))),
        (&|p: &P| p.1 >= b.1, &|p: &P, q: &P| (p.0 + (q.0 - p.0) * (b.1 - p.1) / (q.1 - p.1), b.1)),
        (&|p: &P| p.1 <= b.3, &|p: &P, q: &P| (p.0 + (q.0 - p.0) * (b.3 - p.1) / (q.1 - p.1), b.3)),
    ];
    for (inside, cross) in edges {
        if pts.is_empty() {
            break;
        }
        let input = std::mem::take(&mut pts);
        let n = input.len();
        for i in 0..n {
            let cur = input[i];
            let prev = input[(i + n - 1) % n];
            match (inside(&cur), inside(&prev)) {
                (true, true) => pts.push(cur),
                (true, false) => {
                    pts.push(cross(&prev, &cur));
                    pts.push(cur);
                }
                (false, true) => pts.push(cross(&prev, &cur)),
                (false, false) => {}
            }
        }
    }
    pts.dedup();
    if let Some(&f) = pts.first() {
        if pts.len() > 1 && pts.last() == Some(&f) {
            pts.pop();
        }
        pts.push(f);
    }
    pts
}

/// JTS `Area.ofRingSigned`: positive for clockwise rings (y up).
fn ring_area(r: &[P]) -> f64 {
    if r.len() < 3 {
        return 0.0;
    }
    let x0 = r[0].0;
    let mut sum = 0.0;
    for i in 1..r.len() - 1 {
        sum += (r[i].0 - x0) * (r[i - 1].1 - r[i + 1].1);
    }
    sum / 2.0
}

fn simplify(g: G, tol: f64) -> G {
    let line = |v: Vec<P>| douglas_peucker(&v, tol, 2);
    let ring = |v: Vec<P>| {
        let s = douglas_peucker(&v, tol, 4);
        if s.len() < 4 { v } else { s }
    };
    match g {
        G::Line(v) => G::Line(line(v)),
        G::MultiLine(v) => G::MultiLine(v.into_iter().map(line).collect()),
        G::Polygon(v) => G::Polygon(v.into_iter().map(ring).collect()),
        G::MultiPolygon(v) => G::MultiPolygon(v.into_iter().map(|p| p.into_iter().map(ring).collect()).collect()),
        G::Collection(v) => G::Collection(v.into_iter().map(|x| simplify(x, tol)).collect()),
        other => other,
    }
}

/// Douglas-Peucker keeping the end points; fewer than `min` points out
/// means the input is kept.
fn douglas_peucker(v: &[P], tol: f64, min: usize) -> Vec<P> {
    if v.len() <= 2 {
        return v.to_vec();
    }
    let mut keep = vec![false; v.len()];
    keep[0] = true;
    keep[v.len() - 1] = true;
    let mut stack = vec![(0usize, v.len() - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (mut best, mut at) = (-1.0f64, i);
        for k in i + 1..j {
            let d = seg_dist(v[k], v[i], v[j]);
            if d > best {
                best = d;
                at = k;
            }
        }
        if best > tol {
            keep[at] = true;
            stack.push((i, at));
            stack.push((at, j));
        }
    }
    let out: Vec<P> = v.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
    if out.len() < min { v.to_vec() } else { out }
}

fn seg_dist(p: P, a: P, b: P) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
    let (qx, qy) = (a.0 + t * dx, a.1 + t * dy);
    ((p.0 - qx).powi(2) + (p.1 - qy).powi(2)).sqrt()
}

/// mapbox-vector-tile `JtsAdapter.flatFeatureList`: a collection's members
/// pushed on a stack and popped (reversed), multi-geometries kept whole.
fn flatten(g: G, out: &mut Vec<G>) {
    let mut stack = vec![g];
    while let Some(g) = stack.pop() {
        match g {
            G::Collection(v) => stack.extend(v),
            other => out.push(other),
        }
    }
}

fn zigzag(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

const CMD_HDR_LEN_MAX: usize = (1 << 29) - 1;

fn cmd_hdr(id: u32, len: usize) -> u32 {
    (id & 0x7) | ((len as u32) << 3)
}

/// The cursor of a feature's commands (ints, as ES's `Vec2d` read as ints).
struct Cursor(i32, i32);

fn move_cursor(c: &mut Cursor, cmds: &mut Vec<u32>, p: P) {
    let (x, y) = (p.0 as i32, p.1 as i32);
    cmds.push(zigzag(x.wrapping_sub(c.0)));
    cmds.push(zigzag(y.wrapping_sub(c.1)));
    *c = Cursor(x, y);
}

fn eq_int(c: &Cursor, p: P) -> bool {
    c.0 == p.0 as i32 && c.1 == p.1 as i32
}

/// `PatchedJtsAdapter.ptsToGeomCmds`.
fn pts_cmds(pts: &[P], c: &mut Cursor) -> Vec<u32> {
    if pts.is_empty() {
        return Vec::new();
    }
    let mut cmds = vec![0];
    let mut n = 0;
    for (i, p) in pts.iter().enumerate() {
        if i == 0 || !eq_int(c, *p) {
            n += 1;
            move_cursor(c, &mut cmds, *p);
        }
    }
    if n <= CMD_HDR_LEN_MAX {
        cmds[0] = cmd_hdr(1, n);
        cmds
    } else {
        Vec::new()
    }
}

/// `PatchedJtsAdapter.linesToGeomCmds` (the cursor moves even when the
/// line is dropped, as there).
fn line_cmds(pts: &[P], close: bool, c: &mut Cursor, min_line_to: usize) -> Vec<u32> {
    if pts.is_empty() {
        return Vec::new();
    }
    let first = pts[0];
    let repeat = pts[1..].iter().rev().take_while(|p| p.0 as i32 == first.0 as i32 && p.1 as i32 == first.1 as i32).count();
    let len = pts.len() - repeat;
    if len < 2 {
        return Vec::new();
    }
    let mut cmds = vec![cmd_hdr(1, 1)];
    move_cursor(c, &mut cmds, first);
    let hdr = cmds.len();
    cmds.push(0);
    let mut n = 0;
    for p in &pts[1..len] {
        if !eq_int(c, *p) {
            n += 1;
            move_cursor(c, &mut cmds, *p);
        }
    }
    if n >= min_line_to && n <= CMD_HDR_LEN_MAX {
        cmds[hdr] = cmd_hdr(2, n);
        if close {
            cmds.push(cmd_hdr(7, 1));
        }
        cmds
    } else {
        Vec::new()
    }
}

/// `PatchedJtsAdapter.toFeature`, as protobuf `Feature` bytes (type,
/// packed geometry; no id, no tags). `None`: no geometry left.
fn encode_feature(g: &G) -> Option<Vec<u8>> {
    let mut c = Cursor(0, 0);
    let (ty, cmds) = match g {
        G::Point(p) => (1u32, pts_cmds(std::slice::from_ref(p), &mut c)),
        G::MultiPoint(v) => (1, pts_cmds(v, &mut c)),
        G::Line(l) => (2, line_cmds(l, false, &mut c, 1)),
        G::MultiLine(ls) => (2, ls.iter().flat_map(|l| line_cmds(l, false, &mut c, 1)).collect()),
        G::Polygon(p) => (3, polygon_cmds(p, &mut c)),
        G::MultiPolygon(ps) => (3, ps.iter().flat_map(|p| polygon_cmds(p, &mut c)).collect()),
        G::Collection(_) => return None,
    };
    if cmds.is_empty() {
        return None;
    }
    let mut geom = Vec::new();
    for v in &cmds {
        put_varint(&mut geom, *v as u64);
    }
    let mut out = vec![0x18];
    put_varint(&mut out, ty as u64);
    out.push(0x22);
    put_varint(&mut out, geom.len() as u64);
    out.extend_from_slice(&geom);
    Some(out)
}

fn polygon_cmds(rings: &[Vec<P>], c: &mut Cursor) -> Vec<u32> {
    let Some(shell) = rings.first() else { return Vec::new() };
    let ext_area = ring_area(shell);
    if Tile::round(ext_area) == 0.0 {
        return Vec::new();
    }
    let mut shell = shell.clone();
    if ext_area > 0.0 {
        shell.reverse();
    }
    let mut out = line_cmds(&shell, true, c, 2);
    for h in &rings[1..] {
        let a = ring_area(h);
        if Tile::round(a) == 0.0 {
            continue;
        }
        let mut h = h.clone();
        if a < 0.0 {
            h.reverse();
        }
        if ext_area.abs() <= a.abs() {
            return Vec::new();
        }
        out.extend(line_cmds(&h, true, c, 2));
    }
    out
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustic_core::fieldtypes::shape::{parse_wkt, ParseOpts};

    fn wkt(s: &str) -> Geometry {
        parse_wkt(s, &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap()
    }

    /// Decode a feature's geometry commands back to tile-pixel paths.
    fn decode(b64: &str) -> (u64, Vec<Vec<(i64, i64)>>) {
        use base64::Engine;
        let b = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
        let mut pos = 0;
        let mut varint = |pos: &mut usize| {
            let (mut v, mut s) = (0u64, 0);
            loop {
                let x = b[*pos];
                *pos += 1;
                v |= ((x & 0x7f) as u64) << s;
                s += 7;
                if x < 0x80 {
                    return v;
                }
            }
        };
        assert_eq!(b[pos], 0x18);
        pos += 1;
        let ty = varint(&mut pos);
        assert_eq!(b[pos], 0x22);
        pos += 1;
        let len = varint(&mut pos) as usize;
        let end = pos + len;
        let mut cmds = Vec::new();
        while pos < end {
            cmds.push(varint(&mut pos) as u32);
        }
        let unzig = |v: u32| ((v >> 1) as i64) ^ -((v & 1) as i64);
        let (mut x, mut y) = (0i64, 0i64);
        let mut paths: Vec<Vec<(i64, i64)>> = Vec::new();
        let mut i = 0;
        while i < cmds.len() {
            let (id, n) = (cmds[i] & 7, (cmds[i] >> 3) as usize);
            i += 1;
            match id {
                1 => {
                    for _ in 0..n {
                        x += unzig(cmds[i]);
                        y += unzig(cmds[i + 1]);
                        i += 2;
                        paths.push(vec![(x, y)]);
                    }
                }
                2 => {
                    for _ in 0..n {
                        x += unzig(cmds[i]);
                        y += unzig(cmds[i + 1]);
                        i += 2;
                        paths.last_mut().unwrap().push((x, y));
                    }
                }
                _ => {}
            }
        }
        (ty, paths)
    }

    #[test]
    fn specs_parse_like_es() {
        assert_eq!(parse_spec("mvt(9/255/341)"), Some((9, 255, 341, 4096, 5)));
        assert_eq!(parse_spec("mvt(2/1/1@5000)"), Some((2, 1, 1, 5000, 5)));
        assert_eq!(parse_spec("mvt(2/1/1@5000:10)"), Some((2, 1, 1, 5000, 10)));
        assert_eq!(parse_spec("mvt(2/1/1:10)"), Some((2, 1, 1, 4096, 10)));
        for bad in ["mvt(2/4/1)", "mvt(30/0/0)", "mvt(1/1)", "mvt(1/0/0@0)", "geojson", "mvt(a/b/c)"] {
            assert_eq!(parse_spec(bad), None, "{bad}");
        }
    }

    /// The tile 9/255/341 (Rally's: x 255 of 512 is just west of
    /// Greenwich, y 341 is south of the equator, around 51.4 S): a point at
    /// the tile's Mercator centre lands on pixel (2048, 2048); shapes
    /// outside the tile give nothing; a polygon covering the tile is the
    /// buffered tile box with MVT winding; a line across it is clipped to
    /// the buffered box; a collection gives one feature per member,
    /// reversed; a multipoint one feature.
    fn tile() -> (f64, f64, f64, f64, f64, f64) {
        let (top, left, bottom, right) = crate::fieldtypes::geo::tile_bbox(9, 255, 341);
        let n = 512.0f64;
        let cy = (std::f64::consts::PI * (1.0 - 2.0 * 341.5 / n)).sinh().atan().to_degrees();
        (top, left, bottom, right, (left + right) / 2.0, cy)
    }

    #[test]
    fn rally_tile_features() {
        let spec = parse_spec("mvt(9/255/341)").unwrap();
        let (top, left, bottom, right, cx, cy) = tile();
        assert!(top < 0.0 && bottom < top && left < 0.0 && right <= 0.0, "{top} {left} {bottom} {right}");
        let f = features(&wkt(&format!("POINT ({cx} {cy})")), spec);
        assert_eq!(f.len(), 1);
        let (ty, p) = decode(&f[0]);
        assert_eq!(ty, 1);
        assert!((p[0][0].0 - 2048).abs() <= 2 && (p[0][0].1 - 2048).abs() <= 2, "{p:?}");
        assert!(features(&wkt("POINT (10 10)"), spec).is_empty());
        assert!(features(&wkt("POINT (-0.3 51.4)"), spec).is_empty(), "London is the other hemisphere");
        assert!(features(&wkt("LINESTRING (10 10, 11 11)"), spec).is_empty());
        // Covering polygon: the buffered tile (5 px around 4096).
        let cover = format!("POLYGON (({l} {b}, {r} {b}, {r} {t}, {l} {t}, {l} {b}))", l = left - 1.0, r = right + 1.0, b = bottom - 1.0, t = top + 1.0);
        let f = features(&wkt(&cover), spec);
        assert_eq!(f.len(), 1);
        let (ty, rings) = decode(&f[0]);
        assert_eq!(ty, 3);
        let xs: Vec<i64> = rings[0].iter().map(|p| p.0).collect();
        let ys: Vec<i64> = rings[0].iter().map(|p| p.1).collect();
        let near = |v: i64, want: i64| (v - want).abs() <= 1;
        assert!(near(*xs.iter().min().unwrap(), -5) && near(*xs.iter().max().unwrap(), 4101), "{xs:?}");
        assert!(near(*ys.iter().min().unwrap(), -5) && near(*ys.iter().max().unwrap(), 4101), "{ys:?}");
        let area: i64 = (0..rings[0].len()).map(|i| {
            let (a, b) = (rings[0][i], rings[0][(i + 1) % rings[0].len()]);
            a.0 * b.1 - b.0 * a.1
        }).sum();
        assert!(area > 0, "exterior ring clockwise in tile space (y down): {area}");
        // A line across the tile is clipped to the buffered box.
        let across = format!("LINESTRING ({} {cy}, {} {cy})", left - 1.0, right + 1.0);
        let f = features(&wkt(&across), spec);
        assert_eq!(f.len(), 1);
        let (ty, lines) = decode(&f[0]);
        assert_eq!(ty, 2);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].iter().all(|p| p.0 >= -6 && p.0 <= 4102), "{lines:?}");
        // A collection gives one feature per member, reversed; a multi
        // geometry one feature.
        let f = features(&wkt(&format!("GEOMETRYCOLLECTION (POINT ({cx} {cy}), {across})")), spec);
        assert_eq!(f.iter().map(|x| decode(x).0).collect::<Vec<_>>(), vec![2, 1]);
        let f = features(&wkt(&format!("MULTIPOINT (({cx} {cy}), ({} {cy}))", cx + 0.1)), spec);
        assert_eq!(f.len(), 1);
        assert_eq!(decode(&f[0]).1.len(), 2);
    }

    /// The `fields` API: a doc's values as one collection, features as
    /// base64 strings, nothing for a doc outside the tile; other formats
    /// unchanged.
    #[test]
    fn fields_format_mvt() {
        use crate::mapping::EsType;
        let (_, left, _, right, cx, cy) = tile();
        let src = serde_json::json!({"shape": [format!("LINESTRING ({} {cy}, {} {cy})", left - 1.0, right + 1.0), format!("POINT ({cx} {cy})")]});
        let got = crate::fieldtypes::geo::fetch_values(EsType::GeoShape, &src, "shape", Some("mvt(9/255/341)"));
        assert_eq!(got.len(), 2);
        assert_eq!(got.iter().map(|v| decode(v.as_str().unwrap()).0).collect::<Vec<_>>(), vec![1, 2]);
        let far = serde_json::json!({"shape": "POINT (-0.3 51.4)"});
        assert!(crate::fieldtypes::geo::fetch_values(EsType::GeoShape, &far, "shape", Some("mvt(9/255/341)")).is_empty());
        let plain = crate::fieldtypes::geo::fetch_values(EsType::GeoShape, &far, "shape", None);
        assert_eq!(plain[0]["type"], "Point");
    }

    /// Timing of the Rally `mvt-hits` shapes: 10,000 London building-size
    /// polygons and long lines through the tile.
    /// `cargo test -p elasticsearch-engine --release mvt_feature_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mvt_feature_timing() {
        let spec = parse_spec("mvt(9/255/341)").unwrap();
        let mut shapes = Vec::new();
        for i in 0..10_000 {
            let (x, y) = (-0.7 + (i % 100) as f64 * 0.007, 51.18 + (i / 100) as f64 * 0.0043);
            shapes.push(if i % 10 == 0 {
                let pts: Vec<String> = (0..2000).map(|k| format!("{} {}", -2.0 + k as f64 * 0.002, y + (k as f64 * 0.01).sin() * 0.01)).collect();
                wkt(&format!("LINESTRING ({})", pts.join(", ")))
            } else {
                wkt(&format!("POLYGON (({x} {y}, {} {y}, {} {}, {x} {}, {x} {y}))", x + 0.0005, x + 0.0005, y + 0.0004, y + 0.0004))
            });
        }
        let t = std::time::Instant::now();
        let n: usize = shapes.iter().map(|g| features(g, spec).len()).sum();
        eprintln!("{n} features in {:?}", t.elapsed());
    }
}
