//! Lucene 10.5.1's `Component2D` family (`org.apache.lucene.geo`), the
//! relations ES `geo_shape` queries, doc values and aggregations use:
//! `Point2D`, `Line2D` and `Polygon2D` (over an `EdgeTree`),
//! `Rectangle2D`, and `ComponentTree` over several components, with the
//! same arithmetic (raw `f64` coordinates in degrees, `x` longitude, `y`
//! latitude). [`EncodedRect`] is `SpatialQuery.EncodedRectangle` (the
//! encoded-space relations of `LatLonShapeBoundingBoxQuery`).
//!
//! [`query_components`] turns a query geometry into Lucene geometries as
//! ES `LuceneGeometriesUtils.toLatLonGeometry(geometry, quantize = true)`
//! builds them, and [`ShapeQuery`] is `LatLonShape.newGeometryQuery`'s
//! choice of query: a bounding box query for one rectangle, an AND of
//! per-geometry queries for CONTAINS over several, else one component
//! query.

use crate::geo::{decode_lat, decode_lon, encode_lat, encode_lon};

/// `PointValues.Relation`: how a box relates to the query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    /// `CELL_INSIDE_QUERY`.
    Inside,
    /// `CELL_OUTSIDE_QUERY`.
    Outside,
    /// `CELL_CROSSES_QUERY`.
    Crosses,
}

/// `Component2D.WithinRelation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WithinRelation {
    Candidate,
    NotWithin,
    Disjoint,
}

/// `ShapeField.QueryRelation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryRelation {
    Intersects,
    Within,
    Disjoint,
    Contains,
}

/// Lucene `GeoUtils.orient`.
#[inline]
pub fn orient(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> i32 {
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

/// Lucene `GeoUtils.lineCrossesLine`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn line_crosses_line(a1x: f64, a1y: f64, b1x: f64, b1y: f64, a2x: f64, a2y: f64, b2x: f64, b2y: f64) -> bool {
    orient(a2x, a2y, b2x, b2y, a1x, a1y) * orient(a2x, a2y, b2x, b2y, b1x, b1y) < 0 && orient(a1x, a1y, b1x, b1y, a2x, a2y) * orient(a1x, a1y, b1x, b1y, b2x, b2y) < 0
}

/// Lucene `GeoUtils.lineCrossesLineWithBoundary`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn line_crosses_line_with_boundary(a1x: f64, a1y: f64, b1x: f64, b1y: f64, a2x: f64, a2y: f64, b2x: f64, b2y: f64) -> bool {
    orient(a2x, a2y, b2x, b2y, a1x, a1y) * orient(a2x, a2y, b2x, b2y, b1x, b1y) <= 0 && orient(a1x, a1y, b1x, b1y, a2x, a2y) * orient(a1x, a1y, b1x, b1y, b2x, b2y) <= 0
}

/// `Component2D.disjoint`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn disjoint(min_x1: f64, max_x1: f64, min_y1: f64, max_y1: f64, min_x2: f64, max_x2: f64, min_y2: f64, max_y2: f64) -> bool {
    max_y1 < min_y2 || min_y1 > max_y2 || max_x1 < min_x2 || min_x1 > max_x2
}

/// `Component2D.within`: box 1 within box 2.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn within(min_x1: f64, max_x1: f64, min_y1: f64, max_y1: f64, min_x2: f64, max_x2: f64, min_y2: f64, max_y2: f64) -> bool {
    min_y2 <= min_y1 && max_y2 >= max_y1 && min_x2 <= min_x1 && max_x2 >= max_x1
}

/// `Component2D.containsPoint`.
#[inline]
pub fn contains_point(x: f64, y: f64, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
    x >= min_x && x <= max_x && y >= min_y && y <= max_y
}

/// `Component2D.pointInTriangle`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn point_in_triangle(min_x: f64, max_x: f64, min_y: f64, max_y: f64, x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
    if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
        let a = orient(x, y, ax, ay, bx, by);
        let b = orient(x, y, bx, by, cx, cy);
        if a == 0 || b == 0 || (a < 0) == (b < 0) {
            let c = orient(x, y, cx, cy, ax, ay);
            return c == 0 || ((c < 0) == (b < 0 || a < 0));
        }
        false
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// EdgeTree

#[derive(Clone, Debug)]
struct Edge {
    y1: f64,
    y2: f64,
    x1: f64,
    x2: f64,
    low: f64,
    max: f64,
    left: u32,
    right: u32,
}

const NONE: u32 = u32::MAX;

/// Lucene `EdgeTree`: a polygon's or line's edges in an interval tree on
/// latitude (sorted by `low`, then `max`; balanced on the median).
#[derive(Clone, Debug)]
pub struct EdgeTree {
    edges: Vec<Edge>,
    root: u32,
}

const FALSE: u8 = 0;
const TRUE: u8 = 1;
const ON_EDGE: u8 = 2;

impl EdgeTree {
    fn create(x: &[f64], y: &[f64]) -> EdgeTree {
        let mut edges: Vec<Edge> = (1..x.len())
            .map(|i| {
                let (x1, y1, x2, y2) = (x[i - 1], y[i - 1], x[i], y[i]);
                Edge { x1, y1, x2, y2, low: y1.min(y2), max: y1.max(y2), left: NONE, right: NONE }
            })
            .collect();
        // `Arrays.sort` (stable) by `Double.compare(low)`, then `max`.
        edges.sort_by(|a, b| a.low.total_cmp(&b.low).then(a.max.total_cmp(&b.max)));
        let n = edges.len();
        let root = if n == 0 { NONE } else { Self::build(&mut edges, 0, n as i64 - 1) };
        EdgeTree { edges, root }
    }

    fn build(edges: &mut [Edge], low: i64, high: i64) -> u32 {
        if low > high {
            return NONE;
        }
        let mid = ((low + high) as u64 >> 1) as i64;
        let left = Self::build(edges, low, mid - 1);
        let right = Self::build(edges, mid + 1, high);
        let m = mid as usize;
        edges[m].left = left;
        edges[m].right = right;
        if left != NONE {
            edges[m].max = edges[m].max.max(edges[left as usize].max);
        }
        if right != NONE {
            edges[m].max = edges[m].max.max(edges[right as usize].max);
        }
        m as u32
    }

    /// The root edge's first vertex (Lucene `tree.x1`, `tree.y1`).
    fn root_xy(&self) -> (f64, f64) {
        let e = &self.edges[self.root as usize];
        (e.x1, e.y1)
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        self.root != NONE && self.contains_pnpoly(self.root, x, y) > FALSE
    }

    fn contains_pnpoly(&self, n: u32, x: f64, y: f64) -> u8 {
        let e = &self.edges[n as usize];
        let mut res = FALSE;
        if y <= e.max {
            if (y == e.y1 && y == e.y2) || ((y <= e.y1 && y >= e.y2) != (y >= e.y1 && y <= e.y2)) {
                if (x == e.x1 && x == e.x2) || (((x <= e.x1 && x >= e.x2) != (x >= e.x1 && x <= e.x2)) && orient(e.x1, e.y1, e.x2, e.y2, x, y) == 0) {
                    return ON_EDGE;
                } else if (e.y1 > y) != (e.y2 > y) {
                    res = if x < (e.x2 - e.x1) * (y - e.y1) / (e.y2 - e.y1) + e.x1 { TRUE } else { FALSE };
                }
            }
            if e.left != NONE {
                res ^= self.contains_pnpoly(e.left, x, y);
                if res & 0x02 == 0x02 {
                    return ON_EDGE;
                }
            }
            if e.right != NONE && y >= e.low {
                res ^= self.contains_pnpoly(e.right, x, y);
                if res & 0x02 == 0x02 {
                    return ON_EDGE;
                }
            }
        }
        res
    }

    fn is_point_on_line(&self, n: u32, x: f64, y: f64) -> bool {
        if n == NONE {
            return false;
        }
        let e = &self.edges[n as usize];
        if y <= e.max {
            let (a1x, a1y, b1x, b1y) = (e.x1, e.y1, e.x2, e.y2);
            let outside = (a1y < y && b1y < y) || (a1y > y && b1y > y) || (a1x < x && b1x < x) || (a1x > x && b1x > x);
            if !outside && orient(a1x, a1y, b1x, b1y, x, y) == 0 {
                return true;
            }
            if e.left != NONE && self.is_point_on_line(e.left, x, y) {
                return true;
            }
            if e.right != NONE && y >= e.low && self.is_point_on_line(e.right, x, y) {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn crosses_triangle(&self, n: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, include_boundary: bool) -> bool {
        if n == NONE {
            return false;
        }
        let e = &self.edges[n as usize];
        if min_y <= e.max {
            let (dy, ey, dx, ex) = (e.y1, e.y2, e.x1, e.x2);
            let outside = (dy < min_y && ey < min_y) || (dy > max_y && ey > max_y) || (dx < min_x && ex < min_x) || (dx > max_x && ex > max_x);
            if !outside {
                let hit = if include_boundary {
                    line_crosses_line_with_boundary(dx, dy, ex, ey, ax, ay, bx, by) || line_crosses_line_with_boundary(dx, dy, ex, ey, bx, by, cx, cy) || line_crosses_line_with_boundary(dx, dy, ex, ey, cx, cy, ax, ay)
                } else {
                    line_crosses_line(dx, dy, ex, ey, ax, ay, bx, by) || line_crosses_line(dx, dy, ex, ey, bx, by, cx, cy) || line_crosses_line(dx, dy, ex, ey, cx, cy, ax, ay)
                };
                if hit {
                    return true;
                }
            }
            if e.left != NONE && self.crosses_triangle(e.left, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, include_boundary) {
                return true;
            }
            if e.right != NONE && max_y >= e.low && self.crosses_triangle(e.right, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, include_boundary) {
                return true;
            }
        }
        false
    }

    fn crosses_box(&self, n: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, include_boundary: bool) -> bool {
        if n == NONE {
            return false;
        }
        let e = &self.edges[n as usize];
        if min_y <= e.max {
            let (cy, dy, cx, dx) = (e.y1, e.y2, e.x1, e.x2);
            // `Rectangle.containsPoint(lat, lon, minLat, maxLat, minLon, maxLon)`.
            if contains_point(cx, cy, min_x, max_x, min_y, max_y) || contains_point(dx, dy, min_x, max_x, min_y, max_y) {
                return true;
            }
            let outside = (cy < min_y && dy < min_y) || (cy > max_y && dy > max_y) || (cx < min_x && dx < min_x) || (cx > max_x && dx > max_x);
            if !outside {
                let hit = if include_boundary {
                    line_crosses_line_with_boundary(cx, cy, dx, dy, min_x, min_y, max_x, min_y)
                        || line_crosses_line_with_boundary(cx, cy, dx, dy, max_x, min_y, max_x, max_y)
                        || line_crosses_line_with_boundary(cx, cy, dx, dy, max_x, max_y, min_x, max_y)
                        || line_crosses_line_with_boundary(cx, cy, dx, dy, min_x, max_y, min_x, min_y)
                } else {
                    line_crosses_line(cx, cy, dx, dy, min_x, min_y, max_x, min_y)
                        || line_crosses_line(cx, cy, dx, dy, max_x, min_y, max_x, max_y)
                        || line_crosses_line(cx, cy, dx, dy, max_x, max_y, min_x, max_y)
                        || line_crosses_line(cx, cy, dx, dy, min_x, max_y, min_x, min_y)
                };
                if hit {
                    return true;
                }
            }
            if e.left != NONE && self.crosses_box(e.left, min_x, max_x, min_y, max_y, include_boundary) {
                return true;
            }
            if e.right != NONE && max_y >= e.low && self.crosses_box(e.right, min_x, max_x, min_y, max_y, include_boundary) {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn crosses_line(&self, n: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, a2x: f64, a2y: f64, b2x: f64, b2y: f64, include_boundary: bool) -> bool {
        if n == NONE {
            return false;
        }
        let e = &self.edges[n as usize];
        if min_y <= e.max {
            let (a1x, a1y, b1x, b1y) = (e.x1, e.y1, e.x2, e.y2);
            let outside = (a1y < min_y && b1y < min_y) || (a1y > max_y && b1y > max_y) || (a1x < min_x && b1x < min_x) || (a1x > max_x && b1x > max_x);
            if !outside {
                let hit = if include_boundary { line_crosses_line_with_boundary(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y) } else { line_crosses_line(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y) };
                if hit {
                    return true;
                }
            }
            if e.left != NONE && self.crosses_line(e.left, min_x, max_x, min_y, max_y, a2x, a2y, b2x, b2y, include_boundary) {
                return true;
            }
            if e.right != NONE && max_y >= e.low && self.crosses_line(e.right, min_x, max_x, min_y, max_y, a2x, a2y, b2x, b2y, include_boundary) {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Components

/// Lucene `Point2D`.
#[derive(Clone, Debug)]
pub struct Point2D {
    x: f64,
    y: f64,
}

/// Lucene `Line2D`.
#[derive(Clone, Debug)]
pub struct Line2D {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    tree: EdgeTree,
}

/// Lucene `Polygon2D`.
#[derive(Clone, Debug)]
pub struct Polygon2D {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    holes: Option<Box<Component>>,
    tree: EdgeTree,
}

/// Lucene `Rectangle2D`.
#[derive(Clone, Debug)]
pub struct Rectangle2D {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

#[derive(Clone, Debug)]
struct TreeNode {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    component: Component,
    left: u32,
    right: u32,
}

/// Lucene `ComponentTree`: components in a KD tree on their boxes.
#[derive(Clone, Debug)]
pub struct ComponentTree {
    nodes: Vec<TreeNode>,
    root: u32,
}

/// One Lucene `Component2D`.
#[derive(Clone, Debug)]
pub enum Component {
    Point(Point2D),
    Line(Line2D),
    Polygon(Polygon2D),
    Rectangle(Rectangle2D),
    Circle(Circle2D),
    Tree(ComponentTree),
}

macro_rules! tri_box {
    ($ax:expr, $ay:expr, $bx:expr, $by:expr, $cx:expr, $cy:expr) => {
        ($ax.min($bx).min($cx), $ax.max($bx).max($cx), $ay.min($by).min($cy), $ay.max($by).max($cy))
    };
}

impl Component {
    /// A polygon, or a box of positive width and height: no point
    /// contains it (a point's `withinPoint` is never a `CANDIDATE`).
    /// A point of the component itself (a vertex, a corner): any shape
    /// containing the component contains it. `None` for a tree of several.
    pub fn sample_point(&self) -> Option<(f64, f64)> {
        match self {
            Component::Point(p) => Some((p.x, p.y)),
            Component::Line(l) => (l.tree.root != NONE).then(|| l.tree.root_xy()),
            Component::Polygon(p) => (p.tree.root != NONE).then(|| p.tree.root_xy()),
            Component::Rectangle(r) => Some((r.min_x, r.min_y)),
            Component::Circle(c) => Some((c.center_lon, c.center_lat)),
            Component::Tree(_) => None,
        }
    }

    pub fn has_area(&self) -> bool {
        match self {
            Component::Polygon(_) => true,
            Component::Rectangle(r) => r.min_x < r.max_x && r.min_y < r.max_y,
            _ => false,
        }
    }

    pub fn min_x(&self) -> f64 {
        match self {
            Component::Point(p) => p.x,
            Component::Line(l) => l.min_x,
            Component::Polygon(p) => p.min_x,
            Component::Rectangle(r) => r.min_x,
            Component::Circle(c) => c.min_x(),
            Component::Tree(t) => t.nodes[t.root as usize].min_x,
        }
    }
    pub fn max_x(&self) -> f64 {
        match self {
            Component::Point(p) => p.x,
            Component::Line(l) => l.max_x,
            Component::Polygon(p) => p.max_x,
            Component::Rectangle(r) => r.max_x,
            Component::Circle(c) => c.max_x(),
            Component::Tree(t) => t.nodes[t.root as usize].max_x,
        }
    }
    pub fn min_y(&self) -> f64 {
        match self {
            Component::Point(p) => p.y,
            Component::Line(l) => l.min_y,
            Component::Polygon(p) => p.min_y,
            Component::Rectangle(r) => r.min_y,
            Component::Circle(c) => c.min_lat,
            Component::Tree(t) => t.nodes[t.root as usize].min_y,
        }
    }
    pub fn max_y(&self) -> f64 {
        match self {
            Component::Point(p) => p.y,
            Component::Line(l) => l.max_y,
            Component::Polygon(p) => p.max_y,
            Component::Rectangle(r) => r.max_y,
            Component::Circle(c) => c.max_lat,
            Component::Tree(t) => t.nodes[t.root as usize].max_y,
        }
    }

    /// `contains(x, y)`.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            Component::Point(p) => x == p.x && y == p.y,
            Component::Line(l) => contains_point(x, y, l.min_x, l.max_x, l.min_y, l.max_y) && l.tree.is_point_on_line(l.tree.root, x, y),
            Component::Polygon(p) => {
                if contains_point(x, y, p.min_x, p.max_x, p.min_y, p.max_y) && p.tree.contains(x, y) {
                    return p.holes.as_ref().is_none_or(|h| !h.contains(x, y));
                }
                false
            }
            Component::Rectangle(r) => contains_point(x, y, r.min_x, r.max_x, r.min_y, r.max_y),
            Component::Circle(c) => c.contains(x, y),
            Component::Tree(t) => t.contains(t.root, x, y, false),
        }
    }

    /// `relate(minX, maxX, minY, maxY)`.
    pub fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        match self {
            Component::Point(p) => {
                if contains_point(p.x, p.y, min_x, max_x, min_y, max_y) {
                    Relation::Crosses
                } else {
                    Relation::Outside
                }
            }
            Component::Line(l) => {
                if disjoint(l.min_x, l.max_x, l.min_y, l.max_y, min_x, max_x, min_y, max_y) {
                    return Relation::Outside;
                }
                if within(l.min_x, l.max_x, l.min_y, l.max_y, min_x, max_x, min_y, max_y) {
                    return Relation::Crosses;
                }
                if l.tree.crosses_box(l.tree.root, min_x, max_x, min_y, max_y, true) {
                    return Relation::Crosses;
                }
                Relation::Outside
            }
            Component::Polygon(p) => p.relate(self, min_x, max_x, min_y, max_y),
            Component::Rectangle(r) => {
                if disjoint(r.min_x, r.max_x, r.min_y, r.max_y, min_x, max_x, min_y, max_y) {
                    return Relation::Outside;
                }
                if within(min_x, max_x, min_y, max_y, r.min_x, r.max_x, r.min_y, r.max_y) {
                    return Relation::Inside;
                }
                Relation::Crosses
            }
            Component::Circle(c) => {
                if c.disjoint(min_x, max_x, min_y, max_y) {
                    return Relation::Outside;
                }
                if c.within(min_x, max_x, min_y, max_y) {
                    return Relation::Crosses;
                }
                c.relate(min_x, max_x, min_y, max_y)
            }
            Component::Tree(t) => t.relate(t.root, min_x, max_x, min_y, max_y, false),
        }
    }

    /// `intersectsLine(minX, maxX, minY, maxY, aX, aY, bX, bY)`.
    #[allow(clippy::too_many_arguments)]
    pub fn intersects_line_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        match self {
            Component::Point(p) => contains_point(p.x, p.y, min_x, max_x, min_y, max_y) && orient(ax, ay, bx, by, p.x, p.y) == 0,
            Component::Line(l) => {
                if disjoint(l.min_x, l.max_x, l.min_y, l.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                l.tree.crosses_line(l.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, true)
            }
            Component::Polygon(p) => {
                if disjoint(p.min_x, p.max_x, p.min_y, p.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                if self.contains(ax, ay) || self.contains(bx, by) || p.tree.crosses_line(p.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, true) {
                    return p.holes.as_ref().is_none_or(|h| !h.contains_line_box(min_x, max_x, min_y, max_y, ax, ay, bx, by));
                }
                false
            }
            Component::Rectangle(r) => {
                if disjoint(r.min_x, r.max_x, r.min_y, r.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                self.contains(ax, ay) || self.contains(bx, by) || r.edges_intersect(ax, ay, bx, by)
            }
            Component::Circle(c) => !c.disjoint(min_x, max_x, min_y, max_y) && (c.contains(ax, ay) || c.contains(bx, by) || c.intersects_line(ax, ay, bx, by)),
            Component::Tree(t) => t.intersects_line(t.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, false),
        }
    }

    /// `intersectsTriangle(minX, maxX, minY, maxY, aX, aY, bX, bY, cX, cY)`.
    #[allow(clippy::too_many_arguments)]
    pub fn intersects_triangle_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
        match self {
            Component::Point(p) => point_in_triangle(min_x, max_x, min_y, max_y, p.x, p.y, ax, ay, bx, by, cx, cy),
            Component::Line(l) => {
                if disjoint(l.min_x, l.max_x, l.min_y, l.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                let (tx, ty) = l.tree.root_xy();
                point_in_triangle(min_x, max_x, min_y, max_y, tx, ty, ax, ay, bx, by, cx, cy) || l.tree.crosses_triangle(l.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, true)
            }
            Component::Polygon(p) => {
                if disjoint(p.min_x, p.max_x, p.min_y, p.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                let (tx, ty) = p.tree.root_xy();
                if self.contains(ax, ay)
                    || self.contains(bx, by)
                    || self.contains(cx, cy)
                    || point_in_triangle(min_x, max_x, min_y, max_y, tx, ty, ax, ay, bx, by, cx, cy)
                    || p.tree.crosses_triangle(p.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, true)
                {
                    return p.holes.as_ref().is_none_or(|h| !h.contains_triangle_box(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy));
                }
                false
            }
            Component::Rectangle(r) => {
                if disjoint(r.min_x, r.max_x, r.min_y, r.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                self.contains(ax, ay)
                    || self.contains(bx, by)
                    || self.contains(cx, cy)
                    || point_in_triangle(min_x, max_x, min_y, max_y, r.min_x, r.min_y, ax, ay, bx, by, cx, cy)
                    || r.edges_intersect(ax, ay, bx, by)
                    || r.edges_intersect(bx, by, cx, cy)
                    || r.edges_intersect(cx, cy, ax, ay)
            }
            Component::Circle(c) => {
                !c.disjoint(min_x, max_x, min_y, max_y)
                    && (c.contains(ax, ay)
                        || c.contains(bx, by)
                        || c.contains(cx, cy)
                        || point_in_triangle(min_x, max_x, min_y, max_y, c.center_lon, c.center_lat, ax, ay, bx, by, cx, cy)
                        || c.intersects_line(ax, ay, bx, by)
                        || c.intersects_line(bx, by, cx, cy)
                        || c.intersects_line(cx, cy, ax, ay))
            }
            Component::Tree(t) => t.intersects_triangle(t.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, false),
        }
    }

    /// `containsLine(minX, maxX, minY, maxY, aX, aY, bX, bY)`.
    #[allow(clippy::too_many_arguments)]
    pub fn contains_line_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        match self {
            Component::Point(_) | Component::Line(_) => false,
            Component::Polygon(p) => {
                if disjoint(p.min_x, p.max_x, p.min_y, p.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                if self.contains(ax, ay) && self.contains(bx, by) && !p.tree.crosses_line(p.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, false) {
                    return p.holes.as_ref().is_none_or(|h| !h.intersects_line_box(min_x, max_x, min_y, max_y, ax, ay, bx, by));
                }
                false
            }
            Component::Rectangle(r) => within(min_x, max_x, min_y, max_y, r.min_x, r.max_x, r.min_y, r.max_y),
            Component::Circle(c) => !c.disjoint(min_x, max_x, min_y, max_y) && c.contains(ax, ay) && c.contains(bx, by),
            Component::Tree(t) => t.contains_line(t.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, false),
        }
    }

    /// `containsTriangle(minX, maxX, minY, maxY, aX, aY, bX, bY, cX, cY)`.
    #[allow(clippy::too_many_arguments)]
    pub fn contains_triangle_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
        match self {
            Component::Point(_) | Component::Line(_) => false,
            Component::Polygon(p) => {
                if disjoint(p.min_x, p.max_x, p.min_y, p.max_y, min_x, max_x, min_y, max_y) {
                    return false;
                }
                if self.contains(ax, ay) && self.contains(bx, by) && self.contains(cx, cy) && !p.tree.crosses_triangle(p.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, false) {
                    return p.holes.as_ref().is_none_or(|h| !h.intersects_triangle_box(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy));
                }
                false
            }
            Component::Rectangle(r) => within(min_x, max_x, min_y, max_y, r.min_x, r.max_x, r.min_y, r.max_y),
            Component::Circle(c) => !c.disjoint(min_x, max_x, min_y, max_y) && c.contains(ax, ay) && c.contains(bx, by) && c.contains(cx, cy),
            Component::Tree(t) => t.contains_triangle(t.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, false),
        }
    }

    /// `withinPoint(x, y)`; `Err` for a tree of several components.
    pub fn within_point(&self, x: f64, y: f64) -> Result<WithinRelation, &'static str> {
        Ok(match self {
            Component::Point(_) => {
                if self.contains(x, y) {
                    WithinRelation::Candidate
                } else {
                    WithinRelation::Disjoint
                }
            }
            Component::Line(_) | Component::Polygon(_) | Component::Rectangle(_) | Component::Circle(_) => {
                if self.contains(x, y) {
                    WithinRelation::NotWithin
                } else {
                    WithinRelation::Disjoint
                }
            }
            Component::Tree(t) => return t.single("withinPoint")?.within_point(x, y),
        })
    }

    /// `withinLine(minX, maxX, minY, maxY, aX, aY, ab, bX, bY)`.
    #[allow(clippy::too_many_arguments)]
    pub fn within_line_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, ab: bool, bx: f64, by: f64) -> Result<WithinRelation, &'static str> {
        Ok(match self {
            Component::Point(_) => {
                if self.intersects_line_box(min_x, max_x, min_y, max_y, ax, ay, bx, by) {
                    WithinRelation::Candidate
                } else {
                    WithinRelation::Disjoint
                }
            }
            Component::Line(_) => {
                if ab && self.intersects_line_box(min_x, max_x, min_y, max_y, ax, ay, bx, by) {
                    WithinRelation::NotWithin
                } else {
                    WithinRelation::Disjoint
                }
            }
            Component::Polygon(p) => {
                if disjoint(p.min_x, p.max_x, p.min_y, p.max_y, min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                if self.contains(ax, ay) || self.contains(bx, by) {
                    return Ok(WithinRelation::NotWithin);
                }
                if ab && p.tree.crosses_line(p.tree.root, min_x, max_x, min_y, max_y, ax, ay, bx, by, true) {
                    return Ok(WithinRelation::NotWithin);
                }
                WithinRelation::Disjoint
            }
            Component::Rectangle(r) => {
                if disjoint(r.min_x, r.max_x, r.min_y, r.max_y, min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                if self.contains(ax, ay) || self.contains(bx, by) {
                    return Ok(WithinRelation::NotWithin);
                }
                if ab && r.edges_intersect(ax, ay, bx, by) {
                    return Ok(WithinRelation::NotWithin);
                }
                WithinRelation::Disjoint
            }
            Component::Circle(c) => {
                if c.disjoint(min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                if c.contains(ax, ay) || c.contains(bx, by) || (ab && c.intersects_line(ax, ay, bx, by)) {
                    return Ok(WithinRelation::NotWithin);
                }
                WithinRelation::Disjoint
            }
            Component::Tree(t) => return t.single("withinLine")?.within_line_box(min_x, max_x, min_y, max_y, ax, ay, ab, bx, by),
        })
    }

    /// `withinTriangle(minX, maxX, minY, maxY, aX, aY, ab, bX, bY, bc, cX, cY, ca)`.
    #[allow(clippy::too_many_arguments)]
    pub fn within_triangle_box(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, ab: bool, bx: f64, by: f64, bc: bool, cx: f64, cy: f64, ca: bool) -> Result<WithinRelation, &'static str> {
        Ok(match self {
            Component::Point(p) => {
                if point_in_triangle(min_x, max_x, min_y, max_y, p.x, p.y, ax, ay, bx, by, cx, cy) {
                    WithinRelation::Candidate
                } else {
                    WithinRelation::Disjoint
                }
            }
            Component::Line(l) => {
                if disjoint(l.min_x, l.max_x, l.min_y, l.max_y, min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                edge_tree_within_triangle(&l.tree, min_x, max_x, min_y, max_y, ax, ay, ab, bx, by, bc, cx, cy, ca)
            }
            Component::Polygon(p) => {
                if disjoint(p.min_x, p.max_x, p.min_y, p.max_y, min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                if self.contains(ax, ay) || self.contains(bx, by) || self.contains(cx, cy) {
                    return Ok(WithinRelation::NotWithin);
                }
                edge_tree_within_triangle(&p.tree, min_x, max_x, min_y, max_y, ax, ay, ab, bx, by, bc, cx, cy, ca)
            }
            Component::Rectangle(r) => {
                if disjoint(r.min_x, r.max_x, r.min_y, r.max_y, min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                if self.contains(ax, ay) || self.contains(bx, by) || self.contains(cx, cy) {
                    return Ok(WithinRelation::NotWithin);
                }
                let mut relation = WithinRelation::Disjoint;
                for (x1, y1, x2, y2, from_poly) in [(ax, ay, bx, by, ab), (bx, by, cx, cy, bc), (cx, cy, ax, ay, ca)] {
                    if r.edges_intersect(x1, y1, x2, y2) {
                        if from_poly {
                            return Ok(WithinRelation::NotWithin);
                        }
                        relation = WithinRelation::Candidate;
                    }
                }
                if relation == WithinRelation::Candidate {
                    return Ok(WithinRelation::Candidate);
                }
                if point_in_triangle(min_x, max_x, min_y, max_y, r.min_x, r.min_y, ax, ay, bx, by, cx, cy) {
                    return Ok(WithinRelation::Candidate);
                }
                relation
            }
            Component::Circle(c) => {
                if c.disjoint(min_x, max_x, min_y, max_y) {
                    return Ok(WithinRelation::Disjoint);
                }
                if c.contains(ax, ay) || c.contains(bx, by) || c.contains(cx, cy) {
                    return Ok(WithinRelation::NotWithin);
                }
                if (ab && c.intersects_line(ax, ay, bx, by)) || (bc && c.intersects_line(bx, by, cx, cy)) || (ca && c.intersects_line(cx, cy, ax, ay)) {
                    return Ok(WithinRelation::NotWithin);
                }
                if point_in_triangle(min_x, max_x, min_y, max_y, c.center_lon, c.center_lat, ax, ay, bx, by, cx, cy) {
                    return Ok(WithinRelation::Candidate);
                }
                WithinRelation::Disjoint
            }
            Component::Tree(t) => return t.single("withinTriangle")?.within_triangle_box(min_x, max_x, min_y, max_y, ax, ay, ab, bx, by, bc, cx, cy, ca),
        })
    }

    // Box-computing forms (`Component2D`'s default methods).

    pub fn intersects_line(&self, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        self.intersects_line_box(ax.min(bx), ax.max(bx), ay.min(by), ay.max(by), ax, ay, bx, by)
    }
    pub fn intersects_triangle(&self, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
        let (min_x, max_x, min_y, max_y) = tri_box!(ax, ay, bx, by, cx, cy);
        self.intersects_triangle_box(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy)
    }
    pub fn contains_line(&self, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        self.contains_line_box(ax.min(bx), ax.max(bx), ay.min(by), ay.max(by), ax, ay, bx, by)
    }
    pub fn contains_triangle(&self, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
        let (min_x, max_x, min_y, max_y) = tri_box!(ax, ay, bx, by, cx, cy);
        self.contains_triangle_box(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy)
    }
    pub fn within_line(&self, ax: f64, ay: f64, ab: bool, bx: f64, by: f64) -> Result<WithinRelation, &'static str> {
        self.within_line_box(ax.min(bx), ax.max(bx), ay.min(by), ay.max(by), ax, ay, ab, bx, by)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn within_triangle(&self, ax: f64, ay: f64, ab: bool, bx: f64, by: f64, bc: bool, cx: f64, cy: f64, ca: bool) -> Result<WithinRelation, &'static str> {
        let (min_x, max_x, min_y, max_y) = tri_box!(ax, ay, bx, by, cx, cy);
        self.within_triangle_box(min_x, max_x, min_y, max_y, ax, ay, ab, bx, by, bc, cx, cy, ca)
    }
}

/// The edge part of `Line2D` / `Polygon2D.withinTriangle`.
#[allow(clippy::too_many_arguments)]
fn edge_tree_within_triangle(tree: &EdgeTree, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, ab: bool, bx: f64, by: f64, bc: bool, cx: f64, cy: f64, ca: bool) -> WithinRelation {
    let mut relation = WithinRelation::Disjoint;
    for (x1, y1, x2, y2, from_poly) in [(ax, ay, bx, by, ab), (bx, by, cx, cy, bc), (cx, cy, ax, ay, ca)] {
        if tree.crosses_line(tree.root, min_x, max_x, min_y, max_y, x1, y1, x2, y2, true) {
            if from_poly {
                return WithinRelation::NotWithin;
            }
            relation = WithinRelation::Candidate;
        }
    }
    if relation == WithinRelation::Candidate {
        return WithinRelation::Candidate;
    }
    let (tx, ty) = tree.root_xy();
    if point_in_triangle(min_x, max_x, min_y, max_y, tx, ty, ax, ay, bx, by, cx, cy) {
        return WithinRelation::Candidate;
    }
    relation
}

impl Polygon2D {
    fn relate(&self, me: &Component, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        if disjoint(self.min_x, self.max_x, self.min_y, self.max_y, min_x, max_x, min_y, max_y) {
            return Relation::Outside;
        }
        if within(self.min_x, self.max_x, self.min_y, self.max_y, min_x, max_x, min_y, max_y) {
            return Relation::Crosses;
        }
        if let Some(h) = &self.holes {
            match h.relate(min_x, max_x, min_y, max_y) {
                Relation::Crosses => return Relation::Crosses,
                Relation::Inside => return Relation::Outside,
                Relation::Outside => {}
            }
        }
        let corners = self.number_of_corners(me, min_x, max_x, min_y, max_y);
        if corners == 4 {
            if self.tree.crosses_box(self.tree.root, min_x, max_x, min_y, max_y, true) {
                return Relation::Crosses;
            }
            return Relation::Inside;
        } else if corners == 0 {
            let (tx, ty) = self.tree.root_xy();
            if contains_point(tx, ty, min_x, max_x, min_y, max_y) {
                return Relation::Crosses;
            }
            if self.tree.crosses_box(self.tree.root, min_x, max_x, min_y, max_y, true) {
                return Relation::Crosses;
            }
            return Relation::Outside;
        }
        Relation::Crosses
    }

    fn number_of_corners(&self, me: &Component, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> u32 {
        let mut n = 0;
        if me.contains(min_x, min_y) {
            n += 1;
        }
        if me.contains(max_x, min_y) {
            n += 1;
        }
        if n == 1 {
            return n;
        }
        if me.contains(max_x, max_y) {
            n += 1;
        }
        if n == 2 {
            return n;
        }
        if me.contains(min_x, max_y) {
            n += 1;
        }
        n
    }
}

impl Rectangle2D {
    fn edges_intersect(&self, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        let (min_x, max_x, min_y, max_y) = (self.min_x, self.max_x, self.min_y, self.max_y);
        if ax.max(bx) < min_x || ax.min(bx) > max_x || ay.min(by) > max_y || ay.max(by) < min_y {
            return false;
        }
        line_crosses_line_with_boundary(ax, ay, bx, by, min_x, max_y, max_x, max_y)
            || line_crosses_line_with_boundary(ax, ay, bx, by, max_x, max_y, max_x, min_y)
            || line_crosses_line_with_boundary(ax, ay, bx, by, max_x, min_y, min_x, min_y)
            || line_crosses_line_with_boundary(ax, ay, bx, by, min_x, min_y, min_x, max_y)
    }
}

impl ComponentTree {
    fn single(&self, what: &'static str) -> Result<&Component, &'static str> {
        let n = &self.nodes[self.root as usize];
        if n.left != NONE || n.right != NONE {
            return Err(match what {
                "withinPoint" => "withinPoint is not supported for shapes with more than one component",
                "withinLine" => "withinLine is not supported for shapes with more than one component",
                _ => "withinTriangle is not supported for shapes with more than one component",
            });
        }
        Ok(&n.component)
    }

    #[inline]
    fn go_right(&self, n: &TreeNode, split_x: bool, max_x: f64, max_y: f64) -> bool {
        n.right != NONE && ((!split_x && max_y >= n.component.min_y()) || (split_x && max_x >= n.component.min_x()))
    }

    fn contains(&self, i: u32, x: f64, y: f64, split_x: bool) -> bool {
        let n = &self.nodes[i as usize];
        if y <= n.max_y && x <= n.max_x {
            if n.component.contains(x, y) {
                return true;
            }
            if n.left != NONE && self.contains(n.left, x, y, !split_x) {
                return true;
            }
            if self.go_right(n, split_x, x, y) {
                return self.contains(n.right, x, y, !split_x);
            }
        }
        false
    }

    fn relate(&self, i: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, split_x: bool) -> Relation {
        let n = &self.nodes[i as usize];
        if min_y <= n.max_y && min_x <= n.max_x {
            let r = n.component.relate(min_x, max_x, min_y, max_y);
            if r != Relation::Outside {
                return r;
            }
            if n.left != NONE {
                let r = self.relate(n.left, min_x, max_x, min_y, max_y, !split_x);
                if r != Relation::Outside {
                    return r;
                }
            }
            if self.go_right(n, split_x, max_x, max_y) {
                return self.relate(n.right, min_x, max_x, min_y, max_y, !split_x);
            }
        }
        Relation::Outside
    }

    #[allow(clippy::too_many_arguments)]
    fn intersects_line(&self, i: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, split_x: bool) -> bool {
        let n = &self.nodes[i as usize];
        if min_y <= n.max_y && min_x <= n.max_x {
            if n.component.intersects_line_box(min_x, max_x, min_y, max_y, ax, ay, bx, by) {
                return true;
            }
            if n.left != NONE && self.intersects_line(n.left, min_x, max_x, min_y, max_y, ax, ay, bx, by, !split_x) {
                return true;
            }
            if self.go_right(n, split_x, max_x, max_y) {
                return self.intersects_line(n.right, min_x, max_x, min_y, max_y, ax, ay, bx, by, !split_x);
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn intersects_triangle(&self, i: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, split_x: bool) -> bool {
        let n = &self.nodes[i as usize];
        if min_y <= n.max_y && min_x <= n.max_x {
            if n.component.intersects_triangle_box(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy) {
                return true;
            }
            if n.left != NONE && self.intersects_triangle(n.left, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, !split_x) {
                return true;
            }
            if self.go_right(n, split_x, max_x, max_y) {
                return self.intersects_triangle(n.right, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, !split_x);
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn contains_line(&self, i: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, split_x: bool) -> bool {
        let n = &self.nodes[i as usize];
        if min_y <= n.max_y && min_x <= n.max_x {
            if n.component.contains_line_box(min_x, max_x, min_y, max_y, ax, ay, bx, by) {
                return true;
            }
            if n.left != NONE && self.contains_line(n.left, min_x, max_x, min_y, max_y, ax, ay, bx, by, !split_x) {
                return true;
            }
            if self.go_right(n, split_x, max_x, max_y) {
                return self.contains_line(n.right, min_x, max_x, min_y, max_y, ax, ay, bx, by, !split_x);
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn contains_triangle(&self, i: u32, min_x: f64, max_x: f64, min_y: f64, max_y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, split_x: bool) -> bool {
        let n = &self.nodes[i as usize];
        if min_y <= n.max_y && min_x <= n.max_x {
            if n.component.contains_triangle_box(min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy) {
                return true;
            }
            if n.left != NONE && self.contains_triangle(n.left, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, !split_x) {
                return true;
            }
            if self.go_right(n, split_x, max_x, max_y) {
                return self.contains_triangle(n.right, min_x, max_x, min_y, max_y, ax, ay, bx, by, cx, cy, !split_x);
            }
        }
        false
    }

    /// `ComponentTree.create`: one component as is, several in a tree.
    pub fn create(mut components: Vec<Component>) -> Component {
        if components.len() == 1 {
            return components.pop().unwrap();
        }
        let mut items: Vec<Option<Component>> = components.into_iter().map(Some).collect();
        let mut nodes: Vec<TreeNode> = Vec::with_capacity(items.len());
        let n = items.len();
        let root = Self::build(&mut items, &mut nodes, 0, n as i64 - 1, false);
        let mut tree = ComponentTree { nodes, root };
        // The root's min box covers every component.
        let (mut mx, mut my) = (tree.nodes[root as usize].min_x, tree.nodes[root as usize].min_y);
        for nd in &tree.nodes {
            mx = mx.min(nd.component.min_x());
            my = my.min(nd.component.min_y());
        }
        tree.nodes[root as usize].min_x = mx;
        tree.nodes[root as usize].min_y = my;
        Component::Tree(tree)
    }

    fn build(items: &mut [Option<Component>], nodes: &mut Vec<TreeNode>, low: i64, high: i64, split_x: bool) -> u32 {
        if low > high {
            return NONE;
        }
        let mid = ((low + high) as u64 >> 1) as i64;
        if low < high {
            // `ArrayUtil.select` on the boxes: which component lands at
            // `mid` only orders the walk, never its answers.
            let slice = &mut items[low as usize..=high as usize];
            let key = |c: &Option<Component>| {
                let c = c.as_ref().unwrap();
                if split_x {
                    (c.min_x(), c.max_x())
                } else {
                    (c.min_y(), c.max_y())
                }
            };
            slice.sort_by(|a, b| {
                let (a, b) = (key(a), key(b));
                a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
            });
        }
        let comp = items[mid as usize].take().unwrap();
        let (min_x, max_x, min_y, max_y) = (comp.min_x(), comp.max_x(), comp.min_y(), comp.max_y());
        let me = nodes.len() as u32;
        nodes.push(TreeNode { min_x, max_x, min_y, max_y, component: comp, left: NONE, right: NONE });
        let left = Self::build(items, nodes, low, mid - 1, !split_x);
        let right = Self::build(items, nodes, mid + 1, high, !split_x);
        let (mut nmax_x, mut nmax_y) = (max_x, max_y);
        if left != NONE {
            nmax_x = nmax_x.max(nodes[left as usize].max_x);
            nmax_y = nmax_y.max(nodes[left as usize].max_y);
        }
        if right != NONE {
            nmax_x = nmax_x.max(nodes[right as usize].max_x);
            nmax_y = nmax_y.max(nodes[right as usize].max_y);
        }
        let nd = &mut nodes[me as usize];
        nd.left = left;
        nd.right = right;
        nd.max_x = nmax_x;
        nd.max_y = nmax_y;
        me
    }
}

// ---------------------------------------------------------------------------
// Lucene geometries (`org.apache.lucene.geo.{Point, Line, Polygon, Rectangle}`)

pub use crate::geo::{encode_lat_ceil, encode_lon_ceil};

/// A Lucene lat/lon geometry, as a query is built from (coordinates in
/// degrees; rings closed).
#[derive(Clone, Debug, PartialEq)]
pub enum LatLonGeometry {
    Point { lat: f64, lon: f64 },
    /// `(lon, lat)` vertices.
    Line(Vec<(f64, f64)>),
    /// Outer ring and holes, `(lon, lat)`, closed.
    Polygon(Vec<Vec<(f64, f64)>>),
    Rectangle { min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64 },
    /// Lucene `Circle` (centre, radius in meters).
    Circle { lat: f64, lon: f64, meters: f64 },
}

fn ring_bounds(r: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    // Lucene `Polygon` / `Line`: min / max over the vertices.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (r[0].0, r[0].0, r[0].1, r[0].1);
    for p in r {
        min_x = min_x.min(p.0);
        max_x = max_x.max(p.0);
        min_y = min_y.min(p.1);
        max_y = max_y.max(p.1);
    }
    (min_x, max_x, min_y, max_y)
}

impl LatLonGeometry {
    /// `toComponent2D()`.
    pub fn to_component(&self) -> Component {
        match self {
            LatLonGeometry::Point { lat, lon } => {
                let q_lat = if *lat == 90.0 { *lat } else { decode_lat(encode_lat_ceil(*lat)) };
                let q_lon = if *lon == 180.0 { *lon } else { decode_lon(encode_lon_ceil(*lon)) };
                Component::Point(Point2D { x: q_lon, y: q_lat })
            }
            LatLonGeometry::Line(pts) => {
                let (min_x, max_x, min_y, max_y) = ring_bounds(pts);
                let (xs, ys): (Vec<f64>, Vec<f64>) = pts.iter().copied().unzip();
                Component::Line(Line2D { min_x, max_x, min_y, max_y, tree: EdgeTree::create(&xs, &ys) })
            }
            LatLonGeometry::Polygon(rings) => {
                let outer = &rings[0];
                // Lucene `Polygon`'s box is over its vertices but the last
                // (the closing copy of the first).
                let (min_x, max_x, min_y, max_y) = ring_bounds(&outer[..outer.len().saturating_sub(1).max(1)]);
                let holes = if rings.len() > 1 {
                    let hs: Vec<Component> = rings[1..].iter().map(|h| LatLonGeometry::Polygon(vec![h.clone()]).to_component()).collect();
                    Some(Box::new(ComponentTree::create(hs)))
                } else {
                    None
                };
                let (xs, ys): (Vec<f64>, Vec<f64>) = outer.iter().copied().unzip();
                Component::Polygon(Polygon2D { min_x, max_x, min_y, max_y, holes, tree: EdgeTree::create(&xs, &ys) })
            }
            LatLonGeometry::Rectangle { min_lat, max_lat, min_lon, max_lon } => {
                let mut min_longitude = *min_lon;
                let mut crosses = *min_lon > *max_lon;
                if min_longitude == 180.0 && crosses {
                    min_longitude = -180.0;
                    crosses = false;
                }
                let q_min_lat = decode_lat(encode_lat_ceil(*min_lat));
                let q_max_lat = decode_lat(encode_lat(*max_lat));
                let q_min_lon = decode_lon(encode_lon_ceil(min_longitude));
                let q_max_lon = decode_lon(encode_lon(*max_lon));
                if crosses {
                    let west = Component::Rectangle(Rectangle2D { min_x: decode_lon(encode_lon(-180.0)), max_x: q_max_lon, min_y: q_min_lat, max_y: q_max_lat });
                    let east = Component::Rectangle(Rectangle2D { min_x: q_min_lon, max_x: decode_lon(encode_lon(180.0)), min_y: q_min_lat, max_y: q_max_lat });
                    ComponentTree::create(vec![west, east])
                } else {
                    Component::Rectangle(Rectangle2D { min_x: q_min_lon, max_x: q_max_lon, min_y: q_min_lat, max_y: q_max_lat })
                }
            }
            LatLonGeometry::Circle { lat, lon, meters } => Component::Circle(Circle2D::new(*lat, *lon, *meters)),
        }
    }
}

/// Lucene `Circle2D` with its `HaversinDistance` calculator.
#[derive(Clone, Debug)]
pub struct Circle2D {
    pub center_lat: f64,
    pub center_lon: f64,
    sort_key: f64,
    axis_lat: f64,
    pub min_lat: f64,
    pub max_lat: f64,
    min_lon: f64,
    max_lon: f64,
    crosses_dateline: bool,
}

const EARTH_MEAN_RADIUS_METERS: f64 = 6_371_008.7714;

fn sloppy_sin(a: f64) -> f64 {
    crate::geo::sloppy_cos(a - std::f64::consts::FRAC_PI_2)
}

impl Circle2D {
    pub fn new(lat: f64, lon: f64, meters: f64) -> Circle2D {
        use std::f64::consts::PI;
        let (min_lat_r, max_lat_r, min_lon_r, max_lon_r) = ((-90f64).to_radians(), 90f64.to_radians(), (-180f64).to_radians(), 180f64.to_radians());
        // `Rectangle.fromPointDistance`.
        let rad_lat = lat.to_radians();
        let rad_lon = lon.to_radians();
        let rad_distance = (meters + 7e-2) / EARTH_MEAN_RADIUS_METERS;
        let (mut min_lat, mut max_lat) = (rad_lat - rad_distance, rad_lat + rad_distance);
        let (min_lon, max_lon);
        if min_lat > min_lat_r && max_lat < max_lat_r {
            let delta_lon = crate::geo::sloppy_asin(sloppy_sin(rad_distance) / crate::geo::sloppy_cos(rad_lat));
            let mut a = rad_lon - delta_lon;
            if a < min_lon_r {
                a += 2.0 * PI;
            }
            let mut b = rad_lon + delta_lon;
            if b > max_lon_r {
                b -= 2.0 * PI;
            }
            (min_lon, max_lon) = (a, b);
        } else {
            min_lat = min_lat.max(min_lat_r);
            max_lat = max_lat.min(max_lat_r);
            (min_lon, max_lon) = (min_lon_r, max_lon_r);
        }
        let (min_lat, max_lat, min_lon, max_lon) = (min_lat.to_degrees(), max_lat.to_degrees(), min_lon.to_degrees(), max_lon.to_degrees());
        Circle2D { center_lat: lat, center_lon: lon, sort_key: crate::geo::distance_query_sort_key(meters), axis_lat: axis_lat(lat, meters), min_lat, max_lat, min_lon, max_lon, crosses_dateline: min_lon > max_lon }
    }

    fn min_x(&self) -> f64 {
        if self.crosses_dateline {
            -180.0
        } else {
            self.min_lon
        }
    }
    fn max_x(&self) -> f64 {
        if self.crosses_dateline {
            180.0
        } else {
            self.max_lon
        }
    }

    fn key(&self, x: f64, y: f64) -> f64 {
        crate::geo::sloppy_haversin_sort_key(y, x, self.center_lat, self.center_lon)
    }

    pub fn contains(&self, x: f64, y: f64) -> bool {
        let inb = if self.crosses_dateline {
            contains_point(x, y, self.min_lon, 180.0, self.min_lat, self.max_lat) || contains_point(x, y, -180.0, self.max_lon, self.min_lat, self.max_lat)
        } else {
            contains_point(x, y, self.min_lon, self.max_lon, self.min_lat, self.max_lat)
        };
        inb && self.key(x, y) <= self.sort_key
    }

    fn disjoint(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        if self.crosses_dateline {
            disjoint(self.min_lon, 180.0, self.min_lat, self.max_lat, min_x, max_x, min_y, max_y) && disjoint(-180.0, self.max_lon, self.min_lat, self.max_lat, min_x, max_x, min_y, max_y)
        } else {
            disjoint(self.min_lon, self.max_lon, self.min_lat, self.max_lat, min_x, max_x, min_y, max_y)
        }
    }

    fn within(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        if self.crosses_dateline {
            within(self.min_lon, 180.0, self.min_lat, self.max_lat, min_x, max_x, min_y, max_y) || within(-180.0, self.max_lon, self.min_lat, self.max_lat, min_x, max_x, min_y, max_y)
        } else {
            within(self.min_lon, self.max_lon, self.min_lat, self.max_lat, min_x, max_x, min_y, max_y)
        }
    }

    /// `GeoUtils.relate(minY, maxY, minX, maxX, lat, lon, sortKey, axisLat)`.
    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        const AXISLAT_ERROR: f64 = 0.1 / EARTH_MEAN_RADIUS_METERS * (180.0 / std::f64::consts::PI);
        let (lat, lon, k) = (self.center_lat, self.center_lon, self.sort_key);
        let hk = |la: f64, lo: f64| crate::geo::sloppy_haversin_sort_key(lat, lon, la, lo);
        if (lon < min_x || lon > max_x) && (self.axis_lat + AXISLAT_ERROR < min_y || self.axis_lat - AXISLAT_ERROR > max_y) && hk(min_y, min_x) > k && hk(min_y, max_x) > k && hk(max_y, min_x) > k && hk(max_y, max_x) > k {
            return Relation::Outside;
        }
        let within90 = {
            let mut l = lon;
            if max_x <= l - 180.0 {
                l -= 360.0;
            } else if min_x >= l + 180.0 {
                l += 360.0;
            }
            max_x - l < 90.0 && l - min_x < 90.0
        };
        if within90 && hk(min_y, min_x) <= k && hk(min_y, max_x) <= k && hk(max_y, min_x) <= k && hk(max_y, max_x) <= k {
            return Relation::Inside;
        }
        Relation::Crosses
    }

    /// `HaversinDistance.intersectsLine` (the centre shifted by 360 degrees
    /// too across the dateline).
    fn intersects_line(&self, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        if self.segment_hits(self.center_lon, ax, ay, bx, by) {
            return true;
        }
        if self.crosses_dateline {
            let c = if self.center_lon > 0.0 { self.center_lon - 360.0 } else { self.center_lon + 360.0 };
            return self.segment_hits(c, ax, ay, bx, by);
        }
        false
    }

    /// `Circle2D.intersectsLine(centerX, centerY, ...)`: the segment's point
    /// closest to the centre (planar) is in the circle.
    fn segment_hits(&self, center_x: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        let (apx, apy) = (center_x - ax, self.center_lat - ay);
        let (abx, aby) = (bx - ax, by - ay);
        let magnitude = abx * abx + aby * aby;
        let dot = apx * abx + apy * aby;
        let distance = dot / magnitude;
        if !(0.0..=1.0).contains(&distance) {
            return false;
        }
        let (px, py) = (ax + abx * distance, ay + aby * distance);
        if px >= ax.min(bx) && px <= ax.max(bx) && py >= ay.min(by) && py <= ay.max(by) {
            return self.contains(px, py);
        }
        false
    }
}

/// `Rectangle.axisLat`.
fn axis_lat(center_lat: f64, meters: f64) -> f64 {
    use std::f64::consts::FRAC_PI_2;
    let mut l1 = center_lat.to_radians();
    let r = (meters + 7e-2) / EARTH_MEAN_RADIUS_METERS;
    if l1.abs() + r >= 90f64.to_radians() {
        return if center_lat >= 0.0 { 90.0 } else { -90.0 };
    }
    l1 = if center_lat >= 0.0 { FRAC_PI_2 - l1 } else { l1 + FRAC_PI_2 };
    let mut l2 = (l1.cos() / r.cos()).acos();
    l2 = if center_lat >= 0.0 { FRAC_PI_2 - l2 } else { l2 - FRAC_PI_2 };
    l2.to_degrees()
}

/// `LatLonGeometry.create`: one geometry's component, or a tree of several.
pub fn create(geoms: &[LatLonGeometry]) -> Component {
    ComponentTree::create(geoms.iter().map(LatLonGeometry::to_component).collect())
}

/// Lucene `Polygon` validation as ES's `toLatLonPolygon` meets it.
fn check_ring(r: &[(f64, f64)]) -> Result<(), String> {
    if r.len() < 4 {
        return Err("at least 4 polygon points required".into());
    }
    let (f, l) = (r[0], r[r.len() - 1]);
    if f.1 != l.1 || f.0 != l.0 {
        return Err("first and last points of the polygon must be the same (it must close itself)".into());
    }
    Ok(())
}

/// `GeoUtils.quantizeLat` / `quantizeLon` (ES `LATLON_QUANTIZER`).
#[inline]
fn q(p: (f64, f64)) -> (f64, f64) {
    (decode_lon(encode_lon(p.0)), decode_lat(encode_lat(p.1)))
}

fn check_coord(p: (f64, f64)) -> Result<(), String> {
    if p.1.is_nan() || !(-90.0..=90.0).contains(&p.1) {
        return Err(format!("invalid latitude {}; must be between -90.0 and 90.0", p.1));
    }
    if p.0.is_nan() || !(-180.0..=180.0).contains(&p.0) {
        return Err(format!("invalid longitude {}; must be between -180.0 and 180.0", p.0));
    }
    Ok(())
}

/// ES `LuceneGeometriesUtils.toLatLonGeometry(geometry, quantize = true,
/// checker)` of a normalized query geometry (dateline already cut): the
/// Lucene geometries, quantized to the index grid. `within`: lines are
/// refused (`found an unsupported shape Line`).
pub fn query_geometries(g: &super::shape::Geometry, within_rel: bool) -> Result<Vec<LatLonGeometry>, String> {
    use super::shape::Geometry;
    let mut out = Vec::new();
    fn walk(g: &Geometry, within_rel: bool, out: &mut Vec<LatLonGeometry>) -> Result<(), String> {
        match g {
            Geometry::Point(p) => {
                check_coord(*p)?;
                let (lon, lat) = q(*p);
                out.push(LatLonGeometry::Point { lat, lon });
            }
            Geometry::MultiPoint(ps) => {
                for p in ps {
                    walk(&Geometry::Point(*p), within_rel, out)?;
                }
            }
            Geometry::LineString(l) => {
                if within_rel {
                    return Err("found an unsupported shape Line".into());
                }
                if !l.is_empty() {
                    for p in l {
                        check_coord(*p)?;
                    }
                    if l.len() < 2 {
                        return Err("at least 2 line points required".into());
                    }
                    out.push(LatLonGeometry::Line(l.iter().map(|p| q(*p)).collect()));
                }
            }
            Geometry::MultiLineString(ls) => {
                if within_rel {
                    return Err("found an unsupported shape Line".into());
                }
                for l in ls {
                    walk(&Geometry::LineString(l.clone()), within_rel, out)?;
                }
            }
            Geometry::Polygon(rings) => {
                if rings.first().is_some_and(|r| !r.is_empty()) {
                    let mut qr = Vec::with_capacity(rings.len());
                    for r in rings {
                        for p in r {
                            check_coord(*p)?;
                        }
                        let r: Vec<(f64, f64)> = r.iter().map(|p| q(*p)).collect();
                        check_ring(&r)?;
                        qr.push(r);
                    }
                    out.push(LatLonGeometry::Polygon(qr));
                }
            }
            Geometry::MultiPolygon(ps) => {
                for p in ps {
                    walk(&Geometry::Polygon(p.clone()), within_rel, out)?;
                }
            }
            Geometry::Collection(gs) => {
                for g in gs {
                    walk(g, within_rel, out)?;
                }
            }
            Geometry::Envelope { min_x, max_x, max_y, min_y } => {
                check_coord((*min_x, *min_y))?;
                check_coord((*max_x, *max_y))?;
                if max_y < min_y {
                    return Err(format!("maxLat cannot be less than minLat: {} < {}", max_y, min_y));
                }
                let (a, b) = (q((*min_x, *min_y)), q((*max_x, *max_y)));
                out.push(LatLonGeometry::Rectangle { min_lat: a.1, max_lat: b.1, min_lon: a.0, max_lon: b.0 });
            }
        }
        Ok(())
    }
    walk(g, within_rel, &mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// The encoded rectangle of `LatLonShapeBoundingBoxQuery`.

/// Lucene `MIN_LON_ENCODED` / `MAX_LON_ENCODED`.
pub fn min_lon_encoded() -> i32 {
    encode_lon(-180.0)
}
pub fn max_lon_encoded() -> i32 {
    encode_lon(180.0)
}

/// `SpatialQuery.EncodedRectangle` (`EncodedLatLonRectangle`): a query box
/// in encoded ints; `wraps` when it crosses the dateline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodedRect {
    pub min_x: i32,
    pub max_x: i32,
    pub min_y: i32,
    pub max_y: i32,
    pub wraps: bool,
}

impl EncodedRect {
    /// `EncodedLatLonRectangle(minLat, maxLat, minLon, maxLon)`.
    pub fn new(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> EncodedRect {
        let min_lon2 = if min_lon == 180.0 && min_lon > max_lon { -180.0 } else { min_lon };
        EncodedRect { min_x: encode_lon_ceil(min_lon2), max_x: encode_lon(max_lon), min_y: encode_lat_ceil(min_lat), max_y: encode_lat(max_lat), wraps: min_lon2 > max_lon }
    }

    /// The boxes the node relations compare (east, then west when wrapping):
    /// `(minY, minX, maxY, maxX)`.
    fn boxes(&self) -> ([i32; 4], Option<[i32; 4]>) {
        if self.wraps {
            ([self.min_y, self.min_x, self.max_y, max_lon_encoded()], Some([self.min_y, min_lon_encoded(), self.max_y, self.max_x]))
        } else {
            ([self.min_y, self.min_x, self.max_y, self.max_x], None)
        }
    }

    /// `relateRangeBBox` of a node: `min` / `max` are the node's lower and
    /// upper bounds of the four triangle-box dims (minY, minX, maxY, maxX).
    pub fn relate_range(&self, min: &[i32; 4], max: &[i32; 4]) -> Relation {
        let (east, west) = self.boxes();
        let r = compare_bbox_to_range(&east, min, max);
        match west {
            Some(w) if r == Relation::Outside => compare_bbox_to_range(&w, min, max),
            _ => r,
        }
    }

    /// `intersectRangeBBox` of a node (INTERSECTS / DISJOINT).
    pub fn intersect_range(&self, min: &[i32; 4], max: &[i32; 4]) -> Relation {
        let (east, west) = self.boxes();
        let r = intersect_bbox_with_range(&east, min, max);
        match west {
            Some(w) if r == Relation::Outside => intersect_bbox_with_range(&w, min, max),
            _ => r,
        }
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        if y < self.min_y || y > self.max_y {
            return false;
        }
        if self.wraps {
            !(x > self.max_x && x < self.min_x)
        } else {
            !(x > self.max_x || x < self.min_x)
        }
    }

    pub fn intersects_line(&self, ax: i32, ay: i32, bx: i32, by: i32) -> bool {
        if self.contains(ax, ay) || self.contains(bx, by) {
            return true;
        }
        if ay.max(by) < self.min_y || ay.min(by) > self.max_y {
            return false;
        }
        if self.wraps {
            if ax.min(bx) > self.max_x && ax.max(bx) < self.min_x {
                return false;
            }
        } else if ax.min(bx) > self.max_x || ax.max(bx) < self.min_x {
            return false;
        }
        self.edge_intersects_query(ax, ay, bx, by)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn intersects_triangle(&self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> bool {
        if self.contains(ax, ay) || self.contains(bx, by) || self.contains(cx, cy) {
            return true;
        }
        let t_min_y = ay.min(by).min(cy);
        let t_max_y = ay.max(by).max(cy);
        if t_max_y < self.min_y || t_min_y > self.max_y {
            return false;
        }
        let t_min_x = ax.min(bx).min(cx);
        let t_max_x = ax.max(bx).max(cx);
        if self.wraps {
            if t_min_x > self.max_x && t_max_x < self.min_x {
                return false;
            }
        } else if t_min_x > self.max_x || t_max_x < self.min_x {
            return false;
        }
        point_in_triangle(t_min_x as f64, t_max_x as f64, t_min_y as f64, t_max_y as f64, self.min_x as f64, self.min_y as f64, ax as f64, ay as f64, bx as f64, by as f64, cx as f64, cy as f64)
            || self.edge_intersects_query(ax, ay, bx, by)
            || self.edge_intersects_query(bx, by, cx, cy)
            || self.edge_intersects_query(cx, cy, ax, ay)
    }

    pub fn contains_line(&self, ax: i32, ay: i32, bx: i32, by: i32) -> bool {
        if ay < self.min_y || by < self.min_y || ay > self.max_y || by > self.max_y {
            return false;
        }
        if self.wraps {
            (ax >= self.min_x && bx >= self.min_x) || (ax <= self.max_x && bx <= self.max_x)
        } else {
            ax >= self.min_x && bx >= self.min_x && ax <= self.max_x && bx <= self.max_x
        }
    }

    pub fn contains_triangle(&self, ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> bool {
        if ay < self.min_y || by < self.min_y || cy < self.min_y || ay > self.max_y || by > self.max_y || cy > self.max_y {
            return false;
        }
        if self.wraps {
            (ax >= self.min_x && bx >= self.min_x && cx >= self.min_x) || (ax <= self.max_x && bx <= self.max_x && cx <= self.max_x)
        } else {
            ax >= self.min_x && bx >= self.min_x && cx >= self.min_x && ax <= self.max_x && bx <= self.max_x && cx <= self.max_x
        }
    }

    pub fn within_line(&self, ax: i32, ay: i32, ab: bool, bx: i32, by: i32) -> WithinRelation {
        if self.contains(ax, ay) || self.contains(bx, by) {
            return WithinRelation::NotWithin;
        }
        if ab && edge_intersects_box(ax, ay, bx, by, self.min_x, self.max_x, self.min_y, self.max_y) {
            return WithinRelation::NotWithin;
        }
        WithinRelation::Disjoint
    }

    #[allow(clippy::too_many_arguments)]
    pub fn within_triangle(&self, ax: i32, ay: i32, ab: bool, bx: i32, by: i32, bc: bool, cx: i32, cy: i32, ca: bool) -> WithinRelation {
        if self.contains(ax, ay) || self.contains(bx, by) || self.contains(cx, cy) {
            return WithinRelation::NotWithin;
        }
        let t_min_y = ay.min(by).min(cy);
        let t_max_y = ay.max(by).max(cy);
        if t_max_y < self.min_y || t_min_y > self.max_y {
            return WithinRelation::Disjoint;
        }
        let t_min_x = ax.min(bx).min(cx);
        let t_max_x = ax.max(bx).max(cx);
        if self.wraps {
            if t_min_x > self.max_x && t_max_x < self.min_x {
                return WithinRelation::Disjoint;
            }
        } else if t_min_x > self.max_x || t_max_x < self.min_x {
            return WithinRelation::Disjoint;
        }
        let mut relation = WithinRelation::Disjoint;
        for (x1, y1, x2, y2, from_poly) in [(ax, ay, bx, by, ab), (bx, by, cx, cy, bc), (cx, cy, ax, ay, ca)] {
            if edge_intersects_box(x1, y1, x2, y2, self.min_x, self.max_x, self.min_y, self.max_y) {
                if from_poly {
                    return WithinRelation::NotWithin;
                }
                relation = WithinRelation::Candidate;
            }
        }
        if relation == WithinRelation::Candidate
            || point_in_triangle(t_min_x as f64, t_max_x as f64, t_min_y as f64, t_max_y as f64, self.min_x as f64, self.min_y as f64, ax as f64, ay as f64, bx as f64, by as f64, cx as f64, cy as f64)
        {
            return WithinRelation::Candidate;
        }
        relation
    }

    fn edge_intersects_query(&self, ax: i32, ay: i32, bx: i32, by: i32) -> bool {
        if self.wraps {
            return edge_intersects_box(ax, ay, bx, by, min_lon_encoded(), self.max_x, self.min_y, self.max_y) || edge_intersects_box(ax, ay, bx, by, self.min_x, max_lon_encoded(), self.min_y, self.max_y);
        }
        edge_intersects_box(ax, ay, bx, by, self.min_x, self.max_x, self.min_y, self.max_y)
    }
}

/// `EncodedRectangle.edgeIntersectsBox` (ints compared as Lucene's `orient`
/// does: in doubles).
#[allow(clippy::too_many_arguments)]
fn edge_intersects_box(ax: i32, ay: i32, bx: i32, by: i32, min_x: i32, max_x: i32, min_y: i32, max_y: i32) -> bool {
    if ax.max(bx) < min_x || ax.min(bx) > max_x || ay.min(by) > max_y || ay.max(by) < min_y {
        return false;
    }
    let f = |v: i32| v as f64;
    let (ax, ay, bx, by, min_x, max_x, min_y, max_y) = (f(ax), f(ay), f(bx), f(by), f(min_x), f(max_x), f(min_y), f(max_y));
    line_crosses_line_with_boundary(ax, ay, bx, by, min_x, max_y, max_x, max_y)
        || line_crosses_line_with_boundary(ax, ay, bx, by, max_x, max_y, max_x, min_y)
        || line_crosses_line_with_boundary(ax, ay, bx, by, max_x, min_y, min_x, min_y)
        || line_crosses_line_with_boundary(ax, ay, bx, by, min_x, min_y, min_x, max_y)
}

/// `compareBBoxToRangeBBox`; `bbox` and the node bounds as
/// `(minY, minX, maxY, maxX)`. Encoded ints compare as Lucene's sortable
/// bytes do (signed order).
fn compare_bbox_to_range(bbox: &[i32; 4], min: &[i32; 4], max: &[i32; 4]) -> Relation {
    if range_disjoint(bbox, min, max) {
        return Relation::Outside;
    }
    if min[1] >= bbox[1] && max[3] <= bbox[3] && min[0] >= bbox[0] && max[2] <= bbox[2] {
        return Relation::Inside;
    }
    Relation::Crosses
}

/// `intersectBBoxWithRangeBBox`.
fn intersect_bbox_with_range(bbox: &[i32; 4], min: &[i32; 4], max: &[i32; 4]) -> Relation {
    if range_disjoint(bbox, min, max) {
        return Relation::Outside;
    }
    // Offsets: minX dim 1, minY dim 0, maxX dim 3, maxY dim 2.
    if min[1] >= bbox[1] && min[0] >= bbox[0] {
        if max[1] <= bbox[3] && max[2] <= bbox[2] {
            return Relation::Inside;
        }
        if max[3] <= bbox[3] && max[0] <= bbox[2] {
            return Relation::Inside;
        }
    }
    if max[3] <= bbox[3] && max[2] <= bbox[2] {
        if min[1] >= bbox[1] && min[2] >= bbox[0] {
            return Relation::Inside;
        }
        if min[3] >= bbox[1] && min[0] >= bbox[0] {
            return Relation::Inside;
        }
    }
    Relation::Crosses
}

fn range_disjoint(bbox: &[i32; 4], min: &[i32; 4], max: &[i32; 4]) -> bool {
    min[1] > bbox[3] || max[3] < bbox[1] || min[0] > bbox[2] || max[2] < bbox[0]
}

// ---------------------------------------------------------------------------
// Queries

/// The query `LatLonShape.newGeometryQuery(field, relation, geometries)`
/// builds, ready to relate triangles and tree nodes.
#[derive(Clone, Debug)]
pub enum ShapeQuery {
    /// `LatLonShapeQuery`: a component and the relation.
    Component(Component, QueryRelation),
    /// `LatLonShapeBoundingBoxQuery` (one rectangle).
    Rect(EncodedRect, Component, QueryRelation),
    /// A conjunction (CONTAINS over several geometries, or a dateline-
    /// crossing box).
    And(Vec<ShapeQuery>),
    /// No geometry: matches nothing.
    None,
}

impl ShapeQuery {
    /// `LatLonShape.newGeometryQuery`.
    pub fn new(geoms: &[LatLonGeometry], rel: QueryRelation) -> ShapeQuery {
        if geoms.is_empty() {
            return ShapeQuery::None;
        }
        if geoms.len() == 1 {
            return match &geoms[0] {
                LatLonGeometry::Rectangle { min_lat, max_lat, min_lon, max_lon } => Self::box_query(*min_lat, *max_lat, *min_lon, *max_lon, rel),
                g => ShapeQuery::Component(g.to_component(), rel),
            };
        }
        if rel == QueryRelation::Contains {
            return ShapeQuery::And(
                geoms
                    .iter()
                    .map(|g| match g {
                        LatLonGeometry::Rectangle { min_lat, max_lat, min_lon, max_lon } => Self::box_query(*min_lat, *max_lat, *min_lon, *max_lon, rel),
                        g => ShapeQuery::Component(g.to_component(), rel),
                    })
                    .collect(),
            );
        }
        ShapeQuery::Component(create(geoms), rel)
    }

    /// `LatLonShape.newBoxQuery`.
    fn box_query(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64, rel: QueryRelation) -> ShapeQuery {
        if rel == QueryRelation::Contains && min_lon > max_lon {
            return ShapeQuery::And(vec![Self::box_query(min_lat, max_lat, min_lon, 180.0, rel), Self::box_query(min_lat, max_lat, -180.0, max_lon, rel)]);
        }
        let g = LatLonGeometry::Rectangle { min_lat, max_lat, min_lon, max_lon };
        ShapeQuery::Rect(EncodedRect::new(min_lat, max_lat, min_lon, max_lon), g.to_component(), rel)
    }
}

/// One triangle as `ShapeField.DecodedTriangle` gives it: encoded vertices,
/// edge flags and its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tri {
    pub kind: super::tessellator::TriKind,
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

impl Tri {
    /// `ShapeField.decodeTriangle` of the seven encoded ints.
    #[inline]
    pub fn decode(t: &[i32; 7]) -> Tri {
        let d = super::tessellator::decode_triangle(t);
        Tri { kind: d.kind, ax: d.ax, ay: d.ay, bx: d.bx, by: d.by, cx: d.cx, cy: d.cy, ab: d.ab, bc: d.bc, ca: d.ca }
    }
}

/// The per-triangle tests of one leaf query (`SpatialVisitor`'s
/// `intersects()` / `within()` / `contains()` and the node relation).
pub trait LeafQuery {
    /// The node relation (already transposed for DISJOINT, as Lucene's
    /// `getInnerFunction`): `min` / `max` over the node's triangle boxes
    /// `(minY, minX, maxY, maxX)`.
    fn relate_node(&self, min: &[i32; 4], max: &[i32; 4]) -> Relation;
    /// `getLeafPredicate(relation)` (DISJOINT negates intersects).
    fn leaf_matches(&self, t: &Tri) -> bool;
    /// `contains()` (CONTAINS only).
    fn leaf_within(&self, t: &Tri) -> Result<WithinRelation, String>;
    fn relation(&self) -> QueryRelation;
}

#[inline]
fn dlat(v: i32) -> f64 {
    decode_lat(v)
}
#[inline]
fn dlon(v: i32) -> f64 {
    decode_lon(v)
}

/// Transposed relation (INSIDE <-> OUTSIDE).
#[inline]
pub fn transpose(r: Relation) -> Relation {
    match r {
        Relation::Inside => Relation::Outside,
        Relation::Outside => Relation::Inside,
        Relation::Crosses => Relation::Crosses,
    }
}

/// `LatLonShapeQuery`'s visitor over a component.
pub struct ComponentLeaf<'a> {
    pub component: &'a Component,
    pub rel: QueryRelation,
}

impl LeafQuery for ComponentLeaf<'_> {
    fn relate_node(&self, min: &[i32; 4], max: &[i32; 4]) -> Relation {
        let r = self.component.relate(dlon(min[1]), dlon(max[3]), dlat(min[0]), dlat(max[2]));
        if self.rel == QueryRelation::Disjoint {
            transpose(r)
        } else {
            r
        }
    }

    fn leaf_matches(&self, t: &Tri) -> bool {
        use super::tessellator::TriKind;
        let c = self.component;
        let intersects = |t: &Tri| match t.kind {
            TriKind::Point => c.contains(dlon(t.ax), dlat(t.ay)),
            TriKind::Line => c.intersects_line(dlon(t.ax), dlat(t.ay), dlon(t.bx), dlat(t.by)),
            TriKind::Triangle => c.intersects_triangle(dlon(t.ax), dlat(t.ay), dlon(t.bx), dlat(t.by), dlon(t.cx), dlat(t.cy)),
        };
        match self.rel {
            QueryRelation::Intersects => intersects(t),
            QueryRelation::Disjoint => !intersects(t),
            QueryRelation::Within => match t.kind {
                TriKind::Point => c.contains(dlon(t.ax), dlat(t.ay)),
                TriKind::Line => c.contains_line(dlon(t.ax), dlat(t.ay), dlon(t.bx), dlat(t.by)),
                TriKind::Triangle => c.contains_triangle(dlon(t.ax), dlat(t.ay), dlon(t.bx), dlat(t.by), dlon(t.cx), dlat(t.cy)),
            },
            QueryRelation::Contains => self.leaf_within(t).is_ok_and(|w| w == WithinRelation::Candidate),
        }
    }

    fn leaf_within(&self, t: &Tri) -> Result<WithinRelation, String> {
        use super::tessellator::TriKind;
        let c = self.component;
        match t.kind {
            TriKind::Point => c.within_point(dlon(t.ax), dlat(t.ay)),
            TriKind::Line => c.within_line(dlon(t.ax), dlat(t.ay), t.ab, dlon(t.bx), dlat(t.by)),
            TriKind::Triangle => c.within_triangle(dlon(t.ax), dlat(t.ay), t.ab, dlon(t.bx), dlat(t.by), t.bc, dlon(t.cx), dlat(t.cy), t.ca),
        }
        .map_err(str::to_string)
    }

    fn relation(&self) -> QueryRelation {
        self.rel
    }
}

/// `LatLonShapeBoundingBoxQuery`'s visitor.
pub struct RectLeaf<'a> {
    pub rect: &'a EncodedRect,
    pub rel: QueryRelation,
}

impl LeafQuery for RectLeaf<'_> {
    fn relate_node(&self, min: &[i32; 4], max: &[i32; 4]) -> Relation {
        let r = if matches!(self.rel, QueryRelation::Intersects | QueryRelation::Disjoint) { self.rect.intersect_range(min, max) } else { self.rect.relate_range(min, max) };
        if self.rel == QueryRelation::Disjoint {
            transpose(r)
        } else {
            r
        }
    }

    fn leaf_matches(&self, t: &Tri) -> bool {
        use super::tessellator::TriKind;
        let r = self.rect;
        let intersects = |t: &Tri| match t.kind {
            TriKind::Point => r.contains(t.ax, t.ay),
            TriKind::Line => r.intersects_line(t.ax, t.ay, t.bx, t.by),
            TriKind::Triangle => r.intersects_triangle(t.ax, t.ay, t.bx, t.by, t.cx, t.cy),
        };
        match self.rel {
            QueryRelation::Intersects => intersects(t),
            QueryRelation::Disjoint => !intersects(t),
            QueryRelation::Within => match t.kind {
                TriKind::Point => r.contains(t.ax, t.ay),
                TriKind::Line => r.contains_line(t.ax, t.ay, t.bx, t.by),
                TriKind::Triangle => r.contains_triangle(t.ax, t.ay, t.bx, t.by, t.cx, t.cy),
            },
            QueryRelation::Contains => self.leaf_within(t).is_ok_and(|w| w == WithinRelation::Candidate),
        }
    }

    fn leaf_within(&self, t: &Tri) -> Result<WithinRelation, String> {
        use super::tessellator::TriKind;
        let r = self.rect;
        if r.wraps {
            return Err("withinTriangle is not supported for rectangles crossing the date line".into());
        }
        Ok(match t.kind {
            TriKind::Point => {
                if r.contains(t.ax, t.ay) {
                    WithinRelation::NotWithin
                } else {
                    WithinRelation::Disjoint
                }
            }
            TriKind::Line => r.within_line(t.ax, t.ay, t.ab, t.bx, t.by),
            TriKind::Triangle => r.within_triangle(t.ax, t.ay, t.ab, t.bx, t.by, t.bc, t.cx, t.cy, t.ca),
        })
    }

    fn relation(&self) -> QueryRelation {
        self.rel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> LatLonGeometry {
        LatLonGeometry::Polygon(vec![vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]])
    }

    #[test]
    fn polygon_relations() {
        let c = square(0.0, 0.0, 10.0, 10.0).to_component();
        assert!(c.contains(5.0, 5.0));
        assert!(c.contains(0.0, 5.0), "on the edge");
        assert!(!c.contains(11.0, 5.0));
        assert_eq!(c.relate(1.0, 2.0, 1.0, 2.0), Relation::Inside);
        assert_eq!(c.relate(20.0, 30.0, 1.0, 2.0), Relation::Outside);
        assert_eq!(c.relate(-1.0, 2.0, 1.0, 2.0), Relation::Crosses);
        assert_eq!(c.relate(-1.0, 20.0, -1.0, 20.0), Relation::Crosses, "query inside the box");
        assert!(c.intersects_triangle(9.0, 9.0, 20.0, 9.0, 20.0, 20.0));
        assert!(!c.intersects_triangle(11.0, 11.0, 20.0, 11.0, 20.0, 20.0));
        assert!(c.contains_triangle(1.0, 1.0, 2.0, 1.0, 2.0, 2.0));
        assert!(!c.contains_triangle(1.0, 1.0, 20.0, 1.0, 2.0, 2.0));
        assert!(c.intersects_line(-5.0, 5.0, 15.0, 5.0));
        assert!(!c.contains_line(-5.0, 5.0, 15.0, 5.0));
        // A hole.
        let h = LatLonGeometry::Polygon(vec![vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)], vec![(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)]]).to_component();
        assert!(!h.contains(5.0, 5.0));
        assert_eq!(h.relate(4.5, 5.5, 4.5, 5.5), Relation::Outside);
        assert!(!h.intersects_triangle(4.5, 4.5, 5.5, 4.5, 5.0, 5.5));
    }

    #[test]
    fn rectangle_and_tree() {
        let r = LatLonGeometry::Rectangle { min_lat: 0.0, max_lat: 10.0, min_lon: 170.0, max_lon: -170.0 }.to_component();
        assert!(matches!(r, Component::Tree(_)));
        assert!(r.contains(175.0, 5.0) && r.contains(-175.0, 5.0) && !r.contains(0.0, 5.0));
        assert!(r.within_point(175.0, 5.0).is_err());
        let t = create(&[square(0.0, 0.0, 1.0, 1.0), square(5.0, 5.0, 6.0, 6.0), LatLonGeometry::Point { lat: 20.0, lon: 20.0 }]);
        assert!(t.contains(0.5, 0.5) && t.contains(5.5, 5.5) && !t.contains(3.0, 3.0));
        assert_eq!(t.relate(0.2, 0.4, 0.2, 0.4), Relation::Inside);
        assert_eq!(t.relate(2.0, 3.0, 2.0, 3.0), Relation::Outside);
        assert!(t.intersects_triangle(19.0, 19.0, 21.0, 19.0, 20.0, 21.0));
    }

    #[test]
    fn encoded_rect() {
        let r = EncodedRect::new(0.0, 10.0, 0.0, 10.0);
        let (x, y) = (encode_lon(5.0), encode_lat(5.0));
        assert!(r.contains(x, y));
        assert!(r.intersects_triangle(x, y, encode_lon(20.0), y, x, encode_lat(20.0)));
        assert!(!r.contains_triangle(x, y, encode_lon(20.0), y, x, encode_lat(20.0)));
        let node_min = [encode_lat(1.0), encode_lon(1.0), encode_lat(2.0), encode_lon(2.0)];
        let node_max = [encode_lat(3.0), encode_lon(3.0), encode_lat(4.0), encode_lon(4.0)];
        assert_eq!(r.relate_range(&node_min, &node_max), Relation::Inside);
        assert_eq!(r.intersect_range(&node_min, &node_max), Relation::Inside);
        let w = EncodedRect::new(0.0, 10.0, 170.0, -170.0);
        assert!(w.wraps && w.contains(encode_lon(175.0), y) && !w.contains(encode_lon(0.0), y));
    }

    /// Circle2D against the sloppy haversine distance: points in it are
    /// within the radius (its sort key), and a box it relates as inside
    /// (outside) has its sampled points all in (none in) the circle.
    #[test]
    fn circle2d_matches_haversine() {
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for &(lat, lon, m) in &[(51.5, -0.1, 30_000.0), (0.0, 179.9, 500_000.0), (89.0, 10.0, 300_000.0), (-30.0, 20.0, 5.0)] {
            let c = Circle2D::new(lat, lon, m);
            let key = crate::geo::distance_query_sort_key(m);
            let deg = m / 111_000.0 * 2.0 + 0.01;
            for _ in 0..4000 {
                let y = (lat + (rnd() * 2.0 - 1.0) * deg).clamp(-90.0, 90.0);
                let mut x = lon + (rnd() * 2.0 - 1.0) * deg * 3.0;
                if x > 180.0 {
                    x -= 360.0;
                }
                if x < -180.0 {
                    x += 360.0;
                }
                let inside = crate::geo::sloppy_haversin_sort_key(y, x, lat, lon) <= key;
                assert_eq!(Component::Circle(c.clone()).contains(x, y), inside, "{lat},{lon} {m}: {y},{x}");
            }
            for _ in 0..300 {
                let (x0, y0) = (lon + (rnd() * 2.0 - 1.0) * deg, (lat + (rnd() * 2.0 - 1.0) * deg).clamp(-90.0, 89.0));
                let (w, h) = (rnd() * deg / 2.0, rnd() * deg / 2.0);
                let (min_x, max_x, min_y, max_y) = (x0.max(-180.0), (x0 + w).min(180.0), y0, (y0 + h).min(90.0));
                if min_x > max_x {
                    continue;
                }
                let r = Component::Circle(c.clone()).relate(min_x, max_x, min_y, max_y);
                for i in 0..5 {
                    for j in 0..5 {
                        let (x, y) = (min_x + (max_x - min_x) * i as f64 / 4.0, min_y + (max_y - min_y) * j as f64 / 4.0);
                        let p = Component::Circle(c.clone()).contains(x, y);
                        match r {
                            Relation::Inside => assert!(p, "inside box with a point out"),
                            Relation::Outside => assert!(!p, "outside box with a point in"),
                            Relation::Crosses => {}
                        }
                    }
                }
            }
        }
    }
}
