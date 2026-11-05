//! Shapes for ES `geo_shape`, `shape`, `point` (and query geometries):
//! GeoJSON / WKT parsing, a doc-value encoding with a bounding-box prefix
//! for cheap rejection, and the spatial relations `intersects`,
//! `disjoint`, `within`, `contains` evaluated exactly on the planar
//! geometry (coordinates are `(x, y)` = `(lon, lat)` for geo shapes).
//!
//! Lucene indexes tessellated triangles in a BKD tree; here the shape is
//! kept whole in doc values and relations are computed directly, with the
//! bounding box as the approximation.

use serde_json::Value;

pub type Pt = (f64, f64);

#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    Point(Pt),
    MultiPoint(Vec<Pt>),
    LineString(Vec<Pt>),
    MultiLineString(Vec<Vec<Pt>>),
    /// Outer ring first, then holes; rings are closed.
    Polygon(Vec<Vec<Pt>>),
    MultiPolygon(Vec<Vec<Vec<Pt>>>),
    Collection(Vec<Geometry>),
    /// `(min_x, max_x, max_y, min_y)` like ES `envelope` / WKT `BBOX`.
    Envelope { min_x: f64, max_x: f64, max_y: f64, min_y: f64 },
}

/// Spatial relation of an indexed shape to the query shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialRelation {
    Intersects,
    Disjoint,
    /// The indexed shape is within the query shape.
    Within,
    /// The indexed shape contains the query shape.
    Contains,
}

impl SpatialRelation {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "intersects" => SpatialRelation::Intersects,
            "disjoint" => SpatialRelation::Disjoint,
            "within" => SpatialRelation::Within,
            "contains" => SpatialRelation::Contains,
            _ => return None,
        })
    }
}

/// Parse options.
#[derive(Clone, Copy, Debug)]
pub struct ParseOpts {
    /// Geographic coordinates: validate lat/lon ranges.
    pub geo: bool,
    /// Close unclosed polygon rings instead of failing.
    pub coerce: bool,
    /// Accept (and drop) a third coordinate.
    pub ignore_z: bool,
}

impl Default for ParseOpts {
    fn default() -> Self {
        Self { geo: true, coerce: false, ignore_z: true }
    }
}

type R<T> = Result<T, String>;

thread_local! {
    /// Set while parsing a *query* shape: ES does not range-check those.
    static NO_RANGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set while decoding a stored doc value ([`decode`]): values written
    /// before lines needed two points still read back.
    static LENIENT_LINES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Parse a query geometry (`geo_shape` query): like [`parse`] for geographic
/// shapes but coordinates are not range-checked and open rings are closed.
pub fn parse_query_shape(v: &Value) -> R<Geometry> {
    let prev = NO_RANGE.with(|c| c.replace(true));
    let r = parse(v, &ParseOpts { geo: true, coerce: true, ignore_z: true });
    NO_RANGE.with(|c| c.set(prev));
    r
}

fn num(v: &Value) -> R<f64> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| "numeric value expected".to_string()),
        Value::String(s) => s.trim().parse().map_err(|_| format!("numeric value expected, got [{s}]")),
        _ => Err("numeric value expected".into()),
    }
}

fn pt(v: &Value, o: &ParseOpts) -> R<Pt> {
    let a = v.as_array().ok_or_else(|| "geo coordinates must be numbers".to_string())?;
    if a.len() < 2 {
        return Err("geo coordinates must be numbers".into());
    }
    if a.len() > 2 && !o.ignore_z {
        return Err(format!("found Z value [{}] but [ignore_z_value] parameter is [false]", a[2]));
    }
    check((num(&a[0])?, num(&a[1])?), o)
}

fn check(p: Pt, o: &ParseOpts) -> R<Pt> {
    if !p.0.is_finite() || !p.1.is_finite() {
        return Err("coordinates must be finite".into());
    }
    if o.geo && !NO_RANGE.with(|c| c.get()) {
        if !(-90.0..=90.0).contains(&p.1) {
            return Err(format!("invalid latitude {}; must be between -90.0 and 90.0", fmtf(p.1)));
        }
        if !(-180.0..=180.0).contains(&p.0) {
            return Err(format!("invalid longitude {}; must be between -180.0 and 180.0", fmtf(p.0)));
        }
    }
    Ok(p)
}

fn fmtf(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{x:.1}")
    } else {
        x.to_string()
    }
}

fn pts(v: &Value, o: &ParseOpts) -> R<Vec<Pt>> {
    v.as_array().ok_or_else(|| "coordinates must be an array".to_string())?.iter().map(|x| pt(x, o)).collect()
}

fn ring(mut r: Vec<Pt>, o: &ParseOpts) -> R<Vec<Pt>> {
    if r.first() != r.last() || r.len() < 2 {
        if o.coerce && !r.is_empty() {
            r.push(r[0]);
        } else if let (Some(a), Some(b)) = (r.first(), r.last()) {
            return Err(format!(
                "first and last points of the linear ring must be the same (it must close itself): x[0]={}, x[{}]={}, y[0]={}, y[{}]={}",
                fmtf(a.0),
                r.len() - 1,
                fmtf(b.0),
                fmtf(a.1),
                r.len() - 1,
                fmtf(b.1)
            ));
        }
    }
    if r.len() < 4 {
        return Err(format!("invalid number of points in LinearRing (found [{}] - must be >= 4)", r.len()));
    }
    Ok(r)
}

fn rings(v: &Value, o: &ParseOpts) -> R<Vec<Vec<Pt>>> {
    let a = v.as_array().ok_or_else(|| "coordinates must be an array".to_string())?;
    if a.is_empty() {
        return Err("Invalid LinearRing found. Found a single coordinate when expecting a coordinate array".into());
    }
    a.iter().map(|r| ring(pts(r, o)?, o)).collect()
}

fn parse_distance(v: &Value) -> R<f64> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => {
            let s = s.trim();
            let units: &[(&str, f64)] = &[
                ("km", 1000.0), ("kilometers", 1000.0), ("mi", 1609.344), ("miles", 1609.344), ("yd", 0.9144), ("yards", 0.9144),
                ("ft", 0.3048), ("feet", 0.3048), ("in", 0.0254), ("inch", 0.0254), ("cm", 0.01), ("mm", 0.001), ("nmi", 1852.0),
                ("NM", 1852.0), ("m", 1.0), ("meters", 1.0),
            ];
            for (u, f) in units {
                if let Some(n) = s.strip_suffix(u) {
                    if let Ok(x) = n.trim().parse::<f64>() {
                        return Ok(x * f);
                    }
                }
            }
            s.parse().map_err(|_| format!("failed to parse distance [{s}]"))
        }
        _ => Err("radius must be a distance".into()),
    }
}

/// A circle as a polygon (radius in meters for geo, in units otherwise).
pub fn circle_polygon(c: Pt, radius: f64, geo: bool) -> Geometry {
    let n = 96;
    let mut r = Vec::with_capacity(n + 1);
    for i in 0..n {
        let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        if geo {
            // Destination point on the sphere (same radius as the arc distance).
            const R: f64 = 6_371_008.7714;
            let d = radius / R;
            let (lat1, lon1) = (c.1.to_radians(), c.0.to_radians());
            let lat2 = (lat1.sin() * d.cos() + lat1.cos() * d.sin() * a.cos()).clamp(-1.0, 1.0).asin();
            let lon2 = lon1 + (a.sin() * d.sin() * lat1.cos()).atan2(d.cos() - lat1.sin() * lat2.sin());
            let mut lon = lon2.to_degrees();
            while lon > 180.0 {
                lon -= 360.0;
            }
            while lon < -180.0 {
                lon += 360.0;
            }
            r.push((lon, lat2.to_degrees()));
        } else {
            r.push((c.0 + radius * a.cos(), c.1 + radius * a.sin()));
        }
    }
    r.push(r[0]);
    Geometry::Polygon(vec![r])
}

/// Parse GeoJSON (object) or WKT (string).
pub fn parse(v: &Value, o: &ParseOpts) -> R<Geometry> {
    match v {
        Value::String(s) => parse_wkt(s, o),
        Value::Object(m) => {
            let ty = m.get("type").and_then(Value::as_str).ok_or_else(|| "shape type not included".to_string())?.to_ascii_lowercase();
            if ty == "geometrycollection" {
                let gs = m.get("geometries").and_then(Value::as_array).ok_or_else(|| "geometries not included".to_string())?;
                return Ok(Geometry::Collection(gs.iter().map(|g| parse(g, o)).collect::<R<_>>()?));
            }
            let c = m.get("coordinates").ok_or_else(|| format!("coordinates not included"))?;
            Ok(match ty.as_str() {
                "point" => Geometry::Point(pt(c, o)?),
                "multipoint" => Geometry::MultiPoint(pts(c, o)?),
                "linestring" => {
                    let l = pts(c, o)?;
                    if l.len() < 2 {
                        return Err(format!("invalid number of points in LineString (found [{}] - must be >= 2)", l.len()));
                    }
                    Geometry::LineString(l)
                }
                // ES `GeoJson` `asLineString`: every member needs two points.
                "multilinestring" => Geometry::MultiLineString(
                    c.as_array()
                        .ok_or("coordinates must be an array")?
                        .iter()
                        .map(|l| {
                            let l = pts(l, o)?;
                            if l.len() < 2 {
                                return Err("not enough points to build a line".to_string());
                            }
                            Ok(l)
                        })
                        .collect::<R<_>>()?,
                ),
                "polygon" => Geometry::Polygon(rings(c, o)?),
                "multipolygon" => Geometry::MultiPolygon(c.as_array().ok_or("coordinates must be an array")?.iter().map(|p| rings(p, o)).collect::<R<_>>()?),
                "envelope" => {
                    let a = pts(c, o)?;
                    if a.len() != 2 {
                        return Err(format!("invalid number of points [{}] provided for geo_shape [ENVELOPE] when expecting an array of 2 coordinates", a.len()));
                    }
                    if a[0].1 < a[1].1 {
                        return Err("invalid envelope: maxY is below minY".into());
                    }
                    Geometry::Envelope { min_x: a[0].0, max_x: a[1].0, max_y: a[0].1, min_y: a[1].1 }
                }
                "circle" => {
                    let center = pt(c, o)?;
                    let r = parse_distance(m.get("radius").ok_or("radius is not specified")?)?;
                    circle_polygon(center, r, o.geo)
                }
                other => return Err(format!("unknown geometry type: {other}")),
            })
        }
        _ => Err("shape must be an object consisting of type and coordinates".into()),
    }
}

// ---------------------------------------------------------------------------
// WKT.

struct Wkt<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Wkt<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, c: u8) -> R<()> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' but found: {}", c as char, self.rest()))
        }
    }
    fn rest(&self) -> String {
        String::from_utf8_lossy(&self.s[self.i.min(self.s.len())..]).into_owned()
    }
    fn word(&mut self) -> String {
        self.ws();
        let st = self.i;
        while self.i < self.s.len() && self.s[self.i].is_ascii_alphabetic() {
            self.i += 1;
        }
        String::from_utf8_lossy(&self.s[st..self.i]).to_ascii_uppercase()
    }
    fn number(&mut self) -> R<f64> {
        self.ws();
        let st = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || b"+-.eE".contains(&self.s[self.i])) {
            self.i += 1;
        }
        let t = String::from_utf8_lossy(&self.s[st..self.i]);
        t.parse().map_err(|_| format!("expected number but found: {}", self.rest()))
    }
    fn empty(&mut self) -> bool {
        let save = self.i;
        if self.word() == "EMPTY" {
            true
        } else {
            self.i = save;
            false
        }
    }
    fn coord(&mut self, o: &ParseOpts) -> R<Pt> {
        let x = self.number()?;
        let y = self.number()?;
        if matches!(self.peek(), Some(c) if c == b'-' || c == b'+' || c == b'.' || c.is_ascii_digit()) {
            let z = self.number()?;
            if !o.ignore_z {
                return Err(format!("found Z value [{z}] but [ignore_z_value] parameter is [false]"));
            }
        }
        check((x, y), o)
    }
    fn coords(&mut self, o: &ParseOpts) -> R<Vec<Pt>> {
        self.eat(b'(')?;
        // Sized from the commas before the closing parenthesis (one more
        // for a ring the parse may close), not grown point by point.
        let n = self.s[self.i..].iter().take_while(|&&c| c != b')').filter(|&&c| c == b',').count() + 2;
        let mut v = Vec::with_capacity(n);
        v.push(self.coord(o)?);
        while self.peek() == Some(b',') {
            self.i += 1;
            v.push(self.coord(o)?);
        }
        self.eat(b')')?;
        Ok(v)
    }
    fn list<T>(&mut self, mut f: impl FnMut(&mut Self) -> R<T>) -> R<Vec<T>> {
        self.eat(b'(')?;
        let mut v = vec![f(self)?];
        while self.peek() == Some(b',') {
            self.i += 1;
            v.push(f(self)?);
        }
        self.eat(b')')?;
        Ok(v)
    }
    fn geometry(&mut self, o: &ParseOpts) -> R<Geometry> {
        let w = self.word();
        if self.empty() {
            return Ok(match w.as_str() {
                "GEOMETRYCOLLECTION" => Geometry::Collection(vec![]),
                "MULTIPOINT" => Geometry::MultiPoint(vec![]),
                "MULTILINESTRING" => Geometry::MultiLineString(vec![]),
                "MULTIPOLYGON" => Geometry::MultiPolygon(vec![]),
                _ => Geometry::Collection(vec![]),
            });
        }
        Ok(match w.as_str() {
            "POINT" => {
                self.eat(b'(')?;
                let p = self.coord(o)?;
                self.eat(b')')?;
                Geometry::Point(p)
            }
            "MULTIPOINT" => {
                // Both `(1 2, 3 4)` and `((1 2), (3 4))`.
                self.eat(b'(')?;
                let mut v = Vec::new();
                loop {
                    if self.peek() == Some(b'(') {
                        self.i += 1;
                        v.push(self.coord(o)?);
                        self.eat(b')')?;
                    } else {
                        v.push(self.coord(o)?);
                    }
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                self.eat(b')')?;
                Geometry::MultiPoint(v)
            }
            "LINESTRING" => Geometry::LineString(line(self.coords(o)?)?),
            "MULTILINESTRING" => Geometry::MultiLineString(self.list(|p| line(p.coords(o)?))?),
            "POLYGON" => Geometry::Polygon(self.list(|p| ring(p.coords(o)?, o))?),
            "MULTIPOLYGON" => Geometry::MultiPolygon(self.list(|p| p.list(|q| ring(q.coords(o)?, o)))?),
            "GEOMETRYCOLLECTION" => Geometry::Collection(self.list(|p| p.geometry(o))?),
            "BBOX" | "ENVELOPE" => {
                self.eat(b'(')?;
                let min_x = self.number()?;
                self.eat(b',')?;
                let max_x = self.number()?;
                self.eat(b',')?;
                let max_y = self.number()?;
                self.eat(b',')?;
                let min_y = self.number()?;
                self.eat(b')')?;
                check((min_x, min_y), o)?;
                check((max_x, max_y), o)?;
                Geometry::Envelope { min_x, max_x, max_y, min_y }
            }
            "" => return Err(format!("Unknown geometry type: {}", self.rest())),
            other => return Err(format!("Unknown geometry type: {}", other.to_lowercase())),
        })
    }
}

/// ES `Line`: a WKT line needs at least two points.
fn line(l: Vec<Pt>) -> R<Vec<Pt>> {
    if l.len() < 2 && !LENIENT_LINES.with(std::cell::Cell::get) {
        return Err("at least two points in the line is required".into());
    }
    Ok(l)
}

pub fn parse_wkt(s: &str, o: &ParseOpts) -> R<Geometry> {
    let mut p = Wkt { s: s.as_bytes(), i: 0 };
    let g = p.geometry(o)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("expected end of WKT string but found additional text: {}", p.rest()));
    }
    Ok(g)
}

fn wkt_pts(v: &[Pt]) -> String {
    v.iter().map(|p| format!("{} {}", p.0, p.1)).collect::<Vec<_>>().join(", ")
}

impl Geometry {
    /// WKT of the geometry.
    pub fn to_wkt(&self) -> String {
        match self {
            Geometry::Point(p) => format!("POINT ({} {})", p.0, p.1),
            Geometry::MultiPoint(v) if v.is_empty() => "MULTIPOINT EMPTY".into(),
            Geometry::MultiPoint(v) => format!("MULTIPOINT ({})", wkt_pts(v)),
            Geometry::LineString(v) => format!("LINESTRING ({})", wkt_pts(v)),
            Geometry::MultiLineString(v) if v.is_empty() => "MULTILINESTRING EMPTY".into(),
            Geometry::MultiLineString(v) => format!("MULTILINESTRING ({})", v.iter().map(|l| format!("({})", wkt_pts(l))).collect::<Vec<_>>().join(", ")),
            Geometry::Polygon(r) => format!("POLYGON ({})", r.iter().map(|l| format!("({})", wkt_pts(l))).collect::<Vec<_>>().join(", ")),
            Geometry::MultiPolygon(v) if v.is_empty() => "MULTIPOLYGON EMPTY".into(),
            Geometry::MultiPolygon(v) => format!(
                "MULTIPOLYGON ({})",
                v.iter().map(|p| format!("({})", p.iter().map(|l| format!("({})", wkt_pts(l))).collect::<Vec<_>>().join(", "))).collect::<Vec<_>>().join(", ")
            ),
            Geometry::Collection(v) if v.is_empty() => "GEOMETRYCOLLECTION EMPTY".into(),
            Geometry::Collection(v) => format!("GEOMETRYCOLLECTION ({})", v.iter().map(Geometry::to_wkt).collect::<Vec<_>>().join(", ")),
            Geometry::Envelope { min_x, max_x, max_y, min_y } => format!("BBOX ({min_x}, {max_x}, {max_y}, {min_y})"),
        }
    }

    /// `(min_x, min_y, max_x, max_y)`, `None` for an empty geometry.
    pub fn bbox(&self) -> Option<(f64, f64, f64, f64)> {
        // Every vertex as [`Prims::of`] lists them, without copying them.
        let mut b: Option<(f64, f64, f64, f64)> = None;
        self.each_part(&mut |part| {
            for q in part.vertices() {
                b = Some(match b {
                    None => (q.0, q.1, q.0, q.1),
                    Some(b) => (b.0.min(q.0), b.1.min(q.1), b.2.max(q.0), b.3.max(q.1)),
                });
            }
        });
        b
    }

    /// Each point, line and polygon of the geometry in turn, as
    /// [`Prims::of`] decomposes it (an envelope is its rectangle, or two
    /// across the dateline), borrowed where the geometry holds them.
    pub fn each_part(&self, f: &mut dyn FnMut(Part<'_>)) {
        match self {
            Geometry::Point(p) => f(Part::Point(*p)),
            Geometry::MultiPoint(v) => v.iter().for_each(|p| f(Part::Point(*p))),
            Geometry::LineString(l) => f(Part::Line(l)),
            Geometry::MultiLineString(v) => v.iter().for_each(|l| f(Part::Line(l))),
            Geometry::Polygon(r) => f(Part::Polygon(r)),
            Geometry::MultiPolygon(v) => v.iter().for_each(|r| f(Part::Polygon(r))),
            Geometry::Collection(v) => v.iter().for_each(|g| g.each_part(f)),
            Geometry::Envelope { min_x, max_x, max_y, min_y } => {
                let rect = |a: f64, b: f64| vec![vec![(a, *min_y), (b, *min_y), (b, *max_y), (a, *max_y), (a, *min_y)]];
                if min_x > max_x {
                    f(Part::Polygon(&rect(*min_x, 180.0)));
                    f(Part::Polygon(&rect(-180.0, *max_x)));
                } else {
                    f(Part::Polygon(&rect(*min_x, *max_x)));
                }
            }
        }
    }

    /// Doc-value encoding: `minx,miny,maxx,maxy|WKT`.
    pub fn encode(&self) -> String {
        match self.bbox() {
            Some(b) => format!("{},{},{},{}|{}", b.0, b.1, b.2, b.3, self.to_wkt()),
            None => format!("|{}", self.to_wkt()),
        }
    }

    /// The centroid of the geometry's vertices (for `geo_centroid`-like uses).
    pub fn centroid(&self) -> Option<Pt> {
        let p = Prims::of(self);
        let all: Vec<Pt> = p.points.iter().chain(p.lines.iter().flatten()).chain(p.polys.iter().flat_map(|r| r.first()).flatten()).copied().collect();
        if all.is_empty() {
            return None;
        }
        let n = all.len() as f64;
        Some((all.iter().map(|q| q.0).sum::<f64>() / n, all.iter().map(|q| q.1).sum::<f64>() / n))
    }
}

/// One part of a geometry ([`Geometry::each_part`]).
#[derive(Clone, Copy, Debug)]
pub enum Part<'a> {
    Point(Pt),
    Line(&'a [Pt]),
    /// Outer ring first, then holes.
    Polygon(&'a [Vec<Pt>]),
}

impl Part<'_> {
    /// Its vertices (every ring's for a polygon).
    pub fn vertices(&self) -> impl Iterator<Item = Pt> + '_ {
        let one: Option<Pt> = match self {
            Part::Point(p) => Some(*p),
            _ => None,
        };
        let line: &[Pt] = match self {
            Part::Line(l) => l,
            _ => &[],
        };
        let rings: &[Vec<Pt>] = match self {
            Part::Polygon(r) => r,
            _ => &[],
        };
        one.into_iter().chain(line.iter().copied()).chain(rings.iter().flatten().copied())
    }
}

/// The geometry as ES indexes it ([`normalize_geo`]), borrowed when that
/// changes nothing (no polygon crossing the dateline, no line out of
/// range): the common case copies nothing.
pub fn normalized(g: &Geometry) -> std::borrow::Cow<'_, Geometry> {
    fn line_ok(l: &[Pt]) -> bool {
        !l.iter().any(|p| p.1 > 90.0 || p.1 < -90.0 || p.0 > 180.0 || p.0 < -180.0)
    }
    fn poly_ok(r: &[Vec<Pt>]) -> bool {
        r.first().is_none_or(|outer| !has_long_edge(outer) || signed_area2(outer) >= 0.0)
    }
    fn ok(g: &Geometry) -> bool {
        match g {
            Geometry::LineString(l) => line_ok(l),
            Geometry::MultiLineString(v) => v.iter().all(|l| line_ok(l)),
            Geometry::Polygon(r) => poly_ok(r),
            Geometry::MultiPolygon(v) => v.iter().all(|r| poly_ok(r)),
            Geometry::Collection(v) => v.iter().all(ok),
            _ => true,
        }
    }
    if ok(g) {
        std::borrow::Cow::Borrowed(g)
    } else {
        std::borrow::Cow::Owned(normalize_geo(g))
    }
}


/// Bounding box of an encoded value ([`Geometry::encode`]) without parsing
/// the WKT.
pub fn encoded_bbox(s: &str) -> Option<(f64, f64, f64, f64)> {
    let (head, _) = s.split_once('|')?;
    let mut it = head.split(',');
    let mut v = [0.0f64; 4];
    for x in v.iter_mut() {
        *x = it.next()?.parse().ok()?;
    }
    it.next().is_none().then_some((v[0], v[1], v[2], v[3]))
}

/// Decode an encoded value ([`Geometry::encode`]).
pub fn decode(s: &str) -> Option<Geometry> {
    let (_, w) = s.split_once('|')?;
    parse_wkt(w, &ParseOpts { geo: false, coerce: true, ignore_z: true }).ok()
}

/// ES `DimensionalShapeType`: only the highest dimension of a doc (or of an
/// aggregation) contributes to a centroid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum ShapeDim {
    #[default]
    Point,
    Line,
    Polygon,
}

impl ShapeDim {
    pub fn ordinal(self) -> u8 {
        self as u8
    }

    pub fn from_ordinal(o: u8) -> Option<ShapeDim> {
        match o {
            0 => Some(ShapeDim::Point),
            1 => Some(ShapeDim::Line),
            2 => Some(ShapeDim::Polygon),
            _ => None,
        }
    }
}

/// ES `CompensatedSum` (Kahan summation, same arithmetic).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CompSum {
    pub value: f64,
    pub delta: f64,
}

impl CompSum {
    /// `CompensatedSum.add(value)`.
    pub fn add(&mut self, value: f64) {
        self.add_with(value, 0.0);
    }

    /// `CompensatedSum.add(value, delta)`.
    pub fn add_with(&mut self, value: f64, delta: f64) {
        if !value.is_finite() {
            self.value = value + self.value;
        }
        if self.value.is_finite() {
            let corrected = value + (self.delta + delta);
            let updated = self.value + corrected;
            self.delta = corrected - (updated - self.value);
            self.value = updated;
        }
    }

    /// `CompensatedSum.reset(value, delta)`.
    pub fn reset(&mut self, value: f64, delta: f64) {
        self.value = value;
        self.delta = delta;
    }
}

/// ES `CentroidCalculator`: the centroid of one doc's shape values (all of
/// them, before dateline normalization), weighted by the highest
/// dimension present: points by count, lines by segment length, polygons
/// by ring area (holes negative), in degrees. Same arithmetic, in the same
/// order, as ES (`visitLine`, `visitLinearRing`, `addCoordinate`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CentroidCalc {
    sx: CompSum,
    sy: CompSum,
    sw: CompSum,
    dim: ShapeDim,
}

impl CentroidCalc {
    /// Visit one geometry (ES `CentroidCalculator.add`).
    pub fn add(&mut self, g: &Geometry) {
        match g {
            Geometry::Point(p) => self.point(*p),
            Geometry::MultiPoint(v) => {
                if self.dim == ShapeDim::Point {
                    for p in v {
                        self.point(*p);
                    }
                }
            }
            Geometry::LineString(l) => self.line(l),
            Geometry::MultiLineString(ls) => {
                if self.dim != ShapeDim::Polygon {
                    for l in ls {
                        self.line(l);
                    }
                }
            }
            Geometry::Polygon(r) => self.polygon(r),
            Geometry::MultiPolygon(ps) => {
                for p in ps {
                    self.polygon(p);
                }
            }
            Geometry::Collection(v) => {
                for x in v {
                    self.add(x);
                }
            }
            Geometry::Envelope { min_x, max_x, max_y, min_y } => {
                // ES `visit(Rectangle)`.
                let diff_x = max_x - min_x;
                let diff_y = max_y - min_y;
                let w = (diff_x * diff_y).abs();
                if w != 0.0 {
                    let sum_x = max_x + min_x;
                    let sum_y = max_y + min_y;
                    self.coord(sum_x / 2.0, sum_y / 2.0, w, ShapeDim::Polygon);
                } else {
                    // A degenerate rectangle is a line.
                    self.line(&[(*min_x, *min_y), (*max_x, *max_y)]);
                }
            }
        }
    }

    /// x of the centroid (`compSumX / compSumWeight`, NaN when empty).
    pub fn x(&self) -> f64 {
        self.sx.value / self.sw.value
    }

    /// y of the centroid.
    pub fn y(&self) -> f64 {
        self.sy.value / self.sw.value
    }

    /// ES `sumWeight()`: negative sums (rounding in degenerate polygons) are 0.
    pub fn sum_weight(&self) -> f64 {
        let w = self.sw.value;
        if w < 0.0 {
            0.0
        } else {
            w
        }
    }

    pub fn dim(&self) -> ShapeDim {
        self.dim
    }

    fn point(&mut self, p: Pt) {
        if self.dim == ShapeDim::Point {
            self.coord(p.0, p.1, 1.0, ShapeDim::Point);
        }
    }

    fn line(&mut self, l: &[Pt]) {
        if self.dim != ShapeDim::Polygon {
            self.visit_line(l);
        }
    }

    fn visit_line(&mut self, l: &[Pt]) {
        for w in l.windows(2) {
            let (a, b) = (w[0], w[1]);
            let diff_x = a.0 - b.0;
            let diff_y = a.1 - b.1;
            let x_avg = (a.0 + b.0) / 2.0;
            let y_avg = (a.1 + b.1) / 2.0;
            let weight = (diff_x * diff_x + diff_y * diff_y).sqrt();
            if weight == 0.0 {
                // A degenerate segment counts as a point.
                self.coord(a.0, a.1, 1.0, ShapeDim::Point);
            } else {
                self.coord(x_avg, y_avg, weight, ShapeDim::Line);
            }
        }
    }

    fn polygon(&mut self, rings: &[Vec<Pt>]) {
        let Some(outer) = rings.first() else { return };
        let parts: Vec<(f64, f64, f64)> = rings.iter().enumerate().map(|(i, r)| es_ring_centroid(r, i > 0)).collect();
        let mut sum_weight = 0.0;
        for p in &parts {
            sum_weight += p.2;
        }
        if sum_weight == 0.0 && self.dim != ShapeDim::Polygon {
            self.visit_line(outer);
        } else {
            for (x, y, w) in parts {
                self.coord(x, y, w, ShapeDim::Polygon);
            }
        }
    }

    /// ES `addCoordinate`.
    fn coord(&mut self, x: f64, y: f64, weight: f64, dim: ShapeDim) {
        if x.is_finite() && y.is_finite() {
            if self.dim == dim {
                self.sx.add(x * weight);
                self.sy.add(y * weight);
                self.sw.add(weight);
            } else if dim > self.dim {
                self.sx.reset(x * weight, 0.0);
                self.sy.reset(y * weight, 0.0);
                self.sw.reset(weight, 0.0);
                self.dim = dim;
            }
        }
    }
}

/// ES `visitLinearRing`: `(x, y, weight)` of a closed ring, the weight its
/// area, negative for holes (x / y non-finite for a zero-area ring).
fn es_ring_centroid(r: &[Pt], hole: bool) -> (f64, f64, f64) {
    let sign = if hole { -1.0 } else { 1.0 };
    let mut total = 0.0;
    for w in r.windows(2) {
        total += (w[0].0 * w[1].1) - (w[1].0 * w[0].1);
    }
    total /= 2.0;
    let (mut sum_x, mut sum_y) = (0.0, 0.0);
    for w in r.windows(2) {
        let twice = (w[0].0 * w[1].1) - (w[1].0 * w[0].1);
        sum_x += twice * (w[0].0 + w[1].0);
        sum_y += twice * (w[0].1 + w[1].1);
    }
    (sum_x / (6.0 * total), sum_y / (6.0 * total), sign * total.abs())
}

/// ES `GeoUtils.normalizeLat`.
fn normalize_lat(lat: f64) -> f64 {
    let mut lat = lat;
    if lat > 90.0 || lat < -90.0 {
        lat = centered_modulus(lat, 360.0);
        if lat < -90.0 {
            lat = -180.0 - lat;
        } else if lat > 90.0 {
            lat = 180.0 - lat;
        }
    }
    lat + 0.0
}

/// A doc's centroid as ES's shape doc value header stores it: the
/// [`CentroidCalc`] centroid normalized and Lucene-encoded
/// (`encodeX(normalizeX(getX()))`), its dimension and `sumWeight()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCentroid {
    /// Encoded latitude (`GeoEncodingUtils.encodeLatitude`).
    pub lat: i32,
    /// Encoded longitude.
    pub lon: i32,
    pub dim: ShapeDim,
    pub weight: f64,
}

impl ShapeCentroid {
    /// `None` when nothing was added (or the sums are not finite: ES
    /// rejects such a document).
    pub fn of(c: &CentroidCalc) -> Option<ShapeCentroid> {
        let (x, y) = (c.x(), c.y());
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        Some(ShapeCentroid { lat: crate::geo::encode_lat(normalize_lat(y)), lon: crate::geo::encode_lon(normalize_lon(x)), dim: c.dim(), weight: c.sum_weight() })
    }

    /// Of a doc's geometry (all its values, not normalized).
    pub fn of_geometry(g: &Geometry) -> Option<ShapeCentroid> {
        let mut c = CentroidCalc::default();
        c.add(g);
        ShapeCentroid::of(&c)
    }

    /// Decoded latitude (`decodeY(getCentroidY())`).
    pub fn lat_deg(&self) -> f64 {
        crate::geo::decode_lat(self.lat)
    }

    /// Decoded longitude.
    pub fn lon_deg(&self) -> f64 {
        crate::geo::decode_lon(self.lon)
    }
}

/// ES `Extent` in Lucene-encoded ints: top / bottom latitudes, and the
/// longitudes of the parts west (`neg_*`) and east (`pos_*`) of the prime
/// meridian; a part crossing it sets `neg_right = pos_left = 0` (the
/// extent may not wrap the dateline). Unset sides hold ES's sentinels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeExtent {
    pub top: i32,
    pub bottom: i32,
    pub neg_left: i32,
    pub neg_right: i32,
    pub pos_left: i32,
    pub pos_right: i32,
}

impl Default for ShapeExtent {
    fn default() -> Self {
        ShapeExtent { top: i32::MIN, bottom: i32::MAX, neg_left: i32::MAX, neg_right: i32::MIN, pos_left: i32::MAX, pos_right: i32::MIN }
    }
}

impl ShapeExtent {
    /// ES `Extent.addRectangle` (the box must not cross the dateline).
    pub fn add_rect(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) {
        self.bottom = self.bottom.min(min_y);
        self.top = self.top.max(max_y);
        if min_x < 0 && max_x < 0 {
            self.neg_left = self.neg_left.min(min_x);
            self.neg_right = self.neg_right.max(max_x);
        } else if min_x < 0 {
            self.neg_left = self.neg_left.min(min_x);
            self.pos_right = self.pos_right.max(max_x);
            self.neg_right = 0;
            self.pos_left = 0;
        } else {
            self.pos_left = self.pos_left.min(min_x);
            self.pos_right = self.pos_right.max(max_x);
        }
    }

    /// Union with another extent (as adding its rectangles one by one:
    /// `neg_right` is never above 0 and `pos_left` never below, so ES's
    /// crossing assignment is a max / min).
    pub fn merge(&mut self, o: &ShapeExtent) {
        self.top = self.top.max(o.top);
        self.bottom = self.bottom.min(o.bottom);
        self.neg_left = self.neg_left.min(o.neg_left);
        self.neg_right = self.neg_right.max(o.neg_right);
        self.pos_left = self.pos_left.min(o.pos_left);
        self.pos_right = self.pos_right.max(o.pos_right);
    }

    /// No western (resp. eastern) part (ES `BoundingBox.reset`'s test).
    pub fn has_neg(&self) -> bool {
        !(self.neg_left == i32::MAX && self.neg_right == i32::MIN)
    }

    pub fn has_pos(&self) -> bool {
        !(self.pos_left == i32::MAX && self.pos_right == i32::MIN)
    }

    /// The extent of a *normalized* geo geometry ([`normalize_geo`]) as ES
    /// builds it from the triangles it indexes (`TriangleTreeWriter`):
    /// a point is a degenerate triangle, each line segment a triangle, and a
    /// polygon's triangles have its vertices as corners (one of them
    /// straddles the prime meridian exactly when the polygon's vertices
    /// do), all on the Lucene grid. `None` without any vertex.
    pub fn of_geometry(g: &Geometry) -> Option<ShapeExtent> {
        use crate::geo::{encode_lat, encode_lon};
        let mut e = ShapeExtent::default();
        let mut any = false;
        // Parts in any order: the rectangles combine by min / max.
        g.each_part(&mut |part| match part {
            Part::Point(q) => {
                let (x, y) = (encode_lon(q.0), encode_lat(q.1));
                e.add_rect(x, y, x, y);
                any = true;
            }
            Part::Line(l) => {
                if l.len() == 1 {
                    // Only in values written before lines needed two points.
                    let (x, y) = (encode_lon(l[0].0), encode_lat(l[0].1));
                    e.add_rect(x, y, x, y);
                    any = true;
                }
                for w in l.windows(2) {
                    let (ax, ay, bx, by) = (encode_lon(w[0].0), encode_lat(w[0].1), encode_lon(w[1].0), encode_lat(w[1].1));
                    e.add_rect(ax.min(bx), ay.min(by), ax.max(bx), ay.max(by));
                    any = true;
                }
            }
            Part::Polygon(_) => {
                let mut b: Option<(i32, i32, i32, i32)> = None;
                for q in part.vertices() {
                    let (x, y) = (encode_lon(q.0), encode_lat(q.1));
                    b = Some(match b {
                        None => (x, y, x, y),
                        Some(b) => (b.0.min(x), b.1.min(y), b.2.max(x), b.3.max(y)),
                    });
                }
                if let Some(b) = b {
                    e.add_rect(b.0, b.1, b.2, b.3);
                    any = true;
                }
            }
        });
        any.then_some(e)
    }
}

// ---------------------------------------------------------------------------
// Relations.

/// A geometry decomposed into points, polylines and polygons.
#[derive(Default, Debug)]
pub struct Prims {
    pub points: Vec<Pt>,
    pub lines: Vec<Vec<Pt>>,
    pub polys: Vec<Vec<Vec<Pt>>>,
}

impl Prims {
    pub fn of(g: &Geometry) -> Prims {
        let mut p = Prims::default();
        p.add(g);
        p
    }
    fn add(&mut self, g: &Geometry) {
        match g {
            Geometry::Point(x) => self.points.push(*x),
            Geometry::MultiPoint(v) => self.points.extend(v),
            Geometry::LineString(l) => self.lines.push(l.clone()),
            Geometry::MultiLineString(v) => self.lines.extend(v.iter().cloned()),
            Geometry::Polygon(r) => self.polys.push(r.clone()),
            Geometry::MultiPolygon(v) => self.polys.extend(v.iter().cloned()),
            Geometry::Collection(v) => v.iter().for_each(|x| self.add(x)),
            Geometry::Envelope { min_x, max_x, max_y, min_y } => {
                let rect = |a: f64, b: f64| vec![vec![(a, *min_y), (b, *min_y), (b, *max_y), (a, *max_y), (a, *min_y)]];
                if min_x > max_x {
                    // Crosses the dateline.
                    self.polys.push(rect(*min_x, 180.0));
                    self.polys.push(rect(-180.0, *max_x));
                } else {
                    self.polys.push(rect(*min_x, *max_x));
                }
            }
        }
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty() && self.lines.is_empty() && self.polys.is_empty()
    }
    pub(crate) fn segments(&self) -> Vec<(Pt, Pt)> {
        let mut s = Vec::new();
        for l in &self.lines {
            for w in l.windows(2) {
                s.push((w[0], w[1]));
            }
        }
        for p in &self.polys {
            for r in p {
                for w in r.windows(2) {
                    s.push((w[0], w[1]));
                }
            }
        }
        s
    }
    pub(crate) fn vertices(&self) -> Vec<Pt> {
        self.points.iter().chain(self.lines.iter().flatten()).chain(self.polys.iter().flatten().flatten()).copied().collect()
    }
}

pub const EPS: f64 = 1e-12;

fn orient(a: Pt, b: Pt, c: Pt) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

#[inline]
pub(crate) fn on_segment(p: Pt, a: Pt, b: Pt) -> bool {
    // The box first: it rejects almost every point for a fraction of the
    // orientation's cost (the same conjunction, so the same answer).
    p.0 >= a.0.min(b.0) - EPS
        && p.0 <= a.0.max(b.0) + EPS
        && p.1 >= a.1.min(b.1) - EPS
        && p.1 <= a.1.max(b.1) + EPS
        && orient(a, b, p).abs() <= EPS * (1.0 + (b.0 - a.0).abs() + (b.1 - a.1).abs())
}

fn sgn(x: f64) -> i32 {
    if x > EPS {
        1
    } else if x < -EPS {
        -1
    } else {
        0
    }
}

/// Segments share at least one point.
pub fn segments_intersect(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    let (o1, o2, o3, o4) = (sgn(orient(a, b, c)), sgn(orient(a, b, d)), sgn(orient(c, d, a)), sgn(orient(c, d, b)));
    if o1 != o2 && o3 != o4 && o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0 {
        return true;
    }
    on_segment(c, a, b) || on_segment(d, a, b) || on_segment(a, c, d) || on_segment(b, c, d) || (o1 * o2 < 0 && o3 * o4 < 0)
}

/// Segments cross at a single point interior to both.
pub(crate) fn segments_cross(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    let (o1, o2, o3, o4) = (sgn(orient(a, b, c)), sgn(orient(a, b, d)), sgn(orient(c, d, a)), sgn(orient(c, d, b)));
    o1 * o2 < 0 && o3 * o4 < 0
}

/// 1 inside, 0 on the boundary, -1 outside (of a ring).
fn ring_position(p: Pt, r: &[Pt]) -> i32 {
    let mut inside = false;
    for w in r.windows(2) {
        let (a, b) = (w[0], w[1]);
        if on_segment(p, a, b) {
            return 0;
        }
        if (a.1 > p.1) != (b.1 > p.1) {
            let x = a.0 + (p.1 - a.1) * (b.0 - a.0) / (b.1 - a.1);
            if p.0 < x {
                inside = !inside;
            }
        }
    }
    if inside {
        1
    } else {
        -1
    }
}

/// 1 inside, 0 boundary, -1 outside of a polygon with holes.
pub(crate) fn polygon_position(p: Pt, poly: &[Vec<Pt>]) -> i32 {
    let Some(outer) = poly.first() else { return -1 };
    match ring_position(p, outer) {
        -1 => -1,
        0 => 0,
        _ => {
            for h in &poly[1..] {
                match ring_position(p, h) {
                    1 => return -1,
                    0 => return 0,
                    _ => {}
                }
            }
            1
        }
    }
}

/// Point touches the geometry (inside, on boundary or on a line / point).
pub fn point_in(p: Pt, g: &Prims) -> bool {
    g.points.iter().any(|q| (q.0 - p.0).abs() <= EPS && (q.1 - p.1).abs() <= EPS)
        || g.lines.iter().any(|l| l.windows(2).any(|w| on_segment(p, w[0], w[1])) || (l.len() == 1 && l[0] == p))
        || g.polys.iter().any(|poly| polygon_position(p, poly) >= 0)
}

/// `relate(&Prims { points: vec![p], .. }, q, Intersects)`, without
/// allocating (per-point tests of geo-point fields).
pub fn point_intersects(p: Pt, q: &Prims) -> bool {
    if point_in(p, q) {
        return true;
    }
    // Any vertex within EPS (plain loops: the chained iterator over every
    // vertex cost more than the polygon test).
    let near = |v: &Pt| (v.0 - p.0).abs() <= EPS && (v.1 - p.1).abs() <= EPS;
    q.points.iter().any(near)
        || q.lines.iter().any(|l| l.iter().any(near))
        || q.polys.iter().any(|poly| poly.iter().any(|ring| ring.iter().any(near)))
}

/// `relate(&Prims { points: vec![p], .. }, q, Within)`.
pub fn point_within(p: Pt, q: &Prims) -> bool {
    !q.is_empty() && point_in(p, q)
}

/// Whether every point of the box `[x0, x1] x [y0, y1]` is strictly inside
/// one of `q`'s polygons (so [`point_intersects`] / [`point_within`] hold
/// for all of them): its corners are strictly inside and no ring of the
/// polygon touches the box.
pub fn box_covered(q: &Prims, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
    let c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let edges = [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])];
    q.polys.iter().any(|poly| {
        c.iter().all(|p| polygon_position(*p, poly) == 1)
            && !poly.iter().any(|ring| {
                ring.iter().any(|v| v.0 >= x0 - EPS && v.0 <= x1 + EPS && v.1 >= y0 - EPS && v.1 <= y1 + EPS)
                    || ring.windows(2).any(|w| edges.iter().any(|(a, b)| segments_intersect(w[0], w[1], *a, *b)))
            })
    })
}

/// Whether no point of the box `[x0, x1] x [y0, y1]` passes
/// [`point_intersects`] / [`point_within`] against `q` (conservative: a
/// `false` only means "maybe"). True when nothing of `q` comes near the
/// box (no vertex, and no segment whose bounding box, widened by the
/// boundary tolerance, reaches it) and a corner lies outside every
/// polygon: with no boundary near, the whole box is on the corner's side.
pub fn box_disjoint(q: &Prims, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
    let (bx0, by0, bx1, by1) = (x0 - 2.0 * EPS, y0 - 2.0 * EPS, x1 + 2.0 * EPS, y1 + 2.0 * EPS);
    let near_pt = |v: &Pt| v.0 >= bx0 && v.0 <= bx1 && v.1 >= by0 && v.1 <= by1;
    let near_seg = |a: Pt, b: Pt| a.0.min(b.0) - EPS <= bx1 && a.0.max(b.0) + EPS >= bx0 && a.1.min(b.1) - EPS <= by1 && a.1.max(b.1) + EPS >= by0;
    let near_path = |l: &[Pt]| l.iter().any(near_pt) || l.windows(2).any(|w| near_seg(w[0], w[1]));
    if q.points.iter().any(near_pt) || q.lines.iter().any(|l| near_path(l)) {
        return false;
    }
    if q.polys.iter().any(|poly| poly.iter().any(|ring| near_path(ring))) {
        return false;
    }
    q.polys.iter().all(|poly| polygon_position((x0, y0), poly) == -1)
}

/// The bounding box `(min_x, min_y, max_x, max_y)` of every vertex of `q`.
pub fn prims_bbox(q: &Prims) -> Option<(f64, f64, f64, f64)> {
    q.points.iter().chain(q.lines.iter().flatten()).chain(q.polys.iter().flatten().flatten()).fold(None, |b, p| {
        Some(match b {
            None => (p.0, p.1, p.0, p.1),
            Some((a, b, c, d)) => (a.min(p.0), b.min(p.1), c.max(p.0), d.max(p.1)),
        })
    })
}

/// Do the two geometries share any point?
pub fn intersects(a: &Prims, b: &Prims) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    // Any vertex of one inside / on the other.
    if a.vertices().iter().any(|p| point_in(*p, b)) || b.vertices().iter().any(|p| point_in(*p, a)) {
        return true;
    }
    // Any pair of edges intersecting.
    let sa = a.segments();
    let sb = b.segments();
    sa.iter().any(|(p, q)| sb.iter().any(|(r, s)| segments_intersect(*p, *q, *r, *s)))
}

/// Is every point of `a` in `b` (boundary included)?
pub fn within(a: &Prims, b: &Prims) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if !a.vertices().iter().all(|p| point_in(*p, b)) {
        return false;
    }
    let sb = b.segments();
    // Edges of `a` must not cross `b`'s boundary.
    for (p, q) in a.segments() {
        if sb.iter().any(|(r, s)| segments_cross(p, q, *r, *s)) {
            return false;
        }
        // Midpoint check catches edges leaving a concave polygon through vertices.
        let m = ((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0);
        if !point_in(m, b) {
            return false;
        }
    }
    // A hole of `b` (or any part of b's boundary) strictly inside `a`'s area
    // means `a` is not within `b`.
    if !a.polys.is_empty() {
        for poly in &b.polys {
            for h in poly.iter().skip(1) {
                if h.iter().any(|v| a.polys.iter().any(|ap| polygon_position(*v, ap) == 1)) {
                    return false;
                }
            }
        }
    }
    true
}

/// Relation test between an indexed shape and a query shape.
pub fn relate(doc: &Prims, query: &Prims, rel: SpatialRelation) -> bool {
    match rel {
        SpatialRelation::Intersects => intersects(doc, query),
        SpatialRelation::Disjoint => !intersects(doc, query),
        SpatialRelation::Within => within(doc, query),
        // Lines have no interior: only points and areas of the indexed
        // shape can contain the query.
        SpatialRelation::Contains => {
            let d = Prims { points: doc.points.clone(), lines: Vec::new(), polys: doc.polys.clone() };
            within(query, &d)
        }
    }
}

impl Prims {
    /// Every coordinate snapped to the Lucene lat/lon grid (as indexed).
    pub fn quantized(&self) -> Prims {
        let q = |p: &Pt| -> Pt {
            if (-90.0..=90.0).contains(&p.1) && (-180.0..=180.0).contains(&p.0) {
                let (la, lo) = crate::geo::quantize(p.1, p.0);
                (lo, la)
            } else {
                *p
            }
        };
        Prims {
            points: self.points.iter().map(q).collect(),
            lines: self.lines.iter().map(|l| l.iter().map(q).collect()).collect(),
            polys: self.polys.iter().map(|p| p.iter().map(|r| r.iter().map(q).collect()).collect()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dateline handling.

fn signed_area2(r: &[Pt]) -> f64 {
    r.windows(2).map(|w| w[0].0 * w[1].1 - w[1].0 * w[0].1).sum()
}

fn has_long_edge(r: &[Pt]) -> bool {
    r.windows(2).any(|w| (w[1].0 - w[0].0).abs() > 180.0)
}

/// Longitudes made continuous: consecutive points at most 180 apart.
fn unwrap_lon(r: &[Pt]) -> Vec<Pt> {
    let mut out = Vec::with_capacity(r.len());
    let mut off = 0.0;
    for (i, p) in r.iter().enumerate() {
        if i > 0 {
            let dx = p.0 - r[i - 1].0;
            if dx > 180.0 {
                off -= 360.0;
            } else if dx < -180.0 {
                off += 360.0;
            }
        }
        out.push((p.0 + off, p.1));
    }
    out
}

/// Sutherland-Hodgman clip of a closed ring to `x <= x0` (`le`) or `x >= x0`.
fn clip_ring(r: &[Pt], x0: f64, le: bool) -> Vec<Pt> {
    let pts = &r[..r.len().saturating_sub(1)];
    let inside = |p: &Pt| if le { p.0 <= x0 } else { p.0 >= x0 };
    let mut out: Vec<Pt> = Vec::new();
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        let cut = (x0, a.1 + (b.1 - a.1) * (x0 - a.0) / (b.0 - a.0));
        match (inside(&a), inside(&b)) {
            (true, true) => out.push(b),
            (true, false) => out.push(cut),
            (false, true) => {
                out.push(cut);
                out.push(b);
            }
            _ => {}
        }
    }
    if out.len() < 3 {
        return Vec::new();
    }
    out.push(out[0]);
    out
}

fn shift(r: &[Pt], dx: f64) -> Vec<Pt> {
    r.iter().map(|p| (p.0 + dx, p.1)).collect()
}

fn split_polygon(rings: &[Vec<Pt>]) -> Vec<Vec<Vec<Pt>>> {
    let Some(outer) = rings.first() else { return Vec::new() };
    // The right-hand rule decides what an edge longer than 180 degrees means:
    // a counter-clockwise ring is the long way round, a clockwise one crosses
    // the dateline.
    if !has_long_edge(outer) || signed_area2(outer) >= 0.0 {
        return vec![rings.to_vec()];
    }
    let u = unwrap_lon(outer);
    let (min_x, max_x) = u.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let mut out = Vec::new();
    let main = clip_ring(&clip_ring(&u, 180.0, true), -180.0, false);
    if !main.is_empty() {
        out.push(vec![main]);
    }
    if max_x > 180.0 {
        let e = clip_ring(&u, 180.0, false);
        if !e.is_empty() {
            out.push(vec![shift(&e, -360.0)]);
        }
    }
    if min_x < -180.0 {
        let w = clip_ring(&u, -180.0, true);
        if !w.is_empty() {
            out.push(vec![shift(&w, 360.0)]);
        }
    }
    out
}

/// ES `GeoUtils.centeredModulus`: `dividend` into `]-divisor/2, divisor/2]`.
fn centered_modulus(dividend: f64, divisor: f64) -> f64 {
    let mut r = dividend % divisor;
    if r <= 0.0 {
        r += divisor;
    }
    if r > divisor / 2.0 {
        r -= divisor;
    }
    r
}

/// ES `GeoUtils.normalizeLon`.
fn normalize_lon(lon: f64) -> f64 {
    let lon = if lon > 180.0 || lon <= -180.0 { centered_modulus(lon, 360.0) } else { lon };
    lon + 0.0
}

/// ES `GeoUtils.normalizePoint(lonLat, false, true)`: latitude folded over
/// the poles (moving the longitude half way round), longitude kept as
/// `x + k * 360`.
fn normalize_point_lat(p: Pt) -> Pt {
    let (mut lon, mut lat) = p;
    if lat > 90.0 || lat < -90.0 {
        lat = centered_modulus(lat, 360.0);
        let shift = if lat < -90.0 {
            lat = -180.0 - lat;
            true
        } else if lat > 90.0 {
            lat = 180.0 - lat;
            true
        } else {
            false
        };
        if shift {
            lon += if normalize_lon(lon) > 0.0 { -180.0 } else { 180.0 };
        }
    }
    (lon, lat)
}

/// ES `GeoLineDecomposer.calculateShift`: degrees east that bring `lon`
/// into `[-180, 180]` (`180` inclusive only with `include180`).
fn calculate_shift(lon: f64, include180: bool) -> f64 {
    let normalized = centered_modulus(lon, 360.0);
    // Java `Math.round`: floor(x + 0.5).
    let mut shift = (normalized - lon + 0.5).floor();
    if !include180 && normalized == 180.0 {
        shift -= 360.0;
    }
    shift
}

/// ES `GeoLineDecomposer.intersection`: where the segment `p1x..p2x`
/// (`p1x` in range) meets the antimeridian, strictly inside it.
fn dateline_intersection(p1x: f64, p2x: f64) -> Option<f64> {
    if p1x == p2x {
        return None;
    }
    let t = ((if p1x < p2x { 180.0 } else { -180.0 }) - p1x) / (p2x - p1x);
    (t > 0.0 && t < 1.0).then_some(t)
}

fn shifted(lons: &[f64], lats: &[f64], shift: f64) -> Vec<Pt> {
    lons.iter().zip(lats).map(|(x, y)| (x + shift, *y)).collect()
}

/// ES `GeoLineDecomposer.decompose`: cut a line at every antimeridian
/// crossing, each piece shifted into `[-180, 180]`.
fn decompose_line(mut lons: Vec<f64>, mut lats: Vec<f64>) -> Vec<Vec<Pt>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    let mut shift = 0.0;
    let mut i = 1usize;
    // ES gives up (rejecting the shape) past 10k crossings of one segment;
    // only unvalidated query shapes can get here, so stop cutting instead.
    let mut segment_crossings = 0u32;
    while i < lons.len() {
        let direction = if lons[i] > lons[i - 1] { 1 } else if lons[i] < lons[i - 1] { -1 } else { 0 };
        let new_shift = calculate_shift(lons[i - 1], direction < 0);
        if i - offset > 1 && new_shift != shift {
            // Jumping over the antimeridian: start a new piece.
            out.push(shifted(&lons[offset..i], &lats[offset..i], shift));
            shift = new_shift;
            offset = i - 1;
        } else {
            shift = new_shift;
            match dateline_intersection(lons[i - 1] + shift, lons[i] + shift) {
                Some(t) if segment_crossings < 10_000 => {
                    segment_crossings += 1;
                    let x = (if direction > 0 { 180.0 } else { -180.0 }) - shift;
                    let y = lats[i - 1] + (lats[i] - lats[i - 1]) * t;
                    let mut part = shifted(&lons[offset..=i], &lats[offset..=i], 0.0);
                    if let Some(last) = part.last_mut() {
                        *last = (x, y);
                    }
                    lons[i - 1] = x;
                    lats[i - 1] = y;
                    out.push(part.into_iter().map(|(px, py)| (px + shift, py)).collect());
                    offset = i - 1;
                }
                _ => {
                    segment_crossings = 0;
                    i += 1;
                }
            }
        }
    }
    if offset == 0 {
        out.push(shifted(&lons, &lats, shift));
    } else if offset < lons.len() - 1 {
        out.push(shifted(&lons[offset..], &lats[offset..], shift));
    }
    out
}

/// ES `GeometryNormalizer` for lines (`GeoLineDecomposer`): a line is only
/// decomposed when one of its coordinates is out of range. An in-range
/// line is indexed as written, so an edge longer than 180 degrees goes the
/// long way round (it does not cross the dateline); out-of-range
/// longitudes (`x + k * 360`, query shapes only, since indexed shapes are
/// range checked) are cut at each antimeridian crossing.
fn split_line(l: &[Pt]) -> Vec<Vec<Pt>> {
    let needs = l.iter().any(|p| p.1 > 90.0 || p.1 < -90.0 || p.0 > 180.0 || p.0 < -180.0);
    if !needs {
        return vec![l.to_vec()];
    }
    let (lons, lats): (Vec<f64>, Vec<f64>) = l.iter().map(|p| normalize_point_lat(*p)).unzip();
    decompose_line(lons, lats)
}

/// Geometry with dateline-crossing polygons and lines cut into pieces on
/// either side (ES `GeometryNormalizer`).
pub fn normalize_geo(g: &Geometry) -> Geometry {
    match g {
        Geometry::LineString(l) => {
            let mut parts = split_line(l);
            if parts.len() == 1 { Geometry::LineString(parts.remove(0)) } else { Geometry::MultiLineString(parts) }
        }
        Geometry::MultiLineString(ls) => Geometry::MultiLineString(ls.iter().flat_map(|l| split_line(l)).collect()),
        Geometry::Polygon(r) => {
            let mut parts = split_polygon(r);
            if parts.len() == 1 { Geometry::Polygon(parts.remove(0)) } else { Geometry::MultiPolygon(parts) }
        }
        Geometry::MultiPolygon(ps) => Geometry::MultiPolygon(ps.iter().flat_map(|p| split_polygon(p)).collect()),
        Geometry::Collection(v) => Geometry::Collection(v.iter().map(normalize_geo).collect()),
        other => other.clone(),
    }
}

/// Bounding boxes `(minx, miny, maxx, maxy)` overlap.
pub fn bbox_overlap(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 <= b.2 && a.2 >= b.0 && a.1 <= b.3 && a.3 >= b.1
}

// ---------------------------------------------------------------------------
// Point-only geo shapes as a KD tree.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn g(v: Value) -> Prims {
        Prims::of(&parse(&v, &ParseOpts::default()).unwrap())
    }
    fn w(s: &str) -> Prims {
        Prims::of(&parse_wkt(s, &ParseOpts::default()).unwrap())
    }

    #[test]
    fn parse_forms() {
        let p = parse(&json!({"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}), &ParseOpts::default()).unwrap();
        assert_eq!(p.bbox(), Some((0.0, 0.0, 10.0, 10.0)));
        let enc = p.encode();
        assert_eq!(decode(&enc).unwrap(), p);
        assert_eq!(encoded_bbox(&enc), Some((0.0, 0.0, 10.0, 10.0)));
        assert!(parse(&json!({"type": "polygon", "coordinates": [[[0, 0], [10, 0], [10, 10]]]}), &ParseOpts::default()).is_err());
        assert!(parse(&json!({"type": "polygon", "coordinates": [[[0, 0], [10, 0], [10, 10]]]}), &ParseOpts { coerce: true, ..Default::default() }).is_ok());
        assert!(parse(&json!({"type": "point", "coordinates": [0, 95]}), &ParseOpts::default()).unwrap_err().contains("latitude"));
        assert!(parse(&json!({"type": "point", "coordinates": [0, 95]}), &ParseOpts { geo: false, ..Default::default() }).is_ok());
        for s in [
            "POINT (1 2)", "MULTIPOINT (1 2, 3 4)", "MULTIPOINT ((1 2), (3 4))", "LINESTRING (0 0, 1 1)", "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
            "POLYGON ((0 0, 1 0, 1 1, 0 0))", "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))", "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))",
            "BBOX (0, 10, 10, 0)", "POINT (1 2 3)",
        ] {
            let x = parse_wkt(s, &ParseOpts::default()).unwrap_or_else(|e| panic!("{s}: {e}"));
            assert_eq!(parse_wkt(&x.to_wkt(), &ParseOpts::default()).unwrap(), x, "{s}");
        }
        assert!(parse_wkt("POINT (1 2 3)", &ParseOpts { ignore_z: false, ..Default::default() }).is_err());
        assert!(matches!(parse(&json!({"type": "envelope", "coordinates": [[0, 10], [10, 0]]}), &ParseOpts::default()).unwrap(), Geometry::Envelope { .. }));
        assert!(matches!(parse(&json!({"type": "circle", "coordinates": [0, 0], "radius": "1km"}), &ParseOpts::default()).unwrap(), Geometry::Polygon(_)));
    }

    #[test]
    fn relations() {
        let sq = w("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))");
        let small = w("POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))");
        let far = w("POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))");
        let overlap = w("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))");
        assert!(intersects(&sq, &small) && within(&small, &sq) && !within(&sq, &small));
        assert!(relate(&sq, &small, SpatialRelation::Contains));
        assert!(relate(&sq, &far, SpatialRelation::Disjoint));
        assert!(intersects(&sq, &overlap) && !within(&overlap, &sq));
        let holed = w("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))");
        assert!(!intersects(&w("POINT (5 5)"), &holed));
        assert!(intersects(&w("POINT (1 1)"), &holed));
        assert!(!within(&sq, &holed) && !within(&w("POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"), &holed));
        let line = w("LINESTRING (-5 5, 15 5)");
        assert!(intersects(&line, &sq) && !within(&line, &sq));
        assert!(within(&w("LINESTRING (1 1, 9 9)"), &sq));
        assert!(intersects(&w("POINT (10 5)"), &sq));
        assert!(within(&w("MULTIPOINT (1 1, 2 2)"), &sq));
        assert!(g(json!({"type": "envelope", "coordinates": [[0, 10], [10, 0]]})).polys.len() == 1);
        assert!(bbox_overlap((0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 2.0, 2.0)) && !bbox_overlap((0.0, 0.0, 1.0, 1.0), (1.5, 1.5, 2.0, 2.0)));
    }

    // Expected values below were measured against Elasticsearch 8.19.22.
    fn q(s: &str) -> Prims {
        let g = parse_wkt(s, &ParseOpts { geo: true, coerce: true, ignore_z: true }).unwrap();
        Prims::of(&normalize_geo(&g)).quantized()
    }
    fn gj(v: Value) -> Prims {
        Prims::of(&normalize_geo(&parse_query_shape(&v).unwrap())).quantized()
    }

    #[test]
    fn es_quantization_and_contains() {
        let line = q("LINESTRING (0 0, 20 20)");
        let pt = q("POINT (5 5)");
        let env = q("BBOX (-5, 5, 5, -5)");
        let poly = q("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))");
        // A point exactly on a line does not intersect it once snapped to the grid.
        assert!(!relate(&line, &pt, SpatialRelation::Intersects));
        // A line has no interior, so it never contains a point.
        assert!(!relate(&line, &pt, SpatialRelation::Contains));
        // Points on an area's boundary are contained.
        assert!(relate(&env, &pt, SpatialRelation::Contains));
        assert!(relate(&poly, &pt, SpatialRelation::Contains));
    }

    #[test]
    fn es_dateline_orientation() {
        let strip = q("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))");
        // Counter-clockwise ring with 340 degree edges: the long way round.
        let ccw = gj(json!({"type": "Polygon", "coordinates": [[[170, 10], [-170, 10], [-170, -10], [170, -10], [170, 10]]]}));
        // Clockwise: the short way, across the dateline.
        let cw = gj(json!({"type": "Polygon", "coordinates": [[[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]]}));
        assert!(relate(&strip, &ccw, SpatialRelation::Intersects));
        assert!(!relate(&strip, &cw, SpatialRelation::Intersects));
        let across = q("POINT (179.5 0)");
        assert!(relate(&across, &cw, SpatialRelation::Intersects));
        let env = gj(json!({"type": "envelope", "coordinates": [[170, 10], [-170, -10]]}));
        assert!(relate(&q("POINT (-179 0)"), &env, SpatialRelation::Within));
        assert!(!relate(&strip, &env, SpatialRelation::Intersects));
    }

    /// ES `GeometryNormalizer` / `GeoLineDecomposer` (`GeometryNormalizerTests`):
    /// an in-range line is never cut, even with an edge over 180 degrees (it
    /// goes the long way round); a line with out-of-range longitudes is cut
    /// at the antimeridian and each piece shifted into range.
    #[test]
    fn es_line_dateline_decomposition() {
        let long = Geometry::LineString(vec![(170.0, 0.0), (-170.0, 0.0)]);
        assert_eq!(normalize_geo(&long), long);
        let ml = Geometry::MultiLineString(vec![vec![(179.0, 1.0), (-179.0, 2.0)], vec![(1.0, 1.0), (2.0, 2.0)]]);
        assert_eq!(normalize_geo(&ml), ml);
        let over = Geometry::LineString(vec![(170.0, 0.0), (190.0, 10.0)]);
        assert_eq!(normalize_geo(&over), Geometry::MultiLineString(vec![vec![(170.0, 0.0), (180.0, 5.0)], vec![(-180.0, 5.0), (-170.0, 10.0)]]));
        // Entirely past 180: shifted back without a cut.
        let east = Geometry::LineString(vec![(190.0, 0.0), (200.0, 10.0)]);
        assert_eq!(normalize_geo(&east), Geometry::LineString(vec![(-170.0, 0.0), (-160.0, 10.0)]));
    }

    /// ES `Line` / `GeoJson.asLineString`: lines (also multilinestring
    /// members, GeoJSON and WKT) need two points; Z values are dropped
    /// with `ignore_z_value` (the default) and rejected without it.
    #[test]
    fn es_line_validation() {
        let o = ParseOpts::default();
        let e = parse(&json!({"type": "MultiLineString", "coordinates": [[[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]]}), &o).unwrap_err();
        assert_eq!(e, "not enough points to build a line");
        assert_eq!(parse_wkt("LINESTRING (1 2)", &o).unwrap_err(), "at least two points in the line is required");
        assert_eq!(parse_wkt("MULTILINESTRING ((1 2), (3 4, 5 6))", &o).unwrap_err(), "at least two points in the line is required");
        // Duplicate consecutive points and closed members are fine for lines.
        let g = parse_wkt("MULTILINESTRING ((1 2, 1 2, 3 4), (5 5, 6 6, 5 6, 5 5))", &o).unwrap();
        assert_eq!(g, Geometry::MultiLineString(vec![vec![(1.0, 2.0), (1.0, 2.0), (3.0, 4.0)], vec![(5.0, 5.0), (6.0, 6.0), (5.0, 6.0), (5.0, 5.0)]]));
        let z = json!({"type": "MultiLineString", "coordinates": [[[1.0, 2.0, 30.0], [3.0, 4.0, 40.0]]]});
        assert_eq!(parse(&z, &o).unwrap(), Geometry::MultiLineString(vec![vec![(1.0, 2.0), (3.0, 4.0)]]));
        assert!(parse(&z, &ParseOpts { ignore_z: false, ..o }).unwrap_err().contains("ignore_z_value"));
        // A large multilinestring round-trips through the stored encoding.
        let big = Geometry::MultiLineString((0..3000).map(|i| vec![(8.0 + i as f64 * 1e-4, 49.0), (8.0 + i as f64 * 1e-4, 49.1234567)]).collect());
        let enc = big.encode();
        assert!(enc.len() > 65_536);
        assert_eq!(decode(&enc), Some(big));
    }

    #[test]
    fn es_query_shape_parsing() {
        // Query shapes are not range checked; open rings are closed.
        assert!(parse_query_shape(&json!({"type": "Point", "coordinates": [1, 95]})).is_ok());
        assert!(parse_query_shape(&json!({"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10]]]})).is_ok());
        assert!(parse_query_shape(&json!({"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [0, 0]]]})).is_err());
        assert!(parse_query_shape(&json!({"type": "envelope", "coordinates": [[10, -10], [-10, 10]]})).is_err());
        assert!(parse_wkt("POINT EMPTY", &ParseOpts::default()).is_ok());
    }

    /// Deterministic pseudo-random numbers in `[0, 1)`.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }




}
