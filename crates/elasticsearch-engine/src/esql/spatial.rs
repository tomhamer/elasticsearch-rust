//! ES|QL spatial values (`geo_point`, `cartesian_point`, `geo_shape`,
//! `cartesian_shape`). Values travel as WKT strings formatted the way ES's
//! `WellKnownText.toWKT` writes them (Java `Double.toString` coordinates,
//! `POINT (1.0 -1.0)`), so they print, compare and convert like ES.

use rustic_core::fieldtypes::shape::{self, Geometry, ParseOpts, Pt};
use serde_json::Value;

/// The four ES|QL spatial types.
pub const SPATIAL_TYPES: &[&str] = &["geo_point", "cartesian_point", "geo_shape", "cartesian_shape"];

pub fn is_spatial(ty: &str) -> bool {
    SPATIAL_TYPES.contains(&ty)
}

fn num(d: f64) -> String {
    super::fmt_double(d)
}

fn pts(v: &[Pt]) -> String {
    v.iter().map(|p| format!("{} {}", num(p.0), num(p.1))).collect::<Vec<_>>().join(", ")
}

fn rings(r: &[Vec<Pt>]) -> String {
    r.iter().map(|l| format!("({})", pts(l))).collect::<Vec<_>>().join(", ")
}

/// WKT of a geometry with Java-formatted coordinates.
pub fn to_wkt(g: &Geometry) -> String {
    match g {
        Geometry::Point(p) => format!("POINT ({} {})", num(p.0), num(p.1)),
        Geometry::MultiPoint(v) if v.is_empty() => "MULTIPOINT EMPTY".into(),
        Geometry::MultiPoint(v) => format!("MULTIPOINT ({})", pts(v)),
        Geometry::LineString(v) if v.is_empty() => "LINESTRING EMPTY".into(),
        Geometry::LineString(v) => format!("LINESTRING ({})", pts(v)),
        Geometry::MultiLineString(v) if v.is_empty() => "MULTILINESTRING EMPTY".into(),
        Geometry::MultiLineString(v) => format!("MULTILINESTRING ({})", rings(v)),
        Geometry::Polygon(r) if r.is_empty() => "POLYGON EMPTY".into(),
        Geometry::Polygon(r) => format!("POLYGON ({})", rings(r)),
        Geometry::MultiPolygon(v) if v.is_empty() => "MULTIPOLYGON EMPTY".into(),
        Geometry::MultiPolygon(v) => format!("MULTIPOLYGON ({})", v.iter().map(|p| format!("({})", rings(p))).collect::<Vec<_>>().join(", ")),
        Geometry::Collection(v) if v.is_empty() => "GEOMETRYCOLLECTION EMPTY".into(),
        Geometry::Collection(v) => format!("GEOMETRYCOLLECTION ({})", v.iter().map(to_wkt).collect::<Vec<_>>().join(", ")),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => format!("BBOX ({}, {}, {}, {})", num(*min_x), num(*max_x), num(*max_y), num(*min_y)),
    }
}

/// Lucene `GeoEncodingUtils` latitude round trip (what geo doc values hold).
pub fn quantize_lat(lat: f64) -> f64 {
    const LAT_DECODE: f64 = 180.0 / (1u64 << 32) as f64;
    let enc = (lat / LAT_DECODE).floor().clamp(i32::MIN as f64, i32::MAX as f64);
    enc * LAT_DECODE
}

/// Lucene `GeoEncodingUtils` longitude round trip.
pub fn quantize_lon(lon: f64) -> f64 {
    const LON_DECODE: f64 = 360.0 / (1u64 << 32) as f64;
    let enc = (lon / LON_DECODE).floor().clamp(i32::MIN as f64, i32::MAX as f64);
    enc * LON_DECODE
}

pub fn point_wkt(x: f64, y: f64) -> String {
    to_wkt(&Geometry::Point((x, y)))
}

const OPTS: ParseOpts = ParseOpts { geo: false, coerce: true, ignore_z: true };

/// One source value of a field of ES|QL type `ty` as WKT.
pub fn from_source(v: &Value, ty: &str) -> Option<String> {
    match ty {
        "geo_point" => crate::values::parse_geo_point(v).map(|(lat, lon)| point_wkt(lon, lat)),
        "cartesian_point" => crate::fieldtypes::index::parse_xy_point(v, &OPTS).ok().map(|(x, y)| point_wkt(x, y)),
        _ => shape::parse(v, &OPTS).ok().map(|g| to_wkt(&g)),
    }
}

/// `TO_GEOPOINT` / `TO_CARTESIANPOINT` / `TO_GEOSHAPE` / `TO_CARTESIANSHAPE`
/// of a WKT string: `None` when it does not parse (or is not a point for
/// the point types).
pub fn parse_wkt(s: &str, ty: &str) -> Option<String> {
    let g = shape::parse_wkt(s.trim(), &OPTS).ok()?;
    match (ty, &g) {
        ("geo_point" | "cartesian_point", Geometry::Point(_)) => Some(to_wkt(&g)),
        ("geo_point" | "cartesian_point", _) => None,
        ("geo_point" | "cartesian_point" | "geo_shape" | "cartesian_shape", _) => Some(to_wkt(&g)),
        _ => None,
    }
}

/// [`parse_wkt`], with the parse failure as ES reports it (`Failed to
/// parse WKT: expected number but found: '('`): `Err` when the text is not
/// WKT, `Ok(None)` when it is but not of a point type for the point types.
pub fn parse_wkt_checked(s: &str, ty: &str) -> Result<Option<String>, String> {
    match shape::parse_wkt(s.trim(), &OPTS) {
        Ok(_) => Ok(parse_wkt(s, ty)),
        Err(e) => Err(format!("Failed to parse WKT: {}", es_token_message(&e))),
    }
}

/// ES's WKT parser names the offending token (`StreamTokenizer`): a word
/// (numbers are words) as is, a symbol quoted (`'('`), the end as
/// `END-OF-STREAM`. Our messages carry the remaining text instead.
fn es_token_message(msg: &str) -> String {
    let Some(at) = msg.rfind("found: ").map(|i| i + "found: ".len()).or_else(|| msg.rfind("additional text: ").map(|i| i + "additional text: ".len())) else { return msg.to_string() };
    let rest = msg[at..].trim_start();
    let word = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '+' | '.') || (c as u32) >= 160;
    let tok = match rest.chars().next() {
        None => "END-OF-STREAM".to_string(),
        Some(c) if word(c) => rest.chars().take_while(|c| word(*c)).collect(),
        Some(c) => format!("'{c}'"),
    };
    format!("{}{tok}", &msg[..at])
}

/// The primitives (points, lines, rings) of a WKT value, for spatial
/// relations.
pub fn prims(s: &str) -> Option<shape::Prims> {
    shape::parse_wkt(s.trim(), &OPTS).ok().map(|g| shape::Prims::of(&g))
}

/// `(x, y)` of a `POINT` WKT.
pub fn point_xy(s: &str) -> Option<(f64, f64)> {
    match shape::parse_wkt(s.trim(), &OPTS).ok()? {
        Geometry::Point(p) => Some(p),
        _ => None,
    }
}

/// JTS `Distance.pointToSegment`: the distance from `p` to segment `a`-`b`.
fn seg_dist(p: Pt, a: Pt, b: Pt) -> f64 {
    if a.0 == b.0 && a.1 == b.1 {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    let len2 = (b.0 - a.0) * (b.0 - a.0) + (b.1 - a.1) * (b.1 - a.1);
    let r = ((p.0 - a.0) * (b.0 - a.0) + (p.1 - a.1) * (b.1 - a.1)) / len2;
    if r <= 0.0 {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    if r >= 1.0 {
        return (p.0 - b.0).hypot(p.1 - b.1);
    }
    let s = ((a.1 - p.1) * (b.0 - a.0) - (a.0 - p.0) * (b.1 - a.1)) / len2;
    s.abs() * len2.sqrt()
}

/// JTS 1.20 `DouglasPeuckerLineSimplifier.simplify`: the kept points of a
/// coordinate sequence. Sections are disjoint, so an explicit stack visits
/// them in any order with the same result (and no recursion depth limit on
/// long lines). A ring (`preserve_endpoint` false, closed, 4+ points) may
/// also drop its start point (`simplifyRingEndpoint`).
fn dp_coords(pts: &[Pt], tol: f64, preserve_endpoint: bool) -> Vec<Pt> {
    if pts.is_empty() {
        return Vec::new();
    }
    let mut keep = vec![true; pts.len()];
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((i, j)) = stack.pop() {
        if i + 1 >= j {
            continue;
        }
        let (a, b) = (pts[i], pts[j]);
        let mut max_d = -1.0f64;
        let mut max_k = i;
        for (k, &p) in pts.iter().enumerate().take(j).skip(i + 1) {
            let d = seg_dist(p, a, b);
            if d > max_d {
                max_d = d;
                max_k = k;
            }
        }
        if max_d <= tol {
            keep[i + 1..j].iter_mut().for_each(|k| *k = false);
        } else {
            stack.push((max_k, j));
            stack.push((i, max_k));
        }
    }
    let mut out: Vec<Pt> = pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
    let is_ring = pts.len() >= 4 && pts[0] == pts[pts.len() - 1];
    if !preserve_endpoint && is_ring && out.len() >= 4 && seg_dist(out[0], out[1], out[out.len() - 2]) <= tol {
        out.remove(0);
        out.pop();
        // `CoordinateList.closeRing`: append the start unless already there.
        if let (Some(&first), Some(&last)) = (out.first(), out.last()) {
            if first != last {
                out.push(first);
            }
        }
    }
    out
}

/// A polygon ring after simplification: `None` when it collapses below 4
/// points (JTS makes it a line, which `DPTransformer.transformLinearRing`
/// drops from a polygon).
fn dp_ring(r: &[Pt], tol: f64) -> Option<Vec<Pt>> {
    let s = dp_coords(r, tol, false);
    (s.len() >= 4).then_some(s)
}

/// Twice the signed area of a ring (shoelace).
fn ring_area2(r: &[Pt]) -> f64 {
    r.windows(2).map(|w| w[0].0 * w[1].1 - w[1].0 * w[0].1).sum()
}

/// `GeometryTransformer.transformPolygon` under `DPTransformer`: the shell
/// and surviving holes, `None` when the shell collapses (the polygon is
/// dropped). A zero-area shell is invalid, and `createValidArea`'s
/// `buffer(0)` of it is empty, so it is dropped too.
fn dp_polygon(rings: &[Vec<Pt>], tol: f64) -> Option<Vec<Vec<Pt>>> {
    let shell = dp_ring(rings.first()?, tol)?;
    if ring_area2(&shell) == 0.0 {
        return None;
    }
    let mut out = Vec::with_capacity(rings.len());
    out.push(shell);
    out.extend(rings[1..].iter().filter_map(|h| dp_ring(h, tol)));
    Some(out)
}

fn is_empty_geometry(g: &Geometry) -> bool {
    match g {
        Geometry::Point(_) | Geometry::Envelope { .. } => false,
        Geometry::MultiPoint(v) | Geometry::LineString(v) => v.is_empty(),
        Geometry::MultiLineString(v) | Geometry::Polygon(v) => v.iter().all(|r| r.is_empty()),
        Geometry::MultiPolygon(v) => v.iter().flatten().all(|r| r.is_empty()),
        Geometry::Collection(v) => v.iter().all(is_empty_geometry),
    }
}

/// `ST_SIMPLIFY` (ES 9.5.5 `StSimplify`): JTS 1.20
/// `DouglasPeuckerSimplifier.simplify(geometry, tolerance)`, with JTS's
/// result shapes: points are kept; lines keep their end points; polygon
/// rings may lose their start point and collapse, which drops the ring
/// (a hole) or the polygon (`POLYGON EMPTY` at the top level); multi
/// geometries left with one part become that part (`buildGeometry`), with
/// none the empty multi type; collections keep their type and drop empty
/// members. A `BBOX` is its polygon from the bottom-left corner, counter
/// clockwise (`SpatialCoordinateTypes.wkbToJtsGeometry`). Zero-area shells
/// of a simplified area are dropped.
pub fn simplify(g: &Geometry, tol: f64) -> Geometry {
    // `getResultGeometry`: an empty input is returned as is.
    if is_empty_geometry(g) {
        return g.clone();
    }
    match g {
        Geometry::Point(_) | Geometry::MultiPoint(_) => match g {
            Geometry::MultiPoint(v) if v.len() == 1 => Geometry::Point(v[0]),
            _ => g.clone(),
        },
        Geometry::LineString(v) => Geometry::LineString(dp_coords(v, tol, true)),
        Geometry::MultiLineString(v) => {
            let mut lines: Vec<Vec<Pt>> = v.iter().map(|l| dp_coords(l, tol, true)).filter(|l| !l.is_empty()).collect();
            match lines.len() {
                1 => Geometry::LineString(lines.remove(0)),
                _ => Geometry::MultiLineString(lines),
            }
        }
        Geometry::Polygon(rings) => dp_polygon(rings, tol).map(Geometry::Polygon).unwrap_or(Geometry::Polygon(Vec::new())),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => {
            let ring = vec![(*min_x, *min_y), (*max_x, *min_y), (*max_x, *max_y), (*min_x, *max_y), (*min_x, *min_y)];
            dp_polygon(&[ring], tol).map(Geometry::Polygon).unwrap_or(Geometry::Polygon(Vec::new()))
        }
        Geometry::MultiPolygon(v) => {
            let mut polys: Vec<Vec<Vec<Pt>>> = v.iter().filter_map(|p| dp_polygon(p, tol)).collect();
            match polys.len() {
                1 => Geometry::Polygon(polys.remove(0)),
                _ => Geometry::MultiPolygon(polys),
            }
        }
        Geometry::Collection(v) => Geometry::Collection(v.iter().map(|m| simplify(m, tol)).filter(|m| !is_empty_geometry(m)).collect()),
    }
}

/// `ST_SIMPLIFY` of a WKT value: `None` when it does not parse.
pub fn simplify_wkt(s: &str, tol: f64) -> Option<String> {
    let g = shape::parse_wkt(s.trim(), &OPTS).ok()?;
    Some(to_wkt(&simplify(&g, tol)))
}

/// `ST_SIMPLIFY` of a multi-valued spatial value: one collection of all the
/// values (`SpatialGeometryBlockProcessor.asJtsGeometry(List)`, the folded
/// form ES's csv-spec `stSimplifyLiteralPoints` shows).
pub fn simplify_wkts(values: &[String], tol: f64) -> Option<String> {
    let parts = values.iter().map(|s| shape::parse_wkt(s.trim(), &OPTS).ok()).collect::<Option<Vec<_>>>()?;
    Some(to_wkt(&simplify(&Geometry::Collection(parts), tol)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// ES 9.5.5 `spatial-jts.csv-spec` ST_SIMPLIFY cases (expected values
    /// as ES prints them).
    #[test]
    fn st_simplify_matches_es_csv_spec() {
        let s = |w: &str, t: f64| simplify_wkt(w, t).unwrap();
        // stSimplifyLiteralPolygon: the ring start point is dropped too.
        assert_eq!(
            s("POLYGON ((7.998 53.827, 9.470 53.068, 15.754 53.801, 16.523 57.160, 11.162 57.868, 8.064 57.445, 6.219 55.317, 7.998 53.827))", 0.7),
            "POLYGON ((9.47 53.068, 15.754 53.801, 16.523 57.16, 8.064 57.445, 6.219 55.317, 9.47 53.068))"
        );
        let sq = "POLYGON((0 0, 1 0.1, 2 0, 2 2, 1 1.9, 0 2, 0 0))";
        // stSimplifyNoSimplification / stSimplifyWithSimplification /
        // stSimplifyEmptySimplification / stSimplifyWithIntegerTolerance.
        assert_eq!(s(sq, 0.01), "POLYGON ((0.0 0.0, 1.0 0.1, 2.0 0.0, 2.0 2.0, 1.0 1.9, 0.0 2.0, 0.0 0.0))");
        assert_eq!(s(sq, 0.2), "POLYGON ((0.0 0.0, 2.0 0.0, 2.0 2.0, 0.0 2.0, 0.0 0.0))");
        assert_eq!(s(sq, 2.0), "POLYGON EMPTY");
        // stSimplifyBbox.
        assert_eq!(s("BBOX (0, 2, 2, 0)", 2.0), "POLYGON EMPTY");
        assert_eq!(s("BBOX (0, 2, 2, 0)", 0.5), "POLYGON ((0.0 0.0, 2.0 0.0, 2.0 2.0, 0.0 2.0, 0.0 0.0))");
        // stSimplifyCartesianShape (point row) / stSimplifyCartesianPoint.
        assert_eq!(s("POINT(97.11 75.53)", 0.2), "POINT (97.11 75.53)");
        // stSimplifyPostGisFifthExample: lines keep their end points.
        assert_eq!(
            s("MULTILINESTRING ((20 180, 20 150, 50 150, 50 100, 110 150, 150 140, 170 120), (20 10, 80 30, 90 120), (90 120, 130 130), (130 130, 130 70, 160 40, 180 60, 180 90, 140 80), (50 40, 70 40, 80 70, 70 60, 60 60, 50 50, 50 40))", 40.0),
            "MULTILINESTRING ((20.0 180.0, 50.0 100.0, 170.0 120.0), (20.0 10.0, 90.0 120.0), (90.0 120.0, 130.0 130.0), (130.0 130.0, 160.0 40.0, 180.0 90.0, 140.0 80.0), (50.0 40.0, 80.0 70.0, 50.0 40.0))"
        );
        // stSimplifyLiteralPoints (folded multi-value).
        let pts: Vec<String> = ["POINT (7.998 53.827)", "POINT (9.470 53.068)"].iter().map(|p| p.to_string()).collect();
        assert_eq!(simplify_wkts(&pts, 0.1).unwrap(), "GEOMETRYCOLLECTION (POINT (7.998 53.827), POINT (9.47 53.068))");
    }

    /// The PostGIS circle examples of `spatial-jts.csv-spec`
    /// (`stSimplifyPostGisThirdExample` / `...FourthExample`): a 49-point
    /// circle of radius 10 around (1, 3).
    #[test]
    fn st_simplify_postgis_circle() {
        let circle = "POLYGON((11 3,10.914448613738104 1.694738077799484,10.659258262890683 0.411809548974793,10.238795325112868 -0.826834323650898,9.660254037844387 -1.999999999999999,8.933533402912353 -3.087614290087205,8.071067811865476 -4.071067811865475,7.087614290087206 -4.933533402912351,6.000000000000001 -5.660254037844386,4.826834323650898 -6.238795325112868,3.58819045102521 -6.659258262890681,2.305261922200517 -6.914448613738104,1 -7,-0.305261922200514 -6.914448613738106,-1.588190451025206 -6.659258262890683,-2.826834323650895 -6.238795325112868,-3.999999999999998 -5.660254037844387,-5.087614290087204 -4.933533402912354,-6.071067811865475 -4.071067811865476,-6.93353340291235 -3.087614290087209,-7.660254037844386 -2.000000000000004,-8.238795325112868 -0.826834323650899,-8.659258262890681 0.41180954897479,-8.914448613738104 1.69473807779948,-9 3,-8.914448613738106 4.305261922200513,-8.659258262890685 5.588190451025204,-8.23879532511287 6.826834323650893,-7.660254037844389 7.999999999999997,-6.933533402912354 9.087614290087203,-6.071067811865479 10.071067811865472,-5.087614290087209 10.93353340291235,-4.000000000000004 11.660254037844384,-2.826834323650903 12.238795325112864,-1.588190451025215 12.659258262890681,-0.305261922200525 12.914448613738102,1 13,2.305261922200513 12.914448613738106,3.588190451025203 12.659258262890685,4.826834323650892 12.23879532511287,5.999999999999993 11.66025403784439,7.087614290087199 10.933533402912357,8.071067811865474 10.071067811865477,8.93353340291235 9.08761429008721,9.660254037844384 8.000000000000004,10.238795325112864 6.826834323650904,10.659258262890681 5.588190451025216,10.914448613738102 4.305261922200526,11 3))";
        assert_eq!(
            simplify_wkt(circle, 1.0).unwrap(),
            "POLYGON ((11.0 3.0, 8.071067811865476 -4.071067811865475, 1.0 -7.0, -6.071067811865475 -4.071067811865476, -9.0 3.0, -6.071067811865479 10.071067811865472, 1.0 13.0, 8.071067811865474 10.071067811865477, 11.0 3.0))"
        );
        assert_eq!(simplify_wkt(circle, 100.0).unwrap(), "POLYGON EMPTY");
    }

    /// JTS result shapes beyond the csv-spec: one surviving part becomes
    /// that part; collapsed parts and members are dropped.
    #[test]
    fn st_simplify_result_shapes() {
        let s = |w: &str, t: f64| simplify_wkt(w, t).unwrap();
        // A closed line keeps both end points.
        assert_eq!(s("LINESTRING (0 0, 0.001 0.001, 0 0.002, 0 0)", 0.01), "LINESTRING (0.0 0.0, 0.0 0.0)");
        assert_eq!(s("MULTILINESTRING ((0 0, 0.001 0.0001, 1 0))", 0.01), "LINESTRING (0.0 0.0, 1.0 0.0)");
        assert_eq!(s("MULTIPOINT (1 2)", 0.01), "POINT (1.0 2.0)");
        // A tiny building collapses; a large polygon survives alone.
        let tiny = "((2.3 48.8, 2.3002 48.8, 2.3002 48.8002, 2.3 48.8002, 2.3 48.8))";
        let big = "((0 0, 1 0, 1 1, 0 1, 0 0))";
        assert_eq!(s(&format!("POLYGON {tiny}"), 0.01), "POLYGON EMPTY");
        assert_eq!(s(&format!("MULTIPOLYGON ({tiny})"), 0.01), "MULTIPOLYGON EMPTY");
        assert_eq!(s(&format!("MULTIPOLYGON ({tiny}, {big})"), 0.01), "POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 1.0, 0.0 0.0))");
        // A collapsed hole is dropped.
        assert_eq!(
            s("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0), (0.5 0.5, 0.5001 0.5, 0.5001 0.5001, 0.5 0.5))", 0.01),
            "POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 1.0, 0.0 0.0))"
        );
        assert_eq!(s(&format!("GEOMETRYCOLLECTION (POINT (1 2), POLYGON {tiny})"), 0.01), "GEOMETRYCOLLECTION (POINT (1.0 2.0))");
        // Long lines simplify without recursion.
        let long: Vec<String> = (0..200_000).map(|i| format!("{} {}", i as f64 * 1e-5, if i % 2 == 0 { 0.0 } else { 1e-6 })).collect();
        assert_eq!(s(&format!("LINESTRING ({})", long.join(", ")), 0.01), "LINESTRING (0.0 0.0, 1.9999900000000002 1.0E-6)");
    }

    #[test]
    fn wkt_formatting() {
        assert_eq!(from_source(&json!("POINT(1 -1)"), "geo_point").unwrap(), "POINT (1.0 -1.0)");
        assert_eq!(from_source(&json!({"lat": 45.5, "lon": -73.25}), "geo_point").unwrap(), "POINT (-73.25 45.5)");
        assert_eq!(from_source(&json!("3.5,4"), "cartesian_point").unwrap(), "POINT (3.5 4.0)");
        assert_eq!(from_source(&json!("POINT(4321 -1234)"), "cartesian_point").unwrap(), "POINT (4321.0 -1234.0)");
        assert_eq!(
            from_source(&json!("POLYGON((-1 -1, 1 -1, 1 1, -1 1, -1 -1))"), "geo_shape").unwrap(),
            "POLYGON ((-1.0 -1.0, 1.0 -1.0, 1.0 1.0, -1.0 1.0, -1.0 -1.0))"
        );
        assert_eq!(parse_wkt("POINT(1e10 2.123456789)", "cartesian_point").unwrap(), "POINT (1.0E10 2.123456789)");
        assert_eq!(parse_wkt("bad", "geo_point"), None);
    }
}
