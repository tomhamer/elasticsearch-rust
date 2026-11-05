//! Geometry values (`geo_shape`, `shape`, `point`, `geo_shape` runtime
//! fields) rendered for the `fields` API the way ES writes them: GeoJSON
//! (`GeoJson.toXContent`) by default, WKT (`WellKnownText.toWKT`) with
//! `format: wkt`.

use crate::values::java_double;
use rustic_core::fieldtypes::shape::{self, Geometry, ParseOpts, Pt};
use serde_json::{json, Value};

/// Java `Double.toString`.
fn jd(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let a = v.abs();
    if a == 0.0 || (1e-3..1e7).contains(&a) {
        let s = format!("{v}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap();
        let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
        format!("{m}E{e}")
    }
}

fn c(p: &Pt) -> Value {
    json!([java_double(p.0), java_double(p.1)])
}

fn cs(v: &[Pt]) -> Value {
    Value::Array(v.iter().map(c).collect())
}

fn rings(r: &[Vec<Pt>]) -> Value {
    Value::Array(r.iter().map(|l| cs(l)).collect())
}

/// GeoJSON of a geometry.
pub fn geojson(g: &Geometry) -> Value {
    match g {
        Geometry::Point(p) => json!({ "coordinates": c(p), "type": "Point" }),
        Geometry::MultiPoint(v) => json!({ "coordinates": cs(v), "type": "MultiPoint" }),
        Geometry::LineString(v) => json!({ "coordinates": cs(v), "type": "LineString" }),
        Geometry::MultiLineString(v) => json!({ "coordinates": rings(v), "type": "MultiLineString" }),
        Geometry::Polygon(r) => json!({ "coordinates": rings(r), "type": "Polygon" }),
        Geometry::MultiPolygon(v) => json!({ "coordinates": Value::Array(v.iter().map(|p| rings(p)).collect()), "type": "MultiPolygon" }),
        Geometry::Collection(v) => json!({ "geometries": Value::Array(v.iter().map(geojson).collect()), "type": "GeometryCollection" }),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => json!({ "coordinates": [c(&(*min_x, *max_y)), c(&(*max_x, *min_y))], "type": "Envelope" }),
    }
}

fn wpts(v: &[Pt]) -> String {
    v.iter().map(|p| format!("{} {}", jd(p.0), jd(p.1))).collect::<Vec<_>>().join(", ")
}

fn wpoly(r: &[Vec<Pt>]) -> String {
    format!("({})", r.iter().map(|l| format!("({})", wpts(l))).collect::<Vec<_>>().join(", "))
}

/// WKT of a geometry (multi-geometry members are joined with `,`, points
/// and polygon rings with `, `, as ES does).
pub fn wkt(g: &Geometry) -> String {
    match g {
        Geometry::Point(p) => format!("POINT ({} {})", jd(p.0), jd(p.1)),
        Geometry::MultiPoint(v) if v.is_empty() => "MULTIPOINT EMPTY".into(),
        Geometry::MultiPoint(v) => format!("MULTIPOINT ({})", wpts(v)),
        Geometry::LineString(v) if v.is_empty() => "LINESTRING EMPTY".into(),
        Geometry::LineString(v) => format!("LINESTRING ({})", wpts(v)),
        Geometry::MultiLineString(v) if v.is_empty() => "MULTILINESTRING EMPTY".into(),
        Geometry::MultiLineString(v) => format!("MULTILINESTRING ({})", v.iter().map(|l| format!("({})", wpts(l))).collect::<Vec<_>>().join(",")),
        Geometry::Polygon(r) if r.is_empty() => "POLYGON EMPTY".into(),
        Geometry::Polygon(r) => format!("POLYGON {}", wpoly(r)),
        Geometry::MultiPolygon(v) if v.is_empty() => "MULTIPOLYGON EMPTY".into(),
        Geometry::MultiPolygon(v) => format!("MULTIPOLYGON ({})", v.iter().map(|p| wpoly(p)).collect::<Vec<_>>().join(",")),
        Geometry::Collection(v) if v.is_empty() => "GEOMETRYCOLLECTION EMPTY".into(),
        Geometry::Collection(v) => format!("GEOMETRYCOLLECTION ({})", v.iter().map(wkt).collect::<Vec<_>>().join(",")),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => format!("BBOX ({}, {}, {}, {})", jd(*min_x), jd(*max_x), jd(*max_y), jd(*min_y)),
    }
}

/// A geometry in the requested `fields` format.
pub fn render(g: &Geometry, format: Option<&str>) -> Value {
    match format {
        Some(f) if f.eq_ignore_ascii_case("wkt") => Value::String(wkt(g)),
        _ => geojson(g),
    }
}

/// One source value of a `geo_shape` field as ES fetches it: parsed and
/// dateline-normalized. `None` when it does not parse.
pub fn fetch_geo_shape(v: &Value, format: Option<&str>) -> Option<Value> {
    let g = shape::parse(v, &ParseOpts { geo: true, coerce: true, ignore_z: true }).ok()?;
    Some(render(&shape::normalize_geo(&g), format))
}

/// One source value of a `shape` / `point` field.
pub fn fetch_cartesian(v: &Value, point: bool, format: Option<&str>) -> Option<Value> {
    let o = ParseOpts { geo: false, coerce: true, ignore_z: true };
    let g = if point {
        Geometry::Point(super::index::parse_xy_point(v, &o).ok()?)
    } else {
        shape::parse(v, &o).ok()?
    };
    Some(render(&g, format))
}

/// Parse emitted runtime `geo_shape` values (WKT strings, GeoJSON objects,
/// lists of either) into one geometry (a collection when several).
pub fn runtime_geometry(vals: &[Value]) -> Option<Geometry> {
    fn add(v: &Value, out: &mut Vec<Geometry>) {
        let o = ParseOpts { geo: true, coerce: true, ignore_z: true };
        match v {
            Value::Array(a) if !a.iter().all(Value::is_number) => a.iter().for_each(|x| add(x, out)),
            Value::Null => {}
            other => {
                if let Ok(g) = shape::parse(other, &o) {
                    out.push(g);
                }
            }
        }
    }
    let mut gs = Vec::new();
    for v in vals {
        add(v, &mut gs);
    }
    match gs.len() {
        0 => None,
        1 => gs.pop(),
        _ => Some(Geometry::Collection(gs)),
    }
}

/// Source values at `path` (numeric arrays kept whole: they are points).
fn shape_source_values<'v>(src: &'v Value, path: &str) -> Vec<&'v Value> {
    fn walk<'v>(v: &'v Value, parts: &[&str], out: &mut Vec<&'v Value>) {
        if parts.is_empty() {
            match v {
                Value::Array(a) if !a.is_empty() && a.iter().all(Value::is_number) => out.push(v),
                Value::Array(a) => a.iter().for_each(|x| walk(x, parts, out)),
                Value::Null => {}
                other => out.push(other),
            }
            return;
        }
        match v {
            Value::Object(m) => {
                for (k, c) in m {
                    if let Some(i) = (1..=parts.len()).find(|i| parts[..*i].join(".") == *k) {
                        walk(c, &parts[i..], out);
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, parts, out)),
            _ => {}
        }
    }
    let parts: Vec<&str> = path.split('.').collect();
    let mut out = Vec::new();
    walk(src, &parts, &mut out);
    out
}

/// `fields` API values of a `geo_shape` / `shape` / `point` field.
pub fn fetch_values(ty: crate::mapping::EsType, src: &Value, path: &str, format: Option<&str>) -> Vec<Value> {
    use crate::mapping::EsType;
    // `mvt(z/x/y[@extent][:buffer])`: the doc's values (one geometry, or
    // a collection of them) as vector tile features of that tile.
    if let (EsType::GeoShape, Some(spec)) = (ty, format.and_then(super::mvt_feature::parse_spec)) {
        let mut gs: Vec<Geometry> = shape_source_values(src, path)
            .into_iter()
            .filter_map(|v| shape::parse(v, &ParseOpts { geo: true, coerce: true, ignore_z: true }).ok())
            .map(|g| shape::normalize_geo(&g))
            .collect();
        let g = match gs.len() {
            0 => return Vec::new(),
            1 => gs.pop().unwrap(),
            _ => Geometry::Collection(gs),
        };
        return super::mvt_feature::features(&g, spec).into_iter().map(Value::String).collect();
    }
    shape_source_values(src, path)
        .into_iter()
        .filter_map(|v| match ty {
            EsType::GeoShape => fetch_geo_shape(v, format),
            EsType::GeoPoint => crate::values::parse_geo_point(v).map(|(lat, lon)| match format {
                Some(f) if f.eq_ignore_ascii_case("wkt") => Value::String(format!("POINT ({} {})", jd(lon), jd(lat))),
                _ => json!({ "coordinates": [java_double(lon), java_double(lat)], "type": "Point" }),
            }),
            _ => fetch_cartesian(v, ty == EsType::Point, format),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ES `FastMath` (jafama) `sinh` / `atan`, which `GeoTileUtils` uses for tile
// latitudes: a port so tile boxes match ES bit for bit. `StrictMath` calls
// map to `libm` (both are fdlibm).

pub(crate) mod fast_math {
    use std::sync::OnceLock;

    const TABS: usize = 8192;

    struct AtanTabs {
        max: f64,
        delta: f64,
        indexer: f64,
        tab: Vec<[f64; 5]>,
    }

    fn tabs() -> &'static AtanTabs {
        static T: OnceLock<AtanTabs> = OnceLock::new();
        T.get_or_init(|| {
            let max = libm::tan(74.0f64.to_radians());
            let delta = max / 8191.0;
            let indexer = 1.0 / delta;
            let tab = (0..TABS)
                .map(|i| {
                    let x = i as f64 * delta;
                    let inv = 1.0 / (1.0 + x * x);
                    let inv2 = inv * inv;
                    let inv3 = inv2 * inv;
                    let inv4 = inv2 * inv2;
                    [libm::atan(x), inv, -2.0 * x * inv2 * 0.5, (-2.0 + 6.0 * x * x) * inv3 * 0.16666666666666666, 24.0 * x * (1.0 - x * x) * inv4 * 0.041666666666666664]
                })
                .collect();
            AtanTabs { max, delta, indexer, tab }
        })
    }

    fn b(bits: i64) -> f64 {
        f64::from_bits(bits as u64)
    }

    pub fn atan(value: f64) -> f64 {
        let (value, neg) = if value < 0.0 { (-value, true) } else { (value, false) };
        if value == 1.0 {
            return if neg { -std::f64::consts::FRAC_PI_4 } else { std::f64::consts::FRAC_PI_4 };
        }
        let t = tabs();
        if value <= t.max {
            let index = (value * t.indexer + 0.5) as i32 as usize;
            let delta = value - index as f64 * t.delta;
            let r = &t.tab[index];
            let result = r[0] + delta * (r[1] + delta * (r[2] + delta * (r[3] + delta * r[4])));
            return if neg { -result } else { result };
        }
        if value < b(4904419994206470144) {
            let x = -1.0 / value;
            let x2 = x * x;
            let x4 = x2 * x2;
            let (at0, at1, at2, at3, at4, at5) = (b(4599676419421066509), b(-4626998257160492092), b(4594314991288484863), b(-4630701217362536847), b(4591215095208222830), b(-4633165035261879699));
            let (at6, at7, at8, at9, at10) = (b(4589464229703073105), b(-4634804155249132134), b(4587333258118041067), b(-4637946461342241745), b(4580351289466214929));
            let s1 = x2 * (at0 + x4 * (at2 + x4 * (at4 + x4 * (at6 + x4 * (at8 + x4 * at10)))));
            let s2 = x4 * (at1 + x4 * (at3 + x4 * (at5 + x4 * (at7 + x4 * at9))));
            let result = b(4609753056924675352) - ((x * (s1 + s2) - b(4364452196894661639)) - x);
            return if neg { -result } else { result };
        }
        if value.is_nan() {
            return f64::NAN;
        }
        if neg { -std::f64::consts::FRAC_PI_2 } else { std::f64::consts::FRAC_PI_2 }
    }

    pub fn sinh(value: f64) -> f64 {
        let (value, h) = if value < 0.0 { (-value, -0.5) } else { (value, 0.5) };
        if value < 22.0 {
            if value < b(4481081629233643520) {
                return if h < 0.0 { -value } else { value };
            }
            let t = libm::expm1(value);
            return h * (t + t / (t + 1.0));
        }
        if value < libm::log(f64::MAX) {
            return h * libm::exp(value);
        }
        let t = libm::exp(value * 0.5);
        h * t * t
    }
}

/// ES `GeoTileUtils.toBoundingBox`: `(top, left, bottom, right)` of tile
/// `zoom/x/y`, computed exactly as ES does.
pub fn tile_bbox(zoom: u32, x: u32, y: u32) -> (f64, f64, f64, f64) {
    use std::f64::consts::PI;
    let tiles = (1u64 << zoom) as f64;
    let lat = |n: f64| fast_math::atan(fast_math::sinh(n)).to_degrees();
    let min_n = PI - (2.0 * PI * (y as f64 + 1.0)) / tiles;
    let max_n = PI - (2.0 * PI * y as f64) / tiles;
    let min_x = (x as f64 / tiles * 360.0) - 180.0;
    let max_x = ((x as f64 + 1.0) / tiles * 360.0) - 180.0;
    (lat(max_n), min_x, lat(min_n), max_x)
}

#[cfg(test)]
mod tile_tests {
    #[test]
    fn es_tile_bbox() {
        // ES 8.19 geo_grid ingest output for tile 4/8/5.
        assert_eq!(super::tile_bbox(4, 8, 5), (55.77657301866769, 0.0, 40.979898069620134, 22.5));
    }
}

// ---------------------------------------------------------------------------
// H3 cells for vector tiles.

/// Boundary `(lon, lat)` vertices (open ring, longitudes unwrapped to be
/// continuous) of an H3 cell address.
pub fn h3_boundary(key: &str) -> Option<Vec<(f64, f64)>> {
    let c = crate::aggs::geo_hex::parse_cell(key)?;
    let mut ring = crate::aggs::geo_hex::cell_polys(c).into_iter().next()?;
    ring.pop();
    Some(ring)
}

/// Center `(lon, lat)` of an H3 cell address.
pub fn h3_center(key: &str) -> Option<(f64, f64)> {
    let c = crate::aggs::geo_hex::parse_cell(key)?;
    let ll = crate::h3::to_lat_lng(c);
    Some((ll.lon_deg(), ll.lat_deg()))
}

/// ES `GridAggregation.GEOHEX`: H3 resolution for a vector tile at `zoom`
/// with `grid_precision`.
pub fn h3_resolution_for_zoom(zoom: u32, grid_precision: u32) -> u32 {
    const ZOOM2RES: [u32; 30] = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 11, 12, 13, 14, 14, 15, 15, 15, 15];
    ZOOM2RES[((zoom + grid_precision) as usize).min(ZOOM2RES.len() - 1)]
}
