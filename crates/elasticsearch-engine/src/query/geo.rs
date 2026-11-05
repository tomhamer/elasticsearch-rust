//! Geo queries with ES 8.19 parsing / validation semantics:
//! `geo_bounding_box`, `geo_distance`, `geo_polygon` and `geo_grid`.
//!
//! Errors are split the way ES splits them: problems found while parsing the
//! request are thrown directly (`illegal_argument_exception`,
//! `parsing_exception`, `parse_exception`), problems found while building the
//! Lucene query for a shard (coordinate validation, field types) are
//! `query_shard_exception`s.

use super::dsl::{parse_query, QueryContext};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use rustic_core::geo;
use rustic_core::query::Query;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Helpers.

/// Java `Double.toString` for the values these messages print.
pub(crate) fn jd(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.fract() == 0.0 && x.abs() < 1e7 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

fn shard(ctx: &QueryContext, r: impl Into<String>) -> EsError {
    EsError::query_shard(ctx.index_name, ctx.index_uuid, r)
}

fn parse_exc(r: impl Into<String>) -> EsError {
    EsError::new(400, "parse_exception", r)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Validation {
    Strict,
    Coerce,
    IgnoreMalformed,
}

fn validation_of(v: &Value) -> Result<Validation> {
    let s = v.as_str().unwrap_or("");
    match s.to_ascii_uppercase().as_str() {
        "STRICT" => Ok(Validation::Strict),
        "COERCE" => Ok(Validation::Coerce),
        "IGNORE_MALFORMED" => Ok(Validation::IgnoreMalformed),
        _ => Err(EsError::illegal_argument(format!("operator needs to be either [COERCE, IGNORE_MALFORMED, STRICT], but not [{s}]"))),
    }
}

fn centered_modulus(x: f64, m: f64) -> f64 {
    let mut r = x % m;
    if r <= 0.0 {
        r += m;
    }
    if r > m / 2.0 {
        r -= m;
    }
    r
}

/// ES `GeoUtils.normalizePoint(point, true, true)`.
fn normalize_point(lat: f64, lon: f64) -> (f64, f64) {
    let (mut lat, mut lon) = (lat, lon);
    let norm_lat = lat > 90.0 || lat < -90.0;
    let norm_lon = lon > 180.0 || lon < -180.0;
    if norm_lat {
        lat = centered_modulus(lat, 360.0);
        let mut shift = true;
        if lat < -90.0 {
            lat = -180.0 - lat;
        } else if lat > 90.0 {
            lat = 180.0 - lat;
        } else {
            shift = false;
        }
        if shift {
            if norm_lon {
                lon += 180.0;
            } else {
                lon += if lon > 0.0 { -180.0 } else { 180.0 };
            }
        }
    }
    if norm_lon {
        lon = centered_modulus(lon, 360.0);
    }
    (lat, lon)
}

fn lat_ok(lat: f64) -> bool {
    (-90.0..=90.0).contains(&lat)
}
fn lon_ok(lon: f64) -> bool {
    (-180.0..=180.0).contains(&lon)
}

fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

const B32: &str = "0123456789bcdefghjkmnpqrstuvwxyz";

/// Strict geohash cell `(top, left, bottom, right)`.
fn geohash_cell(h: &str) -> Result<(f64, f64, f64, f64)> {
    if let Some(c) = h.chars().find(|c| !B32.contains(*c)) {
        return Err(EsError::illegal_argument(format!("unsupported symbol [{c}] in geohash [{h}]")));
    }
    geo::geohash_bbox(h).ok_or_else(|| EsError::illegal_argument(format!("unsupported symbol [?] in geohash [{h}]")))
}

/// A parsed point: either an exact point or a geohash cell (whose corner is
/// used by bounding boxes). Coordinates may be NaN (missing `lat` / `lon`).
struct Pt {
    lat: f64,
    lon: f64,
    cell: Option<(f64, f64, f64, f64)>,
}

/// ES `GeoUtils.parseGeoPoint` with `ignore_z_value = false`.
fn parse_point(v: &Value, qname: &str, field: &str) -> Result<Pt> {
    let pt = |lat: f64, lon: f64| Pt { lat, lon, cell: None };
    match v {
        Value::Object(m) => {
            if let Some(k) = m.keys().find(|k| !matches!(k.as_str(), "lat" | "lon" | "geohash")) {
                let _ = k;
                return Err(EsError::parsing(format!("[{qname}] query does not support [{field}]")));
            }
            if let Some(Value::String(h)) = m.get("geohash") {
                let c = geohash_cell(h)?;
                return Ok(Pt { lat: c.2, lon: c.1, cell: Some(c) });
            }
            let g = |k: &str, what: &str| -> Result<f64> {
                match m.get(k) {
                    None => Ok(f64::NAN),
                    Some(x) => num_of(x).ok_or_else(|| parse_exc(format!("{what} must be a number"))),
                }
            };
            Ok(pt(g("lat", "latitude")?, g("lon", "longitude")?))
        }
        Value::Array(a) => {
            let n = |i: usize| a.get(i).and_then(num_of);
            match (n(0), n(1)) {
                (Some(lon), Some(lat)) => {
                    if a.len() > 2 {
                        let z = n(2).unwrap_or(f64::NAN);
                        return Err(parse_exc(format!("Exception parsing coordinates: found Z value [{}] but [ignore_z_value] parameter is [false]", jd(z))));
                    }
                    Ok(pt(lat, lon))
                }
                _ => Err(parse_exc("geo_point expected")),
            }
        }
        Value::String(s) => {
            let t = s.trim();
            let up = t.to_ascii_uppercase();
            if up.starts_with("POINT") {
                let inner = t[5..].trim().trim_start_matches('(').trim_end_matches(')');
                let mut it = inner.split_whitespace().map(|x| x.parse::<f64>());
                return match (it.next(), it.next()) {
                    (Some(Ok(lon)), Some(Ok(lat))) => Ok(pt(lat, lon)),
                    _ => Err(parse_exc(format!("Invalid WKT point [{t}]"))),
                };
            }
            if let Some((a, b)) = t.split_once(',') {
                let lat: f64 = a.trim().parse().map_err(|_| parse_exc("latitude must be a number"))?;
                let lon: f64 = b.trim().parse().map_err(|_| parse_exc("longitude must be a number"))?;
                return Ok(pt(lat, lon));
            }
            let c = geohash_cell(t)?;
            // ES: a geohash point is its cell's lower-left corner.
            Ok(Pt { lat: c.2, lon: c.1, cell: Some(c) })
        }
        _ => Err(parse_exc("geo_point expected")),
    }
}

/// Field of a geo query: `Ok(None)` = unmapped and ignored.
fn geo_field(ctx: &QueryContext, field: &str, ignore_unmapped: bool, qname: &str, unmapped_msg: &str) -> Result<Option<(u16, String)>> {
    match ctx.lookup.get(field) {
        None if ignore_unmapped => Ok(None),
        None => Err(shard(ctx, format!("{unmapped_msg} [{field}]"))),
        Some(f) if f.declared == EsType::GeoPoint => match f.geo_id {
            Some(g) => Ok(Some((g, f.name.clone()))),
            None => Err(shard(ctx, format!("field [{}] has no geo index; reindex to use [{qname}]", f.name))),
        },
        Some(f) => Err(shard(ctx, format!("Field [{}] is of unsupported type [{}] for [{qname}] query", f.name, f.declared.name()))),
    }
}

/// A `geo_shape` field targeted by a point query (`geo_bounding_box`,
/// `geo_distance`): ES runs the field's `geoShapeQuery` with the query's
/// geometry, `intersects`.
fn shape_field(ctx: &QueryContext, field: &str) -> Result<Option<String>> {
    match ctx.lookup.get(field) {
        Some(f) if f.declared == EsType::GeoShape => {
            if !f.indexed && !f.doc_values {
                return Err(shard(ctx, format!("failed to create query: Cannot search on field [{}] since it is not indexed nor has doc values.", f.name)));
            }
            Ok(Some(f.name.clone()))
        }
        _ => Ok(None),
    }
}

/// The `geo_shape` query of `geometry` on `field` (constant score = boost).
fn shape_intersects(field: &str, geometry: Value, o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    let boost = o.get("boost").and_then(num_of).unwrap_or(1.0);
    parse_query(&json!({"geo_shape": {field: {"shape": geometry, "relation": "intersects"}, "boost": boost}}), ctx)
}

fn boost_q(q: Query, o: &Map<String, Value>) -> Query {
    let b = o.get("boost").and_then(num_of).unwrap_or(1.0) as f32;
    if (b - 1.0).abs() > f32::EPSILON {
        Query::Boost { query: Box::new(q), boost: b }
    } else {
        q
    }
}

const COMMON: &[&str] = &["validation_method", "ignore_unmapped", "boost", "_name"];

// ---------------------------------------------------------------------------
// geo_bounding_box.

pub fn geo_bounding_box(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let q = "geo_bounding_box";
    let o = body.as_object().ok_or_else(|| EsError::parsing("[geo_bounding_box] query malformed, no start_object after query name"))?;
    let unexpected = |k: &str| EsError::parsing(format!("failed to parse [{q}] query. unexpected field [{k}]"));
    let mut validation = Validation::Strict;
    let mut ignore_unmapped = false;
    let mut spec: Option<(&String, &Value)> = None;
    for (k, v) in o {
        match k.as_str() {
            "validation_method" => validation = validation_of(v)?,
            "ignore_unmapped" => ignore_unmapped = v.as_bool().unwrap_or(false),
            "boost" | "_name" => {}
            "type" | "ignore_malformed" | "coerce" => return Err(unexpected(k)),
            _ => spec = Some((k, v)),
        }
    }
    let (field, spec) = spec.ok_or_else(|| EsError::parsing(format!("failed to parse [{q}] query. bounding box not provided")))?;
    let so = spec.as_object().ok_or_else(|| EsError::parsing(format!("failed to parse [{q}] query. unexpected token [{}]", "VALUE")))?;
    let (mut top, mut left, mut bottom, mut right) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    let wrap = |e: EsError| parse_exc(format!("failed to parse [{q}] query. [{}]", e.reason));
    for (k, v) in so {
        match k.as_str() {
            "top_left" | "bottom_right" | "top_right" | "bottom_left" => {
                let p = parse_point(v, q, field).map_err(wrap)?;
                let (lat, lon) = match (k.as_str(), p.cell) {
                    ("top_left", Some(c)) => (c.0, c.1),
                    ("bottom_right", Some(c)) => (c.2, c.3),
                    ("top_right", Some(c)) => (c.0, c.3),
                    ("bottom_left", Some(c)) => (c.2, c.1),
                    _ => (p.lat, p.lon),
                };
                match k.as_str() {
                    "top_left" => (top, left) = (lat, lon),
                    "bottom_right" => (bottom, right) = (lat, lon),
                    "top_right" => (top, right) = (lat, lon),
                    _ => (bottom, left) = (lat, lon),
                }
            }
            "top" | "left" | "bottom" | "right" => {
                let x = num_of(v).ok_or_else(|| parse_exc(format!("failed to parse [{q}] query. [{k} must be a number]")))?;
                match k.as_str() {
                    "top" => top = x,
                    "left" => left = x,
                    "bottom" => bottom = x,
                    _ => right = x,
                }
            }
            "wkt" => {
                let w = v.as_str().unwrap_or("").trim();
                let word: String = w.chars().take_while(|c| c.is_ascii_alphabetic()).collect::<String>().to_ascii_uppercase();
                if word != "BBOX" && word != "ENVELOPE" {
                    return Err(parse_exc(format!("failed to parse [{q}] query. [failed to parse WKT bounding box. [{word}] found. expected [ENVELOPE]]")));
                }
                let inner = w[word.len()..].trim().trim_start_matches('(').trim_end_matches(')');
                let n: Vec<f64> = inner.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if n.len() != 4 {
                    return Err(parse_exc(format!("failed to parse [{q}] query. [failed to parse WKT bounding box]")));
                }
                (left, right, top, bottom) = (n[0], n[1], n[2], n[3]);
            }
            other => return Err(unexpected(other)),
        }
    }
    if top.is_nan() {
        return Err(EsError::illegal_argument("top latitude is invalid: NaN"));
    }
    if bottom.is_nan() {
        return Err(EsError::illegal_argument("bottom latitude is invalid: NaN"));
    }
    if left.is_nan() {
        return Err(EsError::illegal_argument("left longitude is invalid: NaN"));
    }
    if right.is_nan() {
        return Err(EsError::illegal_argument("right longitude is invalid: NaN"));
    }
    if top < bottom {
        return Err(EsError::illegal_argument(format!("top is below bottom corner: {} vs. {}", jd(top), jd(bottom))));
    }
    let envelope = json!({"type": "envelope", "coordinates": [[left, top], [right, bottom]]});
    if let Some(q) = crate::script::search::runtime_geo_query(field, &envelope, ctx)? {
        return Ok(boost_q(q, o));
    }
    if let Some(sf) = shape_field(ctx, field)? {
        return shape_intersects(&sf, envelope, o, ctx);
    }
    let Some((g, name)) = geo_field(ctx, field, ignore_unmapped, q, "failed to find geo field")? else { return Ok(Query::MatchNone) };
    match validation {
        Validation::Strict => {
            if !(lat_ok(top) && lat_ok(bottom) && lon_ok(left) && lon_ok(right)) {
                return Err(shard(ctx, "couldn't validate latitude/ longitude values"));
            }
        }
        Validation::Coerce => {
            (top, left) = normalize_point(top, left);
            (bottom, right) = normalize_point(bottom, right);
        }
        Validation::IgnoreMalformed => {
            for lat in [top, bottom] {
                if !lat_ok(lat) {
                    return Err(shard(ctx, format!("Exception creating query on Field [{name}] invalid latitude {}; must be between -90.0 and 90.0", jd(lat))));
                }
            }
            for lon in [left, right] {
                if !lon_ok(lon) {
                    return Err(shard(ctx, format!("Exception creating query on Field [{name}] invalid longitude {}; must be between -180.0 and 180.0", jd(lon))));
                }
            }
        }
    }
    // ES quantizes query geometries before handing them to Lucene, just as
    // indexed points are quantized. Raw lower bounds can otherwise exclude
    // a point indexed exactly on the boundary after its decoded value rounds
    // down (e.g. latitude -40).
    let (top, left) = rustic_core::geo::quantize(top, left);
    let (bottom, right) = rustic_core::geo::quantize(bottom, right);
    Ok(boost_q(Query::GeoBoundingBox { field: g, top, left, bottom, right }, o))
}

// ---------------------------------------------------------------------------
// geo_distance.

/// ES `DistanceUnit` names in enum order (suffix match).
const UNITS: &[(&[&str], f64)] = &[
    (&["in", "inch"], 0.0254),
    (&["yd", "yards"], 0.9144),
    (&["ft", "feet"], 0.3048),
    (&["km", "kilometers"], 1000.0),
    (&["NM", "nmi", "nauticalmiles"], 1852.0),
    (&["mm", "millimeters"], 0.001),
    (&["cm", "centimeters"], 0.01),
    (&["mi", "miles"], 1609.344),
    (&["m", "meters"], 1.0),
];

fn java_double(s: &str) -> Result<f64> {
    let t = s.trim();
    if t.is_empty() {
        return Err(EsError::new(400, "number_format_exception", if s.is_empty() { "empty String".to_string() } else { "empty String".to_string() }));
    }
    let ok = t.chars().all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'));
    match (ok, t.parse::<f64>()) {
        (true, Ok(x)) => Ok(x),
        _ => Err(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))),
    }
}

/// `100km`, `5 mi`, `1.5e2m`, `100` into meters (ES `DistanceUnit.parse`).
pub(crate) fn parse_distance_es(s: &str) -> Result<f64> {
    for (names, f) in UNITS {
        for n in *names {
            if let Some(num) = s.strip_suffix(n) {
                return java_double(num).map(|x| x * f);
            }
        }
    }
    java_double(s)
}

pub fn geo_distance(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let q = "geo_distance";
    let o = body.as_object().ok_or_else(|| EsError::parsing("[geo_distance] query malformed, no start_object after query name"))?;
    let mut validation = Validation::Strict;
    let mut ignore_unmapped = false;
    let mut distance: Option<f64> = None;
    let mut spec: Option<(&String, &Value)> = None;
    for (k, v) in o {
        match k.as_str() {
            "validation_method" => validation = validation_of(v)?,
            "ignore_unmapped" => ignore_unmapped = v.as_bool().unwrap_or(false),
            "boost" | "_name" | "unit" => {}
            "distance" => {
                distance = match v {
                    Value::Number(n) => n.as_f64(),
                    Value::String(s) => Some(parse_distance_es(s)?),
                    Value::Null => None,
                    other => return Err(EsError::x_content_parse(format!("Current token ({}) not numeric, can not use numeric value accessors", if other.is_boolean() { "VALUE_TRUE" } else { "START_OBJECT" }))),
                }
            }
            "distance_type" => {
                let s = v.as_str().unwrap_or("");
                if !matches!(s.to_ascii_lowercase().as_str(), "arc" | "plane") {
                    return Err(EsError::illegal_argument(format!("No geo distance for [{s}]")));
                }
            }
            _ => spec = Some((k, v)),
        }
    }
    let dist = distance.ok_or_else(|| EsError::parsing("geo_distance requires 'distance' to be specified"))?;
    let (field, spec) = spec.ok_or_else(|| EsError::illegal_argument("fieldName must not be null or empty"))?;
    if dist <= 0.0 {
        return Err(EsError::illegal_argument("distance must be greater than zero"));
    }
    let p = parse_point(spec, q, field)?;
    let circle = json!({"type": "circle", "coordinates": [p.lon, p.lat], "radius": format!("{dist}m")});
    if let Some(q) = crate::script::search::runtime_geo_query(field, &circle, ctx)? {
        return Ok(boost_q(q, o));
    }
    if let Some(sf) = shape_field(ctx, field)? {
        return shape_intersects(&sf, circle, o, ctx);
    }
    let Some((g, name)) = geo_field(ctx, field, ignore_unmapped, q, "failed to find geo field")? else { return Ok(Query::MatchNone) };
    let (mut lat, mut lon) = (p.lat, p.lon);
    match validation {
        Validation::Strict => {
            if !(lat_ok(lat) && lon_ok(lon)) {
                return Err(shard(ctx, "couldn't validate latitude/ longitude values"));
            }
        }
        Validation::Coerce => (lat, lon) = normalize_point(lat, lon),
        Validation::IgnoreMalformed => {}
    }
    let _ = name;
    if !(lat_ok(lat) && lon_ok(lon)) {
        return Ok(Query::MatchNone);
    }
    Ok(boost_q(Query::GeoDistance { field: g, lat, lon, meters: dist }, o))
}

// ---------------------------------------------------------------------------
// geo_polygon.

pub fn geo_polygon(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let q = "geo_polygon";
    let o = body.as_object().ok_or_else(|| EsError::parsing("[geo_polygon] query malformed, no start_object after query name"))?;
    let mut validation = Validation::Strict;
    let mut ignore_unmapped = false;
    let mut spec: Option<(&String, &Value)> = None;
    for (k, v) in o {
        match k.as_str() {
            "validation_method" => validation = validation_of(v)?,
            "ignore_unmapped" => ignore_unmapped = v.as_bool().unwrap_or(false),
            "boost" | "_name" => {}
            _ => spec = Some((k, v)),
        }
    }
    let (field, spec) = spec.ok_or_else(|| EsError::illegal_argument("fieldName must not be null"))?;
    let pts = spec.get("points").and_then(Value::as_array).filter(|a| !a.is_empty()).ok_or_else(|| EsError::illegal_argument("polygon must not be null or empty"))?;
    if pts.len() < 3 {
        return Err(EsError::illegal_argument("too few points defined for geo_polygon query"));
    }
    let mut points = Vec::with_capacity(pts.len());
    for p in pts {
        let p = parse_point(p, q, field)?;
        points.push((p.lat, p.lon));
    }
    let g = match ctx.lookup.get(field) {
        None if ignore_unmapped => return Ok(Query::MatchNone),
        None => return Err(shard(ctx, format!("failed to find geo_point field [{field}]"))),
        Some(f) if f.declared == EsType::GeoPoint => f.geo_id.ok_or_else(|| shard(ctx, format!("field [{}] has no geo index; reindex to use [geo_polygon]", f.name)))?,
        Some(f) => return Err(shard(ctx, format!("field [{}] is not a geo_point field", f.name))),
    };
    for p in points.iter_mut() {
        match validation {
            Validation::Strict => {
                if !lat_ok(p.0) {
                    return Err(shard(ctx, format!("illegal latitude value [{}] for [geo_polygon]", jd(p.0))));
                }
                if !lon_ok(p.1) {
                    return Err(shard(ctx, format!("illegal longitude value [{}] for [geo_polygon]", jd(p.1))));
                }
            }
            Validation::Coerce => *p = normalize_point(p.0, p.1),
            Validation::IgnoreMalformed => {
                if !lat_ok(p.0) {
                    return Err(shard(ctx, format!("failed to create query: invalid latitude {}; must be between -90.0 and 90.0", jd(p.0))));
                }
                if !lon_ok(p.1) {
                    return Err(shard(ctx, format!("failed to create query: invalid longitude {}; must be between -180.0 and 180.0", jd(p.1))));
                }
            }
        }
    }
    Ok(boost_q(Query::GeoPolygon { field: g, points }, o))
}

// ---------------------------------------------------------------------------
// geo_grid.

pub fn geo_grid(body: &Value, ctx: &QueryContext) -> Result<Query> {
    let q = "geo_grid";
    let o = body.as_object().ok_or_else(|| EsError::parsing("[geo_grid] query malformed, no start_object after query name"))?;
    let ignore_unmapped = o.get("ignore_unmapped").and_then(Value::as_bool).unwrap_or(false);
    let (field, spec) = o
        .iter()
        .find(|(k, _)| !matches!(k.as_str(), "boost" | "_name" | "ignore_unmapped"))
        .ok_or_else(|| EsError::parsing("[geo_grid] query malformed, no field specified"))?;
    let so = spec.as_object().ok_or_else(|| EsError::parsing("[geo_grid] query malformed, expected an object with one of [geohash], [geotile] or [geohex]"))?;
    let mut cell: Option<(&str, &str)> = None;
    for (k, v) in so {
        if cell.is_some() {
            return Err(EsError::parsing(format!("failed to parse [{q}] query. unexpected field [{k}]")));
        }
        if !matches!(k.as_str(), "geohash" | "geotile" | "geohex") {
            return Err(parse_exc(format!("failed to parse [{q}] query. Invalid grid name [{k}]")));
        }
        cell = Some((k.as_str(), v.as_str().unwrap_or("")));
    }
    let (kind, text) = cell.ok_or_else(|| EsError::parsing("[geo_grid] query malformed, expected one of [geohash], [geotile] or [geohex]"))?;
    let (top, left, bottom, right) = match kind {
        "geohash" => geohash_cell(text)?,
        "geotile" => {
            let bad = || EsError::illegal_argument(format!("Invalid geotile_grid hash string of {text}. Must be three integers in a form \"zoom/x/y\"."));
            let parts: Vec<&str> = text.split('/').collect();
            if parts.len() != 3 {
                return Err(bad());
            }
            let p: Vec<u32> = parts.iter().map(|x| x.parse().map_err(|_| bad())).collect::<Result<_>>()?;
            if p[0] > 29 || p[1] as u64 >= (1u64 << p[0]) || p[2] as u64 >= (1u64 << p[0]) {
                return Err(shard(ctx, format!("failed to create query: Zoom/X/Y combination is not valid: {text}")));
            }
            crate::fieldtypes::geo::tile_bbox(p[0], p[1], p[2])
        }
        _ => return geohex_query(field, text, ignore_unmapped, o, ctx),
    };
    let f = match ctx.lookup.get(field) {
        None if ignore_unmapped => return Ok(Query::MatchNone),
        None => return Err(shard(ctx, format!("failed to find geo field [{field}]"))),
        Some(f) => f.clone(),
    };
    let query = match f.declared {
        EsType::GeoPoint => match f.geo_id {
            Some(g) => Query::ConstScore { query: Box::new(Query::GeoBoundingBox { field: g, top, left, bottom, right }), score: 1.0, point_set: false },
            None => return Ok(Query::MatchNone),
        },
        EsType::GeoShape => parse_query(&json!({"geo_shape": {field.clone(): {"shape": {"type": "envelope", "coordinates": [[left, top], [right, bottom]]}, "relation": "intersects"}, "boost": 1.0}}), ctx)?,
        _ => return Err(shard(ctx, format!("Field [{}] is of unsupported type [{}] for [{q}] query", f.name, f.declared.name()))),
    };
    Ok(boost_q(query, o))
}

/// `geo_grid` with a `geohex` (H3) cell: `geo_point` values whose cell at
/// the address's resolution is that cell (ES `H3LatLonGeometry`), shapes
/// intersecting the cell polygon.
fn geohex_query(field: &str, text: &str, ignore_unmapped: bool, o: &Map<String, Value>, ctx: &QueryContext) -> Result<Query> {
    use crate::aggs::geo_hex;
    if u64::from_str_radix(text, 16).is_err() {
        return Err(EsError::illegal_argument(format!("Invalid h3 address [{text}]"))
            .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{text}\" under radix 16"))));
    }
    let f = match ctx.lookup.get(field) {
        None if ignore_unmapped => return Ok(Query::MatchNone),
        None => return Err(shard(ctx, format!("failed to find geo field [{field}]"))),
        Some(f) => f.clone(),
    };
    let Some(cell) = geo_hex::parse_cell(text) else {
        return Ok(Query::MatchNone);
    };
    let query = match f.declared {
        EsType::GeoPoint => {
            let Some(g) = f.geo_id else { return Ok(Query::MatchNone) };
            let q = Query::Custom(std::sync::Arc::new(GeoHexPointQuery { field: g, pred: std::sync::Arc::new(geo_hex::HexCellPred::new(cell)) }));
            Query::ConstScore { query: Box::new(q), score: 1.0, point_set: false }
        }
        EsType::GeoShape => {
            let polys: Vec<Value> = geo_hex::cartesian_polys(cell).into_iter().map(|r| json!([r.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>()])).collect();
            let q = parse_query(&json!({"geo_shape": {field: {"shape": {"type": "multipolygon", "coordinates": polys}, "relation": "intersects"}, "boost": 1.0}}), ctx)?;
            Query::ConstScore { query: Box::new(q), score: 1.0, point_set: false }
        }
        _ => return Err(shard(ctx, format!("Field [{}] is of unsupported type [{}] for [geo_grid] query", f.name, f.declared.name()))),
    };
    Ok(boost_q(query, o))
}

/// The `geo_grid` geohex query on a `geo_point` field: docs with a point
/// in the cell ([`crate::aggs::geo_hex::HexCellPred`]). Sealed segments walk the KD tree
/// (leaves inside the cell taken whole, boundary leaves through a cell
/// grid, or the doc-order scan when a size-10 search stops early); the
/// memtable tests its points.
#[derive(Debug)]
struct GeoHexPointQuery {
    field: u16,
    pred: std::sync::Arc<crate::aggs::geo_hex::HexCellPred>,
}

impl rustic_core::query::CustomQuery for GeoHexPointQuery {
    fn geo_pred(&self) -> Option<(u16, geo::GeoPred)> {
        let p: std::sync::Arc<dyn geo::PointPredicate> = match self.pred.plain_bbox() {
            Some(b) => std::sync::Arc::new(geo::GridPred::keyed(HexRef(self.pred.clone()), b, format!("geohex {:?}", self.pred))),
            None => std::sync::Arc::new(HexRef(self.pred.clone())),
        };
        Some((self.field, geo::GeoPred::Leaf(p)))
    }

    fn scorer(&self, leaf: &rustic_core::query::LeafReader<'_>, _ctx: &rustic_core::query::SearchContext) -> Option<Box<dyn rustic_core::query::Scorer>> {
        use rustic_core::query::LeafReader;
        let (_, pred) = rustic_core::query::CustomQuery::geo_pred(self)?;
        match leaf {
            LeafReader::Segment(seg) => {
                let g = seg.geo_index(self.field)?;
                rustic_core::query::geo_pred_scorer(g, pred, true, seg.num_docs, 1.0)
            }
            LeafReader::Memtable(..) => {
                let mut docs: Vec<rustic_core::types::DocId> = Vec::new();
                for p in rustic_core::query::GeoDv::open(leaf, self.field).iter() {
                    let d = { p.doc };
                    if docs.last() != Some(&d) && pred.test(geo::decode_lat({ p.lat }), geo::decode_lon({ p.lon })) {
                        docs.push(d);
                    }
                }
                if docs.is_empty() {
                    return None;
                }
                Some(rustic_core::query::hits_scorer(geo::DocHits::Sorted(docs), 1.0))
            }
        }
    }

    fn is_constant_score(&self) -> bool {
        true
    }

    fn scores_uniformly(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "geo_grid"
    }
}

/// A shared [`crate::aggs::geo_hex::HexCellPred`] as a point predicate.
#[derive(Debug)]
struct HexRef(std::sync::Arc<crate::aggs::geo_hex::HexCellPred>);

impl geo::PointPredicate for HexRef {
    fn classify(&self, a: f64, b: f64, c: f64, d: f64) -> geo::Relation {
        geo::PointPredicate::classify(&*self.0, a, b, c, d)
    }
    fn test(&self, lat: f64, lon: f64) -> bool {
        geo::PointPredicate::test(&*self.0, lat, lon)
    }
}
