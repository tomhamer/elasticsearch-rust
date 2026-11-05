//! More ES|QL scalar functions: spatial introspection and grid cells,
//! hashes, URL coding, extra math and multi-value set functions, vector
//! similarity and `TOP_SNIPPETS`.

use super::func::FnCtx;
use super::{cmp_vals, Val};
use crate::error::{EsError, Result};
use rustic_core::fieldtypes::shape::{self, Geometry};

const OPTS: shape::ParseOpts = shape::ParseOpts { geo: false, coerce: true, ignore_z: true };

/// The geometry of a spatial value; a multi-value is a collection.
fn geometry(v: &Val) -> Option<Geometry> {
    match v {
        Val::Str(s) => shape::parse_wkt(s.trim(), &OPTS).ok(),
        Val::Multi(m) => Some(Geometry::Collection(m.iter().filter_map(geometry).collect())),
        _ => None,
    }
}

fn geometry_type(g: &Geometry) -> &'static str {
    match g {
        Geometry::Point(_) => "ST_Point",
        Geometry::MultiPoint(_) => "ST_MultiPoint",
        Geometry::LineString(_) => "ST_LineString",
        Geometry::MultiLineString(_) => "ST_MultiLineString",
        Geometry::Polygon(_) | Geometry::Envelope { .. } => "ST_Polygon",
        Geometry::MultiPolygon(_) => "ST_MultiPolygon",
        Geometry::Collection(_) => "ST_GeometryCollection",
    }
}

fn dimension(g: &Geometry) -> i64 {
    match g {
        Geometry::Point(_) | Geometry::MultiPoint(_) => 0,
        Geometry::LineString(_) | Geometry::MultiLineString(_) => 1,
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) | Geometry::Envelope { .. } => 2,
        Geometry::Collection(c) => c.iter().map(dimension).max().unwrap_or(0),
    }
}

fn points(g: &Geometry, out: &mut Vec<(f64, f64)>) {
    match g {
        Geometry::Point(p) => out.push(*p),
        Geometry::MultiPoint(v) | Geometry::LineString(v) => out.extend(v.iter().copied()),
        Geometry::MultiLineString(v) | Geometry::Polygon(v) => v.iter().for_each(|r| out.extend(r.iter().copied())),
        Geometry::MultiPolygon(v) => v.iter().flatten().for_each(|r| out.extend(r.iter().copied())),
        Geometry::Collection(c) => c.iter().for_each(|x| points(x, out)),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => out.extend([(*min_x, *max_y), (*max_x, *min_y)]),
    }
}

fn is_empty(g: &Geometry) -> bool {
    let mut p = Vec::new();
    points(g, &mut p);
    p.is_empty()
}

/// `(min_x, max_x, min_y, max_y)`.
fn bounds(g: &Geometry) -> Option<(f64, f64, f64, f64)> {
    let mut p = Vec::new();
    points(g, &mut p);
    if p.is_empty() {
        return None;
    }
    Some(p.iter().fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |(a, b, c, d), (x, y)| (a.min(*x), b.max(*x), c.min(*y), d.max(*y))))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(alg: &str, data: &[u8]) -> Option<String> {
    use sha2::Digest;
    Some(match alg.to_ascii_uppercase().as_str() {
        "MD5" => hex(&crate::ingest::util::md5(data)),
        "SHA1" | "SHA-1" => hex(&crate::ingest::util::sha1(data)),
        "SHA-224" | "SHA224" => hex(&sha2::Sha224::digest(data)),
        "SHA-256" | "SHA256" => hex(&sha2::Sha256::digest(data)),
        "SHA-384" | "SHA384" => hex(&sha2::Sha384::digest(data)),
        "SHA-512" | "SHA512" => hex(&sha2::Sha512::digest(data)),
        _ => return None,
    })
}

/// `URLEncoder.encode(s, UTF-8)` (`form`) or RFC 3986 component encoding.
fn url_encode(s: &str, form: bool) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        let keep = b.is_ascii_alphanumeric() || if form { matches!(b, b'.' | b'-' | b'*' | b'_') } else { matches!(b, b'.' | b'-' | b'_' | b'~') };
        if keep {
            out.push(b as char);
        } else if form && b == b' ' {
            out.push('+');
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn url_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' => {
                let h = std::str::from_utf8(b.get(i + 1..i + 3)?).ok()?;
                out.push(u8::from_str_radix(h, 16).ok()?);
                i += 2;
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8(out).ok()
}

fn nums(v: &Val) -> Vec<f64> {
    v.values().into_iter().filter_map(|x| x.as_f64()).collect()
}

fn f(v: &Val) -> Option<f64> {
    match v {
        Val::Multi(_) => None,
        o => o.as_f64(),
    }
}

fn s(v: &Val) -> Option<String> {
    match v {
        Val::Str(s) => Some(s.clone()),
        Val::Null | Val::Multi(_) => None,
        o => Some(o.to_text()),
    }
}

fn mv(v: &Val) -> Vec<Val> {
    match v {
        Val::Null => vec![],
        Val::Multi(m) => m.clone(),
        o => vec![o.clone()],
    }
}

fn dedup(mut v: Vec<Val>) -> Vec<Val> {
    let mut seen = std::collections::HashSet::new();
    v.retain(|x| seen.insert(x.key()));
    v
}

/// `TOP_SNIPPETS`: the passages of a text that best match a query.
fn top_snippets(text: &Val, query: &str, opts: &[(String, Val)]) -> Val {
    let opt = |k: &str| opts.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    let num_snippets = opt("num_snippets").and_then(|v| v.as_i64()).unwrap_or(5).max(1) as usize;
    let num_words = opt("num_words").and_then(|v| v.as_i64()).unwrap_or(300).max(0) as usize;
    let highlight = matches!(opt("highlight"), Some(Val::Bool(true)));
    let pre = opt("pre_tag").map(|v| v.to_text()).unwrap_or_else(|| "<em>".into());
    let post = opt("post_tag").map(|v| v.to_text()).unwrap_or_else(|| "</em>".into());
    let doc_order = opt("order").map(|v| v.to_text()) == Some("none".into());
    let norm = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    let terms: Vec<String> = query.split_whitespace().map(norm).filter(|t| !t.is_empty()).collect();
    if terms.is_empty() {
        return Val::Null;
    }
    // Chunks: `SentenceBoundaryChunkingSettings(num_words, 0)`; 0 = as-is.
    let mut chunks: Vec<String> = Vec::new();
    let chunking = crate::inference::local::Chunking { strategy: "sentence".into(), max_chunk_size: num_words, overlap: 0, sentence_overlap: 0 };
    for value in text.values() {
        let t = value.to_text();
        if num_words == 0 {
            chunks.push(t);
            continue;
        }
        chunks.extend(crate::inference::local::chunk(&t, &chunking).into_iter().map(|(a, b)| t[a..b].to_string()));
    }
    let mut scored: Vec<(usize, usize, String)> = chunks
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let score = c.split_whitespace().filter(|w| terms.contains(&norm(w))).count();
            (score > 0).then_some((score, i, c))
        })
        .collect();
    if !doc_order {
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    }
    scored.truncate(num_snippets);
    let out: Vec<Val> = scored
        .into_iter()
        .map(|(_, _, c)| {
            if !highlight {
                return Val::Str(c);
            }
            let words: Vec<String> = c
                .split_whitespace()
                .map(|w| {
                    let core = w.trim_matches(|ch: char| !ch.is_alphanumeric());
                    if !core.is_empty() && terms.contains(&core.to_lowercase()) {
                        w.replacen(core, &format!("{pre}{core}{post}"), 1)
                    } else {
                        w.to_string()
                    }
                })
                .collect();
            Val::Str(words.join(" "))
        })
        .collect();
    Val::from_list(out)
}

/// Internal: a grid cell address as its long (bound in place of
/// `ST_GEO*(..)::long`; never written in a query).
pub const GRID_LONG: &str = "\u{1}GRID_LONG";

/// The long ES encodes a cell address as: geohash `(chars << 4) | length`
/// (`Geohash.longEncode(String)`), geotile `zoom << 58 | x << 29 | y`
/// (`GeoTileUtils.longEncodeTiles`), geohex the H3 index.
pub fn grid_address_long(kind: &str, addr: &str) -> Option<i64> {
    match kind {
        "geohash" => {
            const BASE32: &str = "0123456789bcdefghjkmnpqrstuvwxyz";
            let len = addr.len();
            let mut level = len as i64 - 1;
            let mut l = 0i64;
            for c in addr.chars() {
                let b = BASE32.find(c)? as i64;
                l |= b << (level * 5);
                level -= 1;
                if level < 0 { break; }
            }
            Some((l << 4) | len as i64)
        }
        "geotile" => {
            let mut parts = addr.split('/').map(|x| x.parse::<i64>().ok());
            let (Some(Some(z)), Some(Some(x)), Some(Some(y)), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else { return None };
            Some((z << 58) | (x << 29) | y)
        }
        "geohex" => u64::from_str_radix(addr, 16).ok().map(|v| v as i64),
        _ => None,
    }
}

const GEOHASH_BASE32: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// `Geohash.longEncode(String)` with Java's arithmetic: no validation, an
/// unknown character contributes `indexOf == -1` (all bits set), at most
/// 12 characters are encoded and the low 4 bits hold the UTF-16 length.
fn geohash_long_encode_java(s: &str) -> i64 {
    let units: Vec<u16> = s.encode_utf16().collect();
    let length = units.len() as i64;
    let mut level = length - 1;
    let mut l = 0i64;
    for u in units {
        let b = GEOHASH_BASE32.iter().position(|&c| c as u16 == u).map(|i| i as i64).unwrap_or(-1);
        l |= b.wrapping_shl((level * 5) as u32);
        level -= 1;
        if level < 0 {
            break;
        }
    }
    l.wrapping_shl(4) | length
}

/// `Geohash.stringEncode(long)`: `None` where Java fails (level 0).
fn geohash_string_encode(v: i64) -> Option<String> {
    let mut level = (v & 15) as usize;
    if level == 0 {
        return None;
    }
    let mut h = (v as u64) >> 4;
    let mut chars = vec![b'0'; level];
    while level > 0 {
        level -= 1;
        chars[level] = GEOHASH_BASE32[(h & 31) as usize];
        h >>= 5;
    }
    String::from_utf8(chars).ok()
}

/// `GeoTileUtils.stringEncode(long)`: `None` where `validateZXY` throws.
fn geotile_string_encode(v: i64) -> Option<String> {
    const MASK: i64 = (1 << 29) - 1;
    let z = ((v as u64) >> 58) as i64;
    let (x, y) = ((v >> 29) & MASK, v & MASK);
    if !(0..=29).contains(&z) || x >= (1i64 << z) || y >= (1i64 << z) {
        return None;
    }
    Some(format!("{z}/{x}/{y}"))
}

/// The address ES prints for a grid cell long (`geoGridToString`).
pub fn grid_long_address(kind: &str, v: i64) -> Option<String> {
    match kind {
        "geohash" => geohash_string_encode(v),
        "geotile" => geotile_string_encode(v),
        "geohex" => Some(format!("{:x}", v as u64)),
        _ => None,
    }
}

/// `TO_GEOHASH` / `TO_GEOTILE` / `TO_GEOHEX` of a string (ES 9.5.5
/// `ToGeohash|ToGeotile|ToGeohex.fromString`, then printed by
/// `geoGridToString`): the canonical address, `Ok(None)` when ES's long
/// cannot be printed, `Err` with the warning ES records
/// (`warnExceptions = IllegalArgumentException`) when it does not parse.
pub fn grid_from_string(kind: &str, s: &str) -> std::result::Result<Option<String>, String> {
    match kind {
        "geohash" => Ok(geohash_string_encode(geohash_long_encode_java(s))),
        "geotile" => {
            // `GeoTileUtils.parseHash`: `split("/", 4)`, three `Integer.parseInt`s.
            let parts: Vec<&str> = s.splitn(4, '/').collect();
            let ints: Option<Vec<i32>> = if parts.len() == 3 { parts.iter().map(|p| p.parse::<i32>().ok()).collect() } else { None };
            let Some(v) = ints else {
                return Err(format!("java.lang.IllegalArgumentException: Invalid geotile_grid hash string of {s}. Must be three integers in a form \"zoom/x/y\"."));
            };
            // `longEncodeTiles`: `(long) z << 58 | (long) x << 29 | y`.
            let long = ((v[0] as i64).wrapping_shl(58)) | ((v[1] as i64).wrapping_shl(29)) | v[2] as i64;
            Ok(geotile_string_encode(long))
        }
        "geohex" => {
            // `H3.stringToH3`: `Long.parseUnsignedLong(s, 16)`.
            if s.starts_with('-') {
                return Err(format!("java.lang.NumberFormatException: Illegal leading minus sign on unsigned string {s}."));
            }
            let digits = s.strip_prefix('+').unwrap_or(s);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("java.lang.NumberFormatException: For input string: \"{s}\" under radix 16"));
            }
            match u64::from_str_radix(digits, 16) {
                Ok(v) => Ok(Some(format!("{v:x}"))),
                Err(_) => Err(format!("java.lang.NumberFormatException: String value {s} exceeds range of unsigned long.")),
            }
        }
        _ => Ok(None),
    }
}

/// `GeometryPointCountVisitor` (`ST_NPOINTS`): every vertex, a `BBOX` 4.
fn npoints(g: &Geometry) -> i64 {
    match g {
        Geometry::Envelope { .. } => 4,
        Geometry::Collection(c) => c.iter().map(npoints).sum(),
        other => {
            let mut p = Vec::new();
            points(other, &mut p);
            p.len() as i64
        }
    }
}

/// The functions evaluated here.
pub const NAMES: &[&str] = &[
    "ST_GEOMETRYTYPE", "ST_DIMENSION", "ST_ISEMPTY", "ST_NPOINTS", "ST_ENVELOPE", "ST_XMIN", "ST_XMAX", "ST_YMIN", "ST_YMAX", "ST_GEOHASH", "ST_GEOTILE", "ST_GEOHEX", "ST_SIMPLIFY",
    "TO_GEOHASH", "TO_GEOTILE", "TO_GEOHEX", "MD5", "SHA1", "SHA256", "HASH", "URL_ENCODE", "URL_ENCODE_COMPONENT", "URL_DECODE", "ACOSH", "ASINH", "ATANH", "SCALB",
    "CLAMP", "CLAMP_MIN", "CLAMP_MAX", "MV_UNION", "MV_INTERSECTION", "MV_DIFFERENCE", "MV_INTERSECTS", "MV_PSERIES_WEIGHTED_SUM",
    "V_COSINE", "V_DOT_PRODUCT", "V_L1_NORM", "V_L2_NORM", "V_HAMMING", "TO_DENSE_VECTOR", "TOP_SNIPPETS", "JSON_EXTRACT", "FIELD_EXTRACT",
];

/// `ST_SIMPLIFY(a0, a1)`: Douglas-Peucker with a constant, non-negative
/// tolerance (checked when binding); a multi-value is one collection.
/// Pure (no context, no warnings), so rows may be simplified in parallel.
pub fn st_simplify(a0: &Val, a1: &Val) -> Val {
    let Some(tol) = f(a1).filter(|t| *t >= 0.0) else { return Val::Null };
    let out = match a0 {
        Val::Str(w) => super::spatial::simplify_wkt(w, tol),
        Val::Multi(m) => {
            let wkts: Vec<String> = m.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
            super::spatial::simplify_wkts(&wkts, tol)
        }
        _ => None,
    };
    out.map(Val::Str).unwrap_or(Val::Null)
}

/// Evaluate one of these functions (`None` when the name is not one).
pub fn call(name: &str, args: &[Val], _ctx: &FnCtx) -> Option<Result<Val>> {
    let a0 = args.first().cloned().unwrap_or(Val::Null);
    let a1 = args.get(1).cloned().unwrap_or(Val::Null);
    let r: Val = match name {
        "ST_GEOMETRYTYPE" => geometry(&a0).map(|g| Val::Str(geometry_type(&g).into())).unwrap_or(Val::Null),
        "ST_DIMENSION" => geometry(&a0).map(|g| Val::Int(dimension(&g))).unwrap_or(Val::Null),
        "ST_ISEMPTY" => geometry(&a0).map(|g| Val::Bool(is_empty(&g))).unwrap_or(Val::Null),
        "ST_NPOINTS" => geometry(&a0).map(|g| Val::Int(npoints(&g))).unwrap_or(Val::Null),
        // `StSimplify`: Douglas-Peucker with a constant, non-negative
        // tolerance (checked when binding); a multi-value is one collection.
        "ST_SIMPLIFY" => st_simplify(&a0, &a1),
        "ST_ENVELOPE" => geometry(&a0)
            .and_then(|g| bounds(&g))
            .map(|(a, b, c, d)| Val::Str(super::spatial::to_wkt(&Geometry::Envelope { min_x: a, max_x: b, max_y: d, min_y: c })))
            .unwrap_or(Val::Null),
        "ST_XMIN" | "ST_XMAX" | "ST_YMIN" | "ST_YMAX" => geometry(&a0)
            .and_then(|g| bounds(&g))
            .map(|(a, b, c, d)| Val::Double(match name {
                "ST_XMIN" => a,
                "ST_XMAX" => b,
                "ST_YMIN" => c,
                _ => d,
            }))
            .unwrap_or(Val::Null),
        "ST_GEOHASH" | "ST_GEOTILE" | "ST_GEOHEX" => {
            let Some(p) = a1.as_i64() else { return Some(Ok(Val::Null)) };
            let cell = |point: &Val| -> Val {
                let Some((x, y)) = s(point).and_then(|w| super::spatial::point_xy(&w)) else { return Val::Null };
                match name {
                    "ST_GEOHASH" => Val::Str(rustic_core::geo::geohash(y, x, p.clamp(1, 12) as usize)),
                    "ST_GEOTILE" => {
                        let (tx, ty) = rustic_core::geo::geotile(y, x, p.clamp(0, 29) as u32);
                        Val::Str(format!("{p}/{tx}/{ty}"))
                    }
                    _ => crate::aggs::geo_hex::point_cell(y, x, p.clamp(0, 15) as u32).map(Val::Str).unwrap_or(Val::Null),
                }
            };
            match &a0 {
                // One cell per point, in value order (`SpatialGridFunction.fromWKB`).
                Val::Multi(points) => Val::from_list(points.iter().map(cell).collect()),
                point => cell(point),
            }
        }
        // `geohash|geotile|geohex::long` (ES `ToLong` passes the cell's long
        // through): the address (`a0`) of a cell of grid type `a1` as its long.
        GRID_LONG => {
            let kind = s(&a1).unwrap_or_default();
            let one = |v: &Val| -> Val { match v { Val::Str(addr) => grid_address_long(&kind, addr).map(Val::Long).unwrap_or(Val::Null), _ => Val::Null } };
            match &a0 {
                Val::Multi(cells) => Val::from_list(cells.iter().map(one).collect()),
                cell => one(cell),
            }
        }
        // ES 9.5.5 `ToGeohash|ToGeotile|ToGeohex`: a cell (held as its
        // address) passes through; a long is the cell's long (`fieldEval`),
        // printed as its address; a string goes through `fromString`, so it
        // prints canonically (`TO_GEOHEX("811FBFFFFFFFFFF")` is
        // `811fbffffffffff`) and a bad one is null with ES's warning.
        "TO_GEOHASH" | "TO_GEOTILE" | "TO_GEOHEX" => {
            let kind = match name {
                "TO_GEOHASH" => "geohash",
                "TO_GEOTILE" => "geotile",
                _ => "geohex",
            };
            let one = |v: &Val| -> std::result::Result<Option<String>, String> {
                match v {
                    Val::Str(s) => grid_from_string(kind, s),
                    Val::Long(l) | Val::Int(l) => Ok(grid_long_address(kind, *l)),
                    _ => Ok(None),
                }
            };
            match &a0 {
                Val::Multi(m) => {
                    // Bad values are dropped (`AbstractConvertFunction`); only
                    // when none converts is the failure reported.
                    let mut out = Vec::with_capacity(m.len());
                    let mut first_err = None;
                    for v in m {
                        match one(v) {
                            Ok(Some(a)) => out.push(Val::Str(a)),
                            Ok(None) => {}
                            Err(e) => first_err = first_err.or(Some(e)),
                        }
                    }
                    match (out.is_empty(), first_err) {
                        (true, Some(e)) => return Some(Err(super::func::eval_warning(e))),
                        _ => Val::from_list(out),
                    }
                }
                v => match one(v) {
                    Ok(a) => a.map(Val::Str).unwrap_or(Val::Null),
                    Err(e) => return Some(Err(super::func::eval_warning(e))),
                },
            }
        }
        "MD5" | "SHA1" | "SHA256" => s(&a0).and_then(|x| digest(if name == "SHA256" { "SHA-256" } else { name }, x.as_bytes())).map(Val::Str).unwrap_or(Val::Null),
        "HASH" => match (s(&a0), s(&a1)) {
            (Some(alg), Some(x)) => match digest(&alg, x.as_bytes()) {
                Some(h) => Val::Str(h),
                None => return Some(Err(super::func::eval_warning(format!("java.security.NoSuchAlgorithmException: {alg} MessageDigest not available")))),
            },
            _ => Val::Null,
        },
        "URL_ENCODE" => s(&a0).map(|x| Val::Str(url_encode(&x, true))).unwrap_or(Val::Null),
        "URL_ENCODE_COMPONENT" => s(&a0).map(|x| Val::Str(url_encode(&x, false))).unwrap_or(Val::Null),
        "URL_DECODE" => match s(&a0) {
            Some(x) => match url_decode(&x) {
                Some(d) => Val::Str(d),
                None => return Some(Err(super::func::eval_warning(format!("java.lang.IllegalArgumentException: URLDecoder: Illegal hex characters in escape (%) pattern")))),
            },
            None => Val::Null,
        },
        "ACOSH" => f(&a0).map(|x| super::func::dbl(x.acosh())).unwrap_or(Val::Null),
        "ASINH" => f(&a0).map(|x| super::func::dbl(x.asinh())).unwrap_or(Val::Null),
        "ATANH" => f(&a0).map(|x| super::func::dbl(x.atanh())).unwrap_or(Val::Null),
        "SCALB" => match (f(&a0), a1.as_i64()) {
            (Some(d), Some(n)) => Val::Double(d * 2f64.powi(n as i32)),
            _ => Val::Null,
        },
        "CLAMP" | "CLAMP_MIN" | "CLAMP_MAX" => {
            if a0.is_null() {
                return Some(Ok(Val::Null));
            }
            let (lo, hi) = match name {
                "CLAMP" => (args.get(1).cloned(), args.get(2).cloned()),
                "CLAMP_MIN" => (args.get(1).cloned(), None),
                _ => (None, args.get(1).cloned()),
            };
            let mut v = a0.clone();
            if let Some(l) = lo.filter(|l| !l.is_null()) {
                if cmp_vals(&v, &l) == std::cmp::Ordering::Less {
                    v = l;
                }
            }
            if let Some(h) = hi.filter(|h| !h.is_null()) {
                if cmp_vals(&v, &h) == std::cmp::Ordering::Greater {
                    v = h;
                }
            }
            v
        }
        "MV_UNION" => Val::from_list(dedup(mv(&a0).into_iter().chain(mv(&a1)).collect())),
        "MV_INTERSECTION" => {
            let b: Vec<String> = mv(&a1).iter().map(Val::key).collect();
            Val::from_list(dedup(mv(&a0).into_iter().filter(|x| b.contains(&x.key())).collect()))
        }
        "MV_DIFFERENCE" => {
            let b: Vec<String> = mv(&a1).iter().map(Val::key).collect();
            Val::from_list(dedup(mv(&a0).into_iter().filter(|x| !b.contains(&x.key())).collect()))
        }
        "MV_INTERSECTS" => {
            if a0.is_null() || a1.is_null() {
                return Some(Ok(Val::Null));
            }
            let b: Vec<String> = mv(&a1).iter().map(Val::key).collect();
            Val::Bool(mv(&a0).iter().any(|x| b.contains(&x.key())))
        }
        "MV_PSERIES_WEIGHTED_SUM" => {
            let mut v = nums(&a0);
            if v.is_empty() {
                return Some(Ok(Val::Null));
            }
            v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let p = f(&a1).unwrap_or(1.0);
            Val::Double(v.iter().enumerate().map(|(i, x)| x / ((i + 1) as f64).powf(p)).sum())
        }
        "V_COSINE" | "V_DOT_PRODUCT" | "V_L1_NORM" | "V_L2_NORM" | "V_HAMMING" => {
            let (x, y) = (nums(&a0), nums(&a1));
            if x.is_empty() || y.is_empty() {
                return Some(Ok(Val::Null));
            }
            if x.len() != y.len() {
                return Some(Err(EsError::illegal_argument(format!("vector dimensions differ: {}!={}", x.len(), y.len()))));
            }
            // Dense vectors are float vectors: compute in f32 like ES.
            let (x, y): (Vec<f32>, Vec<f32>) = (x.iter().map(|v| *v as f32).collect(), y.iter().map(|v| *v as f32).collect());
            let dot: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            Val::Double(match name {
                "V_DOT_PRODUCT" => dot,
                "V_COSINE" => {
                    let (nx, ny) = (x.iter().map(|a| a * a).sum::<f32>(), y.iter().map(|a| a * a).sum::<f32>());
                    if nx == 0.0 || ny == 0.0 { return Some(Ok(Val::Null)) } else { dot / (nx * ny).sqrt() }
                }
                "V_L1_NORM" => x.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum(),
                "V_L2_NORM" => x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt(),
                _ => x.iter().zip(&y).map(|(a, b)| ((*a as i64 as u8) ^ (*b as i64 as u8)).count_ones() as f32).sum(),
            } as f64)
        }
        "TO_DENSE_VECTOR" => {
            let v: Vec<Val> = mv(&a0).into_iter().filter_map(|x| x.as_f64()).map(Val::Double).collect();
            Val::from_list(v)
        }
        "JSON_EXTRACT" | "FIELD_EXTRACT" => match (&a0, &a1) {
            (Val::Null, _) | (_, Val::Null) => Val::Null,
            (Val::Multi(_), _) | (_, Val::Multi(_)) => return Some(Err(iae("single-value function encountered multi-value"))),
            (j, p) => {
                let (j, p) = (j.to_text(), p.to_text());
                return Some(if name == "JSON_EXTRACT" { json_extract(&j, &p) } else { field_extract(&j, &p) });
            }
        },
        "TOP_SNIPPETS" => {
            let Some(q) = s(&a1) else { return Some(Ok(Val::Null)) };
            if a0.is_null() {
                return Some(Ok(Val::Null));
            }
            let opts: Vec<(String, Val)> = match args.get(2) {
                Some(Val::Str(json)) => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(json).map(|m| m.iter().map(|(k, v)| (k.clone(), super::parse::json_to_val(v))).collect()).unwrap_or_default(),
                _ => vec![],
            };
            top_snippets(&a0, &q, &opts)
        }
        _ => return None,
    };
    Some(Ok(r))
}

// ---------------------------------------------------------------- JSON_EXTRACT / FIELD_EXTRACT

/// A `JsonPath` segment.
#[derive(Debug, Clone, PartialEq)]
pub enum Seg {
    Key(String),
    Index(usize),
}

fn invalid_path(path: &str, reason: &str, pos: Option<usize>) -> String {
    match pos {
        Some(p) => format!("Invalid JSON path [{path}]: {reason} at position {p}"),
        None => format!("Invalid JSON path [{path}]: {reason}"),
    }
}

/// `JsonPath.parse`: the segments of a path (empty = the root), or the
/// `IllegalArgumentException` message.
pub fn parse_json_path(orig: &str) -> std::result::Result<Vec<Seg>, String> {
    let c: Vec<char> = orig.chars().collect();
    if c.is_empty() || (c.len() == 1 && c[0] == '$') {
        return Ok(vec![]);
    }
    let mut start = 0usize;
    if c.len() >= 2 && c[0] == '$' {
        match c[1] {
            '.' => start = 2,
            '[' => start = 1,
            _ => return Err(invalid_path(orig, "expected [.] or [[] after [$]", Some(1))),
        }
    }
    let p: Vec<char> = c[start..].to_vec();
    if p.is_empty() {
        return Ok(vec![]);
    }
    let base = start;
    let blank = |ch: char| matches!(ch, ' ' | '\t' | '\n' | '\r');
    let mut segs = Vec::new();
    let mut pos = 0usize;
    let mut has = false;
    let read_key = |pos: usize, segs: &mut Vec<Seg>| -> std::result::Result<usize, String> {
        let mut end = pos;
        while end < p.len() && !matches!(p[end], '.' | '[' | ']') {
            end += 1;
        }
        if end == pos {
            return Err(invalid_path(orig, "empty key name", Some(base + pos)));
        }
        segs.push(Seg::Key(p[pos..end].iter().collect()));
        Ok(end)
    };
    while pos < p.len() {
        let ch = p[pos];
        if ch == '.' {
            if !has {
                return Err(invalid_path(orig, "path cannot start with a dot", Some(base + pos)));
            }
            pos += 1;
            if pos >= p.len() {
                return Err(invalid_path(orig, "path cannot end with a dot", Some(base + pos - 1)));
            }
            if p[pos] == '.' {
                return Err(invalid_path(orig, "consecutive dots", Some(base + pos - 1)));
            }
            pos = read_key(pos, &mut segs)?;
            has = true;
        } else if ch == '[' {
            let bo = base + pos;
            let mut q = pos + 1;
            while q < p.len() && blank(p[q]) {
                q += 1;
            }
            if q >= p.len() {
                return Err(invalid_path(orig, "unterminated bracket", Some(bo)));
            }
            match p[q] {
                '\'' | '"' => {
                    let quote = p[q];
                    q += 1;
                    let mut key = String::new();
                    let mut done = None;
                    while q < p.len() {
                        if p[q] == '\\' && q + 1 < p.len() {
                            key.push(p[q + 1]);
                            q += 2;
                        } else if p[q] == quote {
                            let mut a = q + 1;
                            while a < p.len() && blank(p[a]) {
                                a += 1;
                            }
                            if a >= p.len() || p[a] != ']' {
                                return Err(invalid_path(orig, "expected closing bracket after quoted key", Some(bo)));
                            }
                            done = Some(a + 1);
                            break;
                        } else {
                            key.push(p[q]);
                            q += 1;
                        }
                    }
                    let Some(next) = done else { return Err(invalid_path(orig, "unterminated quoted key", Some(bo))) };
                    segs.push(Seg::Key(key));
                    pos = next;
                }
                ']' => return Err(invalid_path(orig, "empty brackets", Some(bo))),
                _ => {
                    let Some(end) = p[q..].iter().position(|&x| x == ']').map(|e| e + q) else {
                        return Err(invalid_path(orig, "missing closing bracket for array index", Some(bo)));
                    };
                    let mut ce = end;
                    while ce > q && blank(p[ce - 1]) {
                        ce -= 1;
                    }
                    let content: String = p[q..ce].iter().collect();
                    if content.is_empty() {
                        return Err(invalid_path(orig, "empty array index", Some(bo)));
                    }
                    let Ok(i) = content.parse::<i32>() else {
                        return Err(invalid_path(orig, &format!("expected integer array index, got [{content}]"), Some(bo)));
                    };
                    if content.len() > 1 && content.starts_with('0') {
                        return Err(invalid_path(orig, &format!("leading zeros are not allowed in array index [{content}]"), Some(bo)));
                    }
                    if i < 0 {
                        return Err("array index out of bounds".into());
                    }
                    segs.push(Seg::Index(i as usize));
                    pos = end + 1;
                }
            }
            has = true;
        } else {
            if has {
                return Err(invalid_path(orig, &format!("expected [.] or [[] before [{ch}]"), Some(base + pos)));
            }
            pos = read_key(pos, &mut segs)?;
            has = true;
        }
    }
    if segs.is_empty() {
        return Err(invalid_path(orig, "empty path", None));
    }
    Ok(segs)
}

/// Minimal streaming JSON reader over bytes: positions of values.
struct Js<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Js<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.i).copied()
    }
    /// A string starting at the current `"`: its decoded text.
    fn string(&mut self) -> Option<String> {
        let start = self.i + 1;
        let mut j = start;
        while j < self.b.len() {
            match self.b[j] {
                b'\\' => j += 2,
                b'"' => {
                    let raw = std::str::from_utf8(&self.b[self.i..=j]).ok()?;
                    self.i = j + 1;
                    return serde_json::from_str::<String>(raw).ok();
                }
                _ => j += 1,
            }
        }
        None
    }
    /// Skip one value; returns its byte range.
    fn skip(&mut self) -> Option<(usize, usize)> {
        let c = self.peek()?;
        let s = self.i;
        match c {
            b'"' => {
                self.string()?;
            }
            b'{' | b'[' => {
                let close = if c == b'{' { b'}' } else { b']' };
                self.i += 1;
                if self.peek()? == close {
                    self.i += 1;
                    return Some((s, self.i));
                }
                loop {
                    if c == b'{' {
                        if self.peek()? != b'"' {
                            return None;
                        }
                        self.string()?;
                        if self.peek()? != b':' {
                            return None;
                        }
                        self.i += 1;
                    }
                    self.skip()?;
                    match self.peek()? {
                        b',' => self.i += 1,
                        x if x == close => {
                            self.i += 1;
                            break;
                        }
                        _ => return None,
                    }
                }
            }
            _ => {
                let mut j = self.i;
                while j < self.b.len() && !matches!(self.b[j], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    j += 1;
                }
                let tok = std::str::from_utf8(&self.b[self.i..j]).ok()?;
                if !(matches!(tok, "true" | "false" | "null") || serde_json::from_str::<serde_json::Number>(tok).is_ok()) {
                    return None;
                }
                self.i = j;
            }
        }
        Some((s, self.i))
    }
}

enum Nav {
    Found(usize),
    Missing,
    OutOfBounds,
    Invalid,
}

/// Navigate to the value at `segs` (byte offset of the value).
fn navigate(js: &mut Js, segs: &[Seg]) -> Nav {
    for seg in segs {
        match (js.peek(), seg) {
            (Some(b'{'), Seg::Key(k)) => {
                js.i += 1;
                let mut found = false;
                if js.peek() == Some(b'}') {
                    return Nav::Missing;
                }
                loop {
                    if js.peek() != Some(b'"') {
                        return Nav::Invalid;
                    }
                    let Some(name) = js.string() else { return Nav::Invalid };
                    if js.peek() != Some(b':') {
                        return Nav::Invalid;
                    }
                    js.i += 1;
                    if &name == k {
                        found = true;
                        break;
                    }
                    if js.skip().is_none() {
                        return Nav::Invalid;
                    }
                    match js.peek() {
                        Some(b',') => js.i += 1,
                        Some(b'}') => break,
                        _ => return Nav::Invalid,
                    }
                }
                if !found {
                    return Nav::Missing;
                }
            }
            (Some(b'['), Seg::Index(n)) => {
                js.i += 1;
                if js.peek() == Some(b']') {
                    return Nav::OutOfBounds;
                }
                let mut idx = 0;
                loop {
                    if idx == *n {
                        break;
                    }
                    if js.skip().is_none() {
                        return Nav::Invalid;
                    }
                    idx += 1;
                    match js.peek() {
                        Some(b',') => js.i += 1,
                        Some(b']') => return Nav::OutOfBounds,
                        _ => return Nav::Invalid,
                    }
                }
            }
            (None, _) => return Nav::Invalid,
            _ => return Nav::Missing,
        }
    }
    js.ws();
    Nav::Found(js.i)
}

fn iae(msg: &str) -> EsError {
    super::func::eval_warning(format!("java.lang.IllegalArgumentException: {msg}"))
}

/// `JSON_EXTRACT(string, path)`.
pub fn json_extract(text: &str, path: &str) -> Result<Val> {
    let segs = parse_json_path(path).map_err(|m| iae(&m))?;
    let b = text.as_bytes();
    let mut js = Js { b, i: 0 };
    if js.peek().is_none() {
        return Err(iae("empty JSON input"));
    }
    if !matches!(js.peek(), Some(b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-' | b'0'..=b'9')) {
        return Err(iae("invalid JSON input"));
    }
    let at = match navigate(&mut js, &segs) {
        Nav::Found(i) => i,
        Nav::Missing => return Err(iae(&format!("path [{path}] does not exist"))),
        Nav::OutOfBounds => return Err(iae("array index out of bounds")),
        Nav::Invalid => return Err(iae("invalid JSON input")),
    };
    js.i = at;
    let first = b.get(at).copied();
    let Some((s, e)) = js.skip() else { return Err(iae("invalid JSON input")) };
    let raw = &text[s..e];
    Ok(match first {
        Some(b'"') => Val::Str(serde_json::from_str::<String>(raw).map_err(|_| iae("invalid JSON input"))?),
        Some(b'n') => Val::Null,
        _ => Val::Str(raw.to_string()),
    })
}

/// `FieldExtract.validateFieldExtractPath`.
pub fn field_extract_path_error(path: &str) -> Option<&'static str> {
    if path.is_empty() {
        Some("field_extract path must not be empty")
    } else if path.contains(['[', ']']) {
        Some("field_extract path must be a literal flattened sub-field name. Brackets and array indices are not supported")
    } else {
        None
    }
}

/// `FIELD_EXTRACT(flattened, key)`: the sub-field's values as keywords
/// (sorted and deduplicated like the keyed doc values).
pub fn field_extract(root: &str, key: &str) -> Result<Val> {
    if let Some(m) = field_extract_path_error(key) {
        return Err(iae(m));
    }
    let v: serde_json::Value = serde_json::from_str(root).map_err(|_| iae("invalid JSON input"))?;
    let Some(o) = v.as_object() else { return Err(iae(&format!("path [{key}] does not exist"))) };
    let Some(x) = o.get(key) else { return Err(iae(&format!("path [{key}] does not exist"))) };
    fn leaves(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::String(s) => out.push(s.clone()),
            serde_json::Value::Number(n) => out.push(n.to_string()),
            serde_json::Value::Bool(b) => out.push(b.to_string()),
            serde_json::Value::Array(a) => a.iter().for_each(|x| leaves(x, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    leaves(x, &mut out);
    out.sort();
    out.dedup();
    Ok(Val::from_list(out.into_iter().map(Val::Str).collect()))
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    /// `Geohash.longEncode(String)` / `stringEncode(long)`, `GeoTileUtils`
    /// and `H3` string forms, Java arithmetic included.
    #[test]
    fn grid_strings_follow_es() {
        assert_eq!(geohash_long_encode_java("u3bu"), 13686180);
        assert_eq!(grid_from_string("geohash", "u3bu"), Ok(Some("u3bu".into())));
        // An unknown character is `indexOf == -1`: every higher bit set.
        assert_eq!(grid_from_string("geohash", "ua"), Ok(Some("zz".into())));
        assert_eq!(grid_from_string("geohash", ""), Ok(None));
        assert_eq!(grid_long_address("geohash", 13686180).as_deref(), Some("u3bu"));
        assert_eq!(grid_from_string("geotile", "+6/32/022"), Ok(Some("6/32/22".into())));
        assert_eq!(grid_from_string("geotile", "6/64/22"), Ok(None));
        assert!(grid_from_string("geotile", "6/32/22/").unwrap_err().contains("Invalid geotile_grid hash string of 6/32/22/."));
        assert_eq!(grid_long_address("geotile", (6 << 58) | (32 << 29) | 22).as_deref(), Some("6/32/22"));
        assert_eq!(grid_from_string("geohex", "811FBFFFFFFFFFF"), Ok(Some("811fbffffffffff".into())));
        assert_eq!(grid_from_string("geohex", "zz").unwrap_err(), "java.lang.NumberFormatException: For input string: \"zz\" under radix 16");
        assert!(grid_from_string("geohex", "-1").unwrap_err().contains("Illegal leading minus sign"));
        for (kind, addr) in [("geohash", "u0"), ("geotile", "6/32/22"), ("geohex", "811fbffffffffff")] {
            let l = grid_address_long(kind, addr).unwrap();
            assert_eq!(grid_long_address(kind, l).as_deref(), Some(addr), "{kind}");
        }
    }

    #[test]
    fn npoints_counts_a_bbox_as_four() {
        assert_eq!(npoints(&Geometry::Envelope { min_x: 0.0, max_x: 1.0, max_y: 1.0, min_y: 0.0 }), 4);
        assert_eq!(npoints(&Geometry::Collection(vec![Geometry::Point((0.0, 0.0)), Geometry::Envelope { min_x: 0.0, max_x: 1.0, max_y: 1.0, min_y: 0.0 }])), 5);
    }
}
