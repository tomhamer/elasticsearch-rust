//! Value encodings shared by indexing, queries, aggregations and output:
//! IP addresses (16-byte, hex-encoded so byte order == address order),
//! geo points (all ES input forms, geohash), numeric coercion and Java-like
//! number formatting.

use serde_json::Value;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Parse an IPv4/IPv6 address into its 16-byte (v4-mapped) form.
pub fn parse_ip(s: &str) -> Option<[u8; 16]> {
    // Metrics documents repeat the same addresses (a node's 100-odd
    // `host.ip` values in every document): parsed once per thread.
    IP_PARSED.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(b) = c.get(s) {
            return *b;
        }
        let b = parse_ip_uncached(s);
        if c.len() >= IP_CACHE_MAX {
            c.clear();
        }
        c.insert(s.to_string(), b);
        b
    })
}

fn parse_ip_uncached(s: &str) -> Option<[u8; 16]> {
    let ip: IpAddr = s.trim().parse().ok()?;
    Some(match ip {
        IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        IpAddr::V6(v6) => v6.octets(),
    })
}

/// Entries of each per-thread ip cache before it is cleared.
const IP_CACHE_MAX: usize = 4096;

thread_local! {
    static IP_PARSED: std::cell::RefCell<std::collections::HashMap<String, Option<[u8; 16]>, ahash::RandomState>> = Default::default();
    static IP_SHOWN: std::cell::RefCell<std::collections::HashMap<[u8; 16], String, ahash::RandomState>> = Default::default();
}

/// [`rustic_core::fieldtypes::ip::display`] of `b`, cached per thread (the
/// synthetic `_source` of metrics documents renders the same addresses in
/// every document).
pub fn ip_display_bytes(b: [u8; 16]) -> String {
    IP_SHOWN.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(s) = c.get(&b) {
            return s.clone();
        }
        let s = rustic_core::fieldtypes::ip::display(b);
        if c.len() >= IP_CACHE_MAX {
            c.clear();
        }
        c.insert(b, s.clone());
        s
    })
}

/// Hex term for an IP (sorts like the address).
pub fn ip_to_term(b: &[u8; 16]) -> String {
    // A hex table, not `format!` per byte (every ip value of every doc: a
    // k8s doc's 213 `host.ip` values were ~3% of ingest CPU).
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(32);
    for &x in b {
        s.push(HEX[(x >> 4) as usize] as char);
        s.push(HEX[(x & 15) as usize] as char);
    }
    s
}

pub fn term_to_ip(t: &str) -> Option<[u8; 16]> {
    if t.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = u8::from_str_radix(&t[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// Display form of an IP term (`192.168.0.1`, `::1`).
pub fn ip_display(t: &str) -> String {
    match term_to_ip(t) {
        Some(b) => {
            let v6 = Ipv6Addr::from(b);
            match v6.to_ipv4_mapped() {
                Some(v4) => v4.to_string(),
                None => v6.to_string(),
            }
        }
        None => t.to_string(),
    }
}

/// CIDR (`10.0.0.0/8`, `2001:db8::/32`) → inclusive term bounds.
pub fn cidr_bounds(s: &str) -> Option<(String, String)> {
    let (addr, bits) = s.split_once('/')?;
    let bits: u32 = bits.parse().ok()?;
    let ip: IpAddr = addr.parse().ok()?;
    let (bytes, prefix) = match ip {
        IpAddr::V4(v4) => {
            if bits > 32 {
                return None;
            }
            (v4.to_ipv6_mapped().octets(), bits + 96)
        }
        IpAddr::V6(v6) => {
            if bits > 128 {
                return None;
            }
            (v6.octets(), bits)
        }
    };
    let n = u128::from_be_bytes(bytes);
    let mask = if prefix == 0 { 0 } else { u128::MAX << (128 - prefix) };
    let lo = n & mask;
    let hi = lo | !mask;
    Some((ip_to_term(&lo.to_be_bytes()), ip_to_term(&hi.to_be_bytes())))
}

pub fn ipv4(a: u8, b: u8, c: u8, d: u8) -> String {
    ip_to_term(&Ipv4Addr::new(a, b, c, d).to_ipv6_mapped().octets())
}

// ---------------------------------------------------------------------------
// Geo.

const BASE32: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

pub fn geohash_decode(h: &str) -> Option<(f64, f64)> {
    let (mut lat_lo, mut lat_hi, mut lon_lo, mut lon_hi) = (-90.0f64, 90.0f64, -180.0f64, 180.0f64);
    let mut even = true;
    for c in h.bytes() {
        let idx = BASE32.iter().position(|&b| b == c.to_ascii_lowercase())?;
        for bit in (0..5).rev() {
            let on = (idx >> bit) & 1 == 1;
            if even {
                let mid = (lon_lo + lon_hi) / 2.0;
                if on { lon_lo = mid } else { lon_hi = mid }
            } else {
                let mid = (lat_lo + lat_hi) / 2.0;
                if on { lat_lo = mid } else { lat_hi = mid }
            }
            even = !even;
        }
    }
    // ES `GeoPoint.resetFromGeoHash` decodes the morton-encoded hash with
    // zero-filled low bits: the cell's lower-left corner.
    let _ = (lat_hi, lon_hi);
    Some((lat_lo, lon_lo))
}

pub fn geohash_encode(lat: f64, lon: f64, precision: usize) -> String {
    let (mut lat_lo, mut lat_hi, mut lon_lo, mut lon_hi) = (-90.0f64, 90.0f64, -180.0f64, 180.0f64);
    let mut out = String::new();
    let mut even = true;
    let mut bits = 0;
    let mut ch = 0usize;
    while out.len() < precision {
        if even {
            let mid = (lon_lo + lon_hi) / 2.0;
            if lon >= mid {
                ch = (ch << 1) | 1;
                lon_lo = mid;
            } else {
                ch <<= 1;
                lon_hi = mid;
            }
        } else {
            let mid = (lat_lo + lat_hi) / 2.0;
            if lat >= mid {
                ch = (ch << 1) | 1;
                lat_lo = mid;
            } else {
                ch <<= 1;
                lat_hi = mid;
            }
        }
        even = !even;
        bits += 1;
        if bits == 5 {
            out.push(BASE32[ch] as char);
            bits = 0;
            ch = 0;
        }
    }
    out
}

fn valid_point(lat: f64, lon: f64) -> Option<(f64, f64)> {
    if (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) {
        Some((lat, lon))
    } else {
        None
    }
}

/// `GeoUtils.normalizePoint`: latitudes beyond a pole are reflected
/// (moving the longitude half way round) and longitudes wrapped into
/// [-180, 180].
pub fn normalize_point(lat: f64, lon: f64) -> Option<(f64, f64)> {
    if !lat.is_finite() || !lon.is_finite() {
        return None;
    }
    let centered = |x: f64, d: f64| {
        let mut r = x % d;
        if r <= 0.0 {
            r += d;
        }
        if r > d / 2.0 {
            r -= d;
        }
        r
    };
    let (mut lat, mut lon) = (lat, lon);
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        lat = centered(lat, 360.0);
        let mut shift = true;
        if lat < -90.0 {
            lat = -180.0 - lat;
        } else if lat > 90.0 {
            lat = 180.0 - lat;
        } else {
            shift = false;
        }
        if shift {
            lon += 180.0;
        }
        lon = centered(lon, 360.0);
    }
    Some((lat, lon))
}

/// Parse one geo point in any ES form: `{lat, lon}`, `"lat,lon"`,
/// geohash, `[lon, lat]`, `POINT (lon lat)`, GeoJSON `{type: Point, coordinates}`.
pub fn parse_geo_point(v: &Value) -> Option<(f64, f64)> {
    parse_geo_point_with(v, valid_point)
}

/// [`parse_geo_point`] for a field with `ignore_malformed`: out-of-range
/// coordinates are normalized, not refused (ES `GeoPointFieldMapper`,
/// finding 457).
pub fn parse_geo_point_normalized(v: &Value) -> Option<(f64, f64)> {
    parse_geo_point_with(v, normalize_point)
}

fn parse_geo_point_with(v: &Value, valid_point: fn(f64, f64) -> Option<(f64, f64)>) -> Option<(f64, f64)> {
    match v {
        Value::Object(m) => {
            if let (Some(lat), Some(lon)) = (m.get("lat"), m.get("lon")) {
                return valid_point(num_of(lat)?, num_of(lon)?);
            }
            if m.get("type").and_then(Value::as_str).map(|t| t.eq_ignore_ascii_case("point")) == Some(true) {
                let c = m.get("coordinates")?.as_array()?;
                return valid_point(c.get(1)?.as_f64()?, c.first()?.as_f64()?);
            }
            // `{"location": "POINT (lon lat)"}` or `{"location": "lat,lon"}`.
            if let Some(loc) = m.get("location") {
                return parse_geo_point_with(loc, valid_point);
            }
            None
        }
        Value::Array(a) if a.len() >= 2 && a.iter().all(Value::is_number) => valid_point(a[1].as_f64()?, a[0].as_f64()?),
        Value::String(s) => {
            let t = s.trim();
            if let Some(rest) = t.strip_prefix("POINT").or_else(|| t.strip_prefix("point")) {
                let inner = rest.trim().trim_start_matches('(').trim_end_matches(')');
                let mut it = inner.split_whitespace();
                let lon: f64 = it.next()?.parse().ok()?;
                let lat: f64 = it.next()?.parse().ok()?;
                // ES reads WKT through `GeographyValidator`, which refuses
                // out-of-range coordinates: never normalized, even with
                // ignore_malformed (the value is malformed).
                if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
                    return None;
                }
                return valid_point(lat, lon);
            }
            if let Some((a, b)) = t.split_once(',') {
                return valid_point(a.trim().parse().ok()?, b.trim().parse().ok()?);
            }
            geohash_decode(t)
        }
        _ => None,
    }
}

/// Is this array a single point (`[lon, lat]`) rather than a list of points?
pub fn is_point_array(a: &[Value]) -> bool {
    (a.len() == 2 || a.len() == 3) && a.iter().all(Value::is_number)
}

fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Haversine distance in meters (ES `arc`).
pub fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_008.7714;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = (lat2 - lat1).to_radians();
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * R * a.sqrt().asin()
}

/// Parse a distance (`12km`, `5mi`, `100`) into meters.
pub fn parse_distance(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            let s = s.trim();
            let idx = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());
            let (n, unit) = s.split_at(idx);
            let n: f64 = n.trim().parse().ok()?;
            Some(n * unit_meters(unit.trim())?)
        }
        _ => None,
    }
}

pub fn unit_meters(unit: &str) -> Option<f64> {
    Some(match unit {
        "" | "m" | "meters" => 1.0,
        "km" | "kilometers" => 1000.0,
        "mi" | "miles" => 1609.344,
        "yd" | "yards" => 0.9144,
        "ft" | "feet" => 0.3048,
        "in" | "inch" => 0.0254,
        "cm" | "centimeters" => 0.01,
        "mm" | "millimeters" => 0.001,
        "NM" | "nmi" | "nauticalmiles" => 1852.0,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Numbers.

/// Java `Double.toString`-compatible JSON number for a double (shortest
/// repr; scientific with `E` outside [1e-3, 1e7)).
pub fn java_double(v: f64) -> Value {
    if !v.is_finite() {
        return Value::Null;
    }
    serde_json::Number::from_f64(v).map(Value::Number).unwrap_or(Value::Null)
}

/// Java `Double.toString` (`1.0`, `0.2`, `1.0E7`, `-1.7976931348623157E308`).
pub fn java_double_string(d: f64) -> String {
    if d.is_nan() {
        return "NaN".into();
    }
    if d.is_infinite() {
        return if d > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if d == 0.0 {
        return if d.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = d.abs();
    if (1e-3..1e7).contains(&a) {
        let s = format!("{d}");
        return if s.contains('.') { s } else { format!("{s}.0") };
    }
    let s = format!("{d:e}");
    let (m, e) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
    format!("{m}E{e}")
}

/// Java `Float.toString`-compatible JSON number for a float: the shortest
/// decimal that round-trips the `f32` (`2.1234567`, `3.0`), as Jackson
/// writes a Java `float`.
pub fn java_float(v: f32) -> Value {
    if !v.is_finite() {
        return Value::Null;
    }
    // The shortest round-tripping decimal of the `f32` (Ryu, as Rust's
    // `Display` but without the formatting machinery or a heap string:
    // vectors run this per component on every indexed document).
    java_double(ryu::Buffer::new().format_finite(v).parse::<f64>().unwrap_or(v as f64))
}

/// Render an f64 that holds an integral value of a `long` field as a JSON
/// integer.
pub fn num_value(v: f64, integral: bool) -> Value {
    if integral && v.fract() == 0.0 && v.abs() < 9.2e18 {
        Value::from(v as i64)
    } else {
        java_double(v)
    }
}

/// Round an f64 to f32 precision (ES `float` fields), returned as f64.
pub fn to_float(v: f64) -> f64 {
    v as f32 as f64
}

/// Round to IEEE half precision (ES `half_float`, Lucene
/// `HalfFloatPoint`): the input as a float, then to the nearest binary16
/// value, ties to even, including into and out of the subnormal range
/// (production review #452: values between 2^-25 and 2^-24 were flushed to
/// zero instead of rounding up to the smallest subnormal).
pub fn to_half_float(v: f64) -> f64 {
    let f = v as f32;
    if f == 0.0 || !f.is_finite() {
        return f as f64;
    }
    let x = (f as f64).abs();
    // The unit in the last place of the binade (subnormals: 2^-24).
    let e = x.log2().floor() as i32;
    let ulp = 2f64.powi(e.max(-14) - 10);
    let r = (x / ulp).round_ties_even() * ulp;
    let r = if r > 65504.0 { f64::INFINITY } else { r };
    r.copysign(v)
}

/// The binary16 bit pattern of a value [`to_half_float`] returns.
fn half_bits(h: f64) -> u16 {
    let sign = if h.is_sign_negative() { 0x8000u16 } else { 0 };
    let x = h.abs();
    if x == 0.0 {
        return sign;
    }
    if x.is_infinite() {
        return sign | 0x7c00;
    }
    if x < 2f64.powi(-14) {
        return sign | (x / 2f64.powi(-24)) as u16;
    }
    let e = x.log2().floor() as i32;
    let m = ((x / 2f64.powi(e) - 1.0) * 1024.0) as u16;
    sign | (((e + 15) as u16) << 10) | m
}

fn half_from_bits(b: u16) -> f64 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f64;
    sign * match e {
        0 => m * 2f64.powi(-24),
        31 => f64::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The half float right above half value `h` (Lucene `HalfFloatPoint.nextUp`).
fn half_next_up(h: f64) -> f64 {
    if h == f64::INFINITY {
        return h;
    }
    if h == 0.0 {
        return 2f64.powi(-24);
    }
    let b = half_bits(h);
    let r = half_from_bits(if h > 0.0 { b + 1 } else { b - 1 });
    if r == 0.0 { 0.0 } else { r }
}

/// The half float right below half value `h`.
fn half_next_down(h: f64) -> f64 {
    -half_next_up(-h)
}

/// The range bound of a `half_float` field (ES `HALF_FLOAT.rangeQuery`):
/// the smallest half value on the requested side of the (float) bound, so
/// a bound between two half floats never rounds to the wrong side
/// (production review #451). `lower`: the bound's side is above it.
pub fn half_float_bound(v: f64, lower: bool, inclusive: bool) -> f64 {
    let x = v as f32 as f64;
    let h = to_half_float(x);
    if lower {
        let ceil = if h >= x { h } else { half_next_up(h) };
        if inclusive || ceil > x { ceil } else { half_next_up(ceil) }
    } else {
        let floor = if h <= x { h } else { half_next_down(h) };
        if inclusive || floor < x { floor } else { half_next_down(floor) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ips() {
        let t = ip_to_term(&parse_ip("192.168.1.10").unwrap());
        assert_eq!(ip_display(&t), "192.168.1.10");
        assert_eq!(ip_display(&ip_to_term(&parse_ip("2001:db8::1").unwrap())), "2001:db8::1");
        let (lo, hi) = cidr_bounds("192.168.0.0/16").unwrap();
        assert!(lo <= t && t <= hi);
        let other = ip_to_term(&parse_ip("10.0.0.1").unwrap());
        assert!(other < lo);
        assert!(parse_ip("300.1.1.1").is_none());
        assert_eq!(ipv4(10, 0, 0, 1), other);
    }

    #[test]
    fn geo_forms() {
        let want = (41.12, -71.34);
        assert_eq!(parse_geo_point(&json!({"lat": 41.12, "lon": -71.34})), Some(want));
        // Finding 457: normalized with ignore_malformed, refused otherwise.
        assert_eq!(parse_geo_point(&json!({"lat": 10.0, "lon": 190.0})), None);
        assert_eq!(parse_geo_point_normalized(&json!({"lat": 10.0, "lon": 190.0})), Some((10.0, -170.0)));
        assert_eq!(parse_geo_point_normalized(&json!([-190.0, 95.0])), Some((85.0, -10.0)));
        assert_eq!(parse_geo_point_normalized(&json!("-100,0")), Some((-80.0, 180.0)));
        assert_eq!(parse_geo_point_normalized(&json!({"lat": 1.0, "lon": 2.0})), Some((1.0, 2.0)));
        // WKT is validated, never normalized (ES GeographyValidator).
        assert_eq!(parse_geo_point_normalized(&json!("POINT (-77.03653 1000)")), None);
        assert_eq!(parse_geo_point_normalized(&json!("POINT (-71.34 41.12)")), Some((41.12, -71.34)));
        assert_eq!(parse_geo_point(&json!("41.12,-71.34")), Some(want));
        assert_eq!(parse_geo_point(&json!([-71.34, 41.12])), Some(want));
        assert_eq!(parse_geo_point(&json!("POINT (-71.34 41.12)")), Some(want));
        assert_eq!(parse_geo_point(&json!({"type": "Point", "coordinates": [-71.34, 41.12]})), Some(want));
        let (lat, lon) = parse_geo_point(&json!("drm3btev3e86")).unwrap();
        assert!((lat - 41.12).abs() < 1e-4 && (lon + 71.34).abs() < 1e-4);
        assert_eq!(geohash_encode(41.12, -71.34, 12), "drm3btev3e86");
        assert!(parse_geo_point(&json!({"lat": 100, "lon": 0})).is_none());
        let d = haversine(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((d / 1000.0 - 5570.0).abs() < 10.0);
        assert_eq!(parse_distance(&json!("12km")), Some(12000.0));
    }

    #[test]
    fn floats() {
        assert_eq!(to_float(1.1), 1.100000023841858);
        assert_eq!(to_half_float(1.1), 1.099609375);
        assert_eq!(to_half_float(65504.0), 65504.0);
        // Production review #452: between 2^-25 and 2^-24 rounds up to the
        // smallest subnormal; 2^-25 itself ties to even (zero).
        assert_eq!(to_half_float(0.00000004), 2f64.powi(-24));
        assert_eq!(to_half_float(-0.00000004), -2f64.powi(-24));
        assert_eq!(to_half_float(2f64.powi(-25)), 0.0);
        assert_eq!(to_half_float(1.5 * 2f64.powi(-24)), 2.0 * 2f64.powi(-24));
        assert_eq!(to_half_float(2f64.powi(-14) - 2f64.powi(-26)), 2f64.powi(-14));
        assert_eq!(to_half_float(65519.0), 65504.0);
        assert_eq!(to_half_float(65520.0), f64::INFINITY);
        for b in [1u16, 2, 0x3ff, 0x400, 0x3c00, 0x3c01, 0x7bff] {
            assert_eq!(half_bits(half_from_bits(b)), b);
            assert_eq!(to_half_float(half_from_bits(b)), half_from_bits(b));
        }
        // Production review #451: bounds between adjacent half floats.
        let next = 1.0009765625;
        assert_eq!(half_float_bound(1.0001, true, true), next);
        assert_eq!(half_float_bound(1.0008, true, false), next);
        assert_eq!(half_float_bound(1.0, true, false), next);
        assert_eq!(half_float_bound(1.0, true, true), 1.0);
        assert_eq!(half_float_bound(1.0008, false, true), 1.0);
        assert_eq!(half_float_bound(1.0001, false, false), 1.0);
        assert_eq!(half_float_bound(next, false, false), 1.0);
        assert_eq!(half_float_bound(next, false, true), next);
        assert_eq!(half_float_bound(0.0, true, false), 2f64.powi(-24));
        assert_eq!(num_value(3.0, true), json!(3));
        assert_eq!(num_value(3.5, true), json!(3.5));
    }
}
