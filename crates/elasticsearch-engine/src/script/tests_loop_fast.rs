//! Interpreter fast paths used by loop-heavy scripts (Rally geopoint
//! `geoGrid_sort_*` runtime fields): compile-time bound `String` methods,
//! `String + String`, typed primitive locals, `int` bit operations, `i++`,
//! and the loop counter. Every fast path must give exactly the value or
//! error of the general path it short-cuts.

use super::compile::compile_arc;
use super::error::SResult;
use super::*;
use serde_json::json;

/// Rally `geoGrid_sort_geohash` (exact source).
const GEOHASH: &str = "String base32 = '0123456789bcdefghjkmnpqrstuvwxyz'; double lat = doc['location'].lat; double lon = doc['location'].lon; double minLat = -90; double maxLat = 90; double minLon = -180; double maxLon = 180; boolean even = true; int bit = 0; int ch = 0; String hash = ''; while (hash.length() < 5) { double mid; if (even) { mid = (minLon + maxLon) / 2; if (lon >= mid) { ch |= (16 >> bit); minLon = mid; } else { maxLon = mid; } } else { mid = (minLat + maxLat) / 2; if (lat >= mid) { ch |= (16 >> bit); minLat = mid; } else { maxLat = mid; } } even = !even; if (bit < 4) { bit++; } else { hash += base32.substring(ch, ch + 1); bit = 0; ch = 0; } } emit(hash);";

/// Rally `geoGrid_sort_geotile` (exact source).
const GEOTILE: &str = "int zoom = 5; double lon = doc['location'].lon; double lat = doc['location'].lat; if (lat > 85.0511287798066) { lat = 85.0511287798066; } if (lat < -85.0511287798066) { lat = -85.0511287798066; } int tiles = 1 << zoom; int x = (int) Math.floor((lon + 180.0) / 360.0 * tiles); if (x < 0) { x = 0; } else if (x >= tiles) { x = tiles - 1; } double latSin = Math.sin(Math.toRadians(lat)); double yTile = (0.5 - (Math.log((1.0 + latSin) / (1.0 - latSin)) / (4.0 * Math.PI))) * tiles; int y = (int) Math.floor(yTile); if (y < 0) { y = 0; } else if (y >= tiles) { y = tiles - 1; } emit(zoom + '/' + x + '/' + y);";

/// Rally `geoGrid_sort_geohex` (exact source).
const GEOHEX: &str = "double lon = doc['location'].lon; double lat = doc['location'].lat; int res = 5; double size = 180.0 / Math.pow(Math.sqrt(7), res); double q = (2.0 / 3.0 * lon) / size; double r = (-1.0 / 3.0 * lon + Math.sqrt(3.0) / 3.0 * lat) / size; double x = q; double zc = r; double y = -x - zc; int rx = (int) Math.round(x); int ry = (int) Math.round(y); int rz = (int) Math.round(zc); double xd = Math.abs(rx - x); double yd = Math.abs(ry - y); double zd = Math.abs(rz - zc); if (xd > yd && xd > zd) { rx = -ry - rz; } else if (yd > zd) { ry = -rx - rz; } else { rz = -rx - ry; } emit(rx + '/' + rz);";

/// The script with its doc-value reads replaced by `params.lat/lon` (the
/// unit tests have no index; the loop and string work are unchanged).
fn on_params(src: &str) -> String {
    src.replace("doc['location'].lat", "params.lat").replace("doc['location'].lon", "params.lon")
}

/// The same script with every typed local declared `def` (general,
/// dynamically typed paths throughout).
fn all_def(src: &str) -> String {
    let mut s = src.to_string();
    for t in ["String ", "double ", "boolean ", "int "] {
        s = s.replace(t, "def ");
    }
    // Integer literals that initialised `double`s: keep double arithmetic.
    for (from, to) in [("= -90;", "= -90.0;"), ("= 90;", "= 90.0;"), ("= -180;", "= -180.0;"), ("= 180;", "= 180.0;")] {
        s = s.replace(from, to);
    }
    s
}

fn executor(src: &str, ctx: ScriptContext, params: serde_json::Value) -> Executor {
    let prog = compile_arc(src, ctx).unwrap_or_else(|e| panic!("compile {src:?}: {} at {:?}", e.msg, e.pos));
    Executor::new(prog, &params, None, None)
}

fn emitted_at(e: &mut Executor, lat: f64, lon: f64) -> String {
    e.set_param("lat", Val::Double(lat));
    e.set_param("lon", Val::Double(lon));
    if let Err(err) = e.run() {
        panic!("({lat}, {lon}): {} {}", err.kind, err.msg);
    }
    let out = e.take_emitted();
    assert_eq!(out.len(), 1, "({lat}, {lon})");
    match &out[0] {
        Val::Str(s) => s.to_string(),
        other => panic!("({lat}, {lon}): emitted {other:?}"),
    }
}

/// splitmix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Random points, the extremes, and exact geohash cell boundaries.
fn points(n: usize) -> Vec<(f64, f64)> {
    let mut rng = Rng(0x5EED_6E0A);
    let mut out = vec![(0.0, 0.0), (-0.0, -0.0), (90.0, 180.0), (-90.0, -180.0), (90.0, -180.0), (-90.0, 180.0), (85.0511287798066, 0.0), (-85.0511287798066, 0.0), (89.999999, 179.999999), (1e-300, -1e-300)];
    for _ in 0..n {
        out.push((-90.0 + 180.0 * rng.unit(), -180.0 + 360.0 * rng.unit()));
    }
    for _ in 0..n / 4 {
        let (j, k) = ((rng.next() % 4097) as f64, (rng.next() % 8193) as f64);
        out.push((-90.0 + 180.0 * j / 4096.0, -180.0 + 360.0 * k / 8192.0));
    }
    out
}

/// The geohash script transliterated.
fn geohash_ref(lat: f64, lon: f64) -> String {
    let base32 = b"0123456789bcdefghjkmnpqrstuvwxyz";
    let (mut min_lat, mut max_lat, mut min_lon, mut max_lon) = (-90.0f64, 90.0f64, -180.0f64, 180.0f64);
    let (mut even, mut bit, mut ch) = (true, 0i32, 0i32);
    let mut hash = String::new();
    while hash.len() < 5 {
        if even {
            let mid = (min_lon + max_lon) / 2.0;
            if lon >= mid {
                ch |= 16 >> bit;
                min_lon = mid;
            } else {
                max_lon = mid;
            }
        } else {
            let mid = (min_lat + max_lat) / 2.0;
            if lat >= mid {
                ch |= 16 >> bit;
                min_lat = mid;
            } else {
                max_lat = mid;
            }
        }
        even = !even;
        if bit < 4 {
            bit += 1;
        } else {
            hash.push(base32[ch as usize] as char);
            bit = 0;
            ch = 0;
        }
    }
    hash
}

/// Java `(int)` of a double.
fn d2i(d: f64) -> i32 {
    if d.is_nan() { 0 } else { d as i32 }
}

/// The geotile script transliterated (`Math.*` as rustic computes them).
fn geotile_ref(lat: f64, lon: f64) -> String {
    let zoom = 5i32;
    let mut lat = lat;
    if lat > 85.0511287798066 {
        lat = 85.0511287798066;
    }
    if lat < -85.0511287798066 {
        lat = -85.0511287798066;
    }
    let tiles = 1i32 << zoom;
    let mut x = d2i(((lon + 180.0) / 360.0 * tiles as f64).floor());
    if x < 0 {
        x = 0;
    } else if x >= tiles {
        x = tiles - 1;
    }
    let lat_sin = lat.to_radians().sin();
    let y_tile = (0.5 - (((1.0 + lat_sin) / (1.0 - lat_sin)).ln() / (4.0 * std::f64::consts::PI))) * tiles as f64;
    let mut y = d2i(y_tile.floor());
    if y < 0 {
        y = 0;
    } else if y >= tiles {
        y = tiles - 1;
    }
    format!("{zoom}/{x}/{y}")
}

/// The geohex script transliterated (`Math.round` as rustic computes it).
fn geohex_ref(lat: f64, lon: f64) -> String {
    let res = 5i32;
    let size = 180.0 / 7f64.sqrt().powf(res as f64);
    let q = (2.0 / 3.0 * lon) / size;
    let r = (-1.0 / 3.0 * lon + 3f64.sqrt() / 3.0 * lat) / size;
    let (x, zc) = (q, r);
    let y = -x - zc;
    let round = |v: f64| (if v.is_nan() { 0 } else { (v + 0.5).floor() as i64 }) as i32;
    let (mut rx, mut ry, mut rz) = (round(x), round(y), round(zc));
    let xd = (rx as f64 - x).abs();
    let yd = (ry as f64 - y).abs();
    let zd = (rz as f64 - zc).abs();
    if xd > yd && xd > zd {
        rx = -ry - rz;
    } else if yd > zd {
        ry = -rx - rz;
    } else {
        rz = -rx - ry;
    }
    let _ = ry;
    format!("{rx}/{rz}")
}

#[test]
fn rally_geogrid_scripts_match_reference() {
    let params = json!({"lat": 0.0, "lon": 0.0});
    let cases: [(&str, fn(f64, f64) -> String); 3] = [(GEOHASH, geohash_ref), (GEOTILE, geotile_ref), (GEOHEX, geohex_ref)];
    for (src, reference) in cases {
        let typed_src = on_params(src);
        let def_src = all_def(&typed_src);
        assert!(!def_src.contains("double ") && !def_src.contains("int ") && !def_src.contains("String "), "{def_src}");
        let mut typed = executor(&typed_src, ScriptContext::RuntimeKeyword, params.clone());
        let mut dynamic = executor(&def_src, ScriptContext::RuntimeKeyword, params.clone());
        for (lat, lon) in points(4000) {
            let want = reference(lat, lon);
            assert_eq!(emitted_at(&mut typed, lat, lon), want, "{typed_src} at ({lat}, {lon})");
            assert_eq!(emitted_at(&mut dynamic, lat, lon), want, "{def_src} at ({lat}, {lon})");
        }
    }
    // The transliteration is the standard geohash.
    for (lat, lon) in points(500) {
        assert_eq!(geohash_ref(lat, lon), rustic_core::geo::geohash(lat, lon, 5), "({lat}, {lon})");
    }
}

/// A script's outcome vs a direct general-path call: same value (variant
/// included) or same exception (kind, message, JDK frames).
fn assert_same(what: &str, got: SResult<Val>, want: SResult<Val>) {
    match (got, want) {
        (Ok(g), Ok(w)) => assert_eq!(format!("{g:?}"), format!("{w:?}"), "{what}"),
        (Err(g), Err(w)) => {
            assert_eq!((g.kind.as_str(), g.msg.as_str(), g.frames), (w.kind.as_str(), w.msg.as_str(), w.frames), "{what}");
            assert!(g.pos.is_some(), "{what}: error without a position");
        }
        (g, w) => panic!("{what}: script {g:?} vs general {w:?}"),
    }
}

#[test]
fn bound_string_methods_match_general_dispatch() {
    let strings = ["", "a", "abc", "0123456789bcdefghjkmnpqrstuvwxyz", "h\u{e9}llo", "\u{65e5}\u{672c}\u{8a9e}", "a\u{1F600}b", "\u{1F600}"];
    let calls: [(&str, usize); 5] = [("length", 0), ("isEmpty", 0), ("charAt", 1), ("substring", 1), ("substring", 2)];
    let mut general = executor("1", ScriptContext::PainlessTest, json!({}));
    for (m, n) in calls {
        let (dargs, targs) = match n {
            0 => ("", ""),
            1 => ("params.a", "a"),
            _ => ("params.a, params.b", "a, b"),
        };
        let mut forms = vec![
            format!("params.s.{m}({dargs})"),
            format!("String s = params.s; int a = params.a; int b = params.b; return s.{m}({targs});"),
            format!("String s = params.s; return s.{m}({dargs});"),
            format!("def s = params.s; int a = params.a; int b = params.b; return s.{m}({targs});"),
        ];
        if m == "length" {
            // Numeric fast path of `local.length()`.
            forms.push("String s = params.s; return s.length() * 1;".to_string());
            forms.push("String s = params.s; return s.length() < 3;".to_string());
        }
        for form in &forms {
            let mut e = executor(form, ScriptContext::PainlessTest, json!({"s": "", "a": 0, "b": 0}));
            for s in strings {
                let units = s.encode_utf16().count() as i32;
                for a in -1..=units + 1 {
                    for b in -1..=units + 1 {
                        if (n < 2 && b != 0) || (n < 1 && a != 0) {
                            continue;
                        }
                        e.set_param("s", Val::str(s));
                        e.set_param("a", Val::Int(a));
                        e.set_param("b", Val::Int(b));
                        let got = e.run();
                        let args: Vec<Val> = [Val::Int(a), Val::Int(b)].into_iter().take(n).collect();
                        let mut want = methods::call_method(&mut general.rt, Val::str(s), m, args);
                        if form.ends_with("* 1;") {
                            want = want.map(|v| super::runtime::binary(ast::BinOp::Mul, &v, &Val::Int(1)).unwrap());
                        } else if form.ends_with("< 3;") {
                            want = want.map(|v| super::runtime::binary(ast::BinOp::Lt, &v, &Val::Int(3)).unwrap());
                        }
                        assert_same(&format!("{form} with s={s:?} a={a} b={b}"), got, want);
                    }
                }
            }
        }
    }
}

#[test]
fn string_fast_paths_keep_utf16_semantics() {
    let mut e = executor(
        "String s = params.s; return [s.length(), s.substring(1, 3), s.substring(1, 2).length(), (int) s.charAt(1), (int) s.charAt(2), s.substring(3), s.length() - 1];",
        ScriptContext::PainlessTest,
        json!({"s": "a\u{1F600}b"}),
    );
    let v = e.run().unwrap();
    assert_eq!(v.to_json(), json!([4, "\u{1F600}", 1, 0xD83D, 0xDE00, "b", 3]));
    // A lone surrogate (`Val::Utf16` receiver) takes the general path.
    let mut e = executor("String s = 'a' + (char) 55357 + 'b'; return [s.length(), s.substring(1, 2).length(), (int) s.charAt(1), s.isEmpty(), s.length() < 4];", ScriptContext::PainlessTest, json!({}));
    assert_eq!(e.run().unwrap().to_json(), json!([3, 1, 55357, false, true]));
    // `String + String` and mixed concatenations.
    let mut e = executor("String h = ''; String b = 'xyz'; for (int i = 0; i < 3; i++) { h += b.substring(i, i + 1); } return [h, h + 'q', 'q' + h, h + (char) 98, h + 1, '' + '', h + params.e, params.e + params.e];", ScriptContext::PainlessTest, json!({"e": "\u{e9}"}));
    assert_eq!(e.run().unwrap().to_json(), json!(["xyz", "xyzq", "qxyz", "xyzb", "xyz1", "", "xyz\u{e9}", "\u{e9}\u{e9}"]));
    let long = "x".repeat(100);
    let mut e = executor("String s = params.s; return s + s;", ScriptContext::PainlessTest, json!({"s": long.clone()}));
    assert_eq!(e.run().unwrap().to_json(), json!(long.repeat(2)));
    // Null receivers: the general path's NPE; null-safe calls yield null.
    let mut e = executor("String s = null; return s?.length();", ScriptContext::PainlessTest, json!({}));
    assert_eq!(e.run().unwrap().to_json(), serde_json::Value::Null);
    let mut e = executor("String s = null; return s.length();", ScriptContext::PainlessTest, json!({}));
    assert_eq!(e.run().unwrap_err().kind, "null_pointer_exception");
}

#[test]
fn string_fast_path_errors_keep_kind_message_and_position() {
    let src = "String s = 'abc'; int a = 2; int b = 5; return s.substring(a, b);";
    let err = executor(src, ScriptContext::PainlessTest, json!({})).run().unwrap_err();
    assert_eq!(err.kind, "string_index_out_of_bounds_exception");
    assert_eq!(err.msg, "Range [2, 5) out of bounds for length 3");
    assert_eq!(err.pos, Some(src.find(".substring").unwrap()));
    let src = "String s = 'abc'; int i = 3; return s.charAt(i);";
    let err = executor(src, ScriptContext::PainlessTest, json!({})).run().unwrap_err();
    assert_eq!(err.kind, "string_index_out_of_bounds_exception");
    assert_eq!(err.msg, "Index 3 out of bounds for length 3");
    assert_eq!(err.pos, Some(src.find(".charAt").unwrap()));
    // Non-int arguments are cast by the general path exactly as before.
    let src = "String s = 'abc'; def a = 1L; return s.substring(a);";
    assert_eq!(executor(src, ScriptContext::PainlessTest, json!({})).run().unwrap().to_json(), json!("bc"));
    let src = "String s = 'abc'; def a = 1.5; return s.substring(a);";
    assert_eq!(executor(src, ScriptContext::PainlessTest, json!({})).run().unwrap_err().kind, "class_cast_exception");
}

#[test]
fn typed_local_fast_paths_match_java() {
    let cases = [
        ("int i = 2147483647; i++; return i;", json!(-2147483648i64)),
        ("int i = -2147483648; i--; return i;", json!(2147483647)),
        ("int i = 5; int j = i++; int k = ++i; return [i, j, k];", json!([7, 5, 7])),
        ("int i = 5; int j = i--; int k = --i; return [i, j, k];", json!([3, 5, 3])),
        ("long l = 9223372036854775807L; long m = l++; return [l, m];", json!([i64::MIN, i64::MAX])),
        ("long l = 1; ++l; l--; return l;", json!(1)),
        ("int ch = 0; ch |= (16 >> 3); ch ^= 5; ch &= 7; ch <<= 33; return ch;", json!(14)),
        ("int x = -1; x >>>= 28; int y = -16; y >>= 2; return [x, y, -1 >>> 0, 1 << -1];", json!([15, -4, -1, -2147483648i64])),
        ("double d = 0; d = 1 + 2; long l = 0; l = 7; int i = 0; i = 3 * 4; return [d, l, i];", json!([3.0, 7, 12])),
        ("double d = 0; d = 2147483647 + 1; long l = 0; l = 2147483647 + 1; return [d, l];", json!([-2147483648.0, -2147483648i64])),
        ("double d = 0; d = params.x; return d;", json!(5.0)),
        ("double d = 0; d = params.x / 2; return d;", json!(2.0)),
        ("long l = 0; l = params.x * 3; return l;", json!(15)),
        ("int i = 0; double s = 0; while (i < 10) { s = s + i; i++; } return [i, s];", json!([10, 45.0])),
        ("int bit = 0; int n = 0; while (n < 7) { if (bit < 4) { bit++; } else { bit = 0; } n++; } return bit;", json!(2)),
    ];
    for (src, want) in cases {
        let got = executor(src, ScriptContext::PainlessTest, json!({"x": 5})).run().unwrap_or_else(|e| panic!("{src}: {} {}", e.kind, e.msg));
        assert_eq!(got.to_json(), want, "{src}");
    }
    // Primitive stores keep the general path's conversion errors.
    let err = executor("double d = 0; d = params.s; return d;", ScriptContext::PainlessTest, json!({"s": "x"})).run().unwrap_err();
    assert_eq!(err.kind, "class_cast_exception");
    let err = executor("int i = 0; i = params.l; return i;", ScriptContext::PainlessTest, json!({"l": 5_000_000_000i64})).run().unwrap_err();
    assert_eq!(err.kind, "class_cast_exception");
}

#[test]
fn loop_counter_limit_is_unchanged() {
    let ok = executor("int i = 0; while (i < 999999) { i++; } return i;", ScriptContext::PainlessTest, json!({})).run().unwrap();
    assert_eq!(ok.to_json(), json!(999999));
    let src = "int i = 0; while (i < 1000000) { i++; } return i;";
    let err = executor(src, ScriptContext::PainlessTest, json!({})).run().unwrap_err();
    assert_eq!(err.kind, "painless_error");
    assert_eq!(err.msg, "The maximum number of statements that can be executed in a loop has been reached.");
    assert_eq!(err.pos, Some(src.find("while").unwrap()));
    // The budget is per execution.
    let mut e = executor("int i = 0; while (i < 600000) { i++; } return i;", ScriptContext::PainlessTest, json!({}));
    for _ in 0..3 {
        assert_eq!(e.run().unwrap().to_json(), json!(600000));
    }
}

/// `GEOGRID_N=5000000 cargo test -p elasticsearch-engine --release geogrid_script_timing -- --ignored --nocapture`
#[test]
#[ignore]
fn geogrid_script_timing() {
    let n: usize = std::env::var("GEOGRID_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let pts = points(4096);
    for (name, src) in [("geohash", GEOHASH), ("geotile", GEOTILE), ("geohex", GEOHEX)] {
        let mut e = executor(&on_params(src), ScriptContext::RuntimeKeyword, json!({"lat": 0.0, "lon": 0.0}));
        let mut sink = 0usize;
        let start = std::time::Instant::now();
        for i in 0..n {
            let (lat, lon) = pts[i % pts.len()];
            e.set_param("lat", Val::Double(lat));
            e.set_param("lon", Val::Double(lon));
            e.rt.emitted.clear();
            e.run().unwrap();
            sink += e.rt.emitted.len();
        }
        let el = start.elapsed();
        assert_eq!(sink, n);
        eprintln!("{name}: {n} runs in {el:?} = {:.1} ns/run", el.as_nanos() as f64 / n as f64);
    }
}
