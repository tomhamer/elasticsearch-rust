//! `GET|POST /{index}/_mvt/{field}/{zoom}/{x}/{y}`: a Mapbox vector tile with
//! the `hits` (documents in the tile), `aggs` (geotile grid cells) and `meta`
//! layers, as Kibana Maps requests it.
//!
//! The tile is built from two internal searches (hits, grid aggregation) and
//! encoded as protobuf (`vector_tile.proto`, version 2) by [`Tile`].

use crate::{blocking, Req, Resp, RespBody, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ResolveOpts};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::sync::Arc;

pub async fn handle(node: Arc<Node>, req: Req, index: String) -> R {
    if req.method != Method::GET && req.method != Method::POST {
        return crate::method_not_allowed(&req, &["GET", "POST"]);
    }
    blocking(move || tile(&node, &req, &index)).await
}

// ----------------------------------------------------------------- protobuf

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_tag(out: &mut Vec<u8>, field: u32, wire: u8) {
    put_varint(out, ((field as u64) << 3) | wire as u64);
}

fn put_bytes(out: &mut Vec<u8>, field: u32, b: &[u8]) {
    put_tag(out, field, 2);
    put_varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// A scalar feature property.
#[derive(Clone, PartialEq)]
enum Prop {
    Str(String),
    Int(i64),
    Dbl(f64),
    Bool(bool),
}

impl Prop {
    fn from_json(v: &Value) -> Option<Prop> {
        match v {
            Value::String(s) => Some(Prop::Str(s.clone())),
            Value::Bool(b) => Some(Prop::Bool(*b)),
            Value::Number(n) => Some(match n.as_i64() {
                Some(i) => Prop::Int(i),
                None => Prop::Dbl(n.as_f64()?),
            }),
            Value::Array(a) if a.len() == 1 => Prop::from_json(&a[0]),
            Value::Array(_) | Value::Object(_) => Some(Prop::Str(v.to_string())),
            Value::Null => None,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Prop::Str(s) => put_bytes(&mut out, 1, s.as_bytes()),
            Prop::Dbl(d) => {
                put_tag(&mut out, 3, 1);
                out.extend_from_slice(&d.to_le_bytes());
            }
            Prop::Int(i) => {
                put_tag(&mut out, 4, 0);
                put_varint(&mut out, *i as u64);
            }
            Prop::Bool(b) => {
                put_tag(&mut out, 7, 0);
                put_varint(&mut out, *b as u64);
            }
        }
        out
    }
}

#[derive(Clone, Copy, PartialEq)]
enum GeomType {
    Point = 1,
    Line = 2,
    Polygon = 3,
}

struct Feature {
    ty: GeomType,
    geometry: Vec<u32>,
    props: Vec<(String, Prop)>,
}

struct Layer {
    name: &'static str,
    extent: u32,
    features: Vec<Feature>,
}

impl Layer {
    fn encode(&self) -> Vec<u8> {
        let mut keys: Vec<String> = Vec::new();
        let mut values: Vec<Vec<u8>> = Vec::new();
        let mut feats: Vec<Vec<u8>> = Vec::new();
        for f in &self.features {
            let mut tags: Vec<u8> = Vec::new();
            for (k, v) in &f.props {
                let ki = match keys.iter().position(|x| x == k) {
                    Some(i) => i,
                    None => {
                        keys.push(k.clone());
                        keys.len() - 1
                    }
                };
                let ve = v.encode();
                let vi = match values.iter().position(|x| *x == ve) {
                    Some(i) => i,
                    None => {
                        values.push(ve);
                        values.len() - 1
                    }
                };
                put_varint(&mut tags, ki as u64);
                put_varint(&mut tags, vi as u64);
            }
            let mut fe = Vec::new();
            if !tags.is_empty() {
                put_bytes(&mut fe, 2, &tags);
            }
            put_tag(&mut fe, 3, 0);
            put_varint(&mut fe, f.ty as u64);
            let mut g = Vec::new();
            for c in &f.geometry {
                put_varint(&mut g, *c as u64);
            }
            put_bytes(&mut fe, 4, &g);
            feats.push(fe);
        }
        let mut out = Vec::new();
        put_bytes(&mut out, 1, self.name.as_bytes());
        for f in &feats {
            put_bytes(&mut out, 2, f);
        }
        for k in &keys {
            put_bytes(&mut out, 3, k.as_bytes());
        }
        for v in &values {
            put_bytes(&mut out, 4, v);
        }
        put_tag(&mut out, 5, 0);
        put_varint(&mut out, self.extent as u64);
        put_tag(&mut out, 15, 0);
        put_varint(&mut out, 2);
        out
    }
}

// ------------------------------------------------------------------ geometry

/// Command integer: `id | count << 3`.
fn cmd(id: u32, count: usize) -> u32 {
    id | ((count as u32) << 3)
}

fn point_geometry(pts: &[(i64, i64)]) -> Vec<u32> {
    let mut g = vec![cmd(1, pts.len())];
    let (mut cx, mut cy) = (0i64, 0i64);
    for (x, y) in pts {
        g.push(zigzag(x - cx) as u32);
        g.push(zigzag(y - cy) as u32);
        cx = *x;
        cy = *y;
    }
    g
}

/// Encode rings/lines of tile coordinates; `closed` rings get `ClosePath`.
fn path_geometry(paths: &[Vec<(i64, i64)>], closed: bool) -> Vec<u32> {
    let mut g = Vec::new();
    let (mut cx, mut cy) = (0i64, 0i64);
    for p in paths {
        let mut p: Vec<(i64, i64)> = p.clone();
        if closed && p.len() > 1 && p.first() == p.last() {
            p.pop();
        }
        p.dedup();
        if p.len() < if closed { 3 } else { 2 } {
            continue;
        }
        g.push(cmd(1, 1));
        g.push(zigzag(p[0].0 - cx) as u32);
        g.push(zigzag(p[0].1 - cy) as u32);
        cx = p[0].0;
        cy = p[0].1;
        g.push(cmd(2, p.len() - 1));
        for (x, y) in &p[1..] {
            g.push(zigzag(x - cx) as u32);
            g.push(zigzag(y - cy) as u32);
            cx = *x;
            cy = *y;
        }
        if closed {
            g.push(cmd(7, 1));
        }
    }
    g
}

/// Twice the signed (shoelace) area; MVT exterior rings are positive.
fn area2(ring: &[(i64, i64)]) -> i64 {
    let n = ring.len();
    (0..n).map(|i| ring[i].0 * ring[(i + 1) % n].1 - ring[(i + 1) % n].0 * ring[i].1).sum()
}

/// A Web Mercator tile and the lon/lat to tile-pixel projection.
#[derive(Clone, Copy)]
struct TileCoord {
    z: u32,
    x: u32,
    y: u32,
    extent: u32,
}

impl TileCoord {
    /// Tile fractions `(fx, fy)` of a lon/lat.
    fn frac(&self, lon: f64, lat: f64) -> (f64, f64) {
        let n = (1u64 << self.z) as f64;
        let lat = lat.clamp(-85.05112878, 85.05112878).to_radians();
        let fx = (lon + 180.0) / 360.0 * n - self.x as f64;
        let fy = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / PI) / 2.0 * n - self.y as f64;
        (fx, fy)
    }

    fn pixel(&self, lon: f64, lat: f64) -> (i64, i64) {
        let (fx, fy) = self.frac(lon, lat);
        let e = self.extent as f64;
        ((fx * e).round() as i64, (fy * e).round() as i64)
    }

    /// lon/lat of a tile-space fraction.
    fn lonlat(z: u32, tx: f64, ty: f64) -> (f64, f64) {
        let n = (1u64 << z) as f64;
        let lon = tx / n * 360.0 - 180.0;
        let lat = (PI * (1.0 - 2.0 * ty / n)).sinh().atan().to_degrees();
        (lon, lat)
    }

    /// Bounds (west, south, east, north) of the tile grown by `buffer` pixels.
    fn bounds(&self, buffer: f64) -> (f64, f64, f64, f64) {
        let b = buffer / self.extent as f64;
        let n = (1u64 << self.z) as f64;
        let (x0, x1) = ((self.x as f64 - b).max(0.0), (self.x as f64 + 1.0 + b).min(n));
        let (y0, y1) = ((self.y as f64 - b).max(0.0), (self.y as f64 + 1.0 + b).min(n));
        let (w, n_) = Self::lonlat(self.z, x0, y0);
        let (e, s) = Self::lonlat(self.z, x1, y1);
        (w, s, e, n_)
    }
}

/// Pixels of a geotile cell `z/x/y` as `(left, top, right, bottom)` in the
/// tile's coordinate space.
fn cell_pixels(t: &TileCoord, cz: u32, cx: u64, cy: u64) -> (i64, i64, i64, i64) {
    let scale = (1u64 << (cz - t.z)) as f64;
    let e = t.extent as f64;
    let l = ((cx as f64 / scale - t.x as f64) * e).round() as i64;
    let r = (((cx + 1) as f64 / scale - t.x as f64) * e).round() as i64;
    let tp = ((cy as f64 / scale - t.y as f64) * e).round() as i64;
    let b = (((cy + 1) as f64 / scale - t.y as f64) * e).round() as i64;
    (l, tp, r, b)
}

/// GeoJSON -> MVT feature geometry in tile pixels.
fn geojson_feature(t: &TileCoord, g: &Value) -> Option<(GeomType, Vec<u32>)> {
    let ty = g.get("type")?.as_str()?;
    let c = g.get("coordinates");
    let pt = |v: &Value| -> Option<(i64, i64)> {
        let a = v.as_array()?;
        Some(t.pixel(a.first()?.as_f64()?, a.get(1)?.as_f64()?))
    };
    let line = |v: &Value| -> Option<Vec<(i64, i64)>> { v.as_array()?.iter().map(&pt).collect() };
    let polygon = |v: &Value| -> Option<Vec<Vec<(i64, i64)>>> {
        let mut rings: Vec<Vec<(i64, i64)>> = v.as_array()?.iter().map(&line).collect::<Option<_>>()?;
        for (i, r) in rings.iter_mut().enumerate() {
            let mut open = r.clone();
            if open.len() > 1 && open.first() == open.last() {
                open.pop();
            }
            // Exterior rings have a positive area, holes a negative one.
            if (area2(&open) > 0) != (i == 0) {
                r.reverse();
            }
        }
        Some(rings)
    };
    match ty {
        "Point" => Some((GeomType::Point, point_geometry(&[pt(c?)?]))),
        "MultiPoint" => Some((GeomType::Point, point_geometry(&c?.as_array()?.iter().map(&pt).collect::<Option<Vec<_>>>()?))),
        "LineString" => Some((GeomType::Line, path_geometry(&[line(c?)?], false))),
        "MultiLineString" => Some((GeomType::Line, path_geometry(&c?.as_array()?.iter().map(&line).collect::<Option<Vec<_>>>()?, false))),
        "Polygon" => Some((GeomType::Polygon, path_geometry(&polygon(c?)?, true))),
        "MultiPolygon" => {
            let mut rings = Vec::new();
            for p in c?.as_array()? {
                rings.extend(polygon(p)?);
            }
            Some((GeomType::Polygon, path_geometry(&rings, true)))
        }
        // Collections are flattened by `flatten_geometries` first.
        _ => None,
    }
}

/// `_shards` of a tile from its hits and grid searches: the larger failure
/// and skip counts, the smaller success count.
fn merge_shards(a: &Value, b: &Value) -> Value {
    let n = |v: &Value, k: &str| v[k].as_i64().unwrap_or(0);
    json!({
        "total": n(a, "total").max(n(b, "total")),
        "successful": n(a, "successful").min(n(b, "successful")),
        "skipped": n(a, "skipped").max(n(b, "skipped")),
        "failed": n(a, "failed").max(n(b, "failed")),
    })
}

/// Every geometry of a fetched field value list, collection members
/// included (ES's MVT formatter: one feature per value and member).
fn flatten_geometries(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flatten_geometries(x, out)),
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("GeometryCollection") => {
            o.get("geometries").into_iter().for_each(|g| flatten_geometries(g, out));
        }
        Value::Object(_) => out.push(v.clone()),
        _ => {}
    }
}

/// A representative label point (tile pixels) of a GeoJSON geometry.
fn collect_coords(v: &Value, out: &mut Vec<(f64, f64)>) {
    if let Some(a) = v.as_array() {
        if a.len() >= 2 && a[0].is_number() {
            if let (Some(x), Some(y)) = (a[0].as_f64(), a[1].as_f64()) {
                out.push((x, y));
            }
        } else {
            a.iter().for_each(|x| collect_coords(x, out));
        }
    }
}

/// Whether geometry `g`'s pixel bounding box meets the tile grown by
/// `buffer` pixels (ES clips each feature to it: one wholly outside is
/// no feature).
fn touches_tile(t: &TileCoord, g: &Value, buffer: i64) -> bool {
    let mut pts = Vec::new();
    if let Some(c) = g.get("coordinates") {
        collect_coords(c, &mut pts);
    }
    let px: Vec<(i64, i64)> = pts.iter().map(|p| t.pixel(p.0, p.1)).collect();
    let (Some(x0), Some(x1)) = (px.iter().map(|p| p.0).min(), px.iter().map(|p| p.0).max()) else { return false };
    let (y0, y1) = (px.iter().map(|p| p.1).min().unwrap_or(0), px.iter().map(|p| p.1).max().unwrap_or(0));
    let (lo, hi) = (-buffer, t.extent as i64 + buffer);
    x1 >= lo && x0 <= hi && y1 >= lo && y0 <= hi
}

fn label_point(t: &TileCoord, g: &Value) -> Option<(i64, i64)> {
    let mut pts = Vec::new();
    collect_coords(g.get("coordinates")?, &mut pts);
    if pts.is_empty() {
        return None;
    }
    let (minx, maxx) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (miny, maxy) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    Some(t.pixel((minx + maxx) / 2.0, (miny + maxy) / 2.0))
}

// --------------------------------------------------------------------- tile

fn bad(reason: String) -> EsError {
    EsError::illegal_argument(reason)
}

fn parse_err(req: &Req, key: &str, reason: String) -> EsError {
    // `[line:col]` points just past the offending value in a single-line body.
    let body = String::from_utf8_lossy(&req.body);
    let col = body
        .find(&format!("\"{key}\""))
        .and_then(|i| body[i..].find(':').map(|j| i + j + 1))
        .map(|i| {
            let rest = &body[i..];
            let ws = rest.len() - rest.trim_start().len();
            let val = rest.trim_start();
            let len = val.find(|c: char| c == ',' || c == '}' || c == ']').unwrap_or(val.len());
            i + ws + len
        })
        .unwrap_or(1);
    EsError::new(400, "x_content_parse_exception", format!("[1:{col}] [vector-tile] failed to parse field [{key}]")).caused_by(EsError::illegal_argument(reason))
}

/// ES `RestVectorTileAction`: the grid aggregation's `size` is
/// `MultiBucketConsumerService.DEFAULT_MAX_BUCKETS` (65,536), so every
/// cell of a zoom-0 tile at grid precision 8 (4^8 cells) is returned.
const MVT_GRID_SIZE: usize = 65_536;

fn uint_param(body: &Value, req: &Req, key: &str, default: i64, min: i64, max: Option<i64>, name: &str) -> Result<i64, EsError> {
    // ES `VectorTileRequest.parseRestRequest`: a URL parameter overrides the
    // body and is checked by the setter itself (`illegal_argument_exception`,
    // not a body parse failure).
    if let Some(s) = req.params.get(key) {
        let n: i64 = s.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [{key}] with value [{s}]")))?;
        if n < min {
            return Err(EsError::illegal_argument(format!("[{name}] parameter cannot be negative, found [{n}]")));
        }
        if let Some(m) = max.filter(|m| n > *m) {
            return Err(EsError::illegal_argument(format!("[{name}] parameter cannot be bigger than {m}, found [{n}]")));
        }
        return Ok(n);
    }
    let Some(v) = body.get(key).cloned() else { return Ok(default) };
    let Some(n) = v.as_i64() else {
        return Err(parse_err(req, key, format!("failed to parse [{key}] value [{v}]")));
    };
    if n < min {
        return Err(parse_err(req, key, format!("[{name}] parameter cannot be negative, found [{n}]")));
    }
    if let Some(m) = max {
        if n > m {
            return Err(parse_err(req, key, format!("[{name}] parameter cannot be bigger than {m}, found [{n}]")));
        }
    }
    Ok(n)
}

fn tile(node: &Node, req: &Req, index: &str) -> R {
    let seg = |i: usize| req.seg(i).to_string();
    let field = seg(2);
    let parse = |s: String, what: &str| s.parse::<i64>().map_err(|_| bad(format!("failed to parse [{what}] value [{s}]")));
    let (z, x, y) = (parse(seg(3), "zoom")?, parse(seg(4), "x")?, parse(seg(5), "y")?);
    if !(0..=29).contains(&z) || x < 0 || y < 0 || x >= (1i64 << z) || y >= (1i64 << z) {
        return Err(bad(format!("Zoom/X/Y combination is not valid: {z}/{x}/{y}")));
    }
    let body = req.json()?;
    let body = if body.is_null() { json!({}) } else { body };
    if let Some(o) = body.as_object() {
        for k in o.keys() {
            const KNOWN: &[&str] = &["aggs", "buffer", "exact_bounds", "extent", "fields", "grid_agg", "grid_precision", "grid_type", "query", "runtime_mappings", "size", "sort", "track_total_hits", "with_labels"];
            if !KNOWN.contains(&k.as_str()) {
                return Err(EsError::new(400, "x_content_parse_exception", format!("[1:2] [vector-tile] unknown field [{k}]")));
            }
        }
    }
    let grid_precision = uint_param(&body, req, "grid_precision", 8, 0, Some(8), "gridPrecision")?;
    let size = uint_param(&body, req, "size", 10000, 0, None, "size")?;
    let extent = uint_param(&body, req, "extent", 4096, 0, None, "extent")?.max(1) as u32;
    let buffer = uint_param(&body, req, "buffer", 5, 0, None, "buffer")?;
    let flag = |k: &str| body.get(k).and_then(Value::as_bool).or_else(|| req.params.get(k).map(|v| v == "true" || v.is_empty())).unwrap_or(false);
    let (exact_bounds, with_labels) = (flag("exact_bounds"), flag("with_labels"));
    let grid_type = body.get("grid_type").and_then(Value::as_str).map(String::from).or_else(|| req.params.get("grid_type").cloned()).unwrap_or_else(|| "grid".into());
    if !matches!(grid_type.as_str(), "grid" | "point" | "centroid") {
        return Err(parse_err(req, "grid_type", format!("Unknown grid type [{grid_type}]")));
    }
    let grid_agg = body.get("grid_agg").and_then(Value::as_str).unwrap_or("geotile").to_string();
    if !matches!(grid_agg.as_str(), "geotile" | "geohex") {
        return Err(parse_err(req, "grid_agg", format!("Unknown grid agg [{grid_agg}]")));
    }
    let mut fields: Vec<String> = match body.get("fields") {
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        Some(Value::String(s)) => s.split(',').map(String::from).collect(),
        _ => req.params.get("fields").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default(),
    };
    fields.retain(|f| !f.is_empty());

    let t = TileCoord { z: z as u32, x: x as u32, y: y as u32, extent };
    let (w, s, e, n) = t.bounds(buffer as f64);
    let bbox = json!({"geo_shape": {&field: {"shape": {"type": "envelope", "coordinates": [[w, n], [e, s]]}, "relation": "intersects"}}});
    let user_query = body.get("query").cloned().unwrap_or_else(|| json!({"match_all": {}}));
    let query = json!({"bool": {"filter": [user_query, bbox]}});
    let params: HashMap<String, String> = HashMap::new();
    let resolve = ResolveOpts::from_params(&req.params);
    let started = std::time::Instant::now();

    let mut layers: Vec<Layer> = Vec::new();
    let mut hits_total: (i64, String) = (0, "eq".into());
    let mut shards = json!({"total": 1, "successful": 1, "skipped": 0, "failed": 0});
    let mut timed_out = false;

    // ---- hits layer
    let mut want_fields = vec![field.clone()];
    want_fields.extend(fields.iter().cloned());
    let mut hits_body = json!({"size": size, "query": query.clone(), "_source": false, "fields": want_fields, "track_total_hits": body.get("track_total_hits").cloned().unwrap_or(json!(true))});
    for k in ["sort", "runtime_mappings"] {
        if let Some(v) = body.get(k) {
            hits_body[k] = v.clone();
        }
    }
    // On a cluster the searches run through the coordinator (every shard);
    // the local engine holds only this node's copies.
    let out = crate::search_api::coordinated_search(node, req, &hits_body, &params, index, resolve.clone(), req.access.as_deref())?;
    let resp: Value = serde_json::from_str(&out).map_err(|e| EsError::internal(format!("bad search output: {e}")))?;
    shards = resp.get("_shards").cloned().unwrap_or(shards);
    timed_out |= resp["timed_out"].as_bool().unwrap_or(false);
    if let Some(tot) = resp["hits"]["total"].as_object() {
        hits_total = (tot.get("value").and_then(Value::as_i64).unwrap_or(0), tot.get("relation").and_then(Value::as_str).unwrap_or("eq").to_string());
    }
    let mut hit_features: Vec<Feature> = Vec::new();
    let mut hit_bounds: Option<(i64, i64, i64, i64)> = None;
    for h in resp["hits"]["hits"].as_array().cloned().unwrap_or_default() {
        let flds = h.get("fields").cloned().unwrap_or(Value::Null);
        let Some(values) = flds.get(&field) else { continue };
        let mut geoms = Vec::new();
        flatten_geometries(values, &mut geoms);
        let mut props = vec![("_id".to_string(), Prop::Str(h["_id"].as_str().unwrap_or("").into())), ("_index".to_string(), Prop::Str(h["_index"].as_str().unwrap_or("").into()))];
        for f in &fields {
            if let Some(p) = flds.get(f).and_then(Prop::from_json) {
                props.push((f.clone(), p));
            }
        }
        // ES `buildHitsLayer`: a feature per geometry, then one label.
        let mut label: Option<(i64, i64)> = None;
        for geom in &geoms {
            // A value or member wholly outside the tile is no feature;
            // with several, one inside still shows (ES clips each).
            if geoms.len() > 1 && !touches_tile(&t, geom, buffer) {
                continue;
            }
            let Some((ty, g)) = geojson_feature(&t, geom) else { continue };
            if g.is_empty() {
                continue;
            }
            if let Some(lp) = label_point(&t, geom) {
                hit_bounds = Some(match hit_bounds {
                    Some(b) => (b.0.min(lp.0), b.1.min(lp.1), b.2.max(lp.0), b.3.max(lp.1)),
                    None => (lp.0, lp.1, lp.0, lp.1),
                });
                label.get_or_insert(lp);
            }
            hit_features.push(Feature { ty, geometry: g, props: props.clone() });
        }
        if let (true, Some(lp)) = (with_labels, label) {
            let mut props = props;
            props.push(("_mvt_label_position".into(), Prop::Bool(true)));
            hit_features.push(Feature { ty: GeomType::Point, geometry: point_geometry(&[lp]), props });
        }
    }
    if !hit_features.is_empty() {
        layers.push(Layer { name: "hits", extent, features: hit_features });
    }

    // ---- aggs layer
    let mut count_stats: Option<(i64, f64, f64, f64)> = None;
    let mut agg_stats: BTreeMap<String, (i64, f64, f64, f64)> = BTreeMap::new();
    let mut agg_bounds: Option<(i64, i64, i64, i64)> = None;
    if grid_precision > 0 {
        let hex = grid_agg == "geohex";
        let precision = if hex { elasticsearch_engine::fieldtypes::geo::h3_resolution_for_zoom(z as u32, grid_precision as u32) } else { ((z + grid_precision) as u32).min(29) };
        let mut sub = body.get("aggs").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
        if grid_type == "centroid" {
            sub["_mvt_centroid"] = json!({"geo_centroid": {"field": field}});
        }
        let grid = if grid_agg == "geohex" { "geohex_grid" } else { "geotile_grid" };
        let mut aggs_body = json!({"size": 0, "query": query.clone(), "track_total_hits": false, "aggs": {"_mvt_grid": {grid: {"field": field, "precision": precision, "size": MVT_GRID_SIZE}, "aggs": sub}}});
        // ES (`RestVectorTileAction`) bounds the grid to the tile
        // (`setGeoBoundingBox`): the tilers only produce cells inside it
        // (geotile cells align with the tile; geohex cells touching it).
        // Unbounded, a high-zoom tile over large shapes tiled them across
        // the whole world, and the `size` cut could drop cells of this tile.
        let (w, s_, e, n_) = t.bounds(0.0);
        aggs_body["aggs"]["_mvt_grid"][grid]["bounds"] = json!({"top_left": {"lat": n_, "lon": w}, "bottom_right": {"lat": s_, "lon": e}});
        if let Some(rm) = body.get("runtime_mappings") {
            aggs_body["runtime_mappings"] = rm.clone();
        }
        let out = crate::search_api::coordinated_search(node, req, &aggs_body, &params, index, resolve, req.access.as_deref())?;
        let resp: Value = serde_json::from_str(&out).map_err(|e| EsError::internal(format!("bad search output: {e}")))?;
        timed_out |= resp["timed_out"].as_bool().unwrap_or(false);
        // The grid's shard failures are the tile's too: its meta layer
        // reports the less complete of the two searches.
        if let Some(agg_shards) = resp.get("_shards") {
            shards = merge_shards(&shards, agg_shards);
        }
        let mut feats: Vec<Feature> = Vec::new();
        let mut stats_acc: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut counts: Vec<f64> = Vec::new();
        for b in resp["aggregations"]["_mvt_grid"]["buckets"].as_array().cloned().unwrap_or_default() {
            let key = b["key"].as_str().unwrap_or("");
            // geohex cells: the H3 polygon in tile pixels, labelled at its center.
            let hex_cell = if hex {
                match (elasticsearch_engine::fieldtypes::geo::h3_boundary(key), elasticsearch_engine::fieldtypes::geo::h3_center(key)) {
                    (Some(ring), Some(c)) => Some((ring.iter().map(|p| t.pixel(p.0, p.1)).collect::<Vec<_>>(), t.pixel(c.0, c.1))),
                    _ => continue,
                }
            } else {
                None
            };
            let parts: Vec<u64> = key.split('/').filter_map(|p| p.parse().ok()).collect();
            if hex_cell.is_none() && parts.len() != 3 {
                continue;
            }
            let (cz, cx, cy) = if hex_cell.is_some() { (t.z, t.x as u64, t.y as u64) } else { (parts[0] as u32, parts[1], parts[2]) };
            // Only cells inside this tile.
            if cz < t.z || (cx >> (cz - t.z)) != t.x as u64 || (cy >> (cz - t.z)) != t.y as u64 {
                continue;
            }
            let count = b["doc_count"].as_i64().unwrap_or(0);
            let mut props = vec![("_key".to_string(), Prop::Str(key.to_string())), ("_count".to_string(), Prop::Int(count))];
            counts.push(count as f64);
            if let Some(o) = b.as_object() {
                for (name, v) in o {
                    if matches!(name.as_str(), "key" | "doc_count" | "key_as_string" | "_mvt_centroid") {
                        continue;
                    }
                    if let Some(vo) = v.as_object() {
                        for (k2, v2) in vo {
                            if let Some(p) = Prop::from_json(v2).filter(|p| !matches!(p, Prop::Str(_)) || k2 == "value_as_string") {
                                props.push((format!("{name}.{k2}"), p));
                            }
                        }
                        if let Some(val) = vo.get("value").and_then(Value::as_f64) {
                            stats_acc.entry(name.clone()).or_default().push(val);
                        }
                    }
                }
            }
            let (l, tp, r, bt) = match &hex_cell {
                Some((ring, _)) => ring.iter().fold((i64::MAX, i64::MAX, i64::MIN, i64::MIN), |a, p| (a.0.min(p.0), a.1.min(p.1), a.2.max(p.0), a.3.max(p.1))),
                None => cell_pixels(&t, cz, cx, cy),
            };
            let (ty, geom, lp) = match grid_type.as_str() {
                "grid" if hex_cell.is_some() => {
                    let (ring, c) = hex_cell.clone().unwrap();
                    let mut ring = ring;
                    if area2(&ring) <= 0 {
                        ring.reverse();
                    }
                    (GeomType::Polygon, path_geometry(&[ring], true), c)
                }
                "point" if hex_cell.is_some() => {
                    let c = hex_cell.as_ref().unwrap().1;
                    (GeomType::Point, point_geometry(&[c]), c)
                }
                "point" => (GeomType::Point, point_geometry(&[((l + r) / 2, (tp + bt) / 2)]), ((l + r) / 2, (tp + bt) / 2)),
                "centroid" => {
                    let c = &b["_mvt_centroid"]["location"];
                    match (c["lon"].as_f64(), c["lat"].as_f64()) {
                        (Some(lon), Some(lat)) => {
                            let p = t.pixel(lon, lat);
                            (GeomType::Point, point_geometry(&[p]), p)
                        }
                        _ => continue,
                    }
                }
                _ => (GeomType::Polygon, path_geometry(&[vec![(l, bt), (l, tp), (r, tp), (r, bt)]], true), ((l + r) / 2, (tp + bt) / 2)),
            };
            agg_bounds = Some(match agg_bounds {
                Some(bb) => (bb.0.min(l), bb.1.min(tp), bb.2.max(r), bb.3.max(bt)),
                None => (l, tp, r, bt),
            });
            if with_labels && ty == GeomType::Polygon {
                feats.push(Feature { ty, geometry: geom, props: props.clone() });
                props.push(("_mvt_label_position".into(), Prop::Bool(true)));
                feats.push(Feature { ty: GeomType::Point, geometry: point_geometry(&[lp]), props });
            } else {
                feats.push(Feature { ty, geometry: geom, props });
            }
        }
        let stat = |v: &[f64]| (v.len() as i64, v.iter().sum::<f64>(), v.iter().cloned().fold(f64::MAX, f64::min), v.iter().cloned().fold(f64::MIN, f64::max));
        count_stats = Some(stat(&counts));
        for (name, v) in stats_acc {
            agg_stats.insert(name, stat(&v));
        }
        if !feats.is_empty() {
            layers.push(Layer { name: "aggs", extent, features: feats });
        }
    }

    // ---- meta layer
    let mut meta: BTreeMap<String, Prop> = BTreeMap::new();
    for k in ["total", "successful", "skipped", "failed"] {
        meta.insert(format!("_shards.{k}"), Prop::Int(shards[k].as_i64().unwrap_or(0)));
    }
    meta.insert("hits.total.relation".into(), Prop::Str(hits_total.1));
    meta.insert("hits.total.value".into(), Prop::Int(hits_total.0));
    meta.insert("timed_out".into(), Prop::Bool(timed_out));
    meta.insert("took".into(), Prop::Int(started.elapsed().as_millis() as i64));
    let mut put_stats = |name: &str, s: (i64, f64, f64, f64)| {
        meta.insert(format!("aggregations.{name}.count"), Prop::Int(s.0));
        meta.insert(format!("aggregations.{name}.sum"), Prop::Dbl(s.1));
        if s.0 > 0 {
            meta.insert(format!("aggregations.{name}.avg"), Prop::Dbl(s.1 / s.0 as f64));
            meta.insert(format!("aggregations.{name}.min"), Prop::Dbl(s.2));
            meta.insert(format!("aggregations.{name}.max"), Prop::Dbl(s.3));
        }
    };
    if let Some(s) = count_stats {
        put_stats("_count", s);
    }
    for (name, s) in &agg_stats {
        put_stats(name, *s);
    }
    let ext = extent as i64;
    let (bx0, by0, bx1, by1) = if exact_bounds {
        let all = [hit_bounds, agg_bounds].into_iter().flatten().fold(None, |a: Option<(i64, i64, i64, i64)>, b| {
            Some(match a {
                Some(a) => (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)),
                None => b,
            })
        });
        match all {
            Some(b) => (b.0.clamp(0, ext), b.1.clamp(0, ext), b.2.clamp(0, ext), b.3.clamp(0, ext)),
            None => (0, 0, ext, ext),
        }
    } else {
        (0, 0, ext, ext)
    };
    layers.push(Layer {
        name: "meta",
        extent,
        features: vec![Feature { ty: GeomType::Polygon, geometry: path_geometry(&[vec![(bx0, by1), (bx0, by0), (bx1, by0), (bx1, by1)]], true), props: meta.into_iter().collect() }],
    });

    let mut tile_bytes = Vec::new();
    for l in &layers {
        put_bytes(&mut tile_bytes, 3, &l.encode());
    }
    Ok(Resp { status: 200, body: RespBody::Bytes(tile_bytes), headers: vec![("content-type", "application/vnd.mapbox-vector-tile".to_string())] })
}

#[cfg(test)]
mod grid_size_tests {
    use super::*;

    fn varint(b: &[u8], i: &mut usize) -> u64 {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let x = b[*i];
            *i += 1;
            v |= ((x & 0x7f) as u64) << shift;
            if x & 0x80 == 0 {
                return v;
            }
            shift += 7;
        }
    }

    /// `(field, bytes)` of the length-delimited fields of a message.
    fn fields(b: &[u8]) -> Vec<(u64, Vec<u8>)> {
        let (mut i, mut out) = (0, Vec::new());
        while i < b.len() {
            let tag = varint(b, &mut i);
            match tag & 7 {
                0 => {
                    varint(b, &mut i);
                }
                2 => {
                    let n = varint(b, &mut i) as usize;
                    out.push((tag >> 3, b[i..i + n].to_vec()));
                    i += n;
                }
                5 => i += 4,
                1 => i += 8,
                w => panic!("wire type {w}"),
            }
        }
        out
    }

    fn layer_features(tile: &[u8], name: &str) -> usize {
        fields(tile).into_iter().filter(|(f, _)| *f == 3).map(|(_, l)| fields(&l)).find(|l| l.iter().any(|(f, v)| *f == 1 && v == name.as_bytes()))
            .map(|l| l.iter().filter(|(f, _)| *f == 2).count()).unwrap_or(0)
    }

    fn req(path_tail: &str, body: Value, params: &[(&str, &str)]) -> Req {
        let segs: Vec<String> = format!("pts/_mvt/{path_tail}").split('/').map(str::to_string).collect();
        Req { method: Method::POST, path: format!("/pts/_mvt/{path_tail}"), segs, params: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            body: body.to_string().into_bytes().into(), headers: Default::default(), access: None, extensions: axum::http::Extensions::new(), api_version: crate::RestApiVersion::V9 }
    }

    /// The grid is sized as ES sizes it (65,536): one point in each zoom-8
    /// tile gives 65,536 cells of tile 0/0/0 at grid precision 8.
    #[test]
    fn zoom_zero_grid_returns_every_precision_8_cell() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("pts", &json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"p": {"type": "geo_point"}}}})).unwrap();
        let n = 256u32;
        let mut batch = Vec::new();
        for x in 0..n {
            for y in 0..n {
                let lon = (x as f64 + 0.5) / n as f64 * 360.0 - 180.0;
                let lat = (PI * (1.0 - 2.0 * (y as f64 + 0.5) / n as f64)).sinh().atan().to_degrees();
                batch.push(elasticsearch_engine::index::WriteReq::Index { id: Some(format!("{x}-{y}")), source: json!({"p": {"lat": lat, "lon": lon}}).to_string(), routing: None, cond: Default::default() });
            }
            if batch.len() >= 8192 {
                for r in svc.write(std::mem::take(&mut batch), false) {
                    r.unwrap();
                }
            }
        }
        for r in svc.write(batch, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let resp = tile(&node, &req("p/0/0/0", json!({"grid_precision": 8, "size": 0}), &[]), "pts").unwrap();
        let bytes = match resp.body {
            crate::RespBody::Bytes(b) => b,
            _ => panic!("unexpected tile body"),
        };
        assert_eq!(layer_features(&bytes, "aggs"), 65_536);
    }

    /// ES's request checks: grid precision past 8 (a body field: a parse
    /// failure; a URL parameter: an illegal argument), size past the
    /// result window.
    #[test]
    fn grid_precision_and_size_limits_match_es() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        node.create_index("pts", &json!({"mappings": {"properties": {"p": {"type": "geo_point"}}}})).unwrap();
        let e = tile(&node, &req("p/0/0/0", json!({"grid_precision": 9}), &[]), "pts").err().unwrap();
        assert_eq!((e.status, e.kind.as_str()), (400, "x_content_parse_exception"), "{e:?}");
        assert!(format!("{e:?}").contains("[gridPrecision] parameter cannot be bigger than 8, found [9]"), "{e:?}");
        let e = tile(&node, &req("p/0/0/0", json!({}), &[("grid_precision", "9")]), "pts").err().unwrap();
        assert_eq!((e.status, e.kind.as_str()), (400, "illegal_argument_exception"), "{e:?}");
        assert_eq!(e.reason, "[gridPrecision] parameter cannot be bigger than 8, found [9]");
        let e = tile(&node, &req("p/0/0/0", json!({"size": 10001}), &[]), "pts").err().unwrap();
        assert_eq!((e.status, e.kind.as_str()), (400, "search_phase_execution_exception"), "{e:?}");
    }
}

#[cfg(test)]
mod review_tests {
    use super::*;

    fn varint(b: &[u8], i: &mut usize) -> u64 {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let x = b[*i];
            *i += 1;
            v |= ((x & 0x7f) as u64) << shift;
            if x & 0x80 == 0 {
                return v;
            }
            shift += 7;
        }
    }

    /// `(field, bytes)` of the length-delimited fields of a message.
    fn fields(b: &[u8]) -> Vec<(u64, Vec<u8>)> {
        let (mut i, mut out) = (0, Vec::new());
        while i < b.len() {
            let tag = varint(b, &mut i);
            match tag & 7 {
                0 => {
                    varint(b, &mut i);
                }
                2 => {
                    let n = varint(b, &mut i) as usize;
                    out.push((tag >> 3, b[i..i + n].to_vec()));
                    i += n;
                }
                5 => i += 4,
                1 => i += 8,
                w => panic!("wire type {w}"),
            }
        }
        out
    }

    /// Features of layer `name` of a tile.
    fn layer_features(tile: &[u8], name: &str) -> usize {
        fields(tile).into_iter().filter(|(f, _)| *f == 3).map(|(_, l)| fields(&l)).find(|l| l.iter().any(|(f, v)| *f == 1 && v == name.as_bytes()))
            .map(|l| l.iter().filter(|(f, _)| *f == 2).count()).unwrap_or(0)
    }

    /// #390: every value of a multi-valued field and every member of a
    /// geometry collection inside the tile is a feature.
    #[test]
    fn tiles_keep_every_visible_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("geo", &json!({"mappings": {"properties": {"location": {"type": "geo_shape"}}}})).unwrap();
        let pt = |lon: f64, lat: f64| json!({"type": "Point", "coordinates": [lon, lat]});
        let docs = [
            // The first value outside tile 1/1/0 (lon 0..180, north), the second inside.
            json!({"location": [pt(-170.0, 10.0), pt(10.0, 10.0)]}),
            // Two members inside.
            json!({"location": {"type": "GeometryCollection", "geometries": [pt(20.0, 20.0), pt(30.0, 30.0)]}}),
            // Only the later member inside.
            json!({"location": {"type": "GeometryCollection", "geometries": [pt(-100.0, 10.0), pt(40.0, 40.0)]}}),
        ];
        let w = docs.iter().enumerate().map(|(i, d)| elasticsearch_engine::index::WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(w, false) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let req = Req { method: Method::POST, path: "/geo/_mvt/location/1/1/0".into(), segs: ["geo", "_mvt", "location", "1", "1", "0"].iter().map(|s| s.to_string()).collect(), params: Default::default(),
            body: json!({"grid_precision": 0}).to_string().into_bytes().into(), headers: Default::default(), access: None, extensions: axum::http::Extensions::new(), api_version: crate::RestApiVersion::V9 };
        let resp = tile(&node, &req, "geo").unwrap();
        let bytes: Vec<u8> = match resp.body {
            crate::RespBody::Bytes(b) => b.to_vec(),
            crate::RespBody::Raw(s) => s.into_bytes(),
            _ => panic!("unexpected tile body"),
        };
        assert_eq!(layer_features(&bytes, "hits"), 4);
    }

    /// #389: the meta layer reports the grid search's shard failures.
    #[test]
    fn tile_shards_take_the_less_complete_search() {
        let hits = json!({"total": 2, "successful": 2, "skipped": 0, "failed": 0});
        let grid = json!({"total": 2, "successful": 1, "skipped": 0, "failed": 1, "failures": []});
        assert_eq!(merge_shards(&hits, &grid), json!({"total": 2, "successful": 1, "skipped": 0, "failed": 1}));
        assert_eq!(merge_shards(&grid, &hits), json!({"total": 2, "successful": 1, "skipped": 0, "failed": 1}));
    }
}
