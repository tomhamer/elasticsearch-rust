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
        "GeometryCollection" => g.get("geometries")?.as_array()?.iter().find_map(|x| geojson_feature(t, x)),
        _ => None,
    }
}

/// A representative label point (tile pixels) of a GeoJSON geometry.
fn label_point(t: &TileCoord, g: &Value) -> Option<(i64, i64)> {
    fn collect(v: &Value, out: &mut Vec<(f64, f64)>) {
        if let Some(a) = v.as_array() {
            if a.len() >= 2 && a[0].is_number() {
                if let (Some(x), Some(y)) = (a[0].as_f64(), a[1].as_f64()) {
                    out.push((x, y));
                }
            } else {
                a.iter().for_each(|x| collect(x, out));
            }
        }
    }
    let mut pts = Vec::new();
    collect(g.get("coordinates")?, &mut pts);
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

fn uint_param(body: &Value, req: &Req, key: &str, default: i64, min: i64, max: Option<i64>, name: &str) -> Result<i64, EsError> {
    let v = body.get(key).cloned().or_else(|| req.params.get(key).map(|s| s.parse::<i64>().map(Value::from).unwrap_or(Value::String(s.clone()))));
    let Some(v) = v else { return Ok(default) };
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
        let Some(geom) = flds.get(&field).and_then(Value::as_array).and_then(|a| a.first()).cloned() else { continue };
        let Some((ty, g)) = geojson_feature(&t, &geom) else { continue };
        if g.is_empty() {
            continue;
        }
        let mut props = vec![("_id".to_string(), Prop::Str(h["_id"].as_str().unwrap_or("").into())), ("_index".to_string(), Prop::Str(h["_index"].as_str().unwrap_or("").into()))];
        for f in &fields {
            if let Some(p) = flds.get(f).and_then(Prop::from_json) {
                props.push((f.clone(), p));
            }
        }
        if let Some(lp) = label_point(&t, &geom) {
            hit_bounds = Some(match hit_bounds {
                Some(b) => (b.0.min(lp.0), b.1.min(lp.1), b.2.max(lp.0), b.3.max(lp.1)),
                None => (lp.0, lp.1, lp.0, lp.1),
            });
            if with_labels {
                hit_features.push(Feature { ty, geometry: g, props: props.clone() });
                props.push(("_mvt_label_position".into(), Prop::Bool(true)));
                hit_features.push(Feature { ty: GeomType::Point, geometry: point_geometry(&[lp]), props });
                continue;
            }
        }
        hit_features.push(Feature { ty, geometry: g, props });
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
        let mut aggs_body = json!({"size": 0, "query": query.clone(), "track_total_hits": false, "aggs": {"_mvt_grid": {grid: {"field": field, "precision": precision, "size": 65535}, "aggs": sub}}});
        if hex {
            // ES bounds the geohex grid to the tile: cells touching it.
            let (w, s_, e, n_) = t.bounds(0.0);
            aggs_body["aggs"]["_mvt_grid"][grid]["bounds"] = json!({"top_left": {"lat": n_, "lon": w}, "bottom_right": {"lat": s_, "lon": e}});
        }
        if let Some(rm) = body.get("runtime_mappings") {
            aggs_body["runtime_mappings"] = rm.clone();
        }
        let out = crate::search_api::coordinated_search(node, req, &aggs_body, &params, index, resolve, req.access.as_deref())?;
        let resp: Value = serde_json::from_str(&out).map_err(|e| EsError::internal(format!("bad search output: {e}")))?;
        timed_out |= resp["timed_out"].as_bool().unwrap_or(false);
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
