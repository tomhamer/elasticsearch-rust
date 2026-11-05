//! Block-join and geo aggregations: `nested`, `reverse_nested`,
//! `geo_distance`, `geo_bounds`, `geo_centroid`, `geotile_grid`,
//! `geohash_grid`.
//!
//! `nested` / `reverse_nested` move the doc set between levels of the
//! nested-document blocks (core [`LeafReader::nested_children`] /
//! [`LeafReader::nested_ancestor`]); their sub-aggregations then run over
//! the child (or parent) docs like any other doc set. Geo aggregations read
//! the core geo doc values of `geo_point` fields.

use super::{count, field_name, jfmt, Ctx, Docs};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use crate::values;
use rustic_core::geo;
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub(super) const GEO_NESTED: &[&str] = &[
    "nested",
    "reverse_nested",
    "geo_distance",
    "geo_bounds",
    "geo_centroid",
    "geotile_grid",
    "geohash_grid",
    "geohex_grid",
    "geo_line",
    "cartesian_bounds",
    "cartesian_centroid",
];

impl<'a, 'b> Ctx<'a, 'b> {
    pub(super) fn geo_nested_agg(&self, name: &str, t: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        match t {
            "nested" => self.nested_agg(name, o, sub, docs),
            "reverse_nested" => self.reverse_nested_agg(o, sub, docs),
            "geo_bounds" => self.geo_bounds(name, o, docs),
            "geo_centroid" => self.geo_centroid(name, o, docs),
            "geo_distance" => self.geo_distance_agg(name, o, sub, docs),
            "geotile_grid" | "geohash_grid" | "geohex_grid" => self.geo_grid(name, t, o, sub, docs),
            "geo_line" => self.geo_line(name, o, docs),
            "cartesian_bounds" | "cartesian_centroid" => self.cartesian_agg(name, t, o, docs),
            _ => Err(EsError::parsing_at(format!("Unknown aggregation type [{t}]"), 1, 1)),
        }
    }

    // -----------------------------------------------------------------------
    // nested / reverse_nested

    fn nested_agg(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let path = o.get("path").and_then(Value::as_str).ok_or_else(|| EsError::parsing_at(format!("Missing [path] field for nested aggregation [{name}]"), 1, 1))?;
        for s in self.segs {
            match s.lookup.objects.get(path) {
                Some(info) if info.nested => {}
                Some(_) => return Err(EsError::illegal_argument(format!("[nested] nested path [{path}] is not nested"))),
                None => {}
            }
        }
        // Count-only date histograms under every live root (Rally
        // `nested-date-histo`): a segment without deletes whose parents are
        // all its roots has every doc of the path as children; its list is
        // left empty and marked, and the histogram counts it from the
        // field's points index (or lists the path's docs itself).
        let whole: Vec<Option<(u16, u64)>> = if whole_path_subs(sub) {
            (0..docs.len()).map(|si| self.whole_path_children(si, path, docs)).collect()
        } else {
            vec![None; docs.len()]
        };
        let d: Docs = docs
            .iter()
            .enumerate()
            .map(|(si, v)| {
                if whole[si].is_some() {
                    return Vec::new();
                }
                let leaf = &self.segs[si].leaf;
                let Some(n) = leaf.nested_info() else { return Vec::new() };
                let Some(ord) = n.path_ord(path) else { return Vec::new() };
                // Live children of every parent, appended in place.
                let deleted = leaf.deletions_only();
                // Many root parents (a `match_all`): one forward pass over the
                // segment's path ordinals instead of a block walk per parent.
                if v.len() as u64 * 8 >= leaf.num_docs() as u64 {
                    if let Some(out) = root_children_scan(n, ord, leaf.num_docs(), v, deleted.as_deref()) {
                        return out;
                    }
                }
                let mut out = Vec::with_capacity(v.len());
                for &(p, sc) in v {
                    n.for_each_child(p, ord, |c| {
                        if deleted.as_ref().is_none_or(|b| !b.get(c)) {
                            out.push((c, sc));
                        }
                    });
                }
                out
            })
            .collect();
        // The nested docs' list (ES charges the child doc set the same way).
        self.charge(name, d.iter().map(|v| v.len() as u64 * 8 + 24).sum())?;
        if whole.iter().all(Option::is_none) {
            return self.single_bucket(d, sub);
        }
        let doc_count: u64 = d.iter().map(|v| v.len() as u64).sum::<u64>() + whole.iter().flatten().map(|w| w.1).sum::<u64>();
        let mut m = Map::new();
        m.insert("doc_count".into(), json!(doc_count));
        if let Some(sub) = sub {
            let _marked = super::WholePathGuard::set((path.to_string(), whole.iter().map(|w| w.map(|x| x.0)).collect()));
            m.extend(self.compute(sub, &d)?);
        }
        Ok(Value::Object(m))
    }

    /// `(path ordinal, child count)` when segment `si`'s parents `v` are all
    /// its root docs and it has no deletes: its children are then exactly
    /// the docs of `path`.
    fn whole_path_children(&self, si: usize, path: &str, docs: &Docs) -> Option<(u16, u64)> {
        let v = &docs[si];
        let leaf = &self.segs[si].leaf;
        let rustic_core::query::LeafReader::Segment(seg) = leaf else { return None };
        if leaf.deletions_only().is_some() {
            return None;
        }
        let n = leaf.nested_info()?;
        let ord = n.path_ord(path)?;
        let num_docs = leaf.num_docs();
        let non_roots = seg.child_docs().map_or(0, |b| b.count() as u64);
        // An uncollected segment (`with_elided_root`) holds every root.
        if !self.is_elided(docs, si) && (v.len() as u64 != num_docs as u64 - non_roots || !v.iter().all(|&(d, _)| n.is_root(d))) {
            return None;
        }
        // The only nested path: every non-root doc is one of its docs.
        let count = if n.paths().len() == 2 { non_roots } else { (0..num_docs).filter(|&d| n.ord(d) == ord).count() as u64 };
        Some((ord, count))
    }

    fn reverse_nested_agg(&self, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let path = o.get("path").and_then(Value::as_str).unwrap_or("");
        let d: Docs = docs
            .iter()
            .enumerate()
            .map(|(si, v)| {
                let leaf = &self.segs[si].leaf;
                let mut out: Vec<(DocId, f32)> = v.iter().filter_map(|&(c, sc)| leaf.nested_ancestor(c, path).map(|a| (a, sc))).collect();
                out.sort_by_key(|x| x.0);
                out.dedup_by_key(|x| x.0);
                out
            })
            .collect();
        self.single_bucket(d, sub)
    }

    // -----------------------------------------------------------------------
    // geo helpers

    pub(super) fn geo_value_count(&self, field: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        for seg in self.segs {
            if let Some(f) = seg.lookup.get(field) {
                if f.ty != EsType::GeoPoint {
                    return Err(EsError::illegal_argument(format!("Field [{field}] does not support geo_point doc values")));
                }
                if !f.doc_values {
                    return Err(EsError::illegal_argument(format!("Can't load fielddata on [{field}] because fielddata is unsupported on fields of type [geo_point]. Use doc values instead.")));
                }
            }
        }
        let missing = match o.get("missing") {
            Some(value) => {
                values::parse_geo_point(value).ok_or_else(|| EsError::illegal_argument("Invalid geo_point missing value"))?;
                true
            }
            None => false,
        };
        let mut count = 0u64;
        for (seg, selected) in docs.iter().enumerate() {
            let leaf = &self.segs[seg].leaf;
            let field_id = self.segs[seg].lookup.get(field).and_then(|f| f.geo_id);
            // The doc values through a forward cursor (the docs ascend), not
            // a binary search over every point of the segment per doc.
            let dv = field_id.filter(|&id| leaf.has_geo(id)).map(|id| rustic_core::query::GeoDv::open(leaf, id));
            if let (Some(dv), false) = (&dv, missing) {
                if selected.len() == leaf.num_docs() as usize && matches!(leaf, rustic_core::query::LeafReader::Segment(_)) {
                    // Every doc of a sealed leaf (match_all): all its points.
                    count += dv.len() as u64;
                } else {
                    count += selected.iter().map(|&(doc, _)| dv.entries(doc).len() as u64).sum::<u64>();
                }
                continue;
            }
            for &(doc, _) in selected {
                let n = match &dv {
                    Some(dv) => dv.entries(doc).len(),
                    None => self.geo_points(field, seg, doc).len(),
                };
                count += if n == 0 && missing { 1 } else { n as u64 };
            }
        }
        Ok(json!({"value": count}))
    }

    /// Points of geo field `name` of `doc` in segment `seg`.
    pub(super) fn geo_points(&self, name: &str, seg: usize, doc: DocId) -> Vec<(f64, f64)> {
        let Some(f) = self.segs[seg].lookup.get(name) else { return Vec::new() };
        let leaf = &self.segs[seg].leaf;
        if let Some(g) = f.geo_id {
            if leaf.has_geo(g) {
                return leaf.geo_values(g, doc);
            }
        }
        match (f.lat_id, f.lon_id) {
            (Some(a), Some(b)) => leaf.f64_values(a, doc).into_iter().zip(leaf.f64_values(b, doc)).collect(),
            _ => Vec::new(),
        }
    }

    /// Visit the decoded points of the docs `v` (ascending) of segment `s`
    /// of geo field `name`, in doc order: through a forward doc-values
    /// cursor (all of a sealed leaf's points when every doc is selected),
    /// else per doc as [`Ctx::geo_points`].
    pub(super) fn for_each_geo_point(&self, name: &str, s: usize, v: &[(DocId, f32)], mut f: impl FnMut(f64, f64)) {
        let leaf = &self.segs[s].leaf;
        let id = self.segs[s].lookup.get(name).and_then(|rf| rf.geo_id).filter(|&id| leaf.has_geo(id));
        let Some(id) = id else {
            for &(d, _) in v {
                for (lat, lon) in self.geo_points(name, s, d) {
                    f(lat, lon);
                }
            }
            return;
        };
        let dv = rustic_core::query::GeoDv::open(leaf, id);
        if v.len() == leaf.num_docs() as usize && matches!(leaf, rustic_core::query::LeafReader::Segment(_)) {
            for p in dv.iter() {
                f(geo::decode_lat({ p.lat }), geo::decode_lon({ p.lon }));
            }
            return;
        }
        for &(d, _) in v {
            dv.for_each(d, &mut f);
        }
    }

    fn geo_field<'o>(&self, o: &'o Map<String, Value>, name: &str, t: &str) -> Result<&'o str> {
        let f = field_name(o, name, t)?;
        for s in self.segs {
            if let Some(rf) = s.lookup.get(f) {
                // `geo_shape` doc values back every geo aggregation but `geo_distance`.
                let shape_ok = t != "geo_distance" && rf.declared == EsType::GeoShape;
                if rf.ty != EsType::GeoPoint && !shape_ok {
                    return Err(EsError::illegal_argument(format!("Field [{f}] of type [{}] is not supported for aggregation [{t}]", rf.declared.name())));
                }
            }
        }
        Ok(f)
    }

    /// Does any searched index map `f` as `geo_shape`?
    fn any_geo_shape(&self, f: &str) -> bool {
        self.segs.iter().enumerate().any(|(i, s)| s.lookup.get(f).is_some_and(|rf| rf.declared == EsType::GeoShape) || self.is_runtime_shape(f, i))
    }

    fn geo_bounds(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let f = self.geo_field(o, name, "geo_bounds")?;
        let wrap = o.get("wrap_longitude").and_then(Value::as_bool).unwrap_or(true);
        if self.any_geo_shape(f) {
            return Ok(self.geo_extent(f, docs).to_json(wrap));
        }
        let (mut top, mut bottom) = (f64::NEG_INFINITY, f64::INFINITY);
        let (mut pos_left, mut pos_right) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut neg_left, mut neg_right) = (f64::INFINITY, f64::NEG_INFINITY);
        for (s, v) in docs.iter().enumerate() {
            self.for_each_geo_point(f, s, v, |lat, lon| {
                top = top.max(lat);
                bottom = bottom.min(lat);
                if lon >= 0.0 {
                    pos_left = pos_left.min(lon);
                    pos_right = pos_right.max(lon);
                } else {
                    neg_left = neg_left.min(lon);
                    neg_right = neg_right.max(lon);
                }
            });
        }
        if top.is_infinite() {
            return Ok(json!({}));
        }
        // ES `InternalGeoBounds`: with `wrap_longitude`, pick the narrower
        // of the box crossing the dateline and the one that does not.
        let (left, right) = if pos_left.is_infinite() {
            (neg_left, neg_right)
        } else if neg_left.is_infinite() {
            (pos_left, pos_right)
        } else if wrap {
            let unwrapped = pos_right - neg_left;
            let wrapped = (180.0 - pos_left) - (-180.0 - neg_right);
            if unwrapped <= wrapped {
                (neg_left, pos_right)
            } else {
                (pos_left, neg_right)
            }
        } else {
            (neg_left, pos_right)
        };
        Ok(json!({ "bounds": {
            "top_left": { "lat": values::java_double(top), "lon": values::java_double(left) },
            "bottom_right": { "lat": values::java_double(bottom), "lon": values::java_double(right) },
        }}))
    }

    fn geo_centroid(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let f = self.geo_field(o, name, "geo_centroid")?;
        if self.any_geo_shape(f) {
            return Ok(self.geo_shape_centroid(f, docs));
        }
        // ES `GeoCentroidAggregator`: Kahan-compensated sums of the
        // decoded values in doc order, segment after segment.
        let (mut slat, mut slon, mut n) = (Kahan::default(), Kahan::default(), 0u64);
        for (s, v) in docs.iter().enumerate() {
            self.for_each_geo_point(f, s, v, |lat, lon| {
                slat.add(lat);
                slon.add(lon);
                n += 1;
            });
        }
        if n == 0 {
            return Ok(json!({ "count": 0 }));
        }
        Ok(json!({ "location": { "lat": values::java_double(slat.value / n as f64), "lon": values::java_double(slon.value / n as f64) }, "count": n }))
    }

    fn geo_distance_agg(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.geo_field(o, name, "geo_distance")?;
        let origin = o.get("origin").ok_or_else(|| EsError::parsing_at(format!("Missing [origin] in geo_distance aggregator [{name}]"), 1, 1))?;
        let (olat, olon) = values::parse_geo_point(origin).ok_or_else(|| EsError::parsing_at(format!("failed to parse [origin] of geo_distance aggregation [{name}]"), 1, 1))?;
        let unit = o.get("unit").and_then(Value::as_str).unwrap_or("m");
        let scale = values::unit_meters(unit).ok_or_else(|| EsError::illegal_argument(format!("No distance unit match [{unit}]")))?;
        let plane = o.get("distance_type").and_then(Value::as_str) == Some("plane");
        let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(false);
        let ranges = o.get("ranges").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument(format!("No [ranges] specified for the [{name}] aggregation")))?;
        let specs = distance_ranges(ranges);
        self.add_buckets(specs.len())?;
        let count_only = sub.is_none_or(Map::is_empty);
        let mut counts = vec![0u64; specs.len()];
        if count_only {
            // Keep one scratch bit per range, not a distance allocation per
            // document or a copied document list per bucket. Multiple points
            // in one range still contribute only one document to that bucket.
            let mut matched = vec![false; specs.len()];
            for (s, selected) in docs.iter().enumerate() {
                let leaf = &self.segs[s].leaf;
                let dv = self.segs[s].lookup.get(f).and_then(|field| field.geo_id)
                    .filter(|&id| leaf.has_geo(id))
                    .map(|id| rustic_core::query::GeoDv::open(leaf, id));
                for &(doc, _) in selected {
                    matched.fill(false);
                    let mut visit = |lat, lon| {
                        let distance = if plane { geo::plane_meters(olat, olon, lat, lon) }
                            else { geo::haversine_meters(olat, olon, lat, lon) } / scale;
                        for ((_, from, to), hit) in specs.iter().zip(&mut matched) {
                            *hit |= from.is_none_or(|v| distance >= v) && to.is_none_or(|v| distance < v);
                        }
                    };
                    if let Some(dv) = &dv {
                        dv.for_each(doc, &mut visit);
                    } else {
                        for (lat, lon) in self.geo_points(f, s, doc) { visit(lat, lon); }
                    }
                    for (count, hit) in counts.iter_mut().zip(&matched) { *count += u64::from(*hit); }
                }
            }
        }
        // Distances per doc (in `unit`), computed once.
        let dists: Vec<Vec<Vec<f64>>> = if count_only { Vec::new() } else { docs
            .iter()
            .enumerate()
            .map(|(s, v)| {
                v.iter()
                    .map(|&(d, _)| {
                        self.geo_points(f, s, d)
                            .into_iter()
                            .map(|(a, b)| if plane { geo::plane_meters(olat, olon, a, b) } else { geo::haversine_meters(olat, olon, a, b) } / scale)
                            .collect()
                    })
                    .collect()
            })
            .collect() };
        let filters: Vec<(Option<f64>, Option<f64>)> = specs.iter().map(|(_, from, to)| (*from, *to)).collect();
        distance_buckets(specs, keyed, |bucket_index, m| {
            let (from, to) = filters[bucket_index];
            let d: Docs = if count_only { Vec::new() } else { docs
                .iter()
                .enumerate()
                .map(|(s, v)| {
                    v.iter()
                        .zip(&dists[s])
                        .filter(|(_, ds)| ds.iter().any(|x| from.is_none_or(|f| *x >= f) && to.is_none_or(|t| *x < t)))
                        .map(|(e, _)| *e)
                        .collect()
                })
                .collect() };
            m.insert("doc_count".into(), json!(if count_only { counts[bucket_index] } else { count(&d) }));
            if !count_only { m.extend(self.subs(sub, &d)?); }
            Ok(())
        })
    }

    fn geo_grid(&self, name: &str, t: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        use super::geo_shape::{CellPred, Grid};
        let grid = match t {
            "geotile_grid" => Grid::Tile,
            "geohash_grid" => Grid::Hash,
            _ => Grid::Hex,
        };
        let precision = grid_precision(grid, o.get("precision"))?;
        let f = self.geo_field(o, name, t)?;
        let size = o.get("size").and_then(Value::as_u64).unwrap_or(10_000) as usize;
        let bounds = match o.get("bounds") {
            Some(b) => {
                let p = |k: &str| b.get(k).and_then(values::parse_geo_point);
                match (p("top_left"), p("bottom_right")) {
                    (Some((t0, l)), Some((b0, r))) => Some(CellPred::new(grid, precision as u32, (t0, l, b0, r))),
                    _ => None,
                }
            }
            None => None,
        };
        if !self.any_geo_shape(f) {
            if let Some(out) = self.geo_point_grid(name, grid, precision as u32, f, size, bounds.as_ref(), sub, docs)? {
                return Ok(out);
            }
        }
        // `geo_shape` values (and any `geo_point` values of the field in
        // other indices): ES's tilers over each doc's shape doc value
        // (`GeoShapeCellIdSource`), points by their cell; cells as ES's
        // longs, keyed when the buckets are built.
        use super::shape_tiler::{Bounds, HashTiler, HexTiler, TileTiler, Tiler};
        let raw_bounds = o.get("bounds").and_then(|b| {
            let p = |k: &str| b.get(k).and_then(values::parse_geo_point);
            match (p("top_left"), p("bottom_right")) {
                (Some((top, left)), Some((bottom, right))) => Some(Bounds { top, left, bottom, right }),
                _ => None,
            }
        });
        let pu = precision as u32;
        let tiler = match grid {
            Grid::Tile => Tiler::Tile(TileTiler::new(pu, raw_bounds.as_ref())),
            Grid::Hash => Tiler::Hash(HashTiler::new(pu, raw_bounds.as_ref())),
            Grid::Hex => Tiler::Hex(HexTiler::new(pu, raw_bounds.as_ref())),
        };
        let point_cell = |lat: f64, lon: f64| -> Option<u64> {
            match grid {
                Grid::Tile => {
                    let tiles = 1u64 << pu;
                    let (x, y) = (super::shape_tiler::get_x_tile(lon, tiles), super::shape_tiler::get_y_tile(lat, tiles));
                    if let Some(b) = &bounds {
                        if !b.tile(x, y) {
                            return None;
                        }
                    }
                    Some(super::shape_tiler::long_encode_tiles(pu, x, y))
                }
                Grid::Hash => {
                    let h = super::shape_tiler::gh_string_encode(lon, lat, pu);
                    if let Some(b) = &bounds {
                        if !b.key(&h, pu) {
                            return None;
                        }
                    }
                    Some(super::shape_tiler::gh_long_encode(&h))
                }
                Grid::Hex => {
                    let h = super::shape_tiler::geo_to_h3(lat, lon, pu);
                    let k = super::shape_tiler::hex_key(h);
                    if let Some(b) = &bounds {
                        if !b.key(&k, pu) {
                            return None;
                        }
                    }
                    Some(h)
                }
            }
        };
        let mut cells: HashMap<u64, Docs> = HashMap::new();
        let mut buf: Vec<u64> = Vec::new();
        let mut shape_cells = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            // Resolved once per segment: points read only on point fields.
            let points = self.segs[s].lookup.get(f).is_some_and(|rf| rf.geo_id.is_some() || rf.lat_id.is_some());
            let reader = self.shape_reader(f, s);
            // Many docs read from a segment's shape doc values: each doc's
            // cells (the tiling, the cost) computed in chunks on the pool,
            // then grouped in doc order as below (same buckets, doc lists
            // and sub-aggregations).
            if let (false, super::geo_shape::ShapeReader::Column(col)) = (points, &reader) {
                if v.len() >= GRID_PAR_MIN_DOCS {
                    use rayon::prelude::*;
                    let tiler = &tiler;
                    let chunks: Vec<(Vec<u64>, Vec<u32>)> = v
                        .par_chunks(GRID_CHUNK_DOCS)
                        .map(|chunk| {
                            let (mut flat, mut ends, mut sc) = (Vec::new(), Vec::with_capacity(chunk.len()), Vec::new());
                            for &(d, _) in chunk {
                                if let Some(sv) = col.get(d) {
                                    // Sorted and distinct.
                                    tiler.cells(&sv, &mut sc);
                                    flat.extend_from_slice(&sc);
                                }
                                ends.push(flat.len() as u32);
                            }
                            (flat, ends)
                        })
                        .collect();
                    let mut docs_iter = v.iter();
                    for (flat, ends) in &chunks {
                        let mut start = 0usize;
                        for &end in ends {
                            let &(d, sc) = docs_iter.next().expect("one end per doc");
                            for &c in &flat[start..end as usize] {
                                self.push_grouped(name, &mut cells, c, 16, docs.len(), s, (d, sc))?;
                            }
                            start = end as usize;
                        }
                    }
                    continue;
                }
            }
            for &(d, sc) in v {
                buf.clear();
                if points {
                    for (lat, lon) in self.geo_points(f, s, d) {
                        if let Some(c) = point_cell(lat, lon) {
                            buf.push(c);
                        }
                    }
                }
                if let Some(sv) = self.read_shape(&reader, f, s, d) {
                    tiler.cells(&sv.value(), &mut shape_cells);
                    buf.extend_from_slice(&shape_cells);
                }
                buf.sort_unstable();
                buf.dedup();
                for &c in &buf {
                    self.push_grouped(name, &mut cells, c, 16, docs.len(), s, (d, sc))?;
                }
            }
        }
        // ES order: doc count descending, then the cell's long descending.
        let mut list: Vec<(u64, Docs)> = cells.into_iter().collect();
        list.sort_by(|a, b| count(&b.1).cmp(&count(&a.1)).then_with(|| b.0.cmp(&a.0)));
        list.truncate(size);
        self.add_buckets(list.len())?;
        let mut buckets = Vec::with_capacity(list.len());
        for (key, d) in list {
            let mut m = Map::new();
            m.insert("key".into(), json!(tiler.key(key)));
            m.insert("doc_count".into(), json!(count(&d)));
            m.extend(self.subs(sub, &d)?);
            buckets.push(Value::Object(m));
        }
        Ok(json!({ "buckets": buckets }))
    }
}

impl<'a, 'b> Ctx<'a, 'b> {
    /// [`Ctx::geo_grid`] over a `geo_point` field read from the core doc
    /// values (`None`: some segment holds the field otherwise). Cells are
    /// computed as integers (geohash bits, packed tile `x, y`, H3 index)
    /// from a forward doc-values cursor, the `bounds` verdict once per
    /// distinct cell, the strings only for the returned buckets. Same
    /// buckets, counts and order as the string path: geohashes of one
    /// precision order like their bits, tiles like `(x, y)`, H3 addresses
    /// like their index.
    #[allow(clippy::too_many_arguments)]
    fn geo_point_grid(&self, name: &str, grid: super::geo_shape::Grid, precision: u32, f: &str, size: usize, bounds: Option<&super::geo_shape::CellPred>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Option<Value>> {
        use super::geo_shape::Grid;
        let mut ids = Vec::with_capacity(docs.len());
        for s in 0..docs.len() {
            let leaf = &self.segs[s].leaf;
            match self.segs[s].lookup.get(f) {
                None => ids.push(None),
                Some(rf) => match rf.geo_id.filter(|&id| leaf.has_geo(id)) {
                    Some(id) => ids.push(Some(id)),
                    // Declared but without core geo doc values here (e.g. a
                    // missing field): nothing to read, unless it has lat/lon.
                    None if rf.lat_id.is_none() => ids.push(None),
                    None => return Ok(None),
                },
            }
        }
        let cell_of = |lat: f64, lon: f64| -> Option<u64> {
            match grid {
                Grid::Tile => {
                    let (x, y) = geo::geotile(lat, lon, precision);
                    Some(((x as u64) << 32) | y as u64)
                }
                Grid::Hash => Some(geo::geohash_bits(lat, lon, precision as usize)),
                Grid::Hex => super::geo_hex::point_cell_id(lat, lon, precision),
            }
        };
        let key_string = |k: u64| -> String {
            match grid {
                Grid::Tile => format!("{precision}/{}/{}", k >> 32, k & 0xffff_ffff),
                Grid::Hash => geo::geohash_of_bits(k, precision as usize),
                Grid::Hex => super::geo_hex::cell_address(k).unwrap_or_default(),
            }
        };
        let mut verdicts: HashMap<u64, bool> = HashMap::new();
        let mut cells: HashMap<u64, Docs> = HashMap::new();
        let mut keys: Vec<u64> = Vec::with_capacity(4);
        for (s, v) in docs.iter().enumerate() {
            let Some(id) = ids[s] else { continue };
            let dv = rustic_core::query::GeoDv::open(&self.segs[s].leaf, id);
            for &(d, sc) in v {
                keys.clear();
                for p in dv.entries(d) {
                    let Some(k) = cell_of(geo::decode_lat({ p.lat }), geo::decode_lon({ p.lon })) else { continue };
                    if keys.contains(&k) {
                        continue;
                    }
                    let ok = match bounds {
                        None => true,
                        Some(b) => *verdicts.entry(k).or_insert_with(|| b.key(&key_string(k), precision)),
                    };
                    if ok {
                        keys.push(k);
                    }
                }
                for &k in &keys {
                    let kb = if cells.contains_key(&k) { 0 } else { key_string(k).capacity() };
                    self.push_grouped(name, &mut cells, k, kb, docs.len(), s, (d, sc))?;
                }
            }
        }
        let mut list: Vec<(u64, Docs)> = cells.into_iter().collect();
        list.sort_by(|a, b| count(&b.1).cmp(&count(&a.1)).then_with(|| b.0.cmp(&a.0)));
        list.truncate(size);
        self.add_buckets(list.len())?;
        let mut buckets = Vec::with_capacity(list.len());
        for (key, d) in list {
            let mut m = Map::new();
            m.insert("key".into(), json!(key_string(key)));
            m.insert("doc_count".into(), json!(count(&d)));
            m.extend(self.subs(sub, &d)?);
            buckets.push(Value::Object(m));
        }
        Ok(Some(json!({ "buckets": buckets })))
    }
}

/// The `precision` of a grid aggregation with ES defaults and validation.
fn grid_precision(grid: super::geo_shape::Grid, v: Option<&Value>) -> Result<usize> {
    use super::geo_shape::Grid;
    let (default, max) = match grid {
        Grid::Tile => (7, 29),
        Grid::Hash => (5, 12),
        Grid::Hex => (5, 15),
    };
    let precision = match v {
        None => default,
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as usize,
        Some(Value::String(s)) => match s.parse::<usize>() {
            Ok(p) => p,
            Err(_) if grid == Grid::Hash => {
                // A distance: the coarsest geohash whose cells are smaller.
                let m = values::parse_distance(&json!(s)).ok_or_else(|| EsError::illegal_argument(format!("Invalid precision [{s}]")))?;
                (1..=12).find(|&p| geohash_cell_width_m(p) <= m).unwrap_or(12)
            }
            Err(_) if grid == Grid::Hex => {
                return Err(EsError::x_content_parse("[1:1] [geohex_grid] failed to parse field [precision]")
                    .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))));
            }
            Err(_) => return Err(EsError::illegal_argument(format!("Invalid geotile_grid precision of {s}. Must be between 0 and 29."))),
        },
        Some(other) => return Err(EsError::illegal_argument(format!("Invalid precision [{other}]"))),
    };
    match grid {
        Grid::Tile if precision > max => Err(EsError::illegal_argument(format!("Invalid geotile_grid precision of {precision}. Must be between 0 and 29."))),
        Grid::Hash if !(1..=max).contains(&precision) => Err(EsError::illegal_argument(format!("Invalid geohash aggregation precision of {precision}. Must be between 1 and 12."))),
        Grid::Hex if precision > max => Err(EsError::x_content_parse("[1:1] [geohex_grid] failed to parse field [precision]")
            .caused_by(EsError::illegal_argument(format!("Invalid geohex aggregation precision of {precision}. Must be between 0 and 15")))),
        _ => Ok(precision),
    }
}

/// Approximate width in meters of a geohash cell at `p` characters.
fn geohash_cell_width_m(p: usize) -> f64 {
    const W: [f64; 13] = [40_075_000.0, 5_009_400.0, 1_252_300.0, 156_500.0, 39_100.0, 4_900.0, 1_200.0, 152.9, 38.2, 4.8, 1.2, 0.149, 0.037];
    W[p.min(12)]
}

/// ES `CompensatedSum` (Kahan summation) for finite values.
#[derive(Default)]
pub(super) struct Kahan {
    pub(super) value: f64,
    delta: f64,
}

impl Kahan {
    #[inline]
    pub(super) fn add(&mut self, x: f64) {
        if !x.is_finite() {
            self.value += x;
        }
        if self.value.is_finite() {
            let corrected = x + self.delta;
            let updated = self.value + corrected;
            self.delta = corrected - (updated - self.value);
            self.value = updated;
        }
    }
}



/// A count-only `geohash_grid` / `geotile_grid` / `geohex_grid` of a
/// `geo_point` field (options `field`, `precision`, `size`, `bounds` only),
/// counted from the KD trees by the search's aggregation shortcut: points
/// of single-valued fields (one per doc, so cell counts are doc counts)
/// celled as [`Ctx::geo_point_grid`] cells them, and whole leaves whose box
/// lies in one cell counted at once. Same buckets, counts and order.
pub(crate) struct PointGrid {
    pub field: String,
    grid: super::geo_shape::Grid,
    precision: u32,
    size: usize,
    bounds: Option<super::geo_shape::CellPred>,
    /// Bounds verdict per cell (as the aggregation caches it).
    verdicts: HashMap<u64, bool>,
    counts: HashMap<u64, u64>,
    /// geohex: the predicate of a cell, to claim leaves inside it.
    hex: HashMap<u64, std::sync::Arc<super::geo_hex::HexCellPred>>,
    /// The last few points' cells, where points provably share them and
    /// the points counted for them not yet added, latest first (points
    /// come in KD leaf order: a leaf spans a few cells).
    recent: Vec<(u64, LastCell, u64)>,
}

/// Cells [`PointGrid`] keeps in `recent`.
const RECENT_CELLS: usize = 4;

/// Where points provably lie in a [`PointGrid`]'s last cell.
enum LastCell {
    /// `[min_lat, max_lat) x [min_lon, max_lon)`.
    Box(f64, f64, f64, f64),
    Hex(std::sync::Arc<super::geo_hex::HexCellPred>),
    None,
}

impl LastCell {
    /// `unit`: the point's unit vector, computed on first need.
    #[inline]
    fn holds(&self, lat: f64, lon: f64, unit: &mut Option<[f64; 3]>) -> bool {
        match self {
            LastCell::Box(a, b, c, d) => lat >= *a && lat < *b && lon >= *c && lon < *d,
            LastCell::Hex(p) => p.unit_inside(*unit.get_or_insert_with(|| super::geo_hex::unit_vector(lat, lon))),
            LastCell::None => false,
        }
    }
}

impl PointGrid {
    /// `def` (one aggregation's definition) when it is such an aggregation.
    pub fn parse(def: &Map<String, Value>) -> Option<Self> {
        use super::geo_shape::{CellPred, Grid};
        if def.len() != 1 {
            return None;
        }
        let (t, o) = def.iter().next()?;
        let grid = match t.as_str() {
            "geotile_grid" => Grid::Tile,
            "geohash_grid" => Grid::Hash,
            "geohex_grid" => Grid::Hex,
            _ => return None,
        };
        let o = o.as_object()?;
        if !o.keys().all(|k| matches!(k.as_str(), "field" | "precision" | "size" | "bounds")) {
            return None;
        }
        let precision = grid_precision(grid, o.get("precision")).ok()? as u32;
        let size = match o.get("size") {
            None => 10_000,
            Some(v) => v.as_u64()? as usize,
        };
        let bounds = match o.get("bounds") {
            None => None,
            Some(b) => {
                let p = |k: &str| b.get(k).and_then(values::parse_geo_point);
                match (p("top_left"), p("bottom_right")) {
                    (Some((t0, l)), Some((b0, r))) => Some(CellPred::new(grid, precision, (t0, l, b0, r))),
                    _ => return None,
                }
            }
        };
        Some(PointGrid { field: o.get("field")?.as_str()?.to_string(), grid, precision, size, bounds, verdicts: HashMap::new(), counts: HashMap::new(), hex: HashMap::new(), recent: Vec::with_capacity(RECENT_CELLS) })
    }

    fn cell_of(&self, lat: f64, lon: f64) -> Option<u64> {
        use super::geo_shape::Grid;
        match self.grid {
            Grid::Tile => {
                let (x, y) = geo::geotile(lat, lon, self.precision);
                Some(((x as u64) << 32) | y as u64)
            }
            Grid::Hash => Some(geo::geohash_bits(lat, lon, self.precision as usize)),
            Grid::Hex => super::geo_hex::point_cell_id(lat, lon, self.precision),
        }
    }

    fn key_string(&self, k: u64) -> String {
        use super::geo_shape::Grid;
        match self.grid {
            Grid::Tile => format!("{}/{}/{}", self.precision, k >> 32, k & 0xffff_ffff),
            Grid::Hash => geo::geohash_of_bits(k, self.precision as usize),
            Grid::Hex => super::geo_hex::cell_address(k).unwrap_or_default(),
        }
    }

    fn add(&mut self, k: u64, n: u64) {
        let ok = match &self.bounds {
            None => true,
            Some(b) => {
                let key = self.key_string(k);
                *self.verdicts.entry(k).or_insert_with(|| b.key(&key, self.precision))
            }
        };
        if ok {
            *self.counts.entry(k).or_insert(0) += n;
        }
    }

    /// One point: a recent point's cell when the point provably shares it
    /// (inside its box, or a hexagon's edges by the margin), else its own;
    /// counted per recent cell, added when it leaves the recent ones.
    pub fn add_point(&mut self, lat: f64, lon: f64) {
        let mut unit = None;
        if let Some(i) = self.recent.iter().position(|(_, c, _)| c.holds(lat, lon, &mut unit)) {
            self.recent[i].2 += 1;
            if i > 0 {
                self.recent.swap(0, i);
            }
            return;
        }
        let Some(k) = self.cell_of(lat, lon) else { return };
        if let Some(i) = self.recent.iter().position(|(rk, _, _)| *rk == k) {
            self.recent[i].2 += 1;
            return;
        }
        let c = self.last_cell(k);
        if self.recent.len() == RECENT_CELLS {
            let (ok, _, n) = self.recent.pop().unwrap();
            self.add(ok, n);
        }
        self.recent.insert(0, (k, c, 1));
    }

    /// An empty grid of the same aggregation (one segment's share, run on
    /// its own thread), its cell predicates shared.
    pub fn fork(&self) -> PointGrid {
        PointGrid {
            field: self.field.clone(),
            grid: self.grid,
            precision: self.precision,
            size: self.size,
            bounds: self.bounds.clone(),
            verdicts: HashMap::new(),
            counts: HashMap::new(),
            hex: self.hex.clone(),
            recent: Vec::with_capacity(RECENT_CELLS),
        }
    }

    /// Adds a [`PointGrid::fork`]'s counts (finished).
    pub fn merge(&mut self, mut other: PointGrid) {
        other.finish();
        for (k, n) in other.counts {
            *self.counts.entry(k).or_insert(0) += n;
        }
    }

    /// Adds the points counted for the recent cells.
    pub fn finish(&mut self) {
        for (k, _, n) in std::mem::take(&mut self.recent) {
            if n > 0 {
                self.add(k, n);
            }
        }
    }

    /// Where points provably share cell `k`: geohash cells' exact boxes
    /// (bisections are exact in binary), geotiles' boxes a nanodegree in
    /// from their edges (the mercator formula rounds), hexagons' edges.
    fn last_cell(&mut self, k: u64) -> LastCell {
        use super::geo_shape::Grid;
        match self.grid {
            Grid::Hash => {
                let p = self.precision as usize * 5;
                let (mut lat, mut lon) = ((-90.0f64, 90.0f64), (-180.0f64, 180.0f64));
                for i in 0..p {
                    let bit = (k >> (p - 1 - i)) & 1 == 1;
                    let r = if i % 2 == 0 { &mut lon } else { &mut lat };
                    let mid = (r.0 + r.1) / 2.0;
                    if bit {
                        r.0 = mid;
                    } else {
                        r.1 = mid;
                    }
                }
                LastCell::Box(lat.0, lat.1, lon.0, lon.1)
            }
            Grid::Tile => {
                let (top, left, bottom, right) = geo::geotile_bbox(self.precision, (k >> 32) as u32, (k & 0xffff_ffff) as u32);
                let e = 1e-9;
                // Tiles at the mercator limits hold the clamped poles.
                if top >= geo::MERCATOR_MAX_LAT - 1e-6 || bottom <= -geo::MERCATOR_MAX_LAT + 1e-6 {
                    return LastCell::None;
                }
                LastCell::Box(bottom + e, top - e, left + e, right - e)
            }
            Grid::Hex => {
                if !crate::h3::is_valid(k) {
                    return LastCell::None;
                }
                LastCell::Hex(self.hex.entry(k).or_insert_with(|| std::sync::Arc::new(super::geo_hex::HexCellPred::new(k))).clone())
            }
        }
    }

    /// `n` points in the box `(min_lat, max_lat, min_lon, max_lon)` when
    /// they all lie in one cell (counted; `true`), else nothing (`false`):
    /// geohashes and tiles are rectangles split monotonically per axis, so
    /// opposite corners in one cell put the box in it; a geohex leaf is
    /// claimed when the cell of its centre provably covers the box.
    pub fn add_leaf(&mut self, b: (f64, f64, f64, f64), n: u64) -> bool {
        use super::geo_shape::Grid;
        let (a, c) = match self.grid {
            Grid::Tile | Grid::Hash => (self.cell_of(b.1, b.2), self.cell_of(b.0, b.3)),
            Grid::Hex => {
                let Some(k) = self.cell_of((b.0 + b.1) / 2.0, (b.2 + b.3) / 2.0) else { return false };
                if !crate::h3::is_valid(k) {
                    return false;
                }
                let p = self.hex.entry(k).or_insert_with(|| std::sync::Arc::new(super::geo_hex::HexCellPred::new(k)));
                if !p.box_inside(b.0, b.1, b.2, b.3) {
                    return false;
                }
                (Some(k), Some(k))
            }
        };
        match (a, c) {
            (Some(x), Some(y)) if x == y => {
                self.add(x, n);
                true
            }
            _ => false,
        }
    }

    /// Buckets the response would hold.
    pub fn buckets(&self) -> usize {
        self.counts.len().min(self.size)
    }

    /// The aggregation's response (doc count descending, then the cell's
    /// long encoding descending, as `geo_point_grid` orders them).
    pub fn render(&self) -> Value {
        let mut list: Vec<(u64, u64)> = self.counts.iter().map(|(k, n)| (*k, *n)).collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        list.truncate(self.size);
        let buckets: Vec<Value> = list.into_iter().map(|(k, n)| json!({ "key": self.key_string(k), "doc_count": n })).collect();
        json!({ "buckets": buckets })
    }
}

/// One `geo_distance` range: its `key`, `from` and `to` (in the unit).
pub(crate) type DistanceRange = (Option<String>, Option<f64>, Option<f64>);

/// The ranges of a `geo_distance` aggregation (numbers or numeric strings).
fn distance_ranges(ranges: &[Value]) -> Vec<DistanceRange> {
    ranges
        .iter()
        .map(|r| {
            let num = |k: &str| r.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
            (r.get("key").and_then(Value::as_str).map(str::to_string), num("from"), num("to"))
        })
        .collect()
}

/// The `geo_distance` response: per range its key, bounds, and what
/// `fill(range index, bucket)` adds (the doc count, sub-aggregations).
fn distance_buckets(specs: Vec<DistanceRange>, keyed: bool, mut fill: impl FnMut(usize, &mut Map<String, Value>) -> Result<()>) -> Result<Value> {
    let mut buckets = Vec::new();
    let mut keyed_out = Map::new();
    for (bucket_index, (key, from, to)) in specs.into_iter().enumerate() {
        let key = key.unwrap_or_else(|| {
            format!("{}-{}", from.filter(|value| *value != 0.0).map(jfmt).unwrap_or_else(|| "*".into()), to.map(jfmt).unwrap_or_else(|| "*".into()))
        });
        let mut m = Map::new();
        if !keyed {
            m.insert("key".into(), json!(key));
        }
        if let Some(x) = from {
            m.insert("from".into(), values::java_double(x));
        }
        if let Some(x) = to {
            m.insert("to".into(), values::java_double(x));
        }
        fill(bucket_index, &mut m)?;
        if keyed {
            keyed_out.insert(key, Value::Object(m));
        } else {
            buckets.push(Value::Object(m));
        }
    }
    Ok(if keyed { json!({ "buckets": keyed_out }) } else { json!({ "buckets": buckets }) })
}

/// A count-only `geo_distance` aggregation of `arc` distances (no
/// sub-aggregations, script, `missing` or `meta`), whose bucket counts the
/// search may take from KD trees: a bucket counts the docs with a point
/// `p` where `from <= haversine(origin, p) / unit < to`, which
/// [`RingPred`] tests exactly as the aggregation does.
#[derive(Debug)]
pub(crate) struct DistanceCounts {
    pub field: String,
    lat: f64,
    lon: f64,
    scale: f64,
    specs: Vec<DistanceRange>,
    keyed: bool,
}

impl DistanceCounts {
    /// `def` (one aggregation's definition) when it is such an
    /// aggregation; `None` when anything else about it (including an
    /// error ES reports) needs the regular path.
    pub fn parse(def: &Map<String, Value>) -> Option<Self> {
        if def.len() != 1 {
            return None;
        }
        let o = def.get("geo_distance")?.as_object()?;
        if !o.keys().all(|k| matches!(k.as_str(), "field" | "origin" | "unit" | "ranges" | "keyed" | "distance_type")) {
            return None;
        }
        if o.get("distance_type").is_some_and(|t| t.as_str() != Some("arc")) {
            return None;
        }
        let field = o.get("field")?.as_str()?.to_string();
        let (lat, lon) = values::parse_geo_point(o.get("origin")?)?;
        if !(lat.is_finite() && lon.is_finite() && (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon)) {
            return None;
        }
        let scale = match o.get("unit") {
            None => 1.0,
            Some(u) => values::unit_meters(u.as_str()?)?,
        };
        let keyed = match o.get("keyed") {
            None => false,
            Some(k) => k.as_bool()?,
        };
        let ranges = o.get("ranges")?.as_array()?;
        if ranges.is_empty() {
            return None;
        }
        for r in ranges {
            let r = r.as_object()?;
            for (k, v) in r {
                let ok = match k.as_str() {
                    "key" => v.is_string(),
                    "from" | "to" => v.as_f64().is_some_and(f64::is_finite),
                    _ => false,
                };
                if !ok {
                    return None;
                }
            }
        }
        Some(DistanceCounts { field, lat, lon, scale, specs: distance_ranges(ranges), keyed })
    }

    /// Buckets the response has.
    pub fn buckets(&self) -> usize {
        self.specs.len()
    }

    /// Bucket `i`'s predicate.
    pub fn ring(&self, i: usize) -> RingPred {
        let (_, from, to) = self.specs[i];
        RingPred { lat: self.lat, lon: self.lon, scale: self.scale, from, to }
    }

    /// The aggregation's response for these bucket counts.
    pub fn render(&self, counts: &[u64]) -> Value {
        distance_buckets(self.specs.clone(), self.keyed, |i, m| {
            m.insert("doc_count".into(), json!(counts[i]));
            Ok(())
        })
        .unwrap_or(Value::Null)
    }
}

/// Points in one `geo_distance` bucket: `from <= haversine / scale < to`,
/// computed exactly as the aggregation computes it. KD boxes are decided
/// through the circle relations of the query (which keep a margin of
/// 1e-6 relative + 1 m on both sides), so only boxes near the bounds
/// test their points.
#[derive(Debug, Clone)]
pub(crate) struct RingPred {
    lat: f64,
    lon: f64,
    scale: f64,
    from: Option<f64>,
    to: Option<f64>,
}

impl geo::PointPredicate for RingPred {
    fn classify(&self, a: f64, b: f64, c: f64, d: f64) -> geo::Relation {
        use geo::Relation;
        let circle = |unit: f64| geo::GeoShape::Distance { lat: self.lat, lon: self.lon, meters: unit * self.scale }.relate(a, b, c, d);
        // Below `from` everywhere / above it everywhere.
        let low = match self.from {
            Some(f) if f > 0.0 => circle(f),
            _ => Relation::Outside,
        };
        if low == Relation::Inside {
            return Relation::Outside;
        }
        let high = match self.to {
            Some(t) if t <= 0.0 => return Relation::Outside,
            Some(t) => circle(t),
            None => Relation::Inside,
        };
        match (low, high) {
            (_, Relation::Outside) => Relation::Outside,
            (Relation::Outside, Relation::Inside) => Relation::Inside,
            _ => Relation::Crosses,
        }
    }

    #[inline]
    fn test(&self, lat: f64, lon: f64) -> bool {
        let distance = geo::haversine_meters(self.lat, self.lon, lat, lon) / self.scale;
        self.from.is_none_or(|v| distance >= v) && self.to.is_none_or(|v| distance < v)
    }
}

#[cfg(test)]
mod point_grid_tests {
    use super::super::geo_shape::{point_cell, CellPred, Grid};
    use super::super::{run_aggregations, AggEnv, AggSeg};
    use crate::analysis::AnalysisRegistry;
    use crate::document::parse_with_dynamic;
    use crate::error::Result;
    use crate::mapping::{FieldLookup, Mappings};
    use rustic_core::types::{DocId, IndexSettings};
    use rustic_core::Index;
    use serde_json::{json, Value};
    use std::collections::HashMap;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `(lon, lat)` points per doc: Paris (the Rally bounds) and around it,
    /// the dateline, the poles and beyond the web-mercator latitudes, docs
    /// with several points (some in the same cell) and without any.
    fn docs(n: usize, seed: u64) -> Vec<Vec<(f64, f64)>> {
        let mut r = Lcg(seed);
        (0..n)
            .map(|i| {
                if i % 41 == 3 {
                    return Vec::new();
                }
                let k = if i % 6 == 0 { 2 + i % 3 } else { 1 };
                (0..k)
                    .map(|_| match (r.next() * 8.0) as u32 {
                        0 => (-180.0 + 360.0 * r.next(), -90.0 + 180.0 * r.next()),
                        1 => (if r.next() < 0.5 { 180.0 - r.next() } else { -180.0 + r.next() }, -5.0 + 10.0 * r.next()),
                        2 => (-180.0 + 360.0 * r.next(), if r.next() < 0.5 { 84.0 + 6.0 * r.next() } else { -90.0 + 6.0 * r.next() }),
                        3 => (2.20, 48.75 + 0.2 * r.next()),
                        _ => (2.1 + 0.4 * r.next(), 48.65 + 0.4 * r.next()),
                    })
                    .collect()
            })
            .collect()
    }

    fn run(index: &Index, lookup: &FieldLookup, aggs: Value) -> Result<Value> {
        let leaves = index.leaves();
        let segs: Vec<AggSeg> = leaves
            .iter()
            .map(|l| AggSeg { leaf: *l, lookup, index: "idx", max_regex_length: 1000, docs: (0..l.num_docs()).collect(), scores: vec![0.0; l.num_docs() as usize] })
            .collect();
        let fetch = |_s: usize, _d: DocId, _sc: f32, _spec: &Value| -> Result<Value> { Ok(json!({})) };
        let all = |s: usize| -> Vec<DocId> { (0..leaves[s].num_docs()).collect() };
        let env = AggEnv { mapping_context: &[], now: 0, max_buckets: 1 << 20, fetch_hit: &fetch, fetch_source: None, all_docs: &all, background_filtered: false, top_metrics_max_size: 10, max_result_window: 10_000, profile: false };
        run_aggregations(&aggs, &segs, &env)
    }

    /// Integer cells give the string path's buckets: geohash 1-12, geotile
    /// 0-29, geohex 0-15 (a sample), with and without the Rally bounds, on
    /// the memtable and sealed segments, keys, counts and order (ties by
    /// key, descending), `size` truncation.
    #[test]
    fn integer_cells_match_string_cells() {
        let reg = AnalysisRegistry::default();
        let mut m = Mappings::parse(&json!({"properties": {"g": {"type": "geo_point"}}}), &reg).unwrap();
        m.assign_core_ids();
        let lookup = FieldLookup::build(&m, &reg);
        let mut index = Index::new("idx", lookup.core_schema(), IndexSettings::default());
        let ds = docs(4000, 5);
        for (i, pts) in ds.iter().enumerate() {
            let src = match pts.len() {
                0 => json!({}),
                1 => json!({"g": [pts[0].0, pts[0].1]}),
                _ => json!({"g": pts.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>()}),
            };
            let (pd, _) = parse_with_dynamic(&src, &i.to_string(), &m, &lookup, 1000).unwrap();
            index.add_document(&pd.doc).unwrap();
            if i == ds.len() / 2 {
                index.refresh().unwrap();
            }
        }
        let paris = (48.95, 2.20, 48.75, 2.40);
        let cases: Vec<(&str, Grid, u32)> = (1..=12).map(|p| ("geohash_grid", Grid::Hash, p))
            .chain([0, 1, 5, 13, 17, 22, 29].into_iter().map(|p| ("geotile_grid", Grid::Tile, p)))
            .chain([0, 1, 2, 6, 9, 15].into_iter().map(|p| ("geohex_grid", Grid::Hex, p)))
            .collect();
        for sealed in [false, true] {
            if sealed {
                index.refresh().unwrap();
            }
            for &(t, grid, p) in &cases {
                for bounded in [false, true] {
                    for size in [10_000usize, 3] {
                        let mut spec = json!({"field": "g", "precision": p, "size": size});
                        if bounded {
                            spec["bounds"] = json!({"top_left": "POINT (2.20 48.95)", "bottom_right": "POINT (2.40 48.75)"});
                        }
                        let got = run(&index, &lookup, json!({"a": {t: spec}})).unwrap();
                        let got: Vec<(String, u64)> = got["a"]["buckets"].as_array().unwrap().iter().map(|b| (b["key"].as_str().unwrap().to_string(), b["doc_count"].as_u64().unwrap())).collect();
                        // The string path's definition, over the quantized points.
                        let pred = bounded.then(|| CellPred::new(grid, p, paris));
                        let mut counts: HashMap<String, u64> = HashMap::new();
                        for pts in &ds {
                            let mut keys: Vec<String> = Vec::new();
                            for &(lon, lat) in pts {
                                let (lat, lon) = rustic_core::geo::quantize(lat, lon);
                                if let Some(k) = point_cell(grid, p, lat, lon) {
                                    if pred.as_ref().is_none_or(|b| b.key(&k, p)) && !keys.contains(&k) {
                                        keys.push(k);
                                    }
                                }
                            }
                            for k in keys {
                                *counts.entry(k).or_default() += 1;
                            }
                        }
                        let order = |k: &str| -> (u64, u64) {
                            match grid {
                                Grid::Tile => {
                                    let mut it = k.split('/').skip(1).map(|x| x.parse::<u64>().unwrap());
                                    (it.next().unwrap(), it.next().unwrap())
                                }
                                Grid::Hex => (super::super::geo_hex::cell_long(k), 0),
                                Grid::Hash => (0, 0),
                            }
                        };
                        let mut want: Vec<(String, u64)> = counts.into_iter().collect();
                        want.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| match grid {
                            Grid::Hash => b.0.cmp(&a.0),
                            _ => order(&b.0).cmp(&order(&a.0)),
                        }));
                        want.truncate(size);
                        assert_eq!(got, want, "{t} precision={p} bounded={bounded} size={size} sealed={sealed}");
                    }
                }
            }
        }
    }
}

/// The live docs of nested path `ord` under the root docs `parents`
/// (ascending, each with its score), in doc order, each with its root's
/// score: what walking every parent's block yields, read in one forward
/// pass over the path ordinals (children precede their root; a root's
/// block is everything after the previous root). `None` when `parents`
/// holds a non-root doc.
fn root_children_scan(n: &rustic_core::nested::NestedInfo, ord: u16, num_docs: u32, parents: &[(DocId, f32)], deleted: Option<&rustic_core::postings::BitSet>) -> Option<Vec<(DocId, f32)>> {
    let mut out: Vec<(DocId, f32)> = Vec::with_capacity((num_docs as usize).saturating_sub(parents.len()));
    // Children collected since the last root.
    let mut pending = 0usize;
    let mut j = 0usize;
    for d in 0..num_docs {
        let o = n.ord(d);
        if o == 0 {
            if j < parents.len() && parents[j].0 < d {
                // A parent that is not a root doc.
                return None;
            }
            if j < parents.len() && parents[j].0 == d {
                let sc = parents[j].1;
                for c in &mut out[pending..] {
                    c.1 = sc;
                }
                j += 1;
                pending = out.len();
            } else {
                out.truncate(pending);
            }
        } else if o == ord && deleted.is_none_or(|b| !b.get(d)) {
            out.push((d, 0.0));
        }
    }
    out.truncate(pending);
    (j == parents.len()).then_some(out)
}

/// Whether every sub-aggregation is a count-only `date_histogram` (no
/// sub-aggregations, `missing` or `hard_bounds`; no `_doc_count` weights),
/// the shape [`Ctx::whole_path_children`] marks segments for.
/// The path of an `aggs` request that is one `nested` aggregation (no
/// `meta`) whose sub-aggregations are count-only `date_histogram`s
/// ([`whole_path_subs`], Rally nested's `nested-date-histo`): over every
/// root of a segment without deletions they count the path's docs from
/// the points index, so a `match_all` need not collect the roots
/// (`with_elided_root`).
pub fn nested_whole_path_spec(spec: &Value) -> Option<&str> {
    let m = spec.as_object()?;
    if m.len() != 1 {
        return None;
    }
    let d = m.values().next()?.as_object()?;
    let path = d.get("nested")?.as_object()?.get("path")?.as_str()?;
    if d.keys().any(|k| !matches!(k.as_str(), "nested" | "aggs" | "aggregations")) {
        return None;
    }
    let sub = d.get("aggs").or_else(|| d.get("aggregations"))?.as_object()?;
    whole_path_subs(Some(sub)).then_some(path)
}

fn whole_path_subs(sub: Option<&Map<String, Value>>) -> bool {
    let Some(sub) = sub.filter(|m| !m.is_empty()) else { return false };
    if super::DOC_WEIGHTS.with(|w| w.borrow().is_some()) {
        return false;
    }
    sub.values().all(|def| {
        let Some(o) = def.as_object() else { return false };
        o.keys().all(|k| k == "date_histogram" || k == "meta")
            && o.get("date_histogram").and_then(Value::as_object).is_some_and(|h| !h.contains_key("missing") && !h.contains_key("hard_bounds") && !h.contains_key("script"))
    })
}

/// Matching docs of a segment from which a grid over shape doc values tiles
/// them in chunks on the pool, and the chunk size.
const GRID_PAR_MIN_DOCS: usize = 4096;
const GRID_CHUNK_DOCS: usize = 1024;
