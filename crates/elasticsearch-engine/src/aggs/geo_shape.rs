//! Shape doc values for the geo and cartesian aggregations: `geo_bounds`,
//! `geo_centroid`, `geotile_grid`, `geohash_grid` and `geohex_grid` over
//! `geo_shape` fields, `cartesian_bounds` / `cartesian_centroid` over
//! `shape` / `point` fields.
//!
//! ES keeps a per-doc "shape doc value" holding the tessellated shape, its
//! extent and a weighted centroid (`CentroidCalculator`). Here a doc's
//! values are decoded from the encoded geometry doc values, dateline-cut
//! like ES indexes them (`GeoShapeIndexer.normalize`), and the same
//! extent / centroid / grid-cell relations are computed from it.

use super::{Ctx, Docs};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use crate::values;
use rustic_core::fieldtypes::shape::{self, Geometry, Pt};
use rustic_core::geo;
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Doc values.

impl<'a, 'b> Ctx<'a, 'b> {
    /// Declared type of field `f` in the first segment that maps it.
    pub(super) fn declared_type(&self, f: &str) -> Option<EsType> {
        self.segs.iter().find_map(|s| s.lookup.get(f)).map(|rf| rf.declared)
    }

    /// Is `name` a `geo_shape` runtime field of segment `seg`'s index?
    pub(super) fn is_runtime_shape(&self, name: &str, seg: usize) -> bool {
        crate::script::runtime_field(Some(self.segs[seg].index), name).is_some_and(|(rt, i)| rt.fields[i].ty == crate::script::RtType::GeoShape)
    }

    /// The shape of `doc` (all values of a multi-valued field as one
    /// collection), dateline-normalized for `geo_shape`. `None`: no value.
    pub(super) fn doc_shape(&self, name: &str, seg: usize, doc: DocId) -> Option<Geometry> {
        self.doc_shape_as(name, seg, doc, true)
    }

    /// [`Self::doc_shape`]; `normalize: false` keeps the geometry as
    /// indexed (ES computes shape centroids before the dateline cut).
    pub(super) fn doc_shape_as(&self, name: &str, seg: usize, doc: DocId, normalize: bool) -> Option<Geometry> {
        if self.is_runtime_shape(name, seg) {
            let mut cache = self.rt_shapes.borrow_mut();
            let vs = match cache.entry((seg, name.to_string())) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(crate::script::search::ValueSource::runtime(name, self.segs[seg].index)?),
            };
            let vals = vs.values(seg, &self.segs[seg].leaf, doc, 0.0, None).ok()?;
            let gs: Vec<Geometry> = vals
                .iter()
                .filter_map(|v| match v {
                    crate::script::Val::Str(s) => shape::decode(s),
                    _ => None,
                })
                .map(|g| if normalize { shape::normalize_geo(&g) } else { g })
                .collect();
            return match gs.len() {
                0 => None,
                1 => gs.into_iter().next(),
                _ => Some(Geometry::Collection(gs)),
            };
        }
        let f = self.segs[seg].lookup.get(name)?;
        if !crate::fieldtypes::is_shape(f.declared) {
            return None;
        }
        let id = f.core_id?;
        let geo = f.declared == EsType::GeoShape;
        let mut gs: Vec<Geometry> = self.segs[seg]
            .leaf
            .keyword_values(id, doc)
            .iter()
            .filter_map(|v| shape::decode(v))
            .map(|g| if geo && normalize { shape::normalize_geo(&g) } else { g })
            .collect();
        match gs.len() {
            0 => None,
            1 => gs.pop(),
            _ => Some(Geometry::Collection(gs)),
        }
    }

    /// The shape doc value of `doc` for `geo_shape` field `name`: read from
    /// the segment's (or memtable's) doc values, or built from a runtime
    /// field's geometries as ES builds a runtime shape's
    /// (`BinaryShapeDocValuesField`: the normalized values tessellated, the
    /// centroid over the values as emitted). `None`: no value.
    pub(super) fn shape_value(&self, name: &str, seg: usize, doc: DocId) -> Option<DocShape<'_>> {
        if self.is_runtime_shape(name, seg) {
            let g = self.doc_shape_as(name, seg, doc, false)?;
            return runtime_shape_value(&g).map(DocShape::Owned);
        }
        let f = self.segs[seg].lookup.get(name)?;
        if f.declared != EsType::GeoShape {
            return None;
        }
        let id = f.core_id?;
        match self.segs[seg].leaf {
            rustic_core::query::LeafReader::Segment(s) => s.shape_data(id)?.values.as_ref()?.get(doc).map(DocShape::Borrowed),
            rustic_core::query::LeafReader::Memtable(w, _) => w.shape_buf(id)?.value(doc).map(DocShape::Borrowed),
        }
    }

    /// Validate the field of a geo aggregation against `allowed` types.
    pub(super) fn check_field_type(&self, f: &str, t: &str, allowed: &[EsType]) -> Result<()> {
        for s in self.segs {
            if let Some(rf) = s.lookup.get(f) {
                if !allowed.contains(&rf.declared) {
                    let mut ty = rf.declared.name().to_string();
                    if rf.declared == EsType::GeoPoint && rf.time_series_metric.as_deref() == Some("position") {
                        ty.push_str("][position");
                    }
                    return Err(EsError::illegal_argument(format!("Field [{f}] of type [{ty}] is not supported for aggregation [{t}]")));
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // geo_bounds / geo_centroid over geo_shape.

    /// Extent of every value (points and shapes) of geo field `f`: ES
    /// `GeoShapeBoundsAggregator` over each doc's doc-value extent
    /// ([`shape::ShapeExtent`], read from the values' metadata, computed
    /// for values without it and for runtime shapes).
    pub(super) fn geo_extent(&self, f: &str, docs: &Docs) -> GeoExtent {
        let mut ext = GeoExtent::default();
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            let points = self.segs[s].lookup.get(f).is_some_and(|rf| rf.declared == EsType::GeoPoint);
            let r = self.shape_reader(f, s);
            if !points && matches!(r, ShapeReader::None) {
                continue;
            }
            // Every doc of a segment without deletes: the union of its
            // values' extents (the same min / max as doc by doc).
            if let (ShapeReader::Column(c), rustic_core::query::LeafReader::Segment(seg)) = (&r, &self.segs[s].leaf) {
                if !points && v.len() as u32 == seg.num_docs && seg.num_live_docs() == seg.num_docs {
                    if let Some(e) = c.union_extent() {
                        ext.add_extent(&e);
                    }
                    continue;
                }
            }
            for &(d, _) in v {
                if points {
                    for (lat, lon) in self.geo_points(f, s, d) {
                        ext.add(lat, lon);
                    }
                }
                if let Some(e) = self.read_shape(&r, f, s, d).and_then(|x| x.value().extent()) {
                    ext.add_extent(&e);
                }
            }
        }
        ext
    }

    /// `geo_centroid` over `geo_shape` values: ES `GeoShapeCentroidAggregator`
    /// per shard over each doc's doc-value centroid ([`shape::ShapeCentroid`]:
    /// read from the values' metadata, computed for values without it and
    /// for runtime shapes), reduced as `InternalCentroid` reduces shard
    /// results (only the highest dimension's shards count; `geo_point`
    /// indices weighted by count). A part per index stands for its shard.
    pub(super) fn geo_shape_centroid(&self, f: &str, docs: &Docs) -> Value {
        let mut parts: Vec<(&str, CentroidPart)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            let index = self.segs[s].index;
            let is_shape = self.segs[s].lookup.get(f).is_some_and(|rf| rf.declared == EsType::GeoShape) || self.is_runtime_shape(f, s);
            let k = match parts.iter().position(|(i, p)| *i == index && p.is_shape() == is_shape) {
                Some(k) => k,
                None => {
                    parts.push((index, if is_shape { CentroidPart::Shape(ShapeCentroidAcc::default()) } else { CentroidPart::Points(PointCentroidAcc::default()) }));
                    parts.len() - 1
                }
            };
            let r = if is_shape { self.shape_reader(f, s) } else { ShapeReader::None };
            // A segment's doc-value column: one tight loop over its headers.
            if let (CentroidPart::Shape(acc), ShapeReader::Column(c)) = (&mut parts[k].1, &r) {
                for &(d, _) in v {
                    if let Some((lon, lat, dim, weight)) = c.centroid(d) {
                        acc.collect(Some(&shape::ShapeCentroid { lat, lon, dim, weight }));
                    }
                }
                continue;
            }
            for &(d, _) in v {
                match &mut parts[k].1 {
                    CentroidPart::Shape(acc) => {
                        // The doc value's header (a doc without one adds
                        // nothing).
                        if let ShapeReader::Column(c) = &r {
                            if let Some((lon, lat, dim, weight)) = c.centroid(d) {
                                acc.collect(Some(&shape::ShapeCentroid { lat, lon, dim, weight }));
                            }
                        } else if let Some(sv) = self.read_shape(&r, f, s, d) {
                            let x = sv.value();
                            acc.collect(Some(&shape::ShapeCentroid { lat: x.centroid_y(), lon: x.centroid_x(), dim: x.dimension(), weight: x.weight() }));
                        }
                    }
                    CentroidPart::Points(acc) => {
                        for (lat, lon) in self.geo_points(f, s, d) {
                            acc.add(lat, lon);
                        }
                    }
                }
            }
        }
        let (loc, count) = reduce_centroid_parts(parts.iter().map(|(_, p)| p));
        match loc {
            Some((lat, lon)) => json!({ "location": { "lat": values::java_double(lat), "lon": values::java_double(lon) }, "count": count }),
            None => json!({ "count": count }),
        }
    }

    // -----------------------------------------------------------------------
    // cartesian_bounds / cartesian_centroid.

    pub(super) fn cartesian_agg(&self, name: &str, t: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let f = super::field_name(o, name, t)?;
        self.check_field_type(f, t, &[EsType::Point, EsType::Shape])?;
        let point = self.declared_type(f) == Some(EsType::Point);
        let fl = |v: f64| v as f32 as f64;
        if t == "cartesian_bounds" {
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (s, v) in docs.iter().enumerate() {
                for &(d, _) in v {
                    if let Some(g) = self.doc_shape(f, s, d) {
                        for_each_vertex(&g, &mut |(x, y)| {
                            let (x, y) = (fl(x), fl(y));
                            min_x = min_x.min(x);
                            max_x = max_x.max(x);
                            min_y = min_y.min(y);
                            max_y = max_y.max(y);
                        });
                    }
                }
            }
            if min_x.is_infinite() {
                return Ok(json!({}));
            }
            return Ok(json!({ "bounds": {
                "top_left": { "x": values::java_double(min_x), "y": values::java_double(max_y) },
                "bottom_right": { "x": values::java_double(max_x), "y": values::java_double(min_y) },
            }}));
        }
        // cartesian_centroid: points average every value; shapes weight each
        // doc's centroid by its dimension (like geo_centroid).
        let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u64);
        let mut acc = Centroid::default();
        for (s, v) in docs.iter().enumerate() {
            for &(d, _) in v {
                let Some(g) = self.doc_shape(f, s, d) else { continue };
                if point {
                    for_each_vertex(&g, &mut |(x, y)| {
                        sx += fl(x);
                        sy += fl(y);
                        n += 1;
                    });
                } else {
                    let mut c = Centroid::default();
                    c.visit(&g);
                    if let Some((x, y)) = c.value() {
                        acc.add(fl(x), fl(y), c.w, c.dim);
                    }
                    n += 1;
                }
            }
        }
        let loc = if point { (n > 0).then(|| (sx / n as f64, sy / n as f64)) } else { acc.value() };
        Ok(match loc {
            Some((x, y)) if n > 0 => json!({ "location": { "x": values::java_double(x), "y": values::java_double(y) }, "count": n }),
            _ => json!({ "count": n }),
        })
    }
}

/// One segment's shape doc values of a field, resolved once for a pass
/// over its docs (no per-doc field lookups).
pub(crate) enum ShapeReader<'a> {
    None,
    Column(&'a rustic_core::fieldtypes::shape_dv::ShapeDvColumn),
    Memtable(&'a rustic_core::fieldtypes::shape_field::ShapeBuf),
    Runtime,
}

impl<'a, 'b> Ctx<'a, 'b> {
    /// The reader of `geo_shape` field `name` in segment `seg`.
    pub(crate) fn shape_reader(&self, name: &str, seg: usize) -> ShapeReader<'_> {
        if self.is_runtime_shape(name, seg) {
            return ShapeReader::Runtime;
        }
        let Some(f) = self.segs[seg].lookup.get(name) else { return ShapeReader::None };
        if f.declared != EsType::GeoShape {
            return ShapeReader::None;
        }
        let Some(id) = f.core_id else { return ShapeReader::None };
        match self.segs[seg].leaf {
            rustic_core::query::LeafReader::Segment(s) => s.shape_data(id).and_then(|d| d.values.as_ref()).map_or(ShapeReader::None, ShapeReader::Column),
            rustic_core::query::LeafReader::Memtable(w, _) => w.shape_buf(id).map_or(ShapeReader::None, ShapeReader::Memtable),
        }
    }

    /// `doc`'s value through `r` (`name`, `seg`: for runtime fields).
    pub(crate) fn read_shape<'r>(&'r self, r: &ShapeReader<'r>, name: &str, seg: usize, doc: DocId) -> Option<DocShape<'r>> {
        match r {
            ShapeReader::None => None,
            ShapeReader::Column(c) => c.get(doc).map(DocShape::Borrowed),
            ShapeReader::Memtable(b) => b.value(doc).map(DocShape::Borrowed),
            ShapeReader::Runtime => self.shape_value(name, seg, doc),
        }
    }
}

/// A doc's shape doc value: the stored one, or one built for a runtime
/// field.
pub(crate) enum DocShape<'a> {
    Borrowed(rustic_core::fieldtypes::shape_dv::ShapeValue<'a>),
    Owned(Vec<u8>),
}

impl DocShape<'_> {
    pub(crate) fn value(&self) -> rustic_core::fieldtypes::shape_dv::ShapeValue<'_> {
        match self {
            DocShape::Borrowed(v) => *v,
            DocShape::Owned(b) => rustic_core::fieldtypes::shape_dv::ShapeValue::new(b),
        }
    }
}

/// The shape doc value of a geometry as indexed (its values normalized and
/// tessellated, the centroid over `g` as given). `None`: nothing to index.
pub(crate) fn runtime_shape_value(g: &Geometry) -> Option<Vec<u8>> {
    let mut tris = Vec::new();
    rustic_core::fieldtypes::tessellator::geometry_triangles(&shape::normalized(g), &mut tris).ok()?;
    if tris.is_empty() {
        return None;
    }
    let mut c = shape::CentroidCalc::default();
    c.add(g);
    rustic_core::fieldtypes::shape_field::doc_value(&tris, &c).ok()
}

/// Every vertex of a geometry (envelopes: their corners).
pub(super) fn for_each_vertex(g: &Geometry, f: &mut dyn FnMut(Pt)) {
    match g {
        Geometry::Point(p) => f(*p),
        Geometry::MultiPoint(v) | Geometry::LineString(v) => v.iter().for_each(|p| f(*p)),
        Geometry::MultiLineString(v) | Geometry::Polygon(v) => v.iter().flatten().for_each(|p| f(*p)),
        Geometry::MultiPolygon(v) => v.iter().flatten().flatten().for_each(|p| f(*p)),
        Geometry::Collection(v) => v.iter().for_each(|x| for_each_vertex(x, f)),
        Geometry::Envelope { min_x, max_x, max_y, min_y } => {
            f((*min_x, *max_y));
            f((*max_x, *min_y));
        }
    }
}

/// ES `InternalGeoBounds` accumulator (positive / negative longitudes kept
/// apart so `wrap_longitude` can pick the narrower box).
#[derive(Clone, Copy)]
pub(super) struct GeoExtent {
    top: f64,
    bottom: f64,
    pos_left: f64,
    pos_right: f64,
    neg_left: f64,
    neg_right: f64,
}

impl Default for GeoExtent {
    fn default() -> Self {
        GeoExtent {
            top: f64::NEG_INFINITY,
            bottom: f64::INFINITY,
            pos_left: f64::INFINITY,
            pos_right: f64::NEG_INFINITY,
            neg_left: f64::INFINITY,
            neg_right: f64::NEG_INFINITY,
        }
    }
}

impl GeoExtent {
    pub(super) fn add(&mut self, lat: f64, lon: f64) {
        self.top = self.top.max(lat);
        self.bottom = self.bottom.min(lat);
        if lon >= 0.0 {
            self.pos_left = self.pos_left.min(lon);
            self.pos_right = self.pos_right.max(lon);
        } else {
            self.neg_left = self.neg_left.min(lon);
            self.neg_right = self.neg_right.max(lon);
        }
    }

    /// One doc's extent (ES `GeoShapeBoundsAggregator.collect` over
    /// `ShapeValues.BoundingBox`: decoded, unset sides infinite).
    pub(super) fn add_extent(&mut self, e: &shape::ShapeExtent) {
        self.top = self.top.max(geo::decode_lat(e.top));
        self.bottom = self.bottom.min(geo::decode_lat(e.bottom));
        if e.has_neg() {
            self.neg_left = self.neg_left.min(geo::decode_lon(e.neg_left));
            self.neg_right = self.neg_right.max(geo::decode_lon(e.neg_right));
        }
        if e.has_pos() {
            self.pos_left = self.pos_left.min(geo::decode_lon(e.pos_left));
            self.pos_right = self.pos_right.max(geo::decode_lon(e.pos_right));
        }
    }

    /// Cross-shard partial state (`hooks/agg_partial.rs`).
    pub(super) fn state(&self) -> Option<Value> {
        let fin = |v: f64| if v.is_finite() { json!(v) } else { Value::Null };
        self.top.is_finite().then(|| json!({ "top": self.top, "bottom": self.bottom, "pl": fin(self.pos_left), "pr": fin(self.pos_right), "nl": fin(self.neg_left), "nr": fin(self.neg_right) }))
    }

    pub(super) fn to_json(&self, wrap: bool) -> Value {
        if self.top.is_infinite() {
            return json!({});
        }
        let (left, right) = if self.pos_left.is_infinite() {
            (self.neg_left, self.neg_right)
        } else if self.neg_left.is_infinite() {
            (self.pos_left, self.pos_right)
        } else if wrap {
            let unwrapped = self.pos_right - self.neg_left;
            let wrapped = (180.0 - self.pos_left) - (-180.0 - self.neg_right);
            if unwrapped <= wrapped {
                (self.neg_left, self.pos_right)
            } else {
                (self.pos_left, self.neg_right)
            }
        } else {
            (self.neg_left, self.pos_right)
        };
        json!({ "bounds": {
            "top_left": { "lat": values::java_double(self.top), "lon": values::java_double(left) },
            "bottom_right": { "lat": values::java_double(self.bottom), "lon": values::java_double(right) },
        }})
    }
}

// ---------------------------------------------------------------------------
// geo_centroid over shapes: per-shard state and reduce.

/// ES `GeoShapeCentroidAggregator` state of one bucket.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct ShapeCentroidAcc {
    lat: shape::CompSum,
    lon: shape::CompSum,
    weight: shape::CompSum,
    dim: shape::ShapeDim,
    count: u64,
}

impl ShapeCentroidAcc {
    /// A doc with a value; `c`: its doc-value centroid.
    pub(super) fn collect(&mut self, c: Option<&shape::ShapeCentroid>) {
        self.count += 1;
        let Some(c) = c else { return };
        let w = c.weight;
        match self.dim.cmp(&c.dim) {
            std::cmp::Ordering::Less => {
                self.lat.reset(w * c.lat_deg(), 0.0);
                self.lon.reset(w * c.lon_deg(), 0.0);
                self.weight.reset(w, 0.0);
                self.dim = c.dim;
            }
            std::cmp::Ordering::Equal => {
                self.lat.add(w * c.lat_deg());
                self.lon.add(w * c.lon_deg());
                self.weight.add(w);
            }
            std::cmp::Ordering::Greater => {}
        }
    }
}

/// ES `GeoCentroidAggregator` state (`geo_point` values).
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct PointCentroidAcc {
    lat: shape::CompSum,
    lon: shape::CompSum,
    count: u64,
}

impl PointCentroidAcc {
    pub(super) fn add(&mut self, lat: f64, lon: f64) {
        self.lat.add(lat);
        self.lon.add(lon);
        self.count += 1;
    }
}

/// One shard's `geo_centroid` result before the reduce.
pub(super) enum CentroidPart {
    Shape(ShapeCentroidAcc),
    Points(PointCentroidAcc),
}

impl CentroidPart {
    fn is_shape(&self) -> bool {
        matches!(self, CentroidPart::Shape(_))
    }
}

/// ES `InternalCentroid` reduce (9.5.5, with shape weighted sums): shard
/// results with a positive weight combine by dimension (a higher one
/// replaces the sums and the count, a lower one is ignored), plain sums in
/// shard order; point results weight by count. `((lat, lon), count)`.
pub(super) fn reduce_centroid_parts<'p>(parts: impl IntoIterator<Item = &'p CentroidPart>) -> (Option<(f64, f64)>, u64) {
    let (mut first_sum, mut second_sum, mut total_count) = (f64::NAN, f64::NAN, 0u64);
    let (mut c_first, mut c_second, mut c_weight, mut shape_count) = (0.0f64, 0.0f64, 0.0f64, 0u64);
    let mut c_dim = shape::ShapeDim::Point;
    let mut has_shape = false;
    for p in parts {
        match p {
            CentroidPart::Shape(a) if a.count > 0 => {
                let w = a.weight.value;
                if w > 0.0 {
                    if !has_shape || a.dim > c_dim {
                        c_first = a.lat.value;
                        c_second = a.lon.value;
                        c_weight = w;
                        shape_count = a.count;
                        c_dim = a.dim;
                        has_shape = true;
                    } else if a.dim == c_dim {
                        c_first += a.lat.value;
                        c_second += a.lon.value;
                        c_weight += w;
                        shape_count += a.count;
                    }
                }
            }
            CentroidPart::Points(a) if a.count > 0 => {
                let n = a.count as f64;
                let (lat, lon) = (a.lat.value / n, a.lon.value / n);
                if has_shape {
                    c_first += n * lat;
                    c_second += n * lon;
                    c_weight += n;
                    shape_count += a.count;
                } else {
                    total_count += a.count;
                    if first_sum.is_nan() {
                        first_sum = n * lat;
                        second_sum = n * lon;
                    } else {
                        first_sum += n * lat;
                        second_sum += n * lon;
                    }
                }
            }
            _ => {}
        }
    }
    if has_shape {
        ((c_weight > 0.0).then(|| (c_first / c_weight, c_second / c_weight)), shape_count)
    } else {
        ((total_count > 0).then(|| (first_sum / total_count as f64, second_sum / total_count as f64)), total_count)
    }
}

// ---------------------------------------------------------------------------
// Centroid (ES `CentroidCalculator`).

/// Dimensional shape type: only the highest dimension seen contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(super) enum Dim {
    #[default]
    Point,
    Line,
    Polygon,
}

#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Centroid {
    sx: f64,
    sy: f64,
    pub(super) w: f64,
    pub(super) dim: Dim,
}

impl Centroid {
    pub(super) fn add(&mut self, x: f64, y: f64, w: f64, dim: Dim) {
        if dim == self.dim {
            self.sx += x * w;
            self.sy += y * w;
            self.w += w;
        } else if dim > self.dim {
            self.sx = x * w;
            self.sy = y * w;
            self.w = w;
            self.dim = dim;
        }
    }

    /// `(x, y)` of the weighted centroid; `None` when nothing was added.
    pub(super) fn value(&self) -> Option<(f64, f64)> {
        (self.w != 0.0).then(|| (self.sx / self.w, self.sy / self.w))
    }

    fn point(&mut self, p: Pt) {
        if self.dim == Dim::Point {
            self.add(p.0, p.1, 1.0, Dim::Point);
        }
    }

    fn line(&mut self, l: &[Pt]) {
        for w in l.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (dx, dy) = (a.0 - b.0, a.1 - b.1);
            let len = (dx * dx + dy * dy).sqrt();
            if len == 0.0 {
                // A degenerate segment counts as a point.
                self.add(a.0, a.1, 1.0, Dim::Point);
            } else {
                self.add((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0, len, Dim::Line);
            }
        }
    }

    fn polygon(&mut self, rings: &[Vec<Pt>]) {
        let parts: Vec<(f64, f64, f64)> = rings.iter().enumerate().map(|(i, r)| ring_centroid(r, i > 0)).collect();
        let total: f64 = parts.iter().map(|p| p.2).sum();
        if total == 0.0 && self.dim != Dim::Polygon {
            if let Some(r) = rings.first() {
                self.line(r);
            }
        } else {
            for (x, y, w) in parts {
                if w != 0.0 {
                    self.add(x, y, w, Dim::Polygon);
                }
            }
        }
    }

    pub(super) fn visit(&mut self, g: &Geometry) {
        match g {
            Geometry::Point(p) => self.point(*p),
            Geometry::MultiPoint(v) => v.iter().for_each(|p| self.point(*p)),
            Geometry::LineString(l) => {
                if self.dim != Dim::Polygon {
                    self.line(l)
                }
            }
            Geometry::MultiLineString(ls) => {
                for l in ls {
                    if self.dim != Dim::Polygon {
                        self.line(l)
                    }
                }
            }
            Geometry::Polygon(r) => self.polygon(r),
            Geometry::MultiPolygon(ps) => ps.iter().for_each(|p| self.polygon(p)),
            Geometry::Collection(v) => v.iter().for_each(|x| self.visit(x)),
            Geometry::Envelope { min_x, max_x, max_y, min_y } => {
                let (dx, dy) = (max_x - min_x, max_y - min_y);
                let (cx, cy) = ((max_x + min_x) / 2.0, (max_y + min_y) / 2.0);
                if dx != 0.0 && dy != 0.0 {
                    self.add(cx, cy, (dx * dy).abs(), Dim::Polygon);
                } else if dx != 0.0 || dy != 0.0 {
                    if self.dim != Dim::Polygon {
                        self.add(cx, cy, (dx * dx + dy * dy).sqrt(), Dim::Line);
                    }
                } else {
                    self.point((*min_x, *min_y));
                }
            }
        }
    }
}

/// Area-weighted centroid of a closed ring: `(x, y, signed weight)`, the
/// weight negative for holes.
fn ring_centroid(r: &[Pt], hole: bool) -> (f64, f64, f64) {
    let mut area = 0.0;
    let (mut sx, mut sy) = (0.0, 0.0);
    for w in r.windows(2) {
        let (a, b) = (w[0], w[1]);
        let twice = a.0 * b.1 - b.0 * a.1;
        area += twice;
        sx += twice * (a.0 + b.0);
        sy += twice * (a.1 + b.1);
    }
    let area = area / 2.0;
    if area == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let sign = if hole { -1.0 } else { 1.0 };
    (sx / (6.0 * area), sy / (6.0 * area), sign * area.abs())
}

// ---------------------------------------------------------------------------
// Grid cells of shapes.

/// Grid kinds of the geo grid aggregations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Grid {
    Tile,
    Hash,
    Hex,
}

/// `(top, left, bottom, right)` of the `bounds` of a grid aggregation.
pub(super) type BBox = (f64, f64, f64, f64);

/// ES `Geo{Tile,Hash,Hex}BoundedPredicate`: does a cell share an interior
/// point with the bounds?
#[derive(Clone)]
pub(super) struct CellPred {
    grid: Grid,
    bbox: BBox,
    crosses: bool,
    // geotile index ranges [min, max).
    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
}

impl CellPred {
    pub(super) fn new(grid: Grid, precision: u32, bbox: BBox) -> CellPred {
        let (top, left, bottom, right) = bbox;
        let mut p = CellPred { grid, bbox, crosses: right < left, min_x: 0, max_x: 0, min_y: 0, max_y: 0 };
        if grid == Grid::Tile {
            let (x0, y0) = geo::geotile(top, left, precision);
            let t0 = crate::fieldtypes::geo::tile_bbox(precision, x0, y0);
            p.min_x = if t0.3 == left { x0 as i64 + 1 } else { x0 as i64 };
            p.min_y = if t0.2 == top { y0 as i64 + 1 } else { y0 as i64 };
            let (x1, y1) = geo::geotile(bottom, right, precision);
            let t1 = crate::fieldtypes::geo::tile_bbox(precision, x1, y1);
            p.max_x = if t1.1 == right { x1 as i64 } else { x1 as i64 + 1 };
            p.max_y = if t1.0 == bottom { y1 as i64 } else { y1 as i64 + 1 };
        }
        p
    }

    pub(super) fn tile(&self, x: u32, y: u32) -> bool {
        let (x, y) = (x as i64, y as i64);
        if y >= self.min_y && y < self.max_y {
            if self.crosses {
                x < self.max_x || x >= self.min_x
            } else {
                x >= self.min_x && x < self.max_x
            }
        } else {
            false
        }
    }

    /// A cell given by its `(top, left, bottom, right)` box.
    pub(super) fn rect(&self, r: BBox) -> bool {
        let (top, left, bottom, right) = self.bbox;
        if top > r.2 && bottom < r.0 {
            if self.crosses || r.3 < r.1 {
                left < r.3 || right > r.1
            } else {
                left < r.3 && right > r.1
            }
        } else {
            false
        }
    }

    pub(super) fn key(&self, key: &str, precision: u32) -> bool {
        match self.grid {
            Grid::Tile => {
                let mut it = key.split('/').skip(1).map(|v| v.parse::<u32>().unwrap_or(0));
                let (x, y) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
                let _ = precision;
                self.tile(x, y)
            }
            Grid::Hash => geo::geohash_bbox(key).is_some_and(|b| self.rect(b)),
            Grid::Hex => super::geo_hex::hex_bbox_of(key).is_some_and(|b| self.rect(b)),
        }
    }
}

/// The cell of a point.
pub(super) fn point_cell(grid: Grid, precision: u32, lat: f64, lon: f64) -> Option<String> {
    Some(match grid {
        Grid::Tile => {
            let (x, y) = geo::geotile(lat, lon, precision);
            format!("{precision}/{x}/{y}")
        }
        Grid::Hash => geo::geohash(lat, lon, precision as usize),
        Grid::Hex => super::geo_hex::point_cell(lat, lon, precision)?,
    })
}

/// ES `SphericalMercatorUtils.lonToSphericalMercator`.
pub(crate) fn lon_mercator(lon: f64) -> f64 {
    lon * (20037508.34 / 180.0)
}

/// ES `SphericalMercatorUtils.latToSphericalMercator` (8.19: Lucene's
/// `SloppyMath.cos` and ES `FastMath.log`).
pub(crate) fn lat_mercator(lat: f64) -> f64 {
    let cos = crate::sloppy_math::cos((90.0 + lat) * std::f64::consts::PI / 360.0);
    let tan_lower_limit = f64::from_bits(1).tan();
    let tan = tan_lower_limit.max((1.0 - cos * cos).sqrt() / cos);
    let y = crate::sloppy_math::es_log(tan) / (std::f64::consts::PI / 180.0);
    y * (20037508.34 / 180.0)
}
