//! Geo points (Lucene `LatLonPoint` / `LatLonDocValuesField`).
//!
//! Latitude and longitude are quantized to sortable 32-bit ints exactly as
//! Lucene does (`GeoEncodingUtils`), so a point round-trips with < 1e-7°
//! error. Each sealed segment keeps, per geo field:
//!
//! * **doc values**: every `(doc, lat, lon)` sorted by doc (binary-searched
//!   per doc; sorting, aggregations, `geo_values`);
//! * a **KD tree** in the spirit of Lucene's BKD: points are recursively
//!   split on their widest dimension into leaves of at most
//!   [`LEAF_POINTS`] points, each leaf carrying its bounding box. Queries
//!   ([`GeoShape`]) relate every leaf box to the shape: disjoint leaves are
//!   skipped, leaves fully inside contribute all their docs without
//!   per-point checks, only crossing leaves test points.
//!
//! The memtable keeps the same doc-sorted list and scans it linearly.
//! Distances are haversine on the mean earth radius (ES `arc` distance).

use crate::codec::directory::FileSlice;
use crate::error::{Result, RusticError};
use crate::types::DocId;

/// Mean earth radius in meters (Lucene / ES `GeoUtils.EARTH_MEAN_RADIUS`).
pub const EARTH_MEAN_RADIUS_METERS: f64 = 6_371_008.7714;

/// Max points per KD-tree leaf.
pub const LEAF_POINTS: usize = 512;

const LAT_DECODE: f64 = 180.0 / 4294967296.0;
const LON_DECODE: f64 = 360.0 / 4294967296.0;

/// Quantize a latitude in `[-90, 90]` (Lucene `encodeLatitude`).
#[inline]
pub fn encode_lat(lat: f64) -> i32 {
    let lat = if lat >= 90.0 { 90.0f64.next_down_compat() } else { lat.max(-90.0) };
    (lat / LAT_DECODE).floor() as i32
}

/// Quantize a longitude in `[-180, 180]` (Lucene `encodeLongitude`).
#[inline]
pub fn encode_lon(lon: f64) -> i32 {
    let lon = if lon >= 180.0 { 180.0f64.next_down_compat() } else { lon.max(-180.0) };
    (lon / LON_DECODE).floor() as i32
}

/// Lucene `encodeLatitudeCeil`.
#[inline]
pub fn encode_lat_ceil(lat: f64) -> i32 {
    let lat = if lat >= 90.0 { 90.0f64.next_down_compat() } else { lat.max(-90.0) };
    (lat / LAT_DECODE).ceil() as i32
}

/// Lucene `encodeLongitudeCeil`.
#[inline]
pub fn encode_lon_ceil(lon: f64) -> i32 {
    let lon = if lon >= 180.0 { 180.0f64.next_down_compat() } else { lon.max(-180.0) };
    (lon / LON_DECODE).ceil() as i32
}

/// Inverse of [`encode_lat`].
#[inline]
pub fn decode_lat(v: i32) -> f64 {
    v as f64 * LAT_DECODE
}

/// Inverse of [`encode_lon`].
#[inline]
pub fn decode_lon(v: i32) -> f64 {
    v as f64 * LON_DECODE
}

/// The `(lat, lon)` a point reads back as after quantization.
pub fn quantize(lat: f64, lon: f64) -> (f64, f64) {
    (decode_lat(encode_lat(lat)), decode_lon(encode_lon(lon)))
}

trait NextDown {
    fn next_down_compat(self) -> f64;
}

impl NextDown for f64 {
    fn next_down_compat(self) -> f64 {
        // f64::next_down is not stable on rustc 1.87.
        f64::from_bits(self.to_bits() - 1)
    }
}

/// Great-circle (haversine) distance in meters: ES `arc` distance.
pub fn haversine_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dlat = p2 - p1;
    let dlon = (lon2 - lon1).to_radians();
    let s1 = (dlat / 2.0).sin();
    let s2 = (dlon / 2.0).sin();
    let h = s1 * s1 + p1.cos() * p2.cos() * s2 * s2;
    2.0 * EARTH_MEAN_RADIUS_METERS * h.sqrt().min(1.0).asin()
}

/// ES `plane` distance (equirectangular approximation), in meters.
pub fn plane_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let x = (lon2 - lon1).to_radians() * ((lat1 + lat2) / 2.0).to_radians().cos();
    let y = (lat2 - lat1).to_radians();
    (x * x + y * y).sqrt() * EARTH_MEAN_RADIUS_METERS
}

/// A query shape over geo points.
#[derive(Clone, Debug, PartialEq)]
pub enum GeoShape {
    /// Lat/lon rectangle; `left > right` crosses the dateline.
    BoundingBox { top: f64, left: f64, bottom: f64, right: f64 },
    /// Points within `meters` (haversine) of the center.
    Distance { lat: f64, lon: f64, meters: f64 },
    /// Points inside the polygon (`(lat, lon)` vertices, closed or open).
    Polygon { points: Vec<(f64, f64)> },
}

/// How a leaf box relates to a shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    /// Every point of the box is in the shape.
    Inside,
    /// No point of the box is in the shape.
    Outside,
    /// Unknown: test the points.
    Crosses,
}

/// Inclusive lon ranges of a box that may cross the dateline.
fn lon_ranges(left: f64, right: f64) -> [(f64, f64); 2] {
    if left <= right {
        [(left, right), (1.0, -1.0)]
    } else {
        [(left, 180.0), (-180.0, right)]
    }
}

impl GeoShape {
    /// Whether the point is in the shape.
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        match self {
            GeoShape::BoundingBox { top, left, bottom, right } => {
                lat >= *bottom && lat <= *top && lon_ranges(*left, *right).iter().any(|(a, b)| lon >= *a && lon <= *b)
            }
            // Lucene's distance query (ES `geo_distance`): the sloppy
            // haversine sort key against the radius's.
            GeoShape::Distance { lat: clat, lon: clon, meters } => within_distance_query(*clat, *clon, cached_query_key(*meters), lat, lon),
            GeoShape::Polygon { points } => point_in_polygon(points, lat, lon),
        }
    }

    /// Conservative bounding box `(min_lat, max_lat, lon ranges)`.
    fn bbox(&self) -> (f64, f64, [(f64, f64); 2]) {
        match self {
            GeoShape::BoundingBox { top, left, bottom, right } => (*bottom, *top, lon_ranges(*left, *right)),
            GeoShape::Distance { lat, lon, meters } => circle_bbox(*lat, *lon, *meters),
            GeoShape::Polygon { points } => {
                let (mut a, mut b, mut c, mut d) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
                for (la, lo) in points {
                    a = a.min(*la);
                    b = b.max(*la);
                    c = c.min(*lo);
                    d = d.max(*lo);
                }
                (a, b, [(c, d), (1.0, -1.0)])
            }
        }
    }

    /// Relate a lat/lon box (decoded degrees, inclusive) to the shape.
    pub fn relate(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Relation {
        let (blat0, blat1, ranges) = self.bbox();
        let eps = 1e-9;
        if max_lat < blat0 - eps || min_lat > blat1 + eps {
            return Relation::Outside;
        }
        if !ranges.iter().any(|(a, b)| a <= b && max_lon >= a - eps && min_lon <= b + eps) {
            return Relation::Outside;
        }
        match self {
            GeoShape::BoundingBox { .. } => {
                let lat_in = min_lat >= blat0 && max_lat <= blat1;
                let lon_in = ranges.iter().any(|(a, b)| a <= b && min_lon >= *a && max_lon <= *b);
                if lat_in && lon_in {
                    Relation::Inside
                } else {
                    Relation::Crosses
                }
            }
            GeoShape::Distance { lat, lon, meters } => {
                // Outside when even the box's nearest point is too far.
                // (Margins cover the sloppy haversine's error, ~1e-7 relative.)
                // Boxes the exact nearest distance is not computed for
                // (more than 90° of longitude away, crossing the dateline)
                // still lie at least their latitude gap away: a great
                // circle changes latitude no faster than a meridian. For a
                // centre at a pole that bound is the distance itself.
                let dmin = min_box_distance(*lat, *lon, min_lat, max_lat, min_lon, max_lon).unwrap_or_else(|| (lat - lat.clamp(min_lat, max_lat)).abs().to_radians() * EARTH_MEAN_RADIUS_METERS);
                if dmin > meters * (1.0 + 1e-6) + 1.0 {
                    return Relation::Outside;
                }
                // A cap of angular radius < 90° is geodesically convex, so
                // the meridian edges are inside when their ends are; along
                // a parallel the distance grows with |Δlon| (up to the
                // antipodal meridian), so those edges peak at the corners.
                // Hence: all four corners inside => the box is inside.
                let quarter = std::f64::consts::FRAC_PI_2 * EARTH_MEAN_RADIUS_METERS;
                let anti = if *lon > 0.0 { lon - 180.0 } else { lon + 180.0 };
                if *meters < quarter * 0.99 && !(anti >= min_lon && anti <= max_lon) {
                    let corners = [(min_lat, min_lon), (min_lat, max_lon), (max_lat, min_lon), (max_lat, max_lon)];
                    let r = meters * (1.0 - 1e-6) - 1.0;
                    if r > 0.0 && corners.iter().all(|(a, b)| haversine_meters(*lat, *lon, *a, *b) <= r) {
                        return Relation::Inside;
                    }
                }
                Relation::Crosses
            }
            GeoShape::Polygon { .. } => Relation::Crosses,
        }
    }
}

/// The great-circle distance (meters) from `(lat, lon)` to the nearest
/// point of a lat/lon box (decoded degrees, `min_lon <= max_lon`), or
/// `None` when it is not computed (boxes more than 90° of longitude away).
/// On a meridian edge the nearest point is at `atan(tan(lat) / cos(dlon))`
/// clamped to the edge (the distance is unimodal along the meridian).
fn min_box_distance(lat: f64, lon: f64, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Option<f64> {
    if min_lon > max_lon {
        return None;
    }
    if lon >= min_lon && lon <= max_lon {
        return Some((lat - lat.clamp(min_lat, max_lat)).abs().to_radians() * EARTH_MEAN_RADIUS_METERS);
    }
    // The nearer meridian edge (longitude differences wrapped to [-180, 180]).
    let wrap = |d: f64| if d > 180.0 { d - 360.0 } else if d < -180.0 { d + 360.0 } else { d };
    let (d0, d1) = (wrap(min_lon - lon).abs(), wrap(max_lon - lon).abs());
    let dlon = d0.min(d1);
    if dlon >= 90.0 {
        return None;
    }
    let edge = if d0 <= d1 { min_lon } else { max_lon };
    let star = (lat.to_radians().tan() / dlon.to_radians().cos()).atan().to_degrees();
    let phi = star.clamp(min_lat, max_lat);
    Some(haversine_meters(lat, lon, phi, edge))
}

/// Bounding box of a circle (Lucene `Rectangle.fromPointDistance`), padded.
fn circle_bbox(lat: f64, lon: f64, meters: f64) -> (f64, f64, [(f64, f64); 2]) {
    let rad_lat = lat.to_radians();
    let rad_lon = lon.to_radians();
    // Padded beyond the sloppy haversine's error (distance queries use it).
    let rad_dist = (meters / EARTH_MEAN_RADIUS_METERS) * (1.0 + 1e-6) + 1e-9;
    let mut min_lat = rad_lat - rad_dist;
    let mut max_lat = rad_lat + rad_dist;
    let half_pi = std::f64::consts::FRAC_PI_2;
    let pi = std::f64::consts::PI;
    let (min_lon, max_lon);
    if min_lat > -half_pi && max_lat < half_pi {
        let s = rad_dist.sin() / rad_lat.cos();
        if s >= 1.0 || rad_dist >= pi {
            min_lon = -pi;
            max_lon = pi;
        } else {
            let delta = s.asin();
            let mut a = rad_lon - delta;
            let mut b = rad_lon + delta;
            if a < -pi {
                a += 2.0 * pi;
            }
            if b > pi {
                b -= 2.0 * pi;
            }
            min_lon = a;
            max_lon = b;
        }
    } else {
        min_lat = min_lat.max(-half_pi);
        max_lat = max_lat.min(half_pi);
        min_lon = -pi;
        max_lon = pi;
    }
    let pad = 1e-5;
    let (lo, hi) = (min_lon.to_degrees() - pad, max_lon.to_degrees() + pad);
    let ranges = if lo <= hi { [(lo, hi), (1.0, -1.0)] } else { [(lo, 180.0), (-180.0, hi)] };
    (min_lat.to_degrees() - pad, max_lat.to_degrees() + pad, ranges)
}

/// Even-odd ray casting over `(lat, lon)` vertices (x = lon, y = lat).
pub fn point_in_polygon(points: &[(f64, f64)], lat: f64, lon: f64) -> bool {
    let n = points.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (yi, xi) = points[i];
        let (yj, xj) = points[j];
        // On an edge counts as inside.
        if on_segment(xi, yi, xj, yj, lon, lat) {
            return true;
        }
        if (yi > lat) != (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn on_segment(x1: f64, y1: f64, x2: f64, y2: f64, x: f64, y: f64) -> bool {
    let cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
    cross.abs() < 1e-12 && x >= x1.min(x2) && x <= x1.max(x2) && y >= y1.min(y2) && y <= y1.max(y2)
}

/// One encoded point of a doc. Laid out as it is serialized (`doc, lat,
/// lon`, little-endian, 12 bytes) and packed (alignment 1), so a sealed
/// segment's points can be read in place from its mapped file wherever its
/// geo section starts. Copy fields out (`{ p.doc }`) rather than borrow them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, packed)]
pub struct GeoPointEntry {
    pub doc: DocId,
    pub lat: i32,
    pub lon: i32,
}

const _: () = assert!(std::mem::size_of::<GeoPointEntry>() == ENTRY_BYTES && std::mem::align_of::<GeoPointEntry>() == 1);

/// Serialized size of a [`GeoPointEntry`].
const ENTRY_BYTES: usize = 12;

/// Points owned in memory (memtables, builds) or a view of a sealed
/// segment's mapped file. Lucene keeps BKD trees and doc values off-heap the
/// same way: decoding them pinned 24 bytes per point (both copies) for the
/// segment's lifetime, 1.4 GB for 60M points.
#[derive(Clone)]
enum PointBuf {
    Owned(Vec<GeoPointEntry>),
    /// `len / ENTRY_BYTES` serialized entries (little-endian targets only).
    Mapped(FileSlice),
}

impl Default for PointBuf {
    fn default() -> Self {
        PointBuf::Owned(Vec::new())
    }
}

impl std::fmt::Debug for PointBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PointBuf::Owned(v) => write!(f, "Owned({} points)", v.len()),
            PointBuf::Mapped(m) => write!(f, "Mapped({} points)", m.len() / ENTRY_BYTES),
        }
    }
}

impl std::ops::Deref for PointBuf {
    type Target = [GeoPointEntry];
    #[inline]
    fn deref(&self) -> &[GeoPointEntry] {
        match self {
            PointBuf::Owned(v) => v,
            PointBuf::Mapped(m) => {
                let b = m.as_slice();
                // SAFETY: `GeoPointEntry` is `repr(C, packed)` (alignment 1,
                // no padding) of plain integers in the serialized order and
                // endianness (`Mapped` is only built on little-endian
                // targets), so any 12 bytes are a valid entry. The slice
                // keeps the mapping alive for the borrow.
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const GeoPointEntry, b.len() / ENTRY_BYTES) }
            }
        }
    }
}

impl<'a> IntoIterator for &'a PointBuf {
    type Item = &'a GeoPointEntry;
    type IntoIter = std::slice::Iter<'a, GeoPointEntry>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PointBuf {
    /// `n` serialized entries at `range` of `file`: viewed in place where
    /// the layout matches, else decoded.
    fn open(file: &FileSlice, range: std::ops::Range<usize>) -> PointBuf {
        if cfg!(target_endian = "little") {
            return PointBuf::Mapped(file.slice(range));
        }
        PointBuf::Owned(file.as_slice()[range].chunks_exact(ENTRY_BYTES).map(|b| GeoPointEntry {
            doc: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            lat: i32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            lon: i32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        }).collect())
    }

    fn heap_bytes(&self) -> usize {
        match self {
            PointBuf::Owned(v) => v.capacity() * ENTRY_BYTES,
            PointBuf::Mapped(_) => 0,
        }
    }
}

/// Points of one field, sorted by doc (memtable side, and the doc-values
/// half of a sealed [`GeoIndex`]). A sealed segment of format
/// [`crate::codec::GEO_COMPACT_SINCE`]+ whose every doc has exactly one
/// point keeps only `(lat, lon)` per doc ("dense": point `i` is doc `i`, 8
/// bytes instead of 12); read the points through [`GeoValues::entry`],
/// [`GeoValues::iter`] and [`GeoValues::doc_range`].
#[derive(Clone, Debug, Default)]
pub struct GeoValues {
    points: PointBuf,
    /// Dense layout: `len` LE `(lat, lon)` pairs, point `i` of doc `i`.
    dense: Option<(FileSlice, usize)>,
}

/// The points of one doc ([`GeoValues::entries_of`], [`crate::query::GeoDv::entries`]):
/// a run of the doc-sorted points, or a dense layout's one point.
#[derive(Clone, Copy, Debug)]
pub enum GeoEntries<'a> {
    Slice(&'a [GeoPointEntry]),
    One(GeoPointEntry),
}

impl<'a> GeoEntries<'a> {
    /// No point.
    pub const EMPTY: GeoEntries<'static> = GeoEntries::Slice(&[]);
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            GeoEntries::Slice(s) => s.len(),
            GeoEntries::One(_) => 1,
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The points (by value).
    #[inline]
    pub fn iter(&self) -> GeoEntriesIter<'a> {
        GeoEntriesIter { e: *self, i: 0 }
    }
    #[inline]
    pub fn first(&self) -> Option<GeoPointEntry> {
        self.iter().next()
    }
}

/// Iterator of a [`GeoEntries`].
#[derive(Clone, Debug)]
pub struct GeoEntriesIter<'a> {
    e: GeoEntries<'a>,
    i: usize,
}

impl Iterator for GeoEntriesIter<'_> {
    type Item = GeoPointEntry;
    #[inline]
    fn next(&mut self) -> Option<GeoPointEntry> {
        let p = match self.e {
            GeoEntries::Slice(s) => *s.get(self.i)?,
            GeoEntries::One(p) if self.i == 0 => p,
            GeoEntries::One(_) => return None,
        };
        self.i += 1;
        Some(p)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.e.len().saturating_sub(self.i);
        (n, Some(n))
    }
}

impl ExactSizeIterator for GeoEntriesIter<'_> {}

impl<'a> IntoIterator for GeoEntries<'a> {
    type Item = GeoPointEntry;
    type IntoIter = GeoEntriesIter<'a>;
    fn into_iter(self) -> GeoEntriesIter<'a> {
        self.iter()
    }
}

/// Iterator over a range of a [`GeoValues`]' points (by value).
#[derive(Clone, Debug)]
pub struct GeoIter<'a> {
    v: &'a GeoValues,
    i: usize,
    end: usize,
}

impl Iterator for GeoIter<'_> {
    type Item = GeoPointEntry;
    #[inline]
    fn next(&mut self) -> Option<GeoPointEntry> {
        if self.i >= self.end {
            return None;
        }
        let p = self.v.entry(self.i);
        self.i += 1;
        Some(p)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.i;
        (n, Some(n))
    }
}

impl ExactSizeIterator for GeoIter<'_> {}

impl GeoValues {
    /// Append a point of `doc` (docs must arrive in ascending order).
    pub fn push(&mut self, doc: DocId, lat: f64, lon: f64) {
        match &mut self.points {
            PointBuf::Owned(v) if self.dense.is_none() => v.push(GeoPointEntry { doc, lat: encode_lat(lat), lon: encode_lon(lon) }),
            _ => unreachable!("points pushed to a sealed segment"),
        }
    }

    /// Number of points.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.dense {
            Some((_, n)) => *n,
            None => self.points.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Point `i` in doc order (`i < len()`).
    #[inline]
    pub fn entry(&self, i: usize) -> GeoPointEntry {
        match &self.dense {
            None => self.points[i],
            Some((d, _)) => {
                let b = &d.as_slice()[i * 8..i * 8 + 8];
                GeoPointEntry { doc: i as DocId, lat: i32::from_le_bytes([b[0], b[1], b[2], b[3]]), lon: i32::from_le_bytes([b[4], b[5], b[6], b[7]]) }
            }
        }
    }

    /// Point `i`, if there is one.
    #[inline]
    pub fn get_entry(&self, i: usize) -> Option<GeoPointEntry> {
        (i < self.len()).then(|| self.entry(i))
    }

    /// Every point, in doc order (by value).
    #[inline]
    pub fn iter(&self) -> GeoIter<'_> {
        GeoIter { v: self, i: 0, end: self.len() }
    }

    /// Points `r` (clamped to the points).
    #[inline]
    pub fn iter_range(&self, r: std::ops::Range<usize>) -> GeoIter<'_> {
        let end = r.end.min(self.len());
        GeoIter { v: self, i: r.start.min(end), end }
    }

    /// The doc-sorted points as a slice, when not dense (memtables, older
    /// or multi-valued segments).
    #[inline]
    pub fn as_slice(&self) -> Option<&[GeoPointEntry]> {
        self.dense.is_none().then(|| &*self.points)
    }

    /// Every point, copied.
    pub fn to_vec(&self) -> Vec<GeoPointEntry> {
        self.iter().collect()
    }

    /// The first position at or after `from` whose doc is at least `doc`.
    #[inline]
    pub fn lower_bound(&self, doc: DocId, from: usize) -> usize {
        match &self.dense {
            Some((_, n)) => (doc as usize).max(from).min(*n),
            None => {
                let from = from.min(self.points.len());
                from + self.points[from..].partition_point(|p| p.doc < doc)
            }
        }
    }

    /// Positions of `doc`'s points.
    #[inline]
    pub fn doc_range(&self, doc: DocId) -> std::ops::Range<usize> {
        self.doc_range_hint(doc, 0)
    }

    /// [`GeoValues::doc_range`] from a cursor `hint` (the end of the last
    /// range read): galloping forward for ascending docs, a binary search
    /// when moving back.
    #[inline]
    pub fn doc_range_hint(&self, doc: DocId, hint: usize) -> std::ops::Range<usize> {
        if let Some((_, n)) = &self.dense {
            let d = (doc as usize).min(*n);
            return d..(d + usize::from((doc as usize) < *n));
        }
        let p: &[GeoPointEntry] = &self.points;
        let mut i = hint.min(p.len());
        if i > 0 && p[i - 1].doc >= doc {
            i = p.partition_point(|e| e.doc < doc);
        } else if i < p.len() && p[i].doc < doc {
            let mut step = 1;
            let mut hi = i + 1;
            while hi < p.len() && p[hi].doc < doc {
                i = hi;
                hi += step;
                step *= 2;
            }
            let hi = hi.min(p.len());
            i += p[i..hi].partition_point(|e| e.doc < doc);
        }
        let mut j = i;
        while j < p.len() && p[j].doc == doc {
            j += 1;
        }
        i..j
    }

    /// The points at positions `r` (a [`GeoValues::doc_range`]).
    #[inline]
    pub fn entries_of(&self, r: std::ops::Range<usize>) -> GeoEntries<'_> {
        match &self.dense {
            None => GeoEntries::Slice(&self.points[r]),
            Some(_) if r.is_empty() => GeoEntries::EMPTY,
            Some(_) => GeoEntries::One(self.entry(r.start)),
        }
    }

    /// Number of point values, including duplicates, without decoding or allocating.
    pub fn value_count(&self, doc: DocId) -> usize {
        self.doc_range(doc).len()
    }

    /// Decoded points of `doc`.
    pub fn get(&self, doc: DocId) -> Vec<(f64, f64)> {
        self.iter_range(self.doc_range(doc)).map(|p| (decode_lat(p.lat), decode_lon(p.lon))).collect()
    }

    /// Docs with a point in `shape` (linear scan), ascending.
    pub fn query(&self, shape: &GeoShape) -> Vec<DocId> {
        let mut out: Vec<DocId> = Vec::new();
        for p in self.iter() {
            if out.last() == Some(&{ p.doc }) {
                continue;
            }
            if shape.contains(decode_lat(p.lat), decode_lon(p.lon)) {
                out.push(p.doc);
            }
        }
        out
    }

    /// Heap bytes.
    pub fn heap_bytes(&self) -> usize {
        self.points.heap_bytes()
    }
}

/// Encoded bounding box of the points of one block of 256 docs (the
/// geo-distance sort skipper); `count == 0` when no doc of it has a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeoBlock {
    pub min_lat: i32,
    pub max_lat: i32,
    pub min_lon: i32,
    pub max_lon: i32,
    pub count: u32,
}

impl GeoValues {
    /// Bounding box per block of `block` docs over `num_docs` docs.
    pub fn blocks(&self, num_docs: u32, block: usize) -> Vec<GeoBlock> {
        let mut out = vec![GeoBlock { min_lat: i32::MAX, max_lat: i32::MIN, min_lon: i32::MAX, max_lon: i32::MIN, count: 0 }; (num_docs as usize).div_ceil(block)];
        for p in self.iter() {
            let Some(b) = out.get_mut(p.doc as usize / block) else { continue };
            b.min_lat = b.min_lat.min(p.lat);
            b.max_lat = b.max_lat.max(p.lat);
            b.min_lon = b.min_lon.min(p.lon);
            b.max_lon = b.max_lon.max(p.lon);
            b.count += 1;
        }
        out
    }
}

/// Lower bound (meters) of the distance from `(lat, lon)` to any point of
/// `b`, valid for both the haversine and the `plane` distance.
pub fn block_min_meters(lat: f64, lon: f64, b: &GeoBlock, plane: bool) -> f64 {
    if b.count == 0 {
        return f64::INFINITY;
    }
    let (lat0, lat1) = (decode_lat(b.min_lat), decode_lat(b.max_lat));
    let (lon0, lon1) = (decode_lon(b.min_lon), decode_lon(b.max_lon));
    let dlat = if lat < lat0 { lat0 - lat } else if lat > lat1 { lat - lat1 } else { 0.0 };
    let bound = if plane {
        // sqrt(x^2 + y^2) >= |y|.
        dlat.to_radians() * EARTH_MEAN_RADIUS_METERS
    } else {
        // hav(d) = hav(dlat) + cos(lat) cos(lat') hav(dlon) >= the sum of
        // the per-term minima over the box.
        let ang = |a: f64, b: f64| {
            let d = (a - b).abs() % 360.0;
            d.min(360.0 - d)
        };
        let dlon = if lon >= lon0 && lon <= lon1 { 0.0 } else { ang(lon, lon0).min(ang(lon, lon1)) };
        let hav = |deg: f64| {
            let s = (deg.to_radians() / 2.0).sin();
            s * s
        };
        let cos_box = lat0.abs().max(lat1.abs()).min(90.0).to_radians().cos();
        let h = hav(dlat) + lat.to_radians().cos().max(0.0) * cos_box * hav(dlon);
        2.0 * EARTH_MEAN_RADIUS_METERS * h.sqrt().min(1.0).asin()
    };
    // Slack for rounding against the per-point formula (ES's table-based
    // `SloppyMath.haversinMeters` for arc).
    (bound * (1.0 - 1e-9) - 1.0).max(0.0)
}

/// A KD-tree leaf: its encoded bounding box and its slice of points.
#[derive(Clone, Copy, Debug, Default)]
struct Leaf {
    min_lat: i32,
    max_lat: i32,
    min_lon: i32,
    max_lon: i32,
    start: u32,
    len: u32,
    /// Packed tree ([`Tree::Packed`]): the bit where the leaf's
    /// coordinates start and the bits of each point's latitude and
    /// longitude offset from the leaf's minimum.
    coord_bit: u64,
    lat_bits: u8,
    lon_bits: u8,
}

/// The KD-ordered points of a [`GeoIndex`].
#[derive(Clone, Debug)]
enum Tree {
    /// `(doc, lat, lon)` per point, read in place.
    Entries(PointBuf),
    /// Format [`crate::codec::GEO_COMPACT_SINCE`]+: a LE `u32` doc per
    /// point (in leaf order), and each point's `(lat, lon)` bit-packed as
    /// offsets from its leaf's minimum (widths per leaf; 8 zero bytes of
    /// padding so any read is one 8-byte load).
    Packed { docs: FileSlice, coords: FileSlice },
}

impl Default for Tree {
    fn default() -> Self {
        Tree::Entries(PointBuf::default())
    }
}

/// `width` (at most 32) bits of `bytes` at `bit`, LE.
#[inline(always)]
fn get_bits(bytes: &[u8], bit: u64, width: u8) -> u32 {
    if width == 0 {
        return 0;
    }
    let at = (bit / 8) as usize;
    let w = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    ((w >> (bit % 8)) & ((1u64 << width) - 1)) as u32
}

/// Bits needed for offsets `0..=span`.
fn bits_for(span: u32) -> u8 {
    (32 - span.leading_zeros()) as u8
}

/// Whether sealed segments of format [`crate::codec::GEO_COMPACT_SINCE`]+
/// pack the tree's coordinates ([`Tree::Packed`]); the dense doc-ordered
/// values apply either way. Readers handle both layouts.
const PACK_TREE: bool = true;
/// Layout version of a compact geo index entry ([`crate::codec::EXTRA_GEO2`]).
const COMPACT_FORMAT: u32 = 2;

/// The sealed per-segment geo index of one field.
#[derive(Clone, Debug, Default)]
pub struct GeoIndex {
    values: GeoValues,
    leaves: Vec<Leaf>,
    /// Points in leaf order.
    tree: Tree,
    /// No doc has more than one point (counts of points are counts of docs).
    single: bool,
    /// Boxes over runs of consecutive leaves (KD order keeps them compact):
    /// level 1 groups [`BLOCK`] leaves, level 2 groups [`BLOCK`] level-1
    /// blocks, so whole subtrees inside or outside a query are decided at once.
    blocks1: Vec<Block>,
    blocks2: Vec<Block>,
}

/// Leaves (or blocks) per block.
const BLOCK: usize = 16;

/// Geo structures not built because the memory budget refused them (the
/// paths then fall back: a slower relation, no early stop): counted per
/// kind for node stats (`rustic_geo` in `_nodes/stats`) and logged at
/// debug level (target `rustic::geo`).
pub mod structures {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static GEO_SORT_BOUNDS_REFUSED: AtomicU64 = AtomicU64::new(0);
    /// [`super::GeoIndex::pred_count`] walks: leaves counted whole, leaves
    /// whose points were tested, points tested (added once per walk).
    pub static COUNT_LEAVES_WHOLE: AtomicU64 = AtomicU64::new(0);
    pub static COUNT_LEAVES_TESTED: AtomicU64 = AtomicU64::new(0);
    pub static COUNT_POINTS_TESTED: AtomicU64 = AtomicU64::new(0);

    pub fn add(counter: &AtomicU64, n: u64) {
        if n > 0 {
            counter.fetch_add(n, Relaxed);
        }
    }

    /// A reservation of `bytes` for `what` was refused.
    pub fn refused(counter: &AtomicU64, what: &'static str, bytes: u64) {
        counter.fetch_add(1, Relaxed);
        tracing::debug!(target: "rustic::geo", what, bytes, "geo structure reservation refused; falling back");
    }

    pub fn note(counter: &AtomicU64) {
        counter.fetch_add(1, Relaxed);
    }

    /// `(name, count)` of every counter.
    pub fn snapshot() -> Vec<(&'static str, u64)> {
        vec![
            ("geo_sort_bounds_refused", GEO_SORT_BOUNDS_REFUSED.load(Relaxed)),
            ("count_leaves_whole", COUNT_LEAVES_WHOLE.load(Relaxed)),
            ("count_leaves_tested", COUNT_LEAVES_TESTED.load(Relaxed)),
            ("count_points_tested", COUNT_POINTS_TESTED.load(Relaxed)),
        ]
    }
}

/// Points from which a KD walk is split over threads, and the most threads.
const PAR_POINTS: usize = 1 << 21;
const PAR_MAX: usize = 3;

/// [`PAR_POINTS`] outside the search thread pool. A search walks its
/// leaves in parallel on the pool, so a walk there splits only large
/// trees; the ES|QL and aggregation paths walk a shard's leaves one after
/// another on their own thread (one thread per shard), so an index of many
/// mid-sized segments (69 per corpus after a merge that did not collapse)
/// splits each walk from smaller trees.
const PAR_POINTS_OFF_POOL: usize = 1 << 19;

/// The bounding box of a run `[first, end)` of leaves or level-1 blocks.
#[derive(Clone, Copy, Debug, Default)]
struct Block {
    min_lat: i32,
    max_lat: i32,
    min_lon: i32,
    max_lon: i32,
    first: u32,
    end: u32,
}

fn group<T>(items: &[T], bounds: impl Fn(&T) -> (i32, i32, i32, i32)) -> Vec<Block> {
    items
        .chunks(BLOCK)
        .enumerate()
        .map(|(i, c)| {
            let mut b = Block { min_lat: i32::MAX, max_lat: i32::MIN, min_lon: i32::MAX, max_lon: i32::MIN, first: (i * BLOCK) as u32, end: (i * BLOCK + c.len()) as u32 };
            for x in c {
                let (a, bb, cc, d) = bounds(x);
                b.min_lat = b.min_lat.min(a);
                b.max_lat = b.max_lat.max(bb);
                b.min_lon = b.min_lon.min(cc);
                b.max_lon = b.max_lon.max(d);
            }
            b
        })
        .collect()
}

fn blocks_of(leaves: &[Leaf]) -> (Vec<Block>, Vec<Block>) {
    let b1 = group(leaves, |l| (l.min_lat, l.max_lat, l.min_lon, l.max_lon));
    let b2 = group(&b1, |b| (b.min_lat, b.max_lat, b.min_lon, b.max_lon));
    (b1, b2)
}

fn single_valued(points: &[GeoPointEntry]) -> bool {
    points.windows(2).all(|w| w[0].doc < w[1].doc)
}

impl GeoIndex {
    /// Build from doc-sorted points.
    pub fn build(points: Vec<GeoPointEntry>) -> Self {
        let mut tree = points.clone();
        let mut leaves = Vec::new();
        build_kd(&mut tree, 0, &mut leaves);
        let single = single_valued(&points);
        let (blocks1, blocks2) = blocks_of(&leaves);
        GeoIndex { values: GeoValues { points: PointBuf::Owned(points), dense: None }, leaves, tree: Tree::Entries(PointBuf::Owned(tree)), single, blocks1, blocks2 }
    }

    /// The doc-sorted points.
    pub fn values(&self) -> &GeoValues {
        &self.values
    }

    /// Point `i` of leaf `l`.
    #[inline(always)]
    fn leaf_entry(&self, l: &Leaf, i: usize) -> GeoPointEntry {
        match &self.tree {
            Tree::Entries(t) => t[l.start as usize + i],
            Tree::Packed { docs, coords } => {
                let at = (l.start as usize + i) * 4;
                let d = &docs.as_slice()[at..at + 4];
                let bit = l.coord_bit + i as u64 * (l.lat_bits as u64 + l.lon_bits as u64);
                let c = coords.as_slice();
                GeoPointEntry {
                    doc: u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
                    lat: l.min_lat.wrapping_add(get_bits(c, bit, l.lat_bits) as i32),
                    lon: l.min_lon.wrapping_add(get_bits(c, bit + l.lat_bits as u64, l.lon_bits) as i32),
                }
            }
        }
    }

    /// Each point of leaf `l`.
    #[inline(always)]
    fn for_leaf(&self, l: &Leaf, mut f: impl FnMut(GeoPointEntry)) {
        match &self.tree {
            Tree::Entries(t) => t[l.start as usize..(l.start + l.len) as usize].iter().for_each(|p| f(*p)),
            Tree::Packed { .. } => (0..l.len as usize).for_each(|i| f(self.leaf_entry(l, i))),
        }
    }

    /// The doc of each point of leaf `l` (no coordinates decoded).
    #[inline(always)]
    fn for_leaf_docs(&self, l: &Leaf, mut f: impl FnMut(DocId)) {
        match &self.tree {
            Tree::Entries(t) => t[l.start as usize..(l.start + l.len) as usize].iter().for_each(|p| f(p.doc)),
            Tree::Packed { docs, .. } => docs.as_slice()[l.start as usize * 4..(l.start + l.len) as usize * 4]
                .chunks_exact(4).for_each(|d| f(u32::from_le_bytes([d[0], d[1], d[2], d[3]]))),
        }
    }

    /// Points of leaf `l` passing `test`.
    #[inline(always)]
    fn count_leaf(&self, l: &Leaf, mut test: impl FnMut(&GeoPointEntry) -> bool) -> u64 {
        let mut n = 0u64;
        self.for_leaf(l, |p| n += u64::from(test(&p)));
        n
    }

    /// Each point of leaf `l` as `(index in the leaf, lat, lon)`: a packed
    /// leaf decodes its coordinates sequentially and reads no doc (see
    /// [`GeoIndex::leaf_doc`] for the docs of the points that pass).
    #[inline(always)]
    fn for_leaf_coords(&self, l: &Leaf, mut f: impl FnMut(usize, i32, i32)) {
        match &self.tree {
            Tree::Entries(t) => t[l.start as usize..(l.start + l.len) as usize].iter().enumerate().for_each(|(i, p)| f(i, p.lat, p.lon)),
            Tree::Packed { coords, .. } => {
                let c = coords.as_slice();
                let w = l.lat_bits as u64 + l.lon_bits as u64;
                let mut bit = l.coord_bit;
                for i in 0..l.len as usize {
                    let lat = l.min_lat.wrapping_add(get_bits(c, bit, l.lat_bits) as i32);
                    let lon = l.min_lon.wrapping_add(get_bits(c, bit + l.lat_bits as u64, l.lon_bits) as i32);
                    bit += w;
                    f(i, lat, lon);
                }
            }
        }
    }

    /// The doc of point `i` of leaf `l`.
    #[inline(always)]
    fn leaf_doc(&self, l: &Leaf, i: usize) -> DocId {
        match &self.tree {
            Tree::Entries(t) => t[l.start as usize + i].doc,
            Tree::Packed { docs, .. } => {
                let at = (l.start as usize + i) * 4;
                let d = &docs.as_slice()[at..at + 4];
                u32::from_le_bytes([d[0], d[1], d[2], d[3]])
            }
        }
    }

    /// Number of KD leaves.
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Visit KD leaves in ascending conservative distance-bound order.
    /// The callback receives a lower bound in meters and the encoded points.
    /// Returning false stops the walk; callers may do this only when the
    /// bound proves that this leaf and every farther leaf cannot contribute.
    /// Points may include duplicate doc ids on multivalued fields.
    pub fn visit_by_distance(&self, lat: f64, lon: f64, mut visit: impl FnMut(f64, &[GeoPointEntry]) -> bool) {
        use std::cmp::{Ordering, Reverse};
        use std::collections::BinaryHeap;
        // Best first over the block hierarchy: a block's bound is at most
        // its children's (their boxes lie in its box), so leaves come off
        // the heap in the order of sorting every leaf's bound, ties by leaf
        // index (blocks expand before leaves of an equal bound), and a walk
        // that stops after a few leaves bounds a few blocks' children
        // instead of every leaf of the segment.
        struct Item(f64, u8, u32);
        impl Ord for Item {
            fn cmp(&self, o: &Self) -> Ordering {
                self.0.total_cmp(&o.0).then(self.1.cmp(&o.1)).then(self.2.cmp(&o.2))
            }
        }
        impl PartialOrd for Item {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl PartialEq for Item {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o) == Ordering::Equal
            }
        }
        impl Eq for Item {}
        let block_bound = |b: &Block| {
            let g = GeoBlock { min_lat: b.min_lat, max_lat: b.max_lat, min_lon: b.min_lon, max_lon: b.max_lon, count: b.end - b.first };
            block_min_meters(lat, lon, &g, false)
        };
        let leaf_bound = |leaf: &Leaf| {
            let g = GeoBlock { min_lat: leaf.min_lat, max_lat: leaf.max_lat, min_lon: leaf.min_lon, max_lon: leaf.max_lon, count: leaf.len };
            block_min_meters(lat, lon, &g, false)
        };
        let mut heap: BinaryHeap<Reverse<Item>> = BinaryHeap::with_capacity(self.blocks2.len() + 2 * BLOCK);
        heap.extend(self.blocks2.iter().enumerate().map(|(i, b)| Reverse(Item(block_bound(b), 0, i as u32))));
        // A packed leaf is decoded into one reused buffer per visit.
        let mut buf: Vec<GeoPointEntry> = Vec::new();
        while let Some(Reverse(Item(lower_bound, level, i))) = heap.pop() {
            let i = i as usize;
            match level {
                0 => {
                    let b = &self.blocks2[i];
                    heap.extend((b.first..b.end).map(|j| Reverse(Item(block_bound(&self.blocks1[j as usize]), 1, j))));
                }
                1 => {
                    let b = &self.blocks1[i];
                    heap.extend((b.first..b.end).map(|j| Reverse(Item(leaf_bound(&self.leaves[j as usize]), 2, j))));
                }
                _ => {
                    let leaf = &self.leaves[i];
                    let points: &[GeoPointEntry] = if let Tree::Entries(t) = &self.tree {
                        &t[leaf.start as usize..(leaf.start + leaf.len) as usize]
                    } else {
                        buf.clear();
                        self.for_leaf(leaf, |p| buf.push(p));
                        &buf
                    };
                    if !visit(lower_bound, points) {
                        break;
                    }
                }
            }
        }
    }

    /// [`GeoIndex::visit_by_distance`] as it was: every leaf's bound
    /// computed and sorted up front (tests compare the visit orders).
    #[cfg(test)]
    fn visit_by_distance_sorted(&self, lat: f64, lon: f64, mut visit: impl FnMut(f64, &[GeoPointEntry]) -> bool) {
        let mut ordered: Vec<_> = self.leaves.iter().enumerate().map(|(i, leaf)| {
            let bounds = GeoBlock { min_lat: leaf.min_lat, max_lat: leaf.max_lat,
                min_lon: leaf.min_lon, max_lon: leaf.max_lon, count: leaf.len };
            (block_min_meters(lat, lon, &bounds, false), i)
        }).collect();
        ordered.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut buf: Vec<GeoPointEntry> = Vec::new();
        for (lower_bound, i) in ordered {
            let leaf = &self.leaves[i];
            let points: &[GeoPointEntry] = if let Tree::Entries(t) = &self.tree {
                &t[leaf.start as usize..(leaf.start + leaf.len) as usize]
            } else {
                buf.clear();
                self.for_leaf(leaf, |p| buf.push(p));
                &buf
            };
            if !visit(lower_bound, points) {
                break;
            }
        }
    }

    /// Visit the points leaf by leaf: `classify(min_lat, max_lat, min_lon,
    /// max_lon)` (decoded, inclusive) of a leaf's box decides whether its
    /// points are skipped ([`Relation::Outside`]), all in the shape
    /// ([`Relation::Inside`], `f(point, true)`) or to be tested
    /// (`f(point, false)`).
    pub fn visit(&self, classify: impl FnMut(f64, f64, f64, f64) -> Relation, mut f: impl FnMut(&GeoPointEntry, bool)) {
        self.walk(classify, |l, inside| self.for_leaf(l, |p| f(&p, inside)));
    }

    /// The leaves `classify` does not rule out, with whether they are
    /// wholly inside; blocks inside or outside decide their leaves at once
    /// (`classify` must be monotone: a box inside / outside implies the
    /// same for the boxes it contains).
    fn walk(&self, classify: impl FnMut(f64, f64, f64, f64) -> Relation, f: impl FnMut(&Leaf, bool)) {
        self.walk_range(0..self.blocks2.len(), classify, f)
    }

    /// [`GeoIndex::walk`] over the level-2 blocks `range`.
    fn walk_range(&self, range: std::ops::Range<usize>, mut classify: impl FnMut(f64, f64, f64, f64) -> Relation, mut f: impl FnMut(&Leaf, bool)) {
        let mut rel = |b: (i32, i32, i32, i32)| classify(decode_lat(b.0), decode_lat(b.1), decode_lon(b.2), decode_lon(b.3));
        for b2 in &self.blocks2[range] {
            match rel((b2.min_lat, b2.max_lat, b2.min_lon, b2.max_lon)) {
                Relation::Outside => {}
                Relation::Inside => {
                    for b1 in &self.blocks1[b2.first as usize..b2.end as usize] {
                        self.leaves[b1.first as usize..b1.end as usize].iter().for_each(|l| f(l, true));
                    }
                }
                Relation::Crosses => {
                    for b1 in &self.blocks1[b2.first as usize..b2.end as usize] {
                        match rel((b1.min_lat, b1.max_lat, b1.min_lon, b1.max_lon)) {
                            Relation::Outside => {}
                            Relation::Inside => self.leaves[b1.first as usize..b1.end as usize].iter().for_each(|l| f(l, true)),
                            Relation::Crosses => {
                                for l in &self.leaves[b1.first as usize..b1.end as usize] {
                                    match rel((l.min_lat, l.max_lat, l.min_lon, l.max_lon)) {
                                        Relation::Outside => {}
                                        r => f(l, r == Relation::Inside),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Docs with a point in `shape`, ascending. Also reports how many
    /// leaves were visited (not skipped) through `visited`.
    pub fn query_counting(&self, shape: &GeoShape, visited: &mut usize) -> Vec<DocId> {
        let mut out: Vec<DocId> = Vec::new();
        for l in &self.leaves {
            let r = shape.relate(decode_lat(l.min_lat), decode_lat(l.max_lat), decode_lon(l.min_lon), decode_lon(l.max_lon));
            match r {
                Relation::Outside => {}
                Relation::Inside => {
                    *visited += 1;
                    self.for_leaf_docs(l, |d| out.push(d));
                }
                Relation::Crosses => {
                    *visited += 1;
                    self.for_leaf_coords(l, |i, lat, lon| if shape.contains(decode_lat(lat), decode_lon(lon)) { out.push(self.leaf_doc(l, i)) });
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Docs with at least one point (single-valued fields: their count
    /// without a scan).
    pub fn docs_with_points(&self) -> Option<u64> {
        self.single.then_some(self.values.len() as u64)
    }

    /// Number of docs with a point in `shape`, from the KD leaves (leaves
    /// inside the shape counted whole): only for single-valued fields.
    pub fn count(&self, shape: &GeoShape) -> Option<u64> {
        if !self.single {
            return None;
        }
        // Boundary leaves test coordinates only (a packed leaf decodes no doc).
        let mut n = 0u64;
        self.walk(|a, b, c, d| shape.relate(a, b, c, d), |l, inside| {
            if inside {
                n += l.len as u64;
            } else {
                self.for_leaf_coords(l, |_, lat, lon| n += u64::from(shape.contains(decode_lat(lat), decode_lon(lon))));
            }
        });
        Some(n)
    }

    /// Number of docs whose point passes `test`, leaves classified by
    /// `classify` (see [`GeoIndex::visit`]); `None` unless single-valued.
    pub fn visit_count(&self, mut classify: impl FnMut(f64, f64, f64, f64) -> Relation, mut test: impl FnMut(&GeoPointEntry) -> bool) -> Option<u64> {
        if !self.single {
            return None;
        }
        let mut n = 0u64;
        self.walk(classify, |l, inside| {
            n += if inside { l.len as u64 } else { self.count_leaf(l, |p| test(p)) };
        });
        Some(n)
    }

    /// Docs with a point in `shape` (`num_docs`: the segment's docs).
    pub fn query_hits(&self, shape: &GeoShape, num_docs: u32) -> DocHits {
        let mut hits = HitCollector::new(num_docs);
        // Inside leaves read docs only; boundary leaves test coordinates and
        // read the doc of a match (a packed leaf decodes nothing else).
        self.walk(|a, b, c, d| shape.relate(a, b, c, d), |l, inside| {
            if inside {
                self.for_leaf_docs(l, |d| hits.push(d));
            } else {
                self.for_leaf_coords(l, |i, lat, lon| if shape.contains(decode_lat(lat), decode_lon(lon)) { hits.push(self.leaf_doc(l, i)) });
            }
        });
        hits.finish()
    }

    /// Docs with a point in `shape`, ascending.
    pub fn query(&self, shape: &GeoShape) -> Vec<DocId> {
        let mut v = 0;
        self.query_counting(shape, &mut v)
    }

    /// Serialize: `u32 n, n × (doc, lat, lon)` doc-sorted, then `u32
    /// leaves, leaves × (min_lat, max_lat, min_lon, max_lon, start, len)`,
    /// then the leaf-ordered points.
    pub fn encode(&self, out: &mut Vec<u8>) {
        let put = |out: &mut Vec<u8>, p: &GeoPointEntry| {
            out.extend_from_slice(&p.doc.to_le_bytes());
            out.extend_from_slice(&p.lat.to_le_bytes());
            out.extend_from_slice(&p.lon.to_le_bytes());
        };
        out.extend_from_slice(&(self.values.len() as u32).to_le_bytes());
        for p in self.values.iter() {
            put(out, &p);
        }
        out.extend_from_slice(&(self.leaves.len() as u32).to_le_bytes());
        for l in &self.leaves {
            for v in [l.min_lat, l.max_lat, l.min_lon, l.max_lon] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&l.start.to_le_bytes());
            out.extend_from_slice(&l.len.to_le_bytes());
        }
        for l in &self.leaves {
            self.for_leaf(l, |p| put(out, &p));
        }
    }

    /// Serialize in the compact layout (format
    /// [`crate::codec::GEO_COMPACT_SINCE`]+, entry [`crate::codec::EXTRA_GEO2`]):
    /// `u32 2, u32 flags (1: dense values, 2: packed tree), u32 n`; the
    /// values (`n × (lat, lon)` when dense, else `n × (doc, lat, lon)`);
    /// `u32 leaves, leaves × (min_lat, max_lat, min_lon, max_lon, start,
    /// len)`, then with a packed tree `leaves × (u64 coord_bit, u8
    /// lat_bits, u8 lon_bits)`; the tree (`n × u32 doc, u64 coord bytes,
    /// the coordinates, 8 zero bytes` when packed, else `n × (doc, lat,
    /// lon)`). Values are dense when every doc `0..n` has exactly one point.
    pub fn encode_compact(&self, out: &mut Vec<u8>) {
        let n = self.values.len();
        let dense = self.single && self.values.iter().enumerate().all(|(i, p)| p.doc as usize == i);
        let flags = u32::from(dense) | if PACK_TREE { 2 } else { 0 };
        out.extend_from_slice(&COMPACT_FORMAT.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&(n as u32).to_le_bytes());
        for p in self.values.iter() {
            if !dense {
                out.extend_from_slice(&p.doc.to_le_bytes());
            }
            out.extend_from_slice(&p.lat.to_le_bytes());
            out.extend_from_slice(&p.lon.to_le_bytes());
        }
        out.extend_from_slice(&(self.leaves.len() as u32).to_le_bytes());
        for l in &self.leaves {
            for v in [l.min_lat, l.max_lat, l.min_lon, l.max_lon] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&l.start.to_le_bytes());
            out.extend_from_slice(&l.len.to_le_bytes());
        }
        if !PACK_TREE {
            for l in &self.leaves {
                self.for_leaf(l, |p| {
                    out.extend_from_slice(&p.doc.to_le_bytes());
                    out.extend_from_slice(&p.lat.to_le_bytes());
                    out.extend_from_slice(&p.lon.to_le_bytes());
                });
            }
            return;
        }
        // Per leaf: widths from its box, then its points' offsets.
        let mut coords: Vec<u8> = Vec::new();
        let (mut acc, mut acc_bits, mut bit) = (0u64, 0u32, 0u64);
        let mut meta: Vec<(u64, u8, u8)> = Vec::with_capacity(self.leaves.len());
        let push = |coords: &mut Vec<u8>, v: u32, width: u8, acc: &mut u64, acc_bits: &mut u32| {
            if width == 0 {
                return;
            }
            *acc |= (v as u64) << *acc_bits;
            *acc_bits += width as u32;
            while *acc_bits >= 8 {
                coords.push(*acc as u8);
                *acc >>= 8;
                *acc_bits -= 8;
            }
        };
        for l in &self.leaves {
            let lat_bits = bits_for(l.max_lat.wrapping_sub(l.min_lat) as u32);
            let lon_bits = bits_for(l.max_lon.wrapping_sub(l.min_lon) as u32);
            meta.push((bit, lat_bits, lon_bits));
            self.for_leaf(l, |p| {
                push(&mut coords, p.lat.wrapping_sub(l.min_lat) as u32, lat_bits, &mut acc, &mut acc_bits);
                push(&mut coords, p.lon.wrapping_sub(l.min_lon) as u32, lon_bits, &mut acc, &mut acc_bits);
            });
            bit += l.len as u64 * (lat_bits as u64 + lon_bits as u64);
        }
        if acc_bits > 0 {
            coords.push(acc as u8);
        }
        for (b, la, lo) in &meta {
            out.extend_from_slice(&b.to_le_bytes());
            out.push(*la);
            out.push(*lo);
        }
        for l in &self.leaves {
            self.for_leaf_docs(l, |d| out.extend_from_slice(&d.to_le_bytes()));
        }
        out.extend_from_slice(&(coords.len() as u64).to_le_bytes());
        out.extend_from_slice(&coords);
        out.extend_from_slice(&[0u8; 8]);
    }

    /// Inverse of [`GeoIndex::encode_compact`]: values and tree are read in
    /// place from `file`; only the leaves are decoded.
    pub fn open_compact(file: &FileSlice) -> Result<Self> {
        let data = file.as_slice();
        let truncated = || RusticError::Corrupted("geo index truncated".into());
        let u32_at = |pos: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(data.get(pos..pos + 4).ok_or_else(truncated)?.try_into().unwrap()))
        };
        if u32_at(0)? != COMPACT_FORMAT {
            return Err(RusticError::Corrupted("unknown geo index layout".into()));
        }
        let flags = u32_at(4)?;
        let (dense, packed) = (flags & 1 != 0, flags & 2 != 0);
        let n = u32_at(8)? as usize;
        let per_value = if dense { 8 } else { ENTRY_BYTES };
        let values_at = 12..12 + n * per_value;
        let nl = u32_at(values_at.end)? as usize;
        let leaves_at = values_at.end + 4;
        let meta_at = leaves_at + nl * 24;
        let tree_at = meta_at + if packed { nl * 10 } else { 0 };
        let mut leaves = Vec::with_capacity(nl);
        for i in 0..nl {
            let at = |k: usize| u32_at(leaves_at + i * 24 + k * 4);
            let len = at(5)?;
            let start = at(4)?;
            if start as usize + len as usize > n {
                return Err(RusticError::Corrupted("geo index leaf out of range".into()));
            }
            let mut leaf = Leaf { min_lat: at(0)? as i32, max_lat: at(1)? as i32, min_lon: at(2)? as i32, max_lon: at(3)? as i32, start, len, ..Leaf::default() };
            if packed {
                let m = data.get(meta_at + i * 10..meta_at + i * 10 + 10).ok_or_else(truncated)?;
                leaf.coord_bit = u64::from_le_bytes(m[0..8].try_into().unwrap());
                leaf.lat_bits = m[8];
                leaf.lon_bits = m[9];
                if leaf.lat_bits > 32 || leaf.lon_bits > 32 {
                    return Err(RusticError::Corrupted("geo index leaf widths".into()));
                }
            }
            leaves.push(leaf);
        }
        let tree = if packed {
            let docs_end = tree_at + n * 4;
            let len_at = docs_end;
            let coord_len = u64::from_le_bytes(data.get(len_at..len_at + 8).ok_or_else(truncated)?.try_into().unwrap()) as usize;
            let coords_at = len_at + 8..len_at + 8 + coord_len + 8;
            if data.len() < coords_at.end {
                return Err(truncated());
            }
            // Every leaf's last bit inside the coordinates.
            for l in &leaves {
                let end = l.coord_bit + l.len as u64 * (l.lat_bits as u64 + l.lon_bits as u64);
                if end.div_ceil(8) > coord_len as u64 {
                    return Err(RusticError::Corrupted("geo index coordinates out of range".into()));
                }
            }
            Tree::Packed { docs: file.slice(tree_at..docs_end), coords: file.slice(coords_at) }
        } else {
            let range = tree_at..tree_at + n * ENTRY_BYTES;
            if data.len() < range.end {
                return Err(truncated());
            }
            Tree::Entries(PointBuf::open(file, range))
        };
        if data.len() < values_at.end {
            return Err(truncated());
        }
        let values = if dense {
            GeoValues { points: PointBuf::default(), dense: Some((file.slice(values_at), n)) }
        } else {
            GeoValues { points: PointBuf::open(file, values_at), dense: None }
        };
        let single = dense || single_valued(&values.points);
        let (blocks1, blocks2) = blocks_of(&leaves);
        Ok(GeoIndex { values, leaves, tree, single, blocks1, blocks2 })
    }

    /// Inverse of [`GeoIndex::encode`], over the bytes in memory.
    pub fn decode(data: &[u8]) -> Result<Self> {
        Self::open(&FileSlice::from_vec(data.to_vec()))
    }

    /// Inverse of [`GeoIndex::encode`]: the points are read in place from
    /// `file` (a sealed segment's mapped geo section); only the leaf boxes
    /// are decoded.
    pub fn open(file: &FileSlice) -> Result<Self> {
        let data = file.as_slice();
        let truncated = || RusticError::Corrupted("geo index truncated".into());
        let u32_at = |pos: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(data.get(pos..pos + 4).ok_or_else(truncated)?.try_into().unwrap()))
        };
        let n = u32_at(0)? as usize;
        let values_at = 4..4 + n * ENTRY_BYTES;
        let nl = u32_at(values_at.end)? as usize;
        let leaves_at = values_at.end + 4;
        let tree_at = leaves_at + nl * 24..leaves_at + nl * 24 + n * ENTRY_BYTES;
        if data.len() < tree_at.end {
            return Err(truncated());
        }
        let mut leaves = Vec::with_capacity(nl);
        for i in 0..nl {
            let at = |k: usize| u32_at(leaves_at + i * 24 + k * 4);
            let len = at(5)?;
            let start = at(4)?;
            if start as usize + len as usize > n {
                return Err(RusticError::Corrupted("geo index leaf out of range".into()));
            }
            leaves.push(Leaf { min_lat: at(0)? as i32, max_lat: at(1)? as i32, min_lon: at(2)? as i32, max_lon: at(3)? as i32, start, len, ..Leaf::default() });
        }
        let points = PointBuf::open(file, values_at);
        let tree = Tree::Entries(PointBuf::open(file, tree_at));
        let single = single_valued(&points);
        let (blocks1, blocks2) = blocks_of(&leaves);
        Ok(GeoIndex { values: GeoValues { points, dense: None }, leaves, tree, single, blocks1, blocks2 })
    }
}

/// Recursively split `pts` (in place) on the widest dimension until leaves
/// hold at most [`LEAF_POINTS`] points; `base` is `pts`' offset in the tree.
fn build_kd(pts: &mut [GeoPointEntry], base: u32, leaves: &mut Vec<Leaf>) {
    if pts.is_empty() {
        return;
    }
    let (mut a, mut b, mut c, mut d) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for p in pts.iter() {
        a = a.min(p.lat);
        b = b.max(p.lat);
        c = c.min(p.lon);
        d = d.max(p.lon);
    }
    if pts.len() <= LEAF_POINTS {
        leaves.push(Leaf { min_lat: a, max_lat: b, min_lon: c, max_lon: d, start: base, len: pts.len() as u32, ..Leaf::default() });
        return;
    }
    let mid = pts.len() / 2;
    // Latitude ints span half the degrees of longitude ints: compare in degrees.
    if (b as i64 - a as i64) * 2 >= d as i64 - c as i64 {
        pts.select_nth_unstable_by_key(mid, |p| p.lat);
    } else {
        pts.select_nth_unstable_by_key(mid, |p| p.lon);
    }
    let (left, right) = pts.split_at_mut(mid);
    build_kd(left, base, leaves);
    build_kd(right, base + mid as u32, leaves);
}

// ---- point predicates ----------------------------------------------------------

/// A predicate over single points with a KD-leaf classification (custom
/// spatial queries implement it).
pub trait PointPredicate: Send + Sync + std::fmt::Debug {
    /// How a leaf box (decoded, inclusive) relates to the predicate.
    fn classify(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Relation;
    /// Whether the point passes.
    fn test(&self, lat: f64, lon: f64) -> bool;
    /// How many of the `(lat, lon)` points pass ([`PointPredicate::test`]
    /// of each): one call per KD leaf, which implementations can run as
    /// one tight loop.
    fn count_many(&self, pts: &[(f64, f64)]) -> u64 {
        pts.iter().filter(|p| self.test(p.0, p.1)).count() as u64
    }
    /// `out[i] = test(pts[i])` for every point (`out` as long as `pts`).
    fn test_many(&self, pts: &[(f64, f64)], out: &mut [bool]) {
        for (o, p) in out.iter_mut().zip(pts) {
            *o = self.test(p.0, p.1);
        }
    }
}

impl PointPredicate for GeoShape {
    fn classify(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Relation {
        self.relate(min_lat, max_lat, min_lon, max_lon)
    }
    fn test(&self, lat: f64, lon: f64) -> bool {
        self.contains(lat, lon)
    }
}

/// A point predicate with a grid over its bounding box (like Lucene's
/// `GeoEncodingUtils` distance / polygon predicates): a cell is classified
/// the first time a point falls in it, so most points are decided by a
/// cell lookup and only points in boundary cells run the exact test.
/// Same results as `inner`.
#[derive(Debug)]
pub struct GridPred<P: PointPredicate> {
    inner: P,
    /// `(min_lat, max_lat, min_lon, max_lon)` outside of which nothing matches.
    bbox: (f64, f64, f64, f64),
    lat_step: f64,
    lon_step: f64,
    /// Per cell: 0 outside, 1 inside, 2 crosses, 3 not classified yet
    /// (allocated on the first test; empty when the box is degenerate).
    cells: std::sync::OnceLock<GridCells>,
    /// Identifies `inner` (see [`GridPred::keyed`]).
    key: Option<String>,
}

type GridCells = std::sync::Arc<[std::sync::atomic::AtomicU8]>;

/// Cells per grid side (512: half the boundary band of 256, whose points
/// run the exact test).
const GRID: usize = 512;

/// Shapes with more vertices than this are not worth a key: their grids
/// are not shared ([`GridPred::new`]).
pub const GRID_KEY_MAX_VERTICES: usize = 4096;

/// Grids kept by [`GridPred::keyed`] (256 KiB each).
const GRID_CACHE: usize = 32;

/// The classified grids of recent keyed predicates, most recent first:
/// `(key, bbox bits, cells)`.
static GRID_CELLS: parking_lot::Mutex<Vec<(String, [u64; 4], GridCells)>> = parking_lot::Mutex::new(Vec::new());

fn new_cells() -> GridCells {
    (0..GRID * GRID).map(|_| std::sync::atomic::AtomicU8::new(3)).collect()
}

/// The grid of `key` over `bbox`, shared with every predicate made with
/// the same key and box (created unclassified when not cached).
fn shared_cells(key: &str, bbox: (f64, f64, f64, f64)) -> GridCells {
    let bits = [bbox.0.to_bits(), bbox.1.to_bits(), bbox.2.to_bits(), bbox.3.to_bits()];
    let mut cache = GRID_CELLS.lock();
    if let Some(i) = cache.iter().position(|(k, b, _)| *b == bits && k == key) {
        let e = cache.remove(i);
        let cells = e.2.clone();
        cache.insert(0, e);
        return cells;
    }
    let cells = new_cells();
    cache.insert(0, (key.to_string(), bits, cells.clone()));
    cache.truncate(GRID_CACHE);
    cells
}

impl<P: PointPredicate> GridPred<P> {
    pub fn new(inner: P, bbox: (f64, f64, f64, f64)) -> Self {
        let lat_step = (bbox.1 - bbox.0) / GRID as f64;
        let lon_step = (bbox.3 - bbox.2) / GRID as f64;
        GridPred { inner, bbox, lat_step, lon_step, cells: std::sync::OnceLock::new(), key: None }
    }

    /// [`GridPred::new`] whose cell classifications are shared with every
    /// predicate of the same `key` and box: the other leaves of the same
    /// search, and the same query run again (Rally repeats each query), no
    /// longer classify their cells anew. `key` must determine `inner`'s
    /// `classify` exactly (e.g. the shape's coordinates, bit for bit):
    /// a cell's class is a pure function of the predicate and the cell.
    pub fn keyed(inner: P, bbox: (f64, f64, f64, f64), key: String) -> Self {
        GridPred { key: Some(key), ..GridPred::new(inner, bbox) }
    }
}

impl<P: PointPredicate> GridPred<P> {
    fn grid_cells(&self) -> &GridCells {
        self.cells.get_or_init(|| match &self.key {
            _ if !(self.lat_step > 0.0 && self.lon_step > 0.0) => GridCells::from(Vec::new()),
            Some(k) => shared_cells(k, self.bbox),
            None => new_cells(),
        })
    }

    /// Cell `(i, j)`'s class (0 outside, 1 inside, 2 crosses), classified
    /// on first use.
    fn cell_class(&self, cells: &GridCells, i: usize, j: usize) -> u8 {
        use std::sync::atomic::Ordering::Relaxed;
        let cell = &cells[i * GRID + j];
        let mut v = cell.load(Relaxed);
        if v == 3 {
            let (a, c) = (self.bbox.0, self.bbox.2);
            let (la, lb) = (a + i as f64 * self.lat_step, a + (i + 1) as f64 * self.lat_step);
            let (lc, ld) = (c + j as f64 * self.lon_step, c + (j + 1) as f64 * self.lon_step);
            v = match self.inner.classify(la, lb, lc, ld) {
                Relation::Outside => 0,
                Relation::Inside => 1,
                Relation::Crosses => 2,
            };
            cell.store(v, Relaxed);
        }
        v
    }

    /// How a small box relates to the predicate, from the (cached) grid
    /// cells it falls in: `Inside` / `Outside` when every one of them (at
    /// most four) is, `None` otherwise (larger boxes, boxes leaving the
    /// grid, crossing cells: relate them directly). `inner.classify` must
    /// keep a margin past the cell's edges (boxes are mapped to cells by
    /// index, a box edge may sit a rounding error outside its cell).
    pub fn box_class(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Option<Relation> {
        let (a, b, c, d) = self.bbox;
        if min_lat < a || max_lat > b || min_lon < c || max_lon > d || min_lat > max_lat || min_lon > max_lon {
            return None;
        }
        let cells = self.grid_cells();
        if cells.is_empty() {
            return None;
        }
        let row = |x: f64| (((x - a) / self.lat_step) as usize).min(GRID - 1);
        let col = |x: f64| (((x - c) / self.lon_step) as usize).min(GRID - 1);
        let (i0, i1, j0, j1) = (row(min_lat), row(max_lat), col(min_lon), col(max_lon));
        if (i1 - i0 + 1) * (j1 - j0 + 1) > 4 {
            return None;
        }
        let mut seen = None;
        for i in i0..=i1 {
            for j in j0..=j1 {
                let v = self.cell_class(cells, i, j);
                if v == 2 || seen.is_some_and(|s| s != v) {
                    return None;
                }
                seen = Some(v);
            }
        }
        match seen {
            Some(0) => Some(Relation::Outside),
            Some(1) => Some(Relation::Inside),
            _ => None,
        }
    }
}

/// Most grid cells [`GridPred::classify`] reads for a box before it
/// classifies the box exactly instead.
const GRID_CLASSIFY_CELLS: usize = 64;

impl<P: PointPredicate> PointPredicate for GridPred<P> {
    /// From the cells the box overlaps when they agree (each is classified
    /// once and shared, while a KD walk classifies thousands of leaf boxes
    /// per query): all inside, or all outside, is the box's relation too.
    /// Otherwise the exact classification.
    fn classify(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Relation {
        let (a, b, c, d) = self.bbox;
        if max_lat < a || min_lat > b || max_lon < c || min_lon > d {
            // No point of the box passes `test` (outside the box).
            return Relation::Outside;
        }
        let cells = self.grid_cells();
        if !cells.is_empty() {
            // One extra cell each side: rounding can only add cells.
            let row = |x: f64| ((x - a) / self.lat_step).floor().clamp(0.0, (GRID - 1) as f64) as usize;
            let col = |x: f64| ((x - c) / self.lon_step).floor().clamp(0.0, (GRID - 1) as f64) as usize;
            let (i0, i1) = (row(min_lat).saturating_sub(1), (row(max_lat) + 1).min(GRID - 1));
            let (j0, j1) = (col(min_lon).saturating_sub(1), (col(max_lon) + 1).min(GRID - 1));
            if (i1 - i0 + 1) * (j1 - j0 + 1) <= GRID_CLASSIFY_CELLS {
                // Parts of the box beyond the grid's box pass nothing.
                let within = min_lat >= a && max_lat <= b && min_lon >= c && max_lon <= d;
                let (mut all_in, mut all_out) = (within, true);
                'cells: for i in i0..=i1 {
                    for j in j0..=j1 {
                        match self.cell_class(cells, i, j) {
                            0 => all_in = false,
                            1 => all_out = false,
                            _ => {
                                (all_in, all_out) = (false, false);
                                break 'cells;
                            }
                        }
                        if !all_in && !all_out {
                            break 'cells;
                        }
                    }
                }
                if all_in {
                    return Relation::Inside;
                }
                if all_out {
                    return Relation::Outside;
                }
            }
        }
        self.inner.classify(min_lat, max_lat, min_lon, max_lon)
    }

    /// [`GridPred::test`] of each point, the grid resolved once.
    fn test_many(&self, pts: &[(f64, f64)], out: &mut [bool]) {
        let (a, b, c, d) = self.bbox;
        let cells = self.grid_cells();
        for (o, &(lat, lon)) in out.iter_mut().zip(pts) {
            *o = if lat < a || lat > b || lon < c || lon > d {
                false
            } else {
                let class = if cells.is_empty() {
                    2
                } else {
                    let i = (((lat - a) / self.lat_step) as usize).min(GRID - 1);
                    let j = (((lon - c) / self.lon_step) as usize).min(GRID - 1);
                    self.cell_class(cells, i, j)
                };
                match class {
                    0 => false,
                    1 => true,
                    _ => self.inner.test(lat, lon),
                }
            };
        }
    }

    /// [`GridPred::test`] of each point, the grid resolved once.
    fn count_many(&self, pts: &[(f64, f64)]) -> u64 {
        let (a, b, c, d) = self.bbox;
        let cells = self.grid_cells();
        let mut n = 0u64;
        for &(lat, lon) in pts {
            if lat < a || lat > b || lon < c || lon > d {
                continue;
            }
            let class = if cells.is_empty() {
                2
            } else {
                let i = (((lat - a) / self.lat_step) as usize).min(GRID - 1);
                let j = (((lon - c) / self.lon_step) as usize).min(GRID - 1);
                self.cell_class(cells, i, j)
            };
            n += match class {
                0 => 0,
                1 => 1,
                _ => u64::from(self.inner.test(lat, lon)),
            };
        }
        n
    }

    #[inline]
    fn test(&self, lat: f64, lon: f64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let (a, b, c, d) = self.bbox;
        if lat < a || lat > b || lon < c || lon > d {
            return false;
        }
        let cells = self.cells.get_or_init(|| match &self.key {
            _ if !(self.lat_step > 0.0 && self.lon_step > 0.0) => GridCells::from(Vec::new()),
            Some(k) => shared_cells(k, self.bbox),
            None => new_cells(),
        });
        if !cells.is_empty() {
            // Cell edges are inclusive in `classify`: a point on an edge
            // lies in both cells, and the one picked decides it correctly.
            let i = (((lat - a) / self.lat_step) as usize).min(GRID - 1);
            let j = (((lon - c) / self.lon_step) as usize).min(GRID - 1);
            let cell = &cells[i * GRID + j];
            let mut v = cell.load(Relaxed);
            if v == 3 {
                let (la, lb) = (a + i as f64 * self.lat_step, a + (i + 1) as f64 * self.lat_step);
                let (lc, ld) = (c + j as f64 * self.lon_step, c + (j + 1) as f64 * self.lon_step);
                v = match self.inner.classify(la, lb, lc, ld) {
                    Relation::Outside => 0,
                    Relation::Inside => 1,
                    Relation::Crosses => 2,
                };
                cell.store(v, Relaxed);
            }
            match v {
                0 => return false,
                1 => return true,
                _ => {}
            }
        }
        self.inner.test(lat, lon)
    }
}

/// `GeoShape::Distance` membership with the query's sort key computed once.
#[derive(Debug)]
pub struct DistancePred {
    shape: GeoShape,
    lat: f64,
    lon: f64,
    key: f64,
}

impl DistancePred {
    pub fn new(lat: f64, lon: f64, meters: f64) -> Self {
        DistancePred { shape: GeoShape::Distance { lat, lon, meters }, lat, lon, key: distance_query_sort_key(meters) }
    }
}

impl PointPredicate for DistancePred {
    fn classify(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Relation {
        self.shape.relate(min_lat, max_lat, min_lon, max_lon)
    }

    #[inline]
    fn test(&self, lat: f64, lon: f64) -> bool {
        within_distance_query(self.lat, self.lon, self.key, lat, lon)
    }
}

impl GeoShape {
    /// `(min_lat, max_lat, min_lon, max_lon)` outside of which the shape
    /// matches nothing, when one box covers it (`None`: crosses the dateline).
    pub fn plain_bbox(&self) -> Option<(f64, f64, f64, f64)> {
        let (a, b, r) = self.bbox();
        (r[1].0 > r[1].1).then_some((a, b, r[0].0, r[0].1))
    }
}

/// A boolean combination of point predicates on one geo field: evaluated
/// per point, so it equals the per-doc query only on single-valued fields.
#[derive(Clone, Debug)]
pub enum GeoPred {
    /// Every point (the doc has a value).
    All,
    Leaf(std::sync::Arc<dyn PointPredicate>),
    And(Vec<GeoPred>),
    Not(Box<GeoPred>),
}

impl GeoPred {
    pub fn classify(&self, a: f64, b: f64, c: f64, d: f64) -> Relation {
        match self {
            GeoPred::All => Relation::Inside,
            GeoPred::Leaf(p) => p.classify(a, b, c, d),
            GeoPred::And(v) => {
                let mut all_inside = true;
                for p in v {
                    match p.classify(a, b, c, d) {
                        Relation::Outside => return Relation::Outside,
                        Relation::Crosses => all_inside = false,
                        Relation::Inside => {}
                    }
                }
                if all_inside { Relation::Inside } else { Relation::Crosses }
            }
            GeoPred::Not(p) => match p.classify(a, b, c, d) {
                Relation::Inside => Relation::Outside,
                Relation::Outside => Relation::Inside,
                Relation::Crosses => Relation::Crosses,
            },
        }
    }

    pub fn test(&self, lat: f64, lon: f64) -> bool {
        match self {
            GeoPred::All => true,
            GeoPred::Leaf(p) => p.test(lat, lon),
            GeoPred::And(v) => v.iter().all(|p| p.test(lat, lon)),
            GeoPred::Not(p) => !p.test(lat, lon),
        }
    }

    /// How many of the points pass ([`GeoPred::test`] of each).
    pub fn count_many(&self, pts: &[(f64, f64)]) -> u64 {
        match self {
            GeoPred::All => pts.len() as u64,
            GeoPred::Leaf(p) => p.count_many(pts),
            GeoPred::Not(p) => pts.len() as u64 - p.count_many(pts),
            // The parts' answers for every point, one call per part (no
            // dispatch per point), combined.
            GeoPred::And(_) => {
                let mut mask = vec![true; pts.len()];
                self.and_into(pts, &mut mask);
                mask.iter().filter(|m| **m).count() as u64
            }
        }
    }

    /// `mask[i] &= test(pts[i])` for every point.
    fn and_into(&self, pts: &[(f64, f64)], mask: &mut [bool]) {
        match self {
            GeoPred::All => {}
            GeoPred::And(v) => {
                for p in v {
                    p.and_into(pts, mask);
                }
            }
            GeoPred::Leaf(p) => {
                let mut m = vec![false; pts.len()];
                p.test_many(pts, &mut m);
                for (x, y) in mask.iter_mut().zip(m) {
                    *x &= y;
                }
            }
            GeoPred::Not(p) => {
                let mut m = vec![true; pts.len()];
                p.and_into(pts, &mut m);
                for (x, y) in mask.iter_mut().zip(m) {
                    *x &= !y;
                }
            }
        }
    }

    /// Whether a doc without a point can match (a top-level predicate
    /// must require one).
    pub fn requires_point(&self) -> bool {
        match self {
            GeoPred::All | GeoPred::Leaf(_) => true,
            GeoPred::And(v) => v.iter().any(|p| p.requires_point()),
            GeoPred::Not(_) => false,
        }
    }
}

impl GeoIndex {
    /// No doc has more than one point.
    pub fn single_valued(&self) -> bool {
        self.single
    }

    /// Level-2 block ranges to split a large tree's walk over (one range
    /// for small trees; shards already run in parallel).
    fn split_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let unit = if rayon::current_thread_index().is_some() { PAR_POINTS } else { PAR_POINTS_OFF_POOL };
        self.split_ranges_by(unit)
    }

    /// [`GeoIndex::split_ranges`] for hit walks: each range fills a bitset
    /// of the segment's docs that the merge then ORs, so they split only
    /// large trees ([`PAR_POINTS`]) on or off the pool.
    fn split_hit_ranges(&self) -> Vec<std::ops::Range<usize>> {
        self.split_ranges_by(PAR_POINTS)
    }

    fn split_ranges_by(&self, unit: usize) -> Vec<std::ops::Range<usize>> {
        let points = self.values.len();
        let parts = if points >= unit { (points / unit).clamp(2, PAR_MAX) } else { 1 };
        let n = self.blocks2.len();
        let per = n.div_ceil(parts).max(1);
        (0..n).step_by(per).map(|a| a..(a + per).min(n)).collect()
    }

    /// Docs whose point passes `pred` (single-valued fields only); a large
    /// tree is counted over a few threads.
    pub fn pred_count(&self, pred: &GeoPred) -> Option<u64> {
        if !self.single {
            return None;
        }
        // Every point passes exactly one of `p` and `Not(p)`: count the
        // side whose walk takes the polygon's inside whole (a disjoint
        // count otherwise visits every leaf outside it).
        if let GeoPred::Not(inner) = pred {
            let total: u64 = self.leaves.iter().map(|l| l.len as u64).sum();
            return self.pred_count(inner).map(|n| total - n);
        }
        // `All` passes every point of the tree: `And(All, p)` (a disjoint
        // relation) counts as `p`.
        if let GeoPred::And(v) = pred {
            let mut rest = v.iter().filter(|p| !matches!(p, GeoPred::All));
            if let (Some(only), None) = (rest.next(), rest.next()) {
                return self.pred_count(only);
            }
        }
        let count = |r: std::ops::Range<usize>| {
            let mut n = 0u64;
            let mut buf: Vec<(f64, f64)> = Vec::new();
            let (mut whole, mut tested, mut points) = (0u64, 0u64, 0u64);
            self.walk_range(r, |a, b, c, d| pred.classify(a, b, c, d), |l, inside| {
                n += if inside {
                    whole += 1;
                    l.len as u64
                } else {
                    tested += 1;
                    points += l.len as u64;
                    // Decoded first, then tested in one call (no dispatch per point).
                    buf.clear();
                    self.for_leaf_coords(l, |_, lat, lon| buf.push((decode_lat(lat), decode_lon(lon))));
                    pred.count_many(&buf)
                };
            });
            structures::add(&structures::COUNT_LEAVES_WHOLE, whole);
            structures::add(&structures::COUNT_LEAVES_TESTED, tested);
            structures::add(&structures::COUNT_POINTS_TESTED, points);
            n
        };
        let ranges = self.split_ranges();
        if ranges.len() <= 1 {
            return Some(ranges.into_iter().map(count).sum());
        }
        // The ranges on Rayon (the current pool, else the global one): a
        // thread spawned per range cost more than the walk on the
        // mid-sized segments of a many-segment shard.
        use rayon::prelude::*;
        Some(ranges.into_par_iter().map(count).sum())
    }

    /// [`GeoIndex::pred_count`] capped at `limit` (`min(count, limit)`,
    /// exact): the leaves the predicate takes whole count first, from
    /// their sizes alone, the walk stopping once they reach the limit; only
    /// when they fall short are the crossing leaves' points tested.
    pub fn pred_count_at_least(&self, pred: &GeoPred, limit: u64) -> Option<u64> {
        if !self.single {
            return None;
        }
        let whole = std::cell::Cell::new(0u64);
        self.walk(
            |a, b, c, d| if whole.get() >= limit { Relation::Outside } else { pred.classify(a, b, c, d) },
            |l, inside| {
                if inside {
                    whole.set(whole.get() + l.len as u64);
                }
            },
        );
        if whole.get() >= limit {
            return Some(limit);
        }
        self.pred_count(pred).map(|n| n.min(limit))
    }

    /// Docs whose point passes `pred` (single-valued fields only); a large
    /// tree is walked over a few threads (their bitsets OR-ed).
    pub fn pred_hits(&self, pred: &GeoPred, num_docs: u32) -> Option<DocHits> {
        if !self.single {
            return None;
        }
        Some(self.walk_hits(pred, num_docs))
    }

    /// Docs with at least one point passing `pred`, on any field (single
    /// or multi-valued). Only for predicates whose per-doc meaning is "some
    /// point passes" (plain shapes, `intersects`); a large tree is walked
    /// over a few threads like [`GeoIndex::pred_hits`].
    pub fn any_point_hits(&self, pred: &GeoPred, num_docs: u32) -> DocHits {
        self.walk_hits(pred, num_docs)
    }

    /// The docs of the points passing `pred` (duplicates collapsed).
    fn walk_hits(&self, pred: &GeoPred, num_docs: u32) -> DocHits {
        let collect = |r: std::ops::Range<usize>| {
            let mut hits = HitCollector::new(num_docs);
            self.walk_range(r, |a, b, c, d| pred.classify(a, b, c, d), |l, inside| {
                if inside {
                    self.for_leaf_docs(l, |d| hits.push(d));
                } else {
                    self.for_leaf_coords(l, |i, lat, lon| if pred.test(decode_lat(lat), decode_lon(lon)) { hits.push(self.leaf_doc(l, i)) });
                }
            });
            hits
        };
        let ranges = self.split_hit_ranges();
        if ranges.len() <= 1 {
            return collect(0..self.blocks2.len()).finish();
        }
        // On Rayon (the current pool, else the global one), not a thread
        // spawned per range.
        let parts: Vec<HitCollector> = {
            use rayon::prelude::*;
            ranges.into_par_iter().map(collect).collect()
        };
        let mut merged = HitCollector::new(num_docs);
        for part in parts {
            match part.finish() {
                DocHits::Sorted(v) => v.into_iter().for_each(|d| merged.push(d)),
                DocHits::Bits(b) => merged.or_bits(&b),
            }
        }
        merged.finish()
    }

    /// Every leaf `pred` does not rule out, one after another: its decoded
    /// box `(min_lat, max_lat, min_lon, max_lon)`, its number of points,
    /// whether `pred` holds for every point of it, and a reader of its
    /// points' decoded `(lat, lon)` (the caller tests `pred` on those of
    /// leaves not inside). Aggregations that need a leaf's points only when
    /// its box does not decide them read nothing else.
    pub fn visit_pred_leaves(&self, pred: &GeoPred, mut f: impl FnMut((f64, f64, f64, f64), u32, bool, &mut dyn FnMut(&mut dyn FnMut(f64, f64)))) {
        self.walk(|a, b, c, d| pred.classify(a, b, c, d), |l, inside| {
            let bx = (decode_lat(l.min_lat), decode_lat(l.max_lat), decode_lon(l.min_lon), decode_lon(l.max_lon));
            f(bx, l.len, inside, &mut |g: &mut dyn FnMut(f64, f64)| self.for_leaf_coords(l, |_, lat, lon| g(decode_lat(lat), decode_lon(lon))));
        });
    }

    /// Points in the leaves `pred` does not rule out (an upper bound of
    /// its matches, from the leaf boxes alone).
    pub fn pred_estimate(&self, pred: &GeoPred) -> u64 {
        let mut n = 0u64;
        self.walk(|a, b, c, d| pred.classify(a, b, c, d), |l, _| n += l.len as u64);
        n
    }
}

// ---- hit collection ----------------------------------------------------------

/// Docs a geo query matched: ascending ids, or a bitset when dense.
#[derive(Debug)]
pub enum DocHits {
    Sorted(Vec<DocId>),
    /// Bit `d % 64` of word `d / 64` is doc `d`.
    Bits(Vec<u64>),
}

impl DocHits {
    /// Number of docs.
    pub fn len(&self) -> usize {
        match self {
            DocHits::Sorted(v) => v.len(),
            DocHits::Bits(w) => w.iter().map(|x| x.count_ones() as usize).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            DocHits::Sorted(v) => v.is_empty(),
            DocHits::Bits(w) => w.iter().all(|x| *x == 0),
        }
    }

    /// The docs, ascending.
    pub fn into_sorted(self) -> Vec<DocId> {
        match self {
            DocHits::Sorted(v) => v,
            DocHits::Bits(w) => {
                let mut out = Vec::new();
                for (i, mut x) in w.into_iter().enumerate() {
                    while x != 0 {
                        out.push(i as DocId * 64 + x.trailing_zeros());
                        x &= x - 1;
                    }
                }
                out
            }
        }
    }

    /// Keep the docs `keep` accepts.
    pub fn retain(&mut self, mut keep: impl FnMut(DocId) -> bool) {
        match self {
            DocHits::Sorted(v) => v.retain(|d| keep(*d)),
            DocHits::Bits(w) => {
                for (i, word) in w.iter_mut().enumerate() {
                    let mut x = *word;
                    while x != 0 {
                        let b = x.trailing_zeros();
                        x &= x - 1;
                        if !keep(i as DocId * 64 + b) {
                            *word &= !(1u64 << b);
                        }
                    }
                }
            }
        }
    }
}

/// Collects matching docs in any order (duplicates allowed): a vector
/// while the hits are sparse, a bitset over the leaf's docs once they
/// are dense (no sort of millions of ids).
pub struct HitCollector {
    vec: Vec<DocId>,
    bits: Option<Vec<u64>>,
    num_docs: u32,
    limit: usize,
}

impl HitCollector {
    pub fn new(num_docs: u32) -> Self {
        HitCollector { vec: Vec::new(), bits: None, num_docs, limit: (num_docs as usize / 32).max(4096) }
    }

    #[inline]
    pub fn push(&mut self, d: DocId) {
        if let Some(b) = &mut self.bits {
            b[d as usize / 64] |= 1u64 << (d % 64);
            return;
        }
        self.vec.push(d);
        if self.vec.len() > self.limit {
            let mut b = vec![0u64; (self.num_docs as usize).div_ceil(64).max(self.vec.iter().map(|&d| d as usize / 64 + 1).max().unwrap_or(0))];
            for &d in &self.vec {
                b[d as usize / 64] |= 1u64 << (d % 64);
            }
            self.vec = Vec::new();
            self.bits = Some(b);
        }
    }

    /// Add every doc of a bitset.
    pub fn or_bits(&mut self, b: &[u64]) {
        if self.bits.is_none() {
            let words = (self.num_docs as usize).div_ceil(64).max(b.len());
            let mut w = vec![0u64; words];
            for &d in &self.vec {
                w[d as usize / 64] |= 1u64 << (d % 64);
            }
            self.vec = Vec::new();
            self.bits = Some(w);
        }
        let w = self.bits.as_mut().unwrap();
        if w.len() < b.len() {
            w.resize(b.len(), 0);
        }
        w.iter_mut().zip(b).for_each(|(x, y)| *x |= y);
    }

    pub fn finish(self) -> DocHits {
        match self.bits {
            Some(b) => DocHits::Bits(b),
            None => {
                let mut v = self.vec;
                v.sort_unstable();
                v.dedup();
                DocHits::Sorted(v)
            }
        }
    }
}

// ---- grid keys --------------------------------------------------------------

const GEOHASH_BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// [`geohash`] as its `5 * precision` bits (no string).
pub fn geohash_bits(lat: f64, lon: f64, precision: usize) -> u64 {
    let (mut lat_lo, mut lat_hi) = (-90.0f64, 90.0f64);
    let (mut lon_lo, mut lon_hi) = (-180.0f64, 180.0f64);
    let mut bits = 0u64;
    for i in 0..precision * 5 {
        if i % 2 == 0 {
            let mid = (lon_lo + lon_hi) / 2.0;
            if lon >= mid {
                bits = (bits << 1) | 1;
                lon_lo = mid;
            } else {
                bits <<= 1;
                lon_hi = mid;
            }
        } else {
            let mid = (lat_lo + lat_hi) / 2.0;
            if lat >= mid {
                bits = (bits << 1) | 1;
                lat_lo = mid;
            } else {
                bits <<= 1;
                lat_hi = mid;
            }
        }
    }
    bits
}

/// [`geohash_bits`] of the point Lucene decodes from the encoded
/// `(lat_e, lon_e)` (each up to `2^31`, the far edge of the last quantum):
/// the bisections of a decoded value are its offset's top bits (both are
/// exact binary fractions of the same ranges), interleaved longitude first.
pub fn geohash_bits_encoded(lat_e: i64, lon_e: i64, precision: usize) -> u64 {
    let top = (1u64 << 32) - 1;
    let lat_u = ((lat_e + (1 << 31)) as u64).min(top);
    let lon_u = ((lon_e + (1 << 31)) as u64).min(top);
    let mut bits = 0u64;
    let (mut li, mut ai) = (0u32, 0u32);
    for i in 0..precision * 5 {
        let bit = if i % 2 == 0 {
            li += 1;
            (lon_u >> (32 - li)) & 1
        } else {
            ai += 1;
            (lat_u >> (32 - ai)) & 1
        };
        bits = (bits << 1) | bit;
    }
    bits
}

/// The geohash string of [`geohash_bits`].
pub fn geohash_of_bits(bits: u64, precision: usize) -> String {
    (0..precision).map(|i| GEOHASH_BASE32[((bits >> (5 * (precision - 1 - i))) & 31) as usize] as char).collect()
}

/// Geohash of a point at `precision` characters (1..=12).
pub fn geohash(lat: f64, lon: f64, precision: usize) -> String {
    let (mut lat_lo, mut lat_hi) = (-90.0f64, 90.0f64);
    let (mut lon_lo, mut lon_hi) = (-180.0f64, 180.0f64);
    let mut out = String::with_capacity(precision);
    let mut even = true;
    let mut bit = 0;
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
        bit += 1;
        if bit == 5 {
            out.push(GEOHASH_BASE32[ch] as char);
            bit = 0;
            ch = 0;
        }
    }
    out
}

/// Bounding box `(top, left, bottom, right)` of a geohash cell.
pub fn geohash_bbox(hash: &str) -> Option<(f64, f64, f64, f64)> {
    let (mut lat_lo, mut lat_hi) = (-90.0f64, 90.0f64);
    let (mut lon_lo, mut lon_hi) = (-180.0f64, 180.0f64);
    let mut even = true;
    for c in hash.bytes() {
        let v = GEOHASH_BASE32.iter().position(|&b| b == c.to_ascii_lowercase())?;
        for i in (0..5).rev() {
            let bit = (v >> i) & 1 == 1;
            if even {
                let mid = (lon_lo + lon_hi) / 2.0;
                if bit {
                    lon_lo = mid
                } else {
                    lon_hi = mid
                }
            } else {
                let mid = (lat_lo + lat_hi) / 2.0;
                if bit {
                    lat_lo = mid
                } else {
                    lat_hi = mid
                }
            }
            even = !even;
        }
    }
    Some((lat_hi, lon_lo, lat_lo, lon_hi))
}

/// Max latitude of the web-mercator tile grid.
pub const MERCATOR_MAX_LAT: f64 = 85.051_128_779_806_59;

/// Web-mercator tile `(x, y)` of a point at `zoom` (ES `geotile_grid`).
pub fn geotile(lat: f64, lon: f64, zoom: u32) -> (u32, u32) {
    let tiles = 1u64 << zoom;
    let lat = lat.clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
    let x = ((lon + 180.0) / 360.0 * tiles as f64).floor();
    let r = lat.to_radians();
    let y = ((1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * tiles as f64).floor();
    let clamp = |v: f64| v.max(0.0).min((tiles - 1) as f64) as u32;
    (clamp(x), clamp(y))
}

/// Bounding box `(top, left, bottom, right)` of a web-mercator tile.
pub fn geotile_bbox(zoom: u32, x: u32, y: u32) -> (f64, f64, f64, f64) {
    let n = (1u64 << zoom) as f64;
    let lon = |x: f64| x / n * 360.0 - 180.0;
    let lat = |y: f64| (std::f64::consts::PI * (1.0 - 2.0 * y / n)).sinh().atan().to_degrees();
    (lat(y as f64), lon(x as f64), lat(y as f64 + 1.0), lon(x as f64 + 1.0))
}

#[cfg(test)]
mod tests {
    /// The encoded geohash is the bisection of the decoded point, at
    /// every precision, including the last quantum's far edge.
    #[test]
    fn geohash_bits_encoded_matches_bisection() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for i in 0..200_000 {
            let (a, o) = match i {
                0 => (i32::MAX as i64 + 1, i32::MAX as i64 + 1),
                1 => (i32::MIN as i64, i32::MIN as i64),
                _ => ((next() as u32 as i32) as i64, (next() as u32 as i32) as i64),
            };
            let (lat, lon) = ((a as f64) * super::decode_lat(1), (o as f64) * super::decode_lon(1));
            for p in [1usize, 2, 5, 8, 12] {
                assert_eq!(super::geohash_bits_encoded(a, o, p), super::geohash_bits(lat.min(90.0), lon.min(180.0), p), "{a} {o} {p}");
            }
        }
    }

    #[test]
    fn point_value_count_preserves_duplicates_and_missing_docs() {
        let mut values = super::GeoValues::default();
        assert_eq!(values.value_count(0), 0);
        values.push(1, 20.0, 10.0);
        values.push(1, 20.0, 10.0);
        values.push(1, 40.0, 30.0);
        values.push(4, 0.0, 0.0);
        for (doc, expected) in [(0, 0), (1, 3), (2, 0), (3, 0), (4, 1), (5, 0), (u32::MAX, 0)] {
            assert_eq!(values.value_count(doc), expected);
            assert_eq!(values.value_count(doc), values.get(doc).len());
        }
    }

    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn encoding_roundtrip() {
        for (lat, lon) in [(0.0, 0.0), (90.0, 180.0), (-90.0, -180.0), (41.12, -71.34), (-33.8688, 151.2093)] {
            let (a, b) = quantize(lat, lon);
            assert!((a - lat).abs() < 1e-7 && (b - lon).abs() < 1e-7, "{lat},{lon} -> {a},{b}");
        }
        assert!(encode_lat(10.0) < encode_lat(10.000001));
        assert!(encode_lon(-1.0) < encode_lon(1.0));
    }

    #[test]
    fn haversine_known_distances() {
        // London -> Paris ~ 343.5 km.
        let d = haversine_meters(51.5074, -0.1278, 48.8566, 2.3522);
        assert!((d - 343_556.0).abs() < 1_000.0, "{d}");
        assert_eq!(haversine_meters(10.0, 10.0, 10.0, 10.0), 0.0);
    }

    #[test]
    fn mapped_index_reads_in_place_at_any_alignment() {
        // A sealed segment's geo section starts at any byte offset of its
        // file: every offset must read the same points and answer the same
        // queries as the built index, without heap copies of the points.
        let mut r = Lcg(11);
        let mut vals = GeoValues::default();
        for doc in 0..3000u32 {
            for _ in 0..(doc % 3) {
                vals.push(doc, r.next() * 180.0 - 90.0, r.next() * 360.0 - 180.0);
            }
        }
        let built = GeoIndex::build(vals.to_vec());
        let mut enc = Vec::new();
        built.encode(&mut enc);
        let shape = GeoShape::Distance { lat: 10.0, lon: 20.0, meters: 4_000_000.0 };
        for pad in 0..4 {
            let mut file = vec![0xAA; pad];
            file.extend_from_slice(&enc);
            file.extend_from_slice(&[0xBB; 3]);
            let file = FileSlice::from_vec(file);
            let idx = GeoIndex::open(&file.slice(pad..pad + enc.len())).unwrap();
            assert_eq!(idx.values().to_vec(), built.values().to_vec(), "pad {pad}");
            assert_eq!(idx.query(&shape), built.query(&shape), "pad {pad}");
            assert_eq!(idx.count(&shape), built.count(&shape), "pad {pad}");
            assert_eq!(idx.values().get(1001), built.values().get(1001), "pad {pad}");
            assert_eq!(idx.single_valued(), built.single_valued());
            if cfg!(target_endian = "little") {
                assert_eq!(idx.values().heap_bytes(), 0, "pad {pad}");
            }
        }
        // Truncated or out-of-range leaves are corruption, not a panic.
        assert!(GeoIndex::open(&FileSlice::from_vec(enc[..enc.len() - 1].to_vec())).is_err());
        let mut bad = enc.clone();
        let n = u32::from_le_bytes(bad[..4].try_into().unwrap()) as usize;
        let first_leaf_len = 4 + n * ENTRY_BYTES + 4 + 20;
        bad[first_leaf_len..first_leaf_len + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(GeoIndex::open(&FileSlice::from_vec(bad)).is_err());
    }

    #[test]
    fn kd_queries_match_brute_force() {
        let mut r = Lcg(7);
        let mut pts = Vec::new();
        let mut vals = GeoValues::default();
        for doc in 0..5000u32 {
            for _ in 0..(doc % 3) {
                let lat = r.next() * 180.0 - 90.0;
                let lon = r.next() * 360.0 - 180.0;
                vals.push(doc, lat, lon);
            }
        }
        pts.extend(vals.iter());
        let idx = GeoIndex::build(pts);
        let mut buf = Vec::new();
        idx.encode(&mut buf);
        let idx = GeoIndex::decode(&buf).unwrap();
        assert!(idx.num_leaves() > 5);
        for q in 0..60 {
            let shape = match q % 3 {
                0 => GeoShape::Distance { lat: r.next() * 170.0 - 85.0, lon: r.next() * 360.0 - 180.0, meters: r.next() * 3_000_000.0 },
                1 => {
                    let (b, t) = (r.next() * 180.0 - 90.0, r.next() * 180.0 - 90.0);
                    GeoShape::BoundingBox { top: t.max(b), bottom: t.min(b), left: r.next() * 360.0 - 180.0, right: r.next() * 360.0 - 180.0 }
                }
                _ => {
                    let (clat, clon) = (r.next() * 120.0 - 60.0, r.next() * 300.0 - 150.0);
                    let pts: Vec<(f64, f64)> = (0..6)
                        .map(|i| {
                            let a = i as f64 / 6.0 * std::f64::consts::TAU;
                            let rad = 5.0 + r.next() * 20.0;
                            (clat + rad * a.sin(), clon + rad * a.cos())
                        })
                        .collect();
                    GeoShape::Polygon { points: pts }
                }
            };
            let mut want: Vec<DocId> = Vec::new();
            for p in vals.iter() {
                if shape.contains(decode_lat(p.lat), decode_lon(p.lon)) && want.last() != Some(&{ p.doc }) {
                    want.push(p.doc);
                }
            }
            assert_eq!(idx.query(&shape), want, "shape {shape:?}");
            assert_eq!(vals.query(&shape), want);
        }
        // Small queries skip most leaves.
        let mut visited = 0;
        idx.query_counting(&GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 50_000.0 }, &mut visited);
        assert!(visited < idx.num_leaves() / 2, "{visited} of {}", idx.num_leaves());
    }

    #[test]
    fn grids() {
        assert_eq!(geohash(57.64911, 10.40744, 11), "u4pruydqqvj");
        let (t, l, b, rr) = geohash_bbox("u4pruydqqvj").unwrap();
        assert!(b <= 57.64911 && t >= 57.64911 && l <= 10.40744 && rr >= 10.40744);
        assert_eq!(geotile(0.0, 0.0, 1), (1, 1));
        assert_eq!(geotile(52.374, 4.9, 8), (131, 84));
        let (t, l, b, rr) = geotile_bbox(8, 131, 84);
        assert!(b <= 52.374 && t >= 52.374 && l <= 4.9 && rr >= 4.9);
    }

    #[test]
    fn polygon_and_dateline_box() {
        let sq = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        assert!(point_in_polygon(&sq, 5.0, 5.0));
        assert!(!point_in_polygon(&sq, 15.0, 5.0));
        let b = GeoShape::BoundingBox { top: 10.0, left: 170.0, bottom: -10.0, right: -170.0 };
        assert!(b.contains(0.0, 175.0) && b.contains(0.0, -175.0) && !b.contains(0.0, 0.0));
    }
}

// ---- Lucene SloppyMath ------------------------------------------------------

/// Lucene `SloppyMath`'s look-up tables (cos / asin approximations).
struct SloppyTabs {
    sin: Vec<f64>,
    cos: Vec<f64>,
    asin: Vec<f64>,
    d1: Vec<f64>,
    d2: Vec<f64>,
    d3: Vec<f64>,
    d4: Vec<f64>,
}

const SC_TABS: usize = (1 << 11) + 1;
const ASIN_TABS: usize = (1 << 13) + 1;
const PIO2_HI: f64 = f64::from_bits(0x3FF921FB54400000);
const PIO2_LO: f64 = f64::from_bits(0x3DD0B4611A626331);
const SC_DELTA_HI: f64 = 4.0 * PIO2_HI / (SC_TABS - 1) as f64;
const SC_DELTA_LO: f64 = 4.0 * PIO2_LO / (SC_TABS - 1) as f64;

fn sc_indexer() -> f64 {
    1.0 / (SC_DELTA_HI + SC_DELTA_LO)
}

fn asin_max() -> f64 {
    crate::fdlibm::sin(73.0f64.to_radians())
}

fn sloppy_tabs() -> &'static SloppyTabs {
    static TABS: std::sync::OnceLock<SloppyTabs> = std::sync::OnceLock::new();
    TABS.get_or_init(|| {
        let mut t = SloppyTabs { sin: vec![0.0; SC_TABS], cos: vec![0.0; SC_TABS], asin: vec![0.0; ASIN_TABS], d1: vec![0.0; ASIN_TABS], d2: vec![0.0; ASIN_TABS], d3: vec![0.0; ASIN_TABS], d4: vec![0.0; ASIN_TABS] };
        let pi = (SC_TABS - 1) / 2;
        for i in 0..SC_TABS {
            let a = i as f64 * SC_DELTA_HI + i as f64 * SC_DELTA_LO;
            let (mut s, mut c) = (crate::fdlibm::sin(a), crate::fdlibm::cos(a));
            if i == pi || i == 2 * pi {
                s = 0.0;
            } else if i == pi / 2 || i == 3 * pi / 2 {
                c = 0.0;
            }
            t.sin[i] = s;
            t.cos[i] = c;
        }
        let delta = asin_max() / (ASIN_TABS - 1) as f64;
        for i in 0..ASIN_TABS {
            let x = i as f64 * delta;
            t.asin[i] = crate::fdlibm::asin(x);
            let o = 1.0 / (1.0 - x * x);
            let o05 = o.sqrt();
            let o15 = o05 * o;
            let o25 = o15 * o;
            let o35 = o25 * o;
            t.d1[i] = o05;
            t.d2[i] = (x * o15) * (1.0 / 2.0);
            t.d3[i] = ((1.0 + 2.0 * x * x) * o25) * (1.0 / 6.0);
            t.d4[i] = ((5.0 + 2.0 * x * (2.0 + x * (5.0 - 2.0 * x))) * o35) * (1.0 / 24.0);
        }
        t
    })
}

/// Lucene `SloppyMath.cos`.
pub fn sloppy_cos(a: f64) -> f64 {
    let a = a.abs();
    let indexer = sc_indexer();
    if a > ((i32::MAX >> 9) as f64 / indexer) * 0.99 {
        return crate::fdlibm::cos(a);
    }
    let t = sloppy_tabs();
    let mut index = (a * indexer + 0.5) as i32;
    let delta = (a - index as f64 * SC_DELTA_HI) - index as f64 * SC_DELTA_LO;
    index &= (SC_TABS - 2) as i32;
    let (c, s) = (t.cos[index as usize], t.sin[index as usize]);
    c + delta * (-s + delta * (-c * (1.0 / 2.0) + delta * (s * (1.0 / 6.0) + delta * c * (1.0 / 24.0))))
}

/// Lucene `SloppyMath.asin`.
pub fn sloppy_asin(a: f64) -> f64 {
    let neg = a < 0.0;
    let a = a.abs();
    let max = asin_max();
    let r = if a <= max {
        let t = sloppy_tabs();
        let delta_tab = max / (ASIN_TABS - 1) as f64;
        let index = (a * (1.0 / delta_tab) + 0.5) as usize;
        let d = a - index as f64 * delta_tab;
        t.asin[index] + d * (t.d1[index] + d * (t.d2[index] + d * (t.d3[index] + d * t.d4[index])))
    } else if a < 1.0 {
        const PS: [u64; 6] = [0x3fc5555555555555, 0xbfd4d61203eb6f7d, 0x3fc9c1550e884455, 0xbfa48228b5688f3b, 0x3f49efe07501b288, 0x3f023de10dfdf709];
        const QS: [u64; 4] = [0xc0033a271c8a2d4b, 0x40002ae59c598ac8, 0xbfe6066c1b8d0159, 0x3fb3b8c5b12e9282];
        let ps = PS.map(f64::from_bits);
        let qs = QS.map(f64::from_bits);
        let t = (1.0 - a) * 0.5;
        let p = t * (ps[0] + t * (ps[1] + t * (ps[2] + t * (ps[3] + t * (ps[4] + t * ps[5])))));
        let q = 1.0 + t * (qs[0] + t * (qs[1] + t * (qs[2] + t * qs[3])));
        let s = t.sqrt();
        let z = s + s * (p / q);
        f64::from_bits(0x3FF921FB54442D18) - ((z + z) - f64::from_bits(0x3C91A62633145C07))
    } else if a == 1.0 {
        std::f64::consts::FRAC_PI_2
    } else {
        f64::NAN
    };
    if neg {
        -r
    } else {
        r
    }
}

/// Lucene `SloppyMath.haversinSortKey`.
pub fn sloppy_haversin_sort_key(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let x1 = lat1.to_radians();
    let x2 = lat2.to_radians();
    let h1 = 1.0 - sloppy_cos(x1 - x2);
    let h2 = 1.0 - sloppy_cos((lon1 - lon2).to_radians());
    let h = h1 + sloppy_cos(x1) * sloppy_cos(x2) * h2;
    f64::from_bits(h.to_bits() & 0xFFFF_FFFF_FFFF_FFF8)
}

/// Lucene `SloppyMath.haversinMeters` (what ES|QL `ST_DISTANCE` reports
/// for geo points, over their quantized coordinates).
pub fn sloppy_haversin_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let k = sloppy_haversin_sort_key(lat1, lon1, lat2, lon2);
    EARTH_MEAN_RADIUS_METERS * 2.0 * sloppy_asin((k * 0.5).sqrt().min(1.0))
}

#[cfg(test)]
mod hits_tests {
    use super::*;

    #[test]
    fn collector_switches_to_bits_and_matches_query() {
        let mut pts = Vec::new();
        for d in 0..20_000u32 {
            pts.push(GeoPointEntry { doc: d, lat: encode_lat((d % 170) as f64 - 85.0), lon: encode_lon((d % 350) as f64 - 175.0) });
        }
        let idx = GeoIndex::build(pts);
        for shape in [
            GeoShape::BoundingBox { top: 80.0, left: -170.0, bottom: -80.0, right: 170.0 },
            GeoShape::BoundingBox { top: 1.5, left: 0.0, bottom: 0.5, right: 3.0 },
            GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 900_000.0 },
        ] {
            let hits = idx.query_hits(&shape, 20_000);
            let want = idx.query(&shape);
            assert_eq!(idx.count(&shape), Some(want.len() as u64), "{shape:?}");
            assert_eq!(hits.into_sorted(), want, "{shape:?}");
        }
        // The grid accelerator decides points exactly like the shape.
        let shape = GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 900_000.0 };
        let grid = GridPred::new(shape.clone(), shape.plain_bbox().unwrap());
        let mut r = 12345u64;
        for _ in 0..200_000 {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let lat = 10.0 + ((r >> 33) % 2_000_000) as f64 / 100_000.0 - 10.0;
            let lon = 10.0 + ((r >> 13) % 2_000_000) as f64 / 100_000.0 - 10.0;
            assert_eq!(grid.test(lat, lon), shape.contains(lat, lon), "{lat} {lon}");
        }
        let fast = DistancePred::new(10.0, 10.0, 900_000.0);
        let mut r = 777u64;
        for _ in 0..200_000 {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let lat = ((r >> 33) % 2_000_000) as f64 / 100_000.0;
            let lon = ((r >> 13) % 2_000_000) as f64 / 100_000.0;
            assert_eq!(fast.test(lat, lon), shape.contains(lat, lon), "{lat} {lon}");
        }
        // A ring: within 900 km, not within 300 km.
        let ring = GeoPred::And(vec![
            GeoPred::Leaf(std::sync::Arc::new(GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 900_000.0 })),
            GeoPred::Not(Box::new(GeoPred::Leaf(std::sync::Arc::new(GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 300_000.0 })))),
        ]);
        let inner = idx.query(&GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 300_000.0 });
        let want: Vec<DocId> = idx.query(&GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 900_000.0 }).into_iter().filter(|d| !inner.contains(d)).collect();
        assert_eq!(idx.pred_count(&ring), Some(want.len() as u64));
        assert_eq!(idx.pred_hits(&ring, 20_000).unwrap().into_sorted(), want);
        assert!(idx.pred_estimate(&ring) >= want.len() as u64);
        // Several points per doc: no leaf counts.
        let multi = GeoIndex::build(vec![GeoPointEntry { doc: 0, lat: 1, lon: 1 }, GeoPointEntry { doc: 0, lat: 2, lon: 2 }]);
        assert_eq!(multi.count(&GeoShape::BoundingBox { top: 1.0, left: -1.0, bottom: -1.0, right: 1.0 }), None);
        let mut c = HitCollector::new(100_000);
        for d in (0..10_000u32).rev().chain(0..10) {
            c.push(d);
        }
        let h = c.finish();
        assert!(matches!(h, DocHits::Bits(_)));
        assert_eq!(h.into_sorted(), (0..10_000).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod box_distance_tests {
    use super::*;

    /// Distance relations rule out boxes by their latitude gap when the
    /// nearest distance is not computed (far in longitude, polar centres).
    #[test]
    fn far_and_polar_boxes_are_ruled_out_by_latitude() {
        // A cap around the south pole: boxes far in longitude from the
        // centre's (meaningless) longitude but far in latitude are outside;
        // boxes reaching the cap are not.
        let cap = GeoShape::Distance { lat: -90.0, lon: 0.0, meters: 300_000.0 };
        assert_eq!(cap.relate(-60.0, -50.0, 120.0, 130.0), Relation::Outside);
        assert_eq!(cap.relate(10.0, 20.0, -170.0, -160.0), Relation::Outside);
        assert_ne!(cap.relate(-89.9, -89.0, 120.0, 130.0), Relation::Outside);
        // Away from the poles, a box 100° of longitude away and 30° of
        // latitude off a 1000 km circle.
        let c = GeoShape::Distance { lat: 40.0, lon: 0.0, meters: 1_000_000.0 };
        assert_eq!(c.relate(70.0, 80.0, 100.0, 110.0), Relation::Outside);
        // Conservative: no box holding a point of the shape is outside.
        let mut r = 3u64;
        let mut next = || {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20_000 {
            let s = [cap.clone(), c.clone()][(next() * 2.0) as usize % 2].clone();
            let (a, c0) = (-90.0 + 175.0 * next(), -180.0 + 350.0 * next());
            let (b, d) = ((a + 15.0 * next()).min(90.0), (c0 + 20.0 * next()).min(180.0));
            if s.relate(a, b, c0, d) != Relation::Outside {
                continue;
            }
            for i in 0..=10 {
                for j in 0..=10 {
                    let (p, q) = (a + (b - a) * i as f64 / 10.0, c0 + (d - c0) * j as f64 / 10.0);
                    assert!(!s.contains(p, q), "{s:?} box {a},{b},{c0},{d} holds {p},{q}");
                }
            }
        }
    }

    /// `min_box_distance` is the minimum over the box (sampled densely), never more.
    #[test]
    fn min_box_distance_is_the_minimum() {
        let mut r = 5u64;
        let mut next = |m: u64| {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 33) % m
        };
        for _ in 0..2000 {
            let lat = next(16_000) as f64 / 100.0 - 80.0;
            let lon = next(36_000) as f64 / 100.0 - 180.0;
            let a = next(16_000) as f64 / 100.0 - 80.0;
            let b = (a + next(1_000) as f64 / 100.0).min(85.0);
            let c = next(35_000) as f64 / 100.0 - 175.0;
            let d = (c + next(2_000) as f64 / 100.0).min(180.0);
            let Some(m) = min_box_distance(lat, lon, a, b, c, d) else { continue };
            let mut best = f64::MAX;
            for i in 0..=40 {
                for j in 0..=40 {
                    let (p, q) = (a + (b - a) * i as f64 / 40.0, c + (d - c) * j as f64 / 40.0);
                    best = best.min(haversine_meters(lat, lon, p, q));
                }
            }
            assert!(m <= best * (1.0 + 1e-9) + 1e-6, "{lat},{lon} box {a},{b},{c},{d}: {m} > {best}");
            // And close to it (edges are sampled at 1/40 steps).
            assert!(best - m <= (b - a).max(d - c) / 40.0 * 111_200.0 + 1.0, "{lat},{lon} box {a},{b},{c},{d}: {m} vs {best}");
        }
    }
}

#[cfg(test)]
mod par_tests {
    use super::*;

    /// A tree large enough to be walked over several threads gives the
    /// single-threaded results.
    #[test]
    fn split_walks_match_serial() {
        let n = PAR_POINTS as u32 + 200_000;
        let mut r = 99u64;
        let pts: Vec<GeoPointEntry> = (0..n)
            .map(|d| {
                r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let lat = ((r >> 33) % 3_000_000) as f64 / 100_000.0 + 35.0;
                let lon = ((r >> 11) % 4_000_000) as f64 / 100_000.0 - 15.0;
                GeoPointEntry { doc: d, lat: encode_lat(lat), lon: encode_lon(lon) }
            })
            .collect();
        let idx = GeoIndex::build(pts);
        assert!(idx.split_ranges().len() > 1);
        let ring = GeoPred::And(vec![
            GeoPred::Leaf(std::sync::Arc::new(GeoShape::Distance { lat: 50.0, lon: 5.0, meters: 900_000.0 })),
            GeoPred::Not(Box::new(GeoPred::Leaf(std::sync::Arc::new(GeoShape::Distance { lat: 50.0, lon: 5.0, meters: 300_000.0 })))),
        ]);
        let serial = idx.visit_count(|a, b, c, d| ring.classify(a, b, c, d), |p| ring.test(decode_lat(p.lat), decode_lon(p.lon))).unwrap();
        assert!(serial > 0);
        assert_eq!(idx.pred_count(&ring), Some(serial));
        let hits = idx.pred_hits(&ring, n).unwrap();
        assert_eq!(hits.len() as u64, serial);
        let mut want = Vec::new();
        idx.visit(|a, b, c, d| ring.classify(a, b, c, d), |p, inside| {
            if inside || ring.test(decode_lat(p.lat), decode_lon(p.lon)) {
                want.push(p.doc);
            }
        });
        want.sort_unstable();
        assert_eq!(hits.into_sorted(), want);
    }
}

/// Lucene `SloppyMath.haversinMeters(sortKey)`.
pub fn sloppy_haversin_meters_of_key(key: f64) -> f64 {
    EARTH_MEAN_RADIUS_METERS * 2.0 * sloppy_asin((key * 0.5).sqrt().min(1.0))
}

/// Lucene `GeoUtils.distanceQuerySortKey`: the smallest haversine sort key
/// whose distance is at least `radius` (a point is within a distance query
/// when its `haversinSortKey` is at most this).
pub fn distance_query_sort_key(radius: f64) -> f64 {
    let max = sloppy_haversin_meters_of_key(f64::MAX);
    if radius >= max {
        return max;
    }
    let (mut lo, mut hi): (i64, i64) = (0, f64::MAX.to_bits() as i64);
    while lo <= hi {
        let mid = ((lo as u64 + hi as u64) >> 1) as i64;
        let key = f64::from_bits(mid as u64);
        let r = sloppy_haversin_meters_of_key(key);
        if r == radius {
            return key;
        } else if r > radius {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    f64::from_bits(lo as u64)
}

/// [`distance_query_sort_key`], memoized per thread for the last radius.
fn cached_query_key(radius: f64) -> f64 {
    // A few radii per thread: a ring (`d <= R2 and not d <= R1`) tests two
    // per point, and a single slot recomputed the key on every test.
    thread_local! {
        static LAST: std::cell::Cell<[(u64, f64); 4]> = const { std::cell::Cell::new([(u64::MAX, 0.0); 4]) };
    }
    LAST.with(|c| {
        let mut slots = c.get();
        let bits = radius.to_bits();
        if let Some(&(_, k)) = slots.iter().find(|(r, _)| *r == bits) {
            return k;
        }
        let k = distance_query_sort_key(radius);
        slots.rotate_right(1);
        slots[0] = (bits, k);
        c.set(slots);
        k
    })
}

/// Whether a point is within a Lucene distance query (`LatLonPoint`
/// distance / circle queries, which ES uses for `geo_distance` and the
/// ES|QL `ST_DISTANCE` pushdown).
#[inline]
pub fn within_distance_query(clat: f64, clon: f64, key: f64, lat: f64, lon: f64) -> bool {
    sloppy_haversin_sort_key(clat, clon, lat, lon) <= key
}

#[cfg(test)]
mod geohash_bits_tests {
    use super::*;

    #[test]
    fn bits_round_trip_to_the_string() {
        let mut r = 17u64;
        for _ in 0..5000 {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let lat = ((r >> 33) % 18_000_000) as f64 / 100_000.0 - 90.0;
            let lon = ((r >> 11) % 36_000_000) as f64 / 100_000.0 - 180.0;
            for p in 1..=12 {
                assert_eq!(geohash_of_bits(geohash_bits(lat, lon, p), p), geohash(lat, lon, p));
            }
        }
    }
}

#[cfg(test)]
mod lucene_distance_tests {
    use super::*;

    /// Values from Lucene 10 (ES 9.5.5) `SloppyMath` / `GeoUtils`.
    #[test]
    fn matches_lucene() {
        let cases: [([f64; 4], u64, u64); 6] = [
            ([55.0, 7.0, 54.8007158, 7.9364927], 4679007600559170343, 4542534738601712616),
            ([55.0, 7.0, 54.86124896, 8.29365346], 4680514125166272840, 4546053512601079768),
            ([-33.9, 151.2, 51.5, -0.12], 4715327525810814463, 4611189718666387752),
            ([10.0, 10.0, 10.0001, 10.0001], 4624975299858171178, 4434466821519025976),
            ([0.0, 0.0, 0.0, 179.9], 4716134991707366308, 4611686011568015152),
            ([89.9, 0.0, -89.9, 180.0], 4716137976577560110, 4611686018427387904),
        ];
        for (p, hav, key) in cases {
            assert_eq!(sloppy_haversin_meters(p[0], p[1], p[2], p[3]).to_bits(), hav, "{p:?}");
            assert_eq!(sloppy_haversin_sort_key(p[0], p[1], p[2], p[3]).to_bits(), key, "{p:?}");
        }
        let keys: [(f64, u64); 8] = [
            (100000.0, 4548676610789792966),
            (300000.0, 4562754643341039304),
            (400000.0, 4566689609917245768),
            (73204.77053637545, 4544499302941600416),
            (1.0, 4398817337607445649),
            (0.0, 0),
            (2e7, 4611686005754014103),
            (5e7, 4716137976577560110),
        ];
        for (r, k) in keys {
            assert_eq!(distance_query_sort_key(r).to_bits(), k, "{r}");
        }
    }
}

#[cfg(test)]
mod sloppy_tests {
    use super::*;

    /// Values ES 9.5.5 returns for `ST_DISTANCE` (quantized inputs).
    #[test]
    fn matches_es() {
        let q = |lat: f64, lon: f64| quantize(lat, lon);
        let (a, b) = (q(55.0, 7.0), q(54.861249, 8.2936535));
        assert_eq!(sloppy_haversin_meters(a.0, a.1, b.0, b.1), 84076.77943034639);
        let c2 = q(54.8007158, 7.9364927);
        assert_eq!(sloppy_haversin_meters(a.0, a.1, c2.0, c2.1), 63844.98386476892);
        let c = q(55.4224547, 8.1786536);
        assert_eq!(sloppy_haversin_meters(a.0, a.1, c.0, c.1), 88305.77649659535);
        assert_eq!(sloppy_haversin_meters(a.0, a.1, a.0, a.1), 0.0);
    }
}

#[cfg(test)]
mod nearest_visit_tests {
    use super::*;
    #[test]
    fn distance_order_visits_every_point_and_stops_on_request() {
        let points: Vec<_> = (0..5000u32).map(|doc| GeoPointEntry {
            doc, lat: encode_lat((doc % 179) as f64 - 89.0),
            lon: encode_lon((doc % 359) as f64 - 179.0),
        }).collect();
        let index = GeoIndex::build(points);
        for (lat, lon) in [(52.37, 4.8951), (89.9, 179.9), (-89.9, -179.9)] {
            let mut previous = -1.0;
            let mut seen = Vec::new();
            index.visit_by_distance(lat, lon, |bound, points| {
                assert!(bound >= previous); previous = bound;
                for point in points {
                    let distance = sloppy_haversin_meters(lat, lon, decode_lat(point.lat), decode_lon(point.lon));
                    assert!(bound <= distance, "{bound} > {distance}");
                    seen.push(point.doc);
                }
                true
            });
            seen.sort_unstable();
            assert_eq!(seen, (0..5000).collect::<Vec<_>>());
            let mut calls = 0;
            index.visit_by_distance(lat, lon, |_, _| { calls += 1; false });
            assert_eq!(calls, 1);
        }
    }
}

#[cfg(test)]
mod best_first_visit_tests {
    use super::*;

    /// The best-first walk visits leaves in exactly the order (bounds,
    /// points, ties by leaf) of sorting every leaf's bound, over clustered
    /// and repeated points (equal bounds) and origins inside, beside and
    /// antipodal to the points, stopping where the visitor asks.
    #[test]
    fn best_first_visits_match_sorting_every_leaf() {
        let mut seed = 7u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 11) as f64 / (1u64 << 53) as f64 };
        let mut points = Vec::new();
        for doc in 0..300_000u32 {
            let (lat, lon) = match doc % 4 {
                0 => (52.0 + rnd() * 2.0, 4.0 + rnd() * 2.0),
                1 => (-89.0 + rnd() * 178.0, -179.0 + rnd() * 358.0),
                2 => (48.85, 2.35),
                _ => (((doc / 4) % 50) as f64 - 25.0, ((doc / 4) % 70) as f64 - 35.0),
            };
            points.push(GeoPointEntry { doc, lat: encode_lat(lat), lon: encode_lon(lon) });
        }
        let index = GeoIndex::build(points);
        assert!(index.blocks2.len() > 1);
        for (lat, lon) in [(52.37, 4.8951), (48.85, 2.35), (0.0, 0.0), (-52.37, -175.1), (89.9, 179.9), (10.0, 0.0)] {
            for stop in [1usize, 7, 64, usize::MAX] {
                let (mut a, mut b) = (Vec::new(), Vec::new());
                index.visit_by_distance(lat, lon, |bound, pts| { a.push((bound.to_bits(), pts.to_vec())); a.len() < stop });
                index.visit_by_distance_sorted(lat, lon, |bound, pts| { b.push((bound.to_bits(), pts.to_vec())); b.len() < stop });
                assert_eq!(a.len(), b.len(), "({lat}, {lon}) stop {stop}");
                assert!(a == b, "({lat}, {lon}) stop {stop}: visit orders differ");
            }
        }
    }

    /// `cargo test -p rustic-core --release --lib distance_visit_microbench -- --ignored --nocapture`
    /// (`GEO_BENCH_POINTS`, default 2.3M: a geonames shard): Rally
    /// geonames' decay_geo_gauss top 10 near Amsterdam (stop once 10
    /// points are seen), best first against sorting every leaf, best of 20.
    #[test]
    #[ignore]
    fn distance_visit_microbench() {
        let n: u32 = std::env::var("GEO_BENCH_POINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(2_300_000);
        let mut seed = 11u64;
        let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 11) as f64 / (1u64 << 53) as f64 };
        let points: Vec<GeoPointEntry> = (0..n).map(|doc| GeoPointEntry { doc, lat: encode_lat(-60.0 + rnd() * 130.0), lon: encode_lon(-180.0 + rnd() * 360.0) }).collect();
        let index = GeoIndex::build(points);
        let best = |f: &dyn Fn() -> usize| (0..20).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(f()); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min);
        let run = |sorted: bool| {
            let mut seen = 0usize;
            let v = |_: f64, p: &[GeoPointEntry]| { seen += p.len(); seen < 10 };
            if sorted { index.visit_by_distance_sorted(52.37, 4.8951, v) } else { index.visit_by_distance(52.37, 4.8951, v) }
            seen
        };
        eprintln!("{n} points, {} leaves: best first {:.3} ms, sort every leaf {:.3} ms", index.num_leaves(), best(&|| run(false)), best(&|| run(true)));
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;
    use std::sync::Arc;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `kind` 0: every doc one point (dense), 1: some docs without a
    /// point, 2: some docs with several. Clustered like stations: many docs
    /// at a few hundred places, plus spread points, poles and the dateline.
    fn points(n: u32, kind: u8, seed: u64) -> Vec<GeoPointEntry> {
        let mut r = Lcg(seed);
        let places: Vec<(f64, f64)> = (0..300).map(|_| (r.next() * 170.0 - 85.0, r.next() * 360.0 - 180.0)).collect();
        let mut out = Vec::new();
        for d in 0..n {
            if kind == 1 && d % 7 == 3 {
                continue;
            }
            let k = if kind == 2 && d % 5 == 0 { 3 } else { 1 };
            for _ in 0..k {
                let (lat, lon) = match d % 10 {
                    0 => (r.next() * 180.0 - 90.0, r.next() * 360.0 - 180.0),
                    1 => (89.9 + r.next() * 0.1, if r.next() < 0.5 { 179.99 } else { -180.0 }),
                    _ => places[(r.next() * places.len() as f64) as usize],
                };
                out.push(GeoPointEntry { doc: d, lat: encode_lat(lat), lon: encode_lon(lon) });
            }
        }
        out
    }

    fn shapes(r: &mut Lcg) -> Vec<GeoShape> {
        let mut v = vec![
            GeoShape::BoundingBox { top: 90.0, left: 170.0, bottom: 80.0, right: -170.0 },
            GeoShape::BoundingBox { top: 60.0, left: -10.0, bottom: 30.0, right: 40.0 },
            GeoShape::Distance { lat: 89.95, lon: 179.995, meters: 50_000.0 },
        ];
        for _ in 0..20 {
            v.push(GeoShape::Distance { lat: r.next() * 170.0 - 85.0, lon: r.next() * 360.0 - 180.0, meters: r.next() * 3_000_000.0 });
            let (clat, clon) = (r.next() * 120.0 - 60.0, r.next() * 300.0 - 150.0);
            let pts: Vec<(f64, f64)> = (0..7).map(|i| {
                let a = i as f64 / 7.0 * std::f64::consts::TAU;
                let rad = 5.0 + r.next() * 30.0;
                (clat + rad * a.sin(), clon + rad * a.cos())
            }).collect();
            v.push(GeoShape::Polygon { points: pts });
        }
        v
    }

    fn reopen(idx: &GeoIndex, compact: bool) -> (GeoIndex, usize) {
        let mut b = Vec::new();
        if compact { idx.encode_compact(&mut b) } else { idx.encode(&mut b) }
        let len = b.len();
        let file = FileSlice::from_vec(b);
        (if compact { GeoIndex::open_compact(&file) } else { GeoIndex::open(&file) }.unwrap(), len)
    }

    fn sorted(h: DocHits) -> Vec<DocId> {
        h.into_sorted()
    }

    /// The compact layout (dense values, packed tree) answers every read
    /// exactly as the v1 layout: values, per-doc points, counts, hits,
    /// distance-ordered leaves; dense only when every doc has one point.
    #[test]
    fn compact_layout_reads_like_v1() {
        for kind in [0u8, 1, 2] {
            let n = 20_000;
            let built = GeoIndex::build(points(n, kind, 11 + kind as u64));
            let (v1, v1_len) = reopen(&built, false);
            let (v2, v2_len) = reopen(&built, true);
            assert_eq!(v2.values().as_slice().is_none(), kind == 0, "dense iff every doc has one point");
            if kind == 0 {
                assert!(v2_len * 10 < v1_len * 8, "compact {v2_len} vs v1 {v1_len}");
            }
            assert_eq!(v2.values().to_vec(), v1.values().to_vec());
            assert_eq!(v2.single_valued(), v1.single_valued());
            assert_eq!(v2.docs_with_points(), v1.docs_with_points());
            for d in (0..n + 3).step_by(7) {
                assert_eq!(v2.values().get(d), v1.values().get(d), "doc {d}");
                assert_eq!(v2.values().value_count(d), v1.values().value_count(d));
                let (a, b) = (v2.values().doc_range(d), v1.values().doc_range(d));
                assert_eq!(v2.values().entries_of(a).iter().collect::<Vec<_>>(), v1.values().entries_of(b).iter().collect::<Vec<_>>());
                assert_eq!(v2.values().lower_bound(d, 100), v1.values().lower_bound(d, 100));
            }
            let mut r = Lcg(5);
            for shape in shapes(&mut r) {
                assert_eq!(v2.query(&shape), v1.query(&shape), "{shape:?}");
                assert_eq!(v2.count(&shape), v1.count(&shape));
                assert_eq!(sorted(v2.query_hits(&shape, n)), sorted(v1.query_hits(&shape, n)));
                let pred = GeoPred::Leaf(Arc::new(shape.clone()));
                assert_eq!(v2.pred_count(&pred), v1.pred_count(&pred));
                assert_eq!(v2.pred_estimate(&pred), v1.pred_estimate(&pred));
                assert_eq!(v2.pred_hits(&pred, n).map(sorted), v1.pred_hits(&pred, n).map(sorted));
                assert_eq!(sorted(v2.any_point_hits(&pred, n)), sorted(v1.any_point_hits(&pred, n)));
                let mut a = Vec::new();
                v2.visit(|w, x, y, z| shape.relate(w, x, y, z), |p, inside| a.push((*p, inside)));
                let mut b = Vec::new();
                v1.visit(|w, x, y, z| shape.relate(w, x, y, z), |p, inside| b.push((*p, inside)));
                assert_eq!(a, b);
            }
            let (mut a, mut b) = (Vec::new(), Vec::new());
            v2.visit_by_distance(48.0, 2.0, |bound, pts| { a.push((bound.to_bits(), pts.to_vec())); a.len() < 50 });
            v1.visit_by_distance(48.0, 2.0, |bound, pts| { b.push((bound.to_bits(), pts.to_vec())); b.len() < 50 });
            assert_eq!(a, b);
        }
    }

    /// `cargo test -p rustic-core --release --lib geo_compact_microbench -- --ignored --nocapture`
    /// (`GEO_BENCH_POINTS`, default 4M): geopoint-shaped queries (box,
    /// distance, polygon, counts, distance-ordered top 10, doc-order scan)
    /// on the v1 and the compact layout of the same points, best of 5.
    #[test]
    #[ignore]
    fn geo_compact_microbench() {
        let n: u32 = std::env::var("GEO_BENCH_POINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(4_000_000);
        let built = GeoIndex::build(points(n, 0, 3));
        let (v1, l1) = reopen(&built, false);
        let (v2, l2) = reopen(&built, true);
        drop(built);
        eprintln!("geo index bytes: v1 {l1} ({:.1} B/pt), compact {l2} ({:.1} B/pt)", l1 as f64 / n as f64, l2 as f64 / n as f64);
        let mut r = Lcg(9);
        let qs = shapes(&mut r);
        let time = |label: &str, f: &dyn Fn(&GeoIndex) -> u64| {
            let best = |g: &GeoIndex| (0..5).map(|_| { let t = std::time::Instant::now(); let x = f(g); std::hint::black_box(x); t.elapsed().as_secs_f64() }).fold(f64::INFINITY, f64::min);
            let (a, b) = (best(&v1), best(&v2));
            eprintln!("{label:<24} v1 {:>9.3} ms  compact {:>9.3} ms  ({:+.1}%)", a * 1e3, b * 1e3, (b / a - 1.0) * 100.0);
        };
        time("hits (box/dist/poly)", &|g| qs.iter().map(|s| g.query_hits(s, n).len() as u64).sum());
        time("pred_count", &|g| qs.iter().map(|s| g.pred_count(&GeoPred::Leaf(Arc::new(s.clone()))).unwrap_or(0)).sum());
        time("pred_hits", &|g| qs.iter().map(|s| g.pred_hits(&GeoPred::Leaf(Arc::new(s.clone())), n).map_or(0, |h| h.len() as u64)).sum());
        time("count", &|g| qs.iter().map(|s| g.count(s).unwrap_or(0)).sum());
        time("distance top-10 visit", &|g| {
            let mut total = 0u64;
            for (lat, lon) in [(48.0, 2.0), (0.0, 0.0), (-33.0, 151.0), (40.7, -74.0)] {
                let mut seen = 0usize;
                g.visit_by_distance(lat, lon, |_, pts| { seen += pts.len(); seen < 10 });
                total += seen as u64;
            }
            total
        });
        time("doc-order scan", &|g| g.values().iter().map(|p| (p.lat as i64 + p.lon as i64) as u64).fold(0u64, u64::wrapping_add));
        time("per-doc get", &|g| (0..n).step_by(97).map(|d| g.values().value_count(d) as u64).sum());
    }
}

#[cfg(test)]
mod grid_cache_tests {
    use super::*;

    /// Small boxes claimed inside / outside from the grid's cells hold
    /// only points the shape contains / does not contain.
    #[test]
    fn box_class_claims_are_conservative() {
        // A circle: its relation decides cells inside and outside it.
        let shape = GeoShape::Distance { lat: 52.0, lon: 6.0, meters: 400_000.0 };
        let b = shape.plain_bbox().unwrap();
        let g = GridPred::new(shape.clone(), b);
        let mut r = 41u64;
        let mut next = || {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut inside, mut outside) = (0, 0);
        for _ in 0..20_000 {
            let (lat, lon) = (b.0 + (b.1 - b.0) * next(), b.2 + (b.3 - b.2) * next());
            let (h, w) = (0.1 * next() * next(), 0.1 * next() * next());
            let (lat1, lon1) = ((lat + h).min(b.1), (lon + w).min(b.3));
            let Some(rel) = g.box_class(lat, lat1, lon, lon1) else { continue };
            if rel == Relation::Inside { inside += 1 } else { outside += 1 }
            for k in 0..9 {
                let (p, q) = (lat + (lat1 - lat) * (k / 3) as f64 / 2.0, lon + (lon1 - lon) * (k % 3) as f64 / 2.0);
                assert_eq!(shape.contains(p, q), rel == Relation::Inside, "box {lat},{lat1},{lon},{lon1} point {p},{q}");
            }
        }
        assert!(inside > 1000 && outside > 1000, "{inside} {outside}");
    }

    /// Keyed grids of one key and box share their cells (what one
    /// classifies the next reads) and decide points like the shape; a
    /// different key or box gets its own grid.
    #[test]
    fn keyed_grids_share_cells_and_match_the_shape() {
        let shape = GeoShape::Distance { lat: 10.0, lon: 10.0, meters: 900_000.0 };
        let b = shape.plain_bbox().unwrap();
        let key = format!("grid cache test {shape:?}");
        let (a, c) = (GridPred::keyed(shape.clone(), b, key.clone()), GridPred::keyed(shape.clone(), b, key.clone()));
        let mut r = 99u64;
        let mut next = || {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (r >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..50_000 {
            let (lat, lon) = (b.0 + (b.1 - b.0) * next(), b.2 + (b.3 - b.2) * next());
            assert_eq!(a.test(lat, lon), shape.contains(lat, lon));
        }
        let (ca, cc) = (a.cells.get().unwrap(), {
            c.test(10.0, 10.0);
            c.cells.get().unwrap()
        });
        assert!(std::sync::Arc::ptr_eq(ca, cc));
        assert!(ca.iter().any(|x| x.load(std::sync::atomic::Ordering::Relaxed) != 3));
        for _ in 0..50_000 {
            let (lat, lon) = (b.0 + (b.1 - b.0) * next(), b.2 + (b.3 - b.2) * next());
            assert_eq!(c.test(lat, lon), shape.contains(lat, lon));
        }
        let other = GridPred::keyed(shape.clone(), b, format!("{key} other"));
        other.test(10.0, 10.0);
        assert!(!std::sync::Arc::ptr_eq(ca, other.cells.get().unwrap()));
        let moved = GridPred::keyed(shape.clone(), (b.0, b.1 + 1.0, b.2, b.3), key);
        moved.test(10.0, 10.0);
        assert!(!std::sync::Arc::ptr_eq(ca, moved.cells.get().unwrap()));
    }
}
