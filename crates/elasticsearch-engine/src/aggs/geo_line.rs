//! `geo_line`: the bucket's `geo_point` values joined into a GeoJSON
//! `LineString` Feature, ordered by a numeric `sort` field (ES x-pack
//! spatial `GeoLineAggregationBuilder` / `InternalGeoLine`).

use super::{Ctx, Docs};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use crate::values;
use serde_json::{json, Map, Value};

/// ES `InternalGeoLine.MAX_PATH_SIZE`.
const MAX_SIZE: u64 = 10_000;

/// ES rounds output coordinates to 6 decimals (`Math.round(v * 1e6) / 1e6`).
fn round6(v: f64) -> f64 {
    (v * 1e6 + 0.5).floor() / 1e6
}

/// Marks the sub-aggregations of a `time_series` aggregation.
pub(super) struct TimeSeriesScope<'c>(Option<&'c std::cell::Cell<u32>>);

impl<'c> TimeSeriesScope<'c> {
    pub(super) fn enter(c: &'c std::cell::Cell<u32>, on: bool) -> Self {
        if on {
            c.set(c.get() + 1);
            TimeSeriesScope(Some(c))
        } else {
            TimeSeriesScope(None)
        }
    }
}

impl Drop for TimeSeriesScope<'_> {
    fn drop(&mut self) {
        if let Some(c) = self.0 {
            c.set(c.get() - 1);
        }
    }
}

fn sub_field<'o>(o: &'o Map<String, Value>, k: &str) -> Option<&'o str> {
    o.get(k).and_then(|v| v.get("field")).and_then(Value::as_str)
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub(super) fn geo_line(&self, _name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let size = match o.get("size") {
            None => MAX_SIZE,
            Some(v) => match v.as_i64() {
                Some(n) if n > 0 && n as u64 <= MAX_SIZE => n as u64,
                _ => {
                    return Err(EsError::x_content_parse("[1:1] [geo_line] failed to parse field [size]")
                        .caused_by(EsError::illegal_argument(format!("invalid [size] value [{v}] must be a positive integer <= {MAX_SIZE}"))))
                }
            },
        } as usize;
        let include_sort = o.get("include_sort").and_then(Value::as_bool).unwrap_or(false);
        let desc = match o.get("sort_order").and_then(Value::as_str) {
            None => false,
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "asc" => false,
                "desc" => true,
                _ => return Err(EsError::illegal_argument(format!("Unknown SortOrder [{s}]"))),
            },
        };
        // Point: a geo_point field.
        let point = sub_field(o, "point").unwrap_or("");
        let mut point_mapped = false;
        for s in self.segs {
            if let Some(rf) = s.lookup.get(point) {
                if rf.declared != EsType::GeoPoint {
                    return Err(EsError::illegal_argument(format!("Expected numeric type on field [{point}], but got [{}]", rf.declared.name())));
                }
                point_mapped = true;
            }
        }
        if !point_mapped {
            return Err(EsError::illegal_argument("field [point] is not a geo_point type"));
        }
        // Under `time_series` the line follows `@timestamp` (ES `TimeSeriesGeoLineBuckets`).
        let in_ts = self.in_time_series.get() > 0;
        let sort = match sub_field(o, "sort") {
            Some(s) if in_ts && s != "@timestamp" => {
                return Err(EsError::illegal_argument(format!("invalid field [sort]='{s}' configured for time-series aggregations")));
            }
            Some(s) => s,
            None if in_ts => "@timestamp",
            None => return Err(EsError::illegal_argument("missing field [sort] configured for geo_line aggregations")),
        };
        let sf = self.source_field(sort).map_err(|_| EsError::illegal_argument("field [sort] is not a numeric type"))?;
        if let Some(t) = sf.ty {
            if !(t.is_integer() || t.is_float() || t.is_date() || t == EsType::Boolean) {
                return Err(EsError::illegal_argument("field [sort] is not a numeric type"));
            }
        }
        // (sort value, lon, lat, arrival) of the best `size` docs having
        // both, selected as they arrive (ties in arrival order, as the
        // stable sort of every point did): only those are held, charged to
        // the request breaker; every point is counted for `complete`.
        let mut best = super::TopN::new(size, |a: &(f64, f64, f64, u64), b: &(f64, f64, f64, u64)| {
            (if desc { b.0.total_cmp(&a.0) } else { a.0.total_cmp(&b.0) }).then(a.3.cmp(&b.3))
        });
        let label = super::agg_label("geo_line");
        let mut seen = 0u64;
        let mut visited = 0u64;
        // Under `time_series`, ES simplifies the whole route to `size`
        // points (`TimeSeriesGeoLineBuckets`: a streaming triangle-area
        // simplifier over the series' native, time-descending order) rather
        // than keeping its first `size` points.
        if in_ts && sf.ty.is_some() {
            let mut all: Vec<(f64, f64, f64)> = Vec::new();
            for (s, v) in docs.iter().enumerate() {
                for &(d, _) in v {
                    visited += 1;
                    if visited % super::POLL_EVERY == 0 {
                        super::poll_cancelled()?;
                    }
                    let Some(&(lat, lon)) = self.geo_points(point, s, d).first() else { continue };
                    let Some(k) = self.values(&sf, s, d).into_iter().find_map(|k| k.as_f64()) else { continue };
                    self.mem.add(24, &label)?;
                    all.push((k, lon, lat));
                }
            }
            all.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut simp = Simplifier::new(size);
            for p in &all {
                simp.consume(*p);
            }
            let complete = simp.order.len() < size;
            let mut pts: Vec<(f64, f64, f64)> = simp.into_points().into_iter().map(|p| (p.sort, p.x, p.y)).collect();
            if !desc {
                pts.reverse();
            }
            return Ok(line_feature(pts.iter().map(|p| (p.0, p.1, p.2)), complete, include_sort));
        }
        if sf.ty.is_some() {
            for (s, v) in docs.iter().enumerate() {
                for &(d, _) in v {
                    visited += 1;
                    if visited % super::POLL_EVERY == 0 {
                        super::poll_cancelled()?;
                    }
                    let Some(&(lat, lon)) = self.geo_points(point, s, d).first() else { continue };
                    let Some(k) = self.values(&sf, s, d).into_iter().find_map(|k| k.as_f64()) else { continue };
                    seen += 1;
                    best.push(&self.mem, &label, (k, lon, lat, seen), 0)?;
                }
            }
        }
        let complete = seen as usize <= size;
        let pts: Vec<(f64, f64, f64, u64)> = best.finish(&self.mem);
        Ok(line_feature(pts.iter().map(|p| (p.0, p.1, p.2)), complete, include_sort))
    }
}

/// The GeoJSON feature of a line of `(sort, lon, lat)` points.
fn line_feature(pts: impl Iterator<Item = (f64, f64, f64)>, complete: bool, include_sort: bool) -> Value {
    let pts: Vec<(f64, f64, f64)> = pts.collect();
    let mut coords: Vec<Value> = pts.iter().map(|p| json!([values::java_double(round6(p.1)), values::java_double(round6(p.2))])).collect();
    // ES renders a one-point line as a GeoJSON Point.
    let geometry = if coords.len() == 1 {
        json!({ "coordinates": coords.pop().unwrap(), "type": "Point" })
    } else {
        json!({ "coordinates": coords, "type": "LineString" })
    };
    let mut props = Map::new();
    props.insert("complete".into(), json!(complete));
    if include_sort {
        props.insert("sort_values".into(), Value::Array(pts.iter().map(|p| values::java_double(p.0)).collect()));
    }
    json!({
        "type": "Feature",
        "geometry": geometry,
        "properties": props,
    })
}

struct SimplePoint {
    x: f64,
    y: f64,
    sort: f64,
}

/// ES `StreamingGeometrySimplifier` with `SimplificationErrorCalculator
/// .TRIANGLE_AREA` (spherical, Heron), ported step for step: at most `max` points; a new
/// point past `max` evicts the head of a `java.util.PriorityQueue` of the
/// interior points by triangle area (the endpoints stay), and the
/// neighbours' areas are recomputed. The queue is Java's binary heap
/// (`siftUp` / `siftDown` / `removeAt`, `compareTo` = the sign of the area
/// difference), so equal areas leave in Java's order.
struct Simplifier {
    max: usize,
    /// The kept points' ids, in order (`points[0..length]`).
    order: Vec<usize>,
    /// Every point by id.
    pts: Vec<SimplePoint>,
    /// The `error` of every point by id.
    err: Vec<f64>,
    /// `PriorityQueue.queue[0..size]`: point ids.
    heap: Vec<usize>,
}

impl Simplifier {
    fn new(max: usize) -> Self {
        Simplifier { max, order: Vec::new(), pts: Vec::new(), err: Vec::new(), heap: Vec::new() }
    }

    /// The kept points, in order.
    fn into_points(self) -> Vec<SimplePoint> {
        let mut pts: Vec<Option<SimplePoint>> = self.pts.into_iter().map(Some).collect();
        self.order.iter().filter_map(|&i| pts[i].take()).collect()
    }

    /// ES `SimplificationErrorCalculator.TriangleAreaCalculator`: Heron's
    /// formula over Lucene's sloppy haversine side lengths (0 for a flat
    /// triangle rounding made negative).
    fn area(&self, left: usize, middle: usize, right: usize) -> f64 {
        let (l, m, r) = (&self.pts[left], &self.pts[middle], &self.pts[right]);
        let d = |p: &SimplePoint, q: &SimplePoint| rustic_core::geo::sloppy_haversin_meters(p.y, p.x, q.y, q.x);
        let (a, b, c) = (d(l, r), d(r, m), d(m, l));
        let s = 0.5 * (a + b + c);
        let (da, db, dc) = (s - a, s - b, s - c);
        if da >= 0.0 && db >= 0.0 && dc >= 0.0 { (s * da * db * dc).sqrt() } else { 0.0 }
    }

    /// `PointError.compareTo`: `(int) Math.signum(this.error - o.error)`.
    fn cmp(&self, a: usize, b: usize) -> i32 {
        let d = self.err[a] - self.err[b];
        if d > 0.0 {
            1
        } else if d < 0.0 {
            -1
        } else {
            0
        }
    }

    fn sift_up(&mut self, mut k: usize, x: usize) {
        while k > 0 {
            let parent = (k - 1) >> 1;
            let e = self.heap[parent];
            if self.cmp(x, e) >= 0 {
                break;
            }
            self.heap[k] = e;
            k = parent;
        }
        self.heap[k] = x;
    }

    fn sift_down(&mut self, mut k: usize, x: usize, n: usize) {
        let half = n >> 1;
        while k < half {
            let mut child = 2 * k + 1;
            let mut c = self.heap[child];
            let right = child + 1;
            if right < n && self.cmp(c, self.heap[right]) > 0 {
                child = right;
                c = self.heap[child];
            }
            if self.cmp(x, c) <= 0 {
                break;
            }
            self.heap[k] = c;
            k = child;
        }
        self.heap[k] = x;
    }

    fn offer(&mut self, x: usize) {
        self.heap.push(x);
        let k = self.heap.len() - 1;
        self.sift_up(k, x);
    }

    fn poll(&mut self) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        let result = self.heap[0];
        let x = self.heap.pop()?;
        let n = self.heap.len();
        if n > 0 {
            self.sift_down(0, x, n);
        }
        Some(result)
    }

    /// `PriorityQueue.remove(Object)` (identity) / `removeAt`.
    fn remove(&mut self, id: usize) -> bool {
        let Some(i) = self.heap.iter().position(|&h| h == id) else { return false };
        let s = self.heap.len() - 1;
        if s == i {
            self.heap.pop();
        } else {
            let moved = self.heap.pop().unwrap_or(id);
            self.heap[i] = moved;
            self.sift_down(i, moved, s);
            if self.heap[i] == moved {
                self.sift_up(i, moved);
            }
        }
        true
    }

    fn update_error_at(&mut self, index: usize) {
        if index > 0 && index + 1 < self.order.len() {
            let (a, b, c) = (self.order[index - 1], self.order[index], self.order[index + 1]);
            let error = self.area(a, b, c);
            let delta = (error - self.err[b]).abs();
            self.err[b] = error;
            if delta > 1e-10 && self.remove(b) {
                self.offer(b);
            }
        }
    }

    fn consume(&mut self, (sort, x, y): (f64, f64, f64)) {
        let id = self.pts.len();
        self.pts.push(SimplePoint { x, y, sort });
        self.err.push(0.0);
        let length = self.order.len();
        if length > 1 {
            let last = self.order[length - 1];
            self.err[last] = self.area(self.order[length - 2], last, id);
            self.offer(last);
        }
        if self.max == 0 {
            return;
        }
        if length == self.max {
            // ES `removeAndAdd` (a queue always holds a point here once
            // `max` >= 3; with fewer, nothing interior can go).
            let Some(to_remove) = self.poll() else {
                self.order.pop();
                self.order.push(id);
                return;
            };
            let at = self.order.iter().position(|&p| p == to_remove).unwrap_or(1);
            self.order.remove(at);
            self.order.push(id);
            self.update_error_at(at.saturating_sub(1));
            self.update_error_at(at);
            if at < self.max - 1 {
                self.update_error_at(self.max - 2);
            }
        } else {
            self.order.push(id);
        }
    }
}

#[cfg(test)]
mod simplifier_tests {
    use super::*;

    /// #489: the kept points equal ES's `StreamingGeometrySimplifier` with
    /// `TRIANGLE_AREA` over `java.util.PriorityQueue`, ties included (the
    /// expected lists are the Java program's output: ES's simplifier logic
    /// on JDK 21 with Lucene 10.2.2 `SloppyMath`).
    #[test]
    fn simplifier_matches_es_including_ties() {
        let mut cases: Vec<(usize, Vec<(f64, f64, f64)>)> = Vec::new();
        cases.push((5, (0..12).map(|i| (12.0 - i as f64, i as f64, 0.0)).collect()));
        cases.push((6, (0..15).map(|i| (15.0 - i as f64, i as f64, (i % 2) as f64)).collect()));
        let mut st: u64 = 17;
        let mut next = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 33) % 2000) as f64 / 100.0
        };
        let c: Vec<(f64, f64, f64)> = (0..40).map(|i| {
            let lon = next();
            let lat = next();
            (40.0 - i as f64, lon, lat)
        }).collect();
        cases.push((7, c));
        cases.push((4, (0..9).map(|i| (9.0 - i as f64, (i * 2) as f64, if i % 3 == 0 { 1.0 } else { 0.0 })).collect()));
        let want: [&[i64]; 4] = [&[12, 9, 7, 5, 1], &[15, 14, 11, 7, 2, 1], &[40, 39, 38, 26, 7, 4, 1], &[9, 7, 4, 1]];
        for ((max, pts), want) in cases.into_iter().zip(want) {
            let mut s = Simplifier::new(max);
            for p in pts {
                s.consume(p);
            }
            let got: Vec<i64> = s.into_points().iter().map(|p| p.sort as i64).collect();
            assert_eq!(got, want, "max {max}");
        }
    }
}
