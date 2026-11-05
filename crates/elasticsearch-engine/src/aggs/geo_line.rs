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
        // (sort value, lon, lat) of every doc having both.
        let mut pts: Vec<(f64, f64, f64)> = Vec::new();
        if sf.ty.is_some() {
            for (s, v) in docs.iter().enumerate() {
                for &(d, _) in v {
                    let Some(&(lat, lon)) = self.geo_points(point, s, d).first() else { continue };
                    let Some(k) = self.values(&sf, s, d).into_iter().find_map(|k| k.as_f64()) else { continue };
                    pts.push((k, lon, lat));
                }
            }
        }
        pts.sort_by(|a, b| if desc { b.0.total_cmp(&a.0) } else { a.0.total_cmp(&b.0) });
        let complete = pts.len() <= size;
        pts.truncate(size);
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
        Ok(json!({
            "type": "Feature",
            "geometry": geometry,
            "properties": props,
        }))
    }
}
