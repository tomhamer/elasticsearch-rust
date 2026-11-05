//! `circle` processor: approximates a circle with a regular polygon whose
//! sagitta stays below `error_distance` (ES `CircleProcessor`).

use super::doc::{type_name, IngestDocument};
use super::jtext::java_double;
use super::processors::{ia, input};
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use serde_json::{json, Value};

const EARTH_RADIUS: f64 = 6371008.7714;

pub struct CircleP {
    pub field: String,
    pub target: String,
    pub error_distance: f64,
    pub geo: bool,
    pub ignore_missing: bool,
}

struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

fn unit_meters(u: &str) -> Option<f64> {
    Some(match u {
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

fn parse_radius(v: &Value, geo: bool) -> std::result::Result<f64, String> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => {
            let t = s.trim();
            let idx = t.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')).unwrap_or(t.len());
            let (num, unit) = t.split_at(idx);
            let n: f64 = num.parse().map_err(|_| format!("unable to parse radius [{s}]"))?;
            if !geo {
                if unit.is_empty() {
                    return Ok(n);
                }
                return Err(format!("radius [{s}] must not have units for shape type"));
            }
            Ok(n * unit_meters(unit.trim()).ok_or_else(|| format!("unknown distance unit [{}]", unit.trim()))?)
        }
        _ => Err("radius must be a number or string".into()),
    }
}

fn parse_circle(v: &Value, geo: bool) -> std::result::Result<(Circle, bool), String> {
    match v {
        Value::String(s) => {
            let t = s.trim();
            let upper = t.to_uppercase();
            if !upper.starts_with("CIRCLE") {
                let kind = upper.split(|c: char| c == '(' || c.is_whitespace()).next().unwrap_or("").to_string();
                return Err(format!("found [{kind}] instead of circle"));
            }
            let inner = t[6..].trim().trim_start_matches('(').trim_end_matches(')');
            let p: Vec<&str> = inner.split_whitespace().collect();
            if p.len() != 3 {
                return Err("invalid circle definition".into());
            }
            let x: f64 = p[0].parse().map_err(|_| "invalid circle definition".to_string())?;
            let y: f64 = p[1].parse().map_err(|_| "invalid circle definition".to_string())?;
            Ok((Circle { x, y, radius: parse_radius(&Value::String(p[2].to_string()), geo)? }, true))
        }
        Value::Object(m) => {
            let ty = m.get("type").and_then(Value::as_str).unwrap_or("");
            if !ty.eq_ignore_ascii_case("circle") {
                return Err(format!("found [{}] instead of circle", ty.to_uppercase()));
            }
            let c = m.get("coordinates").and_then(Value::as_array).ok_or("invalid circle definition")?;
            let x = c.first().and_then(Value::as_f64).ok_or("invalid circle definition")?;
            let y = c.get(1).and_then(Value::as_f64).ok_or("invalid circle definition")?;
            let r = m.get("radius").ok_or("invalid circle definition")?;
            Ok((Circle { x, y, radius: parse_radius(r, geo)? }, false))
        }
        _ => Err("invalid circle definition".into()),
    }
}

/// ES `CircleUtils.slowHaversin` (meters).
fn slow_haversin(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let h1 = (1.0 - (lat2.to_radians() - lat1.to_radians()).cos()) / 2.0;
    let h2 = (1.0 - (lon2.to_radians() - lon1.to_radians()).cos()) / 2.0;
    let h = h1 + lat1.to_radians().cos() * lat2.to_radians().cos() * h2;
    12742017.5428 * 1.0f64.min(h.sqrt()).asin()
}

fn polygon(c: &Circle, err: f64, geo: bool) -> std::result::Result<Vec<(f64, f64)>, String> {
    let sides = ((2.0 * std::f64::consts::PI / (1.0 - err / c.radius).acos()).ceil() as i64).clamp(4, 1000) as usize;
    let mut pts = Vec::with_capacity(sides + 1);
    if geo && (slow_haversin(c.y, c.x, 90.0, 0.0) < c.radius || slow_haversin(c.y, c.x, -90.0, 0.0) < c.radius) {
        return Err(format!("Circle ({} {} {}) is too large: it contains a pole", java_double(c.x), java_double(c.y), java_double(c.radius)));
    }
    for i in 0..sides {
        let angle = i as f64 * (360.0 / sides as f64);
        if geo {
            let (x, y) = (angle.to_radians().cos(), angle.to_radians().sin());
            let (mut factor, mut step, mut last) = (2.0f64, 1.0f64, 0i32);
            let mut found = None;
            for _ in 0..10_000 {
                let lat = c.y + y * factor;
                let lon = c.x + x * factor;
                let dist = slow_haversin(c.y, c.x, lat, lon);
                if (dist - c.radius).abs() < 0.1 {
                    found = Some((lon, lat));
                    break;
                } else if dist > c.radius {
                    factor -= step;
                    if last == 1 {
                        step /= 2.0;
                    }
                    last = -1;
                } else if dist < c.radius {
                    factor += step;
                    if last == -1 {
                        step /= 2.0;
                    }
                    last = 1;
                } else {
                    break;
                }
            }
            pts.push(found.ok_or_else(|| "unable to approximate circle".to_string())?);
        } else {
            pts.push((c.x + c.radius * angle.to_radians().cos(), c.y + c.radius * angle.to_radians().sin()));
        }
    }
    pts.push(pts[0]);
    Ok(pts)
}

impl Processor for CircleP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let Some(v) = input(doc, &self.field, self.ignore_missing, "cannot process it.")? else { return Ok(()) };
        if !matches!(v, Value::String(_) | Value::Object(_)) {
            return Err(ia(format!("field [{}] of type [{}] cannot be cast to [java.lang.Object]", self.field, type_name(&v))).into());
        }
        let (c, wkt) = parse_circle(&v, self.geo).map_err(|m| ia("invalid circle definition").caused_by(ia(m)))?;
        let pts = polygon(&c, self.error_distance, self.geo).map_err(|m| ia("invalid circle definition").caused_by(ia(m)))?;
        let out = if wkt {
            Value::String(format!("POLYGON (({}))", pts.iter().map(|(x, y)| format!("{} {}", java_double(*x), java_double(*y))).collect::<Vec<_>>().join(", ")))
        } else {
            json!({"type": "Polygon", "coordinates": [pts.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>()]})
        };
        doc.set(&self.target, out)?;
        Ok(())
    }
}

pub fn build_circle(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let target = cfg.opt_str("target_field")?.unwrap_or_else(|| field.clone());
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    let ed = cfg.req_value("error_distance")?;
    let error_distance = match &ed {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse().map_err(|_| cfg.err("error_distance", format!("property cannot be converted to a double [{s}]")))?,
        _ => return Err(cfg.err("error_distance", "property isn't a double")),
    };
    let st = cfg.req_str("shape_type")?;
    let geo = match st.to_uppercase().as_str() {
        "GEO_SHAPE" => true,
        "SHAPE" => false,
        _ => {
            let m = format!("illegal [shape_type] value [{st}]. valid values are [SHAPE, GEO_SHAPE]");
            return Err(EsError::new(500, "exception", format!("java.lang.IllegalArgumentException: {m}")).with("processor_type", "circle").caused_by(ia(m)));
        }
    };
    Ok(Box::new(CircleP { field, target, error_distance, geo, ignore_missing }))
}
