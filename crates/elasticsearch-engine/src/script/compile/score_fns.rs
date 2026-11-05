//! Score-context functions (ES `ScoreScriptUtils`, whitelisted only for
//! `script_score`): `saturation`, `sigmoid`, `randomScore` and the
//! `decay{Numeric,Geo,Date}{Linear,Exp,Gauss}` class bindings.
//!
//! A class binding is instantiated on its call site's first invocation
//! from the leading (constant) arguments, which later invocations do not
//! evaluate again (Painless semantics); the derived constants live in
//! [`Rt::bindings`]. Decays over `doc['f'].value` (and numeric
//! expressions) also get an unboxed fast path that reads the doc value
//! directly once the binding exists.

use super::num::NumFn;
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    Linear,
    Exp,
    Gauss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScoreFn {
    Saturation,
    Sigmoid,
    /// `randomScore(seed)` / `randomScore(seed, field)`.
    Random(bool),
    Numeric(Shape),
    Geo(Shape),
    Date(Shape),
}

impl ScoreFn {
    pub(crate) fn of(name: &str, n: usize) -> Option<ScoreFn> {
        let shape = |s: &str| match s {
            "Linear" => Some(Shape::Linear),
            "Exp" => Some(Shape::Exp),
            "Gauss" => Some(Shape::Gauss),
            _ => None,
        };
        Some(match (name, n) {
            ("saturation", 2) => ScoreFn::Saturation,
            ("sigmoid", 3) => ScoreFn::Sigmoid,
            ("randomScore", 1) => ScoreFn::Random(false),
            ("randomScore", 2) => ScoreFn::Random(true),
            (_, 5) => {
                if let Some(s) = name.strip_prefix("decayNumeric") {
                    ScoreFn::Numeric(shape(s)?)
                } else if let Some(s) = name.strip_prefix("decayGeo") {
                    ScoreFn::Geo(shape(s)?)
                } else if let Some(s) = name.strip_prefix("decayDate") {
                    ScoreFn::Date(shape(s)?)
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    }

    fn class(&self) -> &'static str {
        match self {
            ScoreFn::Numeric(Shape::Linear) => "DecayNumericLinear",
            ScoreFn::Numeric(Shape::Exp) => "DecayNumericExp",
            ScoreFn::Numeric(Shape::Gauss) => "DecayNumericGauss",
            ScoreFn::Geo(Shape::Linear) => "DecayGeoLinear",
            ScoreFn::Geo(Shape::Exp) => "DecayGeoExp",
            ScoreFn::Geo(Shape::Gauss) => "DecayGeoGauss",
            ScoreFn::Date(Shape::Linear) => "DecayDateLinear",
            ScoreFn::Date(Shape::Exp) => "DecayDateExp",
            ScoreFn::Date(Shape::Gauss) => "DecayDateGauss",
            _ => "ScoreScriptUtils",
        }
    }
}

/// `scaling` of a decay binding (ES computes it once per binding).
#[inline]
pub(crate) fn scaling(shape: Shape, scale: f64, decay: f64) -> f64 {
    match shape {
        Shape::Linear => scale / (1.0 - decay),
        Shape::Exp => decay.ln() / scale,
        Shape::Gauss => 0.5 * scale * scale / decay.ln(),
    }
}

/// Decay value at `distance` (already reduced by the offset).
#[inline(always)]
pub(crate) fn apply(shape: Shape, scaling: f64, distance: f64) -> f64 {
    match shape {
        Shape::Linear => ((scaling - distance) / scaling).max(0.0),
        Shape::Exp => (scaling * distance).exp(),
        Shape::Gauss => (0.5 * (distance * distance) / scaling).exp(),
    }
}

/// `GeoDistance.ARC` (`GeoUtils.arcDistance` = Lucene
/// `SloppyMath.haversinMeters`) decay of a point.
#[inline(always)]
fn geo_decay(shape: Shape, b: &[f64; 4], lat: f64, lon: f64) -> f64 {
    let d = crate::sloppy_math::haversin_meters(b[0], b[1], lat, lon);
    apply(shape, b[3], (d - b[2]).max(0.0))
}

#[inline(always)]
fn numeric_decay(shape: Shape, b: &[f64; 4], v: f64) -> f64 {
    apply(shape, b[2], ((v - b[0]).abs() - b[1]).max(0.0))
}

#[inline(always)]
fn date_decay(shape: Shape, b: &[f64; 4], millis: i64) -> f64 {
    let (origin, offset) = (b[0] as i64, b[1] as i64);
    let diff = if millis >= origin { millis.wrapping_sub(origin) } else { origin.wrapping_sub(millis) };
    apply(shape, b[2], (diff.wrapping_sub(offset)).max(0) as f64)
}

fn arg_str(v: &Val) -> SResult<String> {
    match v {
        Val::Str(s) => Ok(s.to_string()),
        Val::Null => Err(npe("Cannot invoke \"String.length()\" because \"value\" is null")),
        other => Err(super::super::error::class_cast(format!("Cannot cast {} to java.lang.String", other.type_name()))),
    }
}

fn arg_f64(v: &Val) -> SResult<f64> {
    v.to_f64()
}

/// `(lat, lon)` of a `GeoPoint` value.
fn geo_point_of(v: &Val) -> SResult<(f64, f64)> {
    if let Val::Map(m) = v {
        let m = m.borrow();
        if let (Some(a), Some(b)) = (m.get(&Key::str("lat")).and_then(Val::as_f64), m.get(&Key::str("lon")).and_then(Val::as_f64)) {
            return Ok((a, b));
        }
    }
    if v.is_null() {
        return Err(npe("Cannot invoke \"org.elasticsearch.common.geo.GeoPoint.lat()\" because \"docValue\" is null"));
    }
    Err(super::super::error::class_cast(format!("Cannot cast {} to org.elasticsearch.common.geo.GeoPoint", v.type_name())))
}

fn date_of(v: &Val) -> SResult<i64> {
    match v {
        Val::Date(ms) | Val::Instant(ms) => Ok(*ms),
        Val::Null => Err(npe("Cannot invoke \"java.time.ZonedDateTime.toInstant()\" because \"docValueDate\" is null")),
        other => Err(super::super::error::class_cast(format!("Cannot cast {} to java.time.ZonedDateTime", other.type_name()))),
    }
}

fn time_value(s: &str, class: &str, what: &str) -> SResult<f64> {
    crate::date::parse_time_value(s)
        .map(|v| v as f64)
        .ok_or_else(|| illegal_arg(format!("failed to parse setting [{class}.{what}] with value [{s}] as a time value: unit is missing or unrecognized")))
}

fn distance(s: &str) -> SResult<f64> {
    crate::values::parse_distance(&serde_json::Value::String(s.to_string())).ok_or_else(|| illegal_arg(format!("failed to parse [{s}]")))
}

/// The binding state of a decay from its four constructor arguments.
fn instantiate(kind: ScoreFn, a: &[Val]) -> SResult<[f64; 4]> {
    Ok(match kind {
        ScoreFn::Numeric(shape) => {
            let (origin, scale, offset, decay) = (arg_f64(&a[0])?, arg_f64(&a[1])?, arg_f64(&a[2])?, arg_f64(&a[3])?);
            [origin, offset, scaling(shape, scale, decay), 0.0]
        }
        ScoreFn::Geo(shape) => {
            let (o, sc, off) = (arg_str(&a[0])?, arg_str(&a[1])?, arg_str(&a[2])?);
            let decay = arg_f64(&a[3])?;
            let (lat, lon) = crate::values::parse_geo_point(&serde_json::Value::String(o.clone())).ok_or_else(|| illegal_arg(format!("failed to parse [{o}]")))?;
            [lat, lon, distance(&off)?, scaling(shape, distance(&sc)?, decay)]
        }
        ScoreFn::Date(shape) => {
            let (o, sc, off) = (arg_str(&a[0])?, arg_str(&a[1])?, arg_str(&a[2])?);
            let decay = arg_f64(&a[3])?;
            let fmt = crate::date::DateFormatter::new("strict_date_optional_time||epoch_millis").map_err(|e| illegal_arg(e.reason))?;
            let origin = crate::date::parse_date_math(&o, &fmt, crate::date::Tz::UTC, crate::date::now_millis(), false).map_err(|e| illegal_arg(e.reason))?;
            let class = kind.class();
            [origin as f64, time_value(&off, class, "offset")?, scaling(shape, time_value(&sc, class, "scale")?, decay), 0.0]
        }
        _ => [0.0; 4],
    })
}

/// ES `randomScore`: murmur3 of the doc's seed string, salted with the
/// index and seed; the lower 24 bits as a float in `[0, 1)`.
fn random_score(rt: &mut Rt, seed: i32, field: Option<&Val>) -> SResult<f64> {
    let index_hash = super::super::env().and_then(|e| e.current.borrow().clone()).map(|n| java_hash(&n)).unwrap_or(0);
    let salt = index_hash.wrapping_shl(10);
    let salted = mix32(salt ^ seed);
    let s = match field {
        Some(f) => f.to_java_string(),
        None => rt.doc.doc.to_string(),
    };
    let h = crate::tsdb::murmur3_32(s.as_bytes(), salted as u32) as i32;
    Ok(((h & 0x00FF_FFFF) as f32 / (1 << 24) as f32) as f64)
}

fn java_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// Lucene `BitMixer.mix32`.
fn mix32(k: i32) -> i32 {
    let mut k = k as u32;
    k = (k ^ (k >> 16)).wrapping_mul(0x85eb_ca6b);
    k = (k ^ (k >> 13)).wrapping_mul(0xc2b2_ae35);
    (k ^ (k >> 16)) as i32
}

impl Compiler {
    /// Binding slot of the decay call whose arguments start at `args`
    /// (shared by the general node and its numeric fast path).
    fn binding_slot(&mut self, args: &[Expr]) -> usize {
        let key = args.as_ptr() as usize;
        if let Some(s) = self.binding_slots.get(&key) {
            return *s;
        }
        let s = self.nbindings;
        self.nbindings += 1;
        self.binding_slots.insert(key, s);
        s
    }

    /// General node of a score function call.
    pub(super) fn score_fn(&mut self, kind: ScoreFn, args: &[Expr], fs: Vec<ExprFn>, pos: usize) -> (ExprFn, St) {
        let f: ExprFn = match kind {
            ScoreFn::Saturation => Box::new(move |rt| {
                let (v, k) = (arg_f64(&fs[0](rt)?).map_err(at(pos))?, arg_f64(&fs[1](rt)?).map_err(at(pos))?);
                Ok(Val::Double(v / (k + v)))
            }),
            ScoreFn::Sigmoid => Box::new(move |rt| {
                let (v, k, a) = (arg_f64(&fs[0](rt)?).map_err(at(pos))?, arg_f64(&fs[1](rt)?).map_err(at(pos))?, arg_f64(&fs[2](rt)?).map_err(at(pos))?);
                Ok(Val::Double(v.powf(a) / (k.powf(a) + v.powf(a))))
            }),
            ScoreFn::Random(with_field) => Box::new(move |rt| {
                let seed = fs[0](rt)?.to_i64().map_err(at(pos))? as i32;
                let fv = if with_field {
                    let name = arg_str(&fs[1](rt)?).map_err(at(pos))?;
                    let slot = rt.doc.slot_of(&name);
                    Some(rt.doc.value(slot).map_err(at(pos))?)
                } else {
                    None
                };
                Ok(Val::Double(random_score(rt, seed, fv.as_ref()).map_err(at(pos))?))
            }),
            ScoreFn::Numeric(shape) | ScoreFn::Geo(shape) | ScoreFn::Date(shape) => {
                let slot = self.binding_slot(args);
                Box::new(move |rt| {
                    let b = match rt.bindings[slot] {
                        Some(b) => b,
                        None => {
                            let mut a = Vec::with_capacity(4);
                            for f in &fs[..4] {
                                a.push(f(rt)?);
                            }
                            let b = instantiate(kind, &a).map_err(at(pos))?;
                            rt.bindings[slot] = Some(b);
                            b
                        }
                    };
                    let v = fs[4](rt)?;
                    Ok(Val::Double(match kind {
                        ScoreFn::Numeric(_) => numeric_decay(shape, &b, arg_f64(&v).map_err(at(pos))?),
                        ScoreFn::Geo(_) => {
                            let (lat, lon) = geo_point_of(&v).map_err(at(pos))?;
                            geo_decay(shape, &b, lat, lon)
                        }
                        _ => date_decay(shape, &b, date_of(&v).map_err(at(pos))?),
                    }))
                })
            }
        };
        (f, St::Double)
    }

    /// Unboxed fast path of a score function call (see [`super::num`]):
    /// `None` at run time until the binding exists or when the operand
    /// is not a plain single doc value.
    pub(super) fn score_fn_num(&mut self, kind: ScoreFn, args: &[Expr]) -> Option<NumFn> {
        match kind {
            ScoreFn::Saturation => {
                let (v, k) = (self.num(&args[0])?.into_d(), self.num(&args[1])?.into_d());
                Some(NumFn::D(Box::new(move |rt| {
                    let v = v(rt)?;
                    Some(v / (k(rt)? + v))
                })))
            }
            ScoreFn::Sigmoid => {
                let (v, k, a) = (self.num(&args[0])?.into_d(), self.num(&args[1])?.into_d(), self.num(&args[2])?.into_d());
                Some(NumFn::D(Box::new(move |rt| {
                    let (v, k, a) = (v(rt)?, k(rt)?, a(rt)?);
                    Some(v.powf(a) / (k.powf(a) + v.powf(a)))
                })))
            }
            ScoreFn::Random(_) => None,
            ScoreFn::Numeric(shape) => {
                let slot = self.binding_slot(args);
                let v = self.num(&args[4])?.into_d();
                Some(NumFn::D(Box::new(move |rt| {
                    let b = rt.bindings[slot]?;
                    Some(numeric_decay(shape, &b, v(rt)?))
                })))
            }
            ScoreFn::Geo(shape) => {
                let slot = self.binding_slot(args);
                let doc = self.doc_value_slot(&args[4])?;
                Some(NumFn::D(Box::new(move |rt| {
                    let b = rt.bindings[slot]?;
                    let (lat, lon) = rt.doc.fields[doc].geo_point(rt.doc.doc)?;
                    Some(geo_decay(shape, &b, lat, lon))
                })))
            }
            ScoreFn::Date(shape) => {
                let slot = self.binding_slot(args);
                let doc = self.doc_value_slot(&args[4])?;
                Some(NumFn::D(Box::new(move |rt| {
                    let b = rt.bindings[slot]?;
                    Some(date_decay(shape, &b, rt.doc.fast_date(doc)?))
                })))
            }
        }
    }

    /// `doc['x'].value` / `doc['x'].getValue()` / `doc.x.value`.
    fn doc_value_slot(&mut self, e: &Expr) -> Option<usize> {
        match &e.kind {
            ExprKind::Field { obj, name, null_safe: false } if name == "value" => self.doc_slot(obj),
            ExprKind::Call { obj, name, args, null_safe: false } if name == "getValue" && args.is_empty() => self.doc_slot(obj),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_at_scale_is_decay() {
        for shape in [Shape::Linear, Shape::Exp, Shape::Gauss] {
            let s = scaling(shape, 10.0, 0.5);
            assert!((apply(shape, s, 10.0) - 0.5).abs() < 1e-12, "{shape:?}");
            assert!((apply(shape, s, 0.0) - 1.0).abs() < 1e-12);
        }
        let b = [100.0, 5.0, scaling(Shape::Linear, 10.0, 0.5), 0.0];
        assert_eq!(numeric_decay(Shape::Linear, &b, 103.0), 1.0);
        assert!((numeric_decay(Shape::Linear, &b, 115.0) - 0.5).abs() < 1e-12);
        let b = [1000.0, 0.0, scaling(Shape::Exp, 86_400_000.0, 0.5), 0.0];
        assert!((date_decay(Shape::Exp, &b, 1000 + 86_400_000) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn score_fn_names() {
        assert_eq!(ScoreFn::of("decayGeoGauss", 5), Some(ScoreFn::Geo(Shape::Gauss)));
        assert_eq!(ScoreFn::of("decayDateLinear", 5), Some(ScoreFn::Date(Shape::Linear)));
        assert_eq!(ScoreFn::of("decayNumericExp", 5), Some(ScoreFn::Numeric(Shape::Exp)));
        assert_eq!(ScoreFn::of("decayNumericExp", 4), None);
        assert_eq!(ScoreFn::of("randomScore", 2), Some(ScoreFn::Random(true)));
    }
}
