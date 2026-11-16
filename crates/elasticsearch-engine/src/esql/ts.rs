//! Time-series aggregation (`TS ... | STATS`), after ES's
//! `TranslateTimeSeriesAggregate`: time-series functions (`RATE`,
//! `AVG_OVER_TIME`, ...) run per time series (`_tsid`) and time bucket,
//! then the outer aggregations combine the series by the user's groups.
//! Aggregations over plain fields read each series' last value
//! (`LAST_OVER_TIME`).

use super::exec::{bind, eval, type_of, Column, Ctx, Table};
use super::parse::{Expr, Field};
use super::{cmp_vals, verification_err, Val};
use crate::error::Result;
use std::collections::{BTreeMap, HashMap};

/// Hidden column holding each row's time series (its dimensions as JSON).
pub const TSID: &str = "\u{0}tsid";

/// Time-series aggregation functions.
pub const TS_FUNCS: &[&str] = &[
    "RATE",
    "IRATE",
    "INCREASE",
    "DELTA",
    "IDELTA",
    "DERIV",
    "AVG_OVER_TIME",
    "SUM_OVER_TIME",
    "MIN_OVER_TIME",
    "MAX_OVER_TIME",
    "COUNT_OVER_TIME",
    "COUNT_DISTINCT_OVER_TIME",
    "FIRST_OVER_TIME",
    "LAST_OVER_TIME",
    "PRESENT_OVER_TIME",
    "ABSENT_OVER_TIME",
    "PERCENTILE_OVER_TIME",
    "STDDEV_OVER_TIME",
    "VARIANCE_OVER_TIME",
    "STDVAR_OVER_TIME",
];

pub fn is_ts_fn(name: &str) -> bool {
    TS_FUNCS.contains(&name)
}

/// Whether an expression calls a time-series function.
pub fn has_ts_fn(e: &Expr) -> bool {
    match e {
        Expr::Call(n, args, _) => is_ts_fn(n) || args.iter().any(has_ts_fn),
        Expr::Bin(_, a, b) => has_ts_fn(a) || has_ts_fn(b),
        Expr::Neg(a) | Expr::Not(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) => has_ts_fn(a),
        _ => false,
    }
}

/// An inner (per series and bucket) aggregation.
#[derive(Clone, Debug, PartialEq)]
struct Inner {
    name: String,
    args: Vec<Expr>,
    pos: usize,
}

/// Replace time-series calls by references to their first-pass columns
/// (registering them); a regular aggregation over a plain expression reads
/// the series' last value.
fn reg(inners: &mut Vec<Inner>, inner: Inner) -> Expr {
    let k = match inners.iter().position(|x| x.name == inner.name && x.args == inner.args) {
        Some(k) => k,
        None => {
            inners.push(inner);
            inners.len() - 1
        }
    };
    Expr::Col(format!("\u{0}ts{k}"), 0)
}

fn split(e: &Expr, inners: &mut Vec<Inner>, top_agg: bool) -> Expr {
    match e {
        Expr::Call(n, args, p) if is_ts_fn(n) => reg(inners, Inner { name: n.clone(), args: args.clone(), pos: *p }),
        Expr::Call(n, args, p) if super::func::AGGS.contains(&n.as_str()) && top_agg => {
            let mut new_args = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                if i == 0 && !matches!(a, Expr::Star) && !has_ts_fn(a) {
                    // Implicit `LAST_OVER_TIME`.
                    new_args.push(reg(inners, Inner { name: "LAST_OVER_TIME".into(), args: vec![a.clone()], pos: *p }));
                } else {
                    new_args.push(split(a, inners, false));
                }
            }
            Expr::Call(n.clone(), new_args, *p)
        }
        Expr::Call(n, args, p) => Expr::Call(n.clone(), args.iter().map(|a| split(a, inners, top_agg)).collect(), *p),
        Expr::Bin(op, a, b) => Expr::Bin(*op, Box::new(split(a, inners, top_agg)), Box::new(split(b, inners, top_agg))),
        Expr::Neg(a) => Expr::Neg(Box::new(split(a, inners, top_agg))),
        Expr::Not(a) => Expr::Not(Box::new(split(a, inners, top_agg))),
        Expr::Cast(a, t) => Expr::Cast(Box::new(split(a, inners, top_agg)), t.clone()),
        o => o.clone(),
    }
}

/// Whether an aggregate expression has an outer (regular) aggregation.
fn has_outer_agg(e: &Expr) -> bool {
    match e {
        Expr::Call(n, args, _) => super::func::AGGS.contains(&n.as_str()) || args.iter().any(has_outer_agg),
        Expr::Bin(_, a, b) => has_outer_agg(a) || has_outer_agg(b),
        Expr::Neg(a) | Expr::Not(a) | Expr::Cast(a, _) => has_outer_agg(a),
        _ => false,
    }
}

/// The time bucket of a grouping: `BUCKET(@timestamp, span)` / `TBUCKET(span)`.
fn bucket_span(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Call(n, args, _) if n == "TBUCKET" && args.len() == 1 => Some(args[0].clone()),
        Expr::Call(n, args, _) if (n == "BUCKET" || n == "TBUCKET") && args.len() == 2 && matches!(&args[0], Expr::Col(c, _) if c == "@timestamp") => Some(args[1].clone()),
        Expr::Call(n, args, _) if n == "DATE_TRUNC" && args.len() == 2 && matches!(&args[1], Expr::Col(c, _) if c == "@timestamp") => Some(args[0].clone()),
        _ => None,
    }
}

/// `requires the [@timestamp] field` error.
fn needs_timestamp(ctx: &Ctx, src_pos: usize) -> crate::error::EsError {
    let src = super::exec::call_src(ctx.full, src_pos);
    verification_err(ctx.full, src_pos, format!("[{src}] requires the [@timestamp] field, which was either not present in the source index, or has been dropped or renamed"))
}

/// One series' samples in one bucket, sorted by time.
struct Group {
    tsid: String,
    bucket: Option<i64>,
    rows: Vec<usize>,
}

/// Counter state of a group (`RateDoubleGroupingAggregatorFunction.ReducedState`).
#[derive(Clone, Copy, Debug)]
struct RateState {
    samples: usize,
    first_ts: f64,
    first_val: f64,
    last_ts: f64,
    last_val: f64,
    resets: f64,
}

fn rate_state(samples: &[(i64, f64)], counter: bool) -> Option<RateState> {
    let (f, l) = (samples.first()?, samples.last()?);
    let mut resets = 0.0;
    if counter {
        for w in samples.windows(2) {
            if w[1].1 < w[0].1 {
                resets += w[0].1;
            }
        }
    }
    Some(RateState { samples: samples.len(), first_ts: f.0 as f64 / 1000.0, first_val: f.1, last_ts: l.0 as f64 / 1000.0, last_val: l.1, resets })
}

fn delta_between(lo: &RateState, hi: &RateState, counter: bool) -> f64 {
    if !counter || hi.first_val >= lo.last_val {
        hi.first_val - lo.last_val
    } else {
        hi.first_val
    }
}

fn interpolate(lo: &RateState, hi: &RateState, start: f64, end: f64, lower: bool, counter: bool) -> f64 {
    let slope = delta_between(lo, hi, counter) / (hi.first_ts - lo.last_ts);
    if lower {
        let base = if !counter || hi.first_val >= lo.last_val { lo.last_val } else { 0.0 };
        base + slope * (start - lo.last_ts)
    } else {
        lo.last_val + slope * (end - lo.last_ts)
    }
}

fn extrapolate(s: &RateState, start: f64, end: f64, lower: bool, counter: bool) -> f64 {
    let span = s.last_ts - s.first_ts;
    let avg = span / s.samples as f64;
    let end_val = s.last_val + s.resets;
    let slope = (end_val - s.first_val) / span;
    if lower {
        let mut gap = s.first_ts - start;
        if gap > 0.0 {
            if gap > avg * 1.1 {
                gap = avg / 2.0;
            }
            let v = s.first_val - gap * slope;
            return if counter { v.max(0.0) } else { v };
        }
        s.first_val
    } else {
        let mut gap = end - s.last_ts;
        if gap > 0.0 {
            if gap > avg * 1.1 {
                gap = avg / 2.0;
            }
            return end_val + gap * slope;
        }
        end_val
    }
}

/// `computeRate`: a bucket's increase (or per-second rate), interpolating
/// its boundary values with the adjacent buckets of the series or
/// extrapolating them.
fn bucket_rate(s: &RateState, prev: Option<&RateState>, next: Option<&RateState>, start: f64, end: f64, per_second: bool, counter: bool) -> Option<f64> {
    let (mut first_ts, mut last_ts) = (start, end);
    let first_val = match prev {
        None => {
            if s.samples == 1 {
                first_ts = s.first_ts;
                s.first_val
            } else {
                extrapolate(s, start, end, true, counter)
            }
        }
        Some(p) => interpolate(p, s, start, end, true, counter),
    };
    let last_val = match next {
        None => {
            if s.samples == 1 {
                last_ts = s.last_ts;
                s.last_val + s.resets
            } else {
                extrapolate(s, start, end, false, counter)
            }
        }
        Some(n) => interpolate(s, n, start, end, false, counter) + s.resets,
    };
    if last_ts == first_ts {
        if s.samples == 1 {
            if let Some(p) = prev {
                return Some(if per_second { delta_between(p, s, counter) / (first_ts - p.last_ts) } else { 0.0 });
            }
        }
        return None;
    }
    let inc = last_val - first_val;
    Some(if per_second { inc / (last_ts - first_ts) } else { inc })
}

/// Rate without buckets (`computeRateWithoutExtrapolate`).
fn plain_rate(s: &RateState, per_second: bool) -> Option<f64> {
    if s.samples < 2 {
        return None;
    }
    let inc = s.last_val + s.resets - s.first_val;
    Some(if per_second { inc / (s.last_ts - s.first_ts) } else { inc })
}

fn num(v: &Val) -> Option<f64> {
    match v {
        Val::Str(s) => super::types::Amd::parse(s).and_then(|a| a.max),
        o => o.as_f64(),
    }
}

/// Least-squares slope (per second) of the samples.
fn deriv(samples: &[(i64, f64)]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len() as f64;
    let t0 = samples[0].0 as f64 / 1000.0;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (t, v) in samples {
        let x = *t as f64 / 1000.0 - t0;
        sx += x;
        sy += v;
        sxx += x * x;
        sxy += x * v;
    }
    let den = n * sxx - sx * sx;
    if den == 0.0 {
        return None;
    }
    Some((n * sxy - sx * sy) / den)
}

/// Whether the series of `rows` reports delta temporality (its
/// temporality dimension, from the series key, is `delta`).
fn series_is_delta(ctx: &Ctx, t: &Table, rows: &[usize]) -> bool {
    let Some(field) = &ctx.ts_temporality else { return false };
    let Some(i) = t.cols.iter().position(|c| c.name == TSID) else { return false };
    let Some(&r) = rows.first() else { return false };
    serde_json::from_str::<serde_json::Value>(&t.rows[r][i].to_text()).ok()
        .and_then(|v| v.get(field.as_str()).and_then(|x| x.as_str().map(str::to_string)))
        .is_some_and(|x| x.eq_ignore_ascii_case("delta"))
}

/// The per-series result of an inner function over one group.
#[allow(clippy::too_many_arguments)]
fn inner_value(ctx: &Ctx, inner: &Inner, arg: &Expr, extra: Option<Val>, t: &Table, ts_i: usize, rows: &[usize], neighbors: (Option<&[usize]>, Option<&[usize]>), range: Option<(i64, i64)>, arg_ty: &str) -> Result<Val> {
    let f = &ctx.fctx;
    let mut samples: Vec<(i64, Val)> = Vec::with_capacity(rows.len());
    for &r in rows {
        let ts = t.rows[r][ts_i].as_i64().unwrap_or(0);
        match eval(arg, &t.rows[r], f)? {
            Val::Null => {}
            Val::Multi(m) => samples.extend(m.into_iter().filter(|x| !x.is_null()).map(|x| (ts, x))),
            v => samples.push((ts, v)),
        }
    }
    samples.sort_by_key(|s| s.0);
    let nums: Vec<(i64, f64)> = samples.iter().filter_map(|(t, v)| num(v).map(|x| (*t, x))).collect();
    let neighbor_state = |rs: Option<&[usize]>, counter: bool| -> Result<Option<RateState>> {
        let Some(rs) = rs else { return Ok(None) };
        let mut ns: Vec<(i64, f64)> = Vec::new();
        for &r in rs {
            let ts = t.rows[r][ts_i].as_i64().unwrap_or(0);
            if let Some(x) = eval(arg, &t.rows[r], f)?.as_f64() {
                ns.push((ts, x));
            }
        }
        ns.sort_by_key(|s| s.0);
        Ok(rate_state(&ns, counter))
    };
    let amds: Vec<super::types::Amd> = if arg_ty == "aggregate_metric_double" { samples.iter().filter_map(|(_, v)| super::types::amd_of(v)).collect() } else { vec![] };
    let amd = !amds.is_empty() || arg_ty == "aggregate_metric_double";
    let keep_int = |d: f64| -> Val {
        match arg_ty {
            "integer" | "counter_integer" => Val::Int(d as i64),
            "long" | "counter_long" => Val::Long(d as i64),
            _ => Val::Double(d),
        }
    };
    Ok(match inner.name.as_str() {
        "LAST_OVER_TIME" => samples.last().map(|s| s.1.clone()).unwrap_or(Val::Null),
        "FIRST_OVER_TIME" => samples.first().map(|s| s.1.clone()).unwrap_or(Val::Null),
        "COUNT_OVER_TIME" => {
            if amd {
                Val::Long(amds.iter().filter_map(|a| a.count).sum())
            } else {
                Val::Long(samples.len() as i64)
            }
        }
        "PRESENT_OVER_TIME" => Val::Bool(!samples.is_empty()),
        "ABSENT_OVER_TIME" => Val::Bool(samples.is_empty()),
        "COUNT_DISTINCT_OVER_TIME" => {
            let mut keys: Vec<String> = samples.iter().map(|s| s.1.key()).collect();
            keys.sort();
            keys.dedup();
            Val::Long(keys.len() as i64)
        }
        "MAX_OVER_TIME" | "MIN_OVER_TIME" => {
            let max = inner.name == "MAX_OVER_TIME";
            if amd {
                let v = amds.iter().filter_map(|a| if max { a.max } else { a.min }).reduce(|x, y| if max { x.max(y) } else { x.min(y) });
                v.map(Val::Double).unwrap_or(Val::Null)
            } else {
                let it = samples.into_iter().map(|s| s.1);
                (if max { it.max_by(cmp_vals) } else { it.min_by(cmp_vals) }).unwrap_or(Val::Null)
            }
        }
        "SUM_OVER_TIME" => {
            if amd {
                amds.iter().filter_map(|a| a.sum).reduce(|x, y| x + y).map(Val::Double).unwrap_or(Val::Null)
            } else if nums.is_empty() {
                Val::Null
            } else if matches!(arg_ty, "integer" | "long") {
                // ES `SumOverTime` is `Sum`: exact longs, `addExact`; an
                // overflow is a warning and a null series value.
                let mut sum: i64 = 0;
                let mut overflow = false;
                for v in samples.iter().filter_map(|(_, v)| v.as_i64()) {
                    match sum.checked_add(v) {
                        Some(x) => sum = x,
                        None => {
                            overflow = true;
                            break;
                        }
                    }
                }
                if overflow {
                    super::exec::sum_overflow_warning(f, inner.pos);
                    Val::Null
                } else {
                    Val::Long(sum)
                }
            } else {
                Val::Double(nums.iter().map(|x| x.1).sum())
            }
        }
        "AVG_OVER_TIME" => {
            if amd {
                let s: Option<f64> = amds.iter().filter_map(|a| a.sum).reduce(|x, y| x + y);
                let c: Option<i64> = amds.iter().filter_map(|a| a.count).reduce(|x, y| x + y);
                match (s, c) {
                    (Some(s), Some(c)) if c > 0 => Val::Double(s / c as f64),
                    _ => Val::Null,
                }
            } else if nums.is_empty() {
                Val::Null
            } else {
                Val::Double(nums.iter().map(|x| x.1).sum::<f64>() / nums.len() as f64)
            }
        }
        "STDDEV_OVER_TIME" | "VARIANCE_OVER_TIME" | "STDVAR_OVER_TIME" => {
            if nums.is_empty() {
                Val::Null
            } else {
                let n = nums.len() as f64;
                let m = nums.iter().map(|x| x.1).sum::<f64>() / n;
                let var = nums.iter().map(|x| (x.1 - m).powi(2)).sum::<f64>() / n;
                Val::Double(if inner.name == "STDDEV_OVER_TIME" { var.sqrt() } else { var })
            }
        }
        "PERCENTILE_OVER_TIME" => {
            let p = extra.and_then(|v| v.as_f64()).unwrap_or(50.0);
            super::func::percentile(nums.iter().map(|x| x.1).collect(), p).map(Val::Double).unwrap_or(Val::Null)
        }
        "DERIV" => deriv(&nums).map(Val::Double).unwrap_or(Val::Null),
        "IRATE" | "IDELTA" => {
            if nums.len() < 2 {
                Val::Null
            } else {
                let (a, b) = (nums[nums.len() - 2], nums[nums.len() - 1]);
                let d = if inner.name == "IRATE" && b.1 < a.1 { b.1 } else { b.1 - a.1 };
                if inner.name == "IRATE" {
                    let dt = (b.0 - a.0) as f64 / 1000.0;
                    if dt == 0.0 { Val::Null } else { Val::Double(d / dt) }
                } else {
                    keep_int(d)
                }
            }
        }
        "RATE" | "INCREASE" | "DELTA" => {
            let counter = inner.name != "DELTA";
            let per_second = inner.name == "RATE";
            // A delta-temporality series (`index.time_series.temporality_field`
            // says `delta`): each sample is the change since the previous
            // one, so RATE / INCREASE read the running total (ES
            // `RateDoubleGroupingAggregatorFunction`'s delta state).
            if counter && series_is_delta(ctx, t, rows) {
                let mut total = 0.0;
                let cumulative: Vec<(i64, f64)> = nums.iter().map(|(ts, v)| { total += v; (*ts, total) }).collect();
                let Some(s) = rate_state(&cumulative, true) else { return Ok(Val::Null) };
                let r = match range {
                    Some((start, end)) => bucket_rate(&s, None, None, start as f64 / 1000.0, end as f64 / 1000.0, per_second, true),
                    None => plain_rate(&s, per_second),
                };
                return Ok(match r {
                    Some(v) if v.is_finite() => Val::Double(v),
                    _ => Val::Null,
                });
            }
            let Some(s) = rate_state(&nums, counter) else { return Ok(Val::Null) };
            let r = match range {
                Some((start, end)) => {
                    let prev = neighbor_state(neighbors.0, counter)?;
                    let next = neighbor_state(neighbors.1, counter)?;
                    bucket_rate(&s, prev.as_ref(), next.as_ref(), start as f64 / 1000.0, end as f64 / 1000.0, per_second, counter)
                }
                None => plain_rate(&s, per_second),
            };
            match r {
                Some(v) if v.is_finite() => Val::Double(v),
                _ => Val::Null,
            }
        }
        _ => Val::Null,
    })
}

/// The type of an inner function's result.
fn inner_type(name: &str, arg_ty: &str) -> String {
    match name {
        "RATE" | "IRATE" | "INCREASE" | "DELTA" | "DERIV" | "AVG_OVER_TIME" | "STDDEV_OVER_TIME" | "VARIANCE_OVER_TIME" | "STDVAR_OVER_TIME" | "PERCENTILE_OVER_TIME" => "double".into(),
        "COUNT_OVER_TIME" | "COUNT_DISTINCT_OVER_TIME" => "long".into(),
        "PRESENT_OVER_TIME" | "ABSENT_OVER_TIME" => "boolean".into(),
        "IDELTA" => if matches!(arg_ty, "integer" | "long" | "counter_integer" | "counter_long") { arg_ty.trim_start_matches("counter_").into() } else { "double".into() },
        "SUM_OVER_TIME" => if matches!(arg_ty, "integer" | "long") { "long".into() } else { "double".into() },
        "MAX_OVER_TIME" | "MIN_OVER_TIME" if arg_ty == "aggregate_metric_double" => "double".into(),
        _ => arg_ty.trim_start_matches("counter_").to_string(),
    }
}

/// Bucket boundaries `[start, end)` of a time bucket starting at `start`.
fn bucket_end(start: i64, span: &Val) -> i64 {
    match span {
        Val::Dur(ms) => start + ms,
        Val::Period(m, d) => super::func::add_period(start, *m, *d),
        _ => start,
    }
}

/// `TS | STATS`: the two-pass time-series aggregation.
pub fn ts_stats(ctx: &mut Ctx, t: Table, aggs: &[Field], by: &[Field], pos: usize) -> Result<Table> {
    // Split the aggregates into per-series inner functions and outer expressions.
    let mut inners: Vec<Inner> = Vec::new();
    let mut outer: Vec<Field> = Vec::with_capacity(aggs.len());
    let mut bare = false;
    for a in aggs {
        let top = has_outer_agg(&a.expr);
        if !top && has_ts_fn(&a.expr) {
            bare = true;
        }
        let filter = a.filter.as_ref().map(|f| split(f, &mut inners, false));
        outer.push(Field { name: a.name.clone(), expr: split(&a.expr, &mut inners, top), filter, pos: a.pos });
    }
    let ts_i = t.cols.iter().position(|c| c.name == "@timestamp");
    // Time-series functions need the series' timestamps.
    if ts_i.is_none() {
        if let Some(i) = inners.iter().find(|i| i.name != "LAST_OVER_TIME") {
            return Err(needs_timestamp(ctx, i.pos));
        }
        for a in aggs {
            if let Expr::Call(n, args, p) = &a.expr {
                if matches!(n.as_str(), "FIRST" | "LAST") && matches!(args.get(1), Some(Expr::Lit(Val::Null)) | None) {
                    return Err(needs_timestamp(ctx, *p));
                }
            }
        }
    }
    let Some(ts_i) = ts_i else {
        return Err(verification_err(ctx.full, pos, "Time-series aggregations require the [@timestamp] field"));
    };
    let tsid_i = t.cols.iter().position(|c| c.name == TSID);
    // Groupings: at most one time bucket; the others are read per series.
    let mut bucket: Option<(usize, Expr, Val)> = None;
    let mut others: Vec<(usize, Expr)> = Vec::new();
    for (gi, g) in by.iter().enumerate() {
        match bucket_span(&g.expr) {
            Some(span_e) if bucket.is_none() => {
                if let Some(m) = super::exec::bucket_column_meta(ctx, &g.expr, &t.cols) {
                    ctx.col_meta.borrow_mut().insert(g.name.clone(), serde_json::json!({"bucket": m}));
                }
                let span = eval(&bind(ctx, &span_e, &[])?, &[], &ctx.fctx)?;
                let span = match &span {
                    Val::Str(s) => super::func::parse_temporal(s).unwrap_or(span),
                    _ => span,
                };
                let be = Expr::Call("BUCKET".into(), vec![Expr::Col("@timestamp".into(), g.pos), Expr::Lit(span.clone())], g.pos);
                bucket = Some((gi, bind(ctx, &be, &t.cols)?, span));
            }
            _ => {
                if let Expr::Call(n, args, p) = &g.expr {
                    if n == "BUCKET" && args.first().is_some_and(|a| !matches!(a, Expr::Col(c, _) if c == "@timestamp")) && type_of(&bind(ctx, &args[0], &t.cols)?, &t.cols).is_some_and(|ty| ty.starts_with("date")) {
                        let _ = p;
                        return Err(crate::error::EsError::illegal_argument(
                            "Time-series aggregations require direct use of @timestamp which was not found. If @timestamp was renamed in EVAL, use the original @timestamp field instead.",
                        ));
                    }
                }
                if bare {
                    return Err(verification_err(ctx.full, g.pos, format!("cannot group by [{}] when using a time-series aggregation function without an outer aggregation; use TBUCKET or WITHOUT, or wrap it in an outer aggregation", g.name)));
                }
                others.push((gi, bind(ctx, &g.expr, &t.cols)?));
            }
        }
    }
    // First pass: group rows by series and bucket.
    let mut index: BTreeMap<(String, Option<i64>), usize> = BTreeMap::new();
    let mut groups: Vec<Group> = Vec::new();
    for (ri, r) in t.rows.iter().enumerate() {
        let tsid = tsid_i.map(|i| r[i].to_text()).unwrap_or_default();
        let b = match &bucket {
            Some((_, e, _)) => eval(e, r, &ctx.fctx)?.as_i64(),
            None => None,
        };
        if bucket.is_some() && b.is_none() {
            continue;
        }
        let gi = *index.entry((tsid.clone(), b)).or_insert_with(|| {
            groups.push(Group { tsid, bucket: b, rows: Vec::new() });
            groups.len() - 1
        });
        groups[gi].rows.push(ri);
    }
    let order: Vec<usize> = index.values().copied().collect();
    // Inner function columns, each with its time window (ms), if given:
    // the optional `window` argument (the third of PERCENTILE_OVER_TIME,
    // whose second is the percentile; the second of the others).
    let mut windows: Vec<Option<i64>> = Vec::new();
    for i in &inners {
        let wi = if i.name == "PERCENTILE_OVER_TIME" { 2 } else { 1 };
        let w = match i.args.get(wi) {
            None => None,
            Some(e) => {
                let v = eval(&bind(ctx, e, &t.cols)?, &[], &ctx.fctx).unwrap_or(Val::Null);
                let v = match &v {
                    Val::Str(s) => super::func::parse_temporal(s).unwrap_or(v),
                    _ => v,
                };
                match v {
                    Val::Null => None,
                    Val::Dur(ms) if ms > 0 => Some(ms),
                    other => {
                        return Err(verification_err(
                            ctx.full,
                            i.pos,
                            format!("the time window of [{}] must be a positive time duration, found [{}]", super::exec::call_src(ctx.full, i.pos), other.to_text()),
                        ))
                    }
                }
            }
        };
        // A window is anchored at the end of each output time bucket.
        if w.is_some() && bucket.is_none() {
            return Err(verification_err(
                ctx.full,
                i.pos,
                format!("the time window of [{}] requires grouping by a time bucket (TBUCKET)", super::exec::call_src(ctx.full, i.pos)),
            ));
        }
        windows.push(w);
    }
    // Each series' rows by timestamp (the windows select from them).
    let mut by_series: HashMap<&str, Vec<(i64, usize)>> = HashMap::new();
    if windows.iter().any(Option::is_some) {
        for grp in &groups {
            let v = by_series.entry(grp.tsid.as_str()).or_default();
            v.extend(grp.rows.iter().map(|&r| (t.rows[r][ts_i].as_i64().unwrap_or(i64::MIN), r)));
        }
        for v in by_series.values_mut() {
            v.sort_unstable();
        }
    }
    let in_range = |tsid: &str, lo: i64, hi: i64| -> Vec<usize> {
        let Some(v) = by_series.get(tsid) else { return Vec::new() };
        let a = v.partition_point(|(ts, _)| *ts < lo);
        let b = v.partition_point(|(ts, _)| *ts < hi);
        v[a..b].iter().map(|(_, r)| *r).collect()
    };
    let mut bound_inner: Vec<(Expr, Option<Val>, String)> = Vec::new();
    for i in &inners {
        let arg = match i.args.first() {
            Some(Expr::Col(n, p)) if super::exec::amd_union_pub(&t.cols, n).is_some() => Expr::Call("TO_AGGREGATE_METRIC_DOUBLE".into(), vec![Expr::Idx(super::exec::amd_union_pub(&t.cols, n).unwrap())], *p),
            Some(a) => bind(ctx, a, &t.cols)?,
            None => Expr::Lit(Val::Null),
        };
        let extra = match i.args.get(1) {
            Some(e) => Some(eval(&bind(ctx, e, &t.cols)?, &[], &ctx.fctx).unwrap_or(Val::Null)),
            None => None,
        };
        let ty = type_of(&arg, &t.cols).unwrap_or_else(|| "null".into());
        bound_inner.push((arg, extra, ty));
    }
    let span = bucket.as_ref().map(|b| b.2.clone());
    let mut cols: Vec<Column> = inners.iter().enumerate().map(|(k, i)| Column { name: format!("\u{0}ts{k}"), ty: inner_type(&i.name, &bound_inner[k].2) }).collect();
    for (gi, e) in &others {
        let ty = type_of(e, &t.cols).unwrap_or_else(|| "null".into());
        cols.push(Column { name: by[*gi].name.clone(), ty: if ty == "text" { "keyword".into() } else { ty } });
    }
    if let Some((gi, _, _)) = &bucket {
        cols.push(Column { name: by[*gi].name.clone(), ty: "date".into() });
    }
    let mut rows: Vec<Vec<Val>> = Vec::with_capacity(order.len());
    let mut series: Vec<String> = Vec::with_capacity(order.len());
    for &g in &order {
        let grp = &groups[g];
        let range = grp.bucket.map(|b| (b, bucket_end(b, span.as_ref().unwrap())));
        let prev = range.and_then(|(s, _)| {
            let span = span.as_ref()?;
            // The bucket just before: the one whose end is this start.
            index.range((grp.tsid.clone(), None)..(grp.tsid.clone(), Some(s))).next_back().filter(|(k, _)| k.1.is_some_and(|pb| bucket_end(pb, span) == s)).map(|(_, v)| groups[*v].rows.as_slice())
        });
        let next = range.and_then(|(_, e)| index.get(&(grp.tsid.clone(), Some(e))).map(|v| groups[*v].rows.as_slice()));
        let mut row = Vec::with_capacity(cols.len());
        for (k, i) in inners.iter().enumerate() {
            let (arg, extra, ty) = &bound_inner[k];
            match (windows[k], range) {
                // The series' samples in the window ending with the bucket,
                // `[end - window, end)`; a rate interpolates with the
                // samples of the windows just before and after it.
                (Some(w), Some((_, end))) => {
                    let start = end - w;
                    let rows = in_range(&grp.tsid, start, end);
                    let before = in_range(&grp.tsid, start - w, start);
                    let after = in_range(&grp.tsid, end, end + w);
                    let nb = ((!before.is_empty()).then_some(before.as_slice()), (!after.is_empty()).then_some(after.as_slice()));
                    row.push(inner_value(ctx, i, arg, extra.clone(), &t, ts_i, &rows, nb, Some((start, end)), ty)?);
                }
                _ => row.push(inner_value(ctx, i, arg, extra.clone(), &t, ts_i, &grp.rows, (prev, next), range, ty)?),
            }
        }
        for (gi, e) in &others {
            // A dimension reads the series' most recent document.
            if matches!(&by[*gi].expr, Expr::Col(c, _) if ctx.ts_dims.contains(c)) {
                let last = grp.rows.iter().max_by_key(|r| t.rows[**r][ts_i].as_i64().unwrap_or(i64::MIN));
                row.push(match last {
                    Some(r) => eval(e, &t.rows[*r], &ctx.fctx)?,
                    None => Val::Null,
                });
                continue;
            }
            let mut vals: Vec<Val> = Vec::new();
            for &r in &grp.rows {
                for v in eval(e, &t.rows[r], &ctx.fctx)?.values() {
                    if !vals.contains(v) {
                        vals.push(v.clone());
                    }
                }
            }
            vals.sort_by(cmp_vals);
            row.push(Val::from_list(vals));
        }
        if bucket.is_some() {
            row.push(grp.bucket.map(Val::Date).unwrap_or(Val::Null));
        }
        rows.push(row);
        series.push(grp.tsid.clone());
    }
    let mid = Table { cols, rows };
    ctx.rows_emitted += mid.rows.len() as u64;
    if bare {
        // One row per series (and bucket): the functions, `_timeseries`, the bucket.
        let mut out_cols: Vec<Column> = Vec::new();
        let mut idx: Vec<(Expr, usize)> = Vec::new();
        for f in &outer {
            let b = bind(ctx, &f.expr, &mid.cols)?;
            let ty = type_of(&b, &mid.cols).unwrap_or_else(|| "null".into());
            out_cols.push(Column { name: f.name.clone(), ty });
            idx.push((b, 0));
        }
        out_cols.push(Column { name: "_timeseries".into(), ty: "keyword".into() });
        let bucket_col = bucket.as_ref().map(|(gi, _, _)| mid.cols.iter().position(|c| c.name == by[*gi].name).unwrap());
        if let Some((gi, _, _)) = &bucket {
            out_cols.push(Column { name: by[*gi].name.clone(), ty: "date".into() });
        }
        let mut out_rows = Vec::with_capacity(mid.rows.len());
        for (ri, r) in mid.rows.iter().enumerate() {
            let mut row: Vec<Val> = Vec::with_capacity(out_cols.len());
            for (b, _) in &idx {
                row.push(eval(b, r, &ctx.fctx)?);
            }
            row.push(Val::Str(series[ri].clone()));
            if let Some(bc) = bucket_col {
                row.push(r[bc].clone());
            }
            out_rows.push(row);
        }
        let _ = HashMap::<(), ()>::new();
        return Ok(Table { cols: out_cols, rows: out_rows });
    }
    // Second pass: the user's aggregations over the series results.
    let new_by: Vec<Field> = by.iter().map(|g| Field { name: g.name.clone(), expr: Expr::Col(g.name.clone(), g.pos), filter: None, pos: g.pos }).collect();
    super::exec::stats_pub(ctx, mid, &outer, &new_by, pos)
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use serde_json::{json, Value};

    fn num(v: &Val) -> f64 {
        match v {
            Val::Double(x) => *x,
            Val::Long(x) | Val::Int(x) => *x as f64,
            _ => f64::NAN,
        }
    }

    /// Review finding 174: an explicit time window of a time-series
    /// function selects each series' samples in the window ending with the
    /// output bucket, not the bucket's own samples: over one-minute
    /// TBUCKETs a 5-minute AVG/SUM_OVER_TIME averages/sums the last five
    /// minutes (a reference computed here), a one-minute window equals the
    /// unwindowed value, a 5-minute RATE differs from the one-minute one
    /// (across a counter reset too), and a window without a time bucket is
    /// refused.
    #[test]
    fn time_series_windows_select_their_own_samples() {
        let tmp = tempfile::tempdir().unwrap();
        let node = crate::node::Node::open(tmp.path()).unwrap();
        let now = crate::date::now_millis();
        let base = now - now.rem_euclid(3_600_000) - 3_600_000;
        node.create_index("m", &json!({
            "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": 0,
                "index.time_series.start_time": crate::ilm::iso_date(base - 3_600_000), "index.time_series.end_time": crate::ilm::iso_date(now + 3_600_000)},
            "mappings": {"properties": {"@timestamp": {"type": "date"}, "host": {"type": "keyword", "time_series_dimension": true},
                "cpu": {"type": "double", "time_series_metric": "gauge"}, "req": {"type": "long", "time_series_metric": "counter"}}}}))
            .unwrap();
        // Minute k: cpu = k * k, req climbs 10 a minute and resets at minute 6.
        let req = |k: i64| if k < 6 { 10 * (k + 1) } else { 10 * (k - 5) };
        let docs: Vec<Value> = (0..10).map(|k| json!({"@timestamp": crate::ilm::iso_date(base + k * 60_000 + 30_000), "host": "a", "cpu": (k * k) as f64, "req": req(k)})).collect();
        let svc = node.get_index("m").unwrap();
        let reqs = docs.iter().map(|d| crate::index::WriteReq::Index { id: None, source: d.to_string(), routing: None, cond: Default::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
        let run = |q: &str| crate::esql::run(&node, q, crate::esql::EsqlOptions { default_limit: 1000, max_limit: 10_000, ..Default::default() });
        let out = run("TS m | STATS a1 = AVG(AVG_OVER_TIME(cpu)), a1w = AVG(AVG_OVER_TIME(cpu, 1 minute)), a5 = AVG(AVG_OVER_TIME(cpu, 5 minutes)), s5 = SUM(SUM_OVER_TIME(cpu, 5 minutes)), r1 = SUM(RATE(req)), r5 = SUM(RATE(req, 5 minutes)) BY b = TBUCKET(1 minute) | SORT b").unwrap();
        let col = |n: &str| out.table.cols.iter().position(|c| c.name == n).unwrap_or_else(|| panic!("{n}: {:?}", out.table.cols));
        let (a1, a1w, a5, s5, r1, r5) = (col("a1"), col("a1w"), col("a5"), col("s5"), col("r1"), col("r5"));
        assert_eq!(out.table.rows.len(), 10);
        let mut rates_differ = false;
        for (k, row) in out.table.rows.iter().enumerate() {
            let k = k as i64;
            let window: Vec<f64> = ((k - 4).max(0)..=k).map(|m| (m * m) as f64).collect();
            assert_eq!(num(&row[a1]), (k * k) as f64, "minute {k}");
            assert_eq!(num(&row[a1w]), (k * k) as f64, "minute {k}: a one-minute window is the bucket");
            let avg5 = window.iter().sum::<f64>() / window.len() as f64;
            assert!((num(&row[a5]) - avg5).abs() < 1e-9, "minute {k}: {} vs {avg5}", num(&row[a5]));
            assert!((num(&row[s5]) - window.iter().sum::<f64>()).abs() < 1e-9, "minute {k}");
            let (x1, x5) = (num(&row[r1]), num(&row[r5]));
            if x5.is_finite() && (x1.is_nan() || (x1 - x5).abs() > 1e-9) {
                rates_differ = true;
            }
        }
        assert!(rates_differ, "a 5-minute rate is not the one-minute bucket's rate: {:?}", out.table.rows);
        let Err(e) = run("TS m | STATS a5 = AVG(AVG_OVER_TIME(cpu, 5 minutes))") else { panic!("a window without a time bucket was accepted") };
        assert!(e.reason.contains("requires grouping by a time bucket"), "{}", e.reason);
    }
}
