//! Query approximation (`SET approximation=...`, ES 9.5 `Approximation`).
//!
//! ES samples the rows reaching the first `STATS` once the source holds
//! more than ten times the target sample size, and reports per-value
//! confidence intervals. This engine always computes the exact result,
//! which is what ES returns whenever it does not need to sample (sample
//! probability 1): every approximable numeric output column `c` gets
//! `_approximation_confidence_interval(c)` = `[c, c]` and
//! `_approximation_certified(c)` = `true`. Queries ES cannot approximate
//! run normally with ES's `approximation not supported` warning.

use super::exec::{call_src, Column, Table};
use super::parse::{Cmd, Expr};
use super::Val;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// `ApproximationVerifier.SUPPORTED_SINGLE_VALUED_AGGS`.
const SINGLE_VALUED: &[&str] = &["AVG", "COUNT", "MEDIAN", "MEDIAN_ABSOLUTE_DEVIATION", "PERCENTILE", "STD_DEV", "SUM", "VARIANCE", "WEIGHTED_AVG"];
/// `SUPPORTED_MULTIVALUED_AGGS`.
const MULTI_VALUED: &[&str] = &["SAMPLE"];
/// `ApproximationPlan.MULTIVALUED_OUTPUT_FUNCTIONS`.
const MV_OUTPUT: &[&str] = &["MV_APPEND", "MV_DIFFERENCE", "MV_INTERSECTION", "MV_SLICE", "MV_UNION"];

/// The effective approximation settings: `Some(confidence_level)` when enabled.
pub fn enabled(setting: &Option<Value>) -> Option<Option<f64>> {
    match setting {
        Some(Value::Bool(true)) => Some(Some(0.90)),
        Some(Value::Object(o)) => Some(match o.get("confidence_level") {
            None => Some(0.90),
            Some(Value::Null) => None,
            Some(v) => v.as_f64(),
        }),
        _ => None,
    }
}

/// `ApproximationSettings.parse(MapExpression)` + `Builder` validation.
pub fn validate(v: &Value) -> Option<String> {
    let Value::Object(o) = v else { return None };
    for (k, x) in o {
        match k.as_str() {
            "rows" => {
                if x.is_null() {
                    continue;
                }
                let Some(n) = x.as_i64().filter(|n| *n >= i32::MIN as i64 && *n <= i32::MAX as i64) else {
                    return Some("Approximation configuration [rows] must be an integer value".into());
                };
                if n < 10_000 {
                    return Some("Approximation configuration [rows] must be at least 10000".into());
                }
            }
            "confidence_level" => {
                if x.is_null() {
                    continue;
                }
                if !x.is_f64() {
                    return Some("Approximation configuration [confidence_level] must be a double value".into());
                }
                let c = x.as_f64().unwrap_or(0.0);
                if !(0.5..=0.95).contains(&c) {
                    return Some("Approximation configuration [confidence_level] must be between 0.5 and 0.95".into());
                }
            }
            other => return Some(format!("Approximation configuration contains unknown key [{other}]")),
        }
    }
    None
}

/// Byte offsets of the top-level pipeline's commands (after any `SET`s).
fn command_spans(full: &str) -> Vec<(usize, usize)> {
    let b = full.as_bytes();
    let mut spans = Vec::new();
    let (mut depth, mut start, mut i) = (0i32, 0usize, 0usize);
    while i < b.len() {
        match b[i] {
            b'"' => {
                if full[i..].starts_with("\"\"\"") {
                    i += 3;
                    while i < b.len() && !full[i..].starts_with("\"\"\"") {
                        i += 1;
                    }
                    i += 3;
                    continue;
                }
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'`' => {
                i += 1;
                while i < b.len() && b[i] != b'`' {
                    i += 1;
                }
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b';' if depth == 0 => {
                spans.clear();
                start = i + 1;
            }
            b'|' if depth == 0 => {
                spans.push((start, i));
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    spans.push((start, b.len()));
    spans
        .into_iter()
        .map(|(s, e)| {
            let t = &full[s..e];
            let lead = t.len() - t.trim_start().len();
            (s + lead, s + lead + t.trim().len())
        })
        .collect()
}

fn warn(full: &str, pos: usize, msg: String) -> String {
    let p = super::pos_prefix(full, pos);
    format!("{p}: approximation not supported: {msg}")
}

fn agg_calls<'e>(e: &'e Expr, out: &mut Vec<(&'e str, usize)>) {
    match e {
        Expr::Call(n, args, p) => {
            if super::func::AGGS.contains(&n.as_str()) {
                out.push((n.as_str(), *p));
            } else {
                args.iter().for_each(|a| agg_calls(a, out));
            }
        }
        Expr::Bin(_, a, b) => {
            agg_calls(a, out);
            agg_calls(b, out);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) => agg_calls(a, out),
        _ => {}
    }
}

fn refs(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Col(n, _) => out.push(n.clone()),
        Expr::Call(_, a, _) | Expr::In(_, a, _) => {
            if let Expr::In(x, _, _) = e {
                refs(x, out);
            }
            a.iter().for_each(|x| refs(x, out));
        }
        Expr::Bin(_, a, b) => {
            refs(a, out);
            refs(b, out);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) | Expr::Like(a, ..) => refs(a, out),
        Expr::Between(a, b, c, _) => {
            refs(a, out);
            refs(b, out);
            refs(c, out);
        }
        _ => {}
    }
}

fn calls_any(e: &Expr, names: &[&str]) -> bool {
    match e {
        Expr::Call(n, a, _) => names.contains(&n.as_str()) || a.iter().any(|x| calls_any(x, names)),
        Expr::Bin(_, a, b) => calls_any(a, names) || calls_any(b, names),
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(a, _) | Expr::IsNull(a, _) => calls_any(a, names),
        _ => false,
    }
}

/// `ApproximationVerifier.verifyPlan` (without `FORK`): `Err(warning)` when
/// the query cannot be approximated.
fn verify(full: &str, cmds: &[Cmd], types: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
    let spans = command_spans(full);
    let span = |i: usize| spans.get(i).copied().unwrap_or((0, 0));
    let text = |i: usize| {
        let (s, e) = span(i);
        full[s..e].to_string()
    };
    let first_stats = cmds.iter().position(|c| matches!(c, Cmd::Stats { .. }));
    let Some(first) = first_stats else {
        return Err(warn(full, span(0).0, "query must have [STATS] with aggregation function(s) that can be approximated".into()));
    };
    // Unsupported commands (bottom-up).
    for (i, c) in cmds.iter().enumerate() {
        let bad = match c {
            Cmd::From { ts, .. } => *ts,
            Cmd::Promql(_) | Cmd::ShowInfo => true,
            _ => false,
        };
        if bad {
            return Err(warn(full, span(i).0, format!("query with [{}] cannot be approximated", text(i))));
        }
    }
    // Chained STATS fail outright.
    if let Some(j) = cmds.iter().enumerate().skip(first + 1).find(|(_, c)| matches!(c, Cmd::Stats { .. })).map(|(j, _)| j) {
        return Err(warn(full, span(j).0, "query with chained [STATS] cannot be approximated".into()));
    }
    for (i, c) in cmds.iter().enumerate().take(first) {
        if matches!(c, Cmd::Limit(_) | Cmd::LimitBy(..)) {
            return Err(warn(full, span(i).0, format!("query with [{}] before [STATS] cannot be approximated", text(i))));
        }
    }
    if let Cmd::Stats { aggs, .. } = &cmds[first] {
        for f in aggs {
            let mut calls = Vec::new();
            agg_calls(&f.expr, &mut calls);
            for (n, p) in calls {
                if !SINGLE_VALUED.contains(&n) && !MULTI_VALUED.contains(&n) {
                    return Err(warn(full, p, format!("aggregation function [{}] cannot be approximated", call_src(full, p))));
                }
            }
            if let Some(t) = types(&f.name) {
                if !numeric(&t) {
                    let (n, p) = {
                        let mut c = Vec::new();
                        agg_calls(&f.expr, &mut c);
                        c.first().copied().unwrap_or(("", f.pos))
                    };
                    let _ = n;
                    let ty = if t == "date" { "DATETIME".to_string() } else { t.to_ascii_uppercase() };
                    return Err(warn(full, p, format!("aggregation function [{}] must return a numeric value; got [{ty}]", call_src(full, p))));
                }
            }
        }
    }
    Ok(())
}

fn numeric(t: &str) -> bool {
    matches!(t, "integer" | "long" | "double" | "unsigned_long")
}

/// Columns that carry confidence intervals (`ApproximationPlan` buckets),
/// tracked from the first `STATS` through `EVAL`, `RENAME` and `MV_EXPAND`.
fn bucket_columns(cmds: &[Cmd], types: &dyn Fn(&str) -> Option<String>) -> HashSet<String> {
    let mut b: HashSet<String> = HashSet::new();
    let mut seen = false;
    for c in cmds {
        match c {
            Cmd::Stats { aggs, .. } if !seen => {
                seen = true;
                for f in aggs {
                    let mut calls = Vec::new();
                    agg_calls(&f.expr, &mut calls);
                    let direct = matches!(&f.expr, Expr::Call(n, _, _) if SINGLE_VALUED.contains(&n.as_str()));
                    let derived = !calls.is_empty() && calls.iter().all(|(n, _)| SINGLE_VALUED.contains(n)) && !calls_any(&f.expr, MV_OUTPUT);
                    if direct || derived {
                        b.insert(f.name.clone());
                    } else {
                        b.remove(&f.name);
                    }
                }
            }
            _ if !seen => {}
            Cmd::Eval(fields) => {
                for f in fields {
                    let mut r = Vec::new();
                    refs(&f.expr, &mut r);
                    let num = types(&f.name).map(|t| numeric(&t)).unwrap_or(false);
                    if num && r.iter().any(|n| b.contains(n)) && !calls_any(&f.expr, MV_OUTPUT) {
                        b.insert(f.name.clone());
                    } else {
                        b.remove(&f.name);
                    }
                }
            }
            Cmd::Rename(pairs, _) => {
                for (from, to) in pairs {
                    let had = b.remove(from);
                    b.remove(to);
                    if had {
                        b.insert(to.clone());
                    }
                }
            }
            _ => {}
        }
    }
    b
}

/// Apply approximation to a finished top-level result: the warning when the
/// query cannot be approximated, else the confidence-interval columns.
pub fn apply(full: &str, cmds: &[Cmd], conf: Option<f64>, table: &mut Table, meta: &mut HashMap<String, Value>, warnings: &mut Vec<String>) {
    if cmds.iter().any(|c| matches!(c, Cmd::Fork(_))) {
        return;
    }
    let col_types: HashMap<String, String> = table.cols.iter().map(|c| (c.name.clone(), c.ty.clone())).collect();
    let types = |n: &str| col_types.get(n).cloned();
    if let Err(w) = verify(full, cmds, &types) {
        warnings.push(w);
        return;
    }
    if conf.is_none() {
        return;
    }
    let buckets = bucket_columns(cmds, &types);
    let targets: Vec<(usize, String, String)> = table
        .cols
        .iter()
        .enumerate()
        .filter(|(_, c)| buckets.contains(&c.name) && matches!(c.ty.as_str(), "integer" | "long" | "double"))
        .map(|(i, c)| (i, c.name.clone(), c.ty.clone()))
        .collect();
    for (_, name, ty) in &targets {
        let ci = format!("_approximation_confidence_interval({name})");
        let cert = format!("_approximation_certified({name})");
        meta.insert(ci.clone(), json!({"approximation": {"column": name, "type": "confidence_interval"}}));
        meta.insert(cert.clone(), json!({"approximation": {"column": name, "type": "certified"}}));
        table.cols.push(Column { name: ci, ty: ty.clone() });
        table.cols.push(Column { name: cert, ty: "boolean".into() });
    }
    for r in table.rows.iter_mut() {
        for (i, _, _) in &targets {
            let v = r[*i].clone();
            match v {
                Val::Null | Val::Multi(_) => {
                    r.push(Val::Null);
                    r.push(Val::Null);
                }
                v => {
                    r.push(Val::Multi(vec![v.clone(), v]));
                    r.push(Val::Bool(true));
                }
            }
        }
    }
}
