//! Less common aggregations: `top_metrics`, `weighted_avg`, `multi_terms`,
//! `rare_terms`, `significant_terms`, `auto_date_histogram`,
//! `variable_width_histogram`, `boxplot`, `median_absolute_deviation`,
//! `rate`, `string_stats`, `t_test`.
//!
//! Lives in a child module of `aggs` so it can use the parent's private
//! context (`Ctx`, `Key`, `Bucket`, ordering helpers) without widening them.

use super::{
    apply_parent_pipelines, cmp_key, count, field_name, fnum, jfmt, key_json, opt_num, parse_order, resolve_path, sketch::TDigest, Bucket, Col, Ctx, Docs,
    FieldRef, Key, Matcher, Rounding,
};
use crate::date::{add_units, round_down, CalUnit, DateFormatter, Tz};
use crate::error::{EsError, Result};
use crate::mapping::EsType;
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

pub(super) const EXTRA: &[&str] = &[
    "top_metrics",
    "weighted_avg",
    "multi_terms",
    "rare_terms",
    "significant_terms",
    "auto_date_histogram",
    "variable_width_histogram",
    "boxplot",
    "median_absolute_deviation",
    "rate",
    "string_stats",
    "t_test",
];

thread_local! {
    /// Enclosing `date_histogram`s (field, rounding), innermost last; `rate`
    /// reads the bucket width from here.
    static DH_PARENT: RefCell<Vec<(String, Rounding)>> = const { RefCell::new(Vec::new()) };
}

/// Marks a `date_histogram` as the parent of the sub-aggregations computed
/// while it is alive.
pub(super) struct DhGuard;

impl DhGuard {
    pub(super) fn push(field: &str, r: &Rounding) -> DhGuard {
        DH_PARENT.with(|p| p.borrow_mut().push((field.to_string(), r.clone())));
        DhGuard
    }
}

impl Drop for DhGuard {
    fn drop(&mut self) {
        DH_PARENT.with(|p| {
            p.borrow_mut().pop();
        });
    }
}

fn key_text(k: &Key, f: &FieldRef, fmt: &Option<DateFormatter>) -> String {
    match (k, f.ty) {
        (Key::Long(v), Some(t)) if t.is_date() => fmt.clone().unwrap_or_default().format(*v, Tz::UTC),
        (Key::Str(s), _) => s.clone(),
        (Key::Ip(b), _) => crate::values::ip_display(&crate::values::ip_to_term(b)),
        (Key::Long(v), _) => v.to_string(),
        (Key::ULong(v), _) => v.to_string(),
        (Key::Dbl(v), _) => jfmt(*v),
        (Key::Bool(b), _) => b.to_string(),
        (Key::Null, _) => "null".into(),
    }
}

/// `key` (+ `key_as_string`) of a terms-like bucket, as `terms` renders it.
fn insert_key(m: &mut Map<String, Value>, k: &Key, f: &FieldRef, fmt: &Option<DateFormatter>) {
    match (k, f.ty) {
        (Key::Bool(v), _) => {
            m.insert("key".into(), json!(*v as i64));
            m.insert("key_as_string".into(), json!(v.to_string()));
        }
        (Key::Long(v), Some(t)) if t.is_date() => {
            m.insert("key".into(), json!(v));
            m.insert("key_as_string".into(), json!(key_text(k, f, fmt)));
        }
        (k, _) => {
            m.insert("key".into(), key_json(k));
        }
    }
}

fn mean_var(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0) } else { f64::NAN };
    (m, var)
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9).
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b).
pub(crate) fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// P(T <= -|t|) for Student's t with `df` degrees of freedom.
pub(crate) fn t_lower_tail(t: f64, df: f64) -> f64 {
    super::cmath::t_cdf(-t.abs(), df)
}

/// ES `CompensatedSum` (Kahan summation).
#[derive(Default, Clone, Copy)]
pub(crate) struct Kahan {
    pub value: f64,
    delta: f64,
}

impl Kahan {
    /// Fold in another compensated sum (a part of the same values summed
    /// separately): its value, then its carried error.
    pub(crate) fn merge(&mut self, o: &Kahan) {
        self.add(o.value);
        self.add(o.delta);
    }

    pub(crate) fn add(&mut self, v: f64) {
        if !v.is_finite() {
            self.value += v;
        }
        if self.value.is_finite() {
            let corrected = v + self.delta;
            let updated = self.value + corrected;
            self.delta = corrected - (updated - self.value);
            self.value = updated;
        }
    }
}

fn sort_value_order(a: &Option<Key>, b: &Option<Key>, desc: bool) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, _) => Ordering::Greater,
        (_, None) => Ordering::Less,
        (Some(x), Some(y)) => {
            let o = cmp_key(x, y);
            if desc { o.reverse() } else { o }
        }
    }
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub(super) fn extra_agg(&self, name: &str, t: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        match t {
            "top_metrics" => self.top_metrics(name, o, docs),
            "weighted_avg" => self.weighted_avg(name, o, docs),
            "multi_terms" => self.multi_terms(name, o, sub, docs),
            "rare_terms" => self.rare_terms(name, o, sub, docs),
            "significant_terms" => self.significant_terms(name, o, sub, docs),
            "auto_date_histogram" => self.auto_date_histogram(name, o, sub, docs),
            "variable_width_histogram" => self.variable_width_histogram(name, o, sub, docs),
            "boxplot" | "median_absolute_deviation" => self.quantile_metric(name, t, o, docs),
            "rate" => self.rate(name, o, docs),
            "string_stats" => self.string_stats(name, o, docs),
            "t_test" => self.t_test(name, o, docs),
            _ => Err(EsError::parsing_at(format!("Unknown aggregation type [{t}]"), 1, 1)),
        }
    }

    fn numbers(&self, f: &FieldRef, docs: &Docs, missing: &Option<Key>) -> Result<Vec<f64>> {
        self.num_values(f, docs, missing)
    }

    // -----------------------------------------------------------------------
    // top_metrics

    fn top_metrics(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let size = o.get("size").and_then(Value::as_u64).unwrap_or(1) as usize;
        if size > self.env.top_metrics_max_size {
            return Err(EsError::illegal_argument(format!(
                "[top_metrics.size] must not be more than [{}] but was [{}]. This limit can be set by changing the [index.top_metrics_max_size] index level setting.",
                self.env.top_metrics_max_size, size
            )));
        }
        let metric_fields: Vec<String> = match o.get("metrics") {
            Some(Value::Array(a)) => a.iter().filter_map(|m| m.get("field").and_then(Value::as_str).map(str::to_string)).collect(),
            Some(Value::Object(m)) => m.get("field").and_then(Value::as_str).map(|s| vec![s.to_string()]).unwrap_or_default(),
            _ => Vec::new(),
        };
        if metric_fields.is_empty() {
            return Err(EsError::illegal_argument(format!("[metrics] must not be empty: [{name}]")));
        }
        // sort: "_score" | {"f": "asc"} | {"f": {"order": "desc"}} | [one of those]
        let sort_v = match o.get("sort") {
            Some(Value::Array(a)) => {
                if a.len() > 1 {
                    return Err(EsError::illegal_argument("[top_metrics] only supports sorting on a single field"));
                }
                a.first().cloned()
            }
            other => other.cloned(),
        };
        let (sort_field, desc) = match &sort_v {
            None => ("_score".to_string(), true),
            Some(Value::String(s)) => (s.clone(), s == "_score"),
            Some(Value::Object(m)) => {
                let (k, v) = m.iter().next().ok_or_else(|| EsError::parsing_at("[sort] must not be empty", 1, 1))?;
                let ord = match v {
                    Value::String(s) => Some(s.as_str()),
                    Value::Object(x) => x.get("order").and_then(Value::as_str),
                    _ => None,
                };
                (k.clone(), ord.map(|s| s == "desc").unwrap_or(k == "_score"))
            }
            Some(_) => return Err(EsError::parsing_at("[sort] must be a string or object", 1, 1)),
        };
        // `_geo_distance` sort: distance (meters) from the origin to each doc's
        // geo_point; the sort value is the distance itself.
        let geo_sort: Option<(String, f64, f64)> = match &sort_v {
            Some(Value::Object(m)) if sort_field == "_geo_distance" => {
                let inner = m.get("_geo_distance").and_then(Value::as_object).ok_or_else(|| EsError::parsing_at("[_geo_distance] must be an object", 1, 1))?;
                let (gf, origin) = inner
                    .iter()
                    .find(|(k, _)| !matches!(k.as_str(), "order" | "unit" | "mode" | "distance_type" | "ignore_unmapped" | "validation_method" | "nested"))
                    .ok_or_else(|| EsError::parsing_at("[_geo_distance] requires a field", 1, 1))?;
                let (olat, olon) = crate::values::parse_geo_point(origin)
                    .ok_or_else(|| EsError::parsing_at(format!("failed to parse [_geo_distance] origin [{origin}]"), 1, 1))?;
                Some((gf.clone(), olat, olon))
            }
            _ => None,
        };
        // `_script` sorts: only numeric scripts.
        let script_sort = match &sort_v {
            Some(Value::Object(m)) if sort_field == "_script" => {
                let sv = m.get("_script").cloned().unwrap_or(Value::Null);
                let ty = sv.get("type").and_then(Value::as_str).unwrap_or("number");
                if ty != "number" {
                    return Err(EsError::illegal_argument(format!("error building sort for [_script]: script sorting only supported on [numeric] scripts but was [{ty}]")));
                }
                let mut so = Map::new();
                so.insert("script".into(), sv.get("script").cloned().unwrap_or(Value::Null));
                Some(self.source(&so, name, "top_metrics")?)
            }
            _ => None,
        };
        let desc = if script_sort.is_some() {
            sort_v.as_ref().and_then(|v| v.get("_script")).and_then(|s| s.get("order")).and_then(Value::as_str) == Some("desc")
        } else {
            desc
        };
        let sf = if sort_field == "_score" {
            None
        } else if geo_sort.is_some() {
            None
        } else if let Some(s) = script_sort {
            Some(s)
        } else {
            let f = self.field(&sort_field)?;
            if let Some(t) = f.ty {
                if !t.is_numeric() && !t.is_date() {
                    return Err(EsError::illegal_argument(format!(
                        "error building sort for field [{sort_field}] of type [{}] in index [{}]: only supported on numeric fields",
                        t.name(),
                        self.segs.first().map(|s| s.index).unwrap_or("")
                    )));
                }
            }
            Some(f)
        };
        let sort_is_date = sf.as_ref().and_then(|f| f.ty).is_some_and(|t| t.is_date());
        let sort_fmt = sf.as_ref().and_then(|f| f.format.clone()).unwrap_or_default();
        let mut mfs: Vec<FieldRef> = Vec::new();
        for m in &metric_fields {
            let f = self.source_field(m)?;
            if f.src.is_some() && f.ty == Some(crate::mapping::EsType::Keyword) {
                return Err(EsError::illegal_argument(format!("top_metrics can only collect bytes that have segment ordinals but Field [{m}] of type [keyword] does not")));
            }
            mfs.push(f);
        }
        let mut rows: Vec<(Option<Key>, usize, DocId)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, sc) in v {
                let k = match &sf {
                    None if geo_sort.is_some() => {
                        let (gf, olat, olon) = geo_sort.as_ref().unwrap();
                        let (olat, olon) = (*olat, *olon);
                        self.geo_points(gf, s, d)
                            .into_iter()
                            .map(|(a, b)| crate::sloppy_math::haversin_meters(olat, olon, a, b))
                            .min_by(|a, b| a.total_cmp(b))
                            .map(Key::Dbl)
                    }
                    None => Some(Key::Dbl(sc as f64)),
                    Some(f) => {
                        let vals = self.values(f, s, d);
                        if desc { vals.into_iter().max_by(cmp_key) } else { vals.into_iter().min_by(cmp_key) }
                    }
                };
                rows.push((k, s, d));
            }
        }
        rows.sort_by(|a, b| sort_value_order(&a.0, &b.0, desc));
        rows.truncate(size);
        let mut top = Vec::new();
        for (k, s, d) in rows {
            let mut metrics = Map::new();
            for (n, f) in metric_fields.iter().zip(&mfs) {
                let v = self.values(f, s, d).into_iter().next();
                let j = match (v, f.ty) {
                    (None, _) => Value::Null,
                    (Some(Key::Long(x)), Some(t)) if t.is_date() => json!(f.format.clone().unwrap_or_default().format(x, Tz::UTC)),
                    (Some(Key::Bool(b)), _) => json!(b),
                    (Some(k), _) => key_json(&k),
                };
                metrics.insert(n.clone(), j);
            }
            let sv = match k {
                None => Value::Null,
                Some(Key::Long(x)) if sort_is_date => json!(sort_fmt.format(x, Tz::UTC)),
                Some(Key::Dbl(x)) => fnum(x),
                Some(k) => key_json(&k),
            };
            top.push(json!({"sort": [sv], "metrics": metrics}));
        }
        Ok(json!({ "top": top }))
    }

    // -----------------------------------------------------------------------
    // weighted_avg

    fn weighted_avg(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let part = |k: &str| -> Result<(FieldRef, Option<Key>)> {
            let p = o.get(k).and_then(Value::as_object).ok_or_else(|| EsError::illegal_argument(format!("[{k}] must be set for weighted_avg aggregation [{name}]")))?;
            let f = self.field(field_name(p, name, "weighted_avg")?)?;
            let m = self.missing_key(&f, p.get("missing"))?;
            Ok((f, m))
        };
        let (vf, vm) = part("value")?;
        let (wf, wm) = part("weight")?;
        let (mut sum, mut wsum) = (0.0, 0.0);
        let mut any = false;
        for (s, v) in docs.iter().enumerate() {
            for &(d, _) in v {
                let ws: Vec<f64> = self.values_or(&wf, s, d, &wm).iter().filter_map(Key::as_f64).collect();
                if ws.len() > 1 {
                    return Err(EsError::new(400, "aggregation_execution_exception", "Encountered more than one weight for a single document. Use a script to combine multiple weights-per-doc into a single value."));
                }
                let Some(w) = ws.first().copied() else { continue };
                for x in self.values_or(&vf, s, d, &vm).iter().filter_map(Key::as_f64) {
                    sum += x * w;
                    wsum += w;
                    any = true;
                }
            }
        }
        let v = if any && wsum != 0.0 { Some(sum / wsum) } else { None };
        let mut m = Map::new();
        m.insert("value".into(), opt_num(v));
        if let (Some(fmt), Some(x)) = (o.get("format").and_then(Value::as_str), v) {
            if vf.ty.map(|t| t.is_date()).unwrap_or(false) {
                m.insert("value_as_string".into(), json!(DateFormatter::new(fmt)?.format(x as i64, Tz::UTC)));
            }
        }
        Ok(Value::Object(m))
    }

    // -----------------------------------------------------------------------
    // multi_terms

    fn multi_terms(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let terms = o.get("terms").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument(format!("[terms] must not be null: [{name}]")))?;
        if terms.len() < 2 {
            return Err(EsError::illegal_argument(format!("The [terms] parameter in the aggregation [{name}] must be present and have at least 2 fields or scripts.")));
        }
        let mut srcs: Vec<(FieldRef, Option<Key>, Option<DateFormatter>)> = Vec::new();
        for t in terms {
            let to = t.as_object().ok_or_else(|| EsError::parsing_at("[terms] entries must be objects", 1, 1))?;
            let f = self.field(field_name(to, name, "multi_terms")?)?;
            let m = self.missing_key(&f, to.get("missing"))?;
            let fmt = to.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone());
            srcs.push((f, m, fmt));
        }
        let size = o.get("size").and_then(Value::as_i64).unwrap_or(10);
        if size <= 0 {
            return Err(EsError::illegal_argument(format!("[size] must be greater than 0. Found [{size}] in [{name}]")));
        }
        let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(1);
        let order = parse_order(o.get("order"), &[("_count", false)])?;
        let show_err = o.get("show_term_doc_count_error").and_then(Value::as_bool).unwrap_or(false);
        // ES selects the top `shard_size` buckets per shard before reducing;
        // with a single shard that is the set of buckets that survive.
        let shard_size = o.get("shard_size").and_then(Value::as_i64).unwrap_or(100).max(1) as usize;
        let mut groups: HashMap<Vec<Key>, Docs> = HashMap::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, sc) in v {
                let mut combos: Vec<Vec<Key>> = vec![Vec::new()];
                for (f, m, _) in &srcs {
                    let mut ks = self.values_or(f, s, d, m);
                    ks.sort_by(cmp_key);
                    ks.dedup();
                    let mut next = Vec::new();
                    for c in &combos {
                        for k in &ks {
                            let mut c2 = c.clone();
                            c2.push(k.clone());
                            next.push(c2);
                        }
                    }
                    combos = next;
                }
                for c in combos {
                    let kb = c.capacity() * std::mem::size_of::<Key>() + c.iter().map(Key::heap_bytes).sum::<usize>();
                    self.push_grouped(name, &mut groups, c, kb, docs.len(), s, (d, sc))?;
                }
            }
        }
        // Each index stands for one shard (indices default to one): every
        // shard keeps its top `shard_size` buckets by `_count`, regardless of
        // the final `order`; the coordinator merges and re-sorts them.
        let mut by_shard: Vec<(&str, HashMap<Vec<Key>, Docs>)> = Vec::new();
        for (k, d) in groups {
            for (s, v) in d.into_iter().enumerate() {
                if v.is_empty() {
                    continue;
                }
                let ix = self.segs[s].index;
                let pos = match by_shard.iter().position(|(n, _)| *n == ix) {
                    Some(p) => p,
                    None => {
                        by_shard.push((ix, HashMap::new()));
                        by_shard.len() - 1
                    }
                };
                by_shard[pos].1.entry(k.clone()).or_insert_with(|| vec![Vec::new(); docs.len()])[s] = v;
            }
        }
        let shard_cmp = |(ka, a): &(Vec<Key>, Bucket), (kb, b): &(Vec<Key>, Bucket)| {
            let c = a.count.cmp(&b.count);
            if c != Ordering::Equal {
                return c.reverse();
            }
            ka.iter().zip(kb.iter()).map(|(x, y)| cmp_key(x, y)).find(|c| *c != Ordering::Equal).unwrap_or(Ordering::Equal)
        };
        let mut shard_other = 0u64;
        let mut merged: HashMap<Vec<Key>, Docs> = HashMap::new();
        for (_, g) in by_shard {
            let mut sb: Vec<(Vec<Key>, Bucket)> = g.into_iter().map(|(k, d)| (k, Bucket { key: Key::Null, count: count(&d), docs: d, subs: None })).collect();
            let shard_total: u64 = sb.iter().map(|(_, b)| b.count).sum();
            sb.sort_by(shard_cmp);
            sb.truncate(shard_size);
            shard_other += shard_total - sb.iter().map(|(_, b)| b.count).sum::<u64>();
            for (k, b) in sb {
                let e = merged.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()]);
                for (s, v) in b.docs.into_iter().enumerate() {
                    e[s].extend(v);
                }
            }
        }
        let mut bs: Vec<(Vec<Key>, Bucket)> =
            merged.into_iter().map(|(k, d)| (k, Bucket { key: Key::Null, count: count(&d), docs: d, subs: None })).filter(|(_, b)| b.count >= min_doc_count).collect();
        let total: u64 = bs.iter().map(|(_, b)| b.count).sum::<u64>() + shard_other;
        bs.sort_by(shard_cmp);
        if order.iter().any(|(k, _)| k != "_count" && k != "_key") {
            for (_, b) in bs.iter_mut() {
                b.subs = Some(self.subs(sub, &b.docs)?);
            }
        }
        bs.sort_by(|(ka, a), (kb, b)| {
            for (k, asc) in &order {
                let c = match k.as_str() {
                    "_count" => a.count.cmp(&b.count),
                    "_key" => ka.iter().zip(kb.iter()).map(|(x, y)| cmp_key(x, y)).find(|c| *c != Ordering::Equal).unwrap_or(Ordering::Equal),
                    path => {
                        let mk = |x: &Bucket| -> Option<f64> {
                            let mut m = x.subs.clone().unwrap_or_default();
                            m.insert("doc_count".into(), json!(x.count));
                            resolve_path(&m, path)
                        };
                        match (mk(a), mk(b)) {
                            (Some(x), Some(y)) => x.total_cmp(&y),
                            (None, Some(_)) => return Ordering::Greater,
                            (Some(_), None) => return Ordering::Less,
                            _ => Ordering::Equal,
                        }
                    }
                };
                let c = if *asc { c } else { c.reverse() };
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        });
        bs.truncate(size as usize);
        self.add_buckets(bs.len())?;
        let kept: u64 = bs.iter().map(|(_, b)| b.count).sum();
        let mut out = Vec::new();
        for (keys, b) in bs {
            let subs = match b.subs {
                Some(s) => s,
                None => self.subs(sub, &b.docs)?,
            };
            let mut m = Map::new();
            let key: Vec<Value> = keys.iter().map(|k| if let Key::Bool(x) = k { json!(x) } else { key_json(k) }).collect();
            let kas: Vec<String> = keys.iter().zip(&srcs).map(|(k, (f, _, fmt))| key_text(k, f, fmt)).collect();
            m.insert("key".into(), Value::Array(key));
            m.insert("key_as_string".into(), json!(kas.join("|")));
            m.insert("doc_count".into(), json!(b.count));
            if show_err {
                m.insert("doc_count_error_upper_bound".into(), json!(0));
            }
            m.extend(subs);
            out.push(m);
        }
        apply_parent_pipelines(sub, &mut out)?;
        Ok(json!({"doc_count_error_upper_bound": 0, "sum_other_doc_count": total - kept, "buckets": out}))
    }

    /// Group docs by the (deduplicated) values of `f`.
    /// (Charged to the request breaker as aggregation `name`.)
    pub(super) fn group_by(&self, name: &str, f: &FieldRef, docs: &Docs, missing: &Option<Key>) -> Result<HashMap<Key, Docs>> {
        let mut groups: HashMap<Key, Docs> = HashMap::new();
        // Single-valued keyword columns group by ordinal (on the search
        // pool when there are many docs), then one key per distinct term;
        // each group's docs stay in doc order.
        let mut ord_segs: Vec<(usize, std::rc::Rc<Col<'_>>)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            if !v.is_empty() {
                if let Some(c) = self.ord_col(f, s) { ord_segs.push((s, c)); }
            }
        }
        {
            // Slices of each segment's docs, in doc order (a large segment
            // groups in parts).
            let slice = (rustic_core::query::scorer::part_split_min_docs() as usize / 2).max(1);
            let mut work: Vec<(usize, &super::KwCol<'_>, usize, usize)> = Vec::new();
            for (s, c) in &ord_segs {
                let Col::Kw(kc) = &**c else { unreachable!() };
                let n = docs[*s].len();
                let mut i0 = 0;
                while i0 < n { let i1 = (i0 + slice).min(n); work.push((*s, kc, i0, i1)); i0 = i1; }
            }
            let keep_missing = missing.is_some();
            let group_seg = |&(s, kc, i0, i1): &(usize, &super::KwCol<'_>, usize, usize)| -> (Vec<(u32, Vec<(DocId, f32)>)>, Vec<(DocId, f32)>) {
                let mut by_ord: HashMap<u32, Vec<(DocId, f32)>> = HashMap::new();
                let mut none: Vec<(DocId, f32)> = Vec::new();
                for &(d, sc) in &docs[s][i0..i1] {
                    let o = kc.ord(d);
                    if o == rustic_core::codec::fastfields::MISSING_ORD {
                        if keep_missing { none.push((d, sc)); }
                    } else {
                        by_ord.entry(o).or_default().push((d, sc));
                    }
                }
                (by_ord.into_iter().collect(), none)
            };
            let total: usize = work.iter().map(|w| w.3 - w.2).sum();
            let parts: Vec<(Vec<(u32, Vec<(DocId, f32)>)>, Vec<(DocId, f32)>)> = if total >= 2 * slice && work.len() > 1 {
                rustic_core::query::searcher::par_map_search(work.len(), |i| group_seg(&work[i]))
            } else {
                work.iter().map(group_seg).collect()
            };
            for ((s, kc, _, _), (ords, none)) in work.iter().zip(parts) {
                let s = *s;
                let ty = f.per_seg[s].as_ref().map(|rf| rf.ty).unwrap_or(EsType::Keyword);
                let mut seg_parts: Vec<(Key, Vec<(DocId, f32)>)> = Vec::with_capacity(ords.len() + 1);
                for (o, ds) in ords {
                    if let Some(t) = kc.ord_term(o) { seg_parts.push((Self::term_key(ty, &t), ds)); }
                }
                if let (Some(m), false) = (missing, none.is_empty()) { seg_parts.push((m.clone(), none)); }
                for (k, ds) in seg_parts {
                    let kb = k.heap_bytes();
                    let n = ds.len();
                    let slot = match groups.entry(k) {
                        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            self.charge_group(name, kb, docs.len())?;
                            e.insert(vec![Vec::new(); docs.len()])
                        }
                    };
                    self.charge(name, 8 * n as u64)?;
                    // Slices come in doc order: appending keeps it, unless
                    // a missing key equal to a term mixed two lists.
                    if slot[s].is_empty() { slot[s] = ds; } else {
                        let ordered = slot[s].last().map(|p| p.0) < ds.first().map(|p| p.0);
                        slot[s].extend(ds);
                        if !ordered { slot[s].sort_by_key(|p| p.0); }
                    }
                }
            }
        }
        let ord_done: std::collections::HashSet<usize> = ord_segs.iter().map(|(s, _)| *s).collect();
        for (s, v) in docs.iter().enumerate() {
            if v.is_empty() || ord_done.contains(&s) {
                continue;
            }
            for &(d, sc) in v {
                let mut ks = self.values_or(f, s, d, missing);
                ks.sort_by(cmp_key);
                ks.dedup();
                for k in ks {
                    let kb = k.heap_bytes();
                    self.push_grouped(name, &mut groups, k, kb, docs.len(), s, (d, sc))?;
                }
            }
        }
        Ok(groups)
    }

    // -----------------------------------------------------------------------
    // rare_terms

    fn rare_terms(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.field(field_name(o, name, "rare_terms")?)?;
        let max = o.get("max_doc_count").and_then(Value::as_i64).unwrap_or(1);
        if max <= 0 {
            return Err(EsError::illegal_argument(format!("[max_doc_count] must be greater than 0. Found [{max}] in [{name}]")));
        }
        if max > 100 {
            return Err(EsError::illegal_argument(format!("[max_doc_count] must be smaller or equal to [100], found [{max}] in [{name}]")));
        }
        let missing = self.missing_key(&f, o.get("missing"))?;
        let include = Matcher::parse(o.get("include"))?;
        let exclude = Matcher::parse(o.get("exclude"))?;
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone());
        let all: Vec<Bucket> = self
            .group_by(name, &f, docs, &missing)?
            .into_iter()
            .filter(|(k, _)| include.as_ref().map(|m| m.matches(k)).unwrap_or(true) && !exclude.as_ref().map(|m| m.matches(k)).unwrap_or(false))
            .map(|(key, d)| Bucket { count: count(&d), key, docs: d, subs: None })
            .collect();
        // A coordinator reduce follows: ship the terms that are too frequent
        // here, so it drops them where other shards found them rare.
        let frequent = crate::search::aggregation_shard_response().then(|| {
            let over: Vec<&Bucket> = all.iter().filter(|b| b.count > max as u64).collect();
            let precision = o.get("precision").and_then(Value::as_f64).filter(|p| *p > 0.0 && *p < 1.0).unwrap_or(0.001);
            let mut bloom = super::sketch::KeyBloom::with_capacity(over.len(), precision);
            for b in &over {
                let mut m = Map::new();
                insert_key(&mut m, &b.key, &f, &fmt);
                bloom.insert(&m["key"].to_string());
            }
            if over.is_empty() { json!({}) } else { json!({ "frequent": bloom.state() }) }
        });
        let mut bs: Vec<Bucket> = all.into_iter().filter(|b| b.count <= max as u64).collect();
        bs.sort_by(|a, b| a.count.cmp(&b.count).then_with(|| cmp_key(&a.key, &b.key)));
        self.add_buckets(bs.len())?;
        let mut out = Vec::new();
        for b in bs {
            let mut m = Map::new();
            insert_key(&mut m, &b.key, &f, &fmt);
            m.insert("doc_count".into(), json!(b.count));
            m.extend(self.subs(sub, &b.docs)?);
            out.push(m);
        }
        apply_parent_pipelines(sub, &mut out)?;
        let mut v = json!({ "buckets": out });
        if let Some(st) = frequent {
            v["__p"] = st;
        }
        Ok(v)
    }

    // -----------------------------------------------------------------------
    // significant_terms

    fn significant_terms(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.field(field_name(o, name, "significant_terms")?)?;
        let size = o.get("size").and_then(Value::as_i64).unwrap_or(10);
        if size <= 0 {
            return Err(EsError::illegal_argument(format!("[size] must be greater than 0. Found [{size}] in [{name}]")));
        }
        let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(3);
        let include = Matcher::parse(o.get("include"))?;
        let exclude = Matcher::parse(o.get("exclude"))?;
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone());
        // A shard partial's candidates keep non-positive scores (see
        // `aggs::SHARD_CANDIDATES`).
        let shard_candidates = o.get(super::SHARD_CANDIDATES).and_then(Value::as_bool) == Some(true);
        let mut heuristic = "jlh".to_string();
        let mut include_negatives = false;
        let mut bg_superset = true;
        let mut normalize_above = 0u64;
        for h in ["jlh", "mutual_information", "chi_square", "gnd", "percentage", "script_heuristic", "p_value"] {
            if let Some(cfg) = o.get(h) {
                if h == "script_heuristic" {
                    return Err(EsError::illegal_argument(format!("[script_heuristic] is not supported by this server: [{name}]")));
                }
                heuristic = h.to_string();
                include_negatives = cfg.get("include_negatives").and_then(Value::as_bool).unwrap_or(false);
                bg_superset = cfg.get("background_is_superset").and_then(Value::as_bool).unwrap_or(true);
                normalize_above = cfg.get("normalize_above").and_then(Value::as_u64).unwrap_or(0);
            }
        }
        let subset_size = count(docs);
        let fg_groups = self.group_by(name, &f, docs, &None)?;
        // Background set: every live doc of the targets, optionally
        // filtered; its size and term counts are the same for every parent
        // bucket, so built once per request.
        let bg_key = format!("{}\n{}", field_name(o, name, "significant_terms")?, o.get("background_filter").map(|q| q.to_string()).unwrap_or_default());
        let background = self.background_counts(bg_key, || {
            let all: Docs = (0..self.segs.len()).map(|s| (self.env.all_docs)(s).into_iter().map(|d| (d, 0.0)).collect()).collect();
            let bg: Docs = match o.get("background_filter") {
                Some(q) => self.filter_docs(q, &all)?,
                None => all,
            };
            Ok((self.count_by(&f, &bg, &None)?, count(&bg)))
        })?;
        let (bg_counts, superset_size) = (&background.0, background.1);
        let score_of = |sub_f: u64, sup_f: u64| -> f64 {
            let (sub_f, sup_f, sub_n, sup_n) = (sub_f as f64, sup_f as f64, subset_size as f64, superset_size as f64);
            match heuristic.as_str() {
                "p_value" => super::pvalue::score(sub_f as u64, sub_n as u64, sup_f as u64, sup_n as u64, bg_superset, normalize_above),
                "percentage" => {
                    if sup_f == 0.0 { 0.0 } else { sub_f / sup_f }
                }
                "gnd" => {
                    let fx = sup_f;
                    let fy = sub_n;
                    let fxy = sub_f;
                    if fxy == 0.0 || fx == 0.0 || fy == 0.0 {
                        return 0.0;
                    }
                    if fx == fy && fy == fxy {
                        return 1.0;
                    }
                    let s = (fx.ln().max(fy.ln()) - fxy.ln()) / (sup_n.ln() - fx.ln().min(fy.ln()));
                    (-s).exp()
                }
                "mutual_information" | "chi_square" => {
                    let (n11, n10, n01, n00) = if bg_superset {
                        (sub_f, sup_f - sub_f, sub_n - sub_f, (sup_n - sub_n) - (sup_f - sub_f))
                    } else {
                        (sub_f, sup_f, sub_n - sub_f, sup_n - sup_f)
                    };
                    let n = n11 + n10 + n01 + n00;
                    let (n1_, n0_, n_1, n_0) = (n10 + n11, n00 + n01, n01 + n11, n10 + n00);
                    if !include_negatives && n_1 > 0.0 && n_0 > 0.0 && n11 / n_1 < n10 / n_0 {
                        return f64::NEG_INFINITY;
                    }
                    if heuristic == "chi_square" {
                        let den = n_1 * n1_ * n0_ * n_0;
                        if den == 0.0 { 0.0 } else { n * (n11 * n00 - n10 * n01).powi(2) / den }
                    } else {
                        let term = |nxy: f64, nx: f64, ny: f64| -> f64 {
                            if nxy == 0.0 || nx == 0.0 || ny == 0.0 { 0.0 } else { nxy / n * ((n * nxy) / (nx * ny)).log2() }
                        };
                        term(n11, n1_, n_1) + term(n01, n0_, n_1) + term(n10, n1_, n_0) + term(n00, n0_, n_0)
                    }
                }
                _ => {
                    if sub_f == 0.0 || sub_n == 0.0 || sup_n == 0.0 || sup_f == 0.0 {
                        return 0.0;
                    }
                    let sp = sub_f / sub_n;
                    let bp = sup_f / sup_n;
                    let abs = sp - bp;
                    if abs <= 0.0 { 0.0 } else { abs * (sp / bp) }
                }
            }
        };
        let mut bs: Vec<(f64, u64, Bucket)> = fg_groups
            .into_iter()
            .filter(|(k, _)| include.as_ref().map(|m| m.matches(k)).unwrap_or(true) && !exclude.as_ref().map(|m| m.matches(k)).unwrap_or(false))
            .map(|(key, d)| {
                let c = count(&d);
                let bgc = bg_counts.get(&key).copied().unwrap_or(0).max(if bg_superset { c } else { 0 });
                (score_of(c, bgc), bgc, Bucket { count: c, key, docs: d, subs: None })
            })
            .filter(|(sc, _, b)| (shard_candidates || *sc > 0.0) && b.count >= min_doc_count)
            .collect();
        if shard_candidates {
            // A shard's candidates as ES picks them: under a parent bucket
            // its (remapped) ordinals are enumerated in the order each term
            // was first collected (segment, then doc), through Lucene's PriorityQueue
            // of `shard_size` ordered by score alone
            // (`BucketSignificancePriorityQueue`), so which of several equal
            // scores at the cut stays is the queue's.
            // At the top level (one owning bucket) ES walks its dense global
            // ordinals in term order instead.
            let top_level = std::ptr::eq(docs as *const Docs as *const u8, self.root_docs as *const u8);
            let first_seen = |d: &Docs| d.iter().enumerate().find_map(|(s, v)| v.first().map(|p| (s, p.0))).unwrap_or((usize::MAX, DocId::MAX));
            if top_level {
                bs.sort_by(|a, b| cmp_key(&a.2.key, &b.2.key));
            } else {
                bs.sort_by(|a, b| first_seen(&a.2.docs).cmp(&first_seen(&b.2.docs)).then_with(|| cmp_key(&a.2.key, &b.2.key)));
            }
            let mut heap = LuceneHeap::new(size as usize, |a: &(f64, u64, Bucket), b: &(f64, u64, Bucket)| a.0 < b.0);
            for b in bs {
                heap.insert_with_overflow(b);
            }
            // In the order the shard lists them (the reduce puts their keys
            // into its map in this order).
            bs = heap.into_list();
        } else {
            bs.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| cmp_key(&a.2.key, &b.2.key)));
            bs.truncate(size as usize);
        }
        self.add_buckets(bs.len())?;
        let mut out = Vec::new();
        for (sc, bgc, b) in bs {
            let mut m = Map::new();
            insert_key(&mut m, &b.key, &f, &fmt);
            m.insert("doc_count".into(), json!(b.count));
            m.insert("score".into(), fnum(sc));
            m.insert("bg_count".into(), json!(bgc));
            m.extend(self.subs(sub, &b.docs)?);
            out.push(m);
        }
        apply_parent_pipelines(sub, &mut out)?;
        Ok(json!({"doc_count": subset_size, "bg_count": superset_size, "buckets": out}))
    }

    // -----------------------------------------------------------------------
    // auto_date_histogram

    fn auto_date_histogram(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.field(field_name(o, name, "auto_date_histogram")?)?;
        let target = o.get("buckets").and_then(Value::as_i64).unwrap_or(10);
        if target <= 0 {
            return Err(EsError::illegal_argument(format!("[buckets] must be greater than 0. Found [{target}] in [{name}]")));
        }
        let target = target as u64;
        let tz = match o.get("time_zone").and_then(Value::as_str) {
            Some(z) => Tz::parse(z)?,
            None => Tz::UTC,
        };
        let fmt = o.get("format").and_then(Value::as_str).map(DateFormatter::new).transpose()?.or_else(|| f.format.clone()).unwrap_or_default();
        let missing = self.missing_key(&f, o.get("missing"))?;
        let min_unit = super::auto_date::minimum_unit(o.get("minimum_interval").and_then(Value::as_str))?;
        // Scalar count-only buckets need neither document groups nor a
        // global (bucket, segment, doc) deduplication set. Open each column
        // once. All other cases retain the general collector below.
        // Scalar count-only buckets need neither document groups nor a
        // global (bucket, segment, doc) deduplication set. Open each column
        // once; a segment whose every doc is in the set reads its range from
        // the column's points index and counts each bucket as one range
        // count (as `date_histogram` does), with no per-doc read and no copy
        // of the values. All other cases retain the general collector below.
        // A segment whose query matches were left uncollected
        // (`with_elided_root`) counts only from its points index; anything
        // else reads its matches back from the column and starts over.
        let elided: Vec<bool> = (0..docs.len()).map(|s| self.is_elided(docs, s)).collect();
        let any_elided = elided.iter().any(|&e| e);
        let restart = || self.auto_date_histogram(name, o, sub, &self.collect_elided(&f, docs));
        let mut scalar = None;
        if sub.is_none_or(|m| m.is_empty()) && missing.is_none() && super::DOC_WEIGHTS.with(|w| w.borrow().is_none()) {
            let mut cols = Vec::new();
            let mut ok = true;
            for (s, seg_docs) in docs.iter().enumerate() {
                if seg_docs.is_empty() && !elided[s] { continue; }
                // Explicitly establish uniqueness rather than relying on
                // every aggregation parent to provide a sorted doc set.
                if seg_docs.windows(2).any(|w| w[0].0 >= w[1].0) { ok = false; break; }
                match self.num_col(&f, s) {
                    Some(col) if matches!(&*col, super::Col::I64(_)) => cols.push((s, col)),
                    _ => { ok = false; break; }
                }
            }
            if ok { scalar = Some(cols); }
        }
        if any_elided && scalar.is_none() {
            return restart();
        }
        if let Some(cols) = scalar {
            // The value range whose docs (with a value) are exactly segment
            // `s`'s doc set, for counting from the points index: every value
            // for a whole segment; the query's required range on this field
            // for the query's own matches (Rally metricbeat filters to one
            // day), once the index's count in it equals the doc set's size
            // (the matches satisfy the range, so equal sizes mean the same
            // docs; no deletions, which the index still holds).
            let is_root = std::ptr::eq(docs as *const Docs as *const u8, self.root_docs as *const u8);
            let cover = |s: usize, c: &rustic_core::codec::fastfields::ColumnReader<'_, i64>| -> Option<(i64, i64)> {
                let n = docs[s].len();
                if !c.has_points() { return None; }
                if !elided[s] && n == c.num_docs() as usize && n == self.segs[s].leaf.num_docs() as usize { return Some((i64::MIN, i64::MAX)); }
                let (field, lo, hi) = (*self.query_ranges.get(s)?)?;
                if !is_root || f.per_seg.get(s)?.as_ref()?.core_id != Some(field) || self.segs[s].leaf.deleted_bits().is_some() || !c.counts_from_points() {
                    return None;
                }
                // An uncollected segment's matches are the range's docs.
                (elided[s] || c.count_range(lo, hi) == n as u64).then_some((lo, hi))
            };
            if cols.iter().any(|(s, col)| elided[*s] && !matches!(&**col, super::Col::I64(c) if cover(*s, c).is_some())) {
                return restart();
            }
            let mut range: Option<(i64, i64)> = None;
            let mut widen = |x: i64| range = Some(match range { None => (x, x), Some((a, b)) => (a.min(x), b.max(x)) });
            for (s, col) in &cols {
                let super::Col::I64(c) = &**col else { continue };
                if let Some((qlo, qhi)) = cover(*s, c) {
                    if let Some((a, b)) = c.points_min_max_in(qlo, qhi) { widen(a); widen(b); }
                } else {
                    for &(d, _) in &docs[*s] { if let Some(x) = c.get(d) { widen(x); } }
                }
            }
            if let Some((lo, hi)) = range {
                let choice = super::auto_date::choose(lo, hi, target, min_unit, 0, tz);
                let nb = (choice.bucket_of(hi) + 1) as usize;
                if nb > self.env.max_buckets { return Err(EsError::too_many_buckets(self.env.max_buckets)); }
                // Preparing the selected intervals avoids timezone/date arithmetic
                // once per document. Verify the endpoints with the original
                // collector, retaining it for irregular DST/rounding layouts.
                let keys: Vec<i64> = (0..=nb).map(|b| choice.key_of(b as i64)).collect();
                let verified = keys[0] <= lo && keys[nb] > hi && keys.windows(2).enumerate().all(|(i, pair)| {
                    pair[0] < pair[1] && choice.bucket_of(pair[0]) == i as i64 && choice.bucket_of(pair[1] - 1) == i as i64
                });
                // Every bucket one range count per covered segment; they must
                // cover every value (any doc they miss sends a collected
                // segment per doc, an uncollected one back to the start).
                let mut per_seg: Vec<Option<Vec<u64>>> = Vec::with_capacity(cols.len());
                for (s, col) in &cols {
                    let super::Col::I64(c) = &**col else { per_seg.push(None); continue };
                    let mut per = None;
                    if let (true, Some((qlo, qhi))) = (verified, cover(*s, c)) {
                        let p: Vec<u64> = keys.windows(2).map(|w| c.count_range(w[0].max(qlo), (w[1] - 1).min(qhi))).collect();
                        let whole = (qlo, qhi) == (i64::MIN, i64::MAX);
                        let with_value = if elided[*s] { c.count_range(qlo, qhi) } else if whole && c.is_sparse() { c.points_len() as u64 } else { docs[*s].len() as u64 };
                        if p.iter().sum::<u64>() == with_value { per = Some(p); }
                    }
                    if per.is_none() && elided[*s] {
                        return restart();
                    }
                    per_seg.push(per);
                }
                self.add_buckets(nb)?;
                let mut counts = vec![0u64; nb];
                let prepared = verified.then(|| super::PreparedDateBuckets::from_verified_keys(keys.clone())).flatten();
                for ((s, col), per) in cols.iter().zip(per_seg) {
                    let super::Col::I64(c) = &**col else { continue };
                    if let Some(per) = per {
                        for (b, k) in per.into_iter().enumerate() { counts[b] += k; }
                        continue;
                    }
                    match &prepared {
                        Some(p) => for &(d, _) in &docs[*s] { if let Some(x) = c.get(d) { counts[p.bucket(x)] += 1; } },
                        None => for &(d, _) in &docs[*s] { if let Some(x) = c.get(d) { counts[choice.bucket_of(x) as usize] += 1; } },
                    }
                }
                let buckets: Vec<Value> = counts.into_iter().enumerate().map(|(b, count)| {
                    let key = choice.key_of(b as i64);
                    json!({"key_as_string":fmt.format(key, tz),"key":key,"doc_count":count})
                }).collect();
                return Ok(json!({"buckets":buckets,"interval":choice.interval()}));
            }
        }
        let mut vals: Vec<(i64, usize, DocId, f32)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, sc) in v {
                for k in self.values_or(&f, s, d, &missing) {
                    let x = match k {
                        Key::Long(x) => x,
                        k => match k.as_f64() {
                            Some(x) => x as i64,
                            None => continue,
                        },
                    };
                    vals.push((x, s, d, sc));
                }
            }
        }
        let (lo, hi) = match (vals.iter().map(|v| v.0).min(), vals.iter().map(|v| v.0).max()) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Ok(json!({"buckets": [], "interval": format!("1{}", if min_unit == CalUnit::Second { "s" } else { unit_suffix(min_unit) })}));
            }
        };
        let choice = super::auto_date::choose(lo, hi, target, min_unit, 0, tz);
        let last_bucket = choice.bucket_of(hi);
        let key_of = |b: i64| choice.key_of(b);
        let nb = (last_bucket + 1) as usize;
        if nb > self.env.max_buckets {
            return Err(EsError::too_many_buckets(self.env.max_buckets));
        }
        let mut groups: Vec<Docs> = vec![vec![Vec::new(); docs.len()]; nb];
        let mut seen: HashSet<(usize, usize, DocId)> = HashSet::new();
        for (x, s, d, sc) in vals {
            let b = choice.bucket_of(x) as usize;
            if seen.insert((b, s, d)) {
                groups[b][s].push((d, sc));
            }
        }
        for g in groups.iter_mut() {
            for v in g.iter_mut() {
                v.sort_by_key(|x| x.0);
            }
        }
        self.add_buckets(nb)?;
        let mut out = Vec::new();
        for (b, d) in groups.into_iter().enumerate() {
            let k = key_of(b as i64);
            let mut m = Map::new();
            m.insert("key_as_string".into(), json!(fmt.format(k, tz)));
            m.insert("key".into(), json!(k));
            m.insert("doc_count".into(), json!(count(&d)));
            m.extend(self.subs(sub, &d)?);
            out.push(m);
        }
        apply_parent_pipelines(sub, &mut out)?;
        Ok(json!({"buckets": out, "interval": choice.interval()}))
    }

    // -----------------------------------------------------------------------
    // variable_width_histogram

    fn variable_width_histogram(&self, name: &str, o: &Map<String, Value>, sub: Option<&Map<String, Value>>, docs: &Docs) -> Result<Value> {
        let f = self.field(field_name(o, name, "variable_width_histogram")?)?;
        let k = o.get("buckets").and_then(Value::as_i64).unwrap_or(10);
        if k <= 0 {
            return Err(EsError::illegal_argument(format!("[buckets] must be greater than 0. Found [{k}] in [{name}]")));
        }
        let mut vals: Vec<(f64, usize, DocId, f32)> = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, sc) in v {
                for x in self.values(&f, s, d).iter().filter_map(Key::as_f64) {
                    vals.push((x, s, d, sc));
                }
            }
        }
        vals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut distinct: Vec<f64> = vals.iter().map(|v| v.0).collect();
        distinct.dedup();
        let k = (k as usize).min(distinct.len());
        // 1-D k-means over the sorted values: contiguous clusters, initialised
        // with equal value counts, refined by Lloyd iterations.
        let n = vals.len();
        let mut bounds: Vec<usize> = (0..=k).map(|i| i * n / k.max(1)).collect();
        // Never split equal values across clusters.
        let fix = |b: &mut Vec<usize>| {
            for i in 1..b.len() - 1 {
                while b[i] > b[i - 1] && b[i] < n && vals[b[i]].0 == vals[b[i] - 1].0 {
                    b[i] += 1;
                }
                if b[i] < b[i - 1] {
                    b[i] = b[i - 1];
                }
            }
        };
        if k > 0 {
            fix(&mut bounds);
            for _ in 0..20 {
                let cents: Vec<Option<f64>> = (0..k)
                    .map(|i| {
                        let sl = &vals[bounds[i]..bounds[i + 1]];
                        (!sl.is_empty()).then(|| sl.iter().map(|v| v.0).sum::<f64>() / sl.len() as f64)
                    })
                    .collect();
                let cs: Vec<f64> = cents.into_iter().flatten().collect();
                let mut nb = vec![0usize];
                for w in cs.windows(2) {
                    let mid = (w[0] + w[1]) / 2.0;
                    nb.push(vals.partition_point(|v| v.0 < mid));
                }
                nb.push(n);
                while nb.len() < k + 1 {
                    nb.push(n);
                }
                if nb == bounds {
                    break;
                }
                bounds = nb;
            }
        }
        let mut out = Vec::new();
        let mut nbk = 0;
        for i in 0..k {
            let sl = &vals[bounds[i]..bounds[i + 1]];
            if sl.is_empty() {
                continue;
            }
            nbk += 1;
            let mut d: Docs = vec![Vec::new(); docs.len()];
            let mut seen = HashSet::new();
            for v in sl {
                if seen.insert((v.1, v.2)) {
                    d[v.1].push((v.2, v.3));
                }
            }
            for v in d.iter_mut() {
                v.sort_by_key(|x| x.0);
            }
            let mut m = Map::new();
            m.insert("min".into(), fnum(sl[0].0));
            m.insert("key".into(), fnum(sl.iter().map(|v| v.0).sum::<f64>() / sl.len() as f64));
            m.insert("max".into(), fnum(sl[sl.len() - 1].0));
            m.insert("doc_count".into(), json!(count(&d)));
            m.extend(self.subs(sub, &d)?);
            out.push(m);
        }
        self.add_buckets(nbk)?;
        apply_parent_pipelines(sub, &mut out)?;
        Ok(json!({ "buckets": out }))
    }

    // -----------------------------------------------------------------------
    // boxplot / median_absolute_deviation

    fn quantile_metric(&self, name: &str, t: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let comp = o.get("compression").and_then(Value::as_f64).unwrap_or(if t == "boxplot" { 100.0 } else { 1000.0 });
        if comp < 0.0 {
            return Err(EsError::illegal_argument(format!("[compression] must be greater than or equal to 0. Found [{}] in [{name}]", jfmt(comp))));
        }
        let high_accuracy = match o.get("execution_hint").and_then(Value::as_str) {
            None | Some("default") => false,
            Some("high_accuracy") => true,
            Some(h) => return Err(EsError::illegal_argument(format!("Invalid execution_hint [{h}], valid values are [default, high_accuracy]"))),
        };
        // `source` resolves runtime fields and scripts too.
        let f = self.source(o, name, t)?;
        let missing = self.missing_key(&f, o.get("missing"))?;
        let vals = self.numbers(&f, docs, &missing)?;
        if high_accuracy && t == "boxplot" && !vals.is_empty() && (vals.len() as f64) <= comp {
            // ES's high-accuracy (merging) digest keeps every value as a
            // singleton centroid here: quantiles interpolate between them.
            // Centroids plus stable-sort scratch, before either allocation.
            self.mem.charge((vals.len() as u64).saturating_mul(32), "<agg [boxplot centroids]>")?;
            let mut c: Vec<(f64, u64)> = vals.iter().map(|v| (*v, 1)).collect();
            c.sort_by(|a, b| a.0.total_cmp(&b.0));
            let q = |p: f64| crate::fieldtypes::aggs::weighted_quantile(&c, p).unwrap_or(f64::NAN);
            let (min, max) = (c[0].0, c[c.len() - 1].0);
            let (q1, q2, q3) = (q(0.25), q(0.5), q(0.75));
            let iqr = q3 - q1;
            let lower = vals.iter().copied().filter(|v| *v >= q1 - 1.5 * iqr).fold(f64::INFINITY, f64::min);
            let upper = vals.iter().copied().filter(|v| *v <= q3 + 1.5 * iqr).fold(f64::NEG_INFINITY, f64::max);
            return Ok(json!({"min": fnum(min), "max": fnum(max), "q1": fnum(q1), "q2": fnum(q2), "q3": fnum(q3), "lower": fnum(lower), "upper": fnum(upper)}));
        }
        if t == "median_absolute_deviation" {
            if vals.is_empty() {
                return Ok(json!({"value": null}));
            }
            let mut td = TDigest::new(comp);
            let mut td_bytes = 0;
            for v in &vals {
                self.reserve_sketch(td.working_bytes(), &mut td_bytes)?;
                td.add(*v);
            }
            let med = td.quantile(0.5).unwrap_or(0.0);
            let mut dev = TDigest::new(comp);
            let mut dev_bytes = 0;
            for v in &vals {
                self.reserve_sketch(dev.working_bytes(), &mut dev_bytes)?;
                dev.add((v - med).abs());
            }
            return Ok(json!({"value": opt_num(dev.quantile(0.5))}));
        }
        if vals.is_empty() {
            // `InternalBoxplot` of an empty digest: min/max are the infinities
            // and the quantiles NaN, which JSON renders as strings.
            return Ok(json!({"min": "Infinity", "max": "-Infinity", "q1": "NaN", "q2": "NaN", "q3": "NaN", "lower": "NaN", "upper": "-Infinity"}));
        }
        let mut td = TDigest::new(comp);
        let mut td_bytes = 0;
        for v in &vals {
            self.reserve_sketch(td.working_bytes(), &mut td_bytes)?;
            td.add(*v);
        }
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let q1 = td.quantile(0.25).unwrap_or(min);
        let q2 = td.quantile(0.5).unwrap_or(min);
        let q3 = td.quantile(0.75).unwrap_or(max);
        let iqr = q3 - q1;
        // Whiskers: the most extreme data points within 1.5 IQR.
        let lo_lim = q1 - 1.5 * iqr;
        let hi_lim = q3 + 1.5 * iqr;
        let lower = vals.iter().copied().filter(|v| *v >= lo_lim).fold(f64::INFINITY, f64::min);
        let upper = vals.iter().copied().filter(|v| *v <= hi_lim).fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({"min": fnum(min), "max": fnum(max), "q1": fnum(q1), "q2": fnum(q2), "q3": fnum(q3), "lower": fnum(lower), "upper": fnum(upper)}))
    }

    // -----------------------------------------------------------------------
    // rate

    fn rate(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let mode = o.get("mode").and_then(Value::as_str).unwrap_or("sum");
        if mode != "sum" && mode != "value_count" {
            return Err(EsError::illegal_argument(format!("Unsupported rate mode [{mode}]")));
        }
        let total: f64 = if o.get("field").is_none() && o.get("script").is_none() {
            if o.contains_key("mode") {
                return Err(EsError::illegal_argument(format!("The mode parameter is only supported with field or script [{name}]")));
            }
            // No field: the rate is over the per-document `_doc_count` (default 1),
            // read from `_source` — not the number of documents.
            let mut total = 0.0;
            for (s, v) in docs.iter().enumerate() {
                for &(d, sc) in v {
                    let h = (self.env.fetch_hit)(s, d, sc, &json!({}))?;
                    total += h.get("_source").and_then(|s| s.get("_doc_count")).and_then(Value::as_f64).unwrap_or(1.0);
                }
            }
            total
        } else {
            // `source` resolves runtime fields and scripts too.
            let f = self.source(o, name, "rate")?;
            let missing = self.missing_key(&f, o.get("missing"))?;
            let v = self.numbers(&f, docs, &missing)?;
            if mode == "sum" { v.iter().sum() } else { v.len() as f64 }
        };
        self.rate_with_total(name, o, docs, total)
    }

    /// `rate` of an already summed `total` over the enclosing date bucket.
    pub(super) fn rate_with_total(&self, name: &str, o: &Map<String, Value>, docs: &Docs, total: f64) -> Result<Value> {
        let parent = DH_PARENT.with(|p| p.borrow().last().cloned());
        let Some((pfield, rounding)) = parent else {
            return Err(EsError::illegal_argument(format!(
                "The rate aggregation can only be used inside a date histogram aggregation or composite aggregation with one date histogram value source [{name}]"
            )));
        };
        let unit = match o.get("unit").and_then(Value::as_str) {
            None => None,
            Some(u) => Some(CalUnit::parse(u).ok_or_else(|| EsError::illegal_argument(format!("Unsupported unit {u}")))?),
        };
        let Some(unit) = unit else { return Ok(json!({"value": fnum(total)})) };
        // The bucket this rate is computed for: round any doc's parent value.
        let pf = self.field(&pfield)?;
        let mut key = None;
        'outer: for (s, v) in docs.iter().enumerate() {
            for &(d, _) in v {
                if let Some(Key::Long(x)) = self.values(&pf, s, d).into_iter().next() {
                    key = Some(rounding.round(x));
                    break 'outer;
                }
            }
        }
        let Some(key) = key else { return Ok(json!({"value": fnum(0.0)})) };
        let month_like = |u: CalUnit| match u {
            CalUnit::Month => Some(1.0),
            CalUnit::Quarter => Some(3.0),
            CalUnit::Year => Some(12.0),
            _ => None,
        };
        let pm = rounding.cal.and_then(month_like);
        let um = month_like(unit);
        let value = match (pm, um) {
            (Some(p), Some(u)) => total * u / p,
            _ => {
                if rounding.cal.is_none() && um.is_some() {
                    return Err(EsError::illegal_argument(format!(
                        "Cannot use month-based rate unit [{}] with fixed interval based histogram, only week, day, hour, minute and second are supported for this histogram",
                        unit_name(unit)
                    )));
                }
                let bucket_ms = (rounding.next(key) - key) as f64;
                let unit_ms = match unit {
                    CalUnit::Month | CalUnit::Quarter | CalUnit::Year => {
                        let st = round_down(key, unit, rounding.tz);
                        (add_units(st, unit, 1, rounding.tz) - st) as f64
                    }
                    u => u.approx_millis() as f64,
                };
                // ES divides by the bucket size expressed in the rate unit.
                if bucket_ms <= 0.0 { total } else { total / (bucket_ms / unit_ms) }
            }
        };
        Ok(json!({ "value": fnum(value) }))
    }

    // -----------------------------------------------------------------------
    // string_stats

    fn string_stats(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let f = self.field(field_name(o, name, "string_stats")?)?;
        let missing = self.missing_key(&f, o.get("missing"))?;
        let show = o.get("show_distribution").and_then(Value::as_bool).unwrap_or(false);
        let mut n = 0u64;
        let (mut minl, mut maxl, mut total) = (usize::MAX, 0usize, 0u64);
        let mut chars: BTreeMap<char, u64> = BTreeMap::new();
        for (s, v) in docs.iter().enumerate() {
            for &(d, _) in v {
                for k in self.values_or(&f, s, d, &missing) {
                    let Key::Str(st) = k else { continue };
                    let l = st.chars().count();
                    n += 1;
                    minl = minl.min(l);
                    maxl = maxl.max(l);
                    total += l as u64;
                    for c in st.chars() {
                        *chars.entry(c).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut m = Map::new();
        m.insert("count".into(), json!(n));
        if n == 0 {
            m.insert("min_length".into(), Value::Null);
            m.insert("max_length".into(), Value::Null);
            m.insert("avg_length".into(), Value::Null);
            m.insert("entropy".into(), fnum(0.0));
        } else {
            let tot = total as f64;
            let entropy: f64 = chars.values().map(|c| *c as f64 / tot).map(|p| -p * p.log2()).sum();
            m.insert("min_length".into(), json!(minl));
            m.insert("max_length".into(), json!(maxl));
            m.insert("avg_length".into(), fnum(tot / n as f64));
            m.insert("entropy".into(), fnum(entropy));
        }
        if show {
            let tot = total as f64;
            let mut dist: Vec<(char, u64)> = chars.into_iter().collect();
            dist.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let dm: Map<String, Value> = dist.into_iter().map(|(c, k)| (c.to_string(), fnum(k as f64 / tot))).collect();
            m.insert("distribution".into(), Value::Object(dm));
        }
        Ok(Value::Object(m))
    }

    // -----------------------------------------------------------------------
    // t_test

    fn t_test(&self, name: &str, o: &Map<String, Value>, docs: &Docs) -> Result<Value> {
        let side = |k: &str| -> Result<&Map<String, Value>> { o.get(k).and_then(Value::as_object).ok_or_else(|| EsError::illegal_argument(format!("[{k}] must be set for t_test aggregation [{name}]"))) };
        let (a, b) = (side("a")?, side("b")?);
        let ty = o.get("type").and_then(Value::as_str).unwrap_or("heteroscedastic");
        let tails = o.get("tails").and_then(Value::as_i64).unwrap_or(2);
        if tails != 1 && tails != 2 {
            return Err(EsError::illegal_argument(format!("[tails] must be 1 or 2. Found [{tails}] in [{name}]")));
        }
        let fa = self.field(field_name(a, name, "t_test")?)?;
        let fb = self.field(field_name(b, name, "t_test")?)?;
        // ES (TTestStatsBuilder) keeps Kahan-compensated (count, sum,
        // sumOfSqrs) accumulators, fed in doc order.
        let stats = |side: &Map<String, Value>, f: &FieldRef| -> Result<(u64, f64, f64)> {
            let d = match side.get("filter") {
                Some(q) => self.filter_docs(q, docs)?,
                None => docs.clone(),
            };
            let (mut n, mut sum, mut sq) = (0u64, Kahan::default(), Kahan::default());
            for (s, v) in d.iter().enumerate() {
                for &(doc, _) in v {
                    for x in self.values(f, s, doc).iter().filter_map(Key::as_f64) {
                        n += 1;
                        sum.add(x);
                        sq.add(x * x);
                    }
                }
            }
            Ok((n, sum.value, sq.value))
        };
        let (sa, sb) = match ty {
            "paired" => {
                if a.contains_key("filter") || b.contains_key("filter") {
                    return Err(EsError::illegal_argument("Paired t-test doesn't support filters"));
                }
                let (mut count, mut sum, mut sq) = (0u64, Kahan::default(), Kahan::default());
                for (s, v) in docs.iter().enumerate() {
                    for &(d, _) in v {
                        let xa: Vec<f64> = self.values(&fa, s, d).iter().filter_map(Key::as_f64).collect();
                        let xb: Vec<f64> = self.values(&fb, s, d).iter().filter_map(Key::as_f64).collect();
                        if xa.is_empty() || xb.is_empty() {
                            continue;
                        }
                        if xa.len() > 1 || xb.len() > 1 {
                            return Err(EsError::illegal_argument("Encountered more than one value for a single document. Use a script to combine multiple values per doc into a single value."));
                        }
                        let diff = xa[0] - xb[0];
                        count += 1;
                        sum.add(diff);
                        sq.add(diff * diff);
                    }
                }
                ((count, sum.value, sq.value), (0, 0.0, 0.0))
            }
            "homoscedastic" | "heteroscedastic" => (stats(a, &fa)?, stats(b, &fb)?),
            other => return Err(EsError::illegal_argument(format!("Unknown value [{other}] for [type] in [{name}], expected one of [paired, homoscedastic, heteroscedastic]"))),
        };
        let mut v = t_test_value(ty, tails, sa, sb);
        // A coordinator reduce follows: ship the accumulators (ES `TTestState`).
        if crate::search::aggregation_shard_response() {
            v["__p"] = json!({ "type": ty, "tails": tails, "a": [sa.0, sa.1, sa.2], "b": [sb.0, sb.1, sb.2] });
        }
        Ok(v)
    }
}

/// `t_test` from its (count, sum, sum of squares) accumulators: of the
/// differences for `paired` (`b` unused), of each side otherwise.
pub(super) fn t_test_value(ty: &str, tails: i64, a: (u64, f64, f64), b: (u64, f64, f64)) -> Value {
    let tdist = |t: f64, df: f64| -> Value {
        if t.is_nan() || df.is_nan() || df <= 0.0 {
            return json!({"value": null});
        }
        json!({"value": fnum(super::cmath::t_cdf(-t, df) * tails as f64)})
    };
    if ty == "paired" {
        let (count, sum, sum_of_sqrs) = a;
        if count < 2 {
            return json!({"value": null});
        }
        let n = (count - 1) as f64;
        let c = count as f64;
        let mean_diff = sum / c;
        let variance = (sum_of_sqrs - ((sum * sum) / c)) / c;
        if variance <= 0.0 {
            return if mean_diff == 0.0 { json!({"value": null}) } else { json!({"value": 0.0}) };
        }
        let std_err = variance.sqrt() / n.sqrt();
        return tdist((mean_diff / std_err).abs(), n);
    }
    let ((n1, s1, q1), (n2, s2, q2)) = (a, b);
    if n1 < 2 || n2 < 2 {
        return json!({"value": null});
    }
    let var = |n: u64, s: f64, q: f64| {
        let v = (q - ((s * s) / n as f64)) / (n - 1) as f64;
        if v < 0.0 { 0.0 } else { v }
    };
    let (v1, v2) = (var(n1, s1, q1), var(n2, s2, q2));
    let mean_diff = s1 / n1 as f64 - s2 / n2 as f64;
    let (sd2, df) = if ty == "homoscedastic" {
        let n = (n1 + n2 - 2) as f64;
        let variance = ((n1 - 1) as f64 * v1 + (n2 - 1) as f64 * v2) / n;
        let nn = 1.0 / n1 as f64 + 1.0 / n2 as f64;
        (variance * nn, n)
    } else {
        let (s2an, s2bn) = (v1 / n1 as f64, v2 / n2 as f64);
        let variance = s2an + s2bn;
        (variance, variance * variance / (s2an * s2an / (n1 - 1) as f64 + s2bn * s2bn / (n2 - 1) as f64))
    };
    if df < 0.0 {
        return json!({"value": null});
    }
    tdist((mean_diff / sd2.sqrt()).abs(), df)
}

fn ym(ms: i64, tz: Tz) -> (i64, i64) {
    use chrono::Datelike;
    let l = tz.to_local(ms);
    (l.year() as i64, l.month0() as i64)
}

pub(crate) fn unit_suffix(u: CalUnit) -> &'static str {
    match u {
        CalUnit::Second => "s",
        CalUnit::Minute => "m",
        CalUnit::Hour => "h",
        CalUnit::Day | CalUnit::Week => "d",
        CalUnit::Month | CalUnit::Quarter => "M",
        CalUnit::Year => "y",
    }
}

fn unit_name(u: CalUnit) -> &'static str {
    match u {
        CalUnit::Second => "second",
        CalUnit::Minute => "minute",
        CalUnit::Hour => "hour",
        CalUnit::Day => "day",
        CalUnit::Week => "week",
        CalUnit::Month => "month",
        CalUnit::Quarter => "quarter",
        CalUnit::Year => "year",
    }
}

#[cfg(test)]
#[path = "extra_tests.rs"]
mod extra_tests;

/// ES's `ObjectArrayPriorityQueue` (Lucene's `PriorityQueue` sifting): a
/// 1-based binary heap of at most `max` elements ordered by `less`, so
/// equal elements end up where ES's do.
pub(super) struct LuceneHeap<T, F: Fn(&T, &T) -> bool> {
    heap: Vec<Option<T>>,
    max: usize,
    less: F,
}

impl<T, F: Fn(&T, &T) -> bool> LuceneHeap<T, F> {
    pub(super) fn new(max: usize, less: F) -> Self {
        let mut heap = Vec::with_capacity(max + 1);
        heap.push(None);
        LuceneHeap { heap, max, less }
    }
    fn lt(&self, a: usize, b: usize) -> bool {
        (self.less)(self.heap[a].as_ref().expect("heap slot"), self.heap[b].as_ref().expect("heap slot"))
    }
    fn size(&self) -> usize {
        self.heap.len() - 1
    }
    fn up_heap(&mut self, mut i: usize) {
        let mut j = i >> 1;
        while j > 0 && self.lt(i, j) {
            self.heap.swap(i, j);
            i = j;
            j >>= 1;
        }
    }
    fn down_heap(&mut self, mut i: usize) {
        let size = self.size();
        loop {
            let mut j = i << 1;
            if j > size {
                break;
            }
            if j + 1 <= size && self.lt(j + 1, j) {
                j += 1;
            }
            if self.lt(j, i) {
                self.heap.swap(i, j);
                i = j;
            } else {
                break;
            }
        }
    }
    /// `insertWithOverflow`: add while not full, else replace the top when
    /// it is less than `item` (ES's `ObjectArrayPriorityQueue`: an item equal
    /// to the top is turned away, unlike Lucene's queue).
    pub(super) fn insert_with_overflow(&mut self, item: T) {
        if self.max == 0 {
            return;
        }
        if self.size() < self.max {
            self.heap.push(Some(item));
            let i = self.size();
            self.up_heap(i);
        } else if (self.less)(self.heap[1].as_ref().expect("heap top"), &item) {
            self.heap[1] = Some(item);
            self.down_heap(1);
        }
    }
    /// `pop`: the least element; the last one takes the top and sinks.
    fn pop(&mut self) -> Option<T> {
        if self.size() == 0 {
            return None;
        }
        let last = self.heap.pop().flatten();
        if self.size() == 0 {
            return last;
        }
        let top = std::mem::replace(&mut self.heap[1], last);
        self.down_heap(1);
        top
    }
    /// The elements as ES lists them: `list[i] = pop()` from the last
    /// index down (greatest first).
    pub(super) fn into_list(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.size());
        while let Some(x) = self.pop() {
            out.push(x);
        }
        out.reverse();
        out
    }
}

/// The order ES's `ObjectObjectPagedHashMap<String, _>` iterates keys put
/// in `keys` order into a map created with `capacity`: Java
/// `String.hashCode` slots, linear probing, in-place growth at a 0.6 load
/// factor, iteration by slot. (The reduce of significant terms feeds its
/// buckets to the priority queue in this order.)
pub(crate) fn java_paged_map_order(capacity: usize, keys: &[String]) -> Vec<usize> {
    fn java_hash(s: &str) -> i32 {
        s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
    }
    const LOAD: f32 = 0.6;
    // `1 + (long) (capacity / 0.6f)`, then the next power of two
    // (`max(1, highestOneBit(buckets - 1) << 1)`).
    let raw: i64 = 1 + (capacity as f32 / LOAD) as i64;
    let x = raw - 1;
    let highest = if x <= 0 { 0 } else { 1i64 << (63 - x.leading_zeros() as i64) };
    let buckets: i64 = std::cmp::max(1, highest << 1);
    let mut max_size = (buckets as f32 * LOAD) as i64;
    let mut mask = buckets - 1;
    let mut slots: Vec<Option<usize>> = vec![None; buckets as usize];
    let mut size = 0i64;
    let place = |slots: &mut Vec<Option<usize>>, mask: i64, k: usize| {
        let mut i = (java_hash(&keys[k]) as i64) & mask;
        while slots[i as usize].is_some() {
            i = (i + 1) & mask;
        }
        slots[i as usize] = Some(k);
    };
    for k in 0..keys.len() {
        if size >= max_size {
            // grow(): double, re-add every used slot in place, then the run
            // that wrapped into the new upper half.
            let old = slots.len() as i64;
            let new = old << 1;
            slots.resize(new as usize, None);
            mask = new - 1;
            for i in 0..old {
                if let Some(e) = slots[i as usize].take() {
                    place(&mut slots, mask, e);
                }
            }
            for i in old..new {
                match slots[i as usize].take() {
                    Some(e) => place(&mut slots, mask, e),
                    None => break,
                }
            }
            max_size = (new as f32 * LOAD) as i64;
        }
        place(&mut slots, mask, k);
        size += 1;
    }
    slots.into_iter().flatten().collect()
}
