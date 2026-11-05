//! Aggregation support for the field types of this module: value
//! extraction for the generic aggregations ([`values`]) and the
//! aggregations that need type-specific logic ([`compute`]): `children` /
//! `parent`, `ip_range`, pre-aggregated `histogram` fields, ranges in
//! `histogram` / `date_histogram`, `aggregate_metric_double` metrics.

use crate::aggs::{AggEnv, AggSeg};
use crate::error::{EsError, Result};
use crate::mapping::{EsType, ResolvedField};
use crate::values;
use rustic_core::fieldtypes::{histogram, ip, range as rng};
use rustic_core::query::LeafReader;
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// Per-segment `(doc, score)` lists (as the aggregation framework uses).
pub type Docs = Vec<Vec<(DocId, f32)>>;

/// Aggregation names implemented here (beyond the generic ones).
pub const AGGS: &[&str] = &["children", "parent", "ip_range"];

/// What [`compute`] needs from the aggregation framework.
pub struct Host<'h, 'a> {
    pub segs: &'h [AggSeg<'a>],
    pub env: &'h AggEnv<'a>,
    pub all_docs: &'h (dyn Fn(usize) -> Vec<DocId> + 'h),
    /// Compute the sub-aggregations over a doc set.
    pub sub: &'h (dyn Fn(&Docs) -> Result<Map<String, Value>> + 'h),
}

/// A doc value for the generic aggregations.
pub enum AggValue {
    Str(String),
    Long(i64),
    Dbl(f64),
}

/// Values of `doc` for generic aggregations when the field's behaviour type
/// is not one the framework reads directly.
pub fn values(leaf: &LeafReader<'_>, rf: &ResolvedField, doc: DocId) -> Option<Vec<AggValue>> {
    if let Some(k) = rf.flattened_key() {
        let id = rf.core_id?;
        let prefix = rustic_core::fieldtypes::flattened::key_prefix(k);
        return Some(leaf.keyword_values(id, doc).into_iter().filter_map(|v| v.strip_prefix(&prefix).map(|s| AggValue::Str(s.to_string()))).collect());
    }
    match rf.declared {
        EsType::ConstantKeyword => Some(rf.constant_value.as_ref().map(|v| vec![AggValue::Str(match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })]).unwrap_or_default()),
        EsType::DateNanos => {
            let id = rf.core_id?;
            Some(leaf.i64_values(id, doc).into_iter().map(|n| AggValue::Long(n.div_euclid(1_000_000))).collect())
        }
        EsType::AggregateMetricDouble => {
            let id = super::aggs_default_metric(rf).and_then(|dm| rf.aux_id(&dm))?;
            Some(leaf.f64_values(id, doc).into_iter().map(AggValue::Dbl).collect())
        }
        _ => None,
    }
}


fn field_of<'x>(host: &'x Host, o: &Map<String, Value>) -> Option<std::sync::Arc<ResolvedField>> {
    let name = o.get("field").and_then(Value::as_str)?;
    host.segs.iter().find_map(|s| s.lookup.get(name).cloned())
}

/// ES `UnsupportedAggregationOnDownsampledIndex` (a contract with Kibana).
fn unsupported_downsampled(f: &ResolvedField, t: &str) -> EsError {
    EsError::new(400, "unsupported_aggregation_on_downsampled_index", format!("Field [{}] of type [{}] is not supported for aggregation [{t}]", f.name, f.declared.name()))
}

fn unsupported(f: &ResolvedField, t: &str) -> EsError {
    EsError::illegal_argument(format!("Field [{}] of type [{}] is not supported for aggregation [{t}]", f.name, f.declared.name()))
}

/// Type-specific aggregations; `None` = let the framework handle it.
pub fn compute(name: &str, t: &str, o: &Map<String, Value>, docs: &Docs, host: &Host) -> Result<Option<Value>> {
    match t {
        "children" | "parent" => {
            let d = if t == "children" { super::join::children_docs(o, docs, host)? } else { super::join::parent_docs(o, docs, host)? };
            let mut m = Map::new();
            m.insert("doc_count".into(), json!(d.iter().map(Vec::len).sum::<usize>()));
            m.extend((host.sub)(&d)?);
            return Ok(Some(Value::Object(m)));
        }
        "ip_range" => return ip_range(name, o, docs, host).map(Some),
        // `top_metrics` names its fields under `metrics`.
        "top_metrics" => {
            let ms: Vec<&Value> = match o.get("metrics") {
                Some(Value::Array(a)) => a.iter().collect(),
                Some(v) => vec![v],
                None => vec![],
            };
            for m in ms {
                if let Some(f) = m.as_object().and_then(|m| field_of(host, m)) {
                    if f.declared == EsType::AggregateMetricDouble {
                        return Err(unsupported_downsampled(&f, t));
                    }
                }
            }
            return Ok(None);
        }
        _ => {}
    }
    let Some(f) = field_of(host, o) else { return Ok(None) };
    let decl = f.declared;
    // `exponential_histogram` (possibly mixed with t-digest histograms,
    // aggregate_metric_double or numbers in other indices), and `range`
    // over any pre-aggregated histogram.
    let any_eh = host.segs.iter().any(|s| s.lookup.get(&f.name).is_some_and(|x| x.declared == EsType::ExponentialHistogram));
    if (any_eh || (super::is_histogram(decl) && t == "range")) && !matches!(t, "missing" | "filter" | "filters" | "global" | "top_hits" | "nested" | "reverse_nested" | "sampler") {
        return mixed_histogram(name, t, o, docs, host, &f).map(Some);
    }
    if super::is_histogram(decl) {
        return histogram_field(name, t, o, docs, host, &f).map(Some);
    }
    // Any index mapping the field as aggregate_metric_double (others may hold raw numbers).
    let any_amd = decl == EsType::AggregateMetricDouble || host.segs.iter().any(|s| s.lookup.get(&f.name).is_some_and(|x| x.declared == EsType::AggregateMetricDouble));
    if any_amd && matches!(t, "min" | "max" | "sum" | "value_count" | "avg") {
        return aggregate_metric(t, docs, host, &f).map(Some);
    }
    // Every other values-source aggregation is rejected on an
    // aggregate_metric_double (a downsampled gauge), as ES does.
    if decl == EsType::AggregateMetricDouble && !matches!(t, "missing" | "filter" | "filters" | "global" | "top_hits" | "nested" | "reverse_nested" | "sampler") {
        return Err(unsupported_downsampled(&f, t));
    }
    if let Some(kind) = super::range_kind(decl) {
        return match t {
            "histogram" | "date_histogram" => range_histogram(name, t, o, docs, host, &f, kind).map(Some),
            "missing" | "value_count" | "cardinality" | "filter" | "global" | "top_hits" => Ok(None),
            _ => Err(unsupported(&f, t)),
        };
    }
    match decl {
        EsType::Ip if matches!(t, "min" | "max" | "avg" | "sum" | "stats" | "extended_stats" | "histogram" | "percentiles") => Err(unsupported(&f, t)),
        EsType::RankFeature | EsType::RankFeatures => Err(EsError::illegal_argument(format!("[{}] fields do not support sorting, scripting or aggregating", decl.name()))),
        EsType::SparseVector => Err(EsError::illegal_argument(format!("[sparse_vector] fields do not support sorting, scripting or aggregating"))),
        EsType::Completion | EsType::Percolator | EsType::Binary | EsType::RankVectors | EsType::GeoShape | EsType::Shape | EsType::Point => {
            // Shape doc values back the geo / cartesian aggregations (`aggs/geo_shape.rs`).
            if matches!(t, "missing" | "value_count" | "filter" | "global" | "top_hits")
                || (super::is_shape(decl) && matches!(t, "geo_bounds" | "geo_centroid" | "geotile_grid" | "geohash_grid" | "geohex_grid" | "cartesian_bounds" | "cartesian_centroid"))
            {
                Ok(None)
            } else {
                Err(unsupported(&f, t))
            }
        }
        EsType::Murmur3 if t != "cardinality" && t != "missing" && t != "value_count" => Err(unsupported(&f, t)),
        _ => Ok(None),
    }
}

fn num(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn opt(v: Option<f64>) -> Value {
    v.map(values::java_double).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// ip_range.

fn ip_range(name: &str, o: &Map<String, Value>, docs: &Docs, host: &Host) -> Result<Value> {
    let field = o.get("field").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument(format!("Required [field] for [ip_range] aggregation [{name}]")))?;
    let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(false);
    let ranges = o.get("ranges").and_then(Value::as_array).ok_or_else(|| EsError::illegal_argument(format!("No [ranges] specified for the [{name}] aggregation")))?;
    struct R {
        key: String,
        from: Option<String>,
        to: Option<String>,
        lo: u128,
        hi_excl: Option<u128>,
        mask: Option<String>,
    }
    let mut rs = Vec::new();
    for r in ranges {
        if let Some(mask) = r.get("mask").and_then(Value::as_str) {
            let (lo, hi) = ip::cidr(mask).ok_or_else(|| EsError::illegal_argument(format!("Expected [ip/prefix] but was [{mask}]")))?;
            let hi_excl = ip::to_u128(hi).checked_add(1);
            let from = ip::display(lo);
            let to = hi_excl.map(|h| ip::display(ip::from_u128(h)));
            rs.push(R { key: r.get("key").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| mask.to_string()), from: Some(from), to, lo: ip::to_u128(lo), hi_excl, mask: Some(mask.to_string()) });
            continue;
        }
        let parse = |k: &str| -> Result<Option<u128>> {
            match r.get(k).and_then(Value::as_str) {
                Some(s) => ip::parse(s).map(|b| Some(ip::to_u128(b))).ok_or_else(|| EsError::illegal_argument(format!("'{s}' is not an IP string literal."))),
                None => Ok(None),
            }
        };
        let lo = parse("from")?;
        let hi = parse("to")?;
        let from = lo.map(|x| ip::display(ip::from_u128(x)));
        let to = hi.map(|x| ip::display(ip::from_u128(x)));
        let key = r.get("key").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("{}-{}", from.clone().unwrap_or_else(|| "*".into()), to.clone().unwrap_or_else(|| "*".into())));
        rs.push(R { key, from, to, lo: lo.unwrap_or(0), hi_excl: hi, mask: None });
    }
    let mut buckets: Vec<Docs> = rs.iter().map(|_| vec![Vec::new(); docs.len()]).collect();
    for (s, v) in docs.iter().enumerate() {
        let seg = &host.segs[s];
        let Some(f) = seg.lookup.get(field) else { continue };
        if f.declared != EsType::Ip {
            return Err(EsError::illegal_argument(format!("Field [{field}] of type [{}] is not supported for aggregation [ip_range]", f.declared.name())));
        }
        let Some(id) = f.core_id else { continue };
        for &(d, sc) in v {
            let vals: Vec<u128> = seg.leaf.keyword_values(id, d).iter().filter_map(|t| values::term_to_ip(t)).map(ip::to_u128).collect();
            for (i, r) in rs.iter().enumerate() {
                if vals.iter().any(|x| *x >= r.lo && r.hi_excl.map(|h| *x < h).unwrap_or(true)) {
                    buckets[i][s].push((d, sc));
                }
            }
        }
    }
    let mut out_arr = Vec::new();
    let mut out_obj = Map::new();
    for (r, d) in rs.iter().zip(buckets) {
        let mut m = Map::new();
        m.insert("key".into(), json!(r.key));
        if let Some(mask) = &r.mask {
            let _ = mask;
        }
        if let Some(f) = &r.from {
            m.insert("from".into(), json!(f));
        }
        if let Some(t) = &r.to {
            m.insert("to".into(), json!(t));
        }
        m.insert("doc_count".into(), json!(d.iter().map(Vec::len).sum::<usize>()));
        m.extend((host.sub)(&d)?);
        if keyed {
            let k = m.remove("key").and_then(|k| k.as_str().map(str::to_string)).unwrap_or_default();
            out_obj.insert(k, Value::Object(m));
        } else {
            out_arr.push(Value::Object(m));
        }
    }
    Ok(if keyed { json!({ "buckets": out_obj }) } else { json!({ "buckets": out_arr }) })
}

// ---------------------------------------------------------------------------
// Pre-aggregated histograms.

fn collect_hist(docs: &Docs, host: &Host, f: &ResolvedField) -> Vec<(f64, u64)> {
    let mut all: Vec<(f64, u64)> = Vec::new();
    for (s, v) in docs.iter().enumerate() {
        let seg = &host.segs[s];
        let Some(rf) = seg.lookup.get(&f.name) else { continue };
        let Some(id) = rf.core_id else { continue };
        for &(d, _) in v {
            for enc in seg.leaf.keyword_values(id, d) {
                all.extend(histogram::decode(&enc));
            }
        }
    }
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all
}

/// Weighted quantile over sorted centroids (t-digest interpolation).
pub fn weighted_quantile(c: &[(f64, u64)], q: f64) -> Option<f64> {
    let total: f64 = c.iter().map(|x| x.1 as f64).sum();
    if c.is_empty() || total == 0.0 {
        return None;
    }
    let (min, max) = (c[0].0, c[c.len() - 1].0);
    if c.len() == 1 {
        return Some(c[0].0);
    }
    let target = q.clamp(0.0, 1.0) * total;
    let mut cum = 0.0;
    let mut prev_center: Option<(f64, f64)> = None;
    for &(m, w) in c {
        let center = cum + w as f64 / 2.0;
        if target < center {
            return Some(match prev_center {
                None => {
                    if w <= 1 {
                        m
                    } else {
                        min + (m - min) * (target / center).clamp(0.0, 1.0)
                    }
                }
                Some((pc, pm)) => pm + (m - pm) * (target - pc) / (center - pc),
            });
        }
        prev_center = Some((center, m));
        cum += w as f64;
    }
    let (pc, pm) = prev_center.unwrap();
    if total - pc <= 0.0 {
        return Some(max);
    }
    Some(pm + (max - pm) * ((target - pc) / (total - pc)).clamp(0.0, 1.0))
}

fn weighted_rank(c: &[(f64, u64)], v: f64) -> Option<f64> {
    let total: f64 = c.iter().map(|x| x.1 as f64).sum();
    if c.is_empty() || total == 0.0 {
        return None;
    }
    let below: f64 = c.iter().filter(|x| x.0 < v).map(|x| x.1 as f64).sum();
    let equal: f64 = c.iter().filter(|x| x.0 == v).map(|x| x.1 as f64).sum();
    Some(((below + equal / 2.0) / total * 100.0).clamp(0.0, 100.0))
}

fn pct_key(p: f64) -> String {
    let s = format!("{p}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

fn histogram_field(name: &str, t: &str, o: &Map<String, Value>, docs: &Docs, host: &Host, f: &ResolvedField) -> Result<Value> {
    let c = collect_hist(docs, host, f);
    let count: u64 = c.iter().map(|x| x.1).sum();
    let sum: f64 = c.iter().map(|x| x.0 * x.1 as f64).sum();
    let keyed = o.get("keyed").and_then(Value::as_bool).unwrap_or(true);
    Ok(match t {
        "value_count" => json!({ "value": count }),
        "sum" => json!({ "value": values::java_double(sum) }),
        "avg" => json!({ "value": opt((count > 0).then(|| sum / count as f64)) }),
        "min" => json!({ "value": opt(c.first().map(|x| x.0)) }),
        "max" => json!({ "value": opt(c.last().map(|x| x.0)) }),
        "percentiles" | "percentile_ranks" => {
            let list: Vec<f64> = if t == "percentiles" {
                o.get("percents").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_else(|| vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
            } else {
                o.get("values").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).ok_or_else(|| EsError::illegal_argument(format!("[values] must not be null: [{name}]")))?
            };
            let vals: Vec<(String, Option<f64>)> = list
                .iter()
                .map(|p| (pct_key(*p), if t == "percentiles" { weighted_quantile(&c, p / 100.0) } else { weighted_rank(&c, *p) }))
                .collect();
            if keyed {
                let mut m = Map::new();
                for (k, v) in vals {
                    m.insert(k, opt(v));
                }
                json!({ "values": m })
            } else {
                let arr: Vec<Value> = vals.into_iter().map(|(k, v)| json!({"key": k.parse::<f64>().unwrap_or(0.0), "value": opt(v)})).collect();
                json!({ "values": arr })
            }
        }
        "boxplot" => {
            let q = |p: f64| weighted_quantile(&c, p);
            let (q1, q2, q3) = (q(0.25), q(0.5), q(0.75));
            let (mn, mx) = (c.first().map(|x| x.0), c.last().map(|x| x.0));
            let (lower, upper) = match (q1, q3, mn, mx) {
                (Some(a), Some(b), Some(lo), Some(hi)) => {
                    let iqr = b - a;
                    (Some(c.iter().map(|x| x.0).find(|x| *x >= a - 1.5 * iqr).unwrap_or(lo)), Some(c.iter().rev().map(|x| x.0).find(|x| *x <= b + 1.5 * iqr).unwrap_or(hi)))
                }
                _ => (None, None),
            };
            json!({ "min": opt(mn), "max": opt(mx), "q1": opt(q1), "q2": opt(q2), "q3": opt(q3), "lower": opt(lower), "upper": opt(upper) })
        }
        "histogram" => {
            let interval = num(o.get("interval")).filter(|x| *x > 0.0).ok_or_else(|| EsError::illegal_argument(format!("[interval] must be >0 for histogram aggregation [{name}]")))?;
            let offset = num(o.get("offset")).unwrap_or(0.0);
            let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(0);
            let mut b: BTreeMap<i64, u64> = BTreeMap::new();
            for (v, n) in &c {
                let k = ((v - offset) / interval).floor() as i64;
                *b.entry(k).or_default() += n;
            }
            if min_doc_count == 0 {
                if let (Some(lo), Some(hi)) = (b.keys().next().copied(), b.keys().last().copied()) {
                    for k in lo..=hi {
                        b.entry(k).or_default();
                    }
                }
            }
            let buckets: Vec<Value> = b
                .into_iter()
                .filter(|(_, n)| *n >= min_doc_count)
                .map(|(k, n)| json!({"key": values::java_double(k as f64 * interval + offset), "doc_count": n}))
                .collect();
            json!({ "buckets": buckets })
        }
        "missing" | "filter" | "global" | "top_hits" => return Err(EsError::internal("unreachable")),
        _ => return Err(unsupported(f, t)),
    })
}

/// The values of a field across indices where it may be an
/// `exponential_histogram`, a t-digest `histogram`, an
/// `aggregate_metric_double` or a plain number.
#[derive(Default)]
struct Mixed {
    ehs: Vec<super::exphist::ExpHist>,
    /// T-digest centroids and plain numbers (`(value, count)`).
    pts: Vec<(f64, u64)>,
    /// `aggregate_metric_double` docs: (min, max, sum, value_count).
    amd: Vec<(Option<f64>, Option<f64>, Option<f64>, Option<f64>)>,
}

fn collect_mixed(docs: &Docs, host: &Host, f: &ResolvedField) -> Mixed {
    let mut m = Mixed::default();
    for (s, v) in docs.iter().enumerate() {
        let seg = &host.segs[s];
        let Some(rf) = seg.lookup.get(&f.name) else { continue };
        match rf.declared {
            EsType::ExponentialHistogram => {
                let Some(id) = rf.core_id else { continue };
                for &(d, _) in v {
                    for enc in seg.leaf.keyword_values(id, d) {
                        if let Some(h) = super::exphist::decode(&enc) {
                            m.ehs.push(h);
                        }
                    }
                }
            }
            EsType::Histogram | EsType::TDigest => {
                let Some(id) = rf.core_id else { continue };
                for &(d, _) in v {
                    for enc in seg.leaf.keyword_values(id, d) {
                        m.pts.extend(histogram::decode(&enc));
                    }
                }
            }
            EsType::AggregateMetricDouble => {
                let get = |metric: &str, d: DocId| rf.aux_id(metric).and_then(|id| seg.leaf.f64_values(id, d).first().copied());
                for &(d, _) in v {
                    m.amd.push((get("min", d), get("max", d), get("sum", d), get("value_count", d)));
                }
            }
            _ => {
                let Some(id) = rf.core_id else { continue };
                for &(d, _) in v {
                    let raw: Vec<f64> = if rf.ty.is_integer() || rf.ty.is_date() || rf.ty == EsType::Boolean {
                        seg.leaf.i64_values(id, d).into_iter().map(|x| x as f64).collect()
                    } else if rf.ty.is_float() {
                        seg.leaf.f64_values(id, d)
                    } else {
                        Vec::new()
                    };
                    m.pts.extend(raw.into_iter().map(|x| (x, 1)));
                }
            }
        }
    }
    m
}

/// Java `Double.toString`-style text of a range bound.
fn bound_text(v: f64) -> String {
    match values::java_double(v) {
        Value::Number(n) => {
            let s = n.to_string();
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{s}.0")
            }
        }
        other => other.as_str().unwrap_or("").to_string(),
    }
}

fn mixed_histogram(name: &str, t: &str, o: &Map<String, Value>, docs: &Docs, host: &Host, f: &ResolvedField) -> Result<Value> {
    use super::exphist;
    let m = collect_mixed(docs, host, f);
    // The weighted points histogram / range aggregations bucket.
    let points = || -> Vec<(f64, u64)> {
        let mut p: Vec<(f64, u64)> = m.ehs.iter().flat_map(|h| h.points()).collect();
        p.extend(m.pts.iter().copied());
        p
    };
    let count = || -> u64 { m.ehs.iter().map(|h| h.value_count()).sum::<u64>() + m.pts.iter().map(|p| p.1).sum::<u64>() + m.amd.iter().map(|a| a.3.unwrap_or(0.0) as u64).sum::<u64>() };
    let sum = || -> f64 {
        let mut k = exphist::KahanSum::default();
        for h in &m.ehs {
            k.add(h.sum);
        }
        for &(v, c) in &m.pts {
            k.add(v * c as f64);
        }
        for a in &m.amd {
            k.add(a.2.unwrap_or(0.0));
        }
        k.value()
    };
    // The union state percentiles read: the merged exponential histograms,
    // plus the t-digest centroids converted (`HistogramUnionState`).
    let union = || -> exphist::ExpHist {
        let mut mg = exphist::Merger::new();
        for h in &m.ehs {
            mg.add(h);
        }
        if !m.pts.is_empty() {
            let mut c = m.pts.clone();
            c.sort_by(|a, b| a.0.total_cmp(&b.0));
            mg.add(&exphist::from_centroids(&c));
        }
        mg.get()
    };
    let nan_null = |v: f64| if v.is_nan() { Value::Null } else { values::java_double(v) };
    let keyed = o.get("keyed").and_then(Value::as_bool);
    Ok(match t {
        "value_count" => json!({ "value": count() }),
        "sum" => json!({ "value": values::java_double(sum()) }),
        "avg" => {
            let n = count();
            json!({ "value": opt((n > 0).then(|| sum() / n as f64)) })
        }
        "min" | "max" => {
            let is_min = t == "min";
            let mut best: Option<f64> = None;
            let mut take = |x: f64| {
                if x.is_nan() {
                    return;
                }
                best = Some(match best {
                    None => x,
                    Some(b) => {
                        if is_min {
                            b.min(x)
                        } else {
                            b.max(x)
                        }
                    }
                });
            };
            for h in &m.ehs {
                take(if is_min { h.min } else { h.max });
            }
            for &(v, _) in &m.pts {
                take(v);
            }
            for a in &m.amd {
                if let Some(x) = if is_min { a.0 } else { a.1 } {
                    take(x);
                }
            }
            json!({ "value": opt(best) })
        }
        "percentiles" | "percentile_ranks" => {
            if o.get("hdr").is_some() {
                return Err(EsError::illegal_argument(format!("Percentiles algorithm hdr must not be used with exponential_histogram field")));
            }
            let u = union();
            let list: Vec<f64> = if t == "percentiles" {
                o.get("percents").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_else(|| vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
            } else {
                o.get("values").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).ok_or_else(|| EsError::illegal_argument(format!("[values] must not be null: [{name}]")))?
            };
            let vals: Vec<(String, Value)> = list
                .iter()
                .map(|p| (pct_key(*p), nan_null(if t == "percentiles" { exphist::quantile(&u, p / 100.0) } else { exphist::percentile_rank(&u, *p) })))
                .collect();
            if keyed.unwrap_or(true) {
                json!({ "values": vals.into_iter().collect::<Map<String, Value>>() })
            } else {
                let arr: Vec<Value> = vals.into_iter().map(|(k, v)| json!({"key": k.parse::<f64>().unwrap_or(0.0), "value": v})).collect();
                json!({ "values": arr })
            }
        }
        "boxplot" => {
            let u = union();
            let mut mg = exphist::Merger::new();
            for h in &m.ehs {
                mg.add(h);
            }
            let eh = mg.get();
            let tmin = m.pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let tmax = m.pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let mn = (if eh.min.is_nan() { f64::INFINITY } else { eh.min }).min(tmin);
            let mx = (if eh.max.is_nan() { f64::NEG_INFINITY } else { eh.max }).max(tmax);
            let (q1, q2, q3) = (exphist::quantile(&u, 0.25), exphist::quantile(&u, 0.5), exphist::quantile(&u, 0.75));
            let centroids: Vec<f64> = u.points().into_iter().map(|p| p.0).collect();
            let iqr = q3 - q1;
            let lower = centroids.iter().copied().find(|c| *c > q1 - 1.5 * iqr).unwrap_or(f64::NAN);
            let upper = match centroids.iter().position(|c| *c > q3 + 1.5 * iqr) {
                Some(0) => f64::NAN,
                Some(i) => centroids[i - 1],
                None => mx,
            };
            let d = |v: f64| if v.is_finite() { values::java_double(v) } else if v.is_nan() { json!("NaN") } else if v > 0.0 { json!("Infinity") } else { json!("-Infinity") };
            json!({ "min": d(mn), "max": d(mx), "q1": d(q1), "q2": d(q2), "q3": d(q3), "lower": d(lower), "upper": d(upper) })
        }
        "histogram" => {
            if o.get("aggs").or_else(|| o.get("aggregations")).is_some() {
                return Err(EsError::illegal_argument("Histogram aggregation on histogram fields does not support sub-aggregations"));
            }
            let interval = num(o.get("interval")).filter(|x| *x > 0.0).ok_or_else(|| EsError::illegal_argument(format!("[interval] must be >0 for histogram aggregation [{name}]")))?;
            let offset = num(o.get("offset")).unwrap_or(0.0);
            let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(0);
            let mut b: BTreeMap<i64, u64> = BTreeMap::new();
            for (v, n) in points() {
                if v.is_nan() {
                    continue;
                }
                let k = ((v - offset) / interval).floor() as i64;
                *b.entry(k).or_default() += n;
            }
            if min_doc_count == 0 {
                if let (Some(lo), Some(hi)) = (b.keys().next().copied(), b.keys().last().copied()) {
                    for k in lo..=hi {
                        b.entry(k).or_default();
                    }
                }
            }
            let buckets: Vec<Value> = b
                .into_iter()
                .filter(|(_, n)| *n >= min_doc_count)
                .map(|(k, n)| json!({"key": values::java_double(k as f64 * interval + offset), "doc_count": n}))
                .collect();
            json!({ "buckets": buckets })
        }
        "range" => {
            if o.get("aggs").or_else(|| o.get("aggregations")).is_some() {
                return Err(EsError::illegal_argument(format!("Range aggregation on {} fields does not support sub-aggregations", f.declared.name())));
            }
            let ranges = o.get("ranges").and_then(Value::as_array).cloned().unwrap_or_default();
            let pts = points();
            let mut buckets = Vec::new();
            let mut keyed_map = Map::new();
            for r in &ranges {
                let from = num(r.get("from"));
                let to = num(r.get("to"));
                let lo = from.unwrap_or(f64::NEG_INFINITY);
                let hi = to.unwrap_or(f64::INFINITY);
                let n: u64 = pts.iter().filter(|(v, _)| *v >= lo && *v < hi).map(|p| p.1).sum();
                let key = r.get("key").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| {
                    format!("{}-{}", from.map(bound_text).unwrap_or_else(|| "*".into()), to.map(bound_text).unwrap_or_else(|| "*".into()))
                });
                let mut b = Map::new();
                if !keyed.unwrap_or(false) {
                    b.insert("key".into(), json!(key));
                }
                if let Some(x) = from {
                    b.insert("from".into(), values::java_double(x));
                }
                if let Some(x) = to {
                    b.insert("to".into(), values::java_double(x));
                }
                b.insert("doc_count".into(), json!(n));
                if keyed.unwrap_or(false) {
                    keyed_map.insert(key, Value::Object(b));
                } else {
                    buckets.push(Value::Object(b));
                }
            }
            if keyed.unwrap_or(false) {
                json!({ "buckets": keyed_map })
            } else {
                json!({ "buckets": buckets })
            }
        }
        _ => return Err(unsupported_downsampled(f, t)),
    })
}

fn aggregate_metric(t: &str, docs: &Docs, host: &Host, f: &ResolvedField) -> Result<Value> {
    let read = |metric: &str| -> Vec<f64> {
        let mut out = Vec::new();
        for (s, v) in docs.iter().enumerate() {
            let seg = &host.segs[s];
            let Some(rf) = seg.lookup.get(&f.name) else { continue };
            if rf.declared != EsType::AggregateMetricDouble {
                // A plain numeric field in another index: each value is its own
                // min / max / sum, and counts once.
                let Some(id) = rf.core_id else { continue };
                for &(d, _) in v {
                    let raw: Vec<f64> = if rf.ty.is_integer() || rf.ty.is_date() || rf.ty == EsType::Boolean {
                        seg.leaf.i64_values(id, d).into_iter().map(|x| x as f64).collect()
                    } else if rf.ty.is_float() {
                        seg.leaf.f64_values(id, d)
                    } else {
                        Vec::new()
                    };
                    if metric == "value_count" {
                        out.extend(raw.iter().map(|_| 1.0));
                    } else {
                        out.extend(raw);
                    }
                }
                continue;
            }
            let Some(id) = rf.aux_id(metric) else { continue };
            for &(d, _) in v {
                out.extend(seg.leaf.f64_values(id, d));
            }
        }
        out
    };
    Ok(match t {
        "min" => {
            let v = read("min");
            json!({ "value": opt(v.iter().copied().reduce(f64::min)) })
        }
        "max" => {
            let v = read("max");
            json!({ "value": opt(v.iter().copied().reduce(f64::max)) })
        }
        "sum" => json!({ "value": values::java_double(read("sum").iter().sum()) }),
        "value_count" => json!({ "value": read("value_count").iter().sum::<f64>() as u64 }),
        _ => {
            let s: f64 = read("sum").iter().sum();
            let n: f64 = read("value_count").iter().sum();
            json!({ "value": opt((n > 0.0).then(|| s / n)) })
        }
    })
}

// ---------------------------------------------------------------------------
// histogram / date_histogram over range fields.

fn range_histogram(name: &str, t: &str, o: &Map<String, Value>, docs: &Docs, host: &Host, f: &ResolvedField, kind: rng::RangeKind) -> Result<Value> {
    use rng::RangeKind as K;
    let (interval, offset, date) = if t == "histogram" {
        let iv = num(o.get("interval")).filter(|x| *x > 0.0).ok_or_else(|| EsError::illegal_argument(format!("[interval] must be >0 for histogram aggregation [{name}]")))?;
        (iv, num(o.get("offset")).unwrap_or(0.0), false)
    } else {
        let spec = o.get("fixed_interval").or_else(|| o.get("calendar_interval")).or_else(|| o.get("interval")).and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument(format!("Required one of fields [fixed_interval, calendar_interval], but none were specified.")))?;
        let ms = interval_millis(spec).ok_or_else(|| EsError::illegal_argument(format!("The supplied interval [{spec}] could not be parsed as a calendar interval.")))?;
        (ms as f64, 0.0, true)
    };
    if kind == K::Ip {
        return Err(unsupported(f, t));
    }
    let to_num = |k: u128| -> f64 {
        match kind {
            K::Float | K::Double => rng::f64_of_key(k),
            _ => rng::i64_of_key(k) as f64,
        }
    };
    let min_doc_count = o.get("min_doc_count").and_then(Value::as_u64).unwrap_or(if date { 0 } else { 0 });
    let mut groups: BTreeMap<i64, Docs> = BTreeMap::new();
    for (s, v) in docs.iter().enumerate() {
        let seg = &host.segs[s];
        let Some(rf) = seg.lookup.get(&f.name) else { continue };
        let Some(id) = rf.core_id else { continue };
        for &(d, sc) in v {
            let mut ks: Vec<i64> = Vec::new();
            for enc in seg.leaf.keyword_values(id, d) {
                let Some((lo, hi)) = rng::decode(&enc) else { continue };
                let (a, b) = (to_num(lo), to_num(hi));
                if !a.is_finite() || !b.is_finite() {
                    continue;
                }
                let ka = ((a - offset) / interval).floor() as i64;
                let kb = ((b - offset) / interval).floor() as i64;
                if kb - ka > host.env.max_buckets as i64 {
                    return Err(EsError::too_many_buckets(host.env.max_buckets));
                }
                ks.extend(ka..=kb);
            }
            ks.sort_unstable();
            ks.dedup();
            for k in ks {
                groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()])[s].push((d, sc));
            }
        }
    }
    if min_doc_count == 0 {
        if let (Some(lo), Some(hi)) = (groups.keys().next().copied(), groups.keys().last().copied()) {
            for k in lo..=hi {
                groups.entry(k).or_insert_with(|| vec![Vec::new(); docs.len()]);
            }
        }
    }
    let fmt = crate::date::DateFormatter::new(o.get("format").and_then(Value::as_str).unwrap_or(crate::date::DEFAULT_FORMAT)).unwrap_or_default();
    let mut buckets = Vec::new();
    for (k, d) in groups {
        let n: usize = d.iter().map(Vec::len).sum();
        if (n as u64) < min_doc_count {
            continue;
        }
        let key = k as f64 * interval + offset;
        let mut m = Map::new();
        if date {
            m.insert("key_as_string".into(), json!(fmt.format(key as i64, crate::date::Tz::UTC)));
            m.insert("key".into(), json!(key as i64));
        } else {
            m.insert("key".into(), values::java_double(key));
        }
        m.insert("doc_count".into(), json!(n));
        m.extend((host.sub)(&d)?);
        buckets.push(Value::Object(m));
    }
    Ok(json!({ "buckets": buckets }))
}

fn interval_millis(s: &str) -> Option<i64> {
    let unit_ms = |u: &str| -> Option<i64> {
        Some(match u {
            "ms" => 1,
            "s" | "second" => 1000,
            "m" | "minute" => 60_000,
            "h" | "hour" => 3_600_000,
            "d" | "day" => 86_400_000,
            "w" | "week" => 7 * 86_400_000,
            _ => return None,
        })
    };
    if let Some(u) = unit_ms(s) {
        return Some(u);
    }
    let idx = s.find(|c: char| !c.is_ascii_digit())?;
    let n: i64 = s[..idx].parse().ok()?;
    Some(n * unit_ms(&s[idx..])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles() {
        let c = vec![(1.0, 1), (2.0, 1), (3.0, 1), (4.0, 1)];
        assert_eq!(weighted_quantile(&c, 0.0), Some(1.0));
        assert_eq!(weighted_quantile(&c, 1.0), Some(4.0));
        assert_eq!(weighted_quantile(&c, 0.5), Some(2.5));
        assert_eq!(weighted_rank(&c, 2.0), Some(37.5));
        assert_eq!(interval_millis("1d"), Some(86_400_000));
        assert_eq!(interval_millis("day"), Some(86_400_000));
    }
}
