//! ML sibling pipeline aggregations: `bucket_correlation`
//! (`count_correlation`), `bucket_count_ks_test` and `change_point`
//! (ES `ChangePointDetector`: spike/dip KDE tests, F-tests for trend /
//! step / trend changes, KS distribution change).

use super::super::{buckets_of, resolve_path};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

fn num(v: f64) -> Value {
    if v.is_nan() {
        json!("NaN")
    } else if v.is_infinite() {
        json!(if v > 0.0 { "Infinity" } else { "-Infinity" })
    } else {
        crate::values::java_double(v)
    }
}

/// A failure of the coordinating reduce phase (ES renders it as a
/// `search_phase_execution_exception` of phase `fetch` without root cause).
pub(super) fn reduce_failure(cause: EsError) -> EsError {
    EsError::new(cause.status, "search_phase_execution_exception", "").with("phase", "fetch").with("grouped", true).caused_by(cause)
}

fn path_of(body: &Value) -> Result<String> {
    body.get("buckets_path").and_then(Value::as_str).map(str::to_string).ok_or_else(|| EsError::illegal_argument("Required [buckets_path]"))
}

/// Parse-time checks (ES builder constructors).
pub(super) fn validate(name: &str, t: &str, body: &Value) -> Result<()> {
    match t {
        "bucket_correlation" => {
            let f = body.get("function").ok_or_else(|| EsError::illegal_argument("Required [function]"))?;
            let cc = f.get("count_correlation").ok_or_else(|| EsError::illegal_argument(format!("[{name}] unknown correlation function")))?;
            let ind = cc.get("indicator").ok_or_else(|| EsError::illegal_argument("Required [indicator]"))?;
            if ind.get("expectations").and_then(Value::as_array).is_none() {
                return Err(EsError::illegal_argument("Required [expectations]"));
            }
            if ind.get("doc_count").and_then(Value::as_i64).is_none() {
                return Err(EsError::illegal_argument("Required [doc_count]"));
            }
        }
        "bucket_count_ks_test" => {
            if let Some(a) = body.get("alternative") {
                let a = a.as_array().cloned().unwrap_or_default();
                if a.is_empty() {
                    return Err(EsError::new(400, "x_content_parse_exception", "Failed to build [bucket_count_ks_test] after last required field arrived")
                        .caused_by(EsError::illegal_argument(format!("[alternative] must not be empty for aggregation [{name}]"))));
                }
                for x in a {
                    let s = x.as_str().unwrap_or_default();
                    if !matches!(s, "less" | "greater" | "two_sided") {
                        return Err(EsError::illegal_argument(format!("unknown alternative [{s}]")));
                    }
                }
            }
            if let Some(s) = body.get("sampling_method").and_then(Value::as_str) {
                if !matches!(s, "upper_tail" | "lower_tail" | "uniform") {
                    return Err(EsError::illegal_argument(format!("unknown sampling_method [{s}]")));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// The referenced multi-bucket aggregation and the per-bucket path.
fn target<'r>(path: &str, results: &'r HashMap<String, Value>) -> Result<(&'r Value, String)> {
    let (agg, rest) = match path.split_once('>') {
        Some((a, r)) => (a, r.to_string()),
        None => (path, "_count".to_string()),
    };
    let v = results.get(agg).ok_or_else(|| EsError::action_validation(&[format!("No aggregation found for path [{path}]").as_str()]))?;
    if v.get("buckets").is_none() {
        return Err(EsError::illegal_argument(format!("The first aggregation in buckets_path must be a multi-bucket aggregation for aggregation [{path}]")));
    }
    Ok((v, rest))
}

pub(super) fn sibling(name: &str, t: &str, body: &Value, results: &HashMap<String, Value>) -> Result<Value> {
    validate(name, t, body)?;
    let path = path_of(body)?;
    let (agg, rest) = target(&path, results)?;
    let buckets = buckets_of(agg);
    match t {
        "bucket_correlation" => {
            let xs: Vec<f64> = buckets.iter().map(|b| resolve_path(b, &rest).unwrap_or(0.0)).collect();
            let ind = &body["function"]["count_correlation"]["indicator"];
            let exp: Vec<f64> = ind["expectations"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            let fractions: Option<Vec<f64>> = ind.get("fractions").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect());
            let doc_count = ind["doc_count"].as_f64().unwrap_or(0.0);
            if exp.len() != xs.len() {
                return Err(reduce_failure(EsError::illegal_argument(format!(
                    "value lengths do not match; indicator.expectations [{}] and number of buckets [{}]. Unable to calculate correlation",
                    exp.len(),
                    xs.len()
                ))));
            }
            Ok(json!({ "value": num(count_correlation(&exp, fractions.as_deref(), doc_count, &xs)) }))
        }
        "bucket_count_ks_test" => {
            let counts: Vec<f64> = buckets.iter().map(|b| resolve_path(b, &rest).unwrap_or(0.0)).collect();
            let fractions: Vec<f64> = match body.get("fractions").and_then(Value::as_array) {
                Some(a) => a.iter().filter_map(Value::as_f64).collect(),
                None => vec![1.0 / counts.len().max(1) as f64; counts.len()],
            };
            if fractions.len() != counts.len() {
                return Err(reduce_failure(EsError::illegal_argument(format!(
                    "value lengths do not match; fractions [{}] and number of buckets [{}]. Unable to calculate ks test",
                    fractions.len(),
                    counts.len()
                ))));
            }
            let alts: Vec<String> = match body.get("alternative").and_then(Value::as_array) {
                Some(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                None => vec!["less".into(), "greater".into(), "two_sided".into()],
            };
            let sampling = body.get("sampling_method").and_then(Value::as_str).unwrap_or("upper_tail");
            let mut fx = vec![0.0];
            fx.extend(&counts);
            let mut fy = vec![0.0];
            fy.extend(&fractions);
            let res = ks_test(&fy, &fx, sampling);
            let mut m = Map::new();
            for a in ["less", "greater", "two_sided"] {
                if alts.iter().any(|x| x == a) {
                    m.insert(a.into(), num(res[a]));
                }
            }
            Ok(Value::Object(m))
        }
        "change_point" => {
            // ES `MlAggsHelper.extractDoubleBucketedValues`: gap policy
            // skip, and the last (possibly partial) bucket is left out.
            if !path.contains('>') && !buckets.is_empty() {
                return Err(reduce_failure(EsError::illegal_argument(format!(
                    "buckets_path must reference either a number value or a single value numeric metric aggregation, got: [Bucket] at aggregation [{path}]"
                ))));
            }
            let mut vals = Vec::new();
            let mut idx = Vec::new();
            for (i, b) in buckets.iter().enumerate() {
                if i + 1 >= buckets.len() {
                    break;
                }
                if let Some(v) = resolve_path(b, &rest).filter(|v| v.is_finite()) {
                    vals.push(v);
                    idx.push(i);
                }
            }
            let mut ct = change_type(&vals);
            // Report the change point as an index into all buckets.
            if let Some(cp) = ct.change_point {
                let bi = idx.get(cp).copied().unwrap_or(cp);
                if let Some(o) = ct.body.as_object_mut() {
                    if o.contains_key("change_point") {
                        o.insert("change_point".into(), json!(bi));
                    }
                }
            }
            let mut out = Map::new();
            if let Some(cp) = ct.change_point {
                if let Some(b) = idx.get(cp).and_then(|i| buckets.get(*i)) {
                    let mut bm = Map::new();
                    let key = b.get("key_as_string").cloned().or_else(|| b.get("key").cloned()).unwrap_or(Value::Null);
                    bm.insert("key".into(), key);
                    bm.insert("doc_count".into(), b.get("doc_count").cloned().unwrap_or(json!(0)));
                    for (k, v) in b {
                        if !matches!(k.as_str(), "key" | "key_as_string" | "doc_count" | "from" | "to" | "from_as_string" | "to_as_string") {
                            bm.insert(k.clone(), v.clone());
                        }
                    }
                    out.insert("bucket".into(), Value::Object(bm));
                }
            }
            out.insert("type".into(), json!({ ct.name: ct.body }));
            Ok(Value::Object(out))
        }
        _ => Err(EsError::parsing_at(format!("Unknown aggregation type [{t}]"), 1, 1)),
    }
}

// ---------------------------------------------------------------------------
// count_correlation: Pearson correlation between the indicator (doc in the
// bucket) and the expectation of the range bucket a doc falls in.

/// ES `CountCorrelationFunction.execute` (same operation order).
fn count_correlation(exp: &[f64], fractions: Option<&[f64]>, doc_count: f64, ys: &[f64]) -> f64 {
    let n = exp.len();
    if n == 0 {
        return f64::NAN;
    }
    let (x_mean, x_var) = match fractions {
        None => {
            // MovingFunctions.unweightedAvg / stdDev.
            let avg = exp.iter().sum::<f64>() / n as f64;
            let var = exp.iter().map(|v| (v - avg).powf(2.0)).sum::<f64>();
            let sd = (var / n as f64).sqrt();
            (avg, sd.powf(2.0))
        }
        Some(f) => {
            let mut m = 0.0;
            for i in 0..n {
                m += exp[i] * f[i];
            }
            let mut v = 0.0;
            for i in 0..n {
                v += (exp[i] - m).powf(2.0) * f[i];
            }
            (m, v)
        }
    };
    if x_mean.is_nan() || x_var.is_nan() {
        return f64::NAN;
    }
    let y_mean = ys.iter().sum::<f64>() / doc_count;
    let ym = y_mean;
    let y_var = (1.0 - y_mean) * ym * ym + y_mean * (1.0 - ym) * (1.0 - ym);
    let mut cov = 0.0;
    for i in 0..n {
        let w = fractions.map(|f| f[i]).unwrap_or(1.0 / n as f64);
        let (e, y) = (exp[i], ys[i]);
        cov = cov - (doc_count * w - y) * (e - x_mean) * ym + y * (e - x_mean) * (1.0 - ym);
    }
    cov /= doc_count;
    if x_var * y_var == 0.0 {
        f64::NAN
    } else {
        cov / (x_var * y_var).sqrt()
    }
}

// ---------------------------------------------------------------------------
// bucket_count_ks_test

fn cdf_points(sampling: &str) -> Vec<f64> {
    let mut upper = Vec::with_capacity(55);
    for i in 1..=10 {
        for j in 0..i {
            upper.push(0.1 * (i - 1) as f64 + 0.05 / i as f64 + 0.1 * j as f64 / i as f64);
        }
    }
    match sampling {
        "lower_tail" => upper.iter().map(|x| 1.0 - x).collect(),
        "uniform" => {
            let interval = 1.0 / 56.0;
            let mut v = vec![interval];
            for i in 1..55 {
                v.push(v[i - 1] + interval);
            }
            v
        }
        _ => upper,
    }
}

fn bisect_right(xs: &[f64], x: f64) -> usize {
    xs.partition_point(|v| *v <= x).max(1)
}

fn interpolate(xs: &[f64], fx: &[f64], x: f64) -> f64 {
    let i = bisect_right(xs, x).min(xs.len() - 1);
    ((x - xs[i - 1]) * fx[i] + (xs[i] - x) * fx[i - 1]) / (xs[i] - xs[i - 1])
}

fn cumulative_normalized(v: &[f64]) -> Option<Vec<f64>> {
    let mut c = Vec::with_capacity(v.len());
    let mut s = 0.0;
    for x in v {
        s += x;
        c.push(s);
    }
    let last = *c.last()?;
    if last <= 0.0 {
        return None;
    }
    Some(c.iter().map(|x| x / last).collect())
}

/// Two-sample sided KS statistics `(less, greater, two_sided)`.
fn sided_statistics(xa: &[f64], xb: &[f64]) -> (f64, f64, f64) {
    let (n, m) = (xa.len() as f64, xb.len() as f64);
    let (mut ia, mut ib) = (0usize, 0usize);
    let (mut less, mut greater, mut two) = (0.0f64, 0.0f64, 0.0f64);
    while ia < xa.len() || ib < xb.len() {
        let v = match (xa.get(ia), xb.get(ib)) {
            (Some(a), Some(b)) => a.min(*b),
            (Some(a), None) => *a,
            (None, Some(b)) => *b,
            _ => break,
        };
        while ia < xa.len() && xa[ia] <= v {
            ia += 1;
        }
        while ib < xb.len() && xb[ib] <= v {
            ib += 1;
        }
        let a = ia as f64 / n;
        let b = ib as f64 / m;
        less = less.max(b - a);
        greater = greater.max(a - b);
        two = two.max((a - b).abs());
    }
    (less.clamp(0.0, 1.0), greater.max(0.0), two)
}

fn one_sided_p(stat: f64, n: usize, m: usize) -> f64 {
    let (n, m) = (n as f64, m as f64);
    let z_constant = n * m / (n + m);
    let continuity = (n + 2.0 * m) / (n * m * (n + m)).sqrt();
    let z = z_constant.sqrt() * stat;
    (-2.0 * z * z - 2.0 * z * continuity / 3.0).exp().clamp(0.0, 1.0)
}

fn ks_once(fx: &[f64], fy: &[f64], mono: &[f64], points: &[f64]) -> HashMap<&'static str, f64> {
    let mut x: Vec<f64> = points.iter().map(|f| interpolate(fx, mono, *f)).collect();
    let mut y: Vec<f64> = points.iter().map(|f| interpolate(fy, mono, *f)).collect();
    x.sort_by(|a, b| a.total_cmp(b));
    y.sort_by(|a, b| a.total_cmp(b));
    let (less, greater, two) = sided_statistics(&x, &y);
    let mut r = HashMap::new();
    r.insert("less", one_sided_p(less, x.len(), y.len()));
    r.insert("greater", one_sided_p(greater, x.len(), y.len()));
    r.insert("two_sided", ks_exact_p(two, x.len(), y.len(), false));
    r
}

/// ES `BucketCountKSTestAggregator.ksTest`.
fn ks_test(fractions: &[f64], values: &[f64], sampling: &str) -> HashMap<&'static str, f64> {
    let nan = || ["less", "greater", "two_sided"].into_iter().map(|a| (a, f64::NAN)).collect::<HashMap<_, _>>();
    let points = cdf_points(sampling);
    let total: f64 = values.iter().sum();
    let n_samples = (total.min(i32::MAX as f64) as usize).min(points.len());
    let (Some(fx), Some(fy)) = (cumulative_normalized(values), cumulative_normalized(fractions)) else { return nan() };
    if n_samples < 23 {
        return nan();
    }
    let mono: Vec<f64> = (1..=fy.len()).map(|i| i as f64).collect();
    if n_samples >= points.len() {
        return ks_once(&fx, &fy, &mono, &points);
    }
    // ES averages 20 random subsamples (geometric mean of the p-values);
    // a fixed seed keeps the answer stable here.
    let mut acc: HashMap<&'static str, f64> = HashMap::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..20 {
        let mut idx: Vec<usize> = (0..points.len()).collect();
        for i in 0..n_samples {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = i + (state % (points.len() - i) as u64) as usize;
            idx.swap(i, j);
        }
        let sample: Vec<f64> = idx[..n_samples].iter().map(|i| points[*i]).collect();
        for (k, v) in ks_once(&fx, &fy, &mono, &sample) {
            *acc.entry(k).or_insert(0.0) += if v == 0.0 { 0.0 } else { v.ln() };
        }
    }
    acc.into_iter().map(|(k, v)| (k, (v / 20.0).exp().clamp(0.0, 1.0))).collect()
}

/// Apache commons-math `KolmogorovSmirnovTest.exactP(d, n, m, strict)`:
/// P(D >= d) (or > d when strict) for the two-sample statistic, by
/// lattice-path counting.
fn ks_exact_p(d: f64, n: usize, m: usize, strict: bool) -> f64 {
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let tol = 1e-12;
    let nm = (n * m) as f64;
    let upper = ((d - tol) * nm).ceil() as i64;
    let lower = ((d + tol) * nm).floor() as i64;
    let l = if strict && lower == upper { upper + 1 } else { upper };
    // Paths from (0,0) to (n,m) that stay strictly inside |i*m - j*n| < l.
    let (n_i, m_i) = (n as i64, m as i64);
    let mut row = vec![0.0f64; m + 1];
    for i in 0..=n {
        for j in 0..=m {
            let ok = ((i as i64) * m_i - (j as i64) * n_i).abs() < l;
            let v = if !ok {
                0.0
            } else if i == 0 && j == 0 {
                1.0
            } else {
                let up = if i > 0 { row[j] } else { 0.0 };
                let left = if j > 0 { row[j - 1] } else { 0.0 };
                up + left
            };
            row[j] = v;
        }
        // Normalise per row to avoid overflow: divide by the binomial ratio later.
    }
    let paths = row[m];
    let total = binomial(n + m, n);
    (1.0 - paths / total).clamp(0.0, 1.0)
}

fn binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    let mut r = 1.0f64;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r.round().max(r)
}

/// Apache `KolmogorovSmirnovTest.kolmogorovSmirnovStatistic(x, y)`.
fn ks_statistic(x: &[f64], y: &[f64]) -> f64 {
    let mut a = x.to_vec();
    let mut b = y.to_vec();
    a.sort_by(|p, q| p.total_cmp(q));
    b.sort_by(|p, q| p.total_cmp(q));
    sided_statistics(&a, &b).2
}

// ---------------------------------------------------------------------------
// change_point

const P_VALUE_THRESHOLD: f64 = 0.01;
const MINIMUM_BUCKETS: usize = 10;
const MAX_SAMPLE_SIZE_KS: usize = 500;
const MAX_CANDIDATE_CPS: usize = 1000;

struct Change {
    name: &'static str,
    body: Value,
    p_value: f64,
    change_point: Option<usize>,
}

impl Change {
    fn indeterminable(reason: String) -> Change {
        Change { name: "indeterminable", body: json!({ "reason": reason }), p_value: 1.0, change_point: None }
    }
    fn stationary() -> Change {
        Change { name: "stationary", body: json!({}), p_value: 1.0, change_point: None }
    }
    fn with_cp(name: &'static str, p: f64, cp: usize) -> Change {
        Change { name, body: json!({ "p_value": num(p), "change_point": cp }), p_value: p, change_point: Some(cp) }
    }
}

fn change_type(values: &[f64]) -> Change {
    if values.len() < 2 * MINIMUM_BUCKETS + 2 {
        return Change::indeterminable(format!("not enough buckets to calculate change_point. Requires at least [22]; found [{}]", values.len()));
    }
    let spike = spike_or_dip(values, P_VALUE_THRESHOLD);
    let change = ChangeDetector { values }.detect(P_VALUE_THRESHOLD);
    if spike.p_value < change.p_value {
        spike
    } else {
        change
    }
}

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if !a.is_finite() {
        return f64::NAN;
    }
    f64::from_bits(a.to_bits() + 1) - a
}

fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

fn sample_variance(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if v.len() < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n;
    // Apache `Variance` (bias corrected, with the compensating term).
    let mut acc = 0.0;
    let mut acc2 = 0.0;
    for x in v {
        let d = x - mean;
        acc += d * d;
        acc2 += d;
    }
    (acc - acc2 * acc2 / n) / (n - 1.0)
}

struct Kde {
    values: Vec<f64>,
    bandwidth: f64,
}

struct ValueAndMagnitude {
    value: f64,
    magnitude: f64,
}

impl ValueAndMagnitude {
    fn more_significant(&self, o: &ValueAndMagnitude) -> bool {
        match self.value.partial_cmp(&o.value) {
            Some(std::cmp::Ordering::Equal) | None => self.magnitude > o.magnitude,
            Some(c) => c == std::cmp::Ordering::Less,
        }
    }
    fn p_value(&self, n: usize) -> f64 {
        if self.value > 1e-10 {
            1.0 - (1.0 - self.value).powi(n as i32)
        } else {
            n as f64 * self.value
        }
    }
}

impl Kde {
    fn new(values: Vec<f64>, smoothing: f64) -> Kde {
        let bandwidth = smoothing * (values.len() as f64).powf(-0.2) * sample_variance(&values).sqrt();
        Kde { values, bandwidth }
    }
    fn lower_bound(&self, x: f64) -> usize {
        self.values.partition_point(|v| *v < x)
    }
    fn evaluate(&self, sf: bool, x: f64) -> ValueAndMagnitude {
        let n = self.values.len();
        if self.bandwidth == 0.0 || n == 0 {
            return ValueAndMagnitude { value: 1.0, magnitude: 0.0 };
        }
        let a = self.lower_bound(x - 3.0 * self.bandwidth).min(n - 1);
        let b = self.lower_bound(x + 3.0 * self.bandwidth).max(a + 1);
        let mut kdfx = if sf { (n - b.min(n)) as f64 } else { a as f64 };
        let mut diff = f64::MAX;
        for i in a..b.min(n) {
            let c = self.values[i];
            let dev = x - c;
            let k = if dev.abs() > 40.0 * self.bandwidth {
                if sf {
                    if dev > 0.0 { 0.0 } else { 1.0 }
                } else if dev > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else if sf {
                0.5 * erfc(dev / (self.bandwidth * std::f64::consts::SQRT_2))
            } else {
                0.5 * erfc(-dev / (self.bandwidth * std::f64::consts::SQRT_2))
            };
            kdfx += k;
            diff = diff.min((c - x).abs());
        }
        ValueAndMagnitude { value: kdfx / n as f64, magnitude: diff }
    }
}

fn find_spike_or_dip(values: &[f64], extent: usize, negate: bool) -> (usize, usize, usize) {
    let n = values.len();
    let extent = extent.min(n - 1);
    let val = |i: usize| if negate { -values[i] } else { values[i] };
    let mut argmax = 0;
    let mut max = val(0);
    for i in 1..n {
        if val(i) > max {
            argmax = i;
            max = val(i);
        }
    }
    let sum = |s: usize, e: usize| (s..e).map(val).sum::<f64>();
    let mut max_start = (argmax as i64 + 1 - extent as i64).max(0) as usize;
    let max_end = (max_start + extent).min(n);
    let mut max_sum = sum(max_start, max_end);
    for start in (max_start + 1)..=argmax {
        if start + extent > n {
            break;
        }
        let s = sum(start, start + extent);
        if s > max_sum {
            max_start = start;
            max_sum = s;
        }
    }
    (argmax, max_start, max_start + extent)
}

fn spike_or_dip(values: &[f64], threshold: f64) -> Change {
    let n = values.len();
    if n < 4 {
        return Change::indeterminable(format!("not enough buckets to check for dip or spike. Requires at least [3]; found [{n}]"));
    }
    let spike_length = ((0.1 * n as f64 + 0.5) as usize).max(2) - 1;
    let dip = find_spike_or_dip(values, spike_length, true);
    let spike = find_spike_or_dip(values, spike_length, false);
    let keep = |excl: (usize, usize, usize), other: usize| -> Vec<f64> {
        let mut v: Vec<f64> = values.iter().enumerate().filter(|(i, _)| !((*i >= excl.1 && *i < excl.2) || *i == other)).map(|(_, x)| *x).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v
    };
    let dip_kde = Kde::new(keep(dip, spike.0), 1.36);
    let spike_kde = Kde::new(keep(spike, dip.0), 1.36);
    let dip_tail = dip_kde.evaluate(false, values[dip.0]);
    let spike_tail = spike_kde.evaluate(true, values[spike.0]);
    let dip_p = dip_tail.p_value(n);
    let spike_p = spike_tail.p_value(n);
    if dip_p < threshold && spike_p < threshold {
        if dip_tail.more_significant(&spike_tail) {
            return Change::with_cp("dip", dip_p, dip.0);
        }
        return Change::with_cp("spike", spike_p, spike.0);
    }
    if dip_p < threshold {
        return Change::with_cp("dip", dip_p, dip.0);
    }
    if spike_p < threshold {
        return Change::with_cp("spike", spike_p, spike.0);
    }
    Change::stationary()
}

#[derive(Clone, Copy, Default)]
struct RunningStats {
    sum_sq: f64,
    sum: f64,
    count: f64,
}

impl RunningStats {
    fn add(&mut self, v: f64, w: f64) {
        self.sum_sq += v * v * w;
        self.count += w;
        self.sum += v * w;
    }
    fn remove(&mut self, v: f64, w: f64) {
        self.sum_sq = (self.sum_sq - v * v * w).max(0.0);
        self.count = (self.count - w).max(0.0);
        self.sum -= v * w;
    }
    fn add_range(&mut self, v: &[f64], w: &[f64], s: usize, e: usize) {
        for i in s..e {
            self.add(v[i], w[i]);
        }
    }
    fn remove_range(&mut self, v: &[f64], w: &[f64], s: usize, e: usize) {
        for i in s..e {
            self.remove(v[i], w[i]);
        }
    }
    fn mean(&self) -> f64 {
        self.sum / self.count
    }
    fn variance(&self) -> f64 {
        ((self.sum_sq - self.sum * self.sum / self.count) / self.count).max(0.0)
    }
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// ES `LeastSquaresOnlineRegression` (degree 1).
#[derive(Clone)]
struct OnlineRegression {
    count: f64,
    stats: [f64; 6],
}

impl OnlineRegression {
    const N: usize = 2;
    fn new() -> Self {
        OnlineRegression { count: 0.0, stats: [0.0; 6] }
    }
    fn adj(x: f64, y: f64) -> [f64; 6] {
        [1.0, x, x * x, y, x * y, y * y]
    }
    fn add(&mut self, x: f64, y: f64, w: f64) {
        let d = Self::adj(x, y);
        self.count += w;
        let alpha = w / self.count;
        let beta = 1.0 - alpha;
        for i in 0..6 {
            self.stats[i] = self.stats[i] * beta + alpha * d[i];
        }
    }
    fn remove(&mut self, x: f64, y: f64, w: f64) {
        let d = Self::adj(x, y);
        self.count = (self.count - w).max(0.0);
        if self.count == 0.0 {
            self.stats = [0.0; 6];
            return;
        }
        let alpha = w / self.count;
        let beta = 1.0 + alpha;
        for i in 0..6 {
            self.stats[i] = self.stats[i] * beta - alpha * d[i];
        }
    }
    fn r_squared(&self) -> f64 {
        if self.count <= 0.0 {
            return 0.0;
        }
        let s = &self.stats;
        let n = Self::N;
        let var = s[3 * n - 1] - s[2 * n - 1] * s[2 * n - 1];
        // Normal equations X r = y, X[i][j] = E[x^(i+j)].
        let (a, b, d) = (s[0], s[1], s[2]);
        let tr = a + d;
        let det = a * d - b * b;
        let disc = ((tr * tr) / 4.0 - det).max(0.0).sqrt();
        let (l1, l2) = ((tr / 2.0 + disc).abs(), (tr / 2.0 - disc).abs());
        let (smax, smin) = (l1.max(l2), l1.min(l2));
        if smax > 1e15 * smin || det == 0.0 {
            return 0.0;
        }
        let (y0, y1) = (s[3], s[4]);
        let r0 = (d * y0 - b * y1) / det;
        let r1 = (a * y1 - b * y0) / det;
        let yr = y0 * r0 + y1 * r1;
        let zr = s[0] * r0 + s[1] * r1;
        let t = s[2 * n - 1] - zr;
        let residual = (s[3 * n - 1] - yr) - t * t;
        (1.0 - residual / var).clamp(0.0, 1.0)
    }
}

fn independent_trials(p: f64, n: usize) -> f64 {
    if p > 1e-10 {
        1.0 - (1.0 - p).powi(n as i32)
    } else {
        n as f64 * p
    }
}

fn f_test_nested(n: f64, v_null: f64, p_null: f64, v_alt: f64, p_alt: f64) -> f64 {
    if v_alt == v_null {
        return 1.0;
    }
    if v_alt == 0.0 {
        return 0.0;
    }
    let f = (v_null - v_alt) / (p_alt - p_null) * (n - p_alt) / v_alt;
    let (d1, d2) = (p_alt - p_null, n - p_alt);
    let sf = if f <= 0.0 {
        1.0
    } else if !f.is_finite() {
        0.0
    } else {
        super::super::extra::inc_beta(0.5 * d2, 0.5 * d1, d2 / (d2 + d1 * f))
    };
    (2.0 * sf).min(1.0)
}

#[derive(Clone, Copy)]
struct DataStats {
    n_values: f64,
    var: f64,
    mean: f64,
    n_candidates: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Stationary,
    NonStationary,
    StepChange,
    TrendChange,
    DistributionChange,
}

#[derive(Clone, Copy)]
struct TestStats {
    kind: Kind,
    p_value: f64,
    var: f64,
    n_params: f64,
    change_point: Option<usize>,
    data: DataStats,
}

impl TestStats {
    fn accept(&self, threshold: f64) -> bool {
        self.p_value < threshold && self.r_squared() >= 0.5
    }
    fn r_squared(&self) -> f64 {
        1.0 - self.var / self.data.var
    }
    fn p_value_vs_stationary(&self) -> f64 {
        independent_trials(f_test_nested(self.data.n_values, self.data.var, 1.0, self.var, self.n_params), self.data.n_candidates)
    }
    fn change(&self, slope: f64) -> Change {
        match self.kind {
            Kind::Stationary => Change::stationary(),
            Kind::NonStationary => {
                let p = self.p_value_vs_stationary();
                Change {
                    name: "non_stationary",
                    body: json!({ "p_value": num(p), "r_value": num(self.r_squared()), "trend": if slope < 0.0 { "decreasing" } else { "increasing" } }),
                    p_value: p,
                    change_point: None,
                }
            }
            Kind::StepChange => Change::with_cp("step_change", self.p_value_vs_stationary(), self.change_point.unwrap_or(0)),
            Kind::TrendChange => {
                let p = self.p_value_vs_stationary();
                let cp = self.change_point.unwrap_or(0);
                // ES renders `r_value` from the p-value field.
                Change { name: "trend_change", body: json!({ "p_value": num(p), "r_value": num(p), "change_point": cp }), p_value: p, change_point: Some(cp) }
            }
            Kind::DistributionChange => Change::with_cp("distribution_change", self.p_value, self.change_point.unwrap_or(0)),
        }
    }
}

struct ChangeDetector<'v> {
    values: &'v [f64],
}

impl ChangeDetector<'_> {
    fn detect(&self, min_p: f64) -> Change {
        let n = self.values.len();
        let threshold = min_p * (-0.04 * (n as f64 - 2.0 * (MINIMUM_BUCKETS + 1) as f64)).exp();
        let stats = self.test_for_change(threshold);
        stats.change(slope(self.values))
    }

    fn candidates(&self) -> Vec<usize> {
        let n = self.values.len();
        let min_values = ((0.1 * n as f64 + 0.5) as usize).max(MINIMUM_BUCKETS);
        let range = n - 2 * min_values;
        if range <= MAX_CANDIDATE_CPS {
            (min_values..n - min_values).collect()
        } else {
            let step = (range as f64 / MAX_CANDIDATE_CPS as f64).ceil() as usize;
            (min_values..n - min_values).filter(|i| i % step == 0).collect()
        }
    }

    fn outlier_weights(&self) -> Vec<f64> {
        let n = self.values.len();
        let i = (0.025 * n as f64).ceil() as usize;
        let mut sorted = self.values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let (a, b) = (sorted[i], sorted[n - i - 1]);
        self.values.iter().map(|v| if *v <= b && *v >= a { 1.0 } else { 0.01 }).collect()
    }

    fn test_for_change(&self, threshold: f64) -> TestStats {
        let values = self.values;
        let cands = self.candidates();
        let w = self.outlier_weights();
        let mut rs = RunningStats::default();
        rs.add_range(values, &w, 0, values.len());
        let data = DataStats { n_values: rs.count, mean: rs.mean(), var: rs.variance(), n_candidates: cands.len() };
        let stationary = TestStats { kind: Kind::Stationary, p_value: 1.0, var: data.var, n_params: 1.0, change_point: None, data };
        if data.var < ulp(2.0 * data.n_values * data.mean).sqrt() {
            return stationary;
        }
        let trend_vs_stationary = self.test_trend_vs(&stationary, &w);
        let mut best = stationary;
        let mut discovered: HashSet<usize> = HashSet::new();
        if trend_vs_stationary.accept(threshold) {
            let tc = self.test_trend_change_vs(&trend_vs_stationary, &w, &cands);
            discovered.extend(tc.change_point);
            best = if tc.accept(threshold) { self.test_vs_step_change(&tc, &w, &cands, threshold) } else { trend_vs_stationary };
        } else {
            let step = self.test_step_change_vs(&stationary, &w, &cands);
            discovered.extend(step.change_point);
            if step.accept(threshold) {
                let tc = self.test_trend_change_vs(&step, &w, &cands);
                best = if tc.accept(threshold) { tc } else { step };
            } else {
                let tc = self.test_trend_change_vs(&stationary, &w, &cands);
                if tc.accept(threshold) {
                    best = tc;
                }
            }
        }
        if best.p_value_vs_stationary() > 1e-5 {
            let dist = self.test_distribution_change(&data, &w, &cands, &mut discovered);
            if dist.p_value < threshold.min(0.1 * best.p_value_vs_stationary()) {
                best = dist;
            }
        }
        best
    }

    fn test_trend_vs(&self, h0: &TestStats, w: &[f64]) -> TestStats {
        let mut reg = OnlineRegression::new();
        for (i, v) in self.values.iter().enumerate() {
            reg.add(i as f64, *v, w[i]);
        }
        let v_trend = h0.data.var * (1.0 - reg.r_squared());
        let p = f_test_nested(h0.data.n_values, h0.var, h0.n_params, v_trend, 3.0);
        TestStats { kind: Kind::NonStationary, p_value: p, var: v_trend, n_params: 3.0, change_point: None, data: h0.data }
    }

    fn test_step_change_vs(&self, h0: &TestStats, w: &[f64], cands: &[usize]) -> TestStats {
        let values = self.values;
        let mut v_step = f64::MAX;
        let mut cp = None;
        let mut lower = RunningStats::default();
        let mut upper = RunningStats::default();
        upper.add_range(values, w, cands[0], values.len());
        lower.add_range(values, w, 0, cands[0]);
        let mut last = cands[0];
        for &c in cands {
            lower.add_range(values, w, last, c);
            upper.remove_range(values, w, last, c);
            last = c;
            let v = (lower.count * lower.variance() + upper.count * upper.variance()) / (lower.count + upper.count);
            if v < v_step {
                v_step = v;
                cp = Some(c);
            }
        }
        let p = independent_trials(f_test_nested(h0.data.n_values, h0.var, h0.n_params, v_step, 2.0), cands.len());
        TestStats { kind: Kind::StepChange, p_value: p, var: v_step, n_params: 2.0, change_point: cp, data: h0.data }
    }

    fn test_trend_change_vs(&self, h0: &TestStats, w: &[f64], cands: &[usize]) -> TestStats {
        let values = self.values;
        let first = cands[0];
        let mut v_change = f64::MAX;
        let mut cp = None;
        let mut lower = RunningStats::default();
        let mut upper = RunningStats::default();
        lower.add_range(values, w, 0, first);
        upper.add_range(values, w, first, values.len());
        let mut lreg = OnlineRegression::new();
        let mut ureg = OnlineRegression::new();
        for i in 0..first {
            lreg.add(i as f64, values[i], w[i]);
        }
        for i in first..values.len() {
            ureg.add((i - first) as f64, values[i], w[i]);
        }
        let mut last = first;
        for &c in cands {
            for i in last..c {
                lower.add(values[i], w[i]);
                upper.remove(values[i], w[i]);
                lreg.add(i as f64, values[i], w[i]);
                ureg.remove((i - first) as f64, values[i], w[i]);
            }
            last = c;
            let vl = lower.variance() * (1.0 - lreg.r_squared());
            let vu = upper.variance() * (1.0 - ureg.r_squared());
            let v = (lower.count * vl + upper.count * vu) / (lower.count + upper.count);
            if v < v_change {
                v_change = v;
                cp = Some(c);
            }
        }
        let p = independent_trials(f_test_nested(h0.data.n_values, h0.var, h0.n_params, v_change, 6.0), cands.len());
        TestStats { kind: Kind::TrendChange, p_value: p, var: v_change, n_params: 6.0, change_point: cp, data: h0.data }
    }

    fn test_vs_step_change(&self, tc: &TestStats, w: &[f64], cands: &[usize], threshold: f64) -> TestStats {
        let ds = tc.data;
        let stationary = TestStats { kind: Kind::Stationary, p_value: 1.0, var: ds.var, n_params: 1.0, change_point: None, data: ds };
        let step = self.test_step_change_vs(&stationary, w, cands);
        let p = f_test_nested(ds.n_values, step.var, 2.0, tc.var, 6.0);
        if p < threshold {
            *tc
        } else {
            step
        }
    }

    fn test_distribution_change(&self, data: &DataStats, w: &[f64], cands: &[usize], discovered: &mut HashSet<usize>) -> TestStats {
        let values = self.values;
        let mut max_diff = 0.0;
        let mut cp = None;
        let mut lower = RunningStats::default();
        let mut upper = RunningStats::default();
        upper.add_range(values, w, cands[0], values.len());
        lower.add_range(values, w, 0, cands[0]);
        let mut last = cands[0];
        for &c in cands {
            lower.add_range(values, w, last, c);
            upper.remove_range(values, w, last, c);
            last = c;
            let scale = c.min(values.len() - c) as f64;
            let diff = scale * ((lower.mean() - upper.mean()).abs() + (lower.std() - upper.std()).abs());
            if diff >= max_diff {
                max_diff = diff;
                cp = Some(c);
            }
        }
        discovered.extend(cp);
        // Sample (n <= 500 keeps every value).
        let (sv, cps): (Vec<f64>, Vec<usize>) = if values.len() <= MAX_SAMPLE_SIZE_KS {
            let mut c: Vec<usize> = discovered.iter().copied().collect();
            c.sort();
            (values.to_vec(), c)
        } else {
            let mut choice: Vec<usize> = (0..values.len()).collect();
            let mut state: u64 = 126_832_678;
            for i in 0..MAX_SAMPLE_SIZE_KS {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let j = i + (state % (values.len() - i) as u64) as usize;
                choice.swap(i, j);
            }
            let mut ch = choice[..MAX_SAMPLE_SIZE_KS].to_vec();
            ch.sort();
            let mut c: Vec<usize> = discovered.iter().map(|d| ch.partition_point(|x| x < d)).collect();
            c.sort();
            (ch.iter().map(|i| values[*i]).collect(), c)
        };
        let mut p_value = 1.0;
        for c in cps {
            if c == 0 || c >= sv.len() {
                continue;
            }
            let stat = ks_statistic(&sv[..c], &sv[c..]);
            let p = ks_exact_p(stat, c, sv.len() - c, false);
            if p < p_value {
                cp = Some(c);
                p_value = p;
            }
        }
        let p_value = independent_trials(p_value, sv.len().div_ceil(50));
        TestStats { kind: Kind::DistributionChange, p_value, var: data.var, n_params: 0.0, change_point: cp, data: *data }
    }
}

/// Apache `SimpleRegression` slope over `(i, values[i])`.
fn slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let (mut sx, mut sy, mut sxy, mut sxx) = (0.0, 0.0, 0.0, 0.0);
    for (i, v) in values.iter().enumerate() {
        let x = i as f64;
        sx += x;
        sy += v;
        sxy += x * v;
        sxx += x * x;
    }
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_matches_es() {
        // TV bucket of the ES YAML fixture.
        let v = count_correlation(&[3.0, 4.0, 2.0], None, 9.0, &[1.0, 3.0, 2.0]);
        assert!((v - 0.2886751345948128).abs() < 1e-12, "{v}");
        let v = count_correlation(&[1.0, 5.0, 2.0], Some(&[0.2, 0.5, 0.3]), 20.0, &[0.0, 2.0, 2.0]);
        assert!((v - 0.057639041770423505).abs() < 1e-12, "{v}");
    }

    #[test]
    fn exact_p_small() {
        // n = m = 3, D = 1: only 2 of the 20 paths reach it.
        assert!((ks_exact_p(1.0, 3, 3, false) - 0.1).abs() < 1e-12);
        assert_eq!(ks_exact_p(0.0, 3, 3, false), 1.0);
    }

    #[test]
    fn change_point_trend() {
        let mut v = vec![200.0, 200.0, 200.0, 250.0, 250.0, 380.0, 450.0, 550.0, 600.0];
        v.extend(std::iter::repeat(600.0).take(13));
        let c = change_type(&v);
        assert_eq!(c.name, "trend_change", "{}", c.body);
        assert_eq!(c.change_point, Some(10));
    }
}
