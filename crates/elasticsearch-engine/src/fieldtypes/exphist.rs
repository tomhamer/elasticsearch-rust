//! `exponential_histogram` (ES 9.2+, `x-pack/plugin/analytics` +
//! `libs/exponential-histogram`): OpenTelemetry-style base-2 exponential
//! histograms. Ported from ES 9.5.5 with the same constants and the same
//! floating-point operation order, so bucket boundaries, merges, quantiles
//! and estimates come out bit-for-bit as ES computes them.
//!
//! Bucket `i` at scale `s` covers `(base^i, base^(i+1)]` with
//! `base = 2^(2^-s)`; negative buckets mirror the positive ones. A zero
//! bucket holds the values in `[-threshold, threshold]`.

use crate::error::EsError;
use serde_json::{Map, Value};

pub const MAX_SCALE: i32 = 38;
pub const MIN_SCALE: i32 = -11;
pub const MAX_INDEX: i64 = (1i64 << 62) - 1;
pub const MIN_INDEX: i64 = -MAX_INDEX;
/// `ExponentialHistogramMerger.DEFAULT_MAX_HISTOGRAM_BUCKETS`.
pub const MAX_BUCKETS: usize = 320;
const MAX_INDEX_BITS: usize = 62;

fn ln2() -> f64 {
    2f64.ln()
}

/// Java `Math.scalb(x, n)` for the exponents used here (exact).
fn scalb(x: f64, n: i32) -> f64 {
    x * 2f64.powi(n)
}

/// Java `Math.clamp(v, min, max)` (NaN passes through).
fn clamp(v: f64, min: f64, max: f64) -> f64 {
    if v.is_nan() {
        v
    } else if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

pub fn exp_scaled(index: i64, scale: i32) -> f64 {
    (scalb(ln2(), -scale) * index as f64).exp()
}

pub fn lower_boundary(index: i64, scale: i32) -> f64 {
    exp_scaled(index, scale)
}

pub fn upper_boundary(index: i64, scale: i32) -> f64 {
    exp_scaled(index + 1, scale)
}

/// `ExponentialScaleUtils.getPointOfLeastRelativeError`.
pub fn point_lre(index: i64, scale: i32) -> f64 {
    let base = 2f64.powf(scalb(1.0, -scale));
    if base.is_finite() {
        2.0 / (base + 1.0) * upper_boundary(index, scale)
    } else if index >= 0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// `Base2ExponentialHistogramIndexer.computeIndex`.
pub fn compute_index(value: f64, scale: i32) -> i64 {
    let a = value.abs();
    if scale > 0 {
        let log_base2_e = 1.0 / ln2();
        return (scalb(a.ln() * log_base2_e, scale)).ceil() as i64 - 1;
    }
    let e = map_to_index_scale_zero(a);
    if scale == 0 {
        e
    } else {
        e >> -scale
    }
}

fn map_to_index_scale_zero(a: f64) -> i64 {
    let bits = a.to_bits() as i64;
    let mut raw_exp = (bits & 0x7FF0_0000_0000_0000) >> 52;
    let sig = bits & 0x000F_FFFF_FFFF_FFFF;
    if raw_exp == 0 {
        raw_exp -= ((sig - 1) as u64).leading_zeros() as i64 - 11 - 1;
    }
    let ieee = raw_exp - 1023;
    if sig == 0 {
        ieee - 1
    } else {
        ieee
    }
}

/// `ExponentialScaleUtils.compareExponentiallyScaledValues`: compares the
/// lower boundaries of two buckets exactly.
pub fn compare_scaled(ia: i64, sa: i32, ib: i64, sb: i32) -> i32 {
    if sa > sb {
        return -compare_scaled(ib, sb, ia, sa);
    }
    let shifts = (sb - sa) as u32;
    let down = ib >> shifts;
    match ia.cmp(&down) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => {
            if shifts > 0 && (ib & ((1i64 << shifts) - 1)) > 0 {
                -1
            } else {
                0
            }
        }
    }
}

/// `ZeroBucket`: threshold-based (a user threshold, at the max scale) or
/// index-based (a bucket boundary, after collapsing buckets).
#[derive(Clone, Copy, Debug)]
pub struct ZeroBucket {
    pub index: i64,
    pub scale: i32,
    pub threshold: f64,
    pub index_based: bool,
    pub count: u64,
}

impl ZeroBucket {
    pub fn minimal(count: u64) -> ZeroBucket {
        ZeroBucket { index: MIN_INDEX, scale: MIN_SCALE, threshold: exp_scaled(MIN_INDEX, MIN_SCALE), index_based: true, count }
    }

    pub fn create(threshold: f64, count: u64) -> ZeroBucket {
        if threshold == 0.0 {
            return ZeroBucket::minimal(count);
        }
        ZeroBucket { index: compute_index(threshold, MAX_SCALE) + 1, scale: MAX_SCALE, threshold, index_based: false, count }
    }

    pub fn from_index(index: i64, scale: i32, count: u64) -> ZeroBucket {
        if index == MIN_INDEX && scale == MIN_SCALE {
            return ZeroBucket::minimal(count);
        }
        ZeroBucket { index, scale, threshold: exp_scaled(index, scale), index_based: true, count }
    }

    pub fn with_count(&self, count: u64) -> ZeroBucket {
        ZeroBucket { count, ..*self }
    }

    pub fn compare_threshold(&self, other: &ZeroBucket) -> i32 {
        compare_scaled(self.index, self.scale, other.index, other.scale)
    }

    pub fn merge(&self, other: &ZeroBucket) -> ZeroBucket {
        if other.count == 0 {
            *self
        } else if self.count == 0 {
            *other
        } else {
            let total = self.count + other.count;
            if self.compare_threshold(other) >= 0 {
                self.with_count(total)
            } else {
                other.with_count(total)
            }
        }
    }

    /// `collapseOverlappingBuckets`: folds the buckets (from `*pos` on)
    /// whose lower boundary lies below the threshold into the zero bucket.
    fn collapse(&self, buckets: &[(i64, u64)], pos: &mut usize, scale: i32) -> ZeroBucket {
        let mut collapsed = 0u64;
        let mut highest = 0i64;
        while *pos < buckets.len() && compare_scaled(buckets[*pos].0, scale, self.index, self.scale) < 0 {
            highest = buckets[*pos].0;
            collapsed += buckets[*pos].1;
            *pos += 1;
        }
        if collapsed == 0 {
            return *self;
        }
        let count = self.count + collapsed;
        let upper = highest + 1;
        if compare_scaled(self.index, self.scale, upper, scale) >= 0 {
            self.with_count(count)
        } else {
            ZeroBucket::from_index(upper, scale, count)
        }
    }

    /// `collapseOverlappingBucketsForAll(positive, negative)`.
    fn collapse_all(&self, pos: &[(i64, u64)], neg: &[(i64, u64)], scale: i32) -> (ZeroBucket, usize, usize) {
        let (mut ip, mut ineg) = (0usize, 0usize);
        let mut current = *self;
        loop {
            let previous = current;
            current = current.collapse(pos, &mut ip, scale);
            current = current.collapse(neg, &mut ineg, scale);
            if previous.compare_threshold(&current) == 0 {
                return (current, ip, ineg);
            }
        }
    }
}

/// An exponential histogram (`ExponentialHistogram`). `min` / `max` are
/// NaN when unknown; buckets are sorted by index with non-zero counts.
#[derive(Clone, Debug)]
pub struct ExpHist {
    pub scale: i32,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub zero: ZeroBucket,
    pub neg: Vec<(i64, u64)>,
    pub pos: Vec<(i64, u64)>,
}

impl ExpHist {
    pub fn empty() -> ExpHist {
        ExpHist { scale: MAX_SCALE, sum: 0.0, min: f64::NAN, max: f64::NAN, zero: ZeroBucket::minimal(0), neg: Vec::new(), pos: Vec::new() }
    }

    pub fn value_count(&self) -> u64 {
        self.zero.count + self.neg.iter().map(|b| b.1).sum::<u64>() + self.pos.iter().map(|b| b.1).sum::<u64>()
    }

    pub fn is_empty(&self) -> bool {
        self.value_count() == 0
    }

    /// The `(center, count)` points aggregations bucket: negative bucket
    /// midpoints, the zero bucket at 0.0, positive midpoints, each clamped
    /// to `[min, max]` (`ExponentialHistogramBackedHistogramAggregator`).
    pub fn points(&self) -> Vec<(f64, u64)> {
        let mut out = Vec::with_capacity(self.neg.len() + self.pos.len() + 1);
        for &(i, c) in self.neg.iter().rev() {
            out.push((clamp(-point_lre(i, self.scale), self.min, self.max), c));
        }
        if self.zero.count > 0 {
            out.push((0.0, self.zero.count));
        }
        for &(i, c) in &self.pos {
            out.push((clamp(point_lre(i, self.scale), self.min, self.max), c));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Estimates (`ExponentialHistogramUtils`).

fn merge_lists(a: &[(i64, u64)], b: &[(i64, u64)], op: impl Fn(i64, i64) -> i64) -> Vec<(i64, i64)> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(a.len() + b.len());
    while i < a.len() || j < b.len() {
        let take_a = i < a.len() && (j >= b.len() || a[i].0 <= b[j].0);
        let take_b = j < b.len() && (i >= a.len() || b[j].0 <= a[i].0);
        let (mut idx, mut ca, mut cb) = (0, 0i64, 0i64);
        if take_a {
            idx = a[i].0;
            ca = a[i].1 as i64;
            i += 1;
        }
        if take_b {
            idx = b[j].0;
            cb = b[j].1 as i64;
            j += 1;
        }
        out.push((idx, op(ca, cb)));
    }
    out
}

pub fn estimate_sum(neg: &[(i64, u64)], pos: &[(i64, u64)], scale: i32) -> f64 {
    let mut sum = 0.0;
    for (idx, c) in merge_lists(pos, neg, |p, n| p - n) {
        if c != 0 {
            let t = point_lre(idx, scale) * c as f64;
            if t.is_finite() {
                sum += t;
            } else {
                sum = t;
            }
        }
    }
    sum
}

pub fn estimate_min(zero: &ZeroBucket, neg: &[(i64, u64)], pos: &[(i64, u64)], scale: i32) -> f64 {
    if let Some(&(i, _)) = neg.last() {
        return -upper_boundary(i, scale);
    }
    if zero.count > 0 {
        return if zero.threshold == 0.0 { 0.0 } else { -zero.threshold };
    }
    if let Some(&(i, _)) = pos.first() {
        return lower_boundary(i, scale);
    }
    f64::NAN
}

pub fn estimate_max(zero: &ZeroBucket, neg: &[(i64, u64)], pos: &[(i64, u64)], scale: i32) -> f64 {
    if let Some(&(i, _)) = pos.last() {
        return upper_boundary(i, scale);
    }
    if zero.count > 0 {
        return zero.threshold;
    }
    if let Some(&(i, _)) = neg.first() {
        return -lower_boundary(i, scale);
    }
    f64::NAN
}

// ---------------------------------------------------------------------------
// Merging (`ExponentialHistogramMerger`).

/// Buckets at `scale` reduced to `target` (`idx >> (scale - target)`,
/// equal indices summed).
fn reduce(b: &[(i64, u64)], scale: i32, target: i32) -> Vec<(i64, u64)> {
    let r = (scale - target) as u32;
    if r == 0 {
        return b.to_vec();
    }
    let mut out: Vec<(i64, u64)> = Vec::with_capacity(b.len());
    for &(i, c) in b {
        let ni = i >> r;
        match out.last_mut() {
            Some(last) if last.0 == ni => last.1 += c,
            _ => out.push((ni, c)),
        }
    }
    out
}

struct DownscaleStats([u32; MAX_INDEX_BITS + 2]);

impl DownscaleStats {
    fn add(&mut self, prev: i64, cur: i64) {
        let x = (prev ^ cur) as u64;
        let lz = x.leading_zeros() as usize;
        if lz > 0 {
            let change = 64 - lz;
            if change >= 1 && change <= self.0.len() {
                self.0[change - 1] += 1;
            }
        }
    }

    fn required_reduction(&self, n: usize) -> i32 {
        if n == 0 {
            return 0;
        }
        let mut total = 0usize;
        for (i, c) in self.0.iter().enumerate() {
            total += *c as usize;
            if total >= n {
                return i as i32 + 1;
            }
        }
        self.0.len() as i32
    }
}

/// Merge `a` and `b` (`setToMerged`) within `max_buckets` buckets.
pub fn merge(a: &ExpHist, b: &ExpHist, max_buckets: usize) -> ExpHist {
    merge_with(a, b, max_buckets, MAX_SCALE, |x, y| x + y)
}

fn merge_with(a: &ExpHist, b: &ExpHist, max_buckets: usize, max_scale: i32, op: impl Fn(i64, i64) -> i64 + Copy) -> ExpHist {
    let zero = a.zero.merge(&b.zero);
    let nan_min = |x: f64, y: f64| if x.is_nan() { y } else if y.is_nan() { x } else { x.min(y) };
    let nan_max = |x: f64, y: f64| if x.is_nan() { y } else if y.is_nan() { x } else { x.max(y) };
    let mut target = max_scale.min(a.scale).min(b.scale);
    let (z, neg, pos) = merge_buckets(zero, a, b, &mut target, max_buckets, op);
    ExpHist { scale: target, sum: a.sum + b.sum, min: nan_min(a.min, b.min), max: nan_max(a.max, b.max), zero: z, neg, pos }
}

fn merge_buckets(zero: ZeroBucket, a: &ExpHist, b: &ExpHist, target: &mut i32, max_buckets: usize, op: impl Fn(i64, i64) -> i64 + Copy) -> (ZeroBucket, Vec<(i64, u64)>, Vec<(i64, u64)>) {
    let build = |t: i32| {
        let to_u = |v: Vec<(i64, i64)>| -> Vec<(i64, u64)> { v.into_iter().map(|(i, c)| (i, c.max(0) as u64)).collect() };
        let pos = to_u(merge_lists(&reduce(&a.pos, a.scale, t), &reduce(&b.pos, b.scale, t), op));
        let neg = to_u(merge_lists(&reduce(&a.neg, a.scale, t), &reduce(&b.neg, b.scale, t), op));
        (pos, neg)
    };
    let put = |list: &[(i64, u64)], out: &mut Vec<(i64, u64)>, used: &mut usize, stats: &mut Option<DownscaleStats>| -> usize {
        let mut overflow = 0;
        let mut prev: Option<i64> = None;
        for &(i, c) in list {
            if c == 0 {
                continue;
            }
            if let (Some(p), Some(s)) = (prev, stats.as_mut()) {
                s.add(p, i);
            }
            if *used < max_buckets {
                out.push((i, c));
                *used += 1;
            } else {
                overflow += 1;
            }
            prev = Some(i);
        }
        overflow
    };
    let (pos, neg) = build(*target);
    let (z, ip, ineg) = zero.collapse_all(&pos, &neg, *target);
    let mut stats = Some(DownscaleStats([0; MAX_INDEX_BITS + 2]));
    let (mut on, mut op_) = (Vec::new(), Vec::new());
    let mut used = 0usize;
    let mut overflow = put(&neg[ineg..], &mut on, &mut used, &mut stats);
    overflow += put(&pos[ip..], &mut op_, &mut used, &mut stats);
    if overflow == 0 {
        return (z, on, op_);
    }
    let reduction = stats.as_ref().map(|s| s.required_reduction(overflow)).unwrap_or(1);
    *target -= reduction;
    let (pos, neg) = build(*target);
    let (z, ip, ineg) = zero.collapse_all(&pos, &neg, *target);
    let (mut on, mut op_) = (Vec::new(), Vec::new());
    let mut used = 0usize;
    let mut none = None;
    put(&neg[ineg..], &mut on, &mut used, &mut none);
    put(&pos[ip..], &mut op_, &mut used, &mut none);
    (z, on, op_)
}

/// `ExponentialHistogramMerger`: accumulates histograms (empty ones are skipped).
pub struct Merger {
    acc: Option<ExpHist>,
    max_buckets: usize,
}

impl Merger {
    pub fn new() -> Merger {
        Merger { acc: None, max_buckets: MAX_BUCKETS }
    }

    pub fn add(&mut self, h: &ExpHist) {
        if h.is_empty() {
            return;
        }
        let a = self.acc.take().unwrap_or_else(ExpHist::empty);
        self.acc = Some(merge(&a, h, self.max_buckets));
    }

    pub fn get(&self) -> ExpHist {
        self.acc.clone().unwrap_or_else(ExpHist::empty)
    }
}

impl Default for Merger {
    fn default() -> Self {
        Merger::new()
    }
}

/// `setToDifference(a, b)` (cumulative downsampling): `a - b`, and whether
/// the pair is truly cumulative.
pub fn difference(a: &ExpHist, b: &ExpHist) -> Result<(ExpHist, bool), String> {
    if b.is_empty() {
        let mut m = Merger::new();
        m.add(a);
        return Ok((m.get(), true));
    }
    let (ac, bc) = (a.value_count(), b.value_count());
    if ac == bc {
        return Ok((ExpHist::empty(), a.sum == b.sum && a.min == b.min && a.max == b.max));
    }
    if ac < bc {
        return Err(format!("Cannot subtract histograms (a-b), where a.count < b.count: {ac} < {bc}"));
    }
    let mut truly = !(a.scale > b.scale || a.min > b.min || a.max < b.max || a.zero.compare_threshold(&b.zero) < 0);
    let bz = a.zero.with_count(b.zero.count);
    let (bz, _, _) = bz.collapse_all(&b.neg, &b.pos, b.scale);
    if bz.compare_threshold(&a.zero) != 0 {
        truly = false;
    }
    let new_zero = a.zero.with_count(a.zero.count.saturating_sub(bz.count));
    let mut target = a.scale.min(b.scale);
    let (z, neg, pos) = merge_buckets(new_zero, a, b, &mut target, MAX_BUCKETS, |x, y| (x - y).max(0));
    let mut out = ExpHist { scale: target, sum: a.sum - b.sum, min: f64::NAN, max: f64::NAN, zero: z, neg, pos };
    let desired = ac - bc;
    if desired != out.value_count() {
        truly = false;
        scale_counts_to(&mut out, desired);
    }
    out.min = if a.min < b.min { a.min } else { a.min.max(estimate_min(&out.zero, &out.neg, &out.pos, out.scale)) };
    out.max = if a.max > b.max { a.max } else { a.max.min(estimate_max(&out.zero, &out.neg, &out.pos, out.scale)) };
    Ok((out, truly))
}

/// `FixedCapacityExponentialHistogram.scaleBucketCountsTo`.
fn scale_counts_to(h: &mut ExpHist, desired: u64) {
    let current = h.value_count();
    if current == 0 {
        return;
    }
    let factor = desired as f64 / current as f64;
    let mut total = 0u64;
    h.zero.count = (h.zero.count as f64 * factor).round() as u64;
    total += h.zero.count;
    for b in h.neg.iter_mut().chain(h.pos.iter_mut()) {
        b.1 = (b.1 as f64 * factor).round() as u64;
        total += b.1;
    }
    // Put the rounding remainder on the largest bucket.
    if total != desired {
        if let Some(b) = h.pos.iter_mut().chain(h.neg.iter_mut()).max_by_key(|b| b.1) {
            b.1 = (b.1 as i64 + desired as i64 - total as i64).max(0) as u64;
        }
    }
    h.neg.retain(|b| b.1 > 0);
    h.pos.retain(|b| b.1 > 0);
}

// ---------------------------------------------------------------------------
// Quantiles and ranks (`ExponentialHistogramQuantile`).

pub fn quantile(h: &ExpHist, q: f64) -> f64 {
    let n = h.value_count();
    if n == 0 {
        return f64::NAN;
    }
    let exact = q * (n - 1) as f64;
    let lo = exact.floor() as u64;
    let hi = exact.ceil() as u64;
    let f = exact - lo as f64;
    let (prev, val) = element_at_rank(h, hi);
    let (prev, val) = (clamp(prev, h.min, h.max), clamp(val, h.min, h.max));
    let r = if lo == hi { val } else { prev + (val - prev) * f };
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

fn mid_for_rank(b: &[(i64, u64)], scale: i32, rank: u64) -> (f64, f64) {
    let mut seen = 0u64;
    let mut prev_idx = i64::MIN;
    for &(i, c) in b {
        seen += c;
        if rank < seen {
            let center = point_lre(i, scale);
            let prev = if rank > 0 {
                if rank - 1 >= seen - c {
                    center
                } else {
                    point_lre(prev_idx, scale)
                }
            } else {
                f64::NAN
            };
            return (prev, center);
        }
        prev_idx = i;
    }
    (f64::NAN, f64::NAN)
}

fn element_at_rank(h: &ExpHist, rank: u64) -> (f64, f64) {
    let nneg: u64 = h.neg.iter().map(|b| b.1).sum();
    let z = h.zero.count;
    let first_neg_mid = h.neg.first().map(|b| point_lre(b.0, h.scale)).unwrap_or(f64::NAN);
    if rank < nneg {
        if rank == 0 {
            let last = h.neg.last().map(|b| point_lre(b.0, h.scale)).unwrap_or(f64::NAN);
            return (f64::NAN, -last);
        }
        let (p, v) = mid_for_rank(&h.neg, h.scale, nneg - rank);
        return (-v, -p);
    }
    if rank < nneg + z {
        return if rank == nneg { (-first_neg_mid, 0.0) } else { (0.0, 0.0) };
    }
    let (p, v) = mid_for_rank(&h.pos, h.scale, rank - nneg - z);
    if rank == 0 || rank - 1 < nneg {
        (-first_neg_mid, v)
    } else if rank - 1 < nneg + z {
        (0.0, v)
    } else {
        (p, v)
    }
}

/// `estimateRank`: values below (or, `inclusive`, at most) `x`.
pub fn estimate_rank(h: &ExpHist, x: f64, inclusive: bool) -> u64 {
    if h.min.is_nan() || x < h.min {
        return 0;
    }
    if x > h.max {
        return h.value_count();
    }
    let walk = |b: &[(i64, u64)], x: f64, incl: bool, lo: f64, hi: f64| -> u64 {
        let mut r = 0;
        for &(i, c) in b {
            let m = clamp(point_lre(i, h.scale), lo, hi);
            if m < x || (incl && m == x) {
                r += c;
            } else {
                break;
            }
        }
        r
    };
    let nneg: u64 = h.neg.iter().map(|b| b.1).sum();
    if x >= 0.0 {
        let mut rank = nneg;
        if x > 0.0 || inclusive {
            rank += h.zero.count;
        }
        rank + walk(&h.pos, x, inclusive, h.min, h.max)
    } else {
        nneg - walk(&h.neg, -x, !inclusive, -h.max, -h.min)
    }
}

/// The percentile rank of `x` (`cdf * 100`, ties counted half).
pub fn percentile_rank(h: &ExpHist, x: f64) -> f64 {
    let n = h.value_count();
    if n == 0 {
        return f64::NAN;
    }
    let less = estimate_rank(h, x, false) as f64;
    let le = estimate_rank(h, x, true) as f64;
    let cdf = (less + (le - less) / 2.0) / n as f64;
    cdf.clamp(0.0, 1.0) * 100.0
}

// ---------------------------------------------------------------------------
// T-Digest centroids -> exponential histogram
// (`TDigestToExponentialHistogramConverter` + the builder's estimates).

pub fn from_centroids(centroids: &[(f64, u64)]) -> ExpHist {
    let scale = MAX_SCALE;
    let mut zero_count = 0u64;
    let mut neg: Vec<(i64, u64)> = Vec::new();
    let mut pos: Vec<(i64, u64)> = Vec::new();
    let push = |list: &mut Vec<(i64, u64)>, v: f64, c: u64| {
        let idx = compute_index(v, scale);
        match list.last_mut() {
            Some(last) if last.0 == idx => last.1 += c,
            _ => list.push((idx, c)),
        }
    };
    // Negative centroids from the one closest to zero outward.
    for &(v, c) in centroids.iter().rev().filter(|x| x.0 < 0.0) {
        if c > 0 {
            push(&mut neg, v, c);
        }
    }
    for &(v, c) in centroids {
        if v == 0.0 {
            zero_count += c;
        } else if v > 0.0 && c > 0 {
            push(&mut pos, v, c);
        }
    }
    let zero = ZeroBucket::create(0.0, zero_count);
    let sum = estimate_sum(&neg, &pos, scale);
    let min = estimate_min(&zero, &neg, &pos, scale);
    let max = estimate_max(&zero, &neg, &pos, scale);
    ExpHist { scale, sum, min, max, zero, neg, pos }
}

/// `ExponentialHistogramParser.isExponentialHistogramSubFieldName`.
pub fn is_root_field(k: &str) -> bool {
    matches!(k, "scale" | "sum" | "min" | "max" | "zero" | "positive" | "negative")
}

/// `ExponentialHistogramToTDigestConverter.convert`: centroids at the
/// arithmetic bucket centers, from -inf to +inf, equal means merged.
pub fn to_centroids(h: &ExpHist) -> Vec<(f64, u64)> {
    let center = |i: i64| (lower_boundary(i, h.scale) + upper_boundary(i, h.scale)) / 2.0;
    let mut out: Vec<(f64, u64)> = Vec::new();
    let mut push = |m: f64, c: u64| {
        let m = if m == 0.0 { 0.0 } else { m };
        match out.last_mut() {
            Some(l) if l.0 == m => l.1 += c,
            _ => out.push((m, c)),
        }
    };
    for &(i, c) in h.neg.iter().rev() {
        push(-center(i), c);
    }
    if h.zero.count > 0 {
        push(0.0, h.zero.count);
    }
    for &(i, c) in &h.pos {
        push(center(i), c);
    }
    out
}

// ---------------------------------------------------------------------------
// Parsing (`ExponentialHistogramParser`, `ExponentialHistogramFieldMapper`).

fn token(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "START_OBJECT",
        Value::Array(_) => "START_ARRAY",
        Value::String(_) => "VALUE_STRING",
        Value::Number(_) => "VALUE_NUMBER",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Null => "VALUE_NULL",
    }
}

/// A parse failure: the `document_parsing_exception` inner reason (after
/// `error parsing field [f], `), or a structural parsing error.
#[derive(Debug)]
pub enum ParseError {
    Field(String),
    Structure(EsError),
    Other(EsError),
}

impl ParseError {
    /// The exception the field mapper raises (wrapped by the caller).
    pub fn into_es(self, field: &str) -> EsError {
        match self {
            ParseError::Field(m) => EsError::document_parsing(format!("[1:1] error parsing field [{field}], {m}")),
            ParseError::Structure(e) | ParseError::Other(e) => e,
        }
    }
}

fn expect_number(v: &Value) -> Result<(), ParseError> {
    if v.is_number() {
        Ok(())
    } else {
        Err(ParseError::Structure(EsError::parsing(format!("Failed to parse object: expecting token of type [VALUE_NUMBER] but found [{}]", token(v)))))
    }
}

fn parse_double_inf(v: &Value) -> Result<f64, ParseError> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::String(s) if s == "Infinity" => Ok(f64::INFINITY),
        Value::String(s) if s == "-Infinity" => Ok(f64::NEG_INFINITY),
        other => Err(ParseError::Field(format!("expected a number but got {}", token(other)))),
    }
}

fn parse_buckets(container: &str, v: &Value) -> Result<Vec<(i64, u64)>, ParseError> {
    let o = v.as_object().ok_or_else(|| ParseError::Structure(EsError::parsing(format!("Failed to parse object: expecting token of type [START_OBJECT] but found [{}]", token(v)))))?;
    let mut indices: Vec<i64> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    for (k, x) in o {
        match k.as_str() {
            "indices" | "counts" => {
                let a = x.as_array().ok_or_else(|| ParseError::Structure(EsError::parsing(format!("Failed to parse object: expecting token of type [START_ARRAY] but found [{}]", token(x)))))?;
                let mut out = Vec::with_capacity(a.len());
                for e in a {
                    expect_number(e)?;
                    let n = e.as_i64().unwrap_or_else(|| e.as_f64().unwrap_or(0.0) as i64);
                    if k == "indices" && !(MIN_INDEX..=MAX_INDEX).contains(&n) {
                        return Err(ParseError::Field(format!("{container}.indices values must all be in range [{MIN_INDEX}, {MAX_INDEX}] but got {n}")));
                    }
                    if k == "counts" && n <= 0 {
                        return Err(ParseError::Field(format!("{container}.counts values must all be greater than zero but got {n}")));
                    }
                    out.push(n);
                }
                if k == "indices" {
                    indices = out;
                } else {
                    counts = out;
                }
            }
            other => return Err(ParseError::Field(format!("with unknown parameter for {container} sub-object [{other}]"))),
        }
    }
    if indices.len() != counts.len() {
        return Err(ParseError::Field(format!(
            "expected same length from [{container}.indices] and [{container}.counts] but got [{} != {}]",
            indices.len(),
            counts.len()
        )));
    }
    let mut b: Vec<(i64, u64)> = indices.into_iter().zip(counts).map(|(i, c)| (i, c as u64)).collect();
    b.sort_by_key(|x| x.0);
    for w in b.windows(2) {
        if w[0].0 == w[1].0 {
            return Err(ParseError::Field(format!("expected entries of [{container}.indices] to be unique, but got {} multiple times", w[1].0)));
        }
    }
    Ok(b)
}

/// Parse a document value. `coerce`: an object starting with `values` /
/// `counts` is a T-Digest histogram, converted.
pub fn parse(field: &str, v: &Value, coerce: bool) -> Result<ExpHist, ParseError> {
    let o = v.as_object().ok_or_else(|| ParseError::Structure(EsError::parsing(format!("Failed to parse object: expecting token of type [START_OBJECT] but found [{}]", token(v)))))?;
    if coerce && o.keys().next().is_some_and(|k| k == "values" || k == "counts") {
        let (vs, cs) = crate::fieldtypes::index::parse_histogram(crate::mapping::EsType::TDigest, v).map_err(ParseError::Other)?;
        let centroids: Vec<(f64, u64)> = vs.into_iter().zip(cs).map(|(x, c)| (x, c.max(0) as u64)).collect();
        return Ok(from_centroids(&centroids));
    }
    let _ = field;
    let mut scale: Option<i32> = None;
    let (mut sum, mut min, mut max): (Option<f64>, Option<f64>, Option<f64>) = (None, None, None);
    let (mut zthr, mut zcount) = (0.0f64, 0u64);
    let mut neg: Vec<(i64, u64)> = Vec::new();
    let mut pos: Vec<(i64, u64)> = Vec::new();
    for (k, x) in o {
        match k.as_str() {
            "scale" => {
                expect_number(x)?;
                let s = x.as_i64().unwrap_or_else(|| x.as_f64().unwrap_or(0.0) as i64);
                if s > MAX_SCALE as i64 || s < MIN_SCALE as i64 {
                    return Err(ParseError::Field(format!("scale field must be in range [{MIN_SCALE}, {MAX_SCALE}] but got {s}")));
                }
                scale = Some(s as i32);
            }
            "sum" => sum = Some(parse_double_inf(x)?),
            "min" => min = Some(parse_double_inf(x)?),
            "max" => max = Some(parse_double_inf(x)?),
            "zero" => {
                let z = x.as_object().ok_or_else(|| ParseError::Structure(EsError::parsing(format!("Failed to parse object: expecting token of type [START_OBJECT] but found [{}]", token(x)))))?;
                for (zk, zv) in z {
                    match zk.as_str() {
                        "threshold" => {
                            expect_number(zv)?;
                            let t = zv.as_f64().unwrap_or(0.0);
                            if t < 0.0 || !t.is_finite() {
                                return Err(ParseError::Field(format!("zero.threshold field must be a non-negative, finite number but got {}", java_double_str(t))));
                            }
                            zthr = t;
                        }
                        "count" => {
                            expect_number(zv)?;
                            let c = zv.as_i64().unwrap_or_else(|| zv.as_f64().unwrap_or(0.0) as i64);
                            if c < 0 {
                                return Err(ParseError::Field(format!("zero.count field must be a non-negative number but got {c}")));
                            }
                            zcount = c as u64;
                        }
                        other => return Err(ParseError::Field(format!("with unknown parameter for zero sub-object [{other}]"))),
                    }
                }
            }
            "positive" => pos = parse_buckets("positive", x)?,
            "negative" => neg = parse_buckets("negative", x)?,
            other => return Err(ParseError::Field(format!("with unknown parameter [{other}]"))),
        }
    }
    let scale = scale.ok_or_else(|| ParseError::Field("expected field called [scale]".to_string()))?;
    // Total count overflow (checked add).
    let mut total: u64 = zcount;
    for &(_, c) in neg.iter().chain(pos.iter()) {
        total = total.checked_add(c).filter(|t| *t <= i64::MAX as u64).ok_or_else(|| {
            ParseError::Other(EsError::illegal_argument(format!("Field [{field}] has a total value count exceeding the allowed maximum value of {}", i64::MAX)))
        })?;
    }
    let zero = ZeroBucket::create(zthr, zcount);
    let empty = total == 0;
    let sum = match sum {
        Some(s) => {
            if empty && s != 0.0 {
                return Err(ParseError::Field(format!("sum field must be zero if the histogram is empty, but got {}", java_double_str(s))));
            }
            s
        }
        None => estimate_sum(&neg, &pos, scale),
    };
    let min = match min {
        Some(m) => {
            if empty {
                return Err(ParseError::Field(format!("min field must be null if the histogram is empty, but got {}", java_double_str(m))));
            }
            m
        }
        None => estimate_min(&zero, &neg, &pos, scale),
    };
    let max = match max {
        Some(m) => {
            if empty {
                return Err(ParseError::Field(format!("max field must be null if the histogram is empty, but got {}", java_double_str(m))));
            }
            m
        }
        None => estimate_max(&zero, &neg, &pos, scale),
    };
    Ok(ExpHist { scale, sum: if empty { 0.0 } else { sum }, min, max, zero, neg, pos })
}

/// Java `Double.toString` for the messages above.
fn java_double_str(d: f64) -> String {
    match crate::values::java_double(d) {
        Value::Number(n) => {
            let s = n.to_string();
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{s}.0")
            }
        }
        other => other.as_str().map(str::to_string).unwrap_or_else(|| d.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Doc value encoding.

/// Compact doc value: `scale;sum;min;max;zero_threshold;zero_count;neg;pos`
/// with doubles as hex bit patterns and buckets as `i:c,i:c`.
pub fn encode(h: &ExpHist) -> String {
    let b = |l: &[(i64, u64)]| l.iter().map(|(i, c)| format!("{i}:{c}")).collect::<Vec<_>>().join(",");
    let zthr = if h.zero.index_based { h.zero.threshold } else { h.zero.threshold };
    format!(
        "{};{:x};{:x};{:x};{:x};{};{};{}",
        h.scale,
        h.sum.to_bits(),
        h.min.to_bits(),
        h.max.to_bits(),
        zthr.to_bits(),
        h.zero.count,
        b(&h.neg),
        b(&h.pos)
    )
}

pub fn decode(s: &str) -> Option<ExpHist> {
    let mut it = s.split(';');
    let scale: i32 = it.next()?.parse().ok()?;
    let f = |x: Option<&str>| x.and_then(|x| u64::from_str_radix(x, 16).ok()).map(f64::from_bits);
    let sum = f(it.next())?;
    let min = f(it.next())?;
    let max = f(it.next())?;
    let zthr = f(it.next())?;
    let zcount: u64 = it.next()?.parse().ok()?;
    let b = |x: Option<&str>| -> Option<Vec<(i64, u64)>> {
        let x = x?;
        if x.is_empty() {
            return Some(Vec::new());
        }
        x.split(',')
            .map(|p| {
                let (i, c) = p.split_once(':')?;
                Some((i.parse().ok()?, c.parse().ok()?))
            })
            .collect()
    };
    let neg = b(it.next())?;
    let pos = b(it.next())?;
    Some(ExpHist { scale, sum, min, max, zero: ZeroBucket::create(zthr, zcount), neg, pos })
}

// ---------------------------------------------------------------------------
// Rendering (`ExponentialHistogramXContent.serialize`).

fn dbl(d: f64) -> Value {
    if d.is_finite() {
        crate::values::java_double(d)
    } else if d.is_nan() {
        Value::String("NaN".into())
    } else if d > 0.0 {
        Value::String("Infinity".into())
    } else {
        Value::String("-Infinity".into())
    }
}

pub fn render(h: &ExpHist) -> Value {
    let mut m = Map::new();
    m.insert("scale".into(), Value::from(h.scale));
    let count = h.value_count();
    if h.sum != 0.0 || count > 0 {
        m.insert("sum".into(), dbl(h.sum));
    }
    if !h.min.is_nan() {
        m.insert("min".into(), dbl(h.min));
    }
    if !h.max.is_nan() {
        m.insert("max".into(), dbl(h.max));
    }
    let zthr = h.zero.threshold;
    if h.zero.count != 0 || zthr != 0.0 {
        let mut z = Map::new();
        if h.zero.count != 0 {
            z.insert("count".into(), Value::from(h.zero.count));
        }
        if zthr != 0.0 {
            z.insert("threshold".into(), dbl(zthr));
        }
        m.insert("zero".into(), Value::Object(z));
    }
    for (k, l) in [("positive", &h.pos), ("negative", &h.neg)] {
        if !l.is_empty() {
            let mut b = Map::new();
            b.insert("indices".into(), Value::Array(l.iter().map(|x| Value::from(x.0)).collect()));
            b.insert("counts".into(), Value::Array(l.iter().map(|x| Value::from(x.1)).collect()));
            m.insert(k.into(), Value::Object(b));
        }
    }
    Value::Object(m)
}

/// Kahan-compensated sum (`CompensatedSum`).
#[derive(Default)]
pub struct KahanSum {
    value: f64,
    delta: f64,
}

impl KahanSum {
    pub fn add(&mut self, v: f64) {
        if !v.is_finite() || !self.value.is_finite() {
            self.value += v;
            return;
        }
        let corrected = v - self.delta;
        let new = self.value + corrected;
        self.delta = (new - self.value) - corrected;
        self.value = new;
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn h(v: Value) -> ExpHist {
        parse("h", &v, true).unwrap()
    }

    #[test]
    fn estimates_and_render() {
        let x = h(json!({"scale": 0, "positive": {"indices": [0, 1], "counts": [1, 1]}, "negative": {"indices": [0], "counts": [4]}}));
        assert_eq!(x.sum, -1.3333333333333335);
        assert_eq!(x.min, -2.0);
        assert_eq!(x.max, 4.0);
        let z = h(json!({"scale": 0, "zero": {"count": 101}}));
        assert_eq!(render(&z), json!({"scale": 0, "sum": 0.0, "min": 0.0, "max": 0.0, "zero": {"count": 101}}));
        assert_eq!(render(&h(json!({"scale": -7, "sum": 0.0}))), json!({"scale": -7}));
        let inf = h(json!({"scale": 0, "positive": {"indices": [2000], "counts": [1]}}));
        assert_eq!(render(&inf)["sum"], json!("Infinity"));
        let d = decode(&encode(&x)).unwrap();
        assert_eq!(render(&d), render(&x));
    }

    fn docs() -> Vec<ExpHist> {
        vec![
            h(json!({"scale":-1,"sum":292.0,"min":-17.0,"max":101.0,"zero":{"count":3},"positive":{"indices":[-1,0,1,2,3],"counts":[1,3,5,3,2]},"negative":{"indices":[0,2],"counts":[1,1]}})),
            h(json!({"scale":38,"sum":0.0})),
            h(json!({"scale":38,"sum":0.0,"min":0.0,"max":0.0,"zero":{"count":5}})),
            h(json!({"scale":5,"sum":-4101.0,"min":-1000.0,"max":-700.0,"negative":{"indices":[302,308,314,318],"counts":[2,1,1,1]}})),
            h(json!({"scale":0,"sum":90000.0,"min":2000.0,"max":13000.0,"positive":{"indices":[10,11,12,13],"counts":[1,2,4,5]}})),
        ]
    }

    #[test]
    fn percentiles_match_es() {
        let mut m = Merger::new();
        for d in docs() {
            m.add(&d);
        }
        let r = m.get();
        assert_eq!(r.value_count(), 41);
        let q = |p: f64| quantile(&r, p);
        assert!((q(0.05) - -409.6).abs() < 0.05, "{}", q(0.05));
        assert!(q(0.25).abs() < 0.01);
        assert!((q(0.5) - 6.4).abs() < 0.01, "{}", q(0.5));
        assert!((q(0.75) - 1638.4).abs() < 0.1, "{}", q(0.75));
        assert!((q(0.95) - 6553.6).abs() < 0.1, "{}", q(0.95));
        assert!((percentile_rank(&r, 0.0) - 26.829).abs() < 0.01);
        assert!((percentile_rank(&r, 100.0) - 65.854).abs() < 0.01);
        assert!((percentile_rank(&r, 5000.0) - 78.049).abs() < 0.01);
    }

    #[test]
    fn histogram_points() {
        let d = docs();
        let keys: Vec<i64> = d[3].points().iter().map(|(v, _)| (v / 5.0).floor() as i64 * 5).collect();
        assert_eq!(keys, vec![-995, -910, -800, -705]);
    }
}
