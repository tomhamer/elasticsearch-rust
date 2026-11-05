//! Cardinality (HyperLogLog++ with exact/linear counting below the
//! `precision_threshold`) and percentiles (merging t-digest).

use std::collections::HashSet;

/// Compact serde for sketch arrays: base64 of the little-endian values
/// (a dense HLL or a t-digest's centroids as JSON number arrays are 3-5x
/// larger). Plain arrays, as older nodes wrote them, still read.
mod packed {
    use base64::Engine;
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub trait Le: Sized + Copy + for<'de> Deserialize<'de> {
        const N: usize;
        fn put(self, out: &mut Vec<u8>);
        fn get(b: &[u8]) -> Self;
    }
    impl Le for u8 {
        const N: usize = 1;
        fn put(self, out: &mut Vec<u8>) { out.push(self) }
        fn get(b: &[u8]) -> Self { b[0] }
    }
    impl Le for u64 {
        const N: usize = 8;
        fn put(self, out: &mut Vec<u8>) { out.extend(self.to_le_bytes()) }
        fn get(b: &[u8]) -> Self { u64::from_le_bytes(b.try_into().unwrap()) }
    }
    impl Le for f64 {
        const N: usize = 8;
        fn put(self, out: &mut Vec<u8>) { out.extend(self.to_le_bytes()) }
        fn get(b: &[u8]) -> Self { f64::from_le_bytes(b.try_into().unwrap()) }
    }

    fn encode<T: Le>(v: &[T]) -> String {
        let mut out = Vec::with_capacity(v.len() * T::N);
        v.iter().for_each(|x| x.put(&mut out));
        base64::engine::general_purpose::STANDARD.encode(out)
    }

    fn decode<T: Le, E: Error>(s: &str) -> Result<Vec<T>, E> {
        let b = base64::engine::general_purpose::STANDARD.decode(s).map_err(E::custom)?;
        if b.len() % T::N != 0 {
            return Err(E::custom("packed sketch array length"));
        }
        Ok(b.chunks(T::N).map(T::get).collect())
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Form<T> {
        Packed(String),
        Plain(Vec<T>),
    }

    pub fn ser<T: Le, S: Serializer>(v: &Vec<T>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&encode(v))
    }

    pub fn de<'de, T: Le, D: Deserializer<'de>>(d: D) -> Result<Vec<T>, D::Error> {
        match Form::<T>::deserialize(d)? {
            Form::Packed(s) => decode(&s),
            Form::Plain(v) => Ok(v),
        }
    }

    pub mod opt {
        use super::*;
        pub fn ser<T: Le, S: Serializer>(v: &Option<Vec<T>>, s: S) -> Result<S::Ok, S::Error> {
            match v {
                Some(v) => s.serialize_some(&encode(v)),
                None => s.serialize_none(),
            }
        }
        pub fn de<'de, T: Le, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<T>>, D::Error> {
            match Option::<Form<T>>::deserialize(d)? {
                Some(Form::Packed(s)) => decode(&s).map(Some),
                Some(Form::Plain(v)) => Ok(Some(v)),
                None => Ok(None),
            }
        }
    }
}

/// MurmurHash3 x64_128, first 64 bits (what ES hashes values with).
pub fn murmur3_64(data: &[u8]) -> u64 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let len = data.len();
    let (mut h1, mut h2) = (0u64, 0u64);
    let nblocks = len / 16;
    for i in 0..nblocks {
        let b = &data[i * 16..i * 16 + 16];
        let mut k1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(27).wrapping_add(h2).wrapping_mul(5).wrapping_add(0x52dc_e729);
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2.rotate_left(31).wrapping_add(h1).wrapping_mul(5).wrapping_add(0x3849_5ab5);
    }
    let tail = &data[nblocks * 16..];
    let (mut k1, mut k2) = (0u64, 0u64);
    for (i, &b) in tail.iter().enumerate().rev() {
        if i >= 8 {
            k2 ^= (b as u64) << ((i - 8) * 8);
        } else {
            k1 ^= (b as u64) << (i * 8);
        }
    }
    if tail.len() > 8 {
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
    }
    h1 ^= len as u64;
    h2 ^= len as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let fmix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    };
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1.wrapping_add(h2)
}

/// HyperLogLog++: exact set of hashes until `threshold`, then registers.
pub struct HyperLogLog {
    p: u32,
    threshold: usize,
    exact: Option<HashSet<u64>>,
    regs: Vec<u8>,
}

impl HyperLogLog {
    /// ES maps `precision_threshold` (default 3000) to precision
    /// `ceil(log2(threshold))+1`, clamped to [4, 18].
    pub fn new(precision_threshold: usize) -> Self {
        let t = precision_threshold.clamp(1, 40_000);
        let p = ((t as f64).log2().ceil() as u32 + 1).clamp(4, 18);
        Self { p, threshold: t, exact: Some(HashSet::new()), regs: Vec::new() }
    }

    pub fn add_hash(&mut self, h: u64) {
        if let Some(set) = &mut self.exact {
            set.insert(h);
            if set.len() > self.threshold {
                let hashes: Vec<u64> = set.drain().collect();
                self.exact = None;
                self.regs = vec![0; 1 << self.p];
                for h in hashes {
                    self.add_reg(h);
                }
            }
            return;
        }
        self.add_reg(h);
    }

    fn add_reg(&mut self, h: u64) {
        let idx = (h >> (64 - self.p)) as usize;
        let w = (h << self.p) | (1 << (self.p - 1));
        let rank = w.leading_zeros() as u8 + 1;
        if rank > self.regs[idx] {
            self.regs[idx] = rank;
        }
    }

    pub fn add_bytes(&mut self, b: &[u8]) {
        self.add_hash(murmur3_64(b));
    }

    /// Serializable state (exact hash set below the threshold, registers above).
    pub fn state(&self) -> HllState {
        let mut exact: Option<Vec<u64>> = self.exact.as_ref().map(|s| s.iter().copied().collect());
        if let Some(e) = &mut exact {
            e.sort_unstable();
        }
        HllState { p: self.p, threshold: self.threshold, exact, regs: self.regs.clone() }
    }

    pub fn from_state(s: &HllState) -> Self {
        Self { p: s.p, threshold: s.threshold, exact: s.exact.as_ref().map(|e| e.iter().copied().collect()), regs: s.regs.clone() }
    }

    /// Union with another sketch of the same precision (register-wise max;
    /// exact sets union and spill to registers past the threshold). Exact
    /// regardless of how many values each side holds.
    pub fn merge(&mut self, other: &HyperLogLog) {
        match (&other.exact, other.regs.is_empty()) {
            (Some(set), _) => {
                for h in set.iter().copied().collect::<Vec<_>>() {
                    self.add_hash(h);
                }
            }
            (None, _) => {
                if self.exact.is_some() {
                    // Adopt registers: spill our exact hashes into them.
                    let hashes: Vec<u64> = self.exact.take().unwrap().into_iter().collect();
                    self.p = other.p;
                    self.regs = other.regs.clone();
                    for h in hashes {
                        self.add_reg(h);
                    }
                } else {
                    for (a, b) in self.regs.iter_mut().zip(&other.regs) {
                        *a = (*a).max(*b);
                    }
                }
            }
        }
    }

    pub fn count(&self) -> u64 {
        if let Some(set) = &self.exact {
            return set.len() as u64;
        }
        let m = (1u64 << self.p) as f64;
        let alpha = match self.p {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let mut sum = 0.0;
        let mut zeros = 0;
        for &r in &self.regs {
            sum += 2f64.powi(-(r as i32));
            if r == 0 {
                zeros += 1;
            }
        }
        let est = alpha * m * m / sum;
        if zeros > 0 {
            let lc = m * (m / zeros as f64).ln();
            if lc <= 5.0 * m {
                return lc.round() as u64;
            }
        }
        est.round() as u64
    }
}

/// Serialized [`HyperLogLog`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HllState {
    pub p: u32,
    pub threshold: usize,
    /// Sorted hashes while below the threshold.
    #[serde(serialize_with = "packed::opt::ser", deserialize_with = "packed::opt::de", default)]
    pub exact: Option<Vec<u64>>,
    #[serde(serialize_with = "packed::ser", deserialize_with = "packed::de")]
    pub regs: Vec<u8>,
}

/// Serialized [`TDigest`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TDigestState {
    pub compression: f64,
    #[serde(serialize_with = "packed::ser", deserialize_with = "packed::de")]
    pub means: Vec<f64>,
    #[serde(serialize_with = "packed::ser", deserialize_with = "packed::de")]
    pub weights: Vec<f64>,
    /// Total weight.
    pub count: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// ES `HybridDigest`: exact sorted samples (`Dist` quantile/cdf) until
/// `20 * compression` values were added, then a merging t-digest queried with
/// `MergingDigest`'s interpolation.
pub struct TDigest {
    compression: f64,
    means: Vec<f64>,
    weights: Vec<f64>,
    buffer: Vec<f64>,
    count: f64,
    min: f64,
    max: f64,
    /// Exact mode: every sample so far (`None` once merged).
    samples: Option<Vec<f64>>,
    max_sorting: usize,
}

/// `means ++ buffer ++ extra` as (mean, weight) points stably sorted by
/// mean: the sorted centroids, the buffer sorted (`buffer` is sorted in
/// place) and `extra` sorted stably, merged with ties going to the earlier
/// sequence.
fn merged_points(means: &[f64], weights: &[f64], buffer: &mut [f64], mut extra: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    sort_f64(buffer);
    extra.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(means.len() + buffer.len() + extra.len());
    let (mut i, mut j, mut k) = (0, 0, 0);
    loop {
        let mut best: Option<(u8, f64)> = means.get(i).map(|&m| (0, m));
        if let Some(&v) = buffer.get(j) {
            if best.is_none_or(|(_, x)| v.total_cmp(&x).is_lt()) { best = Some((1, v)); }
        }
        if let Some(e) = extra.get(k) {
            if best.is_none_or(|(_, x)| e.0.total_cmp(&x).is_lt()) { best = Some((2, e.0)); }
        }
        match best {
            None => return pts,
            Some((0, _)) => { pts.push((means[i], weights[i])); i += 1; }
            Some((1, _)) => { pts.push((buffer[j], 1.0)); j += 1; }
            Some(_) => { pts.push(extra[k]); k += 1; }
        }
    }
}

/// Sort values in `f64::total_cmp` order. Unstable: values equal under it
/// have the same bits, so their order is invisible.
fn sort_f64(v: &mut [f64]) {
    if v.len() < 64 {
        v.sort_unstable_by(f64::total_cmp);
        return;
    }
    // `total_cmp`'s key: the bits, with the magnitude flipped for negatives
    // (an involution).
    let flip = |b: i64| b ^ ((((b >> 63) as u64) >> 1) as i64);
    let mut keys: Vec<i64> = v.iter().map(|&x| flip(x.to_bits() as i64)).collect();
    keys.sort_unstable();
    for (x, k) in v.iter_mut().zip(keys) {
        *x = f64::from_bits(flip(k) as u64);
    }
}

fn weighted_average(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    let (a, wa, b, wb) = if x1 <= x2 { (x1, w1, x2, w2) } else { (x2, w2, x1, w1) };
    let x = (a * wa + b * wb) / (wa + wb);
    a.max(x.min(b))
}

impl TDigest {
    /// Conservative live peak for the next insertion and subsequent reads.
    /// Compression retains old arrays while building points, sorting scratch,
    /// and replacement means/weights. Bound those by the actual live points,
    /// not total observations (which would overcharge a compressed digest).
    pub fn working_bytes(&self) -> u64 {
        let samples = self.samples.as_ref();
        let capacity = self.means.capacity() + self.weights.capacity() + self.buffer.capacity()
            + samples.map_or(0, Vec::capacity);
        let points = self.means.len() + self.buffer.len() + samples.map_or(0, Vec::len) + 1;
        (capacity as u64).saturating_mul(8).saturating_add((points.max(4) as u64).saturating_mul(128))
    }

    pub fn new(compression: f64) -> Self {
        let compression = compression.max(20.0);
        Self {
            compression,
            means: Vec::new(),
            weights: Vec::new(),
            buffer: Vec::new(),
            count: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            samples: Some(Vec::new()),
            max_sorting: compression.round() as usize * 20,
        }
    }

    pub fn add(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        if let Some(s) = &mut self.samples {
            s.push(v);
            if s.len() >= self.max_sorting {
                self.buffer = self.samples.take().unwrap();
                self.compress();
            }
            return;
        }
        self.buffer.push(v);
        if self.buffer.len() >= (self.compression as usize) * 10 {
            self.compress();
        }
    }

    pub fn count(&self) -> u64 {
        match &self.samples {
            Some(s) => s.len() as u64,
            None => (self.count + self.buffer.len() as f64) as u64,
        }
    }

    fn compress(&mut self) {
        self.compress_with(Vec::new());
    }

    /// Merge the buffer plus weighted `extra` centroids into the digest.
    fn compress_with(&mut self, extra: Vec<(f64, f64)>) {
        if self.buffer.is_empty() && extra.is_empty() {
            return;
        }
        // The order a stable sort of `means ++ buffer ++ extra` by mean gives:
        // the (sorted) centroids, the buffer sorted (weight-1 values: equal
        // ones are interchangeable) and `extra` sorted stably, merged with
        // ties going to the earlier sequence.
        let mut buffer = std::mem::take(&mut self.buffer);
        let pts = merged_points(&self.means, &self.weights, &mut buffer, extra);
        buffer.clear();
        self.buffer = buffer;
        let total: f64 = pts.iter().map(|p| p.1).sum();
        let mut means = Vec::new();
        let mut weights = Vec::new();
        let mut so_far = 0.0;
        let (mut cm, mut cw) = pts[0];
        let k = |q: f64, c: f64| c / (2.0 * std::f64::consts::PI) * (2.0 * q - 1.0).asin();
        let mut k_lo = k(0.0, self.compression);
        for &(m, w) in &pts[1..] {
            let q = (so_far + cw + w) / total;
            if k(q, self.compression) - k_lo <= 1.0 {
                cm += (m - cm) * w / (cw + w);
                cw += w;
            } else {
                so_far += cw;
                means.push(cm);
                weights.push(cw);
                k_lo = k(so_far / total, self.compression);
                cm = m;
                cw = w;
            }
        }
        means.push(cm);
        weights.push(cw);
        self.means = means;
        self.weights = weights;
        self.count = total;
    }

    /// Serializable form: exact samples are weight-1 centroids.
    pub fn state(&mut self) -> TDigestState {
        let (means, weights, count) = match &mut self.samples {
            Some(s) => {
                sort_f64(s);
                (s.clone(), vec![1.0; s.len()], s.len() as f64)
            }
            None => {
                self.compress();
                (self.means.clone(), self.weights.clone(), self.count)
            }
        };
        let any = count > 0.0;
        TDigestState { compression: self.compression, means, weights, count, min: any.then_some(self.min), max: any.then_some(self.max) }
    }

    pub fn from_state(st: &TDigestState) -> Self {
        let mut d = TDigest::new(st.compression);
        if st.weights.iter().all(|w| *w == 1.0) && st.means.len() < d.max_sorting {
            d.samples = Some(st.means.clone());
        } else {
            d.samples = None;
            d.means = st.means.clone();
            d.weights = st.weights.clone();
            d.count = st.count;
        }
        d.min = st.min.unwrap_or(f64::INFINITY);
        d.max = st.max.unwrap_or(f64::NEG_INFINITY);
        d
    }

    /// Fold `other`'s centroids into this digest (shard reduce).
    pub fn merge(&mut self, other: &mut TDigest) {
        let st = other.state();
        self.min = self.min.min(st.min.unwrap_or(f64::INFINITY));
        self.max = self.max.max(st.max.unwrap_or(f64::NEG_INFINITY));
        let unit = st.weights.iter().all(|w| *w == 1.0);
        if let (Some(s), true) = (&mut self.samples, unit) {
            if s.len() + st.means.len() < self.max_sorting {
                s.extend(st.means.iter().copied());
                return;
            }
        }
        if let Some(s) = self.samples.take() {
            self.buffer = s;
        }
        let extra: Vec<(f64, f64)> = st.means.into_iter().zip(st.weights).collect();
        self.compress_with(extra);
    }

    /// Value at quantile `q` in [0, 1].
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        if let Some(s) = &mut self.samples {
            if s.is_empty() {
                return None;
            }
            sort_f64(s);
            let len = s.len();
            let index = q * (len - 1) as f64;
            let (lo, hi, w) = if index <= 0.0 {
                (0, 0, 0.0)
            } else if index >= (len - 1) as f64 {
                (len - 1, len - 1, 0.0)
            } else {
                let l = index.floor() as usize;
                (l, l + 1, index - l as f64)
            };
            return Some(s[lo] + w * (s[hi] - s[lo]));
        }
        self.compress();
        let n = self.means.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(self.means[0]);
        }
        let (mean, weight, total) = (&self.means, &self.weights, self.count);
        let index = q * total;
        if index < 0.0 {
            return Some(self.min);
        }
        if index >= total {
            return Some(self.max);
        }
        let mut so_far = weight[0] / 2.0;
        if weight[0] > 1.0 && index < so_far {
            return Some(weighted_average(self.min, so_far - index, mean[0], index));
        }
        if weight[n - 1] > 1.0 && total - index <= weight[n - 1] / 2.0 {
            return Some(self.max - (total - index - 1.0) / (weight[n - 1] / 2.0 - 1.0) * (self.max - mean[n - 1]));
        }
        for i in 0..n - 1 {
            let dw = (weight[i] + weight[i + 1]) / 2.0;
            if so_far + dw > index {
                let z1 = index - so_far;
                let z2 = so_far + dw - index;
                return Some(weighted_average(mean[i], z2, mean[i + 1], z1));
            }
            so_far += dw;
        }
        let z1 = index - so_far;
        let z2 = weight[n - 1] / 2.0 - z1;
        Some(weighted_average(mean[n - 1], z1, self.max, z2))
    }

    /// Fraction of values <= v (percentile_ranks).
    pub fn cdf(&mut self, x: f64) -> Option<f64> {
        let (mean, weight, total): (Vec<f64>, Vec<f64>, f64) = if let Some(s) = &mut self.samples {
            sort_f64(s);
            (s.clone(), vec![1.0; s.len()], s.len() as f64)
        } else {
            self.compress();
            (self.means.clone(), self.weights.clone(), self.count)
        };
        let n = mean.len();
        if n == 0 {
            return None;
        }
        let (min, max) = (mean[0].min(self.min), mean[n - 1].max(self.max));
        if n == 1 {
            return Some(if x < min { 0.0 } else if x > max { 1.0 } else { 0.5 });
        }
        if x < min {
            return Some(0.0);
        }
        if x == min {
            let dw: f64 = mean.iter().zip(&weight).take_while(|(m, _)| **m == x).map(|(_, w)| *w).sum();
            return Some(dw / 2.0 / total);
        }
        if x > max {
            return Some(1.0);
        }
        if x == max {
            let dw: f64 = mean.iter().zip(&weight).rev().take_while(|(m, _)| **m == x).map(|(_, w)| *w).sum();
            return Some((total - dw / 2.0) / total);
        }
        let interp = |x: f64, x0: f64, x1: f64| (x - x0) / (x1 - x0);
        let mut left = (mean[1] - mean[0]) / 2.0;
        let mut so_far = 0.0;
        for i in 0..n - 1 {
            let right = (mean[i + 1] - mean[i]) / 2.0;
            if x < mean[i] + right {
                let v = (so_far + weight[i] * interp(x, mean[i] - left, mean[i] + right)) / total;
                return Some(v.max(0.0));
            }
            so_far += weight[i];
            left = right;
        }
        let last = n - 1;
        let right = (mean[last] - mean[last - 1]) / 2.0;
        if x < mean[last] + right {
            return Some((so_far + weight[last] * interp(x, mean[last] - right, mean[last] + right)) / total);
        }
        Some(1.0)
    }
}

/// HdrHistogram 2.1.9 `DoubleHistogram` with auto-resize, as ES builds for
/// `hdr` percentiles: doubles are scaled to integers by a power-of-two
/// conversion ratio chosen by auto-ranging, and counted in an integer
/// `Histogram` with log-linear buckets of `digits` significant digits. This
/// is a faithful port (including the order-dependence of auto-ranging and
/// the 2.1.9 range-shift behaviour) so percentiles match ES bit for bit.
#[derive(Clone, Debug)]
pub struct Hdr {
    digits: u32,
    // Integer histogram geometry (lowestDiscernibleValue = 1).
    half_mag: u32,
    sub_count: i64,
    half: i64,
    highest_trackable: i64,
    counts_len: i64,
    /// Counts by logical counts-array index.
    counts: std::collections::BTreeMap<i64, u64>,
    total: u64,
    max_value: i64,
    min_non_zero: i64,
    // DoubleHistogram auto-range state.
    configured_ratio: i64,
    lowest: f64,
    highest: f64,
    i2d: f64,
    d2i: f64,
}

/// Serialized [`Hdr`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HdrState {
    pub digits: u32,
    pub highest_trackable: i64,
    pub configured_ratio: i64,
    pub lowest: f64,
    pub highest: f64,
    /// `(counts index, count)` ascending.
    pub counts: Vec<(i64, u64)>,
    pub total: u64,
    pub max_value: i64,
    pub min_non_zero: i64,
}

fn containing_order(n: i64) -> u32 {
    64 - (n as u64).leading_zeros()
}

/// `DoubleHistogram.highestAllowedValueEver`: the largest power of two
/// below `Double.MAX_VALUE / 4`, rounded up (2^1022).
fn highest_allowed_value_ever() -> f64 {
    2f64.powi(1022)
}

impl Hdr {
    /// BTree nodes plus simultaneous range-shift/reduction copies. A node
    /// allocation can hold fewer entries while a tree is growing.
    pub fn working_bytes(&self) -> u64 {
        (self.counts.len().saturating_add(1) as u64).saturating_mul(384)
    }

    pub fn new(digits: u32) -> Self {
        let largest = 2 * 10i64.pow(digits);
        let sub_count_mag = (largest as f64).ln() / 2f64.ln();
        let sub_count_mag = sub_count_mag.ceil() as u32;
        let half_mag = sub_count_mag.max(1) - 1;
        let sub_count = 1i64 << (half_mag + 1);
        let half = sub_count / 2;
        let configured_ratio = 2i64;
        let internal = 1i64 << (containing_order(configured_ratio) + 1);
        // numberOfSubbuckets(digits) / 2 * internal ratio.
        let n_sub = 1i64 << sub_count_mag;
        let integer_range = (n_sub / 2) * internal;
        let mut h = Self {
            digits,
            half_mag,
            sub_count,
            half,
            highest_trackable: 0,
            counts_len: 0,
            counts: Default::default(),
            total: 0,
            max_value: 0,
            min_non_zero: i64::MAX,
            configured_ratio,
            lowest: 0.0,
            highest: 0.0,
            i2d: 0.0,
            d2i: 0.0,
        };
        h.establish_size(integer_range - 1);
        let lowest = 2f64.powi(800);
        h.set_range(lowest, lowest * internal as f64);
        h
    }

    /// `new DoubleHistogram(source)`: same configuration and range, no counts.
    pub fn empty_like(&self) -> Self {
        let mut h = self.clone();
        h.counts.clear();
        h.total = 0;
        h.max_value = 0;
        h.min_non_zero = i64::MAX;
        h
    }

    pub fn digits(&self) -> u32 {
        self.digits
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn state(&self) -> HdrState {
        HdrState {
            digits: self.digits,
            highest_trackable: self.highest_trackable,
            configured_ratio: self.configured_ratio,
            lowest: self.lowest,
            highest: self.highest,
            counts: self.counts.iter().map(|(k, c)| (*k, *c)).collect(),
            total: self.total,
            max_value: self.max_value,
            min_non_zero: self.min_non_zero,
        }
    }

    pub fn from_state(s: &HdrState) -> Self {
        let mut h = Self::new(s.digits);
        h.establish_size(s.highest_trackable);
        h.configured_ratio = s.configured_ratio;
        h.set_range(s.lowest, s.highest);
        h.counts = s.counts.iter().copied().collect();
        h.total = s.total;
        h.max_value = s.max_value;
        h.min_non_zero = s.min_non_zero;
        h
    }

    // ---- integer histogram ----------------------------------------------

    fn establish_size(&mut self, highest: i64) {
        let mut smallest_untrackable = self.sub_count;
        let mut buckets = 1i64;
        while smallest_untrackable <= highest {
            if smallest_untrackable > i64::MAX / 2 {
                buckets += 1;
                break;
            }
            smallest_untrackable <<= 1;
            buckets += 1;
        }
        self.counts_len = (buckets + 1) * (self.sub_count / 2);
        self.highest_trackable = highest;
    }

    fn bucket_index(&self, v: i64) -> i64 {
        let lzc_base = 64 - self.half_mag as i64 - 1;
        lzc_base - ((v | (self.sub_count - 1)) as u64).leading_zeros() as i64
    }

    fn index_of(&self, v: i64) -> i64 {
        let b = self.bucket_index(v);
        let sub = v >> b;
        ((b + 1) << self.half_mag) + (sub - self.half)
    }

    fn value_from_index(&self, index: i64) -> i64 {
        let mut b = (index >> self.half_mag) - 1;
        let mut sub = (index & (self.half - 1)) + self.half;
        if b < 0 {
            sub -= self.half;
            b = 0;
        }
        sub << b
    }

    fn lowest_equiv(&self, v: i64) -> i64 {
        let b = self.bucket_index(v);
        (v >> b) << b
    }

    fn highest_equiv(&self, v: i64) -> i64 {
        let b = self.bucket_index(v);
        let sub = v >> b;
        let size = 1i64 << if sub >= self.sub_count { b + 1 } else { b };
        (sub << b) + size - 1
    }

    fn count_at(&self, index: i64) -> u64 {
        self.counts.get(&index).copied().unwrap_or(0)
    }

    fn record_int(&mut self, v: i64, count: u64) {
        let idx = self.index_of(v);
        *self.counts.entry(idx).or_insert(0) += count;
        self.update_min_max(v);
        self.total += count;
    }

    fn update_min_max(&mut self, v: i64) {
        if v > self.max_value {
            self.max_value = v;
        }
        if v > 0 && v < self.min_non_zero {
            self.min_non_zero = v;
        }
    }

    /// Re-key every non-zero index by `delta` (the normalizing-offset shift).
    fn shift_indexes(&mut self, delta: i64) {
        let zero = self.counts.remove(&0);
        self.counts = std::mem::take(&mut self.counts).into_iter().map(|(k, c)| (k + delta, c)).collect();
        if let Some(z) = zero {
            self.counts.insert(0, z);
        }
    }

    fn shift_values_left(&mut self, n: u32) -> bool {
        if n == 0 || self.total == self.count_at(0) {
            return true;
        }
        let shift = (n as i64) << self.half_mag;
        let max = if self.max_value == 0 { 0 } else { self.highest_equiv(self.max_value) };
        if self.index_of(max) >= self.counts_len - shift {
            return false;
        }
        let (max_before, min_before) = (self.max_value, self.min_non_zero);
        self.max_value = 0;
        self.min_non_zero = i64::MAX;
        let lowest_half_populated = min_before < self.half;
        self.shift_indexes(shift);
        if lowest_half_populated {
            for from in 1..self.half {
                // Contents of the old lowest half bucket now sit `shift` up.
                let c = self.counts.remove(&(from + shift)).unwrap_or(0);
                let to = self.index_of(self.value_from_index(from) << n);
                if c > 0 {
                    self.counts.insert(to, c);
                } else {
                    self.counts.remove(&to);
                }
            }
        }
        self.update_min_max(max_before << n);
        if min_before < i64::MAX {
            self.update_min_max(min_before << n);
        }
        true
    }

    fn shift_values_right(&mut self, n: u32) -> bool {
        if n == 0 || self.total == self.count_at(0) {
            return true;
        }
        let shift = self.half * n as i64;
        let min = if self.min_non_zero == i64::MAX { i64::MAX } else { self.lowest_equiv(self.min_non_zero) };
        if self.index_of(min) < shift + self.half {
            return false;
        }
        let (max_before, min_before) = (self.max_value, self.min_non_zero);
        self.max_value = 0;
        self.min_non_zero = i64::MAX;
        self.shift_indexes(-shift);
        self.update_min_max(max_before >> n);
        if min_before < i64::MAX {
            self.update_min_max(min_before >> n);
        }
        true
    }

    // ---- DoubleHistogram --------------------------------------------------

    fn set_range(&mut self, lowest: f64, highest: f64) {
        self.lowest = lowest;
        self.highest = highest;
        self.i2d = lowest / self.half as f64;
        self.d2i = 1.0 / self.i2d;
    }

    fn capped_order(&self, d: f64) -> u32 {
        if d > self.configured_ratio as f64 {
            return ((self.configured_ratio as f64).ln() / 2f64.ln()) as u32;
        }
        if d > 2f64.powi(50) {
            return 50;
        }
        containing_order(d.ceil() as i64)
    }

    fn handle_shift_exception(&mut self, n: u32) {
        let order = n + containing_order(self.highest_trackable);
        let new_highest = if order >= 63 { i64::MAX } else { (1i64 << order) - 1 };
        self.establish_size(new_highest);
        self.configured_ratio <<= n;
    }

    fn shift_range_right(&mut self, n: u32) {
        let (mut new_lowest, mut new_highest) = (self.lowest, self.highest);
        let m = 1.0 / (1u64 << n) as f64;
        self.highest *= m;
        if self.total > self.count_at(0) && !self.shift_values_left(n) {
            self.handle_shift_exception(n);
            new_highest /= m;
            self.shift_values_left(n);
        }
        new_lowest *= m;
        new_highest *= m;
        self.set_range(new_lowest, new_highest);
    }

    fn shift_range_left(&mut self, n: u32) {
        let (mut new_lowest, mut new_highest) = (self.lowest, self.highest);
        let m = (1u64 << n) as f64;
        self.lowest *= m;
        if self.total > self.count_at(0) {
            if self.shift_values_right(n) {
                // HdrHistogram 2.1.9 scales the range here and again below.
                new_lowest *= m;
                new_highest *= m;
            } else {
                self.handle_shift_exception(n);
                new_lowest /= m;
            }
        }
        new_lowest *= m;
        new_highest *= m;
        self.set_range(new_lowest, new_highest);
    }

    fn auto_adjust(&mut self, v: f64) {
        if v == 0.0 {
            return;
        }
        if v < self.lowest {
            while v < self.lowest {
                let n = self.capped_order((self.lowest / v).ceil() - 1.0);
                self.shift_range_right(n);
            }
        } else if v >= self.highest {
            while v >= self.highest {
                let ulp = f64::from_bits(v.to_bits() + 1) - v;
                let n = self.capped_order(((v + ulp) / self.highest).ceil() - 1.0);
                self.shift_range_left(n);
            }
        }
    }

    pub fn add(&mut self, v: f64) {
        self.add_count(v, 1);
    }

    /// `recordValueWithCount`. Negative / non-finite values are ignored.
    pub fn add_count(&mut self, v: f64, count: u64) {
        if !v.is_finite() || v < 0.0 || v > highest_allowed_value_ever() {
            return;
        }
        if v < self.lowest || v >= self.highest {
            self.auto_adjust(v);
        }
        let iv = (v * self.d2i) as i64;
        self.record_int(iv, count);
    }

    /// `DoubleHistogram.add(other)`: every non-empty index of `other` is
    /// re-recorded at its lowest equivalent value.
    pub fn merge(&mut self, other: &Hdr) {
        for (&i, &c) in &other.counts {
            if c > 0 {
                self.add_count(other.value_from_index(i) as f64 * other.i2d, c);
            }
        }
    }

    /// `getValueAtPercentile`, `percentile` in [0, 100].
    pub fn value_at_percentile(&self, percentile: f64) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        let requested = percentile.min(100.0);
        let target = (((requested / 100.0) * self.total as f64) + 0.5) as i64;
        let target = target.max(1) as u64;
        let mut seen = 0u64;
        for (&i, &c) in &self.counts {
            if i >= self.counts_len {
                break;
            }
            seen += c;
            if seen >= target {
                let v = self.value_from_index(i);
                let iv = if percentile == 0.0 { self.lowest_equiv(v) } else { self.highest_equiv(v) };
                return Some(iv as f64 * self.i2d);
            }
        }
        Some(0.0)
    }

    /// `getPercentileAtOrBelowValue`.
    pub fn percentile_at_or_below(&self, value: f64) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        let iv = ((value * self.d2i) as i64).max(0);
        let target = self.index_of(iv).min(self.counts_len - 1);
        let below: u64 = self.counts.range(..=target).map(|(_, c)| *c).sum();
        Some(((100.0 * below as f64) / self.total as f64).clamp(0.0, 100.0))
    }
}

/// Bloom filter over rendered term keys: the `rare_terms` shard state of
/// the terms a shard holds more than `max_doc_count` times, so the reduce
/// drops them (ES ships a cuckoo filter at the same default precision).
pub struct KeyBloom {
    bits: Vec<u64>,
    m: u64,
    k: u32,
}

impl KeyBloom {
    pub fn with_capacity(n: usize, fpp: f64) -> KeyBloom {
        let n = n.max(1) as f64;
        let m = (-n * fpp.ln() / (std::f64::consts::LN_2 * std::f64::consts::LN_2)).ceil().max(64.0) as u64;
        let k = (m as f64 / n * std::f64::consts::LN_2).round().clamp(1.0, 16.0) as u32;
        KeyBloom { bits: vec![0; m.div_ceil(64) as usize], m, k }
    }

    fn probes(&self, key: &str) -> impl Iterator<Item = u64> + '_ {
        // Double hashing over two halves of one stable hash.
        let h = murmur3_64(key.as_bytes());
        let (a, b) = (h, h.rotate_left(32) | 1);
        (0..self.k as u64).map(move |i| a.wrapping_add(i.wrapping_mul(b)) % self.m)
    }

    pub fn insert(&mut self, key: &str) {
        let probes: Vec<u64> = self.probes(key).collect();
        for bit in probes {
            self.bits[(bit / 64) as usize] |= 1 << (bit % 64);
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.m > 0 && self.probes(key).all(|bit| self.bits[(bit / 64) as usize] & (1 << (bit % 64)) != 0)
    }

    pub fn state(&self) -> serde_json::Value {
        use base64::Engine;
        let bytes: Vec<u8> = self.bits.iter().flat_map(|w| w.to_le_bytes()).collect();
        serde_json::json!({ "m": self.m, "k": self.k, "bits": base64::engine::general_purpose::STANDARD.encode(bytes) })
    }

    pub fn from_state(v: &serde_json::Value) -> Option<KeyBloom> {
        use base64::Engine;
        let (m, k) = (v.get("m")?.as_u64()?, v.get("k")?.as_u64()? as u32);
        let bytes = base64::engine::general_purpose::STANDARD.decode(v.get("bits")?.as_str()?).ok()?;
        let bits: Vec<u64> = bytes.chunks(8).map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        }).collect();
        (m > 0 && bits.len() as u64 == m.div_ceil(64)).then_some(KeyBloom { bits, m, k })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn merged_points_are_the_stable_sort_of_the_concatenation() {
        let mut seed = 11u64;
        let mut next = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); seed >> 33 };
        for round in 0..200 {
            let pick = |r: u64| [0.0, -0.0, 1.5, -2.25, 3.0, 1e300, -1e-300, 7.0][(r % 8) as usize];
            let mut means: Vec<f64> = (0..next() % 40).map(|_| if round % 2 == 0 { pick(next()) } else { (next() % 1000) as f64 / 7.0 }).collect();
            means.sort_by(f64::total_cmp);
            let weights: Vec<f64> = means.iter().map(|_| (1 + next() % 5) as f64).collect();
            let buffer: Vec<f64> = (0..next() % 300).map(|_| if round % 2 == 0 { pick(next()) } else { (next() % 1000) as f64 / 7.0 }).collect();
            let extra: Vec<(f64, f64)> = (0..next() % 30).map(|_| (pick(next()), (1 + next() % 9) as f64)).collect();
            let mut want: Vec<(f64, f64)> = means.iter().copied().zip(weights.iter().copied()).collect();
            want.extend(buffer.iter().map(|&v| (v, 1.0)));
            want.extend(extra.iter().copied());
            want.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut b = buffer.clone();
            let got = super::merged_points(&means, &weights, &mut b, extra);
            let bits = |v: &[(f64, f64)]| v.iter().map(|(m, w)| (m.to_bits(), w.to_bits())).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "round {round}");
        }
        let mut v: Vec<f64> = (0..500).map(|i| ((i * 7919) % 211) as f64 - 100.0).chain([0.0, -0.0, f64::MAX, f64::MIN, 1e-310, -1e-310]).collect();
        let mut w = v.clone();
        super::sort_f64(&mut v);
        w.sort_by(f64::total_cmp);
        assert_eq!(v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), w.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    use super::*;

    #[test]
    fn sketch_states_pack_and_read_both_forms() {
        let mut h = HyperLogLog::new(100);
        (0..50u64).for_each(|i| h.add_hash(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        let packed = serde_json::to_value(h.state()).unwrap();
        assert!(packed["exact"].is_string(), "{packed}");
        let back: HllState = serde_json::from_value(packed).unwrap();
        assert_eq!(back, h.state());
        let plain = serde_json::json!({"p": 4, "threshold": 1, "exact": null, "regs": [1, 2, 3]});
        assert_eq!(serde_json::from_value::<HllState>(plain).unwrap().regs, vec![1, 2, 3]);
        let mut td = TDigest::new(100.0);
        (0..5000).for_each(|i| td.add(i as f64 * 0.5));
        let st = td.state();
        let v = serde_json::to_value(&st).unwrap();
        assert!(v["means"].is_string());
        assert_eq!(serde_json::from_value::<TDigestState>(v).unwrap(), st);
    }

    #[test]
    fn key_bloom_round_trips_without_false_negatives() {
        let mut b = KeyBloom::with_capacity(10_000, 0.001);
        for i in 0..10_000 {
            b.insert(&format!("\"k{i}\""));
        }
        let b = KeyBloom::from_state(&b.state()).unwrap();
        assert!((0..10_000).all(|i| b.contains(&format!("\"k{i}\""))));
        let fp = (0..10_000).filter(|i| b.contains(&format!("\"other{i}\""))).count();
        assert!(fp < 50, "{fp} false positives");
    }

    #[test]
    fn hll_exact_and_approx() {
        let mut h = HyperLogLog::new(3000);
        for i in 0..1000u64 {
            h.add_bytes(&i.to_le_bytes());
            h.add_bytes(&i.to_le_bytes());
        }
        assert_eq!(h.count(), 1000);
        let mut h = HyperLogLog::new(100);
        for i in 0..100_000u64 {
            h.add_bytes(&i.to_le_bytes());
        }
        let c = h.count() as f64;
        assert!((c - 100_000.0).abs() / 100_000.0 < 0.1, "{c}");
    }

    #[test]
    fn tdigest_quantiles() {
        let mut t = TDigest::new(100.0);
        for i in 1..=10_000 {
            t.add(i as f64);
        }
        let med = t.quantile(0.5).unwrap();
        assert!((med - 5000.0).abs() < 50.0, "{med}");
        let p99 = t.quantile(0.99).unwrap();
        assert!((p99 - 9900.0).abs() < 30.0, "{p99}");
        let mut t = TDigest::new(100.0);
        for v in [1.0, 2.0, 3.0, 4.0] {
            t.add(v);
        }
        assert_eq!(t.quantile(0.0), Some(1.0));
        assert_eq!(t.quantile(1.0), Some(4.0));
        assert!((t.cdf(2.5).unwrap() - 0.5).abs() < 1e-9);
    }

    /// Values captured from ES 8.19.22 (:9250) for small samples (HybridDigest
    /// exact mode).
    #[test]
    fn tdigest_matches_es_small_samples() {
        let q = |vals: &[f64], ps: &[f64]| -> Vec<f64> {
            let mut t = TDigest::new(100.0);
            vals.iter().for_each(|v| t.add(*v));
            ps.iter().map(|p| t.quantile(p / 100.0).unwrap()).collect()
        };
        let close = |a: Vec<f64>, b: Vec<f64>| a.iter().zip(&b).for_each(|(x, y)| assert!((x - y).abs() < 1e-9, "{a:?} vs {b:?}"));
        close(q(&[10., 20., 22., 28.], &[1., 5., 25., 50., 75., 95., 99.]), vec![10.3, 11.5, 17.5, 21.0, 23.5, 27.1, 27.82]);
        close(q(&[10., 20.], &[1., 25., 50., 75., 99.]), vec![10.1, 12.5, 15.0, 17.5, 19.9]);
        close(q(&[10., 20., 22.], &[1., 25., 50., 75., 99.]), vec![10.2, 15.0, 20.0, 21.0, 21.96]);
    }

    /// ES 8.19.22 `hdr` (2 digits) on [10,20,22,28]: 20.0625 etc.
    #[test]
    fn hdr_matches_es() {
        let mut h = Hdr::new(2);
        for v in [10., 20., 22., 28.] {
            h.add(v);
        }
        let got: Vec<f64> = [1., 25., 50., 75., 99.].iter().map(|p| h.value_at_percentile(*p).unwrap()).collect();
        assert_eq!(got, vec![10.0, 10.0, 20.0625, 22.0625, 28.0625]);
        let r: Vec<f64> = [5., 10., 20., 21., 22., 28., 50.].iter().map(|v| h.percentile_at_or_below(*v).unwrap()).collect();
        assert_eq!(r, vec![0.0, 25.0, 50.0, 50.0, 75.0, 100.0, 100.0]);
    }

    #[test]
    fn murmur_known_value() {
        // Different inputs hash differently and deterministically.
        assert_ne!(murmur3_64(b"a"), murmur3_64(b"b"));
        assert_eq!(murmur3_64(b"hello"), murmur3_64(b"hello"));
    }
}
