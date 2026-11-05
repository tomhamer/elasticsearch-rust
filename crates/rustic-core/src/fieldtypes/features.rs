//! Feature doc values (ES `rank_feature`, `rank_features`, `sparse_vector`):
//! one `token\0hexbits` string per feature, plus the scoring functions of
//! the `rank_feature` query.

/// `token\0` + hex bits of the f32 weight.
pub fn encode(token: &str, weight: f32) -> String {
    format!("{token}\u{0}{:08x}", weight.to_bits())
}

/// Inverse of [`encode`].
pub fn decode(s: &str) -> Option<(&str, f32)> {
    let (t, w) = s.split_once('\u{0}')?;
    Some((t, f32::from_bits(u32::from_str_radix(w, 16).ok()?)))
}

/// Prefix of every encoded value of `token`.
pub fn token_prefix(token: &str) -> String {
    format!("{token}\u{0}")
}

/// Lucene `FeatureField` keeps 9 significant bits of the weight; mirror it
/// so scores match.
pub fn quantize(w: f32) -> f32 {
    let bits = w.to_bits() >> 15;
    f32::from_bits(bits << 15)
}

/// The term frequency a feature of weight `w` is indexed with (Lucene
/// `FeatureField`: the weight's top 16 bits; at most `0x7F7F`, so it never
/// reaches the postings' "unbounded" block-max sentinel). Weights too
/// small to keep a bit index as the smallest frequency.
pub fn weight_to_freq(w: f32) -> u32 {
    (w.to_bits() >> 15).max(1)
}

/// Inverse of [`weight_to_freq`] (the [`quantize`]d weight).
#[inline]
pub fn freq_to_weight(freq: u32) -> f32 {
    f32::from_bits(freq << 15)
}

/// `rank_feature` scoring functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureFunction {
    /// `S / (S + pivot)`.
    Saturation { pivot: f32 },
    /// `ln(scaling_factor + S)`.
    Log { scaling_factor: f32 },
    /// `S^exp / (S^exp + pivot^exp)`.
    Sigmoid { pivot: f32, exponent: f32 },
    /// `S`.
    Linear,
}

impl FeatureFunction {
    pub fn score(&self, s: f32) -> f32 {
        match *self {
            FeatureFunction::Saturation { pivot } => s / (s + pivot),
            FeatureFunction::Log { scaling_factor } => (scaling_factor + s).ln(),
            FeatureFunction::Sigmoid { pivot, exponent } => {
                let a = s.powf(exponent);
                a / (a + pivot.powf(exponent))
            }
            FeatureFunction::Linear => s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_functions() {
        let e = encode("running", 2.5);
        assert_eq!(decode(&e), Some(("running", 2.5)));
        assert!(e.starts_with(&token_prefix("running")));
        assert!((quantize(2.4097164) - 2.4097164).abs() < 0.01);
        for w in [2.4097164f32, 1e-3, 0.5, 1.0, 40.0, f32::MAX] {
            assert_eq!(freq_to_weight(weight_to_freq(w)), quantize(w));
            assert!(weight_to_freq(w) < 0xFFFF);
        }
        assert_eq!(FeatureFunction::Saturation { pivot: 2.0 }.score(2.0), 0.5);
        assert_eq!(FeatureFunction::Linear.score(3.0), 3.0);
        assert!((FeatureFunction::Log { scaling_factor: 1.0 }.score(std::f32::consts::E - 1.0) - 1.0).abs() < 1e-6);
        assert_eq!(FeatureFunction::Sigmoid { pivot: 2.0, exponent: 1.0 }.score(2.0), 0.5);
    }
}
