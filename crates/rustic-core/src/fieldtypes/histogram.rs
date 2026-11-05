//! Pre-aggregated histogram doc values (ES `histogram`, and the
//! representative-value form of `tdigest` / `exponential_histogram`):
//! parallel `values[]` (strictly ascending) and `counts[]`.

/// Why a histogram value is invalid (ES error texts).
pub fn validate(values: &[f64], counts: &[i64]) -> Result<(), String> {
    if values.len() != counts.len() {
        return Err(format!("expected same length from [values] and [counts] but got [{} != {}]", values.len(), counts.len()));
    }
    for w in values.windows(2) {
        if w[1] <= w[0] {
            return Err(format!("[values] values must be in increasing order, got [{}] but previous value was [{}]", fmt(w[1]), fmt(w[0])));
        }
    }
    for c in counts {
        if *c < 0 {
            return Err(format!("[counts] elements must be >= 0 but got {c}"));
        }
    }
    Ok(())
}

fn fmt(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{x:.1}")
    } else {
        x.to_string()
    }
}

/// Doc-value string: `bits:count;...` (exact f64 bits in hex).
pub fn encode(values: &[f64], counts: &[i64]) -> String {
    let mut s = String::new();
    for (v, c) in values.iter().zip(counts) {
        if *c == 0 {
            continue;
        }
        if !s.is_empty() {
            s.push(';');
        }
        s.push_str(&format!("{:016x}:{c}", v.to_bits()));
    }
    s
}

/// Inverse of [`encode`].
pub fn decode(s: &str) -> Vec<(f64, u64)> {
    s.split(';')
        .filter(|p| !p.is_empty())
        .filter_map(|p| {
            let (v, c) = p.split_once(':')?;
            Some((f64::from_bits(u64::from_str_radix(v, 16).ok()?), c.parse().ok()?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_validate() {
        let s = encode(&[0.1, 0.25, 3.0], &[3, 0, 7]);
        assert_eq!(decode(&s), vec![(0.1, 3), (3.0, 7)]);
        assert!(validate(&[1.0, 2.0], &[1]).unwrap_err().contains("[2 != 1]"));
        assert!(validate(&[2.0, 1.0], &[1, 1]).unwrap_err().contains("increasing order"));
        assert!(validate(&[1.0], &[-1]).is_err());
    }
}
