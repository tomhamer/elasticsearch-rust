//! Numeric encodings: `unsigned_long` (offset by -2^63 into a sortable
//! signed long, ES `UnsignedLongFieldMapper`), `scaled_float` (saturating
//! `round(v * factor)` longs) and `date_nanos` (epoch nanoseconds, valid
//! 1970-01-01 .. 2262-04-11T23:47:16.854775807Z).

/// `u64` → order-preserving `i64` (ES stores `value - 2^63`).
pub fn u64_to_sortable(v: u64) -> i64 {
    (v ^ (1u64 << 63)) as i64
}

/// Inverse of [`u64_to_sortable`].
pub fn sortable_to_u64(v: i64) -> u64 {
    (v as u64) ^ (1u64 << 63)
}

/// `scaled_float` encoding: `round(v * factor)` saturated to the long range.
pub fn scaled_encode(v: f64, factor: f64) -> i64 {
    // Java `Math.round`: half rounds toward positive infinity.
    let x = (v * factor + 0.5).floor();
    if x >= i64::MAX as f64 {
        i64::MAX
    } else if x <= i64::MIN as f64 {
        i64::MIN
    } else {
        x as i64
    }
}

/// `scaled_float` decoding (what queries, aggs and doc values see).
pub fn scaled_decode(v: i64, factor: f64) -> f64 {
    v as f64 / factor
}

/// `scaled_float` value as ES returns it for a source value.
pub fn scaled_round_trip(v: f64, factor: f64) -> f64 {
    scaled_decode(scaled_encode(v, factor), factor)
}

/// Why a date cannot be stored with nanosecond resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanosError {
    BeforeEpoch,
    AfterMax,
}

pub const NANOS_MAX_DISPLAY: &str = "2262-04-11T23:47:16.854775807";

/// Epoch nanoseconds from whole epoch seconds plus the nanoseconds within
/// that second.
pub fn nanos_from_parts(epoch_seconds: i64, nanos_of_second: u32) -> Result<i64, NanosError> {
    if epoch_seconds < 0 {
        return Err(NanosError::BeforeEpoch);
    }
    epoch_seconds
        .checked_mul(1_000_000_000)
        .and_then(|s| s.checked_add(nanos_of_second as i64))
        .ok_or(NanosError::AfterMax)
}

/// Epoch nanoseconds from epoch millis plus extra nanos below the milli.
pub fn nanos_from_millis(millis: i64, sub_milli_nanos: u32) -> Result<i64, NanosError> {
    if millis < 0 {
        return Err(NanosError::BeforeEpoch);
    }
    millis
        .checked_mul(1_000_000)
        .and_then(|s| s.checked_add(sub_milli_nanos as i64))
        .ok_or(NanosError::AfterMax)
}

/// Epoch millis of epoch nanos (floor).
pub fn nanos_to_millis(nanos: i64) -> i64 {
    nanos.div_euclid(1_000_000)
}

/// The fractional-second digits of an ISO-8601-ish date string as
/// nanoseconds (`...12:12:12.123456789Z` → 123456789), if present.
pub fn fraction_nanos(s: &str) -> Option<u32> {
    let b = s.as_bytes();
    // Find "ss.fff" after a "HH:mm:" pattern: the last ':' followed by two
    // digits and a '.' or ','.
    let colon = s.rfind(':')?;
    let rest = &b[colon + 1..];
    if rest.len() < 4 || !rest[0].is_ascii_digit() || !rest[1].is_ascii_digit() || (rest[2] != b'.' && rest[2] != b',') {
        return None;
    }
    let digits: Vec<u8> = rest[3..].iter().take_while(|c| c.is_ascii_digit()).copied().collect();
    if digits.is_empty() {
        return None;
    }
    let mut n: u32 = 0;
    for i in 0..9 {
        n = n * 10 + digits.get(i).map(|d| (d - b'0') as u32).unwrap_or(0);
    }
    Some(n)
}

/// `yyyy-MM-ddTHH:mm:ss.nnnnnnnnnZ` (trailing zero groups trimmed to
/// millis like Java's `strict_date_optional_time_nanos` printer).
pub fn format_nanos(nanos: i64) -> String {
    let secs = nanos.div_euclid(1_000_000_000);
    let frac = nanos.rem_euclid(1_000_000_000) as u32;
    let dt = chrono::DateTime::from_timestamp(secs, frac).unwrap_or_default();
    let base = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
    let mut f = format!("{frac:09}");
    while f.len() > 3 && f.ends_with('0') {
        f.pop();
    }
    format!("{base}.{f}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_order() {
        let vals = [0u64, 1, 1 << 62, (1 << 63) - 1, 1 << 63, u64::MAX];
        for w in vals.windows(2) {
            assert!(u64_to_sortable(w[0]) < u64_to_sortable(w[1]));
        }
        for v in vals {
            assert_eq!(sortable_to_u64(u64_to_sortable(v)), v);
        }
        assert_eq!(u64_to_sortable(0), i64::MIN);
    }

    #[test]
    fn scaled() {
        assert_eq!(scaled_encode(1.26, 10.0), 13);
        assert_eq!(scaled_round_trip(1.26, 10.0), 1.3);
        assert_eq!(scaled_encode(1e300, 100.0), i64::MAX);
        assert_eq!(scaled_encode(-1e300, 100.0), i64::MIN);
    }

    #[test]
    fn nanos() {
        assert_eq!(fraction_nanos("2018-10-29T12:12:12.123456789Z"), Some(123456789));
        assert_eq!(fraction_nanos("2018-10-29T12:12:12.5Z"), Some(500000000));
        assert_eq!(fraction_nanos("2018-10-29T12:12:12Z"), None);
        assert_eq!(fraction_nanos("2018-10-29"), None);
        assert_eq!(nanos_from_parts(1540815132, 123456789), Ok(1540815132123456789));
        assert_eq!(nanos_from_parts(-1, 0), Err(NanosError::BeforeEpoch));
        assert_eq!(nanos_from_parts(9_300_000_000, 0), Err(NanosError::AfterMax));
        assert_eq!(format_nanos(1540815132123456789), "2018-10-29T12:12:12.123456789Z");
        assert_eq!(format_nanos(1540815132120000000), "2018-10-29T12:12:12.120Z");
        assert_eq!(nanos_to_millis(1540815132123456789), 1540815132123);
    }
}
