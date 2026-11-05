//! Range field encoding (ES `integer_range`, `long_range`, `float_range`,
//! `double_range`, `date_range`, `ip_range`): a stored range is a 2-D point
//! `(min, max)`; both ends are mapped to order-preserving `u128` keys so one
//! set of relation tests serves every kind. Doc values hold one 64-hex-char
//! string per range (min key ‖ max key), which sorts by `min` then `max`.

/// The kind of a range field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeKind {
    Integer,
    Long,
    Float,
    Double,
    Date,
    Ip,
}

impl RangeKind {
    pub fn parse(es_type: &str) -> Option<RangeKind> {
        Some(match es_type {
            "integer_range" => RangeKind::Integer,
            "long_range" => RangeKind::Long,
            "float_range" => RangeKind::Float,
            "double_range" => RangeKind::Double,
            "date_range" => RangeKind::Date,
            "ip_range" => RangeKind::Ip,
            _ => return None,
        })
    }

    pub fn is_integral(self) -> bool {
        matches!(self, RangeKind::Integer | RangeKind::Long | RangeKind::Date)
    }

    /// Smallest / largest representable key of the kind.
    pub fn min_key(self) -> u128 {
        match self {
            RangeKind::Integer => key_i64(i32::MIN as i64),
            RangeKind::Long | RangeKind::Date => key_i64(i64::MIN),
            RangeKind::Float => key_f64(f32::NEG_INFINITY as f64),
            RangeKind::Double => key_f64(f64::NEG_INFINITY),
            RangeKind::Ip => 0,
        }
    }

    pub fn max_key(self) -> u128 {
        match self {
            RangeKind::Integer => key_i64(i32::MAX as i64),
            RangeKind::Long | RangeKind::Date => key_i64(i64::MAX),
            RangeKind::Float => key_f64(f32::INFINITY as f64),
            RangeKind::Double => key_f64(f64::INFINITY),
            RangeKind::Ip => u128::MAX,
        }
    }

    /// Key of the next representable value (`gt` → inclusive).
    pub fn next_up(self, k: u128) -> u128 {
        match self {
            RangeKind::Float => key_f64(next_up_f32(f64_of_key(k) as f32) as f64),
            RangeKind::Double => key_f64(next_up_f64(f64_of_key(k))),
            _ => k.saturating_add(1),
        }
    }

    /// Key of the previous representable value (`lt` → inclusive).
    pub fn next_down(self, k: u128) -> u128 {
        match self {
            RangeKind::Float => key_f64(-next_up_f32(-(f64_of_key(k) as f32)) as f64),
            RangeKind::Double => key_f64(-next_up_f64(-f64_of_key(k))),
            _ => k.saturating_sub(1),
        }
    }
}

fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let b = x.to_bits();
    f64::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let b = x.to_bits();
    f32::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

/// Order-preserving key of a signed long.
pub fn key_i64(v: i64) -> u128 {
    ((v as u64) ^ (1u64 << 63)) as u128
}

pub fn i64_of_key(k: u128) -> i64 {
    ((k as u64) ^ (1u64 << 63)) as i64
}

/// Order-preserving key of a double (NaN sorts last).
pub fn key_f64(x: f64) -> u128 {
    let b = x.to_bits();
    (if b >> 63 == 1 { !b } else { b | (1 << 63) }) as u128
}

pub fn f64_of_key(k: u128) -> f64 {
    let b = k as u64;
    f64::from_bits(if b >> 63 == 1 { b & !(1 << 63) } else { !b })
}

/// Key of a 16-byte IP.
pub fn key_ip(b: [u8; 16]) -> u128 {
    u128::from_be_bytes(b)
}

pub fn ip_of_key(k: u128) -> [u8; 16] {
    k.to_be_bytes()
}

/// Doc-value string of a stored range.
pub fn encode(min: u128, max: u128) -> String {
    format!("{min:032x}{max:032x}")
}

/// Inverse of [`encode`].
pub fn decode(s: &str) -> Option<(u128, u128)> {
    if s.len() != 64 {
        return None;
    }
    Some((u128::from_str_radix(&s[..32], 16).ok()?, u128::from_str_radix(&s[32..], 16).ok()?))
}

/// Relation between a stored range and a query range (ES `relation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    Intersects,
    /// The stored range contains the query range.
    Contains,
    /// The stored range is within the query range.
    Within,
    Disjoint,
}

impl Relation {
    pub fn parse(s: &str) -> Option<Relation> {
        Some(match s.to_ascii_lowercase().as_str() {
            "intersects" => Relation::Intersects,
            "contains" => Relation::Contains,
            "within" => Relation::Within,
            "disjoint" => Relation::Disjoint,
            _ => return None,
        })
    }
}

/// Does `doc` (inclusive min, max) stand in `rel` to `query`?
pub fn matches(doc: (u128, u128), query: (u128, u128), rel: Relation) -> bool {
    let intersects = doc.0 <= query.1 && doc.1 >= query.0;
    match rel {
        Relation::Intersects => intersects,
        Relation::Disjoint => !intersects,
        Relation::Contains => doc.0 <= query.0 && doc.1 >= query.1,
        Relation::Within => doc.0 >= query.0 && doc.1 <= query.1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_order() {
        let xs = [i64::MIN, -5, 0, 3, i64::MAX];
        for w in xs.windows(2) {
            assert!(key_i64(w[0]) < key_i64(w[1]));
        }
        for x in xs {
            assert_eq!(i64_of_key(key_i64(x)), x);
        }
        let fs = [f64::NEG_INFINITY, -1.5, -0.0, 0.0, 1e-300, 2.5, f64::INFINITY];
        for w in fs.windows(2) {
            assert!(key_f64(w[0]) <= key_f64(w[1]), "{:?}", w);
        }
        assert_eq!(f64_of_key(key_f64(2.5)), 2.5);
        assert_eq!(f64_of_key(key_f64(-2.5)), -2.5);
        let e = encode(key_i64(1), key_i64(5));
        assert_eq!(decode(&e), Some((key_i64(1), key_i64(5))));
    }

    #[test]
    fn relations() {
        let d = (key_i64(1), key_i64(5));
        let q = (key_i64(3), key_i64(4));
        assert!(matches(d, q, Relation::Intersects));
        assert!(matches(d, q, Relation::Contains));
        assert!(!matches(d, q, Relation::Within));
        assert!(!matches(d, q, Relation::Disjoint));
        assert!(matches(q, d, Relation::Within));
        assert_eq!(RangeKind::Long.next_up(key_i64(3)), key_i64(4));
        assert_eq!(f64_of_key(RangeKind::Double.next_up(key_f64(1.0))), 1.0000000000000002);
    }
}
