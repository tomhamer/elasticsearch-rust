//! IP addresses as 16-byte sortable points (Lucene `InetAddressPoint`):
//! IPv4 is mapped into IPv6 (`::ffff:a.b.c.d`), so byte order equals
//! address order for both families. CIDR blocks become inclusive ranges.

use std::net::{IpAddr, Ipv6Addr};

/// Encode an address (IPv4 mapped to IPv6).
pub fn encode(ip: IpAddr) -> [u8; 16] {
    match ip {
        IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        IpAddr::V6(v6) => v6.octets(),
    }
}

/// Decode, unmapping IPv4-mapped addresses.
pub fn decode(b: [u8; 16]) -> IpAddr {
    let v6 = Ipv6Addr::from(b);
    match v6.to_ipv4_mapped() {
        Some(v4) => IpAddr::V4(v4),
        None => IpAddr::V6(v6),
    }
}

/// Parse an IP string literal.
pub fn parse(s: &str) -> Option<[u8; 16]> {
    s.trim().parse::<IpAddr>().ok().map(encode)
}

/// `a.b.c.d/n` or `v6/n` → inclusive `[lo, hi]`.
pub fn cidr(s: &str) -> Option<([u8; 16], [u8; 16])> {
    let (addr, bits) = s.split_once('/')?;
    let bits: u32 = bits.trim().parse().ok()?;
    let ip: IpAddr = addr.trim().parse().ok()?;
    let prefix = match ip {
        IpAddr::V4(_) if bits <= 32 => bits + 96,
        IpAddr::V6(_) if bits <= 128 => bits,
        _ => return None,
    };
    let n = u128::from_be_bytes(encode(ip));
    let mask = if prefix == 0 { 0 } else { u128::MAX << (128 - prefix) };
    Some(((n & mask).to_be_bytes(), (n | !mask).to_be_bytes()))
}

/// The address as an integer (for arithmetic on range bounds).
pub fn to_u128(b: [u8; 16]) -> u128 {
    u128::from_be_bytes(b)
}

pub fn from_u128(v: u128) -> [u8; 16] {
    v.to_be_bytes()
}

/// Display form (`10.0.0.1`, `::1`).
pub fn display(b: [u8; 16]) -> String {
    decode(b).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_and_cidr() {
        let a = parse("10.0.0.1").unwrap();
        let b = parse("10.0.0.2").unwrap();
        let c = parse("192.168.0.1").unwrap();
        assert!(a < b && b < c);
        assert_eq!(display(a), "10.0.0.1");
        assert_eq!(display(parse("::1").unwrap()), "::1");
        let (lo, hi) = cidr("192.168.0.0/16").unwrap();
        assert!(lo <= c && c <= hi);
        assert!(!(lo <= a && a <= hi));
        assert_eq!(display(lo), "192.168.0.0");
        assert_eq!(display(hi), "192.168.255.255");
        assert!(cidr("10.0.0.0/33").is_none());
        let (lo, hi) = cidr("2001:db8::/32").unwrap();
        assert!(lo < hi && display(lo) == "2001:db8::");
    }
}
