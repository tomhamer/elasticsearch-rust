//! Helpers for processors: message digests (no crypto crate in the tree),
//! user-agent parsing, byte sizes, IP classification, HTML stripping.

use serde_json::{json, Map, Value};

// ------------------------------------------------------------ digests

pub fn md5(input: &[u8]) -> Vec<u8> {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = input.to_vec();
    let bit_len = (input.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk.chunks(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    [a0, b0, c0, d0].iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn pad_be(input: &[u8], block: usize, len_bytes: usize) -> Vec<u8> {
    let mut msg = input.to_vec();
    let bit_len = (input.len() as u128).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % block != block - len_bytes {
        msg.push(0);
    }
    let lb = bit_len.to_be_bytes();
    msg.extend_from_slice(&lb[16 - len_bytes..]);
    msg
}

pub fn sha1(input: &[u8]) -> Vec<u8> {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    for chunk in pad_be(input, 64, 8).chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i / 20 {
                0 => ((b & c) | (!b & d), 0x5A827999),
                1 => (b ^ c ^ d, 0x6ED9EBA1),
                2 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().flat_map(|x| x.to_be_bytes()).collect()
}

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub fn sha256(input: &[u8]) -> Vec<u8> {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    for chunk in pad_be(input, 64, 8).chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K256[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (x, y) in h.iter_mut().zip(v) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().flat_map(|x| x.to_be_bytes()).collect()
}

const K512: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242,
    0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f,
    0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
    0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec, 0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915,
    0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

pub fn sha512(input: &[u8]) -> Vec<u8> {
    let mut h: [u64; 8] = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179];
    for chunk in pad_be(input, 128, 16).chunks(128) {
        let mut w = [0u64; 80];
        for i in 0..16 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&chunk[8 * i..8 * i + 8]);
            w[i] = u64::from_be_bytes(b);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..80 {
            let s1 = v[4].rotate_right(14) ^ v[4].rotate_right(18) ^ v[4].rotate_right(41);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K512[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(28) ^ v[0].rotate_right(34) ^ v[0].rotate_right(39);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (x, y) in h.iter_mut().zip(v) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().flat_map(|x| x.to_be_bytes()).collect()
}

/// Murmur3 x64 128-bit (seed 0), the ES `MurmurHash3` fingerprint method.
pub fn murmur3_128(data: &[u8]) -> Vec<u8> {
    let (c1, c2) = (0x87c37b91114253d5u64, 0x4cf5ad432745937fu64);
    let (mut h1, mut h2) = (0u64, 0u64);
    let nblocks = data.len() / 16;
    for i in 0..nblocks {
        let mut a = [0u8; 8];
        a.copy_from_slice(&data[i * 16..i * 16 + 8]);
        let mut k1 = u64::from_le_bytes(a);
        a.copy_from_slice(&data[i * 16 + 8..i * 16 + 16]);
        let mut k2 = u64::from_le_bytes(a);
        k1 = k1.wrapping_mul(c1).rotate_left(31).wrapping_mul(c2);
        h1 ^= k1;
        h1 = h1.rotate_left(27).wrapping_add(h2).wrapping_mul(5).wrapping_add(0x52dce729);
        k2 = k2.wrapping_mul(c2).rotate_left(33).wrapping_mul(c1);
        h2 ^= k2;
        h2 = h2.rotate_left(31).wrapping_add(h1).wrapping_mul(5).wrapping_add(0x38495ab5);
    }
    let tail = &data[nblocks * 16..];
    let (mut k1, mut k2) = (0u64, 0u64);
    for (i, b) in tail.iter().enumerate() {
        if i < 8 {
            k1 ^= (*b as u64) << (8 * i);
        } else {
            k2 ^= (*b as u64) << (8 * (i - 8));
        }
    }
    if tail.len() > 8 {
        k2 = k2.wrapping_mul(c2).rotate_left(33).wrapping_mul(c1);
        h2 ^= k2;
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(c1).rotate_left(31).wrapping_mul(c2);
        h1 ^= k1;
    }
    h1 ^= data.len() as u64;
    h2 ^= data.len() as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let fmix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51afd7ed558ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
        k ^ (k >> 33)
    };
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let mut out = h1.to_le_bytes().to_vec();
    out.extend_from_slice(&h2.to_le_bytes());
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ------------------------------------------------------------ byte sizes

/// ES `ByteSizeValue` parsing (`1kb`, `1.5 MB`, `512`).
pub fn parse_bytes(s: &str) -> Option<i64> {
    let t = s.trim().to_lowercase();
    let units: [(&str, f64); 7] = [("pb", 1125899906842624.0), ("tb", 1099511627776.0), ("gb", 1073741824.0), ("mb", 1048576.0), ("kb", 1024.0), ("b", 1.0), ("", 1.0)];
    for (u, m) in units {
        if let Some(num) = t.strip_suffix(u) {
            let num = num.trim();
            if num.is_empty() {
                return None;
            }
            if u.is_empty() && num.parse::<i64>().is_err() {
                return None;
            }
            let v: f64 = num.parse().ok()?;
            if v < 0.0 {
                return None;
            }
            return Some((v * m) as i64);
        }
    }
    None
}

/// ES `ByteSizeValue.parseBytesSizeValue` with its exact errors: integer
/// or up-to-two-decimal amounts of a unit (rounded up to whole bytes),
/// plain `b` amounts integer only, `0` / `-1` without a unit.
pub fn parse_byte_size_value(s: &str, setting: &str) -> crate::error::Result<i64> {
    use crate::error::EsError;
    let pe = |m: String| EsError::new(400, "parse_exception", m);
    let lower = s.trim().to_lowercase();
    if lower == "-1" || lower == "0" {
        return Ok(lower.parse().unwrap());
    }
    const UNITS: [(&str, i128); 10] = [
        ("kb", 1 << 10), ("k", 1 << 10), ("mb", 1 << 20), ("m", 1 << 20), ("gb", 1 << 30), ("g", 1 << 30),
        ("tb", 1 << 40), ("t", 1 << 40), ("pb", 1 << 50), ("p", 1 << 50),
    ];
    let as_size = |cause: EsError| pe(format!("failed to parse setting [{setting}] with value [{s}] as a size in bytes")).caused_by(cause);
    let Some((suffix, mult)) = UNITS.iter().find(|(u, _)| lower.ends_with(u)).copied().or_else(|| lower.ends_with('b').then_some(("b", 1))) else {
        return Err(pe(format!("failed to parse setting [{setting}] with value [{s}] as a size in bytes: unit is missing or unrecognized")));
    };
    let num = lower[..lower.len() - suffix.len()].trim();
    if let Ok(n) = num.parse::<i64>() {
        let bytes = n as i128 * mult;
        if bytes < -1 {
            return Err(as_size(EsError::illegal_argument(format!("Values less than -1 bytes are not supported: {num}{suffix}"))));
        }
        if bytes > i64::MAX as i128 {
            return Err(as_size(EsError::illegal_argument(format!("Values greater than {} bytes are not supported: {num}{suffix}", i64::MAX))));
        }
        return Ok(bytes as i64);
    }
    if suffix == "b" {
        return Err(pe(format!("failed to parse setting [{setting}] with value [{s}]"))
            .caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{num}\""))));
    }
    // BigDecimal path: [+-]digits[.digits]
    let body = num.strip_prefix('+').unwrap_or(num);
    let (neg, body) = match body.strip_prefix('-') { Some(b) => (true, b), None => (false, body) };
    let (ip, fp) = body.split_once('.').unwrap_or((body, ""));
    let valid = !(ip.is_empty() && fp.is_empty()) && ip.bytes().all(|c| c.is_ascii_digit()) && fp.bytes().all(|c| c.is_ascii_digit());
    if !valid {
        return Err(pe(format!("failed to parse setting [{setting}] with value [{s}]")));
    }
    let digits = format!("{ip}{fp}");
    let mag: i128 = digits.trim_start_matches('0').parse::<i128>().unwrap_or(0);
    if neg && mag != 0 {
        return Err(pe(format!("failed to parse setting [{setting}] with value [{s}]")));
    }
    if fp.len() > 2 {
        return Err(pe(format!("failed to parse setting [{setting}] with more than two decimals in value [{s}]")));
    }
    let scale = 10i128.pow(fp.len() as u32);
    let bytes = (mag.saturating_mul(mult) + scale - 1) / scale;
    if digits.len() > 30 || bytes > i64::MAX as i128 {
        return Err(pe(format!("failed to parse setting [{setting}] with value beyond {}: [{s}]", i64::MAX)));
    }
    Ok(bytes as i64)
}

// ------------------------------------------------------------ IPs

pub fn parse_ip(s: &str) -> Option<std::net::IpAddr> {
    s.trim().trim_start_matches('[').trim_end_matches(']').parse().ok()
}

/// Private / loopback / link-local / unspecified (no geo data).
pub fn is_internal_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v) => v.is_private() || v.is_loopback() || v.is_link_local() || v.is_unspecified() || v.is_broadcast() || v.octets()[0] >= 224,
        std::net::IpAddr::V6(v) => v.is_loopback() || v.is_unspecified() || (v.segments()[0] & 0xfe00) == 0xfc00 || (v.segments()[0] & 0xffc0) == 0xfe80,
    }
}

/// `ip` in `cidr` (or one of the ES named ranges).
pub fn ip_in_network(ip: &std::net::IpAddr, net: &str) -> bool {
    use std::net::IpAddr;
    match net {
        "loopback" => return ip.is_loopback(),
        "unicast" | "global_unicast" => {
            return !ip.is_multicast() && !ip.is_unspecified() && !ip.is_loopback() && !matches!(ip, IpAddr::V4(v) if v.is_broadcast() || v.is_link_local());
        }
        "multicast" => return ip.is_multicast(),
        "link_local_unicast" => return matches!(ip, IpAddr::V4(v) if v.is_link_local()) || matches!(ip, IpAddr::V6(v) if (v.segments()[0] & 0xffc0) == 0xfe80),
        "link_local_multicast" => return matches!(ip, IpAddr::V4(v) if v.octets()[0] == 224 && v.octets()[1] == 0 && v.octets()[2] == 0) || matches!(ip, IpAddr::V6(v) if (v.segments()[0] & 0xff0f) == 0xff02),
        "interface_local_multicast" => return matches!(ip, IpAddr::V6(v) if (v.segments()[0] & 0xff0f) == 0xff01),
        "unspecified" => return ip.is_unspecified(),
        "private" => return matches!(ip, IpAddr::V4(v) if v.is_private()) || matches!(ip, IpAddr::V6(v) if (v.segments()[0] & 0xfe00) == 0xfc00),
        "public" => {
            return !is_internal_ip(ip) && !ip.is_multicast();
        }
        _ => {}
    }
    let (addr, bits) = match net.split_once('/') {
        Some((a, b)) => (a, b.parse::<u32>().ok()),
        None => (net, None),
    };
    let Some(n) = parse_ip(addr) else { return false };
    match (ip, n) {
        (IpAddr::V4(a), IpAddr::V4(b)) => {
            let bits = bits.unwrap_or(32).min(32);
            let mask = if bits == 0 { 0 } else { u32::MAX << (32 - bits) };
            (u32::from(*a) & mask) == (u32::from(b) & mask)
        }
        (IpAddr::V6(a), IpAddr::V6(b)) => {
            let bits = bits.unwrap_or(128).min(128);
            let mask = if bits == 0 { 0 } else { u128::MAX << (128 - bits) };
            (u128::from(*a) & mask) == (u128::from(b) & mask)
        }
        _ => false,
    }
}

// ------------------------------------------------------------ HTML

/// Entity-decoding tag stripper for text extraction (`attachment`); NOT the `html_strip` processor.
pub fn html_text(s: &str) -> String {
    const BLOCK: &[&str] = &["p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr", "dd", "dt", "dl", "address", "center", "form", "title", "body", "html", "head"];
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let b = s.as_bytes();
    while i < s.len() {
        if b[i] == b'<' {
            if s[i..].starts_with("<!--") {
                match s[i..].find("-->") {
                    Some(e) => {
                        i += e + 3;
                        continue;
                    }
                    None => break,
                }
            }
            if let Some(e) = s[i..].find('>') {
                let tag = &s[i + 1..i + e];
                let name: String = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
                if !name.is_empty() || tag.starts_with('!') || tag.starts_with('?') {
                    let lower = name.as_str();
                    if lower == "script" || lower == "style" {
                        if !tag.starts_with('/') {
                            let close = format!("</{lower}");
                            if let Some(c) = s[i..].to_lowercase().find(&close) {
                                let after = i + c;
                                let end = s[after..].find('>').map(|x| after + x + 1).unwrap_or(s.len());
                                i = end;
                                continue;
                            }
                        }
                    }
                    if BLOCK.contains(&lower) {
                        out.push('\n');
                    }
                    i += e + 1;
                    continue;
                }
            }
        }
        if b[i] == b'&' {
            if let Some(e) = s[i..].find(';').filter(|e| *e <= 10) {
                let ent = &s[i + 1..i + e];
                let rep = match ent {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some('\u{a0}'),
                    _ => {
                        if let Some(h) = ent.strip_prefix("#x").or(ent.strip_prefix("#X")) {
                            u32::from_str_radix(h, 16).ok().and_then(char::from_u32)
                        } else if let Some(d) = ent.strip_prefix('#') {
                            d.parse::<u32>().ok().and_then(char::from_u32)
                        } else {
                            None
                        }
                    }
                };
                if let Some(c) = rep {
                    out.push(c);
                    i += e + 1;
                    continue;
                }
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Lucene `HTMLStripCharFilter` as ES uses it: inline tags vanish, every other tag
/// (open or close) becomes `\n`, `<script>`/`<style>` elements collapse to one `\n`,
/// comments / doctype / processing instructions are dropped, CDATA keeps its text and
/// entities are left alone.
pub fn html_strip(s: &str) -> String {
    // ES only runs the char filter when the text looks like markup.
    if !(s.contains('<') && s.contains('>')) {
        return s.to_string();
    }
    decode_entities(&html_strip_tags(s))
}

const LATIN1: &str = "nbsp iexcl cent pound curren yen brvbar sect uml copy ordf laquo not shy reg macr deg plusmn sup2 sup3 acute micro para middot cedil sup1 ordm raquo frac14 frac12 frac34 iquest Agrave Aacute Acirc Atilde Auml Aring AElig Ccedil Egrave Eacute Ecirc Euml Igrave Iacute Icirc Iuml ETH Ntilde Ograve Oacute Ocirc Otilde Ouml times Oslash Ugrave Uacute Ucirc Uuml Yacute THORN szlig agrave aacute acirc atilde auml aring aelig ccedil egrave eacute ecirc euml igrave iacute icirc iuml eth ntilde ograve oacute ocirc otilde ouml divide oslash ugrave uacute ucirc uuml yacute thorn yuml";

fn named_entity(name: &str) -> Option<char> {
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        "euro" => Some('\u{20ac}'),
        "hellip" => Some('\u{2026}'),
        "ndash" => Some('\u{2013}'),
        "mdash" => Some('\u{2014}'),
        "bull" => Some('\u{2022}'),
        "trade" => Some('\u{2122}'),
        "lsquo" => Some('\u{2018}'),
        "rsquo" => Some('\u{2019}'),
        "ldquo" => Some('\u{201c}'),
        "rdquo" => Some('\u{201d}'),
        n => LATIN1.split(' ').position(|x| x == n).and_then(|i| char::from_u32(0xA0 + i as u32)),
    }
}

/// Decode `&name;` / `&#N;` / `&#xH;` references; unknown ones are left as written.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let decoded = tail.find(';').filter(|e| *e <= 10).and_then(|e| {
            let ent = &tail[1..e];
            let c = if let Some(h) = ent.strip_prefix("#x").or(ent.strip_prefix("#X")) {
                u32::from_str_radix(h, 16).ok().and_then(char::from_u32)
            } else if let Some(d) = ent.strip_prefix('#') {
                d.parse::<u32>().ok().and_then(char::from_u32)
            } else {
                named_entity(ent)
            };
            c.map(|c| (c, e + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &tail[len..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn html_strip_tags(s: &str) -> String {
    String::from_utf16(&html_strip_tags_units(&s.encode_utf16().collect::<Vec<_>>())).expect("Unicode input")
}

/// Scan markup over Java code units; literal text never passes through UTF8.
pub fn html_strip_units(s: &[u16]) -> Vec<u16> {
    if !s.contains(&60) || !s.contains(&62) { return s.to_vec(); }
    let text=html_strip_tags_units(s); let mut out=Vec::new();let mut i=0;
    while i<text.len() {
        if text[i]==38 {
            if let Some(end)=text[i..].iter().take(11).position(|&u|u==59) {
                if let Ok(entity)=String::from_utf16(&text[i..i+end+1]) {
                    let decoded=decode_entities(&entity);
                    if decoded!=entity {out.extend(decoded.encode_utf16());i+=end+1;continue;}
                }
            }
        }
        out.push(text[i]);i+=1;
    }
    out
}

fn html_strip_tags_units(s: &[u16]) -> Vec<u16> {
    const INLINE: &[&str] = &[
        "a", "b", "i", "q", "s", "u", "abbr", "acronym", "bdo", "big", "cite", "code", "dfn", "em", "font", "img", "input", "kbd", "label", "samp", "select", "small", "span", "strike", "strong", "sub", "sup",
        "textarea", "tt", "var",
    ];
    let b = s;
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    let lower_starts = |at: usize, pat: &str| s.get(at..at + pat.len()).map(|x| x.iter().zip(pat.bytes()).all(|(&u,c)| u <= 127 && (u as u8).eq_ignore_ascii_case(&c))).unwrap_or(false);
    while i < b.len() {
        if b[i] != 60 {
            out.push(s[i]);
            i += 1;
            continue;
        }
        let rest = &s[i..];
        if rest.starts_with(&"<!--".encode_utf16().collect::<Vec<_>>()) {
            if let Some(e) = rest[4..].windows(3).position(|w|w==[45,45,62]) {
                i += 4 + e + 3;
                continue;
            }
        } else if rest.starts_with(&"<![CDATA[".encode_utf16().collect::<Vec<_>>()) {
            if let Some(e) = rest[9..].windows(3).position(|w|w==[93,93,62]) {
                out.extend_from_slice(&rest[9..9 + e]);
                i += 9 + e + 3;
                continue;
            }
        } else if rest.starts_with(&"<!".encode_utf16().collect::<Vec<_>>()) || rest.starts_with(&"<?".encode_utf16().collect::<Vec<_>>()) {
            if let Some(e) = rest.iter().position(|&u|u==62) {
                i += e + 1;
                continue;
            }
        } else {
            // tag: `<` `/`? name ... `>` (quote-aware)
            let mut j = i + 1;
            let closing = j < b.len() && b[j] == 47;
            if closing {
                j += 1;
            }
            let ns = j;
            if j < b.len() && (b[j]<=127 && (b[j] as u8).is_ascii_alphabetic()) {
                while j < b.len() && ((b[j]<=127 && (b[j] as u8).is_ascii_alphanumeric()) || b[j] == 58 || b[j] == 45 || b[j] == 95) {
                    j += 1;
                }
                let name = String::from_utf16(&s[ns..j]).unwrap().to_ascii_lowercase();
                // scan attributes to the closing `>`
                let mut q: Option<u16> = None;
                let mut k = j;
                let mut end = None;
                while k < b.len() {
                    match (q, b[k]) {
                        (Some(qc), c) if c == qc => q = None,
                        (Some(_), _) => {}
                        (None, 34) | (None, 39) => q = Some(b[k]),
                        (None, 62) => {
                            end = Some(k);
                            break;
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if let Some(e) = end {
                    if !closing && (name == "script" || name == "style") {
                        // drop through the matching close tag
                        let mut m = e + 1;
                        let mut found = None;
                        while m < b.len() {
                            if b[m] == 60 && lower_starts(m, &format!("</{name}")) {
                                if let Some(g) = s[m..].iter().position(|&u|u==62) {
                                    found = Some(m + g + 1);
                                }
                                break;
                            }
                            m += 1;
                        }
                        out.push(10);
                        i = found.unwrap_or(e + 1);
                        continue;
                    }
                    if !INLINE.contains(&name.as_str()) {
                        out.push(10);
                    }
                    i = e + 1;
                    continue;
                }
            }
        }
        out.push(60);
        i += 1;
    }
    out
}

// ------------------------------------------------------------ user agent

fn ver_after(ua: &str, token: &str) -> Option<String> {
    let i = ua.find(token)? + token.len();
    let v: String = ua[i..].chars().take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '_').collect();
    if v.is_empty() { None } else { Some(v.replace('_', ".")) }
}

/// Basic UA parsing into the ES `user_agent` processor shape.
pub fn parse_user_agent(ua: &str, extract_device_type: bool) -> Map<String, Value> {
    let mut out = Map::new();
    // Browser.
    let browsers: &[(&str, &str)] = &[
        ("Edg/", "Edge"),
        ("Edge/", "Edge"),
        ("OPR/", "Opera"),
        ("Opera/", "Opera"),
        ("YaBrowser/", "Yandex Browser"),
        ("SamsungBrowser/", "Samsung Internet"),
        ("Vivaldi/", "Vivaldi"),
        ("Firefox/", "Firefox"),
        ("FxiOS/", "Firefox iOS"),
        ("CriOS/", "Chrome Mobile iOS"),
        ("Chrome/", "Chrome"),
        ("MSIE ", "IE"),
        ("Trident/", "IE"),
        ("curl/", "curl"),
        ("Wget/", "Wget"),
        ("python-requests/", "Python Requests"),
        ("Go-http-client/", "Go-http-client"),
        ("Googlebot/", "Googlebot"),
        ("bingbot/", "bingbot"),
        ("Elastic-Heartbeat/", "Elastic Heartbeat"),
        ("Version/", "Safari"),
    ];
    let mut name = "Other".to_string();
    let mut version: Option<String> = None;
    for (tok, n) in browsers {
        if ua.contains(tok) {
            name = n.to_string();
            version = if *tok == "Trident/" { ver_after(ua, "rv:") } else { ver_after(ua, tok) };
            if *n == "Chrome" && (ua.contains("Mobile") || ua.contains("Android")) {
                name = "Chrome Mobile".into();
            }
            if *n == "Safari" && ua.contains("Mobile") {
                name = "Mobile Safari".into();
            }
            if *n == "Firefox" && ua.contains("Mobile") {
                name = "Firefox Mobile".into();
            }
            break;
        }
    }
    out.insert("name".into(), Value::String(name.clone()));
    if let Some(v) = &version {
        let parts: Vec<&str> = v.split('.').filter(|p| !p.is_empty()).collect();
        if !parts.is_empty() {
            out.insert("version".into(), Value::String(parts.join(".")));
        }
    }
    // OS.
    let mut os: Option<(String, Option<String>)> = None;
    if ua.contains("Windows NT") {
        let v = ver_after(ua, "Windows NT ");
        let ver = match v.as_deref() {
            Some("10.0") => Some("10".to_string()),
            Some("6.3") => Some("8.1".to_string()),
            Some("6.2") => Some("8".to_string()),
            Some("6.1") => Some("7".to_string()),
            Some("6.0") => Some("Vista".to_string()),
            Some("5.1") | Some("5.2") => Some("XP".to_string()),
            o => o.map(str::to_string),
        };
        os = Some(("Windows".into(), ver));
    } else if ua.contains("iPhone") || ua.contains("iPad") || ua.contains("iPod") {
        os = Some(("iOS".into(), ver_after(ua, "OS ")));
    } else if ua.contains("Android") {
        os = Some(("Android".into(), ver_after(ua, "Android ")));
    } else if ua.contains("Mac OS X") {
        os = Some(("Mac OS X".into(), ver_after(ua, "Mac OS X ")));
    } else if ua.contains("CrOS") {
        os = Some(("Chrome OS".into(), None));
    } else if ua.contains("Ubuntu") {
        os = Some(("Ubuntu".into(), ver_after(ua, "Ubuntu/")));
    } else if ua.contains("Linux") {
        os = Some(("Linux".into(), None));
    }
    if let Some((n, v)) = os {
        let mut o = Map::new();
        o.insert("name".into(), Value::String(n.clone()));
        if let Some(v) = v {
            o.insert("version".into(), Value::String(v.clone()));
            o.insert("full".into(), Value::String(format!("{n} {v}")));
        } else {
            o.insert("full".into(), Value::String(n));
        }
        out.insert("os".into(), Value::Object(o));
    }
    // Device.
    let device = if ua.contains("iPhone") {
        "iPhone"
    } else if ua.contains("iPad") {
        "iPad"
    } else if ua.contains("Macintosh") {
        "Mac"
    } else if ua.contains("Googlebot") || ua.contains("bingbot") || ua.to_lowercase().contains("bot/") {
        "Spider"
    } else if ua.contains("Android") && ua.contains("Mobile") {
        "Generic Smartphone"
    } else if ua.contains("Android") {
        "Generic Tablet"
    } else {
        "Other"
    };
    let mut d = Map::new();
    d.insert("name".into(), Value::String(device.into()));
    if extract_device_type {
        let ty = match device {
            "iPhone" | "Generic Smartphone" => "Phone",
            "iPad" | "Generic Tablet" => "Tablet",
            "Spider" => "Robot",
            "Mac" => "Desktop",
            _ if ua.contains("Windows") || ua.contains("X11") => "Desktop",
            _ => "Other",
        };
        d.insert("type".into(), Value::String(ty.into()));
    }
    out.insert("device".into(), Value::Object(d));
    out
}

// ------------------------------------------------------------ geo cells

/// Bounding box (min_lon, min_lat, max_lon, max_lat) of a geotile `z/x/y`.
pub fn geotile_bbox(id: &str) -> Option<(f64, f64, f64, f64)> {
    let p: Vec<i64> = id.split('/').map(|x| x.parse().ok()).collect::<Option<Vec<_>>>()?;
    if p.len() != 3 || p[0] < 0 || p[0] > 29 {
        return None;
    }
    let n = (1i64 << p[0]) as f64;
    let (x, y) = (p[1] as f64, p[2] as f64);
    if x < 0.0 || y < 0.0 || x >= n || y >= n {
        return None;
    }
    // ES `GeoTileUtils.toBoundingBox` (FastMath sinh/atan), bit for bit.
    let (top, left, bottom, right) = crate::fieldtypes::geo::tile_bbox(p[0] as u32, x as u32, y as u32);
    Some((left, bottom, right, top))
}

pub fn geohash_bbox(h: &str) -> Option<(f64, f64, f64, f64)> {
    const B32: &str = "0123456789bcdefghjkmnpqrstuvwxyz";
    if h.is_empty() || h.len() > 12 {
        return None;
    }
    let (mut lat, mut lon) = ((-90.0f64, 90.0f64), (-180.0f64, 180.0f64));
    let mut even = true;
    for c in h.chars() {
        let v = B32.find(c)?;
        for bit in (0..5).rev() {
            let b = (v >> bit) & 1 == 1;
            let r = if even { &mut lon } else { &mut lat };
            let mid = (r.0 + r.1) / 2.0;
            if b {
                r.0 = mid;
            } else {
                r.1 = mid;
            }
            even = !even;
        }
    }
    Some((lon.0, lat.0, lon.1, lat.1))
}

pub fn bbox_shape(b: (f64, f64, f64, f64), wkt: bool) -> Value {
    let (x0, y0, x1, y1) = b;
    if wkt {
        return Value::String(format!("POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"));
    }
    json!({"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests() {
        assert_eq!(hex(&md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(&hex(&sha512(b"abc"))[..32], "ddaf35a193617abacc417349ae204131");
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn bytes_and_html() {
        assert_eq!(parse_bytes("1kb"), Some(1024));
        assert_eq!(parse_bytes("1.5 KB"), Some(1536));
        assert_eq!(parse_bytes("12"), Some(12));
        assert_eq!(parse_bytes("abc"), None);
        assert_eq!(html_strip("<p>a &amp; b</p><b>c</b>"), "\na & b\nc");
        assert_eq!(html_strip("a &amp; b"), "a &amp; b");
        assert_eq!(html_strip("<i>&nbsp;&#65;&#x42;&copy;&unknown;</i>"), " AB\u{a9}&unknown;");
        assert_eq!(html_text("<p>a &amp; b</p><b>c</b>"), "\na & b\nc");
        let ua = parse_user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36", false);
        assert_eq!(ua["name"], "Chrome");
        assert_eq!(ua["version"], "33.0.1750.149");
        assert_eq!(ua["os"]["full"], "Mac OS X 10.9.2");
        assert_eq!(ua["device"]["name"], "Mac");
    }
}
