//! MurmurHash3 x64 128-bit (seed 0), as ES's `murmur3` field and
//! `cardinality` use (`MurmurHash3.hash128(..).h1`).

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

/// `(h1, h2)` of MurmurHash3_x64_128.
pub fn hash128(data: &[u8], seed: u64) -> (u64, u64) {
    const C1: u64 = 0x87c37b91114253d5;
    const C2: u64 = 0x4cf5ad432745937f;
    let len = data.len();
    let mut h1 = seed;
    let mut h2 = seed;
    let nblocks = len / 16;
    for i in 0..nblocks {
        let b = &data[i * 16..i * 16 + 16];
        let mut k1 = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(b[8..16].try_into().unwrap());
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(27).wrapping_add(h2).wrapping_mul(5).wrapping_add(0x52dce729);
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2.rotate_left(31).wrapping_add(h1).wrapping_mul(5).wrapping_add(0x38495ab5);
    }
    let tail = &data[nblocks * 16..];
    let mut k1: u64 = 0;
    let mut k2: u64 = 0;
    for i in (0..tail.len()).rev() {
        if i >= 8 {
            k2 ^= (tail[i] as u64) << ((i - 8) * 8);
        } else {
            k1 ^= (tail[i] as u64) << (i * 8);
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
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (h1, h2)
}

/// The value ES's `murmur3` field stores for a string (`h1` as a long).
pub fn murmur3_field_hash(s: &str) -> i64 {
    hash128(s.as_bytes(), 0).0 as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // Reference values of MurmurHash3_x64_128 with seed 0.
        assert_eq!(hash128(b"", 0), (0, 0));
        assert_eq!(hash128(b"hello", 0).0, 0xcbd8a7b341bd9b02);
        assert_eq!(hash128(b"The quick brown fox jumps over the lazy dog", 0).0, 0xe34bbc7bbc071b6c);
    }
}
