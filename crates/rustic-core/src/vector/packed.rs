//! Packed vector rows: ES `element_type: bit` and `byte` vectors stored as
//! their bytes rather than one `f32` per value.
//!
//! A packed row keeps the store's `f32` row type (every store, file, copy,
//! snapshot and recovery path moves rows as `f32` slices), but its lanes
//! are not floats: each holds four of the row's bytes, little-endian, and
//! the last lane is zero-padded. A `bit` row of `n` bits is `n / 8` bytes,
//! eight bits per byte with the first bit in the most significant position
//! (as ES packs them); a `byte` row of `n` values is `n` `i8`s. The field's
//! core dimension is its lane count ([`words`]).
//!
//! Only the packed [`Similarity`] variants read these rows, through the
//! kernels here (Hamming by popcount, `i8` dot / squared L2): the lanes may
//! hold any bit pattern, NaNs included, so no float kernel, finiteness
//! check, norm or quantizer may touch them. Zero padding changes no
//! distance. Scores equal what the unpacked rows score under the matching
//! float similarity (see [`Similarity::unpacked`]); integer sums are exact.

use super::Similarity;

#[cfg(target_endian = "big")]
compile_error!("packed vector rows assume a little-endian target");

/// Lanes of a packed row of `bytes` bytes.
#[inline]
pub fn words(bytes: usize) -> usize {
    bytes.div_ceil(4)
}

/// Pack `bytes` into a row's lanes.
pub fn pack(bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; words(bytes.len())];
    for (lane, chunk) in out.iter_mut().zip(bytes.chunks(4)) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        *lane = f32::from_bits(u32::from_le_bytes(word));
    }
    out
}

/// The bytes of a packed row (padding included).
#[inline]
pub fn as_bytes(row: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding or invalid bit patterns as bytes; the
    // byte slice covers exactly the row's lanes, and on a little-endian
    // target lane `i` holds bytes `4 i .. 4 i + 4` in order.
    unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), std::mem::size_of_val(row)) }
}

/// A packed row's `i8` values (padding included).
#[inline]
pub fn as_i8(row: &[f32]) -> &[i8] {
    // SAFETY: as `as_bytes`; `i8` and `u8` share size and alignment.
    unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<i8>(), std::mem::size_of_val(row)) }
}

/// A `bit` row's first `bits` bits as `0.0` / `1.0` values (most
/// significant bit of each byte first): the unpacked form.
pub fn unpack_bits(row: &[f32], bits: usize) -> Vec<f32> {
    let bytes = as_bytes(row);
    (0..bits.min(bytes.len() * 8)).map(|i| ((bytes[i / 8] >> (7 - i % 8)) & 1) as f32).collect()
}

/// A `byte` row's first `dims` values as floats: the unpacked form.
pub fn unpack_i8(row: &[f32], dims: usize) -> Vec<f32> {
    as_i8(row).iter().take(dims).map(|&v| f32::from(v)).collect()
}

/// Whether two packed rows hold the same bytes (`==` on their lanes would
/// compare them as floats).
#[inline]
pub fn same(a: &[f32], b: &[f32]) -> bool {
    as_bytes(a) == as_bytes(b)
}

/// Hamming distance of two packed `bit` rows.
#[inline]
pub fn hamming(a: &[f32], b: &[f32]) -> u32 {
    assert_eq!(a.len(), b.len(), "hamming: length mismatch");
    let (a, b) = (as_bytes(a), as_bytes(b));
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths are equal.
        unsafe { neon::hamming(a.as_ptr(), b.as_ptr(), a.len()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::hamming(a, b)
    }
}

/// Dot product of two packed `byte` rows.
#[inline]
pub fn dot_i8(a: &[f32], b: &[f32]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: length mismatch");
    // The int8 codes' kernel: SDOT where the CPU has it (Graviton 2+: 16
    // multiply-adds per instruction), else NEON widening; exact in `i32`.
    crate::vector::quantize::dot_i8(as_i8(a), as_i8(b))
}

/// Squared Euclidean distance of two packed `byte` rows.
#[inline]
pub fn l2_sq_i8(a: &[f32], b: &[f32]) -> i32 {
    assert_eq!(a.len(), b.len(), "l2_sq_i8: length mismatch");
    let (a, b) = (as_i8(a), as_i8(b));
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths are equal.
        unsafe { neon::l2_sq_i8(a.as_ptr(), b.as_ptr(), a.len()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::l2_sq_i8(a, b)
    }
}

/// Cosine of two packed `byte` rows, as [`super::distance::cosine`] computes
/// it on the unpacked values (`0` when either norm is zero).
#[inline]
pub fn cosine_i8(a: &[f32], b: &[f32]) -> f32 {
    let na = (dot_i8(a, a) as f32).sqrt();
    if na == 0.0 {
        return 0.0;
    }
    let nb2 = dot_i8(b, b) as f32;
    if nb2 == 0.0 {
        return 0.0;
    }
    dot_i8(a, b) as f32 / (na * nb2.sqrt())
}

/// [`Similarity::distance`] of a packed similarity.
#[inline]
pub fn distance(similarity: Similarity, a: &[f32], b: &[f32]) -> f32 {
    match similarity {
        Similarity::BitHamming => hamming(a, b) as f32,
        Similarity::ByteL2 => l2_sq_i8(a, b) as f32,
        Similarity::ByteCosine => 1.0 - cosine_i8(a, b),
        Similarity::ByteDotProduct | Similarity::ByteMaxInnerProduct => -(dot_i8(a, b) as f32),
        _ => unreachable!("float similarity on packed rows"),
    }
}

/// [`Similarity::score`] of a packed similarity.
#[inline]
pub fn score(similarity: Similarity, a: &[f32], b: &[f32]) -> f32 {
    match similarity {
        Similarity::BitHamming => 1.0 / (1.0 + hamming(a, b) as f32),
        Similarity::ByteL2 => 1.0 / (1.0 + l2_sq_i8(a, b) as f32),
        Similarity::ByteCosine => (1.0 + cosine_i8(a, b)) / 2.0,
        Similarity::ByteDotProduct => (1.0 + dot_i8(a, b) as f32) / 2.0,
        Similarity::ByteMaxInnerProduct => {
            let d = dot_i8(a, b) as f32;
            if d >= 0.0 { d + 1.0 } else { 1.0 / (1.0 - d) }
        }
        _ => unreachable!("float similarity on packed rows"),
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod scalar {
    pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
        let mut total = 0u32;
        let mut ca = a.chunks_exact(8);
        let mut cb = b.chunks_exact(8);
        for (x, y) in (&mut ca).zip(&mut cb) {
            let x = u64::from_le_bytes(x.try_into().expect("8 bytes"));
            let y = u64::from_le_bytes(y.try_into().expect("8 bytes"));
            total += (x ^ y).count_ones();
        }
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            total += (x ^ y).count_ones();
        }
        total
    }
    pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        let mut acc = [0i32; 16];
        let mut ca = a.chunks_exact(16);
        let mut cb = b.chunks_exact(16);
        for (x, y) in (&mut ca).zip(&mut cb) {
            for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
                *acc += i32::from(*x) * i32::from(*y);
            }
        }
        let mut total: i32 = acc.iter().sum();
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            total += i32::from(*x) * i32::from(*y);
        }
        total
    }
    pub fn l2_sq_i8(a: &[i8], b: &[i8]) -> i32 {
        let mut acc = [0i32; 16];
        let mut ca = a.chunks_exact(16);
        let mut cb = b.chunks_exact(16);
        for (x, y) in (&mut ca).zip(&mut cb) {
            for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
                let d = i32::from(*x) - i32::from(*y);
                *acc += d * d;
            }
        }
        let mut total: i32 = acc.iter().sum();
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            let d = i32::from(*x) - i32::from(*y);
            total += d * d;
        }
        total
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Popcount of `a ^ b`, 16 bytes per step into `u16` lane counts
    /// (at most 16 per lane per step, flushed every 2048 steps).
    ///
    /// # Safety
    /// `a` and `b` must be valid for `n` reads.
    #[inline]
    pub unsafe fn hamming(a: *const u8, b: *const u8, n: usize) -> u32 {
        unsafe {
            let mut acc = vdupq_n_u16(0);
            let mut total = 0u32;
            let mut i = 0;
            while i + 16 <= n {
                let x = veorq_u8(vld1q_u8(a.add(i)), vld1q_u8(b.add(i)));
                acc = vpadalq_u8(acc, vcntq_u8(x));
                i += 16;
                if i % (16 * 2048) == 0 {
                    total += vaddlvq_u16(acc);
                    acc = vdupq_n_u16(0);
                }
            }
            total += vaddlvq_u16(acc);
            while i < n {
                total += (*a.add(i) ^ *b.add(i)).count_ones();
                i += 1;
            }
            total
        }
    }

    /// `i8` dot product: widening multiplies into `i16` (exact: at most
    /// 16384 in magnitude), pairwise-accumulated into `i32` lanes.
    ///
    /// # Safety
    /// `a` and `b` must be valid for `n` reads.
    #[inline]
    // Packed `byte` dots use `quantize::dot_i8` (SDOT); this kernel is the
    // tests' NEON reference.
    #[cfg_attr(not(test), allow(dead_code))]
    pub unsafe fn dot_i8(a: *const i8, b: *const i8, n: usize) -> i32 {
        unsafe {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut i = 0;
            while i + 16 <= n {
                let x = vld1q_s8(a.add(i));
                let y = vld1q_s8(b.add(i));
                acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
                acc1 = vpadalq_s16(acc1, vmull_high_s8(x, y));
                i += 16;
            }
            let mut total = vaddvq_s32(vaddq_s32(acc0, acc1));
            while i < n {
                total += i32::from(*a.add(i)) * i32::from(*b.add(i));
                i += 1;
            }
            total
        }
    }

    /// `i8` squared L2: widening differences into `i16`, squared and
    /// accumulated into `i32` lanes.
    ///
    /// # Safety
    /// `a` and `b` must be valid for `n` reads.
    #[inline]
    pub unsafe fn l2_sq_i8(a: *const i8, b: *const i8, n: usize) -> i32 {
        unsafe {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut i = 0;
            while i + 16 <= n {
                let x = vld1q_s8(a.add(i));
                let y = vld1q_s8(b.add(i));
                let lo = vsubl_s8(vget_low_s8(x), vget_low_s8(y));
                let hi = vsubl_high_s8(x, y);
                acc0 = vmlal_s16(acc0, vget_low_s16(lo), vget_low_s16(lo));
                acc1 = vmlal_high_s16(acc1, lo, lo);
                acc0 = vmlal_s16(acc0, vget_low_s16(hi), vget_low_s16(hi));
                acc1 = vmlal_high_s16(acc1, hi, hi);
                i += 16;
            }
            let mut total = vaddvq_s32(vaddq_s32(acc0, acc1));
            while i < n {
                let d = i32::from(*a.add(i)) - i32::from(*b.add(i));
                total += d * d;
                i += 1;
            }
            total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn kernels_match_the_unpacked_float_similarities() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for len in [1usize, 3, 4, 7, 15, 16, 17, 31, 64, 100, 128, 768, 1029] {
            for _ in 0..8 {
                let a: Vec<u8> = (0..len).map(|_| rng(&mut seed) as u8).collect();
                let b: Vec<u8> = (0..len).map(|_| rng(&mut seed) as u8).collect();
                let (pa, pb) = (pack(&a), pack(&b));
                assert_eq!(pa.len(), words(len));
                // Bits: Hamming = L2 of the 0/1 expansion.
                let (ea, eb) = (unpack_bits(&pa, len * 8), unpack_bits(&pb, len * 8));
                assert_eq!(hamming(&pa, &pb) as f32, super::super::distance::l2_sq(&ea, &eb), "len {len}");
                assert_eq!(Similarity::BitHamming.score(&pa, &pb), Similarity::L2.score(&ea, &eb));
                // Bytes: exact integer sums equal the float kernels' below 2^24.
                let (va, vb) = (unpack_i8(&pa, len), unpack_i8(&pb, len));
                for (packed, float) in [(Similarity::ByteL2, Similarity::L2), (Similarity::ByteDotProduct, Similarity::DotProduct),
                    (Similarity::ByteMaxInnerProduct, Similarity::MaxInnerProduct), (Similarity::ByteCosine, Similarity::Cosine)] {
                    assert_eq!(packed.score(&pa, &pb), float.score(&va, &vb), "{packed:?} len {len}");
                    assert_eq!(packed.distance(&pa, &pb), float.distance(&va, &vb), "{packed:?} len {len}");
                }
            }
        }
    }

    /// The packed dot (the codes' SDOT kernel where available) equals the
    /// scalar and NEON widening kernels exactly, extremes included
    /// (`-128 * -128` in every lane).
    #[test]
    fn dot_kernels_agree_exactly() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for len in [1usize, 15, 16, 17, 63, 64, 65, 100, 1024, 4096] {
            let rows: [Vec<u8>; 3] = [(0..len).map(|_| rng(&mut seed) as u8).collect(), vec![0x80; len], vec![0x7f; len]];
            for a in &rows { for b in &rows {
                let (pa, pb) = (pack(a), pack(b));
                let (ia, ib) = (as_i8(&pa), as_i8(&pb));
                let expect = scalar::dot_i8(ia, ib);
                assert_eq!(dot_i8(&pa, &pb), expect, "len {len}");
                #[cfg(target_arch = "aarch64")]
                // SAFETY: both rows hold `ia.len()` bytes.
                unsafe { assert_eq!(neon::dot_i8(ia.as_ptr(), ib.as_ptr(), ia.len()), expect, "len {len}"); }
            } }
        }
    }

    #[test]
    fn bits_unpack_most_significant_first_and_rows_compare_by_bytes() {
        let row = pack(&[0b1000_0001, 0x40]);
        assert_eq!(unpack_bits(&row, 16), [1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(unpack_i8(&pack(&[0xff, 0x80, 0x7f]), 3), [-1., -128., 127.]);
        // A NaN-patterned lane still compares equal to itself bytewise.
        let nan = pack(&[0xff, 0xff, 0xff, 0xff]);
        assert!(nan[0].is_nan() && same(&nan, &nan.clone()));
        assert!(!same(&pack(&[0, 0, 0, 0x80]), &pack(&[0, 0, 0, 0])));
    }
}
