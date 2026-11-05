//! SIMD vector distance primitives.
//!
//! On `aarch64` the kernels are explicit NEON (4 independent `float32x4`
//! FMA accumulators, 16 floats per iteration). On `x86_64` an AVX2+FMA
//! kernel is selected at runtime (the feature check is a cached atomic
//! load); everywhere else, and for the tail, a chunked scalar loop with 8
//! independent accumulators that LLVM autovectorizes.
//!
//! All functions require `a.len() == b.len()` (or, for [`norm`], a single
//! slice) and are `#[inline]` so they fuse into callers.

/// Dot product of `a` and `b`. Panics if the lengths differ.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths are equal.
        unsafe { neon::dot(a.as_ptr(), b.as_ptr(), a.len()) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::available() {
            // SAFETY: feature checked above; lengths are equal.
            return unsafe { avx2::dot(a.as_ptr(), b.as_ptr(), a.len()) };
        }
        scalar::dot(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::dot(a, b)
    }
}

/// Four independent dot products with one query load per SIMD lane group.
/// Keep each row's accumulation order identical to `dot` so graph traversal
/// decisions, including ties, do not change.
#[inline]
pub(crate) fn dot_four(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    assert!(rows.iter().all(|r| r.len() == q.len()), "dot_four: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: every slice has the query length; NEON is mandatory.
        unsafe { neon::dot_four(q.as_ptr(), rows.map(|r| r.as_ptr()), q.len()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    { rows.map(|r| dot(q, r)) }
}

/// Squared Euclidean distance `sum((a[i] - b[i])^2)`. Panics if lengths differ.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_sq: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths are equal.
        unsafe { neon::l2_sq(a.as_ptr(), b.as_ptr(), a.len()) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::available() {
            // SAFETY: feature checked above; lengths are equal.
            return unsafe { avx2::l2_sq(a.as_ptr(), b.as_ptr(), a.len()) };
        }
        scalar::l2_sq(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar::l2_sq(a, b)
    }
}

/// `(dot(a, b), |b|^2)` in one pass (cosine against a query whose norm is
/// already known). Panics if lengths differ.
#[inline]
pub fn dot_and_sq_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "dot_and_sq_norm: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; lengths are equal.
        unsafe { neon::dot_sq(a.as_ptr(), b.as_ptr(), a.len()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        (dot(a, b), dot(b, b))
    }
}

/// Cosine similarity `dot(a, b) / (|a| * |b|)`.
///
/// Returns `0.0` if either vector has zero norm. Panics if lengths differ.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    if na == 0.0 {
        return 0.0;
    }
    let (d, nb2) = dot_and_sq_norm(a, b);
    if nb2 == 0.0 {
        return 0.0;
    }
    d / (na * nb2.sqrt())
}

/// Euclidean norm `sqrt(sum(a[i]^2))`.
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod scalar {
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut s = [0.0f32; 8];
        let mut ca = a.chunks_exact(8);
        let mut cb = b.chunks_exact(8);
        for (x, y) in (&mut ca).zip(&mut cb) {
            for i in 0..8 {
                s[i] += x[i] * y[i];
            }
        }
        let mut t = 0.0;
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            t += x * y;
        }
        (s[0] + s[1]) + (s[2] + s[3]) + (s[4] + s[5]) + (s[6] + s[7]) + t
    }

    #[inline]
    pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut s = [0.0f32; 8];
        let mut ca = a.chunks_exact(8);
        let mut cb = b.chunks_exact(8);
        for (x, y) in (&mut ca).zip(&mut cb) {
            for i in 0..8 {
                let d = x[i] - y[i];
                s[i] += d * d;
            }
        }
        let mut t = 0.0;
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            t += (x - y) * (x - y);
        }
        (s[0] + s[1]) + (s[2] + s[3]) + (s[4] + s[5]) + (s[6] + s[7]) + t
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// # Safety
    /// All five pointers must be valid for `n` float reads.
    #[inline]
    pub unsafe fn dot_four(q: *const f32, rows: [*const f32; 4], n: usize) -> [f32; 4] {
        unsafe {
            let mut r0a0 = vdupq_n_f32(0.0);
            let mut r0a1 = vdupq_n_f32(0.0);
            let mut r0a2 = vdupq_n_f32(0.0);
            let mut r0a3 = vdupq_n_f32(0.0);
            let mut r1a0 = vdupq_n_f32(0.0);
            let mut r1a1 = vdupq_n_f32(0.0);
            let mut r1a2 = vdupq_n_f32(0.0);
            let mut r1a3 = vdupq_n_f32(0.0);
            let mut r2a0 = vdupq_n_f32(0.0);
            let mut r2a1 = vdupq_n_f32(0.0);
            let mut r2a2 = vdupq_n_f32(0.0);
            let mut r2a3 = vdupq_n_f32(0.0);
            let mut r3a0 = vdupq_n_f32(0.0);
            let mut r3a1 = vdupq_n_f32(0.0);
            let mut r3a2 = vdupq_n_f32(0.0);
            let mut r3a3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 16 <= n {
                let q0 = vld1q_f32(q.add(i + 0));
                r0a0 = vfmaq_f32(r0a0, q0, vld1q_f32(rows[0].add(i + 0)));
                r1a0 = vfmaq_f32(r1a0, q0, vld1q_f32(rows[1].add(i + 0)));
                r2a0 = vfmaq_f32(r2a0, q0, vld1q_f32(rows[2].add(i + 0)));
                r3a0 = vfmaq_f32(r3a0, q0, vld1q_f32(rows[3].add(i + 0)));
                let q1 = vld1q_f32(q.add(i + 4));
                r0a1 = vfmaq_f32(r0a1, q1, vld1q_f32(rows[0].add(i + 4)));
                r1a1 = vfmaq_f32(r1a1, q1, vld1q_f32(rows[1].add(i + 4)));
                r2a1 = vfmaq_f32(r2a1, q1, vld1q_f32(rows[2].add(i + 4)));
                r3a1 = vfmaq_f32(r3a1, q1, vld1q_f32(rows[3].add(i + 4)));
                let q2 = vld1q_f32(q.add(i + 8));
                r0a2 = vfmaq_f32(r0a2, q2, vld1q_f32(rows[0].add(i + 8)));
                r1a2 = vfmaq_f32(r1a2, q2, vld1q_f32(rows[1].add(i + 8)));
                r2a2 = vfmaq_f32(r2a2, q2, vld1q_f32(rows[2].add(i + 8)));
                r3a2 = vfmaq_f32(r3a2, q2, vld1q_f32(rows[3].add(i + 8)));
                let q3 = vld1q_f32(q.add(i + 12));
                r0a3 = vfmaq_f32(r0a3, q3, vld1q_f32(rows[0].add(i + 12)));
                r1a3 = vfmaq_f32(r1a3, q3, vld1q_f32(rows[1].add(i + 12)));
                r2a3 = vfmaq_f32(r2a3, q3, vld1q_f32(rows[2].add(i + 12)));
                r3a3 = vfmaq_f32(r3a3, q3, vld1q_f32(rows[3].add(i + 12)));
                i += 16;
            }
            while i + 4 <= n {
                let q0 = vld1q_f32(q.add(i));
                r0a0 = vfmaq_f32(r0a0, q0, vld1q_f32(rows[0].add(i)));
                r1a0 = vfmaq_f32(r1a0, q0, vld1q_f32(rows[1].add(i)));
                r2a0 = vfmaq_f32(r2a0, q0, vld1q_f32(rows[2].add(i)));
                r3a0 = vfmaq_f32(r3a0, q0, vld1q_f32(rows[3].add(i)));
                i += 4;
            }
            let mut out = [
                vaddvq_f32(vaddq_f32(vaddq_f32(r0a0, r0a1), vaddq_f32(r0a2, r0a3))),
                vaddvq_f32(vaddq_f32(vaddq_f32(r1a0, r1a1), vaddq_f32(r1a2, r1a3))),
                vaddvq_f32(vaddq_f32(vaddq_f32(r2a0, r2a1), vaddq_f32(r2a2, r2a3))),
                vaddvq_f32(vaddq_f32(vaddq_f32(r3a0, r3a1), vaddq_f32(r3a2, r3a3))),
            ];
            while i < n {
                for r in 0..4 { out[r] += *q.add(i) * *rows[r].add(i); }
                i += 1;
            }
            out
        }
    }

    /// # Safety
    /// `a` and `b` must be valid for `n` reads.
    #[inline]
    pub unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 16 <= n {
                acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
                acc2 = vfmaq_f32(acc2, vld1q_f32(a.add(i + 8)), vld1q_f32(b.add(i + 8)));
                acc3 = vfmaq_f32(acc3, vld1q_f32(a.add(i + 12)), vld1q_f32(b.add(i + 12)));
                i += 16;
            }
            while i + 4 <= n {
                acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
                i += 4;
            }
            let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            while i < n {
                s += *a.add(i) * *b.add(i);
                i += 1;
            }
            s
        }
    }

    /// # Safety
    /// `a` and `b` must be valid for `n` reads.
    #[inline]
    pub unsafe fn l2_sq(a: *const f32, b: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 16 <= n {
                let d0 = vsubq_f32(vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
                let d1 = vsubq_f32(vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
                let d2 = vsubq_f32(vld1q_f32(a.add(i + 8)), vld1q_f32(b.add(i + 8)));
                let d3 = vsubq_f32(vld1q_f32(a.add(i + 12)), vld1q_f32(b.add(i + 12)));
                acc0 = vfmaq_f32(acc0, d0, d0);
                acc1 = vfmaq_f32(acc1, d1, d1);
                acc2 = vfmaq_f32(acc2, d2, d2);
                acc3 = vfmaq_f32(acc3, d3, d3);
                i += 16;
            }
            while i + 4 <= n {
                let d = vsubq_f32(vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
                acc0 = vfmaq_f32(acc0, d, d);
                i += 4;
            }
            let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            while i < n {
                let d = *a.add(i) - *b.add(i);
                s += d * d;
                i += 1;
            }
            s
        }
    }

    /// `(a·b, b·b)`.
    ///
    /// # Safety
    /// `a` and `b` must be valid for `n` reads.
    #[inline]
    pub unsafe fn dot_sq(a: *const f32, b: *const f32, n: usize) -> (f32, f32) {
        unsafe {
            let mut d0 = vdupq_n_f32(0.0);
            let mut d1 = vdupq_n_f32(0.0);
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= n {
                let b0 = vld1q_f32(b.add(i));
                let b1 = vld1q_f32(b.add(i + 4));
                d0 = vfmaq_f32(d0, vld1q_f32(a.add(i)), b0);
                d1 = vfmaq_f32(d1, vld1q_f32(a.add(i + 4)), b1);
                s0 = vfmaq_f32(s0, b0, b0);
                s1 = vfmaq_f32(s1, b1, b1);
                i += 8;
            }
            let mut d = vaddvq_f32(vaddq_f32(d0, d1));
            let mut s = vaddvq_f32(vaddq_f32(s0, s1));
            while i < n {
                let y = *b.add(i);
                d += *a.add(i) * y;
                s += y * y;
                i += 1;
            }
            (d, s)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    #[inline]
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    // `target_feature` lets the intrinsics (and this helper) inline into
    // the AVX2 kernels; without it every horizontal sum was an out-of-line
    // call chain (one per distance).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// Requires AVX2+FMA; `a` and `b` valid for `n` reads.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)), acc1);
                i += 16;
            }
            while i + 8 <= n {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
                i += 8;
            }
            let mut s = hsum(_mm256_add_ps(acc0, acc1));
            while i < n {
                s += *a.add(i) * *b.add(i);
                i += 1;
            }
            s
        }
    }

    /// # Safety
    /// Requires AVX2+FMA; `a` and `b` valid for `n` reads.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_sq(a: *const f32, b: *const f32, n: usize) -> f32 {
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)));
                let d1 = _mm256_sub_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)));
                acc0 = _mm256_fmadd_ps(d0, d0, acc0);
                acc1 = _mm256_fmadd_ps(d1, d1, acc1);
                i += 16;
            }
            while i + 8 <= n {
                let d = _mm256_sub_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)));
                acc0 = _mm256_fmadd_ps(d, d, acc0);
                i += 8;
            }
            let mut s = hsum(_mm256_add_ps(acc0, acc1));
            while i < n {
                let d = *a.add(i) - *b.add(i);
                s += d * d;
                i += 1;
            }
            s
        }
    }
}

/// Hint the CPU to pull the cache lines of `bytes` bytes at `p` into L1.
/// A no-op on targets without a prefetch instruction.
#[inline(always)]
pub(crate) fn prefetch(p: *const u8, bytes: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        let mut off = 0;
        while off < bytes {
            // SAFETY: `prfm` never faults, even on an invalid address.
            unsafe {
                core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.wrapping_add(off), options(nostack, preserves_flags, readonly));
            }
            // 64-byte lines on Neoverse (Graviton); a 128-byte stride left
            // every other line of a row cold.
            off += 64;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mut off = 0;
        while off < bytes {
            // SAFETY: prefetch never faults.
            unsafe {
                core::arch::x86_64::_mm_prefetch(p.wrapping_add(off) as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
            off += 64;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (p, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic PRNG (xorshift64*) so tests are reproducible.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn next_f32(&mut self) -> f32 {
            // Uniform in [-1, 1).
            ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn naive_l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }
    fn naive_norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn chunked_matches_naive_for_lengths_1_to_100() {
        let mut rng = Rng::new(42);
        for len in 1..=100usize {
            let a: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let (nd, nl, nn) = (naive_dot(&a, &b), naive_l2_sq(&a, &b), naive_norm(&a));
            let (cd, cl, cn) = (dot(&a, &b), l2_sq(&a, &b), norm(&a));
            // Summation order differs, so compare with a relative tolerance.
            let tol = |x: f32| 1e-5 * x.abs().max(1.0);
            assert!((nd - cd).abs() < tol(nd), "dot len={len}: naive={nd} simd={cd}");
            assert!((nl - cl).abs() < tol(nl), "l2_sq len={len}: naive={nl} simd={cl}");
            assert!((nn - cn).abs() < tol(nn), "norm len={len}: naive={nn} simd={cn}");
            assert!((naive_dot(&a, &b) - scalar::dot(&a, &b)).abs() < tol(nd));
            assert!((naive_l2_sq(&a, &b) - scalar::l2_sq(&a, &b)).abs() < tol(nl));
            let (fd, fs) = dot_and_sq_norm(&a, &b);
            assert!((fd - nd).abs() < tol(nd));
            assert!((fs - naive_dot(&b, &b)).abs() < tol(fs));
            // cosine is derived from the above; check it too when norms are nonzero.
            if nn > 1e-6 {
                let nc = nd / (nn * naive_norm(&b));
                let cc = cosine(&a, &b);
                assert!((nc - cc).abs() < 1e-5, "cosine len={len}");
            }
        }
    }

    #[test]
    fn four_dots_preserve_single_row_bits() {
        let mut rng = Rng::new(71);
        for len in (0..=135).chain([256, 384, 768, 1536]) {
            let q: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let rows: [Vec<f32>; 4] = std::array::from_fn(|_| (0..len).map(|_| rng.next_f32()).collect());
            let refs = rows.each_ref().map(Vec::as_slice);
            let got = dot_four(&q, refs);
            for r in 0..4 { assert_eq!(got[r].to_bits(), dot(&q, refs[r]).to_bits(), "len={len}, row={r}"); }
        }
    }

    #[test]
    fn zero_norm_cosine_is_zero() {
        let a = [0.0f32; 16];
        let b = [1.0f32; 16];
        assert_eq!(cosine(&a, &b), 0.0);
        assert_eq!(cosine(&b, &a), 0.0);
    }
}

/// Whether the first page of `row` is in memory (Linux `mincore`; elsewhere
/// unknown, `false`). One syscall: a cheap probe of whether a mapped
/// file's rows are already in the page cache.
pub fn resident(row: &[f32]) -> bool {
    #[cfg(target_os = "linux")]
    {
        if row.is_empty() { return true; }
        let aligned = row.as_ptr() as usize & !4095usize;
        let mut vec = 0u8;
        // SAFETY: queries residency of one page the slice borrows; writes
        // one byte into `vec`.
        let ok = unsafe { libc::mincore(aligned as *mut libc::c_void, 4096, &mut vec) } == 0;
        ok && vec & 1 == 1
    }
    #[cfg(not(target_os = "linux"))]
    { let _ = row; false }
}

/// Hint that `row` (possibly in a mapped file) is about to be read: on unix,
/// `madvise(MADV_WILLNEED)` over its pages starts their reads now
/// (asynchronously); elsewhere, and for anonymous memory, a no-op.
pub fn will_need(row: &[f32]) {
    #[cfg(unix)]
    {
        if row.is_empty() { return; }
        let page = 4096usize;
        let start = row.as_ptr() as usize;
        let end = start + std::mem::size_of_val(row);
        let aligned = start & !(page - 1);
        // SAFETY: advisory only; the range lies within (page-rounded) memory
        // the slice borrows, and MADV_WILLNEED never changes its contents.
        unsafe { libc::madvise(aligned as *mut libc::c_void, end - aligned, libc::MADV_WILLNEED); }
    }
    #[cfg(not(unix))]
    let _ = row;
}
