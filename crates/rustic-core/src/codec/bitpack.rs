//! Fixed-width bit packing for blocks of u32 values.
//!
//! Format: one byte `bits` (0..=32), then `ceil(n * bits / 8)` bytes of
//! little-endian packed values. `bits == 0` means every value is zero and
//! no payload follows. This is Lucene's FOR layout without SIMD yet.

use crate::error::{Result, RusticError};

/// Number of payload bytes for `n` values at `bits` width.
#[inline]
pub fn packed_len(n: usize, bits: usize) -> usize {
    (n * bits + 7) / 8
}

/// Append the packed form of `values` to `out`.
pub fn pack(values: &[u32], out: &mut Vec<u8>) {
    let max = values.iter().copied().max().unwrap_or(0);
    let bits = if max == 0 { 0 } else { 32 - max.leading_zeros() as usize };
    out.push(bits as u8);
    if bits == 0 {
        return;
    }
    match bits {
        8 => {
            out.extend(values.iter().map(|&v| v as u8));
            return;
        }
        16 => {
            for &v in values {
                out.extend_from_slice(&(v as u16).to_le_bytes());
            }
            return;
        }
        32 => {
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            return;
        }
        _ => {}
    }
    let start = out.len();
    out.resize(start + packed_len(values.len(), bits), 0);
    let buf = &mut out[start..];
    let mut bitpos = 0usize;
    for &value in values {
        let mut v = value as u64;
        let mut remaining = bits;
        while remaining > 0 {
            let byte = bitpos / 8;
            let off = bitpos % 8;
            let take = (8 - off).min(remaining);
            buf[byte] |= ((v & ((1u64 << take) - 1)) as u8) << off;
            v >>= take;
            bitpos += take;
            remaining -= take;
        }
    }
}

/// Decode `n` values from the start of `data`, appending to `out`.
/// Returns the number of bytes consumed.
///
/// Hot path: one unaligned 64-bit load per value (the payload is always
/// followed by at least the block header of the next block or the section
/// footer, but we never rely on that: the tail falls back to a byte loop).
pub fn unpack(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<usize> {
    let bits = *data
        .first()
        .ok_or_else(|| RusticError::Corrupted("bitpack: missing header".into()))?
        as usize;
    if bits > 32 {
        return Err(RusticError::Corrupted(format!("bitpack: bad width {bits}")));
    }
    if bits == 0 {
        out.extend(std::iter::repeat_n(0u32, n));
        return Ok(1);
    }
    let nbytes = packed_len(n, bits);
    let buf = data
        .get(1..1 + nbytes)
        .ok_or_else(|| RusticError::Corrupted("bitpack: truncated payload".into()))?;
    out.reserve(n);
    match bits {
        8 => out.extend(buf.iter().map(|&b| b as u32)),
        16 => out.extend(buf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)),
        32 => out.extend(buf.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        _ => {
            let mask = (1u64 << bits) - 1;
            // Values whose 8-byte window lies fully inside `buf` take the fast path.
            let fast_n = if buf.len() >= 8 { ((buf.len() - 8) * 8) / bits + 1 } else { 0 }.min(n);
            let mut bitpos = 0usize;
            for _ in 0..fast_n {
                let byte = bitpos >> 3;
                let w = u64::from_le_bytes(buf[byte..byte + 8].try_into().unwrap());
                out.push(((w >> (bitpos & 7)) & mask) as u32);
                bitpos += bits;
            }
            for _ in fast_n..n {
                let mut v = 0u64;
                let mut got = 0usize;
                while got < bits {
                    let byte = bitpos / 8;
                    let off = bitpos % 8;
                    let take = (8 - off).min(bits - got);
                    let chunk = ((buf[byte] >> off) as u64) & ((1u64 << take) - 1);
                    v |= chunk << got;
                    got += take;
                    bitpos += take;
                }
                out.push(v as u32);
            }
        }
    }
    Ok(1 + nbytes)
}

// ---------------------------------------------------------------------------
// 128-value blocks in the 4-lane interleaved layout (`bitpacking`'s
// `BitPacker4x`: NEON on aarch64, SSE3/AVX on x86_64, scalar fallback).
//
// Lane `l` holds values `l, l+4, l+8, ...` packed LSB-first into its own
// stream of little-endian u32 words; the four streams are interleaved word by
// word (16-byte rows), so a block at `bits` width is exactly `16 * bits`
// bytes. Besides the SIMD bulk decode, any single value can be read in O(1)
// ([`get4x`]), which conjunctions and phrase confirmation use to avoid
// decoding whole blocks.
// ---------------------------------------------------------------------------

use bitpacking::{BitPacker, BitPacker4x};

/// Values per SIMD block.
pub const BLOCK128: usize = 128;

#[inline]
fn packer() -> BitPacker4x {
    BitPacker4x::new()
}

/// Bytes of a 4x block at `bits` width.
#[inline(always)]
pub const fn len4x(bits: u8) -> usize {
    16 * bits as usize
}

/// Bits needed for the largest of `values` (exactly 128 values).
#[inline]
pub fn num_bits4x(values: &[u32]) -> u8 {
    packer().num_bits(values)
}

/// Bits needed for the deltas of sorted `values` relative to `initial`.
#[inline]
pub fn num_bits_sorted4x(initial: u32, values: &[u32]) -> u8 {
    packer().num_bits_sorted(initial, values)
}

/// Append 128 values packed at `bits` (no header).
pub fn pack4x(values: &[u32], bits: u8, out: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), BLOCK128);
    if bits == 0 {
        return;
    }
    let start = out.len();
    out.resize(start + len4x(bits), 0);
    packer().compress(values, &mut out[start..], bits);
}

/// Append 128 non-decreasing values packed as deltas from `initial`.
pub fn pack_sorted4x(initial: u32, values: &[u32], bits: u8, out: &mut Vec<u8>) {
    debug_assert_eq!(values.len(), BLOCK128);
    if bits == 0 {
        return;
    }
    let start = out.len();
    out.resize(start + len4x(bits), 0);
    packer().compress_sorted(initial, values, &mut out[start..], bits);
}

/// Decode 128 values (`data` holds at least `len4x(bits)` bytes).
#[inline]
pub fn unpack4x(data: &[u8], bits: u8, out: &mut [u32; BLOCK128]) {
    if bits == 0 {
        *out = [0; BLOCK128];
        return;
    }
    packer().decompress(&data[..len4x(bits)], out, bits);
}

/// Decode 128 delta-coded values, integrating from `initial` (fused SIMD
/// prefix sum).
#[inline]
pub fn unpack_sorted4x(initial: u32, data: &[u8], bits: u8, out: &mut [u32; BLOCK128]) {
    if bits == 0 {
        *out = [initial; BLOCK128];
        return;
    }
    packer().decompress_sorted(initial, &data[..len4x(bits)], out, bits);
}

/// Random access: value `i` of a 4x block packed at `bits`.
#[inline(always)]
pub fn get4x(data: &[u8], bits: u8, i: usize) -> u32 {
    if bits == 0 {
        return 0;
    }
    let b = bits as usize;
    let lane = i & 3;
    let off = (i >> 2) * b;
    let k = off >> 5;
    let s = off & 31;
    let word = |w: usize| -> u32 {
        let o = (w * 4 + lane) * 4;
        u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
    };
    let mut v = (word(k) as u64) >> s;
    if s + b > 32 {
        v |= (word(k + 1) as u64) << (32 - s);
    }
    let mask = if b == 32 { u32::MAX as u64 } else { (1u64 << b) - 1 };
    (v & mask) as u32
}

// ---------------------------------------------------------------------------
// Sequential LSB-first packing with O(1) random access (positions).
//
// Value `i` occupies bits `[i*bits, (i+1)*bits)`. Readers do one unaligned
// 8-byte load per value, so callers must guarantee `SEQ_PAD` readable bytes
// after the last packed byte.
// ---------------------------------------------------------------------------

/// Slack required after sequentially packed data.
pub const SEQ_PAD: usize = 8;

/// Bits needed to represent `max`.
#[inline]
pub fn bits_for(max: u32) -> u8 {
    (32 - max.leading_zeros()) as u8
}

/// Append `values` packed sequentially at `bits` (no header).
pub fn pack_seq(values: &[u32], bits: u8, out: &mut Vec<u8>) {
    if bits == 0 || values.is_empty() {
        return;
    }
    let start = out.len();
    let len = packed_len(values.len(), bits as usize);
    out.resize(start + len + 8, 0);
    let buf = &mut out[start..];
    let mut bp = 0usize;
    for &v in values {
        let byte = bp >> 3;
        let mut w = u64::from_le_bytes(buf[byte..byte + 8].try_into().unwrap());
        w |= (v as u64) << (bp & 7);
        buf[byte..byte + 8].copy_from_slice(&w.to_le_bytes());
        bp += bits as usize;
    }
    out.truncate(start + len);
}

/// Decode `out.len()` values starting at value index `first`. Panics if
/// `data` lacks the `SEQ_PAD` slack.
#[inline]
pub fn unpack_seq_at(data: &[u8], bits: u8, first: usize, out: &mut [u32]) {
    if bits == 0 {
        out.fill(0);
        return;
    }
    let b = bits as usize;
    let end_byte = ((first + out.len()) * b + 7) / 8;
    assert!(data.len() >= end_byte + SEQ_PAD - 1, "bitpack: missing slack");
    let mask: u64 = if b == 32 { u32::MAX as u64 } else { (1u64 << b) - 1 };
    let p = data.as_ptr();
    let mut bp = first * b;
    for o in out.iter_mut() {
        // SAFETY: bp >> 3 + 8 <= end_byte + 7 <= data.len() (asserted).
        let w = u64::from_le(unsafe { (p.add(bp >> 3) as *const u64).read_unaligned() });
        *o = ((w >> (bp & 7)) & mask) as u32;
        bp += b;
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;

    #[test]
    fn get4x_matches_crate_layout() {
        for bits in 0..=32u8 {
            let max = if bits == 32 { u32::MAX } else { (1u64 << bits) as u32 - 1 };
            let vals: Vec<u32> = (0..128u32).map(|i| i.wrapping_mul(2654435761) & max).collect();
            let mut buf = Vec::new();
            pack4x(&vals, bits, &mut buf);
            assert_eq!(buf.len(), len4x(bits));
            let mut out = [0u32; BLOCK128];
            unpack4x(&buf, bits, &mut out);
            assert_eq!(&out[..], &vals[..], "bits {bits}");
            for i in 0..128 {
                assert_eq!(get4x(&buf, bits, i), vals[i], "bits {bits} i {i}");
            }
        }
    }

    #[test]
    fn sorted4x_equals_plain_deltas() {
        // compress_sorted(0, cumulative) == compress(deltas): one payload
        // serves both tf-1 random access and cumulative decode.
        let tf1: Vec<u32> = (0..128u32).map(|i| (i * 7) % 5).collect();
        let mut cum = Vec::new();
        let mut c = 0;
        for &t in &tf1 {
            c += t;
            cum.push(c);
        }
        let bits = num_bits4x(&tf1);
        assert_eq!(bits, num_bits_sorted4x(0, &cum));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        pack4x(&tf1, bits, &mut a);
        pack_sorted4x(0, &cum, bits, &mut b);
        assert_eq!(a, b);
        let mut out = [0u32; BLOCK128];
        unpack_sorted4x(0, &a, bits, &mut out);
        assert_eq!(&out[..], &cum[..]);
    }

    #[test]
    fn seq_random_access() {
        for bits in 1..=32u8 {
            let max = if bits == 32 { u32::MAX } else { (1u64 << bits) as u32 - 1 };
            let vals: Vec<u32> = (0..300u32).map(|i| i.wrapping_mul(2654435761) & max).collect();
            let mut buf = Vec::new();
            pack_seq(&vals, bits, &mut buf);
            buf.extend_from_slice(&[0; SEQ_PAD]);
            for first in [0usize, 1, 7, 128, 297] {
                let mut out = [0u32; 3];
                unpack_seq_at(&buf, bits, first, &mut out);
                assert_eq!(&out[..], &vals[first..first + 3], "bits {bits} first {first}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[u32]) {
        let mut buf = Vec::new();
        pack(values, &mut buf);
        buf.extend_from_slice(&[0xAA, 0xBB]); // trailing garbage must be ignored
        let mut out = Vec::new();
        let used = unpack(&buf, values.len(), &mut out).unwrap();
        assert_eq!(out, values);
        assert_eq!(used, buf.len() - 2);
    }

    #[test]
    fn roundtrips() {
        roundtrip(&[]);
        roundtrip(&[0, 0, 0]);
        roundtrip(&[1, 2, 3, 4, 5, 6, 7]);
        roundtrip(&[255, 1, 128, 0, 77]);
        roundtrip(&[u32::MAX, 0, u32::MAX]);
        let big: Vec<u32> = (0..128).map(|i| i * 1000 + 7).collect();
        roundtrip(&big);
        // Every width 1..=32, lengths that exercise the fast/slow split.
        for bits in 1..=32u32 {
            let max = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
            for n in [1usize, 2, 7, 8, 9, 63, 64, 65, 128, 129] {
                let vals: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761) & max).collect();
                roundtrip(&vals);
            }
        }
    }
}
