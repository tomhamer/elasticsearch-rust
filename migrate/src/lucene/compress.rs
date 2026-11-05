//! Decompressors for Lucene stored fields: LZ4 (Lucene's block variant) and
//! raw DEFLATE, both with preset-dictionary support. Hand-written because the
//! pure-Rust inflate crates do not expose preset dictionaries.

use anyhow::{bail, Result};

/// Appends exactly `n` decompressed bytes to `dest`. Matches may reference
/// anything already in `dest` (that is how the preset dictionary works).
/// Returns the number of compressed bytes consumed.
pub fn lz4_decompress(src: &[u8], n: usize, dest: &mut Vec<u8>) -> Result<usize> {
    let end = dest.len() + n;
    let mut p = 0usize;
    let byte = |p: usize| -> Result<usize> {
        match src.get(p) {
            Some(b) => Ok(*b as usize),
            None => bail!("lz4: truncated input"),
        }
    };
    while dest.len() < end {
        let token = byte(p)?;
        p += 1;
        let mut lit = token >> 4;
        if lit == 0x0f {
            loop {
                let b = byte(p)?;
                p += 1;
                lit += b;
                if b != 0xff {
                    break;
                }
            }
        }
        if p + lit > src.len() {
            bail!("lz4: literal run past end of input");
        }
        dest.extend_from_slice(&src[p..p + lit]);
        p += lit;
        if dest.len() >= end {
            break;
        }
        let dec = byte(p)? | (byte(p + 1)? << 8);
        p += 2;
        let mut ml = token & 0x0f;
        if ml == 0x0f {
            loop {
                let b = byte(p)?;
                p += 1;
                ml += b;
                if b != 0xff {
                    break;
                }
            }
        }
        ml += 4;
        if dec == 0 || dec > dest.len() {
            bail!("lz4: invalid match offset {}", dec);
        }
        let from = dest.len() - dec;
        if dec >= ml {
            dest.extend_from_within(from..from + ml);
        } else {
            dest.reserve(ml);
            for i in 0..ml {
                let b = dest[from + i];
                dest.push(b);
            }
        }
    }
    if dest.len() != end {
        bail!("lz4: produced {} bytes more than expected", dest.len() - end);
    }
    Ok(p)
}

struct Bits<'a> {
    src: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
}

impl<'a> Bits<'a> {
    fn bits(&mut self, need: u32) -> Result<u32> {
        while self.cnt < need {
            let Some(b) = self.src.get(self.pos) else { bail!("inflate: truncated input") };
            self.buf |= (*b as u64) << self.cnt;
            self.cnt += 8;
            self.pos += 1;
        }
        let v = (self.buf & ((1u64 << need) - 1)) as u32;
        self.buf >>= need;
        self.cnt -= need;
        Ok(v)
    }
}

struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut offs = [0u16; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + count[i - 1];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Huffman { count, symbol }
    }

    fn decode(&self, b: &mut Bits) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= b.bits(1)? as i32;
            let count = self.count[len] as i32;
            if code - count < first {
                return Ok(self.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        bail!("inflate: invalid Huffman code")
    }
}

const LEN_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn inflate_codes(b: &mut Bits, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<()> {
    loop {
        let sym = lit.decode(b)? as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let s = sym - 257;
            if s >= 29 {
                bail!("inflate: invalid length symbol");
            }
            let len = LEN_BASE[s] as usize + b.bits(LEN_EXTRA[s] as u32)? as usize;
            let ds = dist.decode(b)? as usize;
            if ds >= 30 {
                bail!("inflate: invalid distance symbol");
            }
            let d = DIST_BASE[ds] as usize + b.bits(DIST_EXTRA[ds] as u32)? as usize;
            if d > out.len() {
                bail!("inflate: distance {} beyond window {}", d, out.len());
            }
            let from = out.len() - d;
            if d >= len {
                out.extend_from_within(from..from + len);
            } else {
                for i in 0..len {
                    let v = out[from + i];
                    out.push(v);
                }
            }
        }
    }
}

/// Raw DEFLATE (no zlib wrapper). `out` may already hold a preset dictionary;
/// decompressed bytes are appended after it.
pub fn inflate_raw(src: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut b = Bits { src, pos: 0, buf: 0, cnt: 0 };
    loop {
        let last = b.bits(1)?;
        match b.bits(2)? {
            0 => {
                b.buf = 0;
                b.cnt = 0;
                if b.pos + 4 > src.len() {
                    bail!("inflate: truncated stored block");
                }
                let len = src[b.pos] as usize | (src[b.pos + 1] as usize) << 8;
                b.pos += 4;
                if b.pos + len > src.len() {
                    bail!("inflate: truncated stored block");
                }
                out.extend_from_slice(&src[b.pos..b.pos + len]);
                b.pos += len;
            }
            1 => {
                let mut l = [0u8; 288];
                for (i, v) in l.iter_mut().enumerate() {
                    *v = match i {
                        0..=143 => 8,
                        144..=255 => 9,
                        256..=279 => 7,
                        _ => 8,
                    };
                }
                let lit = Huffman::new(&l);
                let dist = Huffman::new(&[5u8; 30]);
                inflate_codes(&mut b, out, &lit, &dist)?;
            }
            2 => {
                let nlen = b.bits(5)? as usize + 257;
                let ndist = b.bits(5)? as usize + 1;
                let ncode = b.bits(4)? as usize + 4;
                if nlen > 286 || ndist > 30 {
                    bail!("inflate: bad code counts");
                }
                let mut cl = [0u8; 19];
                for &o in CL_ORDER.iter().take(ncode) {
                    cl[o] = b.bits(3)? as u8;
                }
                let clh = Huffman::new(&cl);
                let mut lengths = vec![0u8; nlen + ndist];
                let mut i = 0;
                while i < nlen + ndist {
                    let sym = clh.decode(&mut b)?;
                    if sym < 16 {
                        lengths[i] = sym as u8;
                        i += 1;
                    } else {
                        let (val, rep) = match sym {
                            16 => {
                                if i == 0 {
                                    bail!("inflate: repeat with no previous length");
                                }
                                (lengths[i - 1], 3 + b.bits(2)? as usize)
                            }
                            17 => (0, 3 + b.bits(3)? as usize),
                            _ => (0, 11 + b.bits(7)? as usize),
                        };
                        if i + rep > nlen + ndist {
                            bail!("inflate: too many code lengths");
                        }
                        for _ in 0..rep {
                            lengths[i] = val;
                            i += 1;
                        }
                    }
                }
                let lit = Huffman::new(&lengths[..nlen]);
                let dist = Huffman::new(&lengths[nlen..]);
                inflate_codes(&mut b, out, &lit, &dist)?;
            }
            _ => bail!("inflate: invalid block type"),
        }
        if last == 1 {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_literals_and_overlapping_match() {
        // token: 2 literals, match len 4+2; "ab" then copy 6 bytes from distance 2; final token: 1 literal "c"
        let src = [0x22, b'a', b'b', 0x02, 0x00, 0x10, b'c'];
        let mut out = Vec::new();
        let used = lz4_decompress(&src, 9, &mut out).unwrap();
        assert_eq!(out, b"ababababc");
        assert_eq!(used, src.len());
    }

    #[test]
    fn lz4_uses_dictionary_prefix() {
        let mut out = b"hello ".to_vec();
        // 0 literals, match of 5 from distance 6, then last literal "!"
        let src = [0x01, 0x06, 0x00, 0x10, b'!'];
        lz4_decompress(&src, 6, &mut out).unwrap();
        assert_eq!(out, b"hello hello!");
    }

    #[test]
    fn inflate_fixed_block() {
        let mut out = Vec::new();
        inflate_raw(&[0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00], &mut out).unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn inflate_stored_block() {
        let mut out = Vec::new();
        inflate_raw(&[0x01, 0x03, 0x00, 0xfc, 0xff, b'a', b'b', b'c'], &mut out).unwrap();
        assert_eq!(out, b"abc");
    }
}
