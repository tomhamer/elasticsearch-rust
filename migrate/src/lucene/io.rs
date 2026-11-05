//! Byte-level primitives shared by every Lucene file format: vInts, the
//! big-endian codec header/footer and the little-endian payload (Lucene 9+).

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;

pub const CODEC_MAGIC: u32 = 0x3fd7_6c17;
pub const FOOTER_LEN: usize = 16;

pub struct In<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> In<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        In { buf, pos: 0 }
    }

    pub fn at(buf: &'a [u8], pos: usize) -> Self {
        In { buf, pos }
    }

    pub fn buf(&self) -> &'a [u8] {
        self.buf
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            bail!("unexpected end of data: need {} bytes at offset {} of {}", n, self.pos, self.buf.len());
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    pub fn be_u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn be_i32(&mut self) -> Result<i32> {
        Ok(self.be_u32()? as i32)
    }

    pub fn be_i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    pub fn le_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }

    pub fn le_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn le_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn le_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    pub fn le_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    pub fn vint(&mut self) -> Result<u32> {
        let mut v = 0u32;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u32) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 28 {
                bail!("invalid vInt at offset {}", self.pos);
            }
        }
    }

    pub fn vlong(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                bail!("invalid vLong at offset {}", self.pos);
            }
        }
    }

    pub fn zint(&mut self) -> Result<i32> {
        let v = self.vint()?;
        Ok(((v >> 1) as i32) ^ -((v & 1) as i32))
    }

    pub fn string(&mut self) -> Result<String> {
        let n = self.vint()? as usize;
        let b = self.bytes(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    pub fn string_set(&mut self) -> Result<Vec<String>> {
        let n = self.vint()? as usize;
        let mut out = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            out.push(self.string()?);
        }
        Ok(out)
    }

    pub fn string_map(&mut self) -> Result<BTreeMap<String, String>> {
        let n = self.vint()? as usize;
        let mut out = BTreeMap::new();
        for _ in 0..n {
            let k = self.string()?;
            let v = self.string()?;
            out.insert(k, v);
        }
        Ok(out)
    }
}

pub fn zigzag64(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

#[derive(Debug, Clone)]
pub struct Header {
    pub codec: String,
    pub version: i32,
    pub id: [u8; 16],
    pub suffix: String,
}

/// `CodecUtil.writeHeader`: magic, codec name, version (all big-endian).
pub fn read_header(inp: &mut In) -> Result<Header> {
    let magic = inp.be_u32()?;
    if magic != CODEC_MAGIC {
        bail!("not a Lucene codec file (magic {:#x})", magic);
    }
    let codec = inp.string()?;
    let version = inp.be_i32()?;
    Ok(Header { codec, version, id: [0; 16], suffix: String::new() })
}

/// `CodecUtil.writeIndexHeader`: header + 16-byte segment id + suffix.
pub fn read_index_header(inp: &mut In) -> Result<Header> {
    let mut h = read_header(inp)?;
    h.id.copy_from_slice(inp.bytes(16)?);
    let n = inp.u8()? as usize;
    h.suffix = String::from_utf8_lossy(inp.bytes(n)?).into_owned();
    Ok(h)
}

pub fn expect_codec(h: &Header, file: &str, allowed: &[&str]) -> Result<()> {
    if !allowed.contains(&h.codec.as_str()) {
        bail!("{}: unsupported format [{}] (supported: {})", file, h.codec, allowed.join(", "));
    }
    Ok(())
}

pub fn base36(mut n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    let mut s = Vec::new();
    while n > 0 {
        s.push(b"0123456789abcdefghijklmnopqrstuvwxyz"[(n % 36) as usize]);
        n /= 36;
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

pub fn parse_base36(s: &str) -> Result<u64> {
    u64::from_str_radix(s, 36).with_context(|| format!("bad base-36 generation [{}]", s))
}
