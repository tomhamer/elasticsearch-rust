//! Sequential reader for `Lucene90CompressingStoredFieldsFormat` data files
//! (`.fdt`), including Elasticsearch's zstd variant. Chunks are scanned front
//! to back, so the `.fdx` index is never needed.

use super::compress::{inflate_raw, lz4_decompress};
use super::io::{read_index_header, zigzag64, In, FOOTER_LEN};
use super::Segment;
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub enum Value {
    Str(String),
    Bytes(Vec<u8>),
    Int(i32),
    Long(i64),
    Other,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Lz4,
    Deflate,
    Zstd,
}

fn read_ints(inp: &mut In, n: usize) -> Result<Vec<u32>> {
    if n == 1 {
        return Ok(vec![inp.vint()?]);
    }
    let bits = inp.u8()?;
    let mut v = vec![0u32; n];
    match bits {
        0 => {
            let x = inp.vint()?;
            v.iter_mut().for_each(|e| *e = x);
        }
        8 | 16 | 32 => {
            let per_long = 64 / bits as usize;
            let stride = 128 / per_long;
            let mut k = 0;
            while k + 128 <= n {
                for i in 0..stride {
                    let l = inp.le_u64()?;
                    for j in 0..per_long {
                        let shift = 64 - bits as usize * (j + 1);
                        v[k + j * stride + i] = ((l >> shift) & ((1u64 << bits) - 1)) as u32;
                    }
                }
                k += 128;
            }
            for e in v.iter_mut().skip(k) {
                *e = match bits {
                    8 => inp.u8()? as u32,
                    16 => inp.le_u16()? as u32,
                    _ => inp.le_u32()?,
                };
            }
        }
        x => bail!("stored fields: bad int encoding {}", x),
    }
    Ok(v)
}

fn decompress(inp: &mut In, mode: Mode, len: usize, out: &mut Vec<u8>) -> Result<()> {
    match mode {
        Mode::Zstd => {
            let clen = inp.vint()? as usize;
            let src = inp.bytes(clen)?;
            let mut dec = ruzstd::StreamingDecoder::new(src)?;
            let before = out.len();
            std::io::Read::read_to_end(&mut dec, out)?;
            if out.len() - before != len {
                bail!("zstd chunk: got {} bytes, expected {}", out.len() - before, len);
            }
        }
        Mode::Lz4 => {
            let dict_len = inp.vint()? as usize;
            let block_len = inp.vint()? as usize;
            let dict_clen = inp.vint()? as usize;
            let mut clens = Vec::new();
            let mut total = dict_len;
            while total < len {
                clens.push(inp.vint()? as usize);
                total += block_len;
            }
            let mut dict = Vec::with_capacity(dict_len);
            lz4_decompress(inp.bytes(dict_clen)?, dict_len, &mut dict)?;
            out.extend_from_slice(&dict);
            let mut produced = dict_len;
            for c in clens {
                let this = block_len.min(len - produced);
                let mut b = dict.clone();
                lz4_decompress(inp.bytes(c)?, this, &mut b)?;
                out.extend_from_slice(&b[dict_len..]);
                produced += this;
            }
        }
        Mode::Deflate => {
            let _dict_len = inp.vint()?;
            let block_len = inp.vint()? as usize;
            let read_block = |inp: &mut In, buf: &mut Vec<u8>| -> Result<()> {
                let c = inp.vint()? as usize;
                if c > 0 {
                    inflate_raw(inp.bytes(c)?, buf)?;
                }
                Ok(())
            };
            let mut dict = Vec::new();
            read_block(inp, &mut dict)?;
            out.extend_from_slice(&dict);
            let mut produced = dict.len();
            while produced < len {
                let mut b = dict.clone();
                read_block(inp, &mut b)?;
                out.extend_from_slice(&b[dict.len()..]);
                produced += block_len;
            }
        }
    }
    Ok(())
}

/// Calls `f(doc_id, fields)` for every live doc in the segment.
pub fn for_each_doc(seg: &Segment, mut f: impl FnMut(u32, Vec<(String, Value)>) -> Result<()>) -> Result<()> {
    let fdt = seg.files.open(".fdt")?;
    let b = fdt.as_slice();
    let mut inp = In::new(b);
    let h = read_index_header(&mut inp)?;
    let mode = if h.codec.contains("Zstd") {
        Mode::Zstd
    } else if h.codec.ends_with("HighData") {
        Mode::Deflate
    } else if h.codec.ends_with("FastData") {
        Mode::Lz4
    } else {
        bail!("{}.fdt: unsupported stored fields format {}", seg.name, h.codec);
    };
    let end = b.len() - FOOTER_LEN;
    // chunk size (needed to split sliced chunks) is the first value in .fdm
    let fdm = seg.files.open(".fdm")?;
    let mut mi = In::new(fdm.as_slice());
    read_index_header(&mut mi)?;
    let chunk_size = mi.vint()? as usize;
    if chunk_size == 0 {
        bail!("{}.fdm: chunk size 0", seg.name);
    }
    while inp.pos() < end {
        let doc_base = inp.vint()?;
        let token = inp.vint()?;
        let n = (token >> 2) as usize;
        let sliced = token & 1 != 0;
        let nfields = read_ints(&mut inp, n)?;
        let lens = read_ints(&mut inp, n)?;
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let mut data = Vec::with_capacity(total);
        if sliced {
            let mut done = 0;
            while done < total {
                let part = chunk_size.min(total - done);
                decompress(&mut inp, mode, part, &mut data)?;
                done += part;
            }
        } else if total > 0 {
            decompress(&mut inp, mode, total, &mut data)?;
        }
        let mut d = In::new(&data);
        for i in 0..n {
            let doc = doc_base + i as u32;
            let mut fields = Vec::with_capacity(nfields[i] as usize);
            for _ in 0..nfields[i] {
                let info = d.vlong()?;
                let name = seg.field_name((info >> 3) as u32).unwrap_or("?").to_string();
                let v = match info & 7 {
                    0 => Value::Str(d.string()?),
                    1 => {
                        let l = d.vint()? as usize;
                        Value::Bytes(d.bytes(l)?.to_vec())
                    }
                    2 => Value::Int(d.zint()?),
                    3 => {
                        let b0 = d.u8()?;
                        d.skip(if b0 == 0xff { 4 } else if b0 & 0x80 != 0 { 0 } else { 3 })?;
                        Value::Other
                    }
                    4 => {
                        let b0 = d.u8()?;
                        let mut v = (b0 & 0x1f) as u64;
                        if b0 & 0x20 != 0 {
                            v |= d.vlong()? << 5;
                        }
                        let unit: i64 = match b0 & 0xc0 {
                            0x40 => 1000,
                            0x80 => 3_600_000,
                            0xc0 => 86_400_000,
                            _ => 1,
                        };
                        Value::Long(zigzag64(v).wrapping_mul(unit))
                    }
                    5 => {
                        let b0 = d.u8()?;
                        d.skip(match b0 {
                            0xff => 8,
                            0xfe => 4,
                            x if x & 0x80 != 0 => 0,
                            _ => 7,
                        })?;
                        Value::Other
                    }
                    t => bail!("{}: unknown stored field type {}", seg.name, t),
                };
                fields.push((name, v));
            }
            if seg.is_live(doc) {
                f(doc, fields)?;
            }
        }
    }
    Ok(())
}
