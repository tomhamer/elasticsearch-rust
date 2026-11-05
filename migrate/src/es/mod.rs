//! Elasticsearch-specific encodings on top of Lucene: SMILE metadata, the
//! binary `_id` encoding, compressed x-content and the on-disk node layout.

pub mod smile;
pub mod state;

use crate::lucene::compress::inflate_raw;
use anyhow::{bail, Result};
use serde_json::Value;

const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Standard base64 with padding, or URL-safe without padding.
pub fn base64(b: &[u8], url: bool) -> String {
    let alpha = if url { B64_URL } else { B64_STD };
    let mut s = String::with_capacity((b.len() + 2) / 3 * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        let chars = c.len() + 1;
        for i in 0..4 {
            if i < chars {
                s.push(alpha[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else if !url {
                s.push('=');
            }
        }
    }
    s
}

/// `org.elasticsearch.index.mapper.Uid.decodeId`.
pub fn decode_id(b: &[u8]) -> String {
    const NUMERIC: u8 = 0xfe;
    const UTF8: u8 = 0xff;
    const BASE64_ESCAPE: u8 = 0xfd;
    match b.first() {
        None => String::new(),
        Some(&NUMERIC) => {
            let mut s = String::with_capacity((b.len() - 1) * 2);
            for (i, &x) in b.iter().enumerate().skip(1) {
                s.push((b'0' + (x >> 4)) as char);
                let lo = x & 0x0f;
                if i == b.len() - 1 && lo == 0x0f {
                    break;
                }
                s.push((b'0' + lo) as char);
            }
            s
        }
        Some(&UTF8) => String::from_utf8_lossy(&b[1..]).into_owned(),
        Some(&BASE64_ESCAPE) => base64(&b[1..], true),
        Some(_) => base64(b, true),
    }
}

/// Decodes a stored x-content blob: optionally `DFL\0`-prefixed DEFLATE,
/// then SMILE or JSON.
pub fn decode_blob(b: &[u8]) -> Result<Value> {
    if b.starts_with(b"DFL\0") {
        let mut out = Vec::new();
        inflate_raw(&b[4..], &mut out)?;
        return decode_blob(&out);
    }
    if b.starts_with(smile::HEADER) {
        return smile::decode(b, true);
    }
    match b.iter().find(|c| !c.is_ascii_whitespace()) {
        Some(b'{') | Some(b'[') => Ok(serde_json::from_slice(b)?),
        _ => bail!("not a JSON or SMILE document"),
    }
}

/// `_source` bytes to JSON text. JSON is passed through untouched so field
/// order and number formatting survive exactly.
pub fn source_to_json(b: &[u8]) -> Result<String> {
    if b.starts_with(smile::HEADER) {
        return Ok(serde_json::to_string(&smile::decode(b, false)?)?);
    }
    match b.iter().find(|c| !c.is_ascii_whitespace()) {
        Some(b'{') => {
            let s = std::str::from_utf8(b)?;
            // bulk bodies are newline-delimited: re-serialise multi-line JSON
            if s.contains('\n') || s.contains('\r') {
                let v: Value = serde_json::from_str(s)?;
                return Ok(serde_json::to_string(&v)?);
            }
            Ok(s.to_string())
        }
        Some(&c) if c == 0xbf || (0xa0..=0xbb).contains(&c) => bail!("_source is CBOR, which is not supported yet"),
        _ => bail!("_source is not JSON or SMILE (YAML/CBOR sources are not supported yet)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert_eq!(decode_id(&[0xfe, 0x12, 0x3f]), "123");
        assert_eq!(decode_id(&[0xfe, 0x12, 0x34]), "1234");
        assert_eq!(decode_id(b"\xffmy id"), "my id");
        // auto-generated ids are base64url of the raw bytes
        assert_eq!(decode_id(&base64_decode_url("kHyq3pIBAAAAhqtjXAKH")), "kHyq3pIBAAAAhqtjXAKH");
    }

    fn base64_decode_url(s: &str) -> Vec<u8> {
        let mut bits = 0u32;
        let mut n = 0;
        let mut out = Vec::new();
        for c in s.bytes() {
            let v = B64_URL.iter().position(|&x| x == c).unwrap() as u32;
            bits = bits << 6 | v;
            n += 6;
            if n >= 8 {
                n -= 8;
                out.push((bits >> n) as u8);
            }
        }
        out
    }

    #[test]
    fn b64() {
        assert_eq!(base64(b"hello", false), "aGVsbG8=");
        assert_eq!(base64(b"hello", true), "aGVsbG8");
    }
}
