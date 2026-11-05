//! LEB128 variable-length integers and fixed-width little-endian helpers.

use crate::error::{Result, RusticError};

fn truncated() -> RusticError {
    RusticError::Corrupted("unexpected end of data".into())
}

#[inline]
pub fn write_vint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub fn read_vint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *data.get(*pos).ok_or_else(truncated)?;
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte < 0x80 {
            return Ok(result);
        }
        shift += 7;
        if shift > 63 {
            return Err(RusticError::Corrupted("varint too long".into()));
        }
    }
}

#[inline]
pub fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32> {
    let bytes = data.get(*pos..*pos + 4).ok_or_else(truncated)?;
    *pos += 4;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

#[inline]
pub fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64> {
    let bytes = data.get(*pos..*pos + 8).ok_or_else(truncated)?;
    *pos += 8;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

#[inline]
pub fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    let bytes = data.get(*pos..*pos + len).ok_or_else(truncated)?;
    *pos += len;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vint_roundtrip() {
        let values = [0u64, 1, 127, 128, 300, 16383, 16384, u32::MAX as u64, u64::MAX];
        let mut buf = Vec::new();
        for &v in &values {
            write_vint(v, &mut buf);
        }
        let mut pos = 0;
        for &v in &values {
            assert_eq!(read_vint(&buf, &mut pos).unwrap(), v);
        }
        assert_eq!(pos, buf.len());
        assert!(read_vint(&buf, &mut pos).is_err());
    }
}
