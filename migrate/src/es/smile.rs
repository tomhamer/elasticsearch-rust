//! Decoder for SMILE (binary JSON) as written by Elasticsearch's Jackson
//! setup: shared key names, optional shared string values, raw binary.

use anyhow::{bail, Result};
use serde_json::{Map, Number, Value};

pub const HEADER: &[u8] = b":)\n";

/// Decodes a SMILE document. With `expand_binary`, binary values that are
/// themselves (possibly DEFLATE-compressed) JSON/SMILE documents are decoded
/// in place, which is how ES stores mappings and pipeline configs.
pub fn decode(b: &[u8], expand_binary: bool) -> Result<Value> {
    if b.len() < 4 || &b[..3] != HEADER {
        bail!("not a SMILE document");
    }
    let flags = b[3];
    let mut d = Dec {
        b,
        p: 4,
        names: Vec::new(),
        values: Vec::new(),
        share_values: flags & 0x02 != 0,
        expand_binary,
    };
    d.value()
}

struct Dec<'a> {
    b: &'a [u8],
    p: usize,
    names: Vec<String>,
    values: Vec<String>,
    share_values: bool,
    expand_binary: bool,
}

fn push_shared(t: &mut Vec<String>, s: &str) {
    if t.len() == 1024 {
        t.clear();
    }
    t.push(s.to_string());
}

impl<'a> Dec<'a> {
    fn byte(&mut self) -> Result<u8> {
        match self.b.get(self.p) {
            Some(&x) => {
                self.p += 1;
                Ok(x)
            }
            None => bail!("SMILE: unexpected end of input"),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("SMILE: unexpected end of input");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    fn text(&mut self, n: usize) -> Result<String> {
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn until_fc(&mut self) -> Result<String> {
        let start = self.p;
        while self.byte()? != 0xfc {}
        Ok(String::from_utf8_lossy(&self.b[start..self.p - 1]).into_owned())
    }

    fn uvint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        loop {
            let x = self.byte()?;
            if x & 0x80 != 0 {
                return Ok((v << 6) | (x & 0x3f) as u64);
            }
            v = (v << 7) | x as u64;
        }
    }

    fn zz(&mut self) -> Result<i64> {
        let v = self.uvint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    fn seven_bit(&mut self, n: usize) -> Result<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 7) | (self.byte()? & 0x7f) as u64;
        }
        Ok(v)
    }

    fn seven_bit_binary(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(len);
        let full = len / 7;
        for _ in 0..full {
            let v = self.seven_bit(8)?;
            for i in (0..7).rev() {
                out.push((v >> (i * 8)) as u8);
            }
        }
        let left = len - full * 7;
        if left > 0 {
            let mut v = (self.byte()? & 0x7f) as u64;
            for i in 1..left {
                v = (v << 7) | (self.byte()? & 0x7f) as u64;
                out.push((v >> (7 - i)) as u8);
            }
            v <<= left;
            out.push((v as u8).wrapping_add(self.byte()?));
        }
        Ok(out)
    }

    fn key(&mut self, t: u8) -> Result<String> {
        let s = match t {
            0x20 => return Ok(String::new()),
            0x30..=0x33 => {
                let i = (((t & 3) as usize) << 8) | self.byte()? as usize;
                return self.names.get(i).cloned().ok_or_else(|| anyhow::anyhow!("SMILE: bad shared name {}", i));
            }
            0x40..=0x7f => {
                let i = (t & 0x3f) as usize;
                return self.names.get(i).cloned().ok_or_else(|| anyhow::anyhow!("SMILE: bad shared name {}", i));
            }
            0x34 => self.until_fc()?,
            0x80..=0xbf => self.text((t & 0x3f) as usize + 1)?,
            0xc0..=0xf7 => self.text((t & 0x3f) as usize + 2)?,
            x => bail!("SMILE: invalid key token {:#x} at {}", x, self.p - 1),
        };
        push_shared(&mut self.names, &s);
        Ok(s)
    }

    fn short_string(&mut self, n: usize) -> Result<Value> {
        let s = self.text(n)?;
        if self.share_values && s.len() <= 64 {
            push_shared(&mut self.values, &s);
        }
        Ok(Value::String(s))
    }

    fn value(&mut self) -> Result<Value> {
        let t = self.byte()?;
        Ok(match t {
            0x01..=0x1f => {
                let i = t as usize - 1;
                Value::String(self.values.get(i).cloned().ok_or_else(|| anyhow::anyhow!("SMILE: bad shared value {}", i))?)
            }
            0x20 => Value::String(String::new()),
            0x21 => Value::Null,
            0x22 => Value::Bool(false),
            0x23 => Value::Bool(true),
            0x24 | 0x25 => Value::from(self.zz()?),
            0x26 => {
                let n = self.uvint()? as usize;
                let bytes = self.seven_bit_binary(n)?;
                num(&bigint_to_string(&bytes))
            }
            0x28 => {
                let f = f32::from_bits(self.seven_bit(5)? as u32);
                if f.is_finite() { num(&format!("{}", f)) } else { Value::Null }
            }
            0x29 => {
                let f = f64::from_bits(self.seven_bit(10)?);
                serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
            }
            0x2a => {
                let scale = self.zz()?;
                let n = self.uvint()? as usize;
                let bytes = self.seven_bit_binary(n)?;
                num(&format!("{}E{}", bigint_to_string(&bytes), -scale))
            }
            0x40..=0x5f => self.short_string((t & 0x1f) as usize + 1)?,
            0x60..=0x7f => self.short_string((t & 0x1f) as usize + 33)?,
            0x80..=0x9f => self.short_string((t & 0x1f) as usize + 2)?,
            0xa0..=0xbf => self.short_string((t & 0x1f) as usize + 34)?,
            0xc0..=0xdf => {
                let v = (t & 0x1f) as i64;
                Value::from((v >> 1) ^ -(v & 1))
            }
            0xe0 | 0xe4 => Value::String(self.until_fc()?),
            0xe8 => {
                let n = self.uvint()? as usize;
                let bytes = self.seven_bit_binary(n)?;
                self.binary(bytes)
            }
            0xec..=0xef => {
                let i = (((t & 3) as usize) << 8) | self.byte()? as usize;
                Value::String(self.values.get(i).cloned().ok_or_else(|| anyhow::anyhow!("SMILE: bad shared value {}", i))?)
            }
            0xfd => {
                let n = self.uvint()? as usize;
                let bytes = self.take(n)?.to_vec();
                self.binary(bytes)
            }
            0xf8 => {
                let mut a = Vec::new();
                loop {
                    if self.b.get(self.p) == Some(&0xf9) {
                        self.p += 1;
                        break;
                    }
                    a.push(self.value()?);
                }
                Value::Array(a)
            }
            0xfa => {
                let mut m = Map::new();
                loop {
                    let k = self.byte()?;
                    if k == 0xfb {
                        break;
                    }
                    let key = self.key(k)?;
                    let v = self.value()?;
                    m.insert(key, v);
                }
                Value::Object(m)
            }
            x => bail!("SMILE: invalid value token {:#x} at {}", x, self.p - 1),
        })
    }

    fn binary(&self, bytes: Vec<u8>) -> Value {
        if self.expand_binary {
            if let Ok(v) = super::decode_blob(&bytes) {
                return v;
            }
        }
        Value::String(super::base64(&bytes, false))
    }
}

fn num(s: &str) -> Value {
    serde_json::from_str::<Number>(s).map(Value::Number).unwrap_or(Value::Null)
}

/// Two's-complement big-endian bytes to a decimal string.
fn bigint_to_string(b: &[u8]) -> String {
    if b.is_empty() {
        return "0".into();
    }
    let neg = b[0] & 0x80 != 0;
    let mut mag: Vec<u8> = if neg {
        let mut v: Vec<u8> = b.iter().map(|x| !x).collect();
        for x in v.iter_mut().rev() {
            let (r, c) = x.overflowing_add(1);
            *x = r;
            if !c {
                break;
            }
        }
        v
    } else {
        b.to_vec()
    };
    let mut digits = Vec::new();
    while mag.iter().any(|&x| x != 0) {
        let mut rem = 0u32;
        for x in mag.iter_mut() {
            let cur = (rem << 8) | *x as u32;
            *x = (cur / 10) as u8;
            rem = cur % 10;
        }
        digits.push(b'0' + rem as u8);
    }
    if digits.is_empty() {
        digits.push(b'0');
    }
    if neg {
        digits.push(b'-');
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_with_shared_names() {
        // {"a":1,"b":[true,"x"],"a2":{"a":null}} using a shared-name back reference for the inner "a"
        let doc = [
            b':', b')', b'\n', 0x01, 0xfa, 0x80, b'a', 0xc2, 0x80, b'b', 0xf8, 0x23, 0x40, b'x', 0xf9, 0x81, b'a', b'2',
            0xfa, 0x40, 0x21, 0xfb, 0xfb,
        ];
        let v = decode(&doc, false).unwrap();
        assert_eq!(v, serde_json::json!({"a": 1, "b": [true, "x"], "a2": {"a": null}}));
    }

    #[test]
    fn vint_and_bigint() {
        let doc = [b':', b')', b'\n', 0x00, 0x24, 0x01, 0x80 | 0x14];
        // zigzag(100) = 200 = 0b11_001000 -> first byte 0x03, last byte 0x80|0x08
        let doc2 = [b':', b')', b'\n', 0x00, 0x24, 0x03, 0x88];
        assert_eq!(decode(&doc2, false).unwrap(), serde_json::json!(100));
        let _ = doc;
        assert_eq!(bigint_to_string(&[0xff, 0xff]), "-1");
        assert_eq!(bigint_to_string(&[0x01, 0x00]), "256");
    }
}
