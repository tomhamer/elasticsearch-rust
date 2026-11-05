//! Scroll and point-in-time ids, byte-compatible with ES's
//! `TransportSearchHelper.buildScrollId` / `parseScrollId` and
//! `SearchContextId.encode` / `decode`, so that malformed ids fail exactly
//! like ES (`Cannot parse scroll id` / `java.io.EOFException` chains with
//! the Java stream exception as the cause).
//!
//! Decoding only validates and extracts the context key; the id string
//! itself keys the live context tables.

use crate::error::EsError;
use crate::esql::async_id::java_b64_decode;

const INCLUDE_CONTEXT_UUID: &str = "include_context_uuid";
const QUERY_THEN_FETCH: &str = "queryThenFetch";
/// Transport version written at the head of PIT ids (8.19).
const PIT_TRANSPORT_VERSION: u32 = 8_841_097;

fn write_vint(out: &mut Vec<u8>, mut v: u32) {
    while v & !0x7f != 0 {
        out.push(((v & 0x7f) | 0x80) as u8);
        v >>= 7;
    }
    out.push(v as u8);
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_vint(out, s.chars().count() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn write_long(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn b64url(bytes: &[u8]) -> String {
    const URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.len();
        let b = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= n {
                out.push(URL[((b >> (18 - 6 * k)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// `EOFException` from a `ByteBufferStreamInput` read past the end.
fn underflow() -> EsError {
    EsError::new(400, "e_o_f_exception", "").with("reason", serde_json::Value::Null).caused_by(EsError::new(400, "buffer_underflow_exception", "").with("reason", serde_json::Value::Null))
}

struct In<'a> {
    b: &'a [u8],
    p: usize,
}

impl In<'_> {
    fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
    fn byte(&mut self) -> Result<u8, EsError> {
        let v = *self.b.get(self.p).ok_or_else(underflow)?;
        self.p += 1;
        Ok(v)
    }
    fn vint(&mut self) -> Result<u32, EsError> {
        let mut v: u32 = 0;
        for shift in [0, 7, 14, 21, 28] {
            let b = self.byte()?;
            v |= ((b & 0x7f) as u32) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Ok(v)
    }
    fn long(&mut self) -> Result<i64, EsError> {
        let mut a = [0u8; 8];
        for x in a.iter_mut() {
            *x = self.byte()?;
        }
        Ok(i64::from_be_bytes(a))
    }
    fn array_size(&mut self) -> Result<usize, EsError> {
        let n = self.vint()? as i32;
        if n < 0 {
            return Err(EsError::new(400, "negative_array_size_exception", format!("array size must be positive but was: {n}")));
        }
        let n = n as usize;
        if n > self.remaining() {
            return Err(EsError::new(400, "e_o_f_exception", format!("tried to read: {n} bytes but only {} remaining", self.remaining())));
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String, EsError> {
        let n = self.array_size()?;
        let mut s = String::new();
        for _ in 0..n {
            let b0 = self.byte()?;
            let (len, init) = match b0 >> 4 {
                0..=7 => (0, b0 as u32),
                12 | 13 => (1, (b0 & 0x1f) as u32),
                14 => (2, (b0 & 0x0f) as u32),
                _ => (0, 0xfffd),
            };
            let mut c = init;
            for _ in 0..len {
                c = (c << 6) | (self.byte()? & 0x3f) as u32;
            }
            s.push(char::from_u32(c).unwrap_or('\u{fffd}'));
        }
        Ok(s)
    }
    fn optional_string(&mut self) -> Result<Option<String>, EsError> {
        if self.byte()? == 0 {
            Ok(None)
        } else {
            self.string().map(Some)
        }
    }
}

/// A scroll id for one search context on this node.
pub fn new_scroll_id(session: &str, context_id: i64, node_id: &str) -> String {
    let mut out = Vec::new();
    write_string(&mut out, INCLUDE_CONTEXT_UUID);
    write_string(&mut out, QUERY_THEN_FETCH);
    write_vint(&mut out, 1);
    write_string(&mut out, session);
    write_long(&mut out, context_id);
    write_string(&mut out, node_id);
    b64url(&out)
}

/// `TransportSearchHelper.parseScrollId`: `illegal_argument_exception
/// "Cannot parse scroll id"` over the decoding failure.
pub fn parse_scroll_id(id: &str) -> Result<(), EsError> {
    let wrap = |e: EsError| EsError::illegal_argument("Cannot parse scroll id").caused_by(e);
    let bytes = java_b64_decode(id, true).map_err(wrap)?;
    let mut r = In { b: &bytes, p: 0 };
    let inner = |r: &mut In| -> Result<(), EsError> {
        let first = r.string()?;
        let include_uuid = first == INCLUDE_CONTEXT_UUID;
        if include_uuid {
            r.string()?;
        }
        let n = r.array_size()?;
        for _ in 0..n {
            if include_uuid {
                r.string()?;
            }
            r.long()?;
            r.string()?;
        }
        if r.remaining() > 0 {
            return Err(EsError::illegal_argument("Not all bytes were read"));
        }
        Ok(())
    };
    inner(&mut r).map_err(wrap)
}

/// A point-in-time id over `shards` `(index, uuid, shard, session, context_id, node)`.
pub fn new_pit_id(shards: &[(String, String, u32, String, i64, String)]) -> String {
    let mut out = Vec::new();
    write_vint(&mut out, PIT_TRANSPORT_VERSION);
    write_vint(&mut out, shards.len() as u32);
    for (index, uuid, shard, session, ctx, node) in shards {
        write_string(&mut out, index);
        write_string(&mut out, uuid);
        write_vint(&mut out, *shard);
        out.push(0); // no cluster alias
        write_string(&mut out, node);
        write_string(&mut out, session);
        write_long(&mut out, *ctx);
    }
    write_vint(&mut out, 0); // alias filters
    b64url(&out)
}

/// `SearchContextId.decode`: base64 failures propagate as the Java
/// `IllegalArgumentException`; stream failures become
/// `illegal_argument_exception "<cause.toString()>"` over the cause.
pub fn parse_pit_id(id: &str) -> Result<(), EsError> {
    let bytes = java_b64_decode(id, true)?;
    let mut r = In { b: &bytes, p: 0 };
    let inner = |r: &mut In| -> Result<(), EsError> {
        r.vint()?; // transport version
        let n = r.array_size()?;
        for _ in 0..n {
            r.string()?;
            r.string()?;
            r.vint()?;
            r.optional_string()?;
            r.string()?;
            r.string()?;
            r.long()?;
        }
        let m = r.array_size()?;
        for _ in 0..m {
            r.string()?;
            // AliasFilter: query (optional writeable) + aliases
            if r.byte()? != 0 {
                return Err(EsError::illegal_argument("unsupported point in time id"));
            }
            let k = r.array_size()?;
            for _ in 0..k {
                r.string()?;
            }
        }
        if r.remaining() > 0 {
            return Err(EsError::illegal_argument("Not all bytes were read"));
        }
        Ok(())
    };
    inner(&mut r).map_err(|e| {
        if e.kind == "illegal_argument_exception" {
            return e;
        }
        let java = match e.kind.as_str() {
            "e_o_f_exception" => "java.io.EOFException",
            "negative_array_size_exception" => "java.lang.NegativeArraySizeException",
            _ => "java.io.IOException",
        };
        let msg = if e.reason.is_empty() { java.to_string() } else { format!("{java}: {}", e.reason) };
        EsError::illegal_argument(msg).caused_by(e)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_errors() {
        let id = new_scroll_id("abc", 7, "node1");
        assert!(id.starts_with("FGluY2x1ZGVfY29udGV4dF91dWlk"));
        parse_scroll_id(&id).unwrap();
        let e = parse_scroll_id("foo").unwrap_err();
        assert_eq!(e.reason, "Cannot parse scroll id");
        assert_eq!(e.caused_by.as_ref().unwrap().reason, "tried to read: 126 bytes but only 1 remaining");
        let e = parse_scroll_id("AA").unwrap_err();
        assert_eq!(e.caused_by.as_ref().unwrap().kind, "e_o_f_exception");
        let e = parse_scroll_id("f.o").unwrap_err();
        assert_eq!(e.caused_by.as_ref().unwrap().reason, "Illegal base64 character 2e");

        let pid = new_pit_id(&[("idx".into(), "uuid".into(), 0, "sess".into(), 3, "node".into())]);
        parse_pit_id(&pid).unwrap();
        assert_eq!(parse_pit_id("foo").unwrap_err().reason, "java.io.EOFException");
        assert_eq!(parse_pit_id("Zm9vYmFy").unwrap_err().reason, "java.io.EOFException: tried to read: 111 bytes but only 4 remaining");
        assert_eq!(parse_pit_id("f.o").unwrap_err().reason, "Illegal base64 character 2e");
    }
}
