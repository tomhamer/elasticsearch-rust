//! Async execution ids (ES|QL, EQL and SQL async searches) and SQL cursors,
//! byte-compatible with ES's `AsyncExecutionId` and `Cursors` encodings so
//! that malformed ids fail the same way (`invalid id: [..]` 400s with the
//! underlying Java decoder/stream exception as the cause).
//!
//! An async id is URL-safe base64 of `writeString(docId) +
//! writeString("nodeId:taskId")`; results are keyed by the doc id (ES looks
//! the stored response up by doc id only).

use crate::error::EsError;
use std::sync::atomic::{AtomicU64, Ordering};

/// Transport version written at the head of SQL cursors (8.19).
pub const SQL_CURSOR_VERSION: i64 = 8_841_097;

static TASK_SEQ: AtomicU64 = AtomicU64::new(1);

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

const STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64_encode(bytes: &[u8], url: bool, pad: bool) -> String {
    let alpha = if url { URL } else { STD };
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.len();
        let b = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= n {
                out.push(alpha[((b >> (18 - 6 * k)) & 63) as usize] as char);
            } else if pad {
                out.push('=');
            }
        }
    }
    out
}

fn iae(msg: impl Into<String>) -> EsError {
    EsError::illegal_argument(msg)
}

/// `java.util.Base64.Decoder.decode` (non-MIME), with its exact error messages.
pub fn java_b64_decode(s: &str, url: bool) -> Result<Vec<u8>, EsError> {
    let src = s.as_bytes();
    let sl = src.len();
    if sl == 0 {
        return Ok(vec![]);
    }
    if sl < 2 {
        return Err(iae("Input byte[] should at least have 2 bytes for base64 bytes"));
    }
    let lookup = |c: u8| -> i32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as i32,
            b'a'..=b'z' => (c - b'a') as i32 + 26,
            b'0'..=b'9' => (c - b'0') as i32 + 52,
            b'-' if url => 62,
            b'_' if url => 63,
            b'+' if !url => 62,
            b'/' if !url => 63,
            b'=' => -2,
            _ => -1,
        }
    };
    let mut out = Vec::new();
    let (mut bits, mut shiftto, mut sp) = (0u32, 18i32, 0usize);
    while sp < sl {
        let c = src[sp];
        sp += 1;
        let b = lookup(c);
        if b < 0 {
            if b == -2 {
                if (shiftto == 6 && (sp == sl || {
                    let n = src[sp];
                    sp += 1;
                    n != b'='
                })) || shiftto == 18
                {
                    return Err(iae("Input byte array has wrong 4-byte ending unit"));
                }
                break;
            }
            // Java prints the signed byte in hex.
            let signed = c as i8 as i32;
            let hex = if signed < 0 { format!("-{:x}", -signed) } else { format!("{signed:x}") };
            return Err(iae(format!("Illegal base64 character {hex}")));
        }
        bits |= (b as u32) << shiftto;
        shiftto -= 6;
        if shiftto < 0 {
            out.extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]);
            shiftto = 18;
            bits = 0;
        }
    }
    match shiftto {
        6 => out.push((bits >> 16) as u8),
        0 => out.extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8]),
        12 => return Err(iae("Last unit does not have enough valid bits")),
        _ => {}
    }
    if sp < sl {
        return Err(iae(format!("Input byte array has incorrect ending byte at {sp}")));
    }
    Ok(out)
}

/// `EOFException` raised by a `ByteBufferStreamInput` read past the end.
fn underflow() -> EsError {
    EsError::new(400, "e_o_f_exception", "").with("reason", serde_json::Value::Null).caused_by(EsError::new(400, "buffer_underflow_exception", "").with("reason", serde_json::Value::Null))
}

struct In<'a> {
    b: &'a [u8],
    p: usize,
}

impl In<'_> {
    fn byte(&mut self) -> Result<u8, EsError> {
        let v = *self.b.get(self.p).ok_or_else(underflow)?;
        self.p += 1;
        Ok(v)
    }
    fn vint(&mut self) -> Result<i32, EsError> {
        let mut v: u32 = 0;
        for shift in [0, 7, 14, 21, 28] {
            let b = self.byte()?;
            v |= ((b & 0x7f) as u32) << shift;
            if b & 0x80 == 0 {
                return Ok(v as i32);
            }
        }
        Ok(v as i32)
    }
    fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
    fn string(&mut self) -> Result<String, EsError> {
        let n = self.vint()?;
        if n < 0 {
            return Err(EsError::new(400, "negative_array_size_exception", format!("array size must be positive but was: {n}")));
        }
        let n = n as usize;
        if n > self.remaining() {
            return Err(EsError::new(400, "e_o_f_exception", format!("tried to read: {n} bytes but only {} remaining", self.remaining())));
        }
        // `n` chars of (modified) UTF-8.
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
}

/// A decoded async execution id.
pub struct AsyncId {
    pub doc_id: String,
    pub node_id: String,
    pub task: i64,
}

fn invalid(id: &str) -> EsError {
    iae(format!("invalid id: [{id}]"))
}

/// Java `Long.parseLong` acceptance.
fn java_parse_long(s: &str) -> Option<i64> {
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<i64>().ok()
}

/// Decode an async id (`AsyncExecutionId.decode`).
pub fn decode(id: &str) -> Result<AsyncId, EsError> {
    let bytes = java_b64_decode(id, true).map_err(|e| invalid(id).caused_by(e))?;
    let mut r = In { b: &bytes, p: 0 };
    let doc_id = r.string().map_err(|e| invalid(id).caused_by(e))?;
    let task = r.string().map_err(|e| invalid(id).caused_by(e))?;
    let (node_id, task_num) = if task.is_empty() || task == "unset" {
        (String::new(), -1)
    } else {
        let colon = task.rfind(':').filter(|c| *c > 0 && *c + 1 < task.len()).ok_or_else(|| iae(format!("malformed task id {task}")))?;
        let num = &task[colon + 1..];
        let n = java_parse_long(num).ok_or_else(|| iae(format!("malformed task id {task}")).caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{num}\""))))?;
        (task[..colon].to_string(), n)
    };
    if r.remaining() > 0 {
        return Err(invalid(id));
    }
    Ok(AsyncId { doc_id, node_id, task: task_num })
}

/// A fresh async id for this node: `(id, doc_id)`.
pub fn new_id(node_id: &str) -> (String, String) {
    // ES doc ids are 22-char URL-safe base64 UUIDs.
    let doc_id = crate::index::new_uuid();
    let mut out = Vec::new();
    write_string(&mut out, &doc_id);
    write_string(&mut out, &format!("{node_id}:{}", TASK_SEQ.fetch_add(1, Ordering::Relaxed)));
    (b64_encode(&out, true, true), doc_id)
}

/// A fresh SQL cursor (`Cursors.encodeToString`-shaped): standard base64 of
/// the transport version followed by an opaque key.
pub fn new_sql_cursor() -> String {
    let mut out = Vec::new();
    write_vint(&mut out, SQL_CURSOR_VERSION as u32);
    write_string(&mut out, "Z");
    write_string(&mut out, &crate::index::new_uuid());
    b64_encode(&out, false, true)
}

/// Validate a SQL cursor the way `Cursors.decodeFromString` does (before the
/// cursor is looked up).
pub fn check_sql_cursor(cursor: &str) -> Result<(), EsError> {
    let bytes = java_b64_decode(cursor, false)?;
    let mut r = In { b: &bytes, p: 0 };
    let v = r.vint().map_err(|_| EsError::new(500, "sql_illegal_argument_exception", "Unsupported cursor version [0], expected [8841097]"))?;
    if v as i64 != SQL_CURSOR_VERSION {
        return Err(EsError::new(500, "sql_illegal_argument_exception", format!("Unsupported cursor version [{v}], expected [{SQL_CURSOR_VERSION}]")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_reference_id() {
        let a = decode("FjBZSkxibm9rVE5hX29NNGNkREZmR1EaOE9lLW1COUVROWlDakZieC1LSDI0Zzo0NzE=").ok().unwrap();
        assert_eq!(a.doc_id, "0YJLbnokTNa_oM4cdDFfGQ");
        assert_eq!(a.node_id, "8Oe-mB9EQ9iCjFbx-KH24g");
        assert_eq!(a.task, 471);
    }

    #[test]
    fn malformed_ids() {
        let cause = |id: &str| decode(id).err().unwrap().caused_by.map(|c| c.reason).unwrap_or_default();
        assert_eq!(cause("foo"), "tried to read: 126 bytes but only 1 remaining");
        assert_eq!(cause("Zm9v"), "tried to read: 102 bytes but only 2 remaining");
        assert_eq!(cause("f.o"), "Illegal base64 character 2e");
        assert_eq!(cause("A"), "Input byte[] should at least have 2 bytes for base64 bytes");
        assert_eq!(cause("Zm9vY"), "Last unit does not have enough valid bits");
        assert_eq!(cause("Zm9v="), "Input byte array has wrong 4-byte ending unit");
        assert_eq!(decode("AA").err().unwrap().caused_by.unwrap().kind, "e_o_f_exception");
        assert_eq!(decode("A2FiYwN4eXo=").err().unwrap().reason, "malformed task id xyz");
    }

    #[test]
    fn roundtrip() {
        let (id, doc) = new_id("node1");
        let a = decode(&id).ok().unwrap();
        assert_eq!(a.doc_id, doc);
        assert_eq!(doc.len(), 22);
        assert!(check_sql_cursor(&new_sql_cursor()).is_ok());
        assert_eq!(check_sql_cursor("foo").err().unwrap().reason, "Unsupported cursor version [126], expected [8841097]");
    }
}
