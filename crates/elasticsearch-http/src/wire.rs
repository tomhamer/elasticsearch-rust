//! Alternative response encodings for JSON bodies: `?format=yaml|cbor|smile`
//! (or an `Accept` header of the same media type), byte-compatible with the
//! Jackson generators Elasticsearch uses (indefinite-length CBOR containers,
//! Smile with shared property names, YAML with always-quoted strings).

use serde_json::Value;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wire {
    Yaml,
    Cbor,
    Smile,
}

impl Wire {
    pub fn content_type(self) -> &'static str {
        match self {
            Wire::Yaml => "application/yaml",
            Wire::Cbor => "application/cbor",
            Wire::Smile => "application/smile",
        }
    }
}

pub fn encode(w: Wire, v: &Value) -> Vec<u8> {
    match w {
        Wire::Yaml => yaml(v).into_bytes(),
        Wire::Cbor => {
            let mut o = Vec::new();
            cbor(v, &mut o);
            o
        }
        // One Smile encoder (the streaming one): the value-token ranges,
        // BigInteger and name dictionary of the Smile format.
        Wire::Smile => {
            let mut o = Vec::new();
            admitted::encode(Wire::Smile, v, &mut o).expect("encoding into a Vec");
            o
        }
    }
}

#[path = "wire_admitted.rs"]
mod admitted;

/// Count using the same allocation-free walk, then encode into a fixed admitted
/// destination. The owner follows Bytes clones and extracted response bodies.
pub fn encode_admitted(w: Wire, v: &Value) -> elasticsearch_engine::error::Result<bytes::Bytes> {
    encode_output(|out| admitted::encode(w, v, out))
}
/// One encoding pass into a buffer whose request-breaker reservation grows
/// ahead of each allocation (its capacity), instead of encoding the whole
/// response twice (once to count it): ES|QL pages of 10k wide rows are
/// ~100 MB of JSON. A breaker refusal mid-way is the same 429.
pub fn json_admitted<T: serde::Serialize + ?Sized>(v: &T, pretty: bool) -> elasticsearch_engine::error::Result<bytes::Bytes> {
    use std::io::Write;
    write_admitted(|out| if pretty {
        serde_json::to_writer_pretty(&mut *out, v).map_err(std::io::Error::other).and_then(|_| out.write_all(b"\n"))
    } else {
        serde_json::to_writer(&mut *out, v).map_err(std::io::Error::other)
    })
}

/// [`json_admitted`] of bytes `write` produces (a hand-written encoder).
pub fn write_admitted(write: impl FnOnce(&mut dyn std::io::Write) -> std::io::Result<()>) -> elasticsearch_engine::error::Result<bytes::Bytes> {
    use elasticsearch_engine::{error::EsError, memory};
    use std::io::Write;
    struct Output { data: Vec<u8>, _memory: memory::Reservation }
    impl AsRef<[u8]> for Output { fn as_ref(&self) -> &[u8] { &self.data } }
    struct Admitted { data: Vec<u8>, memory: memory::Reservation, admitted: usize, error: Option<EsError> }
    impl Admitted {
        fn room(&mut self, more: usize) -> std::io::Result<()> {
            let need = self.data.len().checked_add(more).ok_or_else(|| std::io::Error::other("encoded response overflow"))?;
            if need <= self.data.capacity() { return Ok(()); }
            let target = need.max(self.data.capacity().saturating_mul(2)).max(8192);
            let refuse = |this: &mut Admitted, e: EsError| { this.error = Some(e); std::io::Error::other("encoded response refused") };
            if target > self.admitted {
                if let Err(e) = self.memory.grow((target - self.admitted) as u64, "encoded response") { return Err(refuse(self, e)); }
                self.admitted = target;
            }
            if let Err(e) = self.data.try_reserve_exact(target - self.data.len()) {
                return Err(refuse(self, EsError::new(429, "circuit_breaking_exception", e.to_string())));
            }
            // An allocator may round the capacity up: admit what it holds.
            let capacity = self.data.capacity();
            if capacity > self.admitted {
                if let Err(e) = self.memory.grow((capacity - self.admitted) as u64, "encoded response") { return Err(refuse(self, e)); }
                self.admitted = capacity;
            }
            Ok(())
        }
    }
    impl Write for Admitted {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.room(b.len())?;
            self.data.extend_from_slice(b);
            Ok(b.len())
        }
        fn write_all(&mut self, b: &[u8]) -> std::io::Result<()> { self.write(b).map(|_| ()) }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let overhead = (std::mem::size_of::<Output>() + 4 * std::mem::size_of::<usize>()) as u64;
    let mut out = Admitted { data: Vec::new(), memory: memory::breakers().reserve_request(overhead, "encoded response")?, admitted: 0, error: None };
    let written = write(&mut out);
    if let Err(e) = written {
        return Err(out.error.take().unwrap_or_else(|| EsError::internal(e.to_string())));
    }
    let Admitted { data, memory, .. } = out;
    Ok(bytes::Bytes::from_owner(Output { data, _memory: memory.into_result_owner() }))
}
fn encode_output(encode: impl Fn(&mut dyn std::io::Write)->std::io::Result<()>) -> elasticsearch_engine::error::Result<bytes::Bytes> {
    use elasticsearch_engine::{error::EsError,memory};
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self,b:&[u8])->std::io::Result<usize>{self.0=self.0.checked_add(b.len()).ok_or_else(||std::io::Error::other("encoded response overflow"))?;Ok(b.len())}
        fn flush(&mut self)->std::io::Result<()>{Ok(())}
    }
    struct Output { data: Vec<u8>, _memory: memory::Reservation }
    impl AsRef<[u8]> for Output {fn as_ref(&self)->&[u8]{&self.data}}
    let mut count=Count(0);
    encode(&mut count).map_err(|e|EsError::internal(e.to_string()))?;
    let size=count.0.checked_add(std::mem::size_of::<Output>()+4*std::mem::size_of::<usize>()).ok_or_else(||EsError::new(429,"circuit_breaking_exception","encoded response overflow"))?;
    let owner=memory::breakers().reserve_request(size as u64,"encoded response")?;
    let mut data=Vec::new();
    data.try_reserve_exact(count.0).map_err(|e|EsError::new(429,"circuit_breaking_exception",e.to_string()))?;
    data.resize(count.0,0);
    let mut destination=data.as_mut_slice();
    encode(&mut destination).map_err(|e|EsError::internal(e.to_string()))?;
    if !destination.is_empty(){return Err(EsError::internal("encoded response count changed"));}
    Ok(bytes::Bytes::from_owner(Output{data,_memory:owner.into_result_owner()}))
}

// ---------------------------------------------------------------------------
// YAML
// ---------------------------------------------------------------------------

fn yaml_str(s: &str) -> String {
    yaml_quoted(s, None)
}

/// SnakeYAML `Emitter.writeDoubleQuoted` (as Jackson's YAML generator drives
/// it): escapes, and with `fold = Some((column, indent))` the 80-column
/// line splitting of values (`\` at the break, continuation indented).
fn yaml_quoted(text: &str, fold: Option<(usize, usize)>) -> String {
    const BEST_WIDTH: usize = 80;
    let t: Vec<char> = text.chars().collect();
    let mut o = String::with_capacity(text.len() + 2);
    o.push('"');
    let mut column = fold.map_or(0, |(c, _)| c) + 1;
    let (mut start, mut end) = (0usize, 0usize);
    let write = |o: &mut String, column: &mut usize, data: &str| {
        *column += data.chars().count();
        o.push_str(data);
    };
    while end <= t.len() {
        let ch = t.get(end).copied();
        let special = match ch {
            None => true,
            Some(c) => matches!(c, '"' | '\\' | '\u{85}' | '\u{2028}' | '\u{2029}' | '\u{FEFF}') || !(' '..='~').contains(&c),
        };
        if special {
            if start < end {
                let chunk: String = t[start..end].iter().collect();
                write(&mut o, &mut column, &chunk);
                start = end;
            }
            if let Some(c) = ch {
                let data = match c {
                    '\0' => "\\0".to_string(),
                    '\u{7}' => "\\a".into(),
                    '\u{8}' => "\\b".into(),
                    '\t' => "\\t".into(),
                    '\n' => "\\n".into(),
                    '\u{b}' => "\\v".into(),
                    '\u{c}' => "\\f".into(),
                    '\r' => "\\r".into(),
                    '\u{1b}' => "\\e".into(),
                    '"' => "\\\"".into(),
                    '\\' => "\\\\".into(),
                    '\u{85}' => "\\N".into(),
                    '\u{a0}' => "\\_".into(),
                    '\u{2028}' => "\\L".into(),
                    '\u{2029}' => "\\P".into(),
                    c if printable(c) => c.to_string(),
                    c if (c as u32) <= 0xFF => format!("\\x{:02x}", c as u32),
                    c => format!("\\u{:04x}", c as u32),
                };
                write(&mut o, &mut column, &data);
                start = end + 1;
            }
        }
        if let Some((_, indent)) = fold {
            if 0 < end && end + 1 < t.len() && (ch == Some(' ') || start >= end) && column + end.saturating_sub(start) > BEST_WIDTH {
                let data = if start >= end { "\\".to_string() } else { t[start..end].iter().collect::<String>() + "\\" };
                if start < end {
                    start = end;
                }
                o.push_str(&data);
                o.push('\n');
                o.push_str(&" ".repeat(indent));
                column = indent;
                if t[start] == ' ' {
                    write(&mut o, &mut column, "\\");
                }
            }
        }
        end += 1;
    }
    o.push('"');
    o
}

/// SnakeYAML `StreamReader.isPrintable`.
fn printable(c: char) -> bool {
    let c = c as u32;
    matches!(c, 0x9 | 0xA | 0xD | 0x85) || (0x20..=0x7E).contains(&c) || (0xA0..=0xD7FF).contains(&c) || (0xE000..=0xFFFD).contains(&c) || (0x10000..=0x10FFFF).contains(&c)
}

fn yaml_key(k: &str) -> String {
    let reserved = matches!(k.to_ascii_lowercase().as_str(), "true" | "false" | "yes" | "no" | "on" | "off" | "y" | "n" | "null" | "~" | "");
    let first = k.chars().next().unwrap_or(' ');
    let special = "-?:,[]{}#&*!|>'\"%@`".contains(first) || first.is_ascii_digit() || first.is_whitespace() || k.contains(": ") || k.contains(" #") || k.ends_with(':') || k.ends_with(' ');
    if reserved || special { yaml_str(k) } else { k.to_string() }
}

fn yaml_scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => yaml_str(s),
        _ => unreachable!(),
    }
}

fn yaml_block(v: &Value, indent: usize, out: &mut String) {
    let pad = " ".repeat(indent);
    match v {
        Value::Object(m) => {
            for (k, val) in m {
                out.push_str(&pad);
                out.push_str(&yaml_key(k));
                out.push(':');
                yaml_child(val, indent, out, false);
            }
        }
        Value::Array(a) => {
            for val in a {
                out.push_str(&pad);
                out.push('-');
                yaml_child(val, indent, out, true);
            }
        }
        _ => {}
    }
}

/// Write the value after `key:` / `-`; `in_seq` children of a sequence item start on the same line.
fn yaml_child(val: &Value, indent: usize, out: &mut String, in_seq: bool) {
    match val {
        Value::Object(m) if m.is_empty() => out.push_str(" {}\n"),
        Value::Array(a) if a.is_empty() => out.push_str(" []\n"),
        Value::Object(m) => {
            if in_seq {
                // first entry inline after "- ", the rest aligned under it
                let mut first = true;
                for (k, v) in m {
                    if first {
                        out.push(' ');
                        first = false;
                    } else {
                        out.push_str(&" ".repeat(indent + 2));
                    }
                    out.push_str(&yaml_key(k));
                    out.push(':');
                    yaml_child(v, indent + 2, out, false);
                }
            } else {
                out.push('\n');
                yaml_block(val, indent + 2, out);
            }
        }
        Value::Array(a) => {
            if in_seq {
                let mut first = true;
                for v in a {
                    if first {
                        out.push(' ');
                        first = false;
                    } else {
                        out.push_str(&" ".repeat(indent + 2));
                    }
                    out.push('-');
                    yaml_child(v, indent + 2, out, true);
                }
            } else {
                // Jackson does not indent a sequence under a mapping key.
                out.push('\n');
                yaml_block(val, indent, out);
            }
        }
        Value::String(st) => {
            out.push(' ');
            let column = out.len() - out.rfind('\n').map_or(0, |i| i + 1);
            out.push_str(&yaml_quoted(st, Some((column, indent + 2))));
            out.push('\n');
        }
        scalar => {
            out.push(' ');
            out.push_str(&yaml_scalar(scalar));
            out.push('\n');
        }
    }
}

pub fn yaml(v: &Value) -> String {
    let mut out = String::from("---");
    match v {
        Value::Object(m) if m.is_empty() => out.push_str(" {}\n"),
        Value::Array(a) if a.is_empty() => out.push_str(" []\n"),
        Value::Object(_) | Value::Array(_) => {
            out.push('\n');
            yaml_block(v, 0, &mut out);
        }
        s => {
            out.push(' ');
            out.push_str(&yaml_scalar(s));
            out.push('\n');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CBOR
// ---------------------------------------------------------------------------

fn cbor_head(major: u8, n: u64, o: &mut Vec<u8>) {
    let m = major << 5;
    if n < 24 {
        o.push(m | n as u8);
    } else if n < 0x100 {
        o.extend([m | 24, n as u8]);
    } else if n < 0x1_0000 {
        o.push(m | 25);
        o.extend((n as u16).to_be_bytes());
    } else if n < 0x1_0000_0000 {
        o.push(m | 26);
        o.extend((n as u32).to_be_bytes());
    } else {
        o.push(m | 27);
        o.extend(n.to_be_bytes());
    }
}

fn cbor(v: &Value, o: &mut Vec<u8>) {
    match v {
        Value::Null => o.push(0xf6),
        Value::Bool(b) => o.push(if *b { 0xf5 } else { 0xf4 }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= 0 { cbor_head(0, i as u64, o) } else { cbor_head(1, (-1 - i) as u64, o) }
            } else if let Some(u) = n.as_u64() {
                cbor_head(0, u, o)
            } else {
                o.push(0xfb);
                o.extend(n.as_f64().unwrap_or(0.0).to_be_bytes());
            }
        }
        Value::String(s) => {
            cbor_head(3, s.len() as u64, o);
            o.extend(s.as_bytes());
        }
        Value::Array(a) => {
            o.push(0x9f);
            for x in a {
                cbor(x, o);
            }
            o.push(0xff);
        }
        Value::Object(m) => {
            o.push(0xbf);
            for (k, x) in m {
                cbor_head(3, k.len() as u64, o);
                o.extend(k.as_bytes());
                cbor(x, o);
            }
            o.push(0xff);
        }
    }
}

// ---------------------------------------------------------------------------
// Smile
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cat_row_cbor_and_smile() {
        let v = json!([{"epoch": "1791136264", "count": "0"}]);
        assert_eq!(encode(Wire::Cbor, &v), b"\x9f\xbfeepochj1791136264ecounta0\xff\xff");
        let s = encode(Wire::Smile, &v);
        assert_eq!(&s[..4], &[0x3a, 0x29, 0x0a, 0x05]);
        assert_eq!(&s[4..], b"\xf8\xfa\x84epochI1791136264\x84count@0\xfb\xf9");
    }

    /// One-pass JSON: the bytes `serde_json` writes (compact and pretty, small
    /// and many-times-the-first-buffer bodies), the reservation released with
    /// the body, and a breaker refusal mid-way is a 429 with nothing held.
    #[test]
    fn json_admitted_writes_serde_json_bytes_and_admits_its_buffer() {
        use elasticsearch_engine::memory::{scoped, Breakers, RequestScope};
        let big: Vec<serde_json::Value> = (0..20_000).map(|i| json!({"k": format!("v{i}"), "n": i, "x": null})).collect();
        for v in [json!({}), json!({"a": [1, "b\"c", null, 1.5]}), serde_json::Value::Array(big.clone())] {
            assert_eq!(&json_admitted(&v, false).unwrap()[..], serde_json::to_vec(&v).unwrap().as_slice());
            let mut pretty = serde_json::to_vec_pretty(&v).unwrap();
            pretty.push(b'\n');
            assert_eq!(&json_admitted(&v, true).unwrap()[..], pretty.as_slice());
        }
        let breakers = Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key| match key {
            "indices.breaker.request.limit" => Some("256kb".into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            _ => None,
        });
        scoped(Some(RequestScope::new("json body", breakers)), || {
            let small = json_admitted(&json!({"a": 1}), false).unwrap();
            drop(small);
            assert_eq!(breakers.request.reserved(), 0);
            let error = json_admitted(&serde_json::Value::Array(big), false).unwrap_err();
            assert_eq!(error.status, 429);
            assert_eq!(breakers.request.reserved(), 0);
        });
    }

    #[test]
    fn yaml_basic() {
        let v = json!({"a": "x", "n": {"k": 1}, "l": [1, {"p": 1, "q": 2}], "e": [], "9x": null});
        // Jackson quotes YAML 1.1 boolean-ish keys (y/n/on/no/true/null): ES emits `"n":`.
        assert_eq!(yaml(&v), "---\na: \"x\"\n\"n\":\n  k: 1\nl:\n- 1\n- p: 1\n  q: 2\ne: []\n\"9x\": null\n");
    }
}

#[cfg(test)]
mod smile_format_tests {
    use super::*;
    use serde_json::json;

    fn smile(v: &Value) -> Vec<u8> {
        encode(Wire::Smile, v)
    }

    fn smile_in(b: &[u8]) -> Vec<u8> {
        match crate::xcontent_in::decode(crate::xcontent_in::InFmt::Smile, b) {
            Ok(d) => d.json.to_vec(),
            Err(e) => panic!("{e:?}"),
        }
    }

    /// Review finding 317: non-ASCII string values use Smile's Unicode value
    /// tokens (`0x80` tiny, `0xa0` short, `0xe4` long), never the small-int
    /// or control ranges, at each size-class boundary; the bytes decode back.
    #[test]
    fn unicode_values_use_the_unicode_value_tokens() {
        let cases: Vec<(String, u8)> = vec![
            ("é".into(), 0x80),                       // 2 bytes
            (format!("{}a", "é".repeat(16)), 0x9f),   // 33 bytes
            ("é".repeat(17), 0xa0),                   // 34 bytes
            (format!("{}a", "é".repeat(32)), 0xbf),   // 65 bytes
            ("é".repeat(33), 0xe4),                   // 66 bytes: long
        ];
        for (s, token) in cases {
            let out = smile(&json!([s]));
            assert_eq!(out[5], token, "{} bytes", s.len());
            let decoded = smile_in(&out);
            assert_eq!(serde_json::from_slice::<Value>(&decoded).unwrap(), json!([s]));
        }
    }

    /// Review finding 318: integers above `i64::MAX` are Smile BigIntegers
    /// (`0x26`), exact: they decode to the same values, never signed ones.
    #[test]
    fn unsigned_longs_above_i64_are_big_integers() {
        for u in [9_223_372_036_854_775_808u64, 9_223_372_036_854_775_809, u64::MAX] {
            let out = smile(&json!([u]));
            assert_eq!(out[5], 0x26, "{u}");
            let decoded = smile_in(&out);
            assert_eq!(std::str::from_utf8(&decoded).unwrap(), format!("[{u}]"));
        }
        // A signed long keeps the zigzag long token.
        assert_eq!(smile(&json!([i64::MIN]))[5], 0x25);
    }

    /// Review finding 319: a literal field name longer than 64 bytes takes
    /// a dictionary slot like every other (both directions), so later
    /// shared references name the right fields.
    #[test]
    fn long_field_names_take_a_dictionary_slot() {
        let long = "l".repeat(70);
        let v = json!([{long.clone(): 1, "s": 2}, {long.clone(): 3, "s": 4}]);
        let out = smile(&v);
        // Second object: both names by shared reference, `s` at index 1.
        let second = out.iter().rposition(|b| *b == 0xfa).unwrap();
        assert_eq!(out[second + 1], 0x40, "the long name is entry 0");
        assert_eq!(out[second + 3], 0x41, "the short name is entry 1");
        let decoded = smile_in(&out);
        assert_eq!(serde_json::from_slice::<Value>(&decoded).unwrap(), v);
        // A conforming writer's stream: long name L, short s, then a shared
        // reference to L (index 0) and to s (index 1).
        let mut stream = vec![0x3a, 0x29, 0x0a, 0x01, 0xfa, 0x34];
        stream.extend(long.as_bytes());
        stream.extend([0xfc, 0xc2, 0x80, b's', 0xc4, 0xfb, 0xfa, 0x40, 0xc6, 0x41, 0xc8, 0xfb]);
        let mut stream_arr = vec![0x3a, 0x29, 0x0a, 0x01, 0xf8];
        stream_arr.extend(&stream[4..]);
        stream_arr.push(0xf9);
        let decoded = smile_in(&stream_arr);
        assert_eq!(serde_json::from_slice::<Value>(&decoded).unwrap(), json!([{long.clone(): 1, "s": 2}, {long: 3, "s": 4}]));
    }
}

#[cfg(test)]
mod admitted_tests {
    use super::*;
    #[test]
    fn streaming_encoders_preserve_existing_wire_bytes() {
        let mut names=serde_json::Map::new();
        for i in 0..1030 { names.insert(format!("key{i}"),serde_json::json!({"same":i})); }
        let cases=[serde_json::json!({"é": "a ".repeat(90), "control":"\u{0}\u{85}\u{2028}🦀", "array":[{},[],null,true,-17,1.25]}), Value::Object(names)];
        for value in &cases { for wire in [Wire::Yaml,Wire::Cbor,Wire::Smile] {
            let mut streamed=Vec::new(); admitted::encode(wire,value,&mut streamed).unwrap();
            assert_eq!(streamed,encode(wire,value),"{wire:?}");
        }}
    }
}
