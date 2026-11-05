//! Request bodies in YAML, CBOR and Smile (`application/yaml|cbor|smile` and
//! their `application/vnd.elasticsearch+…` forms), decoded the way ES 9.5.5's
//! Jackson parsers read them.
//!
//! A body is transcoded once, at the HTTP edge, into single-line JSON that
//! every handler then parses as usual (the JSON path is untouched). The
//! transcoder records where each JSON token came from ([`SourceMap`]), so
//! errors resolved against the JSON report positions in the original body.
//! A body that cannot be decoded is replaced by a [`ForeignError`] marker
//! carrying the format's own exception chain, which the JSON body parsers
//! surface in place of a JSON syntax error.
//!
//! * YAML follows Jackson's `YAMLParser` over SnakeYAML: YAML 1.1 scalars
//!   (`yes`/`on` booleans, `0x`/`0b`/leading-zero octal integers, `_`
//!   separators), aliases read as the alias name, tags only for the core
//!   types, strict duplicate detection, a second document is a trailing
//!   token. Syntax errors use SnakeYAML's `MarkedYAMLException` text.
//! * CBOR / Smile: Jackson's binary parsers (Smile requires its header and
//!   honours shared key / value back-references); duplicate detection is on;
//!   positions are byte offsets (ES prints them as line `-1`).
//! * Bulk-style bodies (`_bulk`, `_msearch`, ...) accept Smile documents
//!   separated by `0xFF` (Smile's stream separator); YAML and CBOR have none.

use elasticsearch_engine::jsonpos::{source_desc, ForeignError, SourceMap};
use std::collections::HashSet;
#[path = "xcontent_memory.rs"]
mod memory;
use memory::{Buffer, ByteBuffer, Text};
use elasticsearch_engine::{error::EsError, rustic_core::admission};
use bytes::Bytes;

#[derive(Debug)]
enum ParseError { Syntax(ForeignError, admission::Reservation), Memory(EsError) }
impl From<EsError> for ParseError { fn from(error: EsError) -> Self { Self::Memory(error) } }
impl From<elasticsearch_engine::rustic_core::RusticError> for ParseError { fn from(error: elasticsearch_engine::rustic_core::RusticError) -> Self { Self::Memory(error.into()) } }


/// Internal header naming the client's body type of a request the cluster
/// edge decoded to JSON.
pub const ORIGINAL_CONTENT_TYPE: &str = "x-es-original-content-type";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InFmt {
    Yaml,
    Cbor,
    Smile,
}

/// The non-JSON x-content format of a `Content-Type` header value.
pub fn in_format(ct: &str) -> Option<InFmt> {
    let mime = ct.split(';').next()?.trim();
    let sub = mime.get(..12).filter(|p| p.eq_ignore_ascii_case("application/")).map(|_| &mime[12..])?;
    let sub = if sub.get(..18).is_some_and(|prefix| prefix.eq_ignore_ascii_case("vnd.elasticsearch+")) { &sub[18..] } else { sub };
    if sub.eq_ignore_ascii_case("yaml") {
        Some(InFmt::Yaml)
    } else if sub.eq_ignore_ascii_case("cbor") {
        Some(InFmt::Cbor)
    } else if sub.eq_ignore_ascii_case("smile") {
        Some(InFmt::Smile)
    } else {
        None
    }
}

/// Handlers whose body is a stream of documents (`RestHandler.mediaTypesValid`
/// requiring `XContentType.supportsDelimitedBulkRequests`).
pub fn is_delimited_endpoint(path: &str) -> bool {
    let p = path.trim_end_matches('/');
    p.ends_with("/_bulk") || p.ends_with("/_msearch") || p.ends_with("/_msearch/template") || p.ends_with("/_monitoring/bulk") || (p.contains("/_ml/anomaly_detectors/") && p.ends_with("/_data"))
}

/// `XContentType.supportsDelimitedBulkRequests`: formats with a stream separator.
pub fn supports_delimited(f: InFmt) -> bool {
    f == InFmt::Smile
}

/// A decoded body: JSON text (or the [`ForeignError`] marker) and its source map.
pub struct Decoded {
    pub json: Bytes,
    pub map: Option<SourceMap>,
    failure: Option<DecodeFailure>,
}
struct DecodeFailure { error: ForeignError, _memory: admission::Reservation }

/// Decode one document.
pub fn decode(f: InFmt, body: &[u8]) -> Result<Decoded, EsError> {
    let r = match f {
        InFmt::Yaml => yaml::decode(body),
        InFmt::Cbor => cbor::decode(body),
        InFmt::Smile => smile::decode(body),
    };
    match r {
        Ok((json, map)) => Ok(Decoded { json, map: Some(map), failure: None }),
        Err(ParseError::Memory(error)) => Err(error),
        Err(ParseError::Syntax(error, memory)) => syntax_body(error, memory),
    }
}

/// Decode a bulk-style body: Smile documents each terminated by `0xFF`,
/// rewritten as newline-terminated JSON lines (an unterminated last document
/// stays unterminated, as the bulk parsers require).
pub fn decode_stream(f: InFmt, body: &[u8]) -> Result<Decoded, EsError> {
    if f != InFmt::Smile {
        return decode(f, body);
    }
    let mut out = ByteBuffer::new();
    let mut rest = body;
    while !rest.is_empty() {
        let (doc, terminated) = match rest.iter().position(|b| *b == 0xFF) {
            Some(i) => (&rest[..i], Some(i)),
            None => (rest, None),
        };
        if terminated.is_none() {
            // An unterminated last document is never parsed: the bulk
            // parsers reject the body for its missing separator.
            out.extend_from_slice(b"{}");
            break;
        }
        if !doc.is_empty() {
            match smile::decode(doc) {
                Ok((json, _)) => out.extend_from_slice(&json),
                Err(ParseError::Memory(error)) => return Err(error),
                Err(ParseError::Syntax(error, memory)) => return syntax_body(error, memory),
            }
        }
        match terminated {
            Some(i) => {
                out.push(b'\n');
                out.check()?;
                rest = &rest[i + 1..];
            }
            None => break,
        }
    }
    Ok(Decoded { json: out.finish()?, map: None, failure: None })
}

/// Positions in an error response that was produced for the transcoded JSON
/// (by another node) rewritten to the original body: `line`/`col` fields,
/// `[1:N]` reason prefixes and Jackson's `at [Source: ...]` suffixes.
pub fn remap_error_positions(v: &mut serde_json::Value, json_text: &[u8], map: &SourceMap, memory: &mut elasticsearch_engine::memory::Reservation) -> Result<(),EsError> {
    use serde_json::Value;
    // source_desc reads at most500 bytes; admit its bounded escape and formatting
    // scratch before producing the descriptor (same bound as input decoding).
    memory.grow(8192,"error source description")?;
    let json_desc=source_desc(json_text);
    fn produce(memory:&mut elasticsearch_engine::memory::Reservation, f:impl Fn(&mut dyn std::fmt::Write)->std::fmt::Result)->Result<String,EsError> {
        struct Count(usize);
        impl std::fmt::Write for Count {fn write_str(&mut self,s:&str)->std::fmt::Result{self.0=self.0.checked_add(s.len()).ok_or(std::fmt::Error)?;Ok(())}}
        let mut count=Count(0);f(&mut count).map_err(|_|memory::overflow())?;
        memory.grow((count.0 as u64).saturating_add(32),"rewritten error string")?;
        let mut result=String::new();result.try_reserve_exact(count.0).map_err(|_|memory::overflow())?;
        struct Fixed<'a>{out:&'a mut String,left:usize}
        impl std::fmt::Write for Fixed<'_>{fn write_str(&mut self,s:&str)->std::fmt::Result{if s.len()>self.left{return Err(std::fmt::Error);}self.out.push_str(s);self.left-=s.len();Ok(())}}
        let mut fixed=Fixed{out:&mut result,left:count.0};f(&mut fixed).map_err(|_|memory::overflow())?;
        if fixed.left!=0{return Err(EsError::internal("error rewrite count changed"));}Ok(result)
    }
    let marker=produce(memory,|out|write!(out,"Source: {json_desc}; line: 1, column: "))?;
    fn walk(v:&mut Value,map:&SourceMap,marker:&str,memory:&mut elasticsearch_engine::memory::Reservation)->Result<(),EsError> {
        if let Value::Object(object)=v {
            if object.get("line").and_then(Value::as_u64)==Some(1) {
                if let Some(c)=object.get("col").and_then(Value::as_u64) {
                    if map.binary {object.remove("line");object.remove("col");}
                    else {let(l,c)=map.map_pos(1,c);memory.grow(64,"mapped error position keys")?;object.insert("line".into(),Value::from(l));object.insert("col".into(),Value::from(c));}
                }
            }
            if let Some(Value::String(reason))=object.get("reason") {
                let first=produce(memory,|out|{
                    let mut rest=reason.as_str();
                    while let Some(i)=rest.find("[1:") {
                        out.write_str(&rest[..i])?;let tail=&rest[i+3..];let n=tail.bytes().take_while(u8::is_ascii_digit).count();
                        if n>0 && tail[n..].starts_with(']') {let(l,c)=map.map_pos(1,tail[..n].parse().unwrap_or(1));if map.binary{write!(out,"[-1:{c}]")?;}else{write!(out,"[{l}:{c}]")?;}rest=&tail[n+1..];}
                        else{out.write_str("[1:")?;rest=tail;}
                    }out.write_str(rest)
                })?;
                let rewritten=produce(memory,|out|{
                    let mut rest=first.as_str();
                    while let Some(i)=rest.find(marker){out.write_str(&rest[..i])?;let tail=&rest[i+marker.len()..];let n=tail.bytes().take_while(u8::is_ascii_digit).count();let(l,c)=map.map_pos(1,tail[..n].parse().unwrap_or(1));
                        if map.binary{write!(out,"Source: {}; byte offset: #{c}",map.source)?;}else{write!(out,"Source: {}; line: {l}, column: {c}",map.source)?;}
                        rest=&tail[n..];
                    }out.write_str(rest)
                })?;
                memory.grow(32,"mapped error reason key")?;object.insert("reason".into(),Value::String(rewritten));
            }
        }
        match v {Value::Object(m)=>for value in m.values_mut(){walk(value,map,marker,memory)?;},Value::Array(a)=>for value in a{walk(value,map,marker,memory)?;},_=>{}}
        Ok(())
    }
    walk(v,map,&marker,memory)
}

/// The error ES reports for an undecodable body of a `method` request on
/// `path`: document writes wrap it as `failed to parse`, search-style bodies
/// as their parsing failure, everything else as the parser's exception.
pub fn decode_failure(method: &str, path: &str, decoded: &Decoded) -> Result<Option<Bytes>, EsError> {
    let Some(failure) = &decoded.failure else { return Ok(None) };
    let fe = &failure.error;
    let mut segments = path.split('/').filter(|s| !s.is_empty());
    let _index = segments.next();
    let doc_write = matches!(method, "PUT" | "POST") && matches!(segments.next(), Some("_doc" | "_create"));
    let search = path.split('/').any(|s| matches!(s, "_search" | "_count" | "_async_search" | "_explain" | "_validate" | "_delete_by_query" | "_update_by_query"));
    // The decoded diagnostic already owns its strings. Build only the wrapper
    // reason, then serialize borrowed views directly; do not parse our internal
    // marker twice or construct a second heap-owned EsError/Value tree.
    let reason = if let Some(kind) = &fe.trailing {
        Some(Text::formatted(format_args!("Unexpected token [{kind}] found after the main object."))?)
    } else if doc_write {
        Some(Text::formatted(format_args!("[{}:{}] failed to parse: {}", fe.last_tok.0, fe.last_tok.1, fe.chain.first().map(|p| p.1.as_str()).unwrap_or("")))?)
    } else { None };
    let wrapper = if fe.trailing.is_some() { Some(("parsing_exception", reason.as_deref().unwrap(), if fe.binary { None } else { Some(fe.pos) })) }
        else if doc_write { Some(("document_parsing_exception", reason.as_deref().unwrap(), None)) }
        else if search && fe.chain.len() > 2 && !fe.binary { Some(("parsing_exception", fe.chain.first().map(|p| p.1.as_str()).unwrap_or(""), Some((fe.pos.0, fe.pos.1.saturating_sub(1))))) }
        else { None };
    let view = FailureView { chain: if fe.trailing.is_some() { &[] } else { &fe.chain }, wrapper };
    crate::wire::json_admitted(&FailureResponse(&view), false).map(Some)
}

struct FailureView<'a> { chain: &'a [(String, String)], wrapper: Option<(&'static str, &'a str, Option<(u64,u64)>)> }
impl FailureView<'_> {
    fn len(&self) -> usize { self.chain.len() + usize::from(self.wrapper.is_some()) }
    fn node(&self, index: usize) -> (&str, &str, Option<(u64,u64)>) {
        if index == 0 { if let Some(wrapper) = self.wrapper { return wrapper; } }
        self.chain.get(index - usize::from(self.wrapper.is_some())).map(|p| (p.0.as_str(), p.1.as_str(), None)).unwrap_or(("", "", None))
    }
    fn root(&self) -> usize {
        let mut i = 0;
        while i + 1 < self.len() {
            let (kind, _, _) = self.node(i); let (cause, _, _) = self.node(i + 1);
            let xc = |s: &str| matches!(s, "x_content_parse_exception" | "x_content_e_o_f_exception" | "named_object_not_found_exception");
            let descend = if xc(kind) { xc(cause) || EsError::is_es_exception(cause) }
                else { EsError::is_es_exception(kind) && EsError::is_es_exception(cause) };
            if !descend { break; } i += 1;
        }
        i
    }
}
struct FailureNode<'a,'b> { view: &'a FailureView<'b>, index: usize, causes: bool, roots: bool }
impl serde::Serialize for FailureNode<'_,'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok,S::Error> {
        use serde::ser::SerializeMap;
        let (kind, reason, position) = self.view.node(self.index);
        let cause = self.causes && self.index + 1 < self.view.len();
        let mut map = s.serialize_map(Some(2 + usize::from(cause) + usize::from(self.roots) + 2*usize::from(position.is_some())))?;
        if self.roots { map.serialize_entry("root_cause", &[FailureNode { view: self.view, index: self.view.root(), causes: false, roots: false }])?; }
        map.serialize_entry("type", kind)?; map.serialize_entry("reason", reason)?;
        if let Some((line,col)) = position { map.serialize_entry("line", &line)?; map.serialize_entry("col", &col)?; }
        if cause { map.serialize_entry("caused_by", &FailureNode { view: self.view, index: self.index+1, causes: true, roots: false })?; }
        map.end()
    }
}
struct FailureResponse<'a,'b>(&'a FailureView<'b>);
impl serde::Serialize for FailureResponse<'_,'_> {
    fn serialize<S: serde::Serializer>(&self, s:S)->Result<S::Ok,S::Error> {
        use serde::ser::SerializeMap;
        let mut map=s.serialize_map(Some(2))?;
        map.serialize_entry("error", &FailureNode { view:self.0,index:0,causes:true,roots:true })?;
        map.serialize_entry("status", &400u16)?;map.end()
    }
}

// ---------------------------------------------------------------------------
// JSON output with source positions

struct Frame {
    object: bool,
    entries: usize,
    keys: HashSet<String>,
    memory: Option<admission::Reservation>,
}
struct Out {
    b: ByteBuffer,
    marks: Buffer<(u64, u64, u64)>,
    stack: Buffer<Frame>,
    source_memory: admission::Reservation,
}
impl Out {
    fn new(_cap: usize) -> Result<Self, ParseError> {
        // source_desc displays at most 500 bytes: at most 3000 escaped bytes,
        // geometric string capacity4096, lossy UTF8 scratch1500, and suffixes.
        let source_memory = admission::shared().reserve(8192, "x-content source description")?;
        Ok(Self { b: ByteBuffer::new(), marks: Buffer::new(), stack: Buffer::new(), source_memory })
    }
    fn mark(&mut self, line: u64, col: u64) -> Result<(), ParseError> {
        self.b.check()?;
        self.marks.push((self.b.len() as u64, line, col))?;
        Ok(())
    }
    fn before_value(&mut self) -> Result<(), ParseError> {
        self.b.check()?;
        if let Some(frame) = self.stack.last_mut() { if !frame.object { if frame.entries > 0 { self.b.push(b','); } frame.entries += 1; } }
        self.b.check()?; Ok(())
    }
    fn key(&mut self, key: &str, at: (u64, u64)) -> Result<bool, ParseError> {
        let Some(frame) = self.stack.last_mut() else { return Ok(true) };
        if frame.keys.contains(key) { return Ok(false); }
        // Per-key table allowance covers growth overlap; copied text is
        // charged exactly. Closing a frame destroys keys before its owner.
        let bytes = (key.len() as u64).saturating_add(256);
        if let Some(memory) = &mut frame.memory { memory.grow(bytes, "x-content object keys")?; }
        else { frame.memory = Some(admission::shared().reserve(bytes, "x-content object keys")?); }
        frame.keys.try_reserve(1).map_err(|_| memory::overflow())?;
        let mut copied = String::new();
        copied.try_reserve_exact(key.len()).map_err(|_| memory::overflow())?;
        copied.push_str(key);
        frame.keys.insert(copied);
        if frame.entries > 0 { self.b.push(b','); } frame.entries += 1;
        self.mark(at.0, at.1)?;
        write_str(&mut self.b, key); self.b.push(b':'); self.b.check()?;
        Ok(true)
    }
    fn open(&mut self, object: bool) -> Result<(), ParseError> {
        self.stack.push(Frame { object, entries: 0, keys: HashSet::new(), memory: None })?;
        self.b.push(if object { b'{' } else { b'[' }); self.b.check()?; Ok(())
    }
    fn close(&mut self) -> Result<(), ParseError> {
        if let Some(frame) = self.stack.pop() { self.b.push(if frame.object { b'}' } else { b']' }); }
        self.b.check()?; Ok(())
    }
    fn in_object(&self) -> bool { self.stack.last().is_some_and(|frame| frame.object) }
    fn map(self, source: String, binary: bool) -> Result<(Bytes, SourceMap), ParseError> {
        let json = self.b.finish()?;
        let (marks, memory) = self.marks.into_parts();
        Ok((json, SourceMap::from_admitted(source, binary, marks, [memory, Some(self.source_memory), None])))
    }
}

fn trailing_error(kind: &str, pos: (u64, u64), binary: bool) -> ParseError {
    let memory = match admission::shared().reserve((kind.len() as u64).saturating_add(512), "x-content trailing diagnostic") { Ok(memory) => memory, Err(error) => return error.into() };
    ParseError::Syntax(ForeignError { chain: Vec::new(), pos, last_tok: pos, binary, trailing: Some(kind.into()) }, memory)
}

fn syntax_body(error: ForeignError, memory: admission::Reservation) -> Result<Decoded, EsError> {
    // Bundle before the first fallible encode: a second argument's standalone
    // guard would otherwise drop before the earlier error on an early return.
    let failure=DecodeFailure { error, _memory: memory };
    let mut buffer = ByteBuffer::new();
    buffer.extend_from_slice(elasticsearch_engine::jsonpos::FOREIGN_MAGIC);
    let _ = serde_json::to_writer(&mut buffer, &failure.error);
    Ok(Decoded { json: buffer.finish()?, map: None, failure: Some(failure) })
}

fn write_str(b: &mut ByteBuffer, s: &str) {
    let _ = serde_json::to_writer(&mut *b, s);
}

fn write_f64(b: &mut ByteBuffer, f: f64) {
    if f.is_finite() {
        let _ = serde_json::to_writer(&mut *b, &f);
    } else {
        // Jackson renders non-finite doubles as these strings in JSON.
        write_str(b, if f.is_nan() { "NaN" } else if f > 0.0 { "Infinity" } else { "-Infinity" });
    }
}

fn base64(data: &[u8]) -> Result<Text, ParseError> {
    struct Encoded<'a>(&'a [u8]);
    impl std::fmt::Display for Encoded<'_> {
        fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use std::fmt::Write;
            const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for c in self.0.chunks(3) {
                let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
                out.write_char(T[(n >> 18) as usize & 63] as char)?;
                out.write_char(T[(n >> 12) as usize & 63] as char)?;
                out.write_char(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' })?;
                out.write_char(if c.len() > 2 { T[n as usize & 63] as char } else { '=' })?;
            }
            Ok(())
        }
    }
    Ok(Text::formatted(format_args!("{}", Encoded(data)))?)
}

/// Two's-complement integer conversion with admitted byte/limb storage.
fn bigint_digits(bytes: &[u8]) -> Result<Text, ParseError> {
    let neg = bytes.first().is_some_and(|b| b & 0x80 != 0);
    let mut mag = Buffer::new(); mag.extend_from_slice(bytes)?;
    if neg {
        for b in mag.iter_mut() { *b = !*b; }
        for b in mag.iter_mut().rev() { let (v, c) = b.overflowing_add(1); *b = v; if !c { break; } }
    }
    let mut limbs = Buffer::new(); limbs.push(0u32)?;
    for &byte in mag.iter() {
        let mut carry = byte as u64;
        for limb in limbs.iter_mut() { let value = (*limb as u64) * 256 + carry; *limb = (value % 1_000_000_000) as u32; carry = value / 1_000_000_000; }
        while carry > 0 { limbs.push((carry % 1_000_000_000) as u32)?; carry /= 1_000_000_000; }
    }
    struct Decimal<'a> { negative: bool, limbs: &'a [u32] }
    impl std::fmt::Display for Decimal<'_> {
        fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.negative && self.limbs.iter().any(|l| *l != 0) { out.write_str("-")?; }
            write!(out, "{}", self.limbs.last().unwrap())?;
            for limb in self.limbs.iter().rev().skip(1) { write!(out, "{limb:09}")?; }
            Ok(())
        }
    }
    Ok(Text::formatted(format_args!("{}", Decimal { negative: neg, limbs: &limbs }))?)
}

fn binary_error(arguments: std::fmt::Arguments<'_>, len: usize, off: usize, prefixed: bool) -> ParseError {
    let msg = match Text::formatted(arguments) { Ok(msg) => msg, Err(error) => return error.into() };
    // full/top retain two message copies; include geometric String capacities
    // and the fixed parser exception descriptors before formatting either.
    let memory = match admission::shared().reserve((msg.len() as u64).saturating_mul(4).saturating_add(2048), "binary x-content syntax diagnostic") {
        Ok(memory) => memory, Err(error) => return error.into(),
    };
    let full = format!("{msg}\n at [Source: (byte[])[{len} bytes]; byte offset: #{off}]");
    let top = if prefixed { format!("[-1:{off}] {full}") } else { full.clone() };
    ParseError::Syntax(ForeignError { chain: vec![("x_content_parse_exception".into(), top), ("json_parse_exception".into(), full)], pos: (0, off as u64), last_tok: (0, off as u64), binary: true, trailing: None }, memory)
}

// ---------------------------------------------------------------------------
// YAML

mod yaml {
    use super::*;
    use std::ffi::CStr;
    use std::mem::MaybeUninit;
    use unsafe_libyaml as y;

    struct Ev {
        kind: Kind,
        start: (u64, u64, usize),
        end: (u64, u64, usize),
    }

    enum Kind {
        DocStart,
        Scalar { value: Text, tag: Option<Text>, plain: bool },
        Alias(Text),
        SeqStart,
        SeqEnd,
        MapStart,
        MapEnd,
        Other,
        StreamEnd,
    }

    struct LibErr {
        scanner: bool,
        problem: Text,
        problem_mark: (usize, usize, usize),
        context: Option<Text>,
        context_mark: (usize, usize, usize),
    }

    enum YamlError { Syntax(LibErr), Memory(EsError) }
    impl From<EsError> for YamlError { fn from(error: EsError) -> Self { Self::Memory(error) } }
    struct Event(y::yaml_event_t);
    impl Drop for Event { fn drop(&mut self) { unsafe { y::yaml_event_delete(&mut self.0); } } }

    /// Pull parser over libyaml's event stream.
    struct Parser {
        p: Box<MaybeUninit<y::yaml_parser_t>>,
        _input: Buffer<u8>,
        done: bool,
        allocator: y::admitted::ParserAllocator,
        _memory: admission::Reservation,
    }

    impl Parser {
        fn new(text: &[u8]) -> Result<Parser, ParseError> {
            let memory = admission::shared().reserve((std::mem::size_of::<y::yaml_parser_t>() as u64).saturating_add(64), "YAML parser descriptor")?;
            let mut input = Buffer::new(); input.extend_from_slice(text)?;
            let layout = std::alloc::Layout::new::<MaybeUninit<y::yaml_parser_t>>();
            // The descriptor is admitted above; use the fallible allocator path
            // so physical allocation failure follows the same typed denial path.
            let raw = unsafe { std::alloc::alloc(layout) }.cast::<MaybeUninit<y::yaml_parser_t>>();
            if raw.is_null() {
                return Err(elasticsearch_engine::rustic_core::RusticError::Memory("YAML parser descriptor allocation refused".into()).into());
            }
            // SAFETY: alloc returned a uniquely owned, correctly aligned block
            // of the exact Box layout. MaybeUninit needs no initialization.
            let mut p = unsafe { Box::from_raw(raw) };
            let mut allocator = y::admitted::ParserAllocator::new()?;
            // SAFETY: libyaml initialises the parser in place; the input buffer
            // is owned by `Parser` and outlives every parse call.
            unsafe {
                if allocator.initialize(p.as_mut_ptr()).fail {
                    // Initialization zeroes the parser before allocating. Delete
                    // initialized fields before the session reclaims local scratch.
                    allocator.delete(p.as_mut_ptr());
                    return Err(elasticsearch_engine::rustic_core::RusticError::Memory("YAML parser initialization allocation refused".into()).into());
                }
                y::yaml_parser_set_input_string(p.as_mut_ptr(), input.as_ptr(), input.len() as _);
            }
            Ok(Parser { p, _input: input, done: false, allocator, _memory: memory })
        }

        fn next(&mut self) -> Result<Ev, YamlError> {
            if self.done {
                return Ok(Ev { kind: Kind::StreamEnd, start: (0, 0, 0), end: (0, 0, 0) });
            }
            let mut ev = MaybeUninit::<y::yaml_event_t>::uninit();
            // SAFETY: the parser is initialised; a successful parse fills
            // `ev`, which is deleted after its data is copied out.
            unsafe {
                if self.allocator.parse(self.p.as_mut_ptr(), ev.as_mut_ptr()).fail {
                    self.done = true;
                    let pre: &y::yaml_parser_t = &*self.p.as_ptr();
                    if self.allocator.failed() || pre.error == y::YAML_MEMORY_ERROR {
                        return Err(YamlError::Memory(elasticsearch_engine::rustic_core::RusticError::Memory("YAML parser allocation refused".into()).into()));
                    }
                    let cs = |c: *const u8| -> Result<Option<Text>, EsError> { if c.is_null() { Ok(None) } else { Ok(Some(Text::utf8(CStr::from_ptr(c as *const _).to_bytes())?)) } };
                    let m = |m: y::yaml_mark_t| (m.index as usize, m.line as usize, m.column as usize);
                    return Err(YamlError::Syntax(LibErr {
                        scanner: pre.error == y::YAML_SCANNER_ERROR,
                        problem: match cs(pre.problem as *const u8)? { Some(text) => text, None => Text::utf8(b"")? },
                        problem_mark: m(pre.problem_mark),
                        context: cs(pre.context as *const u8)?,
                        context_mark: m(pre.context_mark),
                    }));
                }
                let mut event = Event(ev.assume_init());
                let e = &mut event.0;
                let s = |p: *mut u8, n: usize| Text::utf8(if p.is_null() { &[] } else { std::slice::from_raw_parts(p, n) });
                let cz = |p: *mut u8| -> Result<Option<Text>, EsError> {
                    if p.is_null() { Ok(None) } else { Ok(Some(Text::utf8(CStr::from_ptr(p as *const _).to_bytes())?)) }
                };
                let kind = match e.type_ {
                    y::YAML_DOCUMENT_START_EVENT => Kind::DocStart,
                    y::YAML_SCALAR_EVENT => {
                        let d = e.data.scalar;
                        Kind::Scalar { value: s(d.value, d.length as usize)?, tag: cz(d.tag)?, plain: d.style == y::YAML_PLAIN_SCALAR_STYLE }
                    }
                    y::YAML_ALIAS_EVENT => Kind::Alias(match cz(e.data.alias.anchor)? { Some(text) => text, None => Text::utf8(b"")? }),
                    y::YAML_SEQUENCE_START_EVENT => Kind::SeqStart,
                    y::YAML_SEQUENCE_END_EVENT => Kind::SeqEnd,
                    y::YAML_MAPPING_START_EVENT => Kind::MapStart,
                    y::YAML_MAPPING_END_EVENT => Kind::MapEnd,
                    y::YAML_STREAM_END_EVENT => {
                        self.done = true;
                        Kind::StreamEnd
                    }
                    _ => Kind::Other,
                };
                let out = Ev {
                    kind,
                    start: (e.start_mark.line as u64, e.start_mark.column as u64, e.start_mark.index as usize),
                    end: (e.end_mark.line as u64, e.end_mark.column as u64, e.end_mark.index as usize),
                };
                Ok(out)
            }
        }
    }

    impl Drop for Parser {
        fn drop(&mut self) {
            // SAFETY: initialised in `new`.
            unsafe { self.allocator.delete(self.p.as_mut_ptr()) }
        }
    }

    /// 1-based position of a 0-based mark.
    fn one(m: (u64, u64, usize)) -> (u64, u64) {
        (m.0 + 1, m.1 + 1)
    }

    pub(super) fn decode(text: &[u8]) -> Result<(Bytes, SourceMap), ParseError> {
        let mut out = Out::new(0)?;
        let src = source_desc(text);
        let at = |p: (u64, u64)| format!("\n at [Source: {src}; line: {}, column: {}]", p.0, p.1);
        let mut parser = Parser::new(text)?;
        let mut docs = 0;
        let mut root_done = false;
        // Map frames waiting for a key (true) or a value (false).
        let mut want_key: Buffer<bool> = Buffer::new();
        let mut last: Option<((u64, u64, usize), (u64, u64, usize))> = None;
        loop {
            let ev = match parser.next() {
                Ok(ev) => ev,
                Err(YamlError::Memory(error)) => return Err(error.into()),
                Err(YamlError::Syntax(error)) => return Err(syntax_error(text, &src, error, last)),
            };
            let node = matches!(ev.kind, Kind::Scalar { .. } | Kind::Alias(_) | Kind::SeqStart | Kind::MapStart);
            if node && root_done && out.stack.is_empty() {
                // A further document: Jackson's next root token.
                let kind = match ev.kind {
                    Kind::MapStart => "START_OBJECT",
                    Kind::SeqStart => "START_ARRAY",
                    _ => "VALUE_STRING",
                };
                let p = one(ev.start);
                return Err(trailing_error(kind, p, false));
            }
            let as_key = node && out.in_object() && want_key.last() == Some(&true);
            match &ev.kind {
                Kind::DocStart => docs += 1,
                Kind::StreamEnd => break,
                Kind::Other => {}
                _ if as_key => {
                    let name = match &ev.kind {
                        Kind::Scalar { value, .. } => value.clone(),
                        Kind::Alias(a) => a.clone(),
                        other => {
                            let what = if matches!(other, Kind::SeqStart) { "SequenceStartEvent" } else { "MappingStartEvent" };
                            let p = one(ev.end);
                            let diagnostic_memory = admission::shared().reserve((src.len() as u64).saturating_add(1024).saturating_mul(16), "YAML field-name diagnostic")?;
                            let msg = format!("Expected a field name (Scalar value in YAML), got this instead: <org.yaml.snakeyaml.events.{what}(anchor=null, tag=null, implicit=true)>{}", at(p));
                            return Err(ParseError::Syntax(ForeignError { chain: vec![("x_content_parse_exception".into(), format!("[{}:{}] {msg}", p.0, p.1)), ("json_parse_exception".into(), msg)], pos: p, last_tok: one(ev.start), binary: false, trailing: None }, diagnostic_memory));
                        }
                    };
                    let s = one(ev.start);
                    if !out.key(&name, s)? {
                        let p = one(ev.end);
                        let diagnostic_memory = admission::shared().reserve((name.len() as u64).saturating_add(src.len() as u64).saturating_add(1024).saturating_mul(16), "YAML duplicate diagnostic")?;
                        let msg = format!("Duplicate field '{name}'{}", at(p));
                        return Err(ParseError::Syntax(ForeignError { chain: vec![("x_content_parse_exception".into(), format!("[{}:{}] {msg}", p.0, p.1)), ("json_parse_exception".into(), msg)], pos: p, last_tok: s, binary: false, trailing: None }, diagnostic_memory));
                    }
                    *want_key.last_mut().unwrap() = false;
                }
                Kind::Scalar { value, tag, plain } => {
                    out.before_value()?;
                    let s = one(ev.start);
                    out.mark(s.0, s.1)?;
                    let _scalar_memory = admission::shared().reserve((value.len() as u64).saturating_mul(4).saturating_add(512), "YAML scalar normalization")?;
                    if let Err(bad) = scalar(&mut out.b, value, tag.as_deref(), *plain) {
                        let p = one(ev.end);
                        let diagnostic_memory = admission::shared().reserve((bad.len() as u64).saturating_add(src.len() as u64).saturating_add(1024).saturating_mul(16), "YAML numeric diagnostic")?;
                        let msg = format!("Malformed numeric value '{bad}'{}", at(p));
                        return Err(ParseError::Syntax(ForeignError {
                            chain: vec![("x_content_parse_exception".into(), format!("[{}:{}] {msg}", p.0, p.1)), ("json_parse_exception".into(), msg), ("number_format_exception".into(), "illegal syntax".into())],
                            pos: p,
                            last_tok: s,
                            binary: false,
                            trailing: None,
                        }, diagnostic_memory));
                    }
                    value_done(&mut want_key, &out);
                }
                Kind::Alias(a) => {
                    out.before_value()?;
                    let s = one(ev.start);
                    out.mark(s.0, s.1)?;
                    write_str(&mut out.b, a);
                    value_done(&mut want_key, &out);
                }
                Kind::SeqStart | Kind::MapStart => {
                    out.before_value()?;
                    let s = one(ev.start);
                    out.mark(s.0, s.1)?;
                    let obj = matches!(ev.kind, Kind::MapStart);
                    out.open(obj)?;
                    want_key.push(obj)?;
                }
                Kind::SeqEnd | Kind::MapEnd => {
                    let s = one(ev.start);
                    out.mark(s.0, s.1)?;
                    out.close()?;
                    want_key.pop();
                    value_done(&mut want_key, &out);
                }
            }
            if node || matches!(ev.kind, Kind::SeqEnd | Kind::MapEnd) {
                last = Some((ev.start, ev.end));
                if out.stack.is_empty() {
                    root_done = true;
                }
            }
        }
        let _ = docs;
        out.map(src, false)
    }

    /// After a value: the enclosing map waits for its next key.
    fn value_done(want_key: &mut [bool], out: &Out) {
        if out.in_object() {
            if let Some(w) = want_key.last_mut() {
                *w = true;
            }
        }
    }

    /// Jackson `YAMLParser._decodeScalar`. `Err(text)`: a float Java cannot parse.
    fn scalar(b: &mut ByteBuffer, v: &str, tag: Option<&str>, plain: bool) -> Result<(), String> {
        if let Some(t) = tag {
            let t = t.strip_prefix("tag:yaml.org,2002:").unwrap_or("");
            return match t {
                "int" => {
                    if !int(b, v) {
                        write_str(b, v);
                    }
                    Ok(())
                }
                "float" => float(b, v),
                "bool" => {
                    match bool_of(v) {
                        Some(x) => b.extend_from_slice(if x { b"true" } else { b"false" }),
                        None => write_str(b, v),
                    }
                    Ok(())
                }
                "null" => {
                    b.extend_from_slice(b"null");
                    Ok(())
                }
                _ => {
                    write_str(b, v);
                    Ok(())
                }
            };
        }
        if !plain {
            write_str(b, v);
            return Ok(());
        }
        if matches!(v, "" | "~" | "null" | "Null" | "NULL") {
            b.extend_from_slice(b"null");
            return Ok(());
        }
        if let Some(x) = bool_of(v) {
            b.extend_from_slice(if x { b"true" } else { b"false" });
            return Ok(());
        }
        let first = v.as_bytes()[0];
        if first.is_ascii_digit() || matches!(first, b'+' | b'-' | b'.') {
            if is_int_like(v) {
                if !int(b, v) {
                    write_str(b, v);
                }
                return Ok(());
            }
            if is_float(v) {
                return float(b, v);
            }
        }
        write_str(b, v);
        Ok(())
    }

    fn bool_of(v: &str) -> Option<bool> {
        match v {
            "yes" | "Yes" | "YES" | "true" | "True" | "TRUE" | "on" | "On" | "ON" => Some(true),
            "no" | "No" | "NO" | "false" | "False" | "FALSE" | "off" | "Off" | "OFF" => Some(false),
            _ => None,
        }
    }

    fn unsigned(v: &str) -> (bool, &str) {
        match v.as_bytes().first() {
            Some(b'-') => (true, &v[1..]),
            Some(b'+') => (false, &v[1..]),
            _ => (false, v),
        }
    }

    /// Decimal / octal digits, `0x` hex or `0b` binary (with `_`).
    fn is_int_like(v: &str) -> bool {
        let (_, u) = unsigned(v);
        if let Some(h) = u.strip_prefix("0x") {
            return !h.is_empty() && h.bytes().all(|c| c.is_ascii_hexdigit() || c == b'_');
        }
        if let Some(h) = u.strip_prefix("0b") {
            return !h.is_empty() && h.bytes().all(|c| matches!(c, b'0' | b'1' | b'_'));
        }
        !u.is_empty() && u.as_bytes()[0].is_ascii_digit() && u.bytes().all(|c| c.is_ascii_digit() || c == b'_')
    }

    /// Write the integer; `false` when the digits do not form one (Jackson
    /// then keeps the text, e.g. `08`).
    fn int(b: &mut ByteBuffer, v: &str) -> bool {
        let (neg, u) = unsigned(v);
        let clean: String = u.chars().filter(|c| *c != '_').collect();
        let (radix, digits) = if let Some(h) = clean.strip_prefix("0x") {
            (16, h)
        } else if let Some(h) = clean.strip_prefix("0b") {
            (2, h)
        } else if clean.len() > 1 && clean.starts_with('0') {
            (8, &clean[1..])
        } else {
            (10, clean.as_str())
        };
        if digits.is_empty() {
            return false;
        }
        match u128::from_str_radix(digits, radix) {
            Ok(n) => {
                if neg && n != 0 {
                    b.push(b'-');
                }
                b.extend_from_slice(n.to_string().as_bytes());
                true
            }
            Err(_) if radix == 10 && digits.bytes().all(|c| c.is_ascii_digit()) => {
                if neg {
                    b.push(b'-');
                }
                b.extend_from_slice(digits.trim_start_matches('0').as_bytes());
                true
            }
            Err(_) => false,
        }
    }

    /// SnakeYAML 1.x float: `[-+]?(\.[0-9]+|[0-9_]+(\.[0-9_]*)?)([eE][-+]?[0-9]+)?`,
    /// `[-+]?\.(inf|Inf|INF)`, `\.(nan|NaN|NAN)`.
    fn is_float(v: &str) -> bool {
        if matches!(v, ".nan" | ".NaN" | ".NAN") {
            return true;
        }
        let (_, u) = unsigned(v);
        if matches!(u, ".inf" | ".Inf" | ".INF") {
            return true;
        }
        let (mant, exp) = match u.find(['e', 'E']) {
            Some(i) => (&u[..i], Some(&u[i + 1..])),
            None => (u, None),
        };
        if let Some(e) = exp {
            let e = e.strip_prefix(['+', '-']).unwrap_or(e);
            if e.is_empty() || !e.bytes().all(|c| c.is_ascii_digit()) {
                return false;
            }
        }
        if let Some(f) = mant.strip_prefix('.') {
            return !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit());
        }
        let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
        !ip.is_empty() && ip.bytes().all(|c| c.is_ascii_digit() || c == b'_') && fp.bytes().all(|c| c.is_ascii_digit() || c == b'_')
    }

    fn float(b: &mut ByteBuffer, v: &str) -> Result<(), String> {
        let clean: String = v.chars().filter(|c| *c != '_').collect();
        if clean.contains("inf") || clean.contains("Inf") || clean.contains("INF") || clean.to_ascii_lowercase().contains("nan") {
            return Err(v.to_string());
        }
        match clean.parse::<f64>() {
            Ok(f) => {
                write_f64(b, f);
                Ok(())
            }
            Err(_) => {
                write_str(b, v);
                Ok(())
            }
        }
    }

    // ----------------------------------------------------- syntax errors

    /// SnakeYAML `Mark.toString` for a 0-based (index, line, column) mark.
    fn lossy_chars(bytes: &[u8]) -> impl Iterator<Item = char> + '_ {
        bytes.utf8_chunks().flat_map(|chunk| chunk.valid().chars().chain((!chunk.invalid().is_empty()).then_some('\u{fffd}')))
    }

    fn mark(text: &[u8], m: (usize, usize, usize)) -> String {
        let pointer = lossy_chars(&text[..m.0.min(text.len())]).count();
        let line_break = |c: char| matches!(c, '\0' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}');
        // SnakeYAML displays at most37 characters to either side, retaining
        //32 when a truncation marker is needed. Keep this context on stack;
        // an error near EOF must not allocate a char array for the whole body.
        let mut left = ['\0'; 37]; let mut left_len = 0;
        let mut right = ['\0'; 37]; let mut right_len = 0;
        for (position, character) in lossy_chars(text).enumerate() {
            if position < pointer {
                if line_break(character) { left_len = 0; }
                else {
                    if left_len == left.len() { left.copy_within(1.., 0); left_len -= 1; }
                    left[left_len] = character; left_len += 1;
                }
            } else {
                if line_break(character) { break; }
                right[right_len] = character; right_len += 1;
                if right_len == right.len() { break; }
            }
        }
        let (head, before) = if left_len ==37 { (" ... ", &left[5..]) } else { ("", &left[..left_len]) };
        let (tail, after) = if right_len ==37 { (" ... ", &right[..32]) } else { ("", &right[..right_len]) };
        let line: String = before.iter().chain(after).collect();
        format!(" in 'reader', line {}, column {}:\n    {head}{line}{tail}\n    {}^", m.1 + 1, m.2 + 1, " ".repeat(before.len() + head.len()))
    }

    /// SnakeYAML's name of the token at byte `at`.
    fn token_at(text: &[u8], at: usize) -> &'static str {
        let rest = &text[at.min(text.len())..];
        match rest.iter().find(|c| !matches!(c, b' ' | b'\t' | b'\r' | b'\n')) {
            None => "<stream end>",
            Some(b'{') => "{",
            Some(b'}') => "}",
            Some(b'[') => "[",
            Some(b']') => "]",
            Some(b',') => ",",
            Some(b'-') => "-",
            Some(b'?') => "?",
            Some(b':') => ":",
            _ => "<scalar>",
        }
    }

    fn syntax_error(text: &[u8], src: &str, e: LibErr, last: Option<((u64, u64, usize), (u64, u64, usize))>) -> ParseError {
        let memory = match admission::shared().reserve((src.len() as u64).saturating_add(e.problem.len() as u64).saturating_add(e.context.as_ref().map_or(0, |c| c.len() as u64)).saturating_add(2048).saturating_mul(16), "YAML syntax diagnostic") { Ok(memory) => memory, Err(error) => return error.into() };
        let ctx = e.context.as_deref().unwrap_or("");
        let tok = token_at(text, e.problem_mark.0);
        let mut context_mark = e.context.is_some();
        let problem = match (ctx, &*e.problem) {
            (_, "found character that cannot start any token") => {
                context_mark = false;
                let c = lossy_chars(&text[e.problem_mark.0.min(text.len())..]).next().unwrap_or(' ');
                let r = match c {
                    '\t' => "\\t(TAB)".to_string(),
                    c => c.to_string(),
                };
                format!("found character '{r}' that cannot start any token. (Do not use {r} for indentation)")
            }
            (_, "mapping values are not allowed in this context") => "mapping values are not allowed here".to_string(),
            (_, "did not find expected node content") => format!("expected the node content, but found '{tok}'"),
            ("while parsing a flow sequence", _) => format!("expected ',' or ']', but got {tok}"),
            ("while parsing a flow mapping", _) => format!("expected ',' or '}}', but got {tok}"),
            ("while parsing a block mapping", _) | ("while parsing a block collection", _) => format!("expected <block end>, but found '{tok}'"),
            (_, p) => p.to_string(),
        };
        let mut lines = Vec::new();
        if let Some(c) = &e.context {
            lines.push(c.to_string());
            if context_mark {
                lines.push(mark(text, e.context_mark));
            }
        }
        lines.push(problem);
        lines.push(mark(text, e.problem_mark));
        let snake = format!("{}\n", lines.join("\n"));
        let pm = (e.problem_mark.1 as u64 + 1, e.problem_mark.2 as u64 + 1);
        let pos = match last {
            _ if e.scanner && e.problem.starts_with("mapping values") => pm,
            Some((_, end)) => one(end),
            None => pm,
        };
        let last_tok = last.map(|(s, _)| one(s)).unwrap_or((1, 1));
        let full = format!("{snake}\n at [Source: {src}; line: {}, column: {}]", pos.0, pos.1);
        let leaf = if e.scanner { "scanner_exception" } else { "parser_exception" };
        ParseError::Syntax(ForeignError {
            chain: vec![("x_content_parse_exception".into(), format!("[{}:{}] {full}", pos.0, pos.1)), ("marked_y_a_m_l_exception".into(), full), (leaf.into(), snake)],
            pos,
            last_tok,
            binary: false,
            trailing: None,
        }, memory)
    }
}

// ---------------------------------------------------------------------------
// CBOR

mod cbor {
    use super::*;

    struct R<'a> {
        b: &'a [u8],
        p: usize,
    }

    type Res<T> = Result<T, ParseError>;

    impl<'a> R<'a> {
        fn err(&self, msg: std::fmt::Arguments<'_>) -> ParseError {
            binary_error(msg, self.b.len(), self.p, true)
        }

        fn byte(&mut self) -> Res<u8> {
            let c = *self.b.get(self.p).ok_or_else(|| self.err(format_args!("Unexpected end-of-input in VALUE_NUMBER_INT")))?;
            self.p += 1;
            Ok(c)
        }

        fn take(&mut self, n: usize) -> Res<&'a [u8]> {
            if self.b.len() - self.p < n {
                return Err(self.err(format_args!("Needed to read {n} bytes, reached end-of-input")));
            }
            let s = &self.b[self.p..self.p + n];
            self.p += n;
            Ok(s)
        }

        /// Argument of an initial byte (`None` = indefinite length).
        fn arg(&mut self, info: u8) -> Res<Option<u64>> {
            Ok(Some(match info {
                0..=23 => info as u64,
                24 => self.byte()? as u64,
                25 => u16::from_be_bytes(self.take(2)?.try_into().unwrap()) as u64,
                26 => u32::from_be_bytes(self.take(4)?.try_into().unwrap()) as u64,
                27 => u64::from_be_bytes(self.take(8)?.try_into().unwrap()),
                31 => return Ok(None),
                _ => return Err(self.err(format_args!("Invalid length for {}: 0x{info:02x}", "CBOR value"))),
            }))
        }

        fn text(&mut self, info: u8, major: u8) -> Res<Buffer<u8>> {
            match self.arg(info)? {
                Some(n) => { let mut out = Buffer::new(); out.extend_from_slice(self.take(n as usize)?)?; Ok(out) },
                None => {
                    let mut out = Buffer::new();
                    loop {
                        let c = self.byte()?;
                        if c == 0xFF {
                            return Ok(out);
                        }
                        if c >> 5 != major {
                            return Err(self.err(format_args!("Mismatched chunk in chunked content")));
                        }
                        let n = self.arg(c & 31)?.ok_or_else(|| self.err(format_args!("Illegal chunked-length indicator within chunked-length value")))?;
                        out.extend_from_slice(self.take(n as usize)?)?;
                    }
                }
            }
        }

        fn value(&mut self, out: &mut Out, depth: usize) -> Res<()> {
            if depth > 1000 {
                return Err(self.err(format_args!("Document nesting depth exceeds the maximum allowed")));
            }
            // Tags other than big integers are transparent.
            let at = self.p as u64;
            let (major, info) = loop {
                let c = self.byte()?;
                let (major, info) = (c >> 5, c & 31);
                if major == 6 {
                    let save = self.p;
                    let tag = self.arg(info)?.unwrap_or(0);
                    if tag != 2 && tag != 3 {
                        continue;
                    }
                    self.p = save;
                }
                break (major, info);
            };
            out.before_value()?;
            out.mark(0, at)?;
            match major {
                0 => {
                    let n = self.arg(info)?.ok_or_else(|| self.err(format_args!("Invalid length indicator")))?;
                    out.b.extend_from_slice(Text::formatted(format_args!("{n}"))?.as_bytes());
                }
                1 => {
                    let n = self.arg(info)?.ok_or_else(|| self.err(format_args!("Invalid length indicator")))?;
                    out.b.extend_from_slice(Text::formatted(format_args!("-{}", n as u128 + 1))?.as_bytes());
                }
                2 => {
                    let d = self.text(info, 2)?;
                    write_str(&mut out.b, &base64(&d)?);
                }
                3 => {
                    let d = self.text(info, 3)?;
                    write_str(&mut out.b, &Text::utf8(&d)?);
                }
                4 => {
                    out.open(false)?;
                    match self.arg(info)? {
                        Some(n) => {
                            for _ in 0..n {
                                self.value(out, depth + 1)?;
                            }
                        }
                        None => loop {
                            if self.b.get(self.p) == Some(&0xFF) {
                                self.p += 1;
                                break;
                            }
                            self.value(out, depth + 1)?;
                        },
                    }
                    out.mark(0, self.p as u64)?;
                    out.close()?;
                }
                5 => {
                    out.open(true)?;
                    let n = self.arg(info)?;
                    let mut i = 0u64;
                    loop {
                        match n {
                            Some(n) if i >= n => break,
                            None if self.b.get(self.p) == Some(&0xFF) => {
                                self.p += 1;
                                break;
                            }
                            _ => {}
                        }
                        i += 1;
                        let kat = self.p as u64;
                        let k = self.key()?;
                        if !out.key(&k, (0, kat))? {
                            return Err(self.err(format_args!("Duplicate field '{k}'")));
                        }
                        self.value(out, depth + 1)?;
                    }
                    out.mark(0, self.p as u64)?;
                    out.close()?;
                }
                6 => {
                    // Tags: big integers become numbers; others are transparent.
                    let tag = self.arg(info)?.unwrap_or(0);
                    if tag == 2 || tag == 3 {
                        let c2 = self.byte()?;
                        if c2 >> 5 != 2 {
                            return Err(self.err(format_args!("Unexpected token after BigInteger tag")));
                        }
                        let d = self.text(c2 & 31, 2)?;
                        let mut bytes = Buffer::new();
                        bytes.push(0u8)?;
                        bytes.extend_from_slice(&d)?;
                        // CBOR negative bignums encode -1-n, exactly the
                        // one's complement of a sign-extended positive n.
                        if tag == 3 { for byte in bytes.iter_mut() { *byte = !*byte; } }
                        let digits = bigint_digits(&bytes)?;
                        out.b.extend_from_slice(digits.as_bytes());
                    }
                }
                _ => match info {
                    20 => out.b.extend_from_slice(b"false"),
                    21 => out.b.extend_from_slice(b"true"),
                    22 | 23 => out.b.extend_from_slice(b"null"),
                    25 => {
                        let h = u16::from_be_bytes(self.take(2)?.try_into().unwrap());
                        write_f64(&mut out.b, half(h) as f64);
                    }
                    26 => {
                        let f = f32::from_be_bytes(self.take(4)?.try_into().unwrap());
                        write_f64(&mut out.b, f as f64);
                    }
                    27 => {
                        let f = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                        write_f64(&mut out.b, f);
                    }
                    31 => {
                        self.p -= 1;
                        let ctx = if out.stack.is_empty() { "Root" } else if out.in_object() { "Object" } else { "Array" };
                        return Err(self.err(format_args!("Unexpected Break (0xFF) token in {ctx} context")));
                    }
                    _ => out.b.extend_from_slice(b"null"),
                },
            }
            out.b.check()?;
            Ok(())
        }

        fn key(&mut self) -> Res<Text> {
            let c = self.byte()?;
            let (major, info) = (c >> 5, c & 31);
            Ok(match major {
                3 | 2 => Text::utf8(&self.text(info, major)?)?,
                0 => Text::formatted(format_args!("{}", self.arg(info)?.unwrap_or(0)))?,
                1 => Text::formatted(format_args!("-{}", self.arg(info)?.unwrap_or(0) as u128 + 1))?,
                _ => {
                    self.p -= 1;
                    return Err(self.err(format_args!("Unsupported major type ({major}) for CBOR Objects, not (yet?) supported, only Strings")));
                }
            })
        }
    }

    fn half(h: u16) -> f32 {
        let exp = (h >> 10) & 0x1F;
        let mant = (h & 0x3FF) as f32;
        let v = match exp {
            0 => mant * 2f32.powi(-24),
            31 => {
                if mant == 0.0 {
                    f32::INFINITY
                } else {
                    f32::NAN
                }
            }
            e => (1.0 + mant / 1024.0) * 2f32.powi(e as i32 - 15),
        };
        if h & 0x8000 != 0 {
            -v
        } else {
            v
        }
    }

    pub(super) fn decode(body: &[u8]) -> Result<(Bytes, SourceMap), ParseError> {
        let mut r = R { b: body, p: 0 };
        let mut out = Out::new(0)?;
        // Skip the self-describe tag (0xd9d9f7).
        if body.starts_with(&[0xd9, 0xd9, 0xf7]) {
            r.p = 3;
        }
        r.value(&mut out, 0)?;
        out.b.check()?;
        if r.p < body.len() {
            // Jackson reads a further root value as a trailing token.
            let at = r.p as u64;
            let kind = match body[r.p] >> 5 {
                4 => "START_ARRAY",
                5 => "START_OBJECT",
                3 => "VALUE_STRING",
                0 | 1 => "VALUE_NUMBER",
                _ => "VALUE_NULL",
            };
            return Err(trailing_error(kind, (0, at), true));
        }
        out.map(format!("(byte[])[{} bytes]", body.len()), true)
    }
}

// ---------------------------------------------------------------------------
// Smile

mod smile {
    use super::*;

    struct R<'a> {
        b: &'a [u8],
        p: usize,
        names: Buffer<Text>,
        values: Buffer<Text>,
        shared_names: bool,
        shared_values: bool,
    }

    type Res<T> = Result<T, ParseError>;

    impl<'a> R<'a> {
        fn err(&self, msg: std::fmt::Arguments<'_>) -> ParseError {
            binary_error(msg, self.b.len(), self.p, true)
        }

        fn byte(&mut self) -> Res<u8> {
            let c = *self.b.get(self.p).ok_or_else(|| self.err(format_args!("Unexpected end-of-input")))?;
            self.p += 1;
            Ok(c)
        }

        fn take(&mut self, n: usize) -> Res<&'a [u8]> {
            if self.b.len() - self.p < n {
                return Err(self.err(format_args!("Needed to read {n} bytes, reached end-of-input")));
            }
            let s = &self.b[self.p..self.p + n];
            self.p += n;
            Ok(s)
        }

        fn vint(&mut self) -> Res<u64> {
            let mut v = 0u64;
            loop {
                let c = self.byte()?;
                if c & 0x80 != 0 {
                    return Ok((v << 6) | (c & 0x3F) as u64);
                }
                v = (v << 7) | c as u64;
            }
        }

        fn zigzag(n: u64) -> i64 {
            ((n >> 1) as i64) ^ -((n & 1) as i64)
        }

        /// `n` raw bytes in 7-bit groups.
        fn seven_bit(&mut self, n: usize) -> Res<Buffer<u8>> {
            // `n` comes from the body: check it against the bytes left
            // before allocating (8 encoded bytes per 7 raw, plus a tail).
            let encoded = (n / 7).checked_mul(8).and_then(|full| full.checked_add(if n % 7 == 0 { 0 } else { n % 7 + 1 }));
            match encoded {
                Some(e) if e <= self.b.len() - self.p => {}
                _ => return Err(self.err(format_args!("Needed to read {n} bytes, reached end-of-input"))),
            }
            let mut out = Buffer::new();
            out.reserve(n)?;
            let mut left = n;
            while left >= 7 {
                let g = self.take(8)?;
                let mut acc = 0u64;
                for &x in g {
                    acc = (acc << 7) | (x & 0x7F) as u64;
                }
                out.extend_from_slice(&acc.to_be_bytes()[1..])?;
                left -= 7;
            }
            if left > 0 {
                let g = self.take(left + 1)?;
                let mut acc = 0u64;
                for &x in &g[..left] {
                    acc = (acc << 7) | (x & 0x7F) as u64;
                }
                // The last byte holds the remaining `left` bits.
                acc = (acc << left) | (g[left] & ((1 << left) - 1)) as u64;
                let bytes = acc.to_be_bytes();
                out.extend_from_slice(&bytes[8 - left..])?;
            }
            Ok(out)
        }

        fn until_end_marker(&mut self) -> Res<Text> {
            let start = self.p;
            let i = self.b[start..].iter().position(|c| *c == 0xFC).ok_or_else(|| self.err(format_args!("Unexpected end-of-input in VALUE_STRING")))?;
            self.p = start + i + 1;
            Ok(Text::utf8(&self.b[start..start + i])?)
        }

        fn short_text(&mut self, n: usize) -> Res<Text> {
            let s = Text::utf8(self.take(n)?)?;
            if self.shared_values && n <= 64 {
                if self.values.len() >= 1024 {
                    self.values.clear();
                }
                self.values.push(s.clone())?;
            }
            Ok(s)
        }

        fn value(&mut self, out: &mut Out, depth: usize) -> Res<()> {
            if depth > 1000 {
                return Err(self.err(format_args!("Document nesting depth exceeds the maximum allowed")));
            }
            let at = self.p as u64;
            let c = self.byte()?;
            out.before_value()?;
            out.mark(0, at)?;
            match c {
                0x01..=0x1F => {
                    let s = self.values.get(c as usize - 1).cloned().ok_or_else(|| self.err(format_args!("Invalid shared text value reference {}; only got {} names in buffer (invalid content)", c - 1, self.values.len())))?;
                    write_str(&mut out.b, &s);
                }
                0x20 => write_str(&mut out.b, ""),
                0x21 => out.b.extend_from_slice(b"null"),
                0x22 => out.b.extend_from_slice(b"false"),
                0x23 => out.b.extend_from_slice(b"true"),
                0x24 | 0x25 => {
                    let n = Self::zigzag(self.vint()?);
                    out.b.extend_from_slice(Text::formatted(format_args!("{n}"))?.as_bytes());
                }
                0x26 => {
                    let n = self.vint()? as usize;
                    let d = self.seven_bit(n)?;
                    out.b.extend_from_slice(bigint_digits(&d)?.as_bytes());
                }
                0x28 => {
                    let g = self.take(5)?;
                    let mut acc = 0u32;
                    for &x in g {
                        acc = (acc << 7) | (x & 0x7F) as u32;
                    }
                    write_f64(&mut out.b, f32::from_bits(acc) as f64);
                }
                0x29 => {
                    let g = self.take(10)?;
                    let mut acc = 0u64;
                    for &x in g {
                        acc = (acc << 7) | (x & 0x7F) as u64;
                    }
                    write_f64(&mut out.b, f64::from_bits(acc));
                }
                0x2A => {
                    // Jackson reads the scale as a 32-bit zigzag int.
                    let scale = Self::zigzag(self.vint()?);
                    if i32::try_from(scale).is_err() {
                        return Err(self.err(format_args!("Invalid BigDecimal scale {scale}")));
                    }
                    let n = self.vint()? as usize;
                    let d = self.seven_bit(n)?;
                    let digits = bigint_digits(&d)?;
                    let (neg, digits) = match digits.strip_prefix('-') {
                        Some(x) => (true, x),
                        None => (false, &*digits),
                    };
                    // Stream into the admitted output. Formatting a decimal
                    // must not create independently growing padded strings.
                    use std::io::Write as _;
                    if neg { out.b.push(b'-'); }
                    if scale <= 0 {
                        out.b.extend_from_slice(digits.as_bytes());
                        if scale < 0 { let _ = write!(&mut out.b, "e{}", -scale); }
                    } else if scale as usize > digits.len() + 64 {
                        let _ = write!(&mut out.b, "{digits}e-{scale}");
                    } else {
                        let scale = scale as usize;
                        if scale >= digits.len() {
                            out.b.extend_from_slice(b"0.");
                            for _ in 0..scale-digits.len() { out.b.push(b'0'); }
                            out.b.extend_from_slice(digits.as_bytes());
                        } else {
                            let (integer, fraction) = digits.split_at(digits.len()-scale);
                            out.b.extend_from_slice(integer.as_bytes());
                            out.b.push(b'.');
                            out.b.extend_from_slice(fraction.as_bytes());
                        }
                    }
                    out.b.check()?;
                }
                0x40..=0x5F => {
                    let s = self.short_text((c & 0x1F) as usize + 1)?;
                    write_str(&mut out.b, &s);
                }
                0x60..=0x7F => {
                    let s = self.short_text((c & 0x1F) as usize + 33)?;
                    write_str(&mut out.b, &s);
                }
                0x80..=0x9F => {
                    let s = self.short_text((c & 0x1F) as usize + 2)?;
                    write_str(&mut out.b, &s);
                }
                0xA0..=0xBF => {
                    let s = self.short_text((c & 0x1F) as usize + 34)?;
                    write_str(&mut out.b, &s);
                }
                0xC0..=0xDF => {
                    let n = Self::zigzag((c & 0x1F) as u64);
                    out.b.extend_from_slice(Text::formatted(format_args!("{n}"))?.as_bytes());
                }
                0xE0 | 0xE4 => {
                    let s = self.until_end_marker()?;
                    write_str(&mut out.b, &s);
                }
                0xE8 => {
                    let n = self.vint()? as usize;
                    let d = self.seven_bit(n)?;
                    write_str(&mut out.b, &base64(&d)?);
                }
                0xEC..=0xEF => {
                    let i = (((c & 0x03) as usize) << 8) | self.byte()? as usize;
                    let s = self.values.get(i).cloned().ok_or_else(|| self.err(format_args!("Invalid shared text value reference {i}; only got {} names in buffer (invalid content)", self.values.len())))?;
                    write_str(&mut out.b, &s);
                }
                0xF8 => {
                    out.open(false)?;
                    loop {
                        if self.b.get(self.p) == Some(&0xF9) {
                            out.mark(0, self.p as u64)?;
                            self.p += 1;
                            break;
                        }
                        self.value(out, depth + 1)?;
                    }
                    out.close()?;
                }
                0xFA => {
                    out.open(true)?;
                    loop {
                        if self.b.get(self.p) == Some(&0xFB) {
                            out.mark(0, self.p as u64)?;
                            self.p += 1;
                            break;
                        }
                        let kat = self.p as u64;
                        let k = self.key()?;
                        if !out.key(&k, (0, kat))? {
                            return Err(self.err(format_args!("Duplicate field '{k}'")));
                        }
                        self.value(out, depth + 1)?;
                    }
                    out.close()?;
                }
                0xFD => {
                    let n = self.vint()? as usize;
                    let d = self.take(n)?;
                    write_str(&mut out.b, &base64(&d)?);
                }
                _ => {
                    self.p -= 1;
                    return Err(self.err(format_args!("Invalid type marker byte 0x{c:x} for expected value token")));
                }
            }
            out.b.check()?;
            Ok(())
        }

        fn add_name(&mut self, s: &Text) -> Res<()> {
            if self.shared_names {
                if self.names.len() >= 1024 {
                    self.names.clear();
                }
                self.names.push(s.clone())?;
            }
            Ok(())
        }

        fn key(&mut self) -> Res<Text> {
            let c = self.byte()?;
            Ok(match c {
                0x20 => Text::utf8(b"")?,
                0x30..=0x33 => {
                    let i = (((c & 0x03) as usize) << 8) | self.byte()? as usize;
                    self.names.get(i).cloned().ok_or_else(|| self.err(format_args!("Invalid shared name reference {i}; only got {} names in buffer (invalid content)", self.names.len())))?
                }
                0x34 => self.until_end_marker()?,
                0x40..=0x7F => {
                    let i = (c & 0x3F) as usize;
                    self.names.get(i).cloned().ok_or_else(|| self.err(format_args!("Invalid shared name reference {i}; only got {} names in buffer (invalid content)", self.names.len())))?
                }
                0x80..=0xBF => {
                    let s = Text::utf8(self.take((c & 0x3F) as usize + 1)?)?;
                    self.add_name(&s)?;
                    s
                }
                0xC0..=0xF7 => {
                    let s = Text::utf8(self.take((c & 0x3F) as usize + 2)?)?;
                    self.add_name(&s)?;
                    s
                }
                _ => {
                    self.p -= 1;
                    return Err(self.err(format_args!("Invalid type marker byte 0x{c:x} for expected field name (or END_OBJECT marker)")));
                }
            })
        }
    }

    pub(super) fn decode(body: &[u8]) -> Result<(Bytes, SourceMap), ParseError> {
        if body.len() < 4 || &body[..3] != b":)\n" {
            let first = body.first().copied().unwrap_or(0);
            return Err(binary_error(format_args!("Input does not start with Smile format header (first byte = 0x{first:x}) and parser has REQUIRE_HEADER enabled: can not parse"), body.len(), 0, false));
        }
        let flags = body[3];
        let mut r = R { b: body, p: 4, names: Buffer::new(), values: Buffer::new(), shared_names: flags & 0x01 != 0, shared_values: flags & 0x02 != 0 };
        let mut out = Out::new(0)?;
        r.value(&mut out, 0)?;
        out.b.check()?;
        if r.p < body.len() && body[r.p] != 0xFF {
            let at = r.p as u64;
            let kind = match body[r.p] {
                0xFA => "START_OBJECT",
                0xF8 => "START_ARRAY",
                _ => "VALUE_STRING",
            };
            return Err(trailing_error(kind, (0, at), true));
        }
        out.map(format!("(byte[])[{} bytes]", body.len()), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(f: InFmt, b: &[u8]) -> String {
        String::from_utf8(decode(f, b).unwrap().json.to_vec()).unwrap()
    }

    #[test]
    fn yaml_scalars_follow_jackson() {
        let y = b"a: yes\nb: 010\nc: 1e3\nd: .5\ne: ~\nf: \"s\"\ng: [1, two]\nh: 0x1F\ni: 2001-01-01\nj: 1_000\nk: on\nl: 08\nm: 0o17\nn: 0b101\no: 1:20\np: !!str 5\nq: -0\nr: 3.\ns: *x\n";
        // `*x` is an undefined alias: libyaml rejects it.
        let ok = json(InFmt::Yaml, &y[..y.len() - 6]);
        assert_eq!(ok, r#"{"a":true,"b":8,"c":1000.0,"d":0.5,"e":null,"f":"s","g":[1,"two"],"h":31,"i":"2001-01-01","j":1000,"k":true,"l":"08","m":"0o17","n":5,"o":"1:20","p":"5","q":0,"r":3.0}"#);
        assert_eq!(json(InFmt::Yaml, b"a: &x 1\nb: *x\n"), r#"{"a":1,"b":"x"}"#);
    }

    #[test]
    fn yaml_errors() {
        let e = ForeignError::from_body(&decode(InFmt::Yaml, b"size: 1\nsize: 2\n").unwrap().json).unwrap();
        assert_eq!(e.chain[0].1, "[2:5] Duplicate field 'size'\n at [Source: (byte[])\"size: 1\nsize: 2\n\"; line: 2, column: 5]");
        let e = ForeignError::from_body(&decode(InFmt::Yaml, b"a: 1\n b: 2\n").unwrap().json).unwrap();
        assert!(e.chain[2].1.starts_with("mapping values are not allowed here\n in 'reader', line 2, column 3:\n     b: 2\n      ^\n"), "{:?}", e.chain);
        let e = ForeignError::from_body(&decode(InFmt::Yaml, b"---\nsize: 0\n---\nsize: 1\n").unwrap().json).unwrap();
        assert_eq!((e.trailing.as_deref(), e.pos), (Some("START_OBJECT"), (4, 1)));
    }

    #[test]
    fn cbor_and_smile_from_es() {
        // `{"query":{"match_all":{}},"size":0}` as ES 9.5.5 encodes it.
        let c = [0xbf, 0x65, b'q', b'u', b'e', b'r', b'y', 0xbf, 0x69, b'm', b'a', b't', b'c', b'h', b'_', b'a', b'l', b'l', 0xbf, 0xff, 0xff, 0x64, b's', b'i', b'z', b'e', 0x00, 0xff];
        assert_eq!(json(InFmt::Cbor, &c), r#"{"query":{"match_all":{}},"size":0}"#);
        let s = [0x3a, 0x29, 0x0a, 0x05, 0xfa, 0x84, b'q', b'u', b'e', b'r', b'y', 0xfa, 0x88, b'm', b'a', b't', b'c', b'h', b'_', b'a', b'l', b'l', 0xfa, 0xfb, 0xfb, 0x83, b's', b'i', b'z', b'e', 0xc0, 0xfb];
        assert_eq!(json(InFmt::Smile, &s), r#"{"query":{"match_all":{}},"size":0}"#);
        // Shared key back-reference, 64-bit int, double, unicode text, long ASCII.
        let d: Vec<u8> = [&b":)\n\x05\xfa\x81aa\xc2\x81bb\xfa\x40\xc4\x81ccBxyz\xfb\x82big%\x0e/S\x11P\xa8\x80f)\x00?z\x00\x00\x00\x00\x00\x00\x00\x82neg\xc5\x80u\x83\xc3\xa9\xe2\x82\xac\xfb"[..]].concat();
        assert_eq!(json(InFmt::Smile, &d), r#"{"aa":1,"bb":{"aa":2,"cc":"xyz"},"big":123456789012,"f":1.25,"neg":-3,"u":"é€"}"#);
        let e = ForeignError::from_body(&decode(InFmt::Cbor, &[0xa2, 0x64, b's', b'i', b'z', b'e', 0x00, 0x64, b's', b'i', b'z', b'e', 0x01]).unwrap().json).unwrap();
        assert!(e.chain[1].1.starts_with("Duplicate field 'size'"));
        let e = ForeignError::from_body(&decode(InFmt::Smile, b"xyz").unwrap().json).unwrap();
        assert_eq!(e.chain[0].1, "Input does not start with Smile format header (first byte = 0x78) and parser has REQUIRE_HEADER enabled: can not parse\n at [Source: (byte[])[3 bytes]; byte offset: #0]");
    }

    #[test]
    fn smile_lengths_from_the_body_are_bounded() {
        // Binary (0xE8) and BigInteger (0x26) whose 7-bit length vint
        // claims ~2^60 bytes: an error, not a huge allocation.
        let huge_len = [0x0F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0xBF];
        for tag in [0xE8u8, 0x26] {
            let mut b = b":)\n\x05\xfa\x80x".to_vec();
            b.push(tag);
            b.extend_from_slice(&huge_len);
            b.extend_from_slice(b"\xfb");
            let e = ForeignError::from_body(&decode(InFmt::Smile, &b).unwrap().json).unwrap();
            assert!(e.chain.iter().any(|(_, m)| m.contains("reached end-of-input")), "{:?}", e.chain);
        }
        // BigDecimal (0x2A) with a scale outside i32: rejected.
        let mut b = b":)\n\x05\xfa\x80x\x2a".to_vec();
        b.extend_from_slice(&huge_len);
        b.extend_from_slice(b"\x81\x80\x81\xfb");
        let e = ForeignError::from_body(&decode(InFmt::Smile, &b).unwrap().json).unwrap();
        assert!(e.chain.iter().any(|(_, m)| m.contains("Invalid BigDecimal scale")), "{:?}", e.chain);
        // A large in-range scale uses exponent form: 1 * 10^-2000000000.
        let b = b":)\n\x05\xfa\x80x\x2a\x1d\x66\x59\x20\x80\x81\x00\x01\xfb";
        assert_eq!(json(InFmt::Smile, b), r#"{"x":1e-2000000000}"#);
    }

    #[test]
    fn smile_stream() {
        let doc = b":)\n\x05\xfa\x80z\xc2\xfb";
        let mut b = doc.to_vec();
        b.push(0xFF);
        b.extend_from_slice(doc);
        b.push(0xFF);
        assert_eq!(String::from_utf8(decode_stream(InFmt::Smile, &b).unwrap().json.to_vec()).unwrap(), "{\"z\":1}\n{\"z\":1}\n");
        b.pop();
        // The unterminated last document is left for the bulk parser to reject.
        assert_eq!(String::from_utf8(decode_stream(InFmt::Smile, &b).unwrap().json.to_vec()).unwrap(), "{\"z\":1}\n{}");
    }
}

#[cfg(test)]
mod admitted_decode_tests {
    use super::*;
    #[test]
    fn expanded_smile_output_denial_is_typed_429() {
        let mut input = b":)\n\x02\xf8\x5f".to_vec();
        input.extend_from_slice(&[0; 32]);
        input.extend_from_slice(&[1; 128]);
        input.push(0xf9);
        let lease = admission::Lease::reserve(admission::shared(), 20_000, "x-content test lease").unwrap();
        let scope = lease.enter();
        let error = decode(InFmt::Smile, &input).err().expect("expanded output must exceed lease");
        assert_eq!(error.status, 429);
        drop(scope);
        lease.expire();
    }
    #[test]
    fn source_map_and_bytes_clones_retain_independent_owners() {
        let owner = admission::Owner::new();
        let scope = owner.enter();
        let decoded = decode(InFmt::Cbor, &[0xa1, 0x61, b'x', 0x01]).unwrap();
        let map = decoded.map.as_ref().unwrap().clone();
        let bytes = decoded.json.clone();
        drop(decoded);
        assert!(owner.bytes().current > 0);
        assert_eq!(bytes.as_ref(), b"{\"x\":1}");
        drop(bytes);
        assert!(owner.bytes().current > 0);
        assert_eq!(map.map_pos(1, 2), (0, 1));
        drop(map);
        assert_eq!(owner.bytes().current, 0);
        drop(scope);
    }
    #[test]
    fn cbor_negative_bignum_does_not_truncate_to_u128() {
        // Tag3, positive magnitude2^128, interpreted as -1-magnitude.
        let mut input = vec![0xc3, 0x51, 1];
        input.extend_from_slice(&[0; 16]);
        let decoded = decode(InFmt::Cbor, &input).unwrap();
        assert_eq!(decoded.json.as_ref(), b"-340282366920938463463374607431768211457");
    }
}

#[cfg(test)]
mod failure_view_tests {
    use super::*;
    #[test]
    fn borrowed_failure_response_preserves_handler_and_root_cause_shapes() {
        for (format, input) in [(InFmt::Yaml, b"a: 1\n b: 2\n".as_slice()), (InFmt::Cbor, b"\xff".as_slice()), (InFmt::Smile, b"xyz".as_slice()), (InFmt::Yaml, b"---\na: 1\n---\nb: 2\n".as_slice())] {
            for (method,path) in [("PUT","/index/_doc/1"),("POST","/index/_search"),("POST","/_other")] {
                let decoded=decode(format,input).unwrap();
                let foreign=ForeignError::from_body(&decoded.json).unwrap();
                let expected=if foreign.trailing.is_some(){foreign.to_es_error()}
                    else { let error=elasticsearch_engine::jsonpos::validate(&decoded.json).unwrap_err();
                        if path.contains("/_doc/"){error.doc_failure(&decoded.json)}
                        else if path.ends_with("/_search"){error.search_failure(&decoded.json)}
                        else{error.to_error(&decoded.json)} };
                let actual=decode_failure(method,path,&decoded).unwrap().unwrap();
                assert_eq!(serde_json::from_slice::<serde_json::Value>(&actual).unwrap(),expected.to_json());
            }
        }
    }
}
