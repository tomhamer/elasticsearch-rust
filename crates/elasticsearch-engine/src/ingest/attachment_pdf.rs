//! PDF text extraction for the `attachment` processor (what Tika's
//! `PDFParser` gives ES): the text shown on each page, in page order, a new
//! line per text line and a blank line between paragraphs, plus the
//! document information dictionary.
//!
//! Objects are found by scanning the file (`N G obj`), so damaged or
//! incrementally updated cross-reference tables do not matter; compressed
//! object streams are expanded. Text is decoded through each font's
//! `ToUnicode` map, else its simple encoding (`/Differences` glyph names
//! over WinAnsi). A document whose pages show text that none of this can
//! decode, an encrypted document, or one without any page fails: the
//! processor never reports such a document as one without text.

use super::attachment::{Budget, ParseError, Text};
use std::collections::{HashMap, HashSet};
use std::io::Read;

/// Decoded stream bytes kept at most (a decompression bomb fails).
const MAX_STREAM: u64 = 64 << 20;
/// Nesting of arrays / dictionaries the object parser follows.
const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
enum Obj {
    Null,
    Bool(bool),
    Num(f64),
    Str(Vec<u8>),
    Name(String),
    Arr(Vec<Obj>),
    Dict(Vec<(String, Obj)>),
    Ref(u32),
    /// A content stream operator (or an unknown keyword).
    Op(String),
}

impl Obj {
    fn get(&self, key: &str) -> Option<&Obj> {
        match self {
            Obj::Dict(d) => d.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Obj::Num(n) => Some(*n),
            _ => None,
        }
    }
    fn name(&self) -> Option<&str> {
        match self {
            Obj::Name(n) => Some(n),
            _ => None,
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'\x0c' | b'\0')
}

fn is_delim(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

struct Lexer<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Lexer<'a> {
    fn new(b: &'a [u8], i: usize) -> Self {
        Lexer { b, i }
    }

    fn skip_ws(&mut self) {
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if is_ws(c) {
                self.i += 1;
            } else if c == b'%' {
                while self.i < self.b.len() && !matches!(self.b[self.i], b'\n' | b'\r') {
                    self.i += 1;
                }
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    /// A run of regular characters (a number or a keyword).
    fn regular(&mut self) -> &'a [u8] {
        let start = self.i;
        while self.i < self.b.len() && !is_ws(self.b[self.i]) && !is_delim(self.b[self.i]) {
            self.i += 1;
        }
        &self.b[start..self.i]
    }

    fn literal_string(&mut self) -> Vec<u8> {
        // At the opening parenthesis.
        self.i += 1;
        let mut out = Vec::new();
        let mut depth = 1usize;
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'(' => {
                    depth += 1;
                    out.push(c);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(c);
                }
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else { break };
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'\r' => {
                            if self.b.get(self.i) == Some(&b'\n') {
                                self.i += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut v = (e - b'0') as u32;
                            for _ in 0..2 {
                                match self.b.get(self.i) {
                                    Some(&d @ b'0'..=b'7') => {
                                        v = v * 8 + (d - b'0') as u32;
                                        self.i += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        other => out.push(other),
                    }
                }
                _ => out.push(c),
            }
        }
        out
    }

    fn hex_string(&mut self) -> Vec<u8> {
        // At `<` (not `<<`).
        self.i += 1;
        let mut digits = Vec::new();
        while self.i < self.b.len() && self.b[self.i] != b'>' {
            let c = self.b[self.i];
            if c.is_ascii_hexdigit() {
                digits.push(c);
            }
            self.i += 1;
        }
        self.i += 1;
        if digits.len() % 2 == 1 {
            digits.push(b'0');
        }
        digits.chunks(2).map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap_or("00"), 16).unwrap_or(0)).collect()
    }

    fn name(&mut self) -> String {
        // At `/`.
        self.i += 1;
        let raw = self.regular();
        let mut out = Vec::with_capacity(raw.len());
        let mut k = 0;
        while k < raw.len() {
            if raw[k] == b'#' && k + 2 < raw.len() {
                if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&raw[k + 1..k + 3]).unwrap_or("zz"), 16) {
                    out.push(v);
                    k += 3;
                    continue;
                }
            }
            out.push(raw[k]);
            k += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// The next object (an operator keyword is [`Obj::Op`]); `None` at the
    /// end of input. `refs`: `N G R` is a reference (object bodies; not
    /// in content streams).
    fn object(&mut self, refs: bool, depth: usize) -> Option<Obj> {
        self.skip_ws();
        let c = self.peek()?;
        if depth > MAX_DEPTH {
            self.i = self.b.len();
            return None;
        }
        Some(match c {
            b'(' => Obj::Str(self.literal_string()),
            b'<' if self.b.get(self.i + 1) == Some(&b'<') => {
                self.i += 2;
                let mut d = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek() {
                        None => break,
                        Some(b'>') => {
                            self.i += if self.b.get(self.i + 1) == Some(&b'>') { 2 } else { 1 };
                            break;
                        }
                        Some(b'/') => {
                            let k = self.name();
                            let v = self.object(refs, depth + 1).unwrap_or(Obj::Null);
                            d.push((k, v));
                        }
                        Some(_) => {
                            // A malformed entry: skip a token.
                            if self.object(refs, depth + 1).is_none() {
                                break;
                            }
                        }
                    }
                }
                Obj::Dict(d)
            }
            b'<' => Obj::Str(self.hex_string()),
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek() {
                        None => break,
                        Some(b']') => {
                            self.i += 1;
                            break;
                        }
                        Some(_) => match self.object(refs, depth + 1) {
                            Some(o) => a.push(o),
                            None => break,
                        },
                    }
                }
                Obj::Arr(a)
            }
            b'/' => Obj::Name(self.name()),
            b')' | b'>' | b']' | b'{' | b'}' => {
                self.i += 1;
                Obj::Op(String::from_utf8_lossy(&[c]).into_owned())
            }
            _ => {
                let tok = self.regular();
                if tok.is_empty() {
                    self.i += 1;
                    return Some(Obj::Null);
                }
                let s = String::from_utf8_lossy(tok).into_owned();
                if let Ok(n) = s.parse::<f64>() {
                    if refs && !s.contains('.') && !s.starts_with(['-', '+']) {
                        // `N G R`?
                        let save = self.i;
                        self.skip_ws();
                        let generation = self.regular();
                        if !generation.is_empty() && generation.iter().all(u8::is_ascii_digit) {
                            self.skip_ws();
                            if self.peek() == Some(b'R') && self.b.get(self.i + 1).is_none_or(|&b| is_ws(b) || is_delim(b)) {
                                self.i += 1;
                                return Some(Obj::Ref(n as u32));
                            }
                        }
                        self.i = save;
                    }
                    return Some(Obj::Num(n));
                }
                match s.as_str() {
                    "true" => Obj::Bool(true),
                    "false" => Obj::Bool(false),
                    "null" => Obj::Null,
                    _ => Obj::Op(s),
                }
            }
        })
    }
}

/// One indirect object: its value and, for a stream, its raw data.
struct Indirect {
    value: Obj,
    stream: Option<(usize, usize)>,
}

struct Doc<'a> {
    data: &'a [u8],
    objects: HashMap<u32, Indirect>,
    /// Objects found in object streams (`/Type /ObjStm`).
    packed: HashMap<u32, Obj>,
    trailers: Vec<Obj>,
    /// All decoding charges the document's budget.
    budget: &'a Budget,
    /// Object streams that could not be decoded (their objects are lost).
    lost_object_streams: usize,
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    memchr::memmem::find(&hay[from..], needle).map(|p| p + from)
}

impl<'a> Doc<'a> {
    fn scan(data: &'a [u8], budget: &'a Budget) -> Doc<'a> {
        let mut doc = Doc { data, objects: HashMap::new(), packed: HashMap::new(), trailers: Vec::new(), budget, lost_object_streams: 0 };
        for pos in memchr::memmem::find_iter(data, b"obj") {
            // `N G obj`, not `endobj`.
            if pos >= 3 && &data[pos - 3..pos] == b"end" {
                continue;
            }
            if data.get(pos + 3).is_some_and(|&b| !is_ws(b) && !is_delim(b)) {
                continue;
            }
            let Some(num) = Self::object_number(data, pos) else { continue };
            let mut lx = Lexer::new(data, pos + 3);
            let Some(value) = lx.object(true, 0) else { continue };
            lx.skip_ws();
            let mut stream = None;
            if data[lx.i..].starts_with(b"stream") {
                let mut start = lx.i + 6;
                if data.get(start) == Some(&b'\r') {
                    start += 1;
                }
                if data.get(start) == Some(&b'\n') {
                    start += 1;
                }
                let direct = value.get("Length").and_then(Obj::num).map(|n| n as usize);
                let end = match direct {
                    Some(len) if start + len <= data.len() && {
                        let mut k = start + len;
                        while k < data.len() && is_ws(data[k]) {
                            k += 1;
                        }
                        data[k..].starts_with(b"endstream")
                    } => Some(start + len),
                    _ => find(data, b"endstream", start).map(|mut e| {
                        if e > start && data[e - 1] == b'\n' {
                            e -= 1;
                        }
                        if e > start && data[e - 1] == b'\r' {
                            e -= 1;
                        }
                        e
                    }),
                };
                if let Some(end) = end {
                    stream = Some((start, end));
                }
            }
            doc.objects.insert(num, Indirect { value, stream });
        }
        for pos in memchr::memmem::find_iter(data, b"trailer") {
            let mut lx = Lexer::new(data, pos + 7);
            if let Some(t @ Obj::Dict(_)) = lx.object(true, 0) {
                doc.trailers.push(t);
            }
        }
        // Cross-reference streams carry the trailer entries.
        let xref: Vec<Obj> = doc.objects.values().filter(|o| o.value.get("Type").and_then(Obj::name) == Some("XRef")).map(|o| o.value.clone()).collect();
        doc.trailers.extend(xref);
        doc.unpack_object_streams();
        doc
    }

    /// The object number before `N G obj` at `pos`.
    fn object_number(data: &[u8], pos: usize) -> Option<u32> {
        let mut k = pos;
        let ws = |k: &mut usize| {
            let end = *k;
            while *k > 0 && is_ws(data[*k - 1]) {
                *k -= 1;
            }
            *k < end
        };
        let digits = |k: &mut usize| {
            let end = *k;
            while *k > 0 && data[*k - 1].is_ascii_digit() {
                *k -= 1;
            }
            (*k < end).then(|| std::str::from_utf8(&data[*k..end]).ok()?.parse::<u32>().ok()).flatten()
        };
        if !ws(&mut k) {
            return None;
        }
        digits(&mut k)?;
        if !ws(&mut k) {
            return None;
        }
        let num = digits(&mut k)?;
        if k > 0 && !is_ws(data[k - 1]) && !is_delim(data[k - 1]) && data[k - 1] != b'>' {
            return None;
        }
        Some(num)
    }

    fn unpack_object_streams(&mut self) {
        let streams: Vec<u32> = self.objects.iter().filter(|(_, o)| o.value.get("Type").and_then(Obj::name) == Some("ObjStm")).map(|(n, _)| *n).collect();
        for num in streams {
            let Ok(data) = self.stream(num) else {
                self.lost_object_streams += 1;
                continue;
            };
            let (n, first) = {
                let o = &self.objects[&num].value;
                (self.resolve(o.get("N")).and_then(Obj::num).unwrap_or(0.0) as usize, self.resolve(o.get("First")).and_then(Obj::num).unwrap_or(0.0) as usize)
            };
            let mut header = Lexer::new(&data, 0);
            let mut entries = Vec::new();
            for _ in 0..n.min(1 << 20) {
                let (Some(Obj::Num(id)), Some(Obj::Num(off))) = (header.object(false, 0), header.object(false, 0)) else { break };
                entries.push((id as u32, off as usize));
            }
            for (id, off) in entries {
                if self.objects.contains_key(&id) || self.packed.contains_key(&id) {
                    continue;
                }
                let mut lx = Lexer::new(&data, first.saturating_add(off).min(data.len()));
                if let Some(v) = lx.object(true, 0) {
                    self.packed.insert(id, v);
                }
            }
        }
    }

    fn get(&self, num: u32) -> Option<&Obj> {
        self.objects.get(&num).map(|o| &o.value).or_else(|| self.packed.get(&num))
    }

    /// Follow references (a bounded chain).
    fn resolve<'b>(&'b self, mut o: Option<&'b Obj>) -> Option<&'b Obj> {
        for _ in 0..16 {
            match o {
                Some(Obj::Ref(n)) => o = self.get(*n),
                other => return other,
            }
        }
        None
    }

    /// A stream object's decoded data.
    fn stream(&self, num: u32) -> Result<Vec<u8>, String> {
        let o = self.objects.get(&num).ok_or_else(|| format!("missing stream object {num}"))?;
        let (start, end) = o.stream.ok_or_else(|| format!("object {num} is not a stream"))?;
        let mut data = self.data[start..end].to_vec();
        let filters: Vec<String> = match self.resolve(o.value.get("Filter")) {
            None | Some(Obj::Null) => Vec::new(),
            Some(Obj::Name(n)) => vec![n.clone()],
            Some(Obj::Arr(a)) => a.iter().filter_map(|f| self.resolve(Some(f)).and_then(Obj::name).map(str::to_string)).collect(),
            Some(_) => return Err(format!("object {num} has an invalid filter")),
        };
        for f in &filters {
            data = match f.as_str() {
                "FlateDecode" | "Fl" => inflate(&data, self.budget)?,
                "ASCIIHexDecode" | "AHx" => ascii_hex(&data),
                "ASCII85Decode" | "A85" => ascii85(&data)?,
                other => return Err(format!("unsupported stream filter [{other}]")),
            };
        }
        if let Some(parms) = self.resolve(o.value.get("DecodeParms")) {
            let parms = match parms {
                Obj::Arr(a) => a.first().and_then(|p| self.resolve(Some(p))).cloned().unwrap_or(Obj::Null),
                p => p.clone(),
            };
            if parms.get("Predictor").and_then(Obj::num).unwrap_or(1.0) >= 10.0 {
                let columns = parms.get("Columns").and_then(Obj::num).unwrap_or(1.0).max(1.0) as usize;
                data = png_unpredict(&data, columns)?;
            }
        }
        Ok(data)
    }

    fn stream_of(&self, o: Option<&Obj>) -> Option<u32> {
        match o {
            Some(Obj::Ref(n)) if self.objects.get(n).is_some_and(|o| o.stream.is_some()) => Some(*n),
            _ => None,
        }
    }

    fn trailer(&self, key: &str) -> Option<&Obj> {
        self.trailers.iter().rev().find_map(|t| t.get(key))
    }
}

fn inflate(data: &[u8], budget: &Budget) -> Result<Vec<u8>, String> {
    let cap = budget.inflate_cap(MAX_STREAM);
    let mut out = Vec::new();
    let mut z = flate2::read::ZlibDecoder::new(data).take(cap);
    match z.read_to_end(&mut out) {
        Ok(_) => {}
        // A truncated or slightly damaged stream keeps what decoded.
        Err(e) if out.is_empty() => {
            let mut raw = flate2::read::DeflateDecoder::new(data).take(cap);
            let mut alt = Vec::new();
            if raw.read_to_end(&mut alt).is_err() && alt.is_empty() {
                return Err(format!("invalid compressed stream: {e}"));
            }
            out = alt;
        }
        Err(_) => {}
    }
    // Every decoded stream counts against the document's total.
    budget.inflate(out.len() as u64, MAX_STREAM, "decoded stream")?;
    Ok(out)
}

fn ascii_hex(data: &[u8]) -> Vec<u8> {
    let mut buf = vec![b'<'];
    buf.extend(data.iter().take_while(|&&b| b != b'>'));
    buf.push(b'>');
    Lexer::new(&buf, 0).hex_string()
}

fn ascii85(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut group = [0u32; 5];
    let mut n = 0;
    let body = data.strip_prefix(b"<~").unwrap_or(data);
    for &c in body {
        match c {
            b'~' => break,
            b'z' if n == 0 => out.extend_from_slice(&[0, 0, 0, 0]),
            b'!'..=b'u' => {
                group[n] = (c - b'!') as u32;
                n += 1;
                if n == 5 {
                    let v = group.iter().fold(0u64, |a, &d| a * 85 + d as u64);
                    out.extend_from_slice(&(v as u32).to_be_bytes());
                    n = 0;
                }
            }
            c if is_ws(c) => {}
            _ => return Err("invalid ASCII85 data".into()),
        }
    }
    if n > 1 {
        for g in group.iter_mut().skip(n) {
            *g = 84;
        }
        let v = group.iter().fold(0u64, |a, &d| a * 85 + d as u64);
        out.extend_from_slice(&(v as u32).to_be_bytes()[..n - 1]);
    }
    Ok(out)
}

fn png_unpredict(data: &[u8], columns: usize) -> Result<Vec<u8>, String> {
    let row = columns;
    let mut out = Vec::with_capacity(data.len());
    let mut prev = vec![0u8; row];
    for chunk in data.chunks(row + 1) {
        if chunk.len() < row + 1 {
            break;
        }
        let (kind, line) = (chunk[0], &chunk[1..]);
        let mut cur = vec![0u8; row];
        for i in 0..row {
            let left = if i > 0 { cur[i - 1] } else { 0 };
            let up = prev[i];
            let upleft = if i > 0 { prev[i - 1] } else { 0 };
            cur[i] = match kind {
                0 => line[i],
                1 => line[i].wrapping_add(left),
                2 => line[i].wrapping_add(up),
                3 => line[i].wrapping_add(((left as u16 + up as u16) / 2) as u8),
                4 => {
                    let p = left as i16 + up as i16 - upleft as i16;
                    let (pa, pb, pc) = ((p - left as i16).abs(), (p - up as i16).abs(), (p - upleft as i16).abs());
                    let pred = if pa <= pb && pa <= pc { left } else if pb <= pc { up } else { upleft };
                    line[i].wrapping_add(pred)
                }
                _ => return Err("invalid PNG predictor".into()),
            };
        }
        out.extend_from_slice(&cur);
        prev = cur;
    }
    Ok(out)
}

// ------------------------------------------------------------ fonts

/// Windows-1252 code points 0x80..0x9F (the rest is Latin-1).
const CP1252_HIGH: [char; 32] = [
    '\u{20AC}', '\u{FFFD}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{FFFD}', '\u{017D}', '\u{FFFD}',
    '\u{FFFD}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\u{FFFD}', '\u{017E}', '\u{0178}',
];

pub(super) fn cp1252(b: u8) -> char {
    match b {
        0x80..=0x9F => CP1252_HIGH[(b - 0x80) as usize],
        _ => b as char,
    }
}

/// A glyph name's character (the Adobe names `/Differences` uses most).
fn glyph_char(name: &str) -> Option<String> {
    if name.chars().count() == 1 && name.is_ascii() {
        return Some(name.to_string());
    }
    let hex = |h: &str| u32::from_str_radix(h, 16).ok().and_then(char::from_u32).map(String::from);
    if let Some(h) = name.strip_prefix("uni").filter(|h| h.len() == 4) {
        return hex(h);
    }
    if let Some(h) = name.strip_prefix('u').filter(|h| (4..=6).contains(&h.len())) {
        return hex(h);
    }
    const NAMES: &[(&str, &str)] = &[
        ("space", " "), ("exclam", "!"), ("quotedbl", "\""), ("numbersign", "#"), ("dollar", "$"), ("percent", "%"), ("ampersand", "&"),
        ("quotesingle", "'"), ("parenleft", "("), ("parenright", ")"), ("asterisk", "*"), ("plus", "+"), ("comma", ","), ("hyphen", "-"),
        ("period", "."), ("slash", "/"), ("zero", "0"), ("one", "1"), ("two", "2"), ("three", "3"), ("four", "4"), ("five", "5"),
        ("six", "6"), ("seven", "7"), ("eight", "8"), ("nine", "9"), ("colon", ":"), ("semicolon", ";"), ("less", "<"), ("equal", "="),
        ("greater", ">"), ("question", "?"), ("at", "@"), ("bracketleft", "["), ("backslash", "\\"), ("bracketright", "]"),
        ("asciicircum", "^"), ("underscore", "_"), ("grave", "`"), ("braceleft", "{"), ("bar", "|"), ("braceright", "}"),
        ("asciitilde", "~"), ("quoteleft", "\u{2018}"), ("quoteright", "\u{2019}"), ("quotedblleft", "\u{201C}"),
        ("quotedblright", "\u{201D}"), ("quotesinglbase", "\u{201A}"), ("quotedblbase", "\u{201E}"), ("endash", "\u{2013}"),
        ("emdash", "\u{2014}"), ("bullet", "\u{2022}"), ("ellipsis", "\u{2026}"), ("fi", "fi"), ("fl", "fl"), ("ff", "ff"),
        ("ffi", "ffi"), ("ffl", "ffl"), ("nbspace", "\u{A0}"), ("copyright", "\u{A9}"), ("registered", "\u{AE}"),
        ("trademark", "\u{2122}"), ("degree", "\u{B0}"), ("section", "\u{A7}"), ("paragraph", "\u{B6}"), ("dagger", "\u{2020}"),
        ("germandbls", "\u{DF}"), ("adieresis", "\u{E4}"), ("odieresis", "\u{F6}"), ("udieresis", "\u{FC}"), ("Adieresis", "\u{C4}"),
        ("Odieresis", "\u{D6}"), ("Udieresis", "\u{DC}"), ("eacute", "\u{E9}"), ("egrave", "\u{E8}"), ("ecircumflex", "\u{EA}"),
        ("edieresis", "\u{EB}"), ("aacute", "\u{E1}"), ("agrave", "\u{E0}"), ("acircumflex", "\u{E2}"), ("atilde", "\u{E3}"),
        ("aring", "\u{E5}"), ("ccedilla", "\u{E7}"), ("iacute", "\u{ED}"), ("igrave", "\u{EC}"), ("icircumflex", "\u{EE}"),
        ("idieresis", "\u{EF}"), ("ntilde", "\u{F1}"), ("oacute", "\u{F3}"), ("ograve", "\u{F2}"), ("ocircumflex", "\u{F4}"),
        ("otilde", "\u{F5}"), ("oslash", "\u{F8}"), ("uacute", "\u{FA}"), ("ugrave", "\u{F9}"), ("ucircumflex", "\u{FB}"),
        ("yacute", "\u{FD}"), ("ydieresis", "\u{FF}"), ("Eacute", "\u{C9}"), ("Egrave", "\u{C8}"), ("Aacute", "\u{C1}"),
        ("Agrave", "\u{C0}"), ("Ccedilla", "\u{C7}"), ("Ntilde", "\u{D1}"), ("Oslash", "\u{D8}"), ("minus", "\u{2212}"),
        ("Euro", "\u{20AC}"), ("sterling", "\u{A3}"), ("yen", "\u{A5}"), ("cent", "\u{A2}"),
    ];
    NAMES.iter().find(|(n, _)| *n == name).map(|(_, c)| c.to_string())
}

/// How a font's string bytes become text.
#[derive(Default)]
struct Font {
    /// `ToUnicode`: code -> text, and the code byte lengths it declares.
    to_unicode: HashMap<u32, String>,
    code_lens: Vec<usize>,
    /// A composite (`Type0`) font: two-byte codes unless the map says else.
    composite: bool,
    /// Simple font code -> text from `/Encoding` `/Differences`.
    differences: HashMap<u8, String>,
}

impl Font {
    fn code_len(&self) -> usize {
        match self.code_lens.as_slice() {
            [] => {
                if self.composite {
                    2
                } else {
                    1
                }
            }
            lens => *lens.iter().min().unwrap_or(&1),
        }
    }

    /// Decode shown bytes; counts the codes nothing maps.
    fn decode(&self, bytes: &[u8], unmapped: &mut usize) -> String {
        let mut out = String::new();
        let mut i = 0;
        let uniform = self.code_lens.iter().all(|l| *l == self.code_len());
        while i < bytes.len() {
            // Mixed code lengths: the shortest code the map knows.
            let mut len = self.code_len();
            if !uniform {
                for l in self.code_lens.iter().copied() {
                    if i + l <= bytes.len() && self.to_unicode.contains_key(&code(&bytes[i..i + l])) {
                        len = l;
                        break;
                    }
                }
            }
            let len = len.min(bytes.len() - i).max(1);
            let c = code(&bytes[i..i + len]);
            i += len;
            if let Some(t) = self.to_unicode.get(&c) {
                out.push_str(t);
            } else if !self.composite && len == 1 {
                match self.differences.get(&(c as u8)) {
                    Some(t) => out.push_str(t),
                    None if self.differences.is_empty() || c >= 0x20 => out.push(cp1252(c as u8)),
                    None => *unmapped += 1,
                }
            } else {
                *unmapped += 1;
            }
        }
        out
    }
}

fn code(b: &[u8]) -> u32 {
    b.iter().fold(0u32, |a, &x| (a << 8) | x as u32)
}

fn utf16be(b: &[u8]) -> String {
    let units: Vec<u16> = b.chunks(2).map(|p| ((p[0] as u16) << 8) | *p.get(1).unwrap_or(&0) as u16).collect();
    String::from_utf16_lossy(&units)
}

/// Parse a `ToUnicode` CMap.
fn parse_cmap(data: &[u8], font: &mut Font) {
    let mut lx = Lexer::new(data, 0);
    let mut ops: Vec<Obj> = Vec::new();
    let mut mode = "";
    while let Some(o) = lx.object(false, 0) {
        if let Obj::Op(op) = &o {
            match op.as_str() {
                "begincodespacerange" | "beginbfchar" | "beginbfrange" => {
                    mode = match op.as_str() {
                        "begincodespacerange" => "space",
                        "beginbfchar" => "char",
                        _ => "range",
                    };
                    ops.clear();
                    continue;
                }
                "endcodespacerange" => {
                    for p in ops.chunks(2) {
                        if let [Obj::Str(lo), Obj::Str(_)] = p {
                            if !font.code_lens.contains(&lo.len()) && (1..=4).contains(&lo.len()) {
                                font.code_lens.push(lo.len());
                            }
                        }
                    }
                }
                "endbfchar" => {
                    for p in ops.chunks(2) {
                        if let [Obj::Str(src), dst] = p {
                            let t = match dst {
                                Obj::Str(d) => utf16be(d),
                                Obj::Name(n) => glyph_char(n).unwrap_or_default(),
                                _ => continue,
                            };
                            font.to_unicode.insert(code(src), t);
                        }
                    }
                }
                "endbfrange" => {
                    for p in ops.chunks(3) {
                        let [Obj::Str(lo), Obj::Str(hi), dst] = p else { continue };
                        let (lo, hi) = (code(lo), code(hi));
                        if hi < lo || hi - lo > 0xFFFF {
                            continue;
                        }
                        match dst {
                            Obj::Str(d) => {
                                let mut units: Vec<u16> = d.chunks(2).map(|p| ((p[0] as u16) << 8) | *p.get(1).unwrap_or(&0) as u16).collect();
                                if units.is_empty() {
                                    continue;
                                }
                                let last = units.len() - 1;
                                let base = units[last];
                                for (k, c) in (lo..=hi).enumerate() {
                                    units[last] = base.wrapping_add(k as u16);
                                    font.to_unicode.insert(c, String::from_utf16_lossy(&units));
                                }
                            }
                            Obj::Arr(a) => {
                                for (c, d) in (lo..=hi).zip(a.iter()) {
                                    if let Obj::Str(d) = d {
                                        font.to_unicode.insert(c, utf16be(d));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
            mode = if op.starts_with("end") { "" } else { mode };
            ops.clear();
            continue;
        }
        if !mode.is_empty() {
            ops.push(o);
        }
    }
}

// ------------------------------------------------------------ content

#[derive(Clone, Copy)]
struct Matrix([f64; 6]);

impl Matrix {
    const ID: Matrix = Matrix([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    fn translate(self, tx: f64, ty: f64) -> Matrix {
        let [a, b, c, d, e, f] = self.0;
        Matrix([a, b, c, d, e + tx * a + ty * c, f + tx * b + ty * d])
    }
}

struct Extractor<'d, 'a, 't, 'b> {
    doc: &'d Doc<'a>,
    fonts: HashMap<u32, std::rc::Rc<Font>>,
    out: &'t mut Text<'b>,
    /// Content streams (pages, forms) that could not be decoded.
    undecodable: usize,
    last_y: Option<f64>,
    /// Glyph codes shown, and those no map decoded.
    shown: usize,
    unmapped: usize,
    /// Form XObjects being run (cycles).
    forms: HashSet<u32>,
}

impl<'d, 'a, 't, 'b> Extractor<'d, 'a, 't, 'b> {
    fn font(&mut self, resources: Option<&Obj>, name: &str) -> std::rc::Rc<Font> {
        let doc = self.doc;
        let fonts = doc.resolve(resources.and_then(|r| r.get("Font")));
        let Some(r) = fonts.and_then(|f| f.get(name)) else { return Default::default() };
        let key = match r {
            Obj::Ref(n) => Some(*n),
            _ => None,
        };
        if let Some(f) = key.and_then(|k| self.fonts.get(&k)) {
            return f.clone();
        }
        let mut font = Font::default();
        if let Some(fd) = doc.resolve(Some(r)) {
            font.composite = fd.get("Subtype").and_then(Obj::name) == Some("Type0");
            if let Some(n) = doc.stream_of(fd.get("ToUnicode")) {
                if let Ok(data) = doc.stream(n) {
                    parse_cmap(&data, &mut font);
                }
            }
            if let Some(enc @ Obj::Dict(_)) = doc.resolve(fd.get("Encoding")) {
                if let Some(Obj::Arr(diffs)) = doc.resolve(enc.get("Differences")) {
                    let mut code = 0u32;
                    for d in diffs {
                        match d {
                            Obj::Num(n) => code = *n as u32,
                            Obj::Name(g) => {
                                if code <= 255 {
                                    if let Some(c) = glyph_char(g) {
                                        font.differences.insert(code as u8, c);
                                    }
                                }
                                code += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        let font = std::rc::Rc::new(font);
        if let Some(k) = key {
            self.fonts.insert(k, font.clone());
        }
        font
    }

    fn line(&mut self, y: f64, size: f64) {
        let h = size.abs().max(1.0);
        if let Some(ly) = self.last_y {
            let gap = (ly - y).abs();
            if gap > h * 0.5 && !self.out.is_empty() {
                // A drop of more than two lines starts a paragraph.
                self.out.trim_end_spaces();
                self.out.push_str(if gap > h * 2.0 { "\n\n" } else { "\n" });
            }
        }
        self.last_y = Some(y);
    }

    fn show(&mut self, font: &Font, bytes: &[u8]) {
        let mut unmapped = 0;
        let text = font.decode(bytes, &mut unmapped);
        self.shown += bytes.len().div_ceil(font.code_len().max(1));
        self.unmapped += unmapped;
        for c in text.chars() {
            match c {
                '\t' | '\n' | '\r' => self.out.push(' '),
                c if c.is_control() => {}
                c => self.out.push(c),
            }
        }
    }

    fn run(&mut self, content: &[u8], resources: Option<&Obj>, depth: usize) {
        let mut lx = Lexer::new(content, 0);
        let mut operands: Vec<Obj> = Vec::new();
        let (mut tm, mut tlm) = (Matrix::ID, Matrix::ID);
        let mut leading = 0.0;
        let mut size = 0.0;
        let mut font: std::rc::Rc<Font> = Default::default();
        let num = |ops: &[Obj], i: usize| ops.get(i).and_then(Obj::num).unwrap_or(0.0);
        while let Some(o) = lx.object(false, 0) {
            // The text budget is spent: nothing further is extracted.
            if self.out.full() {
                return;
            }
            let Obj::Op(op) = o else {
                operands.push(o);
                if operands.len() > 4096 {
                    operands.clear();
                }
                continue;
            };
            let n = operands.len();
            match op.as_str() {
                "BT" => {
                    tm = Matrix::ID;
                    tlm = Matrix::ID;
                }
                "Tf" if n >= 2 => {
                    if let Some(name) = operands[n - 2].name() {
                        font = self.font(resources, name);
                    }
                    size = num(&operands, n - 1);
                }
                "TL" if n >= 1 => leading = num(&operands, n - 1),
                "Td" if n >= 2 => {
                    tlm = tlm.translate(num(&operands, n - 2), num(&operands, n - 1));
                    tm = tlm;
                }
                "TD" if n >= 2 => {
                    leading = -num(&operands, n - 1);
                    tlm = tlm.translate(num(&operands, n - 2), num(&operands, n - 1));
                    tm = tlm;
                }
                "Tm" if n >= 6 => {
                    tlm = Matrix([num(&operands, n - 6), num(&operands, n - 5), num(&operands, n - 4), num(&operands, n - 3), num(&operands, n - 2), num(&operands, n - 1)]);
                    tm = tlm;
                }
                "T*" => {
                    tlm = tlm.translate(0.0, -leading);
                    tm = tlm;
                }
                "Tj" | "'" | "\"" | "TJ" => {
                    if op == "'" || op == "\"" {
                        tlm = tlm.translate(0.0, -leading);
                        tm = tlm;
                    }
                    let scale = (tm.0[1].powi(2) + tm.0[3].powi(2)).sqrt().max(f64::MIN_POSITIVE);
                    self.line(tm.0[5], size * scale);
                    match operands.last() {
                        Some(Obj::Str(s)) => {
                            let s = s.clone();
                            self.show(&font, &s);
                        }
                        Some(Obj::Arr(a)) => {
                            for item in a.clone() {
                                match item {
                                    Obj::Str(s) => self.show(&font, &s),
                                    // A gap wider than a fifth of an em is a word break.
                                    Obj::Num(d) if d < -200.0 && !self.out.as_str().ends_with([' ', '\n']) && !self.out.is_empty() => self.out.push(' '),
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                "Do" if n >= 1 && depth < 8 => {
                    let doc = self.doc;
                    if let Some(name) = operands[n - 1].name() {
                        let xobjects = doc.resolve(resources.and_then(|r| r.get("XObject")));
                        if let Some(num) = doc.stream_of(xobjects.and_then(|x| x.get(name))) {
                            let form = doc.objects[&num].value.get("Subtype").and_then(Obj::name) == Some("Form");
                            if form && self.forms.insert(num) {
                                match doc.stream(num) {
                                    Ok(data) => {
                                        let own = doc.resolve(doc.objects[&num].value.get("Resources")).cloned();
                                        self.run(&data, own.as_ref().or(resources), depth + 1);
                                    }
                                    Err(_) => self.undecodable += 1,
                                }
                                self.forms.remove(&num);
                            }
                        }
                    }
                }
                "BI" => {
                    // Inline image: skip its data up to `EI`.
                    if let Some(p) = find(content, b"ID", lx.i) {
                        let mut k = p + 2;
                        loop {
                            match find(content, b"EI", k) {
                                Some(e) if e > 0 && is_ws(content[e - 1]) && content.get(e + 2).is_none_or(|&b| is_ws(b)) => {
                                    lx.i = e + 2;
                                    break;
                                }
                                Some(e) => k = e + 2,
                                None => {
                                    lx.i = content.len();
                                    break;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            operands.clear();
        }
    }
}

/// A PDF text string (`FE FF` UTF-16BE, UTF-8 with BOM, else PDFDocEncoding
/// taken as Windows-1252).
fn text_string(b: &[u8]) -> String {
    if let Some(rest) = b.strip_prefix(&[0xFE, 0xFF]) {
        return utf16be(rest);
    }
    if let Some(rest) = b.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    b.iter().map(|&c| cp1252(c)).collect()
}

/// `D:YYYYMMDDHHmmSSOHH'mm'` as `YYYY-MM-DDTHH:MM:SSZ` (UTC), as Tika
/// reports PDF dates.
fn pdf_date(s: &str) -> Option<String> {
    let s = s.trim().strip_prefix("D:").unwrap_or(s.trim());
    let digits: String = s.chars().take_while(char::is_ascii_digit).collect();
    if digits.len() < 4 {
        return None;
    }
    let part = |a: usize, b: usize, d: u32| digits.get(a..b).and_then(|p| p.parse::<u32>().ok()).unwrap_or(d);
    let (y, mo, d, h, mi, se) = (part(0, 4, 1970) as i32, part(4, 6, 1), part(6, 8, 1), part(8, 10, 0), part(10, 12, 0), part(12, 14, 0));
    let rest = &s[digits.len()..];
    let offset = match rest.chars().next() {
        Some(sign @ ('+' | '-')) => {
            let nums: Vec<i64> = rest[1..].split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).filter_map(|p| p.parse().ok()).collect();
            let secs = nums.first().copied().unwrap_or(0) * 3600 + nums.get(1).copied().unwrap_or(0) * 60;
            if sign == '-' { -secs } else { secs }
        }
        _ => 0,
    };
    let local = chrono::NaiveDate::from_ymd_opt(y, mo, d)?.and_hms_opt(h, mi, se.min(59))?;
    let utc = local - chrono::Duration::seconds(offset);
    Some(utc.format("%Y-%m-%dT%H:%M:%SZ").to_string())
}

/// Extract a PDF's text and document information (`Err`: why it cannot).
/// Extract a PDF's text into `out`; returns its document information. Like
/// Tika's `PDF2XHTML`, which throws "Unable to extract PDF content" when
/// any page failed, a content stream, form or object stream that cannot be
/// decoded fails the document: its text would be silently incomplete.
pub(super) fn extract(data: &[u8], out: &mut Text) -> Result<Vec<(&'static str, String)>, ParseError> {
    let doc = Doc::scan(data, out.budget());
    if doc.objects.is_empty() {
        return Err("no PDF objects found: the document is damaged".into());
    }
    if doc.trailer("Encrypt").is_some() {
        return Err("the document is encrypted".into());
    }
    // Pages in page-tree order (inherited resources), else every page
    // object in object order.
    let mut pages: Vec<(Obj, Option<Obj>)> = Vec::new();
    let mut missing_pages = 0usize;
    if let Some(root) = doc.resolve(doc.trailer("Root")) {
        let mut visited = HashSet::new();
        let mut stack = vec![(root.get("Pages").cloned(), None::<Obj>)];
        while let Some((node, inherited)) = stack.pop() {
            if let Some(Obj::Ref(n)) = &node {
                if !visited.insert(*n) {
                    continue;
                }
            }
            let Some(node) = doc.resolve(node.as_ref()) else {
                // A page (or subtree) the document does not contain.
                missing_pages += 1;
                continue;
            };
            let resources = doc.resolve(node.get("Resources")).cloned().or(inherited);
            match doc.resolve(node.get("Kids")) {
                Some(Obj::Arr(kids)) => {
                    for kid in kids.iter().rev() {
                        stack.push((Some(kid.clone()), resources.clone()));
                    }
                }
                _ if node.get("Type").and_then(Obj::name) != Some("Pages") => pages.push((node.clone(), resources)),
                _ => {}
            }
            if visited.len() > 1 << 20 {
                break;
            }
        }
    }
    if pages.is_empty() {
        let mut nums: Vec<u32> = doc.objects.keys().chain(doc.packed.keys()).copied().filter(|n| doc.get(*n).and_then(|o| o.get("Type")).and_then(Obj::name) == Some("Page")).collect();
        nums.sort_unstable();
        nums.dedup();
        for n in nums {
            let page = doc.get(n).cloned().unwrap_or(Obj::Null);
            let resources = doc.resolve(page.get("Resources")).cloned();
            pages.push((page, resources));
        }
    }
    if pages.is_empty() {
        return Err("the document has no pages".into());
    }
    if missing_pages > 0 && !pages.is_empty() {
        return Err(format!("Unable to extract PDF content: {missing_pages} pages could not be read").into());
    }
    if doc.lost_object_streams > 0 {
        return Err(format!("Unable to extract PDF content: {} object streams could not be decoded", doc.lost_object_streams).into());
    }
    let mut x = Extractor { doc: &doc, fonts: HashMap::new(), out, undecodable: 0, last_y: None, shown: 0, unmapped: 0, forms: HashSet::new() };
    for (page, resources) in &pages {
        if x.out.full() {
            break;
        }
        let refs: Vec<Obj> = match page.get("Contents") {
            Some(Obj::Arr(a)) => a.clone(),
            Some(r @ Obj::Ref(_)) => match doc.resolve(Some(r)) {
                Some(Obj::Arr(a)) => a.clone(),
                _ => vec![r.clone()],
            },
            _ => Vec::new(),
        };
        let mut content = Vec::new();
        for r in &refs {
            let Some(n) = doc.stream_of(Some(r)) else { continue };
            match doc.stream(n) {
                Ok(d) => {
                    content.extend_from_slice(&d);
                    content.push(b'\n');
                }
                Err(_) => x.undecodable += 1,
            }
        }
        let before = x.out.len();
        x.last_y = None;
        x.run(&content, resources.as_ref(), 0);
        if x.out.len() > before {
            x.out.push('\n');
        }
        if x.out.len() > before {
            x.out.push('\n');
        }
    }
    if x.undecodable > 0 {
        return Err(format!("Unable to extract PDF content: {} content streams could not be decoded", x.undecodable).into());
    }
    if x.out.as_str().trim().is_empty() && x.shown > 0 && !x.out.full() {
        return Err("the document shows text in fonts without a Unicode mapping".into());
    }
    let mut meta = Vec::new();
    if let Some(info) = doc.resolve(doc.trailer("Info")) {
        let text = |k: &str| match doc.resolve(info.get(k)) {
            Some(Obj::Str(s)) => Some(text_string(s)).filter(|t| !t.trim().is_empty()),
            _ => None,
        };
        if let Some(t) = text("Title") {
            meta.push(("title", t));
        }
        if let Some(t) = text("Author") {
            meta.push(("author", t));
        }
        // Tika's subject is ES's keywords unless the document has keywords.
        if let Some(t) = text("Keywords").or_else(|| text("Subject")) {
            meta.push(("keywords", t));
        }
        if let Some(t) = text("Creator") {
            meta.push(("creator_tool", t));
        }
        if let Some(t) = text("CreationDate").and_then(|d| pdf_date(&d)) {
            meta.push(("date", t));
        }
        if let Some(t) = text("ModDate").and_then(|d| pdf_date(&d)) {
            meta.push(("modified", t));
        }
    }
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text and document information of a PDF under an unlimited budget.
    fn extract(data: &[u8]) -> Result<(String, Vec<(&'static str, String)>), String> {
        let budget = Budget::new(-1);
        let mut out = Text::new(&budget);
        let meta = super::extract(data, &mut out).map_err(|e| e.message().to_string())?;
        Ok((out.into_string(), meta))
    }

    /// A one-page PDF showing `lines` (Helvetica, WinAnsi), with a
    /// deflated content stream.
    fn pdf(lines: &[(&str, f64)]) -> Vec<u8> {
        use std::io::Write;
        let mut content = String::from("BT /F1 12 Tf\n");
        let mut y = 0.0;
        for (i, (text, gap)) in lines.iter().enumerate() {
            if i == 0 {
                content.push_str("72 720 Td\n");
            } else {
                content.push_str(&format!("0 {} Td\n", -gap));
            }
            y -= gap;
            content.push_str(&format!("({}) Tj\n", text.replace('(', "\\(").replace(')', "\\)")));
        }
        let _ = y;
        content.push_str("ET\n");
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(content.as_bytes()).unwrap();
        let stream = z.finish().unwrap();
        let mut out = b"%PDF-1.4\n".to_vec();
        out.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        out.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        out.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n");
        out.extend_from_slice(format!("4 0 obj\n<< /Length {} /Filter /FlateDecode >>\nstream\n", stream.len()).as_bytes());
        out.extend_from_slice(&stream);
        out.extend_from_slice(b"\nendstream\nendobj\n");
        out.extend_from_slice(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n");
        out.extend_from_slice(b"6 0 obj\n<< /Title (Quarterly report) /Author (Jane Doe) /CreationDate (D:20240102030405+01'00') >>\nendobj\n");
        out.extend_from_slice(b"trailer\n<< /Root 1 0 R /Info 6 0 R /Size 7 >>\n%%EOF\n");
        out
    }

    #[test]
    fn extracts_lines_paragraphs_and_info() {
        let (text, meta) = extract(&pdf(&[("Hello (PDF) world", 0.0), ("second line", 14.0), ("new paragraph", 40.0)])).unwrap();
        assert_eq!(text.trim(), "Hello (PDF) world\nsecond line\n\nnew paragraph");
        assert!(meta.contains(&("title", "Quarterly report".to_string())), "{meta:?}");
        assert!(meta.contains(&("author", "Jane Doe".to_string())), "{meta:?}");
        assert!(meta.contains(&("date", "2024-01-02T02:04:05Z".to_string())), "{meta:?}");
    }

    #[test]
    fn to_unicode_maps_and_ranges() {
        let mut f = Font::default();
        parse_cmap(b"1 begincodespacerange <00> <FF> endcodespacerange 2 beginbfchar <01> <0054> <02> <00FC> endbfchar 1 beginbfrange <10> <12> <0061> endbfrange", &mut f);
        let mut unmapped = 0;
        assert_eq!(f.decode(&[1, 2, 0x10, 0x11, 0x12], &mut unmapped), "Tüabc");
        assert_eq!(unmapped, 0);
        let mut c = Font { composite: true, ..Default::default() };
        parse_cmap(b"begincodespacerange <0000> <FFFF> endcodespacerange beginbfrange <0003> <0004> [<0048> <0069>] endbfrange", &mut c);
        assert_eq!(c.decode(&[0, 3, 0, 4, 0, 9], &mut unmapped), "Hi");
        assert_eq!(unmapped, 1);
    }

    #[test]
    fn damaged_or_encrypted_documents_fail() {
        assert!(extract(b"%PDF-1.4\nthis is not a pdf body").is_err());
        let mut enc = pdf(&[("secret", 0.0)]);
        let at = memchr::memmem::find(&enc, b"/Size 7").unwrap();
        enc.splice(at..at, b"/Encrypt 9 0 R ".iter().copied());
        assert_eq!(extract(&enc).unwrap_err(), "the document is encrypted");
        // A page whose text no font can decode is not a page without text.
        let mut garbled = pdf(&[("x", 0.0)]);
        let at = memchr::memmem::find(&garbled, b"/Subtype /Type1").unwrap();
        garbled.splice(at..at + 15, b"/Subtype /Type0".iter().copied());
        assert!(extract(&garbled).is_err());
    }

    /// A two-page PDF: page 1 shows `first` (Flate), page 2's content
    /// stream uses `filter` over `second`.
    fn two_pages(first: &str, filter: &str, second: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(format!("BT /F1 12 Tf 72 720 Td ({first}) Tj ET").as_bytes()).unwrap();
        let one = z.finish().unwrap();
        let mut out = b"%PDF-1.4\n".to_vec();
        out.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        out.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>\nendobj\n");
        out.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n");
        out.extend_from_slice(format!("4 0 obj\n<< /Length {} /Filter /FlateDecode >>\nstream\n", one.len()).as_bytes());
        out.extend_from_slice(&one);
        out.extend_from_slice(b"\nendstream\nendobj\n");
        out.extend_from_slice(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
        out.extend_from_slice(b"6 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>\nendobj\n");
        out.extend_from_slice(format!("7 0 obj\n<< /Length {} /Filter /{filter} >>\nstream\n", second.len()).as_bytes());
        out.extend_from_slice(second);
        out.extend_from_slice(b"\nendstream\nendobj\n");
        out.extend_from_slice(b"trailer\n<< /Root 1 0 R /Size 8 >>\n%%EOF\n");
        out
    }

    /// A page that cannot be decoded fails the document even when another
    /// page has text (Tika's "Unable to extract PDF content"); pages that
    /// decode, and the text budget, behave as before.
    #[test]
    fn a_page_that_cannot_be_decoded_fails_the_document() {
        let ok = two_pages("first page", "ASCIIHexDecode", b"4254202f46312031322054662037322037323020546420287365636f6e642920546a204554>");
        assert_eq!(extract(&ok).unwrap().0.trim(), "first page\n\nsecond");
        let lzw = two_pages("first page", "LZWDecode", b"\x80\x0b\x60\x50");
        assert!(extract(&lzw).unwrap_err().starts_with("Unable to extract PDF content"));
        let corrupt = two_pages("first page", "FlateDecode", b"\xff\xff\xff\xff not zlib");
        assert!(extract(&corrupt).unwrap_err().starts_with("Unable to extract PDF content"));
        // A page tree entry the document does not contain.
        let mut missing = ok.clone();
        let at = memchr::memmem::find(&missing, b"[3 0 R 6 0 R]").unwrap();
        missing.splice(at..at + 13, b"[3 0 R 9 0 R]".iter().copied());
        assert!(extract(&missing).unwrap_err().contains("pages could not be read"));
        // The text budget stops extraction (Tika's write limit).
        let budget = Budget::new(5);
        let mut out = Text::new(&budget);
        super::extract(&ok, &mut out).unwrap();
        assert_eq!(out.into_string(), "first");
        // So does the decompression budget, as a hard failure.
        let budget = Budget::with_limits(-1, 10);
        let mut out = Text::new(&budget);
        let _ = super::extract(&ok, &mut out);
        assert!(budget.exceeded().is_some());
    }

    #[test]
    fn pdf_dates_are_utc() {
        assert_eq!(pdf_date("D:20160930151958+02'00'").as_deref(), Some("2016-09-30T13:19:58Z"));
        assert_eq!(pdf_date("D:2016").as_deref(), Some("2016-01-01T00:00:00Z"));
    }
}
