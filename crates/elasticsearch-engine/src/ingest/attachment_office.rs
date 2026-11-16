//! Office and archive formats for the `attachment` processor (what Tika's
//! parsers give ES): OOXML (`docx`, `xlsx`, `pptx`), OpenDocument, EPUB and
//! other ZIP archives, and OLE2 compound files (Word 97-2003 `doc`).
//!
//! Each returns the extracted text with its content type and the document
//! properties ES maps (title, author, keywords, dates, ...), or why the
//! document cannot be read: a damaged container, an encrypted document, or
//! an OLE document of a kind with no text extraction here fails rather
//! than reporting a document without text.

use super::attachment::{Budget, ParseError, Text};
use std::io::Read;

/// Inflated bytes kept per archive entry (a decompression bomb fails; all
/// entries together are bounded by the document's [`Budget`]).
const MAX_ENTRY: u64 = 64 << 20;

/// What an extractor returns besides its text: the content type and the
/// document properties ES maps.
type Extracted = Result<(String, Vec<(&'static str, String)>), ParseError>;

// ------------------------------------------------------------ zip

struct Entry {
    name: String,
    method: u16,
    compressed: usize,
    local: usize,
}

pub(super) struct Zip<'a> {
    data: &'a [u8],
    entries: Vec<Entry>,
    budget: &'a Budget,
}

fn u16le(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32le(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn u64le(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

impl<'a> Zip<'a> {
    pub(super) fn open(data: &'a [u8], budget: &'a Budget) -> Result<Zip<'a>, String> {
        let bad = || "invalid ZIP archive".to_string();
        // End of central directory: the last signature within the comment span.
        let low = data.len().saturating_sub(22 + 65535);
        let eocd = (low..data.len().saturating_sub(21)).rev().find(|&i| data[i..].starts_with(b"PK\x05\x06")).ok_or_else(bad)?;
        let count = u16le(data, eocd + 10).ok_or_else(bad)? as usize;
        let mut at = u32le(data, eocd + 16).ok_or_else(bad)? as usize;
        if at == 0xFFFF_FFFF {
            return Err("ZIP64 archives are not supported".into());
        }
        let mut entries = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            if u32le(data, at) != Some(0x0201_4b50) {
                return Err(bad());
            }
            let method = u16le(data, at + 10).ok_or_else(bad)?;
            let compressed = u32le(data, at + 20).ok_or_else(bad)? as usize;
            let name_len = u16le(data, at + 28).ok_or_else(bad)? as usize;
            let extra = u16le(data, at + 30).ok_or_else(bad)? as usize;
            let comment = u16le(data, at + 32).ok_or_else(bad)? as usize;
            let local = u32le(data, at + 42).ok_or_else(bad)? as usize;
            let name = data.get(at + 46..at + 46 + name_len).ok_or_else(bad)?;
            entries.push(Entry { name: String::from_utf8_lossy(name).into_owned(), method, compressed, local });
            at += 46 + name_len + extra + comment;
        }
        Ok(Zip { data, entries, budget })
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    pub(super) fn has(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// An entry's bytes (`None`: no such entry).
    pub(super) fn read(&self, name: &str) -> Option<Result<Vec<u8>, String>> {
        let e = self.entries.iter().find(|e| e.name == name)?;
        Some((|| {
            let bad = || format!("invalid ZIP entry [{name}]");
            if u32le(self.data, e.local) != Some(0x0403_4b50) {
                return Err(bad());
            }
            let n = u16le(self.data, e.local + 26).ok_or_else(bad)? as usize;
            let x = u16le(self.data, e.local + 28).ok_or_else(bad)? as usize;
            let start = e.local + 30 + n + x;
            let raw = self.data.get(start..start + e.compressed).ok_or_else(bad)?;
            let mut out = Vec::new();
            match e.method {
                0 => out.extend_from_slice(raw),
                8 => {
                    flate2::read::DeflateDecoder::new(raw).take(self.budget.inflate_cap(MAX_ENTRY)).read_to_end(&mut out).map_err(|err| format!("invalid ZIP entry [{name}]: {err}"))?;
                }
                m => return Err(format!("unsupported compression method [{m}] in ZIP entry [{name}]")),
            }
            // Every entry read counts against the document's total.
            self.budget.inflate(out.len() as u64, MAX_ENTRY, &format!("ZIP entry [{name}]"))?;
            Ok(out)
        })())
    }

    fn text(&self, name: &str) -> Result<Option<String>, String> {
        match self.read(name) {
            None => Ok(None),
            Some(r) => r.map(|b| Some(String::from_utf8_lossy(&b).into_owned())),
        }
    }
}

// ------------------------------------------------------------ xml

#[derive(Debug, PartialEq)]
enum Tok<'a> {
    /// Local name (prefix removed), raw attributes, self-closing.
    Start(&'a str, &'a str, bool),
    End(&'a str),
    Text(String),
}

fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

pub(super) fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest.find(';').filter(|e| *e <= 12) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..end];
        let c = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => ent
                .strip_prefix("#x")
                .or_else(|| ent.strip_prefix("#X"))
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| ent.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        match c {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn xml(s: &str) -> Vec<Tok<'_>> {
    let mut out = Vec::new();
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        if b[i] != b'<' {
            let end = s[i..].find('<').map(|p| p + i).unwrap_or(s.len());
            out.push(Tok::Text(unescape(&s[i..end])));
            i = end;
            continue;
        }
        let rest = &s[i..];
        if let Some(body) = rest.strip_prefix("<![CDATA[") {
            let end = body.find("]]>").unwrap_or(body.len());
            out.push(Tok::Text(body[..end].to_string()));
            i += 9 + end + 3;
            continue;
        }
        if rest.starts_with("<!--") {
            i += rest.find("-->").map(|e| e + 3).unwrap_or(rest.len());
            continue;
        }
        if rest.starts_with("<?") || rest.starts_with("<!") {
            i += rest.find('>').map(|e| e + 1).unwrap_or(rest.len());
            continue;
        }
        // A tag; `>` inside quoted attribute values does not end it.
        let mut j = 1;
        let mut quote = None;
        while j < rest.len() {
            let c = rest.as_bytes()[j];
            match quote {
                Some(q) if c == q => quote = None,
                Some(_) => {}
                None if c == b'"' || c == b'\'' => quote = Some(c),
                None if c == b'>' => break,
                None => {}
            }
            j += 1;
        }
        let tag = &rest[1..j.min(rest.len())];
        i += j + 1;
        if let Some(name) = tag.strip_prefix('/') {
            out.push(Tok::End(local(name.trim())));
        } else {
            let empty = tag.ends_with('/');
            let tag = tag.trim_end_matches('/');
            let (name, attrs) = tag.split_once(|c: char| c.is_whitespace()).unwrap_or((tag, ""));
            out.push(Tok::Start(local(name), attrs, empty));
        }
    }
    out
}

fn attr(attrs: &str, key: &str) -> Option<String> {
    let mut rest = attrs;
    while let Some(eq) = rest.find('=') {
        let name = rest[..eq].trim();
        let after = rest[eq + 1..].trim_start();
        let q = after.chars().next()?;
        if q != '"' && q != '\'' {
            return None;
        }
        let end = after[1..].find(q)? + 1;
        if local(name) == key || name == key {
            return Some(unescape(&after[1..end]));
        }
        rest = &after[end + 1..];
    }
    None
}

/// Text of the elements named in `props` (local names) of a metadata
/// document, mapped to ES attachment properties. When several elements map
/// to one property, the one listed first in `props` wins (whatever the
/// document order).
fn properties(doc: &str, props: &[(&str, &'static str)], meta: &mut Vec<(&'static str, String)>) {
    let mut current: Option<&str> = None;
    let mut text = String::new();
    let mut found: Vec<(&'static str, String, usize)> = Vec::new();
    for t in xml(doc) {
        match t {
            Tok::Start(name, _, false) => {
                current = Some(name);
                text.clear();
            }
            Tok::Text(s) if current.is_some() => text.push_str(&s),
            Tok::End(name) if current == Some(name) => {
                if let Some(rank) = props.iter().position(|(n, _)| *n == name) {
                    let (prop, v) = (props[rank].1, text.trim());
                    if !v.is_empty() && !meta.iter().any(|(p, _)| *p == prop) {
                        match found.iter_mut().find(|(p, _, _)| *p == prop) {
                            Some(f) if f.2 > rank => *f = (prop, v.to_string(), rank),
                            Some(_) => {}
                            None => found.push((prop, v.to_string(), rank)),
                        }
                    }
                }
                current = None;
            }
            _ => {}
        }
    }
    meta.extend(found.into_iter().map(|(p, v, _)| (p, v)));
}

/// OOXML `docProps/core.xml` (Tika's OOXML metadata as ES maps it: the
/// subject is the keywords when the document has none).
const CORE_PROPERTIES: &[(&str, &str)] = &[
    ("title", "title"),
    ("creator", "author"),
    ("keywords", "keywords"),
    ("subject", "keywords"),
    ("description", "description"),
    ("lastModifiedBy", "modifier"),
    ("created", "date"),
    ("modified", "modified"),
    ("identifier", "identifier"),
    ("lastPrinted", "print_date"),
];

/// Text of an OOXML part: `text` element contents, `p` ends a line,
/// `tab` / `br` / `cr` as such; deleted text and field instructions are not
/// document text.
fn ooxml_text(part: &str, text_tag: &str, para_tag: &str, out: &mut Text) {
    let mut skip = 0usize;
    let mut in_text = false;
    for t in xml(part) {
        if out.full() {
            return;
        }
        match t {
            // Deleted text, field instructions, and tab stop definitions
            // (`w:tabs` / `w:tab` in paragraph properties) are not text.
            Tok::Start(n, _, empty) if matches!(n, "delText" | "instrText" | "tabs") && !empty => skip += 1,
            Tok::End(n) if matches!(n, "delText" | "instrText" | "tabs") => skip = skip.saturating_sub(1),
            Tok::Start(n, _, empty) if n == text_tag && !empty => in_text = true,
            Tok::End(n) if n == text_tag => in_text = false,
            Tok::Start("tab", _, _) if skip == 0 => out.push('\t'),
            Tok::Start("br" | "cr", _, _) if skip == 0 => out.push('\n'),
            Tok::End(n) if n == para_tag => out.push('\n'),
            Tok::Text(s) if in_text && skip == 0 => out.push_str(&s),
            _ => {}
        }
    }
}

/// Numbered parts (`prefix<N>.xml`) in number order.
fn numbered<'z>(zip: &'z Zip, prefix: &str) -> Vec<&'z str> {
    let mut parts: Vec<(u64, &str)> = zip
        .names()
        .filter_map(|n| n.strip_prefix(prefix).and_then(|r| r.strip_suffix(".xml")).and_then(|d| d.parse().ok()).map(|k| (k, n)))
        .collect();
    parts.sort();
    parts.into_iter().map(|(_, n)| n).collect()
}

fn core_properties(zip: &Zip) -> Result<Vec<(&'static str, String)>, String> {
    let mut meta = Vec::new();
    if let Some(core) = zip.text("docProps/core.xml")? {
        properties(&core, CORE_PROPERTIES, &mut meta);
    }
    Ok(meta)
}

fn docx(zip: &Zip, out: &mut Text) -> Extracted {
    let meta = core_properties(zip)?;
    let body = zip.text("word/document.xml")?.ok_or("missing word/document.xml")?;
    ooxml_text(&body, "t", "p", out);
    // Tika ends the document body with a newline of its own.
    out.push('\n');
    Ok(("application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(), meta))
}

fn pptx(zip: &Zip, out: &mut Text) -> Extracted {
    let meta = core_properties(zip)?;
    for slide in numbered(zip, "ppt/slides/slide") {
        if out.full() {
            break;
        }
        let part = zip.text(slide)?.unwrap_or_default();
        ooxml_text(&part, "t", "p", out);
    }
    Ok(("application/vnd.openxmlformats-officedocument.presentationml.presentation".into(), meta))
}

fn xlsx(zip: &Zip, out: &mut Text) -> Extracted {
    let meta = core_properties(zip)?;
    // Shared strings: each `si`'s text runs. Cells refer to them by index;
    // a reference is written straight into the budgeted text, never copied.
    let mut shared: Vec<String> = Vec::new();
    if let Some(sst) = zip.text("xl/sharedStrings.xml")? {
        let mut cur: Option<String> = None;
        let mut in_t = false;
        for t in xml(&sst) {
            match t {
                Tok::Start("si", _, false) => cur = Some(String::new()),
                Tok::Start("si", _, true) => shared.push(String::new()),
                Tok::End("si") => shared.push(cur.take().unwrap_or_default()),
                Tok::Start("t", _, false) => in_t = true,
                Tok::End("t") => in_t = false,
                Tok::Text(s) if in_t => {
                    if let Some(c) = cur.as_mut() {
                        c.push_str(&s);
                    }
                }
                _ => {}
            }
        }
    }
    let names: Vec<String> = zip
        .text("xl/workbook.xml")?
        .map(|wb| xml(&wb).into_iter().filter_map(|t| if let Tok::Start("sheet", a, _) = t { attr(a, "name") } else { None }).collect())
        .unwrap_or_default();
    'sheets: for (k, sheet) in numbered(zip, "xl/worksheets/sheet").into_iter().enumerate() {
        if out.full() {
            break;
        }
        if let Some(name) = names.get(k) {
            out.push_str(name);
            out.push('\n');
        }
        let part = zip.text(sheet)?.unwrap_or_default();
        // Per row: the cell position, the last written cell's position (the
        // tabs before a cell are written with it; empty cells at a row's end
        // write nothing), whether the row wrote anything.
        let (mut cell, mut written, mut any) = (0usize, 0usize, false);
        let (mut kind, mut value, mut in_v) = (String::new(), String::new(), false);
        for t in xml(&part) {
            if out.full() {
                break 'sheets;
            }
            match t {
                Tok::Start("row", _, _) => (cell, written, any) = (0, 0, false),
                Tok::End("row") => {
                    if any {
                        out.push('\n');
                    }
                    (cell, written, any) = (0, 0, false);
                }
                Tok::Start("c", a, empty) => {
                    kind = attr(a, "t").unwrap_or_default();
                    value.clear();
                    if empty {
                        cell += 1;
                    }
                }
                Tok::End("c") => {
                    let v: &str = if kind == "s" { value.trim().parse::<usize>().ok().and_then(|i| shared.get(i)).map(String::as_str).unwrap_or("") } else { &value };
                    if !v.is_empty() {
                        while written < cell && !out.full() {
                            out.push('\t');
                            written += 1;
                        }
                        out.push_str(v);
                        any = true;
                    }
                    cell += 1;
                }
                Tok::Start("v" | "t", _, false) => in_v = true,
                Tok::End("v" | "t") => in_v = false,
                Tok::Text(s) if in_v => value.push_str(&s),
                _ => {}
            }
        }
        out.push('\n');
    }
    Ok(("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".into(), meta))
}

fn opendocument(zip: &Zip, mime: String, out: &mut Text) -> Extracted {
    let mut meta = Vec::new();
    if let Some(m) = zip.text("meta.xml")? {
        properties(
            &m,
            &[("title", "title"), ("initial-creator", "author"), ("creator", "modifier"), ("keyword", "keywords"), ("subject", "keywords"), ("description", "description"), ("creation-date", "date"), ("date", "modified")],
            &mut meta,
        );
    }
    let body = zip.text("content.xml")?.ok_or("missing content.xml")?;
    let mut in_body = false;
    for t in xml(&body) {
        if out.full() {
            break;
        }
        match t {
            Tok::Start("body", _, _) => in_body = true,
            Tok::Start("s", a, _) if in_body => {
                for _ in 0..attr(a, "c").and_then(|c| c.parse().ok()).unwrap_or(1usize).min(1000) {
                    out.push(' ');
                }
            }
            Tok::Start("tab", _, _) if in_body => out.push('\t'),
            Tok::Start("line-break", _, _) if in_body => out.push('\n'),
            Tok::End("p" | "h") if in_body => out.push('\n'),
            Tok::Text(s) if in_body => out.push_str(&s),
            _ => {}
        }
    }
    Ok((mime, meta))
}

fn epub(zip: &Zip, out: &mut Text) -> Extracted {
    // The package document (`META-INF/container.xml` names it) orders the
    // content documents (spine) and holds the Dublin Core metadata.
    let opf_path = zip
        .text("META-INF/container.xml")?
        .and_then(|c| xml(&c).into_iter().find_map(|t| if let Tok::Start("rootfile", a, _) = t { attr(a, "full-path") } else { None }));
    let mut order: Vec<String> = Vec::new();
    let mut meta = Vec::new();
    if let Some(path) = opf_path {
        if let Some(opf) = zip.text(&path)? {
            let base = path.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
            let toks = xml(&opf);
            let manifest: Vec<(String, String)> = toks.iter().filter_map(|t| if let Tok::Start("item", a, _) = t { Some((attr(a, "id")?, attr(a, "href")?)) } else { None }).collect();
            for t in &toks {
                if let Tok::Start("itemref", a, _) = t {
                    if let Some(href) = attr(a, "idref").and_then(|id| manifest.iter().find(|(i, _)| *i == id).map(|(_, h)| h.clone())) {
                        order.push(format!("{base}{href}"));
                    }
                }
            }
            properties(
                &opf,
                &[("title", "title"), ("creator", "author"), ("subject", "keywords"), ("description", "description"), ("publisher", "publisher"), ("contributor", "contributor"), ("date", "date"), ("identifier", "identifier"), ("type", "type"), ("format", "format"), ("coverage", "coverage"), ("rights", "rights"), ("source", "source"), ("relation", "relation")],
                &mut meta,
            );
        }
    }
    if order.is_empty() {
        order = zip.names().filter(|n| n.ends_with(".xhtml") || n.ends_with(".html") || n.ends_with(".htm")).map(str::to_string).collect();
    }
    for part in order {
        if out.full() {
            break;
        }
        if let Some(html) = zip.text(&part)? {
            let lower = html.to_ascii_lowercase();
            let body = match (lower.find("<body"), lower.rfind("</body>")) {
                (Some(a), Some(b)) if a < b => &html[a..b],
                _ => &html[..],
            };
            out.push_str(super::util::html_text(body).trim());
            out.push('\n');
        }
    }
    Ok(("application/epub+zip".into(), meta))
}

/// Any other archive: each entry's name, and its text when it is a
/// document with any (Tika's `PackageParser`). An entry the archive cannot
/// give (a damaged entry, an unsupported compression) fails the document,
/// as Tika's read error does. An embedded document that is itself damaged
/// or encrypted is skipped, as Tika's `ParsingEmbeddedDocumentExtractor`
/// records and skips its `TikaException`; an embedded document of a kind
/// with no extraction here, or one over a bound, fails the document.
fn archive(zip: &Zip, depth: usize, out: &mut Text) -> Extracted {
    let names: Vec<&str> = zip.names().filter(|n| !n.ends_with('/')).collect();
    for name in names {
        if out.full() {
            break;
        }
        out.push_str(name);
        out.push('\n');
        let bytes = match zip.read(name) {
            Some(Ok(bytes)) => bytes,
            Some(Err(e)) => return Err(ParseError::Fatal(e)),
            None => continue,
        };
        let before = out.len();
        match super::attachment::parse_into(&bytes, depth + 1, out) {
            Ok(_) => {}
            Err(ParseError::Damaged(_)) if out.budget().exceeded().is_none() => {}
            Err(e) => return Err(e),
        }
        if out.len() > before && !out.as_str().ends_with('\n') {
            out.push('\n');
        }
    }
    Ok(("application/zip".into(), Vec::new()))
}

/// A ZIP-based document or archive.
pub(super) fn zip_document(data: &[u8], depth: usize, out: &mut Text) -> Extracted {
    let zip = Zip::open(data, out.budget())?;
    if zip.has("word/document.xml") {
        return docx(&zip, out);
    }
    if zip.has("xl/workbook.xml") {
        return xlsx(&zip, out);
    }
    if zip.has("ppt/presentation.xml") {
        return pptx(&zip, out);
    }
    if let Some(mime) = zip.text("mimetype")?.map(|m| m.trim().to_string()) {
        if mime == "application/epub+zip" {
            return epub(&zip, out);
        }
        if mime.starts_with("application/vnd.oasis.opendocument.") {
            return opendocument(&zip, mime, out);
        }
    }
    if zip.has("[Content_Types].xml") && zip.names().any(|n| n.starts_with("visio/")) {
        return Err(ParseError::Fatal("Visio documents are not supported".into()));
    }
    archive(&zip, depth, out)
}

// ------------------------------------------------------------ OLE2

const FREE: u32 = 0xFFFF_FFFF;
const END_OF_CHAIN: u32 = 0xFFFF_FFFE;

struct DirEntry {
    name: String,
    kind: u8,
    start: u32,
    size: u64,
}

struct Cfb<'a> {
    data: &'a [u8],
    sector: usize,
    fat: Vec<u32>,
    minifat: Vec<u32>,
    mini: Vec<u8>,
    cutoff: u64,
    entries: Vec<DirEntry>,
}

impl<'a> Cfb<'a> {
    fn open(data: &'a [u8]) -> Result<Cfb<'a>, String> {
        let bad = || "invalid OLE2 compound document".to_string();
        if !data.starts_with(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") || data.len() < 512 {
            return Err(bad());
        }
        let shift = u16le(data, 0x1E).ok_or_else(bad)?;
        if !(7..=16).contains(&shift) {
            return Err(bad());
        }
        let sector = 1usize << shift;
        let fat_sectors = u32le(data, 0x2C).ok_or_else(bad)? as usize;
        let first_dir = u32le(data, 0x30).ok_or_else(bad)?;
        let cutoff = u32le(data, 0x38).ok_or_else(bad)? as u64;
        let first_minifat = u32le(data, 0x3C).ok_or_else(bad)?;
        let mut difat_next = u32le(data, 0x44).ok_or_else(bad)?;
        let mut fat_list: Vec<u32> = (0..109).filter_map(|k| u32le(data, 0x4C + 4 * k)).filter(|s| *s != FREE).collect();
        let per = sector / 4;
        let mut guard = 0;
        while difat_next != END_OF_CHAIN && difat_next != FREE && guard < 1 << 16 {
            let off = (difat_next as usize + 1) * sector;
            let block = data.get(off..off + sector).ok_or_else(bad)?;
            fat_list.extend((0..per - 1).filter_map(|k| u32le(block, 4 * k)).filter(|s| *s != FREE));
            difat_next = u32le(block, sector - 4).ok_or_else(bad)?;
            guard += 1;
        }
        fat_list.truncate(fat_sectors);
        let mut fat = Vec::with_capacity(fat_list.len() * per);
        for s in &fat_list {
            let off = (*s as usize + 1) * sector;
            let block = data.get(off..off + sector).ok_or_else(bad)?;
            fat.extend((0..per).filter_map(|k| u32le(block, 4 * k)));
        }
        let mut cfb = Cfb { data, sector, fat, minifat: Vec::new(), mini: Vec::new(), cutoff, entries: Vec::new() };
        let dir = cfb.chain(first_dir)?;
        for e in dir.chunks(128) {
            if e.len() < 128 {
                break;
            }
            let len = (u16le(e, 0x40).unwrap_or(0) as usize).min(64);
            let units: Vec<u16> = e[..len].chunks(2).map(|p| u16::from_le_bytes([p[0], *p.get(1).unwrap_or(&0)])).take_while(|u| *u != 0).collect();
            // Version 3 files (512-byte sectors) may leave the high half of
            // the size uninitialised.
            let size = u64le(e, 0x78).unwrap_or(0);
            let size = if sector == 512 { size & 0xFFFF_FFFF } else { size };
            cfb.entries.push(DirEntry { name: String::from_utf16_lossy(&units), kind: e[0x42], start: u32le(e, 0x74).unwrap_or(END_OF_CHAIN), size });
        }
        let root = cfb.entries.first().filter(|e| e.kind == 5).ok_or_else(bad)?;
        let (root_start, root_size) = (root.start, root.size);
        let mut mini = cfb.chain(root_start)?;
        mini.truncate(root_size as usize);
        cfb.mini = mini;
        let minifat = cfb.chain(first_minifat).unwrap_or_default();
        cfb.minifat = minifat.chunks(4).filter_map(|p| u32le(p, 0)).collect();
        Ok(cfb)
    }

    /// The bytes of a regular sector chain.
    fn chain(&self, mut s: u32) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        let mut steps = 0;
        while s != END_OF_CHAIN && s != FREE {
            if steps > self.fat.len() {
                return Err("cyclic sector chain in OLE2 document".into());
            }
            let off = (s as usize + 1) * self.sector;
            out.extend_from_slice(self.data.get(off..off + self.sector).ok_or("truncated OLE2 document")?);
            s = *self.fat.get(s as usize).ok_or("invalid sector in OLE2 document")?;
            steps += 1;
        }
        Ok(out)
    }

    fn mini_chain(&self, mut s: u32) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        let mut steps = 0;
        while s != END_OF_CHAIN && s != FREE {
            if steps > self.minifat.len() {
                return Err("cyclic mini sector chain in OLE2 document".into());
            }
            let off = s as usize * 64;
            out.extend_from_slice(self.mini.get(off..off + 64).ok_or("truncated OLE2 mini stream")?);
            s = *self.minifat.get(s as usize).ok_or("invalid mini sector in OLE2 document")?;
            steps += 1;
        }
        Ok(out)
    }

    fn stream(&self, name: &str) -> Option<Result<Vec<u8>, String>> {
        let e = self.entries.iter().find(|e| e.kind == 2 && e.name.eq_ignore_ascii_case(name))?;
        Some((|| {
            let mut b = if e.size < self.cutoff { self.mini_chain(e.start)? } else { self.chain(e.start)? };
            if (b.len() as u64) < e.size {
                return Err(format!("truncated stream [{name}] in OLE2 document"));
            }
            b.truncate(e.size as usize);
            Ok(b)
        })())
    }
}

/// `\x05SummaryInformation` properties ES maps (Tika's `SummaryExtractor`).
fn summary_information(b: &[u8], meta: &mut Vec<(&'static str, String)>) {
    let Some(section) = u32le(b, 44).map(|v| v as usize) else { return };
    let Some(count) = u32le(b, section + 4) else { return };
    let mut codepage = 1252u32;
    let mut props = Vec::new();
    for k in 0..count.min(256) as usize {
        let (Some(id), Some(off)) = (u32le(b, section + 8 + 8 * k), u32le(b, section + 12 + 8 * k)) else { break };
        props.push((id, section + off as usize));
    }
    if let Some((_, at)) = props.iter().find(|(id, _)| *id == 1) {
        if u32le(b, *at) == Some(2) {
            codepage = u16le(b, at + 4).unwrap_or(1252) as u32;
        }
    }
    let string = |at: usize| -> Option<String> {
        match u32le(b, at)? {
            0x1E => {
                let len = u32le(b, at + 4)? as usize;
                let raw = b.get(at + 8..at + 8 + len)?;
                let raw = raw.split(|c| *c == 0).next().unwrap_or(raw);
                Some(match codepage {
                    65001 => String::from_utf8_lossy(raw).into_owned(),
                    1200 => String::from_utf16_lossy(&raw.chunks(2).map(|p| u16::from_le_bytes([p[0], *p.get(1).unwrap_or(&0)])).collect::<Vec<_>>()),
                    _ => raw.iter().map(|&c| super::attachment_pdf::cp1252(c)).collect(),
                })
            }
            0x1F => {
                let len = u32le(b, at + 4)? as usize;
                let raw = b.get(at + 8..at + 8 + 2 * len)?;
                let units: Vec<u16> = raw.chunks(2).map(|p| u16::from_le_bytes([p[0], *p.get(1).unwrap_or(&0)])).take_while(|u| *u != 0).collect();
                Some(String::from_utf16_lossy(&units))
            }
            _ => None,
        }
    };
    let date = |at: usize| -> Option<String> {
        if u32le(b, at)? != 0x40 {
            return None;
        }
        let ft = u64le(b, at + 4)?;
        if ft == 0 {
            return None;
        }
        let secs = (ft / 10_000_000) as i64 - 11_644_473_600;
        chrono::DateTime::from_timestamp(secs, 0).map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
    };
    let mut put = |prop: &'static str, v: Option<String>| {
        if let Some(v) = v.map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) {
            if !meta.iter().any(|(p, _)| *p == prop) {
                meta.push((prop, v));
            }
        }
    };
    let get = |id: u32| props.iter().find(|(i, _)| *i == id).map(|(_, at)| *at);
    put("title", get(2).and_then(string));
    put("author", get(4).and_then(string));
    // The subject is the keywords when the document has none.
    put("keywords", get(5).and_then(string));
    put("keywords", get(3).and_then(string));
    put("comments", get(6).and_then(string));
    put("modifier", get(8).and_then(string));
    put("date", get(12).and_then(date));
    put("modified", get(13).and_then(date));
}

/// Word 97-2003 text (`WordDocument` stream and its piece table).
fn word_text(cfb: &Cfb, word: &[u8], out: &mut Text) -> Result<(), String> {
    let bad = || "invalid Word document".to_string();
    if u16le(word, 0) != Some(0xA5EC) {
        return Err(bad());
    }
    let flags = u16le(word, 0x0A).ok_or_else(bad)?;
    if flags & 0x0100 != 0 {
        return Err("the document is encrypted".into());
    }
    let table_name = if flags & 0x0200 != 0 { "1Table" } else { "0Table" };
    // FIB: FibBase (32 bytes), csw + fibRgW, cslw + fibRgLw, cbRgFcLcb + fibRgFcLcb.
    let csw = u16le(word, 32).ok_or_else(bad)? as usize;
    let lw = 34 + 2 * csw;
    let cslw = u16le(word, lw).ok_or_else(bad)? as usize;
    let ccp_text = u32le(word, lw + 2 + 4 * 3).ok_or_else(bad)? as usize;
    let fclcb = lw + 2 + 4 * cslw + 2;
    let fc_clx = u32le(word, fclcb + 8 * 33).ok_or_else(bad)? as usize;
    let lcb_clx = u32le(word, fclcb + 8 * 33 + 4).ok_or_else(bad)? as usize;
    let mut text: Vec<u16> = Vec::new();
    let table = cfb.stream(table_name).transpose()?;
    match table.as_deref().and_then(|t| t.get(fc_clx..fc_clx + lcb_clx)).filter(|_| lcb_clx > 0) {
        Some(clx) => {
            // Skip Prc entries to the Pcdt.
            let mut i = 0;
            while clx.get(i) == Some(&1) {
                i += 3 + u16le(clx, i + 1).ok_or_else(bad)? as usize;
            }
            if clx.get(i) != Some(&2) {
                return Err(bad());
            }
            let lcb = u32le(clx, i + 1).ok_or_else(bad)? as usize;
            let plc = clx.get(i + 5..i + 5 + lcb).ok_or_else(bad)?;
            let n = lcb.saturating_sub(4) / 12;
            for k in 0..n {
                let (cp0, cp1) = (u32le(plc, 4 * k).ok_or_else(bad)? as usize, u32le(plc, 4 * k + 4).ok_or_else(bad)? as usize);
                if cp0 >= ccp_text {
                    break;
                }
                let chars = cp1.min(ccp_text).saturating_sub(cp0);
                let fc = u32le(plc, 4 * (n + 1) + 8 * k + 2).ok_or_else(bad)?;
                if fc & 0x4000_0000 != 0 {
                    let at = ((fc & !0x4000_0000) / 2) as usize;
                    let raw = word.get(at..at + chars).ok_or_else(bad)?;
                    text.extend(raw.iter().map(|&c| super::attachment_pdf::cp1252(c) as u16));
                } else {
                    let at = fc as usize;
                    let raw = word.get(at..at + 2 * chars).ok_or_else(bad)?;
                    text.extend(raw.chunks(2).map(|p| u16::from_le_bytes([p[0], p[1]])));
                }
            }
        }
        None => {
            // No piece table: the text is one 8-bit run from fcMin.
            let fc_min = u32le(word, 0x18).ok_or_else(bad)? as usize;
            let raw = word.get(fc_min..fc_min + ccp_text).ok_or_else(bad)?;
            text.extend(raw.iter().map(|&c| super::attachment_pdf::cp1252(c) as u16));
        }
    }
    // Paragraph marks end lines; fields show their result, not their code.
    let mut fields: Vec<bool> = Vec::new();
    for c in char::decode_utf16(text.iter().copied()).map(|r| r.unwrap_or('\u{FFFD}')) {
        if out.full() {
            break;
        }
        match c {
            '\u{13}' => fields.push(false),
            '\u{14}' => {
                if let Some(f) = fields.last_mut() {
                    *f = true;
                }
            }
            '\u{15}' => {
                fields.pop();
            }
            _ if fields.last() == Some(&false) => {}
            '\r' | '\u{0B}' | '\u{0C}' | '\u{0E}' => out.push('\n'),
            '\u{07}' => out.push('\t'),
            '\u{1E}' => out.push('-'),
            '\t' => out.push('\t'),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    Ok(())
}

/// An OLE2 compound document.
pub(super) fn ole_document(data: &[u8], out: &mut Text) -> Extracted {
    let cfb = Cfb::open(data)?;
    let mut meta = Vec::new();
    if let Some(Ok(si)) = cfb.stream("\u{5}SummaryInformation") {
        summary_information(&si, &mut meta);
    }
    match cfb.stream("WordDocument") {
        Some(word) => {
            word_text(&cfb, &word?, out)?;
            Ok(("application/msword".into(), meta))
        }
        None if cfb.stream("EncryptedPackage").is_some() => Err("the document is encrypted".into()),
        None if cfb.stream("Workbook").is_some() || cfb.stream("Book").is_some() => Err(ParseError::Fatal("Excel 97-2003 documents are not supported".into())),
        None if cfb.stream("PowerPoint Document").is_some() => Err(ParseError::Fatal("PowerPoint 97-2003 documents are not supported".into())),
        None => Err(ParseError::Fatal("OLE2 documents of this kind are not supported".into())),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::io::Write;

    /// Content, content type and properties of a ZIP-based document.
    fn run(data: &[u8]) -> Result<(String, String, Vec<(&'static str, String)>), ParseError> {
        let budget = Budget::new(-1);
        let mut out = Text::new(&budget);
        let (t, m) = zip_document(data, 0, &mut out)?;
        Ok((out.into_string(), t, m))
    }

    /// A stored-or-deflated ZIP of `(name, bytes)`.
    pub(crate) fn zip(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, body) in files {
            let mut z = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            z.write_all(body).unwrap();
            let comp = z.finish().unwrap();
            let crc = crc32fast::hash(body);
            let local = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0, 8, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&comp);
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            central.extend_from_slice(&(body.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&local.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let at = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&at.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    #[test]
    fn docx_text_and_core_properties() {
        let doc = br#"<?xml version="1.0"?><w:document xmlns:w="x"><w:body><w:p><w:r><w:t>Hello &amp; </w:t></w:r><w:r><w:t xml:space="preserve">world</w:t></w:r></w:p><w:p><w:r><w:instrText>PAGE</w:instrText><w:t>Second</w:t><w:tab/><w:t>line</w:t></w:r></w:p></w:body></w:document>"#;
        let core = br#"<cp:coreProperties xmlns:cp="a" xmlns:dc="b" xmlns:dcterms="c"><dc:title></dc:title><dc:creator>David Pilato</dc:creator><cp:lastModifiedBy>David Pilato</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">2016-03-10T08:24:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2016-03-10T08:24:00Z</dcterms:modified></cp:coreProperties>"#;
        let z = zip(&[("[Content_Types].xml", b"<Types/>"), ("word/document.xml", doc), ("docProps/core.xml", core)]);
        let (content, content_type, meta) = run(&z).unwrap();
        assert_eq!(content, "Hello & world\nSecond\tline\n\n");
        assert_eq!(content_type, "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        assert_eq!(meta, vec![("author", "David Pilato".to_string()), ("modifier", "David Pilato".to_string()), ("date", "2016-03-10T08:24:00Z".to_string()), ("modified", "2016-03-10T08:24:00Z".to_string())]);
    }

    #[test]
    fn xlsx_pptx_and_archives() {
        let x = zip(&[
            ("xl/workbook.xml", br#"<workbook><sheets><sheet name="Data" sheetId="1"/></sheets></workbook>"#),
            ("xl/sharedStrings.xml", br#"<sst><si><t>name</t></si><si><r><t>ali</t></r><r><t>ce</t></r></si></sst>"#),
            ("xl/worksheets/sheet1.xml", br#"<worksheet><sheetData><row><c t="s"><v>0</v></c><c><v>42</v></c></row><row><c t="s"><v>1</v></c><c t="inlineStr"><is><t>x</t></is></c></row></sheetData></worksheet>"#),
        ]);
        assert_eq!(run(&x).unwrap().0, "Data\nname\t42\nalice\tx\n\n");
        let p = zip(&[("ppt/presentation.xml", b"<p/>"), ("ppt/slides/slide2.xml", b"<p:sld><a:p><a:r><a:t>two</a:t></a:r></a:p></p:sld>"), ("ppt/slides/slide1.xml", b"<p:sld><a:p><a:r><a:t>one</a:t></a:r></a:p></p:sld>")]);
        assert_eq!(run(&p).unwrap().0, "one\ntwo\n");
        let a = zip(&[("notes.txt", b"plain words")]);
        let (content, content_type, _) = run(&a).unwrap();
        assert_eq!((content.as_str(), content_type.as_str()), ("notes.txt\nplain words\n", "application/zip"));
        assert!(run(b"PK\x03\x04 truncated garbage").is_err());
    }

    #[test]
    fn xml_entities_and_attributes() {
        assert_eq!(unescape("a &lt;b&gt; &#233;&#x41; &bogus; &"), "a <b> éA &bogus; &");
        assert_eq!(attr(r#" r:id="rId1" name='Sheet &amp; 1'"#, "name").as_deref(), Some("Sheet & 1"));
    }

    #[test]
    fn ole_rejects_damage_and_unsupported_kinds() {
        let budget = Budget::new(-1);
        assert!(ole_document(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1 short", &mut Text::new(&budget)).is_err());
    }
}
