//! `attachment` processor (ES `ingest-attachment` module).

use super::doc::IngestDocument;
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};

/// ES `AttachmentProcessor.Property`, in enum order.
const PROPERTIES: &[&str] = &[
    "CONTENT",
    "TITLE",
    "AUTHOR",
    "KEYWORDS",
    "DATE",
    "CONTENT_TYPE",
    "CONTENT_LENGTH",
    "LANGUAGE",
    "MODIFIED",
    "FORMAT",
    "IDENTIFIER",
    "CONTRIBUTOR",
    "COVERAGE",
    "MODIFIER",
    "CREATOR_TOOL",
    "PUBLISHER",
    "RELATION",
    "RIGHTS",
    "SOURCE",
    "TYPE",
    "DESCRIPTION",
    "PRINT_DATE",
    "METADATA_DATE",
    "LATITUDE",
    "LONGITUDE",
    "ALTITUDE",
    "RATING",
    "COMMENTS",
];

const NUMBER_OF_CHARS_INDEXED: i64 = 100_000;

pub struct AttachmentP {
    field: String,
    target: String,
    properties: Vec<&'static str>,
    indexed_chars: i64,
    indexed_chars_field: Option<String>,
    resource_name: Option<String>,
    ignore_missing: bool,
    remove_binary: bool,
}

#[derive(Debug)]
pub(super) struct Parsed {
    pub content: String,
    pub content_type: String,
    /// Document properties ES maps (`title`, `author`, `date`, ...).
    pub meta: Vec<(&'static str, String)>,
}

fn magic_type(b: &[u8]) -> Option<&'static str> {
    let starts = |m: &[u8]| b.starts_with(m);
    Some(if starts(b"%PDF") {
        "application/pdf"
    } else if starts(b"\x89PNG") {
        "image/png"
    } else if starts(b"\xFF\xD8\xFF") {
        "image/jpeg"
    } else if starts(b"GIF8") {
        "image/gif"
    } else if starts(b"PK\x03\x04") {
        "application/zip"
    } else if starts(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") {
        "application/x-tika-msoffice"
    } else if starts(b"\x1F\x8B") {
        "application/gzip"
    } else {
        return None;
    })
}

fn strip_rtf(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    let mut depth_skip: Vec<bool> = vec![false];
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                let cur = *depth_skip.last().unwrap_or(&false);
                depth_skip.push(cur);
            }
            '}' => {
                depth_skip.pop();
            }
            '\\' => {
                let mut word = String::new();
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_alphabetic() {
                        word.push(n);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if word.is_empty() {
                    if let Some(n) = chars.next() {
                        if n == '*' {
                            if let Some(l) = depth_skip.last_mut() {
                                *l = true;
                            }
                        } else if !n.is_ascii_alphanumeric() && n != '\n' && n != '\r' {
                            out.push(n);
                        }
                    }
                    continue;
                }
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_digit() || n == '-' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                if chars.peek() == Some(&' ') {
                    chars.next();
                }
                if matches!(word.as_str(), "fonttbl" | "colortbl" | "stylesheet" | "info" | "pict") {
                    if let Some(l) = depth_skip.last_mut() {
                        *l = true;
                    }
                }
                if matches!(word.as_str(), "par" | "line") && !*depth_skip.last().unwrap_or(&false) {
                    out.push('\n');
                }
            }
            '\r' | '\n' => {}
            c => {
                if !*depth_skip.last().unwrap_or(&false) {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// Archives nested deeper than this are not opened.
const MAX_NESTING: usize = 3;

/// Text an extraction may produce when `indexed_chars` is `-1` (unlimited
/// in ES) or above this: a document with more fails rather than holding an
/// unbounded string.
pub(super) const MAX_TEXT_CHARS: usize = 16 << 20;
/// Bytes all decompression of one document may produce together (archive
/// entries, PDF streams, gzip, nested archives): Tika's zip bomb guard.
pub(super) const MAX_INFLATED: u64 = 256 << 20;

/// Why a document cannot be extracted.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ParseError {
    /// The document is damaged or encrypted (Tika's `TikaException`). An
    /// embedded document like this is skipped, as Tika's
    /// `ParsingEmbeddedDocumentExtractor` records and skips it.
    Damaged(String),
    /// Fails the whole document even when it comes from an embedded one:
    /// an extraction bound was exceeded, the container itself could not be
    /// read, or the format has no text extraction here (skipping it would
    /// silently drop text Tika extracts).
    Fatal(String),
}

impl ParseError {
    pub(super) fn message(&self) -> &str {
        match self {
            ParseError::Damaged(m) | ParseError::Fatal(m) => m,
        }
    }
}

impl From<String> for ParseError {
    fn from(m: String) -> Self {
        ParseError::Damaged(m)
    }
}

impl From<&str> for ParseError {
    fn from(m: &str) -> Self {
        ParseError::Damaged(m.to_string())
    }
}

/// The work one document's extraction may do: the characters it may emit
/// (`indexed_chars`, Tika's write limit) and the bytes all its
/// decompression may produce. Enforced while extracting, before any text is
/// copied or emitted.
pub(super) struct Budget {
    chars_left: std::cell::Cell<usize>,
    /// Reaching the character budget fails the document (`indexed_chars`
    /// `-1` or above [`MAX_TEXT_CHARS`]) instead of truncating it.
    hard: bool,
    inflate_left: std::cell::Cell<u64>,
    exceeded: std::cell::RefCell<Option<String>>,
}

impl Budget {
    pub(super) fn new(indexed_chars: i64) -> Budget {
        Budget::with_limits(indexed_chars, MAX_INFLATED)
    }

    pub(super) fn with_limits(indexed_chars: i64, inflate: u64) -> Budget {
        let (chars, hard) = match usize::try_from(indexed_chars) {
            Ok(n) if n <= MAX_TEXT_CHARS => (n, false),
            _ => (MAX_TEXT_CHARS, true),
        };
        Budget { chars_left: std::cell::Cell::new(chars), hard, inflate_left: std::cell::Cell::new(inflate), exceeded: Default::default() }
    }

    /// The bound that failed the extraction, if one did.
    pub(super) fn exceeded(&self) -> Option<String> {
        self.exceeded.borrow().clone()
    }

    fn fail(&self, why: String) {
        let mut e = self.exceeded.borrow_mut();
        if e.is_none() {
            *e = Some(why);
        }
    }

    /// At most how many decompressed bytes a reader may produce now
    /// (`per`: the reader's own limit); read one more to detect excess.
    pub(super) fn inflate_cap(&self, per: u64) -> u64 {
        per.min(self.inflate_left.get()).saturating_add(1)
    }

    /// Charge `n` decompressed bytes (`per`: the reader's own limit).
    pub(super) fn inflate(&self, n: u64, per: u64, what: &str) -> std::result::Result<(), String> {
        if n > per {
            return Err(format!("{what} is too large"));
        }
        let left = self.inflate_left.get();
        if n > left {
            let why = format!("decompressed content exceeds the limit of [{MAX_INFLATED}] bytes per document");
            self.fail(why.clone());
            return Err(why);
        }
        self.inflate_left.set(left - n);
        Ok(())
    }
}

/// Extracted text, charged against a [`Budget`] as it is written.
pub(super) struct Text<'b> {
    s: String,
    budget: &'b Budget,
}

impl<'b> Text<'b> {
    pub(super) fn new(budget: &'b Budget) -> Text<'b> {
        Text { s: String::new(), budget }
    }

    pub(super) fn budget(&self) -> &'b Budget {
        self.budget
    }

    /// No more text is taken (the budget is spent, or a bound failed).
    pub(super) fn full(&self) -> bool {
        self.budget.chars_left.get() == 0 || self.budget.exceeded.borrow().is_some()
    }

    pub(super) fn push_str(&mut self, t: &str) {
        if t.is_empty() || self.budget.exceeded.borrow().is_some() {
            return;
        }
        let left = self.budget.chars_left.get();
        let (taken, n) = if t.len() <= left {
            (t, t.chars().count())
        } else {
            match t.char_indices().nth(left) {
                Some((i, _)) => (&t[..i], left),
                None => (t, t.chars().count()),
            }
        };
        self.s.push_str(taken);
        self.budget.chars_left.set(left - n);
        if taken.len() < t.len() && self.budget.hard {
            self.budget.fail(format!("extracted text exceeds the limit of [{MAX_TEXT_CHARS}] characters"));
        }
    }

    pub(super) fn push(&mut self, c: char) {
        let mut b = [0u8; 4];
        self.push_str(c.encode_utf8(&mut b));
    }

    pub(super) fn as_str(&self) -> &str {
        &self.s
    }

    pub(super) fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.s.len()
    }

    /// Drop trailing spaces (their characters are given back).
    pub(super) fn trim_end_spaces(&mut self) {
        while self.s.ends_with(' ') {
            self.s.pop();
            self.budget.chars_left.set(self.budget.chars_left.get() + 1);
        }
    }

    pub(super) fn into_string(self) -> String {
        self.s
    }
}

/// Extract a document's text, content type and properties (Tika's
/// `AutoDetectParser` as ES's `TikaImpl` runs it) within `budget`. Text
/// beyond a positive `indexed_chars` is not extracted (Tika's write limit:
/// the text is truncated, the document succeeds). `Err`: why the document
/// cannot be parsed (damaged, encrypted, unsupported, incompletely read, or
/// over a bound); the processor fails the document rather than index it
/// without, or with only part of, its text.
pub(super) fn parse(bytes: &[u8], budget: &Budget) -> std::result::Result<Parsed, ParseError> {
    let mut out = Text::new(budget);
    let r = parse_into(bytes, 0, &mut out);
    if let Some(why) = budget.exceeded() {
        return Err(ParseError::Fatal(why));
    }
    let (content_type, meta) = r?;
    Ok(Parsed { content: out.into_string(), content_type, meta })
}

/// [`parse`] writing the text into `out` (shared by embedded documents).
pub(super) fn parse_into(bytes: &[u8], depth: usize, out: &mut Text) -> std::result::Result<(String, Vec<(&'static str, String)>), ParseError> {
    if let Some(t) = magic_type(bytes) {
        return match t {
            "application/pdf" => Ok((t.into(), super::attachment_pdf::extract(bytes, out)?)),
            "application/zip" if depth < MAX_NESTING => super::attachment_office::zip_document(bytes, depth, out),
            "application/x-tika-msoffice" => super::attachment_office::ole_document(bytes, out),
            "application/gzip" if depth < MAX_NESTING => {
                use std::io::Read;
                const PER: u64 = 64 << 20;
                let budget = out.budget();
                let mut inner = Vec::new();
                flate2::read::MultiGzDecoder::new(bytes).take(budget.inflate_cap(PER)).read_to_end(&mut inner).map_err(|e| format!("invalid gzip data: {e}"))?;
                budget.inflate(inner.len() as u64, PER, "gzip content").map_err(ParseError::Fatal)?;
                let (_, meta) = parse_into(&inner, depth + 1, out)?;
                Ok((t.into(), meta))
            }
            // Images carry no text (Tika reports their type only).
            "image/png" | "image/jpeg" | "image/gif" => Ok((t.into(), Vec::new())),
            other => Err(ParseError::Fatal(format!("archive [{other}] nested too deeply"))),
        };
    }
    let (text, charset) = match std::str::from_utf8(bytes) {
        Ok(s) if s.is_ascii() => (std::borrow::Cow::Borrowed(s), "ISO-8859-1"),
        Ok(s) => (std::borrow::Cow::Borrowed(s), "UTF-8"),
        Err(_) => (std::borrow::Cow::Owned(bytes.iter().map(|b| *b as char).collect::<String>()), "ISO-8859-1"),
    };
    let head = text.trim_start().get(..64.min(text.trim_start().len())).unwrap_or("").to_ascii_lowercase();
    if head.starts_with("{\\rtf") {
        out.push_str(&strip_rtf(&text));
        return Ok(("application/rtf".into(), Vec::new()));
    }
    if head.starts_with("<!doctype html") || head.starts_with("<html") {
        let lower = text.to_ascii_lowercase();
        let title = lower.find("<title>").and_then(|a| lower[a..].find("</title>").map(|b| text[a + 7..a + b].trim().to_string())).filter(|t| !t.is_empty());
        let body = match (lower.find("<body"), lower.rfind("</body>")) {
            (Some(a), Some(b)) if a < b => &text[a..b],
            _ => &text[..],
        };
        out.push_str(super::util::html_text(body).trim());
        out.push('\n');
        return Ok((format!("text/html; charset={charset}"), title.map(|t| vec![("title", t)]).unwrap_or_default()));
    }
    // Tika's text parser ends the extracted text with a newline.
    out.push_str(&text);
    out.push('\n');
    Ok((format!("text/plain; charset={charset}"), Vec::new()))
}

/// Very small stop-word based language guess (ISO 639-1).
pub fn detect_language(text: &str) -> String {
    const PROFILES: &[(&str, &[&str])] = &[
        ("en", &["the", "and", "is", "of", "to", "in", "a", "this", "that", "it", "for", "if", "an", "with", "be", "are", "was", "text", "on"]),
        ("de", &["der", "die", "und", "das", "ist", "nicht", "ein", "eine", "zu", "mit", "den", "von", "ich", "sie", "es"]),
        ("fr", &["le", "la", "les", "et", "est", "un", "une", "des", "du", "en", "que", "pour", "pas", "dans", "ce"]),
        ("es", &["el", "la", "los", "las", "y", "es", "un", "una", "que", "de", "en", "por", "para", "con", "no"]),
        ("it", &["il", "la", "e", "di", "che", "è", "un", "una", "per", "non", "con", "sono", "del", "gli"]),
        ("nl", &["de", "het", "een", "en", "is", "van", "dat", "niet", "op", "te", "zijn", "met", "voor"]),
        ("pt", &["o", "a", "os", "as", "e", "é", "um", "uma", "que", "não", "do", "da", "em", "para"]),
        ("et", &["ja", "on", "see", "ei", "ta", "et", "oli", "kui", "mis", "aga", "ka", "tere"]),
    ];
    let words: Vec<String> = text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect();
    let mut best = ("en", 0usize);
    for (lang, sw) in PROFILES {
        let n = words.iter().filter(|w| sw.contains(&w.as_str())).count();
        if n > best.1 {
            best = (lang, n);
        }
    }
    best.0.to_string()
}

impl Processor for AttachmentP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        let raw = match doc.get(&self.field)? {
            None | Some(Value::Null) if self.ignore_missing => return Ok(()),
            None => return Err(doc.require(&self.field).err().unwrap().into()),
            Some(Value::Null) => return Err(EsError::illegal_argument(format!("field [{}] is null, cannot parse.", self.field)).into()),
            Some(v) => v.clone(),
        };
        let bytes = match &raw {
            Value::String(s) => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(s.trim())
                    .map_err(|e| EsError::illegal_argument(format!("Illegal base64 character in field [{}]: {e}", self.field)))?
            }
            Value::Array(a) => a.iter().map(|x| x.as_i64().unwrap_or(0) as u8).collect(),
            other => {
                return Err(EsError::illegal_argument(format!("Content field [{}] of unknown type [{}], must be string or byte array", self.field, super::doc::type_name(other))).into());
            }
        };
        let _resource = self.resource_name.as_ref().and_then(|r| doc.get(r).ok().flatten().and_then(Value::as_str).map(str::to_string));
        let mut limit = self.indexed_chars;
        if let Some(f) = &self.indexed_chars_field {
            if let Some(v) = doc.get(f)?.and_then(Value::as_i64) {
                limit = v;
            }
        }
        let mut additional = Map::new();
        // An empty input parses to nothing (Tika's `ZeroByteFileException`,
        // which ES ignores); a document that cannot be parsed fails, and
        // its binary stays where it was.
        let parsed = if bytes.is_empty() {
            None
        } else {
            // `indexed_chars` bounds the extraction itself (Tika's write
            // limit), not a copy of the whole text afterwards.
            let budget = Budget::new(limit);
            Some(parse(&bytes, &budget).map_err(|cause| {
                EsError::new(400, "parse_exception", format!("Error parsing document in field [{}]", self.field)).caused_by(EsError::new(500, "tika_exception", cause.message()))
            })?)
        };
        let (content, content_type, meta) = match parsed {
            Some(p) => (p.content, Some(p.content_type), p.meta),
            None => (String::new(), None, Vec::new()),
        };
        let prop = |name: &str| meta.iter().find(|(p, _)| *p == name).map(|(_, v)| v.clone());
        for p in &self.properties {
            match *p {
                "CONTENT" if !content.is_empty() => {
                    additional.insert("content".into(), json!(content.trim()));
                }
                "LANGUAGE" if !content.is_empty() => {
                    additional.insert("language".into(), json!(detect_language(&content)));
                }
                "CONTENT_TYPE" => {
                    if let Some(t) = &content_type {
                        additional.insert("content_type".into(), json!(t));
                    }
                }
                "CONTENT_LENGTH" => {
                    additional.insert("content_length".into(), json!(content.chars().count()));
                }
                other => {
                    let name = other.to_ascii_lowercase();
                    if let Some(v) = prop(&name) {
                        additional.insert(name, json!(v));
                    }
                }
            }
        }
        doc.set(&self.target, Value::Object(additional))?;
        if self.remove_binary {
            doc.remove(&self.field)?;
        }
        Ok(())
    }
}

/// Build an `attachment` processor (ES `AttachmentProcessor.Factory`).
pub fn build(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let field = cfg.req_str("field")?;
    let resource_name = cfg.opt_str("resource_name")?;
    let target = cfg.str_or("target_field", "attachment")?;
    let names = cfg.opt_list_str("properties")?;
    let indexed_chars = cfg.opt_i64("indexed_chars")?.unwrap_or(NUMBER_OF_CHARS_INDEXED);
    let ignore_missing = cfg.bool_or("ignore_missing", false)?;
    let indexed_chars_field = cfg.opt_str("indexed_chars_field")?;
    let remove_binary = cfg.bool_or("remove_binary", false)?;
    let properties = match names {
        None => PROPERTIES.to_vec(),
        Some(ns) => {
            let mut v = Vec::new();
            for n in ns {
                let up = n.to_uppercase();
                match PROPERTIES.iter().find(|p| **p == up) {
                    Some(p) => v.push(*p),
                    None => return Err(cfg.err("properties", format!("illegal field option [{n}]. valid values are [{}]", PROPERTIES.join(", ")))),
                }
            }
            v.sort_by_key(|p| PROPERTIES.iter().position(|x| x == p));
            v.dedup();
            v
        }
    };
    Ok(Box::new(AttachmentP { field, target, properties, indexed_chars, indexed_chars_field, resource_name, ignore_missing, remove_binary }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCX: &[u8] = include_bytes!("attachment_fixtures/sample.docx");
    const DOC: &[u8] = include_bytes!("attachment_fixtures/sample.doc");
    const TEXT: &str = "Rustic attachment fixture\n\nThe quick brown fox jumps over the lazy dog.";

    fn unlimited(bytes: &[u8]) -> std::result::Result<Parsed, ParseError> {
        parse(bytes, &Budget::new(-1))
    }

    use super::super::attachment_office::tests::zip;

    /// A workbook whose `cells` cells all refer to one shared string of
    /// `len` characters (the reference fan-out of finding 113).
    fn repeated_cell_workbook(cells: usize, len: usize) -> Vec<u8> {
        let shared = format!("<sst><si><t>{}</t></si></sst>", "x".repeat(len));
        let mut sheet = String::from("<worksheet><sheetData>");
        for row in 0..cells / 100 {
            sheet.push_str(&format!("<row r=\"{}\">", row + 1));
            for _ in 0..100 {
                sheet.push_str("<c t=\"s\"><v>0</v></c>");
            }
            sheet.push_str("</row>");
        }
        sheet.push_str("</sheetData></worksheet>");
        zip(&[
            ("xl/workbook.xml", br#"<workbook><sheets><sheet name="S"/></sheets></workbook>"#),
            ("xl/sharedStrings.xml", shared.as_bytes()),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ])
    }

    /// The fan-out is bounded while it is extracted: `indexed_chars` stops
    /// it (Tika's write limit: truncated text, the document succeeds), and
    /// `-1` (unlimited in ES) still has a hard total that fails cleanly. A
    /// 100,000-cell workbook referring to a 32,767-character string would
    /// expand to ~3.3 GB.
    #[test]
    fn repeated_cell_workbook_is_bounded_while_extracting() {
        let book = repeated_cell_workbook(100_000, 32_767);
        for limit in [100_000i64, 50, 1, 0] {
            let started = std::time::Instant::now();
            let p = parse(&book, &Budget::new(limit)).unwrap();
            assert_eq!(p.content.chars().count(), limit as usize, "indexed_chars {limit}");
            assert!(p.content.starts_with(&"S\nxxxxx"[..(limit as usize).min(7)]), "indexed_chars {limit}");
            assert!(started.elapsed() < std::time::Duration::from_secs(20), "indexed_chars {limit} took {:?}", started.elapsed());
        }
        let e = parse(&book, &Budget::new(-1)).unwrap_err();
        assert!(matches!(&e, ParseError::Fatal(m) if m.contains("extracted text exceeds the limit")), "{e:?}");
    }

    /// Many entries share one character budget and one decompression
    /// budget: the archive's text stops at `indexed_chars`, and entries
    /// that together inflate past the document's total fail it.
    #[test]
    fn multi_entry_archive_shares_one_budget() {
        let body = "word ".repeat(2_000);
        let names: Vec<String> = (0..50).map(|i| format!("part{i:02}.txt")).collect();
        let files: Vec<(&str, &[u8])> = names.iter().map(|n| (n.as_str(), body.as_bytes())).collect();
        let archive = zip(&files);
        let all = unlimited(&archive).unwrap();
        assert_eq!(all.content.matches("word").count(), 50 * 2_000);
        for limit in [100_000i64, 30, 0] {
            let p = parse(&archive, &Budget::new(limit)).unwrap();
            assert_eq!(p.content.chars().count(), limit as usize, "indexed_chars {limit}");
            assert_eq!(p.content, all.content.chars().take(limit as usize).collect::<String>());
        }
        // 50 entries of 10,000 bytes against a 100,000-byte total.
        let e = parse(&archive, &Budget::with_limits(-1, 100_000)).unwrap_err();
        assert!(matches!(&e, ParseError::Fatal(m) if m.contains("decompressed content exceeds")), "{e:?}");
        // Nested archives draw on the same total.
        let nested = zip(&[("inner.zip", &archive[..])]);
        assert!(matches!(parse(&nested, &Budget::with_limits(-1, 100_000)), Err(ParseError::Fatal(_))));
        assert!(unlimited(&nested).is_ok());
    }

    /// Partial extraction: an archive entry the container cannot give fails
    /// the document (Tika's read error does), as does an embedded document
    /// of a kind with no extraction here. An embedded document that is
    /// itself damaged is skipped, as Tika's embedded-document extractor
    /// records and skips its exception.
    #[test]
    fn partial_archives_fail_unless_tika_would_skip_the_part() {
        let good = zip(&[("a.txt", b"alpha text"), ("b.txt", b"beta text")]);
        assert_eq!(unlimited(&good).unwrap().content, "a.txt\nalpha text\nb.txt\nbeta text\n");
        // Corrupt the second entry's compressed data.
        let mut bad = good.clone();
        // The first `b.txt` is the entry's local header; its data follows.
        let at = memchr::memmem::find(&bad, b"b.txt").unwrap() + 5;
        for b in &mut bad[at..at + 6] {
            *b = 0xFF;
        }
        let e = unlimited(&bad).unwrap_err();
        assert!(matches!(&e, ParseError::Fatal(m) if m.contains("b.txt")), "{e:?}");
        // A damaged embedded document: skipped, the rest extracted.
        let damaged = zip(&[("a.txt", b"alpha text"), ("broken.pdf", b"%PDF-1.4 nothing here")]);
        assert_eq!(unlimited(&damaged).unwrap().content, "a.txt\nalpha text\nbroken.pdf\n");
        // An embedded document with no extraction here: fails.
        let mut not_word = DOC.to_vec();
        let at = not_word.windows(24).position(|w| w == "WordDocument".encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>().as_slice()).unwrap();
        not_word[at..at + 24].copy_from_slice(&"WordDokument".encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>());
        let unsupported = zip(&[("a.txt", b"alpha text"), ("legacy.doc", &not_word[..])]);
        assert!(matches!(unlimited(&unsupported), Err(ParseError::Fatal(m)) if m.contains("not supported")));
    }

    #[test]
    fn office_documents_yield_their_text() {
        let p = unlimited(DOCX).unwrap();
        assert_eq!(p.content.trim(), TEXT);
        assert_eq!(p.content_type, "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        let p = unlimited(DOC).unwrap();
        assert_eq!(p.content, format!("{TEXT}\n"));
        assert_eq!(p.content_type, "application/msword");
        // A gzip of a document is that document's text.
        let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut z, DOCX).unwrap();
        let p = unlimited(&z.finish().unwrap()).unwrap();
        assert_eq!((p.content.trim(), p.content_type.as_str()), (TEXT, "application/gzip"));
    }

    #[test]
    fn damaged_binary_documents_fail_instead_of_parsing_empty() {
        for (what, bytes) in [
            ("truncated docx", &DOCX[..DOCX.len() / 2]),
            ("truncated doc", &DOC[..DOC.len() / 3]),
            ("pdf without objects", &b"%PDF-1.7\nnot really a pdf"[..]),
            ("zip header only", &b"PK\x03\x04\x14\x00garbage"[..]),
            ("ole header only", &b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1\x00\x00"[..]),
        ] {
            assert!(unlimited(bytes).is_err(), "{what} parsed");
        }
        // An OLE file that is not a Word document is refused, not empty.
        let mut not_word = DOC.to_vec();
        let at = not_word.windows(24).position(|w| w == "WordDocument".encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>().as_slice()).unwrap();
        not_word[at..at + 24].copy_from_slice(&"WordDokument".encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<_>>());
        assert!(unlimited(&not_word).unwrap_err().message().contains("not supported"));
    }

    #[test]
    fn plain_text() {
        let p = unlimited(b"This is an english text to test if the pipeline works").unwrap();
        assert_eq!(p.content_type, "text/plain; charset=ISO-8859-1");
        assert_eq!(p.content.chars().count(), 54);
        assert_eq!(detect_language(&p.content), "en");
        assert_eq!(strip_rtf("{\\rtf1\\ansi{\\fonttbl{\\f0 Arial;}}\\f0 Hello\\par World}"), "Hello\nWorld");
    }
}
