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

struct Parsed {
    content: String,
    content_type: String,
    title: Option<String>,
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

fn parse(bytes: &[u8]) -> Parsed {
    if let Some(t) = magic_type(bytes) {
        return Parsed { content: String::new(), content_type: t.into(), title: None };
    }
    let (text, charset) = match std::str::from_utf8(bytes) {
        Ok(s) if s.is_ascii() => (s.to_string(), "ISO-8859-1"),
        Ok(s) => (s.to_string(), "UTF-8"),
        Err(_) => (bytes.iter().map(|b| *b as char).collect::<String>(), "ISO-8859-1"),
    };
    let head = text.trim_start().get(..64.min(text.trim_start().len())).unwrap_or("").to_ascii_lowercase();
    if head.starts_with("{\\rtf") {
        return Parsed { content: strip_rtf(&text), content_type: "application/rtf".into(), title: None };
    }
    if head.starts_with("<!doctype html") || head.starts_with("<html") {
        let lower = text.to_ascii_lowercase();
        let title = lower.find("<title>").and_then(|a| lower[a..].find("</title>").map(|b| text[a + 7..a + b].trim().to_string())).filter(|t| !t.is_empty());
        let body = match (lower.find("<body"), lower.rfind("</body>")) {
            (Some(a), Some(b)) if a < b => &text[a..b],
            _ => &text[..],
        };
        let stripped = super::util::html_text(body);
        return Parsed { content: stripped.trim().to_string() + "\n", content_type: format!("text/html; charset={charset}"), title };
    }
    // Tika's text parser ends the extracted text with a newline.
    Parsed { content: format!("{text}\n"), content_type: format!("text/plain; charset={charset}"), title: None }
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
        let parsed = if bytes.is_empty() { None } else { Some(parse(&bytes)) };
        let (mut content, content_type, title) = match parsed {
            Some(p) => (p.content, Some(p.content_type), p.title),
            None => (String::new(), None, None),
        };
        if limit >= 0 && content.chars().count() as i64 > limit {
            content = content.chars().take(limit as usize).collect();
        }
        for p in &self.properties {
            match *p {
                "CONTENT" if !content.is_empty() => {
                    additional.insert("content".into(), json!(content.trim()));
                }
                "LANGUAGE" if !content.is_empty() => {
                    additional.insert("language".into(), json!(detect_language(&content)));
                }
                "TITLE" => {
                    if let Some(t) = &title {
                        additional.insert("title".into(), json!(t));
                    }
                }
                "CONTENT_TYPE" => {
                    if let Some(t) = &content_type {
                        additional.insert("content_type".into(), json!(t));
                    }
                }
                "CONTENT_LENGTH" => {
                    additional.insert("content_length".into(), json!(content.chars().count()));
                }
                _ => {}
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

    #[test]
    fn plain_text() {
        let p = parse(b"This is an english text to test if the pipeline works");
        assert_eq!(p.content_type, "text/plain; charset=ISO-8859-1");
        assert_eq!(p.content.chars().count(), 54);
        assert_eq!(detect_language(&p.content), "en");
        assert_eq!(strip_rtf("{\\rtf1\\ansi{\\fonttbl{\\f0 Arial;}}\\f0 Hello\\par World}"), "Hello\nWorld");
    }
}
