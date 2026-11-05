//! JSON text held in a Java String may itself contain unpaired UTF16 units.
//! Escape those units only inside JSON strings, then retain scalar tokens.
use crate::error::{EsError, Result};
use crate::source_json::Source;
use serde::de::{DeserializeSeed, MapAccess, Visitor};
use serde_json::value::RawValue;
use std::collections::BTreeMap;

fn json_text(units: &[u16]) -> Result<String> {
    let mut out = String::new();
    let (mut quoted, mut escaped) = (false, false);
    for decoded in char::decode_utf16(units.iter().copied()) {
        match decoded {
            Ok(c) => {
                out.push(c);
                if quoted && escaped { escaped = false; }
                else if quoted && c == '\\' { escaped = true; }
                else if c == '"' { quoted = !quoted; }
            }
            Err(e) => {
                if !quoted || escaped { return Err(EsError::illegal_argument("invalid surrogate outside a JSON string value")); }
                use std::fmt::Write;
                write!(out, "\\u{:04x}", e.unpaired_surrogate()).unwrap();
            }
        }
    }
    Ok(out)
}

struct Members { duplicates: bool }
impl<'de> DeserializeSeed<'de> for Members {
    type Value = BTreeMap<String, Box<RawValue>>;
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for Members {
    type Value = BTreeMap<String, Box<RawValue>>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a JSON object") }
    fn visit_map<M: MapAccess<'de>>(self, mut input: M) -> std::result::Result<Self::Value, M::Error> {
        let mut out = BTreeMap::new();
        while let Some((key,value)) = input.next_entry::<String,Box<RawValue>>()? {
            if !self.duplicates && out.contains_key(&key) { return Err(serde::de::Error::custom(format!("Duplicate field '{key}'"))); }
            out.insert(key,value);
        }
        Ok(out)
    }
}

fn value(raw: Box<RawValue>, duplicates: bool, depth: usize) -> Result<Source> {
    if depth > 128 { return Err(EsError::illegal_argument("JSON nesting limit exceeded")); }
    Ok(match raw.get().as_bytes().first() {
        Some(b'{') => {
            let mut parser = serde_json::Deserializer::from_str(raw.get());
            let children = Members { duplicates }.deserialize(&mut parser)?;
            Source::Object(children.into_iter().map(|(key,raw)| Ok((key,value(raw,duplicates,depth+1)?))).collect::<Result<_>>()?)
        }
        Some(b'[') => {
            let children: Vec<Box<RawValue>> = serde_json::from_str(raw.get())?;
            Source::Array(children.into_iter().map(|raw|value(raw,duplicates,depth+1)).collect::<Result<_>>()?)
        }
        _ => Source::parse(raw.get())?,
    })
}

pub(super) fn parse(input: &Source, duplicates: bool, strict: bool) -> Result<Source> {
    let text = json_text(&crate::script::source_from_json(input).string_units())?;
    let mut stream = serde_json::Deserializer::from_str(&text).into_iter::<Box<RawValue>>();
    let raw = stream.next().ok_or_else(||EsError::illegal_argument("empty JSON input"))??;
    if strict && !text[stream.byte_offset()..].trim().is_empty() { return Err(EsError::illegal_argument("input is not valid JSON and strict_json_parsing is true")); }
    value(raw,duplicates,0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_text_preserves_raw_and_escaped_units_and_duplicate_policy() {
        let input=Source::parse(r#""{\"a\":\"\ud800\",\"b\":\"\\udfff\"}""#).unwrap();
        assert!(parse(&input,false,true).unwrap().equivalent(&Source::parse(r#"{"a":"\ud800","b":"\udfff"}"#).unwrap()));
        let input=Source::from_value(&serde_json::json!("{\"a\":1,\"a\":2} trailing"));
        assert!(parse(&input,false,false).is_err());
        assert!(parse(&input,true,true).is_err());
        assert!(parse(&input,true,false).unwrap().equivalent(&Source::parse("{\"a\":2}").unwrap()));
        let invalid=Source::parse(r#""{\"\ud800\":1}""#).unwrap();
        assert!(parse(&invalid,false,true).is_err());
    }
}
