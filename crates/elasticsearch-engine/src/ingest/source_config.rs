//! Structural pipeline metadata and raw script inputs have different string
//! contracts. Only script source/parameters bypass Unicode-only metadata.
use crate::error::Result;
use crate::source_json::Source;
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) fn script_spec(raw: &Source) -> Source {
    match raw {
        Source::Object(m) => Source::Object(m.iter().filter(|(k,_)| matches!(k.as_str(), "source"|"inline"|"id"|"lang"|"params")).map(|(k,v)|(k.clone(),v.clone())).collect()),
        _ => raw.clone(),
    }
}

fn script_metadata(raw: &Source) -> Result<Source> {
    let raw = match raw {
        Source::Scalar(v) if v.get().starts_with('"') => Source::Object(BTreeMap::from([("source".into(),raw.clone())])),
        _ => raw.clone(),
    };
    let wrapped=Source::Object(BTreeMap::from([("script".into(),raw)]));
    let (metadata,_) = crate::source_json::update_request(&wrapped.text())?;
    Ok(Source::from_value(&metadata["script"]))
}

fn processor(raw: &Source) -> Result<Source> {
    let Source::Object(entries)=raw else { return Ok(raw.clone()); };
    let mut result=BTreeMap::new();
    for (kind, original) in entries {
        let mut config=original.clone();
        if kind=="script" {
            let script=script_metadata(&script_spec(original))?;
            match (&mut config,script) {
                (Source::Object(m),Source::Object(s)) => { for (k,v) in s {m.insert(k,v);} }
                (dst, script) => *dst=script,
            }
        }
        if let Source::Object(m)=&mut config {
            if matches!(kind.as_str(),"set"|"append"|"csv") {
                if let Some(value)=m.get_mut(if kind=="csv" {"empty_value"} else {"value"}) {
                    if serde_json::from_str::<Value>(&value.text()).is_err() {
                        // Compile the processor's structural/type contract only.
                        // ProcessorNode retains the complete original value;
                        // source execution never reads this metadata projection.
                        *value=Source::from_value(&match value {Source::Array(_)=>serde_json::json!([]),Source::Object(_)=>serde_json::json!({}),Source::Scalar(_)=>serde_json::json!("")});
                    }
                }
            }
            let raw_text_fields:&[&str]=match kind.as_str(){"gsub"=>&["pattern","replacement"],"redact"=>&["prefix","suffix"],"split"|"join"=>&["separator"],"kv"=>&["field_split","value_split","trim_key","trim_value"],_=>&[]};
            for key in raw_text_fields{if let Some(value)=m.get_mut(*key){if serde_json::from_str::<Value>(&value.text()).is_err(){*value=Source::from_value(&serde_json::json!(""));}}}
            if let Some(v)=m.get_mut("if") {
                if !matches!(v,Source::Scalar(s) if s.get()=="null") {*v=script_metadata(v)?;}
            }
            if let Some(v)=m.get_mut("on_failure") {*v=processors(v)?;}
            if kind=="foreach" {
                if let Some(v)=m.get_mut("processor") {*v=processor(v)?;}
            }
        }
        result.insert(kind.clone(),config);
    }
    Ok(Source::Object(result))
}

fn processors(raw: &Source) -> Result<Source> {
    match raw {
        Source::Array(a) => Ok(Source::Array(a.iter().map(processor).collect::<Result<_>>()?)),
        _ => Ok(raw.clone()),
    }
}

pub(super) fn metadata(raw: &Source) -> Result<Value> {
    let mut normalized=raw.clone();
    if let Source::Object(m)=&mut normalized {
        for key in ["processors","on_failure"] {
            if let Some(v)=m.get_mut(key) {*v=processors(v)?;}
        }
    }
    Ok(serde_json::from_str(&normalized.text())?)
}
