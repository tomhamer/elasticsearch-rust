//! The `inference` ingest processor: runs a text-embedding / text-expansion
//! model (an `_inference` endpoint id or a built-in trained model id) over
//! document fields.
//!
//! Config: `model_id`, `target_field` (default `ml.inference.<model_id>`),
//! `field_map` (`{doc_field: model_input_field}`), `input_output`
//! (`[{input_field, output_field}]`), `inference_config` (accepted).

use super::doc::IngestDocument;
use super::{Cfg, PResult, PipelineContext, Processor};
use crate::error::{EsError, Result};
use crate::inference::Embedded;
use serde_json::{json, Map, Value};

struct InferenceP {
    model_id: String,
    target: String,
    results_field: String,
    field_map: Vec<(String, String)>,
    input_output: Vec<(String, String)>,
}

fn endpoint_for(model_id: &str) -> String {
    match model_id.trim_end_matches("_linux-x86_64") {
        ".multilingual-e5-small" => ".multilingual-e5-small-elasticsearch".into(),
        ".elser_model_2" => ".elser-2-elasticsearch".into(),
        other => other.to_string(),
    }
}

fn embed_one(model_id: &str, text: &str) -> Result<Value> {
    let svc = crate::inference::current().ok_or_else(|| EsError::resource_not_found(format!("Could not find trained model [{model_id}]")))?;
    let id = endpoint_for(model_id);
    if svc.get_endpoint(&id).is_none() {
        return Err(EsError::resource_not_found(format!("Could not find trained model [{model_id}]")));
    }
    Ok(match svc.embed_typed(&id, &[text.to_string()], "ingest")? {
        Embedded::Dense(mut v) => json!(v.pop().unwrap_or_default()),
        Embedded::Sparse(mut v) => Value::Object(v.pop().unwrap_or_default()),
    })
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The model input field of NLP models (`text_field`).
const MODEL_INPUT: &str = "text_field";

/// ES `InferenceResults.writeResultToField`: an absent destination is set,
/// an existing one is appended to (finding 247: earlier results are kept).
fn set_or_append(doc: &mut IngestDocument, path: &str, value: Value) -> Result<()> {
    if doc.has(path) {
        doc.append(path, value, true)
    } else {
        doc.set(path, value)
    }
}

impl InferenceP {
    /// The model's input text (finding 253): ES copies the document and
    /// applies `field_map` with `putIfAbsent(model_field, value)`, so the
    /// document's own `text_field` wins and a mapped field only supplies it
    /// when absent.
    fn input(&self, doc: &IngestDocument) -> Result<Value> {
        if let Some(v) = doc.get(MODEL_INPUT)?.filter(|v| !v.is_null()) {
            return Ok(v.clone());
        }
        for (src, dest) in &self.field_map {
            if dest == MODEL_INPUT {
                if let Some(v) = doc.get(src)?.filter(|v| !v.is_null()) {
                    return Ok(v.clone());
                }
            }
        }
        Err(EsError::new(400, "status_exception", format!("Input field [{MODEL_INPUT}] does not exist in the source document")))
    }
}

impl Processor for InferenceP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        if !self.input_output.is_empty() {
            for (inp, out) in &self.input_output {
                let Some(v) = doc.get(inp)?.cloned() else {
                    return Err(EsError::illegal_argument(format!("field [{inp}] doesn't exist")).into());
                };
                let r = embed_one(&self.model_id, &text_of(&v))?;
                set_or_append(doc, out, r)?;
            }
            return Ok(());
        }
        let v = self.input(doc)?;
        let r = embed_one(&self.model_id, &text_of(&v))?;
        let mut result = Map::new();
        result.insert(self.results_field.clone(), r);
        result.insert("model_id".into(), json!(self.model_id));
        set_or_append(doc, &self.target, Value::Object(result))?;
        Ok(())
    }
}

fn pairs(m: Option<Map<String, Value>>, k1: &str, k2: &str) -> Vec<(String, String)> {
    m.map(|m| m.into_iter().filter_map(|(a, b)| b.as_str().map(|b| (a, b.to_string()))).collect()).unwrap_or_else(|| {
        let _ = (k1, k2);
        Vec::new()
    })
}

pub fn build(cfg: &mut Cfg) -> Result<Box<dyn Processor>> {
    let model_id = cfg.req_str("model_id")?;
    let target = cfg.opt_str("target_field")?.unwrap_or_else(|| format!("ml.inference.{model_id}"));
    let field_map = pairs(cfg.opt_map("field_map")?, "", "");
    let mut input_output = Vec::new();
    if let Some(Value::Array(a)) = cfg.opt_value("input_output") {
        for e in a {
            let (Some(i), Some(o)) = (e.get("input_field").and_then(Value::as_str), e.get("output_field").and_then(Value::as_str)) else {
                return Err(cfg.err("input_output", "each entry requires [input_field] and [output_field]"));
            };
            input_output.push((i.to_string(), o.to_string()));
        }
    }
    let inference_config = cfg.opt_map("inference_config")?;
    let results_field = match inference_config.as_ref().and_then(|m| m.values().next()).and_then(|v| v.get("results_field")) {
        None => "predicted_value".to_string(),
        Some(Value::String(field)) => field.clone(),
        Some(_) => return Err(cfg.err("inference_config", "[results_field] must be a string")),
    };
    cfg.ignore(&["ignore_missing"]);
    Ok(Box::new(InferenceP { model_id, target, results_field, field_map, input_output }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(v: Value) -> IngestDocument {
        IngestDocument::new(v.as_object().unwrap().clone(), "i", None, None, "now".into())
    }

    fn processor(field_map: &[(&str, &str)]) -> InferenceP {
        InferenceP {
            model_id: "m".into(),
            target: "ml.inference.m".into(),
            results_field: "predicted_value".into(),
            field_map: field_map.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
            input_output: Vec::new(),
        }
    }

    /// Finding 253: the document's own `text_field` is the model input; a
    /// `field_map` entry only supplies it when absent, whatever the order.
    #[test]
    fn field_map_supplies_a_missing_model_input_only() {
        let p = processor(&[("fallback_text", "text_field"), ("other", "unrelated")]);
        let d = doc(json!({"text_field": "current contract", "fallback_text": "old unrelated text"}));
        assert_eq!(p.input(&d).unwrap(), json!("current contract"));
        let d = doc(json!({"fallback_text": "old unrelated text", "other": "x"}));
        assert_eq!(p.input(&d).unwrap(), json!("old unrelated text"));
        let reversed = processor(&[("other", "unrelated"), ("fallback_text", "text_field")]);
        assert_eq!(reversed.input(&d).unwrap(), json!("old unrelated text"));
        assert!(p.input(&doc(json!({"x": 1}))).is_err());
    }

    /// Finding 247: an existing destination keeps its earlier result; the
    /// new one is appended (ES `InferenceResults.writeResult`).
    #[test]
    fn results_are_appended_to_existing_outputs() {
        let mut d = doc(json!({"ml": {"inference": {"m": {"predicted_value": [1.0], "model_id": "old", "note": "kept"}}}, "out": [0.5]}));
        set_or_append(&mut d, "ml.inference.m", json!({"predicted_value": [2.0], "model_id": "m"})).unwrap();
        let t = d.get("ml.inference.m").unwrap().unwrap();
        assert_eq!(t, &json!([{"predicted_value": [1.0], "model_id": "old", "note": "kept"}, {"predicted_value": [2.0], "model_id": "m"}]));
        set_or_append(&mut d, "fresh", json!({"predicted_value": 1})).unwrap();
        assert_eq!(d.get("fresh").unwrap().unwrap(), &json!({"predicted_value": 1}));
        set_or_append(&mut d, "out", json!([0.7])).unwrap();
        assert_eq!(d.get("out").unwrap().unwrap(), &json!([0.5, 0.7]));
    }
}
