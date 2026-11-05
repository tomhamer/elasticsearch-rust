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

impl Processor for InferenceP {
    fn process(&self, doc: &mut IngestDocument, _: &PipelineContext) -> PResult {
        if !self.input_output.is_empty() {
            for (inp, out) in &self.input_output {
                let Some(v) = doc.get(inp)?.cloned() else {
                    return Err(EsError::illegal_argument(format!("field [{inp}] doesn't exist")).into());
                };
                let r = embed_one(&self.model_id, &text_of(&v))?;
                doc.set(out, r)?;
            }
            return Ok(());
        }
        let from = self.field_map.first().map(|(f, _)| f.clone()).unwrap_or_else(|| "text_field".to_string());
        let Some(v) = doc.get(&from)?.cloned() else {
            return Err(EsError::illegal_argument(format!("field [{from}] doesn't exist")).into());
        };
        let r = embed_one(&self.model_id, &text_of(&v))?;
        doc.set(&format!("{}.{}", self.target, self.results_field), r)?;
        doc.set(&format!("{}.model_id", self.target), json!(self.model_id))?;
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
