//! Ingest pipelines: storage (`_state/ingest.json`), compilation of
//! pipeline configs into processor trees, execution (conditions,
//! `ignore_failure`, processor and pipeline `on_failure`, `drop`,
//! `terminate`, nested `pipeline` calls), the index-request integration
//! (`?pipeline=`, `index.default_pipeline`, `index.final_pipeline`,
//! `_index` re-routing) and both simulate APIs.

pub mod doc;
mod source_config;
mod source_data;
pub mod source_doc;
mod source_uri;
mod source_pipeline;
mod source_template;
pub mod grok;
pub mod jtext;
pub mod geo_circle;
pub mod geo_h3;
pub mod net;
pub mod p_data;
pub mod lite;
pub mod processors;
pub mod util;
pub mod geoip;
pub mod useragent;
pub mod attachment;
pub mod inference;

use crate::error::{EsError, Result};
use crate::node::Node;
pub use doc::IngestDocument;
use parking_lot::RwLock;
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

// ------------------------------------------------------------ errors

/// Why a processor stopped.
#[derive(Debug)]
pub enum ProcessorError {
    /// A failure; `origin` = (processor type, tag, pipeline) of the
    /// innermost processor that raised it.
    Failed(EsError, Option<(String, Option<String>, String)>),
    /// `drop`: the document is discarded.
    Dropped,
    /// `terminate`: stop the current pipeline, keep the document.
    Terminated,
}

impl From<EsError> for ProcessorError {
    fn from(e: EsError) -> Self {
        ProcessorError::Failed(e, None)
    }
}

pub type PResult = std::result::Result<(), ProcessorError>;

/// One processor implementation.
pub trait Processor: Send + Sync {
    fn process(&self, doc: &mut IngestDocument, ctx: &PipelineContext) -> PResult;
}

// ------------------------------------------------------------ scripts

/// Scripts run on the engine's Painless (`crate::script`); an alternative
/// engine can be registered, and the [`lite`] subset is the fallback for
/// sources Painless cannot compile.
pub trait IngestScriptEngine: Send + Sync {
    /// Run `source` with `ctx` (source + metadata + `_ingest`) mutable.
    fn execute(&self, source: &str, lang: &str, params: &Value, ctx: &mut Value) -> Result<()>;
    /// Evaluate a condition against `ctx`.
    fn condition(&self, source: &str, lang: &str, params: &Value, ctx: &mut Value) -> Result<bool>;
}

static SCRIPT_ENGINE: OnceLock<Arc<dyn IngestScriptEngine>> = OnceLock::new();

/// Register the script engine used by ingest (first registration wins).
pub fn register_script_engine(e: Arc<dyn IngestScriptEngine>) {
    let _ = SCRIPT_ENGINE.set(e);
}

/// A compiled script reference (`if` or `script` processor).
pub struct Script {
    pub source: String,
    pub lang: String,
    pub params: Value,
    raw_script: Option<crate::source_json::Source>,
    lite: std::result::Result<lite::LiteScript, EsError>,
}

impl Script {
    pub fn new(source: String, lang: String, params: Value) -> Script {
        let lite = lite::LiteScript::compile(&source);
        Script { source, lang, params, raw_script: None, lite }
    }

    /// Compile metadata separately from Java string values in script inputs.
    pub fn from_source(raw: &crate::source_json::Source) -> Result<Script> {
        use crate::source_json::Source;
        let raw = match raw {
            Source::Scalar(token) if token.get().starts_with('"') => Source::Object(BTreeMap::from([("source".into(),raw.clone())])),
            _ => raw.clone(),
        };
        let wrapper=Source::Object(BTreeMap::from([("script".into(),raw.clone())]));
        let (metadata, _)=crate::source_json::update_request(&wrapper.text())?;
        let mut script=Self::from_config(&metadata["script"])?;
        // A stored source can itself contain lone units. Preserve that token
        // while binding the caller's raw parameter values.
        if raw.get("id").is_some() {
            if let Some(Source::Object(stored))=script.raw_script.as_mut() {
                if let Some(params)=raw.get("params") {stored.insert("params".into(),params.clone());}
            } else {script.raw_script=Some(raw);}
        } else {script.raw_script=Some(raw);}
        Ok(script)
    }

    fn compile_source(&self, context: crate::script::ScriptContext) -> Result<crate::script::CompiledScript> {
        let metadata=json!({"source":self.source,"lang":self.lang,"params":self.params});
        crate::script::compile_json_source(&metadata,self.raw_script.as_ref(),context,None)
    }

    /// Execute without converting the context through Unicode-only Value.
    /// Serialization validates field names and reference cycles before replacing
    /// the caller's context, so a failed script cannot partially mutate it.
    /// [`Script::execute_source`] on an ingest document's `ctx` itself (no
    /// copy to run on): it is replaced only when the script succeeds and
    /// leaves an object (ES: "ingest context must be an object"), else left
    /// as it was.
    pub fn execute_on_ctx(&self, ctx: &mut crate::source_json::Source) -> Result<()> {
        self.execute_source_checked(ctx, true)
    }

    pub fn execute_source(&self, context: &mut crate::source_json::Source) -> Result<()> {
        self.execute_source_checked(context, false)
    }

    fn execute_source_checked(&self, context: &mut crate::source_json::Source, object: bool) -> Result<()> {
        let compiled=self.compile_source(crate::script::ScriptContext::Ingest)?;
        let mut executor=compiled.executor(None,None);
        // A script that reaches ctx only by constant keys gets just those
        // parts (converting a whole document both ways was most of a
        // `set` + script pipeline's time); the result is the same.
        static WHOLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let whole = *WHOLE.get_or_init(|| std::env::var("RUSTIC_SCRIPT_WHOLE_CTX").is_ok_and(|v| v == "1"));
        if let (false, Some(paths), crate::source_json::Source::Object(map)) = (whole, compiled.program.ctx_paths.clone(), &*context) {
            let (value, made) = crate::script::partial_ctx(map, &paths);
            executor.set_ctx(value.clone());
            executor.run().map_err(|e|compiled.error(&e))?;
            if self_referencing(&value, &mut Vec::new()) {
                return Err(EsError::illegal_argument(format!("Failed to generate the source document for ingest pipeline [{}]", INGEST_PIPELINE.with(|p| p.borrow().clone()))));
            }
            return crate::script::merge_partial(context, &value, &made);
        }
        self.execute_source_whole(context, compiled, executor, object)
    }

    /// [`Script::execute_source`] converting all of ctx.
    fn execute_source_whole(&self, context: &mut crate::source_json::Source, compiled: crate::script::CompiledScript, mut executor: crate::script::Executor, object: bool) -> Result<()> {
        let value=crate::script::source_from_json(context);
        executor.set_ctx(value.clone());
        executor.run().map_err(|e|compiled.error(&e))?;
        if self_referencing(&value, &mut Vec::new()) {
            return Err(EsError::illegal_argument(format!("Failed to generate the source document for ingest pipeline [{}]", INGEST_PIPELINE.with(|p| p.borrow().clone()))));
        }
        let output=crate::script::source_to_json(&value)?;
        if object && !matches!(&output, crate::source_json::Source::Object(_)) {
            return Err(EsError::illegal_argument("ingest context must be an object"));
        }
        *context=output;
        Ok(())
    }

    /// Parse the `if` / `script` shapes: a string, or `{source|id, lang, params}`.
    pub fn from_config(v: &Value) -> Result<Script> {
        match v {
            Value::String(s) => Ok(Script::new(s.clone(), "painless".into(), json!({}))),
            Value::Object(m) => {
                let lang = m.get("lang").and_then(Value::as_str).unwrap_or("painless").to_string();
                let params = m.get("params").cloned().unwrap_or(json!({}));
                if m.contains_key("id") && m.contains_key("source") {
                    return Err(EsError::new(400, "parse_exception", "Only one of [id] or [source] may be configured"));
                }
                if let Some(id) = m.get("id").and_then(Value::as_str) {
                    let data = crate::enrich::with_current(|n| n.data.clone());
                    let found = data.and_then(|d| crate::script::stored::for_data_dir(&d).get(id));
                    return match found {
                        Some(st) if st.lang == "painless" => {
                            let mut script=Script::new(st.source,st.lang,params);
                            if let Some(token)=st.raw_source {
                                script.raw_script=Some(crate::source_json::Source::Object(BTreeMap::from([
                                    ("source".into(),crate::source_json::Source::parse(&token)?),
                                    ("params".into(),crate::source_json::Source::from_value(&script.params)),
                                ])));
                            }
                            Ok(script)
                        },
                        Some(st) => Err(EsError::illegal_argument(format!("script_lang not supported [{}]", st.lang))),
                        None => Err(EsError::resource_not_found(format!("unable to find script [{id}] in cluster state"))),
                    };
                }
                let src = m
                    .get("source")
                    .or(m.get("inline"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| EsError::new(400, "parse_exception", "Need [id] or [source] parameter to refer to scripts"))?;
                if lang == "mustache" {
                    return Err(EsError::new(500, "general_script_exception", format!("Failed to compile inline script [{src}] using lang [mustache]"))
                        .caused_by(EsError::illegal_argument("mustache engine does not know how to handle context [ingest]")));
                }
                if lang != "painless" {
                    return Err(EsError::illegal_argument(format!("script_lang not supported [{lang}]")));
                }
                Ok(Script::new(src.to_string(), lang, params))
            }
            _ => Err(EsError::new(400, "parse_exception", "script must be a string or an object")),
        }
    }

    /// Compile now so definition-time errors (400 `script_exception`) surface at PUT.
    pub fn validate(&self, ctx: crate::script::ScriptContext) -> Result<()> {
        if SCRIPT_ENGINE.get().is_some() {
            return Ok(());
        }
        match self.compile_source(ctx) {
            Ok(_) => Ok(()),
            Err(_) if self.raw_script.is_none() && self.lite.is_ok() => Ok(()),
            // An API our Painless does not know yet is not a definition error: defer to run time
            // so valid ES pipelines (e.g. `metadata()` in the built-in APM pipeline) can still be stored.
            Err(e) if e.caused_by.as_ref().map(|c| c.reason.starts_with("Unknown ")).unwrap_or(false) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn with_ctx<T>(doc: &mut IngestDocument, f: impl FnOnce(&mut Value) -> Result<T>) -> Result<T> {
        // ES 8.x exposes only the source and metadata as `ctx` (no `_ingest`).
        f(&mut doc.ctx)
    }

    /// Painless via `crate::script` (compiled once, cached there); `None`
    /// when it cannot compile the source (the lite subset is tried then).
    fn painless(&self, ctx: crate::script::ScriptContext) -> Option<std::result::Result<crate::script::CompiledScript, EsError>> {
        match self.compile_source(ctx) {
            Ok(c) => Some(Ok(c)),
            Err(e) => {
                if self.raw_script.is_none() && self.lite.is_ok() {
                    None
                } else {
                    Some(Err(e))
                }
            }
        }
    }

    fn run_painless(cs: &crate::script::CompiledScript, ctx: &mut Value) -> Result<Value> {
        let mut exec = cs.executor(None, None);
        let v = crate::script::Val::from_json(ctx);
        exec.set_ctx(v.clone());
        let r = exec.run().map_err(|e| cs.error(&e))?;
        // A self-referencing ctx (`ctx.a = ctx`) cannot become a document.
        if self_referencing(&v, &mut Vec::new()) {
            return Err(EsError::illegal_argument(format!("Failed to generate the source document for ingest pipeline [{}]", INGEST_PIPELINE.with(|p| p.borrow().clone()))));
        }
        *ctx = v.to_json();
        Ok(r.to_json())
    }

    pub fn execute(&self, doc: &mut IngestDocument) -> Result<()> {
        if let Some(e) = SCRIPT_ENGINE.get() {
            let e = e.clone();
            return Self::with_ctx(doc, |c| e.execute(&self.source, &self.lang, &self.params, c));
        }
        if let Some(cs) = self.painless(crate::script::ScriptContext::Ingest) {
            let cs = cs?;
            return Self::with_ctx(doc, |c| Self::run_painless(&cs, c).map(|_| ()));
        }
        let s = self.lite.as_ref().map_err(Clone::clone)?;
        Self::with_ctx(doc, |c| s.run(c, &self.params).map(|_| ()))
    }

    pub fn condition(&self, doc: &mut IngestDocument) -> Result<bool> {
        if let Some(e) = SCRIPT_ENGINE.get() {
            let e = e.clone();
            return Self::with_ctx(doc, |c| e.condition(&self.source, &self.lang, &self.params, c));
        }
        if let Some(cs) = self.painless(crate::script::ScriptContext::IngestConditional) {
            let cs = cs?;
            return Self::with_ctx(doc, |c| {
                let mut exec = cs.executor(None, None);
                let v = crate::script::Val::from_json(c);
                exec.set_ctx(v.clone());
                let b = exec.run_bool().map_err(|e| cs.error(&e))?;
                *c = v.to_json();
                Ok(b)
            });
        }
        let s = self.lite.as_ref().map_err(Clone::clone)?;
        Self::with_ctx(doc, |c| s.condition(c, &self.params))
    }

    /// Evaluate as an expression returning a value (calculation processor).
    pub fn eval_value(&self, doc: &mut IngestDocument) -> Result<Value> {
        let s = self.lite.as_ref().map_err(Clone::clone)?;
        Self::with_ctx(doc, |c| s.run(c, &self.params))
    }
}

thread_local! {
    /// The pipeline running the current script (for error messages).
    static INGEST_PIPELINE: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Whether a script value graph contains a reference cycle.
fn self_referencing(v: &crate::script::Val, path: &mut Vec<*const ()>) -> bool {
    use crate::script::Val;
    let (ptr, children): (*const (), Vec<Val>) = match v {
        Val::Map(m) => (std::rc::Rc::as_ptr(m) as *const (), m.borrow().values().cloned().collect()),
        Val::List(l) | Val::Set(l) | Val::Array { items: l, .. } => (std::rc::Rc::as_ptr(l) as *const (), l.borrow().clone()),
        _ => return false,
    };
    if path.contains(&ptr) {
        return true;
    }
    path.push(ptr);
    let cyc = children.iter().any(|c| self_referencing(c, path));
    path.pop();
    cyc
}

// ------------------------------------------------------------ config

/// Reads a processor's config map, tracking unknown keys.
pub struct Cfg {
    pub kind: String,
    pub tag: Option<String>,
    pub m: Map<String, Value>,
    pub raw: Option<crate::source_json::Source>,
}

pub fn config_error(kind: &str, tag: Option<&str>, prop: Option<&str>, reason: impl Into<String>) -> EsError {
    let reason = reason.into();
    let msg = match prop {
        Some(p) => format!("[{p}] {reason}"),
        None => reason,
    };
    let mut e = EsError::new(400, "parse_exception", msg);
    if let Some(p) = prop {
        e = e.with("property_name", p);
    }
    e = e.with("processor_type", kind);
    if let Some(t) = tag {
        e = e.with("processor_tag", t);
    }
    e
}

impl Cfg {
    pub fn err(&self, prop: &str, reason: impl Into<String>) -> EsError {
        config_error(&self.kind, self.tag.as_deref(), Some(prop), reason)
    }

    pub fn has(&self, k: &str) -> bool {
        self.m.contains_key(k)
    }

    pub fn opt_value(&mut self, k: &str) -> Option<Value> {
        self.m.shift_remove(k)
    }

    pub fn req_value(&mut self, k: &str) -> Result<Value> {
        match self.m.shift_remove(k) {
            Some(v) if !v.is_null() => Ok(v),
            _ => Err(self.err(k, "required property is missing")),
        }
    }

    pub fn opt_str(&mut self, k: &str) -> Result<Option<String>> {
        match self.m.shift_remove(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(other) => Err(self.err(k, format!("property isn't a string, but of type [{}]", doc::type_name(&other)))),
        }
    }

    pub fn req_str(&mut self, k: &str) -> Result<String> {
        match self.opt_str(k)? {
            Some(s) => Ok(s),
            None => Err(self.err(k, "required property is missing")),
        }
    }

    pub fn str_or(&mut self, k: &str, d: &str) -> Result<String> {
        Ok(self.opt_str(k)?.unwrap_or_else(|| d.to_string()))
    }

    pub fn bool_or(&mut self, k: &str, d: bool) -> Result<bool> {
        match self.m.shift_remove(k) {
            None | Some(Value::Null) => Ok(d),
            Some(Value::Bool(b)) => Ok(b),
            Some(Value::String(s)) if s == "true" || s == "false" => Ok(s == "true"),
            Some(other) => Err(self.err(k, format!("property isn't a boolean, but of type [{}]", doc::type_name(&other)))),
        }
    }

    pub fn opt_i64(&mut self, k: &str) -> Result<Option<i64>> {
        match self.m.shift_remove(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => Ok(n.as_i64().or(n.as_f64().map(|f| f as i64))),
            Some(Value::String(s)) => s.trim().parse().map(Some).map_err(|_| self.err(k, format!("property cannot be converted to an int [{s}]"))),
            Some(other) => Err(self.err(k, format!("property isn't a number, but of type [{}]", doc::type_name(&other)))),
        }
    }

    pub fn opt_list_str(&mut self, k: &str) -> Result<Option<Vec<String>>> {
        match self.m.shift_remove(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Array(a)) => {
                let mut out = Vec::new();
                for x in a {
                    match x {
                        Value::String(s) => out.push(s),
                        other => out.push(doc::value_to_string(&other)),
                    }
                }
                Ok(Some(out))
            }
            Some(other) => Err(self.err(k, format!("property isn't a list, but of type [{}]", doc::type_name(&other)))),
        }
    }

    pub fn req_list_str(&mut self, k: &str) -> Result<Vec<String>> {
        match self.opt_list_str(k)? {
            Some(v) => Ok(v),
            None => Err(self.err(k, "required property is missing")),
        }
    }

    /// A string or a list of strings.
    pub fn str_or_list(&mut self, k: &str) -> Result<Option<Vec<String>>> {
        match self.m.get(k) {
            Some(Value::String(_)) => Ok(self.opt_str(k)?.map(|s| vec![s])),
            _ => self.opt_list_str(k),
        }
    }

    pub fn opt_map(&mut self, k: &str) -> Result<Option<Map<String, Value>>> {
        match self.m.shift_remove(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Object(m)) => Ok(Some(m)),
            Some(other) => Err(self.err(k, format!("property isn't a map, but of type [{}]", doc::type_name(&other)))),
        }
    }

    /// Accept and ignore keys.
    pub fn ignore(&mut self, keys: &[&str]) {
        for k in keys {
            self.m.shift_remove(*k);
        }
    }

    pub fn finish(self) -> Result<()> {
        if self.m.is_empty() {
            return Ok(());
        }
        let keys: Vec<&str> = self.m.keys().map(String::as_str).collect();
        Err(config_error(&self.kind, self.tag.as_deref(), None, format!("processor [{}] doesn't support one or more provided configuration parameters {:?}", self.kind, keys).replace('"', "")))
    }
}

// ------------------------------------------------------------ pipelines

pub struct ProcessorNode {
    pub source_config: crate::source_json::Source,
    pub kind: String,
    pub tag: Option<String>,
    pub description: Option<String>,
    pub cond: Option<Script>,
    pub ignore_failure: bool,
    pub on_failure: Vec<ProcessorNode>,
    pub imp: Box<dyn Processor>,
    /// Execution counters (`_nodes/stats` ingest processor stats).
    pub stats: IngestMetric,
    /// A `script` processor's inline script, built from its configuration
    /// once (`None`: a stored script, read at each run as it may change,
    /// or a configuration whose error each run reports).
    pub script: std::sync::OnceLock<Option<Script>>,
}

pub struct Pipeline {
    pub id: String,
    pub description: Option<String>,
    pub version: Option<i64>,
    pub processors: Vec<ProcessorNode>,
    pub on_failure: Vec<ProcessorNode>,
    /// Execution counters (`_nodes/stats` ingest pipeline stats).
    pub stats: IngestMetric,
    /// Source bytes of documents this pipeline ran on first / produced.
    pub ingested_bytes: std::sync::atomic::AtomicU64,
    pub produced_bytes: std::sync::atomic::AtomicU64,
}

/// ES `IngestMetric`: executions, time, in-flight and failed counts.
#[derive(Default, Debug)]
pub struct IngestMetric {
    pub count: std::sync::atomic::AtomicU64,
    pub time_nanos: std::sync::atomic::AtomicU64,
    pub current: std::sync::atomic::AtomicU64,
    pub failed: std::sync::atomic::AtomicU64,
}

impl IngestMetric {
    fn pre(&self) -> std::time::Instant {
        self.current.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::time::Instant::now()
    }

    fn post(&self, start: std::time::Instant, failed: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        self.current.fetch_sub(1, Relaxed);
        self.count.fetch_add(1, Relaxed);
        self.time_nanos.fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
        if failed {
            self.failed.fetch_add(1, Relaxed);
        }
    }

    /// `{count, time_in_millis, current, failed}`.
    pub fn to_json(&self) -> Value {
        use std::sync::atomic::Ordering::Relaxed;
        json!({"count": self.count.load(Relaxed), "time_in_millis": self.time_nanos.load(Relaxed) / 1_000_000, "current": self.current.load(Relaxed), "failed": self.failed.load(Relaxed)})
    }
}

const PIPELINE_KEYS: &[&str] = &["description", "processors", "on_failure", "version", "_meta", "deprecated", "created_date_millis", "modified_date_millis", "field_access_pattern"];

/// `IngestService.validateNoSystemPropertiesInPipelineConfig`: the tracking
/// timestamps are set by the node, never by the request.
pub fn validate_no_system_properties(config: &Value) -> Result<()> {
    for k in ["created_date_millis", "created_date", "modified_date_millis", "modified_date"] {
        if config.get(k).is_some() {
            return Err(EsError::new(400, "parse_exception", format!("Provided a pipeline property which is managed by the system: {k}.")));
        }
    }
    Ok(())
}

/// Add `processor_type` (and `processor_tag`) to a definition-time error,
/// ahead of the error's own metadata like ES's header ordering.
pub(super) fn with_processor_meta(mut e: EsError, kind: &str, tag: Option<&str>) -> EsError {
    let mut front: Vec<(String, Value)> = vec![("processor_type".into(), json!(kind))];
    if let Some(t) = tag {
        front.push(("processor_tag".into(), json!(t)));
    }
    if e.extra.iter().any(|(k, _)| k == "processor_type") {
        return e;
    }
    front.extend(std::mem::take(&mut e.extra));
    e.extra = front;
    e
}

/// Compile a list of processor entries; an entry map with several keys yields
/// one processor per key (ES `readProcessorConfigs`).
pub fn compile_processor_list(entries: &[Value]) -> Result<Vec<ProcessorNode>> {
    compile_processor_list_source(entries, None)
}

fn compile_processor_list_source(entries: &[Value], raw: Option<&crate::source_json::Source>) -> Result<Vec<ProcessorNode>> {
    let mut out = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let raw_entry = match raw { Some(crate::source_json::Source::Array(a)) => a.get(i), _ => None };
        let o = entry.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "processor must be an object"))?;
        for (k, v) in o {
            let mut single = Map::new();
            single.insert(k.clone(), v.clone());
            out.push(compile_processor_source(&Value::Object(single), raw_entry)?);
        }
    }
    Ok(out)
}

/// Compile one `{"type": {config}}` entry.
pub fn compile_processor(entry: &Value) -> Result<ProcessorNode> {
    compile_processor_source(entry, None)
}

pub(super) fn compile_processor_source(entry: &Value, raw: Option<&crate::source_json::Source>) -> Result<ProcessorNode> {
    let o = entry.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "processor must be an object"))?;
    if o.len() != 1 {
        return Err(EsError::new(400, "parse_exception", format!("processor definition must contain exactly one processor type, found [{}]", o.len())));
    }
    let (kind, conf) = o.iter().next().unwrap();
    let raw = raw.and_then(|v| v.get(kind));
    let mut m = match conf {
        Value::Object(m) => m.clone(),
        Value::String(s) if kind == "script" => {
            let mut m = Map::new();
            m.insert("source".into(), Value::String(s.clone()));
            m
        }
        _ => return Err(config_error(kind, None, None, format!("processor [{kind}] config must be an object"))),
    };
    let tag = m.shift_remove("tag").and_then(|t| t.as_str().map(str::to_string));
    let mut cfg = Cfg { kind: kind.clone(), tag, m: Map::new(), raw: raw.cloned() };
    let description = m.shift_remove("description").and_then(|t| t.as_str().map(str::to_string));
    let cond = match m.shift_remove("if") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let sc = match raw.and_then(|r|r.get("if")) {
                Some(r) => Script::from_source(r),
                None => Script::from_config(&v),
            }.map_err(|e| with_processor_meta(e, kind, cfg.tag.as_deref()))?;
            sc.validate(crate::script::ScriptContext::IngestConditional).map_err(|e| with_processor_meta(e, kind, cfg.tag.as_deref()))?;
            Some(sc)
        }
    };
    let ignore_failure = match m.shift_remove("ignore_failure") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => b,
        Some(Value::String(s)) => s == "true",
        Some(other) => return Err(cfg.err("ignore_failure", format!("property isn't a boolean, but of type [{}]", doc::type_name(&other)))),
    };
    let on_failure = match m.shift_remove("on_failure") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => {
            if a.is_empty() {
                return Err(cfg.err("on_failure", "processors list cannot be empty"));
            }
            compile_processor_list_source(&a, raw.and_then(|r|r.get("on_failure")))?
        }
        Some(_) => return Err(cfg.err("on_failure", "property isn't a list")),
    };
    cfg.m = m;
    let imp = processors::build(&mut cfg)?;
    let tag = cfg.tag.clone();
    cfg.finish()?;
    Ok(ProcessorNode { source_config:raw.cloned().unwrap_or_else(||crate::source_json::Source::from_value(conf)), kind: kind.clone(), tag, description, cond, ignore_failure, on_failure, imp, stats: IngestMetric::default(), script: std::sync::OnceLock::new() })
}

/// Compile a pipeline definition.
pub fn compile_pipeline(id: &str, config: &Value) -> Result<Pipeline> {
    compile_pipeline_source_inner(id, config, None)
}

/// Compile pipeline scripts without decoding Java string values as UTF-8.
pub fn compile_pipeline_source(id: &str, raw: &crate::source_json::Source) -> Result<Pipeline> {
    let metadata = source_config::metadata(raw)?;
    compile_pipeline_source_inner(id, &metadata, Some(raw))
}

fn compile_pipeline_source_inner(id: &str, config: &Value, raw: Option<&crate::source_json::Source>) -> Result<Pipeline> {
    let o = config.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "[processors] required property is missing").with("property_name", "processors"))?;
    let unknown: Vec<&str> = o.keys().filter(|k| !PIPELINE_KEYS.contains(&k.as_str())).map(String::as_str).collect();
    if !unknown.is_empty() {
        return Err(EsError::new(400, "parse_exception", format!("pipeline [{id}] doesn't support one or more provided configuration parameters [{}]", unknown.join(", "))));
    }
    let procs = match o.get("processors") {
        None | Some(Value::Null) => return Err(EsError::new(400, "parse_exception", "[processors] required property is missing").with("property_name", "processors")),
        Some(Value::Array(a)) => a,
        Some(other) => return Err(EsError::new(400, "parse_exception", format!("[processors] property isn't a list, but of type [{}]", doc::type_name(other))).with("property_name", "processors")),
    };
    let processors = compile_processor_list_source(procs, raw.and_then(|r|r.get("processors")))?;
    let on_failure = match o.get("on_failure") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => {
            if a.is_empty() {
                return Err(EsError::new(400, "parse_exception", "pipeline [".to_string() + id + "] cannot have an empty on_failure option defined"));
            }
            compile_processor_list_source(a, raw.and_then(|r|r.get("on_failure")))?
        }
        Some(other) => return Err(EsError::new(400, "parse_exception", format!("[on_failure] property isn't a list, but of type [{}]", doc::type_name(other))).with("property_name", "on_failure")),
    };
    match o.get("description") {
        None | Some(Value::Null) | Some(Value::String(_)) => {}
        Some(other) => return Err(EsError::new(400, "parse_exception", format!("[description] property isn't a string, but of type [{}]", doc::type_name(other))).with("property_name", "description")),
    }
    match o.get("deprecated") {
        None | Some(Value::Null) | Some(Value::Bool(_)) => {}
        Some(other) => return Err(EsError::new(400, "parse_exception", format!("[deprecated] property isn't a boolean, but of type [{}]", doc::type_name(other))).with("property_name", "deprecated")),
    }
    let bad_version = |txt: String| EsError::new(400, "parse_exception", format!("[version] property cannot be converted to an int [{txt}]")).with("property_name", "version");
    let version = match o.get("version") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(0.0);
            if f.fract() == 0.0 && f.abs() < 2147483648.0 { Some(f as i64) } else { return Err(bad_version(jtext::java_double(f))) }
        }
        Some(Value::Number(n)) => match n.as_i64() {
            Some(i) if i >= i32::MIN as i64 && i <= i32::MAX as i64 => Some(i),
            _ => return Err(bad_version(n.to_string())),
        },
        Some(Value::String(s)) => Some(s.trim().parse::<i32>().map(|v| v as i64).map_err(|_| bad_version(s.clone()))?),
        Some(other) => return Err(EsError::new(400, "parse_exception", format!("[version] property isn't a number, but of type [{}]", doc::type_name(other))).with("property_name", "version")),
    };
    if let Some(m) = o.get("_meta") {
        if !m.is_object() && !m.is_null() {
            return Err(EsError::new(400, "parse_exception", format!("[_meta] property isn't a map, but of type [{}]", doc::type_name(m))).with("property_name", "_meta"));
        }
    }
    Ok(Pipeline { id: id.to_string(), description: o.get("description").and_then(Value::as_str).map(str::to_string), version, processors, on_failure, stats: IngestMetric::default(), ingested_bytes: Default::default(), produced_bytes: Default::default() })
}

// ------------------------------------------------------------ execution

/// Per-execution state: pipeline lookup (with simulate substitutions),
/// the call stack (cycle detection), verbose trace, executed pipelines.
pub struct PipelineContext<'a> {
    pub store: &'a IngestService,
    pub substitutions: Option<&'a HashMap<String, Arc<Pipeline>>>,
    pub stack: RefCell<Vec<String>>,
    pub trace: Option<RefCell<Vec<Value>>>,
    pub executed: RefCell<Vec<String>>,
    /// The authenticated user (`set_security_user`); see [`IngestContext`].
    pub user: Option<crate::security::Authn>,
    /// While > 0, processors do not add verbose trace entries (foreach bodies).
    pub quiet: std::cell::Cell<usize>,
    /// The pipeline stack when the current failure was raised (failure store `pipeline_trace`).
    pub fail_stack: RefCell<Option<Vec<String>>>,
    /// (processor type, tag, pipeline) of the failure that ended the request.
    pub fail_origin: RefCell<Option<(String, Option<String>, String)>>,
    /// The index whose failure store receives a failure: the request's
    /// target, or the latest reroute destination.
    pub fs_index: RefCell<Option<String>>,
    /// Count pipeline / processor stats (real indexing, not simulate).
    pub count_stats: std::cell::Cell<bool>,
}

/// Per-request ingest context supplied by the caller (the HTTP layer enters a
/// guard around `run_ingest` / simulate on the request thread).
#[derive(Clone, Debug, Default)]
pub struct IngestContext {
    pub user: Option<crate::security::Authn>,
}

thread_local! {
    static INGEST_CTX: RefCell<IngestContext> = RefCell::new(IngestContext::default());
}

/// Restores the previous [`IngestContext`] when dropped.
pub struct IngestContextGuard(IngestContext);

impl Drop for IngestContextGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.0);
        INGEST_CTX.with(|c| *c.borrow_mut() = prev);
    }
}

impl IngestContext {
    /// Make `ctx` the current ingest context on this thread until the guard drops.
    pub fn enter(ctx: IngestContext) -> IngestContextGuard {
        IngestContextGuard(INGEST_CTX.with(|c| c.replace(ctx)))
    }
    pub fn current() -> IngestContext {
        INGEST_CTX.with(|c| c.borrow().clone())
    }
}

impl<'a> PipelineContext<'a> {
    pub fn new(store: &'a IngestService) -> PipelineContext<'a> {
        PipelineContext { store, substitutions: None, stack: RefCell::new(Vec::new()), trace: None, executed: RefCell::new(Vec::new()), user: IngestContext::current().user, quiet: std::cell::Cell::new(0), fail_stack: RefCell::new(None), fail_origin: RefCell::new(None), fs_index: RefCell::new(None), count_stats: std::cell::Cell::new(false) }
    }

    pub fn pipeline(&self, id: &str) -> Option<Arc<Pipeline>> {
        if let Some(s) = self.substitutions {
            if let Some(p) = s.get(id) {
                return Some(p.clone());
            }
        }
        self.store.pipeline(id)
    }

    pub fn current_pipeline(&self) -> String {
        self.stack.borrow().last().cloned().unwrap_or_default()
    }

    fn record(&self, v: Value) -> Option<usize> {
        if self.quiet.get() > 0 {
            return None;
        }
        self.trace.as_ref().map(|t| {
            let mut t = t.borrow_mut();
            t.push(v);
            t.len() - 1
        })
    }
}

fn innermost_reason(e: &EsError) -> String {
    let mut cur = e;
    if cur.kind == "script_exception" {
        while let Some(c) = &cur.caused_by {
            cur = c;
        }
    }
    cur.reason.clone()
}

/// Error JSON without `root_cause` (bulk item / simulate shape).
pub fn error_body(e: &EsError) -> Value {
    let mut v = e.error_object(false);
    if let Some(o) = v.as_object_mut() {
        o.remove("root_cause");
    }
    v
}

/// `_index/_id/_version/_routing/_source/_ingest` rendering of simulate docs.
pub fn simulate_doc_json(d: &IngestDocument) -> Value {
    let mut m = Map::new();
    let ctx = d.ctx_map();
    for k in ["_index", "_version", "_id", "_routing", "_version_type", "_if_seq_no", "_if_primary_term"] {
        if let Some(v) = ctx.get(k) {
            if v.is_null() {
                continue;
            }
            let v = match (k, v) {
                ("_version", Value::Number(n)) => Value::String(n.to_string()),
                _ => v.clone(),
            };
            m.insert(k.into(), v);
        }
    }
    m.insert("_source".into(), d.source_value());
    m.insert("_ingest".into(), d.ingest.clone());
    Value::Object(m)
}

fn trace_base(n: &ProcessorNode, cond: Option<(&Script, bool)>) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("processor_type".into(), json!(n.kind));
    m.insert("status".into(), json!("success"));
    if let Some(d) = &n.description {
        m.insert("description".into(), json!(d));
    }
    if let Some(t) = &n.tag {
        m.insert("tag".into(), json!(t));
    }
    if let Some((s, r)) = cond {
        m.insert("if".into(), json!({"condition": s.source, "result": r}));
    }
    m
}

fn put_failure_meta(doc: &mut IngestDocument, e: &EsError, origin: &(String, Option<String>, String)) {
    let m = doc.ingest_map_mut();
    m.insert("on_failure_message".into(), json!(innermost_reason(e)));
    m.insert("on_failure_processor_type".into(), json!(origin.0));
    m.insert("on_failure_processor_tag".into(), origin.1.clone().map(Value::String).unwrap_or(Value::Null));
    m.insert("on_failure_pipeline".into(), json!(origin.2));
}

fn remove_failure_meta(doc: &mut IngestDocument, prev: Vec<(String, Option<Value>)>) {
    let m = doc.ingest_map_mut();
    for (k, v) in prev {
        match v {
            Some(v) => {
                m.insert(k, v);
            }
            None => {
                m.shift_remove(&k);
            }
        }
    }
}

fn snapshot_failure_meta(doc: &IngestDocument) -> Vec<(String, Option<Value>)> {
    ["on_failure_message", "on_failure_processor_type", "on_failure_processor_tag", "on_failure_pipeline"]
        .iter()
        .map(|k| (k.to_string(), doc.ingest.get(*k).cloned()))
        .collect()
}

pub(crate) fn run_node(n: &ProcessorNode, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
    if !cx.count_stats.get() {
        return run_node_inner(n, doc, cx);
    }
    let t = n.stats.pre();
    let r = run_node_inner(n, doc, cx);
    n.stats.post(t, matches!(r, Err(ProcessorError::Failed(..))));
    r
}

fn run_node_inner(n: &ProcessorNode, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
    let mut cond_res = None;
    if let Some(c) = &n.cond {
        let r = match c.condition(doc) {
            Ok(r) => r,
            Err(e) => {
                // A failing condition fails the processor.
                return handle_failure(n, doc, cx, e, None, None);
            }
        };
        if !r {
            if cx.trace.is_some() {
                let mut m = trace_base(n, Some((c, false)));
                m.insert("status".into(), json!("skipped"));
                cx.record(Value::Object(m));
            }
            return Ok(());
        }
        cond_res = Some((c, true));
    }
    // Pipeline processors record before their nested processors.
    let slot = if n.kind == "pipeline" { cx.record(Value::Object(trace_base(n, cond_res))) } else { None };
    match n.imp.process(doc, cx) {
        Ok(()) => {
            if cx.trace.is_some() && n.kind != "pipeline" {
                let mut m = trace_base(n, cond_res);
                m.insert("doc".into(), simulate_doc_json(doc));
                cx.record(Value::Object(m));
            }
            Ok(())
        }
        Err(ProcessorError::Dropped) => {
            if cx.trace.is_some() && n.kind != "pipeline" {
                let mut m = trace_base(n, cond_res);
                m.insert("status".into(), json!("dropped"));
                cx.record(Value::Object(m));
            }
            Err(ProcessorError::Dropped)
        }
        Err(ProcessorError::Terminated) => {
            if cx.trace.is_some() && n.kind != "pipeline" {
                let mut m = trace_base(n, cond_res);
                m.insert("doc".into(), simulate_doc_json(doc));
                cx.record(Value::Object(m));
            }
            Err(ProcessorError::Terminated)
        }
        Err(ProcessorError::Failed(e, origin)) => handle_failure(n, doc, cx, e, origin, slot.map(|s| (s, cond_res.map(|c| c.1)))),
    }
}

fn handle_failure(
    n: &ProcessorNode,
    doc: &mut IngestDocument,
    cx: &PipelineContext,
    e: EsError,
    origin: Option<(String, Option<String>, String)>,
    slot: Option<(usize, Option<bool>)>,
) -> PResult {
    let origin = origin.unwrap_or_else(|| (n.kind.clone(), n.tag.clone(), cx.current_pipeline()));
    if cx.trace.is_some() {
        let cond = n.cond.as_ref().map(|c| (c, true));
        let mut m = trace_base(n, cond);
        if n.ignore_failure {
            m.insert("status".into(), json!("error_ignored"));
            m.insert("ignored_error".into(), json!({"error": e.error_object(false)}));
            m.insert("doc".into(), simulate_doc_json(doc));
        } else {
            m.insert("status".into(), json!("error"));
            m.insert("error".into(), e.error_object(false));
        }
        match slot {
            Some((i, _)) => {
                if let Some(t) = &cx.trace {
                    t.borrow_mut()[i] = Value::Object(m);
                }
            }
            None => {
                cx.record(Value::Object(m));
            }
        }
    }
    if n.ignore_failure {
        *cx.fail_stack.borrow_mut() = None;
        return Ok(());
    }
    if n.on_failure.is_empty() {
        return Err(ProcessorError::Failed(e, Some(origin)));
    }
    *cx.fail_stack.borrow_mut() = None;
    let prev = snapshot_failure_meta(doc);
    put_failure_meta(doc, &e, &origin);
    let r = run_nodes(&n.on_failure, doc, cx);
    remove_failure_meta(doc, prev);
    r
}

fn run_nodes(nodes: &[ProcessorNode], doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
    for n in nodes {
        run_node(n, doc, cx)?;
        if doc.rerouted {
            // reroute halts this pipeline and every enclosing one
            return Err(ProcessorError::Terminated);
        }
    }
    Ok(())
}

/// Run a pipeline (also used by the `pipeline` processor).
pub fn run_pipeline(p: &Pipeline, doc: &mut IngestDocument, cx: &PipelineContext) -> PResult {
    {
        let st = cx.stack.borrow();
        if st.iter().any(|x| x == &p.id) {
            // IngestDocument.executePipeline: GraphStructureException
            return Err(ProcessorError::Failed(EsError::new(500, "graph_structure_exception", format!("Cycle detected for pipeline: {}", p.id)), None));
        }
        if st.len() >= 100 {
            return Err(ProcessorError::Failed(EsError::illegal_state("Too many nested pipelines. Cannot have more than 100 nested pipelines"), None));
        }
    }
    let timer = if cx.count_stats.get() { Some(p.stats.pre()) } else { None };
    cx.stack.borrow_mut().push(p.id.clone());
    cx.executed.borrow_mut().push(p.id.clone());
    let prev_pipeline = INGEST_PIPELINE.with(|c| c.replace(p.id.clone()));
    let prev = doc.ingest.get("pipeline").cloned();
    doc.ingest_map_mut().insert("pipeline".into(), json!(p.id));
    let r = match run_nodes(&p.processors, doc, cx) {
        Err(ProcessorError::Failed(e, origin)) if !p.on_failure.is_empty() => {
            let origin = origin.unwrap_or_else(|| ("pipeline".into(), None, p.id.clone()));
            let prevm = snapshot_failure_meta(doc);
            put_failure_meta(doc, &e, &origin);
            *cx.fail_stack.borrow_mut() = None;
            let r = run_nodes(&p.on_failure, doc, cx);
            remove_failure_meta(doc, prevm);
            r
        }
        Err(ProcessorError::Terminated) => Ok(()),
        other => other,
    };
    if matches!(r, Err(ProcessorError::Failed(..))) && cx.fail_stack.borrow().is_none() {
        *cx.fail_stack.borrow_mut() = Some(cx.stack.borrow().clone());
    }
    match prev {
        Some(v) => {
            doc.ingest_map_mut().insert("pipeline".into(), v);
        }
        None => {
            doc.ingest_map_mut().shift_remove("pipeline");
        }
    }
    cx.stack.borrow_mut().pop();
    INGEST_PIPELINE.with(|c| *c.borrow_mut() = prev_pipeline);
    if let Some(t) = timer {
        p.stats.post(t, matches!(r, Err(ProcessorError::Failed(..))));
    }
    r
}

trait IllegalState {
    fn illegal_state(m: impl Into<String>) -> EsError;
}
impl IllegalState for EsError {
    fn illegal_state(m: impl Into<String>) -> EsError {
        EsError::new(500, "illegal_state_exception", m)
    }
}

// ------------------------------------------------------------ store

struct Stored {
    config: Value,
    compiled: std::result::Result<Arc<Pipeline>, EsError>,
}

/// The node's pipelines, persisted in `_state/ingest.json`.
pub struct IngestService {
    path: PathBuf,
    pipelines: RwLock<BTreeMap<String, Stored>>,
    /// Documents run through the ingest service (`_nodes/stats` `ingest.total`).
    pub total: IngestMetric,
}

impl IngestService {
    pub fn open(data: &Path) -> IngestService {
        let path = data.join("_state").join("ingest.json");
        let mut map = BTreeMap::new();
        if let Ok(b) = std::fs::read(&path) {
            if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(&b) {
                for (id, cfg) in m {
                    let compiled = compile_pipeline(&id, &cfg).map(Arc::new);
                    if let Err(e) = &compiled {
                        tracing::warn!("stored pipeline [{id}] failed to compile: {e}");
                    }
                    map.insert(id, Stored { config: cfg, compiled });
                }
            }
        }
        IngestService { path, pipelines: RwLock::new(map), total: IngestMetric::default() }
    }

    /// An in-memory store (tests, simulate).
    pub fn in_memory() -> IngestService {
        IngestService { path: PathBuf::new(), pipelines: RwLock::new(BTreeMap::new()), total: IngestMetric::default() }
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.read().is_empty()
    }

    fn persist(&self, map: &BTreeMap<String, Stored>) -> Result<()> {
        if self.path.as_os_str().is_empty() {
            return Ok(());
        }
        let v: Map<String, Value> = map.iter().map(|(k, s)| (k.clone(), s.config.clone())).collect();
        let tmp = self.path.with_extension("json.tmp");
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&tmp, serde_json::to_vec(&Value::Object(v))?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    /// Create or replace a pipeline (validated by compiling it).
    pub fn put(&self, id: &str, config: Value, if_version: Option<i64>) -> Result<()> {
        validate_no_system_properties(&config)?;
        let compiled = compile_pipeline(id, &config)?;
        let mut map = self.pipelines.write();
        // `IngestService.isNoOpPipelineUpdate`: the same definition (system
        // properties aside) leaves the pipeline, and its dates, untouched.
        if if_version.is_none() {
            if let Some(cur) = map.get(id) {
                let mut bare = cur.config.clone();
                if let Some(o) = bare.as_object_mut() {
                    o.remove("created_date_millis");
                    o.remove("modified_date_millis");
                }
                if bare == config {
                    return Ok(());
                }
            }
        }
        if let Some(req) = if_version {
            match map.get(id) {
                None => return Err(EsError::illegal_argument(format!("version conflict, required version [{req}] for pipeline [{id}] but no pipeline was found"))),
                Some(s) => {
                    let ver_of = |v: &Value| v.as_i64().or_else(|| v.as_str().and_then(|x| x.trim().parse().ok()));
                    let cur = s.config.get("version").and_then(ver_of);
                    if cur != Some(req) {
                        return Err(EsError::illegal_argument(format!(
                            "version conflict, required version [{req}] for pipeline [{id}] but current version is [{}]",
                            cur.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
                        )));
                    }
                    let nv = config.get("version").and_then(ver_of);
                    if nv == Some(req) {
                        return Err(EsError::illegal_argument(format!("cannot update pipeline [{id}] with the same version [{req}]")));
                    }
                }
            }
        }
        let mut config = config;
        if if_version.is_some() && config.get("version").map(Value::is_null).unwrap_or(true) {
            if let Some(o) = config.as_object_mut() {
                o.insert("version".into(), json!(if_version.unwrap() + 1));
            }
        }
        // Tracking info: `created_date_millis` is kept from the existing
        // pipeline (only when it has one), `modified_date_millis` is now.
        let now = crate::date::now_millis();
        let created = match map.get(id) {
            None => Some(json!(now)),
            Some(cur) => cur.config.get("created_date_millis").cloned(),
        };
        if let Some(o) = config.as_object_mut() {
            if let Some(c) = created {
                o.insert("created_date_millis".into(), c);
            }
            o.insert("modified_date_millis".into(), json!(now));
        }
        map.insert(id.to_string(), Stored { config, compiled: Ok(Arc::new(compiled)) });
        self.persist(&map)
    }

    /// Restore a pipeline from trusted cluster metadata, preserving the
    /// system-managed dates assigned by the master. Public writes use `put`.
    pub fn restore_from_cluster_state(&self, id: &str, config: Value) -> Result<()> {
        let compiled = compile_pipeline(id, &config)?;
        let mut map = self.pipelines.write();
        map.insert(id.to_string(), Stored { config, compiled: Ok(Arc::new(compiled)) });
        self.persist(&map)
    }

    /// Install built-in pipelines (template registries) in one write. A
    /// pipeline that does not compile is still stored (and fails when used).
    pub fn put_builtin(&self, pipelines: Vec<(String, Value)>) {
        let mut map = self.pipelines.write();
        for (id, config) in pipelines {
            let compiled = compile_pipeline(&id, &config).map(Arc::new);
            if let Err(e) = &compiled {
                tracing::warn!("built-in pipeline [{id}] failed to compile: {e}");
            }
            map.insert(id, Stored { config, compiled });
        }
        if let Err(e) = self.persist(&map) {
            tracing::warn!("persisting built-in pipelines failed: {e}");
        }
    }

    /// Pipeline configs matching a comma list of ids / wildcards (all when `None`).
    pub fn get(&self, ids: Option<&str>) -> Vec<(String, Value)> {
        let map = self.pipelines.read();
        match ids {
            None => map.iter().map(|(k, s)| (k.clone(), s.config.clone())).collect(),
            Some(expr) => {
                let pats: Vec<&str> = expr.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
                if pats.iter().any(|p| *p == "*") {
                    return map.iter().map(|(k, s)| (k.clone(), s.config.clone())).collect();
                }
                map.iter().filter(|(k, _)| pats.iter().any(|p| crate::mapping::wildcard_match(p, k))).map(|(k, s)| (k.clone(), s.config.clone())).collect()
            }
        }
    }

    /// Ids matching `expr` (wildcards).
    pub fn matching(&self, expr: &str) -> Vec<String> {
        self.get(Some(expr)).into_iter().map(|(k, _)| k).collect()
    }

    /// Ids matching the single delete pattern `expr` (ES does not split DELETE ids on commas).
    pub fn matching_delete(&self, expr: &str) -> Vec<String> {
        self.pipelines.read().keys().filter(|k| crate::mapping::wildcard_match(expr, k)).cloned().collect()
    }

    /// Delete pipelines matching the pattern; no match is a 404.
    pub fn delete(&self, expr: &str) -> Result<Vec<String>> {
        let ids = self.matching_delete(expr);
        if ids.is_empty() {
            return Err(EsError::resource_not_found(format!("pipeline [{expr}] is missing")));
        }
        let mut map = self.pipelines.write();
        for id in &ids {
            map.remove(id);
        }
        self.persist(&map)?;
        Ok(ids)
    }

    /// Compiled pipeline by id.
    pub fn pipeline(&self, id: &str) -> Option<Arc<Pipeline>> {
        self.pipelines.read().get(id).and_then(|s| s.compiled.as_ref().ok().cloned())
    }

    pub fn exists(&self, id: &str) -> bool {
        self.pipelines.read().contains_key(id)
    }

    pub fn count(&self) -> usize {
        self.pipelines.read().len()
    }

    /// `_nodes/stats` `ingest` section: node totals plus every pipeline's
    /// counters and its top-level processors' counters (`type` or `type:tag`).
    pub fn stats_json(&self) -> Value {
        use std::sync::atomic::Ordering::Relaxed;
        let mut pipelines = Map::new();
        for (id, s) in self.pipelines.read().iter() {
            let Ok(p) = &s.compiled else { continue };
            let mut v = p.stats.to_json();
            v["ingested_as_first_pipeline_in_bytes"] = json!(p.ingested_bytes.load(Relaxed));
            v["produced_as_first_pipeline_in_bytes"] = json!(p.produced_bytes.load(Relaxed));
            let procs: Vec<Value> = p
                .processors
                .iter()
                .map(|n| {
                    let key = match &n.tag {
                        Some(t) => format!("{}:{t}", n.kind),
                        None => n.kind.clone(),
                    };
                    json!({ key: {"type": n.kind, "stats": n.stats.to_json()} })
                })
                .collect();
            v["processors"] = Value::Array(procs);
            pipelines.insert(id.clone(), v);
        }
        json!({"total": self.total.to_json(), "pipelines": pipelines})
    }

    /// Per processor type `[count, time_in_millis, current, failed]` summed
    /// over every pipeline (`_cluster/stats` `nodes.ingest.processor_stats`).
    pub fn processor_type_stats(&self) -> BTreeMap<String, [u64; 4]> {
        use std::sync::atomic::Ordering::Relaxed;
        let mut out: BTreeMap<String, [u64; 4]> = BTreeMap::new();
        for s in self.pipelines.read().values() {
            let Ok(p) = &s.compiled else { continue };
            for n in &p.processors {
                let e = out.entry(n.kind.clone()).or_default();
                e[0] += n.stats.count.load(Relaxed);
                e[1] += n.stats.time_nanos.load(Relaxed) / 1_000_000;
                e[2] += n.stats.current.load(Relaxed);
                e[3] += n.stats.failed.load(Relaxed);
            }
        }
        out
    }
}

/// Shared node-wide stores, keyed by data directory (the node owns one).
fn registry() -> &'static parking_lot::Mutex<HashMap<PathBuf, Arc<IngestService>>> {
    static R: OnceLock<parking_lot::Mutex<HashMap<PathBuf, Arc<IngestService>>>> = OnceLock::new();
    R.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

// ------------------------------------------------------------ node integration

/// One document headed for an index.
pub struct IngestRequest<'a> {
    pub index: &'a str,
    pub id: Option<&'a str>,
    pub routing: Option<&'a str>,
    pub source: &'a str,
    /// `?pipeline=` / bulk `pipeline` (`_none` disables the default pipeline).
    pub pipeline: Option<&'a str>,
}

/// A failed ingest run (see [`Node::run_ingest_ext`]).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IngestFailure {
    pub error: EsError,
    /// The index whose failure store receives the document.
    pub index: String,
    pub context: crate::datastream::FailureContext,
}

#[derive(Debug, PartialEq)]
pub enum IngestOutcome {
    /// No pipeline applies: index as-is.
    Unchanged,
    /// A `drop` processor discarded the document.
    Dropped { index: String },
    /// Pipelines ran; index this instead.
    Changed {
        index: String,
        id: Option<String>,
        routing: Option<String>,
        source: String,
        /// `_dynamic_templates` set by the pipelines (path -> template name).
        dynamic_templates: Option<std::collections::HashMap<String, String>>,
    },
}

fn none_if_none(s: Option<String>) -> Option<String> {
    s.filter(|p| !p.is_empty() && p != "_none")
}

fn parse_source_map(source: &str) -> Result<Map<String, Value>> {
    // Text after the first value is ignored, as ES's `XContentParser.map()`
    // does (`crate::source_json::leading_value`).
    let v: Value = serde_json::from_str(source)
        .or_else(|e| crate::source_json::leading_value(source).map_or(Err(e), serde_json::from_str::<Value>))
        .map_err(|e| EsError::document_parsing(format!("[{}:{}] failed to parse", e.line(), e.column())).caused_by(EsError::json_parse(&e)))?;
    match v {
        Value::Object(m) => Ok(m),
        _ => Err(EsError::document_parsing("[1:1] failed to parse: Root document must be an object")),
    }
}

impl Node {
    /// The node's pipeline store.
    pub fn ingest(&self) -> Arc<IngestService> {
        let mut r = registry().lock();
        r.entry(self.data.clone()).or_insert_with(|| Arc::new(IngestService::open(&self.data))).clone()
    }

    /// (`index.default_pipeline`, `index.final_pipeline`) for a write
    /// target: the concrete index's settings, or the matching templates'
    /// when the index does not exist yet.
    pub fn index_pipelines(&self, target: &str) -> (Option<String>, Option<String>) {
        if let Ok(svc) = self.write_index(target, false) {
            return (none_if_none(svc.setting("index.default_pipeline")), none_if_none(svc.setting("index.final_pipeline")));
        }
        let (settings, _, _) = self.template_for(target);
        let flat = crate::settings::normalize_index_settings(&settings);
        let get = |k: &str| flat.get(k).and_then(Value::as_str).map(str::to_string);
        (none_if_none(get("index.default_pipeline")), none_if_none(get("index.final_pipeline")))
    }

    /// Delete pipelines unless an index uses one as default/final pipeline.
    pub fn delete_pipeline(&self, expr: &str) -> Result<()> {
        let store = self.ingest();
        for id in store.matching_delete(expr) {
            for (setting, what) in [("index.default_pipeline", "default"), ("index.final_pipeline", "final")] {
                let users: Vec<String> = self.all_indices().into_iter().filter(|s| s.setting(setting).as_deref() == Some(id.as_str())).map(|s| s.name.clone()).collect();
                if !users.is_empty() {
                    return Err(EsError::illegal_argument(format!(
                        "pipeline [{id}] cannot be deleted because it is the {what} pipeline for {} index(es) including [{}]",
                        users.len(),
                        users.join(", ")
                    )));
                }
            }
        }
        store.delete(expr).map(|_| ())
    }

    /// Run the pipelines that apply to an index request.
    pub fn run_ingest(&self, r: IngestRequest) -> Result<IngestOutcome> {
        self.run_ingest_ext(r).map_err(|f| f.error)
    }

    /// [`Self::run_ingest`] reporting, on failure, what the failure store
    /// needs: the index whose failure store applies and the pipeline trace.
    pub fn run_ingest_ext(&self, r: IngestRequest) -> std::result::Result<IngestOutcome, IngestFailure> {
        self.run_ingest_source_ext(r)
    }

    /// Shared by indexing and `_ingest/_simulate`: returns `None` when no
    /// pipeline applies, else the final document or a drop.
    fn execute_request(&self, r: &IngestRequest, cx: &PipelineContext, pre: Option<IngestDocument>) -> Result<Option<Exec>> {
        self.execute_request_ix(r, cx, pre).map_err(|(e, _)| e)
    }

    /// Like [`Self::execute_request`], but a failure also reports the document's
    /// `_index` at the time (simulate-ingest shows it).
    fn execute_request_ix(&self, r: &IngestRequest, cx: &PipelineContext, pre: Option<IngestDocument>) -> std::result::Result<Option<Exec>, (EsError, String)> {
        let (mut def, mut fin) = self.index_pipelines(r.index);
        let mut current: Option<String> = match r.pipeline {
            Some("_none") => None,
            Some(p) => Some(p.to_string()),
            None => def.take(),
        };
        if current.is_none() && fin.is_none() && pre.is_none() {
            return Ok(None);
        }
        let mut doc = match pre {
            Some(d) => d,
            None => IngestDocument::new(parse_source_map(r.source).map_err(|e| (e, r.index.to_string()))?, r.index, r.id, r.routing, now_timestamp()),
        };
        let mut seen = vec![r.index.to_string()];
        let mut first: Option<Arc<Pipeline>> = None;
        loop {
            if let Some(pid) = current.take() {
                let p = cx.pipeline(&pid).ok_or_else(|| (EsError::illegal_argument(format!("pipeline with id [{pid}] does not exist")), doc.meta_str("_index").unwrap_or_default()))?;
                if cx.count_stats.get() && first.is_none() {
                    p.ingested_bytes.fetch_add(r.source.len() as u64, std::sync::atomic::Ordering::Relaxed);
                    first = Some(p.clone());
                }
                match run_pipeline(&p, &mut doc, cx) {
                    Ok(()) => {}
                    Err(ProcessorError::Dropped) => return Ok(Some(Exec::Dropped(doc.meta_str("_index").unwrap_or_default()))),
                    Err(ProcessorError::Terminated) => {}
                    Err(ProcessorError::Failed(e, origin)) => {
                        *cx.fail_origin.borrow_mut() = origin;
                        return Err((e, doc.meta_str("_index").unwrap_or_default()));
                    }
                }
            }
            let now_index = doc.meta_str("_index").unwrap_or_default();
            if doc.rerouted {
                doc.rerouted = false;
                if now_index != *seen.last().unwrap() {
                    if seen.contains(&now_index) {
                        seen.push(now_index.clone());
                        return Err((EsError::new(500, "illegal_state_exception", format!("index cycle detected while processing pipeline [{}] for document [{}]: [{}]", cx.executed.borrow().last().cloned().unwrap_or_default(), r.id.unwrap_or(""), seen.join(", "))), now_index.clone()));
                    }
                    seen.push(now_index.clone());
                    *cx.fs_index.borrow_mut() = Some(now_index.clone());
                    let (d2, f2) = self.index_pipelines(&now_index);
                    fin = f2;
                    current = d2;
                    continue;
                }
            } else if now_index != *seen.last().unwrap() {
                // a plain `_index` change keeps the pipelines but picks up the new index's final pipeline
                fin = self.index_pipelines(&now_index).1;
            }
            break;
        }
        if let Some(fid) = fin {
            let p = cx.pipeline(&fid).ok_or_else(|| (EsError::illegal_argument(format!("pipeline with id [{fid}] does not exist")), doc.meta_str("_index").unwrap_or_default()))?;
            let before = doc.meta_str("_index");
            match run_pipeline(&p, &mut doc, cx) {
                Ok(()) | Err(ProcessorError::Terminated) => {}
                Err(ProcessorError::Dropped) => return Ok(Some(Exec::Dropped(doc.meta_str("_index").unwrap_or_default()))),
                Err(ProcessorError::Failed(e, origin)) => {
                    *cx.fail_origin.borrow_mut() = origin;
                    return Err((e, doc.meta_str("_index").unwrap_or_default()));
                }
            }
            let after = doc.meta_str("_index");
            if after != before {
                let msg = format!("final pipeline [{fid}] can't change the target index (from [{}] to [{}]) for document [{}]", before.unwrap_or_default(), after.clone().unwrap_or_default(), r.id.unwrap_or(""));
                return Err((EsError::new(500, "illegal_state_exception", msg), after.unwrap_or_default()));
            }
        }
        if let Some(p) = first {
            let produced = serde_json::to_vec(&doc.source_value()).map(|v| v.len() as u64).unwrap_or(0);
            p.produced_bytes.fetch_add(produced, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Some(Exec::Doc(doc)))
    }

    /// `POST /_ingest/pipeline[/{id}]/_simulate`.
    pub fn simulate_pipeline(&self, id: Option<&str>, body: &Value, verbose: bool) -> Result<Value> {
        let _enrich = crate::enrich::NodeScope::enter(self);
        let store = self.ingest();
        let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body is required"))?;
        let pipeline: Arc<Pipeline> = match id {
            Some(id) => store.pipeline(id).ok_or_else(|| EsError::illegal_argument(format!("pipeline [{id}] does not exist")))?,
            None => {
                let cfg = o.get("pipeline").ok_or_else(|| EsError::new(400, "parse_exception", "[pipeline] required property is missing").with("property_name", "pipeline"))?;
                Arc::new(compile_pipeline("_simulate_pipeline", cfg)?)
            }
        };
        let docs = match o.get("docs") {
            Some(Value::Array(a)) => a,
            Some(_) => return Err(EsError::new(400, "parse_exception", "[docs] property isn't a list").with("property_name", "docs")),
            None => return Err(EsError::new(400, "parse_exception", "[docs] required property is missing").with("property_name", "docs")),
        };
        if docs.is_empty() {
            return Err(EsError::illegal_argument("must specify at least one document in [docs]"));
        }
        let mut parsed = Vec::with_capacity(docs.len());
        for d in docs {
            parsed.push(simulate_input_doc(d, None, true)?);
        }
        let mut out = Vec::new();
        for mut doc in parsed {
            let cx = PipelineContext { trace: if verbose { Some(RefCell::new(Vec::new())) } else { None }, ..PipelineContext::new(&store) };
            let r = run_pipeline(&pipeline, &mut doc, &cx);
            if verbose {
                let results = cx.trace.map(|t| t.into_inner()).unwrap_or_default();
                out.push(json!({"processor_results": results}));
                continue;
            }
            out.push(match r {
                Ok(()) | Err(ProcessorError::Terminated) => json!({"doc": simulate_doc_json(&doc)}),
                Err(ProcessorError::Dropped) => Value::Null,
                Err(ProcessorError::Failed(e, _)) => json!({"error": e.error_object(false)}),
            });
        }
        Ok(json!({"docs": out}))
    }

    /// `POST /_ingest[/{index}]/_simulate` (simulate ingest, 8.12+).
    pub fn simulate_ingest(&self, index: Option<&str>, pipeline: Option<&str>, body: &Value) -> Result<Value> {
        let _enrich = crate::enrich::NodeScope::enter(self);
        let store = self.ingest();
        let o = body.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "request body is required"))?;
        let mut subs: HashMap<String, Arc<Pipeline>> = HashMap::new();
        if let Some(ps) = o.get("pipeline_substitutions") {
            let m = ps.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "[pipeline_substitutions] must be an object"))?;
            for (id, cfg) in m {
                validate_no_system_properties(cfg)?;
                subs.insert(id.clone(), Arc::new(compile_pipeline(id, cfg)?));
            }
        }
        let docs = match o.get("docs") {
            Some(Value::Array(a)) => a,
            _ => return Err(EsError::action_validation(&["docs is missing"])),
        };
        let mut out = Vec::new();
        for d in docs {
            let doc = simulate_input_doc(d, index, false)?;
            let idx = doc.meta_str("_index").unwrap_or_default();
            let id = doc.meta_str("_id");
            let cx = PipelineContext { substitutions: Some(&subs), ..PipelineContext::new(&store) };
            let req = IngestRequest { index: &idx, id: id.as_deref(), routing: None, source: "", pipeline };
            let mut entry = Map::new();
            let mut error: Option<EsError> = None;
            let mut final_doc: Option<IngestDocument> = None;
            match self.execute_request_ix(&req, &cx, Some(doc)) {
                Ok(Some(Exec::Doc(d))) => final_doc = Some(d),
                Ok(Some(Exec::Dropped(di))) => {
                    out.push(json!({"doc": {"_index": di, "_id": id.clone().unwrap_or_else(|| "auto-generated".into()), "_version": -3, "result": "noop", "_shards": {"total": 0, "successful": 0, "failed": 0}}}));
                    continue;
                }
                Ok(None) => {}
                Err((e, at)) => {
                    out.push(json!({"doc": {"_id": id.clone().unwrap_or_else(|| "auto-generated".into()), "_index": at, "error": error_body(&e)}}));
                    continue;
                }
            }
            let src_fallback = Value::Null;
            let (out_index, out_id, source) = match &final_doc {
                Some(d) => (d.meta_str("_index").unwrap_or(idx.clone()), d.meta_str("_id"), d.source_value()),
                None => (idx.clone(), id.clone(), src_fallback),
            };
            entry.insert("_id".into(), out_id.map(Value::String).unwrap_or(Value::Null));
            entry.insert("_index".into(), json!(out_index));
            entry.insert("_version".into(), json!(-3));
            entry.insert("_source".into(), source.clone());
            entry.insert("executed_pipelines".into(), json!(cx.executed.borrow().clone()));
            if error.is_none() {
                if let Err(e) = self.validate_against_mapping(&out_index, &source) {
                    error = Some(e);
                }
            }
            if let Some(e) = error {
                entry.insert("error".into(), error_body(&e));
            }
            out.push(json!({"doc": Value::Object(entry)}));
        }
        Ok(json!({"docs": out}))
    }

    /// Parse a would-be source against the target index's mapping (no side effects).
    fn validate_against_mapping(&self, index: &str, source: &Value) -> Result<()> {
        // ES reports where in the (compact) source the offending field's value starts.
        let locate = |mut e: EsError| {
            if e.kind == "strict_dynamic_mapping_exception" {
                let field = e.reason.split("dynamic introduction of [").nth(1).and_then(|r| r.split(']').next()).unwrap_or("").to_string();
                let compact = source.to_string();
                if let Some(at) = compact.find(&format!("\"{field}\":")) {
                    let col = compact[..at + field.len() + 3].chars().count() + 1;
                    e.reason = e.reason.replacen("[1:1]", &format!("[1:{col}]"), 1);
                }
            }
            e
        };
        let Ok(svc) = self.write_index(index, false) else {
            // Not created yet: validate against what its matching templates would give it.
            let (settings, mut mappings, _) = self.template_for(index);
            if !mappings.as_object().is_some_and(|m| !m.is_empty()) {
                return Ok(());
            }
            // A data stream's backing index maps `@timestamp` itself.
            if self.data_stream_template(index).is_some() && mappings.pointer("/properties/@timestamp").is_none() {
                mappings["properties"]["@timestamp"] = json!({"type": "date"});
            }
            let flat = crate::settings::normalize_index_settings(&settings);
            let registry = crate::index::registry_for(&flat)?;
            let mut m = crate::mapping::Mappings::parse(&mappings, &registry)?;
            m.assign_core_ids();
            let lookup = crate::mapping::FieldLookup::build(&m, &registry);
            return crate::document::parse_with_dynamic(source, "_simulate", &m, &lookup, 1000).map(|_| ()).map_err(locate);
        };
        let meta = svc.meta();
        let lookup = svc.lookup();
        let limit = svc.setting_usize("index.mapping.total_fields.limit", 1000);
        crate::document::parse_with_dynamic(source, "_simulate", &meta.mappings, &lookup, limit).map(|_| ()).map_err(locate)
    }
}

impl Node {
    /// No ingest pipeline is defined: a write that names none runs none
    /// (one check per bulk instead of one per item).
    pub fn ingest_is_idle(&self) -> bool {
        registry_is_empty(self)
    }
}

fn registry_is_empty(node: &Node) -> bool {
    {
        let r = registry().lock();
        if let Some(s) = r.get(&node.data) {
            return s.is_empty();
        }
    }
    node.ingest().is_empty()
}

enum Exec {
    Doc(IngestDocument),
    Dropped(String),
}

pub fn now_timestamp() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)
}

/// A simulate request doc → ingest document.
fn simulate_input_doc(d: &Value, default_index: Option<&str>, pipeline_api: bool) -> Result<IngestDocument> {
    let o = d.as_object().ok_or_else(|| EsError::new(400, "parse_exception", "[docs] entries must be objects"))?;
    let src = match o.get("_source") {
        Some(Value::Object(m)) => m.clone(),
        Some(_) => return Err(EsError::new(400, "parse_exception", "[_source] property isn't a map").with("property_name", "_source")),
        None => return Err(EsError::new(400, "parse_exception", "[_source] required property is missing").with("property_name", "_source")),
    };
    let s = |k: &str| o.get(k).filter(|v| !v.is_null()).map(doc::value_to_string);
    let index = s("_index").or(default_index.map(str::to_string)).unwrap_or_else(|| if pipeline_api { "_index".into() } else { String::new() });
    let id = s("_id").or(if pipeline_api { Some("_id".into()) } else { None });
    let routing = s("_routing");
    let mut doc = IngestDocument::new(src, &index, id.as_deref(), routing.as_deref(), now_timestamp());
    if pipeline_api {
        let m = doc.ctx_map_mut();
        m.insert("_version".into(), o.get("_version").cloned().unwrap_or(json!(-3)));
        for k in ["_version_type", "_if_seq_no", "_if_primary_term"] {
            if let Some(v) = o.get(k) {
                m.insert(k.into(), v.clone());
            }
        }
        if let Some(Value::Object(ing)) = o.get("_ingest") {
            for (k, v) in ing {
                doc.ingest_map_mut().insert(k.clone(), v.clone());
            }
        }
    }
    Ok(doc)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod partial_ctx_script_tests {
    use super::*;
    use crate::source_json::Source;

    /// Scripts reaching ctx by constant keys run on part of it and give
    /// the same document (and the same errors) as on all of it.
    #[test]
    fn partial_and_whole_ctx_scripts_agree() {
        let doc = r#"{"kubernetes":{"pod":{"uid":"u","name":"p","cpu":{"pct":0.50}},"node":{"name":"n"}},"x":[1,2.50],"y":"\u00e9","@timestamp":"2026-01-01"}"#;
        for src in [
            "ctx['kubernetes']['pod']['uid'] = 'value-' + 12",
            "ctx.kubernetes.pod.cpu.pct += 1; ctx.z = ctx.x.size()",
            "ctx.kubernetes.node = null; ctx.newobj = ['a': 1]",
            "if (ctx.kubernetes?.pod != null) { ctx.kubernetes.pod.uid = ctx.kubernetes.pod.name }",
            "ctx.missing.deeper = 1",
            "ctx.y.foo = 1",
            "ctx.kubernetes.pod.self = ctx.kubernetes.pod",
            "def m = ctx.kubernetes; m.extra = 1",
            "ctx.remove('x')",
        ] {
            let script = Script::new(src.to_string(), "painless".into(), json!({}));
            let compiled = script.compile_source(crate::script::ScriptContext::Ingest).unwrap();
            let mut partial = Source::parse(doc).unwrap();
            let a = script.execute_source(&mut partial);
            let mut whole = Source::parse(doc).unwrap();
            let b = script.execute_source_whole(&mut whole, compiled.clone(), compiled.executor(None, None), false);
            match (a, b) {
                (Ok(()), Ok(())) => assert_eq!(partial.text(), whole.text(), "{src}"),
                (Err(a), Err(b)) => {
                    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{src}");
                    assert_eq!(partial.text(), Source::parse(doc).unwrap().text(), "{src}: unchanged on failure");
                }
                (a, b) => panic!("{src}: {a:?} vs {b:?}"),
            }
        }
        let first = Script::new("ctx['kubernetes']['pod']['uid'] = 'v'".into(), "painless".into(), json!({}));
        assert!(first.compile_source(crate::script::ScriptContext::Ingest).unwrap().program.ctx_paths.is_some(), "the touch script runs on part of ctx");
    }
}
