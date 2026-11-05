//! Painless-compatible scripting.
//!
//! Scripts are parsed ([`parser`]), compiled once per `(source, context)`
//! into a closure tree ([`compile`]) and cached in an LRU. Executing a
//! script goes through an [`Executor`]: the program bound to its `params`,
//! an index mapping (so `doc['x']` names resolve to column readers once per
//! leaf) and per-doc state (`_score`, the current doc, `ctx`, ...).
//!
//! Entry points for other modules:
//! - [`compile`] / [`compile_json`] → [`CompiledScript`];
//! - [`CompiledScript::executor`] + the `execute_*` helpers per context;
//! - [`search`]: `script` / `script_score` queries, `function_score`
//!   script functions, `script_fields`, `_script` sort, runtime fields;
//! - [`update`]: scripted `_update` and the `ctx` contexts used by
//!   `_update_by_query` / `_reindex` ([`execute_ctx`]);
//! - [`stored`]: stored scripts (`_scripts/{id}`).

pub(crate) mod allocation;
pub mod ast;
pub mod compile;
pub mod ctx_access;
pub mod doc;
pub mod error;
pub mod execute_api;
pub mod expression;
pub mod jbig;
pub mod jmisc;
pub mod jtime;
pub mod lexer;
pub mod methods;
pub mod parser;
pub mod regex;
mod regex_text;
pub mod runtime;
pub mod search;
mod source;
pub(crate) use source::{from_source as source_from_json, to_source as source_to_json, partial_ctx, merge_partial};
pub(crate) use source::decode_java_utf8;
mod simple_case;
mod locale_case;
mod locale_case_data;
mod locale_word_data;
mod search_geo_topk;
mod search_numeric_topk;
mod search_vector_score;
mod search_compound_topk;
pub mod stored;
pub mod timefmt;
pub mod update;
pub mod value;
pub mod whitelist;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_loop_fast;

pub use error::ScriptErr;
pub use runtime::{Executor, Program};
pub use value::Val;

use crate::date::DateFormatter;
use crate::error::{EsError, Result};
use crate::mapping::FieldLookup;
use parking_lot::Mutex;
use rustic_core::types::DocId;
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Where a script runs; decides return-value conversion and shows up in
/// errors / `_script_context`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptContext {
    Score,
    Filter,
    Field,
    NumberSort,
    StringSort,
    Aggs,
    BucketScript,
    BucketSelector,
    Update,
    UpdateByQuery,
    Reindex,
    Ingest,
    IngestConditional,
    RuntimeLong,
    RuntimeDouble,
    RuntimeKeyword,
    RuntimeBoolean,
    RuntimeDate,
    RuntimeIp,
    RuntimeGeoShape,
    PainlessTest,
    /// Watcher condition / transform / template scripts (`Json`, `String.sha*`).
    Watcher,
    WatcherCondition,
    WatcherTemplate,
    /// `scripted_metric` scripts (`state` / `states` bound to the ctx slot).
    MetricInit,
    MetricMap,
    MetricCombine,
    MetricReduce,
}

impl ScriptContext {
    pub fn from_name(name: &str) -> Option<Self> {
        const ALL: &[ScriptContext] = &[
            ScriptContext::Score, ScriptContext::Filter, ScriptContext::Field,
            ScriptContext::NumberSort, ScriptContext::StringSort, ScriptContext::Aggs,
            ScriptContext::BucketScript, ScriptContext::BucketSelector, ScriptContext::Update,
            ScriptContext::UpdateByQuery, ScriptContext::Reindex, ScriptContext::Ingest, ScriptContext::IngestConditional,
            ScriptContext::RuntimeLong, ScriptContext::RuntimeDouble, ScriptContext::RuntimeKeyword,
            ScriptContext::RuntimeBoolean, ScriptContext::RuntimeDate, ScriptContext::RuntimeIp,
            ScriptContext::RuntimeGeoShape, ScriptContext::PainlessTest, ScriptContext::Watcher,
            ScriptContext::WatcherCondition, ScriptContext::WatcherTemplate,
            ScriptContext::MetricInit, ScriptContext::MetricMap, ScriptContext::MetricCombine, ScriptContext::MetricReduce,
        ];
        ALL.iter().copied().find(|context| context.name() == name)
    }
    pub(crate) fn compilation_rate_limited(self) -> bool {
        !matches!(self, Self::Ingest | Self::IngestConditional | Self::RuntimeLong | Self::RuntimeDouble | Self::RuntimeKeyword | Self::RuntimeBoolean | Self::RuntimeDate | Self::RuntimeIp | Self::RuntimeGeoShape)
    }
    pub(crate) fn default_cache_size(self) -> usize {
        if matches!(self, Self::Ingest | Self::IngestConditional) { 200 } else { 100 }
    }
    pub fn name(&self) -> &'static str {
        match self {
            ScriptContext::Score => "score",
            ScriptContext::Filter => "filter",
            ScriptContext::Field => "field",
            ScriptContext::NumberSort => "number_sort",
            ScriptContext::StringSort => "string_sort",
            ScriptContext::Aggs => "aggs",
            ScriptContext::BucketScript => "bucket_aggregation",
            ScriptContext::BucketSelector => "aggregation_selector",
            ScriptContext::Update => "update",
            ScriptContext::UpdateByQuery => "update_by_query",
            ScriptContext::Reindex => "reindex",
            ScriptContext::Ingest => "ingest",
            ScriptContext::IngestConditional => "processor_conditional",
            ScriptContext::RuntimeLong => "long_field",
            ScriptContext::RuntimeDouble => "double_field",
            ScriptContext::RuntimeKeyword => "keyword_field",
            ScriptContext::RuntimeBoolean => "boolean_field",
            ScriptContext::RuntimeDate => "date_field",
            ScriptContext::RuntimeIp => "ip_field",
            ScriptContext::RuntimeGeoShape => "geometry_field",
            ScriptContext::PainlessTest => "painless_test",
            ScriptContext::Watcher => "watcher_transform",
            ScriptContext::WatcherCondition => "watcher_condition",
            ScriptContext::WatcherTemplate => "template",
            ScriptContext::MetricInit => "aggs_init",
            ScriptContext::MetricMap => "aggs_map",
            ScriptContext::MetricCombine => "aggs_combine",
            ScriptContext::MetricReduce => "aggs_reduce",
        }
    }
}

/// All context names (`GET _script_context`).
pub const CONTEXT_NAMES: &[(&str, &str)] = &[
    ("aggregation_selector", "boolean"),
    ("aggs", "java.lang.Object"),
    ("aggs_combine", "java.lang.Object"),
    ("aggs_init", "void"),
    ("aggs_map", "void"),
    ("aggs_reduce", "java.lang.Object"),
    ("boolean_field", "void"),
    ("bucket_aggregation", "java.lang.Double"),
    ("date_field", "void"),
    ("double_field", "void"),
    ("field", "java.lang.Object"),
    ("filter", "boolean"),
    ("ingest", "void"),
    ("processor_conditional", "boolean"),
    ("ip_field", "void"),
    ("geometry_field", "void"),
    ("keyword_field", "void"),
    ("long_field", "void"),
    ("number_sort", "double"),
    ("painless_test", "java.lang.Object"),
    ("reindex", "void"),
    ("score", "double"),
    ("string_sort", "java.lang.String"),
    ("template", "java.lang.String"),
    ("update", "void"),
    ("update_by_query", "void"),
    ("watcher_condition", "boolean"),
    ("watcher_transform", "java.lang.Object"),
];

// ---------------------------------------------------------------------------
// Compilation cache.

pub mod rate_limit;
pub mod cache_policy;
pub mod telemetry;

static COMPILATIONS: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

type CacheKey = (String, ScriptContext);

fn cache() -> &'static Mutex<cache_policy::Caches> {
    static C: OnceLock<Mutex<cache_policy::Caches>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(cache_policy::Caches::default()))
}

/// `(compilations, cache_evictions)` counters (`_nodes/stats`, `_info/script`).
pub fn stats() -> (u64, u64) {
    (COMPILATIONS.load(Ordering::Relaxed), EVICTIONS.load(Ordering::Relaxed))
}

/// One compilation in progress. Identical concurrent misses wait for its
/// result (ES's cache `computeIfAbsent`, which also hands every waiter the
/// loader's failure) instead of compiling again and spending more budget.
#[derive(Default)]
pub(super) struct Pending {
    result: Mutex<Option<std::result::Result<Arc<Program>, EsError>>>,
    ready: parking_lot::Condvar,
}
impl Pending {
    fn publish(&self, result: std::result::Result<Arc<Program>, EsError>) {
        *self.result.lock() = Some(result);
        self.ready.notify_all();
    }
    fn wait(&self) -> std::result::Result<Arc<Program>, EsError> {
        let mut result = self.result.lock();
        while result.is_none() { self.ready.wait(&mut result); }
        result.clone().expect("published")
    }
}

/// `key`'s program from the cache, else compiled by `compile`. The cache lock
/// covers only the lookup and the insert: admission and compilation (which
/// can take long, e.g. regex automata) run outside it, so a stream of new
/// scripts never stalls cache hits of other scripts on this node. One
/// compilation, counted once against the rate limit, per actual compile.
fn cached(key: CacheKey, compile: impl FnOnce() -> std::result::Result<Arc<Program>, EsError>) -> std::result::Result<Arc<Program>, EsError> {
    let ctx = key.1;
    let pending = {
        let mut caches = cache().lock();
        if let Some(p) = caches.context(ctx).get(&key) {
            return Ok(p.clone());
        }
        if let Some(running) = caches.inflight.get(&key).cloned() {
            drop(caches);
            return running.wait();
        }
        let pending = Arc::new(Pending::default());
        caches.inflight.insert(key.clone(), pending.clone());
        pending
    };
    // Waiters must never be stranded, even if compilation unwinds.
    struct Leader { key: Option<CacheKey>, pending: Arc<Pending> }
    impl Drop for Leader {
        fn drop(&mut self) {
            if let Some(key) = self.key.take() {
                cache().lock().inflight.remove(&key);
                self.pending.publish(Err(EsError::internal("script compilation did not complete")));
            }
        }
    }
    let mut leader = Leader { key: Some(key), pending };
    let result = rate_limit::check(ctx).and_then(|_| compile());
    let key = leader.key.take().expect("leader key");
    let mut caches = cache().lock();
    caches.inflight.remove(&key);
    if let Ok(p) = &result {
        COMPILATIONS.fetch_add(1, Ordering::Relaxed);
        telemetry::compiled(ctx);
        let c = caches.context(ctx);
        if c.len() == c.cap().get() {
            EVICTIONS.fetch_add(1, Ordering::Relaxed);
            if let Some(((_, evicted_ctx), _)) = c.peek_lru() { telemetry::evicted(*evicted_ctx); }
        }
        c.put(key, p.clone());
    }
    drop(caches);
    leader.pending.publish(result.clone());
    result
}

/// Admit a Mustache template through the shared script cache (`template`
/// context): `check` validates it on a miss, which counts one compilation.
/// The cached entry is a marker; templates are rendered from their source.
pub fn admit_template(source: &str, check: impl FnOnce() -> Result<()>) -> Result<()> {
    static MARKER: OnceLock<Arc<Program>> = OnceLock::new();
    let marker = || MARKER.get_or_init(|| compile::compile_arc("return null", ScriptContext::PainlessTest).expect("template cache marker")).clone();
    cached((format!("\u{0}mustache\u{0}{source}"), ScriptContext::WatcherTemplate), || check().map(|_| marker())).map(drop)
}

/// Compiled program for `source` in `ctx`, from the LRU when possible.
pub fn program(source: &str, ctx: ScriptContext) -> std::result::Result<Arc<Program>, EsError> {
    cached((source.to_string(), ctx), || compile::compile_arc(source, ctx).map_err(|e| {
        // Syntax errors (lexer / parser) report the whole script as the snippet.
        let syntax = e.msg.starts_with("unexpected token") || e.msg.starts_with("invalid sequence of tokens") || e.msg.starts_with("unexpected character") || e.msg.starts_with("unexpected end of script");
        let starts = if syntax { vec![0] } else { stmt_starts_or_empty(source) };
        error::to_es_error(&e, source, &starts)
    }))
}

fn stmt_starts_or_empty(source: &str) -> Vec<usize> {
    // Statement boundaries are only known after a successful parse; for a
    // failed compile use line starts so the snippet stays short.
    let mut v = vec![0];
    for (i, c) in source.char_indices() {
        if c == ';' || c == '\n' {
            v.push(i + 1);
        }
    }
    v.retain(|x| *x < source.len().max(1));
    v
}

/// A program bound to its parameters.
#[derive(Clone, Debug)]
pub struct CompiledScript {
    pub program: Arc<Program>,
    pub params: Arc<Value>,
    pub(crate) source_params: Option<Arc<crate::source_json::Source>>,
}

impl CompiledScript {
    pub fn source(&self) -> &str {
        &self.program.source
    }

    /// A fresh executor (one per thread / leaf; reuse it across docs).
    pub fn executor(&self, lookup: Option<Arc<FieldLookup>>, runtime: Option<Arc<RuntimeFields>>) -> Executor {
        let mut exec=Executor::new(self.program.clone(), &self.params, lookup, runtime);
        if let Some(crate::source_json::Source::Object(params))=self.source_params.as_deref() {
            for (name,value) in params {exec.set_param(name,source::from_source(value));}
        }
        exec
    }

    /// ES `script_exception` for a runtime failure of this script.
    pub fn error(&self, e: &ScriptErr) -> EsError {
        if let Some(error) = &e.breaker { return error.clone(); }
        if self.program.expression.is_some() {
            return expression::to_es_error(e, &self.program.source);
        }
        error::to_es_error(e, &self.program.source, &self.program.stmt_starts)
    }
}

/// `script::compile(source, params, context)`.
pub fn compile(source: &str, params: &Value, ctx: ScriptContext) -> Result<CompiledScript> {
    let program = program(source, ctx)?;
    Ok(CompiledScript { program, params: Arc::new(if params.is_null() { json!({}) } else { params.clone() }), source_params: None })
}

/// Parsed `script` request object: `"src"` or `{"source"|"id", "params", "lang"}`.
#[derive(Clone, Debug, Default)]
pub struct ScriptSpec {
    pub source: Option<String>,
    pub id: Option<String>,
    pub params: Value,
    pub lang: Option<String>,
}

pub fn parse_spec(v: &Value) -> Result<ScriptSpec> {
    match v {
        Value::String(s) => Ok(ScriptSpec { source: Some(s.clone()), params: json!({}), ..Default::default() }),
        Value::Object(o) => {
            let mut spec = ScriptSpec { params: json!({}), ..Default::default() };
            for (k, x) in o {
                match k.as_str() {
                    "source" | "inline" => {
                        spec.source = Some(match x {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                    }
                    "id" => spec.id = x.as_str().map(str::to_string),
                    "params" => {
                        if !x.is_object() && !x.is_null() {
                            return Err(EsError::parsing("[params] must be an object"));
                        }
                        if x.is_object() {
                            spec.params = x.clone();
                        }
                    }
                    "lang" => spec.lang = x.as_str().map(str::to_string),
                    "options" => {}
                    other => return Err(EsError::parsing(format!("[script] unknown field [{other}]")).with("line", 1).with("col", 1)),
                }
            }
            if spec.source.is_some() && spec.id.is_some() {
                return Err(EsError::parsing("must specify either [source] for an inline script or [id] for a stored script"));
            }
            if spec.source.is_none() && spec.id.is_none() {
                return Err(EsError::parsing("must specify either [source] for an inline script or [id] for a stored script"));
            }
            Ok(spec)
        }
        _ => Err(EsError::parsing("[script] must be a string or an object")),
    }
}

/// Compile a request `script` value (inline or stored by id). Stored
/// scripts come from `stored`, else from the current [`ScriptEnv`].
pub fn compile_json(v: &Value, ctx: ScriptContext, stored: Option<&stored::StoredScripts>) -> Result<CompiledScript> {
    let spec = parse_spec(v)?;
    if let Some(l) = spec.lang.as_deref() {
        if l != "painless" && l != expression::LANG && !(l == "mustache" && spec.id.is_some()) {
            return Err(EsError::illegal_argument(format!("script_lang not supported [{l}]")));
        }
    }
    let (source, lang, raw_source) = match (&spec.source, &spec.id) {
        (Some(s), _) => (s.clone(), spec.lang.clone().unwrap_or_else(|| "painless".into()), None),
        (None, Some(id)) => {
            let found = match stored {
                Some(s) => s.get(id),
                None => env().and_then(|e| e.stored.as_ref().and_then(|s| s.get(id))),
            };
            match found {
                Some(s) if s.lang == "painless" || s.lang == expression::LANG => (s.source, s.lang, s.raw_source),
                Some(s) => return Err(EsError::illegal_argument(format!("script_lang not supported [{}]", s.lang))),
                None => return Err(EsError::resource_not_found(format!("unable to find script [{id}] in cluster state"))),
            }
        }
        _ => unreachable!(),
    };
    if lang == expression::LANG {
        return compile_expression(&source, &spec.params, ctx);
    }
    if let Some(raw) = raw_source {
        let program = source::compile_raw_literal_source(&crate::source_json::decode_string(&raw), ctx)?;
        return Ok(CompiledScript { program, params: Arc::new(if spec.params.is_null() { json!({}) } else { spec.params }), source_params: None });
    }
    compile(&source, &spec.params, ctx)
}

/// Resolve opaque request script slots only at search-DSL call sites.
/// Ordinary compile_json remains strict and never interprets user integers.
pub fn search_raw_slot(v:&Value)->Option<crate::source_json::Source> {
    v.as_u64().and_then(|id|env().and_then(|e|e.raw_search_scripts.get(id as usize).cloned()))
}

pub fn search_script_source(v:&Value)->Result<(Value,Option<crate::source_json::Source>)> {
    let raw=search_raw_slot(v);
    let Some(mut raw)=raw else{return Ok((v.clone(),None))};
    if matches!(&raw,crate::source_json::Source::Scalar(token) if token.get().starts_with('"')) {
        raw=crate::source_json::Source::Object(std::collections::BTreeMap::from([("source".into(),raw)]));
    }
    let wrapper=crate::source_json::Source::Object(std::collections::BTreeMap::from([("script".into(),raw.clone())]));
    let (metadata,_)=crate::source_json::update_request(&wrapper.text())?;
    Ok((metadata["script"].clone(),Some(raw)))
}

pub fn compile_search_script(v:&Value,ctx:ScriptContext)->Result<CompiledScript> {
    let (metadata,raw)=search_script_source(v)?;
    compile_json_source(&metadata,raw.as_ref(),ctx,None)
}

/// Compile request metadata with the original script subtree retained separately.
pub fn compile_json_source(v:&Value, raw:Option<&crate::source_json::Source>, ctx:ScriptContext, stored:Option<&stored::StoredScripts>) -> Result<CompiledScript> {
    let source_params = raw.and_then(|r|r.get("params")).cloned().map(Arc::new);
    if let Some(text) = raw.and_then(|r| r.get("source").or_else(||r.get("inline"))) {
        if let crate::source_json::Source::Scalar(text) = text {
            if text.get().starts_with('"') && serde_json::from_str::<String>(text.get()).is_err() {
                let spec = parse_spec(v)?;
                if spec.lang.as_deref().is_some_and(|lang|lang != "painless") {
                    return Err(EsError::illegal_argument("Raw UTF16 source requires painless"));
                }
                let program = source::compile_raw_literal_source(&crate::source_json::decode_string(text.get()),ctx)?;
                return Ok(CompiledScript { program, params: Arc::new(if spec.params.is_null() { json!({}) } else { spec.params }), source_params });
            }
        }
    }
    let mut script = compile_json(v,ctx,stored)?;
    script.source_params = source_params;
    Ok(script)
}

/// `lang: expression` counterpart of [`compile`].
pub fn compile_expression(source: &str, params: &Value, ctx: ScriptContext) -> Result<CompiledScript> {
    if !expression::supports(ctx) {
        return Err(EsError::illegal_argument(format!("expression engine does not know how to handle script context [{}]", ctx.name())));
    }
    let key = (format!("\u{0}expression\u{0}{source}"), ctx);
    let program = cached(key, || Ok(Arc::new(expression::compile(source, ctx).map_err(|e| expression::to_es_error(&e, source))?)))?;
    Ok(CompiledScript { program, params: Arc::new(if params.is_null() { json!({}) } else { params.clone() }), source_params: None })
}

// ---------------------------------------------------------------------------
// Context helpers.

fn score_of(v: &Val) -> std::result::Result<f64, Box<ScriptErr>> {
    match v {
        Val::Double(d) => Ok(*d),
        other => other.as_f64().ok_or_else(|| error::class_cast(format!("cannot implicitly cast def [{}] to double", other.def_name()))),
    }
}

/// Score context: `doc` + `_score` → double.
#[inline]
pub fn execute_score(exec: &mut Executor, doc: DocId, score: f32) -> std::result::Result<f64, Box<ScriptErr>> {
    exec.set_doc(doc);
    exec.set_score(score as f64);
    if let Some(d) = exec.run_f64() {
        return Ok(d);
    }
    let v = exec.run()?;
    if let Val::Double(d) = &v {
        let d = *d;
        std::mem::forget(v);
        return Ok(d);
    }
    score_of(&v).map_err(|e| e.at(exec.program().result_pos))
}

/// Filter context → boolean.
#[inline]
pub fn execute_filter(exec: &mut Executor, doc: DocId) -> std::result::Result<bool, Box<ScriptErr>> {
    exec.set_doc(doc);
    if let Some(b) = exec.run_cond() {
        return Ok(b);
    }
    exec.run()?.as_bool()
}

/// Field context (script_fields) → JSON.
pub fn execute_field(exec: &mut Executor, doc: DocId, score: f32) -> std::result::Result<Value, Box<ScriptErr>> {
    exec.set_doc(doc);
    exec.set_score(score as f64);
    Ok(exec.run()?.to_json())
}

/// Run a `ctx` script (update / update_by_query / reindex / ingest): `ctx`
/// is passed in as JSON (e.g. `{"_source": {...}, "op": "index", "_id": ..}`)
/// and returned after the script modified it.
pub fn execute_ctx(script: &CompiledScript, ctx: Map<String, Value>) -> Result<Map<String, Value>> {
    let mut exec = script.executor(None, None);
    let ctx_val = Val::from_json(&Value::Object(ctx));
    exec.set_ctx(ctx_val.clone());
    exec.run().map_err(|e| script.error(&e))?;
    match ctx_val.to_json() {
        Value::Object(m) => Ok(m),
        _ => Ok(Map::new()),
    }
}

/// Execute source transformations without converting UTF-16 strings to Value.
pub fn execute_ctx_source(script: &CompiledScript, mut ctx: Map<String, Value>, raw: &str) -> Result<(Map<String,Value>,String)> {
    ctx.remove("_source");
    let ctx_val=Val::from_json(&Value::Object(ctx));
    let Val::Map(map)=&ctx_val else { unreachable!() };
    map.borrow_mut().insert(value::Key::str("_source"),source::from_source(&crate::source_json::Source::parse(raw)?));
    let mut exec=script.executor(None,None);
    exec.set_ctx(ctx_val.clone());
    exec.run().map_err(|e|script.error(&e))?;
    let src=map.borrow_mut().shift_remove(&value::Key::str("_source")).unwrap_or(Val::Null);
    if !matches!(src,Val::Map(_)) { return Err(EsError::illegal_argument("ctx._source must be a map")); }
    let source=source::to_source(&src)?.text();
    let Value::Object(mut out)=ctx_val.to_json() else { unreachable!() };
    out.insert("_source".into(),serde_json::json!({}));
    Ok((out,source))
}

// ---------------------------------------------------------------------------
// Runtime fields.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtType {
    Long,
    Double,
    Keyword,
    Boolean,
    Date,
    Ip,
    /// `type: lookup`: fetches fields of docs of another index (fields API only).
    Lookup,
    /// `type: geo_shape`: the emitted geometries of a doc as one encoded
    /// shape value (`fieldtypes/geo.rs`).
    GeoShape,
}

impl RtType {
    fn context(self) -> ScriptContext {
        match self {
            Self::Long => ScriptContext::RuntimeLong,
            Self::Double => ScriptContext::RuntimeDouble,
            Self::Keyword | Self::Lookup => ScriptContext::RuntimeKeyword,
            Self::Boolean => ScriptContext::RuntimeBoolean,
            Self::Date => ScriptContext::RuntimeDate,
            Self::Ip => ScriptContext::RuntimeIp,
            Self::GeoShape => ScriptContext::RuntimeGeoShape,
        }
    }
}

/// Definition of a `lookup` runtime field.
#[derive(Debug, Clone)]
pub struct LookupSpec {
    pub target_index: String,
    pub input_field: String,
    pub target_field: String,
    pub fetch_fields: Vec<String>,
}

type LookupFn = dyn Fn(&str, &str, &str, Option<&str>) -> Result<Vec<Value>> + Sync;

thread_local! {
    static TERMS_LOOKUP: std::cell::Cell<Option<*const LookupFn>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with the request's terms-lookup callback reachable from
/// [`lookup_values`] (lookup runtime fields fetch through it).
pub fn with_terms_lookup<R>(lookup: Option<&(dyn Fn(&str, &str, &str, Option<&str>) -> Result<Vec<Value>> + Sync)>, f: impl FnOnce() -> R) -> R {
    // SAFETY: the pointer is only dereferenced inside `f`, while `lookup` is borrowed.
    let ptr = lookup.map(|l| unsafe { std::mem::transmute::<&(dyn Fn(&str, &str, &str, Option<&str>) -> Result<Vec<Value>> + Sync), &'static LookupFn>(l) as *const LookupFn });
    let prev = TERMS_LOOKUP.with(|c| c.replace(ptr));
    struct Restore(Option<*const LookupFn>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TERMS_LOOKUP.with(|c| c.set(self.0));
        }
    }
    let _r = Restore(prev);
    f()
}

/// The request's terms-lookup callback on this thread ([`with_terms_lookup`]),
/// to make it reachable from a worker thread running part of the request.
#[derive(Clone, Copy)]
pub struct TermsLookupPtr(Option<*const LookupFn>);
// SAFETY: `LookupFn` is `Sync`; the pointer is dereferenced only while the
// request that set it waits for its workers (see `with_terms_lookup`).
unsafe impl Send for TermsLookupPtr {}
unsafe impl Sync for TermsLookupPtr {}

/// This thread's terms-lookup callback.
pub fn terms_lookup_ptr() -> TermsLookupPtr {
    TermsLookupPtr(TERMS_LOOKUP.with(|c| c.get()))
}

/// Run `f` with a captured terms-lookup callback ([`terms_lookup_ptr`]).
pub fn with_terms_lookup_ptr<R>(ptr: TermsLookupPtr, f: impl FnOnce() -> R) -> R {
    let prev = TERMS_LOOKUP.with(|c| c.replace(ptr.0));
    struct Restore(Option<*const LookupFn>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TERMS_LOOKUP.with(|c| c.set(self.0));
        }
    }
    let _r = Restore(prev);
    f()
}

/// Source values at `path` of doc `id` of `index` (the request's realtime-get lookup).
pub fn lookup_values(index: &str, id: &str, path: &str) -> Result<Vec<Value>> {
    match TERMS_LOOKUP.with(|c| c.get()) {
        // SAFETY: see `with_terms_lookup`.
        Some(p) => unsafe { (*p)(index, id, path, None) },
        None => Ok(Vec::new()),
    }
}

impl RtType {
    pub fn name(&self) -> &'static str {
        match self {
            RtType::Long => "long",
            RtType::Double => "double",
            RtType::Keyword => "keyword",
            RtType::Boolean => "boolean",
            RtType::Date => "date",
            RtType::Ip => "ip",
            RtType::Lookup => "lookup",
            RtType::GeoShape => "geo_shape",
        }
    }
}

#[derive(Debug)]
pub struct RuntimeField {
    pub name: String,
    pub ty: RtType,
    pub script: Option<CompiledScript>,
    pub format: DateFormatter,
    pub lookup: Option<LookupSpec>,
}

#[derive(Debug, Default)]
pub struct RuntimeFields {
    pub fields: Vec<RuntimeField>,
}

impl RuntimeFields {
    /// Parse `{"name": {"type": "long", "script": ...}}` (search
    /// `runtime_mappings` or the mapping's `runtime` section); later
    /// definitions override earlier ones.
    pub fn parse_into(&mut self, v: &Value) -> Result<()> {
        self.parse_into_source(v, None)
    }

    pub fn parse_into_source(&mut self, v: &Value, raw: Option<&crate::source_json::Source>) -> Result<()> {
        let o = v.as_object().ok_or_else(|| EsError::mapper_parsing("runtime mappings must be an object"))?;
        for (name, def) in o {
            let d = def.as_object().ok_or_else(|| EsError::mapper_parsing(format!("Expected map for runtime field [{name}] definition but got a {}", if def.is_string() { "java.lang.String" } else { "value" })))?;
            let ty = match d.get("type").and_then(Value::as_str) {
                Some("long") => RtType::Long,
                Some("double") => RtType::Double,
                Some("keyword") => RtType::Keyword,
                Some("boolean") => RtType::Boolean,
                Some("date") => RtType::Date,
                Some("ip") => RtType::Ip,
                Some("lookup") => RtType::Lookup,
                Some("geo_shape") => RtType::GeoShape,
                Some(other) => return Err(EsError::mapper_parsing(format!("No handler for type [{other}] declared on runtime field [{name}]"))),
                None => return Err(EsError::mapper_parsing(format!("No type specified for runtime field [{name}]"))),
            };
            let lookup = if ty == RtType::Lookup {
                let req = |k: &str| d.get(k).and_then(Value::as_str).map(str::to_string).ok_or_else(|| EsError::mapper_parsing(format!("Failed to parse mapping: [{k}] is required for lookup runtime field [{name}]")));
                let fetch_fields = match d.get("fetch_fields") {
                    Some(Value::Array(a)) => a
                        .iter()
                        .filter_map(|x| match x {
                            Value::String(s) => Some(s.clone()),
                            Value::Object(o) => o.get("field").and_then(Value::as_str).map(str::to_string),
                            _ => None,
                        })
                        .collect(),
                    _ => return Err(EsError::mapper_parsing(format!("Failed to parse mapping: [fetch_fields] is required for lookup runtime field [{name}]"))),
                };
                Some(LookupSpec { target_index: req("target_index")?, input_field: req("input_field")?, target_field: req("target_field")?, fetch_fields })
            } else {
                None
            };
            let script = match d.get("script") {
                Some(s) => Some(compile_json_source(s, raw.and_then(|r|r.get(name)).and_then(|r|r.get("script")), ty.context(), None).map_err(|e| {
                    if e.kind == "script_exception" {
                        EsError::mapper_parsing(format!("Failed to parse mapping: compile error")).caused_by(e)
                    } else {
                        e
                    }
                })?),
                None => None,
            };
            let format = match d.get("format").and_then(Value::as_str) {
                Some(f) => DateFormatter::new(f)?,
                None => DateFormatter::new("strict_date_optional_time||epoch_millis")?,
            };
            self.fields.retain(|f| f.name != *name);
            self.fields.push(RuntimeField { name: name.clone(), ty, script, format, lookup });
        }
        Ok(())
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn get(&self, name: &str) -> Option<&RuntimeField> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// `params._source` path lookup (dotted path through objects, or a literal
/// dotted key).
fn source_path(src: &Val, path: &str) -> Vec<Val> {
    fn walk(v: &Val, parts: &[&str], out: &mut Vec<Val>) {
        if parts.is_empty() {
            match v {
                Val::List(l) => out.extend(l.borrow().iter().filter(|x| !x.is_null()).cloned()),
                Val::Array { items, .. } => out.extend(items.borrow().iter().filter(|x| !x.is_null()).cloned()),
                Val::Null => {}
                other => out.push(other.clone()),
            }
            return;
        }
        match v {
            Val::Map(m) => {
                let m = m.borrow();
                for i in (1..=parts.len()).rev() {
                    let key = parts[..i].join(".");
                    if let Some(c) = m.get(&value::Key::str(&key)) {
                        walk(c, &parts[i..], out);
                    }
                }
            }
            Val::List(l) => {
                for x in l.borrow().iter() {
                    walk(x, parts, out);
                }
            }
            Val::Array { items, .. } => {
                for x in items.borrow().iter() {
                    walk(x, parts, out);
                }
            }
            _ => {}
        }
    }
    let parts: Vec<&str> = path.split('.').collect();
    let mut out = Vec::new();
    walk(src, &parts, &mut out);
    out
}

impl RuntimeField {
    pub fn executor(&self, lookup: Option<Arc<FieldLookup>>, runtime: Option<Arc<RuntimeFields>>) -> Executor {
        match &self.script {
            Some(s) => s.executor(lookup, runtime),
            None => {
                // This is an implementation-owned executor for a scriptless
                // runtime field, not a user compilation. A depleted user rate
                // budget must never turn this infallible API into a panic.
                static EMPTY: OnceLock<Arc<Program>> = OnceLock::new();
                let p = EMPTY.get_or_init(|| compile::compile_arc("null", ScriptContext::RuntimeKeyword).expect("internal null program")).clone();
                Executor::new(p, &json!({}), lookup, runtime)
            }
        }
    }

    fn convert(&self, v: Val) -> std::result::Result<Val, Box<ScriptErr>> {
        Ok(match self.ty {
            RtType::Long => match &v {
                Val::Date(m) | Val::Instant(m) => Val::Long(*m),
                Val::Str(s) => Val::Long(s.trim().parse::<f64>().map_err(|_| error::illegal_arg(format!("For input string: \"{s}\"")))? as i64),
                _ => Val::Long(v.to_i64()?),
            },
            RtType::Double => match &v {
                Val::Str(s) => Val::Double(s.trim().parse::<f64>().map_err(|_| error::illegal_arg(format!("For input string: \"{s}\"")))?),
                _ => Val::Double(v.to_f64()?),
            },
            RtType::Lookup | RtType::GeoShape => v,
            // Runtime keyword values cross Lucene's UTF-8 BytesRef boundary.
            // ES replaces unpaired UTF-16 units here; script-field results and
            // stored source retain their original Java string units.
            RtType::Keyword | RtType::Ip => match v {
                Val::Str(_) => v,
                other => Val::string(String::from_utf16_lossy(&other.string_units())),
            },
            RtType::Boolean => match &v {
                Val::Bool(_) => v,
                Val::Str(s) => Val::Bool(s.as_ref() == "true"),
                other => return Err(error::class_cast(format!("Cannot cast {} to boolean", other.type_name()))),
            },
            RtType::Date => match &v {
                Val::Date(m) | Val::Instant(m) => Val::Date(*m),
                Val::Str(s) => Val::Date(self.format.parse(s).map_err(|e| error::illegal_arg(e.reason))?),
                _ => Val::Date(v.to_i64()?),
            },
        })
    }

    /// [`Self::values`] handed to `f` without allocating per doc (the
    /// executor's emit buffer is reused).
    pub fn with_values<R>(&self, exec: &mut Executor, doc: DocId, f: impl FnOnce(&[Val]) -> R) -> std::result::Result<R, Box<ScriptErr>> {
        if self.script.is_none() || matches!(self.ty, RtType::Lookup | RtType::GeoShape) {
            return Ok(f(&self.values(exec, doc)?));
        }
        exec.set_doc(doc);
        exec.rt.emitted.clear();
        exec.run()?;
        let mut raw = std::mem::take(&mut exec.rt.emitted);
        for v in raw.iter_mut() {
            *v = self.convert(std::mem::replace(v, Val::Null))?;
        }
        if raw.len() > 1 {
            raw.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        let r = f(&raw);
        raw.clear();
        exec.rt.emitted = raw;
        Ok(r)
    }

    /// Values of this field for `doc` (sorted like doc values).
    pub fn values(&self, exec: &mut Executor, doc: DocId) -> std::result::Result<Vec<Val>, Box<ScriptErr>> {
        if self.ty == RtType::Lookup {
            return Ok(Vec::new());
        }
        exec.set_doc(doc);
        let raw = match &self.script {
            Some(_) => {
                exec.take_emitted();
                exec.run()?;
                exec.take_emitted()
            }
            None => {
                let src = exec.rt.source()?;
                source_path(&src, &self.name)
            }
        };
        if self.ty == RtType::GeoShape {
            // All emitted geometries form the doc's one shape value.
            let json: Vec<Value> = raw.iter().map(Val::to_json).collect();
            return Ok(crate::fieldtypes::geo::runtime_geometry(&json).map(|g| Val::string(g.encode())).into_iter().collect());
        }
        let mut out = raw.into_iter().map(|v| self.convert(v)).collect::<std::result::Result<Vec<_>, _>>()?;
        if out.len() > 1 {
            out.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(out)
    }

    /// `fields` API rendering of one value.
    pub fn render(&self, v: &Val) -> Value {
        self.render_as(v, None)
    }

    /// `fields` API rendering of one value in `format`.
    pub fn render_as(&self, v: &Val, format: Option<&str>) -> Value {
        match (self.ty, v) {
            (RtType::GeoShape, Val::Str(s)) => match rustic_core::fieldtypes::shape::decode(s) {
                Some(g) => crate::fieldtypes::geo::render(&g, format),
                None => Value::Null,
            },
            (RtType::Date, Val::Date(m)) => Value::String(self.format.format(*m, crate::date::Tz::UTC)),
            (_, v) => v.to_json(),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-request environment (thread-local while a search / update runs).

/// State a request shares with the scripts it compiles: stored scripts,
/// runtime fields, the mapping of the index being parsed, and the first
/// runtime error raised inside a scorer.
pub struct ScriptEnv {
    pub stored: Option<Arc<stored::StoredScripts>>,
    /// Request-level `runtime_mappings`.
    pub runtime_request: Option<Value>,
    pub raw_script_fields: Option<crate::source_json::Source>,
    pub raw_runtime_mappings: Option<crate::source_json::Source>,
    pub raw_sort: Option<crate::source_json::Source>,
    pub raw_search_scripts: Vec<crate::source_json::Source>,
    pub per_index: RefCell<HashMap<String, IndexEnv>>,
    pub current: RefCell<Option<String>>,
    pub error: Arc<Mutex<Option<EsError>>>,
    /// The search's `timeout` deadline (scripts poll it in loops).
    pub cancel: Option<Arc<runtime::Cancel>>,
}

#[derive(Clone)]
pub struct IndexEnv {
    pub lookup: Arc<FieldLookup>,
    pub runtime: Option<Arc<RuntimeFields>>,
}

impl ScriptEnv {
    /// A copy for a worker thread running part of the same request: the
    /// same settings and per-index environments, the same error sink and
    /// timeout.
    pub fn fork(&self) -> ScriptEnv {
        ScriptEnv {
            stored: self.stored.clone(),
            runtime_request: self.runtime_request.clone(),
            raw_script_fields: self.raw_script_fields.clone(),
            raw_runtime_mappings: self.raw_runtime_mappings.clone(),
            raw_sort: self.raw_sort.clone(),
            raw_search_scripts: self.raw_search_scripts.clone(),
            per_index: RefCell::new(self.per_index.borrow().clone()),
            current: RefCell::new(self.current.borrow().clone()),
            error: self.error.clone(),
            cancel: self.cancel.clone(),
        }
    }
}

thread_local! {
    static ENV: RefCell<Option<Rc<ScriptEnv>>> = const { RefCell::new(None) };
}

pub fn env() -> Option<Rc<ScriptEnv>> {
    ENV.with(|e| e.borrow().clone())
}

/// Run `f` with `env` installed (restoring the previous one afterwards).
pub fn with_env<R>(env: ScriptEnv, f: impl FnOnce() -> R) -> R {
    let prev = ENV.with(|e| e.borrow_mut().replace(Rc::new(env)));
    struct Restore(Option<Rc<ScriptEnv>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let p = self.0.take();
            ENV.with(|e| *e.borrow_mut() = p);
        }
    }
    let _r = Restore(prev);
    f()
}

impl ScriptEnv {
    pub fn new(stored: Option<Arc<stored::StoredScripts>>, runtime_request: Option<Value>) -> ScriptEnv {
        ScriptEnv { stored, runtime_request, raw_script_fields: None, raw_runtime_mappings: None, raw_sort: None, raw_search_scripts: Vec::new(), per_index: RefCell::new(HashMap::new()), current: RefCell::new(None), error: Arc::new(Mutex::new(None)), cancel: None }
    }

    pub fn take_error(&self) -> Option<EsError> {
        self.error.lock().take()
    }

    /// A script of this request stopped at the deadline.
    pub fn timed_out(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.timed_out.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// Make `index` the current index of the installed env (its mapping and
/// runtime fields: the mapping's `runtime` section overlaid with the
/// request's `runtime_mappings`).
/// `mapping_runtime` reads the mapping's `runtime` section, called only
/// when a script env is installed and the index is new to it.
pub fn enter_index(index: &str, lookup: &Arc<FieldLookup>, mapping_runtime: impl FnOnce() -> Option<Value>) -> Result<()> {
    let Some(env) = env() else { return Ok(()) };
    *env.current.borrow_mut() = Some(index.to_string());
    if env.per_index.borrow().contains_key(index) {
        return Ok(());
    }
    let mut rf = RuntimeFields::default();
    if let Some(m) = mapping_runtime() {
        rf.parse_into(&m)?;
    }
    if let Some(r) = &env.runtime_request {
        rf.parse_into_source(r, env.raw_runtime_mappings.as_ref())?;
    }
    let runtime = if rf.is_empty() { None } else { Some(Arc::new(rf)) };
    env.per_index.borrow_mut().insert(index.to_string(), IndexEnv { lookup: lookup.clone(), runtime });
    Ok(())
}

/// The env entry of `index` (or of the current index).
pub fn index_env(index: Option<&str>) -> Option<IndexEnv> {
    let env = env()?;
    let name = match index {
        Some(i) => i.to_string(),
        None => env.current.borrow().clone()?,
    };
    env.per_index.borrow().get(&name).cloned()
}

/// Runtime field `name` of the current index, if any.
pub fn runtime_field(index: Option<&str>, name: &str) -> Option<(Arc<RuntimeFields>, usize)> {
    let ie = index_env(index)?;
    let rt = ie.runtime?;
    let i = rt.index_of(name)?;
    Some((rt, i))
}

/// Record the first runtime failure of a scorer (searches check it after
/// collection).
pub fn report_error(slot: &Arc<Mutex<Option<EsError>>>, e: EsError) {
    let mut g = slot.lock();
    if g.is_none() {
        *g = Some(e);
    }
}

// `_scripts/painless/_execute` lives in [`execute_api`].
pub use execute_api::{painless_execute, painless_execute_source};

pub(crate) fn ingest_string_transform(value:&crate::source_json::Source,method:&str)->Result<crate::source_json::Source> {
    let value=source::from_source(value);
    if !matches!(value,Val::Str(_)|Val::Utf16(_)){return Err(crate::error::EsError::illegal_argument("ingest string processor requires a string"));}
    let result=source::string_method(&value,&value.string_units(),method,&[]).map_err(|e|crate::error::EsError::illegal_argument(e.msg.clone()))?;
    source::to_source(&result)
}

pub(crate) fn ingest_regex_transform(value:&crate::source_json::Source,pattern:&crate::source_json::Source,replacement:Option<&crate::source_json::Source>,trailing:bool)->Result<crate::source_json::Source> {
    let value=source::from_source(value);
    if !matches!(value,Val::Str(_)|Val::Utf16(_)){return Err(crate::error::EsError::illegal_argument("ingest string processor requires a string"));}
    let pattern=std::sync::Arc::new(regex::compile_pattern_units(&source::from_source(pattern).string_units(),0).map_err(|e|crate::error::EsError::illegal_argument(e.msg.clone()))?);
    let result=if let Some(replacement)=replacement {Val::utf16(regex::replace_template_units(&pattern,&value.string_units(),&source::from_source(replacement).string_units(),true).map_err(|e|crate::error::EsError::illegal_argument(e.msg.clone()))?)}else{Val::new_list(regex::split_units(&pattern,&value.string_units(),if trailing{-1}else{0}).map_err(|e|e.breaker.clone().unwrap_or_else(||EsError::illegal_argument(e.msg.clone())))?.into_iter().map(Val::utf16).collect())};
    source::to_source(&result)
}

/// Decode only percent-encoded byte runs. Unescaped Java characters must not
/// pass through a UTF8 encoder, which would replace unpaired surrogate units.
pub(crate) fn ingest_url_decode(value:&crate::source_json::Source)->Result<crate::source_json::Source> {
    let value=source_from_json(value);
    if !matches!(&value,Val::Str(_)|Val::Utf16(_)){return Err(EsError::illegal_argument("URL decode field must be a string"));}
    let units=value.string_units();let mut out=Vec::new();let mut i=0;
    while i<units.len(){
        match units[i]{
            43=>{out.push(32);i+=1;},
            37=>{let mut bytes=Vec::new();while i<units.len()&&units[i]==37{
                if i+2>=units.len(){return Err(EsError::illegal_argument("URLDecoder: Incomplete trailing escape (%) pattern"));}
                let digit=|u:u16|match u{48..=57=>Some((u-48)as u8),65..=70=>Some((u-55)as u8),97..=102=>Some((u-87)as u8),_=>None};
                let (Some(a),Some(b))=(digit(units[i+1]),digit(units[i+2]))else{return Err(EsError::illegal_argument("URLDecoder: Illegal hex characters in escape (%) pattern"));};
                bytes.push(a*16+b);i+=3;
            }out.extend(source::decode_java_utf8(&bytes));},
            u=>{out.push(u);i+=1;},
        }
    }
    source_to_json(&Val::utf16(out))
}

pub(crate) fn ingest_regex_split(value:&crate::source_json::Source,pattern:&crate::source_json::Source,limit:i64)->Result<Vec<crate::source_json::Source>> {
    let value=source_from_json(value);
    if !matches!(&value,Val::Str(_)|Val::Utf16(_)){return Err(EsError::illegal_argument("split field must be a string"));}
    let pattern=regex::compile_pattern_units(&source_from_json(pattern).string_units(),0).map_err(|e|EsError::illegal_argument(e.msg.clone()))?;
    regex::split_units(&pattern,&value.string_units(),limit).map_err(|e|e.breaker.clone().unwrap_or_else(||EsError::illegal_argument(e.msg.clone())))?.into_iter().map(|units|source_to_json(&Val::utf16(units))).collect()
}

#[cfg(test)]
mod context_name_tests {
    #[test]
    fn runtime_and_watcher_context_names_are_distinct_and_round_trip() {
        for (name, _) in super::CONTEXT_NAMES {
            assert_eq!(super::ScriptContext::from_name(name).unwrap().name(), *name);
        }
        let _budget = super::rate_limit::fixture_budget("10000/1m");
        let params = serde_json::json!({});
        let long = super::compile("emit(123456789)", &params, super::RtType::Long.context()).unwrap();
        let double = super::compile("emit(123456789)", &params, super::RtType::Double.context()).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&long.program, &double.program));
        assert_eq!(long.program.ctx.name(), "long_field");
        assert_eq!(double.program.ctx.name(), "double_field");
    }
    #[test]
    fn pipeline_script_contexts_match_es_factory_names() {
        assert_eq!(super::ScriptContext::BucketScript.name(), "bucket_aggregation");
        assert_eq!(super::ScriptContext::BucketSelector.name(), "aggregation_selector");
    }
}

#[cfg(test)]
mod cache_lock_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    #[test]
    fn slow_compile_blocks_neither_cache_hits_nor_duplicates_work() {
        // Ingest is exempt from the compilation budget, so the global
        // admission state of other tests is untouched.
        let ctx = ScriptContext::Ingest;
        let warm = ("return 'cache-lock-warm-7d1e'".to_string(), ctx);
        let warmed = cached(warm.clone(), || compile::compile_arc(&warm.0, ctx).map_err(|e| EsError::illegal_argument(e.msg))).unwrap();
        let slow = ("return 'cache-lock-slow-7d1e'".to_string(), ctx);
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let compiles = Arc::new(AtomicUsize::new(0));
        let leader = {
            let (slow, compiles) = (slow.clone(), compiles.clone());
            std::thread::spawn(move || cached(slow.clone(), || {
                compiles.fetch_add(1, Ordering::SeqCst);
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                compile::compile_arc(&slow.0, ctx).map_err(|e| EsError::illegal_argument(e.msg))
            }))
        };
        entered.recv_timeout(Duration::from_secs(10)).expect("compile started");
        // The leader is inside compilation: a hit on another script returns at once.
        let started = Instant::now();
        let hit = cached(warm.clone(), || panic!("a cached program must not recompile")).unwrap();
        assert!(Arc::ptr_eq(&hit, &warmed));
        assert!(started.elapsed() < Duration::from_secs(2), "cache hit waited {:?} on an unrelated compile", started.elapsed());
        // An identical miss waits for the leader rather than compiling again.
        let follower = {
            let slow = slow.clone();
            std::thread::spawn(move || cached(slow, || panic!("identical concurrent miss compiled twice")))
        };
        while !cache().lock().inflight.contains_key(&slow) { std::thread::yield_now(); }
        std::thread::sleep(Duration::from_millis(50));
        release.send(()).unwrap();
        let first = leader.join().unwrap().unwrap();
        let second = follower.join().unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(compiles.load(Ordering::SeqCst), 1);
        assert!(!cache().lock().inflight.contains_key(&slow));
    }

    #[test]
    fn waiters_share_the_leaders_failure_and_the_key_is_retried_later() {
        let ctx = ScriptContext::Ingest;
        let key = ("cache-lock-failure-7d1e".to_string(), ctx);
        let error = cached(key.clone(), || Err(EsError::illegal_argument("boom"))).unwrap_err();
        assert_eq!(error.reason, "boom");
        assert!(!cache().lock().inflight.contains_key(&key));
        let retried = cached(key, || compile::compile_arc("return 1", ctx).map_err(|e| EsError::illegal_argument(e.msg)));
        assert!(retried.is_ok());
    }
}

#[cfg(test)]
mod template_admission_tests {
    use super::*;
    #[test]
    fn templates_count_once_per_uncached_source_against_the_template_budget() {
        let _budget = rate_limit::fixture_budget("1/1m");
        crate::template::admit("{\"query\":{\"match\":{\"f\":\"{{q}}-tmpl-7d1e\"}}}").unwrap();
        // Cached: no further compilation.
        crate::template::admit("{\"query\":{\"match\":{\"f\":\"{{q}}-tmpl-7d1e\"}}}").unwrap();
        let error = crate::template::admit("{\"query\":{\"match\":{\"f\":\"{{q}}-tmpl-other-7d1e\"}}}").unwrap_err();
        assert_eq!(error.status, 429);
        assert!(error.reason.contains("script.context.template.max_compilations_rate") || error.reason.contains("[script.max_compilations_rate]"), "{}", error.reason);
    }
}
