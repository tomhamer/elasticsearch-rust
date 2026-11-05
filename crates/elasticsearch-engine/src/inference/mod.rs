//! The `_inference` API: endpoint registry (persisted under
//! `<path.data>/_state/inference*.json`, secrets kept in a separate file like
//! ES's `.secrets-inference` index), request validation with ES wording, and
//! inference dispatch to local models ([`local`]) or hosted providers
//! ([`providers`]).

pub mod local;
pub mod mock;
pub mod models;
pub mod providers;
pub mod real;
pub mod region_policy;
pub mod semantic;
pub mod semantic_map;
pub mod semantic_query;
pub mod chunk_query;
pub mod highlight;
pub mod torch;

use crate::error::{EsError, Result};
use parking_lot::RwLock;
use providers::{Call, Task};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

const SERVICES_JSON: &str = include_str!("data/services.json");

/// Models the `elasticsearch` service knows without upload.
pub const BUILTIN_MODELS: &[(&str, Task)] = &[
    (".multilingual-e5-small", Task::TextEmbedding),
    (".multilingual-e5-small_linux-x86_64", Task::TextEmbedding),
    (".elser_model_2", Task::SparseEmbedding),
    (".elser_model_2_linux-x86_64", Task::SparseEmbedding),
    (".rerank-v1", Task::Rerank),
];

fn status_ex(status: u16, reason: impl Into<String>) -> EsError {
    EsError::new(status, "status_exception", reason)
}

/// One stored endpoint.
#[derive(Clone, Debug)]
pub struct Endpoint {
    pub inference_id: String,
    pub task_type: Task,
    pub service: String,
    pub service_settings: Map<String, Value>,
    pub task_settings: Map<String, Value>,
    pub chunking_settings: Option<Value>,
    pub secrets: Map<String, Value>,
    pub preconfigured: bool,
}

impl Endpoint {
    /// The stored representation (all service settings, no secrets).
    fn to_stored_json(&self) -> Value {
        let mut v = self.to_json();
        v["service_settings"] = Value::Object(self.service_settings.clone());
        v
    }

    /// The GET representation (secrets never included).
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("inference_id".into(), json!(self.inference_id));
        m.insert("task_type".into(), json!(self.task_type.name()));
        m.insert("service".into(), json!(self.service));
        let ss = if mock::is_mock(&self.service) { mock::filtered_settings(self) } else { self.service_settings.clone() };
        m.insert("service_settings".into(), Value::Object(ss));
        if let Some(c) = &self.chunking_settings {
            m.insert("chunking_settings".into(), c.clone());
        }
        if !self.task_settings.is_empty() {
            m.insert("task_settings".into(), Value::Object(self.task_settings.clone()));
        }
        Value::Object(m)
    }

    fn is_local(&self) -> bool {
        matches!(self.service.as_str(), "elasticsearch" | "elser")
    }

    fn model_id(&self) -> &str {
        self.service_settings.get("model_id").and_then(Value::as_str).unwrap_or("")
    }
}

/// Per-service configuration: supported tasks, allowed `service_settings`
/// keys (secrets are stripped out first) and required secrets.
struct ServiceSpec {
    tasks: &'static [Task],
    settings: &'static [&'static str],
    task_settings: &'static [&'static str],
    secrets: &'static [&'static str],
    required_secret: bool,
}

const T_EMB: Task = Task::TextEmbedding;
const T_SPARSE: Task = Task::SparseEmbedding;
const T_RERANK: Task = Task::Rerank;
const T_COMP: Task = Task::Completion;
const T_CHAT: Task = Task::ChatCompletion;

/// `service_settings` a service requires, in the order ES reports them
/// (validated before the secrets and before any unknown-setting check).
fn required_settings(service: &str, task: Task) -> &'static [&'static str] {
    let _ = task;
    match service {
        "openai" | "googleaistudio" | "anthropic" | "deepseek" | "jinaai" | "voyageai" => &["model_id"],
        "mistral" => &["model"],
        "hugging_face" | "huggingface" => &["url"],
        "azureopenai" => &["resource_name", "deployment_id", "api_version"],
        "azureaistudio" => &["target", "endpoint_type", "provider"],
        "watsonxai" => &["model_id", "project_id", "url", "api_version"],
        "googlevertexai" => &["location", "project_id", "model_id"],
        "alibabacloud-ai-search" => &["service_id", "host", "workspace"],
        _ => &[],
    }
}

fn spec(service: &str) -> Option<ServiceSpec> {
    let rate = "rate_limit";
    let _ = rate;
    Some(match service {
        "openai" => ServiceSpec { tasks: &[T_EMB, T_COMP, T_CHAT], settings: &["model_id", "url", "organization_id", "dimensions", "similarity", "rate_limit", "max_input_tokens"], task_settings: &["user", "headers"], secrets: &["api_key"], required_secret: true },
        "azureopenai" => ServiceSpec { tasks: &[T_EMB, T_COMP], settings: &["resource_name", "deployment_id", "api_version", "dimensions", "similarity", "rate_limit", "max_input_tokens"], task_settings: &["user"], secrets: &["api_key", "entra_id"], required_secret: true },
        "azureaistudio" => ServiceSpec { tasks: &[T_EMB, T_COMP], settings: &["target", "provider", "endpoint_type", "dimensions", "similarity", "rate_limit", "max_input_tokens"], task_settings: &["user", "temperature", "top_p", "do_sample", "max_new_tokens"], secrets: &["api_key"], required_secret: true },
        "cohere" => ServiceSpec { tasks: &[T_EMB, T_RERANK, T_COMP], settings: &["model_id", "url", "embedding_type", "similarity", "dimensions", "rate_limit", "max_input_tokens"], task_settings: &["input_type", "truncate", "top_n", "return_documents", "max_chunks_per_doc"], secrets: &["api_key"], required_secret: true },
        "hugging_face" | "huggingface" => ServiceSpec { tasks: &[T_EMB, T_SPARSE, T_RERANK, T_COMP, T_CHAT], settings: &["url", "model_id", "dimensions", "similarity", "rate_limit", "max_input_tokens"], task_settings: &["return_documents", "top_n"], secrets: &["api_key"], required_secret: true },
        "mistral" => ServiceSpec { tasks: &[T_EMB, T_COMP, T_CHAT], settings: &["model", "model_id", "max_input_tokens", "rate_limit", "dimensions", "similarity"], task_settings: &[], secrets: &["api_key"], required_secret: true },
        "googleaistudio" => ServiceSpec { tasks: &[T_EMB, T_COMP], settings: &["model_id", "rate_limit", "dimensions", "similarity"], task_settings: &[], secrets: &["api_key"], required_secret: true },
        "googlevertexai" => ServiceSpec { tasks: &[T_EMB, T_RERANK, T_COMP, T_CHAT], settings: &["model_id", "location", "project_id", "dimensions", "similarity", "rate_limit", "max_input_tokens"], task_settings: &["auto_truncate", "top_n", "thinking_config"], secrets: &["service_account_json"], required_secret: true },
        "anthropic" => ServiceSpec { tasks: &[T_COMP], settings: &["model_id", "rate_limit", "url"], task_settings: &["max_tokens", "temperature", "top_p", "top_k"], secrets: &["api_key"], required_secret: true },
        "amazonbedrock" => ServiceSpec { tasks: &[T_EMB, T_COMP], settings: &["region", "model", "provider", "dimensions", "similarity", "rate_limit", "max_input_tokens"], task_settings: &["max_new_tokens", "temperature", "top_p", "top_k"], secrets: &["access_key", "secret_key"], required_secret: true },
        "alibabacloud-ai-search" => ServiceSpec { tasks: &[T_EMB, T_SPARSE, T_RERANK, T_COMP], settings: &["service_id", "host", "workspace", "http_schema", "rate_limit", "dimensions"], task_settings: &["input_type", "return_token"], secrets: &["api_key"], required_secret: true },
        "watsonxai" => ServiceSpec { tasks: &[T_EMB, T_RERANK], settings: &["url", "api_version", "project_id", "model_id", "max_input_tokens", "rate_limit"], task_settings: &["truncate_input_tokens", "top_n", "return_documents"], secrets: &["api_key"], required_secret: true },
        "jinaai" => ServiceSpec { tasks: &[T_EMB, T_RERANK], settings: &["model_id", "url", "similarity", "dimensions", "rate_limit", "embedding_type"], task_settings: &["task", "input_type", "return_documents", "top_n", "late_chunking"], secrets: &["api_key"], required_secret: true },
        "voyageai" => ServiceSpec { tasks: &[T_EMB, T_RERANK], settings: &["model_id", "url", "dimensions", "similarity", "embedding_type", "rate_limit"], task_settings: &["input_type", "truncation", "return_documents", "top_k"], secrets: &["api_key"], required_secret: true },
        "deepseek" => ServiceSpec { tasks: &[T_COMP, T_CHAT], settings: &["model_id", "url"], task_settings: &[], secrets: &["api_key"], required_secret: true },
        "amazon_sagemaker" => ServiceSpec { tasks: &[T_EMB, T_SPARSE, T_RERANK, T_COMP, T_CHAT], settings: &["access_key", "secret_key", "region", "endpoint_name", "api", "batch_size", "dimensions"], task_settings: &["custom_attributes", "enable_explanations", "inference_id", "session_id", "target_variant"], secrets: &["access_key", "secret_key"], required_secret: false },
        "elasticsearch" => ServiceSpec { tasks: &[T_EMB, T_SPARSE, T_RERANK], settings: &["model_id", "num_allocations", "num_threads", "adaptive_allocations", "deployment_id", "dimensions", "similarity", "element_type"], task_settings: &["return_documents"], secrets: &[], required_secret: false },
        "elser" => ServiceSpec { tasks: &[T_SPARSE], settings: &["model_id", "num_allocations", "num_threads", "adaptive_allocations"], task_settings: &[], secrets: &[], required_secret: false },
        _ => return None,
    })
}

/// Java `Map.toString` rendering of unknown settings (ES error text).
fn java_map(m: &Map<String, Value>) -> String {
    let parts: Vec<String> = m
        .iter()
        .map(|(k, v)| {
            let r = match v {
                Value::String(s) => s.clone(),
                Value::Object(o) => java_map(o),
                Value::Array(a) => format!("[{}]", a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
                other => other.to_string(),
            };
            format!("{k}={r}")
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// Provider failure text (`providers::execute` yields `status|detail`).
fn provider_error_text(status: u16, id: &str, msg: &str) -> String {
    let detail = msg.rsplit_once("Error message: [").map(|(_, r)| r.trim_end_matches(']').to_string()).unwrap_or_else(|| msg.to_string());
    let what = match status {
        401 => "an authentication error",
        429 => "a rate limit",
        500..=599 => "a server error",
        _ => "an unsuccessful",
    };
    format!("Received {what} status code for request from inference entity id [{id}] status [{status}]. Error message: [{detail}]")
}

fn default_rate(service: &str, task: Task) -> i64 {
    match (service, task) {
        ("openai", Task::TextEmbedding) => 3000,
        ("openai", _) => 500,
        ("cohere", _) => 10000,
        ("azureopenai", _) => 1440,
        ("hugging_face" | "huggingface", _) => 3000,
        ("mistral", _) => 240,
        ("googleaistudio", _) => 360,
        ("googlevertexai", _) => 30000,
        ("anthropic", _) => 50,
        ("amazonbedrock", _) => 240,
        ("alibabacloud-ai-search", _) => 3000,
        ("watsonxai", _) => 120,
        ("jinaai" | "voyageai", _) => 2000,
        _ => 240,
    }
}

/// Order and default the hosted-service settings the way ES renders them.
fn canonical_settings(service: &str, task: Task, mut ss: Map<String, Value>, probed_dims: Option<usize>) -> Map<String, Value> {
    let embed = task.is_dense();
    let rate = ss.shift_remove("rate_limit").unwrap_or_else(|| json!({"requests_per_minute": default_rate(service, task)}));
    let sim = ss.shift_remove("similarity").unwrap_or_else(|| json!(if service == "openai" { "dot_product" } else { "cosine" }));
    let dims = ss.shift_remove("dimensions").or(probed_dims.map(|d| json!(d)));
    let mut out = Map::new();
    let mut take = |out: &mut Map<String, Value>, ss: &mut Map<String, Value>, k: &str| {
        if let Some(v) = ss.shift_remove(k) {
            out.insert(k.to_string(), v);
        }
    };
    let sd = |out: &mut Map<String, Value>| {
        if embed {
            out.insert("similarity".into(), sim.clone());
            if let Some(d) = &dims {
                out.insert("dimensions".into(), d.clone());
            }
        }
    };
    match service {
        "cohere" if embed => {
            take(&mut out, &mut ss, "url");
            sd(&mut out);
            take(&mut out, &mut ss, "model_id");
            out.insert("rate_limit".into(), rate);
            let et = ss.shift_remove("embedding_type").unwrap_or(json!("float"));
            out.insert("embedding_type".into(), et);
        }
        "cohere" => {
            take(&mut out, &mut ss, "url");
            take(&mut out, &mut ss, "model_id");
            out.insert("rate_limit".into(), rate);
        }
        _ => {
            for k in ["model_id", "url", "organization_id"] {
                take(&mut out, &mut ss, k);
            }
            sd(&mut out);
            out.insert("rate_limit".into(), rate);
        }
    }
    for (k, v) in ss {
        out.entry(k).or_insert(v);
    }
    out
}

fn valid_id(id: &str) -> bool {
    let b = id.as_bytes();
    !b.is_empty() && b.iter().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-' || *c == b'_') && b[0].is_ascii_alphanumeric() && b[b.len() - 1].is_ascii_alphanumeric()
}

fn defaults() -> Vec<Endpoint> {
    let alloc = json!({"enabled": true, "min_number_of_allocations": 0, "max_number_of_allocations": 32});
    let chunking = json!({"strategy": "sentence", "max_chunk_size": 250, "sentence_overlap": 1});
    let mk = |id: &str, task: Task, model: &str, chunk: bool, ts: Value| Endpoint {
        inference_id: id.into(),
        task_type: task,
        service: "elasticsearch".into(),
        service_settings: json!({"num_threads": 1, "model_id": model, "adaptive_allocations": alloc}).as_object().unwrap().clone(),
        task_settings: ts.as_object().cloned().unwrap_or_default(),
        chunking_settings: chunk.then(|| chunking.clone()),
        secrets: Map::new(),
        preconfigured: true,
    };
    vec![
        mk(".elser-2-elasticsearch", Task::SparseEmbedding, ".elser_model_2_linux-x86_64", true, json!({})),
        mk(".multilingual-e5-small-elasticsearch", Task::TextEmbedding, ".multilingual-e5-small_linux-x86_64", true, json!({})),
        mk(".rerank-v1-elasticsearch", Task::Rerank, ".rerank-v1", false, json!({"return_documents": true})),
    ]
}

/// Result of embedding inputs for indexing/search.
pub enum Embedded {
    Dense(Vec<Vec<f32>>),
    Sparse(Vec<Map<String, Value>>),
}

pub struct InferenceService {
    dir: PathBuf,
    eps: RwLock<BTreeMap<String, Endpoint>>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<InferenceService>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<InferenceService>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

static CURRENT: RwLock<Option<Arc<InferenceService>>> = RwLock::new(None);

thread_local! {
    /// The inference service of the node whose request this thread is
    /// serving (several in-process nodes share one process in tests).
    static SCOPED: std::cell::RefCell<Option<Arc<InferenceService>>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` with `svc` as this thread's [`current`] service.
pub fn scoped<R>(svc: Option<Arc<InferenceService>>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<InferenceService>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            SCOPED.with(|s| *s.borrow_mut() = prev);
        }
    }
    let _restore = Restore(SCOPED.with(|s| std::mem::replace(&mut *s.borrow_mut(), svc)));
    f()
}

/// The service [`scoped`] set on this thread, if any (to hand to a worker thread).
pub fn scoped_current() -> Option<Arc<InferenceService>> {
    SCOPED.with(|s| s.borrow().clone())
}

/// Record the node's inference service as the one query/index-time hooks
/// (`semantic_text`, `knn.query_vector_builder`) resolve endpoints from.
pub fn attach(data: &Path) {
    if let Some(c) = CURRENT.read().as_ref() {
        if c.dir.parent() == Some(data) {
            return;
        }
    }
    *CURRENT.write() = Some(service_for(data));
}

/// The attached inference service, if any request has attached one.
pub fn current() -> Option<Arc<InferenceService>> {
    scoped_current().or_else(|| CURRENT.read().clone())
}

/// The inference service of the node whose data path is `data`.
pub fn service_for(data: &Path) -> Arc<InferenceService> {
    // Every REST request asks (several in-process nodes may share a
    // thread): the last answer of this thread, before the shared registry
    // (a process-wide lock and a path copy per request).
    thread_local! {
        static LAST: std::cell::RefCell<Option<(std::path::PathBuf, Arc<InferenceService>)>> = const { std::cell::RefCell::new(None) };
    }
    if let Some(hit) = LAST.with(|l| l.borrow().as_ref().filter(|(p, _)| p == data).map(|(_, s)| s.clone())) {
        return hit;
    }
    let service = {
        let mut r = registry().lock().unwrap_or_else(|e| e.into_inner());
        match r.get(data) {
            Some(s) => s.clone(),
            None => r.entry(data.to_path_buf()).or_insert_with(|| Arc::new(InferenceService::open(data))).clone(),
        }
    };
    LAST.with(|l| *l.borrow_mut() = Some((data.to_path_buf(), service.clone())));
    service
}

impl InferenceService {
    /// The node's `_state` directory (trained models live next to it).
    pub fn state_dir(&self) -> &Path {
        &self.dir
    }

    pub fn open(data: &Path) -> InferenceService {
        let dir = data.join("_state");
        let _ = std::fs::create_dir_all(&dir);
        let mut eps = BTreeMap::new();
        for d in defaults() {
            eps.insert(d.inference_id.clone(), d);
        }
        let stored: Value = std::fs::read(dir.join("inference.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let secrets: Value = std::fs::read(dir.join("inference-secrets.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        if let Some(arr) = stored.as_array() {
            for e in arr {
                if let Some(ep) = Self::decode(e, &secrets) {
                    eps.insert(ep.inference_id.clone(), ep);
                }
            }
        }
        InferenceService { dir, eps: RwLock::new(eps) }
    }

    fn decode(e: &Value, secrets: &Value) -> Option<Endpoint> {
        let id = e.get("inference_id")?.as_str()?.to_string();
        Some(Endpoint {
            task_type: Task::parse(e.get("task_type")?.as_str()?)?,
            service: e.get("service")?.as_str()?.to_string(),
            service_settings: e.get("service_settings")?.as_object()?.clone(),
            task_settings: e.get("task_settings").and_then(Value::as_object).cloned().unwrap_or_default(),
            chunking_settings: e.get("chunking_settings").cloned(),
            secrets: secrets.get(&id).and_then(Value::as_object).cloned().unwrap_or_default(),
            inference_id: id,
            preconfigured: false,
        })
    }

    fn persist(&self) -> Result<()> {
        let eps = self.eps.read();
        let user: Vec<&Endpoint> = eps.values().filter(|e| !e.preconfigured).collect();
        let list: Vec<Value> = user.iter().map(|e| e.to_stored_json()).collect();
        let mut sec = Map::new();
        for e in &user {
            if !e.secrets.is_empty() {
                sec.insert(e.inference_id.clone(), Value::Object(e.secrets.clone()));
            }
        }
        std::fs::write(self.dir.join("inference.json"), serde_json::to_vec_pretty(&Value::Array(list))?)?;
        std::fs::write(self.dir.join("inference-secrets.json"), serde_json::to_vec_pretty(&Value::Object(sec))?)?;
        Ok(())
    }

    /// The user-created endpoints (with their secrets) as cluster metadata:
    /// `{"endpoints": [stored json], "secrets": {id: secrets}}`. ES keeps
    /// them in the `.inference` system index, readable from every node; a
    /// cluster publishes this so every node's engine can resolve them
    /// (semantic_text at index time, inference queries on each shard).
    pub fn cluster_state(&self) -> Value {
        let eps = self.eps.read();
        let user: Vec<&Endpoint> = eps.values().filter(|e| !e.preconfigured).collect();
        let mut sec = Map::new();
        for e in &user {
            if !e.secrets.is_empty() {
                sec.insert(e.inference_id.clone(), Value::Object(e.secrets.clone()));
            }
        }
        json!({"endpoints": user.iter().map(|e| e.to_stored_json()).collect::<Vec<_>>(), "secrets": sec})
    }

    /// Make the user-created endpoints exactly those of [`cluster_state`]
    /// (preconfigured defaults stay). Persisted only when something changed.
    pub fn restore_cluster_state(&self, state: &Value) -> Result<()> {
        let secrets = state.get("secrets").cloned().unwrap_or_else(|| json!({}));
        let want: BTreeMap<String, Endpoint> = state["endpoints"].as_array().into_iter().flatten()
            .filter_map(|e| Self::decode(e, &secrets)).map(|e| (e.inference_id.clone(), e)).collect();
        {
            let mut eps = self.eps.write();
            let same = eps.values().filter(|e| !e.preconfigured).count() == want.len()
                && want.iter().all(|(id, w)| eps.get(id).is_some_and(|e| !e.preconfigured && e.to_stored_json() == w.to_stored_json() && e.secrets == w.secrets));
            if same {
                return Ok(());
            }
            eps.retain(|_, e| e.preconfigured);
            for (id, e) in want {
                eps.entry(id).or_insert(e);
            }
        }
        self.persist()
    }

    /// Reset stored endpoints and secrets while preserving configured defaults.
    pub fn reset(&self) -> Result<()> {
        self.eps.write().retain(|_, endpoint| endpoint.preconfigured);
        self.persist()
    }

    // ------------------------------------------------------------ read

    fn parse_task(t: &str) -> Result<Task> {
        Task::parse(t).ok_or_else(|| status_ex(400, format!("Unknown task_type [{t}]")))
    }

    pub fn get_endpoint(&self, id: &str) -> Option<Endpoint> {
        self.eps.read().get(id).cloned()
    }

    /// `TransportInferenceUsageAction.collectUsage`: per (service, task)
    /// endpoint counts, `_all` per task type and per default model
    /// (`_{service}_{model}`), with semantic_text stats from `fields`
    /// (`(index, inference_id)` of each non-system, non-hidden index's
    /// inference fields). Ordered by stats key, as ES's `TreeMap`.
    pub fn usage(&self, fields: &[(String, String)]) -> Value {
        const LINUX_SUFFIX: &str = "_linux-x86_64";
        let strip = |m: &str| m.strip_suffix(LINUX_SUFFIX).unwrap_or(m).to_string();
        let eps: Vec<Endpoint> = self.eps.read().values().cloned().collect();
        let by_id: HashMap<&str, &Endpoint> = eps.iter().map(|e| (e.inference_id.as_str(), e)).collect();
        let model_of = |e: &Endpoint| e.service_settings.get("model_id").and_then(Value::as_str).map(strip);
        let task_key = |t: Task| t.name().to_ascii_uppercase();
        // key -> (service, task, count, matching fields)
        let mut stats: BTreeMap<String, (String, String, u64, Vec<(String, String)>)> = BTreeMap::new();
        let fields_where = |pred: &dyn Fn(&Endpoint) -> bool| -> Vec<(String, String)> {
            fields.iter().filter(|(_, id)| by_id.get(id.as_str()).is_some_and(|e| pred(e))).cloned().collect()
        };
        for e in &eps {
            let t = task_key(e.task_type);
            stats.entry(format!("{}:{t}", e.service)).or_insert_with(|| (e.service.clone(), t.clone(), 0, Vec::new())).2 += 1;
            stats.entry(format!("_all:{t}")).or_insert_with(|| ("_all".into(), t.clone(), 0, Vec::new())).2 += 1;
        }
        for v in stats.values_mut() {
            let (svc, t) = (v.0.clone(), v.1.clone());
            v.3 = fields_where(&|e: &Endpoint| (svc == "_all" || e.service == svc) && task_key(e.task_type) == t);
        }
        for t in [Task::TextEmbedding, Task::SparseEmbedding, Task::Rerank, Task::Completion, Task::ChatCompletion, Task::Embedding] {
            let tk = task_key(t);
            stats.entry(format!("_all:{tk}")).or_insert_with(|| ("_all".into(), tk.clone(), 0, fields_where(&|e: &Endpoint| e.task_type == t)));
        }
        let defaults: std::collections::BTreeSet<String> = eps.iter().filter(|e| e.preconfigured).filter_map(model_of).collect();
        let mut dm: BTreeMap<String, (String, Task, String, u64)> = BTreeMap::new();
        for e in &eps {
            let Some(m) = model_of(e).filter(|m| defaults.contains(m)) else { continue };
            let key = format!("_{}_{}", e.service, m.replace('.', "_"));
            dm.entry(format!("{key}|{}", task_key(e.task_type))).or_insert_with(|| (key, e.task_type, m.clone(), 0)).3 += 1;
        }
        for (key, task, m, n) in dm.into_values() {
            let fs = fields_where(&|e: &Endpoint| e.task_type == task && model_of(e).as_deref() == Some(m.as_str()) && key == format!("_{}_{}", e.service, m.replace('.', "_")));
            stats.insert(key.clone(), (key, task_key(task), n, fs));
        }
        let models: Vec<Value> = stats
            .into_values()
            .map(|(service, task, count, fs)| {
                let mut o = json!({"service": service, "task_type": task, "count": count});
                if !fs.is_empty() {
                    let indices: std::collections::BTreeSet<&str> = fs.iter().map(|(i, _)| i.as_str()).collect();
                    let ids: std::collections::BTreeSet<&str> = fs.iter().map(|(_, id)| id.as_str()).collect();
                    o["semantic_text"] = json!({"field_count": fs.len(), "indices_count": indices.len(), "inference_id_count": ids.len()});
                }
                o
            })
            .collect();
        json!({"available": true, "enabled": true, "models": models})
    }

    /// `GET /_inference[/{task_type}[/{id}]]`; `id` may be `_all`.
    pub fn get(&self, task: Option<&str>, id: Option<&str>) -> Result<Value> {
        let task = task.map(Self::parse_task).transpose()?;
        let eps = self.eps.read();
        match id {
            None | Some("_all") | Some("*") => Ok(json!({"endpoints": eps.values().filter(|e| task.is_none_or(|t| e.task_type == t)).map(Endpoint::to_json).collect::<Vec<_>>()})),
            Some(id) if id.contains('*') => {
                let re = id.replace('*', "");
                let hits: Vec<Value> = eps.values().filter(|e| task.is_none_or(|t| e.task_type == t) && (e.inference_id.starts_with(&re) || e.inference_id.contains(&re))).map(Endpoint::to_json).collect();
                Ok(json!({"endpoints": hits}))
            }
            Some(id) => match eps.get(id) {
                Some(e) if task.is_none_or(|t| e.task_type == t) => Ok(json!({"endpoints": [e.to_json()]})),
                Some(e) => Err(status_ex(400, format!("Requested task type [{}] does not match the inference endpoint's task type [{}]", task.unwrap().name(), e.task_type.name()))),
                None => Err(EsError::resource_not_found(format!("Inference endpoint not found [{id}]"))),
            },
        }
    }

    /// `GET /_inference/_services[/{task_type}]`.
    pub fn services(task: Option<&str>) -> Result<Value> {
        let task = task.map(Self::parse_task).transpose()?;
        let all: Value = serde_json::from_str(SERVICES_JSON).map_err(|e| EsError::internal(e.to_string()))?;
        let Some(t) = task else { return Ok(all) };
        let out: Vec<Value> = all.as_array().into_iter().flatten().filter(|s| s["task_types"].as_array().is_some_and(|a| a.iter().any(|x| x == t.name()))).cloned().collect();
        Ok(Value::Array(out))
    }

    // ------------------------------------------------------------ write

    /// `PUT /_inference/{task_type}/{id}`.
    pub fn put(&self, task: &str, id: &str, body: &Value) -> Result<Value> {
        let task = Self::parse_task(task)?;
        if !valid_id(id) {
            return Err(EsError::action_validation(&[&format!("Invalid inference_id; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric")]));
        }
        if id.starts_with('.') || self.eps.read().get(id).is_some_and(|e| e.preconfigured) {
            return Err(status_ex(400, format!("[{id}] is a reserved inference ID")));
        }
        if self.eps.read().contains_key(id) {
            return Err(EsError::new(400, "resource_already_exists_exception", format!("Inference endpoint [{id}] already exists")));
        }
        let o = body.as_object().ok_or_else(|| EsError::parsing("request body must be an object"))?;
        let service = o.get("service").and_then(Value::as_str).ok_or_else(|| status_ex(400, "Inference endpoint configuration is missing the [service] setting"))?;
        if mock::is_registered(service) {
            let ep = mock::parse(task, id, service, o)?;
            let out = ep.to_json();
            self.eps.write().insert(id.to_string(), ep);
            self.persist()?;
            return Ok(out);
        }
        let Some(sp) = spec(service) else { return Err(status_ex(400, format!("Unknown service [{service}]"))) };
        if !sp.tasks.contains(&task) {
            return Err(status_ex(400, format!("The [{service}] service does not support task type [{}]", task.name())));
        }
        if o.get("service_settings").is_none_or(Value::is_null) {
            return Err(status_ex(400, "Missing required field [service_settings]"));
        }
        let mut ss = o.get("service_settings").and_then(Value::as_object).cloned().unwrap_or_default();
        let mut ts = o.get("task_settings").and_then(Value::as_object).cloned().unwrap_or_default();
        // Model construction: required settings, then secrets.
        if service == "cohere" && !ss.contains_key("model_id") {
            return Err(EsError::new(400, "validation_exception", "Validation Failed: 1: The [service_settings.model_id] field is required for the Cohere V2 API.;"));
        }
        let missing: Vec<String> = required_settings(service, task).iter().filter(|k| ss.get(**k).is_none_or(Value::is_null)).map(|k| format!("[service_settings] does not contain the required setting [{k}]")).collect();
        if !missing.is_empty() {
            let refs: Vec<&str> = missing.iter().map(String::as_str).collect();
            let mut e = EsError::action_validation(&refs);
            e.kind = "validation_exception".into();
            return Err(e);
        }
        let mut secrets = Map::new();
        for k in sp.secrets {
            if let Some(v) = ss.shift_remove(*k) {
                secrets.insert((*k).to_string(), v);
            }
        }
        if sp.required_secret && secrets.is_empty() {
            let need = sp.secrets.first().copied().unwrap_or("api_key");
            return Err(EsError::new(400, "validation_exception", format!("Validation Failed: 1: [secret_settings] does not contain the required setting [{need}];")));
        }
        // Leftovers (`throwIfNotEmptyMap`): config, service and task settings.
        for k in o.keys() {
            if !matches!(k.as_str(), "service" | "service_settings" | "task_settings" | "chunking_settings") {
                return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{service}] service", java_map(&Map::from_iter([(k.clone(), o[k].clone())])))));
            }
        }
        // `access_key`/`secret_key` are allowed settings for sagemaker only.
        let unknown: Map<String, Value> = ss.iter().filter(|(k, _)| !sp.settings.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
        if !unknown.is_empty() {
            return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{service}] service", java_map(&unknown))));
        }
        if let Some(rl) = ss.get("rate_limit").and_then(Value::as_object) {
            let bad: Map<String, Value> = rl.iter().filter(|(k, _)| k.as_str() != "requests_per_minute").map(|(k, v)| (k.clone(), v.clone())).collect();
            if !bad.is_empty() {
                return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{service}] service", java_map(&bad))));
            }
        }
        let unknown_t: Map<String, Value> = ts.iter().filter(|(k, _)| !sp.task_settings.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();
        if !unknown_t.is_empty() {
            return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{service}] service", java_map(&unknown_t))));
        }
        let local = matches!(service, "elasticsearch" | "elser");
        let mut chunking = o.get("chunking_settings").cloned();
        if matches!(task, Task::TextEmbedding | Task::Embedding | Task::SparseEmbedding) && chunking.is_none() {
            chunking = Some(json!({"strategy": "sentence", "max_chunk_size": 250, "sentence_overlap": 1}));
        }
        if matches!(task, Task::Rerank | Task::Completion | Task::ChatCompletion) {
            chunking = None;
        }
        if let Some(c) = chunking.as_ref().and_then(Value::as_object) {
            if c.get("strategy").and_then(Value::as_str) == Some("word") && !c.contains_key("overlap") {
                return Err(EsError::new(400, "validation_exception", "Validation Failed: 1: [chunking_settings] does not contain the required setting [overlap];"));
            }
        }
        let mut service_out = service.to_string();
        if local {
            service_out = "elasticsearch".into();
            let model = match ss.get("model_id").and_then(Value::as_str) {
                Some(m) => m.to_string(),
                None if service == "elser" => ".elser_model_2_linux-x86_64".to_string(),
                None => return Err(EsError::illegal_argument("Error parsing service settings, model_id must be provided")),
            };
            let known = BUILTIN_MODELS.iter().any(|(m, t)| *m == model && *t == task) || models::exists(&self.dir, &model);
            if !known {
                return Err(EsError::resource_not_found(format!("No known trained model with model_id [{model}], you may need to create it or load it into the cluster with eland")));
            }
            let mut out = Map::new();
            out.insert("num_allocations".into(), ss.get("num_allocations").cloned().unwrap_or(json!(1)));
            out.insert("num_threads".into(), ss.get("num_threads").cloned().unwrap_or(json!(1)));
            out.insert("model_id".into(), json!(model));
            for (k, v) in &ss {
                out.entry(k.clone()).or_insert(v.clone());
            }
            ss = out;
            if task == Task::Rerank && !ts.contains_key("return_documents") {
                ts.insert("return_documents".into(), json!(true));
            }
        }
        let mut ep = Endpoint { inference_id: id.into(), task_type: task, service: service_out, service_settings: ss, task_settings: ts, chunking_settings: chunking, secrets, preconfigured: false };
        if !local {
            // Validation call: embeds a probe string; learns dimensions.
            let probe = vec!["how big".to_string()];
            let c = Call { service: &ep.service, task: ep.task_type, service_settings: &ep.service_settings, secrets: &ep.secrets, task_settings: ep.task_settings.clone(), input: &probe, query: Some("query") };
            match providers::execute(&c, Duration::from_secs(30)) {
                Ok(res) => {
                    let dims = res["text_embedding"][0]["embedding"].as_array().map(Vec::len);
                    ep.service_settings = canonical_settings(&ep.service, task, std::mem::take(&mut ep.service_settings), dims);
                }
                Err((status, msg)) => {
                    let msg = if msg.starts_with("Error message: [") { provider_error_text(status, id, &msg) } else { msg };
                    let label = match (service, task) {
                        ("openai", Task::TextEmbedding) => "OpenAI embeddings".to_string(),
                        ("openai", _) => "OpenAI chat completions".to_string(),
                        (s, t) => format!("{s} {}", t.name()),
                    };
                    let conn = msg.starts_with("Connection refused");
                    let root = if conn { format!("Failed to send {label} request. Cause: {msg}") } else { msg.clone() };
                    let mut cause = status_ex(400, root.clone());
                    if conn {
                        cause = cause.caused_by(EsError::new(400, "connect_exception", msg));
                    }
                    return Err(status_ex(400, "Could not complete inference endpoint creation as validation call to service threw an exception.").caused_by(cause));
                }
            }
        }
        let out = ep.to_json();
        self.eps.write().insert(id.to_string(), ep);
        self.persist()?;
        Ok(out)
    }

    /// The task type of a put/update: the URL's, or the body's `task_type`.
    fn resolve_task(url_task: Option<&str>, body: &Value) -> Result<String> {
        let body_task = body.get("task_type").and_then(Value::as_str);
        match (url_task, body_task) {
            (Some(u), Some(b)) if u != b => {
                Err(status_ex(400, format!("Cannot resolve conflicting task_type parameter in the request URL [{u}] and the request body [{b}]")))
            }
            (Some(u), _) => Ok(u.to_string()),
            (None, Some(b)) => Self::parse_task(b).map(|t| t.name().to_string()),
            (None, None) => Err(status_ex(400, "model is missing required setting [task_type]")),
        }
    }

    fn check_id(id: &str) -> Result<()> {
        if valid_id(id) {
            Ok(())
        } else {
            Err(EsError::action_validation(&[&format!("Invalid inference_id; '{id}' can contain lowercase alphanumeric (a-z and 0-9), hyphens or underscores; must start and end with alphanumeric")]))
        }
    }

    /// `PUT /_inference[/{task_type}]/{id}`: the task type comes from the
    /// URL or the body (`task_type`).
    pub fn put_resolving(&self, url_task: Option<&str>, id: &str, body: &Value) -> Result<Value> {
        if let Some(t) = url_task {
            Self::parse_task(t)?;
        }
        Self::check_id(id)?;
        let task = Self::resolve_task(url_task, body)?;
        let mut body = body.clone();
        if let Some(o) = body.as_object_mut() {
            o.shift_remove("task_type");
        }
        self.put(&task, id, &body)
    }

    /// `PUT /_inference[/{task_type}]/{id}/_update`: new `service_settings`
    /// (secrets included) and `task_settings`, merged into the endpoint.
    pub fn update(&self, url_task: Option<&str>, id: &str, body: &Value) -> Result<Value> {
        if let Some(t) = url_task {
            Self::parse_task(t)?;
        }
        let o = body.as_object().cloned().unwrap_or_default();
        if o.is_empty() {
            return Err(status_ex(400, "Request body is empty"));
        }
        Self::check_id(id)?;
        let bad: Map<String, Value> = o.iter().filter(|(k, _)| !matches!(k.as_str(), "service_settings" | "task_settings" | "task_type")).map(|(k, v)| (k.clone(), v.clone())).collect();
        if !bad.is_empty() {
            return Err(status_ex(400, format!("Request contained fields which cannot be updated, remove these fields and try again [{}]", java_map(&bad))));
        }
        for k in ["service_settings", "task_settings"] {
            if let Some(v) = o.get(k).filter(|v| !v.is_object() && !v.is_null()) {
                let what = if k == "service_settings" { "service settings" } else { "task settings" };
                return Err(status_ex(400, format!("Unable to parse {what} in the request [{}]", java_map(&Map::from_iter([(k.to_string(), v.clone())])))));
            }
        }
        let task = Self::parse_task(&Self::resolve_task(url_task, body)?)?;
        let mut eps = self.eps.write();
        let ep = match eps.get_mut(id) {
            Some(ep) if !ep.preconfigured => ep,
            _ => return Err(status_ex(404, format!("The inference endpoint [{id}] does not exist and cannot be updated"))),
        };
        if ep.task_type != task {
            return Err(status_ex(400, format!(
                "Task type must match the task type of the existing endpoint, received [{}] but existing endpoint has [{}]",
                task.name(),
                ep.task_type.name()
            )));
        }
        if mock::is_mock(&ep.service) {
            if let Some(k) = o.get("service_settings").and_then(|v| v.get("api_key")) {
                ep.secrets.insert("api_key".into(), k.clone());
            }
            if let Some(ts) = o.get("task_settings").and_then(Value::as_object) {
                for (k, v) in ts {
                    ep.task_settings.insert(k.clone(), v.clone());
                }
            }
            let out = ep.to_json();
            drop(eps);
            self.persist()?;
            return Ok(out);
        }
        let Some(sp) = spec(&ep.service) else { return Err(status_ex(400, format!("Unknown service [{}]", ep.service))) };
        if let Some(ss) = o.get("service_settings").and_then(Value::as_object) {
            for (k, v) in ss {
                if sp.secrets.contains(&k.as_str()) {
                    ep.secrets.insert(k.clone(), v.clone());
                } else if ep.is_local() || sp.settings.contains(&k.as_str()) {
                    ep.service_settings.insert(k.clone(), v.clone());
                } else {
                    return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{}] service", java_map(&Map::from_iter([(k.clone(), v.clone())])), ep.service)));
                }
            }
        }
        if let Some(ts) = o.get("task_settings").and_then(Value::as_object) {
            for (k, v) in ts {
                if !ep.is_local() && !sp.task_settings.contains(&k.as_str()) {
                    return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{}] service", java_map(&Map::from_iter([(k.clone(), v.clone())])), ep.service)));
                }
                ep.task_settings.insert(k.clone(), v.clone());
            }
        }
        let out = ep.to_json();
        drop(eps);
        self.persist()?;
        Ok(out)
    }

    /// `DELETE /_inference[/{task_type}]/{id}`.
    pub fn delete(&self, task: Option<&str>, id: &str, dry_run: bool, force: bool) -> Result<Value> {
        let task = task.map(Self::parse_task).transpose()?;
        let ep = self.eps.read().get(id).cloned().ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
        if let Some(t) = task {
            if t != ep.task_type {
                return Err(status_ex(400, format!("Requested task type [{}] does not match the inference endpoint's task type [{}]", t.name(), ep.task_type.name())));
            }
        }
        if ep.preconfigured {
            return Err(status_ex(400, format!("[{id}] is a reserved inference endpoint and cannot be deleted")));
        }
        let _ = force;
        if dry_run {
            return Ok(json!({"acknowledged": false, "pipelines": [], "indexes": [], "dry_run": true}));
        }
        self.eps.write().remove(id);
        self.persist()?;
        Ok(json!({"acknowledged": true, "pipelines": [], "indexes": []}))
    }

    // ------------------------------------------------------------ infer

    /// Raw inference of an endpoint (the `POST _inference` result object).
    pub fn infer(&self, task: Option<&str>, id: &str, body: &Value) -> Result<Value> {
        let ep = self.eps.read().get(id).cloned().ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
        if let Some(t) = task.map(Self::parse_task).transpose()? {
            if t != ep.task_type {
                return Err(status_ex(400, format!("Incompatible task_type, the requested type [{}] does not match the model type [{}]", t.name(), ep.task_type.name())));
            }
        }
        let o = body.as_object().cloned().unwrap_or_default();
        let input: Vec<String> = match o.get("input") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) if !a.is_empty() => a.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| EsError::parsing("[inference] input must be a string or an array of strings"))).collect::<Result<_>>()?,
            Some(Value::Array(_)) => return Err(EsError::action_validation(&["Field [input] cannot be an empty array"])),
            Some(Value::Null) | None => return Err(EsError::action_validation(&["Field [input] cannot be null"])),
            Some(other) => {
                let ty = match other {
                    Value::Number(_) => "VALUE_NUMBER",
                    Value::Bool(_) => "VALUE_BOOLEAN",
                    _ => "START_OBJECT",
                };
                return Err(EsError::new(400, "x_content_parse_exception", format!("[1:10] [cluster:internal/xpack/inference] input doesn't support values of type: {ty}")));
            }
        };
        let query = o.get("query").and_then(Value::as_str);
        if ep.task_type == Task::Rerank && query.is_none() {
            return Err(EsError::action_validation(&["Field [query] cannot be null for task type [rerank]"]));
        }
        let mut ts = ep.task_settings.clone();
        if let Some(t) = o.get("task_settings").and_then(Value::as_object) {
            for (k, v) in t {
                ts.insert(k.clone(), v.clone());
            }
        }
        if ep.is_local() {
            let kind = o.get("input_type").and_then(Value::as_str).unwrap_or("");
            return Ok(self.infer_local(&ep, &input, query, &ts, kind));
        }
        if mock::is_mock(&ep.service) {
            let over = o.get("task_settings").and_then(Value::as_object).cloned().unwrap_or_default();
            return mock::infer(&ep, &input, &over);
        }
        let timeout = o.get("timeout").and_then(Value::as_str).and_then(crate::ilm::parse_time_millis).map(|ms| Duration::from_millis(ms as u64)).unwrap_or(Duration::from_secs(30));
        let c = Call { service: &ep.service, task: ep.task_type, service_settings: &ep.service_settings, secrets: &ep.secrets, task_settings: ts, input: &input, query };
        providers::execute(&c, timeout).map_err(|(status, msg)| {
            let msg = if msg.starts_with("Error message: [") { provider_error_text(status, id, &msg) } else { msg };
            status_ex(if status == 502 { 500 } else { status }, msg)
        })
    }

    fn infer_local(&self, ep: &Endpoint, input: &[String], query: Option<&str>, ts: &Map<String, Value>, kind: &str) -> Value {
        match ep.task_type {
            Task::TextEmbedding => {
                let model_dir = self.dir.parent().unwrap_or(&self.dir).join("ml-models").join(ep.model_id().trim_end_matches("_linux-x86_64"));
                if let Some(rows) = real::try_embed(&model_dir, input, kind) {
                    return json!({"text_embedding": rows.into_iter().map(|r| json!({"embedding": r})).collect::<Vec<_>>()});
                }
                json!({"text_embedding": input.iter().map(|t| json!({"embedding": local::embed_dense(t, None)})).collect::<Vec<_>>()})
            }
            Task::SparseEmbedding => json!({"sparse_embedding": input.iter().map(|t| json!({"is_truncated": false, "embedding": local::embed_sparse(t)})).collect::<Vec<_>>()}),
            Task::Rerank => {
                let q = query.unwrap_or("");
                let ret = ts.get("return_documents").and_then(Value::as_bool).unwrap_or(false);
                let mut rows: Vec<(usize, f64)> = input.iter().enumerate().map(|(i, d)| (i, local::rerank_score(q, d))).collect();
                rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top = ts.get("top_n").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(rows.len());
                json!({"rerank": rows.into_iter().take(top).map(|(i, s)| {
                    let mut o = json!({"index": i, "relevance_score": s});
                    if ret { o["text"] = json!(input[i]); }
                    o
                }).collect::<Vec<_>>()})
            }
            _ => json!({}),
        }
    }

    /// Embed `texts` with an endpoint (indexing / query-time use).
    pub fn embed(&self, id: &str, texts: &[String]) -> Result<Embedded> {
        self.embed_typed(id, texts, "")
    }

    /// Like [`embed`](Self::embed) with an `input_type` (`ingest` / `search`).
    pub fn embed_typed(&self, id: &str, texts: &[String], kind: &str) -> Result<Embedded> {
        let res = self.infer(None, id, &json!({"input": texts, "input_type": kind}))?;
        if let Some(a) = res.get("text_embedding").or_else(|| res.get("embeddings")).and_then(Value::as_array) {
            return Ok(Embedded::Dense(a.iter().map(|e| e["embedding"].as_array().map(|v| v.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()).unwrap_or_default()).collect()));
        }
        if let Some(a) = res.get("sparse_embedding").and_then(Value::as_array) {
            return Ok(Embedded::Sparse(a.iter().map(|e| e["embedding"].as_object().cloned().unwrap_or_default()).collect()));
        }
        Err(status_ex(400, format!("Inference endpoint [{id}] does not produce embeddings")))
    }

    /// Embeddings of `text` from an `_ml` trained model (`text_embedding` /
    /// `text_expansion` models referenced by `model_id` / `inference_id`);
    /// `None` when no such model exists.
    pub fn embed_trained_model(&self, model_id: &str, text: &str) -> Result<Option<Embedded>> {
        let (res, field) = match models::infer_text_with_field(&self.dir, model_id, text) {
            Ok(r) => r,
            Err(e) if e.status == 404 => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(match res.get(&field) {
            Some(Value::Object(m)) => Some(Embedded::Sparse(vec![m.clone()])),
            Some(Value::Array(a)) => Some(Embedded::Dense(vec![a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()])),
            _ => None,
        })
    }

    /// Rerank `docs` for `query` with an endpoint: `(index, score)` best first.
    pub fn rerank(&self, id: &str, query: &str, docs: &[String]) -> Result<Vec<(usize, f64)>> {
        let ep = self.eps.read().get(id).cloned().ok_or_else(|| EsError::resource_not_found(format!("Inference endpoint not found [{id}]")))?;
        if ep.is_local() {
            // Local stand-in: score directly (no HTTP round trip).
            let q = query;
            let mut rows: Vec<(usize, f64)> = docs.iter().enumerate().map(|(i, d)| (i, local::rerank_score(q, d))).collect();
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(rows);
        }
        let res = self.infer(None, id, &json!({"input": docs, "query": query}))?;
        Ok(res["rerank"].as_array().into_iter().flatten().filter_map(|r| Some((r["index"].as_u64()? as usize, r["relevance_score"].as_f64()?))).collect())
    }
}

/// Model settings (`model_settings` of a `semantic_text` field) of an endpoint.
pub fn model_settings(ep: &Endpoint) -> Option<Value> {
    match ep.task_type {
        Task::TextEmbedding | Task::Embedding => {
            let dims = ep.service_settings.get("dimensions").cloned().unwrap_or_else(|| json!(local::E5_DIMS));
            let sim = ep.service_settings.get("similarity").cloned().unwrap_or_else(|| json!(if ep.is_local() || mock::is_mock(&ep.service) { "cosine" } else { "dot_product" }));
            let et = ep.service_settings.get("element_type").cloned().unwrap_or_else(|| json!("float"));
            Some(json!({"service": ep.service, "task_type": ep.task_type.name(), "dimensions": dims, "similarity": sim, "element_type": et}))
        }
        Task::SparseEmbedding => Some(json!({"service": ep.service, "task_type": "sparse_embedding"})),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc() -> (tempfile::TempDir, InferenceService) {
        let d = tempfile::tempdir().unwrap();
        let s = InferenceService::open(d.path());
        (d, s)
    }

    #[test]
    fn feature_reset_removes_persisted_endpoints_and_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("_state");
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(state.join("inference.json"), json!([{
            "inference_id":"saved-endpoint", "task_type":"text_embedding", "service":"openai",
            "service_settings":{"model_id":"saved-model"}, "task_settings":{}
        }]).to_string()).unwrap();
        std::fs::write(state.join("inference-secrets.json"), json!({"saved-endpoint":{"api_key":"test-secret"}}).to_string()).unwrap();
        let service = InferenceService::open(dir.path());
        assert!(service.get(None, Some("saved-endpoint")).is_ok());
        service.reset().unwrap();
        let reopened = InferenceService::open(dir.path());
        assert!(reopened.get(None, Some("saved-endpoint")).is_err());
        assert_eq!(reopened.get(None, Some("_all")).unwrap()["endpoints"].as_array().unwrap().len(), 3);
        let secrets: Value = serde_json::from_slice(&std::fs::read(state.join("inference-secrets.json")).unwrap()).unwrap();
        assert_eq!(secrets, json!({}));
    }

    #[test]
    fn cluster_state_round_trips_endpoints_and_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("_state");
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(state.join("inference.json"), json!([{
            "inference_id":"saved-endpoint", "task_type":"text_embedding", "service":"openai",
            "service_settings":{"model_id":"saved-model"}, "task_settings":{}
        }]).to_string()).unwrap();
        std::fs::write(state.join("inference-secrets.json"), json!({"saved-endpoint":{"api_key":"test-secret"}}).to_string()).unwrap();
        let master = InferenceService::open(dir.path());
        let published = master.cluster_state();
        // Another node: learns the endpoint (and its secret), persistently.
        let (other_dir, other) = svc();
        assert!(other.get_endpoint("saved-endpoint").is_none());
        other.restore_cluster_state(&published).unwrap();
        let ep = other.get_endpoint("saved-endpoint").unwrap();
        assert_eq!(ep.secrets.get("api_key"), Some(&json!("test-secret")));
        assert!(InferenceService::open(other_dir.path()).get_endpoint("saved-endpoint").is_some());
        // Deleting on the master removes it everywhere; defaults stay.
        other.restore_cluster_state(&json!({"endpoints": [], "secrets": {}})).unwrap();
        assert!(other.get_endpoint("saved-endpoint").is_none());
        assert_eq!(other.get(None, Some("_all")).unwrap()["endpoints"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn defaults_listed_and_error_wording() {
        let (_d, s) = svc();
        let all = s.get(None, Some("_all")).unwrap();
        assert_eq!(all["endpoints"].as_array().unwrap().len(), 3);
        assert_eq!(all["endpoints"][2]["task_settings"], json!({"return_documents": true}));
        assert_eq!(s.get(Some("badtype"), None).unwrap_err().reason, "Unknown task_type [badtype]");
        assert_eq!(s.get(None, Some("nope")).unwrap_err().reason, "Inference endpoint not found [nope]");
        assert_eq!(s.delete(Some("text_embedding"), ".elser-2-elasticsearch", false, false).unwrap_err().reason, "Requested task type [text_embedding] does not match the inference endpoint's task type [sparse_embedding]");
    }

    #[test]
    fn put_validation_messages() {
        let (_d, s) = svc();
        let e = s.put("text_embedding", ".bad", &json!({})).unwrap_err();
        assert!(e.reason.starts_with("Validation Failed: 1: Invalid inference_id; '.bad' can contain"), "{}", e.reason);
        assert_eq!(s.put("text_embedding", "e1", &json!({"service": "nosuch"})).unwrap_err().reason, "Unknown service [nosuch]");
        assert_eq!(s.put("text_embedding", "e1", &json!({"service_settings": {}})).unwrap_err().reason, "Inference endpoint configuration is missing the [service] setting");
        assert_eq!(s.put("sparse_embedding", "e1", &json!({"service": "openai", "service_settings": {"api_key": "k"}})).unwrap_err().reason, "The [openai] service does not support task type [sparse_embedding]");
        // ES 8.19: required service settings are checked before leftovers.
        assert_eq!(s.put("text_embedding", "e1", &json!({"service": "openai", "service_settings": {"api_key": "k", "bogus": 1}})).unwrap_err().reason, "Validation Failed: 1: [service_settings] does not contain the required setting [model_id];");
        assert_eq!(s.put("text_embedding", "e1", &json!({"service": "openai", "service_settings": {"api_key": "k", "model_id": "m", "bogus": 1}})).unwrap_err().reason, "Configuration contains settings [{bogus=1}] unknown to the [openai] service");
        let e = s.put("text_embedding", "e1", &json!({"service": "openai", "service_settings": {"model_id": "m"}})).unwrap_err();
        assert_eq!(e.reason, "Validation Failed: 1: [secret_settings] does not contain the required setting [api_key];");
    }

    #[test]
    fn local_endpoint_lifecycle_and_infer() {
        let (d, s) = svc();
        let r = s.put("text_embedding", "my-e5", &json!({"service": "elasticsearch", "service_settings": {"model_id": ".multilingual-e5-small", "num_allocations": 1, "num_threads": 1}})).unwrap();
        assert_eq!(r["inference_id"], "my-e5");
        let out = s.infer(Some("text_embedding"), "my-e5", &json!({"input": ["hello world", "bye"]})).unwrap();
        assert_eq!(out["text_embedding"].as_array().unwrap().len(), 2);
        assert_eq!(out["text_embedding"][0]["embedding"].as_array().unwrap().len(), 384);
        // persisted
        let s2 = InferenceService::open(d.path());
        assert!(s2.get_endpoint("my-e5").is_some());
        assert_eq!(s.infer(Some("completion"), "my-e5", &json!({"input": "x"})).unwrap_err().reason, "Incompatible task_type, the requested type [completion] does not match the model type [text_embedding]");
        assert_eq!(s.infer(None, "my-e5", &json!({})).unwrap_err().reason, "Validation Failed: 1: Field [input] cannot be null;");
        assert_eq!(s.delete(None, "my-e5", false, false).unwrap(), json!({"acknowledged": true, "pipelines": [], "indexes": []}));
    }

    #[test]
    fn rerank_default_endpoint_orders_by_overlap() {
        let (_d, s) = svc();
        let out = s.infer(None, ".rerank-v1-elasticsearch", &json!({"query": "fox", "input": ["database revenue", "the quick fox"]})).unwrap();
        assert_eq!(out["rerank"][0]["index"], 1);
        assert_eq!(out["rerank"][0]["text"], "the quick fox");
    }
}
