//! The `_inference` API: endpoint registry (persisted under
//! `<path.data>/_state/inference*.json`, secrets kept in a separate file like
//! ES's `.secrets-inference` index), request validation with ES wording, and
//! inference dispatch to local models ([`local`]) or hosted providers
//! ([`providers`]).

pub mod backend;
pub mod local;
pub mod mock;
pub mod model_store;
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
pub mod tree_infer;

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
        let mut ss = if mock::is_mock(&self.service) { mock::filtered_settings(self) } else { self.service_settings.clone() };
        // Internal: whether `dimensions` is the user's (ES does not render it).
        ss.shift_remove("dimensions_set_by_user");
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
    backend: Arc<backend::NodeState>,
}

/// Apply the node settings the inference service reads
/// (`xpack.inference.builtin_fallback`) to the service of `data`.
pub fn configure(data: &Path, settings: &Map<String, Value>) {
    service_for(data).configure(settings);
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

    /// Apply the node settings this service reads ([`backend::FALLBACK_SETTING`]).
    pub fn configure(&self, settings: &Map<String, Value>) {
        backend::configure_node(&self.backend, settings);
    }

    /// Enable or disable the built-in model stand-ins on this node.
    pub fn set_builtin_fallback(&self, enabled: bool) {
        backend::set_fallback(&self.dir, enabled);
    }

    /// The implementation serving a local endpoint's built-in model, or why
    /// none can (`None` for other endpoints).
    pub fn implementation(&self, ep: &Endpoint) -> Option<std::result::Result<backend::Implementation, String>> {
        if !ep.is_local() {
            return None;
        }
        backend::Builtin::of(ep.model_id()).map(|b| backend::resolve(&self.dir, b))
    }

    /// The response warning for a call to `id` when a stand-in serves it.
    pub fn fallback_warning(&self, id: &str) -> Option<String> {
        let ep = self.get_endpoint(id)?;
        match self.implementation(&ep)? {
            Ok(imp) if imp.is_fallback() => Some(backend::warning(ep.model_id(), imp)),
            _ => None,
        }
    }

    /// `GET` rendering of an endpoint: with the stand-ins enabled, built-in
    /// endpoints name the implementation that serves them.
    fn render(&self, e: &Endpoint) -> Value {
        let mut v = e.to_json();
        if backend::fallback_enabled(&self.dir) {
            if let Some(Ok(imp)) = self.implementation(e) {
                v["implementation"] = json!(imp.name());
            }
        }
        v
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
        let backend = backend::node(&dir);
        InferenceService { dir, eps: RwLock::new(eps), backend }
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
        backend::clear_pins(&self.dir);
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
            None | Some("_all") | Some("*") => Ok(json!({"endpoints": eps.values().filter(|e| task.is_none_or(|t| e.task_type == t)).map(|e| self.render(e)).collect::<Vec<_>>()})),
            Some(id) if id.contains('*') => {
                let re = id.replace('*', "");
                let hits: Vec<Value> = eps.values().filter(|e| task.is_none_or(|t| e.task_type == t) && (e.inference_id.starts_with(&re) || e.inference_id.contains(&re))).map(|e| self.render(e)).collect();
                Ok(json!({"endpoints": hits}))
            }
            Some(id) => match eps.get(id) {
                Some(e) if task.is_none_or(|t| e.task_type == t) => Ok(json!({"endpoints": [self.render(e)]})),
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
        // ES builds endpoint chunking with the same `ChunkingSettingsBuilder`
        // validators as a field's (finding 126: word overlap at most half the
        // chunk size, size bounds, known strategy and keys).
        if let Some(c) = chunking.as_ref() {
            semantic_map::chunking_settings(c)?;
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
            // A width the user configured is requested from the provider
            // and must be what it returns (finding 356); a learned width
            // is not requested later.
            let user_dims = ep.service_settings.get("dimensions").and_then(Value::as_u64);
            if task.is_dense() {
                ep.service_settings.insert("dimensions_set_by_user".into(), json!(user_dims.is_some()));
            }
            // ES validates with an internal ingest-type call.
            let probe_type = task.is_dense().then_some("internal_ingest");
            let c = Call { service: &ep.service, task: ep.task_type, service_settings: &ep.service_settings, secrets: &ep.secrets, task_settings: ep.task_settings.clone(), input: &probe, query: Some("query"), input_type: probe_type };
            match providers::execute(&c, Duration::from_secs(30)) {
                Ok(res) => {
                    let dims = res["text_embedding"][0]["embedding"].as_array().map(Vec::len);
                    // Services that request the configured width must
                    // get it; for the others the provider's width is the
                    // endpoint's (ES `updateModelWithEmbeddingDetails`).
                    let honours_dims = matches!(ep.service.as_str(), "openai" | "azureopenai" | "voyageai" | "jinaai");
                    if !honours_dims && dims.is_some() {
                        ep.service_settings.shift_remove("dimensions");
                        ep.service_settings.insert("dimensions_set_by_user".into(), json!(false));
                    }
                    if let (true, Some(want), Some(got)) = (honours_dims, user_dims, dims) {
                        if want as usize != got {
                            return Err(status_ex(
                                400,
                                format!("The retrieved embeddings size [{got}] does not match the size specified in the settings [{want}]. Please recreate the [{id}] configuration with the correct dimensions"),
                            ));
                        }
                    }
                    let set_by_user = ep.service_settings.shift_remove("dimensions_set_by_user");
                    ep.service_settings = canonical_settings(&ep.service, task, std::mem::take(&mut ep.service_settings), dims);
                    if let Some(v) = set_by_user {
                        ep.service_settings.insert("dimensions_set_by_user".into(), v);
                    }
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
        // ES's v9 update contract (`TransportUpdateInferenceModelAction`):
        // each service's update parser takes the secrets, the settings it
        // may change (rate limit, max input tokens, organization) and, for
        // the in-cluster services, the allocation counts; any key left over
        // fails the request (`validateConsumedUpdateSettings`). The model,
        // URL, dimensions and the rest are the endpoint's identity: changing
        // them would mix embeddings of different models under one inference
        // id. Everything is validated on a copy and installed only when the
        // whole update is valid.
        let mock = mock::is_mock(&ep.service);
        let sp = if mock { None } else { Some(spec(&ep.service).ok_or_else(|| status_ex(400, format!("Unknown service [{}]", ep.service)))?) };
        let secret = |k: &str| if mock { k == "api_key" } else { sp.as_ref().is_some_and(|sp| sp.secrets.contains(&k)) };
        let updatable = |k: &str| {
            if ep.is_local() {
                matches!(k, "num_allocations" | "adaptive_allocations")
            } else {
                !mock && matches!(k, "rate_limit" | "max_input_tokens" | "organization_id") && sp.as_ref().is_some_and(|sp| sp.settings.contains(&k))
            }
        };
        let mut next = ep.clone();
        let mut leftover = Map::new();
        if let Some(ss) = o.get("service_settings").and_then(Value::as_object) {
            for (k, v) in ss {
                if secret(k) {
                    next.secrets.insert(k.clone(), v.clone());
                } else if updatable(k) {
                    next.service_settings.insert(k.clone(), v.clone());
                } else {
                    leftover.insert(k.clone(), v.clone());
                }
            }
        }
        if !leftover.is_empty() {
            return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{}] service", java_map(&leftover), ep.service)));
        }
        if let Some(ts) = o.get("task_settings").and_then(Value::as_object) {
            for (k, v) in ts {
                if !mock && !ep.is_local() && !sp.as_ref().is_some_and(|sp| sp.task_settings.contains(&k.as_str())) {
                    return Err(status_ex(400, format!("Configuration contains settings [{}] unknown to the [{}] service", java_map(&Map::from_iter([(k.clone(), v.clone())])), ep.service)));
                }
                next.task_settings.insert(k.clone(), v.clone());
            }
        }
        *ep = next;
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
            return self.infer_local(&ep, &input, query, &ts, kind);
        }
        if mock::is_mock(&ep.service) {
            let over = o.get("task_settings").and_then(Value::as_object).cloned().unwrap_or_default();
            return mock::infer(&ep, &input, &over);
        }
        let timeout = o.get("timeout").and_then(Value::as_str).and_then(crate::ilm::parse_time_millis).map(|ms| Duration::from_millis(ms as u64)).unwrap_or(Duration::from_secs(30));
        // The root `input_type` takes precedence over the endpoint's
        // `task_settings.input_type` (finding 344).
        let input_type = o.get("input_type").and_then(Value::as_str).map(str::to_ascii_lowercase).filter(|t| !t.is_empty() && t != "unspecified");
        if let (Some(t), "voyageai") = (input_type.as_deref(), ep.service.as_str()) {
            if !matches!(t, "ingest" | "search" | "internal_ingest" | "internal_search") {
                return Err(EsError::action_validation(&[&format!("Input type [{t}] is not supported for [Voyage AI]")]));
            }
        }
        let c = Call { service: &ep.service, task: ep.task_type, service_settings: &ep.service_settings, secrets: &ep.secrets, task_settings: ts, input: &input, query, input_type: input_type.as_deref() };
        let res = providers::execute(&c, timeout).map_err(|(status, msg)| {
            let msg = if msg.starts_with("Error message: [") { provider_error_text(status, id, &msg) } else { msg };
            status_ex(if status == 502 { 500 } else { status }, msg)
        })?;
        // Vectors must have the endpoint's declared width (finding 356): a
        // `bit` endpoint's `dimensions` counts bits, its embeddings are
        // packed bytes (ES `DenseEmbeddingBitResults`: dims / 8).
        if let (true, Some(want)) = (ep.task_type.is_dense(), embedding_len(&ep.service_settings)) {
            if let Some(got) = res["text_embedding"].as_array().and_then(|a| a.iter().filter_map(|e| e["embedding"].as_array().map(Vec::len)).find(|n| *n as u64 != want)) {
                return Err(status_ex(500, format!("Inference endpoint [{id}] returned embeddings of size [{got}], but it is configured for [{want}] dimensions")));
            }
        }
        Ok(res)
    }

    /// Inference on the `elasticsearch` service. Built-in models run on
    /// their real implementation, or on a stand-in only when the node
    /// enabled them; other model ids run on their trained-model deployment.
    /// Nothing is substituted silently: a model that cannot run fails.
    fn infer_local(&self, ep: &Endpoint, input: &[String], query: Option<&str>, ts: &Map<String, Value>, kind: &str) -> Result<Value> {
        let model_id = ep.model_id();
        let Some(builtin) = backend::Builtin::of(model_id) else {
            return self.infer_deployed_model(ep, input, query);
        };
        let imp = backend::resolve(&self.dir, builtin).map_err(|why| backend::not_deployed(model_id, &why))?;
        if imp.is_fallback() {
            crate::warnings::add(backend::warning(model_id, imp));
        }
        Ok(match ep.task_type {
            Task::TextEmbedding | Task::Embedding if builtin == backend::Builtin::E5 => {
                backend::check_pin(&self.dir, &ep.inference_id, imp)?;
                let rows = match imp {
                    backend::Implementation::Candle => real::embed(&backend::model_dir(&self.dir, builtin), input, kind)
                        .map_err(|e| status_ex(500, format!("Failed to run inference with model [{model_id}]: {e}")))?,
                    _ => input.iter().map(|t| local::embed_dense(t, None)).collect(),
                };
                json!({"text_embedding": rows.into_iter().map(|r| json!({"embedding": r})).collect::<Vec<_>>()})
            }
            Task::SparseEmbedding if builtin == backend::Builtin::Elser => {
                backend::check_pin(&self.dir, &ep.inference_id, imp)?;
                json!({"sparse_embedding": input.iter().map(|t| json!({"is_truncated": false, "embedding": local::embed_sparse(t)})).collect::<Vec<_>>()})
            }
            Task::Rerank if builtin == backend::Builtin::Rerank => {
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
            t => return Err(status_ex(400, format!("Model [{model_id}] does not support task type [{}]", t.name()))),
        })
    }

    /// Inference of a non-built-in `model_id` through its trained-model
    /// deployment (an uploaded PyTorch model), as ES's internal service does.
    fn infer_deployed_model(&self, ep: &Endpoint, input: &[String], query: Option<&str>) -> Result<Value> {
        let model_id = ep.model_id();
        let _ = query;
        match ep.task_type {
            Task::TextEmbedding | Task::Embedding | Task::SparseEmbedding => {
                let mut rows = Vec::with_capacity(input.len());
                for t in input {
                    let (res, field) = models::infer_text_with_field(&self.dir, model_id, t)?;
                    rows.push(res.get(&field).cloned().unwrap_or(Value::Null));
                }
                Ok(if ep.task_type == Task::SparseEmbedding {
                    json!({"sparse_embedding": rows.into_iter().map(|r| json!({"is_truncated": false, "embedding": r})).collect::<Vec<_>>()})
                } else {
                    json!({"text_embedding": rows.into_iter().map(|r| json!({"embedding": r})).collect::<Vec<_>>()})
                })
            }
            t => Err(status_ex(400, format!("Inference with model [{model_id}] and task type [{}] is not supported", t.name()))),
        }
    }

    /// Embed `texts` with an endpoint (indexing / query-time use).
    pub fn embed(&self, id: &str, texts: &[String]) -> Result<Embedded> {
        self.embed_typed(id, texts, "")
    }

    /// Like [`embed`](Self::embed) with an `input_type` (`ingest` / `search`).
    pub fn embed_typed(&self, id: &str, texts: &[String], kind: &str) -> Result<Embedded> {
        let res = self.infer(None, id, &json!({"input": texts, "input_type": kind}))?;
        // One embedding per input, of the endpoint's shape: anything else is
        // a malformed provider answer and fails the request (finding 127).
        let malformed = |why: String| status_ex(500, why);
        if let Some(a) = res.get("text_embedding").or_else(|| res.get("embeddings")).and_then(Value::as_array) {
            if a.len() != texts.len() {
                return Err(malformed(providers::count_mismatch(a.len(), texts.len())));
            }
            let dims = self.get_endpoint(id).and_then(|ep| embedding_len(&ep.service_settings));
            let mut rows = Vec::with_capacity(a.len());
            for e in a {
                let v = e["embedding"].as_array().ok_or_else(|| malformed(format!("Inference endpoint [{id}] returned an embedding that is not an array")))?;
                let v: Vec<f32> = v.iter().map(|x| x.as_f64().map(|f| f as f32)).collect::<Option<_>>().ok_or_else(|| malformed(format!("Inference endpoint [{id}] returned a non-numeric embedding value")))?;
                if v.is_empty() || dims.is_some_and(|d| d as usize != v.len()) || rows.first().is_some_and(|f: &Vec<f32>| f.len() != v.len()) {
                    return Err(malformed(format!("Inference endpoint [{id}] returned an embedding of [{}] dimensions{}", v.len(), dims.map(|d| format!(", expected [{d}]")).unwrap_or_default())));
                }
                rows.push(v);
            }
            return Ok(Embedded::Dense(rows));
        }
        if let Some(a) = res.get("sparse_embedding").and_then(Value::as_array) {
            if a.len() != texts.len() {
                return Err(malformed(providers::count_mismatch(a.len(), texts.len())));
            }
            return a.iter().map(|e| e["embedding"].as_object().cloned().ok_or_else(|| malformed(format!("Inference endpoint [{id}] returned a sparse embedding that is not an object")))).collect::<Result<Vec<_>>>().map(Embedded::Sparse);
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

/// The length of each embedding a dense endpoint returns: its
/// `dimensions`, or for a `bit` endpoint (whose `dimensions` counts bits)
/// the packed bytes, `dimensions / 8` (ES `DenseEmbeddingBitResults`).
fn embedding_len(service_settings: &Map<String, Value>) -> Option<u64> {
    let dims = service_settings.get("dimensions").and_then(Value::as_u64)?;
    let bits = service_settings.get("element_type").and_then(Value::as_str).is_some_and(|t| t.eq_ignore_ascii_case("bit"));
    Some(if bits { dims.div_ceil(8) } else { dims })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc() -> (tempfile::TempDir, InferenceService) {
        let d = tempfile::tempdir().unwrap();
        let s = InferenceService::open(d.path());
        (d, s)
    }

    /// Finding 117: `_update` follows ES's v9 contract. The model, URL and
    /// other identity settings cannot change (ES's unknown-settings error),
    /// nothing of a rejected update is applied, and secret rotation, task
    /// settings and rate limits still update.
    #[test]
    fn update_refuses_identity_changes_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("_state");
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(state.join("inference.json"), json!([{
            "inference_id":"emb", "task_type":"text_embedding", "service":"openai",
            "service_settings":{"model_id":"model-a", "dimensions": 4, "similarity": "cosine"}, "task_settings":{"user": "u1"}
        }]).to_string()).unwrap();
        std::fs::write(state.join("inference-secrets.json"), json!({"emb":{"api_key":"k1"}}).to_string()).unwrap();
        let s = InferenceService::open(dir.path());
        let before = s.get(None, Some("emb")).unwrap();
        for change in [json!({"model_id": "model-b"}), json!({"url": "http://elsewhere"}), json!({"dimensions": 4, "model_id": "model-b"}), json!({"model_id": "model-a"})] {
            // A rejected update applies none of its parts.
            let body = json!({"service_settings": {"api_key": "k-new"}, "task_settings": {"user": "u2"}});
            let mut body = body;
            for (k, v) in change.as_object().unwrap() {
                body["service_settings"][k] = v.clone();
            }
            let e = s.update(Some("text_embedding"), "emb", &body).unwrap_err();
            assert_eq!(e.status, 400, "{change}");
            assert!(e.reason.starts_with("Configuration contains settings [{") && e.reason.ends_with("] unknown to the [openai] service"), "{change}: {}", e.reason);
            assert_eq!(s.get(None, Some("emb")).unwrap(), before, "{change}");
            assert_eq!(s.eps.read()["emb"].secrets.get("api_key"), Some(&json!("k1")), "{change}");
        }
        // Secret rotation, task settings and the rate limit update.
        s.update(Some("text_embedding"), "emb", &json!({"service_settings": {"api_key": "k2", "rate_limit": {"requests_per_minute": 10}}, "task_settings": {"user": "u3"}})).unwrap();
        let ep = s.eps.read()["emb"].clone();
        assert_eq!(ep.secrets.get("api_key"), Some(&json!("k2")));
        assert_eq!(ep.task_settings.get("user"), Some(&json!("u3")));
        assert_eq!(ep.model_id(), "model-a");
        // ... and persist; the model stays.
        let reopened = InferenceService::open(dir.path());
        assert_eq!(reopened.eps.read()["emb"].model_id(), "model-a");
        assert_eq!(reopened.eps.read()["emb"].secrets.get("api_key"), Some(&json!("k2")));
    }

    /// A fake Voyage embeddings server: answers with the width it was
    /// asked for (`output_dimension`, else the model default 1024) and
    /// records every request body.
    fn fake_voyage() -> (String, Arc<Mutex<Vec<Value>>>) {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/embeddings", l.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let rec = seen.clone();
        std::thread::spawn(move || {
            for conn in l.incoming() {
                let Ok(mut c) = conn else { continue };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head_end, len) = loop {
                    let n = c.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break (None, 0);
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..p]).to_ascii_lowercase();
                        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                        break (Some(p + 4), len);
                    }
                };
                let Some(start) = head_end else { continue };
                while buf.len() < start + len {
                    let n = c.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                let body: Value = serde_json::from_slice(&buf[start..(start + len).min(buf.len())]).unwrap_or(Value::Null);
                let n = body.get("output_dimension").and_then(Value::as_u64).unwrap_or(1024) as usize;
                let k = body["input"].as_array().map_or(1, Vec::len);
                rec.lock().unwrap().push(body);
                let resp = json!({"data": (0..k).map(|i| json!({"index": i, "embedding": vec![0.5; n]})).collect::<Vec<_>>()}).to_string();
                let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", resp.len(), resp);
            }
        });
        (url, seen)
    }

    /// A strict OpenAI embeddings server: refuses the `dimensions` option
    /// (as `text-embedding-ada-002` does) and answers 1536-wide vectors.
    fn fake_strict_openai() -> (String, Arc<Mutex<Vec<Value>>>) {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/embeddings", l.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let rec = seen.clone();
        std::thread::spawn(move || {
            for conn in l.incoming() {
                let Ok(mut c) = conn else { continue };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head_end, len) = loop {
                    let n = c.read(&mut chunk).unwrap_or(0);
                    if n == 0 { break (None, 0); }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..p]).to_ascii_lowercase();
                        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                        break (Some(p + 4), len);
                    }
                };
                let Some(start) = head_end else { continue };
                while buf.len() < start + len {
                    let n = c.read(&mut chunk).unwrap_or(0);
                    if n == 0 { break; }
                    buf.extend_from_slice(&chunk[..n]);
                }
                let body: Value = serde_json::from_slice(&buf[start..(start + len).min(buf.len())]).unwrap_or(Value::Null);
                let k = body["input"].as_array().map_or(1, Vec::len);
                let refused = body.get("dimensions").is_some();
                rec.lock().unwrap().push(body);
                let (status, resp) = if refused {
                    ("400 Bad Request", json!({"error": {"message": "This model does not support specifying dimensions.", "type": "invalid_request_error"}}).to_string())
                } else {
                    ("200 OK", json!({"data": (0..k).map(|i| json!({"index": i, "embedding": vec![0.01; 1536]})).collect::<Vec<_>>()}).to_string())
                };
                let _ = write!(c, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", resp.len(), resp);
            }
        });
        (url, seen)
    }

    /// Finding 547: an OpenAI endpoint created without `dimensions` learns
    /// its width from the probe but never sends it later (ada-002 refuses
    /// the option), also when its persisted settings predate the flag.
    #[test]
    fn openai_learned_width_is_never_requested() {
        let (_d, s) = svc();
        let (url, seen) = fake_strict_openai();
        let out = s.put("text_embedding", "ada", &json!({"service": "openai", "service_settings": {"model_id": "text-embedding-ada-002", "url": url, "api_key": "k"}})).unwrap();
        assert_eq!(out["service_settings"]["dimensions"], json!(1536), "{out}");
        let res = s.infer(None, "ada", &json!({"input": ["q"]})).unwrap();
        assert_eq!(res["text_embedding"][0]["embedding"].as_array().map(Vec::len), Some(1536));
        assert!(seen.lock().unwrap().iter().all(|b| b.get("dimensions").is_none()));
        // An endpoint stored before the provenance flag existed.
        s.eps.write().get_mut("ada").unwrap().service_settings.shift_remove("dimensions_set_by_user");
        assert!(s.infer(None, "ada", &json!({"input": ["q"]})).is_ok());
        assert!(seen.lock().unwrap().iter().all(|b| b.get("dimensions").is_none()));
        // A width the user set is still requested.
        let out = s.put("text_embedding", "small", &json!({"service": "openai", "service_settings": {"model_id": "text-embedding-3-small", "dimensions": 1536, "url": url, "api_key": "k"}}));
        assert!(out.is_err(), "the strict server refuses the explicit width");
        assert!(seen.lock().unwrap().last().unwrap().get("dimensions").is_some());
    }

    /// Findings 344 and 356 end to end on a Voyage endpoint: the
    /// configured width is requested (and is what the endpoint reports),
    /// the request's root input type wins over the saved one and is
    /// translated, and unsupported input types are refused.
    #[test]
    fn voyage_dimensions_and_input_types_reach_the_provider() {
        let (_d, s) = svc();
        let (url, seen) = fake_voyage();
        let out = s
            .put("text_embedding", "v512", &json!({"service": "voyageai", "service_settings": {"model_id": "voyage-3-large", "dimensions": 512, "url": url, "api_key": "k"}, "task_settings": {"input_type": "ingest"}}))
            .unwrap();
        assert_eq!(out["service_settings"]["dimensions"], json!(512), "{out}");
        assert!(out["service_settings"].get("dimensions_set_by_user").is_none(), "{out}");
        let probe = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!((&probe["output_dimension"], &probe["input_type"]), (&json!(512), &json!("document")), "{probe}");
        let res = s.infer(None, "v512", &json!({"input": ["q"], "input_type": "search"})).unwrap();
        assert_eq!(res["text_embedding"][0]["embedding"].as_array().unwrap().len(), 512);
        let body = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!((&body["output_dimension"], &body["input_type"]), (&json!(512), &json!("query")), "{body}");
        s.infer(None, "v512", &json!({"input": ["d"]})).unwrap();
        assert_eq!(seen.lock().unwrap().last().unwrap()["input_type"], json!("document"));
        let e = s.infer(None, "v512", &json!({"input": ["q"], "input_type": "classification"})).unwrap_err();
        assert!(e.reason.contains("Input type [classification] is not supported for [Voyage AI]"), "{}", e.reason);
        // Semantic search asks for query embeddings of the same endpoint.
        assert!(matches!(s.embed_typed("v512", &["q".to_string()], "search").unwrap(), Embedded::Dense(ref v) if v[0].len() == 512));
        assert_eq!(seen.lock().unwrap().last().unwrap()["input_type"], json!("query"));
        // No configured width: none is requested, the default is learned
        // and later calls do not ask for it.
        let out = s.put("text_embedding", "vdef", &json!({"service": "voyageai", "service_settings": {"model_id": "voyage-3-large", "url": url, "api_key": "k"}})).unwrap();
        assert_eq!(out["service_settings"]["dimensions"], json!(1024), "{out}");
        s.infer(None, "vdef", &json!({"input": ["x"]})).unwrap();
        assert!(seen.lock().unwrap().last().unwrap().get("output_dimension").is_none());
    }

    /// A `bit` endpoint's `dimensions` counts bits: its packed embeddings
    /// (dims / 8 bytes) pass the width check, a float endpoint's must
    /// match exactly.
    #[test]
    fn bit_endpoint_width_counts_bits() {
        mock::enable();
        let (_d, s) = svc();
        s.put("text_embedding", "bits", &json!({"service": "text_embedding_test_service", "service_settings": {"model": "m", "dimensions": 16, "api_key": "k", "similarity": "L2_NORM", "element_type": "bit"}, "task_settings": {}})).unwrap();
        let res = s.infer(None, "bits", &json!({"input": ["inference test", "another inference test"]})).unwrap();
        assert_eq!(res["text_embedding"][0]["embedding"].as_array().map(Vec::len), Some(2), "{res}");
        // semantic_text's ingest and query embeddings take the same width.
        assert!(matches!(s.embed_typed("bits", &["inference test".to_string()], "ingest").unwrap(), Embedded::Dense(ref v) if v[0].len() == 2));
        s.put("text_embedding", "floats", &json!({"service": "text_embedding_test_service", "service_settings": {"model": "m", "dimensions": 16, "api_key": "k", "similarity": "cosine", "element_type": "float"}, "task_settings": {}})).unwrap();
        let res = s.infer(None, "floats", &json!({"input": ["x"]})).unwrap();
        assert_eq!(res["text_embedding"][0]["embedding"].as_array().map(Vec::len), Some(16), "{res}");
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
        s.set_builtin_fallback(true);
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
        s.set_builtin_fallback(true);
        let out = s.infer(None, ".rerank-v1-elasticsearch", &json!({"query": "fox", "input": ["database revenue", "the quick fox"]})).unwrap();
        assert_eq!(out["rerank"][0]["index"], 1);
        assert_eq!(out["rerank"][0]["text"], "the quick fox");
    }

    /// Without the opt-in, built-in models that cannot run fail like an
    /// undeployed model in ES; nothing is substituted.
    #[test]
    fn builtin_models_fail_clearly_without_opt_in() {
        let (_d, s) = svc();
        let e = s.infer(None, ".elser-2-elasticsearch", &json!({"input": "hello"})).unwrap_err();
        assert_eq!((e.status, e.kind.as_str()), (409, "status_exception"));
        assert_eq!(e.reason, "Trained model deployment [.elser_model_2_linux-x86_64] is not allocated to any nodes");
        let cause = e.caused_by.as_ref().unwrap().reason.clone();
        assert!(cause.contains("no ELSER v2 implementation") && cause.contains("xpack.inference.builtin_fallback"), "{cause}");
        // E5: no model files in this data path (and/or no candle-models feature).
        let e = s.infer(None, ".multilingual-e5-small-elasticsearch", &json!({"input": "hello"})).unwrap_err();
        assert_eq!(e.reason, "Trained model deployment [.multilingual-e5-small_linux-x86_64] is not allocated to any nodes");
        let cause = e.caused_by.as_ref().unwrap().reason.clone();
        assert!(cause.contains("candle-models") || cause.contains("not found in"), "{cause}");
        assert!(s.infer(None, ".rerank-v1-elasticsearch", &json!({"query": "q", "input": ["a"]})).is_err());
        assert!(s.embed(".elser-2-elasticsearch", &["x".to_string()]).is_err());
        // GET is unchanged (ES's rendering) and names no implementation.
        let all = s.get(None, None).unwrap();
        assert!(all["endpoints"].as_array().unwrap().iter().all(|e| e.get("implementation").is_none()));
        assert!(s.fallback_warning(".elser-2-elasticsearch").is_none());
    }

    #[test]
    fn opt_in_stand_ins_report_their_implementation() {
        let (_d, s) = svc();
        let mut settings = Map::new();
        settings.insert(backend::FALLBACK_SETTING.into(), json!("lexical"));
        s.configure(&settings);
        let out = s.infer(None, ".elser-2-elasticsearch", &json!({"input": "running dogs"})).unwrap();
        assert!(out["sparse_embedding"][0]["embedding"]["dogs"].is_number());
        let got = s.get(None, Some(".elser-2-elasticsearch")).unwrap();
        assert_eq!(got["endpoints"][0]["implementation"], "lexical-fallback");
        assert!(s.fallback_warning(".elser-2-elasticsearch").unwrap().contains("[lexical-fallback] stand-in"));
        if !real::ACTIVE {
            let got = s.get(None, Some(".multilingual-e5-small-elasticsearch")).unwrap();
            assert_eq!(got["endpoints"][0]["implementation"], "hashing-fallback");
        }
        // Turning the setting off again makes the model unavailable.
        s.configure(&Map::new());
        assert_eq!(s.infer(None, ".elser-2-elasticsearch", &json!({"input": "x"})).unwrap_err().status, 409);
    }

    /// An endpoint that embedded with one implementation refuses another.
    #[test]
    fn endpoint_refuses_implementation_change() {
        let (_d, s) = svc();
        s.set_builtin_fallback(true);
        s.put("text_embedding", "my-e5", &json!({"service": "elasticsearch", "service_settings": {"model_id": ".multilingual-e5-small", "num_allocations": 1, "num_threads": 1}})).unwrap();
        s.infer(None, "my-e5", &json!({"input": "hello"})).unwrap();
        let pinned = backend::pin_of(&s.dir, "my-e5").unwrap();
        let other = if pinned == "candle" { backend::Implementation::HashingFallback } else { backend::Implementation::Candle };
        let e = backend::check_pin(&s.dir, "my-e5", other).unwrap_err();
        assert!(e.reason.contains(&format!("[{pinned}] but would now use [{}]", other.name())), "{}", e.reason);
        // Simulate the real model becoming the implementation: the endpoint refuses it.
        if pinned != "candle" {
            backend::clear_pin(&s.dir, "my-e5");
            backend::check_pin(&s.dir, "my-e5", backend::Implementation::Candle).unwrap();
            let e = s.infer(None, "my-e5", &json!({"input": "hello"})).unwrap_err();
            assert_eq!(e.status, 409);
            assert!(e.reason.contains("produced its existing embeddings with implementation [candle] but would now use [hashing-fallback]"), "{}", e.reason);
        }
    }
}
