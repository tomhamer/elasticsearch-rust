//! Bulk-by-scroll operations: `_reindex`, `_update_by_query`,
//! `_delete_by_query`.
//!
//! All three scan a point-in-time snapshot of the source indices in `_doc`
//! order (batches of `size` / `scroll_size`, default 1000), apply one write
//! batch per scroll batch (one translog fsync per batch) and report ES's
//! BulkByScroll response (`took, timed_out, total, updated, created,
//! deleted, batches, version_conflicts, noops, retries, throttled_millis,
//! requests_per_second, throttled_until_millis, failures`). Writes carry
//! `if_seq_no` from the snapshot (update/delete by query), so concurrent
//! modifications surface as version conflicts. `slices` runs one scan per slice
//! (sliced by `_id` hash, one after another) and reports per-slice statuses;
//! `requests_per_second` throttles between batches and can be changed while
//! running (`_rethrottle`, see [`Task::set_rps`]). `source.remote` reads the
//! source through [`crate::remote`].

use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::node::{Node, ResolveOpts};
use crate::search::{self, parse_source_filter, ExecOpts, SearchRequest, SortSpec, SourceFilter};
use crate::shard::{VersionType, WriteCond};
use crate::remote::RemoteSpec;
use crate::tasks::Task;
use rustic_core::query::TrackTotalHits;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// How a node acknowledges BulkByScroll write batches. ES's
/// `BulkByScrollTask` sends each batch through the ordinary bulk action,
/// which answers only once the in-sync replicas have the writes
/// (`wait_for_active_shards`, `ReplicationOperation`); a cluster installs
/// one that waits for its replication, so a batch counts (and the task
/// completes) only once it would survive the loss of the primary.
pub trait ReplicationBarrier: Send + Sync {
    /// Block until every operation the local copy `svc` has sequenced is
    /// on its in-sync copies. An error: the writes are not acknowledged
    /// (the copy stopped being the primary, replication failed).
    fn await_replicated(&self, svc: &crate::index::IndexService) -> Result<()>;
}

/// ES `ScrollableHitSource.SearchFailure` of a source page's shard failure
/// (`index`, `shard`, `node`, `status`, `reason`), with its REST status.
pub fn search_failure(f: &Value) -> (u16, Value) {
    let reason = f.get("reason").cloned().unwrap_or(Value::Null);
    let status = f.get("status").and_then(Value::as_u64).map(|s| s as u16).unwrap_or_else(|| match reason["type"].as_str().unwrap_or("") {
        "no_shard_available_action_exception" | "unavailable_shards_exception" | "node_not_connected_exception" | "node_disconnected_exception" | "node_closed_exception" | "shard_not_found_exception" | "illegal_index_shard_state_exception" => 503,
        "es_rejected_execution_exception" | "circuit_breaking_exception" => 429,
        "index_not_found_exception" | "search_context_missing_exception" | "resource_not_found_exception" => 404,
        "task_cancelled_exception" => 400,
        t if t.contains("parsing") || t.contains("illegal_argument") || t.contains("query_shard") => 400,
        _ => 500,
    });
    let mut m = Map::new();
    for k in ["index", "shard", "node"] {
        if let Some(v) = f.get(k).filter(|v| !v.is_null()) {
            m.insert(k.into(), v.clone());
        }
    }
    m.insert("status".into(), json!(status));
    m.insert("reason".into(), reason);
    (status, Value::Object(m))
}

/// What a source page says about its own completeness: shard failures
/// (as [`search_failure`]s) and `timed_out`. ES's BulkByScroll stops at a
/// page with either, without processing its hits, and reports them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PageCheck {
    pub failures: Vec<(u16, Value)>,
    pub timed_out: bool,
}

impl PageCheck {
    pub fn of(page: &Value) -> PageCheck {
        let mut failures: Vec<(u16, Value)> = page["_shards"]["failures"].as_array().into_iter().flatten().map(search_failure).collect();
        let failed = page["_shards"]["failed"].as_u64().unwrap_or(0) as usize;
        // Counted failures without their entries still make the page partial.
        for _ in failures.len()..failed {
            failures.push(search_failure(&json!({"reason": {"type": "search_phase_execution_exception", "reason": "a source shard failed"}})));
        }
        PageCheck { failures, timed_out: page["timed_out"].as_bool().unwrap_or(false) }
    }

    pub fn complete(&self) -> bool {
        self.failures.is_empty() && !self.timed_out
    }
}

/// ES `BulkByScrollResponse` REST status: 408 when timed out, raised to the
/// worst bulk / search failure status.
pub fn response_status(failures: &[(u16, Value)], timed_out: bool) -> u16 {
    let base = if timed_out { 408 } else { 200 };
    failures.iter().map(|f| f.0).fold(base, u16::max)
}

/// `slices`: a fixed count or `auto` (one per source shard).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slices {
    N(u32),
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Reindex,
    UpdateByQuery,
    DeleteByQuery,
}

impl Kind {
    pub fn action(&self) -> &'static str {
        match self {
            Kind::Reindex => "indices:data/write/reindex",
            Kind::UpdateByQuery => "indices:data/write/update/byquery",
            Kind::DeleteByQuery => "indices:data/write/delete/byquery",
        }
    }
}

/// The caller's read of a lookup document's index (terms lookup,
/// `more_like_this` like documents, `percolate` by id) for a concrete
/// index and the name the query used: its DLS/FLS (`None`: unrestricted),
/// or an error when the caller may not read it.
#[derive(Clone)]
pub struct LookupAccess(pub std::sync::Arc<dyn Fn(&str, &str) -> Result<Option<std::sync::Arc<crate::security::roles::IndexAccess>>> + Send + Sync>);

impl std::fmt::Debug for LookupAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LookupAccess")
    }
}

#[derive(Clone, Debug)]
pub struct ByQueryRequest {
    pub kind: Kind,
    /// Source index expression.
    pub source: String,
    pub resolve: ResolveOpts,
    pub query: Option<Value>,
    pub batch_size: usize,
    pub max_docs: Option<u64>,
    pub source_filter: SourceFilter,
    /// `source._source.exclude_vectors` (also rehydrates `_inference_fields` when false).
    pub source_exclude_vectors: Option<bool>,
    pub conflicts_proceed: bool,
    pub dest_index: Option<String>,
    pub dest_op_create: bool,
    pub dest_version_type: Option<VersionType>,
    /// `keep` (default), `discard` or `=<value>`.
    pub dest_routing: Option<String>,
    pub script: Option<Value>,
    pub raw_script: Option<crate::source_json::Source>,
    pub refresh: bool,
    /// `routing` URL param of by-query requests (applied to writes).
    pub routing: Option<String>,
    /// Ingest pipeline for writes (`dest.pipeline` / `?pipeline=`).
    pub pipeline: Option<String>,
    pub slices: Slices,
    /// Explicit `slice: {id, max}` (one slice of a larger scan).
    pub slice: Option<(u32, u32)>,
    /// Source sort (`source.sort` / `?sort=`); default `_doc`.
    pub sort: Vec<SortSpec>,
    pub sort_extra: HashMap<String, Value>,
    /// `requests_per_second` (`-1` = unthrottled).
    pub rps: f64,
    pub terminate_after: Option<u64>,
    pub require_alias: bool,
    pub remote: Option<RemoteSpec>,
    /// Raw `source.query` / `_source` for the remote search body.
    pub remote_source: Option<Value>,
    pub scroll: String,
    /// The caller's lookup reads (`None`: security off).
    pub lookup_access: Option<LookupAccess>,
}

impl ByQueryRequest {
    pub fn description(&self) -> String {
        match self.kind {
            Kind::Reindex => match &self.remote {
                Some(r) => format!("reindex from [{}][{}] to [{}]", r.base(), self.source.replace(',', ", "), self.dest_index.as_deref().unwrap_or("")),
                None => format!("reindex from [{}] to [{}]", self.source.replace(',', ", "), self.dest_index.as_deref().unwrap_or("")),
            },
            Kind::UpdateByQuery => format!("update-by-query [{}]", self.source.replace(',', ", ")),
            Kind::DeleteByQuery => format!("delete-by-query [{}]", self.source.replace(',', ", ")),
        }
    }
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// request parsing
// ---------------------------------------------------------------------------

/// 1-based `(line, col)` of byte offset `at` in `text`.
fn line_col(text: &str, at: usize) -> (usize, usize) {
    let before = &text[..at.min(text.len())];
    let line = before.matches('\n').count() + 1;
    let col = before.rsplit('\n').next().map(|l| l.chars().count()).unwrap_or(0) + 1;
    (line, col)
}

/// Position of the first `"key"` token at/after byte `from` in `text`:
/// the key's start, or (`value`) the start of its value.
fn locate(text: &str, from: usize, key: &str, value: bool) -> Option<(usize, usize, usize)> {
    let needle = format!("\"{key}\"");
    let mut at = from;
    while let Some(i) = text.get(at..)?.find(&needle) {
        let start = at + i;
        let after = &text[start + needle.len()..];
        let trimmed = after.trim_start();
        if trimmed.starts_with(':') {
            if !value {
                let (l, c) = line_col(text, start);
                return Some((start, l, c));
            }
            let vstart = text.len() - trimmed[1..].trim_start().len();
            let (l, c) = line_col(text, vstart);
            return Some((start, l, c));
        }
        at = start + needle.len();
    }
    None
}

/// 1-based `(line, col)` of the first `"key"` token in a request body (`(1, 1)` if absent);
/// the key's own start, or (`value`) the start of its value.
pub fn key_position(text: &str, key: &str, value: bool) -> (usize, usize) {
    locate(text, 0, key, value).map(|(_, l, c)| (l, c)).unwrap_or((1, 1))
}

/// `[1:65]`-style prefix for a field of `text` (falls back to `[1:1]`).
fn pos_prefix(text: &str, from: usize, key: &str, value: bool) -> (String, usize) {
    match locate(text, from, key, value) {
        Some((start, l, c)) => (format!("[{l}:{c}]"), start),
        None => ("[1:1]".into(), from),
    }
}

/// ObjectParser's `[1:N] [parent] failed to parse field [key]` wrapper around `cause`.
fn field_failure(text: &str, from: usize, parent: &str, key: &str, cause: EsError) -> EsError {
    let (p, _) = pos_prefix(text, from, key, true);
    EsError::x_content_parse(format!("{p} [{parent}] failed to parse field [{key}]")).caused_by(cause)
}

fn parse_conflicts(s: &str) -> Result<bool> {
    match s {
        "proceed" => Ok(true),
        "abort" => Ok(false),
        o => Err(EsError::illegal_argument(format!("conflicts may only be \"proceed\" or \"abort\" but was [{o}]"))),
    }
}

fn parse_max_docs(v: &Value) -> Result<Option<u64>> {
    let n = match v {
        Value::Number(n) => n.as_i64().unwrap_or(-1),
        Value::String(s) => s.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [max_docs] with value [{s}]")))?,
        _ => return Err(EsError::parsing("[max_docs] must be a number")),
    };
    if n == -1 {
        return Ok(None);
    }
    if n <= 0 {
        return Err(EsError::action_validation(&[&format!(
            "maxDocs should be greater than 0 if the request is limited to some number of documents or -1 if it isn't but it was [{n}]"
        )]));
    }
    Ok(Some(n as u64))
}

fn parse_batch(v: &Value, name: &str) -> Result<usize> {
    let n = v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| EsError::illegal_argument(format!("Failed to parse int parameter [{name}] with value [{v}]")))?;
    if n <= 0 {
        return Err(EsError::action_validation(&[&format!("[{}] cannot be [{n}] in a scroll context", if name == "scroll_size" { "size" } else { name })]));
    }
    Ok(n as usize)
}

/// `requests_per_second`: `-1` (unthrottled) or a float > 0.
pub fn parse_rps(s: &str) -> Result<f64> {
    let bad = || EsError::illegal_argument("[requests_per_second] must be a float greater than 0. Use -1 to disable throttling.");
    let v: f64 = s.trim().parse().map_err(|_| bad())?;
    if v == -1.0 {
        return Ok(-1.0);
    }
    if v.is_nan() || v <= 0.0 {
        return Err(bad());
    }
    Ok(v)
}

fn parse_slices(s: &str) -> Result<Slices> {
    if s == "auto" {
        return Ok(Slices::Auto);
    }
    let bad = |cause: Option<EsError>| {
        let e = EsError::illegal_argument(format!("[slices] must be a positive integer or the string \"auto\", but was [{s}]"));
        match cause {
            Some(c) => e.caused_by(c),
            None => e,
        }
    };
    match s.parse::<i64>() {
        Ok(n) if n >= 1 => Ok(Slices::N(n.min(u32::MAX as i64) as u32)),
        Ok(_) => Err(bad(None)),
        Err(_) => Err(bad(Some(EsError::new(400, "number_format_exception", format!("For input string: \"{s}\""))))),
    }
}

fn common_params(r: &mut ByQueryRequest, params: &HashMap<String, String>) -> Result<()> {
    if let Some(c) = params.get("conflicts") {
        r.conflicts_proceed = parse_conflicts(c)?;
    }
    if let Some(m) = params.get("max_docs") {
        r.max_docs = parse_max_docs(&json!(m))?;
    }
    if let Some(s) = params.get("slices") {
        r.slices = parse_slices(s)?;
    }
    if let Some(s) = params.get("requests_per_second") {
        r.rps = parse_rps(s)?;
    }
    if let Some(s) = params.get("scroll") {
        r.scroll = s.clone();
    }
    r.refresh = matches!(params.get("refresh").map(String::as_str), Some("true") | Some(""));
    r.require_alias = params.get("require_alias").map(|v| v == "true" || v.is_empty()).unwrap_or(false);
    Ok(())
}

fn blank(kind: Kind, source: String, resolve: ResolveOpts) -> ByQueryRequest {
    ByQueryRequest {
        kind,
        source,
        resolve,
        query: None,
        batch_size: 1000,
        max_docs: None,
        source_filter: SourceFilter::All,
        source_exclude_vectors: None,
        conflicts_proceed: false,
        dest_index: None,
        dest_op_create: false,
        dest_version_type: None,
        dest_routing: None,
        script: None,
        raw_script: None,
        refresh: false,
        routing: None,
        pipeline: None,
        slices: Slices::N(1),
        slice: None,
        sort: Vec::new(),
        sort_extra: HashMap::new(),
        rps: -1.0,
        terminate_after: None,
        require_alias: false,
        remote: None,
        remote_source: None,
        scroll: "5m".into(),
        lookup_access: None,
    }
}

fn parse_slice(v: &Value) -> Result<(u32, u32)> {
    let id = v.get("id").and_then(Value::as_u64);
    let max = v.get("max").and_then(Value::as_u64);
    match (id, max) {
        (Some(id), Some(max)) if max > 1 && id < max => Ok((id as u32, max as u32)),
        (Some(_), Some(max)) if max <= 1 => Err(EsError::illegal_argument("max must be greater than 1")),
        (Some(id), Some(max)) => Err(EsError::illegal_argument(format!("id must be lower than max; id [{id}] max [{max}]"))),
        _ => Err(EsError::parsing("[slice] requires [id] and [max]")),
    }
}

fn parse_sort(v: &Value, r: &mut ByQueryRequest) -> Result<()> {
    let sr = search::parse_request(&json!({ "sort": v }), &HashMap::new())?;
    r.sort = sr.sort;
    r.sort_extra = sr.sort_extra;
    Ok(())
}

/// Parse a `_reindex` body (`text` is its raw text, for error positions) + params.
pub fn parse_reindex(body: &Value, text: &str, params: &HashMap<String, String>) -> Result<ByQueryRequest> {
    let o = body.as_object().ok_or_else(|| EsError::action_validation(&["source is missing"]))?;
    let mut r = blank(Kind::Reindex, String::new(), ResolveOpts::default());
    let mut src_from = 0usize;
    for (k, v) in o {
        match k.as_str() {
            "source" => {
                let so = match v.as_object() {
                    Some(so) => so,
                    None => {
                        let (p, _) = pos_prefix(text, 0, "source", true);
                        let ty = match v {
                            Value::String(_) => "VALUE_STRING",
                            Value::Number(_) => "VALUE_NUMBER",
                            Value::Bool(_) => "VALUE_BOOLEAN",
                            Value::Array(_) => "START_ARRAY",
                            _ => "VALUE_NULL",
                        };
                        return Err(EsError::x_content_parse(format!("{p} [reindex] source doesn't support values of type: {ty}")));
                    }
                };
                src_from = locate(text, 0, "source", true).map(|x| x.0).unwrap_or(0);
                for (sk, sv) in so {
                    match sk.as_str() {
                        "index" => r.source = str_list(sv).join(","),
                        "query" => {
                            if let Err(e) = crate::search_strict::check_query(sv, "query", None) {
                                let canon = crate::jsonpos::java_hashmap_json(v);
                                let e = crate::jsonpos::resolve(e, canon.as_bytes());
                                return Err(EsError::x_content_parse("[reindex] failed to parse field [source]").at_key("source", None).cursor("end").prefixed("cursor").caused_by(e));
                            }
                            r.query = Some(sv.clone())
                        }
                        "size" => r.batch_size = parse_batch(sv, "size")?,
                        "_source" => {
                            r.source_filter = parse_source_filter(sv)?;
                            r.source_exclude_vectors = sv.get("exclude_vectors").and_then(Value::as_bool);
                            r.remote_source = Some(sv.clone());
                        }
                        "slice" => r.slice = Some(parse_slice(sv)?),
                        "sort" => parse_sort(sv, &mut r)?,
                        "remote" => {
                            let ro = sv.as_object().ok_or_else(|| EsError::parsing("[remote] must be an object"))?;
                            let spec = RemoteSpec::parse(ro).map_err(|e| field_failure(text, src_from, "reindex", "source", e))?;
                            r.remote = Some(spec);
                        }
                        other => {
                            let canon = crate::jsonpos::java_hashmap_json(v);
                            let e = crate::jsonpos::resolve(EsError::unknown_key(other, sv, None), canon.as_bytes());
                            return Err(EsError::x_content_parse("[reindex] failed to parse field [source]").at_key("source", None).cursor("end").prefixed("cursor").caused_by(e));
                        }
                    }
                }
            }
            "dest" => {
                let d = v.as_object().ok_or_else(|| EsError::parsing("[dest] must be an object"))?;
                let dest_from = locate(text, 0, "dest", true).map(|x| x.0).unwrap_or(0);
                for (dk, dv) in d {
                    let inner = |cause: EsError| {
                        let in_ = field_failure(text, dest_from, "dest", dk, cause);
                        let (p, _) = pos_prefix(text, 0, "dest", true);
                        EsError::x_content_parse(format!("{p} [reindex] failed to parse field [dest]")).caused_by(in_)
                    };
                    match dk.as_str() {
                        "index" => r.dest_index = dv.as_str().map(str::to_string),
                        "op_type" => match dv.as_str() {
                            Some("create") => r.dest_op_create = true,
                            Some("index") => r.dest_op_create = false,
                            o => return Err(inner(EsError::illegal_argument(format!("opType must be 'create' or 'index', found: [{}]", o.unwrap_or(""))))),
                        },
                        "version_type" => r.dest_version_type = Some(VersionType::parse(dv.as_str().unwrap_or("")).map_err(|_| inner(EsError::illegal_argument(format!("No version type match [{}]", dv.as_str().unwrap_or("")))))?),
                        "routing" => r.dest_routing = dv.as_str().map(str::to_string),
                        "pipeline" => r.pipeline = dv.as_str().map(str::to_string),
                        other => {
                            let (p, _) = pos_prefix(text, dest_from, other, false);
                            return Err(EsError::x_content_parse(format!("{p} [dest] unknown field [{other}]")));
                        }
                    }
                }
            }
            "conflicts" => r.conflicts_proceed = parse_conflicts(v.as_str().unwrap_or("")).map_err(|e| field_failure(text, 0, "reindex", "conflicts", e))?,
            "max_docs" => {
                let n = v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
                if let Some(n) = n {
                    if n != -1 && n < 1 {
                        return Err(field_failure(text, 0, "reindex", "max_docs", EsError::illegal_argument("[max_docs] should be >= [slices]")));
                    }
                }
                r.max_docs = parse_max_docs(v)?;
            }
            "size" => return Err(field_failure(text, 0, "reindex", "size", EsError::illegal_argument("invalid parameter [size], use [max_docs] instead"))),
            "script" => r.script = Some(v.clone()),
            other => {
                let (p, _) = pos_prefix(text, 0, other, false);
                return Err(EsError::x_content_parse(format!("{p} [reindex] unknown field [{other}]")));
            }
        }
    }
    common_params(&mut r, params)?;
    let mut errs: Vec<String> = Vec::new();
    if r.source.is_empty() {
        errs.push("use _all if you really want to copy from all existing indexes".into());
    }
    match &r.dest_index {
        None => errs.push("index must be specified".into()),
        Some(d) if r.remote.is_none() && r.source.split(',').any(|s| s == d) => errs.push(format!("reindex cannot write into an index its reading from [{d}]")),
        _ => {}
    }
    if r.dest_op_create && r.dest_version_type.map(|v| v != VersionType::Internal).unwrap_or(false) {
        errs.push("create operations only support internal versioning. use index instead".into());
    }
    if r.remote.is_some() {
        if let Slices::N(n) = r.slices {
            if n > 1 {
                errs.push(format!("reindex from remote sources doesn't support slices > 1 but was [{n}]"));
            }
        }
        if r.slices == Slices::Auto {
            // `auto` resolves to 1 for remote sources.
            r.slices = Slices::N(1);
        }
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    if let Some(s) = &r.script {
        crate::script::parse_spec(s)?;
    }
    Ok(r)
}

fn json_token(v: &Value) -> &'static str {
    match v {
        Value::Number(_) => "VALUE_NUMBER",
        Value::String(_) => "VALUE_STRING",
        Value::Bool(_) => "VALUE_BOOLEAN",
        Value::Array(_) => "START_ARRAY",
        Value::Object(_) => "START_OBJECT",
        Value::Null => "VALUE_NULL",
    }
}

/// Parse an `_update_by_query` / `_delete_by_query` request.
pub fn parse_by_query(kind: Kind, index: &str, body: &Value, params: &HashMap<String, String>) -> Result<ByQueryRequest> {
    let mut r = blank(kind, index.to_string(), ResolveOpts::from_params(params));
    r.routing = params.get("routing").cloned();
    r.pipeline = params.get("pipeline").cloned();
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            match k.as_str() {
                "query" => {
                    crate::search_strict::check_query(v, "query", None)?;
                    r.query = Some(v.clone())
                }
                "conflicts" => r.conflicts_proceed = parse_conflicts(v.as_str().unwrap_or(""))?,
                "max_docs" => r.max_docs = parse_max_docs(v)?,
                "size" => {
                    if r.max_docs.is_none() {
                        r.max_docs = parse_max_docs(v)?
                    }
                }
                "script" if kind == Kind::UpdateByQuery => {
                    if !(v.is_string() || v.is_object()) {
                        return Err(EsError::illegal_argument("Script value should be a String or a Map"));
                    }
                    r.script = Some(v.clone())
                }
                "slice" => r.slice = Some(parse_slice(v)?),
                "sort" => parse_sort(v, &mut r)?,
                other => return Err(EsError::unknown_key(other, v, None)),
            }
        }
    }
    if params.contains_key("size") {
        return Err(EsError::illegal_argument("invalid parameter [size], use [max_docs] instead"));
    }
    if let Some(q) = params.get("q") {
        let mut qs = Map::new();
        qs.insert("query".into(), json!(q));
        for (p, k) in [("df", "default_field"), ("default_operator", "default_operator"), ("analyzer", "analyzer"), ("lenient", "lenient"), ("analyze_wildcard", "analyze_wildcard")] {
            if let Some(v) = params.get(p) {
                qs.insert(k.into(), if k == "lenient" || k == "analyze_wildcard" { json!(v == "true") } else { json!(v) });
            }
        }
        if body.get("query").is_none() {
            r.query = Some(json!({ "query_string": qs }));
        }
    }
    if let Some(s) = params.get("scroll_size") {
        r.batch_size = parse_batch(&json!(s), "scroll_size")?;
    }
    if let Some(s) = params.get("sort") {
        if r.sort.is_empty() {
            let mut p = HashMap::new();
            p.insert("sort".to_string(), s.clone());
            let sr = search::parse_request(&json!({}), &p)?;
            r.sort = sr.sort;
            r.sort_extra = sr.sort_extra;
        }
    }
    if let Some(t) = params.get("terminate_after") {
        r.terminate_after = t.parse::<u64>().ok().filter(|n| *n > 0);
    }
    common_params(&mut r, params)?;
    let mut errs: Vec<String> = Vec::new();
    if params.get("from").and_then(|f| f.parse::<i64>().ok()).map(|f| f > 0).unwrap_or(false) {
        errs.push("using [from] is not allowed in a scroll context".into());
        errs.push("from is not supported in this context".into());
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    if kind == Kind::DeleteByQuery && r.query.is_none() {
        return Err(EsError::action_validation(&["query is missing"]));
    }
    if let Some(s) = &r.script {
        crate::script::parse_spec(s)?;
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// statistics
/// ES `BulkByScrollTask` retry policy: a bulk item rejected (429) is
/// retried with `BackoffPolicy.exponentialBackoff`
/// (`retry_backoff_initial_time` 500ms, `max_retries` 11) before it counts
/// as a failure.
pub const BULK_MAX_RETRIES: u32 = 11;
#[cfg(not(test))]
const BULK_RETRY_INITIAL_MS: u64 = 500;
#[cfg(test)]
const BULK_RETRY_INITIAL_MS: u64 = 1;

/// ES `ExponentialBackoff`: the delay before retry `i` (from 0) is
/// `initial + 10ms * (floor(e^(0.8 i)) - 1)`.
pub fn bulk_retry_delay(i: u32) -> Duration {
    #[cfg(test)]
    let step = 0u64;
    #[cfg(not(test))]
    let step = 10u64;
    Duration::from_millis(BULK_RETRY_INITIAL_MS + step * ((0.8 * i as f64).exp() as u64).saturating_sub(1))
}

/// Sleep `delay`, waking early (`false`) once `cancelled` holds.
pub fn sleep_unless_cancelled(delay: Duration, cancelled: &dyn Fn() -> bool) -> bool {
    let until = Instant::now() + delay;
    loop {
        if cancelled() {
            return false;
        }
        let now = Instant::now();
        if now >= until {
            return true;
        }
        std::thread::sleep((until - now).min(Duration::from_millis(50)));
    }
}

/// `write` the requests, re-sending the rejected (429) ones with backoff
/// (a cancelled task stops waiting: the rejections stand); each retry round
/// counts in `retries`. Results in request order.
fn write_retrying(mut write: impl FnMut(Vec<WriteReq>) -> Vec<Result<crate::shard::WriteResult>>, reqs: Vec<WriteReq>, retries: &mut u64, cancelled: &dyn Fn() -> bool) -> Vec<Result<crate::shard::WriteResult>> {
    let mut results: Vec<Option<Result<crate::shard::WriteResult>>> = (0..reqs.len()).map(|_| None).collect();
    let mut pending: Vec<usize> = (0..reqs.len()).collect();
    for attempt in 0..=BULK_MAX_RETRIES {
        let out = write(pending.iter().map(|&i| reqs[i].clone()).collect());
        let mut again = Vec::new();
        for (&i, r) in pending.iter().zip(out) {
            if matches!(&r, Err(e) if e.status == 429) { again.push(i); }
            results[i] = Some(r);
        }
        // The last attempt's rejections, or a cancelled task's, stand.
        if again.is_empty() || attempt == BULK_MAX_RETRIES {
            break;
        }
        *retries += 1;
        if !sleep_unless_cancelled(bulk_retry_delay(attempt), cancelled) {
            break;
        }
        pending = again;
    }
    results.into_iter().map(|r| r.unwrap_or_else(|| Err(EsError::internal("bulk item got no result")))).collect()
}

// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub total: u64,
    pub updated: u64,
    pub created: u64,
    pub deleted: u64,
    pub batches: u64,
    pub version_conflicts: u64,
    pub noops: u64,
    pub throttled_micros: u64,
    pub failures: Vec<(u16, Value)>,
    pub canceled: Option<String>,
    pub started: bool,
    /// Bulk items retried after a rejection (ES `retries.bulk`).
    pub retries_bulk: u64,
    /// A source page timed out (ES `timed_out`; the run stopped there).
    pub timed_out: bool,
}

impl Stats {
    /// ES `BulkByScrollTask.Status.getSuccessfullyProcessed`: what counts
    /// toward `max_docs`.
    pub fn successful(&self) -> u64 {
        self.created + self.updated + self.deleted
    }

    fn merge(&mut self, o: &Stats) {
        self.total += o.total;
        self.updated += o.updated;
        self.created += o.created;
        self.deleted += o.deleted;
        self.batches += o.batches;
        self.version_conflicts += o.version_conflicts;
        self.noops += o.noops;
        self.throttled_micros += o.throttled_micros;
        self.retries_bulk += o.retries_bulk;
        self.failures.extend(o.failures.iter().cloned());
        self.timed_out |= o.timed_out;
        if self.canceled.is_none() {
            self.canceled = o.canceled.clone();
        }
    }

    /// BulkByScroll task status (`Status.innerXContent`).
    pub fn status_json(&self, kind: Kind, rps: f64, until_ms: u64, slice_id: Option<u32>) -> Map<String, Value> {
        let mut m = Map::new();
        if let Some(id) = slice_id {
            m.insert("slice_id".into(), json!(id));
        }
        m.insert("total".into(), json!(self.total));
        if kind != Kind::DeleteByQuery {
            m.insert("updated".into(), json!(self.updated));
        }
        if kind == Kind::Reindex {
            m.insert("created".into(), json!(self.created));
        }
        m.insert("deleted".into(), json!(self.deleted));
        m.insert("batches".into(), json!(self.batches));
        m.insert("version_conflicts".into(), json!(self.version_conflicts));
        m.insert("noops".into(), json!(self.noops));
        m.insert("retries".into(), json!({"bulk": self.retries_bulk, "search": 0}));
        m.insert("throttled_millis".into(), json!(self.throttled_micros / 1000));
        m.insert("requests_per_second".into(), json!(rps));
        if let Some(c) = &self.canceled {
            m.insert("canceled".into(), json!(c));
        }
        m.insert("throttled_until_millis".into(), json!(until_ms));
        m
    }
}

/// All slices of a run (a single entry when unsliced).
struct Run {
    kind: Kind,
    /// The slices started so far, in order (slice `i` is entry `i`).
    stats: Vec<Stats>,
    /// The slice count (the status lists every slice, `null` until it starts).
    n_slices: u32,
    /// `slices > 1`: report a `slices` array.
    sliced: bool,
    /// Explicit `slice.id` of an unsliced run.
    slice_id: Option<u32>,
}

impl Run {
    fn totals(&self) -> Stats {
        let mut t = Stats::default();
        for s in &self.stats {
            t.merge(s);
        }
        t
    }

    fn status(&self, rps: f64, until_ms: u64) -> Map<String, Value> {
        let n = self.n_slices.max(1) as f64;
        let mut m = self.totals().status_json(self.kind, rps, until_ms, self.slice_id);
        if self.sliced {
            let each = if rps < 0.0 { rps } else { rps / n };
            let slices: Vec<Value> = (0..self.n_slices as usize)
                .map(|i| match self.stats.get(i) {
                    Some(s) if s.started => Value::Object(s.status_json(self.kind, each, until_ms, Some(i as u32))),
                    _ => Value::Null,
                })
                .collect();
            m.insert("slices".into(), Value::Array(slices));
        }
        m
    }
}

/// `TimeValue.toString()` of a millisecond count (`0s`, `5ms`, `1.1s`, `2m`).
pub fn time_value_str(ms: u64) -> String {
    if ms == 0 {
        return "0s".into();
    }
    let fmt = |v: f64, suffix: &str| {
        let s = format!("{:.1}", v);
        format!("{}{suffix}", s.strip_suffix(".0").unwrap_or(&s))
    };
    match ms {
        0..=999 => format!("{ms}ms"),
        1000..=59_999 => fmt(ms as f64 / 1000.0, "s"),
        60_000..=3_599_999 => fmt(ms as f64 / 60_000.0, "m"),
        3_600_000..=86_399_999 => fmt(ms as f64 / 3_600_000.0, "h"),
        _ => fmt(ms as f64 / 86_400_000.0, "d"),
    }
}

/// Add the human-readable `throttled` / `throttled_until` fields a stored task response carries.
pub fn with_human(v: &mut Value) {
    fn rebuild(m: &mut Map<String, Value>) {
        let old = std::mem::take(m);
        let tm = old.get("throttled_millis").and_then(Value::as_u64).unwrap_or(0);
        let tu = old.get("throttled_until_millis").and_then(Value::as_u64).unwrap_or(0);
        for (k, val) in old {
            if k == "throttled_millis" {
                m.insert("throttled".into(), json!(time_value_str(tm)));
            }
            if k == "throttled_until_millis" {
                m.insert("throttled_until".into(), json!(time_value_str(tu)));
            }
            m.insert(k, val);
        }
    }
    if let Value::Object(m) = v {
        rebuild(m);
        if let Some(Value::Array(sl)) = m.get_mut("slices") {
            for s in sl.iter_mut().filter_map(Value::as_object_mut) {
                rebuild(s);
            }
        }
    }
}

fn failure_json(index: &str, id: &str, e: &EsError) -> Value {
    let mut cause = e.error_object(false);
    if let Some(o) = cause.as_object_mut() {
        o.remove("root_cause");
    }
    json!({"index": index, "id": id, "cause": cause, "status": e.status})
}

struct Hit {
    index: String,
    /// The UUID of the index incarnation the hit was read from (local
    /// sources): a write built from the hit goes to that incarnation only,
    /// never to an index recreated under the same name since.
    index_uuid: Option<String>,
    id: String,
    version: u64,
    seq_no: i64,
    /// The hit's `_primary_term`: with `seq_no`, the condition its write
    /// carries (ES `copyMetadata`).
    primary_term: Option<u64>,
    routing: Option<String>,
    source: Option<String>,
}

fn parse_hits(out: &str) -> Result<(Vec<Hit>, PageCheck)> {
    if let Ok(v) = serde_json::from_str::<Value>(out) { return Ok((hits_of(&v), PageCheck::of(&v))); }
    let mut sources = crate::source_json::ResponseSources::default();
    let v = sources.parse(out.as_bytes())?;
    let check = PageCheck::of(&v);
    let mut hits = hits_of(&v);
    if let Some(raw_hits) = v["hits"]["hits"].as_array() {
        for (hit, raw) in hits.iter_mut().zip(raw_hits) {
            if let Some(token) = raw.get("_source") {
                // The hit's source, kept lossless by the parse (the hit
                // holds its ordinal). One that does not resolve is an
                // error: indexing the ordinal would write "0", "1", ...
                hit.source = Some(sources.source_text(token).ok_or_else(|| EsError::new(500, "illegal_state_exception", format!("unresolved source of [{}][{}]", hit.index, hit.id)))?);
            }
        }
    }
    Ok((hits, check))
}

fn hits_of(v: &Value) -> Vec<Hit> {
    v["hits"]["hits"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|h| Hit {
            index: h["_index"].as_str().unwrap_or("").to_string(),
            index_uuid: None,
            id: h["_id"].as_str().unwrap_or("").to_string(),
            version: h["_version"].as_u64().unwrap_or(1),
            seq_no: h["_seq_no"].as_i64().unwrap_or(0),
            primary_term: h["_primary_term"].as_u64(),
            routing: h["_routing"].as_str().map(str::to_string),
            // `"_source": null` is no source either (ES `didn't store _source`).
            source: h.get("_source").filter(|s| !s.is_null()).map(|s| s.to_string()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

/// Sleep until the batch's share of the rate budget has passed (`docs / rps`
/// seconds after the batch started), re-reading the rate so `_rethrottle`
/// takes effect immediately.
fn throttle(task: &Task, share: f64, batch_start: Instant, docs: usize, run: &mut Run, idx: usize, publish: &dyn Fn(&Run, u64)) {
    loop {
        let rps = task.rps();
        if rps <= 0.0 || task.is_cancelled() {
            break;
        }
        let target = batch_start + Duration::from_secs_f64(docs as f64 / (rps / share));
        let now = Instant::now();
        if now >= target {
            break;
        }
        publish(run, (target - now).as_millis() as u64);
        let nap = (target - now).min(Duration::from_millis(10));
        std::thread::sleep(nap);
        run.stats[idx].throttled_micros += nap.as_micros() as u64;
    }
    publish(run, 0);
}

/// What one slice's state is charged on the request breaker: its counters
/// and its entry in the task status.
const SLICE_STATE_BYTES: u64 = 1024;

/// ES `DefaultSearchContext.preProcess`: a sliced search with more slices
/// than a target index's `index.max_slices_per_scroll` (default 1024)
/// fails on that index's shards.
fn check_slice_limit(snap: &[search::Target], slices: u32) -> Result<()> {
    for t in snap {
        let max = t.svc.setting_usize("index.max_slices_per_scroll", 1024);
        if slices as usize > max {
            return Err(EsError::search_phase(
                &t.svc.name,
                EsError::illegal_argument(format!(
                    "The number of slices [{slices}] is too large. It must be less than [{max}]. This limit can be set by changing the [index.max_slices_per_scroll] index level setting."
                )),
            ));
        }
    }
    Ok(())
}

/// Run the operation; `task` receives status updates, cancellation and rethrottling.
pub fn execute(node: &Node, r: &ByQueryRequest, task: &Task) -> Result<(u16, Value)> {
    let start = Instant::now();
    task.set_rps(r.rps);
    let mut ropts = r.resolve;
    if r.source == "_all" || r.source.is_empty() {
        ropts.allow_no_indices = true;
    }
    let snap = if r.remote.is_some() {
        Vec::new()
    } else {
        let targets = node.resolve(if r.source.is_empty() { "_all" } else { &r.source }, ropts)?;
        node.snapshot(&targets)?
    };
    let n_slices = match (r.slice, r.slices) {
        (Some(_), _) => 1,
        (None, Slices::N(n)) => n,
        (None, Slices::Auto) => {
            if r.remote.is_some() {
                1
            } else {
                snap.iter().map(|t| t.svc.setting_usize("index.number_of_shards", 1) as u32).min().unwrap_or(1).clamp(1, 20)
            }
        }
    };
    // Every slice's search carries `slice.max`: over a source index's
    // `index.max_slices_per_scroll` it fails as ES's shard search does,
    // before any slice state exists.
    let slice_max = match r.slice {
        Some((_, max)) => max,
        None => n_slices,
    };
    if slice_max > 1 {
        check_slice_limit(&snap, slice_max)?;
    }
    // The per-slice state (and the status table naming every slice) is
    // admitted on the request breaker before it is built. Slices run one
    // after another: a slice's counters exist only once it starts.
    let _slice_state = crate::memory::breakers().reserve_request(u64::from(n_slices) * SLICE_STATE_BYTES, "<bulk_by_scroll_slices>")?;
    let dest = match (&r.kind, &r.dest_index) {
        (Kind::Reindex, Some(d)) => {
            if r.require_alias && !node.all_indices().iter().any(|s| s.aliases().contains_key(d.as_str())) {
                return Err(EsError::new(404, "index_not_found_exception", format!("no such index [{d}] and [require_alias] request flag is [true] and [{d}] is not an alias"))
                    .with("resource.type", "index_expression")
                    .with("resource.id", d.as_str())
                    .with("index_uuid", "_na_")
                    .with("index", d.as_str()));
            }
            Some((d.clone(), node.write_index(d, true)?))
        }
        _ => None,
    };
    let mut run = Run { kind: r.kind, stats: Vec::new(), n_slices, sliced: n_slices > 1, slice_id: r.slice.map(|s| s.0) };
    let publish = |run: &Run, until: u64| task.set_status(Value::Object(run.status(task.rps(), until)));
    let mut touched: Vec<std::sync::Arc<crate::index::IndexService>> = Vec::new();
    let script = match &r.script {
        Some(s) => {
            let ctx = if r.kind == Kind::Reindex { crate::script::ScriptContext::Reindex } else { crate::script::ScriptContext::UpdateByQuery };
            Some(crate::script::compile_json_source(s, r.raw_script.as_ref(), ctx, Some(&crate::script::stored::for_data_dir(&node.data)))?)
        }
        None => None,
    };
    let cap_total = match (r.max_docs, r.terminate_after) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    for idx in 0..n_slices as usize {
        let slice = if n_slices > 1 { Some((idx as u32, n_slices)) } else { r.slice };
        let max_docs = cap_total.map(|m| if n_slices > 1 { m / n_slices as u64 + u64::from((idx as u64) < m % n_slices as u64) } else { m });
        run.stats.push(Stats { started: true, ..Stats::default() });
        publish(&run, 0);
        let abort = run_slice(node, r, task, &snap, dest.as_ref(), script.as_ref(), slice, max_docs, n_slices as f64, idx, &mut run, &mut touched, &publish)?;
        if abort || task.is_cancelled() {
            break;
        }
    }
    if r.refresh {
        for s in &touched {
            s.refresh()?;
        }
    }
    // The final replication barrier: the task completes (and reports its
    // counts) only once every write it made is acknowledged by the in-sync
    // copies. Each batch already waited for its own writes; this covers
    // anything a batch left behind.
    for s in &touched {
        if let Err(e) = node.await_replicated(s) {
            if run.stats.is_empty() {
                run.stats.push(Stats::default());
            }
            run.stats[0].failures.push((e.status, failure_json(&s.name, "", &e)));
        }
    }
    let rps = task.rps();
    publish(&run, 0);
    let timed_out = run.stats.iter().any(|s| s.timed_out);
    let mut m = Map::new();
    m.insert("took".into(), json!(start.elapsed().as_millis() as u64));
    m.insert("timed_out".into(), json!(timed_out));
    m.extend(run.status(rps, 0));
    let failures: Vec<(u16, Value)> = run.stats.iter().flat_map(|s| s.failures.iter().cloned()).collect();
    // Keep ES's field order: `slices` precedes `failures`.
    m.insert("failures".into(), Value::Array(failures.iter().map(|f| f.1.clone()).collect()));
    let status = response_status(&failures, timed_out);
    Ok((status, Value::Object(m)))
}

/// Where a slice reads its batches from.
enum Source {
    Local { cursors: Option<Vec<search::Cursor>> },
    Remote { scroll_id: Option<String> },
}

#[allow(clippy::too_many_arguments)]
fn run_slice(
    node: &Node,
    r: &ByQueryRequest,
    task: &Task,
    snap: &[search::Target],
    dest: Option<&(String, std::sync::Arc<crate::index::IndexService>)>,
    script: Option<&crate::script::CompiledScript>,
    slice: Option<(u32, u32)>,
    max_docs: Option<u64>,
    share: f64,
    idx: usize,
    run: &mut Run,
    touched: &mut Vec<std::sync::Arc<crate::index::IndexService>>,
    publish: &dyn Fn(&Run, u64),
) -> Result<bool> {
    let lookup = |i: &str, id: &str, path: &str, _routing: Option<&str>| -> Result<Vec<Value>> {
        let svc = node.write_index(i, false)?;
        // A lookup is the caller's read of the lookup index (ES runs it as
        // a get under the caller's privileges and DLS/FLS).
        let acc = match &r.lookup_access {
            Some(a) => (a.0)(&svc.name, i)?,
            None => None,
        };
        let Some(g) = svc.get(id)? else { return Ok(vec![]) };
        // DLS judges exactly the fetched document (its shard and version).
        if let Some(acc) = &acc {
            if !search::fetched_doc_visible(&svc, id, None, &g, acc, false)? {
                return Ok(vec![]);
            }
        }
        let mut src: Value = g.source.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(Value::Null);
        if let Some(fa) = acc.as_ref().and_then(|a| a.fields.as_ref()) {
            src = fa.filter_source(&src);
        }
        Ok(crate::highlight::source_values(&src, path).into_iter().cloned().collect())
    };
    let batch = match max_docs {
        Some(m) => r.batch_size.min(m as usize).max(1),
        None => r.batch_size,
    };
    let sort = if r.sort.is_empty() {
        vec![SortSpec { field: "_doc".into(), descending: false, missing: None, mode: None, unmapped_type: None, format: None, numeric_type: None, extra: None }]
    } else {
        r.sort.clone()
    };
    let sreq = SearchRequest {
        query: r.query.clone(),
        size: batch,
        sort,
        sort_extra: r.sort_extra.clone(),
        version: true,
        seq_no_primary_term: true,
        slice,
        source: if r.kind == Kind::DeleteByQuery { SourceFilter::None } else { r.source_filter.clone() },
        exclude_vectors: r.source_exclude_vectors,
        vectors_by_default: r.kind != Kind::DeleteByQuery,
        track_total_hits: Some(TrackTotalHits::True),
        ..Default::default()
    };
    // `https://` remotes connect with the node's `reindex.ssl.*` settings.
    let remote = match &r.remote {
        Some(spec) => Some(spec.with_node_tls(&node.node_settings.read())?),
        None => None,
    };
    let mut source = if remote.is_some() { Source::Remote { scroll_id: None } } else { Source::Local { cursors: None } };
    let mut abort = false;
    let mut first = true;
    let result = (|| -> Result<bool> {
        loop {
            if task.is_cancelled() {
                run.stats[idx].canceled = Some("by user request".into());
                break;
            }
            let batch_start = Instant::now();
            let (mut hits, total, check) = match &mut source {
                Source::Local { cursors } => {
                    let out = search::execute(snap, &sreq, &ExecOpts { terms_lookup: Some(&lookup), cursors: cursors.as_deref(), max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
                    *cursors = Some(out.cursors);
                    let (mut hits, check) = parse_hits(&out.json)?;
                    // The incarnation each hit came from: the snapshot's.
                    for h in &mut hits {
                        h.index_uuid = snap.iter().find(|t| t.hit_index() == h.index).map(|t| t.svc.uuid.clone());
                    }
                    node.by_query_page_fetched();
                    (hits, out.total, check)
                }
                Source::Remote { scroll_id } => {
                    let spec = remote.as_ref().unwrap();
                    let text = match scroll_id {
                        None => {
                            // Version probe, as the reindex action does before searching.
                            spec.call("GET", "/", None)?;
                            let mut body = Map::new();
                            if let Some(q) = &r.query {
                                body.insert("query".into(), q.clone());
                            }
                            if let Some(s) = &r.remote_source {
                                body.insert("_source".into(), s.clone());
                            }
                            let ver = matches!(r.dest_version_type, Some(VersionType::External) | Some(VersionType::ExternalGte));
                            let idxs = if r.source.is_empty() { "_all".to_string() } else { r.source.clone() };
                            let sort = if r.sort.is_empty() { "_doc%3Aasc".to_string() } else { sort_param(&r.sort) };
                            spec.call(
                                "POST",
                                &format!("/{idxs}/_search?scroll={}&sort={sort}&size={batch}&allow_partial_search_results=false&version={ver}", r.scroll),
                                Some(&Value::Object(body).to_string()),
                            )?
                        }
                        Some(id) => spec.call("POST", "/_search/scroll", Some(&json!({"scroll": r.scroll, "scroll_id": id}).to_string()))?,
                    };
                    let v: Value = serde_json::from_str(&text)?;
                    *scroll_id = v["_scroll_id"].as_str().map(str::to_string);
                    let total = match &v["hits"]["total"] {
                        Value::Object(o) => o.get("value").and_then(Value::as_u64).unwrap_or(0),
                        Value::Number(n) => n.as_u64().unwrap_or(0),
                        _ => 0,
                    };
                    (hits_of(&v), total, PageCheck::of(&v))
                }
            };
            // ES `AbstractAsyncBulkByScrollAction.onScrollResponse`: a page
            // with shard failures or a timeout ends the run with them; its
            // hits are not processed.
            if !check.complete() {
                run.stats[idx].timed_out |= check.timed_out;
                run.stats[idx].failures.extend(check.failures);
                abort = true;
                break;
            }
            if first {
                run.stats[idx].total = max_docs.map(|m| m.min(total)).unwrap_or(total);
                first = false;
            }
            if let Some(m) = max_docs {
                // ES counts the successfully processed documents (created,
                // updated, deleted) toward `max_docs`: conflicts skipped
                // with `conflicts: proceed` and noops do not use it up.
                let left = m.saturating_sub(run.stats[idx].successful()) as usize;
                hits.truncate(left);
            }
            if hits.is_empty() {
                break;
            }
            // ES `AbstractAsyncBulkByScrollAction.accept`: a hit without
            // `_source` (not stored, or not fetched) fails the request
            // before its batch is written; it is never rewritten as `{}`.
            if r.kind != Kind::DeleteByQuery {
                if let Some(h) = hits.iter().find(|h| h.source.is_none()) {
                    return Err(EsError::illegal_argument(format!("[{}][{}] didn't store _source", h.index, h.id)));
                }
            }
            let docs = hits.len();
            run.stats[idx].batches += 1;
            let st = &mut run.stats[idx];
            // Group writes by destination index service (preserving order).
            let mut groups: Vec<(std::sync::Arc<crate::index::IndexService>, Vec<(String, WriteReq)>)> = Vec::new();
            for mut h in hits {
                let (svc, req) = match r.kind {
                    Kind::DeleteByQuery | Kind::UpdateByQuery => {
                        // The write goes to the incarnation the hit was read
                        // from: an index deleted since fails it, one
                        // recreated under the same name conflicts (its
                        // documents are not the hit's, whatever their
                        // seq_no and term).
                        let svc = match (node.get_index(&h.index), h.index_uuid.as_deref()) {
                            (Some(svc), Some(uuid)) if svc.uuid != uuid => {
                                let e = EsError::version_conflict(
                                    &h.index,
                                    uuid,
                                    &h.id,
                                    format!("version conflict, required seqNo [{}], primary term [{}]. the index was replaced by [{}/{}] since the document was read", h.seq_no, h.primary_term.unwrap_or(0), h.index, svc.uuid),
                                );
                                st.version_conflicts += 1;
                                if !r.conflicts_proceed {
                                    st.failures.push((e.status, failure_json(&h.index, &h.id, &e)));
                                    abort = true;
                                }
                                continue;
                            }
                            // The snapshot's own service of that incarnation
                            // (never a later lookup by name).
                            (Some(svc), Some(uuid)) => snap.iter().find(|t| t.svc.uuid == uuid).map(|t| t.svc.clone()).unwrap_or(svc),
                            (Some(svc), None) => svc,
                            (None, _) => {
                                let e = EsError::index_not_found(&h.index);
                                st.failures.push((e.status, failure_json(&h.index, &h.id, &e)));
                                abort = true;
                                continue;
                            }
                        };
                        // The hit's own term (what if_primary_term is compared with).
                        let doc_term = match h.primary_term {
                            Some(t) => t,
                            None => {
                                let sh = svc.route(&h.id, h.routing.as_deref()).ok().and_then(|n| svc.shard(n));
                                sh.map(|sh| sh.shard.doc_primary_term(&h.id).unwrap_or_else(|| sh.shard.primary_term())).unwrap_or(1)
                            }
                        };
                        let cond = WriteCond { if_seq_no: Some(h.seq_no), if_primary_term: Some(doc_term), ..Default::default() };
                        let routing = h.routing.clone();
                        let req = if r.kind == Kind::DeleteByQuery {
                            WriteReq::Delete { id: h.id.clone(), routing, cond }
                        } else if let Some(sc) = script {
                            let Some(src) = h.source.as_deref() else { continue };
                            let meta = crate::script::update::DocMeta {
                                index: h.index.clone(),
                                id: Some(h.id.clone()),
                                routing: h.routing.clone(),
                                version: Some(h.version),
                                seq_no: Some(h.seq_no),
                                primary_term: Some(doc_term),
                            };
                            let (out, raw_source) = crate::script::update::run_doc_script_source(sc, src, &meta, "index")?;
                            match out.op {
                                crate::script::update::DocOp::Noop => {
                                    st.noops += 1;
                                    continue;
                                }
                                crate::script::update::DocOp::Delete => WriteReq::Delete { id: h.id.clone(), routing, cond },
                                _ => WriteReq::Index { id: Some(h.id.clone()), source: raw_source, routing, cond },
                            }
                        } else {
                            let Some(source) = h.source.clone() else { continue };
                            WriteReq::Index { id: Some(h.id.clone()), source, routing, cond }
                        };
                        // Ingest pipelines (`?pipeline=`, index default/final).
                        let req = match req {
                            WriteReq::Index { id, source, routing, mut cond } => {
                                match node.run_ingest(crate::ingest::IngestRequest { index: &h.index, id: id.as_deref(), routing: routing.as_deref(), source: &source, pipeline: r.pipeline.as_deref(), cond: crate::ingest::WriteMeta::of(&cond) }) {
                                    Ok(crate::ingest::IngestOutcome::Dropped { .. }) => {
                                        st.noops += 1;
                                        continue;
                                    }
                                    Ok(crate::ingest::IngestOutcome::Changed { source, cond: after, .. }) => {
                                        after.apply(&mut cond);
                                        WriteReq::Index { id, source, routing, cond }
                                    }
                                    Ok(crate::ingest::IngestOutcome::Unchanged) => WriteReq::Index { id, source, routing, cond },
                                    Err(e) => {
                                        st.failures.push((e.status, failure_json(&svc.name, &h.id, &e)));
                                        abort = true;
                                        continue;
                                    }
                                }
                            }
                            other => other,
                        };
                        (svc, req)
                    }
                    Kind::Reindex => {
                        let (mut dname, mut svc) = dest.cloned().unwrap();
                        let Some(mut src) = h.source.clone() else { continue };
                        let mut delete = false;
                        // ES `Reindexer`: the destination request is built
                        // first (id, `dest.routing` policy, external version),
                        // then the script's metadata changes apply over it: a
                        // cleared `_id` generates one (production review
                        // #191), a changed `_routing` wins over the policy
                        // (#199), and a scripted delete keeps the version
                        // condition (#241).
                        let source_routing = h.routing.clone();
                        let mut id: Option<String> = Some(h.id.clone());
                        let mut routing = match r.dest_routing.as_deref() {
                            None | Some("keep") => source_routing.clone(),
                            Some("discard") => None,
                            Some(v) => Some(v.strip_prefix('=').unwrap_or(v).to_string()),
                        };
                        let mut version = match r.dest_version_type {
                            Some(VersionType::External) | Some(VersionType::ExternalGte) => Some(h.version),
                            _ => None,
                        };
                        if let Some(sc) = script {

                            let meta = crate::script::update::DocMeta { index: h.index.clone(), id: Some(h.id.clone()), routing: source_routing.clone(), version: Some(h.version), ..Default::default() };
                            let (out, raw_source) = crate::script::update::run_doc_script_source(sc, &src, &meta, "index")?;
                            match out.op {
                                crate::script::update::DocOp::Noop => {
                                    st.noops += 1;
                                    continue;
                                }
                                crate::script::update::DocOp::Delete => delete = true,
                                _ => {}
                            }
                            src = raw_source;
                            if out.meta.index != h.index {
                                dname = out.meta.index.clone();
                                svc = node.write_index(&dname, true)?;
                            }
                            if out.meta.id.as_deref() != Some(h.id.as_str()) {
                                id = out.meta.id;
                            }
                            if out.meta.routing != source_routing {
                                routing = out.meta.routing;
                            }
                            if let Some(v) = out.meta.version.filter(|v| *v != h.version) {
                                version = Some(v);
                            }
                        }
                        let key = id.clone().unwrap_or_default();
                        if delete {
                            // ES deletes with the request's version under internal
                            // (equality) semantics: an older or newer destination
                            // copy conflicts.
                            let Some(id) = id else {
                                let e = EsError::action_validation(&["id is missing"]);
                                st.failures.push((e.status, failure_json(&svc.name, &h.id, &e)));
                                abort = true;
                                continue;
                            };
                            let cond = WriteCond { version_eq: version, ..Default::default() };
                            let req = WriteReq::Delete { id: id.clone(), routing, cond };
                            match groups.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, &svc)) {
                                Some((_, v)) => v.push((id, req)),
                                None => groups.push((svc, vec![(id, req)])),
                            }
                            continue;
                        }
                        // The destination request's conditions (op type and
                        // external version) are what its pipelines see, and
                        // the version / concurrency metadata they set apply
                        // to the write (ES `updateIndexRequestMetadata`; old
                        // review finding 72).
                        let mut cond = WriteCond { create: r.dest_op_create, ..Default::default() };
                        if let Some(v) = version {
                            cond.version = Some(v);
                            cond.version_type = Some(r.dest_version_type.filter(|t| *t != VersionType::Internal).unwrap_or(VersionType::External));
                        }
                        // Ingest pipelines (`dest.pipeline`, dest default/final).
                        match node.run_ingest(crate::ingest::IngestRequest { index: &dname, id: id.as_deref(), routing: routing.as_deref(), source: &src, pipeline: r.pipeline.as_deref(), cond: crate::ingest::WriteMeta::of(&cond) }) {
                            Ok(crate::ingest::IngestOutcome::Dropped { .. }) => {
                                st.noops += 1;
                                continue;
                            }
                            Ok(crate::ingest::IngestOutcome::Changed { index, id: new_id, routing: new_routing, source, cond: after, .. }) => {
                                after.apply(&mut cond);
                                if index != dname {
                                    svc = node.write_index(&index, true)?;
                                    dname = index;
                                }
                                if let Some(new_id) = new_id {
                                    id = Some(new_id);
                                }
                                routing = new_routing;
                                src = source;
                            }
                            Ok(crate::ingest::IngestOutcome::Unchanged) => {}
                            Err(e) => {
                                st.failures.push((e.status, failure_json(&svc.name, &key, &e)));
                                abort = true;
                                continue;
                            }
                        }
                        if let Err(e) = node.check_data_stream_write(&dname, r.dest_op_create, Some(&src)) {
                            st.failures.push((e.status, failure_json(&svc.name, &key, &e)));
                            abort = true;
                            continue;
                        }
                        h.id = id.clone().unwrap_or_default();
                        (svc, WriteReq::Index { id, source: src, routing, cond })
                    }
                };
                match groups.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, &svc)) {
                    Some((_, v)) => v.push((h.id.clone(), req)),
                    None => groups.push((svc, vec![(h.id.clone(), req)])),
                }
            }
            for (svc, items) in groups {
                if !touched.iter().any(|s| std::sync::Arc::ptr_eq(s, &svc)) {
                    touched.push(svc.clone());
                }
                let (ids, reqs): (Vec<String>, Vec<WriteReq>) = items.into_iter().unzip();
                // A by-query's writes go to the incarnation its hits were
                // read from: if the name now resolves to another index (the
                // source was deleted and recreated while the page was
                // buffered), every write of the group is a conflict.
                if r.kind != Kind::Reindex && node.get_index(&svc.name).is_none_or(|cur| cur.uuid != svc.uuid) {
                    let now = node.get_index(&svc.name).map(|c| c.uuid.clone()).unwrap_or_else(|| "_na_".into());
                    for id in &ids {
                        let e = EsError::version_conflict(&svc.name, &svc.uuid, id, format!("version conflict, the index [{}/{}] the document was read from was replaced by [{}/{now}]", svc.name, svc.uuid, svc.name));
                        st.version_conflicts += 1;
                        if !r.conflicts_proceed {
                            st.failures.push((e.status, failure_json(&svc.name, id, &e)));
                            abort = true;
                        }
                    }
                    continue;
                }
                let written = write_retrying(|reqs| svc.write(reqs, true), reqs, &mut st.retries_bulk, &|| task.is_cancelled());
                // The batch is acknowledged once its in-sync copies have it
                // (ES's bulk action); until then nothing in it counts.
                let nack: Option<EsError> = node.await_replicated(&svc).err();
                for (id, res) in ids.iter().zip(written) {
                    match res {
                        Ok(_) if nack.is_some() => {
                            if let Some(e) = &nack {
                                st.failures.push((e.status, failure_json(&svc.name, id, e)));
                            }
                            abort = true;
                        }
                        Ok(w) => match w.result {
                            "created" => st.created += 1,
                            "updated" => st.updated += 1,
                            // ES counts every delete item, found or not.
                            "deleted" | "not_found" => st.deleted += 1,
                            "noop" => st.noops += 1,
                            _ => {}
                        },
                        Err(e) if e.status == 409 => {
                            st.version_conflicts += 1;
                            if !r.conflicts_proceed {
                                st.failures.push((e.status, failure_json(&svc.name, id, &e)));
                                abort = true;
                            }
                        }
                        Err(e) => {
                            st.failures.push((e.status, failure_json(&svc.name, id, &e)));
                            abort = true;
                        }
                    }
                }
            }
            publish(run, 0);
            if abort {
                break;
            }
            if max_docs.map(|m| run.stats[idx].successful() >= m).unwrap_or(false) {
                break;
            }
            throttle(task, share, batch_start, docs, run, idx, publish);
        }
        Ok(abort)
    })();
    if let (Source::Remote { scroll_id: Some(id) }, Some(spec)) = (&source, &remote) {
        let _ = spec.request("DELETE", "/_search/scroll", Some(&json!({"scroll_id": [id]}).to_string()));
    }
    result
}

/// URL form of a sort for a remote `_search` (`field%3Aasc`).
fn sort_param(sort: &[SortSpec]) -> String {
    sort.iter().map(|s| format!("{}%3A{}", s.field, if s.descending { "desc" } else { "asc" })).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ES retry policy: rejected (429) bulk items are re-sent with backoff
    /// and counted in `retries.bulk`; other failures are not retried, and
    /// an item still rejected after the last retry fails.
    #[test]
    fn rejected_bulk_items_are_retried_with_backoff() {
        let item = |id: &str| WriteReq::Index { id: Some(id.into()), source: "{}".into(), routing: None, cond: Default::default() };
        let ok = || Ok(crate::shard::WriteResult { id: String::new(), version: 1, seq_no: 0, primary_term: 1, result: "created" });
        let mut calls = 0;
        let mut retries = 0;
        let out = write_retrying(|reqs| {
            calls += 1;
            reqs.iter().map(|r| match r {
                WriteReq::Index { id: Some(id), .. } if id == "busy" && calls < 3 => Err(EsError::new(429, "es_rejected_execution_exception", "busy")),
                WriteReq::Index { id: Some(id), .. } if id == "bad" => Err(EsError::new(400, "mapper_parsing_exception", "bad")),
                _ => ok(),
            }).collect()
        }, vec![item("a"), item("busy"), item("bad")], &mut retries, &|| false);
        assert_eq!(calls, 3);
        assert_eq!(retries, 2);
        assert!(out[0].is_ok() && out[1].is_ok(), "the rejected item was not retried to success");
        assert_eq!(out[2].as_ref().err().map(|e| e.status), Some(400));
        let mut retries = 0;
        let out = write_retrying(|reqs| reqs.iter().map(|_| Err(EsError::new(429, "es_rejected_execution_exception", "busy"))).collect(), vec![item("x")], &mut retries, &|| false);
        assert_eq!(retries, BULK_MAX_RETRIES as u64);
        assert_eq!(out[0].as_ref().err().map(|e| e.status), Some(429));
        // ES ExponentialBackoff delays (production constants: 500 ms + 10 ms * (floor(e^0.8i) - 1)).
        assert_eq!(bulk_retry_delay(0), Duration::from_millis(BULK_RETRY_INITIAL_MS));
        // A cancelled task stops waiting at once.
        let t0 = Instant::now();
        assert!(!sleep_unless_cancelled(Duration::from_secs(30), &|| true));
        assert!(t0.elapsed() < Duration::from_secs(1));
    }

    /// A source serde cannot hold (a lone surrogate) is copied verbatim,
    /// not replaced by its side-table ordinal.
    #[test]
    fn incomplete_source_pages_are_detected() {
        let ok = json!({"timed_out": false, "_shards": {"total": 2, "successful": 2, "failed": 0}, "hits": {"hits": []}});
        assert!(PageCheck::of(&ok).complete());
        let failed = json!({"timed_out": false, "_shards": {"total": 2, "successful": 1, "failed": 1, "failures": [
            {"shard": 1, "index": "i", "node": "n", "reason": {"type": "no_shard_available_action_exception", "reason": "gone"}}]}});
        let c = PageCheck::of(&failed);
        assert!(!c.complete());
        assert_eq!(c.failures, vec![(503, json!({"index": "i", "shard": 1, "node": "n", "status": 503, "reason": {"type": "no_shard_available_action_exception", "reason": "gone"}}))]);
        // A failed count without entries is still partial.
        let counted = json!({"_shards": {"total": 2, "successful": 1, "failed": 1}});
        assert_eq!(PageCheck::of(&counted).failures.len(), 1);
        let slow = json!({"timed_out": true, "_shards": {"total": 1, "successful": 1, "failed": 0}});
        let c = PageCheck::of(&slow);
        assert!(c.timed_out && !c.complete() && c.failures.is_empty());
    }

    #[test]
    fn response_status_is_es_bulk_by_scroll_status() {
        assert_eq!(response_status(&[], false), 200);
        assert_eq!(response_status(&[], true), 408);
        assert_eq!(response_status(&[(409, Value::Null)], true), 409);
        assert_eq!(response_status(&[(409, Value::Null), (503, Value::Null)], false), 503);
    }

    #[test]
    fn parse_hits_reports_page_completeness() {
        let (hits, check) = parse_hits(r#"{"timed_out":true,"_shards":{"total":1,"successful":1,"failed":0},"hits":{"hits":[{"_index":"i","_id":"1","_version":1,"_seq_no":0,"_source":{"a":1}}]}}"#).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(check.timed_out);
    }

    #[test]
    fn parse_hits_keeps_sources_serde_cannot_parse() {
        let out = r#"{"hits":{"hits":[{"_index":"s","_id":"1","_source":{"t":"a \ud800 b","n":1}},{"_index":"s","_id":"2","_source":{"t":"plain"}}]}}"#;
        let (hits, check) = parse_hits(out).unwrap();
        assert!(check.complete());
        assert_eq!(hits.len(), 2);
        let first = hits[0].source.as_deref().unwrap();
        assert!(first.contains(r#""t":"a \ud800 b""#) && first.contains(r#""n":1"#), "{first}");
        assert!(hits[1].source.as_deref().unwrap().contains(r#""t":"plain""#), "{:?}", hits[1].source);
    }

    #[test]
    fn positions_match_es() {
        let t = r#"{"source": {"index": "s"}, "dest": {"index": "d"}, "conflicts": "x"}"#;
        assert_eq!(pos_prefix(t, 0, "conflicts", true).0, "[1:65]");
        let t = r#"{"source": {"index": "s"}, "dest": {"index": "d"}, "foo": 1}"#;
        assert_eq!(pos_prefix(t, 0, "foo", false).0, "[1:52]");
    }

    #[test]
    fn rps_and_slices_validation() {
        assert_eq!(parse_rps("-1").unwrap(), -1.0);
        assert_eq!(parse_rps("2.5").unwrap(), 2.5);
        assert!(parse_rps("0").is_err() && parse_rps("abc").is_err());
        assert_eq!(parse_slices("auto").unwrap(), Slices::Auto);
        assert_eq!(parse_slices("0").unwrap_err().reason, "[slices] must be a positive integer or the string \"auto\", but was [0]");
        assert!(parse_slices("x").unwrap_err().caused_by.is_some());
    }

    #[test]
    fn time_values() {
        assert_eq!(time_value_str(0), "0s");
        assert_eq!(time_value_str(5), "5ms");
        assert_eq!(time_value_str(1100), "1.1s");
        assert_eq!(time_value_str(2000), "2s");
        let mut v = json!({"retries": {}, "throttled_millis": 5, "requests_per_second": -1.0, "throttled_until_millis": 0, "slices": [{"throttled_millis": 0, "throttled_until_millis": 0}]});
        with_human(&mut v);
        assert_eq!(v["throttled"], "5ms");
        assert_eq!(v["slices"][0]["throttled_until"], "0s");
    }
    fn put_docs(node: &Node, index: &str, docs: &[(&str, Value)]) {
        let svc = node.get_index(index).unwrap();
        let reqs = docs.iter().map(|(id, d)| WriteReq::Index { id: Some(id.to_string()), source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
        for r in svc.write(reqs, true) {
            r.unwrap();
        }
        svc.refresh().unwrap();
    }

    fn find(node: &Node, index: &str, query: Value) -> Vec<Value> {
        let svc = node.get_index(index).unwrap();
        svc.refresh().unwrap();
        let req = search::parse_request(&json!({"query": query, "size": 100, "version": true, "seq_no_primary_term": true}), &HashMap::new()).unwrap();
        let opts = ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = search::execute(&[search::Target::new(svc)], &req, &opts).unwrap();
        let v: Value = serde_json::from_str(&out.json).unwrap();
        v["hits"]["hits"].as_array().cloned().unwrap_or_default()
    }

    fn params(p: &[(&str, &str)]) -> HashMap<String, String> {
        p.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn by_query(node: &Node, kind: Kind, index: &str, body: Value, p: &[(&str, &str)]) -> Result<(u16, Value)> {
        let r = parse_by_query(kind, index, &body, &params(p))?;
        let task = node.tasks.register(kind.action(), None, true, false, Map::new());
        execute(node, &r, &task)
    }

    fn reindex(node: &Node, body: Value, p: &[(&str, &str)]) -> Result<(u16, Value)> {
        let r = parse_reindex(&body, &body.to_string(), &params(p))?;
        let task = node.tasks.register(Kind::Reindex.action(), None, true, false, Map::new());
        execute(node, &r, &task)
    }

    /// The same request run as a `wait_for_completion=false` job does: on
    /// another thread, under a stored task.
    fn by_query_async(node: &Node, kind: Kind, index: &str, body: Value, p: &[(&str, &str)]) -> Result<(u16, Value)> {
        let r = parse_by_query(kind, index, &body, &params(p))?;
        let task = node.tasks.register(kind.action(), None, true, true, Map::new());
        std::thread::scope(|sc| sc.spawn(|| execute(node, &r, &task)).join().unwrap())
    }

    fn too_many_slices(e: &EsError, n: u32, max: u32) {
        let text = e.to_json().to_string();
        assert_eq!(e.status, 400, "{text}");
        assert!(text.contains(&format!("The number of slices [{n}] is too large. It must be less than [{max}]. This limit can be set by changing the [index.max_slices_per_scroll] index level setting.")), "{text}");
    }

    /// Review finding 112: an explicit `slices` count (or manual
    /// `slice.max`) over `index.max_slices_per_scroll` fails as ES's sliced
    /// search does, before any slice state or destination exists, for
    /// update/delete-by-query (sync and async) and local reindex; `auto`,
    /// explicit counts within the limit and a manual slice still run.
    #[test]
    fn explicit_slices_are_bounded_by_max_slices_per_scroll() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("src", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}})).unwrap();
        put_docs(&node, "src", &[("1", json!({"n": 1})), ("2", json!({"n": 2})), ("3", json!({"n": 3})), ("4", json!({"n": 4}))]);
        let all = json!({"query": {"match_all": {}}});
        for kind in [Kind::UpdateByQuery, Kind::DeleteByQuery] {
            too_many_slices(&by_query(&node, kind, "src", all.clone(), &[("slices", "1025")]).unwrap_err(), 1025, 1024);
            too_many_slices(&by_query(&node, kind, "src", all.clone(), &[("slices", "4000000000")]).unwrap_err(), 4_000_000_000, 1024);
            too_many_slices(&by_query_async(&node, kind, "src", all.clone(), &[("slices", "1025")]).unwrap_err(), 1025, 1024);
            too_many_slices(&by_query(&node, kind, "src", json!({"query": {"match_all": {}}, "slice": {"id": 0, "max": 2000}}), &[]).unwrap_err(), 2000, 1024);
        }
        too_many_slices(&reindex(&node, json!({"source": {"index": "src"}, "dest": {"index": "dst"}}), &[("slices", "1025")]).unwrap_err(), 1025, 1024);
        assert!(node.get_index("dst").is_none(), "refused before the destination is created");
        assert_eq!(find(&node, "src", json!({"match_all": {}})).len(), 4, "nothing was deleted");
        // Controls (each refreshes, so the next reads current seq_nos).
        let refreshed = [("refresh", "true")];
        let (_, v) = by_query(&node, Kind::UpdateByQuery, "src", all.clone(), &[("slices", "3"), refreshed[0]]).unwrap();
        assert_eq!(v["updated"], 4, "{v}");
        let (_, v) = by_query_async(&node, Kind::UpdateByQuery, "src", all.clone(), &[("slices", "auto"), refreshed[0]]).unwrap();
        assert_eq!(v["updated"], 4, "{v}");
        let (_, v) = by_query(&node, Kind::UpdateByQuery, "src", json!({"query": {"match_all": {}}, "slice": {"id": 0, "max": 1024}}), &refreshed).unwrap();
        assert!(v["updated"].as_u64().unwrap() <= 4, "{v}");
        let (_, v) = reindex(&node, json!({"source": {"index": "src"}, "dest": {"index": "dst"}}), &[("slices", "2")]).unwrap();
        assert_eq!(v["created"], 4, "{v}");
    }

    /// Review finding 112: the slice state is admitted on the request
    /// breaker before it is built: with a small breaker and a slice count
    /// the index allows, the request is refused cleanly and nothing runs.
    #[test]
    fn slice_state_is_admitted_before_it_is_built() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("wide", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0, "index.max_slices_per_scroll": 100000}})).unwrap();
        put_docs(&node, "wide", &[("1", json!({"n": 1}))]);
        let b = crate::memory::Breakers::leak_for_tests(1 << 30);
        b.apply_settings(&|k| match k {
            "indices.breaker.request.limit" => Some("1mb".into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            "indices.breaker.total.use_real_memory" => Some("false".into()),
            _ => None,
        });
        let scope = crate::memory::RequestScope::new("by_query", b);
        for kind in [Kind::UpdateByQuery, Kind::DeleteByQuery] {
            let e = crate::memory::scoped(Some(scope.clone()), || by_query(&node, kind, "wide", json!({"query": {"match_all": {}}}), &[("slices", "50000")])).unwrap_err();
            assert_eq!((e.status, e.kind.as_str()), (429, "circuit_breaking_exception"), "{e:?}");
        }
        let e = crate::memory::scoped(Some(scope.clone()), || reindex(&node, json!({"source": {"index": "wide"}, "dest": {"index": "wdst"}}), &[("slices", "50000")])).unwrap_err();
        assert_eq!(e.status, 429, "{e:?}");
        assert!(node.get_index("wdst").is_none());
        let hits = find(&node, "wide", json!({"match_all": {}}));
        assert_eq!((hits.len(), hits[0]["_version"].as_u64()), (1, Some(1)), "nothing ran: {hits:?}");
    }

    /// Review finding 119: a document whose `_source` is not stored is
    /// never rewritten from an empty source: update-by-query (with and
    /// without a script) and reindex fail with ES's error and the indexed
    /// data is unchanged. Stored and synthetic sources still update.
    #[test]
    fn unavailable_source_fails_instead_of_writing_empty_documents() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("nosrc", &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0},
            "mappings": {"_source": {"enabled": false}, "properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}})).unwrap();
        put_docs(&node, "nosrc", &[("1", json!({"k": "a", "n": 1})), ("2", json!({"k": "b", "n": 2}))]);
        let intact = |node: &Node| {
            assert_eq!(find(node, "nosrc", json!({"term": {"k": "a"}})).len(), 1);
            let hits = find(node, "nosrc", json!({"exists": {"field": "n"}}));
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|h| h["_version"] == 1), "{hits:?}");
            assert!(find(node, "nosrc", json!({"exists": {"field": "extra"}})).is_empty());
        };
        intact(&node);
        let all = json!({"query": {"match_all": {}}});
        let e = by_query(&node, Kind::UpdateByQuery, "nosrc", all.clone(), &[]).unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.reason.starts_with("[nosrc][") && e.reason.ends_with("] didn't store _source"), "{}", e.reason);
        intact(&node);
        let scripted = json!({"query": {"match_all": {}}, "script": {"source": "ctx._source.extra = 1"}});
        let e = by_query(&node, Kind::UpdateByQuery, "nosrc", scripted.clone(), &[]).unwrap_err();
        assert!(e.reason.ends_with("didn't store _source"), "{}", e.reason);
        intact(&node);
        let e = reindex(&node, json!({"source": {"index": "nosrc"}, "dest": {"index": "copy"}}), &[]).unwrap_err();
        assert!(e.reason.ends_with("didn't store _source"), "{}", e.reason);
        assert!(node.get_index("copy").is_none_or(|_| find(&node, "copy", json!({"match_all": {}})).is_empty()), "nothing copied as {{}}");
        intact(&node);
        // Positive controls: stored and synthetic sources.
        for (name, mode) in [("stored", "stored"), ("synth", "synthetic")] {
            node.create_index(name, &json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0, "index.mapping.source.mode": mode},
                "mappings": {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}}})).unwrap();
            put_docs(&node, name, &[("1", json!({"k": "a", "n": 1})), ("2", json!({"k": "b", "n": 2}))]);
            let (_, v) = by_query(&node, Kind::UpdateByQuery, name, scripted.clone(), &[]).unwrap();
            assert_eq!(v["updated"], 2, "{name}: {v}");
            assert_eq!(find(&node, name, json!({"term": {"k": "a"}})).len(), 1, "{name}");
            let (_, v) = by_query(&node, Kind::UpdateByQuery, name, all.clone(), &[]).unwrap();
            assert_eq!(v["updated"], 2, "{name}: {v}");
            assert_eq!(find(&node, name, json!({"exists": {"field": "n"}})).len(), 2, "{name}");
        }
    }

    /// Review finding 120: a by-query page read from one incarnation of an
    /// index (paused after parsing) never mutates the index recreated under
    /// the same name before its writes, even when the replacement holds the
    /// same id at the same seq_no: the task reports a conflict and the
    /// replacement is unchanged. Covered for update- and delete-by-query,
    /// with matching and non-matching replacement seq_nos.
    #[test]
    fn buffered_hits_never_mutate_a_recreated_index() {
        for kind in [Kind::UpdateByQuery, Kind::DeleteByQuery] {
            for same_seq_no in [true, false] {
                let tmp = tempfile::tempdir().unwrap();
                let node = Node::open(tmp.path()).unwrap();
                let settings = json!({"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}});
                node.create_index("src", &settings).unwrap();
                put_docs(&node, "src", &[("1", json!({"m": "old"}))]);
                let old_uuid = node.get_index("src").unwrap().uuid.clone();
                let weak = std::sync::Arc::downgrade(&node);
                let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let f = fired.clone();
                node.set_by_query_page_hook(Some(std::sync::Arc::new(move || {
                    if f.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        return;
                    }
                    let node = weak.upgrade().unwrap();
                    node.delete_index("src").unwrap();
                    node.create_index("src", &settings).unwrap();
                    if !same_seq_no {
                        put_docs(&node, "src", &[("0", json!({"m": "filler"}))]);
                    }
                    put_docs(&node, "src", &[("1", json!({"m": "new"}))]);
                })));
                let mut body = json!({"query": {"match_all": {}}, "max_docs": 1});
                if kind == Kind::UpdateByQuery {
                    body["script"] = json!({"source": "ctx._source.touched = true"});
                }
                let (status, v) = by_query(&node, kind, "src", body, &[]).unwrap();
                node.set_by_query_page_hook(None);
                assert!(fired.load(std::sync::atomic::Ordering::SeqCst));
                let label = format!("{kind:?} same_seq_no={same_seq_no}");
                assert_ne!(node.get_index("src").unwrap().uuid, old_uuid, "{label}");
                assert_eq!(status, 409, "{label}: {v}");
                assert_eq!(v["version_conflicts"], 1, "{label}: {v}");
                assert_eq!(v["failures"][0]["cause"]["type"], "version_conflict_engine_exception", "{label}: {v}");
                assert_eq!((v["updated"].as_u64().unwrap_or(0), v["deleted"].as_u64().unwrap_or(0)), (0, 0), "{label}: {v}");
                let hits = find(&node, "src", json!({"ids": {"values": ["1"]}}));
                assert_eq!(hits.len(), 1, "{label}: the replacement document is kept");
                assert_eq!(hits[0]["_source"], json!({"m": "new"}), "{label}: and unchanged");
                assert_eq!(hits[0]["_version"], 1, "{label}");
            }
        }
    }
}
