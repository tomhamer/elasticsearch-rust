//! Elasticsearch-shaped errors.
//!
//! Every failure that reaches a client is an [`EsError`]: an HTTP status,
//! an exception `type` (`index_not_found_exception`, ...), a `reason`, and
//! the extra metadata keys ES adds (`index`, `index_uuid`, `resource.id`,
//! ...). [`EsError::to_json`] renders the exact ES body:
//! `{"error":{"root_cause":[...],"type":..,"reason":..,...},"status":N}`.

use serde_json::{json, Map, Value};
use std::fmt;

/// Result alias used across the engine.
pub type Result<T> = std::result::Result<T, EsError>;

/// Sentinel `reason` rendered as JSON `null` (a Java exception without a message).
pub const NULL_REASON: &str = "\u{0}null";

/// The fields ES renders on every `circuit_breaking_exception`, in order,
/// with the value used when a producer named none. A refusal without a
/// breaker of its own is the node's memory budget (accounting): PERMANENT.
pub const BREAKING_FIELDS: [&str; 3] = ["bytes_wanted", "bytes_limit", "durability"];
fn breaking_default(field: &str) -> Value {
    if field == "durability" { json!("PERMANENT") } else { json!(0) }
}

/// An Elasticsearch exception.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EsError {
    pub status: u16,
    pub kind: String,
    pub reason: String,
    /// Extra metadata keys rendered next to `type`/`reason`, in order.
    pub extra: Vec<(String, Value)>,
    pub caused_by: Option<Box<EsError>>,
    /// Only admitted imported errors use this owner. Legacy engine producers
    /// retain their existing allocation contract; this is not a global allocator.
    #[serde(skip)]
    pub(crate) allocation_owner: Option<std::sync::Arc<ErrorAllocation>>,
}

pub(crate) struct ErrorAllocation {
    budget: &'static crate::memory::Breakers,
    _reservation: crate::memory::Reservation,
}
impl fmt::Debug for ErrorAllocation {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {f.debug_struct("ErrorAllocation").field("reservation",&self._reservation).finish_non_exhaustive()}
}
impl PartialEq for EsError {
    fn eq(&self, other:&Self)->bool { self.status==other.status && self.kind==other.kind && self.reason==other.reason && self.extra==other.extra && self.caused_by==other.caused_by }
}
/// A clone is always the same error. An imported error's copy is charged
/// to its breakers without tripping: the copy is made either way, and a
/// refusal must never stand in for the original (a remote 400 cloned per
/// document would otherwise reach the client as a 429 it retries forever).
/// Use [`EsError::try_clone_owned`] where a refusal can be propagated.
impl Clone for EsError {
    fn clone(&self)->Self {
        let mut copy=self.unowned_clone();
        if let Some(owner)=&self.allocation_owner {
            let bytes=self.allocation_bytes().unwrap_or(0) as u64;
            copy.attach_allocation(&ErrorAllocation::unchecked(owner.budget,bytes,"imported error clone"));
        }
        copy
    }
}
impl ErrorAllocation {
    /// Bytes of an allocation that has to happen (a clone, metadata on an
    /// existing error), charged without breaking.
    fn unchecked(budget:&'static crate::memory::Breakers,bytes:u64,_label:&str)->std::sync::Arc<Self> {
        std::sync::Arc::new(ErrorAllocation{budget,_reservation:budget.reserve_without_breaking(crate::memory::Kind::Request,bytes).into_result_owner()})
    }
}

impl fmt::Display for EsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]: {}", self.status, self.kind, self.reason)
    }
}

impl std::error::Error for EsError {}

/// Order `(index, failure)` shard failures as ES's `buildShardFailures`
/// does: by shard (`ShardId.compareTo`: shard number, then index name).
/// Within this engine every target is shard 0 of its index, so by index
/// name; a stable sort keeps one index's failures in phase order.
pub fn in_shard_order(failures: &mut [(String, EsError)]) {
    failures.sort_by(|a, b| a.0.cmp(&b.0));
}

impl EsError {
    pub fn new(status: u16, kind: &str, reason: impl Into<String>) -> Self {
        Self { status, kind: kind.to_string(), reason: reason.into(), extra: Vec::new(), caused_by: None, allocation_owner: None }
    }

    /// Copy the imported kind/reason before releasing a transport owner. The
    /// reservation belongs to the resulting error, including queue/cache clones.
    pub fn try_owned(status:u16,kind:&str,reason:&str)->Result<Self> {
        Self::try_owned_in(crate::memory::breakers(),status,kind,reason)
    }
    fn try_owned_in(budget:&'static crate::memory::Breakers,status:u16,kind:&str,reason:&str)->Result<Self>{
        let bytes=kind.len().checked_add(reason.len()).and_then(|n|n.checked_add(std::mem::size_of::<ErrorAllocation>()+2*std::mem::size_of::<usize>()))
            .ok_or_else(||Self::new(429,"circuit_breaking_exception","error allocation overflow"))?;
        let reservation=budget.reserve_request(bytes as u64,"imported error payload")?.into_result_owner();
        Ok(Self{status,kind:kind.to_owned(),reason:reason.to_owned(),extra:Vec::new(),caused_by:None,
            allocation_owner:Some(std::sync::Arc::new(ErrorAllocation{budget,_reservation:reservation}))})
    }
    fn json_allocation_bytes(v:&Value,depth:usize)->Option<usize>{
            if depth>128{return None}
            match v {
                Value::String(s)=>Some(s.len()),
                Value::Array(v)=>v.iter().try_fold(v.len().checked_mul(std::mem::size_of::<Value>())?,|n,v|n.checked_add(Self::json_allocation_bytes(v,depth+1)?)),
                // serde_json's insertion-ordered map owns an entry vector and
                // hash table. Charge spare capacity and table control storage
                // separately from every key/value's recursively counted data.
                Value::Object(v)=>v.iter().try_fold(v.len().checked_next_power_of_two()?.checked_mul(4*(std::mem::size_of::<(String,Value)>()+std::mem::size_of::<usize>())+16)?.checked_add(128)?,|n,(k,v)|n.checked_add(k.len())?.checked_add(Self::json_allocation_bytes(v,depth+1)?)),
                _=>Some(0),
            }
    }
    fn copy_json(value:&Value)->Value {
        match value {
            Value::String(s)=>Value::String(s.to_owned()),
            Value::Array(values)=>{let mut out=Vec::with_capacity(values.len());for value in values{out.push(Self::copy_json(value));}Value::Array(out)},
            Value::Object(values)=>{let mut out=Map::with_capacity(values.len());for(key,value)in values{out.insert(key.to_owned(),Self::copy_json(value));}Value::Object(out)},
            value=>value.clone(),
        }
    }
    fn copy_into_owner(value:&Self,owner:&std::sync::Arc<ErrorAllocation>)->Self {
        let mut extra=Vec::with_capacity(value.extra.len());
        for(key,value)in &value.extra{extra.push((key.to_owned(),Self::copy_json(value)));}
        Self{status:value.status,kind:value.kind.to_owned(),reason:value.reason.to_owned(),extra,
            caused_by:value.caused_by.as_ref().map(|c|Box::new(Self::copy_into_owner(c,owner))),allocation_owner:Some(owner.clone())}
    }
    fn allocation_bytes(&self)->Option<usize> {self.allocation_bytes_at(0)}
    fn allocation_bytes_at(&self,depth:usize)->Option<usize> {
        if depth>128{return None}
        let base=self.kind.len().checked_add(self.reason.len())?.checked_add(self.extra.capacity().max(self.extra.len()).checked_mul(2*std::mem::size_of::<(String,Value)>())?)?;
        let mut bytes=self.extra.iter().try_fold(base,|n,(k,v)|n.checked_add(k.len())?.checked_add(Self::json_allocation_bytes(v,0)?))?;
        if let Some(c)=&self.caused_by{bytes=bytes.checked_add(std::mem::size_of::<EsError>())?.checked_add(c.allocation_bytes_at(depth+1)?)?;}
        bytes.checked_add(std::mem::size_of::<ErrorAllocation>()+2*std::mem::size_of::<usize>())
    }
    /// Field-by-field copy with no allocation owner, causes included.
    fn unowned_clone(&self)->Self {
        Self{status:self.status,kind:self.kind.clone(),reason:self.reason.clone(),extra:self.extra.clone(),
            caused_by:self.caused_by.as_ref().map(|c|Box::new(c.unowned_clone())),allocation_owner:None}
    }
    fn attach_allocation(&mut self,owner:&std::sync::Arc<ErrorAllocation>){
        self.allocation_owner=Some(owner.clone());
        if let Some(c)=&mut self.caused_by{c.attach_allocation(owner)}
    }
    /// Fallible copy of an imported error. Callers at queue/cache boundaries
    /// should use this explicitly so refusal is propagated as a typed failure.
    pub fn try_clone_owned(&self)->Result<Self>{
        match &self.allocation_owner {Some(owner)=>self.owned_copy(owner.budget),None=>Ok(self.unowned_clone())}
    }
    fn owned_copy(&self,budget:&'static crate::memory::Breakers)->Result<Self>{
        let bytes=self.allocation_bytes().ok_or_else(||Self::new(429,"circuit_breaking_exception","error allocation overflow"))?;
        let reservation=budget.reserve_request(bytes as u64,"imported error clone")?.into_result_owner();
        let owner=std::sync::Arc::new(ErrorAllocation{budget,_reservation:reservation});
        Ok(Self::copy_into_owner(self,&owner))
    }

    /// The error an ES error object (`error` of a response body) stands
    /// for: `type`, `reason`, the extra keys and `caused_by` (`root_cause`
    /// and stack traces are derived again when rendered).
    pub fn from_error_json(status: u16, v: &Value) -> Self {
        let Some(o) = v.as_object() else {
            return Self::new(status, "exception", v.as_str().unwrap_or("request failed"));
        };
        let reason = match o.get("reason") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => NULL_REASON.to_string(),
            Some(other) => other.to_string(),
        };
        let mut e = Self::new(status, o.get("type").and_then(Value::as_str).unwrap_or("exception"), reason);
        for (k, val) in o {
            if !matches!(k.as_str(), "type" | "reason" | "root_cause" | "caused_by" | "stack_trace" | "suppressed") {
                e.extra.push((k.clone(), val.clone()));
            }
        }
        if let Some(s) = o.get("suppressed") {
            e.extra.push(("suppressed".into(), s.clone()));
        }
        if let Some(c) = o.get("caused_by") {
            e.caused_by = Some(Box::new(Self::from_error_json(status, c)));
        }
        e
    }

    /// The input value's construction belongs to the caller. Imported errors
    /// copy it into storage of their own, avoiding hidden spare capacity in
    /// caller-owned maps/arrays. Attaching metadata never changes the error:
    /// the bytes are charged without breaking (see `Clone`).
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Self {
        let value=value.into();
        if let Some(owner)=&self.allocation_owner {
            let next_capacity=self.extra.capacity().max(self.extra.len().saturating_add(1)).saturating_mul(2);
            let bytes=self.allocation_bytes().unwrap_or(0).saturating_add(key.len())
                .saturating_add(Self::json_allocation_bytes(&value,0).unwrap_or(0))
                .saturating_add(next_capacity.saturating_mul(std::mem::size_of::<(String,Value)>()));
            let next=ErrorAllocation::unchecked(owner.budget,bytes as u64,"imported error metadata");
            // Keep the previous owner until after the allocation/old-vector
            // overlap. Replacing an owner is never itself an admission credit.
            let previous=self.allocation_owner.clone();
            self.extra.push((key.to_string(),Self::copy_json(&value)));
            self.attach_allocation(&next);
            drop(previous);
        } else {self.extra.push((key.to_string(),value));}
        self
    }

    pub fn caused_by(mut self, cause: EsError) -> Self {
        if let Some(owner)=&self.allocation_owner {
            let bytes=self.allocation_bytes().unwrap_or(0).saturating_add(cause.allocation_bytes().unwrap_or(0))
                .saturating_add(std::mem::size_of::<EsError>());
            let next=ErrorAllocation::unchecked(owner.budget,bytes as u64,"imported error cause");
            let previous=self.allocation_owner.clone();
            self.caused_by=Some(Box::new(Self::copy_into_owner(&cause,&next)));
            self.attach_allocation(&next);
            drop(previous);
        } else {self.caused_by=Some(Box::new(cause));}
        self
    }

    fn body(&self, with_cause: bool) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("type".into(), Value::String(self.kind.clone()));
        m.insert("reason".into(), if self.reason == NULL_REASON { Value::Null } else { Value::String(self.reason.clone()) });
        let breaking = self.kind == "circuit_breaking_exception";
        for (k, v) in &self.extra {
            // `@`-prefixed extras are position hints (see `jsonpos`), never rendered.
            if k.starts_with('@') || (breaking && BREAKING_FIELDS.contains(&k.as_str())) {
                continue;
            }
            m.insert(k.clone(), v.clone());
        }
        if breaking {
            // ES `CircuitBreakingException.metadataToXContent`: always all
            // three, after the other metadata (0, 0 for a trip that names no
            // sizes, as ES's two-argument constructor does).
            for f in BREAKING_FIELDS {
                m.insert(f.to_string(), self.breaking_value(f));
            }
        }
        if with_cause {
            if let Some(c) = &self.caused_by {
                m.insert("caused_by".into(), Value::Object(c.body(true)));
            }
        }
        m
    }

    /// One of [`BREAKING_FIELDS`] of a `circuit_breaking_exception` as ES
    /// renders it: the producer's value, else ES's default.
    pub fn breaking_value(&self, field: &str) -> Value {
        self.extra.iter().find(|(k, _)| k == field).map(|(_, v)| v.clone()).unwrap_or_else(|| breaking_default(field))
    }

    /// Is `kind` an `ElasticsearchException` subclass (as opposed to a JDK /
    /// Jackson / x-content exception ES only wraps)?
    pub fn is_es_exception(kind: &str) -> bool {
        !matches!(
            kind,
            "illegal_argument_exception"
                | "illegal_state_exception"
                | "number_format_exception"
                | "x_content_parse_exception"
                | "x_content_e_o_f_exception"
                | "named_object_not_found_exception"
                | "json_parse_exception"
                | "json_e_o_f_exception"
                | "e_o_f_exception"
                | "buffer_underflow_exception"
                | "array_index_out_of_bounds_exception"
                | "index_out_of_bounds_exception"
                | "string_index_out_of_bounds_exception"
                | "negative_array_size_exception"
                | "unsupported_operation_exception"
                | "null_pointer_exception"
                | "class_cast_exception"
                | "io_exception"
                | "uncheckedio_exception"
                | "date_time_parse_exception"
                | "date_time_exception"
                | "arithmetic_exception"
                | "runtime_exception"
                | "exception"
                | "timestamp_error"
                | "security_exception_java"
                | "pattern_syntax_exception"
                | "too_complex_to_determinize_exception"
                | "parse_exception_lucene"
                | "i_o_exception"
                | "interrupted_exception"
                // Java exceptions raised by Painless scripts.
                | "wrong_method_type_exception"
                | "no_such_element_exception"
                | "unsupported_temporal_type_exception"
                | "concurrent_modification_exception"
                | "lambda_conversion_exception"
                | "missing_format_argument_exception"
                | "illegal_format_conversion_exception"
                | "unknown_format_conversion_exception"
                | "zone_rules_exception"
                | "painless_error"
        )
    }

    /// The innermost cause (ES `root_cause`): `ElasticsearchException.guessRootCauses`.
    /// An ES exception's root is its cause's root when the cause is an ES
    /// exception; an `XContentParseException`'s when the cause is an ES or
    /// x-content exception; anything else is its own root. Search phase
    /// failures report the shard failure.
    pub fn root(&self) -> &EsError {
        let mut e = self;
        if e.kind == "search_phase_execution_exception" {
            if let Some(c) = &e.caused_by {
                return c;
            }
        }
        while let Some(c) = &e.caused_by {
            let xcontent = matches!(e.kind.as_str(), "x_content_parse_exception" | "x_content_e_o_f_exception" | "named_object_not_found_exception");
            let cause_xcontent = matches!(c.kind.as_str(), "x_content_parse_exception" | "x_content_e_o_f_exception" | "named_object_not_found_exception");
            let descend = if xcontent { cause_xcontent || Self::is_es_exception(&c.kind) } else { Self::is_es_exception(&e.kind) && Self::is_es_exception(&c.kind) };
            if !descend {
                break;
            }
            e = c;
        }
        e
    }

    /// `SearchPhaseExecutionException.guessRootCauses`: a search phase failure
    /// carrying only its (grouped) `failed_shards` reports the root cause of
    /// each shard failure's `reason`, not itself.
    fn shard_failure_roots(&self, trace: bool) -> Option<Vec<Value>> {
        if self.kind != "search_phase_execution_exception" || self.caused_by.is_some() {
            return None;
        }
        let shards = self.extra.iter().find(|(k, _)| k == "failed_shards")?.1.as_array()?;
        let xc = |k: &str| matches!(k, "x_content_parse_exception" | "x_content_e_o_f_exception" | "named_object_not_found_exception");
        let mut out = Vec::new();
        for s in shards {
            let Some(mut cur) = s.get("reason").and_then(Value::as_object) else { continue };
            while let Some(c) = cur.get("caused_by").and_then(Value::as_object) {
                let (k, ck) = (cur.get("type").and_then(Value::as_str).unwrap_or(""), c.get("type").and_then(Value::as_str).unwrap_or(""));
                let descend = if xc(k) { xc(ck) || Self::is_es_exception(ck) } else { Self::is_es_exception(k) && Self::is_es_exception(ck) };
                if !descend {
                    break;
                }
                cur = c;
            }
            let mut r = cur.clone();
            if trace {
                r.insert("stack_trace".into(), Value::String(json_trace(cur)));
            }
            r.remove("caused_by");
            out.push(Value::Object(r));
        }
        if out.is_empty() { None } else { Some(out) }
    }

    /// The `error` object (without `status`).
    pub fn error_object(&self, stack_trace: bool) -> Value {
        let mut m = Map::new();
        let root = self.root();
        // A failed coordinator reduce (`phase: fetch`, empty reason) has no shard root cause.
        let roots = if self.kind == "search_phase_execution_exception" && self.reason.is_empty() {
            vec![]
        } else if let Some(shard_roots) = self.shard_failure_roots(stack_trace) {
            shard_roots
        } else {
            let mut r = root.body(false);
            if stack_trace {
                r.insert("stack_trace".into(), Value::String(json_trace(&root.body(true))));
            }
            vec![Value::Object(r)]
        };
        m.insert("root_cause".into(), Value::Array(roots));
        let mut own = Value::Object(self.body(true));
        if stack_trace {
            // `?error_trace`: every exception object carries its stack trace.
            add_traces(&mut own);
        }
        if let Value::Object(o) = own {
            for (k, v) in o {
                m.insert(k, v);
            }
        }
        Value::Object(m)
    }

    /// The `reason` object of a shard failure (`_shards.failures[].reason`).
    pub fn shard_reason(&self) -> Value {
        Value::Object(self.body(true))
    }

    /// Full ES error response body.
    pub fn to_json(&self) -> Value {
        json!({ "error": self.error_object(false), "status": self.status })
    }

    pub fn to_json_with_trace(&self, trace: bool) -> Value {
        json!({ "error": self.error_object(trace), "status": self.status })
    }

    // ---- constructors for the exceptions clients depend on ----

    pub fn index_not_found(index: &str) -> Self {
        Self::new(404, "index_not_found_exception", format!("no such index [{index}]"))
            .with("resource.type", "index_or_alias")
            .with("resource.id", index)
            .with("index_uuid", "_na_")
            .with("index", index)
    }

    pub fn index_already_exists(index: &str, uuid: &str) -> Self {
        Self::new(400, "resource_already_exists_exception", format!("index [{index}/{uuid}] already exists"))
            .with("index_uuid", uuid)
            .with("index", index)
    }

    pub fn invalid_index_name(index: &str, why: &str) -> Self {
        Self::new(400, "invalid_index_name_exception", format!("Invalid index name [{index}], {why}"))
            .with("index_uuid", "_na_")
            .with("index", index)
    }

    pub fn index_closed(index: &str, uuid: &str) -> Self {
        Self::new(400, "index_closed_exception", "closed").with("index_uuid", uuid).with("index", index)
    }

    pub fn illegal_argument(reason: impl Into<String>) -> Self {
        Self::new(400, "illegal_argument_exception", reason)
    }

    pub fn parsing(reason: impl Into<String>) -> Self {
        Self::new(400, "parsing_exception", reason)
    }

    pub fn parsing_at(reason: impl Into<String>, line: u64, col: u64) -> Self {
        Self::new(400, "parsing_exception", reason).with("line", line).with("col", col)
    }

    pub fn x_content_parse(reason: impl Into<String>) -> Self {
        Self::new(400, "x_content_parse_exception", reason)
    }

    pub fn json_parse(err: &serde_json::Error) -> Self {
        Self::new(400, "x_content_parse_exception", format!("[{}:{}] {}", err.line(), err.column(), err))
    }

    pub fn mapper_parsing(reason: impl Into<String>) -> Self {
        Self::new(400, "mapper_parsing_exception", reason)
    }

    pub fn document_parsing(reason: impl Into<String>) -> Self {
        Self::new(400, "document_parsing_exception", reason)
    }

    pub fn strict_dynamic(field: &str, path: &str) -> Self {
        let p = if path.is_empty() { "_doc".to_string() } else { path.to_string() };
        Self::new(
            400,
            "strict_dynamic_mapping_exception",
            format!("[1:1] mapping set to strict, dynamic introduction of [{field}] within [{p}] is not allowed"),
        )
    }

    pub fn version_conflict(index: &str, uuid: &str, id: &str, reason: String) -> Self {
        Self::new(409, "version_conflict_engine_exception", format!("[{id}]: {reason}"))
            .with("index_uuid", uuid)
            .with("shard", "0")
            .with("index", index)
    }

    pub fn document_missing(index: &str, uuid: &str, id: &str) -> Self {
        Self::new(404, "document_missing_exception", format!("[{id}]: document missing"))
            .with("index_uuid", uuid)
            .with("shard", "0")
            .with("index", index)
    }

    pub fn resource_not_found(reason: impl Into<String>) -> Self {
        Self::new(404, "resource_not_found_exception", reason)
    }

    pub fn action_validation(errors: &[&str]) -> Self {
        let mut reason = String::from("Validation Failed: ");
        for (i, e) in errors.iter().enumerate() {
            reason.push_str(&format!("{}: {};", i + 1, e));
        }
        Self::new(400, "action_request_validation_exception", reason)
    }

    pub fn query_shard(index: &str, uuid: &str, reason: impl Into<String>) -> Self {
        Self::new(400, "query_shard_exception", reason).with("index_uuid", uuid).with("index", index)
    }

    /// Wrap a shard-level failure the way ES reports a search that failed on
    /// every shard.
    pub fn search_phase(index: &str, cause: EsError) -> Self {
        let status = cause.status;
        let reason_obj = Value::Object(cause.body(true));
        Self::new(status, "search_phase_execution_exception", "all shards failed")
            .with("phase", "query")
            .with("grouped", true)
            .with(
                "failed_shards",
                json!([{ "shard": 0, "index": index, "node": crate::node::NODE_ID, "reason": reason_obj }]),
            )
            .caused_by_opt(match cause.caused_by.as_deref() {
                Some(c) if c.kind == "kql_parsing_exception" => Some(c.clone()),
                // ES `deduplicateCause`: a shard's own `QueryShardException` is
                // not repeated as the top-level cause.
                _ if cause.kind == "query_shard_exception" => None,
                _ => Some(cause),
            })
    }

    /// [`EsError::search_phase`] over several failed shards (`(index,
    /// cause)`): ES's status is the first failure's, or the last 5xx one's,
    /// in shard order (whatever order the phases failed them in).
    pub fn search_phase_all(failures: &[(String, EsError)]) -> Self {
        let mut failures = failures.to_vec();
        in_shard_order(&mut failures);
        let failures = &failures[..];
        let Some((index, first)) = failures.first() else { return Self::internal("all shards failed") };
        if failures.len() == 1 {
            return Self::search_phase(index, first.clone());
        }
        let status = failures.iter().rev().find(|(_, e)| e.status >= 500).map_or(first.status, |(_, e)| e.status);
        let shards: Vec<Value> = failures.iter().map(|(ix, e)| json!({ "shard": 0, "index": ix, "node": crate::node::NODE_ID, "reason": Value::Object(e.body(true)) })).collect();
        Self::new(status, "search_phase_execution_exception", "all shards failed")
            .with("phase", "query")
            .with("grouped", true)
            .with("failed_shards", Value::Array(shards))
            .caused_by(first.clone())
    }

    fn caused_by_opt(self, cause: Option<EsError>) -> Self {
        match cause {
            Some(c) => self.caused_by(c),
            None => self,
        }
    }

    pub fn too_many_buckets(limit: usize) -> Self {
        Self::new(
            400,
            "too_many_buckets_exception",
            format!(
                "Trying to create too many buckets. Must be less than or equal to: [{limit}] but this number of buckets was exceeded. This limit can be set by changing the [search.max_buckets] cluster level setting."
            ),
        )
        .with("max_buckets", limit)
    }

    pub fn unsupported(what: &str) -> Self {
        Self::illegal_argument(format!("{what} is not supported by this server yet"))
    }

    pub fn internal(reason: impl Into<String>) -> Self {
        Self::new(500, "exception", reason)
    }

    pub fn alias_missing(alias: &str) -> Self {
        Self::new(404, "aliases_not_found_exception", format!("aliases [{alias}] missing")).with("resource.type", "aliases").with("resource.id", alias)
    }

    pub fn search_context_missing(id: &str) -> Self {
        Self::new(404, "search_context_missing_exception", format!("No search context found for id [{id}]"))
    }

    /// ES `IndexShardClosedException`: an operation reached a shard copy
    /// that was closed (fenced) while the caller still held its handle.
    pub fn shard_closed(index: &str, uuid: &str, shard: u32) -> Self {
        Self::new(500, "index_shard_closed_exception", format!("CurrentState[CLOSED] Closed"))
            .with("index_uuid", uuid)
            .with("shard", shard.to_string())
            .with("index", index)
    }

    /// Whether this is the typed "shard copy already closed" error
    /// ([`EsError::shard_closed`]).
    pub fn is_already_closed(&self) -> bool {
        self.kind == "index_shard_closed_exception"
    }

    /// ES `ShardNotInPrimaryModeException`: a primary-only operation
    /// (index / delete / update) reached a replica copy.
    pub fn shard_not_primary(index: &str, uuid: &str, shard: u32) -> Self {
        Self::new(500, "shard_not_in_primary_mode_exception", "CurrentState[STARTED] shard is not in primary mode")
            .with("index_uuid", uuid)
            .with("shard", shard.to_string())
            .with("index", index)
    }

    /// ES `ShardLockObtainFailedException`: the shard directory is already
    /// held by another open copy.
    pub fn shard_lock_failed(index: &str, uuid: &str, shard: u32, why: &str) -> Self {
        Self::new(500, "shard_lock_obtain_failed_exception", format!("[{index}][{shard}]: obtaining shard lock for [{why}] failed, lock already held"))
            .with("index_uuid", uuid)
            .with("shard", shard.to_string())
            .with("index", index)
    }

    /// ES `ShardNotFoundException`: the shard is not allocated on this node.
    pub fn shard_not_found(index: &str, uuid: &str, shard: u32) -> Self {
        Self::new(404, "shard_not_found_exception", format!("no such shard"))
            .with("index_uuid", uuid)
            .with("shard", shard.to_string())
            .with("index", index)
    }
}

/// A Java-style stack trace of an exception object and its `caused_by`
/// chain (frames are not tracked; ES clients match on the
/// `Caused by: <class>: <message>` lines).
fn json_trace(o: &Map<String, Value>) -> String {
    let mut out = String::new();
    let mut cur = Some(o);
    let mut first = true;
    while let Some(e) = cur {
        let kind = e.get("type").and_then(Value::as_str).unwrap_or("exception");
        let reason = e.get("reason").and_then(Value::as_str).unwrap_or("null");
        let class = java_class(kind);
        if !first {
            out.push_str("\nCaused by: ");
        }
        first = false;
        out.push_str(&format!("{class}: {reason}\n\tat {class}.<init>(Unknown Source)"));
        cur = e.get("caused_by").and_then(Value::as_object);
    }
    out
}

/// Add `stack_trace` to every exception object (with `type` and `reason`)
/// in an error body: itself, its `caused_by` chain and shard failure reasons.
fn add_traces(v: &mut Value) {
    match v {
        Value::Object(m) => {
            if m.get("type").is_some_and(Value::is_string) && m.contains_key("reason") && !m.contains_key("stack_trace") {
                let t = json_trace(m);
                m.insert("stack_trace".into(), Value::String(t));
            }
            for (k, x) in m.iter_mut() {
                if k != "stack_trace" {
                    add_traces(x);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(add_traces),
        _ => {}
    }
}

fn java_class(kind: &str) -> String { JavaClass(kind).to_string() }

/// Allocation-free exception class formatting for admitted HTTP renderers.
pub struct JavaClass<'a>(pub &'a str);
impl fmt::Display for JavaClass<'_> {
 fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
    let kind=self.0;
    let jdk = match kind {
        "number_format_exception" => Some("java.lang.NumberFormatException"),
        "illegal_argument_exception" => Some("java.lang.IllegalArgumentException"),
        "illegal_state_exception" => Some("java.lang.IllegalStateException"),
        "null_pointer_exception" => Some("java.lang.NullPointerException"),
        "class_cast_exception" => Some("java.lang.ClassCastException"),
        "arithmetic_exception" => Some("java.lang.ArithmeticException"),
        "unsupported_operation_exception" => Some("java.lang.UnsupportedOperationException"),
        "array_index_out_of_bounds_exception" => Some("java.lang.ArrayIndexOutOfBoundsException"),
        "index_out_of_bounds_exception" => Some("java.lang.IndexOutOfBoundsException"),
        "string_index_out_of_bounds_exception" => Some("java.lang.StringIndexOutOfBoundsException"),
        "negative_array_size_exception" => Some("java.lang.NegativeArraySizeException"),
        "runtime_exception" => Some("java.lang.RuntimeException"),
        "exception" => Some("java.lang.Exception"),
        "io_exception" => Some("java.io.IOException"),
        "e_o_f_exception" => Some("java.io.EOFException"),
        "date_time_parse_exception" => Some("java.time.format.DateTimeParseException"),
        "date_time_exception" => Some("java.time.DateTimeException"),
        "x_content_parse_exception" => Some("org.elasticsearch.xcontent.XContentParseException"),
        "json_parse_exception" => Some("com.fasterxml.jackson.core.JsonParseException"),
        "status_exception" => Some("org.elasticsearch.ElasticsearchStatusException"),
        "parse_exception" => Some("org.elasticsearch.ElasticsearchParseException"),
        _ => None,
    };
    if let Some(j) = jdk { return out.write_str(j); }
    out.write_str("org.elasticsearch.")?;
    for part in kind.split('_') {
        let mut chars=part.chars();
        if let Some(first)=chars.next() {
            for upper in first.to_uppercase(){ write!(out,"{upper}")?; }
            out.write_str(chars.as_str())?;
        }
    }
    Ok(())
 }
}

impl From<rustic_core::RusticError> for EsError {
    fn from(e: rustic_core::RusticError) -> Self {
        match e {
            // Lucene `IndexSearcher.TooManyNestedClauses` / `TooManyClauses`.
            rustic_core::RusticError::TooManyClauses { max,nested } => EsError::new(400, if nested { "too_many_nested_clauses" } else { "too_many_clauses" },
                if nested { format!("Query contains too many nested clauses; maxClauseCount is set to {max}") } else { format!("maxClauseCount is set to {max}") }),
            rustic_core::RusticError::IntervalExpansion(reason) => EsError::new(500,"illegal_state_exception",reason),
            rustic_core::RusticError::Memory(reason)=>crate::memory::core_refusal(reason),
            other => EsError::internal(other.to_string()),
        }
    }
}

impl From<std::io::Error> for EsError {
    fn from(e: std::io::Error) -> Self {
        EsError::internal(format!("io: {e}"))
    }
}

impl From<serde_json::Error> for EsError {
    fn from(e: serde_json::Error) -> Self {
        EsError::json_parse(&e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_not_found_shape() {
        let e = EsError::index_not_found("foo");
        assert_eq!(
            e.to_json(),
            json!({
                "error": {
                    "root_cause": [{
                        "type": "index_not_found_exception",
                        "reason": "no such index [foo]",
                        "resource.type": "index_or_alias",
                        "resource.id": "foo",
                        "index_uuid": "_na_",
                        "index": "foo"
                    }],
                    "type": "index_not_found_exception",
                    "reason": "no such index [foo]",
                    "resource.type": "index_or_alias",
                    "resource.id": "foo",
                    "index_uuid": "_na_",
                    "index": "foo"
                },
                "status": 404
            })
        );
    }

    #[test]
    fn caused_by_and_root_cause() {
        let inner = EsError::illegal_argument("bad value");
        let e = EsError::mapper_parsing("failed to parse").caused_by(inner);
        let j = e.to_json();
        assert_eq!(j["error"]["root_cause"][0]["type"], "mapper_parsing_exception");
        assert_eq!(j["error"]["caused_by"]["type"], "illegal_argument_exception");
        let s = EsError::search_phase("idx", EsError::query_shard("idx", "u", "failed to create query"));
        let j = s.to_json();
        assert_eq!(j["error"]["root_cause"][0]["type"], "query_shard_exception");
        assert_eq!(j["error"]["type"], "search_phase_execution_exception");
        assert_eq!(j["error"]["failed_shards"][0]["reason"]["type"], "query_shard_exception");
        assert_eq!(j["status"], 400);
    }

    #[test]
    fn validation_reason() {
        let e = EsError::action_validation(&["index is missing", "id is missing"]);
        assert_eq!(e.reason, "Validation Failed: 1: index is missing;2: id is missing;");
    }
}

#[cfg(test)]
mod from_error_json_tests {
    use super::*;

    #[test]
    fn error_object_round_trips() {
        let e = EsError::new(400, "search_phase_execution_exception", "all shards failed")
            .with("phase", "query")
            .with("failed_shards", json!([{"shard": 0, "index": "i", "reason": {"type": "script_exception", "reason": "boom"}}]))
            .caused_by(EsError::new(400, "script_exception", "boom"));
        let back = EsError::from_error_json(400, &e.error_object(false));
        assert_eq!(back.to_json(), e.to_json());
    }

    /// Combined review M28: ES's all-shards-failed status and cause come
    /// from the failures in shard order, not in the order the phases
    /// failed them: `qa` tripping in its aggregation after `qb` failed to
    /// parse is still a 429, caused by the trip.
    #[test]
    fn all_shards_failed_takes_the_first_failure_in_shard_order() {
        let parse = EsError::new(400, "query_shard_exception", "failed to create query");
        let trip = EsError::new(429, "circuit_breaking_exception", "[request] Data too large");
        let e = EsError::search_phase_all(&[("qb".into(), parse.clone()), ("qa".into(), trip.clone())]);
        assert_eq!(e.status, 429);
        assert_eq!(e.caused_by.as_deref().map(|c| c.kind.as_str()), Some("circuit_breaking_exception"));
        let v = e.to_json();
        let order: Vec<&str> = v["error"]["failed_shards"].as_array().unwrap().iter().filter_map(|f| f["index"].as_str()).collect();
        assert_eq!(order, ["qa", "qb"]);
        // The other way round, and a 5xx anywhere wins.
        assert_eq!(EsError::search_phase_all(&[("qb".into(), trip.clone()), ("qa".into(), parse.clone())]).status, 400);
        let internal = EsError::new(500, "exception", "boom");
        assert_eq!(EsError::search_phase_all(&[("qc".into(), internal), ("qa".into(), trip)]).status, 500);
    }
}

#[cfg(test)]
mod breaking_shape_tests {
    use super::*;
    /// Review finding 10: every circuit_breaking_exception renders ES's
    /// bytes_wanted, bytes_limit and durability after its other metadata,
    /// whatever produced it (a breaker, an allocation refusal, core memory,
    /// an import that named no sizes).
    #[test]
    fn every_breaking_error_has_es_fields_in_es_order() {
        let fields = |e: &EsError| { let v = e.to_json(); (v["error"]["bytes_wanted"].clone(), v["error"]["bytes_limit"].clone(), v["error"]["durability"].clone()) };
        let breaker = crate::memory::breaking("[request] Data too large".into(), 10, 5, crate::memory::Durability::Transient);
        assert_eq!(fields(&breaker), (json!(10), json!(5), json!("TRANSIENT")));
        let core = EsError::from(rustic_core::RusticError::Memory("[core] [x] needs 9 bytes".into()));
        assert_eq!((core.status, fields(&core)), (429, (json!(9), json!(0), json!("PERMANENT"))));
        // An admission refusal against its limit: the sizes it names, TRANSIENT.
        let admission = EsError::from(rustic_core::RusticError::Memory("[core] [mutable row plane] needs 4096 bytes, retained 8589930000, limit 8589934592".into()));
        assert_eq!((admission.status, admission.kind.as_str(), fields(&admission)), (429, "circuit_breaking_exception", (json!(4096), json!(8589934592u64), json!("TRANSIENT"))));
        let share = EsError::from(rustic_core::RusticError::Memory("[core] [merge] needs 7 bytes, background share 50 of limit 100".into()));
        assert_eq!(fields(&share), (json!(7), json!(100), json!("TRANSIENT")));
        let bare = EsError::new(429, "circuit_breaking_exception", "allocation refused").with("index", "i");
        assert_eq!(fields(&bare), (json!(0), json!(0), json!("PERMANENT")));
        let keys: Vec<String> = bare.to_json()["error"].as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, ["root_cause", "type", "reason", "index", "bytes_wanted", "bytes_limit", "durability"]);
        assert_eq!(bare.to_json()["error"]["root_cause"][0]["durability"], "PERMANENT");
        let wrapped = EsError::search_phase("i", breaker.clone());
        assert_eq!(wrapped.to_json()["error"]["caused_by"]["bytes_wanted"], 10);
        assert!(EsError::new(400, "parsing_exception", "x").to_json()["error"].get("bytes_wanted").is_none());
    }

    #[test]
    fn native_refusals_name_the_accounting_breaker() {
        let b = crate::memory::Breakers::leak_for_tests(1000);
        let held = b.reserve_native(900, "held").unwrap();
        let _query = b.reserve_without_breaking(crate::memory::Kind::Request, 90);
        let error = b.reserve_native(20, "control output").unwrap_err();
        assert!(error.reason.starts_with("[accounting] Data too large, data for [control output] would be ["), "{}", error.reason);
        let v = error.to_json();
        assert_eq!((&v["error"]["bytes_limit"], &v["error"]["durability"]), (&json!(1000), &json!("PERMANENT")));
        drop(held);
    }
}

#[cfg(test)]
mod clause_limit_tests {
    use super::*;
    #[test]
    fn rewrite_clause_errors_remain_request_errors() {
        for (nested,kind) in [(false,"too_many_clauses"),(true,"too_many_nested_clauses")] {
            let error = EsError::from(rustic_core::RusticError::TooManyClauses { max:1024,nested });
            assert_eq!(error.status,400);
            assert_eq!(error.kind,kind);
            assert_eq!(error.reason,if nested {"Query contains too many nested clauses; maxClauseCount is set to 1024"} else {"maxClauseCount is set to 1024"});
        }
    }
}

#[cfg(test)]
mod imported_error_ownership_tests {
    use super::*;
    fn budget()->&'static crate::memory::Breakers {
        let b=crate::memory::Breakers::leak_for_tests(1<<20);
        b.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        b
    }
    #[test]
    fn imported_error_clones_own_copies_and_refusal_keeps_original() {
        let b=budget();
        let error=EsError::try_owned_in(b,409,"version_conflict_engine_exception","retained remote reason").unwrap();
        let original=b.request.used();assert!(original>0);
        let copy=error.try_clone_owned().unwrap();
        assert_eq!(copy,error);assert!(b.request.used()>original);
        drop(copy);assert_eq!(b.request.used(),original);
        b.apply_settings(&|key|match key {
            "indices.breaker.total.use_real_memory"=>Some("false".into()),
            "indices.breaker.request.limit"=>Some(format!("{original}b")),
            _=>None,
        });
        let refusal=error.try_clone_owned().unwrap_err();
        assert_eq!(refusal.status,429);assert_eq!(error.reason,"retained remote reason");
        assert_eq!(b.request.used(),original);
        drop(error);assert_eq!(b.request.used(),0);
    }
    /// Review finding 9: an imported request error cloned (once per bulk
    /// item), given metadata or a cause while the request breaker is full
    /// stays that error; nothing turns it into a 429. The clones' bytes are
    /// still charged, and released with them.
    #[test]
    fn imported_error_clone_and_attach_never_become_a_breaker_error() {
        let b=budget();
        let error=EsError::try_owned_in(b,400,"mapper_parsing_exception","failed to parse field [f]").unwrap();
        let original=b.request.used();assert!(original>0);
        b.apply_settings(&|key|match key {
            "indices.breaker.total.use_real_memory"=>Some("false".into()),
            "indices.breaker.request.limit"=>Some(format!("{original}b")),
            _=>None,
        });
        let copies:Vec<EsError>=(0..3).map(|_|error.clone()).collect();
        for copy in &copies {
            assert_eq!((copy.status,copy.kind.as_str(),copy.reason.as_str()),(400,"mapper_parsing_exception","failed to parse field [f]"));
        }
        assert!(b.request.used()>original,"clones are charged");
        drop(copies);assert_eq!(b.request.used(),original);
        let decorated=error.clone().with("index","i").caused_by(EsError::new(400,"illegal_argument_exception","bad value"));
        assert_eq!(decorated.status,400);assert_eq!(decorated.kind,"mapper_parsing_exception");
        assert_eq!(decorated.extra,vec![("index".to_string(),json!("i"))]);
        assert_eq!(decorated.caused_by.as_ref().unwrap().kind,"illegal_argument_exception");
        assert_eq!(error.try_clone_owned().unwrap_err().status,429,"the explicit fallible copy still refuses");
        drop(decorated);drop(error);assert_eq!(b.request.used(),0);
    }
    #[test]
    fn imported_metadata_causes_and_wire_preserve_values_and_owners() {
        let b=budget();
        let expected=EsError::new(400,"outer","reason")
            .with("field",json!({"items":["one","two"]}))
            .caused_by(EsError::new(500,"inner","cause"));
        let error=EsError::try_owned_in(b,400,"outer","reason").unwrap()
            .with("field",json!({"items":["one","two"]}))
            .caused_by(EsError::new(500,"inner","cause"));
        assert_eq!(error,expected);
        assert_eq!(serde_json::to_vec(&error).unwrap(),serde_json::to_vec(&expected).unwrap());
        let cause=error.caused_by.as_ref().unwrap().try_clone_owned().unwrap();
        drop(error);assert!(b.request.used()>0);assert_eq!(cause.reason,"cause");
        drop(cause);assert_eq!(b.request.used(),0);
    }
}
