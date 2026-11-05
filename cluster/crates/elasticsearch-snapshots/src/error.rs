//! ES-shaped errors. Every error carries the Elasticsearch exception `type`
//! (snake_case, as rendered in REST error bodies), the `reason` string and the
//! HTTP status ES would answer with.

use serde_json::{Value, json};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotError {
    /// ES exception type, e.g. `repository_exception`.
    pub kind: &'static str,
    pub reason: String,
    pub status: u16,
    /// Extra top-level fields ES adds to the error object (e.g. `index`, `resource.id`).
    pub extra: Vec<(String, Value)>,
}

pub type Result<T, E = SnapshotError> = std::result::Result<T, E>;

impl SnapshotError {
    pub fn new(kind: &'static str, status: u16, reason: impl Into<String>) -> Self {
        Self { kind, reason: reason.into(), status, extra: Vec::new() }
    }

    pub fn with_extra(mut self, key: &str, value: Value) -> Self {
        self.extra.push((key.to_string(), value));
        self
    }

    /// `RepositoryException`: `[repo] msg`, HTTP 500.
    pub fn repository(repo: &str, msg: impl AsRef<str>) -> Self {
        Self::new("repository_exception", 500, format!("[{repo}] {}", msg.as_ref()))
    }

    /// `RepositoryException` raised for bad user input (ES maps these to 500 too,
    /// but settings validation surfaces as 500 `repository_exception`).
    pub fn repository_bad_request(repo: &str, msg: impl AsRef<str>) -> Self {
        Self::new("repository_exception", 400, format!("[{repo}] {}", msg.as_ref()))
    }

    pub fn repository_missing(repo: &str) -> Self {
        Self::new("repository_missing_exception", 404, format!("[{repo}] missing"))
    }

    pub fn repository_verification(repo: &str, msg: impl AsRef<str>) -> Self {
        Self::new("repository_verification_exception", 500, format!("[{repo}] {}", msg.as_ref()))
    }

    pub fn snapshot_missing(repo: &str, snapshot: &str) -> Self {
        Self::new("snapshot_missing_exception", 404, format!("[{repo}:{snapshot}] is missing"))
    }

    pub fn invalid_snapshot_name(repo: &str, snapshot: &str, desc: impl AsRef<str>) -> Self {
        Self::new(
            "invalid_snapshot_name_exception",
            400,
            format!("[{repo}:{snapshot}] Invalid snapshot name [{snapshot}], {}", desc.as_ref()),
        )
    }

    /// `SnapshotException`: `[repo:snapshot] msg` (500).
    pub fn snapshot(repo: &str, snapshot: &str, msg: impl AsRef<str>) -> Self {
        Self::new("snapshot_exception", 500, format!("[{repo}:{snapshot}] {}", msg.as_ref()))
    }

    /// `SnapshotRestoreException`: `[repo:snapshot/uuid] msg` (500).
    pub fn snapshot_restore(snapshot_ref: &str, msg: impl AsRef<str>) -> Self {
        Self::new("snapshot_restore_exception", 500, format!("[{snapshot_ref}] {}", msg.as_ref()))
    }

    pub fn concurrent_snapshot_execution(repo: &str, snapshot: &str, msg: impl AsRef<str>) -> Self {
        Self::new(
            "concurrent_snapshot_execution_exception",
            503,
            format!("[{repo}:{snapshot}] {}", msg.as_ref()),
        )
    }

    pub fn illegal_argument(msg: impl Into<String>) -> Self {
        Self::new("illegal_argument_exception", 400, msg)
    }

    pub fn illegal_state(msg: impl Into<String>) -> Self {
        Self::new("illegal_state_exception", 500, msg)
    }

    pub fn validation(errors: &[String]) -> Self {
        let mut s = String::from("Validation Failed: ");
        for (i, e) in errors.iter().enumerate() {
            s.push_str(&format!("{}: {};", i + 1, e));
        }
        Self::new("action_request_validation_exception", 400, s)
    }

    pub fn index_not_found(index: &str) -> Self {
        Self::new("index_not_found_exception", 404, format!("no such index [{index}]"))
            .with_extra("resource.type", json!("index_or_alias"))
            .with_extra("resource.id", json!(index))
            .with_extra("index", json!(index))
    }

    pub fn resource_not_found(msg: impl Into<String>) -> Self {
        Self::new("resource_not_found_exception", 404, msg)
    }

    pub fn io(context: impl AsRef<str>, e: impl fmt::Display) -> Self {
        Self::new("i_o_exception", 500, format!("{}: {e}", context.as_ref()))
    }

    /// Blob not found (`NoSuchFileException` in ES).
    pub fn no_such_blob(name: &str) -> Self {
        Self::new("no_such_file_exception", 404, format!("[{name}] blob not found"))
    }

    /// Blob already exists (`FileAlreadyExistsException`).
    pub fn blob_exists(name: &str) -> Self {
        Self::new("file_already_exists_exception", 409, format!("blob [{name}] already exists, cannot overwrite"))
    }

    pub fn cancelled(msg: impl Into<String>) -> Self {
        Self::new("task_cancelled_exception", 400, msg)
    }

    pub fn is_not_found(&self) -> bool {
        self.kind == "no_such_file_exception"
    }

    pub fn is_already_exists(&self) -> bool {
        self.kind == "file_already_exists_exception"
    }

    /// The `{"error":{...},"status":N}` REST body.
    pub fn to_json(&self) -> Value {
        let mut cause = serde_json::Map::new();
        cause.insert("type".into(), json!(self.kind));
        cause.insert("reason".into(), json!(self.reason));
        for (k, v) in &self.extra {
            cause.insert(k.clone(), v.clone());
        }
        let mut err = cause.clone();
        let mut out = serde_json::Map::new();
        err.insert("root_cause".into(), Value::Array(vec![Value::Object(cause)]));
        // root_cause first, like ES.
        let mut ordered = serde_json::Map::new();
        ordered.insert("root_cause".into(), err.remove("root_cause").unwrap());
        for (k, v) in err {
            ordered.insert(k, v);
        }
        out.insert("error".into(), Value::Object(ordered));
        out.insert("status".into(), json!(self.status));
        Value::Object(out)
    }
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.reason)
    }
}

impl std::error::Error for SnapshotError {}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::NotFound => Self::new("no_such_file_exception", 404, e.to_string()),
            std::io::ErrorKind::AlreadyExists => Self::new("file_already_exists_exception", 409, e.to_string()),
            _ => Self::new("i_o_exception", 500, e.to_string()),
        }
    }
}

impl From<serde_json::Error> for SnapshotError {
    fn from(e: serde_json::Error) -> Self {
        Self::new("parse_exception", 400, format!("failed to parse: {e}"))
    }
}
