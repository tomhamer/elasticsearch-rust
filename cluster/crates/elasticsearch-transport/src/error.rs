//! Transport errors.
//!
//! Two layers, mirroring ES:
//! - [`TransportError`] is what a caller of `send_request` sees: the local
//!   failure modes of the transport itself (connect, timeout, disconnect,
//!   unknown action, serialization) plus [`TransportError::Remote`] when the
//!   remote handler returned a failure.
//! - [`RemoteError`] is the serializable failure a handler returns. It carries
//!   ES's `{type, reason, status, caused_by}` so the HTTP layer can render a
//!   failure that happened on another node exactly as ES would (ES unwraps its
//!   `RemoteTransportException` before rendering, so we never wrap).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;

/// A failure produced by a remote (or local) action handler.
#[derive(Clone, Debug)]
pub struct RemoteError(std::sync::Arc<RemoteErrorOwner>);
#[derive(Debug)]
pub(crate) struct RemoteErrorOwner {
    data: RemoteErrorData,
    owner: Option<std::sync::Arc<crate::control::Owner>>,
}
/// Immutable ES failure fields. The transport error and all cause clones retain
/// the same decoded allocation reservation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteErrorData {
    /// ES exception type in snake case, e.g. `index_not_found_exception`.
    pub error_type: String,
    pub reason: String,
    /// HTTP status the REST layer should answer with.
    pub status: u16,
    /// Extra top-level fields ES prints next to `type`/`reason`
    /// (`index`, `index_uuid`, `shard`, `resource.type`, ...).
    pub metadata: BTreeMap<String, String>,
    pub caused_by: Option<Box<RemoteError>>,
}

impl std::ops::Deref for RemoteError {
    type Target = RemoteErrorData;
    fn deref(&self) -> &Self::Target { &self.0.data }
}
impl PartialEq for RemoteError { fn eq(&self, other: &Self) -> bool { self.0.data == other.0.data } }
impl Eq for RemoteError {}
impl Serialize for RemoteError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok,S::Error> { self.0.data.serialize(s) }
}
impl<'de> Deserialize<'de> for RemoteError {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self,D::Error> {
        RemoteErrorData::deserialize(d).map(|data|Self(std::sync::Arc::new(RemoteErrorOwner { data, owner: None })))
    }
}

/// The fields ES renders on every `circuit_breaking_exception`, in order
/// (`CircuitBreakingException.metadataToXContent`). They cross the wire as
/// metadata strings and render as ES's numbers / enum name.
pub const BREAKING_FIELDS: [&str; 3] = ["bytes_wanted", "bytes_limit", "durability"];
const BREAKING: &str = "circuit_breaking_exception";

impl RemoteErrorData {
    /// A breaking field's value as ES renders it: the carried metadata,
    /// else ES's default (0 / PERMANENT).
    pub fn breaking_value(&self, field: &str) -> Value {
        match (field, self.metadata.get(field)) {
            ("durability", Some(v)) => Value::String(v.clone()),
            ("durability", None) => Value::String("PERMANENT".into()),
            (_, v) => Value::from(v.and_then(|v| v.parse::<u64>().ok()).unwrap_or(0)),
        }
    }
    fn hides(&self, key: &str) -> bool { self.error_type == BREAKING && BREAKING_FIELDS.contains(&key) }
}

struct JsonView<'a>{error:&'a RemoteError,root:bool}
impl Serialize for JsonView<'_>{
    fn serialize<S:serde::Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{
        use serde::ser::SerializeMap;
        let e=self.error;let mut out=serializer.serialize_map(None)?;
        if !e.metadata.contains_key("type"){out.serialize_entry("type",&e.error_type)?;}
        if !e.metadata.contains_key("reason"){out.serialize_entry("reason",&e.reason)?;}
        for(k,v)in &e.metadata{
            if (k=="caused_by"&&e.caused_by.is_some())||(k=="root_cause"&&self.root)||e.hides(k){continue}
            out.serialize_entry(k,v)?;
        }
        if e.error_type==BREAKING{for field in BREAKING_FIELDS{out.serialize_entry(field,&e.breaking_value(field))?;}}
        if let Some(c)=&e.caused_by{out.serialize_entry("caused_by",&JsonView{error:c,root:false})?;}
        if self.root{out.serialize_entry("root_cause",&[JsonView{error:e.root_cause(),root:false}])?;}
        out.end()
    }
}

impl RemoteError {
    /// Borrowed REST error object, including ES root_cause, without allocating
    /// intermediate JSON trees. Counted serializers can admit the output first.
    pub fn json_with_root(&self)->impl Serialize+'_ {JsonView{error:self,root:true}}

    pub fn new(error_type: impl Into<String>, reason: impl Into<String>, status: u16) -> Self {
        Self(std::sync::Arc::new(RemoteErrorOwner { data: RemoteErrorData { error_type: error_type.into(), reason: reason.into(), status, metadata: BTreeMap::new(), caused_by: None }, owner: None }))
    }

    /// Shorthand for a 500 `exception`-style failure.
    pub fn internal(reason: impl Into<String>) -> Self {
        RemoteError::new("exception", reason, 500)
    }

    pub fn with_cause(self, cause: RemoteError) -> Self {
        self.modify(|data| data.caused_by = Some(Box::new(cause)), std::mem::size_of::<RemoteError>())
    }

    /// A `circuit_breaking_exception`'s sizes and durability, carried to the
    /// node that renders it.
    pub fn with_breaking(self, bytes_wanted: u64, bytes_limit: u64, durability: &str) -> Self {
        self.with_metadata("bytes_wanted", bytes_wanted.to_string()).with_metadata("bytes_limit", bytes_limit.to_string()).with_metadata("durability", durability)
    }

    pub fn with_metadata(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into(); let value = value.into();
        let extra = key.len().checked_add(value.len()).and_then(|n|n.checked_add(crate::control::tree_slot::<String,String>()));
        match extra { Some(extra) => self.modify(|data| { data.metadata.insert(key,value); }, extra), None => Self::new("circuit_breaking_exception", "control metadata allocation overflow",429) }
    }

    pub(crate) fn attach_owner(&mut self, owner: &std::sync::Arc<crate::control::Owner>) {
        // Called only on a newly decoded, unpublished tree: every Arc is unique.
        let payload = std::sync::Arc::get_mut(&mut self.0).expect("fresh control decode");
        payload.owner = Some(owner.clone());
        if let Some(cause) = &mut payload.data.caused_by { cause.attach_owner(owner); }
    }
    fn modify(self, change: impl FnOnce(&mut RemoteErrorData), extra: usize) -> Self {
        let owner = if let Some(previous) = &self.0.owner {
            let bytes = self.error_type.len().checked_add(self.reason.len())
                .and_then(|n|self.metadata.iter().try_fold(n,|n,(k,v)|n.checked_add(k.len())?.checked_add(v.len())?.checked_add(crate::control::tree_slot::<String,String>())))
                .and_then(|n|n.checked_add(std::mem::size_of::<RemoteErrorOwner>() + 2*std::mem::size_of::<usize>()))
                .and_then(|n|n.checked_add(if self.caused_by.is_some() { std::mem::size_of::<RemoteError>() } else { 0 }))
                .and_then(|n|n.checked_add(extra));
            let Some(bytes) = bytes else { return Self::new("circuit_breaking_exception","control allocation overflow",429) };
            match crate::control::own(bytes, previous.admission.as_ref()) {
                Ok(owner) => Some(owner),
                Err(error) => return error.to_remote_error(),
            }
        } else { None };
        // Cloning a cause only shares its admitted payload; the Box itself is
        // included in this payload's allocation bound.
        let mut data = self.0.data.clone();
        change(&mut data);
        Self(std::sync::Arc::new(RemoteErrorOwner { data, owner }))
    }

    /// The innermost cause (ES's `root_cause` is built from it).
    pub fn root_cause(&self) -> &RemoteError {
        let mut e = self;
        while let Some(c) = &e.caused_by {
            e = c;
        }
        e
    }

    /// The ES JSON error object: `{"type", "reason", <metadata>, "caused_by"}`.
    /// (Without the `root_cause` array; the REST layer adds that at the top.)
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("type".into(), Value::String(self.error_type.clone()));
        m.insert("reason".into(), Value::String(self.reason.clone()));
        for (k, v) in &self.metadata {
            if self.hides(k) { continue }
            m.insert(k.clone(), Value::String(v.clone()));
        }
        if self.error_type == BREAKING {
            for field in BREAKING_FIELDS { m.insert(field.into(), self.breaking_value(field)); }
        }
        if let Some(c) = &self.caused_by {
            m.insert("caused_by".into(), c.to_json());
        }
        Value::Object(m)
    }
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.error_type, self.reason)?;
        if let Some(c) = &self.caused_by {
            write!(f, "; caused by: {c}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RemoteError {}

/// Every way a transport call can fail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TransportError {
    /// ES `ConnectTransportException`: TCP connect / TLS / handshake I/O failed
    /// or timed out.
    #[error("[{addr}] connect_exception: {reason}")]
    ConnectTransport { addr: SocketAddr, reason: String },
    /// The handshake completed but the peer is not acceptable (e.g. another
    /// cluster). ES wraps this in an `IllegalStateException`.
    #[error("handshake with [{addr}] failed: {reason}")]
    HandshakeFailed { addr: SocketAddr, reason: String },
    /// ES `ReceiveTimeoutTransportException`.
    #[error("[{addr}][{action}] request_id [{request_id}] timed out after [{timeout_ms}ms]")]
    ReceiveTimeout { addr: SocketAddr, action: String, request_id: u64, timeout_ms: u64 },
    /// ES `NodeDisconnectedException`: the connection closed while the request
    /// was in flight (or could not be written).
    #[error("[{addr}][{action}] disconnected")]
    NodeDisconnected { addr: SocketAddr, action: String },
    /// ES `ActionNotFoundTransportException`.
    #[error("No handler for action [{action}]")]
    ActionNotFound { action: String },
    /// The remote handler failed.
    #[error("{0}")]
    Remote(RemoteError),
    /// A request or response could not be (de)serialized.
    #[error("serialization error: {0}")]
    Serialization(String),
    /// The encoded message exceeds `max_frame_size`.
    #[error("message of [{size}] bytes exceeds the maximum frame size of [{max}] bytes")]
    FrameTooLarge { size: usize, max: usize },
    /// `close()` was called on the service.
    #[error("transport service is closed")]
    Closed,
    /// Bind / TLS configuration failures at startup.
    #[error("transport configuration error: {0}")]
    Config(String),
}

impl TransportError {
    /// The ES-shaped error for rendering this failure over REST.
    pub fn to_remote_error(&self) -> RemoteError {
        // Statuses as in ES: ConnectTransportException (and its subclass
        // NodeDisconnectedException) is 502. ReceiveTimeoutTransportException
        // does not override `status()` and has no cause, so ES answers 500
        // (`ElasticsearchException.status()` falls through to
        // INTERNAL_SERVER_ERROR); only EsRejectedExecutionException is 429.
        let (ty, status) = match self {
            TransportError::Remote(e) => return e.clone(),
            TransportError::ConnectTransport { .. } => ("connect_transport_exception", 502),
            TransportError::HandshakeFailed { .. } => ("illegal_state_exception", 500),
            TransportError::ReceiveTimeout { .. } => ("receive_timeout_transport_exception", 500),
            TransportError::NodeDisconnected { .. } => ("node_disconnected_exception", 502),
            TransportError::ActionNotFound { .. } => ("action_not_found_transport_exception", 500),
            TransportError::Serialization(_) => ("transport_serialization_exception", 500),
            TransportError::FrameTooLarge { .. } => ("illegal_argument_exception", 400),
            TransportError::Closed => ("transport_exception", 500),
            TransportError::Config(_) => ("settings_exception", 500),
        };
        RemoteError::new(ty, self.to_string(), status)
    }

    /// Whether the failure means "the node is (currently) unreachable", the
    /// condition higher layers treat as a node failure / retry elsewhere.
    pub fn is_connection_failure(&self) -> bool {
        matches!(
            self,
            TransportError::ConnectTransport { .. } | TransportError::NodeDisconnected { .. } | TransportError::Closed
        )
    }
}

/// `type` used on the wire for unknown actions; mapped back to
/// [`TransportError::ActionNotFound`] by the caller.
pub(crate) const ACTION_NOT_FOUND_TYPE: &str = "action_not_found_transport_exception";

#[cfg(test)]
mod status_tests {
    use super::TransportError;

    /// ES 8.19 `ReceiveTimeoutTransportException` is a plain
    /// `ActionTransportException`: 500, never 429 (429 is reserved for
    /// `es_rejected_execution_exception`).
    #[test]
    fn receive_timeout_is_500() {
        let e = TransportError::ReceiveTimeout { addr: "127.0.0.1:9300".parse().unwrap(), action: "indices:data/read/search[phase/query]".into(), request_id: 7, timeout_ms: 100 };
        let r = e.to_remote_error();
        assert_eq!(r.error_type, "receive_timeout_transport_exception");
        assert_eq!(r.status, 500);
        assert_eq!(r.reason, "[127.0.0.1:9300][indices:data/read/search[phase/query]] request_id [7] timed out after [100ms]");
    }
}

/// A serde decoder cannot return a typed transport error. This reserved custom
/// error tag is emitted only by before-allocation admission seeds, never by
/// fields decoded from the peer. Classify the exact Custom variant/prefix.
const DECODE_ADMISSION: &str = "rustic.internal.decode-admission.v1:";
pub fn decode_admission_error(reason: impl std::fmt::Display) -> String {
    format!("{DECODE_ADMISSION}{reason}")
}

pub(crate) fn typed_decode_error(error: bincode::Error, action: &str) -> RemoteError {
    if let bincode::ErrorKind::Custom(message) = &*error {
        if let Some(reason) = message.strip_prefix(DECODE_ADMISSION) {
            return RemoteError::new("circuit_breaking_exception", reason, 429);
        }
    }
    RemoteError::new("transport_serialization_exception", format!("failed to deserialize payload for action [{action}]: {error}"), 500)
}

pub(crate) fn response_decode_error(error: bincode::Error, action: &str) -> TransportError {
    if let bincode::ErrorKind::Custom(message) = &*error {
        if let Some(reason) = message.strip_prefix(DECODE_ADMISSION) {
            return TransportError::Remote(RemoteError::new("circuit_breaking_exception", reason, 429));
        }
    }
    TransportError::Serialization(format!("failed to deserialize response for [{action}]: {error}"))
}

#[cfg(test)]
mod breaking_shape_tests {
    use super::*;
    /// ES renders bytes_wanted, bytes_limit (numbers) and durability on every
    /// circuit_breaking_exception, after the other metadata; carried values
    /// survive the wire, missing ones are ES's defaults.
    #[test]
    fn breaking_fields_render_as_es_numbers_after_other_metadata() {
        let carried = RemoteError::new(BREAKING, "[request] Data too large", 429).with_metadata("index", "i").with_breaking(10, 5, "TRANSIENT");
        let wire: RemoteError = bincode::deserialize(&bincode::serialize(&carried).unwrap()).unwrap();
        let v = wire.to_json();
        assert_eq!(v, serde_json::json!({"type": BREAKING, "reason": "[request] Data too large", "index": "i", "bytes_wanted": 10, "bytes_limit": 5, "durability": "TRANSIENT"}));
        let view = serde_json::to_value(wire.json_with_root()).unwrap();
        assert_eq!(view["bytes_wanted"], 10);
        assert_eq!(view["root_cause"][0]["durability"], "TRANSIENT");
        let bare = RemoteError::new(BREAKING, "control metadata allocation overflow", 429).to_json();
        assert_eq!((&bare["bytes_wanted"], &bare["bytes_limit"], &bare["durability"]), (&serde_json::json!(0), &serde_json::json!(0), &serde_json::json!("PERMANENT")));
        assert!(RemoteError::new("exception", "x", 500).to_json().get("bytes_wanted").is_none());
    }
}

#[cfg(test)]
mod admission_decode_tests {
    use super::*;
    #[test]
    fn only_reserved_custom_decode_admission_is_a_breaker() {
        let denied = Box::new(bincode::ErrorKind::Custom(decode_admission_error("request limit")));
        let error = typed_decode_error(denied, "replica");
        assert_eq!(error.status, 429);
        let ordinary = Box::new(bincode::ErrorKind::Custom(format!("unknown field `{DECODE_ADMISSION}request limit`")));
        assert_eq!(typed_decode_error(ordinary, "replica").status, 500);
        assert_eq!(typed_decode_error(Box::new(bincode::ErrorKind::SizeLimit), "replica").status, 500);
    }
}
