//! Coordinator timestamp failures use a master-created, replicated failure index.
use super::*;
use elasticsearch_engine::{datastream::{failure_document, failure_store_eligible, FailureStoreTarget}, error::EsError};

const ENSURE: &str = "internal:data_stream/ensure_failure_store";
pub(super) const TOKEN_HEADER: &str = "x-rustic-internal-failure-token";
pub(super) const TOKENS_HEADER: &str = "x-rustic-internal-failure-tokens";
pub(super) const ACK_HEADER: &str = "x-rustic-internal-failure-store-ack";
pub(super) const POSITIONS_HEADER: &str = "x-rustic-internal-failure-store-positions";

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Token { id: String, timestamp: String }

fn new_token() -> Token {
    Token { id: new_doc_id(), timestamp: now_timestamp() }
}

fn now_timestamp() -> String {
    elasticsearch_engine::date::DateFormatter::new("strict_date_optional_time").unwrap()
        .format(elasticsearch_engine::date::now_millis(), elasticsearch_engine::date::Tz::UTC)
}

/// A bulk request's failure-store tokens ([`TOKENS_HEADER`]): one base
/// token for the whole request, item `i`'s being the base id suffixed with
/// its position in the original request (`positions[i]` in a subgroup,
/// `i` otherwise), so a retried group carries the same failure document
/// ids. One per item, generated, serialized and parsed for every bulk and
/// each of its shard subgroups whether or not any target had a failure
/// store, cost a sixth of the coordinating node's bulk CPU.
#[derive(Clone, Serialize, Deserialize)]
struct BulkTokens { id: String, timestamp: String, #[serde(default, skip_serializing_if = "Option::is_none")] positions: Option<Vec<usize>> }

fn bulk_tokens(req: &RestReq) -> Option<BulkTokens> {
    req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(TOKENS_HEADER))
        .and_then(|(_, value)| serde_json::from_str(value).ok())
}

impl BulkTokens {
    /// The original request's position of this request's item `position`.
    fn original(&self, position: usize) -> usize {
        self.positions.as_ref().map_or(Some(position), |p| p.get(position).copied()).unwrap_or(position)
    }
    fn token(&self, position: usize) -> Token {
        Token { id: format!("{}-{}", self.id, self.original(position)), timestamp: self.timestamp.clone() }
    }
}

pub(super) fn token(req: &RestReq, position: Option<usize>) -> Token {
    position.and_then(|position| bulk_tokens(req).map(|tokens| tokens.token(position)))
        .or_else(|| req.headers.iter().find(|(name, _)| name.eq_ignore_ascii_case(TOKEN_HEADER))
            .and_then(|(_, value)| serde_json::from_str(value).ok())).unwrap_or_else(new_token)
}

pub(super) fn prepare_single(req: &mut RestReq) -> elasticsearch_engine::error::Result<()> {
    if !req.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case(TOKEN_HEADER)) {
        req.push_header(TOKEN_HEADER, &serde_json::to_string(&new_token()).unwrap())?;
    }
    Ok(())
}

/// Give a bulk request its failure-store tokens (kept when it already has
/// them: a retried or forwarded request).
pub(super) fn prepare_bulk(req: &mut RestReq, _count: usize) -> elasticsearch_engine::error::Result<()> {
    if bulk_tokens(req).is_some() { return Ok(()); }
    let tokens = BulkTokens { id: new_doc_id(), timestamp: now_timestamp(), positions: None };
    req.replace_header(TOKENS_HEADER, &serde_json::to_string(&tokens).unwrap())
}

/// The tokens of `child`, the subgroup of `parent` holding its items at
/// `positions`.
pub(super) fn subgroup(parent: &RestReq, child: &mut RestReq, positions: &[usize]) -> elasticsearch_engine::error::Result<()> {
    let Some(tokens) = bulk_tokens(parent) else { return Ok(()) };
    let positions = positions.iter().map(|&p| tokens.original(p)).collect();
    child.replace_header(TOKENS_HEADER, &serde_json::to_string(&BulkTokens { positions: Some(positions), ..tokens }).unwrap())
}

pub(super) fn register(node: &Arc<ClusterNode>) {
    let node = node.clone();
    node.transport.clone().register_handler(ENSURE, move |stream: SharedText| {
        let node = node.clone();
        async move { ensure(&node, stream).await }
    });
}

async fn ensure(node: &Arc<ClusterNode>, stream: SharedText) -> Result<SharedText, RemoteError> {
    if !node.is_master() {
        return Err(RemoteError::new("not_master_exception", "not the elected master", 503));
    }
    await_mirror(node).await;
    if let Some(stream) = node.engine.data_stream(&stream).filter(|stream| !stream.failure_needs_rollover()) {
        if let Some(index) = stream.failure_write_index().filter(|index| node.state().metadata.indices.contains_key(*index)) {
            // Recovery may temporarily remove the master's local engine shell.
            // An allocated failure index remains the write target; this is not
            // a request to advance the data stream's generation.
            wait_for_primary(node, index, Duration::from_secs(30)).await;
            return SharedText::try_new(index).map_err(|e|RemoteError::new(&e.kind,e.reason,e.status));
        }
    }
    let (index, version) = node.mutate_engine_metadata(move |engine| {
        if engine.data_stream(&stream).is_none() { engine.create_data_stream_ext(&stream, true)?; }
        engine.failure_store_write_index(&stream).and_then(|service|SharedText::try_new(&service.name))
    }).await.map_err(|error| RemoteError::new(&error.kind, error.reason, error.status))?;
    if !node.wait_applied_everywhere(version, Duration::from_secs(30)).await {
        return Err(RemoteError::new("timeout_exception", "failure store metadata was not applied everywhere", 503));
    }
    wait_for_primary(node, &index, Duration::from_secs(30)).await;
    Ok(index)
}

/// Redirect errors discovered before the normal primary write. The generated
/// failure document is written through the ordinary primary/replica path, with
/// one stable ID for that write's retries.
pub(super) async fn timestamp_failure(node: &Arc<ClusterNode>, req: &RestReq, index: &str, error: EsError) -> RestResp {
    if !failure_store_eligible(&error) {
        return RestResp::json(error.status, &error.to_json_with_trace(false));
    }
    let stream = match node.engine.failure_store_target(index) {
        FailureStoreTarget::None => return RestResp::json(error.status, &error.to_json_with_trace(false)),
        FailureStoreTarget::Disabled => return RestResp::json(error.status, &error.with("failure_store", "not_enabled").to_json_with_trace(false)),
        FailureStoreTarget::Enabled(stream) => stream,
    };
    let Some(source) = req.json().filter(Value::is_object) else {
        return RestResp::json(error.status, &error.to_json_with_trace(false));
    };
    let params = req.params();
    let segments = req.segments();
    let document = failure_document(index, segments.get(2).map(String::as_str), params.get("routing").map(String::as_str), &source, &error, None);
    store(node, req, stream, document, token(req, None)).await
}

pub(super) async fn bulk_prewrite_failure(
    node: &Arc<ClusterNode>, req: &RestReq, index: &str, id: Option<&str>, routing: Option<&str>,
    source: &str, error: EsError, position: usize,
) -> RestResp {
    if !failure_store_eligible(&error) || error.reason.starts_with("only write ops with an op_type of create") {
        return RestResp::json(error.status, &error.to_json_with_trace(false));
    }
    let stream = match node.engine.failure_store_target(index) {
        FailureStoreTarget::None => return RestResp::json(error.status, &error.to_json_with_trace(false)),
        FailureStoreTarget::Disabled => return RestResp::json(error.status, &error.with("failure_store", "not_enabled").to_json_with_trace(false)),
        FailureStoreTarget::Enabled(stream) => stream,
    };
    let Ok(source) = serde_json::from_str::<Value>(source) else { return RestResp::json(error.status, &error.to_json_with_trace(false)); };
    if !source.is_object() { return RestResp::json(error.status, &error.to_json_with_trace(false)); }
    let document = failure_document(index, id, routing, &source, &error, None);
    store(node, req, stream, document, token(req, Some(position))).await
}

pub(super) async fn ingest_failure(node: &Arc<ClusterNode>, req: &RestReq, original_index: &str,
    failure: elasticsearch_engine::ingest::IngestFailure) -> RestResp {
    let stream = match node.engine.failure_store_target(&failure.index) {
        FailureStoreTarget::None => return RestResp::json(failure.error.status, &failure.error.to_json_with_trace(false)),
        FailureStoreTarget::Disabled => {
            let mut body = failure.error.to_json_with_trace(false);
            body["error"]["failure_store"] = json!("not_enabled");
            return RestResp::json(failure.error.status, &body);
        }
        FailureStoreTarget::Enabled(stream) => stream,
    };
    let Some(source) = req.json().filter(Value::is_object) else {
        return RestResp::json(failure.error.status, &failure.error.to_json_with_trace(false));
    };
    let params = req.params();
    let segments = req.segments();
    let document = failure_document(original_index, segments.get(2).map(String::as_str), params.get("routing").map(String::as_str),
        &source, &failure.error, Some(&failure.context));
    store(node, req, stream, document, token(req, None)).await
}

pub(super) async fn store(node: &Arc<ClusterNode>, req: &RestReq, stream: String, mut document: Value, token: Token) -> RestResp {
    document["@timestamp"] = json!(token.timestamp);
    let params = req.params();
    let result = if node.is_master() {
        match SharedText::try_new(&stream){Ok(stream)=>ensure(node,stream).await,Err(e)=>return err(e)}
    } else {
        let master = node.state().master_node_id.clone();
        match master.and_then(|id| node.address_of(&id)) {
            Some(address) => node.transport.send_request::<_, SharedText>(address, ENSURE, &stream, Duration::from_secs(60)).await.map_err(|error| error.to_remote_error()),
            None => Err(RemoteError::new("master_not_discovered_exception", "failure store requires an elected master", 503)),
        }
    };
    let concrete = match result { Ok(index) => index, Err(error) => return RestResp::from_remote(&error) };
    let mut failure = req.clone();
    // This is a new destination write, not local execution of the original
    // bulk subgroup. Carrying EXEC_LOCAL to a different primary bypasses its
    // replication gate and returns the engine's single-copy acknowledgement.
    let prepared = (|| -> elasticsearch_engine::error::Result<()> {
    failure.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(super::ingest::PREPARED_HEADER)
        && !name.eq_ignore_ascii_case(super::EXEC_LOCAL)
        && !name.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER)
        && !name.eq_ignore_ascii_case(super::DATA_STREAM_BACKING_WRITE_HEADER))?;
    failure.set_method("PUT")?;
    let mut uri = format!("/{}/_doc/{}?op_type=create", encode_component(&concrete), encode_component(&token.id));
    if let Some(refresh) = params.get("refresh") {
        uri.push_str(&format!("&refresh={}", encode_component(refresh)));
    }
    failure.set_uri(&uri)?;
    failure.set_json(&document)?;
    Ok(())
    })();
    if let Err(e)=prepared {return RestResp::json(e.status,&e.to_json())}
    // Box the recursive write: its target is a concrete failure index, so it
    // cannot re-enter this data-stream timestamp redirection.
    let mut response = Box::pin(write(node, &failure, &concrete)).await;
    if response.is_success() {
        if let Some(mut body) = response.body_json() {
            body["failure_store"] = json!("used");
            response.replace_json(&body);
            response.retain_headers(|(key, _)| !key.eq_ignore_ascii_case("content-length"));
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_failure_tokens_survive_regrouping_and_retries() {
        let mut parent = RestReq::try_new("POST","/_bulk",&[],&[],0,Default::default(),None).unwrap();
        prepare_bulk(&mut parent, 3).unwrap();
        let original = token(&parent, Some(2));
        assert_ne!(original.id, token(&parent, Some(0)).id);
        let mut child = parent.clone();
        subgroup(&parent, &mut child, &[2, 0]).unwrap();
        prepare_bulk(&mut child, 2).unwrap();
        let retried = token(&child, Some(0));
        assert_eq!(retried.id, original.id);
        assert_eq!(retried.timestamp, original.timestamp);
    }

    /// Every item of a bulk has its own token, and a subgroup of a subgroup
    /// (a retried shard group regrouped again) still maps each item to the
    /// token of its position in the original request.
    #[test]
    fn bulk_failure_tokens_are_unique_and_follow_nested_subgroups() {
        let mut parent = RestReq::try_new("POST","/_bulk",&[],&[],0,Default::default(),None).unwrap();
        prepare_bulk(&mut parent, 1000).unwrap();
        let ids: std::collections::HashSet<String> = (0..1000).map(|i| token(&parent, Some(i)).id).collect();
        assert_eq!(ids.len(), 1000);
        let mut child = parent.clone();
        subgroup(&parent, &mut child, &[7, 3, 999, 500]).unwrap();
        let mut grandchild = child.clone();
        subgroup(&child, &mut grandchild, &[2, 0]).unwrap();
        assert_eq!(token(&grandchild, Some(0)).id, token(&parent, Some(999)).id);
        assert_eq!(token(&grandchild, Some(1)).id, token(&parent, Some(7)).id);
        // A request without bulk tokens (a single write) keeps its own.
        let mut single = RestReq::try_new("PUT","/i/_doc/1",&[],&[],0,Default::default(),None).unwrap();
        prepare_single(&mut single).unwrap();
        assert_eq!(token(&single, None).id, token(&single, None).id);
    }
}
