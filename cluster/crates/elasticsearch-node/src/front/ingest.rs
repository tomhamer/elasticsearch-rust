//! Execute bulk ingest once at the authenticated coordinator, before routing.
use super::{bulk, ClusterNode, RestReq};
use elasticsearch_engine::ingest::{IngestContext, IngestOutcome, IngestRequest};
use elasticsearch_http::{PreparedBulkIngest, PreparedBulkIngestItems};
use serde_json::{json, Value};
use std::sync::Arc;

/// Bulks of at least this many items run their pipelines in parallel.
const PARALLEL_INGEST_ITEMS: usize = 8;

pub const PREPARED_HEADER: &str = "x-rustic-internal-prepared-ingest";

pub fn prepared(req: &RestReq) -> PreparedBulkIngestItems {
    req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(PREPARED_HEADER))
        .and_then(|(_, value)| serde_json::from_str(value).ok()).unwrap_or_default()
}

fn attach(req: &mut RestReq, items: PreparedBulkIngestItems) -> elasticsearch_engine::error::Result<()> {
    req.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(PREPARED_HEADER))?;
    if !items.0.is_empty() {
        req.push_header(PREPARED_HEADER, &serde_json::to_string(&items).expect("ingest serialization"))?;
    }
    Ok(())
}

/// Subgroups have new positions; preserve the prepared state through retries.
pub fn subgroup(parent: &RestReq, child: &mut RestReq, positions: &[usize]) -> elasticsearch_engine::error::Result<()> {
    let items = prepared(parent);
    attach(child, PreparedBulkIngestItems(positions.iter().enumerate()
        .filter_map(|(new, old)| items.0.get(old).cloned().map(|state| (new, state))).collect()))
}

/// Whether preparing the bulk `req` would change nothing: no pipeline is
/// registered, neither the URL nor (conservatively, any mention in the
/// body) an item names one, no index sets a default or final pipeline, and
/// no alias carries an index routing.
fn no_pipeline_can_run(node: &Arc<ClusterNode>, req: &RestReq) -> bool {
    const NEEDLE: &[u8] = b"pipeline";
    req.params().get("pipeline").is_none()
        && node.engine.ingest_is_idle()
        && !contains(&req.body, NEEDLE)
        && !node.engine.all_indices().iter().any(|svc| ["index.default_pipeline", "index.final_pipeline"].iter()
            .any(|key| svc.setting(key).is_some_and(|p| p != "_none"))
            // The prepared state also carries an alias's index routing
            // (`index_routing_for`), which bulk_route routes by.
            || svc.aliases().values().any(|def| def.get("index_routing").is_some_and(|v| !v.is_null())))
}

/// `needle` (non-empty) occurs in `hay`.
fn contains(hay: &[u8], needle: &[u8]) -> bool {
    // SIMD substring search: every bulk body (10 MB for 1,000 metrics docs)
    // is scanned once.
    memchr::memmem::find(hay, needle).is_some()
}

pub async fn prepare(node: &Arc<ClusterNode>, mut req: RestReq, user: Option<elasticsearch_engine::security::Authn>) -> Result<RestReq, super::RestResp> {
    let segments = req.segments();
    if !matches!(req.method.as_str(), "POST" | "PUT") || segments.last().map(String::as_str) != Some("_bulk") {
        return prepare_document(node, req, user).await;
    }
    // No pipeline can run (none exists, the request names none, no index
    // has a default or final one): the body is not parsed and rebuilt here
    // (bulk_route splits it anyway).
    if no_pipeline_can_run(node, &req) {
        return Ok(req);
    }
    let default = segments.first().filter(|s| !s.starts_with('_')).cloned();
    let Some(mut items) = bulk::parse(&req.body, default.as_deref()).filter(|items| !items.is_empty()) else { return Ok(req) };
    let engine = node.engine.clone();
    // No pipeline runs for this request: it names none, and every document
    // goes to an existing concrete index without a default or final
    // pipeline (or a data stream whose write index has none). Pipelines elsewhere in the cluster (built-in ones, other
    // indices) do not matter; skip the per-document pass, which would
    // record an unchanged outcome for each item.
    if req.params().get("pipeline").is_none() && !contains(&req.body, b"pipeline") {
        let mut checked: Vec<&str> = Vec::new();
        let pipeline_free = items.iter().all(|item| {
            if !matches!(item.op.as_str(), "index" | "create") { return true }
            let Some(index) = item.index.as_deref() else { return false };
            if checked.contains(&index) { return true }
            // An existing index, or a data stream (whose write index's
            // settings `index_pipelines` reads): a metrics bulk into a data
            // stream took the per-document pass for every item.
            let free = (engine.get_index(index).is_some() || engine.data_stream(index).is_some()) && engine.index_pipelines(index) == (None, None);
            if free { checked.push(index); }
            free
        });
        if pipeline_free {
            return Ok(req);
        }
    }
    let params = req.params();
    let work = tokio::task::spawn_blocking(move || {
        let _user = IngestContext::enter(IngestContext { user: user.clone() });
        // One item's pipeline run: its prepared state, or `None` when it is
        // not an ingest target (left as written).
        let one = |position: usize, item: &mut bulk::BulkItem| -> Option<(usize, PreparedBulkIngest)> {
            if !matches!(item.op.as_str(), "index" | "create") { return None }
            let index = item.index.clone()?;
            let mut line = serde_json::from_str::<Value>(&item.action_line).ok()?;
            let meta = line.as_object_mut().and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut)?;
            // Preserve index requirements and routing validation before ingest.
            let alias = meta.get("require_alias").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or_else(|| params.get("require_alias").is_some_and(|s| s == "true"));
            let stream = meta.get("require_data_stream").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or_else(|| params.get("require_data_stream").is_some_and(|s| s == "true"));
            if alias && !engine.all_indices().iter().any(|svc| svc.aliases().contains_key(&index)) { return None }
            if engine.require_data_stream(&index, stream).is_err() { return None }
            let routing = meta.get("routing").and_then(Value::as_str).or_else(|| params.get("routing").map(String::as_str));
            let routing = engine.index_routing_for(&index, routing).ok()?;
            let pipeline = meta.get("pipeline").and_then(Value::as_str).or_else(|| params.get("pipeline").map(String::as_str));
            let source = item.source_line.as_deref().unwrap_or_default();
            let outcome = engine.run_ingest_ext(IngestRequest { index: &index, id: item.id.as_deref(), routing: routing.as_deref(), source, pipeline });
            let state = match outcome {
                Ok(IngestOutcome::Unchanged) => PreparedBulkIngest::Write { routing: routing.clone(), dynamic_templates: None },
                Ok(IngestOutcome::Changed { index, id, routing, source, dynamic_templates }) => {
                    item.index = Some(index.clone()); item.id = id.clone(); item.source_line = Some(source);
                    meta.insert("_index".into(), json!(index));
                    // Requirements were checked against the client's original target.
                    meta.insert("require_alias".into(), json!(false));
                    meta.insert("require_data_stream".into(), json!(false));
                    match id { Some(id) => { meta.insert("_id".into(), json!(id)); }, None => { meta.remove("_id"); } }
                    match &routing { Some(routing) => { meta.insert("routing".into(), json!(routing)); }, None => { meta.remove("routing"); } }
                    PreparedBulkIngest::Write { routing, dynamic_templates }
                }
                Ok(IngestOutcome::Dropped { index }) => {
                    item.index = Some(index.clone()); meta.insert("_index".into(), json!(index));
                    PreparedBulkIngest::Dropped
                }
                Err(failure) => PreparedBulkIngest::Failed(failure),
            };
            item.action_line = line.to_string();
            Some((position, state))
        };
        // Items run their pipelines independently: a bulk of several runs
        // them on the shared pool (the per-document pipeline was a bulk's
        // whole latency: tsdb_k8s_queries' 50-document touch bulks).
        // Only while the pool has workers to spare: with every core busy
        // (many concurrent bulks) the fan-out only adds hand-offs and
        // wake-ups (as the shard's parallel parse decides).
        static INGESTING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        struct Busy;
        impl Drop for Busy { fn drop(&mut self) { INGESTING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel); } }
        let busy = INGESTING.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let _busy = Busy;
        let cores = rayon::current_num_threads();
        let fan_out = busy == 0 || (busy + 1) * 2 <= cores;
        let states: Vec<(usize, PreparedBulkIngest)> = if items.len() >= PARALLEL_INGEST_ITEMS && fan_out {
            use rayon::prelude::*;
            items.par_iter_mut().enumerate().filter_map(|(position, item)| {
                let _user = IngestContext::enter(IngestContext { user: user.clone() });
                one(position, item)
            }).collect()
        } else {
            items.iter_mut().enumerate().filter_map(|(position, item)| one(position, item)).collect()
        };
        let mut prepared = PreparedBulkIngestItems::default();
        for (position, state) in states {
            prepared.0.insert(position, state);
        }
        (items, prepared)
    }).await;
    let (items, prepared) = work.map_err(|error| super::RestResp::error(500, "illegal_state_exception", &format!("coordinator ingest failed: {error}")))?;
    req.set_body(&bulk::build_body(items.iter())).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
    attach(&mut req, prepared).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
    Ok(req)
}

/// Whether a single-document write has create semantics: `_create`,
/// `op_type=create`, or an id-less `POST /{index}/_doc` without an explicit
/// `op_type` (ES `RestIndexAction` makes only that auto-id case a create; an
/// explicit `op_type=index` stays an index, e.g. overwriting a time series
/// document whose `_id` derives from its dimensions).
fn document_creates(segments: &[String], op_type: Option<&str>) -> bool {
    segments.get(1).map(String::as_str) == Some("_create") || op_type == Some("create") || (segments.len() == 2 && op_type.is_none())
}

async fn prepare_document(node: &Arc<ClusterNode>, mut req: RestReq, user: Option<elasticsearch_engine::security::Authn>) -> Result<RestReq, super::RestResp> {
    let segments = req.segments();
    let params = req.params();
    let valid = matches!(req.method.as_str(), "POST" | "PUT") && matches!(segments.len(), 2 | 3)
        && matches!(segments.get(1).map(String::as_str), Some("_doc" | "_create"))
        && (segments.len() == 3 || (req.method == "POST" && segments[1] == "_doc"));
    // Conditional/version validation belongs to the existing REST handler and
    // must precede ingest. Leave those requests on that validated path.
    if !valid || ["if_seq_no", "if_primary_term", "version", "version_type"].iter().any(|key| params.contains_key(*key))
        || params.get("op_type").is_some_and(|value| !matches!(value.as_str(), "create" | "index")) {
        return Ok(req);
    }
    let Some(source) = req.json().filter(Value::is_object) else { return Ok(req) };
    let original_index = segments[0].clone();
    let mut metadata = serde_json::Map::new();
    metadata.insert("_index".into(), json!(original_index));
    if let Some(id) = segments.get(2) { metadata.insert("_id".into(), json!(id)); }
    for key in ["routing", "pipeline", "require_alias", "require_data_stream"] {
        if let Some(value) = params.get(key) { metadata.insert(key.into(), json!(value)); }
    }
    let create = document_creates(&segments, params.get("op_type").map(String::as_str));
    let action = if create { "create" } else { "index" };
    let mut bulk = req.clone();
    bulk.set_uri(&format!("/_bulk{}", if req.query().is_empty() { String::new() } else { format!("?{}", req.query()) })).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
    bulk.set_body(format!("{}\n{}\n", json!({action:metadata}), source).as_bytes()).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
    let processed = Box::pin(prepare(node, bulk, user)).await?;
    let state = prepared(&processed);
    let Some(state) = state.0.get(&0).cloned() else { return Ok(req) };
    super::failure_store::prepare_single(&mut req).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
    match state {
        PreparedBulkIngest::Failed(failure) => return Err(super::failure_store::ingest_failure(node, &req, &original_index, failure).await),
        PreparedBulkIngest::Dropped => {
            let item = bulk::parse(&processed.body, None).and_then(|items| items.into_iter().next()).unwrap();
            return Err(super::RestResp::json(200, &elasticsearch_http::ingest_dropped_response(item.index.as_deref().unwrap_or(&original_index), item.id.as_deref())));
        }
        PreparedBulkIngest::Write { routing, .. } => {
            let item = bulk::parse(&processed.body, None).and_then(|items| items.into_iter().next()).unwrap();
            let index = item.index.as_deref().unwrap_or(&original_index);
            let mut params = params;
            if create { params.insert("op_type".into(), "create".into()); }
            if index != original_index {
                params.remove("require_alias"); params.remove("require_data_stream");
            }
            match routing { Some(routing) => { params.insert("routing".into(), routing); }, None => { params.remove("routing"); } }
            let endpoint = if segments[1] == "_create" && item.id.is_none() { "_doc" } else { &segments[1] };
            let mut uri = format!("/{}/{}", super::encode_component(index), endpoint);
            if let Some(id) = &item.id { uri.push_str(&format!("/{}", super::encode_component(id))); }
            if !params.is_empty() {
                let mut query: Vec<_> = params.iter().map(|(key, value)| format!("{}={}", super::encode_component(key), super::encode_component(value))).collect();
                query.sort(); uri.push_str(&format!("?{}", query.join("&")));
            }
            req.set_uri(&uri).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
            let prepared_source = item.source_line.unwrap_or_default();
            // An unchanged pipeline must not alter the client's JSON offsets
            // or stored source bytes merely to construct the routing envelope.
            if prepared_source != source.to_string() { req.set_body(prepared_source.as_bytes()).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?; }
        }
    }
    attach(&mut req, prepared(&processed)).map_err(|e|super::RestResp::json(e.status,&e.to_json()))?;
    Ok(req)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_index_op_type_is_not_a_create() {
        let segs = |p: &str| p.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect::<Vec<_>>();
        assert!(document_creates(&segs("/i/_doc"), None));
        assert!(!document_creates(&segs("/i/_doc"), Some("index")));
        assert!(document_creates(&segs("/i/_doc"), Some("create")));
        assert!(!document_creates(&segs("/i/_doc/1"), None));
        assert!(document_creates(&segs("/i/_doc/1"), Some("create")));
        assert!(document_creates(&segs("/i/_create/1"), None));
    }

    #[test]
    fn subgroup_preserves_prepared_items_in_new_positions() {
        let mut parent = RestReq::try_new("POST","/_bulk",&[],&[],0,Default::default(),None).unwrap();
        attach(&mut parent, PreparedBulkIngestItems(std::collections::BTreeMap::from([
            (0, PreparedBulkIngest::Dropped),
            (2, PreparedBulkIngest::Write { routing: None, dynamic_templates: Some(std::collections::HashMap::from([("x".into(), "template".into())])) }),
        ]))).unwrap();
        let mut child = parent.clone();
        subgroup(&parent, &mut child, &[2, 1, 0]).unwrap();
        let states = prepared(&child);
        assert!(matches!(states.0.get(&0), Some(PreparedBulkIngest::Write { routing: None, dynamic_templates: Some(_) })));
        assert!(!states.0.contains_key(&1));
        assert!(matches!(states.0.get(&2), Some(PreparedBulkIngest::Dropped)));
    }
}

#[cfg(test)]
mod skip_tests {
    #[test]
    fn contains_finds_a_needle_anywhere() {
        assert!(super::contains(b"{\"index\":{\"pipeline\":\"p\"}}", b"pipeline"));
        assert!(super::contains(b"pipeline", b"pipeline"));
        assert!(!super::contains(b"pipelin", b"pipeline"));
        assert!(!super::contains(b"ppppipelinx", b"pipeline"));
        assert!(super::contains(b"ppipeline", b"pipeline"));
    }
}
