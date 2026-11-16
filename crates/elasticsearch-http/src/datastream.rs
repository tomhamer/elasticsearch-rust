//! `_data_stream` create/get/delete and `_rollover`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::node::Node;
use std::sync::Arc;

/// `/_data_stream[/{name}]`.
pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let name = req.seg(1).to_string();
    if !req.seg(2).is_empty() {
        return crate::not_found_handler(&req);
    }
    let m = req.method.clone();
    match m {
        Method::PUT if !name.is_empty() => blocking(move || {
            node.create_data_stream(&name)?;
            Ok(Resp::ack())
        })
        .await,
        Method::GET => blocking(move || Ok(Resp::ok(node.get_data_streams(if name.is_empty() { None } else { Some(&name) })?))).await,
        Method::DELETE if !name.is_empty() => blocking(move || {
            node.delete_data_streams(&name)?;
            Ok(Resp::ack())
        })
        .await,
        _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
    }
}

/// `PUT|POST /_create_from/{source}/{dest}`.
pub async fn create_from(node: Arc<Node>, req: Req) -> R {
    let (source, dest) = (req.seg(1).to_string(), req.seg(2).to_string());
    if source.is_empty() || dest.is_empty() || !req.seg(3).is_empty() {
        return crate::not_found_handler(&req);
    }
    if req.method != Method::PUT && req.method != Method::POST {
        return method_not_allowed(&req, &["PUT", "POST"]);
    }
    blocking(move || {
        let body = if req.body.is_empty() { serde_json::Value::Null } else { req.json()? };
        Ok(Resp::ok(elasticsearch_engine::migrate::create_from(&node, &source, &dest, &body)?))
    })
    .await
}

/// `POST /_migration/reindex`, `GET /_migration/reindex/{index}/_status`,
/// `POST /_migration/reindex/{index}/_cancel`.
pub async fn migration_reindex(node: Arc<Node>, req: Req) -> R {
    let (name, action) = (req.seg(2).to_string(), req.seg(3).to_string());
    if !req.seg(4).is_empty() {
        return crate::not_found_handler(&req);
    }
    match (name.is_empty(), action.as_str()) {
        (true, "") => {
            if req.method != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            blocking(move || {
                let body = if req.body.is_empty() { serde_json::Value::Null } else { req.json()? };
                Ok(Resp::ok(elasticsearch_engine::migrate::migrate_reindex(&node, &body)?))
            })
            .await
        }
        (false, "_status") => {
            if req.method != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            blocking(move || Ok(Resp::ok(elasticsearch_engine::migrate::migrate_reindex_status(&node, &name)?))).await
        }
        (false, "_cancel") => {
            if req.method != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            blocking(move || Ok(Resp::ok(elasticsearch_engine::migrate::migrate_reindex_cancel(&node, &name)?))).await
        }
        _ => crate::not_found_handler(&req),
    }
}

/// `POST /{index}/_downsample/{target_index}`.
pub async fn downsample(node: Arc<Node>, req: Req, index: String) -> R {
    let target = req.seg(2).to_string();
    if target.is_empty() || !req.seg(3).is_empty() {
        return crate::not_found_handler(&req);
    }
    if req.method != Method::POST {
        return method_not_allowed(&req, &["POST"]);
    }
    // ES downsamples under DOWNSAMPLE_ORIGIN: the target needs no caller privileges.
    let rest = req.internal_cluster_rest();
    blocking(move || {
        let body = if req.body.is_empty() { serde_json::Value::Null } else { req.json()? };
        let out = match rest {
            Some(rest) => elasticsearch_engine::migrate::downsample_with(&node, &index, &target, &body, &RestDownsampleOps(rest))?,
            None => elasticsearch_engine::migrate::downsample(&node, &index, &target, &body)?,
        };
        Ok(Resp::ok(out))
    })
    .await
}

/// Downsample index operations through the cluster coordinator: the source
/// is read from every shard, the target is created in the cluster metadata
/// and its documents go to its primaries.
struct RestDownsampleOps(crate::ClusterRest);

impl RestDownsampleOps {
    fn call(&self, method: &str, uri: &str, body: Option<serde_json::Value>) -> Result<serde_json::Value, elasticsearch_engine::error::EsError> {
        (self.0 .0)(method, uri, body)
    }

    /// Visit every hit of `index` in `sort` order through a scroll of 1000
    /// hits a page (one page held at a time), cleared at the end. The
    /// scroll is strict (`allow_partial_search_results=false`) and every
    /// page, the first and each continuation, must come from every shard
    /// in time: a page with failed shards or a timeout fails the scan, so
    /// a partly read source never becomes a complete downsample.
    fn scroll(
        &self,
        index: &str,
        body: serde_json::Value,
        f: &mut dyn FnMut(&serde_json::Value) -> Result<(), elasticsearch_engine::error::EsError>,
    ) -> Result<(), elasticsearch_engine::error::EsError> {
        let mut resp = self.call("POST", &format!("/{index}/_search?scroll=1m&allow_partial_search_results=false"), Some(body))?;
        let mut scroll_id = resp["_scroll_id"].as_str().map(str::to_string);
        let result = 'pages: loop {
            if let Err(e) = page_complete(index, &resp) {
                break Err(e);
            }
            let hits = match resp["hits"]["hits"].as_array_mut() {
                Some(h) if !h.is_empty() => std::mem::take(h),
                _ => break Ok(()),
            };
            for h in &hits {
                if let Err(e) = f(h) {
                    break 'pages Err(e);
                }
            }
            drop(hits);
            let Some(id) = scroll_id.clone() else { break Ok(()) };
            match self.call("POST", "/_search/scroll", Some(serde_json::json!({"scroll": "1m", "scroll_id": id}))) {
                Ok(r) => {
                    resp = r;
                    if let Some(id) = resp["_scroll_id"].as_str() {
                        scroll_id = Some(id.to_string());
                    }
                }
                Err(e) => break Err(e),
            }
        };
        if let Some(id) = scroll_id {
            let _ = self.call("DELETE", "/_search/scroll", Some(serde_json::json!({"scroll_id": [id]})));
        }
        result
    }
}

/// A downsample source page read from every shard in time; else the
/// error that fails the scan (ES's strict search: no partial results).
fn page_complete(index: &str, resp: &serde_json::Value) -> Result<(), elasticsearch_engine::error::EsError> {
    let shards = &resp["_shards"];
    let failed = shards["failed"].as_u64().unwrap_or(0);
    if failed > 0 {
        let reason = shards["failures"].as_array().and_then(|a| a.first()).map(|f| f["reason"]["reason"].as_str().unwrap_or_default().to_string()).unwrap_or_default();
        return Err(elasticsearch_engine::error::EsError::new(
            503,
            "search_phase_execution_exception",
            format!("downsample of [{index}]: [{failed}] of [{}] source shards failed, the source cannot be read completely [{reason}]", shards["total"].as_u64().unwrap_or(0)),
        ));
    }
    if resp["timed_out"].as_bool() == Some(true) {
        return Err(elasticsearch_engine::error::EsError::new(
            504,
            "search_phase_execution_exception",
            format!("downsample of [{index}]: a source search page timed out, the source cannot be read completely"),
        ));
    }
    Ok(())
}

/// The NDJSON `_bulk` body indexing `docs` with generated ids.
fn bulk_index_body(docs: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for d in docs {
        out.push_str("{\"index\":{}}\n");
        out.push_str(&d.to_string());
        out.push('\n');
    }
    out
}

/// The first item failure of a `_bulk` response, as an error.
fn bulk_failure(resp: &serde_json::Value) -> Option<elasticsearch_engine::error::EsError> {
    if resp["errors"].as_bool() != Some(true) {
        return None;
    }
    resp["items"].as_array()?.iter().find_map(|item| {
        let r = item.as_object()?.values().next()?;
        let e = r.get("error")?;
        let status = r["status"].as_u64().unwrap_or(500) as u16;
        Some(elasticsearch_engine::error::EsError::new(status, e["type"].as_str().unwrap_or("exception"), e["reason"].as_str().unwrap_or("bulk item failed")))
    })
}

impl elasticsearch_engine::migrate::DownsampleOps for RestDownsampleOps {
    fn create_index(&self, name: &str, body: &serde_json::Value) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("PUT", &format!("/{name}"), Some(body.clone())).map(|_| ())
    }
    fn scan(&self, index: &str, f: &mut dyn FnMut(&serde_json::Value)) -> Result<(), elasticsearch_engine::error::EsError> {
        self.scroll(index, serde_json::json!({"size": 1000, "sort": ["_doc"]}), &mut |h| {
            if let Some(s) = h.get("_source") {
                f(s);
            }
            Ok(())
        })
    }
    fn scan_series(
        &self,
        index: &str,
        docvalue_fields: &[serde_json::Value],
        f: &mut dyn FnMut(&serde_json::Value) -> Result<(), elasticsearch_engine::error::EsError>,
    ) -> Result<(), elasticsearch_engine::error::EsError> {
        // The index sort: every shard's time series merged by `_tsid`, each
        // newest first.
        let mut body = serde_json::json!({"size": 1000, "sort": [{"_tsid": "asc"}, {"@timestamp": "desc"}]});
        if !docvalue_fields.is_empty() {
            body["docvalue_fields"] = serde_json::Value::Array(docvalue_fields.to_vec());
        }
        self.scroll(index, body, f)
    }
    fn index_docs(&self, index: &str, docs: Vec<serde_json::Value>) -> Result<(), elasticsearch_engine::error::EsError> {
        let resp = self.call("POST", &format!("/{index}/_bulk"), Some(serde_json::Value::String(bulk_index_body(&docs))))?;
        match bulk_failure(&resp) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
    fn refresh(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("POST", &format!("/{index}/_refresh"), None).map(|_| ())
    }
    fn block_writes(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("PUT", &format!("/{index}/_settings"), Some(serde_json::json!({"index": {"blocks": {"write": true}}}))).map(|_| ())
    }
    fn delete_index(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("DELETE", &format!("/{index}"), None).map(|_| ())
    }
    fn mark_success(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
        self.call("PUT", &format!("/{index}/_settings"), Some(serde_json::json!({"index.downsample.status": "success"}))).map(|_| ())
    }
}

#[cfg(test)]
mod downsample_ops_tests {
    use super::*;

    #[test]
    fn bulk_body_is_ndjson_with_auto_ids() {
        let body = bulk_index_body(&[serde_json::json!({"a": 1}), serde_json::json!({"b": "x"})]);
        assert_eq!(body, "{\"index\":{}}\n{\"a\":1}\n{\"index\":{}}\n{\"b\":\"x\"}\n");
    }

    #[test]
    fn bulk_failure_reports_first_item_error() {
        let ok = serde_json::json!({"errors": false, "items": [{"index": {"status": 201}}]});
        assert!(bulk_failure(&ok).is_none());
        let bad = serde_json::json!({"errors": true, "items": [{"index": {"status": 201}}, {"index": {"status": 400, "error": {"type": "document_parsing_exception", "reason": "boom"}}}]});
        let e = bulk_failure(&bad).expect("failure");
        assert_eq!(e.status, 400);
    }

    use elasticsearch_engine::migrate::{DownsampleOps, LocalDownsampleOps};
    use elasticsearch_engine::node::Node;
    use serde_json::{json, Value};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// A source page as a cluster scroll answers it: `failed` of 2 shards
    /// failed, `timed_out` as given.
    fn page(hits: Vec<Value>, failed: u64, timed_out: bool) -> Value {
        let failures: Vec<Value> = (0..failed).map(|i| json!({"shard": i, "index": "src", "reason": {"type": "no_shard_available_action_exception", "reason": "shard gone"}})).collect();
        json!({"_scroll_id": "sid", "timed_out": timed_out, "_shards": {"total": 2, "successful": 2 - failed, "skipped": 0, "failed": failed, "failures": failures}, "hits": {"hits": hits}})
    }

    /// Cluster REST serving `pages` in order to the scroll (search, then
    /// each continuation); every call is logged.
    fn scripted(pages: Vec<Value>) -> (RestDownsampleOps, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(VecDeque::from(pages)));
        let l = log.clone();
        let rest = crate::ClusterRest(Arc::new(move |m: &str, u: &str, _b: Option<Value>| {
            l.lock().unwrap().push(format!("{m} {u}"));
            if m == "DELETE" {
                return Ok(json!({"succeeded": true}));
            }
            Ok(queue.lock().unwrap().pop_front().unwrap_or_else(|| page(Vec::new(), 0, false)))
        }));
        (RestDownsampleOps(rest), log)
    }

    fn hit(i: u64) -> Value {
        json!({"_id": i.to_string(), "_source": {"@timestamp": 1_700_000_000_000u64 + i}, "sort": ["t", 1_700_000_000_000u64 + i]})
    }

    fn scan_count(ops: &RestDownsampleOps) -> (Result<(), elasticsearch_engine::error::EsError>, usize) {
        let mut n = 0;
        let r = ops.scan_series("src", &[], &mut |_| {
            n += 1;
            Ok(())
        });
        (r, n)
    }

    /// Review finding 110: the cluster scan is strict and checks every
    /// page. The first page or any continuation reporting a failed shard
    /// or a timeout fails the scan (the scroll is still cleared); complete
    /// pages are read through.
    #[test]
    fn partial_source_pages_fail_the_downsample_scan() {
        let (ops, log) = scripted(vec![page(vec![hit(1), hit(2)], 0, false), page(vec![hit(3)], 0, false)]);
        let (r, n) = scan_count(&ops);
        assert!(r.is_ok() && n == 3, "{r:?} {n}");
        assert!(log.lock().unwrap()[0].contains("allow_partial_search_results=false"), "{:?}", log.lock().unwrap());
        // Initial page: one of two shards failed, the other's hits present.
        let (ops, log) = scripted(vec![page(vec![hit(1), hit(2)], 1, false)]);
        let (r, n) = scan_count(&ops);
        let e = r.expect_err("an initial page with a failed shard");
        assert_eq!((e.status, e.kind.as_str(), n), (503, "search_phase_execution_exception", 0), "{e:?}");
        assert!(log.lock().unwrap().iter().any(|c| c == "DELETE /_search/scroll"), "the scroll is cleared");
        // A continuation page with a failed shard.
        let (ops, _) = scripted(vec![page(vec![hit(1), hit(2)], 0, false), page(vec![hit(3)], 1, false)]);
        let (r, n) = scan_count(&ops);
        assert_eq!((r.map_err(|e| e.status), n), (Err(503), 2));
        // A continuation page that timed out.
        let (ops, _) = scripted(vec![page(vec![hit(1), hit(2)], 0, false), page(vec![hit(3)], 0, true)]);
        let (r, n) = scan_count(&ops);
        assert_eq!((r.map_err(|e| e.status), n), (Err(504), 2));
        // An initial page that timed out.
        let (ops, _) = scripted(vec![page(vec![hit(1)], 0, true)]);
        assert!(scan_count(&ops).0.is_err());
    }

    /// The cluster read of the source (scripted pages built from the
    /// source's real hits, one page failing as `fail` says), every other
    /// operation on the local engine.
    struct ClusterRead<'a> {
        rest: RestDownsampleOps,
        local: LocalDownsampleOps<'a>,
    }

    impl DownsampleOps for ClusterRead<'_> {
        fn create_index(&self, name: &str, body: &Value) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.create_index(name, body)
        }
        fn scan(&self, index: &str, f: &mut dyn FnMut(&Value)) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.scan(index, f)
        }
        fn scan_series(&self, index: &str, dv: &[Value], f: &mut dyn FnMut(&Value) -> Result<(), elasticsearch_engine::error::EsError>) -> Result<(), elasticsearch_engine::error::EsError> {
            self.rest.scan_series(index, dv, f)
        }
        fn index_docs(&self, index: &str, docs: Vec<Value>) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.index_docs(index, docs)
        }
        fn refresh(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.refresh(index)
        }
        fn block_writes(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.block_writes(index)
        }
        fn delete_index(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.delete_index(index)
        }
        fn mark_success(&self, index: &str) -> Result<(), elasticsearch_engine::error::EsError> {
            self.local.mark_success(index)
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Fail {
        None,
        InitialShard,
        ContinuationShard,
        ContinuationTimeout,
    }

    /// The source's hits in index order as two scroll pages, `fail` applied.
    fn cluster_read<'a>(node: &'a Node, source: &str, fail: Fail) -> ClusterRead<'a> {
        let mut hits = Vec::new();
        LocalDownsampleOps(node)
            .scan_series(source, &[], &mut |h| {
                hits.push(h.clone());
                Ok(())
            })
            .unwrap();
        assert!(hits.len() >= 2, "{hits:?}");
        let second = hits.split_off(hits.len() / 2);
        let pages = match fail {
            Fail::None => vec![page(hits, 0, false), page(second, 0, false)],
            // The failed shard's hits are missing; the other's are there.
            Fail::InitialShard => vec![page(hits, 1, false)],
            Fail::ContinuationShard => vec![page(hits, 0, false), page(second, 1, false)],
            Fail::ContinuationTimeout => vec![page(hits, 0, false), page(second, 0, true)],
        };
        ClusterRead { rest: scripted(pages).0, local: LocalDownsampleOps(node) }
    }

    /// The cluster's data stream lifecycle downsample step, as
    /// `ClusterSteps::downsample` runs it (`_downsample` through the
    /// cluster read), every other step unused.
    struct Steps {
        node: Arc<Node>,
        fail: Mutex<Fail>,
    }

    impl elasticsearch_engine::ilm::ClusterLifecycleSteps for Steps {
        fn single_node(&self, _: &str) -> Option<String> {
            None
        }
        fn allocated_on(&self, _: &str, _: &str) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn primaries_active(&self, _: &str) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn forcemerge(&self, _: &str, _: usize) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn shrink(&self, _: &str, _: &str, _: &Value) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn holds_all_docs(&self, _: &str, _: &str) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn write_history(&self, _: Vec<Value>) {}
        fn allocation_routed(&self, _: &str, _: &std::collections::BTreeMap<String, String>, _: bool) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn downsample(&self, source: &str, target: &str, body: &Value) -> elasticsearch_engine::ilm::StepProgress {
            let fail = *self.fail.lock().unwrap();
            let ops = cluster_read(&self.node, source, fail);
            match elasticsearch_engine::migrate::downsample_with(&self.node, source, target, body, &ops) {
                Ok(_) => elasticsearch_engine::ilm::StepProgress::Done,
                Err(e) => elasticsearch_engine::ilm::StepProgress::Failed(e.reason),
            }
        }
        fn snapshot_includes(&self, _: &str, _: &str, _: &str, _: &str) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn clone_index(&self, _: &str, _: &str, _: &Value) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn snapshot_index(&self, _: &str, _: &str, _: &str, _: &Value) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn mount_snapshot(&self, _: &str, _: &str, _: &Value, _: bool) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn delete_snapshot(&self, _: &str, _: &str) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
        fn delete_snapshots_matching(&self, _: &str, _: &str, _: elasticsearch_engine::ilm::snapshot_steps::SnapshotSelect) -> elasticsearch_engine::ilm::StepProgress {
            elasticsearch_engine::ilm::StepProgress::Pending(None)
        }
    }

    fn iso(ms: i64) -> String {
        elasticsearch_engine::ilm::iso_date(ms)
    }

    /// Review finding 110, end to end: a manual `_downsample` and the data
    /// stream lifecycle whose cluster read of the source is partial (a
    /// failed shard on the first page or a continuation, or a timed-out
    /// continuation) fail; no target is left marked successful and the
    /// lifecycle keeps the original backing index. A complete read is the
    /// positive control.
    #[test]
    fn partial_cluster_reads_never_complete_a_downsample_or_delete_the_source() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.ilm.drive_externally();
        node.templates.write().index.insert(
            "metrics".into(),
            json!({"index_patterns": ["metrics-*"], "data_stream": {}, "template": {
                "settings": {"index.mode": "time_series", "index.routing_path": ["host"], "index.number_of_replicas": "0"},
                "mappings": {"properties": {
                    "@timestamp": {"type": "date"},
                    "host": {"type": "keyword", "time_series_dimension": true},
                    "cpu": {"type": "double", "time_series_metric": "gauge"}}},
                "lifecycle": {"downsampling": [{"after": "1d", "fixed_interval": "1h"}]}}}),
        );
        node.cluster_settings.write().0.insert("data_stream.past_tsdb_index_creation_enabled".into(), json!("true"));
        node.create_data_stream("metrics-a").unwrap();
        let now = elasticsearch_engine::date::now_millis();
        let hour = now - now.rem_euclid(3_600_000) - 5 * 3_600_000;
        let svc = node.create_past_time_series_service("metrics-a", hour).unwrap();
        let first = svc.name.clone();
        let docs: Vec<Value> = (0..6).map(|i| json!({"@timestamp": iso(hour + i * 60_000), "host": if i % 2 == 0 { "a" } else { "b" }, "cpu": (i + 1) as f64})).collect();
        LocalDownsampleOps(&node).index_docs(&first, docs).unwrap();
        svc.refresh().unwrap();
        let mut origin = elasticsearch_engine::settings::FlatSettings::new();
        origin.insert("index.lifecycle.origination_date".into(), json!((now - 2 * 86_400_000).to_string()));
        svc.put_settings(origin, false).unwrap();
        drop(svc);
        let ds = node.data_stream("metrics-a").unwrap();
        // First pass: the write block.
        elasticsearch_engine::ilm::dlm::downsample_backing_index(&node, &ds, &first).unwrap();
        let target = format!("downsample-1h-{first}");
        // Manual `_downsample` over each partial read.
        for fail in [Fail::InitialShard, Fail::ContinuationShard, Fail::ContinuationTimeout] {
            let ops = cluster_read(&node, &first, fail);
            assert!(elasticsearch_engine::migrate::downsample_with(&node, &first, "manual", &json!({"fixed_interval": "1h"}), &ops).is_err(), "{fail:?}");
            assert!(node.get_index("manual").is_none(), "{fail:?}: the partial target is removed, never marked successful");
        }
        // The lifecycle over each partial read keeps the source.
        let steps = Arc::new(Steps { node: node.clone(), fail: Mutex::new(Fail::InitialShard) });
        let dyn_steps: Arc<dyn elasticsearch_engine::ilm::ClusterLifecycleSteps> = steps.clone();
        node.ilm.set_cluster_steps(Some(dyn_steps));
        for fail in [Fail::InitialShard, Fail::ContinuationShard, Fail::ContinuationTimeout] {
            *steps.fail.lock().unwrap() = fail;
            assert!(elasticsearch_engine::ilm::dlm::downsample_backing_index(&node, &ds, &first).is_err(), "{fail:?}");
            assert!(node.get_index(&first).is_some(), "{fail:?}: the source is kept");
            assert!(node.get_index(&target).is_none_or(|t| t.setting("index.downsample.status").as_deref() != Some("success")), "{fail:?}");
            assert_eq!(node.data_stream("metrics-a").unwrap().indices[0], first, "{fail:?}");
        }
        // Positive control: a complete read downsamples and swaps.
        *steps.fail.lock().unwrap() = Fail::None;
        elasticsearch_engine::ilm::dlm::downsample_backing_index(&node, &ds, &first).unwrap();
        assert!(node.get_index(&first).is_none(), "the source goes once its downsample is complete");
        assert_eq!(node.data_stream("metrics-a").unwrap().indices[0], target);
        node.ilm.set_cluster_steps(None);
    }
}

/// `POST /{target}/_rollover[/{new_index}]`.
pub async fn rollover(node: Arc<Node>, req: Req, target: String) -> R {
    if req.method != Method::POST {
        return method_not_allowed(&req, &["POST"]);
    }
    blocking(move || {
        let body = req.json()?;
        let new_index = req.segs.get(2).cloned();
        let dry_run = req.flag("dry_run")?;
        let lazy = req.flag("lazy")?;
        let v = node.rollover_ext(&target, new_index.as_deref(), &body, dry_run, lazy)?;
        let mut r = Resp::ok(v);
        // `RestRolloverIndexAction`: `max_size` is deprecated.
        if body.pointer("/conditions/max_size").is_some_and(|v| !v.is_null()) {
            r.headers.push(crate::search_api::warning_header(
                "Use of the [max_size] rollover condition has been deprecated in favour of the [max_primary_shard_size] condition and will be removed in a later version",
            ));
        }
        Ok(r)
    })
    .await
}
