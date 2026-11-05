use axum::{body::Body, http::Request};
use base64::Engine;
use elasticsearch_engine::{node::Node, security::{SecConfig, hash::Hasher}};
use elasticsearch_http::{AuthorizedDispatch, Resp};
use serde_json::json;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use tower::ServiceExt;

#[tokio::test]
async fn coordinator_continuation_runs_only_after_authentication_and_authorization() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mut cfg = SecConfig::default();
    cfg.enabled = true;
    for (user, role) in [("admin", "superuser"), ("reader", "reader")] {
        cfg.file_users.insert(user.into(), Hasher::Pbkdf2(1000).hash("password"));
        cfg.file_user_roles.insert(user.into(), vec![role.into()]);
    }
    cfg.file_roles.insert("reader".into(), json!({"cluster": [], "indices": [{"names": ["public"], "privileges": ["read"]}]}));
    node.security.configure(cfg);
    let router = elasticsearch_http::router(node);
    let calls = Arc::new(AtomicUsize::new(0));
    for (user, expected) in [(None, 401), (Some("reader"), 403), (Some("admin"), 200)] {
        let mut request = Request::builder().uri("/_cluster/health");
        if let Some(user) = user {
            let value = base64::engine::general_purpose::STANDARD.encode(format!("{user}:password"));
            request = request.header("Authorization", format!("Basic {value}"));
        }
        let mut request = request.body(Body::empty()).unwrap();
        let count = calls.clone();
        request.extensions_mut().insert(AuthorizedDispatch(Arc::new(move |req| {
            let count = count.clone();
            Box::pin(async move {
                assert!(req.access.is_some());
                count.fetch_add(1, Ordering::SeqCst);
                Ok(Resp::ok(json!({"status": "green"})))
            })
        })));
        let response = router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status().as_u16(), expected, "user={user:?}");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn alias_catalog_survives_missing_local_shard_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    assert!(node.all_indices().is_empty());
    let router = elasticsearch_http::router(node);
    let catalog = elasticsearch_http::AliasCatalog(vec![
        ("visible-index".into(), false, std::collections::BTreeMap::from([("visible-alias".into(), json!({}))])),
        ("hidden-index".into(), true, std::collections::BTreeMap::from([("hidden-alias".into(), json!({"is_hidden":true}))])),
    ]);
    for (uri, expected) in [
        ("/_cat/aliases?format=json", 2),
        ("/_cat/aliases/visible-*?format=json", 1),
        ("/_cat/aliases?format=json&expand_wildcards=open", 1),
    ] {
        let mut req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        req.extensions_mut().insert(catalog.clone());
        let response = router.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), expected, "{rows}");
        assert!(rows.as_array().unwrap().iter().any(|r| r["alias"] == "visible-alias"));
    }
    let response = router.oneshot(Request::builder().uri("/_cat/aliases?format=json")
        .header("x-es-alias-catalog", "visible-alias").body(Body::empty()).unwrap()).await.unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap(), json!([]));
}

#[tokio::test]
async fn system_identity_authenticates_internal_work_without_credentials() {
    let dir = tempfile::tempdir().unwrap();
    let node = Node::open(dir.path()).unwrap();
    let mut cfg = SecConfig::default();
    cfg.enabled = true;
    node.security.configure(cfg);
    let router = elasticsearch_http::router(node);
    let put = |system: bool| {
        let mut r = Request::builder().method("PUT").uri("/restored").header("content-type", "application/json").body(Body::from("{}")).unwrap();
        if system {
            r.extensions_mut().insert(elasticsearch_http::SystemIdentity);
        }
        r
    };
    // A client without credentials is rejected; the in-process marker (which
    // only the cluster layer attaches) runs as `_system`.
    assert_eq!(router.clone().oneshot(put(false)).await.unwrap().status().as_u16(), 401);
    assert_eq!(router.clone().oneshot(put(true)).await.unwrap().status().as_u16(), 200);
    // The marker is an extension, not a header: a client header does nothing.
    let spoofed = Request::builder().method("PUT").uri("/spoofed").header("x-rustic-internal-system", "true").body(Body::from("{}")).unwrap();
    assert_eq!(router.oneshot(spoofed).await.unwrap().status().as_u16(), 401);
}

#[tokio::test]
async fn prepared_bulk_items_answer_like_the_body() {
    use elasticsearch_http::{PreparedBulk, PreparedBulkItem};
    let body = concat!(
        "{\"index\":{\"_index\":\"pb\",\"_id\":\"1\"}}\n{\"f\":1}\n",
        "{\"index\":{\"_index\":\"pb\",\"_id\":\"2\",\"op_type\":\"create\"}}\n{\"f\":2}\n",
        "{\"create\":{\"_index\":\"pb\",\"_id\":\"2\"}}\n{\"f\":3}\n",
        "{\"update\":{\"_index\":\"pb\",\"_id\":\"1\"}}\n{\"doc\":{\"g\":1}}\n",
        "{\"delete\":{\"_index\":\"pb\",\"_id\":\"2\"}}\n",
        "{\"delete\":{\"_index\":\"pb\",\"_id\":\"3\"}}\n",
    );
    let mut items = Vec::new();
    let mut lines = body.lines();
    while let Some(line) = lines.next() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let (action, meta) = v.as_object().unwrap().iter().next().unwrap();
        let source = (action != "delete").then(|| lines.next().unwrap().to_string());
        items.push(PreparedBulkItem { action: action.clone(), meta: meta.as_object().unwrap().clone(), source });
    }
    let mut answers = Vec::new();
    for prepared in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let router = elasticsearch_http::router(node.clone());
        let request = Request::builder().method("POST").uri("/_bulk").header("content-type", "application/x-ndjson");
        let mut request = if prepared { request.body(Body::empty()).unwrap() } else { request.body(Body::from(body)).unwrap() };
        if prepared {
            request.extensions_mut().insert(PreparedBulk::new(items.clone(), body.len() as u64));
        }
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status().as_u16(), 200, "prepared={prepared}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        v.as_object_mut().unwrap().remove("took");
        // Each fresh node creates a different index UUID. Validate the actual
        // identity before normalizing only that nondeterministic field; keep
        // every error/status/detail in the prepared-vs-body comparison.
        let index = node.get_index("pb").expect("bulk created pb");
        assert!(!index.uuid.is_empty());
        let conflict = &mut v["items"][2]["create"];
        assert_eq!(conflict["status"], 409, "prepared={prepared}");
        assert_eq!(conflict["error"]["index"], "pb");
        assert_eq!(conflict["error"]["index_uuid"].as_str(), Some(index.uuid.as_str()), "prepared={prepared}");
        conflict["error"]["index_uuid"] = json!("<validated-pb-uuid>");
        answers.push(v);
    }
    assert_eq!(answers[0]["items"].as_array().unwrap().len(), 6, "{}", answers[0]);
    assert_eq!(answers[0], answers[1]);
}

#[tokio::test]
async fn prepared_bulk_items_are_authorized_like_the_body() {
    use elasticsearch_http::{PreparedBulk, PreparedBulkItem};
    let body = concat!(
        "{\"index\":{\"_index\":\"pb\",\"_id\":\"1\"}}\n{\"f\":1}\n",
        "{\"index\":{\"_index\":\"other\",\"_id\":\"2\"}}\n{\"f\":2}\n",
        "{\"delete\":{\"_index\":\"other\",\"_id\":\"3\"}}\n",
        "{\"create\":{\"_index\":\"pb\",\"_id\":\"4\"}}\n{\"f\":4}\n",
    );
    let mut items = Vec::new();
    let mut lines = body.lines();
    while let Some(line) = lines.next() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let (action, meta) = v.as_object().unwrap().iter().next().unwrap();
        let source = (action != "delete").then(|| lines.next().unwrap().to_string());
        items.push(PreparedBulkItem { action: action.clone(), meta: meta.as_object().unwrap().clone(), source });
    }
    let auth = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode("writer:password"));
    let mut answers = Vec::new();
    for prepared in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let mut cfg = SecConfig::default();
        cfg.enabled = true;
        cfg.file_users.insert("writer".into(), Hasher::Pbkdf2(1000).hash("password"));
        cfg.file_user_roles.insert("writer".into(), vec!["writer".into()]);
        cfg.file_roles.insert("writer".into(), json!({"cluster": [], "indices": [{"names": ["pb"], "privileges": ["write", "create_index"]}]}));
        node.security.configure(cfg);
        let router = elasticsearch_http::router(node);
        let request = Request::builder().method("POST").uri("/_bulk").header("content-type", "application/x-ndjson").header("Authorization", auth.clone());
        let mut request = if prepared { request.body(Body::empty()).unwrap() } else { request.body(Body::from(body)).unwrap() };
        if prepared {
            request.extensions_mut().insert(PreparedBulk::new(items.clone(), body.len() as u64));
        }
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status().as_u16(), 200, "prepared={prepared}");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        v.as_object_mut().unwrap().remove("took");
        answers.push(v);
    }
    let items = answers[0]["items"].as_array().unwrap();
    assert_eq!(items.len(), 4, "{}", answers[0]);
    assert_eq!(items[1]["index"]["status"], 403, "{}", answers[0]);
    assert_eq!(items[3]["create"]["status"], 201, "{}", answers[0]);
    assert_eq!(answers[0], answers[1]);
}
