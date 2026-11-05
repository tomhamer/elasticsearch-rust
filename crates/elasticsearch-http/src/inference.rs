//! `/_inference` REST handlers over `elasticsearch_engine::inference`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::inference::{service_for, InferenceService};
use elasticsearch_engine::node::Node;
use serde_json::{json, Value};
use std::sync::Arc;

fn body_obj(req: &Req) -> Result<Value, EsError> {
    let b = req.json()?;
    Ok(if b.is_null() { json!({}) } else { b })
}

fn sse(result: &Value) -> Resp {
    let mut out = String::new();
    if let Some(c) = result["completion"].as_array() {
        for r in c {
            out.push_str(&format!("event: message\ndata: {}\n\n", json!({"completion": [r]})));
        }
    } else {
        out.push_str(&format!("event: message\ndata: {result}\n\n"));
    }
    out.push_str("event: message\ndata: [DONE]\n\n");
    let mut r = Resp::text(out);
    r.headers.push(("content-type", "text/event-stream".to_string()));
    r
}

/// ES reports the *inner* cause as the root cause of an endpoint-creation
/// validation failure; the generic error renderer reports the outer one.
fn fix_root(r: R) -> R {
    match r {
        Err(e) if e.kind == "status_exception" && e.caused_by.is_some() => {
            let mut inner = &e;
            while let Some(c) = &inner.caused_by {
                if c.kind != "status_exception" {
                    break;
                }
                inner = c;
            }
            let (kind, reason) = (inner.kind.clone(), inner.reason.clone());
            let mut body = e.to_json();
            body["error"]["root_cause"] = json!([{"type": kind, "reason": reason}]);
            Ok(Resp::json(e.status, body))
        }
        other => other,
    }
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let segs = req.segs.clone();
    let n = segs.len();
    let s = |i: usize| segs.get(i).cloned().unwrap_or_default();
    let svc = service_for(&node.data);
    let m = req.method.clone();
    // /_inference
    if n == 1 {
        return match m {
            Method::GET => blocking(move || Ok(Resp::ok(svc.get(None, None)?))).await,
            _ => method_not_allowed(&req, &["GET"]),
        };
    }
    // /_inference/_services[/{task}]
    if s(1) == "_services" && n <= 3 {
        return match m {
            Method::GET => {
                let t = (n == 3).then(|| s(2));
                blocking(move || Ok(Resp::ok(InferenceService::services(t.as_deref())?))).await
            }
            _ => method_not_allowed(&req, &["GET"]),
        };
    }
    // /_inference/_region_policy (`Rest{Get,Put,Delete}RegionPolicyAction`).
    if s(1) == "_region_policy" && n == 2 {
        use elasticsearch_engine::inference::region_policy;
        let dir = svc.state_dir().to_path_buf();
        return match m {
            Method::GET => blocking(move || Ok(Resp::ok(region_policy::get(&dir)?))).await,
            Method::DELETE => blocking(move || Ok(Resp::ok(region_policy::delete(&dir)?))).await,
            Method::PUT => {
                if req.body_str()?.trim().is_empty() {
                    return Err(EsError::new(400, "parse_exception", "body is required"));
                }
                req.flag("force")?;
                let body = req.json()?;
                let user = req.access.as_ref().map(|a| a.user().username.clone());
                blocking(move || Ok(Resp::ok(region_policy::put(&dir, &body, user.as_deref())?))).await
            }
            _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
        };
    }
    // Creating and calling endpoints needs an enterprise (or trial) license.
    if matches!(m, Method::PUT | Method::POST) {
        elasticsearch_engine::license::check(&node, "inference", elasticsearch_engine::license::Mode::Enterprise)?;
    }
    // streaming: /_inference/{id}/_stream, /_inference/{task}/{id}/_stream
    if s(n - 1) == "_stream" && (n == 3 || n == 4) {
        if m != Method::POST && m != Method::GET {
            return method_not_allowed(&req, &["GET", "POST"]);
        }
        let (task, id) = if n == 3 { (None, s(1)) } else { (Some(s(1)), s(2)) };
        let body = body_obj(&req)?;
        return blocking(move || {
            let res = svc.infer(task.as_deref(), &id, &body)?;
            Ok(sse(&res))
        })
        .await;
    }
    // /_inference/{id}/_update, /_inference/{task}/{id}/_update
    if s(n - 1) == "_update" && (n == 3 || n == 4) {
        if m != Method::PUT {
            return method_not_allowed(&req, &["PUT"]);
        }
        let (task, id) = if n == 3 { (None, s(1)) } else { (Some(s(1)), s(2)) };
        if req.body_str()?.trim().is_empty() {
            return Err(EsError::new(400, "parse_exception", "request body is required"));
        }
        let body = body_obj(&req)?;
        return blocking(move || Ok(Resp::ok(svc.update(task.as_deref(), &id, &body)?))).await;
    }
    let (task, id) = if n == 2 { (None, s(1)) } else if n == 3 { (Some(s(1)), s(2)) } else { return crate::not_found_handler(&req) };
    match m {
        Method::GET => blocking(move || Ok(Resp::ok(svc.get(task.as_deref(), Some(&id))?))).await,
        Method::PUT => {
            if req.body_str()?.trim().is_empty() {
                return Err(EsError::new(400, "parse_exception", "request body is required"));
            }
            let body = body_obj(&req)?;
            fix_root(
                blocking(move || {
                    let out = svc.put_resolving(task.as_deref(), &id, &body)?;
                    // `TransportPutInferenceModelAction`: semantic_text mappings
                    // that already use this id must keep compatible settings.
                    let ms = svc.get_endpoint(&id).and_then(|ep| elasticsearch_engine::inference::model_settings(&ep));
                    let mut bad: Vec<String> = Vec::new();
                    for ix in node.all_indices() {
                        let lk = ix.lookup();
                        let clash = lk.fields.values().any(|f| {
                            if !f.declared.is_semantic() {
                                return false;
                            }
                            let Some(p) = f.ext.as_ref().map(|e| e.params.clone()) else { return false };
                            if p.get("inference_id").and_then(Value::as_str) != Some(id.as_str()) {
                                return false;
                            }
                            match (p.get("model_settings").filter(|v| !v.is_null()), &ms) {
                                (Some(a), Some(b)) => ["task_type", "dimensions", "similarity", "element_type"].iter().any(|k| a.get(*k) != b.get(*k)),
                                (Some(_), None) => true,
                                _ => false,
                            }
                        });
                        if clash {
                            bad.push(ix.name.clone());
                        }
                    }
                    if !bad.is_empty() {
                        bad.sort();
                        let _ = svc.delete(None, &id, false, true);
                        return Err(EsError::new(400, "status_exception", format!(
                            "Inference endpoint [{id}] could not be created because the inference_id is being used in mappings with incompatible settings for indices: [{}]. Please either use a different inference_id or update the index mappings to refer to a different inference_id.",
                            bad.join(", ")
                        )));
                    }
                    Ok(Resp::ok(out))
                })
                .await,
            )
        }
        Method::DELETE => {
            let dry = req.flag("dry_run")?;
            let force = req.flag("force")?;
            blocking(move || Ok(Resp::ok(svc.delete(task.as_deref(), &id, dry, force)?))).await
        }
        Method::POST => {
            let body = body_obj(&req)?;
            blocking(move || Ok(Resp::ok(svc.infer(task.as_deref(), &id, &body)?))).await
        }
        _ => method_not_allowed(&req, &["GET", "PUT", "POST", "DELETE"]),
    }
}
