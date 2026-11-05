//! Universal Profiling APIs (`/_profiling/...`) over
//! `elasticsearch_engine::profiling`: `status`, `stacktraces`,
//! `flamegraph`, `topn/functions`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::profiling::{self, costs, graph, stacktraces};
use std::sync::Arc;
use std::time::Duration;

/// The instance cost tables ES ships with the plugin.
fn ensure_costs() {
    if costs::costs_installed() {
        return;
    }
    fn gunzip(b: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut out = Vec::new();
        let _ = flate2::read::GzDecoder::new(b).read_to_end(&mut out);
        out
    }
    let aws = gunzip(include_bytes!("../../elasticsearch-engine/src/profiling/resources/profiling-costs-aws.json.gz"));
    let azure = gunzip(include_bytes!("../../elasticsearch-engine/src/profiling/resources/profiling-costs-azure.json.gz"));
    costs::install_costs(&[("aws", &aws), ("azure", &azure)]);
}

pub(crate) fn status_options(path: &str, params: &std::collections::HashMap<String, String>) -> Result<(bool, Duration), EsError> {
    crate::util::check_params_map(path, params, &["master_timeout", "timeout", "wait_for_resources_created"])?;
    let wait = match params.get("wait_for_resources_created").map(String::as_str) {
        None | Some("false") => false,
        Some("true" | "") => true,
        Some(value) => return Err(EsError::illegal_argument(format!("Failed to parse value [{value}] as only [true] or [false] are allowed."))),
    };
    let time = |key: &str| -> Result<Duration, EsError> {
        match params.get(key) {
            None => Ok(Duration::from_secs(30)),
            Some(value) => elasticsearch_engine::date::parse_time_value(value)
                .map(|ms| Duration::from_millis(ms.max(0) as u64))
                .ok_or_else(|| EsError::parsing(format!("failed to parse setting [{key}] with value [{value}] as a time value: unit is missing or unrecognized"))),
        }
    };
    time("master_timeout")?;
    Ok((wait, time("timeout")?))
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let path: Vec<&str> = req.segs.iter().skip(1).map(String::as_str).collect();
    match path.as_slice() {
        ["status"] => {
            if req.method != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            let (wait, timeout) = status_options(&req.path, &req.params)?;
            // On a cluster `has_data` searches every shard through the
            // coordinator (as the profiling origin).
            let rest = req.internal_cluster_rest();
            blocking(move || {
                let data = rest.map(|rest| move || {
                    let uri = format!("/{}/_search?ignore_unavailable=true&allow_no_indices=true", profiling::ALL_EVENTS);
                    (rest.0)("POST", &uri, Some(serde_json::json!({"size": 0, "track_total_hits": 1})))
                        .is_ok_and(|v| v["hits"]["total"]["value"].as_i64().unwrap_or(0) > 0)
                });
                let (status, body) = profiling::status_with(&node, wait, timeout, data.as_ref().map(|f| f as &dyn Fn() -> bool))?;
                Ok(Resp::json(status, body))
            })
            .await
        }
        ["stacktraces"] | ["flamegraph"] | ["topn", "functions"] => {
            if req.method != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            crate::util::check_params(&req, &[])?;
            let kind = path.join("/");
            // On a cluster its reads go through the coordinator, as the caller.
            let rest: Option<elasticsearch_engine::transform::RestCall> = req.extensions.get::<crate::ClusterRest>().map(|r| r.0.clone());
            blocking(move || {
                let body = if req.body.is_empty() { None } else { Some(req.body_str()?) };
                let mut r = stacktraces::parse_request(body)?;
                r.adjust_sample_count = kind != "stacktraces";
                stacktraces::validate(&r)?;
                ensure_costs();
                let resp = stacktraces::execute_on_cluster(&node, &r, rest)?;
                Ok(Resp::ok(match kind.as_str() {
                    "stacktraces" => resp.to_json(),
                    "flamegraph" => graph::flamegraph(&resp),
                    _ => graph::topn_functions(&resp, r.limit),
                }))
            })
            .await
        }
        _ => crate::not_found_handler(&req),
    }
}
