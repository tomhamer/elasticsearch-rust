//! `/_license` (get / put / delete, `start_trial`, `start_basic`,
//! `trial_status`, `basic_status`) and `GET /_xpack` (info) over
//! `elasticsearch_engine::license`.

use crate::{blocking, method_not_allowed, Req, Resp, RespBody, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::license::{self as lic, Mode, Outcome};
use elasticsearch_engine::node::{Node, ES_VERSION};
use serde_json::{json, Map, Value};
use std::sync::Arc;

const ACCEPT_ENTERPRISE_DEPRECATION: &str = "Including [accept_enterprise] in get license requests is deprecated. The parameter will be removed in the next major version";

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let a = req.seg(0).to_string();
    let b = req.seg(1).to_string();
    if req.segs.len() > 2 {
        return crate::not_found_handler(&req);
    }
    let m = req.method.clone();
    match (a.as_str(), b.as_str()) {
        ("_xpack", "") => match m {
            Method::GET => blocking(move || xpack_info(&node, &req)).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("_license", "") => match m {
            Method::GET => blocking(move || get_license(&node, &req)).await,
            Method::PUT | Method::POST => blocking(move || put_license(&node, &req)).await,
            Method::DELETE => blocking(move || {
                lic::service(&node).start_basic(&node, true);
                Ok(Resp::ack())
            })
            .await,
            _ => method_not_allowed(&req, &["DELETE", "POST", "GET", "PUT"]),
        },
        // Licensed-feature usage tracking (`XPackLicenseState` feature
        // trackers): no licensed feature reports usage on this node yet.
        ("_license", "feature_usage") => match m {
            Method::GET => Ok(Resp::ok(json!({"features": []}))),
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("_license", "trial_status") => match m {
            Method::GET => blocking(move || Ok(Resp::ok(json!({"eligible_to_start_trial": lic::service(&node).eligible_for_trial()})))).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("_license", "basic_status") => match m {
            Method::GET => blocking(move || Ok(Resp::ok(json!({"eligible_to_start_basic": lic::service(&node).eligible_for_basic(&node)})))).await,
            _ => method_not_allowed(&req, &["GET"]),
        },
        ("_license", "start_trial") => match m {
            Method::POST => blocking(move || start_trial(&node, &req)).await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        ("_license", "start_basic") => match m {
            Method::POST => blocking(move || start_basic(&node, &req)).await,
            _ => method_not_allowed(&req, &["POST"]),
        },
        _ => crate::not_found_handler(&req),
    }
}

fn warning(msg: &str) -> (&'static str, String) {
    ("Warning", format!("299 Elasticsearch-{}-{} \"{}\"", ES_VERSION, crate::cluster::BUILD_HASH, msg))
}

/// `accept_enterprise`: deprecated, only `true` is accepted.
fn accept_enterprise(req: &Req) -> Result<Option<(&'static str, String)>, EsError> {
    match req.bool_param("accept_enterprise")? {
        None => Ok(None),
        Some(true) => Ok(Some(warning(ACCEPT_ENTERPRISE_DEPRECATION))),
        Some(false) => Err(EsError::illegal_argument("The [accept_enterprise] parameters may not be false")),
    }
}

/// `GET /_license`: ES always pretty-prints this response.
fn get_license(node: &Node, req: &Req) -> R {
    let warn = accept_enterprise(req)?;
    let l = lic::current(node);
    let body = json!({"license": l.to_rest(elasticsearch_engine::date::now_millis())});
    let mut s = es_pretty(&body, 0);
    s.push('\n');
    let mut r = Resp { status: 200, body: RespBody::Raw(s), headers: Vec::new() };
    r.headers.extend(warn);
    Ok(r)
}

fn put_license(node: &Node, req: &Req) -> R {
    if req.body.is_empty() || req.body_str().map(|s| s.trim().is_empty()).unwrap_or(false) {
        return Err(EsError::illegal_argument("The license must be provided in the request body"));
    }
    let ack = req.flag("acknowledge")?;
    let body = req.json()?;
    let lv = match (body.get("license"), body.get("licenses")) {
        (Some(l), _) if l.is_object() => l.clone(),
        (_, Some(Value::Array(a))) => match a.last() {
            Some(l) => l.clone(),
            None => {
                return Err(EsError::new(
                    500,
                    "null_pointer_exception",
                    "Cannot invoke \"org.elasticsearch.license.License.type()\" because the return value of \"org.elasticsearch.license.PutLicenseRequest.license()\" is null",
                ))
            }
        },
        _ => return Err(EsError::new(400, "parse_exception", "failed to parse licenses expected field")),
    };
    let l = lic::parse_license(&lv)?;
    // Licensing is hardwired to enterprise/trial; see AGENTS.md "Licensing:
    // always enterprise". A basic license
    // is acknowledged like any other and, like any other, not applied.
    if l.kind.eq_ignore_ascii_case("basic") {
        return Ok(Resp::ok(json!({"acknowledged": true, "license_status": "valid"})));
    }
    let out = match lic::service(node).put(l, ack) {
        Outcome::Done => json!({"acknowledged": true, "license_status": "valid"}),
        Outcome::Expired => json!({"acknowledged": true, "license_status": "expired"}),
        Outcome::Invalid => json!({"acknowledged": true, "license_status": "invalid"}),
        Outcome::NeedsAck(msgs) => json!({
            "acknowledged": false,
            "license_status": "valid",
            "acknowledge": ack_object("This license update requires acknowledgement. To acknowledge the license, please read the following messages and update the license again, this time with the \"acknowledge=true\" parameter:", msgs)
        }),
        Outcome::TrialAlreadyActivated | Outcome::AlreadyBasic => json!({"acknowledged": true, "license_status": "valid"}),
    };
    Ok(Resp::ok(out))
}

fn ack_object(message: &str, msgs: Vec<(&'static str, Vec<String>)>) -> Value {
    let mut o = Map::new();
    o.insert("message".into(), json!(message));
    for (k, v) in msgs {
        o.insert(k.into(), json!(v));
    }
    Value::Object(o)
}

fn start_trial(node: &Node, req: &Req) -> R {
    let kind = req.param("type").unwrap_or("trial").to_string();
    let kl = kind.to_lowercase();
    if !lic::LICENSE_TYPES.contains(&kl.as_str()) {
        return Err(EsError::illegal_argument(format!(
            "unrecognised license type [ {kind}], supported license types are [{}]",
            lic::LICENSE_TYPES.join(",")
        )));
    }
    if !lic::TRIAL_TYPES.contains(&kl.as_str()) {
        return Err(EsError::illegal_argument(format!("Cannot start trial of type [{kl}]. Valid trial types are [{}]", lic::TRIAL_TYPES.join(","))));
    }
    let ack = req.flag("acknowledge")?;
    Ok(match lic::service(node).start_trial(node, &kl, ack) {
        Outcome::Done => Resp::ok(json!({"acknowledged": true, "trial_was_started": true, "type": kl})),
        Outcome::NeedsAck(msgs) => Resp::ok(json!({
            "acknowledged": false,
            "trial_was_started": false,
            "error_message": "Operation failed: Needs acknowledgement.",
            "acknowledge": ack_object("This API initiates a free 30-day trial for all subscription features. By starting this trial, you agree that it is subject to the terms and conditions at https://www.elastic.co/legal/trial_license/. To begin your free trial, call /start_trial again and specify the \"acknowledge=true\" parameter.", msgs)
        })),
        _ => Resp::json(403, json!({"acknowledged": true, "trial_was_started": false, "error_message": "Operation failed: Trial was already activated."})),
    })
}

fn start_basic(node: &Node, req: &Req) -> R {
    let ack = req.flag("acknowledge")?;
    Ok(match lic::service(node).start_basic(node, ack) {
        Outcome::Done => Resp::ok(json!({"acknowledged": true, "basic_was_started": true})),
        Outcome::NeedsAck(msgs) => Resp::ok(json!({
            "acknowledged": false,
            "basic_was_started": false,
            "error_message": "Operation failed: Needs acknowledgement.",
            "acknowledge": ack_object("This license update requires acknowledgement. To acknowledge the license, please read the following messages and call /start_basic again, this time with the \"acknowledge=true\" parameter:", msgs)
        })),
        _ => Resp::json(403, json!({"acknowledged": true, "basic_was_started": false, "error_message": "Operation failed: Current license is basic."})),
    })
}

/// `GET /_xpack`: build, license and feature availability (from the
/// license mode, as `XPackLicenseState` decides it).
fn xpack_info(node: &Node, req: &Req) -> R {
    let warn = accept_enterprise(req)?;
    let (mut build, mut license, mut features) = (true, true, true);
    if let Some(c) = req.param("categories") {
        build = false;
        license = false;
        features = false;
        for p in c.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match p.to_uppercase().as_str() {
                "_ALL" => (build, license, features) = (true, true, true),
                "_NONE" => {}
                "BUILD" => build = true,
                "LICENSE" => license = true,
                "FEATURES" => features = true,
                other => {
                    return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.protocol.xpack.XPackInfoRequest.Category.{other}")));
                }
            }
        }
    }
    let human = req.bool_param("human")?.unwrap_or(true);
    let now = elasticsearch_engine::date::now_millis();
    let l = lic::current(node);
    let mut out = Map::new();
    if build {
        out.insert("build".into(), json!({"hash": crate::cluster::BUILD_HASH, "date": crate::cluster::BUILD_DATE}));
    }
    if license {
        out.insert("license".into(), l.to_xpack_info(now));
    }
    if features {
        let mode = l.effective_mode(now);
        let mut f = Map::new();
        for (name, min) in lic::XPACK_FEATURES {
            f.insert((*name).into(), json!({"available": mode.allows(*min), "enabled": feature_enabled(node, name, mode)}));
        }
        out.insert("features".into(), Value::Object(f));
    }
    if human {
        out.insert("tagline".into(), json!("You know, for X"));
    }
    let mut r = Resp::ok(Value::Object(out));
    r.headers.extend(warn);
    Ok(r)
}

fn feature_enabled(node: &Node, name: &str, _mode: Mode) -> bool {
    match name {
        "security" => node.security.enabled(),
        "watcher" => elasticsearch_engine::watcher::watcher_enabled(node),
        // `GpuInfoTransportAction`: a supported GPU, not `use_gpu: false`.
        "gpu_vector_indexing" => elasticsearch_engine::gpu::info_enabled(node),
        "logsdb" => elasticsearch_engine::datastream::cluster_logsdb_enabled(node),
        _ => true,
    }
}

/// Jackson-style pretty printing (`"key" : value`, two-space indent), as
/// ES renders responses it always pretty-prints.
pub fn es_pretty(v: &Value, indent: usize) -> String {
    let pad = |n: usize| " ".repeat(n);
    match v {
        Value::Object(m) if m.is_empty() => "{ }".into(),
        Value::Object(m) => {
            let items: Vec<String> = m.iter().map(|(k, x)| format!("{}{} : {}", pad(indent + 2), Value::String(k.clone()), es_pretty(x, indent + 2))).collect();
            format!("{{\n{}\n{}}}", items.join(",\n"), pad(indent))
        }
        Value::Array(a) if a.is_empty() => "[ ]".into(),
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(|x| format!("{}{}", pad(indent + 2), es_pretty(x, indent + 2))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), pad(indent))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretty() {
        let v = json!({"license": {"status": "active", "max_nodes": 1000, "max_resource_units": null}});
        assert_eq!(es_pretty(&v, 0), "{\n  \"license\" : {\n    \"status\" : \"active\",\n    \"max_nodes\" : 1000,\n    \"max_resource_units\" : null\n  }\n}");
    }
}
