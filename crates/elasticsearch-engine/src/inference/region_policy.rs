//! The inference region policy (`/_inference/_region_policy`, ES 9.5):
//! one document (ES keeps it in the `.inference` index under id
//! `region_policy_elastic`) restricting which cloud regions / geographies
//! the Elastic Inference Service may route to. We persist it next to the
//! endpoint registry (`<state>/inference_region_policy.json`).

use crate::error::{EsError, Result};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

fn file(state_dir: &Path) -> PathBuf {
    state_dir.join("inference_region_policy.json")
}

fn load(state_dir: &Path) -> Option<Value> {
    std::fs::read(file(state_dir)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

/// `TransportGetRegionPolicyAction.noRegionPolicyConfiguredException`.
pub fn not_configured() -> EsError {
    EsError::resource_not_found("No region policy is configured for this deployment")
}

fn xparse(pos: &str, parser: &str, reason: impl std::fmt::Display) -> EsError {
    EsError::x_content_parse(format!("[{pos}] [{parser}] {reason}"))
}

fn field_failure(parser: &str, field: &str, cause: EsError) -> EsError {
    xparse("1:2", parser, format!("failed to parse field [{field}]")).caused_by(cause)
}

/// `CspRegion.STRICT_PARSER`.
fn parse_region(v: &Value) -> Result<Value> {
    let o = v.as_object().ok_or_else(|| xparse("1:2", "csp_region", "expected an object"))?;
    let mut out = Map::new();
    for (k, x) in o {
        match k.as_str() {
            "csp" | "region" => {
                let s = match x {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => return Err(xparse("1:2", "csp_region", format!("[{k}] doesn't support values of type: {}", if x.is_null() { "VALUE_NULL" } else { "START_OBJECT" }))),
                };
                out.insert(k.clone(), json!(s));
            }
            _ => return Err(xparse("1:2", "csp_region", format!("unknown field [{k}]"))),
        }
    }
    for req in ["csp", "region"] {
        if !out.contains_key(req) {
            return Err(EsError::illegal_argument(format!("Required [{req}]")));
        }
    }
    // Render in declaration order (`csp`, `region`).
    Ok(json!({"csp": out["csp"], "region": out["region"]}))
}

/// `RegionPolicy.STRICT_PARSER`.
fn parse_policy(v: &Value) -> Result<Value> {
    let o = v.as_object().ok_or_else(|| xparse("1:2", "region_policy", "expected an object"))?;
    let mut out = Map::new();
    for (k, x) in o {
        match k.as_str() {
            "allowed_geos" => {
                let a = x.as_array().ok_or_else(|| field_failure("region_policy", k, EsError::illegal_argument("expected an array")))?;
                let geos: Vec<Value> = a
                    .iter()
                    .map(|g| match g {
                        Value::String(s) => Ok(json!(s)),
                        Value::Number(n) => Ok(json!(n.to_string())),
                        Value::Bool(b) => Ok(json!(b.to_string())),
                        _ => Err(field_failure("region_policy", k, EsError::illegal_argument("expected a string"))),
                    })
                    .collect::<Result<_>>()?;
                out.insert(k.clone(), Value::Array(geos));
            }
            "allowed_regions" => {
                let a = x.as_array().ok_or_else(|| field_failure("region_policy", k, EsError::illegal_argument("expected an array")))?;
                let regions: Vec<Value> = a.iter().map(parse_region).collect::<Result<_>>().map_err(|e| field_failure("region_policy", k, e))?;
                out.insert(k.clone(), Value::Array(regions));
            }
            _ => return Err(xparse("1:2", "region_policy", format!("unknown field [{k}]"))),
        }
    }
    if out.contains_key("allowed_geos") && out.contains_key("allowed_regions") {
        return Err(EsError::illegal_argument("The following fields are not allowed together: [allowed_geos, allowed_regions] "));
    }
    // `toXContent` order: `allowed_geos`, then `allowed_regions`.
    let mut ordered = Map::new();
    for k in ["allowed_geos", "allowed_regions"] {
        if let Some(x) = out.remove(k) {
            ordered.insert(k.into(), x);
        }
    }
    Ok(Value::Object(ordered))
}

/// `PutRegionPolicyAction.Request.parseRequest` + `validate`.
pub fn parse_request(body: &Value) -> Result<Value> {
    let o = body.as_object().ok_or_else(|| xparse("1:1", "put_region_policy_request", "expected an object"))?;
    let mut policy = None;
    for (k, v) in o {
        match k.as_str() {
            "region_policy" => policy = Some(parse_policy(v).map_err(|e| field_failure("put_region_policy_request", "region_policy", e))?),
            _ => return Err(xparse("1:2", "put_region_policy_request", format!("unknown field [{k}]"))),
        }
    }
    let policy = policy.ok_or_else(|| EsError::illegal_argument("Required [region_policy]"))?;
    let mut errs = Vec::new();
    for k in ["allowed_regions", "allowed_geos"] {
        if policy.get(k).and_then(Value::as_array).is_some_and(Vec::is_empty) {
            errs.push(format!("[region_policy.{k}] must not be empty"));
        }
    }
    if !errs.is_empty() {
        let r: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&r));
    }
    Ok(policy)
}

fn now_instant() -> String {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as i128).unwrap_or(0);
    crate::tsdb::instant_string(nanos)
}

/// `RegionPolicyDoc.toXContent` (the API response shape).
fn render(doc: &Value) -> Value {
    let mut m = Map::new();
    for k in ["region_policy", "created_at", "created_by", "updated_at", "updated_by"] {
        if let Some(v) = doc.get(k).filter(|v| !v.is_null()) {
            m.insert(k.into(), v.clone());
        }
    }
    Value::Object(m)
}

/// `GET /_inference/_region_policy`.
pub fn get(state_dir: &Path) -> Result<Value> {
    load(state_dir).map(|d| render(&d)).ok_or_else(not_configured)
}

/// `PUT /_inference/_region_policy`: create, or replace keeping
/// `created_at` / `created_by` and stamping `updated_at` / `updated_by`.
pub fn put(state_dir: &Path, body: &Value, user: Option<&str>) -> Result<Value> {
    let policy = parse_request(body)?;
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let doc = match load(state_dir) {
        None => json!({"region_policy": policy, "created_at": now_instant(), "created_by": user}),
        Some(old) => json!({
            "region_policy": policy,
            "created_at": old["created_at"],
            "created_by": old["created_by"],
            "updated_at": now_instant(),
            "updated_by": user,
        }),
    };
    std::fs::create_dir_all(state_dir)?;
    std::fs::write(file(state_dir), serde_json::to_vec(&doc)?)?;
    Ok(render(&doc))
}

/// `DELETE /_inference/_region_policy`.
pub fn delete(state_dir: &Path) -> Result<Value> {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match std::fs::remove_file(file(state_dir)) {
        Ok(()) => Ok(json!({"acknowledged": true})),
        Err(_) => Err(not_configured()),
    }
}
