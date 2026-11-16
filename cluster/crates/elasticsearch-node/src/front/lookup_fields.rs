//! Runtime `lookup` fields of a distributed search, resolved on the
//! coordinator (ES `FetchLookupFieldsPhase` runs on the coordinating node):
//! the shards return the input field's values and the coordinator runs one
//! `term` search per distinct input value on the target index through the
//! ordinary search route, as the caller, filling the lookup field of every
//! hit. A shard cannot answer it: the target index's documents may live on
//! other nodes.
use super::*;
use serde_json::Map;

/// Marks the planned search [`run`] sends back through `read`: its lookup
/// fields are resolved here, so a `fields` pattern still matching a mapped
/// lookup field must not plan it again.
pub(super) const PLANNED_HEADER: &str = "x-rustic-internal-lookup-planned";

/// Whether `req` is a search [`run`] already planned.
pub(super) fn planned(req: &RestReq) -> bool {
    req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(PLANNED_HEADER))
}

/// One requested `lookup` runtime field.
#[derive(Clone)]
struct LookupField {
    name: String,
    target_index: String,
    input_field: String,
    target_field: String,
    fetch_fields: Vec<Value>,
}

/// A search whose `fields` ask for `lookup` runtime fields (request-level
/// `runtime_mappings` or the target indices' mapped `runtime` section):
/// the body the shards run (lookup fields removed, their input fields
/// requested), the lookups, and the input fields to drop from the hits.
pub(super) struct Plan {
    body: Value,
    lookups: Vec<LookupField>,
    drop_inputs: Vec<String>,
}

/// `fields` entry name (`"f"` or `{"field": "f", ...}`).
fn entry_name(e: &Value) -> Option<&str> {
    e.as_str().or_else(|| e.get("field").and_then(Value::as_str))
}

fn glob(p: &str, s: &str) -> bool {
    match p.split_once('*') {
        None => p == s,
        Some((head, tail)) => s.strip_prefix(head).is_some_and(|rest| (0..=rest.len()).filter(|i| rest.is_char_boundary(*i)).any(|i| glob(tail, &rest[i..]))),
    }
}

/// The `lookup` runtime fields mapped on `indices` (each index mapping's
/// `runtime` section; the first definition of a name wins).
pub(super) fn mapped_lookups(state: &elasticsearch_cluster::state::ClusterState, indices: &BTreeSet<String>) -> Map<String, Value> {
    let mut out = Map::new();
    for name in indices {
        let Some(md) = state.metadata.indices.get(name) else { continue };
        let m: &Value = &md.mappings;
        let runtime = m.get("runtime").or_else(|| m.get("_doc").and_then(|d| d.get("runtime")));
        for (k, v) in runtime.and_then(Value::as_object).into_iter().flatten() {
            if v.get("type").and_then(Value::as_str) == Some("lookup") && !out.contains_key(k) {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    out
}

/// `mapped`: the targets' mapped lookup fields ([`mapped_lookups`]); a
/// request-level runtime field of the same name shadows one.
pub(super) fn plan(body: &Value, mapped: &Map<String, Value>) -> Option<Plan> {
    let empty = Map::new();
    let rm = body.get("runtime_mappings").and_then(Value::as_object).unwrap_or(&empty);
    let mut defs: Map<String, Value> = mapped.iter().filter(|(k, _)| !rm.contains_key(*k)).map(|(k, v)| (k.clone(), v.clone())).collect();
    defs.extend(rm.iter().filter(|(_, v)| v.get("type").and_then(Value::as_str) == Some("lookup")).map(|(k, v)| (k.clone(), v.clone())));
    let lookup_names: Vec<&String> = defs.keys().collect();
    if lookup_names.is_empty() {
        return None;
    }
    let rm = &defs;
    let requested: Vec<Value> = match body.get("fields") {
        Some(Value::Array(a)) => a.clone(),
        Some(v) => vec![v.clone()],
        None => Vec::new(),
    };
    let asked = |name: &str| requested.iter().filter_map(entry_name).any(|p| glob(p, name));
    let mut lookups = Vec::new();
    for name in &lookup_names {
        if !asked(name) {
            continue;
        }
        let spec = &rm[name.as_str()];
        let s = |k: &str| spec.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        lookups.push(LookupField {
            name: (*name).clone(),
            target_index: s("target_index"),
            input_field: s("input_field"),
            target_field: s("target_field"),
            fetch_fields: spec.get("fetch_fields").and_then(Value::as_array).cloned().unwrap_or_default(),
        });
    }
    if lookups.is_empty() {
        return None;
    }
    let mut out = body.clone();
    if let Some(m) = out.get_mut("runtime_mappings").and_then(Value::as_object_mut) {
        m.retain(|k, _| !lookup_names.contains(&k));
        if m.is_empty() {
            out.as_object_mut()?.remove("runtime_mappings");
        }
    }
    let mut fields: Vec<Value> = requested.iter().filter(|e| entry_name(e).is_none_or(|n| !lookup_names.iter().any(|l| l.as_str() == n))).cloned().collect();
    let mut drop_inputs = Vec::new();
    for l in &lookups {
        if !requested.iter().filter_map(entry_name).any(|p| glob(p, &l.input_field)) && !drop_inputs.contains(&l.input_field) {
            drop_inputs.push(l.input_field.clone());
            fields.push(json!(l.input_field));
        }
    }
    out["fields"] = Value::Array(fields);
    Some(Plan { body: out, lookups, drop_inputs })
}

/// Run the planned search, then resolve its hits' lookup fields.
pub(super) async fn run(node: &Arc<ClusterNode>, req: &RestReq, targets: &str, plan: Plan) -> RestResp {
    let mut rewritten = req.clone();
    if let Err(e) = rewritten.set_json(&plan.body).and_then(|_| rewritten.push_header(PLANNED_HEADER, "1")) {
        return RestResp::json(e.status, &e.to_json());
    }
    let mut resp = Box::pin(read(node, &rewritten, targets)).await;
    if !resp.is_success() {
        return resp;
    }
    let Some(mut v) = resp.body_json() else { return resp };
    let mut cache: std::collections::HashMap<(String, String, String), Vec<Value>> = HashMap::new();
    let hits_len = v["hits"]["hits"].as_array().map_or(0, Vec::len);
    for i in 0..hits_len {
        for l in &plan.lookups {
            let inputs: Vec<Value> = v["hits"]["hits"][i]["fields"][&l.input_field].as_array().cloned().unwrap_or_default();
            let mut found: Vec<Value> = Vec::new();
            for input in inputs {
                let key = (l.target_index.clone(), l.target_field.clone(), input.to_string());
                if !cache.contains_key(&key) {
                    match term_search(node, req, l, &input).await {
                        Ok(r) => {
                            cache.insert(key.clone(), r);
                        }
                        Err(e) => return e,
                    }
                }
                found.extend(cache[&key].iter().cloned());
            }
            let hit = &mut v["hits"]["hits"][i];
                if !found.is_empty() {
                if !hit.get("fields").is_some_and(Value::is_object) {
                    hit["fields"] = json!({});
                }
                hit["fields"][&l.name] = Value::Array(found);
            } else if let Some(f) = hit.get_mut("fields").and_then(Value::as_object_mut) {
                // A shard's own (local-only) answer for a mapped lookup
                // field a `fields` pattern matched never stands.
                f.remove(&l.name);
            }
        }
        let hit = &mut v["hits"]["hits"][i];
        if let Some(f) = hit.get_mut("fields").and_then(Value::as_object_mut) {
            f.retain(|k, _| !plan.drop_inputs.contains(k));
            if f.is_empty() {
                hit.as_object_mut().map(|h| h.remove("fields"));
            }
        }
    }
    resp.replace_json(&v);
    resp
}

/// ES `LookupField.toSearchRequest`: a `term` query on the target field,
/// size 1, the fetch fields through the fields API; the matching document's
/// fields as one object (nothing when it has none of them).
async fn term_search(node: &Arc<ClusterNode>, req: &RestReq, l: &LookupField, value: &Value) -> Result<Vec<Value>, RestResp> {
    let query = if l.target_field == "_id" { json!({"ids": {"values": [value]}}) } else { json!({"term": {l.target_field.clone(): value}}) };
    let mut search = req.clone();
    let fail = |e: elasticsearch_engine::error::EsError| RestResp::json(e.status, &e.to_json());
    search.set_method("POST").map_err(fail)?;
    search.set_uri(&format!("/{}/_search", encode_component(&l.target_index))).map_err(fail)?;
    search.set_json(&json!({"query": query, "size": 1, "_source": false, "fields": l.fetch_fields})).map_err(fail)?;
    search.retain_headers(|(name, _)| !name.eq_ignore_ascii_case(EXEC_LOCAL) && !name.eq_ignore_ascii_case(crate::shard_engines::SHARD_HEADER)).map_err(fail)?;
    let response = Box::pin(dispatch(node, search)).await;
    if !response.is_success() {
        return Err(response);
    }
    let body = response.body_json().unwrap_or(Value::Null);
    Ok(body["hits"]["hits"].as_array().into_iter().flatten().filter_map(|h| h.get("fields").filter(|f| f.as_object().is_some_and(|o| !o.is_empty())).cloned()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_moves_lookups_to_the_coordinator() {
        let body = json!({"query": {"match_all": {}}, "fields": ["who", "other"], "runtime_mappings": {
            "who": {"type": "lookup", "target_index": "ips", "input_field": "client", "target_field": "ip", "fetch_fields": ["name"]},
            "k": {"type": "keyword", "script": "emit('x')"}
        }});
        let p = plan(&body, &Map::new()).unwrap();
        assert_eq!(p.body["runtime_mappings"], json!({"k": {"type": "keyword", "script": "emit('x')"}}));
        assert_eq!(p.body["fields"], json!(["other", "client"]));
        assert_eq!(p.drop_inputs, vec!["client".to_string()]);
        assert_eq!(p.lookups.len(), 1);
        // Not asked for: nothing to plan.
        assert!(plan(&json!({"runtime_mappings": body["runtime_mappings"].clone()}), &Map::new()).is_none());
        // The input asked for too: it stays in the hits.
        let p = plan(&json!({"fields": ["*"], "runtime_mappings": {"who": body["runtime_mappings"]["who"].clone()}}), &Map::new()).unwrap();
        assert!(p.drop_inputs.is_empty());
        assert!(p.body.get("runtime_mappings").is_none());
        // A mapped lookup field: planned without request runtime mappings;
        // a request-level field of the same name shadows it.
        let mapped: Map<String, Value> = [("who".to_string(), body["runtime_mappings"]["who"].clone())].into_iter().collect();
        let p = plan(&json!({"fields": ["who"]}), &mapped).unwrap();
        assert_eq!(p.lookups[0].target_index, "ips");
        assert_eq!(p.body["fields"], json!(["client"]));
        assert!(p.body.get("runtime_mappings").is_none());
        assert!(plan(&json!({"fields": ["who"], "runtime_mappings": {"who": {"type": "keyword"}}}), &mapped).is_none());
    }
}
