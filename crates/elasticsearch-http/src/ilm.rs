//! Index lifecycle REST APIs: `_ilm/*`, `/{index}/_ilm/*`, `_slm/*`,
//! data stream lifecycle (`_data_stream/{n}/_lifecycle`,
//! `/{index}/_lifecycle/explain`, `_lifecycle/stats`), data stream
//! `_modify` / `_migrate` / `_promote` / `_stats`, resize
//! (`_clone` / `_shrink` / `_split`) and `_freeze` / `_unfreeze`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::ilm::{self, dlm, policy, resize, runner, slm};
use elasticsearch_engine::node::{Node, ResolveOpts, ES_VERSION};
use serde_json::{json, Map, Value};
use std::sync::Arc;

fn human(req: &Req) -> bool {
    matches!(req.param("human"), Some("") | Some("true"))
}

fn include_hidden(req: &Req) -> bool {
    req.param("expand_wildcards").map(|w| w.split(',').any(|x| x == "hidden" || x == "all")).unwrap_or(false)
}

/// `/_ilm/...`, `/_slm/...`, `/_lifecycle/stats`.
pub async fn handle_global(node: Arc<Node>, req: Req) -> R {
    let a = req.seg(0).to_string();
    let b = req.seg(1).to_string();
    let c = req.seg(2).to_string();
    let d = req.seg(3).to_string();
    let m = req.method.clone();
    match (a.as_str(), b.as_str()) {
        ("_ilm", "policy") => match (&m, c.is_empty()) {
            (&Method::GET, _) => blocking(move || get_policies(&node, if c.is_empty() { None } else { Some(&c) })).await,
            (&Method::PUT, false) => blocking(move || put_policy(&node, &req, &c)).await,
            (&Method::DELETE, false) => blocking(move || delete_policy(&node, &c)).await,
            _ => method_not_allowed(&req, if c.is_empty() { &["GET"] } else { &["GET", "PUT", "DELETE"] }),
        },
        ("_ilm", "status") if m == Method::GET => Ok(Resp::ok(json!({"operation_mode": node.ilm.mode()}))),
        ("_ilm", "start") | ("_ilm", "stop") if m == Method::POST => blocking(move || set_ilm_mode(&node, b == "start")).await,
        ("_ilm", "move") if m == Method::POST && !c.is_empty() => blocking(move || {
            runner::move_to_step(&node, &c, &req.json()?)?;
            Ok(Resp::ack())
        })
        .await,
        ("_ilm", "migrate_to_data_tiers") if m == Method::POST => blocking(move || migrate_to_data_tiers(&node, &req)).await,
        ("_ilm", "status" | "start" | "stop" | "move" | "migrate_to_data_tiers") => {
            method_not_allowed(&req, if b == "status" { &["GET"] } else { &["POST"] })
        }
        ("_slm", "policy") => {
            let human = human(&req);
            match (&m, c.is_empty(), d.as_str()) {
                (&Method::GET, _, "") => blocking(move || Ok(Resp::ok(slm::get_policies(&node, if c.is_empty() { None } else { Some(&c) }, human)?))).await,
                (&Method::PUT, false, "") => blocking(move || {
                    slm::put_policy(&node, &c, &req.json()?, std::str::from_utf8(&req.body).unwrap_or(""))?;
                    Ok(Resp::ack())
                })
                .await,
                (&Method::DELETE, false, "") => blocking(move || {
                    slm::delete_policy(&node, &c)?;
                    Ok(Resp::ack())
                })
                .await,
                (&Method::PUT, false, "_execute") | (&Method::POST, false, "_execute") => blocking(move || Ok(Resp::ok(slm::execute(&node, &c)?))).await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        ("_slm", "stats") if m == Method::GET => Ok(Resp::ok(slm::stats(&node))),
        ("_slm", "status") if m == Method::GET => Ok(Resp::ok(json!({"operation_mode": node.ilm.slm_mode()}))),
        ("_slm", "start") | ("_slm", "stop") if m == Method::POST => blocking(move || {
            slm::set_mode(&node, b == "start")?;
            Ok(Resp::ack())
        })
        .await,
        ("_slm", "_execute_retention") if m == Method::POST => blocking(move || {
            slm::execute_retention(&node)?;
            Ok(Resp::ack())
        })
        .await,
        ("_lifecycle", "stats") if m == Method::GET => {
            let mut v = dlm::stats(&node);
            // Human-readable durations only with `?human` (ES `XContentBuilder.humanReadable`).
            if !human(&req) {
                if let Some(o) = v.as_object_mut() {
                    o.remove("last_run_duration");
                    o.remove("time_between_starts");
                }
            }
            Ok(Resp::ok(v))
        }
        _ => crate::not_found_handler(&req),
    }
}

/// `/{index}/_ilm/{explain|retry|remove}`, `/{index}/_lifecycle/explain`,
/// `/{index}/_clone|_shrink|_split/{target}`, `/{index}/_freeze|_unfreeze`.
pub async fn handle_index(node: Arc<Node>, req: Req, index: String) -> R {
    let b = req.seg(1).to_string();
    let c = req.seg(2).to_string();
    let m = req.method.clone();
    match b.as_str() {
        "_ilm" => match (c.as_str(), &m) {
            ("explain", &Method::GET) | ("explain", &Method::POST) => blocking(move || {
                let only_managed = req.flag("only_managed")?;
                let only_errors = req.flag("only_errors")?;
                let mut v = runner::explain(&node, &index, ResolveOpts::from_params(&req.params), only_managed, only_errors)?;
                if human(&req) {
                    runner::humanize_explain(&mut v);
                }
                Ok(Resp::ok(v))
            })
            .await,
            ("retry", &Method::POST) => blocking(move || {
                runner::retry(&node, &index)?;
                Ok(Resp::ack())
            })
            .await,
            ("remove", &Method::POST) => blocking(move || Ok(Resp::ok(runner::remove_policy(&node, &index, ResolveOpts::from_params(&req.params))?))).await,
            ("explain", _) => method_not_allowed(&req, &["GET"]),
            ("retry", _) | ("remove", _) => method_not_allowed(&req, &["POST"]),
            _ => crate::not_found_handler(&req),
        },
        "_lifecycle" if c == "explain" => {
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            blocking(move || {
                let include_defaults = req.flag("include_defaults")?;
                let mut opts = ResolveOpts::from_params(&req.params);
                if req.param("expand_wildcards").is_none() {
                    opts.expand_hidden = true;
                }
                let mut v = dlm::explain(&node, &index, opts, include_defaults)?;
                if human(&req) {
                    runner::humanize_explain(&mut v);
                }
                Ok(Resp::ok(v))
            })
            .await
        }
        "_clone" | "_shrink" | "_split" => {
            if m != Method::POST && m != Method::PUT {
                return method_not_allowed(&req, &["POST", "PUT"]);
            }
            if c.is_empty() {
                return crate::not_found_handler(&req);
            }
            let ty = resize::ResizeType::parse(&b).unwrap();
            blocking(move || {
                if b == "_split" {
                    if let Some(svc) = node.get_index(&index) {
                        if elasticsearch_engine::tsdb::is_time_series(&svc.meta().settings) {
                            return Err(elasticsearch_engine::tsdb::mode_error("index-split", &index));
                        }
                    }
                }
                let body = req.json()?;
                let copy = req.bool_param("copy_settings")?;
                Ok(Resp::ok(resize::resize_with(&node, &index, &c, ty, &body, copy)?))
            })
            .await
        }
        "_freeze" | "_unfreeze" => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            blocking(move || freeze(&node, &req, &index, b == "_freeze")).await
        }
        _ => crate::not_found_handler(&req),
    }
}

/// Whether a `/_data_stream/...` request is served here (everything but
/// create / delete, which `datastream.rs` handles).
pub fn owns_data_stream_route(req: &Req) -> bool {
    !req.seg(2).is_empty() || matches!(req.seg(1), "_stats" | "_modify") || req.method == Method::GET
}

/// `/_data_stream/...` routes beyond create / delete.
pub async fn handle_data_stream(node: Arc<Node>, req: Req) -> R {
    let b = req.seg(1).to_string();
    let c = req.seg(2).to_string();
    let m = req.method.clone();
    let r = match (b.as_str(), c.as_str()) {
        ("_stats", "") => {
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            blocking(move || Ok(Resp::ok(dlm::ds_stats(&node, "", include_hidden(&req))?))).await
        }
        ("_modify", "") => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            blocking(move || {
                dlm::modify(&node, &req.json()?)?;
                Ok(Resp::ack())
            })
            .await
        }
        ("_migrate", alias) if !alias.is_empty() => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            let alias = alias.to_string();
            blocking(move || {
                dlm::migrate(&node, &alias)?;
                Ok(Resp::ack())
            })
            .await
        }
        (name, "_stats") if !name.is_empty() => {
            if m != Method::GET {
                return method_not_allowed(&req, &["GET"]);
            }
            let name = name.to_string();
            blocking(move || Ok(Resp::ok(dlm::ds_stats(&node, &name, include_hidden(&req))?))).await
        }
        // `POST /_data_stream/_promote/{name}`.
        ("_promote", name) if !name.is_empty() && req.seg(3).is_empty() => {
            if m != Method::POST {
                return method_not_allowed(&req, &["POST"]);
            }
            let name = name.to_string();
            blocking(move || {
                dlm::promote(&node, &name)?;
                Ok(Resp::ack())
            })
            .await
        }
        (name, "_options") if !name.is_empty() && req.seg(3).is_empty() => {
            let name = name.to_string();
            match m {
                Method::GET => blocking(move || Ok(Resp::ok(node.get_data_stream_options(&name, include_hidden(&req))?))).await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    node.put_data_stream_options(&name, include_hidden(&req), Some(&body))?;
                    Ok(Resp::ack())
                })
                .await,
                Method::DELETE => blocking(move || {
                    node.put_data_stream_options(&name, include_hidden(&req), None)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        (name, "_settings") if !name.is_empty() && req.seg(3).is_empty() => {
            let name = name.to_string();
            match m {
                Method::GET => blocking(move || Ok(Resp::ok(node.get_data_stream_settings(&name, include_hidden(&req))?))).await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    let dry_run = req.flag("dry_run")?;
                    Ok(Resp::ok(node.put_data_stream_settings(&name, include_hidden(&req), &body, dry_run)?))
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT"]),
            }
        }
        (name, "_mappings") if !name.is_empty() && req.seg(3).is_empty() => {
            let name = name.to_string();
            match m {
                Method::GET => blocking(move || Ok(Resp::ok(node.get_data_stream_mappings(&name, include_hidden(&req))?))).await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    let dry_run = req.flag("dry_run")?;
                    Ok(Resp::ok(node.put_data_stream_mappings(&name, include_hidden(&req), &body, dry_run)?))
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT"]),
            }
        }
        (name, "_lifecycle") if !name.is_empty() => {
            let name = name.to_string();
            match m {
                Method::GET => blocking(move || {
                    let defaults = req.flag("include_defaults")?;
                    Ok(Resp::ok(dlm::get_lifecycle(&node, &name, include_hidden(&req), defaults)?))
                })
                .await,
                Method::PUT => blocking(move || {
                    let body = req.json()?;
                    dlm::put_lifecycle(&node, &name, include_hidden(&req), Some(&body))?;
                    Ok(Resp::ack())
                })
                .await,
                Method::DELETE => blocking(move || {
                    dlm::put_lifecycle(&node, &name, include_hidden(&req), None)?;
                    Ok(Resp::ack())
                })
                .await,
                _ => method_not_allowed(&req, &["GET", "PUT", "DELETE"]),
            }
        }
        (_, "") if m == Method::GET => blocking(move || {
            let name = req.seg(1).to_string();
            let defaults = req.flag("include_defaults")?;
            let hidden = include_hidden(&req) || (!name.is_empty() && !name.contains('*') && name != "_all");
            let mut out = node.get_data_streams_opts(if name.is_empty() { None } else { Some(&name) }, hidden, defaults)?;
            if req.flag("verbose")? {
                // `maximum_timestamp`: the newest `@timestamp` over the backing indices (0 when empty).
                let ro = elasticsearch_engine::node::ResolveOpts { ignore_unavailable: true, allow_no_indices: true, expand_open: true, expand_closed: false, expand_hidden: true };
                for ds in out["data_streams"].as_array_mut().into_iter().flatten() {
                    let idx: Vec<String> = ds["indices"].as_array().into_iter().flatten().filter_map(|i| i["index_name"].as_str().map(str::to_string)).collect();
                    let body = json!({"size": 0, "aggs": {"m": {"max": {"field": "@timestamp"}}}});
                    let max = crate::search_api::run_search(&node, &body, &Default::default(), Some(&idx.join(",")), ro.clone(), None)
                        .ok()
                        .and_then(|r| serde_json::from_str::<Value>(&r).ok())
                        .and_then(|v| v["aggregations"]["m"]["value"].as_f64())
                        .unwrap_or(0.0);
                    ds["maximum_timestamp"] = json!(max as i64);
                }
            }
            Ok(Resp::ok(out))
        })
        .await,
        _ => crate::not_found_handler(&req),
    };
    r
}

// ---------------------------------------------------------------------------
// ILM policies
// ---------------------------------------------------------------------------

fn get_policies(node: &Node, names: Option<&str>) -> R {
    let st_names: Vec<String> = match names {
        None | Some("") => vec!["*".into()],
        Some(n) => n.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect(),
    };
    if st_names.len() > 1 && st_names.iter().any(|n| n.contains('*')) {
        return Err(EsError::illegal_argument("wildcard only supports a single value, please use comma-separated values or a single wildcard value"));
    }
    let all: Vec<(String, ilm::PolicyMeta)> = {
        let st = node.ilm.st_snapshot();
        st.policies.into_iter().collect()
    };
    let mut out = Map::new();
    for n in &st_names {
        if n.contains('*') {
            for (k, pm) in all.iter().filter(|(k, _)| elasticsearch_engine::mapping::wildcard_match(n, k)) {
                out.insert(k.clone(), policy_json(node, k, pm));
            }
        } else {
            match all.iter().find(|(k, _)| k == n) {
                Some((k, pm)) => {
                    out.insert(k.clone(), policy_json(node, k, pm));
                }
                None => return Err(EsError::resource_not_found(format!("Lifecycle policy not found: {n}"))),
            }
        }
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn policy_json(node: &Node, name: &str, pm: &ilm::PolicyMeta) -> Value {
    json!({
        "version": pm.version,
        "modified_date": ilm::iso_date(pm.modified_date),
        "policy": pm.policy,
        "in_use_by": runner::policy_usage(node, name)
    })
}

fn put_policy(node: &Node, req: &Req, name: &str) -> R {
    let body = req.json()?;
    let raw = std::str::from_utf8(&req.body).unwrap_or("");
    let p = policy::with_source(raw, || policy::parse_put_body(&body))?;
    // `RestPutLifecycleAction`: a `max_size` rollover condition is deprecated.
    let warnings: Vec<(&'static str, String)> = body
        .pointer("/policy/phases")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter(|(_, ph)| ph.pointer("/actions/rollover/max_size").is_some_and(|v| !v.is_null()))
        .map(|(phase, _)| crate::search_api::warning_header(&format!("Use of the [max_size] rollover condition found in phase [{phase}]. This condition has been deprecated in favour of the [max_primary_shard_size] condition and will be removed in a later version")))
        .collect();
    let errs = policy::validation_errors(&p);
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    policy::validate_name(name)?;
    node.ilm.put_policy(node, name, p)?;
    let mut r = Resp::ack();
    r.headers.extend(warnings);
    Ok(r)
}

fn delete_policy(node: &Node, name: &str) -> R {
    let mut users: Vec<String> = node.all_indices().iter().filter(|s| runner::policy_name(s).as_deref() == Some(name)).map(|s| s.name.clone()).collect();
    users.sort();
    if !users.is_empty() {
        return Err(EsError::illegal_argument(format!("Cannot delete policy [{name}]. It is in use by one or more indices: [{}]", users.join(", "))));
    }
    node.ilm.delete_policy(name)?;
    Ok(Resp::ack())
}

fn set_ilm_mode(node: &Node, start: bool) -> R {
    node.ilm.set_mode(start)?;
    Ok(Resp::ack())
}

// ---------------------------------------------------------------------------
// migrate_to_data_tiers
// ---------------------------------------------------------------------------

fn tier_preference(value: &str) -> Option<&'static str> {
    match value {
        "hot" => Some("data_hot"),
        "warm" => Some("data_warm,data_hot"),
        "cold" => Some("data_cold,data_warm,data_hot"),
        "frozen" => Some("data_frozen"),
        _ => None,
    }
}

fn migrate_to_data_tiers(node: &Node, req: &Req) -> R {
    let body = req.json()?;
    let dry_run = req.flag("dry_run")?;
    let mode = node.ilm.mode();
    if mode != ilm::STOPPED && !dry_run {
        return Err(EsError::new(500, "illegal_state_exception", format!("stop ILM before migrating to data tiers, current state is [{mode}]")));
    }
    let attr = body["node_attribute"].as_str().unwrap_or("data").to_string();
    let legacy = body["legacy_template_to_delete"].as_str().map(str::to_string);
    let mut removed_legacy = None;
    if let Some(l) = &legacy {
        if node.templates.read().legacy.contains_key(l) {
            removed_legacy = Some(l.clone());
        }
    }
    // policies
    let mut migrated_policies = Vec::new();
    let policies = node.ilm.st_snapshot().policies;
    for (name, pm) in policies {
        let mut p = pm.policy.clone();
        let mut changed = false;
        if let Some(phases) = p["phases"].as_object_mut() {
            for (_, def) in phases.iter_mut() {
                let mut remove_allocate = false;
                if let Some(alloc) = def["actions"].get_mut("allocate") {
                    for kind in ["include", "exclude", "require"] {
                        if let Some(o) = alloc[kind].as_object_mut() {
                            if o.remove(&attr).is_some() {
                                changed = true;
                            }
                        }
                    }
                    let empty = ["include", "exclude", "require"].iter().all(|k| alloc[*k].as_object().map(|o| o.is_empty()).unwrap_or(true))
                        && alloc.get("number_of_replicas").is_none()
                        && alloc.get("total_shards_per_node").is_none();
                    remove_allocate = empty;
                }
                if remove_allocate {
                    if let Some(a) = def["actions"].as_object_mut() {
                        a.remove("allocate");
                    }
                }
            }
        }
        if changed {
            migrated_policies.push(name.clone());
            if !dry_run {
                node.ilm.replace_policy(&name, p)?;
            }
        }
    }
    // indices
    let mut migrated_indices = Vec::new();
    for svc in node.all_indices() {
        let mut pref: Option<&'static str> = None;
        let mut remove = Vec::new();
        for kind in ["require", "include"] {
            let k = format!("index.routing.allocation.{kind}.{attr}");
            if let Some(v) = svc.setting(&k) {
                remove.push(k);
                if pref.is_none() {
                    pref = tier_preference(v.split(',').next().unwrap_or(""));
                }
            }
        }
        let k = format!("index.routing.allocation.exclude.{attr}");
        if svc.setting(&k).is_some() {
            remove.push(k);
        }
        if remove.is_empty() {
            continue;
        }
        migrated_indices.push(svc.name.clone());
        if !dry_run {
            let mut flat: elasticsearch_engine::settings::FlatSettings = remove.into_iter().map(|k| (k, Value::Null)).collect();
            if let Some(p) = pref {
                flat.insert("index.routing.allocation.include._tier_preference".into(), json!(p));
            }
            svc.put_settings(flat, false)?;
        }
    }
    // templates
    let mut composable = Vec::new();
    let mut component = Vec::new();
    {
        let mut t = node.templates.write();
        let strip = |tmpl: &mut Value| -> bool {
            let settings = &mut tmpl["template"]["settings"];
            let mut flat = elasticsearch_engine::settings::normalize_index_settings(settings);
            let before = flat.len();
            flat.retain(|k, _| !(k.starts_with("index.routing.allocation.") && k.ends_with(&format!(".{attr}")) && !k.ends_with("._tier_preference")));
            if flat.len() != before {
                *settings = elasticsearch_engine::settings::unflatten(&flat);
                true
            } else {
                false
            }
        };
        let names: Vec<String> = t.index.keys().cloned().collect();
        for n in names {
            let mut v = t.index[&n].clone();
            if strip(&mut v) {
                composable.push(n.clone());
                if !dry_run {
                    t.index.insert(n, v);
                }
            }
        }
        let names: Vec<String> = t.component.keys().cloned().collect();
        for n in names {
            let mut v = t.component[&n].clone();
            if strip(&mut v) {
                component.push(n.clone());
                if !dry_run {
                    t.component.insert(n, v);
                }
            }
        }
        if !dry_run {
            if let Some(l) = &removed_legacy {
                t.legacy.remove(l);
            }
        }
    }
    if !dry_run {
        node.persist_state()?;
    }
    let mut out = json!({
        "dry_run": dry_run,
        "migrated_ilm_policies": migrated_policies,
        "migrated_indices": migrated_indices,
        "migrated_legacy_templates": [],
        "migrated_composable_templates": composable,
        "migrated_component_templates": component
    });
    // ES renders only the non-empty migration lists
    if let Some(o) = out.as_object_mut() {
        o.retain(|_, v| v.as_array().map(|a| !a.is_empty()).unwrap_or(true));
    }
    if let Some(l) = removed_legacy {
        out["removed_legacy_template"] = json!(l);
    }
    Ok(Resp::ok(out))
}

// ---------------------------------------------------------------------------
// _freeze / _unfreeze
// ---------------------------------------------------------------------------

const FREEZE_REMOVED: &str = "It is no longer possible to freeze indices, but existing frozen indices can still be unfrozen";
const UNFREEZE_DEPRECATED: &str = "Frozen indices are deprecated because they provide no benefit given improvements in heap memory utilization. They will be removed in a future release.";

fn compat_v7(req: &Req) -> bool {
    ["accept", "content-type"].iter().any(|h| req.headers.get(*h).and_then(|v| v.to_str().ok()).map(|v| v.contains("compatible-with=7")).unwrap_or(false))
}

fn freeze(node: &Node, req: &Req, index: &str, freeze: bool) -> R {
    if freeze {
        if !compat_v7(req) {
            // 8.x registers `_freeze` only for REST API compatibility v7.
            return crate::not_found_handler(req);
        }
        let mut o = ResolveOpts::from_params(&req.params);
        o.expand_closed = true;
        node.resolve(index, o)?;
        return Ok(Resp::json(
            410,
            json!({"error": {"root_cause": [{"type": "unsupported_operation_exception", "reason": FREEZE_REMOVED}], "type": "unsupported_operation_exception", "reason": FREEZE_REMOVED}, "status": 410}),
        ));
    }
    let mut o = ResolveOpts::from_params(&req.params);
    o.expand_closed = true;
    let targets = node.resolve(index, o)?;
    for t in targets {
        let keys = ["index.frozen", "index.search.throttled"];
        if keys.iter().any(|k| t.svc.setting(k).is_some()) {
            let flat: elasticsearch_engine::settings::FlatSettings = keys.iter().map(|k| (k.to_string(), Value::Null)).collect();
            t.svc.put_settings(flat, false)?;
        }
    }
    let mut r = Resp::ok(json!({"acknowledged": true, "shards_acknowledged": true}));
    r.headers.push(("Warning", format!("299 Elasticsearch-{ES_VERSION}-{} \"{UNFREEZE_DEPRECATED}\"", crate::cluster::BUILD_HASH)));
    Ok(r)
}
