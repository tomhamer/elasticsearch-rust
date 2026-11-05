//! Index APIs: create/delete/get/exists, mappings, settings, aliases,
//! refresh/flush/forcemerge, analyze, stats, open/close, templates,
//! resolve.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::IndexService;
use elasticsearch_engine::node::{Node, ResolveOpts};
use elasticsearch_engine::settings;
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub async fn handle_global(node: Arc<Node>, req: Req) -> R {
    let api = req.seg(0).to_string();
    let m = req.method.clone();
    match (api.as_str(), &m) {
        ("_mapping", &Method::GET) | ("_mappings", &Method::GET) => blocking(move || get_mapping(&node, &req, None)).await,
        ("_mapping", &Method::PUT) | ("_mapping", &Method::POST) => blocking(move || put_mapping(&node, &req, "_all")).await,
        // `RestOpenIndexAction` / `RestCloseIndexAction` without an index.
        ("_open", &Method::POST) | ("_close", &Method::POST) if req.segs.len() == 1 => Err(EsError::action_validation(&["index is missing"])),
        ("_settings", &Method::GET) => blocking(move || get_settings(&node, &req, None, req.segs.get(1).cloned())).await,
        ("_settings", &Method::PUT) => blocking(move || put_settings(&node, &req, "_all")).await,
        ("_aliases", &Method::POST) => blocking(move || post_aliases(&node, &req)).await,
        ("_aliases", &Method::GET) | ("_alias", &Method::GET) | ("_alias", &Method::HEAD) => blocking(move || get_alias(&node, &req, None, req.segs.get(1).cloned())).await,
        ("_refresh", &Method::POST) | ("_refresh", &Method::GET) => blocking(move || refresh(&node, &req, None)).await,
        ("_flush", &Method::POST) | ("_flush", &Method::GET) => blocking(move || flush(&node, &req, None)).await,
        ("_forcemerge", &Method::POST) => blocking(move || forcemerge(node, &req, None)).await,
        ("_analyze", &Method::GET) | ("_analyze", &Method::POST) => blocking(move || analyze(&node, &req, None)).await,
        ("_stats", &Method::GET) => blocking(move || stats(&node, &req, None)).await,
        ("_index_template", _) | ("_component_template", _) | ("_template", _) => blocking(move || templates(&node, &req)).await,
        ("_resolve", &Method::GET) if req.seg(1) == "index" => blocking(move || resolve_index(&node, &req)).await,
        ("_all", _) => {
            let req2 = req;
            handle_index(node, req2, "_all".into()).await
        }
        _ => crate::not_found_handler(&req),
    }
}

pub async fn handle_index(node: Arc<Node>, req: Req, index: String) -> R {
    let api = req.seg(1).to_string();
    let m = req.method.clone();
    match (api.as_str(), &m) {
        ("", &Method::PUT) => blocking(move || create(&node, &req, &index)).await,
        ("", &Method::DELETE) => blocking(move || delete(&node, &req, &index)).await,
        ("", &Method::GET) => blocking(move || get_index(&node, &req, &index)).await,
        ("", &Method::HEAD) => blocking(move || exists(&node, &req, &index)).await,
        ("", _) => method_not_allowed(&req, &["GET", "HEAD", "PUT", "DELETE"]),
        ("_mapping", &Method::GET) | ("_mappings", &Method::GET) => blocking(move || get_mapping(&node, &req, Some(&index))).await,
        ("_mapping", &Method::PUT) | ("_mapping", &Method::POST) => blocking(move || put_mapping(&node, &req, &index)).await,
        ("_settings", &Method::GET) => blocking(move || get_settings(&node, &req, Some(&index), req.segs.get(2).cloned())).await,
        ("_settings", &Method::PUT) => blocking(move || put_settings(&node, &req, &index)).await,
        ("_alias", &Method::GET) | ("_alias", &Method::HEAD) | ("_aliases", &Method::GET) => blocking(move || get_alias(&node, &req, Some(&index), req.segs.get(2).cloned())).await,
        ("_alias", &Method::PUT) | ("_alias", &Method::POST) | ("_aliases", &Method::PUT) | ("_aliases", &Method::POST) => blocking(move || put_alias(&node, &req, &index)).await,
        ("_alias", &Method::DELETE) | ("_aliases", &Method::DELETE) => blocking(move || delete_alias(&node, &req, &index)).await,
        ("_refresh", &Method::POST) | ("_refresh", &Method::GET) => blocking(move || refresh(&node, &req, Some(&index))).await,
        ("_flush", &Method::POST) | ("_flush", &Method::GET) => blocking(move || flush(&node, &req, Some(&index))).await,
        ("_forcemerge", &Method::POST) => blocking(move || forcemerge(node, &req, Some(&index))).await,
        ("_analyze", &Method::GET) | ("_analyze", &Method::POST) => blocking(move || analyze(&node, &req, Some(&index))).await,
        ("_stats", &Method::GET) => blocking(move || stats(&node, &req, Some(&index))).await,
        ("_open", &Method::POST) => blocking(move || open_close(&node, &req, &index, true)).await,
        ("_close", &Method::POST) => blocking(move || open_close(&node, &req, &index, false)).await,
        _ => crate::not_found_handler(&req),
    }
}

fn opts(req: &Req) -> ResolveOpts {
    ResolveOpts::from_params(&req.params)
}

fn create(node: &Node, req: &Req, index: &str) -> R {
    let body = req.json()?;
    let svc = node.create_index(index, &body)?;
    // A primary whose recovery failed (analyzers that do not build) is
    // never active: ES answers `shards_acknowledged: false` after waiting.
    Ok(Resp::ok(json!({"acknowledged": true, "shards_acknowledged": svc.failed_shards().is_empty(), "index": svc.name})))
}

fn delete(node: &Node, req: &Req, index: &str) -> R {
    elasticsearch_engine::datastream::reject_selectors(index)?;
    let wildcard = index == "_all" || index.split(',').any(|p| p.contains('*') || p == "_all");
    if wildcard && node.destructive_requires_name() {
        return Err(EsError::illegal_argument("Wildcard expressions or all indices are not allowed"));
    }
    let mut o = opts(req);
    o.expand_closed = true;
    // Delete never goes through aliases: a concrete alias name is an error
    // (skipped with ignore_unavailable) and wildcards only match index names.
    let mut names: Vec<String> = Vec::new();
    let is_alias = |n: &str| node.all_indices().iter().any(|s| s.aliases().contains_key(n));
    for part in index.split(',').filter(|p| !p.is_empty()) {
        let wild = part == "_all" || part.contains('*');
        if !wild && node.get_index(part).is_none() && node.data_stream(part).is_some() {
            if o.ignore_unavailable {
                continue;
            }
            return Err(EsError::index_not_found(part));
        }
        if !wild && node.get_index(part).is_none() && is_alias(part) {
            if o.ignore_unavailable {
                continue;
            }
            return Err(EsError::illegal_argument(format!(
                "The provided expression [{part}] matches an alias, specify the corresponding concrete indices instead."
            )));
        }
        let mut po = o;
        if wild {
            po.allow_no_indices = true;
        }
        let mut matched = 0;
        for t in node.resolve(part, po)? {
            if wild && part != "_all" && !elasticsearch_engine::mapping::wildcard_match(part.trim_start_matches('-'), &t.svc.name) {
                continue;
            }
            matched += 1;
            if !names.contains(&t.svc.name) {
                names.push(t.svc.name.clone());
            }
        }
        if wild && matched == 0 && !o.allow_no_indices {
            return Err(EsError::index_not_found(part));
        }
    }
    if names.is_empty() && !o.allow_no_indices {
        return Err(EsError::index_not_found(index));
    }
    for n in names {
        node.delete_index(&n)?;
    }
    Ok(Resp::ack())
}

fn exists(node: &Node, req: &Req, index: &str) -> R {
    let mut o = opts(req);
    o.allow_no_indices = false;
    o.expand_closed = true;
    match node.resolve(index, o) {
        Ok(t) if !t.is_empty() => Ok(Resp::empty(200)),
        _ => Ok(Resp::empty(404)),
    }
}

fn settings_json(svc: &IndexService, flat: bool, filter: Option<&str>, include_defaults: bool) -> Value {
    let all = svc.meta().settings;
    let pats: Option<Vec<String>> = filter.filter(|f| *f != "_all").map(|f| f.split(',').map(str::to_string).collect());
    let keep = |k: &str| -> bool {
        match &pats {
            None => true,
            Some(p) => p.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, k) || elasticsearch_engine::mapping::wildcard_match(&format!("index.{p}"), k)),
        }
    };
    let mut s = all.clone();
    s.retain(|k, _| keep(k));
    let mut out = Map::new();
    out.insert("settings".into(), if flat { settings::as_flat_json(&s) } else { settings::unflatten(&s) });
    if include_defaults {
        let mut d = settings::index_defaults(&all);
        // The effective source mode (setting, else the mapping's
        // `_source.mode`, else the index-mode default) is what the
        // `index.mapping.source.mode` default reports.
        let eff = elasticsearch_engine::synthetic::source_mode(
            elasticsearch_engine::settings::get_str(&all, "index.mapping.source.mode"),
            &svc.meta().mappings,
            elasticsearch_engine::settings::get_str(&all, "index.mode"),
        );
        d.insert("index.mapping.source.mode".into(), serde_json::Value::String(eff.to_string()));
        d.retain(|k, _| keep(k));
        out.insert("defaults".into(), if flat { settings::as_flat_json(&d) } else { settings::unflatten(&d) });
    }
    Value::Object(out)
}

fn get_index(node: &Node, req: &Req, index: &str) -> R {
    let mut o = opts(req);
    if req.param("expand_wildcards").is_none() {
        o.expand_closed = true;
    }
    let flat = req.flag("flat_settings")?;
    let human = req.flag("human")?;
    let features = req.param("features").map(|s| s.split(',').map(str::to_string).collect::<std::collections::BTreeSet<_>>());
    let targets = node.resolve(index, o)?;
    let mut out = Map::new();
    for t in targets {
        let meta = t.svc.meta();
        let mut m = Map::new();
        let want = |f: &str| features.as_ref().map(|s| s.contains(f)).unwrap_or(true);
        m.insert("aliases".into(), if want("aliases") { json!(meta.aliases) } else { json!({}) });
        m.insert("mappings".into(), if want("mappings") { meta.mappings.to_json() } else { json!({}) });
        let mut settings = if want("settings") { settings_json(&t.svc, flat, None, false)["settings"].clone() } else { json!({}) };
        if human && want("settings") {
            if let Some(o) = settings.as_object_mut() {
                if let Some(idx) = o.get_mut("index").and_then(Value::as_object_mut) {
                    if let Some(cd) = idx.get("creation_date").and_then(Value::as_str) {
                        if let Ok(ts) = cd.parse::<i64>() {
                            idx.insert("creation_date_string".into(), json!(elasticsearch_engine::date::human_date(ts)));
                        }
                    }
                    if let Some(ver) = idx.get_mut("version").and_then(Value::as_object_mut) {
                        if let Some(cs) = ver.get("created").and_then(Value::as_str) {
                            ver.insert("created_string".into(), json!(elasticsearch_engine::date::human_version(cs)));
                        }
                    }
                }
            }
        }
        m.insert("settings".into(), settings);
        if let Some(ds) = node.data_stream_of_index(&t.svc.name) {
            m.insert("data_stream".into(), json!(ds));
        }
        out.insert(t.svc.name.clone(), Value::Object(m));
    }
    Ok(Resp::ok(Value::Object(out)))
}

#[path = "field_mapping.rs"]
mod field_mapping;

fn get_mapping(node: &Node, req: &Req, index: Option<&str>) -> R {
    if req.seg(if index.is_some() { 2 } else { 1 }) == "field" {
        return field_mapping::get_field_mapping(node, req, index);
    }
    if req.seg(if index.is_some() { 2 } else { 1 }) != "" {
        return Ok(Resp::json(400, json!({"error": format!("no handler found for uri [{}] and method [{}]", req.path, req.method)})));
    }
    if let Some(i) = index {
        crate::search_api::search_validate::reject_remote(i)?;
    }
    let mut o = opts(req);
    if index.is_some_and(|i| !i.contains('*') && i != "_all" && !i.contains(',')) {
        o.expand_closed = true;
    }
    if let Some(i) = index {
        elasticsearch_engine::node::reject_remote_indices(i)?;
    }
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    let mut out = Map::new();
    for t in targets {
        out.insert(t.svc.name.clone(), json!({"mappings": t.svc.meta().mappings.to_json()}));
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn put_mapping(node: &Node, req: &Req, index: &str) -> R {
    let body = req.json()?;
    if body.is_null() {
        return Err(EsError::new(400, "parse_exception", "request body is required"));
    }
    // ES 9 (`indices.put_mapping.no_types_check`): the REST layer no longer
    // rejects typed sources; the mapper unwraps a `_doc` root.
    let body = match body.as_object() {
        Some(o) if o.len() == 1 && o.get("_doc").is_some_and(Value::is_object) => o["_doc"].clone(),
        _ => body,
    };
    let mut o = opts(req);
    if req.param("allow_no_indices").is_none() {
        o.allow_no_indices = false;
    }
    // A concrete (closed) index can still take a mapping update.
    if !index.contains('*') && index != "_all" {
        o.expand_closed = true;
    }
    // `write_index_only`: an alias applies the update to its write index alone.
    if req.flag("write_index_only")? && !index.contains(',') && !index.contains('*') && node.get_index(index).is_none() {
        let svc = node.write_index(index, false)?;
        crate::ccr::check_put_mapping(&[&svc])?;
        svc.put_mapping(&body)?;
        return Ok(Resp::ack());
    }
    let targets = node.resolve(index, o)?;
    crate::ccr::check_put_mapping(&targets.iter().map(|t| &*t.svc).collect::<Vec<_>>())?;
    for t in &targets {
        t.svc.check_put_mapping(&body)?;
    }
    for t in &targets {
        t.svc.put_mapping(&body)?;
    }
    Ok(Resp::ack())
}

fn get_settings(node: &Node, req: &Req, index: Option<&str>, name: Option<String>) -> R {
    let mut o = opts(req);
    o.expand_closed = true;
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    let flat = req.flag("flat_settings")?;
    let defaults = req.flag("include_defaults")?;
    let mut out = Map::new();
    for t in targets {
        out.insert(t.svc.name.clone(), settings_json(&t.svc, flat, name.as_deref(), defaults));
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn put_settings(node: &Node, req: &Req, index: &str) -> R {
    let body = req.json()?;
    let flat = settings::normalize_index_settings(&body);
    if flat.is_empty() {
        return Err(EsError::action_validation(&["no settings to update"]));
    }
    let mut o = opts(req);
    o.expand_closed = true;
    let targets = node.resolve(index, o)?;
    let preserve = req.flag("preserve_existing")?;
    let reopen = req.flag("reopen")?;
    for t in &targets {
        t.svc.put_settings_reopen(flat.clone(), preserve, reopen)?;
    }
    Ok(Resp::ack())
}

fn post_aliases(node: &Node, req: &Req) -> R {
    let body = req.json()?;
    let actions = body.get("actions").and_then(Value::as_array).ok_or_else(|| EsError::action_validation(&["No action specified"]))?;
    crate::ccr::check_alias_actions(node, actions)?;
    let results = node.update_aliases(actions)?;
    let errors = results.iter().any(|r| !r["error"].is_null());
    if errors {
        Ok(Resp::ok(json!({"acknowledged": true, "errors": true, "action_results": results})))
    } else {
        Ok(Resp::ok(json!({"acknowledged": true, "errors": false})))
    }
}

fn put_alias(node: &Node, req: &Req, index: &str) -> R {
    let name = req.seg(2).to_string();
    let body = req.json()?;
    // `RestIndexPutAliasAction`: known value fields only (an object value is
    // the `filter`); arrays are rejected. `is_hidden` (ES 7.7+) persists
    // like an `_aliases` add action's.
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            match v {
                Value::Object(_) => {}
                Value::Array(_) => return Err(EsError::parsing_at_key("Unexpected token [START_ARRAY]", k, None)),
                _ if matches!(
                    k.as_str(),
                    "index" | "alias" | "routing" | "indexRouting" | "index-routing" | "index_routing" | "searchRouting" | "search-routing" | "search_routing" | "is_write_index" | "is_hidden"
                ) => {}
                _ => return Err(EsError::illegal_argument(format!("Unsupported field [{k}]"))),
            }
        }
    }
    let mut action = body.as_object().cloned().unwrap_or_default();
    action.insert("index".into(), json!(index));
    if !name.is_empty() {
        action.insert("alias".into(), json!(name));
    } else if !action.contains_key("alias") {
        return Err(EsError::action_validation(&["[name] is required"]));
    }
    if let Some(bad) = name.chars().find(|c| matches!(c, '*' | '?' | '#' | '"' | '|' | '\\' | '/' | ' ')) {
        return Err(EsError::new(400, "invalid_alias_name_exception", format!("Invalid alias name [{name}]: must not contain the following characters [{bad}]")));
    }
    let actions = [json!({"add": action})];
    crate::ccr::check_alias_actions(node, &actions)?;
    node.update_aliases(&actions)?;
    Ok(Resp::ack())
}

fn delete_alias(node: &Node, req: &Req, index: &str) -> R {
    let name = req.seg(2).to_string();
    // `_all` names every alias.
    let pats: Vec<String> = name.split(',').filter(|n| !n.is_empty()).map(|n| if n == "_all" { "*".to_string() } else { n.to_string() }).collect();
    let targets = node.resolve(index, opts(req))?;
    let mut found = false;
    for t in &targets {
        if t.svc.aliases().keys().any(|a| pats.iter().any(|n| elasticsearch_engine::mapping::wildcard_match(n, a))) {
            found = true;
        }
    }
    if !found {
        return Err(EsError::alias_missing(&name));
    }
    let actions = [json!({"remove": {"index": index, "aliases": pats}})];
    crate::ccr::check_alias_actions(node, &actions)?;
    node.update_aliases(&actions)?;
    Ok(Resp::ack())
}

fn get_alias(node: &Node, req: &Req, index: Option<&str>, name: Option<String>) -> R {
    let mut o = opts(req);
    if req.param("expand_wildcards").is_none() {
        o.expand_closed = true;
    }
    let targets = node.resolve(index.unwrap_or("_all"), o)?;
    let patterns: Vec<String> = match name.as_ref() {
        Some(n) => n
            .split(',')
            .filter(|p| !p.is_empty())
            .map(|p| elasticsearch_engine::node::resolve_date_math(p).unwrap_or_else(|_| p.to_string()))
            .collect(),
        None => vec![],
    };
    let all_aliases: std::collections::BTreeSet<String> = targets.iter().flat_map(|t| t.svc.aliases().into_keys()).collect();
    // Patterns apply in order. `-x` removes matches only after a wildcard
    // pattern was seen; before that it is a literal (usually missing) name.
    let mut missing: Vec<String> = Vec::new();
    let mut wildcard_seen = false;
    let mut steps: Vec<(bool, String)> = Vec::new(); // (exclude, pattern)
    for p in &patterns {
        let exclude = p.starts_with('-') && p.len() > 1 && (wildcard_seen || p.contains('*'));
        let pat = if exclude { p[1..].to_string() } else { p.clone() };
        let wild = pat.contains('*') || p == "_all";
        if !exclude && !wild && !all_aliases.contains(&pat) {
            missing.push(p.clone());
        }
        if p.contains('*') || p == "_all" {
            wildcard_seen = true;
        }
        steps.push((exclude, pat));
    }
    let mut out = Map::new();
    for t in &targets {
        let aliases = t.svc.aliases();
        let mut cur: std::collections::BTreeSet<String> = if patterns.is_empty() { aliases.keys().cloned().collect() } else { Default::default() };
        for (exclude, pat) in &steps {
            for a in aliases.keys() {
                if pat == "_all" || elasticsearch_engine::mapping::wildcard_match(pat, a) {
                    if *exclude {
                        cur.remove(a);
                    } else {
                        cur.insert(a.clone());
                    }
                }
            }
        }
        if patterns.is_empty() || !cur.is_empty() {
            let am: Map<String, Value> = aliases.into_iter().filter(|(k, _)| cur.contains(k)).collect();
            // Backing indices of a data stream report under the stream name.
            let by_name = index.is_some_and(|i| i.split(',').any(|p| p == t.svc.name));
            let key = if by_name { None } else { node.data_stream_of_index(&t.svc.name) }.unwrap_or_else(|| t.svc.name.clone());
            out.entry(key).or_insert_with(|| json!({"aliases": am}));
        }
    }
    if !missing.is_empty() {
        missing.sort();
        let names = missing.join(",");
        let msg = if missing.len() == 1 { format!("alias [{names}] missing") } else { format!("aliases [{names}] missing") };
        out.insert("error".into(), json!(msg));
        out.insert("status".into(), json!(404));
        return Ok(Resp::json(404, Value::Object(out)));
    }
    if req.method == Method::HEAD {
        return Ok(if out.is_empty() { Resp::empty(404) } else { Resp::empty(200) });
    }
    Ok(Resp::ok(Value::Object(out)))
}

fn shards_resp(n: usize) -> Value {
    json!({"total": n, "successful": n, "failed": 0})
}

/// Configured shard copies (primaries x (1 + replicas)) of the targets.
fn configured_shards(targets: &[elasticsearch_engine::search::Target]) -> usize {
    targets
        .iter()
        .map(|t| t.svc.setting_usize("index.number_of_shards", 1) * (t.svc.setting_usize("index.number_of_replicas", 1) + 1))
        .sum()
}

fn refresh(node: &Node, req: &Req, index: Option<&str>) -> R {
    let targets = node.resolve(index.unwrap_or("_all"), opts(req))?;
    for t in &targets {
        t.svc.refresh()?;
    }
    // A refresh of everything also reaches the dot-prefixed indices this
    // node treats as hidden (ES only hides `index.hidden` ones).
    if matches!(index, None | Some("_all") | Some("*")) {
        for svc in node.all_indices() {
            if svc.is_open() && svc.name.starts_with('.') && !targets.iter().any(|t| Arc::ptr_eq(&t.svc, &svc)) {
                svc.refresh()?;
            }
        }
    }
    Ok(Resp::ok(json!({"_shards": shards_resp(configured_shards(&targets))})))
}

fn flush(node: &Node, req: &Req, index: Option<&str>) -> R {
    if req.flag("force")? && req.param("wait_if_ongoing").map(|v| v == "false").unwrap_or(false) {
        return Err(EsError::action_validation(&["wait_if_ongoing must be true for a force flush"]));
    }
    let targets = node.resolve(index.unwrap_or("_all"), opts(req))?;
    for t in &targets {
        t.svc.flush()?;
    }
    Ok(Resp::ok(json!({"_shards": shards_resp(configured_shards(&targets))})))
}

fn forcemerge(node: Arc<Node>, req: &Req, index: Option<&str>) -> R {
    let targets = node.resolve(index.unwrap_or("_all"), opts(req))?;
    // ES `ForceMergeRequest`: `max_num_segments` defaults to -1, and any
    // value <= 0 merges only what the merge policy wants
    // (`IndexWriter#maybeMerge`); a positive value merges down to it.
    let max_num: i32 = match req.param("max_num_segments") {
        None => -1,
        Some(v) => v.parse().map_err(|_| EsError::illegal_argument(format!("Failed to parse int parameter [max_num_segments] with value [{v}]")))?,
    };
    let expunge = req.flag("only_expunge_deletes")?;
    if expunge && max_num != -1 {
        return Err(EsError::action_validation(&["cannot set only_expunge_deletes and max_num_segments at the same time, those two parameters are mutually exclusive"]));
    }
    // 0: the policy's merges only ([`Shard::force_merge_with`]); expunging
    // deletes keeps its own pass.
    let max: usize = if expunge { 1 } else if max_num <= 0 { 0 } else { max_num as usize };
    let shards = configured_shards(&targets);
    // ES `flush` (default true): commit once merged.
    let flush = req.param("flush") != Some("false");
    if req.param("wait_for_completion") == Some("false") {
        let task = node.tasks.register("indices:admin/forcemerge", Some(format!("Force merge [{}]", index.unwrap_or("_all"))), false, true, Map::new());
        let id = task.task_id();
        std::thread::spawn(move || {
            let outcome = targets.iter().try_for_each(|t| t.svc.force_merge_with(max, expunge, flush));
            node.tasks.finish(&task, outcome.map(|_| json!({"_shards": shards_resp(shards)})).map_err(|e| e.to_json_with_trace(false)["error"].clone()));
        });
        return Ok(Resp::ok(json!({"task": id})));
    }
    for t in &targets {
        t.svc.force_merge_with(max, expunge, flush)?;
    }
    Ok(Resp::ok(json!({"_shards": shards_resp(shards)})))
}

fn analyze(node: &Node, req: &Req, index: Option<&str>) -> R {
    let mut body = req.json()?;
    if body.is_null() {
        body = json!({});
        for k in ["analyzer", "text", "field", "tokenizer", "normalizer"] {
            if let Some(v) = req.param(k) {
                body[k] = json!(v);
            }
        }
    }
    let svc = match index {
        Some(i) => Some(node.write_index(i, false)?),
        None => None,
    };
    if let Some(s) = &svc {
        // The request runs on a shard copy: none is active when the primaries failed to allocate.
        if s.active_shards() == 0 && !s.failed_shards().is_empty() {
            return Err(EsError::new(503, "no_shard_available_action_exception", "No shard available for [org.elasticsearch.action.admin.indices.analyze.AnalyzeAction$Request/unset]"));
        }
    }
    let registry = svc.as_ref().map(|s| s.registry()).unwrap_or_default();
    let texts: Vec<String> = match body.get("text") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().map(elasticsearch_engine::query::dsl::scalar_string).collect(),
        _ => return Err(EsError::action_validation(&["text is missing"])),
    };
    let mut keyword_like = false;
    let analyzer: Arc<elasticsearch_engine::rustic_core::analysis::Analyzer> = if let Some(a) = body.get("analyzer").and_then(Value::as_str) {
        keyword_like = a == "keyword";
        registry.analyzer(a).ok_or_else(|| EsError::illegal_argument(format!("failed to find global analyzer [{a}]")))?
    } else if let Some(f) = body.get("field").and_then(Value::as_str) {
        let svc = svc.as_ref().ok_or_else(|| EsError::illegal_argument("analysis with a [field] requires an index"))?;
        let lk = svc.lookup();
        match lk.get(f) {
            Some(rf) if rf.ty.is_text() => match &rf.index_analyzer {
                Some(elasticsearch_engine::analysis::IndexAnalyzer::Custom(a)) => a.clone(),
                Some(elasticsearch_engine::analysis::IndexAnalyzer::Builtin(n)) => registry.analyzer(n).unwrap_or_else(|| registry.analyzer("standard").unwrap()),
                None => registry.analyzer("standard").unwrap(),
            },
            Some(rf) => {
                keyword_like = true;
                rf.normalizer.clone().unwrap_or_else(|| registry.analyzer("keyword").unwrap())
            }
            None => registry.analyzer("standard").unwrap(),
        }
    } else if let Some(n) = body.get("normalizer").and_then(Value::as_str) {
        keyword_like = true;
        registry.normalizer(n).ok_or_else(|| EsError::illegal_argument(format!("failed to find normalizer under [{n}]")))?
    } else if body.get("tokenizer").is_some() || body.get("filter").is_some() || body.get("char_filter").is_some() {
        let tok = body.get("tokenizer").cloned().unwrap_or(json!("keyword"));
        keyword_like = tok.as_str() == Some("keyword");
        let filters = body.get("filter").and_then(Value::as_array).cloned().unwrap_or_default();
        let cfs = body.get("char_filter").and_then(Value::as_array).cloned().unwrap_or_default();
        Arc::new(registry.adhoc(&tok, &filters, &cfs)?)
    } else {
        registry.analyzer(registry.default_analyzer_name()).unwrap()
    };
    let standard_types = !keyword_like
        && body.get("analyzer").and_then(Value::as_str).map(|a| !matches!(a, "whitespace" | "keyword" | "simple" | "stop")).unwrap_or(true)
        && body.get("tokenizer").and_then(Value::as_str).map(|t| t == "standard").unwrap_or(true);
    let max_tokens = svc.as_ref().map(|s| s.setting_usize("index.analyze.max_token_count", 10_000)).unwrap_or(10_000);
    let max_chars = svc.as_ref().map(|s| s.setting_usize("index.analyze.max_char_count", 1_000_000)).unwrap_or(1_000_000);
    let render = |a: &elasticsearch_engine::rustic_core::analysis::Analyzer| -> Result<Vec<Value>, EsError> {
        let tokens = analyze_tokens(a, &texts, standard_types);
        let chars: usize = tokens.iter().map(|t| t["token"].as_str().map(|s| s.chars().count()).unwrap_or(0)).sum();
        if chars > max_chars {
            return Err(EsError::new(
                400,
                "status_exception",
                format!("The number of characters produced by calling _analyze has exceeded the allowed maximum of [{max_chars}]. This limit can be set by changing the [index.analyze.max_char_count] index level setting."),
            ));
        }
        if tokens.len() > max_tokens {
            return Err(EsError::new(
                400,
                "status_exception",
                format!("The number of tokens produced by calling _analyze has exceeded the allowed maximum of [{max_tokens}]. This limit can be set by changing the [index.analyze.max_token_count] index level setting."),
            ));
        }
        Ok(tokens)
    };
    let explain = body.get("explain").and_then(Value::as_bool).unwrap_or(false);
    if !explain {
        return Ok(Resp::ok(json!({"tokens": render(&analyzer)?})));
    }
    let with_extras = |ts: Vec<Value>| -> Vec<Value> {
        ts.into_iter()
            .map(|mut t| {
                let bytes = t["token"].as_str().unwrap_or("").bytes().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
                t["bytes"] = json!(format!("[{bytes}]"));
                t["positionLength"] = json!(1);
                t["termFrequency"] = json!(1);
                t
            })
            .collect()
    };
    let adhoc = body.get("analyzer").is_none() && body.get("field").is_none() && body.get("normalizer").is_none() && (body.get("tokenizer").is_some() || body.get("filter").is_some() || body.get("char_filter").is_some());
    if !adhoc {
        let name = body.get("analyzer").and_then(Value::as_str).unwrap_or("default");
        return Ok(Resp::ok(json!({"detail": {"custom_analyzer": false, "analyzer": {"name": name, "tokens": with_extras(render(&analyzer)?)}}})));
    }
    let tok = body.get("tokenizer").cloned().unwrap_or(json!("keyword"));
    let filters = body.get("filter").and_then(Value::as_array).cloned().unwrap_or_default();
    let cfs = body.get("char_filter").and_then(Value::as_array).cloned().unwrap_or_default();
    let step_name = |v: &Value| match v {
        Value::String(s) => s.clone(),
        o => format!("__anonymous__{}", o.get("type").and_then(Value::as_str).unwrap_or("")),
    };
    let mut charfilters = Vec::new();
    for (i, cf) in cfs.iter().enumerate() {
        let a = registry.adhoc(&json!("keyword"), &[], &cfs[..=i])?;
        let filtered: Vec<Value> = texts.iter().map(|t| json!(a.analyze(t).first().map(|k| k.text.clone()).unwrap_or_default())).collect();
        charfilters.push(json!({"name": step_name(cf), "filtered_text": filtered}));
    }
    let tokenizer = json!({"name": step_name(&tok), "tokens": with_extras(render(&registry.adhoc(&tok, &[], &cfs)?)?)});
    let mut tokenfilters = Vec::new();
    for (i, f) in filters.iter().enumerate() {
        let a = registry.adhoc(&tok, &filters[..=i], &cfs)?;
        tokenfilters.push(json!({"name": step_name(f), "tokens": with_extras(render(&a)?)}));
    }
    let mut detail = json!({"custom_analyzer": true, "tokenizer": tokenizer});
    detail["charfilters"] = json!(charfilters);
    detail["tokenfilters"] = json!(tokenfilters);
    Ok(Resp::ok(json!({"detail": detail})))
}

fn analyze_tokens(analyzer: &elasticsearch_engine::rustic_core::analysis::Analyzer, texts: &[String], standard_types: bool) -> Vec<Value> {
    let mut tokens = Vec::new();
    let mut base_pos = 0u32;
    let mut base_off = 0usize;
    for t in texts {
        let toks = analyzer.analyze(t);
        let mut last = 0;
        for tk in &toks {
            let to_chars = |b: u32| -> usize {
                let b = (b as usize).min(t.len());
                t.get(..b).map(|p| p.encode_utf16().count()).unwrap_or(b)
            };
            let ty = if !standard_types {
                "word"
            } else if tk.text.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') {
                "<NUM>"
            } else {
                "<ALPHANUM>"
            };
            tokens.push(json!({
                "token": tk.text,
                "start_offset": base_off + to_chars(tk.offset_from),
                "end_offset": base_off + to_chars(tk.offset_to),
                "type": ty,
                "position": base_pos + tk.position,
            }));
            last = last.max(tk.position + 1);
        }
        // ES: position_increment_gap 100 between array values for text, offset gap 1.
        base_pos += last + if texts.len() > 1 { 100 } else { 0 };
        base_off += t.encode_utf16().count() + 1;
    }
    tokens
}

/// Sparse-vector value counts only. Dense counts come from vector_stats;
/// querying dense presence here would calculate and then discard the result.
pub(crate) fn sparse_vector_count(svc: &Arc<IndexService>) -> u64 {
    // Every mapped `sparse_vector` is a resolved field: without one, skip
    // rendering the mappings to JSON (per stats call) only to find none.
    if !svc.lookup().fields.values().any(|f| f.declared == elasticsearch_engine::mapping::EsType::SparseVector) {
        return 0;
    }
    vector_counts_selected(svc, None, false).1
}

/// Vector value counts over one local shard copy (`shard`), or every local
/// copy (`None`).
pub fn vector_counts_in(svc: &Arc<IndexService>, shard: Option<u32>) -> (u64, u64) {
    vector_counts_selected(svc, shard, true)
}

fn vector_counts_selected(svc: &Arc<IndexService>, shard: Option<u32>, include_dense: bool) -> (u64, u64) {
    fn collect(props: &Map<String, Value>, prefix: &str, dense: &mut Vec<String>, sparse: &mut Vec<String>) {
        for (name, def) in props {
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
            match def.get("type").and_then(Value::as_str) {
                Some("dense_vector") => dense.push(path.clone()),
                Some("sparse_vector") => sparse.push(path.clone()),
                _ => {}
            }
            if let Some(p) = def.get("properties").and_then(Value::as_object) {
                collect(p, &path, dense, sparse);
            }
        }
    }
    if !svc.is_open() {
        return (0, 0);
    }
    let mj = svc.meta().mappings.to_json();
    let (mut dense, mut sparse) = (Vec::new(), Vec::new());
    if let Some(p) = mj.get("properties").and_then(Value::as_object) {
        collect(p, "", &mut dense, &mut sparse);
    }
    if dense.is_empty() && sparse.is_empty() {
        return (0, 0);
    }
    let mut target = elasticsearch_engine::search::Target::new(svc.clone());
    if let Some(n) = shard {
        match svc.shard(n) {
            Some(copy) => target.shard = Some(copy),
            None => return (0, 0),
        }
    }
    let target = [target];
    let count = |fields: &[String]| -> u64 {
        fields
            .iter()
            .map(|f| elasticsearch_engine::search::count(&target, Some(&json!({"exists": {"field": f}})), None).unwrap_or(0))
            .sum()
    };
    (if include_dense { count(&dense) } else { 0 }, count(&sparse))
}

/// Metrics backed entirely by atomic counters (response names, not URL aliases).
pub(crate) fn counter_metrics_only(metrics: Option<&[String]>) -> bool {
    const COUNTER_METRICS: &[&str] = &["merges", "indexing", "get", "search", "refresh", "flush", "warmer", "query_cache", "request_cache", "recovery", "bulk", "shard_stats"];
    metrics.is_some_and(|ms| ms.iter().all(|m| COUNTER_METRICS.contains(&m.as_str())))
}

pub(crate) fn counter_index_section(svc: &IndexService) -> Value {
    let mut s = stats_section(&svc.counters(), 0, 0, 0, 0, 0, 0, svc.local_shards().len().max(1));
    s["request_cache"] = crate::request_cache::index_stats_json(&svc.uuid);
    s
}

/// Local shard statistics. Skip translog locking only when that section will
/// be removed by the caller's metric filter.
pub(crate) fn index_stats_selected(svc: &IndexService, translog: bool) -> Value {
    let (docs, deleted) = svc.doc_count();
    let shards = svc.local_shards().len().max(1);
    let (tl, unc) = if translog { translog_totals(svc) } else { (0, 0) };
    let mut s = stats_section(&svc.counters(), docs, deleted, svc.store_size(), svc.num_segments(), tl, unc, shards);
    s["request_cache"] = crate::request_cache::index_stats_json(&svc.uuid);
    s
}

/// Translog bytes and uncommitted operations (at least 0) over the local
/// shards, from one [`elasticsearch_engine::shard::Shard::write_stats`]
/// sample per shard: the same figures as `IndexService::translog_size` and
/// `IndexService::uncommitted_ops`, which each take that sample (a bounded
/// write-lock wait plus a translog directory listing) again.
fn translog_totals(svc: &IndexService) -> (u64, i64) {
    let (mut bytes, mut ops) = (0u64, 0i64);
    for s in svc.local_shards() {
        let w = s.shard.write_stats();
        bytes += w.translog_bytes;
        ops += w.uncommitted_ops;
    }
    (bytes, ops.max(0))
}

/// Selected statistics for one shard copy (`level=shards`).
fn shard_stats_selected(sh: &elasticsearch_engine::index::IndexShard, translog: bool) -> Value {
    let (docs, deleted) = sh.doc_count();
    let segments = sh.shard.reader_view().segments_for_stats();
    // Translog figures without waiting behind a force merge or flush (see
    // `Shard::write_stats`).
    let write = if translog { sh.shard.write_stats() } else { Default::default() };
    stats_section(&sh.shard.counters, docs, deleted, sh.shard.store_size(), segments, write.translog_bytes, write.uncommitted_ops, 1)
}

#[allow(clippy::too_many_arguments)]
/// The shard's search busy time divided by the time since this metric was
/// first read (0 until the shard has searched).
fn recent_search_load(c: &elasticsearch_engine::shard::Counters) -> f64 {
    use std::sync::atomic::Ordering::Relaxed;
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let busy = c.query_time_nanos.load(Relaxed) + c.fetch_time_nanos.load(Relaxed);
    let up = START.get_or_init(std::time::Instant::now).elapsed().as_nanos().max(1) as f64;
    (busy as f64 / up).min(1.0)
}

fn stats_section(c: &elasticsearch_engine::shard::Counters, docs: u64, deleted: u64, store: u64, segs: usize, tl: u64, unc: i64, shards: usize) -> Value {
    use elasticsearch_engine::shard::Counters as C;
    // Bulk shard requests are not tracked separately: each write op counts.
    let ops = C::get(&c.index_total) + C::get(&c.delete_total);
    let ops_time = C::millis(&c.index_time_nanos) + C::millis(&c.delete_time_nanos);
    json!({
        "docs": {"count": docs, "deleted": deleted, "total_size_in_bytes": store},
        "shard_stats": {"total_count": shards},
        "store": {"size_in_bytes": store, "total_data_set_size_in_bytes": store, "reserved_in_bytes": 0},
        "indexing": {
            "index_total": C::get(&c.index_total), "index_time_in_millis": C::millis(&c.index_time_nanos),
            "index_current": C::get(&c.index_current), "index_failed": C::get(&c.index_failed), "index_failed_due_to_version_conflict": 0,
            "delete_total": C::get(&c.delete_total), "delete_time_in_millis": C::millis(&c.delete_time_nanos),
            "delete_current": C::get(&c.delete_current), "noop_update_total": C::get(&c.noop_update_total), "is_throttled": C::get(&c.index_throttled) > 0, "throttle_time_in_millis": C::millis(&c.index_throttle_nanos),
            "write_load": 0.0
        },
        "get": {
            "total": C::get(&c.get_total), "time_in_millis": C::millis(&c.get_time_nanos),
            "exists_total": C::get(&c.get_exists_total), "exists_time_in_millis": C::millis(&c.get_exists_time_nanos),
            "missing_total": C::get(&c.get_missing_total), "missing_time_in_millis": C::millis(&c.get_missing_time_nanos), "current": C::get(&c.get_current)
        },
        "search": {
            "open_contexts": 0, "query_total": C::get(&c.search_total), "query_time_in_millis": C::millis(&c.query_time_nanos),
            "query_current": C::get(&c.query_current), "query_failure": 0, "fetch_total": C::get(&c.fetch_total), "fetch_time_in_millis": C::millis(&c.fetch_time_nanos), "fetch_current": C::get(&c.fetch_current),
            "fetch_failure": 0, "scroll_total": 0, "scroll_time_in_millis": 0, "scroll_current": 0, "suggest_total": 0, "suggest_time_in_millis": 0, "suggest_current": 0,
            // ES 9.1+ (`mapper.search_load_per_shard`): share of time spent searching.
            "recent_search_load": recent_search_load(c)
        },
        "merges": {
            "current": C::get(&c.merge_current), "current_docs": C::get(&c.merge_current_docs), "current_size_in_bytes": 0, "total": C::get(&c.merge_total), "total_time_in_millis": C::millis(&c.merge_time_nanos),
            "total_docs": C::get(&c.merge_docs), "total_size_in_bytes": C::get(&c.merge_size_bytes),
            "total_stopped_time_in_millis": 0,
            // Node-wide (merges' own throttle sleeps; exact for a node holding one merging shard).
            "total_throttled_time_in_millis": elasticsearch_engine::rustic_core::merge::MERGE_OWN_THROTTLE_NS.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000,
            "total_auto_throttle_in_bytes": elasticsearch_engine::rustic_core::merge::merge_auto_throttle_rate() * shards as u64
        },
        "refresh": {
            "total": C::get(&c.refresh_total), "total_time_in_millis": C::millis(&c.refresh_time_nanos),
            "external_total": C::get(&c.refresh_total), "external_total_time_in_millis": C::millis(&c.refresh_time_nanos), "listeners": 0
        },
        "flush": {"total": C::get(&c.flush_total), "periodic": 0, "total_time_in_millis": C::millis(&c.flush_time_nanos), "total_time_excluding_waiting_on_lock_in_millis": C::millis(&c.flush_time_nanos)},
        "warmer": {"current": 0, "total": C::get(&c.refresh_total), "total_time_in_millis": 0},
        "query_cache": {"memory_size_in_bytes": 0, "total_count": 0, "hit_count": 0, "miss_count": 0, "cache_size": 0, "cache_count": 0, "evictions": 0},
        "fielddata": {"memory_size_in_bytes": 0, "evictions": 0, "fields": {}, "global_ordinals": {"build_time_in_millis": 0}},
        "completion": {"size_in_bytes": 0, "fields": {}},
        "segments": {"count": segs, "memory_in_bytes": 0, "terms_memory_in_bytes": 0, "stored_fields_memory_in_bytes": 0, "term_vectors_memory_in_bytes": 0,
                     "norms_memory_in_bytes": 0, "points_memory_in_bytes": 0, "doc_values_memory_in_bytes": 0, "index_writer_memory_in_bytes": 0,
                     "version_map_memory_in_bytes": 0, "fixed_bit_set_memory_in_bytes": 0, "max_unsafe_auto_id_timestamp": -1, "file_sizes": {}},
        "translog": {"operations": unc, "size_in_bytes": tl, "uncommitted_operations": unc, "uncommitted_size_in_bytes": tl, "earliest_last_modified_age": 0},
        "request_cache": {"memory_size_in_bytes": 0, "evictions": 0, "hit_count": 0, "miss_count": 0},
        "recovery": {"current_as_source": 0, "current_as_target": 0, "throttle_time_in_millis": 0},
        "bulk": {"total_operations": ops, "total_time_in_millis": ops_time, "total_size_in_bytes": 0, "avg_time_in_millis": if ops > 0 { ops_time / ops } else { 0 }, "avg_size_in_bytes": 0}
    })
}

/// The per-copy sections a `level=shards` stats entry ends with: commit
/// point, sequence numbers, retention leases, shard path and search idleness.
pub(crate) fn shard_copy_extras(node: &Node, svc: &IndexService, docs: u64) -> Vec<(String, Value)> {
    let seq: i64 = if docs == 0 { -1 } else { docs as i64 - 1 };
    let data = node.data.to_string_lossy().to_string();
    vec![
        ("commit".into(), json!({"id": svc.uuid, "generation": 1, "user_data": {"local_checkpoint": seq.to_string(), "max_seq_no": seq.to_string(), "max_unsafe_auto_id_timestamp": "-1", "min_retained_seq_no": (seq + 1).to_string(), "es_version": "8537000", "history_uuid": svc.uuid, "translog_uuid": svc.uuid}, "num_docs": docs})),
        ("seq_no".into(), json!({"max_seq_no": seq, "local_checkpoint": seq, "global_checkpoint": seq})),
        ("retention_leases".into(), json!({"primary_term": 1, "version": 1, "leases": [{"id": format!("peer_recovery/{}", elasticsearch_engine::node::NODE_ID), "retaining_seq_no": seq + 1, "timestamp": svc.creation_date, "source": "peer recovery"}]})),
        ("shard_path".into(), json!({"state_path": data, "data_path": data, "is_custom_data_path": false})),
        ("search_idle".into(), json!(false)),
        ("search_idle_time".into(), json!(0)),
    ]
}

/// Open scroll / PIT contexts per index name (`search.open_contexts`,
/// `search.scroll_current`).
fn open_contexts(node: &Node) -> (std::collections::HashMap<String, u64>, std::collections::HashMap<String, u64>) {
    let mut open: std::collections::HashMap<String, u64> = Default::default();
    let mut scroll: std::collections::HashMap<String, u64> = Default::default();
    for s in node.scrolls.lock().values() {
        for t in &s.targets {
            *open.entry(t.svc.name.clone()).or_insert(0) += 1;
            *scroll.entry(t.svc.name.clone()).or_insert(0) += 1;
        }
    }
    for p in node.pits.lock().values() {
        for t in &p.targets {
            *open.entry(t.svc.name.clone()).or_insert(0) += 1;
        }
    }
    (open, scroll)
}

pub(crate) fn add_stats(a: &mut Value, b: &Value) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                match x.get_mut(k) {
                    Some(e) => add_stats(e, v),
                    None => {
                        x.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (a @ Value::Number(_), Value::Number(n)) => {
            *a = match (a.as_u64(), n.as_u64()) {
                (Some(x), Some(y)) => json!(x + y),
                _ => json!(a.as_f64().unwrap_or(0.0) + n.as_f64().unwrap_or(0.0)),
            };
        }
        _ => {}
    }
}

fn stats(node: &Node, req: &Req, index: Option<&str>) -> R {
    let targets = node.resolve(index.unwrap_or("_all"), opts(req))?;
    let raw_metrics: Option<Vec<String>> = req
        .segs
        .iter()
        .skip(if index.is_some() { 2 } else { 1 })
        .next()
        .map(|m| m.split(',').map(str::to_string).collect())
        .or_else(|| req.param("metric").map(|m| m.split(',').map(str::to_string).collect()));
    const KNOWN: &[&str] = &["_all", "store", "indexing", "search", "get", "merge", "flush", "refresh", "query_cache", "fielddata", "docs", "warmer", "completion", "segments", "translog", "request_cache", "recovery", "suggest", "bulk", "shard_stats", "mappings", "dense_vector", "sparse_vector"];
    if let Some(ms) = &raw_metrics {
        let bad: Vec<&String> = ms.iter().filter(|m| !KNOWN.contains(&m.as_str())).collect();
        if !bad.is_empty() {
            // Suggest the closest known metric (one edit away), like ES's did-you-mean.
            let near = |a: &str, b: &str| -> bool {
                let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
                if a.len().abs_diff(b.len()) > 1 {
                    return false;
                }
                let (mut i, mut j, mut diff) = (0, 0, 0);
                while i < a.len() && j < b.len() {
                    if a[i] == b[j] {
                        i += 1;
                        j += 1;
                        continue;
                    }
                    diff += 1;
                    if diff > 1 {
                        return false;
                    }
                    match a.len().cmp(&b.len()) {
                        std::cmp::Ordering::Greater => i += 1,
                        std::cmp::Ordering::Less => j += 1,
                        std::cmp::Ordering::Equal => {
                            i += 1;
                            j += 1;
                        }
                    }
                }
                diff + (a.len() - i) + (b.len() - j) <= 1
            };
            let hint = bad.first().and_then(|b| KNOWN.iter().find(|k| near(b, k))).map(|k| format!(" -> did you mean [{k}]?")).unwrap_or_default();
            return Err(EsError::illegal_argument(format!("request [{}] contains unrecognized metric: [{}]{hint}", req.path, bad.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))));
        }
    }
    // Metric names map to response keys (`merge` -> `merges`).
    let metrics: Option<Vec<String>> = raw_metrics.map(|ms| ms.into_iter().map(|m| if m == "merge" { "merges".to_string() } else { m }).collect());
    let level = req.param("level").map(|s| s.to_string());
    let cluster_level = level.as_deref() == Some("cluster");
    let translog = metrics.as_ref().is_none_or(|ms| ms.iter().any(|m| m == "_all" || m == "translog"));
    // Counter-only requests must not wait for translog/write serialization
    // during maintenance. Full stats and per-copy extras retain their reads.
    let counters_only = level.as_deref() != Some("shards") && counter_metrics_only(metrics.as_deref());
    let pat_list = |k: &str| -> Vec<String> { req.param(k).map(|v| v.split(',').map(str::to_string).collect()).unwrap_or_default() };
    let fields_p = pat_list("fields");
    let fielddata_p = if pat_list("fielddata_fields").is_empty() { fields_p.clone() } else { pat_list("fielddata_fields") };
    let completion_p = if pat_list("completion_fields").is_empty() { fields_p.clone() } else { pat_list("completion_fields") };
    let groups_p = pat_list("groups");
    let seg_file_sizes = req.flag("include_segment_file_sizes")?;
    let filt = |v: Value| -> Value {
        match (&metrics, v) {
            (Some(ms), Value::Object(o)) if !ms.iter().any(|m| m == "_all") => Value::Object(o.into_iter().filter(|(k, _)| ms.contains(k)).collect()),
            (_, v) => v,
        }
    };
    let mut all = json!({});
    let mut per = Map::new();
    let (open, scroll) = open_contexts(node);
    // Stats come from the shard copies: an index whose primaries failed to
    // allocate has none and is not listed (ES), its shards count as not successful.
    let mut failed_copies = 0u64;
    for t in &targets {
        let failed = t.svc.failed_shards().len();
        failed_copies += failed as u64;
        if failed > 0 && t.svc.active_shards() == 0 {
            continue;
        }
        let mut s = if counters_only {
            counter_index_section(&t.svc)
        } else { index_stats_selected(&t.svc, translog) };
        s["search"]["open_contexts"] = json!(open.get(&t.svc.name).copied().unwrap_or(0));
        s["search"]["scroll_current"] = json!(scroll.get(&t.svc.name).copied().unwrap_or(0));
        let docs = s["docs"]["count"].as_u64().unwrap_or(0);
        if !counters_only {
            let sv = sparse_vector_count(&t.svc);
            s["dense_vector"] = elasticsearch_engine::vector_stats::index_stats(&t.svc, true);
            s["sparse_vector"] = json!({"value_count": sv});
        }
        // A closed index has no loaded segments unless asked for.
        // A closed index, or a partially cached searchable snapshot (frozen
        // tier), has no loaded segments unless asked for.
        let unloaded = (!t.svc.is_open() || elasticsearch_engine::snapshot::is_partial_snapshot(&t.svc)) && !req.flag("include_unloaded_segments")?;
        if unloaded {
            s["segments"]["count"] = json!(0);
        }
        if !groups_p.is_empty() {
            let mut g = Map::new();
            for (name, (q, f)) in crate::usage::groups(&t.svc.uuid) {
                if groups_p.iter().any(|p| p == "_all" || elasticsearch_engine::mapping::wildcard_match(p, &name)) {
                    g.insert(name, json!({"query_total": q, "query_time_in_millis": 0, "query_current": 0, "fetch_total": f, "fetch_time_in_millis": 0, "fetch_current": 0}));
                }
            }
            s["search"]["groups"] = Value::Object(g);
        }
        // Text `fielddata` fields a sort / aggregation loaded; completion fields hold their FST.
        let fd_size = |_f: &str| -> u64 { 128 + docs * 8 };
        let used = crate::usage::fielddata_fields(&t.svc.uuid);
        let fd_total: u64 = used.iter().map(|f| fd_size(f)).sum();
        s["fielddata"]["memory_size_in_bytes"] = json!(fd_total);
        if !fielddata_p.is_empty() {
            let m: Map<String, Value> = used.iter().filter(|f| fielddata_p.iter().any(|p| p == "_all" || elasticsearch_engine::mapping::wildcard_match(p, f))).map(|f| (f.clone(), json!({"memory_size_in_bytes": fd_size(f)}))).collect();
            let ords: Map<String, Value> = m.keys().map(|f| (f.clone(), json!({"build_time_in_millis": 0, "shard_max_value_count": docs.max(1)}))).collect();
            s["fielddata"]["fields"] = Value::Object(m);
            s["fielddata"]["global_ordinals"] = json!({"build_time_in_millis": 0, "fields": ords});
        } else if let Some(o) = s["fielddata"].as_object_mut() {
            o.remove("fields");
        }
        let lookup = t.svc.lookup();
        let completion: Vec<String> = lookup.matching("*").into_iter().filter(|n| lookup.get(n).is_some_and(|f| f.declared == elasticsearch_engine::mapping::EsType::Completion)).collect();
        let cs = |_f: &str| -> u64 { if docs > 0 { 64 + docs * 32 } else { 0 } };
        s["completion"]["size_in_bytes"] = json!(completion.iter().map(|f| cs(f)).sum::<u64>());
        if !completion_p.is_empty() {
            let m: Map<String, Value> = completion.iter().filter(|f| completion_p.iter().any(|p| p == "_all" || elasticsearch_engine::mapping::wildcard_match(p, f))).map(|f| (f.clone(), json!({"size_in_bytes": cs(f)}))).collect();
            s["completion"]["fields"] = Value::Object(m);
        } else if let Some(o) = s["completion"].as_object_mut() {
            o.remove("fields");
        }
        if seg_file_sizes {
            s["segments"]["file_sizes"] = if unloaded { json!({}) } else { crate::observability::segment_file_sizes(&t.svc) };
        }
        // `_all` sums the indices without their per-field vector breakdown.
        // Temporarily take the per-field breakdown instead of cloning every
        // statistic just to omit this one subtree from the aggregate.
        let vector_fields = s.get_mut("dense_vector")
            .and_then(|v| v.get_mut("off_heap"))
            .and_then(Value::as_object_mut)
            .and_then(|off| off.remove("fielddata"));
        add_stats(&mut all, &s);
        if let Some(fields) = vector_fields {
            s["dense_vector"]["off_heap"]["fielddata"] = fields;
        }
        all["segments"]["max_unsafe_auto_id_timestamp"] = json!(-1);
        if cluster_level {
            continue;
        }
        let replicas = t.svc.setting_usize("index.number_of_replicas", 1);
        let health = if replicas == 0 { "green" } else { "yellow" };
        let mut entry = json!({"uuid": t.svc.uuid, "health": health, "status": if t.svc.is_open() {"open"} else {"close"}});
        if level.as_deref() == Some("shards") {
            let nshards = t.svc.number_of_shards();
            let mut shards = Map::new();
            for i in 0..nshards {
                let mut ss = s.clone();
                let mut docs = docs;
                if nshards > 1 {
                    // This copy's own numbers.
                    if let Some(sh) = t.svc.shard(i) {
                        let own = shard_stats_selected(&sh, translog);
                        for k in ["docs", "store", "indexing", "get", "merges", "refresh", "flush", "translog"] {
                            ss[k] = own[k].clone();
                        }
                        ss["segments"]["count"] = own["segments"]["count"].clone();
                        for k in ["query_total", "query_time_in_millis", "query_current", "fetch_total", "fetch_time_in_millis", "fetch_current"] {
                            ss["search"][k] = own["search"][k].clone();
                        }
                        docs = own["docs"]["count"].as_u64().unwrap_or(0);
                    }
                }
                let mut copy = Map::new();
                copy.insert("routing".into(), json!({"state": "STARTED", "primary": true, "node": elasticsearch_engine::node::NODE_ID, "relocating_node": null}));
                if let Value::Object(o) = filt(ss) {
                    for (k, v) in o {
                        copy.insert(k, v);
                    }
                }
                for (k, v) in shard_copy_extras(node, &t.svc, docs) {
                    copy.insert(k, v);
                }
                // Durable retention leases of the copy (e.g. CCR followers').
                if let Some(sh) = t.svc.shard(i) {
                    if let Some(Value::Array(a)) = copy.get_mut("retention_leases").and_then(|r| r.get_mut("leases")) {
                        for l in sh.shard.retention_leases() {
                            a.push(json!({"id": l.id, "retaining_seq_no": l.retaining_seq_no, "timestamp": l.timestamp_millis, "source": l.source}));
                        }
                    }
                }
                shards.insert(i.to_string(), json!([Value::Object(copy)]));
            }
            entry["shards"] = Value::Object(shards);
        }
        let selected = filt(s);
        entry["primaries"] = selected.clone();
        entry["total"] = selected;
        per.insert(t.svc.name.clone(), entry);
    }
    let all = filt(all);
    // _shards.total counts configured shards (primaries + replicas), not indices.
    let shard_total: u64 = targets.iter().map(|t| {
        let p = t.svc.setting_usize("index.number_of_shards", 1) as u64;
        let r = t.svc.setting_usize("index.number_of_replicas", 1) as u64;
        p * (r + 1)
    }).sum();
    let mut out = json!({
        "_shards": {"total": shard_total, "successful": shard_total.saturating_sub(failed_copies), "failed": 0},
        "_all": {"primaries": all.clone(), "total": all}
    });
    if !cluster_level {
        out["indices"] = Value::Object(per);
    }
    Ok(Resp::ok(out))
}

/// `GET /{index}/_stats[/{metrics}]` (`params`: the decoded query string)
/// run in process on this engine, without the HTTP dispatch: the JSON body
/// the route answers, `?human` applied as the route's response shaping
/// would (`filter_path` / `pretty` are left to the caller).
///
/// A cluster coordinator sums each shard engine's copy of an index it
/// already resolved, authorized and validated through engine 0's route: a
/// full in-process HTTP round trip per copy (request admission, security,
/// usage counting, an executor hop, serialization and a re-parse) costs
/// more than the statistics themselves.
pub fn index_stats_in_process(node: &Node, index: &str, metrics: Option<&str>, params: std::collections::HashMap<String, String>) -> Result<Value, EsError> {
    let human = matches!(params.get("human").map(String::as_str), Some("") | Some("true"));
    let mut segs = vec![index.to_string(), "_stats".to_string()];
    if let Some(m) = metrics.filter(|m| !m.is_empty()) {
        segs.push(m.to_string());
    }
    let path = format!("/{}", segs.join("/"));
    let req = Req { method: Method::GET, path, segs, params, body: Default::default(), headers: Default::default(),
        access: None, extensions: Default::default(), api_version: crate::RestApiVersion::V9 };
    let mut value = match stats(node, &req, Some(index))?.body {
        crate::RespBody::Json(v) => v,
        crate::RespBody::RetainedJson { value, .. } => value,
        _ => return Err(EsError::internal("index stats rendered no JSON body")),
    };
    if human {
        crate::humanize(&mut value);
    }
    Ok(value)
}

fn open_close(node: &Node, req: &Req, index: &str, open: bool) -> R {
    let mut o = opts(req);
    o.expand_closed = true;
    let mut targets = node.resolve(index, o)?;
    // Data streams cannot be closed or opened: expressions naming one match nothing.
    let named = |n: &str| index.split(',').any(|p| p == n);
    targets.retain(|t| match node.data_stream_of_index(&t.svc.name) {
        None => true,
        Some(ds) => named(&t.svc.name) || named(&ds),
    });
    if targets.is_empty() && !index.contains('*') && index != "_all" && !index.is_empty() {
        return Err(EsError::index_not_found(index));
    }
    let mut indices = Map::new();
    let mut shards_acked = true;
    for t in &targets {
        t.svc.set_open(open)?;
        if !open {
            // Pinned query_then_fetch views must not outlive the open index.
            node.drop_fetch_contexts(&t.svc.name);
        }
        shards_acked &= t.svc.failed_shards().is_empty();
        indices.insert(t.svc.name.clone(), json!({"closed": true}));
    }
    if open {
        Ok(Resp::ok(json!({"acknowledged": true, "shards_acknowledged": shards_acked})))
    } else {
        Ok(Resp::ok(json!({"acknowledged": true, "shards_acknowledged": true, "indices": indices})))
    }
}

/// `routing` on a template alias means both index and search routing.
fn expand_alias_routing(aliases: &mut Map<String, Value>) {
    for (_, def) in aliases.iter_mut() {
        if let Some(d) = def.as_object_mut() {
            if let Some(r) = d.remove("routing") {
                d.entry("index_routing").or_insert(r.clone());
                d.entry("search_routing").or_insert(r);
            }
        }
    }
}

/// Template settings are reported with the `index.` prefix (nested, or flat
/// with `flat_settings`).
/// `?settings_filter=a.*,b`: drop the flat settings matching any pattern
/// (ES `SettingsFilter`, `Regex.simpleMatch`).
fn settings_filter(req: &Req) -> Vec<String> {
    req.param("settings_filter").map(|f| f.split(',').map(str::trim).filter(|p| !p.is_empty()).map(str::to_string).collect()).unwrap_or_default()
}

fn normalize_template_settings(mut v: Value, flat: bool, composable: bool, defaults: bool, filter: &[String]) -> Value {
    let slot = if composable { v.get_mut("template") } else { Some(&mut v) };
    if let Some(o) = slot.and_then(Value::as_object_mut) {
        if let Some(s) = o.get("settings").cloned() {
            let mut norm = settings::normalize_index_settings(&s);
            norm.retain(|k, _| !filter.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, k)));
            o.insert("settings".into(), if flat { settings::as_flat_json(&norm) } else { settings::unflatten(&norm) });
        }
        if let Some(al) = o.get_mut("aliases").and_then(Value::as_object_mut) {
            expand_alias_routing(al);
        }
        // A failure store lifecycle is enabled unless stated otherwise.
        if let Some(lc) = o.get_mut("data_stream_options").and_then(|d| d.get_mut("failure_store")).and_then(|f| f.get_mut("lifecycle")).and_then(Value::as_object_mut) {
            lc.entry("enabled").or_insert(json!(true));
        }
        // A data stream lifecycle is enabled unless stated otherwise.
        if let Some(lc) = o.get_mut("lifecycle").and_then(Value::as_object_mut) {
            lc.entry("enabled").or_insert(json!(true));
            if defaults {
                let retention = lc.get("data_retention").and_then(Value::as_str).map(str::to_string);
                lc.insert("rollover".into(), elasticsearch_engine::ilm::dlm::rollover_defaults(retention.as_deref()));
            }
        }
    }
    v
}

/// A composable template as ES renders it: fields in `ComposableIndexTemplate`
/// order, `composed_of` always present, `data_stream` defaults filled in.
fn render_index_template(v: Value) -> Value {
    let Value::Object(mut o) = v else { return v };
    if let Some(ds) = o.get_mut("data_stream").and_then(Value::as_object_mut) {
        ds.entry("hidden").or_insert(json!(false));
        ds.entry("allow_custom_routing").or_insert(json!(false));
    }
    o.entry("composed_of").or_insert(json!([]));
    let mut out = Map::new();
    for k in ["index_patterns", "template", "composed_of", "priority", "version", "_meta", "data_stream", "allow_auto_create", "ignore_missing_component_templates", "deprecated"] {
        if let Some(v) = o.remove(k) {
            out.insert(k.to_string(), v);
        }
    }
    out.extend(o);
    Value::Object(out)
}

/// Template tracking dates last (`timestampFieldsFromUnixEpochMillis`:
/// the `?human` ISO date before each `*_millis`).
fn render_tracking(v: Value, human: bool) -> Value {
    let Value::Object(mut o) = v else { return v };
    let created = o.remove("created_date_millis");
    let modified = o.remove("modified_date_millis");
    for (base, x) in [("created_date", created), ("modified_date", modified)] {
        if let Some(x) = x.filter(|x| !x.is_null()) {
            if human {
                if let Some(n) = x.as_i64() {
                    o.insert(base.into(), json!(elasticsearch_engine::date::human_date(n)));
                }
            }
            o.insert(format!("{base}_millis"), x);
        }
    }
    Value::Object(o)
}

/// ES always returns `index_patterns` as an array, even when stored as a string.
fn normalize_patterns(v: Value) -> Value {
    let mut v = v;
    if let Some(o) = v.as_object_mut() {
        if let Some(p) = o.get("index_patterns") {
            o.insert("index_patterns".into(), json!(elasticsearch_engine::node::pattern_list(p)));
        }
    }
    v
}

fn templates(node: &Node, req: &Req) -> R {
    let kind = req.seg(0).to_string();
    let name = req.segs.get(1).cloned();
    let m = req.method.clone();
    let get_map = |t: &elasticsearch_engine::node::Templates| -> std::collections::BTreeMap<String, Value> {
        match kind.as_str() {
            "_index_template" => t.index.clone(),
            "_component_template" => t.component.clone(),
            _ => t.legacy.clone(),
        }
    };
    // `_simulate` / `_simulate_index` are actions, not template names.
    if kind == "_index_template" && matches!(name.as_deref(), Some("_simulate") | Some("_simulate_index")) {
        if m != Method::POST {
            // GET/HEAD on the action path: no such template exists.
            if matches!(m, Method::GET | Method::HEAD) {
                return Err(EsError::resource_not_found("index template matching [_simulate] not found"));
            }
            return method_not_allowed(req, &["POST"]);
        }
        let target = req.segs.get(2).map(String::as_str);
        if name.as_deref() == Some("_simulate_index") {
            let index = target.ok_or_else(|| EsError::action_validation(&["index name is missing"]))?;
            let body = req.json()?;
            return Ok(Resp::ok(node.simulate_index(index, &body, req.flag("include_defaults")?)?));
        }
        let body = req.json()?;
        return Ok(Resp::ok(node.simulate_template(target, &body, req.flag("include_defaults")?)?));
    }
    match m {
        Method::PUT | Method::POST => {
            let name = name.ok_or_else(|| EsError::action_validation(&["name is missing"]))?;
            let body = req.json()?;
            if body.is_null() {
                return Err(EsError::action_validation(&["template source is missing"]));
            }
            // Parser: unknown top-level fields (`ComposableIndexTemplate` /
            // `ComponentTemplate` parsers; tracking dates only as `*_millis`).
            if kind != "_template" {
                let (parser, fields): (&str, &[&str]) = if kind == "_index_template" {
                    ("index_template", &["index_patterns", "template", "composed_of", "priority", "version", "_meta", "data_stream", "allow_auto_create", "ignore_missing_component_templates", "deprecated", "created_date_millis", "modified_date_millis"])
                } else {
                    ("component_template", &["template", "version", "_meta", "deprecated", "created_date_millis", "modified_date_millis"])
                };
                if let Some(o) = body.as_object() {
                    if let Some(k) = o.keys().find(|k| !fields.contains(&k.as_str())) {
                        let hint = elasticsearch_engine::jsonpos::suggest(k, fields.iter().copied());
                        return Err(EsError::new(400, "x_content_parse_exception", format!("[1:2] [{parser}] unknown field [{k}]{hint}")));
                    }
                }
            }
            if kind == "_component_template" {
                let errs: Vec<&str> = [("created_date_millis", "Provided a template property which is managed by the system: created_date_millis"), ("modified_date_millis", "Provided a template property which is managed by the system: modified_date_millis")]
                    .iter()
                    .filter(|(k, _)| body.get(*k).is_some_and(|v| !v.is_null()))
                    .map(|(_, m)| *m)
                    .collect();
                if !errs.is_empty() {
                    return Err(EsError::action_validation(&errs));
                }
            }
            if kind != "_component_template" && elasticsearch_engine::node::pattern_list(&body["index_patterns"]).is_empty() {
                return Err(EsError::action_validation(&["index patterns are missing"]));
            }
            if kind == "_index_template" {
                for (k, what) in [("created_date_millis", "created_date"), ("modified_date_millis", "modified_date")] {
                    if body.get(k).is_some_and(|v| !v.is_null()) {
                        return Err(EsError::new(400, "invalid_index_template_exception", format!("index_template [{name}] invalid, cause [provided a template property which is managed by the system: {what}]")));
                    }
                    }
                }
            // `PutComposableIndexTemplateAction` / `PutComponentTemplateAction.validate`.
            if kind != "_template" {
                if let Some(e) = body.pointer("/template/lifecycle").and_then(elasticsearch_engine::ilm::dlm::downsampling_rounds_error) {
                    return Err(EsError::action_validation(&[&format!("template downsampling rounds are not valid: {e}")]));
                }
            }
            // Validate mappings parse.
            let mapping = match kind.as_str() {
                "_template" => body.get("mappings").cloned(),
                _ => body.get("template").and_then(|t| t.get("mappings")).cloned(),
            };
            if let Some(mp) = mapping {
                elasticsearch_engine::mapping::Mappings::parse(&mp, &Default::default())?;
            }
            let create = req.flag("create")?;
            if kind != "_template" {
                if let Some(dso) = body.get("template").and_then(|t| t.get("data_stream_options")) {
                    elasticsearch_engine::datastream::validate_data_stream_options(dso, "template")?;
                }
            }
            if kind == "_index_template" {
                node.validate_composed_of(&body, &name)?;
                // A lifecycle and data stream options need a data stream template.
                if !body.get("data_stream").map(Value::is_object).unwrap_or(false) {
                    if elasticsearch_engine::ilm::dlm::template_lifecycle(node, &body).is_some() {
                        return Err(EsError::illegal_argument(format!("index template [{name}] specifies lifecycle configuration that can only be used in combination with a data stream")));
                    }
                    let composed = node.template_data_stream_options(&body);
                    if composed.as_ref().and_then(Value::as_object).map(|o| !o.is_empty()).unwrap_or(false) {
                        return Err(EsError::illegal_argument(format!("index template [{name}] specifies data stream options that can only be used in combination with a data stream")));
                    }
                }
                // logsdb: `@timestamp` must be a date field (not an object).
                let composed = node.template_for_composition(&body);
                let is_logs = |v: Option<&str>| matches!(v, Some("logsdb" | "logsdb_columnar"));
                if is_logs(composed.get("settings").and_then(|s| s.get("mode")).and_then(Value::as_str))
                    || is_logs(composed.get("settings").and_then(|s| s.get("index")).and_then(|i| i.get("mode")).and_then(Value::as_str))
                {
                    let ts = composed.get("mappings").and_then(|m| m.get("properties")).and_then(|p| p.get("@timestamp"));
                    if let Some(ts) = ts {
                        if ts.get("type").is_none() && (ts.get("properties").is_some() || ts.get("fields").is_some()) {
                            let comps: Vec<String> = body.get("composed_of").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
                            return Err(EsError::illegal_argument(format!("composable template [{name}] template after composition with component templates [{}] is invalid", comps.join(", "))));
                        }
                    }
                }
            }
            // The similarities an index created from it would build (ES
            // validates a template by building one).
            let template_settings = match kind.as_str() {
                "_template" => body.get("settings").cloned(),
                "_index_template" => node.template_for_composition(&body).get("settings").cloned(),
                _ => body.get("template").and_then(|t| t.get("settings")).cloned(),
            };
            if let Some(s) = template_settings {
                elasticsearch_engine::similarity::check_ranges(&elasticsearch_engine::settings::normalize_index_settings(&s))?;
            }
            // Strict columnar index modes: the template's (composed)
            // settings and mappings must be valid for the mode.
            if kind != "_template" {
                let (settings, mappings) = if kind == "_index_template" {
                    let c = node.template_for_composition(&body);
                    (c.get("settings").cloned().unwrap_or(Value::Null), c.get("mappings").cloned().unwrap_or(Value::Null))
                } else {
                    let t = body.get("template");
                    (t.and_then(|t| t.get("settings")).cloned().unwrap_or(Value::Null), t.and_then(|t| t.get("mappings")).cloned().unwrap_or(Value::Null))
                };
                let flat = elasticsearch_engine::settings::normalize_index_settings(&settings);
                let mode = flat.get("index.mode").and_then(Value::as_str).map(str::to_ascii_lowercase);
                if let Some(m) = mode.filter(|m| matches!(m.as_str(), "columnar" | "logsdb_columnar")) {
                    let recovery_off = flat.get("index.recovery.use_synthetic_source").is_some_and(|v| v.as_bool() == Some(false) || v.as_str() == Some("false"));
                    if recovery_off {
                        let msg = format!("The setting [index.recovery.use_synthetic_source] must not be false when [index.mode] is set to [{}].", m.to_ascii_uppercase());
                        return Err(if kind == "_index_template" {
                            EsError::new(400, "invalid_index_template_exception", format!("index_template [{name}] invalid, cause [Validation Failed: 1: {msg};]"))
                        } else {
                            EsError::illegal_argument(msg)
                        });
                    }
                    if mappings.is_object() {
                        let mut reg = elasticsearch_engine::analysis::AnalysisRegistry::default();
                        reg.index_mode = Some(m.clone());
                        if let Err(e) = elasticsearch_engine::mapping::Mappings::parse(&mappings, &reg) {
                            if kind == "_index_template" {
                                return Err(EsError::illegal_argument(format!("composable template [{name}] template after composition is invalid")).caused_by(EsError::illegal_argument(e.reason.clone()).caused_by(e)));
                            }
                            return Err(e);
                        }
                    }
                }
            }
            let mut t = node.templates.write();
            let target = match kind.as_str() {
                "_index_template" => &mut t.index,
                "_component_template" => &mut t.component,
                _ => &mut t.legacy,
            };
            if create && target.contains_key(&name) {
                return Err(EsError::illegal_argument(format!("index template [{name}] already exists")));
            }
            let mut body = body;
            if kind != "_template" {
                // Tracking info: `created_date_millis` kept from the existing
                // template, `modified_date_millis` = now; re-putting an
                // identical template changes nothing.
                let strip = |v: &Value| {
                    let mut v = v.clone();
                    if let Some(o) = v.as_object_mut() {
                        o.remove("created_date_millis");
                        o.remove("modified_date_millis");
                    }
                    v
                };
                let now = elasticsearch_engine::date::now_millis();
                match target.get(&name) {
                    Some(existing) if strip(existing) == body => return Ok(Resp::ack()),
                    Some(existing) => {
                        if let Some(c) = existing.get("created_date_millis").filter(|v| !v.is_null()).cloned() {
                            body["created_date_millis"] = c;
                        }
                        body["modified_date_millis"] = json!(now);
                    }
                    None => {
                        body["created_date_millis"] = json!(now);
                        body["modified_date_millis"] = json!(now);
                    }
                }
            }
            target.insert(name, body);
            drop(t);
            node.persist_state()?;
            Ok(Resp::ack())
        }
        Method::DELETE => {
            let name = name.ok_or_else(|| EsError::action_validation(&["name is missing"]))?;
            let mut t = node.templates.write();
            let target = match kind.as_str() {
                "_index_template" => &mut t.index,
                "_component_template" => &mut t.component,
                _ => &mut t.legacy,
            };
            let keys: Vec<String> = target.keys().filter(|k| name.split(',').any(|n| elasticsearch_engine::mapping::wildcard_match(n, k))).cloned().collect();
            if keys.is_empty() {
                let what = if kind == "_component_template" { "component template" } else { "index template" };
                return Err(EsError::resource_not_found(format!("{what} matching [{name}] not found")));
            }
            for k in keys {
                target.remove(&k);
            }
            drop(t);
            node.persist_state()?;
            // The template registries reinstall a deleted built-in template
            // on the cluster-state change.
            elasticsearch_engine::registry::ensure(node);
            Ok(Resp::ack())
        }
        Method::GET | Method::HEAD => {
            let flat_settings = req.flag("flat_settings")?;
            let include_defaults = req.flag("include_defaults")?;
            let human = req.flag("human")?;
            let filter = settings_filter(req);
            let all = get_map(&node.templates.read());
            let sel: Vec<(String, Value)> = all
                .into_iter()
                .filter(|(k, _)| name.as_ref().map(|n| n.split(',').any(|p| elasticsearch_engine::mapping::wildcard_match(p, k))).unwrap_or(true))
                .collect();
            if kind == "_index_template" && name.as_deref().map(|n| n.contains(',')).unwrap_or(false) {
                return Err(EsError::new(400, "illegal_argument_exception", "index template name must not contain commas"));
            }
            let wildcard_get = name.as_deref().map(|n| n.contains('*')).unwrap_or(false);
            if sel.is_empty() && wildcard_get && kind != "_template" && m != Method::HEAD {
                let key = if kind == "_component_template" { "component_templates" } else { "index_templates" };
                return Ok(Resp::json(404, json!({ key: [] })));
            }
            if sel.is_empty() && name.is_some() {
                if m == Method::HEAD {
                    return Ok(Resp::empty(404));
                }
                let n = name.unwrap();
                let what = match kind.as_str() {
                    "_component_template" => "component template",
                    "_index_template" => "index template",
                    _ => "index template",
                };
                if kind == "_template" {
                    return Ok(Resp::json(404, json!({})));
                }
                return Err(EsError::resource_not_found(format!("{what} matching [{n}] not found")));
            }
            if m == Method::HEAD {
                return Ok(Resp::empty(200));
            }
            Ok(Resp::ok(match kind.as_str() {
                "_index_template" => json!({"index_templates": sel.into_iter().map(|(k, v)| json!({"name": k, "index_template": render_tracking(render_index_template(normalize_template_settings(normalize_patterns(v), flat_settings, true, include_defaults, &filter)), human)})).collect::<Vec<_>>()}),
                "_component_template" => json!({"component_templates": sel.into_iter().map(|(k, v)| json!({"name": k, "component_template": render_tracking(normalize_template_settings(v, flat_settings, true, include_defaults, &filter), human)})).collect::<Vec<_>>()}),
                _ => Value::Object(sel.into_iter().map(|(k, mut v)| {
                    if let Some(o) = v.as_object_mut() {
                        o.entry("order").or_insert(json!(0));
                        o.entry("settings").or_insert(json!({}));
                        o.entry("mappings").or_insert(json!({}));
                        o.entry("aliases").or_insert(json!({}));
                        if let Some(p) = o.get("index_patterns") {
                            o.insert("index_patterns".into(), json!(elasticsearch_engine::node::pattern_list(p)));
                        }
                        // Legacy templates store settings without the `index.`
                        // prefix; ES reports them prefixed.
                        if let Some(s) = o.get("settings").cloned() {
                            let mut norm = settings::normalize_index_settings(&s);
                            norm.retain(|k, _| !filter.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, k)));
                            o.insert("settings".into(), if flat_settings { settings::as_flat_json(&norm) } else { settings::unflatten(&norm) });
                        }
                        // `routing` on a template alias means both index and search routing.
                        if let Some(al) = o.get_mut("aliases").and_then(Value::as_object_mut) {
                            expand_alias_routing(al);
                        }
                    }
                    (k, v)
                }).collect()),
            }))
        }
        _ => method_not_allowed(req, &["GET", "HEAD", "PUT", "POST", "DELETE"]),
    }
}

fn resolve_index(node: &Node, req: &Req) -> R {
    let name = req.seg(2).to_string();
    let mut o = opts(req);
    o.ignore_unavailable = true;
    o.expand_closed = true;
    let targets = node.resolve(&name, o)?;
    let mut indices = Vec::new();
    let mut aliases: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for t in &targets {
        let al: Vec<String> = t.svc.aliases().keys().cloned().collect();
        for a in &al {
            aliases.entry(a.clone()).or_default().push(t.svc.name.clone());
        }
        let mut idx = json!({"name": t.svc.name, "attributes": [if t.svc.is_open() {"open"} else {"closed"}]});
        if !al.is_empty() {
            idx["aliases"] = json!(al);
        }
        indices.push(idx);
    }
    Ok(Resp::ok(json!({
        "indices": indices,
        "aliases": aliases.into_iter().map(|(k, v)| json!({"name": k, "indices": v})).collect::<Vec<_>>(),
        "data_streams": node.data_streams_matching(if name.is_empty() { "*" } else { &name }, true)
            .into_iter()
            .map(|d| json!({"name": d.name, "backing_indices": d.indices, "timestamp_field": "@timestamp"}))
            .collect::<Vec<_>>()
    })))
}

#[cfg(test)]
mod vector_count_tests {
    use super::*;
    use elasticsearch_engine::index::WriteReq;
    use elasticsearch_engine::shard::WriteCond;

    #[test]
    fn counter_stats_do_not_read_locked_core() {
        let dir = tempfile::tempdir().unwrap();
        let node = Arc::new(Node::open(dir.path()).unwrap());
        let svc = node.create_index("counter-stats", &json!({})).unwrap();
        let request = || Req { method: Method::GET, path: "/counter-stats/_stats/merge,flush,indexing".into(),
            segs: vec!["counter-stats".into(), "_stats".into(), "merge,flush,indexing".into()],
            params: Default::default(), body: Default::default(), headers: Default::default(),
            access: None, extensions: Default::default(), api_version: crate::RestApiVersion::V9 };
        let expected = stats(&node, &request(), Some("counter-stats")).unwrap();
        let shard = svc.local_shards()[0].clone();
        let guard = shard.shard.core.write();
        let (tx, rx) = std::sync::mpsc::channel();
        let copy = node.clone();
        let req = request();
        let worker = std::thread::spawn(move || { tx.send(stats(&copy, &req, Some("counter-stats"))).unwrap(); });
        let result = rx.recv_timeout(std::time::Duration::from_secs(1));
        drop(guard);
        worker.join().unwrap();
        let response = result.expect("counter metrics waited for an unrelated core lock").unwrap();
        assert_eq!(response.status, 200);
        match (expected.body, response.body) {
            (crate::RespBody::Json(expected), crate::RespBody::Json(actual)) => assert_eq!(actual, expected),
            _ => panic!("expected JSON stats"),
        }
    }

    /// Docs / segments stats and `_cat/segments` read the published reader
    /// view when the core is write-locked (a flush, merge install or write
    /// batch): they return without waiting and seal nothing.
    #[test]
    fn segment_and_doc_stats_and_cat_segments_do_not_wait_for_a_locked_core() {
        let dir = tempfile::tempdir().unwrap();
        let node = Arc::new(Node::open(dir.path()).unwrap());
        let svc = node.create_index("locked-stats", &json!({"settings": {"refresh_interval": "-1"}})).unwrap();
        let docs: Vec<WriteReq> = (0..5).map(|i| WriteReq::Index { id: Some(i.to_string()), source: json!({"n": i}).to_string(), routing: None, cond: WriteCond::default() }).collect();
        assert!(svc.write(docs, true).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        let req = |path: &str, segs: &[&str]| Req { method: Method::GET, path: path.into(),
            segs: segs.iter().map(|s| s.to_string()).collect(),
            params: Default::default(), body: Default::default(), headers: Default::default(),
            access: None, extensions: Default::default(), api_version: crate::RestApiVersion::V9 };
        let shard = svc.local_shards()[0].clone();
        let guard = shard.shard.core.write();
        let (tx, rx) = std::sync::mpsc::channel();
        let copy = node.clone();
        let stats_req = req("/locked-stats/_stats/docs,segments", &["locked-stats", "_stats", "docs,segments"]);
        let cat_req = req("/_cat/segments/locked-stats", &["_cat", "segments", "locked-stats"]);
        let worker = std::thread::spawn(move || {
            let stats = stats(&copy, &stats_req, Some("locked-stats")).map(|r| r.status);
            let cat = crate::cat::segments(&copy, &cat_req, Some("locked-stats")).map(|r| r.status);
            tx.send((stats, cat)).unwrap();
        });
        let result = rx.recv_timeout(std::time::Duration::from_secs(10));
        drop(guard);
        worker.join().unwrap();
        let (stats, cat) = result.expect("segment / doc stats or _cat/segments waited for the core lock");
        assert_eq!(stats.unwrap(), 200);
        assert_eq!(cat.unwrap(), 200);
        // Nothing was sealed for them: the refreshed docs are still one
        // published memtable.
        assert_eq!(shard.shard.reader_view().num_memtables(), 1);
    }

    #[test]
    fn counts_per_shard_copy_and_over_all_local_copies() {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index("v", &json!({"settings": {"number_of_shards": 2}, "mappings": {"properties": {"s": {"type": "sparse_vector"}, "k": {"type": "keyword"}}}})).unwrap();
        let docs: Vec<WriteReq> = (0..6).map(|i| WriteReq::Index { id: Some(i.to_string()), source: if i < 5 { json!({"s": {"a": 1.0}}).to_string() } else { json!({"k": "x"}).to_string() }, routing: None, cond: WriteCond::default() }).collect();
        svc.write(docs, true);
        svc.refresh().unwrap();
        let (_, all) = vector_counts_in(&svc, None);
        assert_eq!(all, 5);
        assert_eq!(sparse_vector_count(&svc), all);
        // The per-copy counts add up to the index's.
        assert_eq!(vector_counts_in(&svc, Some(0)).1 + vector_counts_in(&svc, Some(1)).1, 5);
        // A shard with no local copy counts nothing.
        assert_eq!(vector_counts_in(&svc, Some(7)), (0, 0));
    }
}

#[cfg(test)]
mod stats_in_process_tests {
    use super::*;
    use elasticsearch_engine::index::WriteReq;
    use elasticsearch_engine::shard::WriteCond;

    fn req(path: &str, params: &[(&str, &str)]) -> Req {
        Req { method: Method::GET, path: path.into(), segs: path.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect(),
            params: params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body: Default::default(), headers: Default::default(),
            access: None, extensions: Default::default(), api_version: crate::RestApiVersion::V9 }
    }

    /// Drop the process-wide merge auto-throttle rate, which merges in tests
    /// running in parallel move between the two renderings.
    fn stable(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.retain(|k, _| !k.starts_with("total_auto_throttle"));
                m.values_mut().for_each(stable);
            }
            Value::Array(a) => a.iter_mut().for_each(stable),
            _ => {}
        }
    }

    fn json_of(r: Resp) -> Value {
        match r.body {
            crate::RespBody::Json(v) => v,
            _ => panic!("expected JSON stats"),
        }
    }

    fn node_with_docs(name: &str, mappings: Value, docs: Vec<Value>) -> (tempfile::TempDir, Arc<Node>, Arc<IndexService>) {
        let dir = tempfile::tempdir().unwrap();
        let node = Node::open(dir.path()).unwrap();
        let svc = node.create_index(name, &json!({"settings": {"refresh_interval": "-1"}, "mappings": mappings})).unwrap();
        let writes: Vec<WriteReq> = docs.into_iter().enumerate().map(|(i, d)| WriteReq::Index { id: Some(i.to_string()), source: d.to_string(), routing: None, cond: WriteCond::default() }).collect();
        assert!(svc.write(writes, true).iter().all(|r| r.is_ok()));
        svc.refresh().unwrap();
        (dir, node, svc)
    }

    /// The in-process call renders what the route's handler renders, for
    /// the coordinator's per-copy metric list, every metric, `level=shards`
    /// and `?human` (applied as the route's response shaping applies it).
    #[test]
    fn in_process_index_stats_match_the_handler() {
        let (_dir, node, _svc) = node_with_docs("inproc", json!({}), (0..5).map(|n| json!({"n": n, "t": format!("text {n}")})).collect());
        const COPY_METRICS: &str = "completion,fielddata,indexing,get,search,merge,refresh,flush,warmer,query_cache,segments,translog,request_cache,recovery,bulk,shard_stats,dense_vector,sparse_vector";
        for (metrics, params) in [
            (Some(COPY_METRICS), vec![]),
            (None, vec![]),
            (Some("_all"), vec![]),
            (Some("docs,store"), vec![]),
            (Some(COPY_METRICS), vec![("level", "shards")]),
            (Some(COPY_METRICS), vec![("human", "true")]),
            (None, vec![("human", "")]),
        ] {
            let path = match metrics { Some(m) => format!("/inproc/_stats/{m}"), None => "/inproc/_stats".to_string() };
            let mut expected = json_of(stats(&node, &req(&path, &params), Some("inproc")).unwrap());
            if params.iter().any(|(k, _)| *k == "human") {
                crate::humanize(&mut expected);
            }
            let mut actual = index_stats_in_process(&node, "inproc", metrics, params.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()).unwrap();
            stable(&mut actual);
            stable(&mut expected);
            assert_eq!(actual, expected, "{path} {params:?}");
            assert_eq!(actual["indices"]["inproc"]["primaries"]["docs"]["count"].as_u64().unwrap_or(5), 5, "{path}");
        }
        // Errors are the handler's errors (an unknown index).
        let missing = index_stats_in_process(&node, "nope", None, Default::default()).unwrap_err();
        assert_eq!(missing.status, stats(&node, &req("/nope/_stats", &[]), Some("nope")).err().map(|e| e.status).unwrap_or(0));
    }

    /// `request_cache` stats (ES `RequestCacheStats`): a repeated
    /// `request_cache=true` size-0 aggregation is a miss, then a hit, and
    /// the cached response's bytes count toward the index's memory; the
    /// index and node stats APIs render them.
    #[test]
    fn request_cache_stats_count_hits_misses_and_memory() {
        let (_dir, node, svc) = node_with_docs("rc", json!({"properties": {"ts": {"type": "date"}}}), (0..20).map(|n| json!({"ts": 1_600_000_000_000i64 + n * 86_400_000})).collect());
        let before = crate::request_cache::index_stats(&svc.uuid);
        let body = json!({"size": 0, "aggs": {"over_time": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}}});
        let params: std::collections::HashMap<String, String> = [("request_cache".to_string(), "true".to_string())].into_iter().collect();
        let first = crate::search_api::run_search(&node, &body, &params, Some("rc"), Default::default(), None).unwrap();
        let second = crate::search_api::run_search(&node, &body, &params, Some("rc"), Default::default(), None).unwrap();
        let strip = |s: &str| { let mut v: Value = serde_json::from_str(s).unwrap(); v.as_object_mut().unwrap().remove("took"); v };
        assert_eq!(strip(&first), strip(&second));
        let after = crate::request_cache::index_stats(&svc.uuid);
        assert_eq!(after.miss_count - before.miss_count, 1);
        assert_eq!(after.hit_count - before.hit_count, 1);
        assert!(after.memory_size_in_bytes > 0);
        let stats = json_of(stats(&node, &req("/rc/_stats/request_cache", &[]), Some("rc")).unwrap());
        let section = &stats["indices"]["rc"]["primaries"]["request_cache"];
        assert_eq!(section["hit_count"].as_u64(), Some(after.hit_count), "{stats}");
        assert_eq!(section["miss_count"].as_u64(), Some(after.miss_count));
        assert_eq!(section["memory_size_in_bytes"].as_u64(), Some(after.memory_size_in_bytes));
        // Clearing the cache releases the memory.
        crate::request_cache::clear(&[svc.uuid.clone()]);
        assert_eq!(crate::request_cache::index_stats(&svc.uuid).memory_size_in_bytes, 0);
    }

    /// One write-stats sample per shard gives the figures the two separate
    /// `IndexService` sums gave.
    #[test]
    fn translog_totals_match_the_index_sums() {
        let (_dir, _node, svc) = node_with_docs("tl", json!({}), (0..7).map(|n| json!({"n": n})).collect());
        assert_eq!(translog_totals(&svc), (svc.translog_size(), svc.uncommitted_ops().max(0)));
        let s = index_stats_selected(&svc, true);
        assert_eq!(s["translog"]["size_in_bytes"].as_u64(), Some(svc.translog_size()));
        assert_eq!(s["translog"]["uncommitted_operations"].as_i64(), Some(svc.uncommitted_ops().max(0)));
        let none = index_stats_selected(&svc, false);
        assert_eq!((none["translog"]["size_in_bytes"].as_u64(), none["translog"]["operations"].as_i64()), (Some(0), Some(0)));
    }

    /// The sparse count shortcut (no `sparse_vector` field resolved) agrees
    /// with counting through the mappings, with and without such a field.
    #[test]
    fn sparse_vector_count_shortcut_agrees_with_the_mapping_walk() {
        let (_d1, _n1, plain) = node_with_docs("plain", json!({"properties": {"k": {"type": "keyword"}}}), vec![json!({"k": "a"}), json!({"k": "b"})]);
        assert_eq!(sparse_vector_count(&plain), 0);
        assert_eq!(sparse_vector_count(&plain), vector_counts_selected(&plain, None, false).1);
        let (_d2, _n2, sparse) = node_with_docs("sparse", json!({"properties": {"o": {"properties": {"sv": {"type": "sparse_vector"}}}, "k": {"type": "keyword"}}}),
            vec![json!({"o": {"sv": {"a": 1.0, "b": 0.5}}}), json!({"k": "x"}), json!({"o": {"sv": {"c": 2.0}}})]);
        // The shortcut's premise: a mapped (object-nested) sparse_vector is a
        // resolved field, so its index takes the counting path.
        assert!(sparse.lookup().fields.values().any(|f| f.declared == elasticsearch_engine::mapping::EsType::SparseVector));
        assert_eq!(sparse_vector_count(&sparse), vector_counts_selected(&sparse, None, false).1);
    }
}
