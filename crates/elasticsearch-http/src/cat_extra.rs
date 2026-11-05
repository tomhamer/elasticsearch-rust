//! Remaining `_cat` APIs: fielddata, pending_tasks, recovery, repositories,
//! snapshots, tasks, component_templates, circuit_breaker, and the
//! transform / ML APIs. Rendering is shared with [`crate::cat`].

use crate::cat::{render, Table};
use crate::observability::{self, recovery_info, time_fmt};
use crate::util::bytes_fmt;
use crate::{Req, Resp, R};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ResolveOpts, NODE_ID};
use serde_json::{json, Value};

type Col = (&'static str, &'static [&'static str], bool);

pub fn handle(node: &Node, req: &Req, what: &str, arg: Option<&str>) -> R {
    match what {
        "fielddata" => fielddata(node, req, arg),
        "pending_tasks" => render(req, Table { cols: vec![("insertOrder", &["o"], true), ("timeInQueue", &["t"], true), ("priority", &["p"], true), ("source", &["s"], true)], rows: vec![], numeric: vec!["insertOrder"] }),
        "recovery" => recovery(node, req, arg),
        "repositories" => repositories(node, req),
        "snapshots" => snapshots(node, req, arg),
        "tasks" => tasks(node, req),
        "component_templates" => component_templates(node, req, arg),
        "circuit_breaker" => circuit_breaker(node, req, arg),
        "transforms" => transforms(node, req, arg),
        "ml" => {
            let kind = req.seg(2).to_string();
            let kind = if kind == "data_frame" { format!("data_frame/{}", req.seg(3)) } else { kind };
            let id_seg = if kind.starts_with("data_frame") { 4 } else { 3 };
            let id = req.segs.get(id_seg).cloned();
            match kind.as_str() {
                k if k.starts_with("data_frame") => crate::ml::dfa::cat(node, req, id.as_deref()),
                "anomaly_detectors" => ml_jobs(node, req, id.as_deref()),
                "datafeeds" => ml_datafeeds(node, req, id.as_deref()),
                "trained_models" => ml_trained_models(node, req, id.as_deref()),
                _ => ml_stub(req, id.as_deref(), &kind),
            }
        }
        _ => crate::not_found_handler(req),
    }
}

fn fielddata(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let filter: Option<Vec<String>> = arg.map(str::to_string).or_else(|| req.param("fields").map(str::to_string)).map(|f| f.split(',').map(|x| x.trim().to_string()).collect());
    // Text fields with `fielddata: true` hold their terms on heap in ES; size
    // them from the terms dictionary.
    let mut sizes: std::collections::BTreeMap<String, u64> = Default::default();
    for svc in node.all_indices() {
        if !svc.is_open() {
            continue;
        }
        let mj = svc.meta().mappings.to_json();
        let mut fields = Vec::new();
        fn walk(props: &serde_json::Map<String, Value>, prefix: &str, out: &mut Vec<String>) {
            for (k, v) in props {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                if v.get("type").and_then(Value::as_str) == Some("text") && v.get("fielddata").map(|f| f == &json!(true) || f == &json!("true")).unwrap_or(false) {
                    out.push(path.clone());
                }
                if let Some(p) = v.get("properties").and_then(Value::as_object) {
                    walk(p, &path, out);
                }
                if let Some(p) = v.get("fields").and_then(Value::as_object) {
                    walk(p, &path, out);
                }
            }
        }
        if let Some(p) = mj.get("properties").and_then(Value::as_object) {
            walk(p, "", &mut fields);
        }
        let lk = svc.lookup();
        let copies = svc.local_shards();
        // Published reader views: no shard lock is held while reading.
        let views: Vec<_> = copies.iter().map(|c| c.shard.reader_view()).collect();
        let leaves: Vec<_> = views.iter().flat_map(|v| v.leaves()).collect();
        for f in fields {
            if let Some(f2) = &filter {
                if !f2.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, &f)) {
                    continue;
                }
            }
            // Only fielddata a sort or aggregation loaded on this node.
            if !elasticsearch_engine::fielddata_cache::loaded(&svc.name, &f) {
                continue;
            }
            let Some(id) = lk.get(&f).and_then(|r| r.core_id) else { continue };
            let mut n = 0u64;
            for l in &leaves {
                for (t, _) in l.term_range(id, None, None) {
                    n += t.len() as u64 + 8;
                }
            }
            if n > 0 {
                *sizes.entry(f).or_insert(0) += n;
            }
        }
    }
    let bytes = req.param("bytes");
    let rows = sizes.into_iter().map(|(f, n)| vec![NODE_ID.to_string(), "127.0.0.1".into(), "127.0.0.1".into(), node.name.clone(), f, bytes_fmt(n, bytes)]).collect();
    render(
        req,
        Table {
            cols: vec![("id", &[], true), ("host", &["h"], true), ("ip", &[], true), ("node", &["n"], true), ("field", &["f"], true), ("size", &[], true)],
            rows,
            numeric: vec![],
        },
    )
}

fn recovery(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let mut o = ResolveOpts { expand_closed: true, ..Default::default() };
    if arg.is_none() {
        o.allow_no_indices = true;
    }
    let targets = node.resolve(arg.unwrap_or("_all"), o)?;
    let active_only = req.flag("active_only")?;
    let bytes = req.param("bytes");
    let mut rows = Vec::new();
    if !active_only {
        for t in &targets {
            let (ty, start, stop) = recovery_info(node, &t.svc);
            let iso = |ms: i64| elasticsearch_engine::date::from_millis(ms).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
            rows.push(vec![
                t.svc.name.clone(),
                "0".into(),
                iso(start),
                start.to_string(),
                iso(stop),
                stop.to_string(),
                time_fmt((stop - start).max(0) as u64),
                ty.to_lowercase(),
                "done".into(),
                "n/a".into(),
                "n/a".into(),
                "127.0.0.1".into(),
                node.name.clone(),
                "n/a".into(),
                "n/a".into(),
                "0".into(),
                "0".into(),
                "0.0%".into(),
                "0".into(),
                bytes_fmt(0, bytes),
                bytes_fmt(0, bytes),
                "0.0%".into(),
                bytes_fmt(0, bytes),
                "0".into(),
                "0".into(),
                "100.0%".into(),
            ]);
        }
    }
    let cols: Vec<Col> = vec![
        ("index", &["i", "idx"], true),
        ("shard", &["s", "sh"], true),
        ("start_time", &["start"], false),
        ("start_time_millis", &["start_millis"], false),
        ("stop_time", &["stop"], false),
        ("stop_time_millis", &["stop_millis"], false),
        ("time", &["t", "ti"], true),
        ("type", &["ty"], true),
        ("stage", &["st"], true),
        ("source_host", &["shost"], true),
        ("source_node", &["snode"], true),
        ("target_host", &["thost"], true),
        ("target_node", &["tnode"], true),
        ("repository", &["rep"], true),
        ("snapshot", &["snap"], true),
        ("files", &["f"], true),
        ("files_recovered", &["fr"], true),
        ("files_percent", &["fp"], true),
        ("files_total", &["tf"], true),
        ("bytes", &["b"], true),
        ("bytes_recovered", &["br"], true),
        ("bytes_percent", &["bp"], true),
        ("bytes_total", &["tb"], true),
        ("translog_ops", &["to"], true),
        ("translog_ops_recovered", &["tor"], true),
        ("translog_ops_percent", &["top"], true),
    ];
    render(req, Table { cols, rows, numeric: vec!["shard", "start_time_millis", "stop_time_millis", "files", "files_recovered", "files_total", "translog_ops", "translog_ops_recovered"] })
}

fn repositories(node: &Node, req: &Req) -> R {
    let repos = node.snapshots.get_repositories(None).unwrap_or_default();
    let rows: Vec<Vec<String>> = repos
        .as_object()
        .map(|m| m.iter().map(|(k, v)| vec![k.clone(), v["type"].as_str().unwrap_or("fs").to_string()]).collect())
        .unwrap_or_default();
    render(req, Table { cols: vec![("id", &["id", "repoId"], true), ("type", &["t", "type"], true)], rows, numeric: vec![] })
}

fn snapshots(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let ignore = req.flag("ignore_unavailable")?;
    let repo = arg.or(req.param("repository")).unwrap_or("_all");
    let repo_names: Vec<String> = if matches!(repo, "_all" | "*") {
        node.snapshots.repository_names()
    } else {
        let mut found = Vec::new();
        for r in repo.split(',') {
            match node.snapshots.get_repositories(Some(r)) {
                Ok(v) => found.extend(v.as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()).unwrap_or_default()),
                Err(_) if ignore => {}
                Err(e) => return Err(e),
            }
        }
        found
    };
    let hms = |ms: i64| elasticsearch_engine::date::from_millis(ms).format("%H:%M:%S").to_string();
    let mut metas: Vec<(i64, Vec<String>)> = Vec::new();
    for r in &repo_names {
        for m in node.snapshots.get_snapshots(r, "_all", ignore).unwrap_or_default() {
            let (st, en) = (m["start_time_in_millis"].as_i64().unwrap_or(0), m["end_time_in_millis"].as_i64().unwrap_or(0));
            let shards = |k: &str| m["shards"][k].as_u64().unwrap_or(0).to_string();
            let dur = en - st;
            metas.push((st, vec![
                m["snapshot"].as_str().unwrap_or("").to_string(),
                r.clone(),
                m["state"].as_str().unwrap_or("SUCCESS").to_string(),
                (st / 1000).to_string(), hms(st), (en / 1000).to_string(), hms(en),
                if dur < 1000 { format!("{dur}ms") } else { format!("{:.1}s", dur as f64 / 1000.0) },
                m["indices"].as_array().map(|a| a.len()).unwrap_or(0).to_string(),
                shards("successful"), shards("failed"), shards("total"),
                String::new(),
            ]));
        }
    }
    metas.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1[0].cmp(&b.1[0])));
    let rows: Vec<Vec<String>> = metas.into_iter().map(|(_, r)| r).collect();
    let cols: Vec<Col> = vec![
        ("id", &["snapshot"], true),
        ("repository", &["re", "repo"], true),
        ("status", &["s"], true),
        ("start_epoch", &["ste", "startEpoch"], true),
        ("start_time", &["sti", "startTime"], true),
        ("end_epoch", &["ete", "endEpoch"], true),
        ("end_time", &["eti", "endTime"], true),
        ("duration", &["dur"], true),
        ("indices", &["i"], true),
        ("successful_shards", &["ss"], true),
        ("failed_shards", &["fs"], true),
        ("total_shards", &["ts"], true),
        ("reason", &["r"], false),
    ];
    render(req, Table { cols, rows, numeric: vec!["start_epoch", "end_epoch", "indices", "successful_shards", "failed_shards", "total_shards"] })
}

fn tasks(node: &Node, req: &Req) -> R {
    let detailed = req.flag("detailed")?;
    let actions: Option<Vec<String>> = req.param("actions").map(|a| a.split(',').map(str::to_string).collect());
    let parent_filter = req.param("parent_task_id").map(str::to_string);
    let opaque = req.headers.get("x-opaque-id").and_then(|v| v.to_str().ok()).map(str::to_string);
    // The listing itself is a task (as in ES): a parent and its node-level child.
    let mut headers = serde_json::Map::new();
    if let Some(o) = &opaque {
        headers.insert("X-Opaque-Id".into(), json!(o));
    }
    let parent = node.tasks.register("cluster:monitor/tasks/lists", None, false, false, headers.clone());
    let child = node.tasks.register("cluster:monitor/tasks/lists[n]", None, false, false, headers);
    let mut rows = Vec::new();
    for t in node.tasks.running() {
        if let Some(a) = &actions {
            if !a.iter().any(|p| elasticsearch_engine::mapping::wildcard_match(p, &t.action)) {
                continue;
            }
        }
        let info = t.info(detailed);
        let mut parent_id = info.get("parent_task_id").and_then(Value::as_str).map(str::to_string);
        if t.id == child.id {
            parent_id = Some(parent.task_id());
        }
        if let Some(pf) = &parent_filter {
            if parent_id.as_deref() != Some(pf.as_str()) {
                continue;
            }
        }
        let start = info["start_time_in_millis"].as_i64().unwrap_or(0);
        let running_ns = info["running_time_in_nanos"].as_u64().unwrap_or(0);
        let ts = elasticsearch_engine::date::from_millis(start).format("%H:%M:%S").to_string();
        let xo = t.headers.get("X-Opaque-Id").and_then(Value::as_str).map(str::to_string).unwrap_or_default();
        rows.push(vec![
            t.id.to_string(),
            t.action.clone(),
            t.task_id(),
            parent_id.unwrap_or_else(|| "-".into()),
            "transport".into(),
            start.to_string(),
            ts,
            running_ns.to_string(),
            if running_ns >= 1_000_000 { time_fmt(running_ns / 1_000_000) } else { format!("{}micros", (running_ns / 1000).max(1)) },
            NODE_ID.into(),
            "127.0.0.1".into(),
            "9300".into(),
            node.name.clone(),
            elasticsearch_engine::node::ES_VERSION.into(),
            xo,
            t.description_text().unwrap_or_default(),
        ]);
    }
    node.tasks.finish(&child, Ok(json!({})));
    node.tasks.finish(&parent, Ok(json!({})));
    let cols: Vec<Col> = vec![
        ("id", &[], false),
        ("action", &["ac"], true),
        ("task_id", &["ti"], true),
        ("parent_task_id", &["pti"], true),
        ("type", &["ty"], true),
        ("start_time", &["start"], true),
        ("timestamp", &["ts", "hms", "hhmmss"], true),
        ("running_time_ns", &[], false),
        ("running_time", &["time"], true),
        ("node_id", &["ni"], false),
        ("ip", &["i"], true),
        ("port", &["po"], false),
        ("node", &["n"], true),
        ("version", &["v"], false),
        ("x_opaque_id", &["x"], false),
        ("description", &["desc"], detailed),
    ];
    render(req, Table { cols, rows, numeric: vec!["id", "start_time", "running_time_ns", "port"] })
}

fn component_templates(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let t = node.templates.read();
    let mut rows = Vec::new();
    for (name, v) in &t.component {
        if let Some(a) = arg {
            if !a.split(',').any(|p| elasticsearch_engine::mapping::wildcard_match(p, name)) {
                continue;
            }
        }
        let tpl = &v["template"];
        let count = |x: &Value| -> usize {
            match x {
                Value::Object(m) => m.len(),
                _ => 0,
            }
        };
        // `RestCatComponentTemplateAction.countMappingInTemplate`: per top-level
        // mapping entry, its object's entry count (a scalar counts once).
        fn count_sub(v: &Value) -> usize {
            match v {
                Value::Array(a) => a.iter().map(count_sub).sum(),
                Value::Object(o) => o.len(),
                _ => 1,
            }
        }
        let mapping_count: usize = match tpl.get("mappings") {
            Some(Value::Object(m)) => {
                let m = match m.get("_doc") {
                    Some(Value::Object(inner)) if m.len() == 1 => inner,
                    _ => m,
                };
                m.values().map(count_sub).sum()
            }
            _ => 0,
        };
        let settings_count = if tpl["settings"].is_object() { elasticsearch_engine::settings::flatten(&tpl["settings"]).len() } else { 0 };
        let included: Vec<&String> = t.index.iter().filter(|(_, it)| it["composed_of"].as_array().map(|a| a.iter().any(|x| x.as_str() == Some(name.as_str()))).unwrap_or(false)).map(|(n, _)| n).collect();
        rows.push(vec![
            name.clone(),
            v["version"].as_i64().map(|x| x.to_string()).unwrap_or_default(),
            count(&tpl["aliases"]).to_string(),
            mapping_count.to_string(),
            settings_count.to_string(),
            count(&v["_meta"]).to_string(),
            format!("[{}]", included.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")),
        ]);
    }
    render(
        req,
        Table {
            cols: vec![
                ("name", &["n"], true),
                ("version", &["v"], true),
                ("alias_count", &["a"], true),
                ("mapping_count", &["m"], true),
                ("settings_count", &["s"], true),
                ("metadata_count", &["me"], true),
                ("included_in", &["i"], true),
            ],
            rows,
            numeric: vec!["alias_count", "mapping_count", "settings_count", "metadata_count"],
        },
    )
}

fn circuit_breaker(node: &Node, req: &Req, arg: Option<&str>) -> R {
    // `CircuitBreakerService` registration order (ES 9.5.5), live figures.
    const ORDER: &[&str] = &["request", "fielddata", "eql_sequence", "model_inference", "inflight_requests", "parent"];
    let mut list = crate::observability::breaker_rows(node);
    list.sort_by_key(|(n, ..)| ORDER.iter().position(|o| o == n).unwrap_or(ORDER.len()));
    let bytes = req.param("bytes");
    let fmt = |n: i64| if n < 0 { format!("{n}b") } else { bytes_fmt(n as u64, bytes) };
    let rows = list
        .iter()
        .filter(|(n, ..)| arg.map(|a| a.split(',').any(|p| elasticsearch_engine::mapping::wildcard_match(p, n))).unwrap_or(true))
        .map(|(n, limit, estimated, tripped, overhead)| {
            vec![NODE_ID.to_string(), node.name.clone(), n.to_string(), fmt(*limit), limit.to_string(), bytes_fmt(*estimated, bytes), estimated.to_string(), tripped.to_string(), format!("{overhead:?}")]
        })
        .collect();
    render(
        req,
        Table {
            cols: vec![
                ("node_id", &["id"], true),
                ("node_name", &["nn"], false),
                ("breaker", &["br"], true),
                ("limit", &["l"], true),
                ("limit_bytes", &["lb"], false),
                ("estimated", &["e"], true),
                ("estimated_bytes", &["eb"], false),
                ("tripped", &["t"], true),
                ("overhead", &["o"], false),
            ],
            rows,
            numeric: vec!["limit_bytes", "estimated_bytes", "tripped"],
        },
    )
}

/// Transform / ML listings: no such features here, so always empty (a
/// concrete id is a 404 as in ES).
fn ml_stub(req: &Req, id: Option<&str>, kind: &str) -> R {
    if let Some(i) = id {
        if !matches!(i, "_all" | "*") && !i.contains('*') && req.param("allow_no_match") != Some("true") {
            let msg = match kind {
                "transforms" => format!("Transform with id [{i}] could not be found"),
                "anomaly_detectors" => format!("No known job with id '{i}'"),
                "datafeeds" => format!("No datafeed with id [{i}] exists"),
                "trained_models" => format!("Could not find trained model [{i}]"),
                _ => format!("No known data frame analytics with id [{i}]"),
            };
            return Err(EsError::resource_not_found(msg));
        }
    }
    let cols: Vec<Col> = match kind {
        "transforms" => vec![
            ("id", &[], true),
            ("state", &["s"], true),
            ("checkpoint", &["c"], true),
            ("documents_processed", &["docp", "documentsProcessed"], true),
            ("checkpoint_progress", &["cp", "checkpointProgress"], true),
            ("last_search_time", &["lst", "lastSearchTime"], true),
            ("changes_last_detection_time", &["cldt"], true),
            ("create_time", &["ct", "createTime"], false),
            ("version", &["v"], false),
            ("source_index", &["si", "sourceIndex"], false),
            ("dest_index", &["di", "destIndex"], false),
            ("pipeline", &["p"], false),
            ("description", &["d"], false),
            ("transform_type", &["tt"], false),
            ("frequency", &["f"], false),
            ("max_page_search_size", &["mpsz"], false),
            ("docs_per_second", &["dps"], false),
            ("reason", &["r"], false),
        ],
        "anomaly_detectors" => vec![
            ("id", &[], true),
            ("state", &["s"], true),
            ("data.processed_records", &["dpr", "dataProcessedRecords"], true),
            ("model.bytes", &["mb", "modelBytes"], true),
            ("model.memory_status", &["mms", "modelMemoryStatus"], true),
            ("forecasts.total", &["ft", "forecastsTotal"], true),
            ("buckets.count", &["bc", "bucketsCount"], false),
            ("node.name", &["nn", "nodeName"], false),
        ],
        "datafeeds" => vec![
            ("id", &[], true),
            ("state", &["s"], true),
            ("buckets.count", &["bc", "bucketsCount"], true),
            ("search.count", &["sc", "searchCount"], true),
            ("search.time", &["st", "searchTime"], false),
            ("node.name", &["nn", "nodeName"], false),
        ],
        "trained_models" => vec![
            ("id", &[], true),
            ("created_by", &["c", "createdBy"], true),
            ("heap_size", &["hs", "modelHeapSize"], true),
            ("operations", &["o", "modelOperations"], true),
            ("license", &["l"], true),
            ("create_time", &["ct"], true),
            ("version", &["v"], false),
            ("description", &["d"], false),
            ("type", &["t"], false),
        ],
        _ => vec![
            ("id", &[], true),
            ("type", &["t"], true),
            ("create_time", &["ct", "createTime"], true),
            ("state", &["s"], true),
            ("version", &["v"], false),
            ("source_index", &["si", "sourceIndex"], false),
            ("dest_index", &["di", "destIndex"], false),
            ("description", &["d"], false),
            ("model_memory_limit", &["mml"], false),
            ("progress", &["p"], false),
            ("node.name", &["nn", "nodeName"], false),
        ],
    };
    let _ = observability::ROLES;
    let _ = Resp::text;
    render(req, Table { cols, rows: vec![], numeric: vec![] })
}

/// `_cat/transforms[/{id}]` (columns and aliases of ES `RestCatTransformAction`).
fn transforms(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let rows = crate::transform::cat_rows(node, req, arg)?;
    let cols: Vec<Col> = vec![
        ("id", &[], true),
        ("state", &["s"], true),
        ("checkpoint", &["c"], true),
        ("documents_processed", &["docp", "documentsProcessed"], true),
        ("checkpoint_progress", &["cp", "checkpointProgress"], true),
        ("last_search_time", &["lst", "lastSearchTime"], true),
        ("changes_last_detection_time", &["cldt"], true),
        ("create_time", &["ct", "createTime"], false),
        ("version", &["v"], false),
        ("source_index", &["si", "sourceIndex"], false),
        ("dest_index", &["di", "destIndex"], false),
        ("pipeline", &["p"], false),
        ("description", &["d"], false),
        ("transform_type", &["tt"], false),
        ("frequency", &["f"], false),
        ("max_page_search_size", &["mpsz"], false),
        ("docs_per_second", &["dps"], false),
        ("reason", &["r"], false),
        ("search_total", &["st"], false),
        ("search_failure", &["sf"], false),
        ("search_time", &["stime"], false),
        ("index_total", &["it"], false),
        ("index_failure", &["if"], false),
        ("index_time", &["itime"], false),
        ("documents_indexed", &["doci"], false),
        ("delete_time", &["dtime"], false),
        ("documents_deleted", &["docd"], false),
        ("trigger_count", &["tc"], false),
        ("pages_processed", &["pp"], false),
        ("processing_time", &["pt"], false),
        ("checkpoint_duration_time_exp_avg", &["cdtea", "checkpointTimeExpAvg"], false),
        ("indexed_documents_exp_avg", &["idea"], false),
        ("processed_documents_exp_avg", &["pdea"], false),
    ];
    render(
        req,
        Table {
            cols,
            rows,
            numeric: vec![
                "checkpoint", "documents_processed", "search_total", "search_failure", "search_time", "index_total", "index_failure", "index_time", "documents_indexed",
                "delete_time", "documents_deleted", "trigger_count", "pages_processed", "processing_time",
            ],
        },
    )
}

fn ml_node_cells(stats: &Value) -> Vec<String> {
    let n = &stats["node"];
    let g = |k: &str| n[k].as_str().unwrap_or("").to_string();
    vec![g("id"), g("name"), g("ephemeral_id"), g("transport_address")]
}

/// `_cat/ml/anomaly_detectors[/{job_id}]` (ES `RestCatJobsAction`).
fn ml_jobs(node: &Node, req: &Req, id: Option<&str>) -> R {
    let allow = req.bool_param("allow_no_match")?.or(req.bool_param("allow_no_jobs")?).unwrap_or(true);
    let rows = elasticsearch_engine::ml::anomaly::cat_job_rows(node, id, allow)?;
    let bytes = req.param("bytes");
    let rows: Vec<Vec<String>> = rows
        .iter()
        .map(|(st, _cfg)| {
            let dc = &st["data_counts"];
            let ms = &st["model_size_stats"];
            let n = |v: &Value| v.as_i64().map(|x| x.to_string()).unwrap_or_default();
            let opened = st["open_time"].as_str().and_then(|t| t.strip_suffix('s')).and_then(|t| t.parse::<i64>().ok()).map(|secs| time_fmt((secs * 1000) as u64)).unwrap_or_default();
            let log_time = ms["log_time"].as_i64().map(|t| elasticsearch_engine::date::from_millis(t).format("%a %b %d %H:%M:%S UTC %Y").to_string()).unwrap_or_default();
            let mut r = vec![
                st["job_id"].as_str().unwrap_or("").to_string(),
                st["state"].as_str().unwrap_or("").to_string(),
                opened,
                st["assignment_explanation"].as_str().unwrap_or("").to_string(),
                n(&dc["processed_record_count"]),
                n(&dc["processed_field_count"]),
                bytes_fmt(dc["input_bytes"].as_u64().unwrap_or(0), bytes),
                n(&dc["input_record_count"]),
                n(&dc["input_field_count"]),
                n(&dc["invalid_date_count"]),
                n(&dc["missing_field_count"]),
                n(&dc["out_of_order_timestamp_count"]),
                n(&dc["empty_bucket_count"]),
                n(&dc["sparse_bucket_count"]),
                n(&dc["bucket_count"]),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                bytes_fmt(ms["model_bytes"].as_u64().unwrap_or(0), bytes),
                ms["memory_status"].as_str().unwrap_or("").to_string(),
                String::new(),
                ms["model_bytes_memory_limit"].as_u64().map(|x| x.to_string()).unwrap_or_default(),
                n(&ms["total_by_field_count"]),
                n(&ms["total_over_field_count"]),
                n(&ms["total_partition_field_count"]),
                n(&ms["bucket_allocation_failures_count"]),
                // ES prints the memory status in this column
                ms["memory_status"].as_str().unwrap_or("").to_string(),
                String::new(),
                n(&ms["categorized_doc_count"]),
                n(&ms["total_category_count"]),
                n(&ms["frequent_category_count"]),
                n(&ms["rare_category_count"]),
                n(&ms["dead_category_count"]),
                n(&ms["failed_category_count"]),
                log_time,
                String::new(),
                n(&st["forecasts_stats"]["total"]),
            ];
            r.extend(std::iter::repeat_n(String::new(), 12));
            r.extend(ml_node_cells(st));
            r.push(n(&st["timing_stats"]["bucket_count"]));
            r.push("0s".into());
            r.extend(std::iter::repeat_n(String::new(), 3));
            r.push("0.0".into());
            r
        })
        .collect();
    let cols: Vec<Col> = vec![
        ("id", &[], true),
        ("state", &["s"], true),
        ("opened_time", &["ot"], false),
        ("assignment_explanation", &["ae"], false),
        ("data.processed_records", &["dpr", "dataProcessedRecords"], true),
        ("data.processed_fields", &["dpf", "dataProcessedFields"], false),
        ("data.input_bytes", &["dib", "dataInputBytes"], false),
        ("data.input_records", &["dir", "dataInputRecords"], false),
        ("data.input_fields", &["dif", "dataInputFields"], false),
        ("data.invalid_dates", &["did", "dataInvalidDates"], false),
        ("data.missing_fields", &["dmf", "dataMissingFields"], false),
        ("data.out_of_order_timestamps", &["doot", "dataOutOfOrderTimestamps"], false),
        ("data.empty_buckets", &["deb", "dataEmptyBuckets"], false),
        ("data.sparse_buckets", &["dsb", "dataSparseBuckets"], false),
        ("data.buckets", &["db", "dataBuckets"], false),
        ("data.earliest_record", &["der", "dataEarliestRecord"], false),
        ("data.latest_record", &["dlr", "dataLatestRecord"], false),
        ("data.last", &["dl", "dataLast"], false),
        ("data.last_empty_bucket", &["dleb", "dataLastEmptyBucket"], false),
        ("data.last_sparse_bucket", &["dlsb", "dataLastSparseBucket"], false),
        ("model.bytes", &["mb", "modelBytes"], true),
        ("model.memory_status", &["mms", "modelMemoryStatus"], true),
        ("model.bytes_exceeded", &["mbe", "modelBytesExceeded"], false),
        ("model.memory_limit", &["mml", "modelMemoryLimit"], false),
        ("model.by_fields", &["mbf", "modelByFields"], false),
        ("model.over_fields", &["mof", "modelOverFields"], false),
        ("model.partition_fields", &["mpf", "modelPartitionFields"], false),
        ("model.bucket_allocation_failures", &["mbaf", "modelBucketAllocationFailures"], false),
        ("model.output_memory_allocator_bytes", &["momab", "modelOutputMemoryAllocatorBytes"], false),
        ("model.categorization_status", &["mcs", "modelCategorizationStatus"], false),
        ("model.categorized_doc_count", &["mcdc", "modelCategorizedDocCount"], false),
        ("model.total_category_count", &["mtcc", "modelTotalCategoryCount"], false),
        ("model.frequent_category_count", &["mfcc", "modelFrequentCategoryCount"], false),
        ("model.rare_category_count", &["mrcc", "modelRareCategoryCount"], false),
        ("model.dead_category_count", &["mdcc", "modelDeadCategoryCount"], false),
        ("model.failed_category_count", &["mfcc", "modelFailedCategoryCount"], false),
        ("model.log_time", &["mlt", "modelLogTime"], false),
        ("model.timestamp", &["mt", "modelTimestamp"], false),
        ("forecasts.total", &["ft", "forecastsTotal"], true),
        ("forecasts.memory.min", &["fmmin", "forecastsMemoryMin"], false),
        ("forecasts.memory.max", &["fmmax", "forecastsMemoryMax"], false),
        ("forecasts.memory.avg", &["fmavg", "forecastsMemoryAvg"], false),
        ("forecasts.memory.total", &["fmt", "forecastsMemoryTotal"], false),
        ("forecasts.records.min", &["frmin", "forecastsRecordsMin"], false),
        ("forecasts.records.max", &["frmax", "forecastsRecordsMax"], false),
        ("forecasts.records.avg", &["fravg", "forecastsRecordsAvg"], false),
        ("forecasts.records.total", &["frt", "forecastsRecordsTotal"], false),
        ("forecasts.time.min", &["ftmin", "forecastsTimeMin"], false),
        ("forecasts.time.max", &["ftmax", "forecastsTimeMax"], false),
        ("forecasts.time.avg", &["ftavg", "forecastsTimeAvg"], false),
        ("forecasts.time.total", &["ftt", "forecastsTimeTotal"], false),
        ("node.id", &["ni", "nodeId"], false),
        ("node.name", &["nn", "nodeName"], false),
        ("node.ephemeral_id", &["ne", "nodeEphemeralId"], false),
        ("node.address", &["na", "nodeAddress"], false),
        ("buckets.count", &["bc", "bucketsCount"], true),
        ("buckets.time.total", &["btt", "bucketsTimeTotal"], false),
        ("buckets.time.min", &["btmin", "bucketsTimeMin"], false),
        ("buckets.time.max", &["btmax", "bucketsTimeMax"], false),
        ("buckets.time.exp_avg", &["btea", "bucketsTimeExpAvg"], false),
        ("buckets.time.exp_avg_hour", &["bteah", "bucketsTimeExpAvgHour"], false),
    ];
    render(
        req,
        Table {
            cols,
            rows,
            numeric: vec![
                "data.processed_records", "data.processed_fields", "data.input_bytes", "data.input_records", "data.input_fields", "data.invalid_dates", "data.missing_fields",
                "data.out_of_order_timestamps", "data.empty_buckets", "data.sparse_buckets", "data.buckets", "model.bytes", "model.memory_limit", "model.by_fields", "model.over_fields",
                "model.partition_fields", "model.bucket_allocation_failures", "model.categorized_doc_count", "model.total_category_count", "model.frequent_category_count",
                "model.rare_category_count", "model.dead_category_count", "model.failed_category_count", "forecasts.total", "buckets.count", "buckets.time.exp_avg_hour",
            ],
        },
    )
}

/// `_cat/ml/datafeeds[/{datafeed_id}]` (ES `RestCatDatafeedsAction`).
fn ml_datafeeds(node: &Node, req: &Req, id: Option<&str>) -> R {
    let allow = req.bool_param("allow_no_match")?.or(req.bool_param("allow_no_datafeeds")?).unwrap_or(true);
    let rows = elasticsearch_engine::ml::anomaly::cat_datafeed_rows(node, id, allow)?;
    let rows: Vec<Vec<String>> = rows
        .iter()
        .map(|st| {
            let ts = &st["timing_stats"];
            let ms = ts["total_search_time_ms"].as_f64().unwrap_or(0.0);
            let mut r = vec![
                st["datafeed_id"].as_str().unwrap_or("").to_string(),
                st["state"].as_str().unwrap_or("").to_string(),
                st["assignment_explanation"].as_str().unwrap_or("").to_string(),
                ts["bucket_count"].as_i64().unwrap_or(0).to_string(),
                ts["search_count"].as_i64().unwrap_or(0).to_string(),
                if ms <= 0.0 { "0s".to_string() } else { time_fmt(ms as u64) },
                format!("{:.1}", ts["average_search_time_per_bucket_ms"].as_f64().unwrap_or(0.0)),
                format!("{:.1}", ts["exponential_average_search_time_per_hour_ms"].as_f64().unwrap_or(0.0)),
            ];
            r.extend(ml_node_cells(st));
            r
        })
        .collect();
    let cols: Vec<Col> = vec![
        ("id", &[], true),
        ("state", &["s"], true),
        ("assignment_explanation", &["ae"], false),
        ("buckets.count", &["bc", "bucketsCount"], true),
        ("search.count", &["sc", "searchCount"], true),
        ("search.time", &["st", "searchTime"], false),
        ("search.bucket_avg", &["sba", "searchBucketAvg"], false),
        ("search.exp_avg_hour", &["seah", "searchExpAvgHour"], false),
        ("node.id", &["ni", "nodeId"], false),
        ("node.name", &["nn", "nodeName"], false),
        ("node.ephemeral_id", &["ne", "nodeEphemeralId"], false),
        ("node.address", &["na", "nodeAddress"], false),
    ];
    render(req, Table { cols, rows, numeric: vec!["buckets.count", "search.count", "search.bucket_avg", "search.exp_avg_hour"] })
}

/// `_cat/ml/trained_models[/{model_id}]` (ES `RestCatTrainedModelsAction`).
fn ml_trained_models(node: &Node, req: &Req, id: Option<&str>) -> R {
    let allow = req.bool_param("allow_no_match")?.unwrap_or(true);
    let state = node.data.join("_state");
    let res = elasticsearch_engine::inference::models::get(&state, id, allow)?;
    let bytes = req.param("bytes");
    let rows: Vec<Vec<String>> = res["trained_model_configs"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            let s = |k: &str| c[k].as_str().map(str::to_string).unwrap_or_else(|| if c[k].is_null() { String::new() } else { c[k].to_string() });
            vec![
                s("model_id"),
                s("created_by"),
                bytes_fmt(c["model_size_bytes"].as_u64().unwrap_or(0), bytes),
                c["estimated_operations"].as_u64().unwrap_or(0).to_string(),
                s("license_level").to_uppercase(),
                c["create_time"].as_i64().map(elasticsearch_engine::ilm::iso_date).unwrap_or_default(),
                s("version"),
                s("description"),
                s("model_type"),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
                "0".into(),
                if c["tags"].as_array().map(|t| t.iter().any(|x| x == "prepackaged")).unwrap_or(false) { "prepackaged".into() } else { "__none__".into() },
                String::new(),
                String::new(),
                String::new(),
            ]
        })
        .collect();
    let cols: Vec<Col> = vec![
        ("id", &[], true),
        ("created_by", &["c", "createdBy"], false),
        ("heap_size", &["hs", "modelHeapSize"], true),
        ("operations", &["o", "modelOperations"], true),
        ("license", &["l"], false),
        ("create_time", &["ct"], true),
        ("version", &["v"], false),
        ("description", &["d"], false),
        ("type", &["t"], true),
        ("ingest.pipelines", &["ip", "ingestPipelines"], true),
        ("ingest.count", &["ic", "ingestCount"], false),
        ("ingest.time", &["it", "ingestTime"], false),
        ("ingest.current", &["icurr", "ingestCurrent"], false),
        ("ingest.failed", &["if", "ingestFailed"], false),
        ("data_frame.id", &["dfid", "dataFrameAnalytics"], true),
        ("data_frame.create_time", &["dft", "dataFrameAnalyticsTime"], false),
        ("data_frame.source_index", &["dfsi", "dataFrameAnalyticsSrcIndex"], false),
        ("data_frame.analysis", &["dfa", "dataFrameAnalyticsAnalysis"], false),
    ];
    render(req, Table { cols, rows, numeric: vec!["heap_size", "operations", "ingest.pipelines", "ingest.count", "ingest.time", "ingest.current", "ingest.failed"] })
}
