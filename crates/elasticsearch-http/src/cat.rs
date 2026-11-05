//! `_cat` APIs: text tables (`?v`, `?h=`, `?s=`, `?bytes=`, `?format=json`).

use crate::util::bytes_fmt;
use crate::{blocking, Req, Resp, R};
use elasticsearch_engine::node::{Node, ResolveOpts, NODE_ID};
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub(crate) struct Table {
    /// (name, aliases, default-visible)
    pub(crate) cols: Vec<(&'static str, &'static [&'static str], bool)>,
    pub(crate) rows: Vec<Vec<String>>,
    /// Numeric columns sort numerically.
    pub(crate) numeric: Vec<&'static str>,
}

static CATALOG: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| serde_json::from_str(include_str!("data/cat_catalog.json")).unwrap_or(Value::Null));

fn catalog_key(req: &Req) -> String {
    match req.seg(1) {
        "ml" => match req.seg(2) {
            "data_frame" => "ml/data_frame/analytics".into(),
            x => format!("ml/{x}"),
        },
        x => x.to_string(),
    }
}

/// One output column: Elasticsearch's name/aliases/description/alignment, and
/// where its value comes from in the handler's table (if the handler tracks it).
struct EffCol {
    name: String,
    aliases: Vec<String>,
    desc: String,
    default: bool,
    right: bool,
    src: Option<usize>,
}

fn effective_cols(req: &Req, t: &Table) -> Vec<EffCol> {
    let cat = CATALOG.get(catalog_key(req)).filter(|c| c["cols"].as_array().map(|a| !a.is_empty()).unwrap_or(false));
    let Some(cat) = cat else {
        return t
            .cols
            .iter()
            .enumerate()
            .map(|(i, (n, al, d))| EffCol {
                name: n.to_string(),
                aliases: al.iter().map(|x| x.to_string()).collect(),
                desc: n.replace(['.', '_'], " "),
                default: *d,
                right: t.numeric.contains(n),
                src: Some(i),
            })
            .collect();
    };
    let strs = |v: &Value| -> Vec<String> { v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default() };
    let default = strs(&cat["default"]);
    let right = strs(&cat["right"]);
    let known = strs(&cat["align_known"]);
    let mut catalog_cols: Vec<Value> = cat["cols"].as_array().cloned().unwrap_or_default();
    // `RestShardsAction`: REST API v8 requests keep the `sync_id` column.
    if catalog_key(req) == "shards" && req.compat_v8() {
        if let Some(i) = catalog_cols.iter().position(|c| c[0] == "node") {
            catalog_cols.insert(i + 1, serde_json::json!(["sync_id", ["sync_id"], "sync id"]));
        }
    }
    catalog_cols
        .iter()
        .map(|c| {
            let name = c[0].as_str().unwrap_or("").to_string();
            let src = t.cols.iter().position(|(n, _, _)| *n == name);
            let right = if known.contains(&name) { right.contains(&name) } else { t.numeric.contains(&name.as_str()) };
            EffCol {
                aliases: strs(&c[1]),
                desc: c[2].as_str().unwrap_or("").to_string(),
                default: match src {
                    Some(i) => t.cols[i].2,
                    None => default.contains(&name),
                },
                right,
                src,
                name,
            }
        })
        .collect()
}

/// Value of a column the handler does not track: what an idle node reports.
fn zero_value(name: &str, bytes: Option<&str>) -> String {
    if name.contains("size_in_bytes") || name.ends_with("_bytes") {
        "0".into()
    } else if name.ends_with("size") || name.ends_with("memory") || name.ends_with("memory_size") {
        bytes_fmt(0, bytes)
    } else if name.ends_with("time") {
        "0s".into()
    } else {
        "0".into()
    }
}

/// Numeric value of a cell with a unit suffix (`1.2kb`, `15ms`, `66.7%`), for sorting.
fn sort_number(v: &str) -> Option<f64> {
    let v = v.trim();
    let split = v.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+')).unwrap_or(v.len());
    let (num, unit) = v.split_at(split);
    let n: f64 = num.parse().ok()?;
    let f = match unit {
        "" | "%" | "b" => 1.0,
        "kb" => 1024.0,
        "mb" => 1024.0 * 1024.0,
        "gb" => 1024.0 * 1024.0 * 1024.0,
        "tb" => 1024f64.powi(4),
        "pb" => 1024f64.powi(5),
        "nanos" => 1e-6,
        "micros" => 1e-3,
        "ms" => 1.0,
        "s" => 1000.0,
        "m" => 60_000.0,
        "h" => 3_600_000.0,
        "d" => 86_400_000.0,
        _ => return None,
    };
    Some(n * f)
}

fn cmp_cells(a: &str, b: &str) -> std::cmp::Ordering {
    match (sort_number(a), sort_number(b)) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        _ => a.cmp(b),
    }
}

pub(crate) fn render(req: &Req, t: Table) -> R {
    let cols = effective_cols(req, &t);
    if req.flag("help")? {
        let nw = cols.iter().map(|c| c.name.len()).max().unwrap_or(0);
        let aw = cols.iter().map(|c| c.aliases.join(",").len()).max().unwrap_or(0);
        let dw = cols.iter().map(|c| c.desc.len()).max().unwrap_or(0);
        let mut out = String::new();
        for c in &cols {
            out.push_str(&format!("{:<nw$} | {:<aw$} | {:<dw$}\n", c.name, c.aliases.join(","), c.desc));
        }
        return Ok(Resp::text(out));
    }
    let bytes = req.param("bytes");
    let zero: Vec<String> = cols.iter().map(|c| zero_value(&c.name, bytes)).collect();
    // Row cells in output-column order.
    let mut rows: Vec<Vec<String>> = t
        .rows
        .into_iter()
        .map(|r| cols.iter().enumerate().map(|(i, c)| match c.src { Some(s) => r.get(s).cloned().unwrap_or_default(), None => zero[i].clone() }).collect())
        .collect();
    let find = |n: &str| cols.iter().position(|c| c.name == n || c.aliases.iter().any(|a| a == n));
    // Column selection.
    let mut headers: Vec<String> = Vec::new();
    let selected: Vec<usize> = match req.param("h") {
        Some(h) => {
            let mut sel = Vec::new();
            for c in h.split(',').map(str::trim).filter(|c| !c.is_empty()) {
                if c.contains('*') {
                    for (i, col) in cols.iter().enumerate() {
                        if elasticsearch_engine::mapping::wildcard_match(c, &col.name) || col.aliases.iter().any(|a| elasticsearch_engine::mapping::wildcard_match(c, a)) {
                            if !sel.contains(&i) {
                                sel.push(i);
                                headers.push(col.name.clone());
                            }
                        }
                    }
                } else if let Some(i) = find(c) {
                    sel.push(i);
                    headers.push(c.to_string());
                }
            }
            sel
        }
        None => {
            let v: Vec<usize> = cols.iter().enumerate().filter(|(_, c)| c.default).map(|(i, _)| i).collect();
            headers = v.iter().map(|&i| cols[i].name.clone()).collect();
            v
        }
    };
    if let Some(s) = req.param("s") {
        let keys: Vec<(usize, bool)> = s
            .split(',')
            .filter(|k| !k.is_empty())
            .filter_map(|k| {
                let (name, desc) = match k.split_once(':') {
                    Some((n, d)) => (n, d == "desc"),
                    None => (k, false),
                };
                find(name).map(|i| (i, desc))
            })
            .collect();
        rows.sort_by(|a, b| {
            for (i, desc) in &keys {
                let o = cmp_cells(&a[*i], &b[*i]);
                let o = if *desc { o.reverse() } else { o };
                if o != std::cmp::Ordering::Equal {
                    return o;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    let accept = req.headers.get("accept").and_then(|v| v.to_str().ok()).unwrap_or("");
    let structured = match req.param("format") {
        Some(f) => matches!(f, "json" | "yaml" | "cbor" | "smile"),
        None => ["json", "yaml", "cbor", "smile"].iter().any(|m| accept.contains(&format!("application/{m}")) || accept.contains(&format!("elasticsearch+{m}"))),
    };
    if structured {
        let arr: Vec<Value> = rows
            .iter()
            .map(|r| {
                let mut m = Map::new();
                for &i in &selected {
                    m.insert(cols[i].name.clone(), if r[i].is_empty() && cols[i].name != "filter" { Value::Null } else { json!(r[i]) });
                }
                Value::Object(m)
            })
            .collect();
        return Ok(Resp::ok(Value::Array(arr)));
    }
    let verbose = req.flag("v")?;
    let mut table: Vec<Vec<String>> = Vec::new();
    if verbose {
        table.push(headers.clone());
    }
    for r in &rows {
        table.push(selected.iter().map(|&i| r[i].clone()).collect());
    }
    let widths: Vec<usize> = (0..selected.len()).map(|c| table.iter().map(|r| r[c].chars().count()).max().unwrap_or(0)).collect();
    // Numeric and byte-size columns are right-aligned, as in ES.
    let right: Vec<bool> = selected.iter().map(|&i| cols[i].right).collect();
    let mut out = String::new();
    for (ri, r) in table.iter().enumerate() {
        // ES pads every column but the last left-aligned one; cells are separated by one space.
        let last = r.len().saturating_sub(1);
        let line: Vec<String> = r.iter().enumerate().map(|(c, v)| {
            let num = right[c] && !(verbose && ri == 0);
            if num { format!("{:>w$}", v, w = widths[c]) } else if c == last { v.clone() } else { format!("{:<w$}", v, w = widths[c]) }
        }).collect();
        out.push_str(&line.join(" "));
        out.push('\n');
    }
    Ok(Resp::text(out))
}

const CAT_ROOT: &str = "=^.^=\n/_cat/allocation\n/_cat/shards\n/_cat/shards/{index}\n/_cat/master\n/_cat/nodes\n/_cat/tasks\n/_cat/indices\n/_cat/indices/{index}\n/_cat/segments\n/_cat/segments/{index}\n/_cat/count\n/_cat/count/{index}\n/_cat/recovery\n/_cat/recovery/{index}\n/_cat/health\n/_cat/pending_tasks\n/_cat/aliases\n/_cat/aliases/{alias}\n/_cat/thread_pool\n/_cat/thread_pool/{thread_pools}\n/_cat/plugins\n/_cat/fielddata\n/_cat/fielddata/{fields}\n/_cat/nodeattrs\n/_cat/repositories\n/_cat/snapshots/{repository}\n/_cat/templates\n/_cat/component_templates\n/_cat/transforms\n/_cat/transforms/{transform_id}\n";

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    blocking(move || {
        let what = req.seg(1).to_string();
        let arg = req.segs.get(2).cloned();
        match what.as_str() {
            "" => Ok(Resp::text(CAT_ROOT.into())),
            "indices" => indices(&node, &req, arg.as_deref()),
            "health" => health(&node, &req),
            "count" => count(&node, &req, arg.as_deref()),
            "aliases" => aliases(&node, &req, arg.as_deref()),
            "shards" => shards(&node, &req, arg.as_deref()),
            "nodes" => nodes(&node, &req),
            "master" => render(&req, Table { cols: vec![("id", &[], true), ("host", &["h"], true), ("ip", &[], true), ("node", &["n"], true)], rows: vec![vec![NODE_ID.into(), "127.0.0.1".into(), "127.0.0.1".into(), node.name.clone()]], numeric: vec![] }),
            "templates" => templates(&node, &req, arg.as_deref()),
            "allocation" => allocation(&node, &req, arg.as_deref()),
            "thread_pool" => thread_pool(&node, &req, arg.as_deref()),
            "plugins" => {
                // No bootstrap plugins either way; ES still parses the flag.
                req.flag("include_bootstrap")?;
                render(&req, Table {
                    cols: vec![("id", &[], false), ("name", &["n"], true), ("component", &["c"], true), ("version", &["v"], true), ("description", &["d"], false)],
                    rows: vec![],
                    numeric: vec![],
                })
            }
            "nodeattrs" => render(&req, Table {
                cols: vec![("node", &["name"], true), ("id", &["nodeId"], false), ("pid", &["p"], false), ("host", &["h"], true), ("ip", &["i"], true), ("port", &["po"], false), ("attr", &["attr.name"], true), ("value", &["attr.value"], true)],
                rows: crate::observability::node_attributes()
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| {
                                vec![
                                    node.name.clone(),
                                    NODE_ID.into(),
                                    std::process::id().to_string(),
                                    "127.0.0.1".into(),
                                    "127.0.0.1".into(),
                                    "9300".into(),
                                    k.clone(),
                                    v.as_str().unwrap_or("").to_string(),
                                ]
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                numeric: vec![],
            }),
            "segments" => segments(&node, &req, arg.as_deref()),
            "fielddata" | "pending_tasks" | "recovery" | "repositories" | "snapshots" | "tasks" | "component_templates" | "circuit_breaker" | "transforms" | "ml" => crate::cat_extra::handle(&node, &req, &what, arg.as_deref()),
            _ => crate::not_found_handler(&req),
        }
    })
    .await
}

fn targets(node: &Node, req: &Req, arg: Option<&str>) -> Result<Vec<elasticsearch_engine::search::Target>, elasticsearch_engine::error::EsError> {
    targets_with(node, req, arg, false)
}

/// `hidden`: hidden indices are listed when no `expand_wildcards` is given.
fn targets_with(node: &Node, req: &Req, arg: Option<&str>, hidden: bool) -> Result<Vec<elasticsearch_engine::search::Target>, elasticsearch_engine::error::EsError> {
    let mut o = ResolveOpts { expand_closed: true, expand_hidden: hidden, ..Default::default() };
    if let Some(ew) = req.param("expand_wildcards") {
        o.expand_open = false;
        o.expand_closed = false;
        for w in ew.split(',') {
            match w.trim() {
                "open" => o.expand_open = true,
                "closed" => o.expand_closed = true,
                "hidden" => o.expand_hidden = true,
                "all" => {
                    o.expand_open = true;
                    o.expand_closed = true;
                    o.expand_hidden = true;
                }
                "none" | "" => {}
                other => return Err(elasticsearch_engine::error::EsError::illegal_argument(format!("No enum constant org.elasticsearch.action.support.IndicesOptions.WildcardStates.{}", other.to_uppercase()))),
            }
        }
    }
    node.resolve(arg.unwrap_or("_all"), o)
}

fn indices(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let bytes = req.param("bytes");
    let mut rows = Vec::new();
    if let Some(h) = req.param("health") {
        if !matches!(h, "green" | "yellow" | "red" | "unknown" | "unavailable") {
            return Err(elasticsearch_engine::error::EsError::illegal_argument(format!("unknown health status [{h}]")));
        }
    }
    for t in targets(node, req, arg)? {
        let replicas = t.svc.setting_usize("index.number_of_replicas", 1);
        // Stats come from the shard copies: none when the primaries failed to allocate.
        let open = t.svc.is_open() && t.svc.active_shards() > 0;
        let (docs, del) = if open { t.svc.doc_count() } else { (0, 0) };
        let size = t.svc.store_size();
        // closed indices are replicated since 7.2: same health rule as open ones
        let health = if !t.svc.failed_shards().is_empty() {
            "red"
        } else if replicas > 0 {
            "yellow"
        } else {
            "green"
        };
        if let Some(h) = req.param("health") {
            if h != health {
                continue;
            }
        }
        rows.push(vec![
            health.into(),
            if open { "open".into() } else { "close".into() },
            t.svc.name.clone(),
            t.svc.uuid.clone(),
            t.svc.number_of_shards().to_string(),
            replicas.to_string(),
            if open { docs.to_string() } else { String::new() },
            if open { del.to_string() } else { String::new() },
            if open { bytes_fmt(size, bytes) } else { String::new() },
            if open { bytes_fmt(size, bytes) } else { String::new() },
            if open { bytes_fmt(size, bytes) } else { String::new() },
            t.svc.creation_date.to_string(),
            elasticsearch_engine::date::from_millis(t.svc.creation_date as i64).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
        ]);
    }
    render(
        req,
        Table {
            cols: vec![
                ("health", &["h"], true),
                ("status", &["s"], true),
                ("index", &["i", "idx"], true),
                ("uuid", &["id", "uuid"], true),
                ("pri", &["p", "shards.primary", "shardsPrimary"], true),
                ("rep", &["r", "shards.replica", "shardsReplica"], true),
                ("docs.count", &["dc", "docsCount"], true),
                ("docs.deleted", &["dd", "docsDeleted"], true),
                ("store.size", &["ss", "storeSize"], true),
                ("pri.store.size", &[], true),
                ("dataset.size", &[], true),
                ("creation.date", &["cd"], false),
                ("creation.date.string", &["cds"], false),
            ],
            rows,
            numeric: vec!["pri", "rep", "docs.count", "docs.deleted", "creation.date"],
        },
    )
}

fn now_parts() -> (String, String) {
    let ms = elasticsearch_engine::date::now_millis();
    let epoch = (ms / 1000).to_string();
    let dt = elasticsearch_engine::date::from_millis(ms);
    (epoch, dt.format("%H:%M:%S").to_string())
}

fn health(node: &Node, req: &Req) -> R {
    let all = node.all_indices();
    let failed: usize = all.iter().map(|s| s.failed_shards().len()).sum();
    let unassigned: usize = all.iter().map(|s| s.setting_usize("index.number_of_replicas", 1)).sum::<usize>() + failed;
    let n = all.len().saturating_sub(failed);
    let status = if failed > 0 {
        "red"
    } else if unassigned > 0 {
        "yellow"
    } else {
        "green"
    };
    let pct = if n + unassigned == 0 { 100.0 } else { n as f64 * 100.0 / (n + unassigned) as f64 };
    let (epoch, ts) = now_parts();
    let ts_on = req.bool_param("ts")?.unwrap_or(true);
    let mut cols: Vec<(&'static str, &'static [&'static str], bool)> = vec![("epoch", &["t", "time"], ts_on), ("timestamp", &["ts", "hms"], ts_on)];
    cols.extend([
        ("cluster", &["cl"][..], true),
        ("status", &["st"][..], true),
        ("node.total", &["nt", "nodeTotal"][..], true),
        ("node.data", &["nd", "nodeData"][..], true),
        ("shards", &["t", "sh", "shards.total"][..], true),
        ("pri", &["p", "shards.primary"][..], true),
        ("relo", &["r", "shards.relocating"][..], true),
        ("init", &["i", "shards.initializing"][..], true),
        ("unassign", &["u", "shards.unassigned"][..], true),
        ("unassign.pri", &["up"][..], true),
        ("pending_tasks", &["pt", "pendingTasks"][..], true),
        ("max_task_wait_time", &["mtwt"][..], true),
        ("active_shards_percent", &["asp"][..], true),
    ]);
    render(
        req,
        Table {
            cols,
            rows: vec![vec![
                epoch,
                ts,
                node.cluster_name.clone(),
                status.into(),
                "1".into(),
                "1".into(),
                n.to_string(),
                n.to_string(),
                "0".into(),
                "0".into(),
                unassigned.to_string(),
                "0".into(),
                "0".into(),
                "-".into(),
                format!("{:.1}%", pct),
            ]],
            numeric: vec!["epoch", "node.total", "node.data", "shards", "pri", "relo", "init", "unassign", "unassign.pri", "pending_tasks"],
        },
    )
}

fn count(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let mut t = node.resolve(arg.unwrap_or("_all"), ResolveOpts::default())?;
    if let Some(a) = req.access.as_deref() {
        t = a.apply(t)?;
    }
    let n = elasticsearch_engine::search::count(&t, None, None)?;
    let (epoch, ts) = now_parts();
    render(req, Table { cols: vec![("epoch", &["t", "time"], true), ("timestamp", &["ts", "hms"], true), ("count", &["dc", "docs.count"], true)], rows: vec![vec![epoch, ts, n.to_string()]], numeric: vec![] })
}

fn aliases(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let mut rows = Vec::new();
    // Default for cat aliases is `all`; without `hidden`/`all` hidden indices and hidden aliases are skipped.
    let show_hidden = req.param("expand_wildcards").map(|w| w.split(',').any(|x| x == "all" || x == "hidden")).unwrap_or(true);
    let catalog = req.extensions.get::<crate::AliasCatalog>().map(|c| c.0.clone()).unwrap_or_else(|| {
        node.all_indices().into_iter().map(|svc| {
            let hidden = svc.name.starts_with('.') || svc.setting("index.hidden").is_some_and(|v| v == "true");
            (svc.name.clone(), hidden, svc.aliases())
        }).collect()
    });
    for (index, index_hidden, aliases) in catalog {
        if index_hidden && !show_hidden {
            continue;
        }
        for (a, def) in aliases {
            if !show_hidden && def.get("is_hidden").map(|v| v == &json!(true) || v == &json!("true")).unwrap_or(false) {
                continue;
            }
            if let Some(p) = arg {
                if !p.split(',').any(|x| elasticsearch_engine::mapping::wildcard_match(x, &a)) {
                    continue;
                }
            }
            let s = |k: &str| def.get(k).map(elasticsearch_engine::query::dsl::scalar_string).unwrap_or_else(|| "-".into());
            rows.push(vec![
                a.clone(),
                index.clone(),
                if def.get("filter").is_some() { "*".into() } else { "-".into() },
                s("index_routing"),
                s("search_routing"),
                s("is_write_index"),
            ]);
        }
    }
    rows.sort_by(|a, b| (&a[1], &a[0]).cmp(&(&b[1], &b[0])));
    render(
        req,
        Table {
            cols: vec![("alias", &["a"], true), ("index", &["i", "idx"], true), ("filter", &["f", "fi"], true), ("routing.index", &["ri", "routingIndex"], true), ("routing.search", &["rs", "routingSearch"], true), ("is_write_index", &["w", "isWriteIndex"], true)],
            rows,
            numeric: vec![],
        },
    )
}

/// `_cat/shards` columns: (name, aliases, shown by default).
const SHARD_COLS: &[(&str, &[&str], bool)] = &[
    ("index", &["i", "idx"], true),
    ("shard", &["s", "sh"], true),
    ("prirep", &["p", "pr", "primaryOrReplica"], true),
    ("state", &["st"], true),
    ("docs", &["d", "dc"], true),
    ("store", &["sto"], true),
    ("dataset", &[], true),
    ("ip", &[], true),
    ("id", &[], false),
    ("node", &["n"], true),
    ("sync_id", &[], false),
    ("unassigned.reason", &["ur"], false),
    ("unassigned.at", &["ua"], false),
    ("unassigned.for", &["uf"], false),
    ("unassigned.details", &["ud"], false),
    ("recoverysource.type", &["rs"], false),
    ("completion.size", &["cs", "completionSize"], false),
    ("fielddata.memory_size", &["fm", "fielddataMemory"], false),
    ("fielddata.evictions", &["fe", "fielddataEvictions"], false),
    ("query_cache.memory_size", &["qcm", "queryCacheMemory"], false),
    ("query_cache.evictions", &["qce", "queryCacheEvictions"], false),
    ("flush.total", &["ft", "flushTotal"], false),
    ("flush.total_time", &["ftt", "flushTotalTime"], false),
    ("get.current", &["gc", "getCurrent"], false),
    ("get.time", &["gti", "getTime"], false),
    ("get.total", &["gto", "getTotal"], false),
    ("get.exists_time", &["geti", "getExistsTime"], false),
    ("get.exists_total", &["geto", "getExistsTotal"], false),
    ("get.missing_time", &["gmti", "getMissingTime"], false),
    ("get.missing_total", &["gmto", "getMissingTotal"], false),
    ("indexing.delete_current", &["idc", "indexingDeleteCurrent"], false),
    ("indexing.delete_time", &["idti", "indexingDeleteTime"], false),
    ("indexing.delete_total", &["idto", "indexingDeleteTotal"], false),
    ("indexing.index_current", &["iic", "indexingIndexCurrent"], false),
    ("indexing.index_time", &["iiti", "indexingIndexTime"], false),
    ("indexing.index_total", &["iito", "indexingIndexTotal"], false),
    ("indexing.index_failed", &["iif", "indexingIndexFailed"], false),
    ("indexing.index_failed_due_to_version_conflict", &["iifvc", "indexingIndexFailedDueToVersionConflict"], false),
    ("merges.current", &["mc", "mergesCurrent"], false),
    ("merges.current_docs", &["mcd", "mergesCurrentDocs"], false),
    ("merges.current_size", &["mcs", "mergesCurrentSize"], false),
    ("merges.total", &["mt", "mergesTotal"], false),
    ("merges.total_docs", &["mtd", "mergesTotalDocs"], false),
    ("merges.total_size", &["mts", "mergesTotalSize"], false),
    ("merges.total_time", &["mtt", "mergesTotalTime"], false),
    ("refresh.total", &["rto", "refreshTotal"], false),
    ("refresh.time", &["rti", "refreshTime"], false),
    ("refresh.external_total", &["rto", "refreshTotal"], false),
    ("refresh.external_time", &["rti", "refreshTime"], false),
    ("refresh.listeners", &["rli", "refreshListeners"], false),
    ("search.fetch_current", &["sfc", "searchFetchCurrent"], false),
    ("search.fetch_time", &["sfti", "searchFetchTime"], false),
    ("search.fetch_total", &["sfto", "searchFetchTotal"], false),
    ("search.open_contexts", &["so", "searchOpenContexts"], false),
    ("search.query_current", &["sqc", "searchQueryCurrent"], false),
    ("search.query_time", &["sqti", "searchQueryTime"], false),
    ("search.query_total", &["sqto", "searchQueryTotal"], false),
    ("search.scroll_current", &["scc", "searchScrollCurrent"], false),
    ("search.scroll_time", &["scti", "searchScrollTime"], false),
    ("search.scroll_total", &["scto", "searchScrollTotal"], false),
    ("segments.count", &["sc", "segmentsCount"], false),
    ("segments.memory", &["sm", "segmentsMemory"], false),
    ("segments.index_writer_memory", &["siwm", "segmentsIndexWriterMemory"], false),
    ("segments.version_map_memory", &["svmm", "segmentsVersionMapMemory"], false),
    ("segments.fixed_bitset_memory", &["sfbm", "fixedBitsetMemory"], false),
    ("seq_no.max", &["sqm", "maxSeqNo"], false),
    ("seq_no.local_checkpoint", &["sql", "localCheckpoint"], false),
    ("seq_no.global_checkpoint", &["sqg", "globalCheckpoint"], false),
    ("warmer.current", &["wc", "warmerCurrent"], false),
    ("warmer.total", &["wto", "warmerTotal"], false),
    ("warmer.total_time", &["wtt", "warmerTotalTime"], false),
    ("path.data", &["pd", "dataPath"], false),
    ("path.state", &["ps", "statsPath"], false),
    ("bulk.total_operations", &["bto", "bulkTotalOperations"], false),
    ("bulk.total_time", &["btti", "bulkTotalTime"], false),
    ("bulk.total_size_in_bytes", &["btsi", "bulkTotalSizeInBytes"], false),
    ("bulk.avg_time", &["bati", "bulkAvgTime"], false),
    ("bulk.avg_size_in_bytes", &["basi", "bulkAvgSizeInBytes"], false),
    ("dense_vector.value_count", &["dvc", "denseVectorCount"], false),
    ("sparse_vector.value_count", &["svc", "sparseVectorCount"], false),
];

fn shards(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let bytes = req.param("bytes");
    let mut rows = Vec::new();
    // Statistic columns the engine does not track report zero.
    let zero = |name: &str| -> String {
        if name.ends_with("size") || name.ends_with("memory") || name.ends_with("memory_size") {
            bytes_fmt(0, bytes)
        } else if name.ends_with("time") {
            "0s".into()
        } else {
            "0".into()
        }
    };
    // `RestShardsAction`: `sync_id` is a REST API v8 column only.
    let cols: Vec<(&str, &[&str], bool)> = SHARD_COLS.iter().copied().filter(|(n, _, _)| *n != "sync_id" || req.compat_v8()).collect();
    let row = |vals: Vec<(&str, String)>| -> Vec<String> {
        cols
            .iter()
            .map(|(n, _, _)| match vals.iter().find(|(k, _)| k == n) {
                Some((_, v)) => v.clone(),
                None if matches!(*n, "sync_id" | "unassigned.reason" | "unassigned.at" | "unassigned.for" | "unassigned.details" | "recoverysource.type" | "path.data" | "path.state") => String::new(),
                None => zero(n),
            })
            .collect()
    };
    for t in targets_with(node, req, arg, arg.is_none())? {
        // Every primary is a local copy (unassigned when its allocation
        // failed); replicas are unassigned.
        for sh in 0..t.svc.number_of_shards() {
            if let Some(details) = t.svc.allocation_failure_details(sh) {
                rows.push(row(vec![
                    ("index", t.svc.name.clone()),
                    ("shard", sh.to_string()),
                    ("prirep", "p".into()),
                    ("state", "UNASSIGNED".into()),
                    ("docs", String::new()),
                    ("store", String::new()),
                    ("dataset", String::new()),
                    ("ip", String::new()),
                    ("id", String::new()),
                    ("node", String::new()),
                    ("unassigned.reason", "ALLOCATION_FAILED".into()),
                    ("unassigned.details", details),
                ]));
            } else {
                let (d, z) = t.svc.shard(sh).map(|c| (c.doc_count().0, c.shard.store_size())).unwrap_or((0, 0));
                rows.push(row(vec![
                    ("index", t.svc.name.clone()),
                    ("shard", sh.to_string()),
                    ("prirep", "p".into()),
                    ("state", "STARTED".into()),
                    ("docs", d.to_string()),
                    ("store", bytes_fmt(z, bytes)),
                    ("dataset", bytes_fmt(z, bytes)),
                    ("ip", "127.0.0.1".into()),
                    ("id", NODE_ID.into()),
                    ("node", node.name.clone()),
                ]));
            }
            for _ in 0..t.svc.setting_usize("index.number_of_replicas", 1) {
                rows.push(row(vec![
                    ("index", t.svc.name.clone()),
                    ("shard", sh.to_string()),
                    ("prirep", "r".into()),
                    ("state", "UNASSIGNED".into()),
                    ("docs", String::new()),
                    ("store", String::new()),
                    ("dataset", String::new()),
                    ("ip", String::new()),
                    ("id", String::new()),
                    ("node", String::new()),
                    ("unassigned.reason", "INDEX_CREATED".into()),
                ]));
            }
        }
    }
    render(req, Table { cols, rows, numeric: vec!["shard", "docs"] })
}

pub(crate) fn segments(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let bytes = req.param("bytes");
    let mut rows = Vec::new();
    // `allow_closed=true` lists closed indices too (wildcards then follow `expand_wildcards`).
    let opts = if req.flag("allow_closed")? {
        ResolveOpts { ignore_unavailable: req.flag("ignore_unavailable")?, ..ResolveOpts::from_params(&req.params) }
    } else {
        ResolveOpts { ignore_unavailable: req.flag("ignore_unavailable")?, ..Default::default() }
    };
    for t in node.resolve(arg.unwrap_or("_all"), opts)? {
        for sh in t.svc.local_shards() {
            // The published reader view, published memtables included (as
            // the segments they seal into): no seal, no shard lock.
            // Searchable units and the last commit's segments (a merged-away
            // one stays `committed`, not `searchable`, until the next commit,
            // as in ES). `size.memory` is 0 in ES since 8.0.
            let view = sh.shard.reader_view();
            for seg in view.segment_stats() {
                rows.push(vec![
                    t.svc.name.clone(), sh.id.to_string(), "p".into(), "127.0.0.1".into(), NODE_ID.into(),
                    format!("_{}", seg.id), seg.id.to_string(), seg.docs.saturating_sub(seg.deleted).to_string(), seg.deleted.to_string(),
                    bytes_fmt(seg.bytes, bytes), "0".into(), seg.committed.to_string(), seg.search.to_string(), elasticsearch_engine::version::LUCENE_VERSION.into(), "false".into(),
                ]);
            }
        }
    }
    render(
        req,
        Table {
            cols: vec![
                ("index", &["i", "idx"], true),
                ("shard", &["s", "sh"], true),
                ("prirep", &["p", "pr", "primaryOrReplica"], true),
                ("ip", &[], true),
                ("id", &[], false),
                ("segment", &["seg"], true),
                ("generation", &["g", "gen"], true),
                ("docs.count", &["dc", "docsCount"], true),
                ("docs.deleted", &["dd", "docsDeleted"], true),
                ("size", &["si"], true),
                ("size.memory", &["sm", "sizeMemory"], true),
                ("committed", &["ic", "isCommitted"], true),
                ("searchable", &["is", "isSearchable"], true),
                ("version", &["v"], true),
                ("compound", &["ico", "isCompound"], true),
            ],
            rows,
            numeric: vec!["shard", "generation", "docs.count", "docs.deleted", "size.memory"],
        },
    )
}

fn nodes(node: &Node, req: &Req) -> R {
    let (mem_total, mem_free) = crate::util::sys_memory().unwrap_or((0, 0));
    let mem_used = mem_total.saturating_sub(mem_free);
    let ram_pct = if mem_total == 0 { 0 } else { mem_used * 100 / mem_total };
    let (_, disk_total, disk_avail) = crate::util::disk_usage(&node.data).unwrap_or((String::new(), 0, 0));
    render(
        req,
        Table {
            cols: vec![
                ("ip", &["i"], true),
                ("heap.current", &["hc", "heapCurrent"], false),
                ("heap.percent", &["hp", "heapPercent"], true),
                ("heap.max", &["hm", "heapMax"], false),
                ("ram.current", &["rc", "ramCurrent"], false),
                ("file_desc.current", &["fdc", "fileDescriptorCurrent"], false),
                ("file_desc.percent", &["fdp", "fileDescriptorPercent"], false),
                ("file_desc.max", &["fdm", "fileDescriptorMax"], false),
                ("ram.percent", &["rp", "ramPercent"], true),
                ("cpu", &[], true),
                ("load_1m", &["l"], true),
                ("load_5m", &[], true),
                ("load_15m", &[], true),
                ("node.role", &["r", "role", "nodeRole"], true),
                ("master", &["m"], true),
                ("name", &["n"], true),
                ("id", &["nodeId"], false),
                ("pid", &["p"], false),
                ("port", &["po"], false),
                ("http_address", &["http"], false),
                ("version", &["v"], false),
                ("disk.total", &["dt", "diskTotal"], false),
                ("disk.used", &["du", "diskUsed"], false),
                ("disk.avail", &["d", "da", "disk", "diskAvail"], false),
                ("disk.used_percent", &["dup", "diskUsedPercent"], false),
                ("available_processors", &["ap"], false),
            ],
            rows: vec![vec![
                "127.0.0.1".into(), "0b".into(), "0".into(), "0b".into(), bytes_fmt(mem_used, req.param("bytes")), "0".into(), "0".into(), "0".into(), ram_pct.to_string(), "0".into(), "0.00".into(), "0.00".into(), "0.00".into(),
                "cdfhilmrstw".into(), "*".into(), node.name.clone(),
                if req.flag("full_id")? { NODE_ID.into() } else { NODE_ID[..4].into() },
                std::process::id().to_string(), "9300".into(), "127.0.0.1:9200".into(), elasticsearch_engine::node::ES_VERSION.into(),
                bytes_fmt(disk_total, req.param("bytes")), bytes_fmt(disk_total - disk_avail, req.param("bytes")), bytes_fmt(disk_avail, req.param("bytes")),
                format!("{:.2}", if disk_total == 0 { 0.0 } else { (disk_total - disk_avail) as f64 * 100.0 / disk_total as f64 }),
                crate::sysinfo::processors().to_string(),
            ]],
            numeric: vec!["heap.percent", "ram.percent", "cpu", "file_desc.current", "file_desc.percent", "file_desc.max", "pid", "port", "available_processors"],
        },
    )
}

fn templates(node: &Node, req: &Req, arg: Option<&str>) -> R {
    if let Some(a) = arg {
        if a.contains(',') {
            return Err(elasticsearch_engine::error::EsError::illegal_argument(format!("invalid name [{a}]: must not contain ','")));
        }
    }
    let wanted = |n: &str| arg.map(|a| elasticsearch_engine::mapping::wildcard_match(a, n)).unwrap_or(true);
    let t = node.templates.read();
    let mut rows = Vec::new();
    for (n, v) in t.index.iter().filter(|(n, _)| wanted(n)) {
        rows.push(vec![n.clone(), format!("[{}]", elasticsearch_engine::node::pattern_list(&v["index_patterns"]).join(", ")), v["priority"].as_i64().unwrap_or(0).to_string(), v["version"].as_i64().map(|x| x.to_string()).unwrap_or_default(), format!("[{}]", v["composed_of"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default())]);
    }
    for (n, v) in t.legacy.iter().filter(|(n, _)| wanted(n)) {
        rows.push(vec![n.clone(), format!("[{}]", elasticsearch_engine::node::pattern_list(&v["index_patterns"]).join(", ")), v["order"].as_i64().unwrap_or(0).to_string(), v["version"].as_i64().map(|x| x.to_string()).unwrap_or_default(), String::new()]);
    }
    render(req, Table { cols: vec![("name", &["n"], true), ("index_patterns", &["t"], true), ("order", &["o", "p"], true), ("version", &["v"], true), ("composed_of", &["c"], true)], rows, numeric: vec!["order"] })
}

fn allocation(node: &Node, req: &Req, arg: Option<&str>) -> R {
    let bytes = req.param("bytes");
    let mut rows = Vec::new();
    let node_ok = arg.map(|a| a.split(',').any(|x| matches!(x, "_all" | "_local" | "_master" | "*") || x == NODE_ID || x == node.name || elasticsearch_engine::mapping::wildcard_match(x, &node.name))).unwrap_or(true);
    let all = node.all_indices();
    let unassigned: usize = all.iter().map(|s| s.number_of_shards() as usize * s.setting_usize("index.number_of_replicas", 1)).sum();
    if node_ok {
        let store: u64 = all.iter().map(|s| s.store_size()).sum();
        let (_, total, avail) = crate::util::disk_usage(&node.data).unwrap_or((String::new(), 0, 0));
        let used = total.saturating_sub(avail);
        rows.push(vec![
            all.iter().map(|s| s.number_of_shards() as usize).sum::<usize>().to_string(),
            "0".into(),
            "0.0".into(),
            bytes_fmt(store, bytes),
            bytes_fmt(store, bytes),
            bytes_fmt(used, bytes),
            bytes_fmt(avail, bytes),
            bytes_fmt(total, bytes),
            if total == 0 { "0".into() } else { (used * 100 / total).to_string() },
            "127.0.0.1".into(),
            "127.0.0.1".into(),
            node.name.clone(),
            "cdfhilmrstw".into(),
        ]);
    }
    if unassigned > 0 && arg.is_none() {
        let mut r = vec![unassigned.to_string()];
        r.extend(std::iter::repeat_n(String::new(), 10));
        r.push("UNASSIGNED".into());
        r.push(String::new());
        rows.push(r);
    }
    render(
        req,
        Table {
            cols: vec![
                ("shards", &["s"], true),
                ("shards.undesired", &[], true),
                ("write_load.forecast", &["wlf", "writeLoadForecast"], true),
                ("disk.indices.forecast", &["dif", "diskIndicesForecast"], true),
                ("disk.indices", &["di", "diskIndices"], true),
                ("disk.used", &["du", "diskUsed"], true),
                ("disk.avail", &["da", "diskAvail"], true),
                ("disk.total", &["dt", "diskTotal"], true),
                ("disk.percent", &["dp", "diskPercent"], true),
                ("host", &["h"], true),
                ("ip", &[], true),
                ("node", &["n"], true),
                ("node.role", &["r", "role", "nodeRole"], true),
            ],
            rows,
            numeric: vec!["shards", "shards.undesired", "write_load.forecast", "disk.percent"],
        },
    )
}

fn thread_pool(node: &Node, req: &Req, arg: Option<&str>) -> R {
    const POOLS: &[(&str, bool)] = &[
        ("analyze", true), ("auto_complete", true), ("fetch_shard_started", false), ("fetch_shard_store", false), ("flush", false), ("force_merge", true),
        ("generic", false), ("get", true), ("management", false), ("refresh", false), ("search", true), ("search_coordination", true),
        ("snapshot", false), ("snapshot_meta", false), ("system_critical_read", true), ("system_critical_write", true),
        ("system_read", true), ("system_write", true), ("warmer", false), ("write", true),
    ];
    let cpus = crate::sysinfo::processors();
    let wanted = |n: &str| arg.map(|a| a.split(',').any(|p| elasticsearch_engine::mapping::wildcard_match(p.trim(), n))).unwrap_or(true);
    let rows: Vec<Vec<String>> = POOLS
        .iter()
        .filter(|(n, _)| wanted(n))
        .map(|(n, fixed)| {
            // The executors engine work really runs on report their own
            // figures (size, active, queue, rejected, completed).
            let live = crate::pools::Pool::ALL.iter().find(|q| q.name() == *n).map(|q| crate::pools::executor(*q).stats());
            let (size, qsize, core, max, ka) = match &live {
                Some(st) => (st.threads.to_string(), st.queue_size.map_or("-1".to_string(), |q| q.to_string()), String::new(), String::new(), String::new()),
                None if *fixed => (cpus.to_string(), "1000".to_string(), String::new(), String::new(), String::new()),
                None => (String::new(), "-1".to_string(), "1".to_string(), cpus.max(1).to_string(), "5m".to_string()),
            };
            let (active, pool_size, queue, rejected, largest, completed) = match &live {
                Some(st) => (st.active.to_string(), st.threads.to_string(), st.queue.to_string(), st.rejected.to_string(), st.threads.to_string(), st.completed.to_string()),
                None => ("0".into(), "0".into(), "0".into(), "0".into(), "0".into(), "0".into()),
            };
            let fixed_type = live.is_some() || *fixed;
            vec![
                node.name.clone(), NODE_ID[..4].to_string(), NODE_ID[..4].to_string(), std::process::id().to_string(), "127.0.0.1".into(), "127.0.0.1".into(), "9300".into(),
                n.to_string(), if fixed_type { "fixed".into() } else { "scaling".into() }, active, pool_size, queue, qsize, rejected, largest, completed,
                core, max, size, ka,
            ]
        })
        .collect();
    render(
        req,
        Table {
            cols: vec![
                ("node_name", &["nn"], true), ("node_id", &["id"], false), ("ephemeral_node_id", &["eid"], false), ("pid", &["p"], false), ("host", &["h"], false),
                ("ip", &["i"], false), ("port", &["po"], false), ("name", &["n"], true), ("type", &["t"], false), ("active", &["a"], true),
                ("pool_size", &["psz"], false), ("queue", &["q"], true), ("queue_size", &["qs"], false), ("rejected", &["r"], true), ("largest", &["l"], false),
                ("completed", &["c"], false), ("core", &["cr"], false), ("max", &["mx"], false), ("size", &["sz"], false), ("keep_alive", &["ka"], false),
            ],
            rows,
            numeric: vec!["pid", "active", "pool_size", "queue", "queue_size", "rejected", "largest", "completed", "core", "max", "size"],
        },
    )
}
