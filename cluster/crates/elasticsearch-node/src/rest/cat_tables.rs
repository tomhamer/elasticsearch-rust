//! Cluster-wide `_cat` tables. Column sets, aliases, default visibility and
//! alignment follow ES 8.x's `Rest*Action.getTableWithHeader`; the data
//! comes from the cluster state plus per-copy stats gathered from every node.

use super::cat::{self, cols, Row};
use super::cat_columns as C;
use super::{gather_shard_stats, recovery, LocalShardStats};
use crate::front::{RestReq, RestResp};
use crate::master::now_millis;
use crate::node::ClusterNode;
use elasticsearch_cluster::health::{self, HealthInputs};
use elasticsearch_cluster::state::{IndexState, ShardRouting};
use elasticsearch_engine::node::ResolveOpts;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn now_parts() -> (String, String) {
    let ms = now_millis();
    let hms = chrono::DateTime::from_timestamp_millis(ms).map(|d| d.format("%H:%M:%S").to_string()).unwrap_or_default();
    ((ms / 1000).to_string(), hms)
}

/// `expand_wildcards` for `_cat` index tables (default: open + closed, no hidden).
pub fn cat_resolve_opts(req: &RestReq, default_closed: bool) -> Result<ResolveOpts, RestResp> {
    let p = req.params();
    let mut o = ResolveOpts { expand_closed: default_closed, ..Default::default() };
    if let Some(v) = p.get("ignore_unavailable") {
        o.ignore_unavailable = v == "true";
    }
    if let Some(v) = p.get("allow_no_indices") {
        o.allow_no_indices = v != "false";
    }
    if let Some(ew) = p.get("expand_wildcards") {
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
                other => return Err(RestResp::error(400, "illegal_argument_exception", &format!("No enum constant org.elasticsearch.action.support.IndicesOptions.WildcardStates.{}", other.to_uppercase()))),
            }
        }
    }
    Ok(o)
}

/// Concrete index names for an expression (None = every index the options allow).
pub fn resolve(node: &ClusterNode, expr: Option<&str>, opts: ResolveOpts) -> Result<Vec<String>, RestResp> {
    match node.engine.resolve(expr.unwrap_or("_all"), opts) {
        Ok(t) => Ok(t.iter().map(|t| t.svc.name.clone()).collect()),
        Err(e) => Err(RestResp::json(e.status, &e.to_json_with_trace(false))),
    }
}

fn stats_of<'a>(stats: &'a BTreeMap<crate::front::SharedText, LocalShardStats>, s: &ShardRouting) -> Option<&'a LocalShardStats> {
    s.allocation_id().and_then(|a| stats.get(a))
}

// ------------------------------------------------------------------ health


pub fn cat_health(node: &ClusterNode, req: &RestReq) -> RestResp {
    let st = node.state();
    let h = health::cluster_health(&st, None, HealthInputs::default());
    let row = health::cat_health_row(&h, now_millis());
    let mut c = cols(C::HEALTH_COLS);
    if req.params().get("ts").map(|v| v == "false").unwrap_or(false) {
        for col in c.iter_mut().filter(|c| c.name == "epoch" || c.name == "timestamp") {
            col.default = false;
        }
    }
    cat::render(req, &c, vec![row])
}

// ------------------------------------------------------------------ master

pub fn cat_master(node: &ClusterNode, req: &RestReq) -> RestResp {
    let st = node.state();
    let rows = st.master().map(|m| vec![vec![("id", Some(m.id.clone())), ("host", Some(m.host.clone())), ("ip", Some(m.ip.clone())), ("node", Some(m.name.clone()))]]).unwrap_or_default();
    let c = cols(C::MASTER_COLS);
    cat::render(req, &c, rows)
}

// --------------------------------------------------------------- nodeattrs

pub fn cat_nodeattrs(node: &ClusterNode, req: &RestReq) -> RestResp {
    let st = node.state();
    let mut rows = Vec::new();
    for n in st.nodes.values() {
        for (k, v) in &n.attributes {
            rows.push(vec![
                ("node", Some(n.name.clone())),
                ("id", Some(if full_id(req) { n.id.clone() } else { n.id.chars().take(4).collect() })),
                ("pid", Some(std::process::id().to_string())),
                ("host", Some(n.host.clone())),
                ("ip", Some(n.ip.clone())),
                ("port", n.transport_address.rsplit(':').next().map(str::to_string)),
                ("attr", Some(k.clone())),
                ("value", Some(v.clone())),
            ]);
        }
    }
    let c = cols(C::NODEATTRS_COLS);
    cat::render(req, &c, rows)
}

fn projection_bytes(parts:impl IntoIterator<Item=usize>)->Result<usize,RestResp>{
    parts.into_iter().try_fold(0usize,usize::checked_add).ok_or_else(||RestResp::error(429,"circuit_breaking_exception","CAT projection size overflow"))
}

// --------------------------------------------------------------- fielddata

pub const ACTION_CAT_FIELDDATA: &str = "internal:cat/fielddata";

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FielddataReq {
    /// The engine `_cat/fielddata` path and query (JSON rows, bytes).
    pub uri: crate::front::SharedText,
    /// The caller's credentials: every engine authorizes what it serves.
    pub headers: crate::front::SharedHeaders,
}

#[derive(serde::Serialize)]
#[serde(transparent)]
pub struct FielddataSizes {
    values:BTreeMap<crate::front::SharedText,u64>,
    #[serde(skip)]
    memory:elasticsearch_engine::memory::Reservation,
}
impl FielddataSizes {
    fn new()->Result<Self,RestResp>{Ok(Self{values:BTreeMap::new(),memory:elasticsearch_engine::memory::breakers().reserve_request(0,"fielddata map").map_err(|e|RestResp::from_es_error(&e))?.into_result_owner()})}
    fn insert(&mut self,field:&str,size:u64)->Result<(),RestResp>{
        if let Some(v)=self.values.get_mut(field){*v+=size;return Ok(())}
        let bytes=2*(11*std::mem::size_of::<(crate::front::SharedText,u64)>()+20*std::mem::size_of::<usize>());
        self.memory.grow(bytes as u64,"fielddata map node").map_err(|e|RestResp::from_es_error(&e))?;
        let field=crate::front::SharedText::try_new(field).map_err(|e|RestResp::from_es_error(&e))?;
        self.values.insert(field,size);Ok(())
    }
    fn parse(text:&str)->Result<Self,RestResp>{
        let tree=crate::owned_json::OwnedJson::parse(text).map_err(|e|if e.status==429{RestResp::from_es_error(&e)}else{RestResp::error(500,"exception",&e.reason)})?;
        let mut out=Self::new()?;
        let map=tree.as_object().ok_or_else(||RestResp::error(500,"exception","fielddata response must be an object"))?;
        for (field,size) in map {out.insert(field,size.as_u64().ok_or_else(||RestResp::error(500,"exception","invalid fielddata size"))?)?;}
        Ok(out)
    }
}

/// Fielddata held by this node, per field, summed over the engines holding
/// its shard copies (shard 0 and the shard engines).
pub async fn local_fielddata(node: &ClusterNode, r: &FielddataReq) -> Result<FielddataSizes, RestResp> {
    let engines=node.shard_engines.admitted_ids().map_err(|e|RestResp::from_es_error(&e))?;
    let mut sizes = FielddataSizes::new()?;
    for k in std::iter::once(0u32).chain(engines.iter().copied()) {
        let mut req=RestReq::try_new("GET",&r.uri,&r.headers,&[],1,Default::default(),None).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
        req.set_shard(k).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
        let resp = crate::front::local_engine(node, &req).await;
        if !resp.is_success() {
            return Err(resp);
        }
        let text=std::str::from_utf8(&resp.body).map_err(|_|RestResp::error(500,"exception","invalid fielddata JSON UTF-8"))?;
        let tree=crate::owned_json::OwnedJson::parse(text).map_err(|e|RestResp::from_es_error(&e))?;
        for row in tree.as_array().into_iter().flatten() {
            let (Some(field), Some(size)) = (row["field"].as_str(), row["size"].as_str().and_then(|s| s.parse::<u64>().ok())) else { continue };
            // A node lists only fields it has loaded (`indices.fielddata.fields`).
            if size > 0 {
                sizes.insert(field,size)?;
            }
        }
    }
    Ok(sizes)
}

/// `GET /_cat/fielddata`: one row per node and field with loaded fielddata
/// (ES builds it from every node's `indices.fielddata.fields` stats).
pub async fn cat_fielddata(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let p = req.params();
    let fields = req.segments().get(2).cloned().or_else(|| p.get("fields").cloned());
    let uri=match &fields {
        Some(field)=>crate::front::SharedText::try_format(format_args!("/_cat/fielddata?format=json&bytes=b&h=field,size&fields={}",crate::front::UriComponent(field))),
        None=>crate::front::SharedText::try_format(format_args!("/_cat/fielddata?format=json&bytes=b&h=field,size")),
    };
    let uri=match uri {Ok(uri)=>uri,Err(e)=>return RestResp::json(e.status,&e.to_json())};
    let headers=match crate::reindex_multi::credentials(req) {Ok(v)=>v,Err(e)=>return RestResp::json(e.status,&e.to_json())};
    let r = FielddataReq { uri, headers };
    let st = node.state();
    let unit = p.get("bytes").cloned();
    let full = full_id(req);
    let mut row_memory=match elasticsearch_engine::memory::breakers().reserve_request(0,"fielddata CAT rows"){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let mut rows = Vec::new();
    for id in st.nodes.keys() {
        let sizes=if *id==node.local.id {local_fielddata(node,&r).await} else if let Some(addr)=node.address_of(id){
            match node.transport.send_request::<_,crate::front::SharedText>(addr,ACTION_CAT_FIELDDATA,&r,std::time::Duration::from_secs(30)).await {
                Ok(s)=>FielddataSizes::parse(&s),
                Err(e)=>{let e=e.to_remote_error();Err(RestResp::error(e.status,&e.error_type,&e.reason))},
            }
        }else{FielddataSizes::new()};
        let sizes=match sizes{Ok(v)=>v,Err(e)=>return e};
        let Some(n) = st.nodes.get(id) else { continue };
        for (field, size) in &sizes.values {
            let bytes=match projection_bytes([6*std::mem::size_of::<(&str,Option<String>)>(),std::mem::size_of::<Row>(),n.id.len(),n.host.len(),n.ip.len(),n.name.len(),field.len(),128]){Ok(v)=>v,Err(e)=>return e};
            if let Err(e)=row_memory.grow(bytes as u64,"fielddata CAT row"){return RestResp::from_es_error(&e)}
            // Each replacement is charged before allocating, retaining old
            // and new outer-vector storage while the allocator moves rows.
            if rows.len()==rows.capacity(){
                if let Err(e)=row_memory.grow(match rows.len().checked_add(1).and_then(|n|n.checked_mul(std::mem::size_of::<Row>())){Some(v)=>v as u64,None=>return RestResp::error(429,"circuit_breaking_exception","CAT row slots overflow")},"fielddata CAT row slots"){return RestResp::from_es_error(&e)}
                if rows.try_reserve_exact(1).is_err(){return RestResp::error(429,"circuit_breaking_exception","fielddata row allocation failed")}
            }
            rows.push(vec![
                ("id", Some(if full { n.id.clone() } else { n.id.chars().take(4).collect() })),
                ("host", Some(n.host.clone())),
                ("ip", Some(n.ip.clone())),
                ("node", Some(n.name.clone())),
                ("field", Some(field.to_string())),
                ("size", Some(cat::bytes(*size, unit.as_deref()))),
            ]);
        }
    }
    cat::render(req, &cols(C::FIELDDATA_COLS), rows)
}

fn full_id(req: &RestReq) -> bool {
    matches!(req.params().get("full_id").map(String::as_str), Some("") | Some("true"))
}

// ----------------------------------------------------------- pending tasks

pub fn cat_pending_tasks(req: &RestReq) -> RestResp {
    let c = cols(C::PENDING_TASKS_COLS);
    cat::render(req, &c, vec![])
}

// ------------------------------------------------------------------- nodes


async fn node_stats_entry(node: &ClusterNode, id: &str, metrics: &str) -> Result<crate::owned_json::OwnedJson,RestResp> {
    let uri = crate::front::SharedText::try_format(format_args!("/_nodes/_local/stats/{metrics}")).map_err(|e|RestResp::from_es_error(&e))?;
    node_monitor_entry(node, id, &uri).await
}

async fn node_monitor_entry(node: &ClusterNode, id: &str, uri: &str) -> Result<crate::owned_json::OwnedJson,RestResp> {
    if id == node.local.id {
        return super::nodes::local_entry(node, &uri).await;
    }
    let Some(addr) = node.address_of(id) else { return crate::owned_json::OwnedJson::parse("null").map_err(|e|RestResp::from_es_error(&e)) };
    #[derive(serde::Serialize)]
    struct LocalReq<'a> {
        uri: &'a str,
    }
    match node.transport.send_request::<_, crate::front::SharedText>(addr, super::nodes::ACTION_LOCAL, &LocalReq { uri }, std::time::Duration::from_secs(10)).await {
        Ok(s) => crate::owned_json::OwnedJson::parse(&s).map_err(|e|RestResp::from_es_error(&e)),
        Err(e) => {let e=e.to_remote_error();Err(RestResp::error(e.status,&e.error_type,&e.reason))},
    }
}

pub async fn cat_nodes(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let st = node.state();
    let p = req.params();
    let full = full_id(req);
    let unit = p.get("bytes").cloned();
    let unit = unit.as_deref();
    let row_slots=match st.nodes.len().checked_mul(std::mem::size_of::<Row>()){Some(v)=>v,None=>return RestResp::error(429,"circuit_breaking_exception","CAT row slots overflow")};
    let mut row_memory=match elasticsearch_engine::memory::breakers().reserve_request(row_slots as u64,"CAT node row slots"){Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let mut rows = Vec::new();
    if rows.try_reserve_exact(st.nodes.len()).is_err(){return RestResp::error(429,"circuit_breaking_exception","CAT node row allocation failed")}
    for (id,n) in &st.nodes {
        let (stats,os)=tokio::join!(node_stats_entry(node,id,"os,process,fs,jvm,indices"),node_monitor_entry(node,id,"/_nodes/_local/os"));
        let stats=match stats{Ok(v)=>v,Err(e)=>return e};
        let os=match os{Ok(v)=>v,Err(e)=>return e};
        // This row has fewer than 128 cells. Fixed f64 formatting can emit
        // 309 integer digits; 512 bytes covers each numeric/time cell.
        let bytes=match projection_bytes([128*(std::mem::size_of::<(&str,Option<String>)>()+512),n.id.len(),n.ip.len(),n.transport_address.len(),n.http_address.len(),n.version.len(),n.name.len(),n.roles.len()]){Ok(v)=>v,Err(e)=>return e};
        if let Err(e)=row_memory.grow(bytes as u64,"CAT node row"){return RestResp::from_es_error(&e)}
        let u = |v: &Value| v.as_u64();
        let mem_total = u(&stats["os"]["mem"]["total_in_bytes"]).unwrap_or(0);
        let mem_used = u(&stats["os"]["mem"]["used_in_bytes"]).unwrap_or(0);
        let fs_total = u(&stats["fs"]["total"]["total_in_bytes"]).unwrap_or(0);
        let fs_avail = u(&stats["fs"]["total"]["available_in_bytes"]).unwrap_or(0);
        let heap_used = u(&stats["jvm"]["mem"]["heap_used_in_bytes"]).unwrap_or(mem_used / 16);
        let heap_max = u(&stats["jvm"]["mem"]["heap_max_in_bytes"]).filter(|m| *m > 0).unwrap_or(mem_total.max(1) / 2);
        let fd_cur = stats["process"]["open_file_descriptors"].as_i64();
        let fd_max = stats["process"]["max_file_descriptors"].as_i64();
        let pct = |a: u64, b: u64| if b == 0 { "0".to_string() } else { (a * 100 / b).to_string() };
        let load = |k: &str| stats["os"]["cpu"]["load_average"][k].as_f64().map(|v| format!("{v:.2}"));
        let idx = &stats["indices"];
        let num = |v: &Value| v.as_u64().map(|x| x.to_string());
        let size = |v: &Value| v.as_u64().map(|x| cat::bytes(x, unit));
        let time = |v: &Value| v.as_u64().map(|x| cat::time_ms(x, p.get("time").map(String::as_str)));
        let row: Row = vec![
            ("id", Some(if full { n.id.clone() } else { n.id.chars().take(4).collect() })),
            ("pid", Some(stats["process"]["id"].as_u64().map(|v| v.to_string()).unwrap_or_else(|| std::process::id().to_string()))),
            ("ip", Some(n.ip.clone())),
            ("port", n.transport_address.rsplit(':').next().map(str::to_string)),
            ("http_address", Some(n.http_address.clone())),
            ("version", Some(n.version.clone())),
            ("type", Some("tar".into())),
            ("build", Some("unknown".into())),
            ("jdk", Some("n/a".into())),
            ("disk.total", Some(cat::bytes(fs_total, unit))),
            ("disk.used", Some(cat::bytes(fs_total.saturating_sub(fs_avail), unit))),
            ("disk.avail", Some(cat::bytes(fs_avail, unit))),
            ("disk.used_percent", Some(if fs_total == 0 { "0.00".into() } else { format!("{:.2}", 100.0 * (fs_total - fs_avail) as f64 / fs_total as f64) })),
            ("heap.current", Some(cat::bytes(heap_used, unit))),
            ("heap.percent", Some(pct(heap_used.min(heap_max), heap_max))),
            ("heap.max", Some(cat::bytes(heap_max, unit))),
            ("ram.current", Some(cat::bytes(mem_used, unit))),
            ("ram.percent", Some(pct(mem_used, mem_total))),
            ("ram.max", Some(cat::bytes(mem_total, unit))),
            ("file_desc.current", fd_cur.map(|v| v.to_string())),
            ("file_desc.percent", Some(match (fd_cur, fd_max) {
                (Some(c), Some(m)) if m > 0 && c >= 0 => (c * 100 / m).to_string(),
                _ => "0".into(),
            })),
            ("file_desc.max", fd_max.map(|v| v.to_string())),
            ("cpu", stats["os"]["cpu"]["percent"].as_i64().map(|v| v.to_string())),
            ("available_processors", os["os"]["available_processors"].as_u64().map(|v| v.to_string())),
            ("load_1m", load("1m")),
            ("load_5m", load("5m")),
            ("load_15m", load("15m")),
            ("uptime", stats["jvm"]["uptime_in_millis"].as_u64().map(|v| cat::time_ms(v, None))),
            ("node.role", Some(n.role_abbreviations())),
            ("master", Some(if st.master_node_id.as_deref() == Some(n.id.as_str()) { "*".into() } else { "-".into() })),
            ("name", Some(n.name.clone())),
            ("completion.size", size(&idx["completion"]["size_in_bytes"])),
            ("fielddata.memory_size", size(&idx["fielddata"]["memory_size_in_bytes"])),
            ("fielddata.evictions", num(&idx["fielddata"]["evictions"])),
            ("query_cache.memory_size", size(&idx["query_cache"]["memory_size_in_bytes"])),
            ("query_cache.evictions", num(&idx["query_cache"]["evictions"])),
            ("query_cache.hit_count", num(&idx["query_cache"]["hit_count"])),
            ("query_cache.miss_count", num(&idx["query_cache"]["miss_count"])),
            ("request_cache.memory_size", size(&idx["request_cache"]["memory_size_in_bytes"])),
            ("request_cache.evictions", num(&idx["request_cache"]["evictions"])),
            ("request_cache.hit_count", num(&idx["request_cache"]["hit_count"])),
            ("request_cache.miss_count", num(&idx["request_cache"]["miss_count"])),
            ("flush.total", num(&idx["flush"]["total"])),
            ("flush.total_time", time(&idx["flush"]["total_time_in_millis"])),
            ("get.current", num(&idx["get"]["current"])),
            ("get.time", time(&idx["get"]["time_in_millis"])),
            ("get.total", num(&idx["get"]["total"])),
            ("indexing.delete_current", num(&idx["indexing"]["delete_current"])),
            ("indexing.delete_time", time(&idx["indexing"]["delete_time_in_millis"])),
            ("indexing.delete_total", num(&idx["indexing"]["delete_total"])),
            ("indexing.index_current", num(&idx["indexing"]["index_current"])),
            ("indexing.index_time", time(&idx["indexing"]["index_time_in_millis"])),
            ("indexing.index_total", num(&idx["indexing"]["index_total"])),
            ("indexing.index_failed", num(&idx["indexing"]["index_failed"])),
            ("merges.current", num(&idx["merges"]["current"])),
            ("merges.total", num(&idx["merges"]["total"])),
            ("refresh.total", num(&idx["refresh"]["total"])),
            ("refresh.time", time(&idx["refresh"]["total_time_in_millis"])),
            ("search.fetch_total", num(&idx["search"]["fetch_total"])),
            ("search.open_contexts", num(&idx["search"]["open_contexts"])),
            ("search.query_total", num(&idx["search"]["query_total"])),
            ("search.scroll_total", num(&idx["search"]["scroll_total"])),
            ("segments.count", num(&idx["segments"]["count"])),
            ("segments.memory", size(&idx["segments"]["memory_in_bytes"])),
            ("bulk.total_operations", num(&idx["bulk"]["total_operations"])),
            ("shard_stats.total_count", Some(st.routing_table.shards_on_node(&n.id).count().to_string())),
            ("mappings.total_count", num(&idx["mappings"]["total_count"])),
            ("mappings.total_estimated_overhead_in_bytes", size(&idx["mappings"]["total_estimated_overhead_in_bytes"])),
        ];
        rows.push(row);
    }
    cat::render(req, &cols(C::NODES_COLS), rows)
}

// ------------------------------------------------------------------ shards


pub async fn cat_shards(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let st = node.state();
    let segs = req.segments();
    let indices: Option<Vec<String>> = match segs.get(2) {
        Some(e) => {
            let mut o = match cat_resolve_opts(req, true) {
                Ok(o) => o,
                Err(r) => return r,
            };
            if !req.params().contains_key("expand_wildcards") {
                o.expand_hidden = true;
            }
            match resolve(node, Some(e), o) {
                Ok(v) => Some(v),
                Err(r) => return r,
            }
        }
        None => None,
    };
    let stats = match gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let bytes = req.params().get("bytes").cloned();
    let lookup = |s: &ShardRouting| stats_of(&stats, s).map(|x| health::CatShardStats { docs: Some(x.docs), store_bytes: Some(x.store_bytes), dataset_bytes: Some(x.store_bytes), max_seq_no: None, local_checkpoint: None, global_checkpoint: None });
    let mut rows = health::cat_shards_rows(&st, indices.as_deref(), now_millis(), &lookup, bytes.as_deref());
    // Per-copy defaults ES always reports for assigned copies.
    for r in rows.iter_mut() {
        let assigned = r.iter().any(|(n, v)| *n == "node" && v.is_some());
        if assigned {
            for k in ["get.current", "indexing.index_current", "indexing.delete_current", "merges.current", "search.open_contexts", "search.query_current", "search.fetch_current", "search.scroll_current", "warmer.current", "refresh.listeners"] {
                if !r.iter().any(|(n, _)| *n == k) {
                    r.push((k, Some("0".into())));
                }
            }
        }
    }
    let mut columns = cols(C::SHARDS_COLS);
    if !req.json_content_type().contains("compatible-with=8") && !req.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("accept") && v.contains("compatible-with=8")) {
        columns.retain(|c| c.name != "sync_id");
    }
    cat::render(req, &columns, rows)
}

// -------------------------------------------------------------- allocation


pub async fn cat_allocation(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let st = node.state();
    let filter: Option<BTreeSet<String>> = req.segments().get(2).map(|spec| super::nodes::resolve_nodes(&st, &node.local.id, spec));
    let mut info = elasticsearch_cluster::allocation::ClusterInfo::default();
    let data: Vec<String> = st.data_nodes().map(|n| n.id.clone()).collect();
    let disks = futures::future::join_all(data.iter().map(|id| super::nodes::fetch_disk(node, id))).await;
    for (id, d) in data.iter().zip(disks) {
        if let Ok(Some(d)) = d {
            info.nodes.insert(id.clone(), d);
        }
    }
    let stats = match gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    for s in st.routing_table.all_shards() {
        if let Some(x) = stats_of(&stats, s) {
            info.shard_sizes.insert((s.index.clone(), s.shard, s.primary), x.store_bytes);
        }
    }
    let rows = health::cat_allocation_rows(&st, &info, req.params().get("bytes").map(String::as_str));
    let names: BTreeMap<String, String> = st.nodes.values().map(|n| (n.name.clone(), n.id.clone())).collect();
    let rows: Vec<Row> = rows
        .into_iter()
        .filter(|r| {
            let Some(f) = &filter else { return true };
            let name = r.iter().find(|(n, _)| *n == "node").and_then(|(_, v)| v.clone()).unwrap_or_default();
            names.get(&name).map(|id| f.contains(id)).unwrap_or(false)
        })
        .map(|mut r| {
            let assigned = r.iter().any(|(n, v)| *n == "node" && v.as_deref() != Some("UNASSIGNED"));
            for (n, v) in r.iter_mut() {
                if assigned && v.is_none() {
                    match *n {
                        "shards.undesired" => *v = Some("0".into()),
                        "write_load.forecast" => *v = Some("0.0".into()),
                        _ => {}
                    }
                }
            }
            r
        })
        .collect();
    cat::render(req, &cols(C::ALLOCATION_COLS), rows)
}

// ----------------------------------------------------------------- indices


pub async fn cat_indices(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let p = req.params();
    if let Some(h) = p.get("health") {
        if !matches!(h.as_str(), "green" | "yellow" | "red" | "unknown" | "unavailable") {
            return RestResp::error(400, "illegal_argument_exception", &format!("unknown health status [{h}]"));
        }
    }
    let opts = match cat_resolve_opts(req, true) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let names = match resolve(node, req.segments().get(2).map(String::as_str), opts) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let st = node.state();
    let stats = match gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let h = health::cluster_health(&st, Some(&names), HealthInputs::default());
    let unit = p.get("bytes").cloned();
    let mut rows = Vec::new();
    for name in &names {
        let Some(im) = st.metadata.indices.get(name) else { continue };
        let open = im.state == IndexState::Open;
        let status = h.indices.iter().find(|x| &x.index == name).map(|x| x.status.name().to_string());
        if let Some(want) = p.get("health") {
            if status.as_deref() != Some(want.as_str()) {
                continue;
            }
        }
        let (mut docs, mut del, mut store, mut pri_store) = (0u64, 0u64, 0u64, 0u64);
        let mut have = false;
        if let Some(irt) = st.routing_table.indices.get(name) {
            for t in &irt.shards {
                for c in t.active_copies() {
                    if let Some(s) = stats_of(&stats, c) {
                        store += s.store_bytes;
                        if c.primary {
                            have = true;
                            docs += s.docs;
                            del += s.deleted;
                            pri_store += s.store_bytes;
                        }
                    }
                }
            }
        }
        let show = open && have;
        let b = |v: u64| if show { Some(cat::bytes(v, unit.as_deref())) } else { None };
        rows.push(vec![
            ("health", status),
            ("status", Some(if open { "open" } else { "close" }.into())),
            ("index", Some(name.clone())),
            ("uuid", Some(im.uuid.clone())),
            ("pri", Some(im.number_of_shards.to_string())),
            ("rep", Some(im.number_of_replicas.to_string())),
            ("docs.count", show.then(|| docs.to_string())),
            ("docs.deleted", show.then(|| del.to_string())),
            ("creation.date", Some(im.creation_date.to_string())),
            ("creation.date.string", Some(super::util::iso_millis(im.creation_date))),
            ("store.size", b(store)),
            ("pri.store.size", b(pri_store)),
            ("dataset.size", b(store)),
        ]);
    }
    cat::render(req, &cols(C::INDICES_COLS), rows)
}

// ------------------------------------------------------------------- count

pub async fn cat_count(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let names = match resolve(node, req.segments().get(2).map(String::as_str), ResolveOpts::from_params(&req.params())) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let st = node.state();
    let stats = match gather_shard_stats(node, &st).await{Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)};
    let mut count = 0u64;
    for name in &names {
        if let Some(irt) = st.routing_table.indices.get(name) {
            for t in &irt.shards {
                if let Some(s) = t.active_primary().and_then(|p| stats_of(&stats, p)) {
                    count += s.docs;
                }
            }
        }
    }
    let (epoch, ts) = now_parts();
    let c = cols(C::COUNT_COLS);
    cat::render(req, &c, vec![vec![("epoch", Some(epoch)), ("timestamp", Some(ts)), ("count", Some(count.to_string()))]])
}

// ---------------------------------------------------------------- recovery


pub async fn cat_recovery(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    let p = req.params();
    let expr = req.segments().get(2).cloned();
    let mut entries = match recovery::collect(node, req, expr.as_deref()).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    let unit = p.get("bytes").cloned();
    let tunit = p.get("time").cloned();
    let count=match entries.prepare_cat_rows(){Ok(n)=>n,Err(e)=>return e};
    let mut rows = Vec::new();
    if rows.try_reserve_exact(count).is_err(){return RestResp::error(429,"circuit_breaking_exception","recovery CAT rows allocation failed")};
    for (index, list) in &entries {
        for r in list {
            let s = |v: &Value| v.as_str().map(str::to_string).filter(|s| !s.is_empty()).unwrap_or_else(|| "n/a".into());
            let n = |v: &Value| v.as_i64().unwrap_or(0).to_string();
            let start = r["start_time_in_millis"].as_i64().unwrap_or(0);
            let stop = r["stop_time_in_millis"].as_i64().unwrap_or(0);
            let size = &r["index"]["size"];
            let files = &r["index"]["files"];
            let total_b = size["total_in_bytes"].as_u64().unwrap_or(0);
            let reused_b = size["reused_in_bytes"].as_u64().unwrap_or(0);
            rows.push(vec![
                ("index", Some(index.clone())),
                ("shard", Some(n(&r["id"]))),
                ("start_time", Some(super::util::iso_millis(start))),
                ("start_time_millis", Some(start.to_string())),
                ("stop_time", Some(super::util::iso_millis(stop))),
                ("stop_time_millis", Some(stop.to_string())),
                ("time", Some(cat::time_ms(r["total_time_in_millis"].as_u64().unwrap_or(0), tunit.as_deref()))),
                ("type", Some(r["type"].as_str().unwrap_or("").to_lowercase())),
                ("stage", Some(r["stage"].as_str().unwrap_or("").to_lowercase())),
                ("source_host", Some(s(&r["source"]["host"]))),
                ("source_node", Some(s(&r["source"]["name"]))),
                ("target_host", Some(s(&r["target"]["host"]))),
                ("target_node", Some(s(&r["target"]["name"]))),
                ("repository", Some(s(&r["source"]["repository"]))),
                ("snapshot", Some(s(&r["source"]["snapshot"]))),
                ("files", Some((files["total"].as_i64().unwrap_or(0) - files["reused"].as_i64().unwrap_or(0)).to_string())),
                ("files_recovered", Some(n(&files["recovered"]))),
                ("files_percent", Some(files["percent"].as_str().unwrap_or("0.0%").to_string())),
                ("files_total", Some(n(&files["total"]))),
                ("bytes", Some(cat::bytes(total_b.saturating_sub(reused_b), unit.as_deref()))),
                ("bytes_recovered", Some(cat::bytes(size["recovered_in_bytes"].as_u64().unwrap_or(0), unit.as_deref()))),
                ("bytes_percent", Some(size["percent"].as_str().unwrap_or("0.0%").to_string())),
                ("bytes_total", Some(cat::bytes(total_b, unit.as_deref()))),
                ("translog_ops", Some(n(&r["translog"]["total"]))),
                ("translog_ops_recovered", Some(n(&r["translog"]["recovered"]))),
                ("translog_ops_percent", Some(r["translog"]["percent"].as_str().unwrap_or("0.0%").to_string())),
            ]);
        }
    }
    cat::render(req, &cols(C::RECOVERY_COLS), rows)
}


#[cfg(test)]
mod admitted_fielddata_tests {
    use super::*;
    #[test]
    fn typed_fielddata_map_preserves_sum_and_wire_shape() {
        let mut sizes=FielddataSizes::new().unwrap();
        sizes.insert("a",3).unwrap();sizes.insert("a",7).unwrap();sizes.insert("β",11).unwrap();
        let wire=crate::front::SharedText::try_json(&sizes).unwrap();
        assert_eq!(wire.as_str(),r#"{"a":10,"β":11}"#);
        let decoded=FielddataSizes::parse(&wire).unwrap();
        assert_eq!(decoded.values.get("a"),Some(&10));
        assert_eq!(decoded.values.get("β"),Some(&11));
        assert_eq!(FielddataSizes::parse(r#"{"a":-1}"#).err().unwrap().status,500);
        assert_eq!(FielddataSizes::parse("[").err().unwrap().status,500);
    }
}
