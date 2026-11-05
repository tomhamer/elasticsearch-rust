//! Searches whose target indices are not all held by one node.
//!
//! A search is a set of per-shard searches keyed `index#shard` (shard `k`
//! of an index lives in engine `k`, see `shard_engines`). We cover the
//! target shards with as few holder nodes as possible, send each node a
//! search restricted to its shards, and merge the responses the way ES's coordinating
//! node reduces query-phase results for hits: by the request's `sort`
//! values, else by `_score` descending, with `from`/`size` applied after
//! the merge. Totals, `max_score`, `_shards`, `took`, `timed_out` and
//! `terminated_early` are combined; `_count` responses are summed.
//!
//! `search_after` merges exactly (each node applies it, the merge re-sorts).
//!
//! Each node response is treated as one shard result of ES's reduce:
//! aggregations are reduced by [`agg_reduce`] (the per-node request is
//! rewritten to ask for `shard_size` buckets and mergeable helpers),
//! `collapse` keeps the best hit per key after the merge, suggesters are
//! merged per entry, `rescore` runs on each node as it does per shard, and
//! a lone top-level `knn` clause is cut to the global `k`. Several `knn`
//! clauses, or kNN combined with a query, aggregations, sort, ..., run ES's
//! DFS phase first: each clause's global top k ([`knn_dfs_body`]) is folded
//! into the query ([`knn_rewrite`]). `collapse.inner_hits` runs ES's expand
//! phase after the merge: one fanned-out search per hit and entry
//! ([`expand_body`], [`attach_inner_hits`]).

use super::agg_reduce;
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

fn aggs_of(body: &Value) -> Option<&Value> {
    body.get("aggs").or_else(|| body.get("aggregations"))
}

/// `(id, max)` of a well-formed `slice` (`0 <= id < max`, `max > 1`);
/// anything else is left for the engine to reject.
pub fn slice_spec(body: &Value) -> Option<(u64, u64)> {
    let s = body.get("slice")?;
    let num = |v: &Value| v.as_u64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()));
    let (id, max) = (num(s.get("id")?)?, num(s.get("max")?)?);
    (max > 1 && id < max).then_some((id, max))
}

/// The body shard `k` of an `n`-shard index runs for a sliced request
/// (`SliceBuilder#toFilter`): with `max >= n` slice `id` lives on shard
/// `id % n` as local slice `id / n` of `max / n (+1)`; with `max < n`
/// shards `k % max == id` match everything. Other shards match nothing.
pub fn slice_for_shard(body: &Value, k: u32, n: u32) -> Value {
    let mut b = body.clone();
    let Some((id, max)) = slice_spec(body) else { return b };
    let (k, n) = (k as u64, n.max(1) as u64);
    let local = if max >= n {
        (id % n == k).then(|| (id / n, max / n + u64::from(max % n > k)))
    } else {
        (k % max == id).then_some((0, 1))
    };
    let Some(o) = b.as_object_mut() else { return b };
    match local {
        Some((lid, lmax)) if lmax > 1 => {
            let mut s = o["slice"].clone();
            s["id"] = json!(lid);
            s["max"] = json!(lmax);
            o.insert("slice".into(), s);
        }
        Some(_) => {
            o.remove("slice");
        }
        None => {
            o.remove("slice");
            let q = o.remove("query").unwrap_or(json!({"match_all": {}}));
            o.insert("query".into(), json!({"bool": {"must": [q], "filter": [{"match_none": {}}]}}));
        }
    }
    b
}

/// Error raised while reducing aggregations in the last [`merge_search`]
/// on this thread. ES reduces on the coordinating node inside the fetch
/// phase, so a reduce failure (matrix_stats over indices missing a field,
/// a pipeline that cannot resolve) is reported as a
/// `search_phase_execution_exception` (phase `fetch`, no failed shards)
/// caused by it, with the request's status and the cause as root cause.
pub fn take_agg_error() -> Option<elasticsearch_engine::error::EsError> {
    agg_reduce::take_error().map(|(status, kind, reason)| {
        elasticsearch_engine::error::EsError::new(status, "search_phase_execution_exception", "")
            .with("phase", "fetch")
            .with("grouped", true)
            .with("failed_shards", serde_json::json!([]))
            .caused_by(elasticsearch_engine::error::EsError::new(status, &kind, reason))
    })
}

/// A circuit breaker trip of the last [`merge_search`]'s reduce, as ES
/// reports a failed coordinator reduce: 429
/// `search_phase_execution_exception` (phase `fetch`, no failed shards)
/// caused by the whole `circuit_breaking_exception` (`bytes_wanted`,
/// `bytes_limit`, `durability` kept); check before [`take_agg_error`].
pub fn take_agg_breaking() -> Option<elasticsearch_engine::error::EsError> {
    agg_reduce::take_breaking().map(|e| {
        elasticsearch_engine::error::EsError::new(e.status, "search_phase_execution_exception", "")
            .with("phase", "fetch")
            .with("grouped", true)
            .with("failed_shards", serde_json::json!([]))
            .caused_by(e)
    })
}

/// Request features the cross-node reduce cannot reproduce.
pub fn unsupported_feature(body: &Value, params: &std::collections::HashMap<String, String>) -> Option<&'static str> {
    if let Some(why) = aggs_of(body).and_then(agg_reduce::unsupported) {
        return Some(why);
    }
    if !collapse_inner_hits(body).is_empty() && body.get("knn").is_some() {
        // The expand phase would have to re-run the kNN clauses per group.
        return Some("collapse with inner_hits combined with knn");
    }
    if let Some(s) = body.get("suggest").and_then(Value::as_object) {
        for (name, e) in s {
            if name != "text" && !["term", "phrase", "completion"].iter().any(|k| e.get(*k).is_some()) {
                return Some("this suggester");
            }
        }
    }
    if knn_needs_dfs(body, params) && knn_clauses(body).iter().any(|c| c.get("inner_hits").is_some()) {
        return Some("knn with inner_hits");
    }
    for (k, name) in [("pit", "point in time"), ("slice", "slice"), ("retriever", "retriever"), ("rank", "rank")] {
        if body.get(k).is_some() {
            return Some(name);
        }
    }
    if params.contains_key("scroll") {
        return Some("scroll");
    }
    None
}

/// The top-level `knn` clauses of a search (one object or an array).
pub fn knn_clauses(body: &Value) -> Vec<Value> {
    match body.get("knn") {
        Some(Value::Array(a)) => a.clone(),
        Some(k @ Value::Object(_)) => vec![k.clone()],
        _ => Vec::new(),
    }
}

/// URL parameters that make up the `q` query (`RestActions#urlParamsToQueryBuilder`).
pub const Q_PARAMS: [&str; 6] = ["q", "df", "analyzer", "analyze_wildcard", "default_operator", "lenient"];

/// Does this search need ES's DFS kNN phase across shards? A lone `knn`
/// object whose hits nothing else consumes merges exactly by cutting the
/// per-shard top k's to the global k; several clauses, or kNN combined with
/// a query, aggregations, sort, rescore, collapse, ..., need the global top
/// k of each clause first (`DfsQueryPhase#rewriteShardSearchRequest`).
pub fn knn_needs_dfs(body: &Value, params: &std::collections::HashMap<String, String>) -> bool {
    match body.get("knn") {
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(_)) => {
            ["query", "aggs", "aggregations", "sort", "rescore", "collapse", "post_filter", "suggest", "min_score"].iter().any(|f| body.get(*f).is_some())
                || params.contains_key("sort")
                || params.contains_key("q")
        }
        _ => false,
    }
}

/// The DFS-phase body for one `knn` clause: a lone kNN search for its
/// global top `k` (`k` defaults to the request's `size`). Boost and name are
/// applied when the results are combined ([`knn_rewrite`]).
pub fn knn_dfs_body(clause: &Value, size: usize) -> Value {
    let mut c = clause.clone();
    let k = c.get("k").and_then(Value::as_u64).unwrap_or(size.max(1) as u64);
    if let Some(o) = c.as_object_mut() {
        o.remove("boost");
        o.remove("_name");
        o.insert("k".into(), json!(k));
    }
    json!({"knn": c, "from": 0, "size": k, "_source": false})
}

/// The `q` URL query as a `query_string` query (`RestActions#urlParamsToQueryBuilder`).
pub fn q_query(params: &std::collections::HashMap<String, String>) -> Option<Value> {
    let q = params.get("q")?;
    let mut qs = json!({"query": q});
    for (p, f) in [("df", "default_field"), ("analyzer", "analyzer"), ("default_operator", "default_operator")] {
        if let Some(v) = params.get(p) {
            qs[f] = json!(v);
        }
    }
    for (p, f) in [("analyze_wildcard", "analyze_wildcard"), ("lenient", "lenient")] {
        if let Some(v) = params.get(p) {
            qs[f] = json!(v == "true" || v.is_empty());
        }
    }
    Some(json!({"query_string": qs}))
}

/// The `collapse.inner_hits` entries of a search (one object or an array).
pub fn collapse_inner_hits(body: &Value) -> Vec<Value> {
    match body.get("collapse").and_then(|c| c.get("inner_hits")) {
        Some(Value::Array(a)) => a.clone(),
        Some(o @ Value::Object(_)) => vec![o.clone()],
        _ => Vec::new(),
    }
}

/// Can the coordinator expand these entries itself? Every entry needs a
/// non-empty `name` and a nested `collapse` must not have `inner_hits` or
/// `collapse`; otherwise the engine produces ES's exact validation error.
pub fn collapse_inner_hits_valid(body: &Value) -> bool {
    let c = &body["collapse"];
    if !c.is_object() || c.get("field").and_then(Value::as_str).is_none() {
        return false;
    }
    if !matches!(c.get("inner_hits"), Some(Value::Array(_)) | Some(Value::Object(_))) {
        return false;
    }
    collapse_inner_hits(body).iter().all(|e| {
        e.is_object()
            && e.get("name").and_then(Value::as_str).map(|s| !s.is_empty()).unwrap_or(false)
            && e.get("collapse").map(|n| n.get("inner_hits").is_none() && n.get("collapse").is_none()).unwrap_or(true)
    })
}

/// The search without `collapse.inner_hits` (the first round of ES's expand).
pub fn strip_inner_hits(body: &Value) -> Value {
    let mut b = body.clone();
    if let Some(c) = b.get_mut("collapse").and_then(Value::as_object_mut) {
        c.remove("inner_hits");
    }
    b
}

/// One inner-hits search of ES's `ExpandSearchPhase`: the group filter
/// (`match` on the collapse value, or `must_not exists` for a missing one)
/// AND the original query (the body's, or the `q` URL query), the original
/// `post_filter` and `runtime_mappings`, and the entry's own options.
pub fn expand_body(body: &Value, params: &std::collections::HashMap<String, String>, field: &str, value: Option<&Value>, entry: &Value) -> Value {
    let mut group = match value {
        Some(v) => json!({"bool": {"filter": [{"match": {field: {"query": v}}}]}}),
        None => json!({"bool": {"must_not": [{"exists": {"field": field}}]}}),
    };
    if let Some(q) = q_query(params).or_else(|| body.get("query").cloned()) {
        group["bool"]["must"] = json!([q]);
    }
    let mut b = Map::new();
    b.insert("query".into(), group);
    b.insert("from".into(), entry.get("from").cloned().unwrap_or(json!(0)));
    b.insert("size".into(), entry.get("size").cloned().unwrap_or(json!(3)));
    for k in [
        "sort",
        "_source",
        "fields",
        "docvalue_fields",
        "stored_fields",
        "script_fields",
        "highlight",
        "explain",
        "track_scores",
        "version",
        "seq_no_primary_term",
        "collapse",
    ] {
        if let Some(v) = entry.get(k) {
            b.insert(k.into(), v.clone());
        }
    }
    for k in ["post_filter", "runtime_mappings"] {
        if let Some(v) = body.get(k) {
            b.insert(k.into(), v.clone());
        }
    }
    Value::Object(b)
}

/// The collapse value of a hit (`fields.<field>[0]`), `None` when missing.
pub fn collapse_value<'a>(hit: &'a Value, field: &str) -> Option<&'a Value> {
    hit.get("fields").and_then(|f| f.get(field)).and_then(|v| v.get(0)).filter(|v| !v.is_null())
}

/// Attach the expand results: `results[h * entries.len() + e]` is the
/// response of entry `e` for hit `h`.
pub fn attach_inner_hits(resp: &mut Value, entries: &[Value], results: &[Value]) {
    let Some(hits) = resp.pointer_mut("/hits/hits").and_then(Value::as_array_mut) else { return };
    for (h, hit) in hits.iter_mut().enumerate() {
        let mut inner = Map::new();
        for (e, entry) in entries.iter().enumerate() {
            let name = entry.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
            let r = results.get(h * entries.len() + e).map(|r| r["hits"].clone()).unwrap_or(Value::Null);
            inner.insert(name, json!({ "hits": r }));
        }
        if let Some(o) = hit.as_object_mut() {
            o.insert("inner_hits".into(), Value::Object(inner));
        }
    }
}

/// The request with its `knn` clauses replaced by their global top k's,
/// as ES does on each shard after the DFS phase: a `bool` that `should`s
/// the original query (the body's, or the `q` URL query) with one clause
/// per kNN search scoring exactly its top docs with their kNN scores
/// (`KnnScoreDocQueryBuilder`, boost and `_name` kept). `top[i]` holds the
/// [`scored_hits`] of clause `i`. A doc is named by index, `_routing` and
/// id: a document with the same id under another routing (another shard)
/// is a different document and must not score.
pub fn knn_rewrite(body: &Value, params: &std::collections::HashMap<String, String>, top: &[Vec<ScoredHit>]) -> Value {
    let clauses = knn_clauses(body);
    let mut b = body.clone();
    let Some(o) = b.as_object_mut() else { return b };
    o.remove("knn");
    let mut should: Vec<Value> = Vec::new();
    if let Some(qs) = q_query(params) {
        o.remove("query");
        should.push(qs);
    } else if let Some(q) = o.remove("query") {
        should.push(q);
    }
    for (c, docs) in clauses.iter().zip(top) {
        let mut q = if docs.is_empty() {
            json!({"match_none": {}})
        } else {
            let scored: Vec<Value> = docs
                .iter()
                .map(|(i, id, routing, s)| {
                    let routed = match routing {
                        Some(r) => json!({"term": {"_routing": r}}),
                        None => json!({"bool": {"must_not": [{"exists": {"field": "_routing"}}]}}),
                    };
                    json!({"constant_score": {"filter": {"bool": {"filter": [{"term": {"_index": i}}, {"ids": {"values": [id]}}, routed]}}, "boost": s}})
                })
                .collect();
            json!({"bool": {"should": scored}})
        };
        if let Some(boost) = c.get("boost").filter(|v| v.is_number()) {
            q[if docs.is_empty() { "match_none" } else { "bool" }]["boost"] = boost.clone();
        }
        if let Some(name) = c.get("_name") {
            q[if docs.is_empty() { "match_none" } else { "bool" }]["_name"] = name.clone();
        }
        should.push(q);
    }
    o.insert("query".into(), json!({"bool": {"should": should}}));
    b
}

/// `(index, id, _routing, score)` of a hit.
pub type ScoredHit = (String, String, Option<String>, f64);

/// The [`ScoredHit`]s of a merged search response.
pub fn scored_hits(resp: &Value) -> Vec<ScoredHit> {
    resp["hits"]["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|h| Some((h["_index"].as_str()?.to_string(), h["_id"].as_str()?.to_string(), h["_routing"].as_str().map(str::to_string), h["_score"].as_f64()?)))
        .collect()
}

/// Cover `indices` with holder nodes, preferring `prefer` (the local node)
/// and then nodes holding the most still-uncovered indices.
pub fn cover(indices: &BTreeSet<String>, holders: &BTreeMap<String, BTreeSet<String>>, prefer: &str) -> Option<BTreeMap<String, Vec<String>>> {
    let mut left: BTreeSet<String> = indices.clone();
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    while !left.is_empty() {
        let mut nodes: BTreeMap<&str, usize> = BTreeMap::new();
        for i in &left {
            for n in holders.get(i)? {
                *nodes.entry(n.as_str()).or_default() += 1;
            }
        }
        let best = nodes.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| (*a.0 == prefer).cmp(&(*b.0 == prefer))).then_with(|| b.0.cmp(a.0)))?;
        let node = best.0.to_string();
        let take: Vec<String> = left.iter().filter(|i| holders.get(*i).map(|h| h.contains(&node)).unwrap_or(false)).cloned().collect();
        for i in &take {
            left.remove(i);
        }
        out.insert(node, take);
    }
    Some(out)
}

/// Select one active copy per search shard, distributing adjacent shards and
/// successive requests over their holders. Unlike metadata/context coverage,
/// ordinary search must not funnel all work into the HTTP coordinator.
/// The caller preserves explicit preferences and context ownership separately.
pub fn search_cover(indices: &BTreeSet<String>, holders: &BTreeMap<String, BTreeSet<String>>,
    rotation: usize) -> Option<BTreeMap<String, Vec<String>>> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (ordinal, key) in indices.iter().enumerate() {
        let copies = holders.get(key)?;
        if copies.is_empty() { return None; }
        let selected = copies.iter().nth(rotation.wrapping_add(ordinal) % copies.len())?;
        groups.entry(selected.clone()).or_default().push(key.clone());
    }
    Some(groups)
}

#[derive(Clone, Debug)]
pub(super) struct SortKey {
    field: String,
    desc: bool,
    missing_first: bool,
}

/// The request's sort as ES applies it (`_score` desc, fields asc by default).
pub(super) fn sort_keys(body: &Value, params: &std::collections::HashMap<String, String>) -> Option<Vec<SortKey>> {
    let mut keys = Vec::new();
    let one = |field: &str, spec: Option<&Value>| -> SortKey {
        let default_desc = field == "_score";
        let (order, missing) = match spec {
            Some(Value::String(o)) => (Some(o.as_str()), None),
            Some(Value::Object(m)) => (m.get("order").and_then(Value::as_str), m.get("missing").and_then(Value::as_str)),
            _ => (None, None),
        };
        let desc = order.map(|o| o == "desc").unwrap_or(default_desc);
        SortKey { field: field.to_string(), desc, missing_first: missing == Some("_first") }
    };
    match body.get("sort") {
        None => {
            if let Some(s) = params.get("sort") {
                for part in s.split(',') {
                    let (f, o) = part.split_once(':').unwrap_or((part, ""));
                    let mut k = one(f, None);
                    if !o.is_empty() {
                        k.desc = o == "desc";
                    }
                    keys.push(k);
                }
                return Some(keys);
            }
            return None;
        }
        Some(Value::String(f)) => keys.push(one(f, None)),
        Some(Value::Array(a)) => {
            for e in a {
                match e {
                    Value::String(f) => keys.push(one(f, None)),
                    Value::Object(m) => {
                        let (f, spec) = m.iter().next()?;
                        keys.push(one(f, Some(spec)));
                    }
                    _ => return None,
                }
            }
        }
        Some(Value::Object(m)) => {
            for (f, spec) in m {
                keys.push(one(f, Some(spec)));
            }
        }
        _ => return None,
    }
    Some(keys)
}

pub(super) fn cmp_values(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            // Integers compare exactly; mixed falls back to f64.
            match (x.as_i64(), y.as_i64()) {
                (Some(p), Some(q)) => p.cmp(&q),
                _ => x.as_f64().unwrap_or(0.0).partial_cmp(&y.as_f64().unwrap_or(0.0)).unwrap_or(Ordering::Equal),
            }
        }
        // A missing double renders as its sentinel `"Infinity"` /
        // `"-Infinity"` (ES): against a number it compares as that number.
        // Two strings are keyword values (ES compares their bytes), also
        // "Infinity" itself, and the two sentinels order alike either way.
        (Value::Number(_), Value::String(_)) | (Value::String(_), Value::Number(_)) => match (inf_value(a).or_else(|| a.as_f64()), inf_value(b).or_else(|| b.as_f64())) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            _ => Ordering::Equal,
        },
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

fn inf_value(v: &Value) -> Option<f64> {
    match v.as_str()? {
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

pub(super) fn hit_order(a: &Value, b: &Value, keys: Option<&[SortKey]>) -> Ordering {
    match keys {
        Some(keys) => sort_order(a["sort"].as_array().map(Vec::as_slice), b["sort"].as_array().map(Vec::as_slice), keys),
        None => score_order(a["_score"].as_f64(), b["_score"].as_f64()),
    }
}

/// Order of two hits' `sort` values under `keys`.
pub(super) fn sort_order(sa: Option<&[Value]>, sb: Option<&[Value]>, keys: &[SortKey]) -> Ordering {
    for (i, k) in keys.iter().enumerate() {
        let x = sa.and_then(|s| s.get(i)).unwrap_or(&Value::Null);
        let y = sb.and_then(|s| s.get(i)).unwrap_or(&Value::Null);
        let ord = match (x.is_null(), y.is_null()) {
            (true, true) => Ordering::Equal,
            // Missing values go last unless `missing: _first`.
            (true, false) => if k.missing_first { Ordering::Less } else { Ordering::Greater },
            (false, true) => if k.missing_first { Ordering::Greater } else { Ordering::Less },
            _ => {
                let o = cmp_values(x, y);
                if k.desc { o.reverse() } else { o }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Order of two hits by descending `_score`, missing scores last.
pub(super) fn score_order(a: Option<f64>, b: Option<f64>) -> Ordering {
    let (x, y) = (a.unwrap_or(f64::NEG_INFINITY), b.unwrap_or(f64::NEG_INFINITY));
    y.partial_cmp(&x).unwrap_or(Ordering::Equal)
}

/// `from` / `size` of the request (body wins over the URL, ES defaults 0/10).
pub fn from_size(body: &Value, params: &std::collections::HashMap<String, String>) -> (usize, usize) {
    let get = |k: &str, d: usize| body.get(k).and_then(Value::as_u64).map(|v| v as usize).or_else(|| params.get(k).and_then(|v| v.parse().ok())).unwrap_or(d);
    (get("from", 0), get("size", 10))
}

/// The per-node request body: everything up to `from + size`, from 0.
pub fn node_body(body: &Value, from: usize, size: usize) -> Value {
    let mut b = if body.is_object() { body.clone() } else { json!({}) };
    b["from"] = json!(0);
    b["size"] = json!(from + size);
    for k in ["aggs", "aggregations"] {
        if let Some(a) = body.get(k) {
            b[k] = agg_reduce::rewrite_root(a);
        }
    }
    b
}

/// Merge per-node search responses (already in ES shape).
/// (index, `_id`) → fused rank of a final retriever search's rank plan.
fn rank_plan_order(body: &Value) -> Option<std::collections::HashMap<(String, String), usize>> {
    let order = body.get(elasticsearch_engine::retriever::RANK_PLAN_KEY)?.get("order")?.as_array()?;
    Some(order.iter().enumerate().filter_map(|(i, e)| Some(((e[0].as_str()?.to_string(), e[1].as_str()?.to_string()), i))).collect())
}

pub fn merge_search(responses: &[Value], body: &Value, params: &std::collections::HashMap<String, String>) -> Value {
    merge_search_with_limit(responses, body, params, 65_536)
}

/// The coordinating reduce runs after shard HTTP calls, outside the engine's
/// per-shard script scope. Retain the same request-owned raw script table here.
pub fn merge_search_with_source(responses:&[Value],body:&Value,params:&std::collections::HashMap<String,String>,max_buckets:usize,raw:&[u8])->Value {
    let mut environment=elasticsearch_engine::script::ScriptEnv::new(None,None);
    if let Some(source)=std::str::from_utf8(raw).ok().and_then(elasticsearch_engine::source_json::lossless_search_source){
        environment.raw_search_scripts=elasticsearch_engine::source_json::search_scripts(&source);
    }
    elasticsearch_engine::script::with_env(environment,||merge_search_with_limit(responses,body,params,max_buckets))
}

/// `v`'s member `key` found by a scan of its members, not a hash: a hit
/// has a handful, and hashing the key cost more than comparing it to each
/// (250,000 hits merged for a `size: 10000` page over 25 shards).
pub(super) fn member<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_object()?.iter().find(|(name, _)| name.as_str() == key).map(|(_, value)| value)
}

pub fn merge_search_with_limit(responses: &[Value], body: &Value, params: &std::collections::HashMap<String, String>, max_buckets: usize) -> Value {
    let (from, size) = from_size(body, params);
    let keys = sort_keys(body, params);
    let as_int = params.get("rest_total_hits_as_int").map(|v| v == "true").unwrap_or(false);
    // Borrowed: only the page is copied out (a `size: 10000` search over
    // 25 shards merges 250,000 hits for 10,000).
    let mut hits: Vec<&Value> = Vec::new();
    let mut total: u64 = 0;
    let mut gte = false;
    let mut have_total = false;
    let mut max_score: Option<f64> = None;
    let (mut took, mut timed_out, mut terminated_early) = (0u64, false, None::<bool>);
    let mut shards = [0u64; 4];
    let mut failures: Vec<Value> = Vec::new();
    for r in responses {
        took = took.max(r["took"].as_u64().unwrap_or(0));
        timed_out |= r["timed_out"].as_bool().unwrap_or(false);
        if let Some(t) = r.get("terminated_early").and_then(Value::as_bool) {
            terminated_early = Some(terminated_early.unwrap_or(false) || t);
        }
        for (i, k) in ["total", "successful", "skipped", "failed"].iter().enumerate() {
            shards[i] += r["_shards"][k].as_u64().unwrap_or(0);
        }
        if let Some(f) = r["_shards"]["failures"].as_array() {
            failures.extend(f.iter().cloned());
        }
        match &r["hits"]["total"] {
            Value::Number(n) => {
                total += n.as_u64().unwrap_or(0);
                have_total = true;
            }
            Value::Object(o) => {
                total += o.get("value").and_then(Value::as_u64).unwrap_or(0);
                gte |= o.get("relation").and_then(Value::as_str) == Some("gte");
                have_total = true;
            }
            _ => {}
        }
        if let Some(s) = r["hits"]["max_score"].as_f64() {
            max_score = Some(max_score.map_or(s, |m| m.max(s)));
        }
        if let Some(h) = r["hits"]["hits"].as_array() {
            hits.extend(h.iter());
        }
    }
    // Stable sort keeps each node's own tie order (index then doc order).
    // The keys each comparison reads are looked up once per hit.
    match keys.as_deref() {
        Some(keys) => {
            let mut keyed: Vec<(Option<&[Value]>, Option<&str>, &Value)> = hits.iter().map(|h| (member(h, "sort").and_then(Value::as_array).map(Vec::as_slice), member(h, "_index").and_then(Value::as_str), *h)).collect();
            keyed.sort_by(|a, b| sort_order(a.0, b.0, keys).then_with(|| a.1.cmp(&b.1)));
            hits = keyed.into_iter().map(|k| k.2).collect();
        }
        None => {
            let mut keyed: Vec<(Option<f64>, Option<&str>, &Value)> = hits.iter().map(|h| (member(h, "_score").and_then(Value::as_f64), member(h, "_index").and_then(Value::as_str), *h)).collect();
            keyed.sort_by(|a, b| score_order(a.0, b.0).then_with(|| a.1.cmp(&b.1)));
            hits = keyed.into_iter().map(|k| k.2).collect();
        }
    }
    if keys.is_none() {
        if let Some(order) = rank_plan_order(body) {
            // Compound retrievers: score ties keep the fused rank order.
            let rank = |h: &Value| order.get(&(h["_index"].as_str().unwrap_or("").to_string(), h["_id"].as_str().unwrap_or("").to_string())).copied().unwrap_or(usize::MAX);
            hits.sort_by(|a, b| hit_order(a, b, None).then_with(|| rank(a).cmp(&rank(b))));
        }
    }
    if let Some(field) = body["collapse"]["field"].as_str() {
        // The best hit of each collapse key wins, as in ES's merge of
        // per-shard collapsed top docs; totals stay uncollapsed.
        let mut seen: BTreeSet<String> = BTreeSet::new();
        hits.retain(|h| seen.insert(h["fields"][field].get(0).map(Value::to_string).unwrap_or_default()));
    }
    if let Some(knn) = body.get("knn").filter(|k| k.is_object()) {
        let k = knn["k"].as_u64().map(|k| k as usize).unwrap_or(size.max(1));
        hits.truncate(k);
        total = total.min(k as u64);
    }
    // ES `SearchPhaseController.getTotalHits`: the summed per-shard counts
    // are capped at `track_total_hits` (default 10000) as a lower bound.
    // Scroll totals are exact; its first-page result becomes the total stored
    // by the composite scroll for every continuation page.
    let param = params.get("track_total_hits").map(|v| v.parse::<i64>().map(|n| json!(n)).unwrap_or(json!(v == "true")));
    let track_up_to = if params.contains_key("scroll") { None } else { match param.as_ref().or_else(|| body.get("track_total_hits")) {
        Some(Value::Number(n)) => n.as_i64().filter(|n| *n != i32::MAX as i64),
        Some(Value::Bool(_)) => None,
        Some(Value::String(s)) => s.parse::<i64>().ok().filter(|n| *n != i32::MAX as i64),
        _ if as_int => None,
        _ => Some(10_000),
    }};
    if let Some(n) = track_up_to.filter(|n| *n >= 0) {
        if total > n as u64 {
            total = n as u64;
            gte = true;
        }
    }
    let page: Vec<Value> = hits.into_iter().skip(from).take(size).cloned().collect();
    let mut out = Map::new();
    out.insert("took".into(), json!(took));
    out.insert("timed_out".into(), json!(timed_out));
    if let Some(t) = terminated_early {
        out.insert("terminated_early".into(), json!(t));
    }
    // The can-match phase keeps one shard of a search with aggregations
    // even when it would skip them all (engines answering part of the
    // search report their raw skipped count).
    if aggs_of(body).is_some() && shards[0] > 0 && shards[2] >= shards[0] {
        shards[2] = shards[0] - 1;
    }
    let mut sh = json!({"total": shards[0], "successful": shards[1], "skipped": shards[2], "failed": shards[3]});
    if !failures.is_empty() {
        sh["failures"] = Value::Array(failures);
    }
    out.insert("_shards".into(), sh);
    let mut h = Map::new();
    if have_total {
        h.insert("total".into(), if as_int { json!(total) } else { json!({"value": total, "relation": if gte { "gte" } else { "eq" }}) });
    }
    h.insert("max_score".into(), max_score.map(|m| json!(m)).unwrap_or(Value::Null));
    h.insert("hits".into(), Value::Array(page));
    out.insert("hits".into(), Value::Object(h));
    let flag = |k: &str| params.get(k).map(|v| v.is_empty() || v == "true").unwrap_or(false);
    if let Some(spec) = aggs_of(body) {
        if responses.iter().any(|r| r.get("aggregations").is_some()) {
            let parts: Vec<&Value> = responses.iter().map(|r| &r["aggregations"]).collect();
            out.insert("aggregations".into(), agg_reduce::reduce_root_with_limit(spec, &parts, flag("typed_keys"), max_buckets));
        }
    }
    if let Some(spec) = body.get("suggest") {
        if responses.iter().any(|r| r.get("suggest").is_some()) {
            out.insert("suggest".into(), merge_suggest(spec, responses));
        }
    }
    let mut profile = responses.iter().find_map(|r| r.get("profile").cloned());
    if let Some(profile) = profile.as_mut() {
        let shards = responses.iter().flat_map(|r| r["profile"]["shards"].as_array().into_iter().flatten().cloned()).collect();
        profile["shards"] = Value::Array(shards);
        out.insert("profile".into(), profile.clone());
    }
    Value::Object(out)
}

/// Merge suggester results: entries are aligned by position (every node
/// analyzes the same text), options are combined per suggester type the
/// way ES's `Suggest.reduce` does and cut back to `size`.
fn merge_suggest(spec: &Value, responses: &[Value]) -> Value {
    let mut out = Map::new();
    let mut names: Vec<&String> = Vec::new();
    for r in responses {
        for k in r["suggest"].as_object().map(|m| m.keys().collect::<Vec<_>>()).unwrap_or_default() {
            if !names.contains(&k) {
                names.push(k);
            }
        }
    }
    for name in names {
        let bare = name.split_once('#').map(|(_, n)| n).unwrap_or(name);
        let e = &spec[bare];
        let kind = ["term", "phrase", "completion"].into_iter().find(|k| e.get(*k).is_some()).unwrap_or("term");
        let conf = &e[kind];
        let size = conf["size"].as_u64().unwrap_or(5) as usize;
        let by_freq = conf["sort"].as_str() == Some("frequency");
        let skip_dups = conf["skip_duplicates"].as_bool().unwrap_or(false);
        let lists: Vec<&Vec<Value>> = responses.iter().filter_map(|r| r["suggest"][name.as_str()].as_array()).collect();
        let n = lists.iter().map(|l| l.len()).max().unwrap_or(0);
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let mut entry = lists.iter().find_map(|l| l.get(i)).cloned().unwrap_or(Value::Null);
            let mut opts: Vec<Value> = Vec::new();
            let mut index: BTreeMap<String, usize> = BTreeMap::new();
            for o in lists.iter().filter_map(|l| l.get(i)).flat_map(|e| e["options"].as_array().into_iter().flatten()) {
                let key = if kind == "completion" && !skip_dups { format!("{}\u{0}{}", o["_index"], o["_id"]) } else { o["text"].to_string() };
                let score = |v: &Value| v.get("score").or_else(|| v.get("_score")).and_then(Value::as_f64).unwrap_or(0.0);
                match index.get(&key) {
                    None => {
                        index.insert(key, opts.len());
                        opts.push(o.clone());
                    }
                    Some(&at) => {
                        let cur = &mut opts[at];
                        let freq = cur.get("freq").and_then(Value::as_u64).map(|f| f + o["freq"].as_u64().unwrap_or(0));
                        if score(o) > score(cur) {
                            *cur = o.clone();
                        }
                        if let Some(f) = freq {
                            cur["freq"] = json!(f);
                        }
                    }
                }
            }
            let score = |v: &Value| v.get("score").or_else(|| v.get("_score")).and_then(Value::as_f64).unwrap_or(0.0);
            let freq = |v: &Value| v["freq"].as_u64().unwrap_or(0);
            opts.sort_by(|a, b| {
                let by_score = score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal);
                let by_f = freq(b).cmp(&freq(a));
                let first = if by_freq { by_f.then(by_score) } else { by_score.then(by_f) };
                first.then_with(|| a["text"].as_str().cmp(&b["text"].as_str()))
            });
            opts.truncate(size);
            if entry.is_object() {
                entry["options"] = Value::Array(opts);
            }
            entries.push(entry);
        }
        out.insert(name.clone(), Value::Array(entries));
    }
    Value::Object(out)
}

/// A part that searched one shard of each of `n` indices covers exactly
/// `n` shards, so the merged sums match ES (an engine may report more: the
/// index's `number_of_shards`). A shard the part reports neither successful
/// nor failed was not searched: it counts as failed, never as successful.
pub(super) fn normalize_part_shards(part: &mut Value, n: usize) {
    let Some(s) = part.get_mut("_shards").and_then(Value::as_object_mut) else { return };
    let n = n as u64;
    let failed = s.get("failed").and_then(Value::as_u64).unwrap_or(0).min(n);
    let successful = s.get("successful").and_then(Value::as_u64).unwrap_or(n).min(n - failed);
    let skipped = s.get("skipped").and_then(Value::as_u64).unwrap_or(0).min(successful);
    let unsearched = n - failed - successful;
    s.insert("total".into(), json!(n));
    s.insert("successful".into(), json!(successful));
    s.insert("skipped".into(), json!(skipped));
    s.insert("failed".into(), json!(failed + unsearched));
    if unsearched > 0 {
        let mut failures = s.remove("failures").and_then(|f| f.as_array().cloned()).unwrap_or_default();
        failures.push(json!({"shard": null, "index": null, "node": null, "reason": {"type": "shard_not_found_exception", "reason": "no such shard"}}));
        s.insert("failures".into(), Value::Array(failures));
    }
}

/// Merge `_count` responses.
pub fn merge_count(responses: &[Value]) -> Value {
    let mut count = 0u64;
    let mut shards = [0u64; 4];
    for r in responses {
        count += r["count"].as_u64().unwrap_or(0);
        for (i, k) in ["total", "successful", "skipped", "failed"].iter().enumerate() {
            shards[i] += r["_shards"][k].as_u64().unwrap_or(0);
        }
    }
    json!({"count": count, "_shards": {"total": shards[0], "successful": shards[1], "skipped": shards[2], "failed": shards[3]}})
}

/// `_mget` / `_mtermvectors` whose docs live in different shards or on
/// different nodes: ES runs each doc as a single-shard op, so route every
/// doc to the shard its `_id` / `routing` hashes to, send each (node, shard)
/// the same request with its docs, and put the results back in request
/// order. The `ids` form is rewritten as `docs`. Docs whose `_index` does
/// not name exactly one index (missing index, alias with several indices)
/// go to the local node, whose engine renders the per-doc error. `None`
/// when the request is not a multi-doc read the engine can split.
pub(super) async fn multi_doc(node: &std::sync::Arc<crate::node::ClusterNode>, req: &super::RestReq) -> Option<super::RestResp> {
    use crate::shard_engines as se;
    let segs = req.segments();
    let op = segs.iter().rev().find(|s| *s == "_mget" || *s == "_mtermvectors")?.clone();
    let mut body = req.json()?;
    let path_index = if segs.len() >= 2 && segs[1] == op { Some(segs[0].clone()) } else { None };
    // A body ES rejects (`MultiGetRequest#add`: a top-level key other than
    // `docs` / `ids`, an unknown or mistyped element field) fails as a
    // whole before any shard is read: the local engine renders the error.
    if op == "_mget" && elasticsearch_http::mget_body::parse(&body, path_index.as_deref(), None).is_err() {
        return None;
    }
    if body.get("docs").is_none() {
        // `{"ids": [...]}` against `/{index}/_mget`: the engine renders
        // every malformed variant itself.
        path_index.as_ref()?;
        let ids = body.get("ids")?.as_array()?.clone();
        if ids.iter().any(|i| !i.is_string()) || body.as_object()?.len() != 1 {
            return None;
        }
        body = json!({ "docs": ids.into_iter().map(|i| json!({ "_id": i })).collect::<Vec<_>>() });
    }
    let docs = body.get("docs")?.as_array()?.clone();
    let params = req.params();
    let opts = elasticsearch_engine::node::ResolveOpts::from_params(&params);
    let exprs: Vec<Option<String>> = docs.iter().map(|d| d.get("_index").and_then(Value::as_str).map(str::to_string).or_else(|| path_index.clone())).collect();
    let engine = node.engine.clone();
    let names: Vec<Option<String>> = tokio::task::spawn_blocking(move || {
        exprs
            .iter()
            .map(|e| {
                let e = e.as_ref()?;
                let ts = engine.resolve(e, opts).ok()?;
                if ts.len() == 1 { Some(ts[0].svc.name.clone()) } else { None }
            })
            .collect()
    })
    .await
    .ok()?;
    let st = node.state();
    let local = node.local.id.clone();
    let scalar = |v: &Value| v.as_str().map(str::to_string).or_else(|| (v.is_number() || v.is_boolean()).then(|| v.to_string()));
    // (index, shard) of every doc; a routing error or an artificial doc
    // (no `_id`) goes to shard 0, whose engine renders / answers it.
    let keys: Vec<Option<(String, u32)>> = names
        .iter()
        .zip(&docs)
        .map(|(n, d)| {
            let n = n.as_ref()?;
            let k = match d.get("_id").and_then(|v| scalar(v)) {
                Some(id) => {
                    let routing = d.get("routing").or_else(|| d.get("_routing")).and_then(|v| scalar(v)).or_else(|| params.get("routing").cloned());
                    se::doc_shard(&st.metadata, n, &id, routing.as_deref()).unwrap_or(0)
                }
                None => 0,
            };
            Some((n.clone(), k))
        })
        .collect();
    let mut holders: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (n, k) in keys.iter().flatten() {
        let key = format!("{n}#{k}");
        if holders.contains_key(&key) {
            continue;
        }
        let h: BTreeSet<String> = match st.routing_table.shard_table(n, *k) {
            Some(t) => t.searchable_copies(&st.nodes).filter(|c| c.node_id.as_deref() != Some(local.as_str()) || super::local_holds(node, c)).filter_map(|c| c.node_id.clone()).collect(),
            None => BTreeSet::new(),
        };
        holders.insert(key, h);
    }
    let mut out = vec![Value::Null; docs.len()];
    // Docs of a shard with no active copy fail on their own, as in ES.
    for (pos, key) in keys.iter().enumerate() {
        if let Some((n, k)) = key {
            if holders.get(&format!("{n}#{k}")).map(|h| h.is_empty()).unwrap_or(false) {
                // A stateless shard without a search copy fails like an
                // unassigned shard in ES (`TransportShardMultiGetAction`).
                if st.routing_table.shard_table(n, *k).is_some_and(|t| t.stateless_without_search_copy(&st.nodes)) {
                    out[pos] = super::stateless_read::multi_doc_item(&op, n, docs[pos].get("_id").unwrap_or(&Value::Null));
                    continue;
                }
                let reason = format!("[{n}][{k}] no active shard copy");
                out[pos] = json!({
                    "_index": n,
                    "_id": docs[pos].get("_id").cloned().unwrap_or(Value::Null),
                    "error": {"root_cause": [{"type": "no_shard_available_action_exception", "reason": reason}], "type": "no_shard_available_action_exception", "reason": reason},
                });
            }
        }
    }
    // Stateless realtime gets need the indexing shard (ES
    // `TransportShardMultiGetAction.shardMultiGetFromTranslog`).
    if params.get("realtime").map(String::as_str) != Some("false") {
        for (pos, key) in keys.iter().enumerate() {
            if let (Some((n, k)), true) = (key, out[pos].is_null()) {
                if st.routing_table.shard_table(n, *k).is_some_and(|t| t.is_stateless(&st.nodes) && t.active_primary().is_none()) {
                    let uuid = st.metadata.indices.get(n).map(|m| m.uuid.clone()).unwrap_or_default();
                    out[pos] = json!({"_index": n, "_id": docs[pos].get("_id").cloned().unwrap_or(Value::Null), "error": super::stateless_read::primary_inactive(n, &uuid, *k)});
                }
            }
        }
    }
    let live: BTreeSet<String> = holders.iter().filter(|(_, h)| !h.is_empty()).map(|(k, _)| k.clone()).collect();
    let groups = cover(&live, &holders, &local)?;
    let owner: BTreeMap<&str, &str> = groups.iter().flat_map(|(n, ks)| ks.iter().map(move |k| (k.as_str(), n.as_str()))).collect();
    // (node, shard) → positions of its docs
    let mut per: BTreeMap<(String, u32), Vec<usize>> = BTreeMap::new();
    for (pos, key) in keys.iter().enumerate() {
        if !out[pos].is_null() {
            continue;
        }
        let (n, k) = match key {
            Some((name, k)) => (owner.get(format!("{name}#{k}").as_str()).copied().unwrap_or(local.as_str()), *k),
            None => (local.as_str(), 0),
        };
        per.entry((n.to_string(), k)).or_default().push(pos);
    }
    // The request for the docs at `positions` (all on shard `k`), reading
    // only copies readable on the answering node.
    let make = |positions: &[usize], k: u32| -> elasticsearch_engine::error::Result<super::RestReq> {
        let mut b = body.clone();
        b["docs"] = Value::Array(positions.iter().map(|p| docs[*p].clone()).collect());
        let mut sub = req.clone();
        sub.set_json(&b)?;
        sub.retain_headers(|(h, _)| !h.eq_ignore_ascii_case("content-length") && !h.eq_ignore_ascii_case("accept-encoding") && !h.eq_ignore_ascii_case("content-type"))?;
        sub.push_header("content-type", &req.json_content_type())?;
        sub.set_shard(k)?;
        let indices: Vec<String> = positions.iter().filter_map(|p| keys[*p].as_ref().map(|(i, _)| i.clone())).collect::<BTreeSet<_>>().into_iter().collect();
        if !indices.is_empty() {
            sub.set_read_copies(&indices)?;
        }
        Ok(sub)
    };
    let key_of = |p: usize| keys[p].as_ref().map(|(i, k)| format!("{i}#{k}"));
    // Each holder of `positions`' shard copies, this node first.
    let cands_for = |positions: &[usize], first: &str| -> Vec<String> {
        let shard_keys: BTreeSet<String> = positions.iter().filter_map(|p| key_of(*p)).collect();
        let mut cands = vec![first.to_string()];
        if let Some(h) = shard_keys.iter().next().and_then(|key| holders.get(key)) {
            let mut rest: Vec<String> = h.iter().filter(|c| c.as_str() != first && shard_keys.iter().all(|key| holders.get(key).is_some_and(|h| h.contains(*c)))).cloned().collect();
            rest.sort_by_key(|c| *c != local);
            cands.extend(rest);
        }
        cands
    };
    // Any failure but a request error moves on to the next copy (ES
    // `TransportShardMultiGetAction` retries the shard's other copies),
    // bounded by its node leaving the cluster.
    let attempt = |sub: elasticsearch_engine::error::Result<super::RestReq>, cands: Vec<String>| async move {
        let sub=match sub {Ok(s)=>s,Err(e)=>return super::RestResp::json(e.status,&e.to_json())};
        let mut last: Option<super::RestResp> = None;
        for c in cands {
            let r = super::copy_request(node, &sub, &c, None).await;
            if super::copies::answered(&r) || super::copies::is_request_error(&r) {
                return r;
            }
            if last.as_ref().map_or(true, |prev| super::copies::replaces(prev, &r)) {
                last = Some(r);
            }
        }
        last.unwrap_or_else(|| super::RestResp::error(503, "no_shard_available_action_exception", "no active shard copy"))
    };
    let here: &str = &local;
    let futs = per.into_iter().map(|((n, k), positions)| {
        let (make, cands_for, attempt, key_of) = (&make, &cands_for, &attempt, &key_of);
        async move {
            let r = attempt(make(&positions, k), cands_for(&positions, &n)).await;
            if super::copies::answered(&r) || super::copies::is_request_error(&r) {
                return vec![(positions, r)];
            }
            // No node answered for the whole group: each index of it on its
            // own copies (no node needs to hold all of them).
            let mut by_key: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for p in &positions {
                by_key.entry(key_of(*p).unwrap_or_default()).or_default().push(*p);
            }
            if by_key.len() < 2 {
                return vec![(positions, r)];
            }
            let singles = by_key.into_values().map(|ps| async move {
                let r = attempt(make(&ps, k), cands_for(&ps, here)).await;
                (ps, r)
            });
            futures::future::join_all(singles).await
        }
    });
    let mut raw_sources=elasticsearch_engine::source_json::ResponseSources::default();
    for (positions, r) in futures::future::join_all(futs).await.into_iter().flatten() {
        if super::copies::is_request_error(&r) {
            return Some(r);
        }
        if !r.is_success() {
            // ES: the docs of a shard no copy answered for fail one by one.
            let error = r.body_json().map(|v| v["error"].clone()).filter(Value::is_object)
                .unwrap_or_else(|| json!({"type": "no_shard_available_action_exception", "reason": "no active shard copy"}));
            for p in positions {
                let index = keys[p].as_ref().map(|(i, _)| json!(i)).unwrap_or(Value::Null);
                out[p] = json!({"_index": index, "_id": docs[p].get("_id").cloned().unwrap_or(Value::Null), "error": error});
            }
            continue;
        }
        let v = match raw_sources.parse(&r.body) {Ok(v)=>v,Err(_)=>return Some(r)};
        let got = v["docs"].as_array().cloned().unwrap_or_default();
        if got.len() != positions.len() {
            return Some(r);
        }
        for (p, d) in positions.into_iter().zip(got) {
            out[p] = d;
        }
    }
    let mut response=super::RestResp::json(200,&Value::Null);response.replace_body_vec(raw_sources.render(&json!({"docs":out})));
    Some(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn keyword_sort_values_compare_as_bytes_and_sentinels_only_against_numbers() {
        use serde_json::json;
        // Keyword values: "Infinity" is just a string, so the order is
        // total and transitive.
        let mut v = vec![json!("apple"), json!("Infinity"), json!("-Infinity"), json!("zebra"), json!("Apple")];
        v.sort_by(cmp_values);
        assert_eq!(v, vec![json!("-Infinity"), json!("Apple"), json!("Infinity"), json!("apple"), json!("zebra")]);
        assert_eq!(cmp_values(&json!("Infinity"), &json!("apple")), Ordering::Less);
        // A numeric key's missing sentinel against numbers.
        assert_eq!(cmp_values(&json!("Infinity"), &json!(1e300)), Ordering::Greater);
        assert_eq!(cmp_values(&json!(-5), &json!("-Infinity")), Ordering::Greater);
        assert_eq!(cmp_values(&json!("-Infinity"), &json!("Infinity")), Ordering::Less);
    }

    #[test]
    fn slice_split_per_shard() {
        let b = json!({"slice": {"id": "4", "max": 7, "field": "f"}, "query": {"match_all": {}}});
        // max >= n: slice 4 of 7 over 3 shards lives on shard 1 as 1 of 2.
        let s1 = slice_for_shard(&b, 1, 3);
        assert_eq!(s1["slice"], json!({"id": 1, "max": 2, "field": "f"}));
        let s0 = slice_for_shard(&b, 0, 3);
        assert!(s0.get("slice").is_none());
        assert_eq!(s0["query"]["bool"]["filter"], json!([{"match_none": {}}]));
        // max < n: slice 1 of 2 over 5 shards = all of shards 1 and 3.
        let c = json!({"slice": {"id": 1, "max": 2}});
        assert!(slice_for_shard(&c, 3, 5).get("query").is_none());
        assert!(slice_for_shard(&c, 3, 5).get("slice").is_none());
        assert!(slice_for_shard(&c, 2, 5).get("query").is_some());
        // Malformed slices go through untouched.
        let bad = json!({"slice": {"id": 3, "max": 2}});
        assert_eq!(slice_for_shard(&bad, 1, 3), bad);
    }

    fn set(v: &[&str]) -> BTreeSet<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn search_cover_spreads_shards_and_rotates_active_copies() {
        let keys = set(&["index#0", "index#1"]);
        let holders = keys.iter().map(|key| (key.clone(), set(&["n1", "n2"]))).collect();
        let first = search_cover(&keys, &holders, 0).unwrap();
        let next = search_cover(&keys, &holders, 1).unwrap();
        assert_eq!(first["n1"], vec!["index#0"]);
        assert_eq!(first["n2"], vec!["index#1"]);
        assert_eq!(next["n1"], vec!["index#1"]);
        assert_eq!(next["n2"], vec!["index#0"]);
        assert_eq!(first.values().map(Vec::len).sum::<usize>(),keys.len());
    }
    #[test]
    fn search_cover_never_invents_a_copy_and_handles_missing_shards() {
        let keys = set(&["a#0", "a#1", "b#0"]);
        let mut holders = BTreeMap::from([
            ("a#0".into(),set(&["n1"])), ("a#1".into(),set(&["n2"])),
            ("b#0".into(),set(&["n1","n3"])),
        ]);
        for rotation in [0,1,2,usize::MAX] {
            let groups = search_cover(&keys,&holders,rotation).unwrap();
            let selected: Vec<_> = groups.iter().flat_map(|(node, keys)| keys.iter().map(move |key| (node,key))).collect();
            assert_eq!(selected.len(),keys.len());
            assert_eq!(selected.iter().map(|(_,key)| (*key).clone()).collect::<BTreeSet<_>>(), keys);
            assert!(selected.iter().all(|(node,key)| holders[*key].contains(*node)));
        }
        holders.insert("a#1".into(),set(&[]));
        assert!(search_cover(&keys,&holders,0).is_none());
        holders.remove("a#1");
        assert!(search_cover(&keys,&holders,0).is_none());
    }

    #[test]
    fn cover_prefers_fewest_nodes_then_local() {
        let mut h = BTreeMap::new();
        h.insert("a".to_string(), set(&["n1", "n2"]));
        h.insert("b".to_string(), set(&["n2", "n3"]));
        h.insert("c".to_string(), set(&["n3"]));
        let c = cover(&set(&["a", "b", "c"]), &h, "n1").unwrap();
        let total: usize = c.values().map(Vec::len).sum();
        assert_eq!(total, 3);
        assert!(c.len() <= 2, "{c:?}");
        let mut h2 = BTreeMap::new();
        h2.insert("a".to_string(), set(&["n1", "n2"]));
        assert_eq!(cover(&set(&["a"]), &h2, "n2").unwrap().keys().next().unwrap(), "n2");
        let mut h3 = BTreeMap::new();
        h3.insert("a".to_string(), set(&[]));
        assert!(cover(&set(&["a"]), &h3, "n1").is_none());
    }

    #[test]
    fn normalizes_shard_engine_totals() {
        let part = |failed: u64| json!({"took": 1, "timed_out": false, "_shards": {"total": 3, "successful": 3 - failed, "skipped": 0, "failed": failed},
            "hits": {"total": {"value": 0, "relation": "eq"}, "max_score": null, "hits": []}});
        let (mut a, mut b) = (part(0), part(3));
        normalize_part_shards(&mut a, 1);
        normalize_part_shards(&mut b, 1);
        assert_eq!(b["_shards"], json!({"total": 1, "successful": 0, "skipped": 0, "failed": 1}));
        let m = merge_search(&[a, b], &json!({}), &HashMap::new());
        assert_eq!(m["_shards"]["total"], 2);
        assert_eq!(m["_shards"]["successful"], 1);
        let mut none = json!({"count": 1});
        normalize_part_shards(&mut none, 1);
        assert_eq!(none, json!({"count": 1}));
    }

    #[test]
    fn an_unsearched_shard_is_never_counted_successful() {
        // An engine that covered zero of the part's shards.
        let mut empty = json!({"_shards": {"total": 0, "successful": 0, "skipped": 0, "failed": 0}, "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}});
        normalize_part_shards(&mut empty, 2);
        assert_eq!(empty["_shards"]["total"], 2);
        assert_eq!(empty["_shards"]["successful"], 0);
        assert_eq!(empty["_shards"]["failed"], 2);
        assert_eq!(empty["_shards"]["failures"][0]["reason"]["type"], "shard_not_found_exception");
        let mut one = json!({"_shards": {"total": 1, "successful": 1, "skipped": 1, "failed": 0}});
        normalize_part_shards(&mut one, 2);
        assert_eq!(one["_shards"], json!({"total": 2, "successful": 1, "skipped": 1, "failed": 1, "failures": [{"shard": null, "index": null, "node": null, "reason": {"type": "shard_not_found_exception", "reason": "no such shard"}}]}));
    }

    #[test]
    fn merges_profiles_in_shard_order() {
        let r = |id: &str| json!({"hits": {"total": {"value": 1, "relation": "eq"}, "hits": []}, "profile": {"shards": [{"id": id, "searches": []}]}});
        let m = merge_search(&[r("a"), r("b")], &json!({"profile": true}), &HashMap::new());
        assert_eq!(m["profile"]["shards"].as_array().unwrap().iter().map(|s| s["id"].clone()).collect::<Vec<_>>(), vec![json!("a"), json!("b")]);
        assert!(merge_search(&[json!({"hits": {"hits": []}})], &json!({}), &HashMap::new()).get("profile").is_none());
    }

    #[test]
    fn caps_merged_total_at_track_total_hits() {
        let r = || json!({"took": 1, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0}, "hits": {"total": {"value": 10000, "relation": "gte"}, "hits": []}});
        let m = merge_search(&[r(), r()], &json!({}), &HashMap::new());
        assert_eq!(m["hits"]["total"], json!({"value": 10000, "relation": "gte"}));
        let m = merge_search(&[r(), r()], &json!({"track_total_hits": 15000}), &HashMap::new());
        assert_eq!(m["hits"]["total"], json!({"value": 15000, "relation": "gte"}));
        let m = merge_search(&[r(), r()], &json!({"track_total_hits": true}), &HashMap::new());
        assert_eq!(m["hits"]["total"], json!({"value": 20000, "relation": "gte"}));
        let small = || json!({"took": 1, "hits": {"total": {"value": 3, "relation": "eq"}, "hits": []}});
        let m = merge_search(&[small(), small()], &json!({}), &HashMap::new());
        assert_eq!(m["hits"]["total"], json!({"value": 6, "relation": "eq"}));
    }

    #[test]
    fn scroll_totals_stay_exact_above_the_default_limit() {
        let shard = |count| json!({"hits": {"total": {"value": count, "relation": "eq"}, "hits": []}});
        let parts = [shard(7_000), shard(8_007)];
        let params = HashMap::from([("scroll".to_string(), "10s".to_string())]);
        assert_eq!(merge_search(&parts, &json!({}), &params)["hits"]["total"], json!({"value": 15_007, "relation": "eq"}));
        assert_eq!(merge_search(&parts, &json!({}), &HashMap::new())["hits"]["total"], json!({"value": 10_000, "relation": "gte"}));
        let params = HashMap::from([("track_total_hits".to_string(), "true".to_string())]);
        assert_eq!(merge_search(&parts, &json!({"track_total_hits": 1}), &params)["hits"]["total"], json!({"value": 15_007, "relation": "eq"}));
        let params = HashMap::from([("track_total_hits".to_string(), "100".to_string())]);
        assert_eq!(merge_search(&parts, &json!({"track_total_hits": true}), &params)["hits"]["total"], json!({"value": 100, "relation": "gte"}));
        assert_eq!(merge_search(&parts, &json!({"track_total_hits": "100"}), &HashMap::new())["hits"]["total"], json!({"value": 100, "relation": "gte"}));
    }

    #[test]
    fn merges_by_score_with_from_size() {
        let r1 = json!({"took": 3, "timed_out": false, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": 2, "relation": "eq"}, "max_score": 2.0, "hits": [{"_index": "a", "_id": "1", "_score": 2.0}, {"_index": "a", "_id": "2", "_score": 0.5}]}});
        let r2 = json!({"took": 5, "timed_out": false, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": 1, "relation": "eq"}, "max_score": 1.0, "hits": [{"_index": "b", "_id": "9", "_score": 1.0}]}});
        let m = merge_search(&[r1.clone(), r2.clone()], &json!({"from": 1, "size": 1}), &HashMap::new());
        assert_eq!(m["took"], 5);
        assert_eq!(m["_shards"]["total"], 2);
        assert_eq!(m["hits"]["total"]["value"], 3);
        assert_eq!(m["hits"]["max_score"], 2.0);
        assert_eq!(m["hits"]["hits"][0]["_id"], "9");
        let mut p = HashMap::new();
        p.insert("rest_total_hits_as_int".to_string(), "true".to_string());
        assert_eq!(merge_search(&[r1, r2], &json!({}), &p)["hits"]["total"], 3);
    }

    #[test]
    fn merges_by_sort_values() {
        let r1 = json!({"hits": {"hits": [{"_index": "a", "_id": "1", "sort": [5]}, {"_index": "a", "_id": "2", "sort": [1]}]}});
        let r2 = json!({"hits": {"hits": [{"_index": "b", "_id": "3", "sort": [3]}, {"_index": "b", "_id": "4", "sort": [null]}]}});
        let m = merge_search(&[r1.clone(), r2.clone()], &json!({"sort": [{"n": "desc"}]}), &HashMap::new());
        let ids: Vec<&str> = m["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["1", "3", "2", "4"]);
        let m = merge_search(&[r1, r2], &json!({"sort": "n"}), &HashMap::new());
        let ids: Vec<&str> = m["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["2", "3", "1", "4"]);
    }

    /// Text fielddata sorts render terms as strings (like keyword sorts);
    /// they merge by string order, missing last or `_first`.
    #[test]
    fn merges_text_fielddata_sort_values() {
        let r1 = json!({"hits": {"hits": [{"_index": "t", "_id": "b", "sort": ["apple"]}, {"_index": "t", "_id": "c", "sort": ["mango"]}]}});
        let r2 = json!({"hits": {"hits": [{"_index": "t", "_id": "a", "sort": ["bar"]}, {"_index": "t", "_id": "d", "sort": [null]}]}});
        let ids = |m: Value| -> Vec<String> { m["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect() };
        assert_eq!(ids(merge_search(&[r1.clone(), r2.clone()], &json!({"sort": "foo"}), &HashMap::new())), ["b", "a", "c", "d"]);
        assert_eq!(ids(merge_search(&[r1.clone(), r2.clone()], &json!({"sort": [{"foo": {"missing": "_first"}}]}), &HashMap::new())), ["d", "b", "a", "c"]);
        let r1 = json!({"hits": {"hits": [{"_index": "t", "_id": "b", "sort": ["zebra"]}, {"_index": "t", "_id": "c", "sort": ["mango"]}]}});
        let r2 = json!({"hits": {"hits": [{"_index": "t", "_id": "a", "sort": ["quick"]}, {"_index": "t", "_id": "d", "sort": [null]}]}});
        assert_eq!(ids(merge_search(&[r1, r2], &json!({"sort": [{"foo": "desc"}]}), &HashMap::new())), ["b", "a", "c", "d"]);
    }

    #[test]
    fn rejects_what_needs_partial_results() {
        let none = HashMap::new();
        assert_eq!(unsupported_feature(&json!({"query": {}}), &none), None);
        assert_eq!(unsupported_feature(&json!({"aggs": {"t": {"terms": {"field": "k"}}}, "rescore": {}, "collapse": {"field": "k"}}), &none), None);
        assert_eq!(unsupported_feature(&json!({"aggs": {"p": {"percentiles": {"field": "k"}}}}), &none), None);
        assert_eq!(unsupported_feature(&json!({"aggs": {"p": {"boxplot": {"field": "k"}}}}), &none), None);
        assert_eq!(unsupported_feature(&json!({"aggs": {"p": {"no_such_aggregation": {"field": "k"}}}}), &none).is_some(), true);
        assert_eq!(unsupported_feature(&json!({"collapse": {"field": "k", "inner_hits": {"name": "x"}}}), &none), None);
        assert_eq!(
            unsupported_feature(&json!({"collapse": {"field": "k", "inner_hits": [{"name": "x"}]}, "knn": {"field": "v"}}), &none),
            Some("collapse with inner_hits combined with knn")
        );
        assert_eq!(unsupported_feature(&json!({"knn": {"field": "v", "query_vector": [1.0], "k": 3}}), &none), None);
        assert_eq!(unsupported_feature(&json!({"knn": {"field": "v"}, "query": {"match_all": {}}}), &none), None);
        assert_eq!(unsupported_feature(&json!({"knn": [{"field": "v", "inner_hits": {}}]}), &none), Some("knn with inner_hits"));
        assert_eq!(unsupported_feature(&json!({"pit": {"id": "x"}}), &none), Some("point in time"));
        let mut p = HashMap::new();
        p.insert("scroll".to_string(), "1m".to_string());
        assert_eq!(unsupported_feature(&json!({}), &p), Some("scroll"));
    }

    #[test]
    fn node_body_rewrites_aggs_and_merge_reduces_them() {
        let body = json!({"size": 0, "aggs": {"t": {"terms": {"field": "k", "size": 1}}, "a": {"avg": {"field": "v"}}}});
        let nb = node_body(&body, 0, 0);
        assert_eq!(nb["aggs"]["t"]["terms"]["size"], 11);
        assert!(nb["aggs"].get("__fanout_avg__a").is_some());
        let part = |k: &str, n: u64, sum: f64| json!({"hits": {"total": {"value": n, "relation": "eq"}, "hits": []}, "aggregations": {
            "t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 0, "buckets": [{"key": k, "doc_count": n}]},
            "a": {"value": sum / n as f64}, "__fanout_avg__a": {"count": n, "sum": sum}}});
        let m = merge_search(&[part("x", 3, 3.0), part("y", 1, 5.0)], &body, &HashMap::new());
        assert_eq!(m["aggregations"], json!({"t": {"doc_count_error_upper_bound": 0, "sum_other_doc_count": 1, "buckets": [{"key": "x", "doc_count": 3}]}, "a": {"value": 2.0}}));
    }

    #[test]
    fn collapse_inner_hits_helpers() {
        let one = json!({"collapse": {"field": "g", "inner_hits": {"name": "a"}}});
        let many = json!({"collapse": {"field": "g", "inner_hits": [{"name": "a"}, {"name": "b", "size": 1}]}});
        assert_eq!(collapse_inner_hits(&one).len(), 1);
        assert_eq!(collapse_inner_hits(&many).len(), 2);
        assert!(collapse_inner_hits(&json!({"collapse": {"field": "g"}})).is_empty());
        assert!(collapse_inner_hits_valid(&one) && collapse_inner_hits_valid(&many));
        assert!(!collapse_inner_hits_valid(&json!({"collapse": {"field": "g", "inner_hits": {}}})));
        assert!(!collapse_inner_hits_valid(&json!({"collapse": {"field": "g", "inner_hits": [{"name": "a"}, {"size": 1}]}})));
        assert!(!collapse_inner_hits_valid(&json!({"collapse": {"field": "g", "inner_hits": {"name": "a", "collapse": {"field": "n", "inner_hits": {}}}}})));
        assert!(!collapse_inner_hits_valid(&json!({"collapse": {"field": "g", "inner_hits": {"name": "a", "collapse": {"field": "n", "collapse": {}}}}})));
        assert!(collapse_inner_hits_valid(&json!({"collapse": {"field": "g", "inner_hits": {"name": "a", "collapse": {"field": "n"}}}})));
        let s = strip_inner_hits(&json!({"collapse": {"field": "g", "inner_hits": {"name": "a"}, "max_concurrent_group_searches": 4}, "size": 2}));
        assert_eq!(s, json!({"collapse": {"field": "g", "max_concurrent_group_searches": 4}, "size": 2}));
    }

    #[test]
    fn collapse_expand_body() {
        let none = HashMap::new();
        let body = json!({"query": {"term": {"t": "x"}}, "post_filter": {"range": {"n": {"gte": 3}}}, "runtime_mappings": {"r": {"type": "long"}},
                          "sort": ["n"], "from": 5, "size": 9, "aggs": {}});
        let e = json!({"name": "a", "size": 2, "from": 1, "sort": [{"n": "desc"}], "_source": ["n"], "collapse": {"field": "m"}, "version": true});
        let b = expand_body(&body, &none, "g", Some(&json!("k1")), &e);
        assert_eq!(
            b,
            json!({"query": {"bool": {"filter": [{"match": {"g": {"query": "k1"}}}], "must": [{"term": {"t": "x"}}]}},
                   "from": 1, "size": 2, "sort": [{"n": "desc"}], "_source": ["n"], "version": true, "collapse": {"field": "m"},
                   "post_filter": {"range": {"n": {"gte": 3}}}, "runtime_mappings": {"r": {"type": "long"}}})
        );
        // Missing value, no query, entry defaults.
        let b = expand_body(&json!({}), &none, "g", None, &json!({"name": "a"}));
        assert_eq!(b, json!({"query": {"bool": {"must_not": [{"exists": {"field": "g"}}]}}, "from": 0, "size": 3}));
        // `q` URL query wins over the body.
        let mut p = HashMap::new();
        p.insert("q".to_string(), "n:5".to_string());
        p.insert("df".to_string(), "t".to_string());
        let b = expand_body(&json!({}), &p, "g", Some(&json!(7)), &json!({"name": "a"}));
        assert_eq!(b["query"]["bool"]["must"], json!([{"query_string": {"query": "n:5", "default_field": "t"}}]));
        assert_eq!(b["query"]["bool"]["filter"], json!([{"match": {"g": {"query": 7}}}]));
    }

    #[test]
    fn collapse_attach_inner_hits() {
        let mut resp = json!({"hits": {"hits": [{"_id": "1", "fields": {"g": ["a"]}}, {"_id": "2"}]}});
        assert_eq!(collapse_value(&resp["hits"]["hits"][0], "g"), Some(&json!("a")));
        assert_eq!(collapse_value(&resp["hits"]["hits"][1], "g"), None);
        assert_eq!(collapse_value(&json!({"fields": {"g": []}}), "g"), None);
        let entries = vec![json!({"name": "x"}), json!({"name": "y"})];
        let r = |id: &str| json!({"took": 1, "hits": {"total": {"value": 1, "relation": "eq"}, "max_score": null, "hits": [{"_id": id}]}});
        attach_inner_hits(&mut resp, &entries, &[r("1x"), r("1y"), r("2x"), r("2y")]);
        assert_eq!(resp["hits"]["hits"][0]["inner_hits"]["x"]["hits"]["hits"][0]["_id"], "1x");
        assert_eq!(resp["hits"]["hits"][0]["inner_hits"]["y"]["hits"]["hits"][0]["_id"], "1y");
        assert_eq!(resp["hits"]["hits"][1]["inner_hits"]["y"]["hits"]["hits"][0]["_id"], "2y");
        assert!(resp["hits"]["hits"][1]["inner_hits"]["x"].get("took").is_none());
        assert!(resp["hits"]["hits"][1]["inner_hits"]["x"]["hits"]["total"].is_object());
    }

    #[test]
    fn collapse_keeps_best_hit_per_key() {
        let hit = |i: &str, id: &str, s: f64, k: Value| json!({"_index": i, "_id": id, "_score": s, "fields": {"k": [k]}});
        let r1 = json!({"hits": {"total": {"value": 3, "relation": "eq"}, "hits": [hit("a", "1", 3.0, json!("x")), hit("a", "2", 1.0, json!("y"))]}});
        let r2 = json!({"hits": {"total": {"value": 2, "relation": "eq"}, "hits": [hit("b", "3", 2.0, json!("y")), hit("b", "4", 0.5, json!("x"))]}});
        let m = merge_search(&[r1, r2], &json!({"collapse": {"field": "k"}}), &HashMap::new());
        let ids: Vec<&str> = m["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["1", "3"]);
        assert_eq!(m["hits"]["total"]["value"], 5);
    }

    #[test]
    fn knn_is_cut_to_global_k() {
        let r = |i: &str, s: &[f64]| json!({"hits": {"total": {"value": s.len(), "relation": "eq"}, "max_score": s[0],
            "hits": s.iter().map(|x| json!({"_index": i, "_id": x.to_string(), "_score": x})).collect::<Vec<_>>()}});
        let m = merge_search(&[r("a", &[0.9, 0.5]), r("b", &[0.8, 0.7])], &json!({"knn": {"field": "v", "query_vector": [1.0], "k": 2}, "size": 10}), &HashMap::new());
        assert_eq!(m["hits"]["total"]["value"], 2);
        assert_eq!(m["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_score"].as_f64().unwrap()).collect::<Vec<_>>(), vec![0.9, 0.8]);
    }

    #[test]
    fn knn_dfs_needed_only_when_hits_are_combined() {
        let none = HashMap::new();
        let k = json!({"field": "v", "query_vector": [1.0], "k": 3});
        assert!(!knn_needs_dfs(&json!({"knn": k}), &none));
        assert!(!knn_needs_dfs(&json!({"knn": k, "size": 5, "_source": false}), &none));
        assert!(knn_needs_dfs(&json!({"knn": [k]}), &none));
        assert!(knn_needs_dfs(&json!({"knn": k, "query": {"match_all": {}}}), &none));
        assert!(knn_needs_dfs(&json!({"knn": k, "aggs": {}}), &none));
        let q: HashMap<String, String> = [("q".to_string(), "x".to_string())].into();
        assert!(knn_needs_dfs(&json!({"knn": k}), &q));
        assert!(!knn_needs_dfs(&json!({"query": {}}), &q));
    }

    #[test]
    fn knn_dfs_body_asks_for_k() {
        let b = knn_dfs_body(&json!({"field": "v", "query_vector": [1.0], "boost": 2.0, "_name": "n"}), 7);
        assert_eq!(b, json!({"knn": {"field": "v", "query_vector": [1.0], "k": 7}, "from": 0, "size": 7, "_source": false}));
        assert_eq!(knn_dfs_body(&json!({"field": "v", "k": 3}), 10)["size"], 3);
    }

    #[test]
    fn knn_rewrite_shoulds_query_with_scored_top_docs() {
        let body = json!({"knn": [{"field": "v", "k": 2, "boost": 0.5, "_name": "a"}, {"field": "w", "k": 1}], "query": {"term": {"t": "x"}}, "size": 3});
        let top = vec![vec![("i".to_string(), "1".to_string(), None, 0.9), ("i".to_string(), "1".to_string(), Some("r".to_string()), 0.4)], vec![]];
        let r = knn_rewrite(&body, &HashMap::new(), &top);
        assert!(r.get("knn").is_none());
        assert_eq!(r["size"], 3);
        let should = r["query"]["bool"]["should"].as_array().unwrap();
        assert_eq!(should[0], json!({"term": {"t": "x"}}));
        assert_eq!(should[1]["bool"]["boost"], 0.5);
        assert_eq!(should[1]["bool"]["_name"], "a");
        assert_eq!(should[1]["bool"]["should"][0]["constant_score"]["boost"], 0.9);
        assert_eq!(should[1]["bool"]["should"][0]["constant_score"]["filter"]["bool"]["filter"][1], json!({"ids": {"values": ["1"]}}));
        // Same id, another routing: told apart by `_routing`.
        assert_eq!(should[1]["bool"]["should"][0]["constant_score"]["filter"]["bool"]["filter"][2], json!({"bool": {"must_not": [{"exists": {"field": "_routing"}}]}}));
        assert_eq!(should[1]["bool"]["should"][1]["constant_score"]["filter"]["bool"]["filter"][2], json!({"term": {"_routing": "r"}}));
        assert_eq!(should[2], json!({"match_none": {}}));
        let p: HashMap<String, String> = [("q".to_string(), "x".to_string()), ("df".to_string(), "t".to_string()), ("lenient".to_string(), "true".to_string())].into();
        let r = knn_rewrite(&json!({"knn": {"field": "v"}}), &p, &[vec![]]);
        assert_eq!(r["query"]["bool"]["should"][0], json!({"query_string": {"query": "x", "default_field": "t", "lenient": true}}));
    }

    #[test]
    fn suggest_options_merge() {
        let body = json!({"suggest": {"s": {"text": "helo", "term": {"field": "t", "size": 2}}}});
        let r = |opts: Value| json!({"hits": {"hits": []}, "suggest": {"s": [{"text": "helo", "offset": 0, "length": 4, "options": opts}]}});
        let a = r(json!([{"text": "hello", "score": 0.75, "freq": 2}, {"text": "help", "score": 0.75, "freq": 1}]));
        let b = r(json!([{"text": "help", "score": 0.75, "freq": 4}, {"text": "held", "score": 0.5, "freq": 9}]));
        let m = merge_search(&[a, b], &body, &HashMap::new());
        assert_eq!(m["suggest"]["s"][0]["options"], json!([{"text": "help", "score": 0.75, "freq": 5}, {"text": "hello", "score": 0.75, "freq": 2}]));
    }

    #[test]
    fn counts_sum() {
        let m = merge_count(&[json!({"count": 2, "_shards": {"total": 1, "successful": 1}}), json!({"count": 3, "_shards": {"total": 1, "successful": 1}})]);
        assert_eq!(m["count"], 5);
        assert_eq!(m["_shards"]["total"], 2);
    }
}

// -------------------------------------------------------------- field caps

/// Per (field, type) accumulator for [`merge_field_caps`].
#[derive(Default)]
struct CapsAcc {
    metadata_field: bool,
    /// `(index, searchable, aggregatable, dimension, metric)` per index.
    per_index: Vec<(String, bool, bool, bool, Option<String>)>,
    meta: BTreeMap<String, BTreeSet<String>>,
}

/// Merge per-index `_field_caps` responses the way ES's
/// `FieldCapabilities.Builder` does: capabilities are ANDed across the
/// indices of one type, `indices` is listed per type only when a field has
/// more than one type (counting `unmapped`), and the `non_*_indices` lists
/// appear only when the indices of a type disagree.
pub fn merge_field_caps(parts: &[(String, Value)], include_unmapped: bool) -> Value {
    let mut indices: BTreeSet<String> = BTreeSet::new();
    let mut fields: BTreeMap<String, BTreeMap<String, CapsAcc>> = BTreeMap::new();
    for (index, body) in parts {
        // An `index_filter` that excludes the index answers with no indices.
        let listed = body.get("indices").and_then(Value::as_array).map(|a| a.iter().any(|v| v.as_str() == Some(index))).unwrap_or(false);
        if !listed {
            continue;
        }
        indices.insert(index.clone());
        let Some(fs) = body.get("fields").and_then(Value::as_object) else { continue };
        for (field, types) in fs {
            let Some(types) = types.as_object() else { continue };
            for (ty, caps) in types {
                if ty == "unmapped" {
                    continue;
                }
                let acc = fields.entry(field.clone()).or_default().entry(ty.clone()).or_default();
                let flag = |k: &str| caps.get(k).and_then(Value::as_bool).unwrap_or(false);
                acc.metadata_field = flag("metadata_field");
                acc.per_index.push((index.clone(), flag("searchable"), flag("aggregatable"), flag("time_series_dimension"), caps.get("time_series_metric").and_then(Value::as_str).map(str::to_string)));
                if let Some(m) = caps.get("meta").and_then(Value::as_object) {
                    for (k, v) in m {
                        let set = acc.meta.entry(k.clone()).or_default();
                        match v {
                            Value::Array(a) => set.extend(a.iter().filter_map(Value::as_str).map(str::to_string)),
                            Value::String(s) => {
                                set.insert(s.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    let mut out_fields = Map::new();
    for (field, types) in fields {
        let mapped_in: BTreeSet<&String> = types.values().flat_map(|a| a.per_index.iter().map(|p| &p.0)).collect();
        let unmapped: Vec<&String> = if include_unmapped { indices.iter().filter(|i| !mapped_in.contains(i)).collect() } else { Vec::new() };
        let multi = types.len() + usize::from(!unmapped.is_empty()) > 1;
        let mut out_types = Map::new();
        for (ty, mut acc) in types {
            acc.per_index.sort_by(|a, b| a.0.cmp(&b.0));
            let mut o = Map::new();
            o.insert("type".into(), json!(ty));
            o.insert("metadata_field".into(), json!(acc.metadata_field));
            let searchable = acc.per_index.iter().all(|p| p.1);
            let aggregatable = acc.per_index.iter().all(|p| p.2);
            o.insert("searchable".into(), json!(searchable));
            o.insert("aggregatable".into(), json!(aggregatable));
            let dimension = acc.per_index.iter().all(|p| p.3);
            if dimension {
                o.insert("time_series_dimension".into(), json!(true));
            }
            let first_metric = acc.per_index.first().and_then(|p| p.4.clone());
            let metric_agrees = acc.per_index.iter().all(|p| p.4 == first_metric);
            if metric_agrees {
                if let Some(m) = &first_metric {
                    o.insert("time_series_metric".into(), json!(m));
                }
            }
            let all: Vec<&String> = acc.per_index.iter().map(|p| &p.0).collect();
            if multi {
                o.insert("indices".into(), json!(all));
            }
            if !searchable && acc.per_index.iter().any(|p| p.1) {
                o.insert("non_searchable_indices".into(), json!(acc.per_index.iter().filter(|p| !p.1).map(|p| &p.0).collect::<Vec<_>>()));
            }
            if !aggregatable && acc.per_index.iter().any(|p| p.2) {
                o.insert("non_aggregatable_indices".into(), json!(acc.per_index.iter().filter(|p| !p.2).map(|p| &p.0).collect::<Vec<_>>()));
            }
            if !dimension && acc.per_index.iter().any(|p| p.3) {
                o.insert("non_dimension_indices".into(), json!(acc.per_index.iter().filter(|p| !p.3).map(|p| &p.0).collect::<Vec<_>>()));
            }
            if !metric_agrees {
                o.insert("metric_conflicts_indices".into(), json!(all));
            }
            if !acc.meta.is_empty() {
                let meta: Map<String, Value> = acc.meta.iter().map(|(k, v)| (k.clone(), json!(v.iter().collect::<Vec<_>>()))).collect();
                o.insert("meta".into(), Value::Object(meta));
            }
            out_types.insert(ty, Value::Object(o));
        }
        if !unmapped.is_empty() {
            out_types.insert("unmapped".into(), json!({"type": "unmapped", "metadata_field": false, "searchable": false, "aggregatable": false, "indices": unmapped}));
        }
        out_fields.insert(field, Value::Object(out_types));
    }
    json!({"indices": indices.iter().collect::<Vec<_>>(), "fields": out_fields})
}

#[cfg(test)]
mod field_caps_tests {
    use super::*;

    #[test]
    fn merges_types_and_flags() {
        let a = json!({"indices": ["a"], "fields": {"f": {"keyword": {"type": "keyword", "metadata_field": false, "searchable": true, "aggregatable": true}}, "n": {"long": {"type": "long", "metadata_field": false, "searchable": true, "aggregatable": true}}}});
        let b = json!({"indices": ["b"], "fields": {"f": {"text": {"type": "text", "metadata_field": false, "searchable": true, "aggregatable": false}}, "n": {"long": {"type": "long", "metadata_field": false, "searchable": false, "aggregatable": true}}}});
        let c = json!({"indices": [], "fields": {}});
        let m = merge_field_caps(&[("a".into(), a), ("b".into(), b), ("c".into(), c)], false);
        assert_eq!(m["indices"], json!(["a", "b"]));
        assert_eq!(m["fields"]["f"]["keyword"]["indices"], json!(["a"]));
        assert_eq!(m["fields"]["f"]["text"]["indices"], json!(["b"]));
        assert_eq!(m["fields"]["n"]["long"]["searchable"], json!(false));
        assert_eq!(m["fields"]["n"]["long"]["non_searchable_indices"], json!(["b"]));
        assert!(m["fields"]["n"]["long"].get("indices").is_none());
        assert!(m["fields"]["n"]["long"].get("non_aggregatable_indices").is_none());
    }

    #[test]
    fn unmapped_entries() {
        let a = json!({"indices": ["a"], "fields": {"f": {"keyword": {"type": "keyword", "metadata_field": false, "searchable": true, "aggregatable": true}}}});
        let b = json!({"indices": ["b"], "fields": {}});
        let m = merge_field_caps(&[("a".into(), a), ("b".into(), b)], true);
        assert_eq!(m["fields"]["f"]["keyword"]["indices"], json!(["a"]));
        assert_eq!(m["fields"]["f"]["unmapped"]["indices"], json!(["b"]));
    }
}


/// Add the numbers of `from` into `into` (index stats sections). ES takes
/// the max of `max_unsafe_auto_id_timestamp` and the min of
/// `earliest_last_modified_age`; everything else numeric is summed.
fn add_stats(into: &mut Value, from: &Value, key: &str) {
    match (into, from) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                match a.get_mut(k) {
                    Some(slot) => add_stats(slot, v, k),
                    None => {
                        a.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (slot @ Value::Number(_), Value::Number(n)) => {
            let (x, y) = (slot.as_i64(), n.as_i64());
            *slot = match (key, x, y) {
                ("max_unsafe_auto_id_timestamp", Some(x), Some(y)) => json!(x.max(y)),
                ("earliest_last_modified_age", Some(x), Some(y)) => json!(x.min(y)),
                (_, Some(x), Some(y)) => json!(x + y),
                _ => json!(slot.as_f64().unwrap_or(0.0) + n.as_f64().unwrap_or(0.0)),
            };
        }
        _ => {}
    }
}

/// Merge `GET /{indices}/_stats` responses from nodes that each covered a
/// disjoint set of whole indices: per-index entries are taken as they are,
/// `_all` and `_shards` are summed.
pub fn merge_index_stats(parts: &[Value]) -> Value {
    let mut shards = json!({"total": 0, "successful": 0, "failed": 0});
    let mut all = json!({"primaries": {}, "total": {}});
    let mut indices = Map::new();
    for p in parts {
        add_stats(&mut shards, &p["_shards"], "_shards");
        add_stats(&mut all, &p["_all"], "_all");
        if let Some(m) = p["indices"].as_object() {
            for (k, v) in m {
                indices.insert(k.clone(), v.clone());
            }
        }
    }
    let mut out = Map::new();
    out.insert("_shards".into(), shards);
    out.insert("_all".into(), all);
    if parts.iter().any(|p| p.get("indices").is_some()) {
        out.insert("indices".into(), Value::Object(indices));
    }
    Value::Object(out)
}

#[cfg(test)]
mod index_stats_merge_tests {
    use super::*;

    #[test]
    fn whole_indices_from_different_nodes_merge_like_es() {
        let a = json!({"_shards": {"total": 2, "successful": 2, "failed": 0},
            "_all": {"primaries": {"docs": {"count": 3}, "dense_vector": {"value_count": 3}, "indexing": {"max_unsafe_auto_id_timestamp": 5}}, "total": {"docs": {"count": 6}}},
            "indices": {"a": {"uuid": "ua", "primaries": {"docs": {"count": 3}}}}});
        let b = json!({"_shards": {"total": 1, "successful": 1, "failed": 0},
            "_all": {"primaries": {"docs": {"count": 4}, "dense_vector": {"value_count": 4}, "indexing": {"max_unsafe_auto_id_timestamp": 9}}, "total": {"docs": {"count": 4}}},
            "indices": {"b": {"uuid": "ub", "primaries": {"docs": {"count": 4}}}}});
        let m = merge_index_stats(&[a, b]);
        assert_eq!(m["_shards"], json!({"total": 3, "successful": 3, "failed": 0}));
        assert_eq!(m["_all"]["primaries"]["docs"]["count"], 7);
        assert_eq!(m["_all"]["primaries"]["dense_vector"]["value_count"], 7);
        assert_eq!(m["_all"]["primaries"]["indexing"]["max_unsafe_auto_id_timestamp"], 9);
        assert_eq!(m["_all"]["total"]["docs"]["count"], 10);
        assert_eq!(m["indices"]["a"]["uuid"], "ua");
        assert_eq!(m["indices"]["b"]["primaries"]["docs"]["count"], 4);
    }
}

#[cfg(test)]
mod rank_plan_merge_tests {
    use super::*;

    #[test]
    fn score_ties_keep_the_fused_rank_order() {
        let body = json!({"size": 3, "_rank_plan": {"docs": {}, "union": null, "order": [["i", "b"], ["i", "a"], ["i", "c"]]}});
        let a = json!({"hits": {"total": {"value": 2, "relation": "eq"}, "hits": [{"_index": "i", "_id": "a", "_score": 0.5}, {"_index": "i", "_id": "c", "_score": 0.1}]}});
        let b = json!({"hits": {"total": {"value": 1, "relation": "eq"}, "hits": [{"_index": "i", "_id": "b", "_score": 0.5}]}});
        let m = merge_search(&[a, b], &body, &Default::default());
        let ids: Vec<&str> = m["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["b", "a", "c"]);
    }
}

#[cfg(test)]
mod skipped_merge_tests {
    use super::*;

    #[test]
    fn an_aggregation_search_keeps_one_shard() {
        let part = |skipped: u64| json!({"_shards": {"total": 1, "successful": 1, "skipped": skipped, "failed": 0}, "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}});
        let aggs = json!({"size": 0, "aggs": {"c": {"value_count": {"field": "f"}}}});
        assert_eq!(merge_search(&[part(1), part(1)], &aggs, &Default::default())["_shards"]["skipped"], 1);
        assert_eq!(merge_search(&[part(1), part(0)], &aggs, &Default::default())["_shards"]["skipped"], 1);
        assert_eq!(merge_search(&[part(1), part(1)], &json!({"size": 0}), &Default::default())["_shards"]["skipped"], 2);
    }
}
