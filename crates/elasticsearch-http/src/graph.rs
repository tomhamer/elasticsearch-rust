//! Graph explore API (`/{index}/_graph/explore`), following x-pack's
//! `TransportGraphExploreAction`.
//!
//! Each hop is one search: the hop's guiding query (AND the include terms of
//! its vertex requests, AND, after the first hop, an OR over the vertices the
//! previous hop found, each boosted by its weight) feeds a `sampler` (or
//! `diversified_sampler`) of the best `sample_size` docs per shard. Under the
//! sampler, the first hop runs one `significant_terms` (or `terms` without
//! significance) agg per vertex field; later hops run a `terms` agg over the
//! previous hop's vertices per field, each with the next hop's
//! `significant_terms` / `terms` aggs beneath it. Vertex weights are the
//! bucket's share of the hop's total signal (significance score or doc
//! count), decayed by the source vertex's weight (at most 0.95) after the
//! first hop; connections carry that signal and the co-occurrence doc count.

use crate::{method_not_allowed, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::node::{Node, ResolveOpts};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// `IndexSearcher.getMaxClauseCount()`: above this many vertex terms a hop
/// ORs plain `terms` queries per field instead of boosted `term` queries.
const MAX_CLAUSES: usize = 1024;

pub async fn handle(node: Arc<Node>, req: Req, index: String) -> R {
    if req.seg(2) != "explore" || req.segs.len() != 3 {
        return crate::not_found_handler(&req);
    }
    if !matches!(req.method, Method::GET | Method::POST) {
        return method_not_allowed(&req, &["GET", "POST"]);
    }
    crate::util::check_params(&req, &["routing", "timeout"])?;
    crate::blocking(move || explore(&node, &req, &index)).await
}

#[derive(Clone, Debug)]
struct VertexRequest {
    field: String,
    size: usize,
    min_doc_count: u64,
    shard_min_doc_count: u64,
    /// `(term, boost)`
    include: Vec<(String, f32)>,
    exclude: Vec<String>,
}

#[derive(Clone, Debug)]
struct Hop {
    query: Option<Value>,
    vertices: Vec<VertexRequest>,
}

struct Controls {
    use_significance: bool,
    sample_size: usize,
    diversity: Option<(String, u64)>,
    timeout: Option<String>,
    detailed: bool,
}

#[derive(Clone, Debug)]
struct Vertex {
    field: String,
    term: String,
    weight: f64,
    depth: usize,
    bg: u64,
    fg: u64,
    removed: bool,
}

struct Connection {
    from: usize,
    to: usize,
    weight: f64,
    doc_count: u64,
}

fn parse_err(msg: impl Into<String>) -> EsError {
    EsError::new(400, "parse_exception", msg)
}

fn parse_vertices(v: &Value) -> Result<Vec<VertexRequest>> {
    let arr = v.as_array().ok_or_else(|| parse_err("Illegal value for vertices: expected an array"))?;
    let mut out = Vec::new();
    for vr in arr {
        let o = vr.as_object().ok_or_else(|| parse_err("Illegal vertex definition: expected an object"))?;
        let mut r = VertexRequest { field: String::new(), size: 10, min_doc_count: 3, shard_min_doc_count: 2, include: Vec::new(), exclude: Vec::new() };
        for (k, val) in o {
            match k.as_str() {
                "field" => r.field = val.as_str().ok_or_else(|| parse_err("[field] must be a string"))?.to_string(),
                "size" => r.size = num(val, "size")? as usize,
                "min_doc_count" => r.min_doc_count = num(val, "min_doc_count")?,
                "shard_min_doc_count" => r.shard_min_doc_count = num(val, "shard_min_doc_count")?,
                "include" => {
                    for t in val.as_array().ok_or_else(|| parse_err("Illegal value for include: expected an array"))? {
                        match t {
                            Value::Object(tb) => {
                                let term = tb.get("term").map(scalar).ok_or_else(|| parse_err("Expected a term in include"))?;
                                let boost = tb.get("boost").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                                r.include.push((term, boost));
                            }
                            other => r.include.push((scalar(other), 1.0)),
                        }
                    }
                }
                "exclude" => {
                    for t in val.as_array().ok_or_else(|| parse_err("Illegal value for exclude: expected an array"))? {
                        r.exclude.push(scalar(t));
                    }
                }
                other => {
                    return Err(parse_err(match val {
                        Value::Number(_) => format!("Unknown numeric property: [{other}]"),
                        Value::String(_) => format!("Unknown string property: [{other}]"),
                        Value::Array(_) => format!("Illegal property in graph vertices definition {other}"),
                        _ => format!("Unknown property: [{other}]"),
                    }))
                }
            }
            if !r.include.is_empty() && !r.exclude.is_empty() {
                return Err(parse_err("Graph vertices definition cannot contain both include and exclude clauses"));
            }
        }
        if r.field.is_empty() {
            return Err(parse_err("Missing [field] in vertex definition"));
        }
        out.push(r);
    }
    Ok(out)
}

fn num(v: &Value, name: &str) -> Result<u64> {
    match v {
        Value::Number(n) => n.as_u64().or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
    .ok_or_else(|| parse_err(format!("Illegal value for [{name}]")))
}

fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The hop chain: the root hop (`query`, `vertices`) followed by nested
/// `connections`.
fn parse_hops(body: &Value, hops: &mut Vec<Hop>) -> Result<()> {
    let Some(o) = body.as_object() else { return Ok(()) };
    let mut hop = Hop { query: None, vertices: Vec::new() };
    let mut next = None;
    for (k, v) in o {
        match k.as_str() {
            "query" => hop.query = Some(v.clone()),
            "vertices" => hop.vertices = parse_vertices(v)?,
            "connections" => next = Some(v),
            "controls" if hops.is_empty() => {}
            other => return Err(parse_err(format!("Illegal property in graph definition {other}"))),
        }
    }
    hops.push(hop);
    if let Some(n) = next {
        parse_hops(n, hops)?;
    }
    Ok(())
}

fn parse_controls(body: &Value) -> Result<Controls> {
    let mut c = Controls { use_significance: true, sample_size: 100, diversity: None, timeout: None, detailed: false };
    let Some(o) = body.get("controls") else { return Ok(c) };
    let o = o.as_object().ok_or_else(|| parse_err("[controls] must be an object"))?;
    for (k, v) in o {
        match k.as_str() {
            "use_significance" => c.use_significance = v.as_bool().ok_or_else(|| parse_err("[use_significance] must be a boolean"))?,
            "sample_size" => c.sample_size = num(v, "sample_size")? as usize,
            "timeout" => c.timeout = Some(scalar(v)),
            "return_detailed_stats" => c.detailed = v.as_bool().unwrap_or(false),
            "sample_diversity" => {
                let field = v.get("field").and_then(Value::as_str).ok_or_else(|| parse_err("[sample_diversity] requires a [field]"))?;
                let max = v.get("max_docs_per_value").map(|m| num(m, "max_docs_per_value")).transpose()?.unwrap_or(1);
                c.diversity = Some((field.to_string(), max));
            }
            other => return Err(parse_err(format!("Unknown control property: [{other}]"))),
        }
    }
    if c.sample_size == 0 {
        return Err(EsError::illegal_argument("sample_size must be at least 1"));
    }
    Ok(c)
}

struct Graph {
    vertices: Vec<Vertex>,
    by_key: HashMap<(String, String), usize>,
    connections: Vec<(Connection, bool)>,
    conn_index: HashMap<(usize, usize), usize>,
    /// Per hop: field -> vertices found on that hop (insertion order).
    findings: Vec<Vec<(String, Vec<usize>)>>,
}

impl Graph {
    fn get(&self, field: &str, term: &str) -> Option<usize> {
        self.by_key.get(&(field.to_string(), term.to_string())).copied().filter(|&i| !self.vertices[i].removed)
    }

    fn add_vertex(&mut self, field: &str, term: &str, weight: f64, depth: usize, bg: u64, fg: u64) -> usize {
        let i = self.vertices.len();
        self.vertices.push(Vertex { field: field.into(), term: term.into(), weight, depth, bg, fg, removed: false });
        self.by_key.insert((field.into(), term.into()), i);
        while self.findings.len() <= depth {
            self.findings.push(Vec::new());
        }
        let hop = &mut self.findings[depth];
        match hop.iter_mut().find(|(f, _)| f == field) {
            Some((_, v)) => v.push(i),
            None => hop.push((field.to_string(), vec![i])),
        }
        i
    }

    fn remove_vertex(&mut self, i: usize) {
        let v = &mut self.vertices[i];
        v.removed = true;
        let (field, depth) = (v.field.clone(), v.depth);
        self.by_key.remove(&(field.clone(), v.term.clone()));
        if let Some(hop) = self.findings.get_mut(depth) {
            if let Some((_, list)) = hop.iter_mut().find(|(f, _)| *f == field) {
                list.retain(|&x| x != i);
            }
            hop.retain(|(_, l)| !l.is_empty());
        }
    }

    fn add_connection(&mut self, from: usize, to: usize, weight: f64, doc_count: u64) -> usize {
        let c = Connection { from, to, weight, doc_count };
        match self.conn_index.get(&(from, to)) {
            Some(&i) => {
                self.connections[i] = (c, true);
                i
            }
            None => {
                self.connections.push((c, true));
                let i = self.connections.len() - 1;
                self.conn_index.insert((from, to), i);
                i
            }
        }
    }
}

/// Bucket key as Java's `getKeyAsString`.
fn key_string(b: &Value) -> String {
    if let Some(s) = b.get("key_as_string").and_then(Value::as_str) {
        return s.to_string();
    }
    match b.get("key") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn buckets<'a>(agg: Option<&'a Value>) -> Option<&'a Vec<Value>> {
    agg?.get("buckets")?.as_array()
}

fn bucket_signal(b: &Value, sig: bool) -> f64 {
    if sig {
        b.get("score").and_then(Value::as_f64).unwrap_or(0.0)
    } else {
        b.get("doc_count").and_then(Value::as_f64).unwrap_or(0.0)
    }
}

fn u(b: &Value, k: &str) -> u64 {
    b.get(k).and_then(Value::as_u64).unwrap_or(0)
}

/// The per-field agg of a vertex request (`significant_terms` or `terms`).
fn field_agg(vr: &VertexRequest, c: &Controls, root: bool, self_loop: bool) -> Value {
    let mut size = vr.size + usize::from(self_loop);
    let mut a = Map::new();
    a.insert("field".into(), json!(vr.field));
    a.insert("execution_hint".into(), json!("map"));
    if c.use_significance || !root {
        a.insert("min_doc_count".into(), json!(vr.min_doc_count));
        a.insert("shard_min_doc_count".into(), json!(vr.shard_min_doc_count));
    }
    if !vr.include.is_empty() {
        let mut inc: Vec<String> = vr.include.iter().map(|(t, _)| t.clone()).collect();
        inc.sort();
        inc.dedup();
        if root {
            size = inc.len();
        }
        a.insert("include".into(), json!(inc));
    } else if !vr.exclude.is_empty() {
        let mut exc = vr.exclude.clone();
        exc.sort();
        exc.dedup();
        a.insert("exclude".into(), json!(exc));
    }
    a.insert("size".into(), json!(size));
    if c.use_significance {
        if !root && size < 10 {
            a.insert("shard_size".into(), json!(10));
        }
        json!({"significant_terms": a})
    } else {
        json!({"terms": a})
    }
}

fn sampler(c: &Controls, aggs: Map<String, Value>) -> Value {
    let s = match &c.diversity {
        Some((f, max)) => json!({"diversified_sampler": {"shard_size": c.sample_size, "field": f, "max_docs_per_value": max}}),
        None => json!({"sampler": {"shard_size": c.sample_size}}),
    };
    let mut s = s.as_object().cloned().unwrap_or_default();
    s.insert("aggs".into(), Value::Object(aggs));
    Value::Object(s)
}

/// ES `addNormalizedBoosts` / `addUserDefinedIncludesToQuery`: the include
/// terms of a hop, boosts rebased so the smallest is 1.
fn include_clauses(hop: &Hop, should: &mut Vec<Value>) {
    for vr in &hop.vertices {
        if vr.include.is_empty() {
            continue;
        }
        if should.len() + vr.include.len() > MAX_CLAUSES {
            let terms: Vec<&String> = vr.include.iter().map(|(t, _)| t).collect();
            should.push(json!({"constant_score": {"filter": {"terms": {vr.field.clone(): terms}}}}));
            continue;
        }
        let min = vr.include.iter().map(|(_, b)| *b).fold(f32::MAX, f32::min);
        for (t, b) in &vr.include {
            should.push(json!({"term": {vr.field.clone(): {"value": t, "boost": b / min}}}));
        }
    }
}

fn explore(node: &Node, req: &Req, index: &str) -> R {
    let start = std::time::Instant::now();
    let body = req.json()?;
    if body.is_null() {
        return Err(parse_err("Body missing for graph request"));
    }
    if !body.is_object() {
        return Err(parse_err("Graph explore request body must be an object"));
    }
    elasticsearch_engine::license::check(node, "graph", elasticsearch_engine::license::Mode::Platinum)?;
    let mut controls = parse_controls(&body)?;
    if let Some(t) = req.param("timeout") {
        controls.timeout = Some(t.to_string());
    }
    let mut hops = Vec::new();
    parse_hops(&body, &mut hops)?;
    if hops.iter().any(|h| h.vertices.is_empty()) {
        return Err(EsError::action_validation(&["Graph explore hop must have at least one VertexRequest"]));
    }
    let detailed = controls.detailed;
    let mut params: HashMap<String, String> = HashMap::new();
    if let Some(r) = req.param("routing") {
        params.insert("routing".into(), r.to_string());
    }
    let opts = ResolveOpts::from_params(&req.params);
    let mut g = Graph { vertices: Vec::new(), by_key: HashMap::new(), connections: Vec::new(), conn_index: HashMap::new(), findings: Vec::new() };
    let mut timed_out = false;
    let mut failures: Vec<Value> = Vec::new();
    let run = |q: Value, aggs: Value, timed_out: &mut bool, failures: &mut Vec<Value>| -> Result<Value> {
        let mut src = json!({"size": 0, "query": q, "aggs": {"sample": aggs}});
        if let Some(t) = &controls.timeout {
            src["timeout"] = json!(t);
        }
        let out = crate::search_api::coordinated_search(node, req, &src, &params, index, opts, req.access.as_deref())?;
        let v: Value = serde_json::from_str(&out).map_err(|e| EsError::internal(e.to_string()))?;
        *timed_out |= v["timed_out"].as_bool().unwrap_or(false);
        if let Some(f) = v["_shards"]["failures"].as_array() {
            failures.extend(f.iter().cloned());
        }
        Ok(v["aggregations"]["sample"].clone())
    };

    // Root hop.
    let root = &hops[0];
    let mut must = vec![root.query.clone().unwrap_or_else(|| json!({"match_all": {}}))];
    let mut inc = Vec::new();
    include_clauses(root, &mut inc);
    if !inc.is_empty() {
        must.push(json!({"bool": {"should": inc}}));
    }
    let mut aggs = Map::new();
    for (i, vr) in root.vertices.iter().enumerate() {
        aggs.insert(format!("field{i}"), field_agg(vr, &controls, true, false));
    }
    let sample = run(json!({"bool": {"must": must}}), sampler(&controls, aggs), &mut timed_out, &mut failures)?;
    let sig = controls.use_significance;
    let total: f64 = (0..root.vertices.len()).filter_map(|i| buckets(sample.get(format!("field{i}")))).flatten().map(|b| bucket_signal(b, sig)).sum();
    for (i, vr) in root.vertices.iter().enumerate() {
        for b in buckets(sample.get(format!("field{i}"))).into_iter().flatten() {
            let w = bucket_signal(b, sig) / total;
            let (bg, fg) = if sig { (u(b, "bg_count"), u(b, "doc_count")) } else { (0, 0) };
            let term = key_string(b);
            if g.get(&vr.field, &term).is_none() {
                g.add_vertex(&vr.field, &term, w, 0, bg, fg);
            }
        }
    }

    // Further hops.
    let mut current = 0usize;
    loop {
        let last_findings: Vec<(String, Vec<usize>)> = g.findings.get(current).cloned().unwrap_or_default();
        if current + 1 >= hops.len() || last_findings.is_empty() {
            break;
        }
        let last = hops[current].clone();
        current += 1;
        let hop = hops[current].clone();
        let mut should = Vec::new();
        include_clauses(&hop, &mut should);
        let n: usize = should.len() + last_findings.iter().map(|(_, v)| v.len()).sum::<usize>();
        for (field, vs) in &last_findings {
            if n < MAX_CLAUSES {
                for &vi in vs {
                    let v = &g.vertices[vi];
                    should.push(json!({"constant_score": {"filter": {"term": {field.clone(): v.term}}, "boost": v.weight}}));
                }
            } else {
                let terms: Vec<&String> = vs.iter().map(|&vi| &g.vertices[vi].term).collect();
                should.push(json!({"constant_score": {"filter": {"terms": {field.clone(): terms}}}}));
            }
        }
        let q = json!({"bool": {"must": [hop.query.clone().unwrap_or_else(|| json!({"match_all": {}})), {"bool": {"should": should}}]}});
        let mut aggs = Map::new();
        for (fi, lvr) in last.vertices.iter().enumerate() {
            let Some((_, vs)) = last_findings.iter().find(|(f, _)| *f == lvr.field) else { continue };
            let mut terms: Vec<String> = vs.iter().map(|&vi| g.vertices[vi].term.clone()).collect();
            terms.sort();
            terms.dedup();
            let mut sub = Map::new();
            for (k, vr) in hop.vertices.iter().enumerate() {
                sub.insert(format!("field{k}"), field_agg(vr, &controls, false, vr.field == lvr.field));
            }
            aggs.insert(
                format!("field{fi}"),
                json!({"terms": {"field": lvr.field, "include": terms, "shard_min_doc_count": 1, "min_doc_count": 1, "execution_hint": "map", "size": terms.len()}, "aggs": sub}),
            );
        }
        let sample = run(q, sampler(&controls, aggs), &mut timed_out, &mut failures)?;
        // Total signal leaving the last hop's vertices (self joins excluded).
        let mut total = 0.0;
        for (j, lvr) in last.vertices.iter().enumerate() {
            for lb in buckets(sample.get(format!("field{j}"))).into_iter().flatten() {
                let lterm = key_string(lb);
                for (k, vr) in hop.vertices.iter().enumerate() {
                    for b in buckets(lb.get(format!("field{k}"))).into_iter().flatten() {
                        if !(vr.field == lvr.field && key_string(b) == lterm) {
                            total += bucket_signal(b, sig);
                        }
                    }
                }
            }
        }
        if total > 0.0 {
            let mut new_vertices: Vec<usize> = Vec::new();
            let mut new_connections: Vec<usize> = Vec::new();
            for (j, lvr) in last.vertices.iter().enumerate() {
                for lb in buckets(sample.get(format!("field{j}"))).into_iter().flatten() {
                    let Some(from) = g.get(&lvr.field, &key_string(lb)) else { continue };
                    let (from_field, from_term) = (g.vertices[from].field.clone(), g.vertices[from].term.clone());
                    for (k, vr) in hop.vertices.iter().enumerate() {
                        for b in buckets(lb.get(format!("field{k}"))).into_iter().flatten() {
                            let term = key_string(b);
                            // ES skips self joins only when scoring by significance.
                            if sig && vr.field == from_field && term == from_term {
                                continue;
                            }
                            // Decayed by the source vertex's current weight (a
                            // self join may have just raised it).
                            let signal = bucket_signal(b, sig) / total * f64::min(0.95, g.vertices[from].weight);
                            let to = match g.get(&vr.field, &term) {
                                None => {
                                    let (bg, fg) = if sig { (u(b, "bg_count"), u(b, "doc_count")) } else { (0, 0) };
                                    let t = g.add_vertex(&vr.field, &term, signal, current, bg, fg);
                                    new_vertices.push(t);
                                    t
                                }
                                Some(t) => {
                                    g.vertices[t].weight += signal;
                                    if sig {
                                        g.vertices[t].fg = g.vertices[t].fg.max(u(b, "doc_count"));
                                    }
                                    t
                                }
                            };
                            new_connections.push(g.add_connection(from, to, signal, u(b, "doc_count")));
                        }
                    }
                }
            }
            // Keep the `size` best new vertices per field.
            let mut evict: Vec<usize> = Vec::new();
            for vr in &hop.vertices {
                if new_vertices.len() <= vr.size {
                    continue;
                }
                let mut mine: Vec<usize> = new_vertices.iter().copied().filter(|&v| g.vertices[v].field == vr.field).collect();
                mine.sort_by(|&a, &b| g.vertices[b].weight.total_cmp(&g.vertices[a].weight));
                evict.extend(mine.into_iter().skip(vr.size));
            }
            if !evict.is_empty() {
                for ci in new_connections {
                    let to = g.connections[ci].0.to;
                    if evict.contains(&to) {
                        g.connections[ci].1 = false;
                        let key = (g.connections[ci].0.from, to);
                        g.conn_index.remove(&key);
                        if !g.vertices[to].removed {
                            g.remove_vertex(to);
                        }
                    }
                }
            }
        }
    }

    // Response.
    let mut numbers: HashMap<usize, usize> = HashMap::new();
    let mut vertices = Vec::new();
    for (i, v) in g.vertices.iter().enumerate().filter(|(_, v)| !v.removed) {
        numbers.insert(i, vertices.len());
        let mut o = json!({"field": v.field, "term": v.term, "weight": v.weight, "depth": v.depth});
        if detailed {
            o["fg"] = json!(v.fg);
            o["bg"] = json!(v.bg);
        }
        vertices.push(o);
    }
    let connections: Vec<Value> = g
        .connections
        .iter()
        .filter(|(_, live)| *live)
        .filter_map(|(c, _)| Some(json!({"source": numbers.get(&c.from)?, "target": numbers.get(&c.to)?, "weight": c.weight, "doc_count": c.doc_count})))
        .collect();
    Ok(Resp::json(
        200,
        json!({"took": start.elapsed().as_millis() as u64, "timed_out": timed_out, "failures": failures, "vertices": vertices, "connections": connections}),
    ))
}
