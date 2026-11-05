//! Scripted updates: `_update` with `script` / `upsert` / `scripted_upsert`,
//! and the `ctx` script contract `_update_by_query` / `_reindex` use
//! ([`run_doc_script`]).

use super::{compile_json, CompiledScript, ScriptContext};
use crate::error::{EsError, Result};
use crate::index::WriteReq;
use crate::shard::{WriteCond, WriteResult};
use serde_json::{json, Map, Value};

/// What a `ctx` script decided for one document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocOp {
    Index,
    Noop,
    Delete,
    Create,
}

/// Document metadata exposed to (and, for reindex, changeable by) a script.
#[derive(Clone, Debug, Default)]
pub struct DocMeta {
    pub index: String,
    pub id: Option<String>,
    pub routing: Option<String>,
    pub version: Option<u64>,
    pub seq_no: Option<i64>,
    pub primary_term: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct DocOutcome {
    pub op: DocOp,
    pub source: Value,
    pub meta: DocMeta,
}

fn op_of(s: &str, ctx: ScriptContext) -> Result<DocOp> {
    Ok(match s {
        "index" => DocOp::Index,
        "create" => DocOp::Create,
        "noop" | "none" => DocOp::Noop,
        "delete" => DocOp::Delete,
        other => {
            let allowed = match ctx {
                ScriptContext::Update => "[index, noop, delete]",
                _ => "[noop, index, delete]",
            };
            return Err(EsError::illegal_argument(format!("Operation type [{other}] not allowed, only {allowed} are allowed")));
        }
    })
}

/// Run a `ctx` script over one document (update, update_by_query, reindex,
/// ingest-like). `initial_op` is `"index"` for existing docs and `"create"`
/// for scripted upserts.
pub fn run_doc_script(script: &CompiledScript, source: Value, meta: &DocMeta, initial_op: &str) -> Result<DocOutcome> {
    Ok(run_doc_script_inner(script,source,meta,initial_op,None)?.0)
}
pub fn run_doc_script_source(script: &CompiledScript, raw: &str, meta: &DocMeta, initial_op: &str) -> Result<(DocOutcome,String)> {
    // A script can introduce unpaired units even when its input source is
    // ordinary Unicode. Always serialize the resulting source losslessly.
    let (out, source) = run_doc_script_inner(script, json!({}), meta, initial_op, Some(raw))?;
    Ok((out, source.expect("raw source execution returns source text")))
}
fn run_doc_script_inner(script: &CompiledScript, source: Value, meta: &DocMeta, initial_op: &str, raw:Option<&str>) -> Result<(DocOutcome,Option<String>)> {
    let mut ctx = Map::new();
    ctx.insert("_source".into(), source);
    ctx.insert("op".into(), json!(initial_op));
    ctx.insert("_index".into(), json!(meta.index));
    ctx.insert("_id".into(), meta.id.clone().map(Value::String).unwrap_or(Value::Null));
    ctx.insert("_routing".into(), meta.routing.clone().map(Value::String).unwrap_or(Value::Null));
    if let Some(v) = meta.version {
        ctx.insert("_version".into(), json!(v));
    }
    if let Some(v) = meta.seq_no {
        ctx.insert("_seq_no".into(), json!(v));
    }
    if let Some(v) = meta.primary_term {
        ctx.insert("_primary_term".into(), json!(v));
    }
    ctx.insert("_now".into(), json!(crate::date::now_millis()));
    let (out, raw_out) = match raw {
        Some(raw)=>{let (out,source)=super::execute_ctx_source(script,ctx,raw)?;(out,Some(source))}
        None=>(super::execute_ctx(script,ctx)?,None),
    };
    let op = match out.get("op") {
        Some(Value::String(s)) => op_of(s, script.program.ctx)?,
        Some(Value::Null) | None => DocOp::Index,
        Some(other) => return Err(EsError::illegal_argument(format!("Operation type [{other}] not allowed"))),
    };
    let source = match out.get("_source") {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        Some(Value::Null) | None => json!({}),
        Some(other) => return Err(EsError::illegal_argument(format!("ctx._source must be a map, got [{other}]"))),
    };
    let mut m = meta.clone();
    if let Some(Value::String(i)) = out.get("_index") {
        m.index = i.clone();
    }
    m.id = match out.get("_id") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) => None,
        Some(v) => Some(v.to_string()),
        None => m.id,
    };
    m.routing = match out.get("_routing") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) => None,
        Some(v) => Some(v.to_string()),
        None => m.routing,
    };
    if let Some(v) = out.get("_version").and_then(Value::as_u64) {
        m.version = Some(v);
    }
    Ok((DocOutcome { op, source, meta: m }, raw_out))
}

/// `_update` with a `script` on one shard copy (see [`crate::index::IndexService::update`]).
pub fn scripted_update(svc: &crate::index::IndexShard, id: &str, body: &Map<String, Value>, raw:Option<&crate::source_json::Source>, routing: Option<String>, retry_on_conflict: usize, cond: WriteCond) -> Result<(WriteResult, Option<String>)> {
    for k in body.keys() {
        if !matches!(k.as_str(), "script" | "upsert" | "scripted_upsert" | "detect_noop" | "_source" | "lang" | "params") {
            if k == "doc" {
                return Err(EsError::action_validation(&["can't provide both script and doc"]));
            }
            return Err(EsError::parsing(format!("[UpdateRequest] unknown field [{k}]")).with("line", 1).with("col", 1));
        }
    }
    let stored = super::stored::for_index_dir(svc.path());
    let script = super::compile_json_source(&body["script"], raw.and_then(|r|r.get("script")), ScriptContext::Update, stored.as_deref())?;
    let upsert = body.get("upsert");
    let scripted_upsert = body.get("scripted_upsert").and_then(Value::as_bool).unwrap_or(false);
    let mut attempt = 0;
    loop {
        let cur = svc.shard.get(id)?;
        let (req, created, outcome_src) = match &cur {
            Some(g) => {
                let meta = DocMeta {
                    index: svc.name.clone(),
                    id: Some(id.to_string()),
                    routing: g.routing.clone(),
                    version: Some(g.version),
                    seq_no: Some(g.seq_no),
                    primary_term: Some(svc.shard.doc_primary_term(id).unwrap_or_else(|| svc.shard.primary_term())),
                };
                let (out, raw_source) = run_doc_script_source(&script,g.source.as_deref().unwrap_or("{}"),&meta,"index")?;
                let mut c = cond.clone();
                if c.if_seq_no.is_none() {
                    c.if_seq_no = Some(g.seq_no);
                    c.if_primary_term = Some(svc.shard.doc_primary_term(id).unwrap_or_else(|| svc.shard.primary_term()));
                }
                match out.op {
                    DocOp::Noop => {
                        return Ok((WriteResult { id: id.to_string(), version: g.version, seq_no: g.seq_no, primary_term: g.primary_term, result: "noop" }, g.source.clone()));
                    }
                    DocOp::Delete => (WriteReq::Delete { id: id.to_string(), routing: routing.clone().or(g.routing.clone()), cond: c }, false, None),
                    DocOp::Index | DocOp::Create => {
                        (WriteReq::Index { id: Some(id.to_string()), source: raw_source.clone(), routing: routing.clone().or(g.routing.clone()), cond: c }, false, Some(raw_source))
                    }
                }
            }
            None => {
                if cond.if_seq_no.is_some() {
                    return Err(EsError::document_missing(&svc.name, &svc.uuid, id));
                }
                let Some(u) = upsert else {
                    return Err(EsError::document_missing(&svc.name, &svc.uuid, id));
                };
                let create = WriteCond { create: true, ..Default::default() };
                if scripted_upsert {
                    let meta = DocMeta { index: svc.name.clone(), id: Some(id.to_string()), routing: routing.clone(), ..Default::default() };
                    let raw_upsert=raw.and_then(|r|r.get("upsert")).map(|s|s.text()).unwrap_or_else(||u.to_string());
                    let (out, source) = run_doc_script_source(&script, &raw_upsert, &meta, "create")?;
                    match out.op {
                        DocOp::Noop | DocOp::Delete => {
                            return Ok((WriteResult { id: id.to_string(), version: 0, seq_no: -2, primary_term: svc.shard.primary_term(), result: "noop" }, None));
                        }
                        _ => {
                            (WriteReq::Index { id: Some(id.to_string()), source: source.clone(), routing: routing.clone(), cond: create }, true, Some(source))
                        }
                    }
                } else {
                    let s = raw.and_then(|r|r.get("upsert")).map(|s|s.text()).unwrap_or_else(||u.to_string());
                    (WriteReq::Index { id: Some(id.to_string()), source: s.clone(), routing: routing.clone(), cond: create }, true, Some(s))
                }
            }
        };
        let r = svc.write(vec![req], true).remove(0);
        match r {
            Ok(mut w) => {
                if created {
                    w.result = "created";
                }
                return Ok((w, outcome_src));
            }
            Err(e) if e.status == 409 && attempt < retry_on_conflict && cond.if_seq_no.is_none() => attempt += 1,
            Err(e) => return Err(e),
        }
    }
}
