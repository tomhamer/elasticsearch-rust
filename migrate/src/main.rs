//! es-lucene-import: offline upgrade path from a Lucene-backed Elasticsearch
//! 8.x node to the Rust server. Reads the node's data directory directly (no
//! running Java process), and replays cluster metadata and every live
//! document's `_source` through the target's REST API.

mod es;
mod lucene;
mod sink;

use anyhow::{anyhow, bail, Context, Result};
use es::state::{self, get_bytes, get_str, IndexMeta};
use serde_json::{json, Map, Value};
use sink::{truncate, Sink};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
es-lucene-import: upgrade a Lucene-backed Elasticsearch 8.x node to the Rust server

USAGE:
  es-lucene-import plan    --data <path.data> [filters]
  es-lucene-import import  --data <path.data> --target <url> [--user u:p] [filters] [options]
  es-lucene-import dump    --data <path.data> --out <dir> [filters]
  es-lucene-import inspect <lucene index dir> [--docs N]

Stop Elasticsearch (or at least POST /_flush) first: operations that exist
only in the translog are not imported, and the tool refuses to run if it
finds any (override with --allow-unflushed).

FILTERS:
  --include <glob,...>   indices / data streams to import (default: all non-dot)
  --exclude <glob,...>   indices / data streams to skip

OPTIONS:
  --skip-metadata        do not import templates, pipelines, scripts, ILM policies
  --overwrite            replace templates/pipelines/... that already exist on the target
  --include-managed      also import Elasticsearch's own built-ins (_meta.managed: true)
  --batch-mb <n>         bulk request size (default 8)
  --allow-unflushed      import even if a shard has unflushed translog operations
";

struct Opts {
    data: Option<PathBuf>,
    target: Option<String>,
    out: Option<PathBuf>,
    user: Option<String>,
    include: Vec<String>,
    exclude: Vec<String>,
    skip_metadata: bool,
    overwrite: bool,
    include_managed: bool,
    batch_bytes: usize,
    allow_unflushed: bool,
    docs: usize,
    positional: Vec<String>,
}

fn parse_args(args: &[String]) -> Result<Opts> {
    let mut o = Opts {
        data: None,
        target: None,
        out: None,
        user: None,
        include: vec![],
        exclude: vec![],
        skip_metadata: false,
        overwrite: false,
        include_managed: false,
        batch_bytes: 8 << 20,
        allow_unflushed: false,
        docs: 5,
        positional: vec![],
    };
    let mut i = 0;
    let list = |s: &str| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect::<Vec<_>>();
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || -> Result<String> {
            i += 1;
            args.get(i).cloned().ok_or_else(|| anyhow!("{} needs a value", a))
        };
        match a {
            "--data" => o.data = Some(PathBuf::from(val()?)),
            "--target" => o.target = Some(val()?),
            "--out" => o.out = Some(PathBuf::from(val()?)),
            "--user" => o.user = Some(val()?),
            "--include" => o.include.extend(list(&val()?)),
            "--exclude" => o.exclude.extend(list(&val()?)),
            "--batch-mb" => o.batch_bytes = val()?.parse::<usize>()? << 20,
            "--docs" => o.docs = val()?.parse()?,
            "--skip-metadata" => o.skip_metadata = true,
            "--overwrite" => o.overwrite = true,
            "--include-managed" => o.include_managed = true,
            "--allow-unflushed" => o.allow_unflushed = true,
            "-h" | "--help" => {
                print!("{}", USAGE);
                std::process::exit(0);
            }
            x if x.starts_with("--") => bail!("unknown option {}", x),
            x => o.positional.push(x.to_string()),
        }
        i += 1;
    }
    Ok(o)
}

fn glob(pat: &str, s: &str) -> bool {
    let (p, t) = (pat.as_bytes(), s.as_bytes());
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}

impl Opts {
    fn selected(&self, name: &str) -> bool {
        if self.exclude.iter().any(|p| glob(p, name)) {
            return false;
        }
        if self.include.is_empty() {
            return !name.starts_with('.');
        }
        self.include.iter().any(|p| glob(p, name))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprint!("{}", USAGE);
        std::process::exit(2);
    }
    let cmd = args[0].clone();
    let res = parse_args(&args[1..]).and_then(|o| match cmd.as_str() {
        "inspect" => inspect(&o),
        "plan" => run(&o, None),
        "import" => {
            let t = o.target.clone().ok_or_else(|| anyhow!("import needs --target <url>"))?;
            run(&o, Some(Sink::http(&t, o.user.as_deref())))
        }
        "dump" => {
            let out = o.out.clone().ok_or_else(|| anyhow!("dump needs --out <dir>"))?;
            run(&o, Some(Sink::dir(out)?))
        }
        "-h" | "--help" => {
            print!("{}", USAGE);
            Ok(())
        }
        x => Err(anyhow!("unknown command {}\n\n{}", x, USAGE)),
    });
    if let Err(e) = res {
        eprintln!("error: {:#}", e);
        std::process::exit(1);
    }
}

fn short(v: &lucene::stored::Value) -> String {
    use lucene::stored::Value::*;
    match v {
        Str(s) => format!("str {:?}", truncate(s, 80)),
        Bytes(b) => {
            let txt = if b.starts_with(es::smile::HEADER) || b.starts_with(b"DFL\0") {
                es::decode_blob(b).map(|v| truncate(&v.to_string(), 120)).unwrap_or_else(|e| format!("<{}>", e))
            } else {
                truncate(&String::from_utf8_lossy(b), 120)
            };
            format!("bytes[{}] {}", b.len(), txt)
        }
        Int(i) => format!("int {}", i),
        Long(l) => format!("long {}", l),
        Other => "float/double".into(),
    }
}

fn inspect(o: &Opts) -> Result<()> {
    let dir = PathBuf::from(o.positional.first().ok_or_else(|| anyhow!("inspect needs a Lucene index directory"))?);
    let (_, path) = lucene::latest_commit_file(&dir)?.ok_or_else(|| anyhow!("no segments_N in {}", dir.display()))?;
    let commit = lucene::read_commit(&path)?;
    println!(
        "commit gen {} lucene {}.{}.{} created-major {} segments {}",
        commit.generation,
        commit.lucene_version.0,
        commit.lucene_version.1,
        commit.lucene_version.2,
        commit.created_major,
        commit.segments.len()
    );
    for (k, v) in &commit.user_data {
        println!("  user_data {} = {}", k, v);
    }
    let mut shown = 0;
    for sc in &commit.segments {
        let seg = lucene::open_segment(&dir, sc)?;
        println!(
            "segment {} codec {} maxDoc {} hard-del {} soft-del {} live {} compound {} sf-mode {:?}",
            sc.name,
            sc.codec,
            seg.max_doc,
            seg.hard_deleted,
            seg.soft_deleted,
            seg.live_count(),
            seg.info.compound,
            seg.info.attributes.iter().find(|(k, _)| k.contains("StoredFields")).map(|(_, v)| v)
        );
        let mut n = 0u32;
        lucene::stored::for_each_doc(&seg, |doc, fields| {
            n += 1;
            if shown < o.docs {
                shown += 1;
                println!("  doc {}", doc);
                for (name, v) in &fields {
                    let extra = if name == "_id" {
                        if let lucene::stored::Value::Bytes(b) = v { format!(" -> {:?}", es::decode_id(b)) } else { String::new() }
                    } else {
                        String::new()
                    };
                    println!("    {} = {}{}", name, short(v), extra);
                }
            }
            Ok(())
        })?;
        if n != seg.live_count() {
            bail!("segment {}: stored fields yielded {} live docs, expected {}", sc.name, n, seg.live_count());
        }
    }
    Ok(())
}

#[derive(Default)]
struct Totals {
    docs: u64,
    nested_children: u64,
    no_source: u64,
    failed: u64,
    conflicts: u64,
}

/// Settings that only make sense on the source cluster, or are applied after
/// the documents are loaded.
fn split_settings(s: &Map<String, Value>) -> (Map<String, Value>, Map<String, Value>) {
    let drop = [
        "index.uuid",
        "index.version.created",
        "index.version.compatibility",
        "index.version.upgraded",
        "index.provided_name",
        "index.history.uuid",
        "index.verified_before_close",
        "index.frozen",
    ];
    let drop_prefix = ["index.resize.", "index.shrink.", "index.routing.allocation.initial_recovery.", "index.store.snapshot."];
    let deferred_prefix = ["index.blocks.", "index.default_pipeline", "index.final_pipeline"];
    let (mut create, mut after) = (Map::new(), Map::new());
    for (k, v) in s {
        if drop.contains(&k.as_str()) || drop_prefix.iter().any(|p| k.starts_with(p)) {
            continue;
        }
        if deferred_prefix.iter().any(|p| k.starts_with(p)) {
            after.insert(k.clone(), v.clone());
        } else {
            create.insert(k.clone(), v.clone());
        }
    }
    (create, after)
}

fn synthetic_source_reason(m: &IndexMeta) -> Option<String> {
    let s = |k: &str| m.settings.get(k).and_then(|v| v.as_str()).map(str::to_string);
    if let Some(mode) = s("index.mode") {
        if mode == "time_series" || mode == "logsdb" {
            return Some(format!("index.mode={}", mode));
        }
    }
    if s("index.mapping.source.mode").as_deref() == Some("synthetic") {
        return Some("index.mapping.source.mode=synthetic".into());
    }
    if m.mappings.pointer("/_source/mode").and_then(|v| v.as_str()) == Some("synthetic") {
        return Some("_source.mode=synthetic".into());
    }
    None
}

struct Ctx<'a> {
    o: &'a Opts,
    node: PathBuf,
    sink: Option<Sink>,
    warnings: Vec<String>,
}

fn warn(w: &mut Vec<String>, s: String) {
    eprintln!("warning: {}", s);
    w.push(s);
}

impl Ctx<'_> {
    /// PUT unless the object already exists on the target (and --overwrite is off).
    fn put_meta(&mut self, kind: &str, path: &str, body: Value) -> Result<()> {
        let overwrite = self.o.overwrite;
        let Some(sink) = self.sink.as_mut() else {
            println!("  {} {}", kind, path);
            return Ok(());
        };
        if !overwrite && sink.exists(path)? {
            println!("  {} {}: exists on target, kept", kind, path);
            return Ok(());
        }
        let (status, resp) = sink.request("PUT", path, Some(&body))?;
        if status >= 300 {
            warn(&mut self.warnings, format!("{} {} failed with {}: {}", kind, path, status, truncate(&resp.to_string(), 300)));
        } else {
            println!("  {} {}: ok", kind, path);
        }
        Ok(())
    }
}

/// Built-ins Elasticsearch installs itself (`_meta.managed: true`); the
/// target server ships its own versions.
fn is_managed(v: &Value) -> bool {
    v.pointer("/_meta/managed").and_then(|m| m.as_bool()) == Some(true)
}

/// Cluster state keeps template mappings in the legacy `{"_doc": {...}}` form.
fn unwrap_template_mappings(mut v: Value) -> Value {
    if let Some(m) = v.pointer_mut("/template/mappings") {
        *m = state::unwrap_type(m.take());
    }
    v
}

fn import_metadata(cx: &mut Ctx, g: &Value) -> Result<()> {
    println!("metadata:");
    let obj = |v: Option<&Value>| v.and_then(|x| x.as_object()).cloned().unwrap_or_default();
    let include_managed = cx.o.include_managed;
    let mut skipped = 0;
    for (name, v) in obj(g.pointer("/component_template/component_template")) {
        if is_managed(&v) && !include_managed {
            skipped += 1;
            continue;
        }
        cx.put_meta("component template", &format!("_component_template/{}", name), unwrap_template_mappings(v))?;
    }
    for (name, v) in obj(g.pointer("/index_template/index_template")) {
        if is_managed(&v) && !include_managed {
            skipped += 1;
            continue;
        }
        cx.put_meta("index template", &format!("_index_template/{}", name), unwrap_template_mappings(v))?;
    }
    for (name, v) in obj(g.get("templates")) {
        let mut body = Map::new();
        for k in ["index_patterns", "order", "version", "settings", "aliases"] {
            if let Some(x) = v.get(k) {
                body.insert(k.into(), x.clone());
            }
        }
        if let Some(m) = state::first_mapping(v.get("mappings")) {
            body.insert("mappings".into(), m);
        }
        cx.put_meta("legacy template", &format!("_template/{}", name), Value::Object(body))?;
    }
    let pipelines = g.pointer("/ingest/pipeline").and_then(|p| p.as_array()).cloned().unwrap_or_default();
    for p in pipelines {
        let (Some(id), Some(cfg)) = (p.get("id").and_then(|i| i.as_str()), p.get("config")) else { continue };
        if is_managed(cfg) && !include_managed {
            skipped += 1;
            continue;
        }
        cx.put_meta("ingest pipeline", &format!("_ingest/pipeline/{}", id), cfg.clone())?;
    }
    for (id, v) in obj(g.get("stored_scripts")) {
        cx.put_meta("stored script", &format!("_scripts/{}", id), json!({ "script": v }))?;
    }
    for (name, v) in obj(g.pointer("/index_lifecycle/policies")) {
        if v.get("policy").map_or(false, is_managed) && !include_managed {
            skipped += 1;
            continue;
        }
        let mut policy = v.get("policy").cloned().unwrap_or(v.clone());
        if let Some(p) = policy.as_object_mut() {
            p.remove("name");
        }
        cx.put_meta("ILM policy", &format!("_ilm/policy/{}", name), json!({ "policy": policy }))?;
    }
    if skipped > 0 {
        println!("  skipped {} Elasticsearch-managed built-ins (_meta.managed; use --include-managed to import them)", skipped);
    }
    if let Some(s) = g.get("settings").and_then(|s| s.as_object()).filter(|s| !s.is_empty()) {
        warn(&mut cx.warnings, format!("persistent cluster settings not imported (review and apply by hand): {}", Value::Object(s.clone())));
    }
    Ok(())
}

/// Streams one shard's live documents into bulk requests.
fn import_shard(cx: &mut Ctx, meta: &IndexMeta, shard: u32, dest: &str, data_stream: bool, t: &mut Totals) -> Result<()> {
    let shard_dir = cx.node.join("indices").join(&meta.uuid).join(shard.to_string());
    let dir = shard_dir.join("index");
    if !dir.is_dir() {
        warn(&mut cx.warnings, format!("{}[{}]: no shard copy on this node ({}), skipped", meta.name, shard, dir.display()));
        return Ok(());
    }
    // refuse silently losing translog-only operations
    let (_, commit_path) = lucene::latest_commit_file(&dir)?.ok_or_else(|| anyhow!("{}: no commit", dir.display()))?;
    let commit = lucene::read_commit(&commit_path)?;
    let committed_max: i64 = commit.user_data.get("max_seq_no").and_then(|s| s.parse().ok()).unwrap_or(-1);
    if let Some(tl) = state::read_translog_checkpoint(&shard_dir)? {
        let beyond = tl.max_seq_no.max(tl.global_checkpoint);
        if beyond > committed_max {
            let msg = format!(
                "{}[{}]: translog holds operations up to seq_no {} beyond the last commit ({}); start Elasticsearch, POST /_flush, stop it cleanly and rerun",
                meta.name, shard, beyond, committed_max
            );
            if !cx.o.allow_unflushed {
                bail!(msg);
            }
            warn(&mut cx.warnings, msg);
        }
    }
    let op = if data_stream { "create" } else { "index" };
    let label = if data_stream { dest.to_string() } else { meta.name.clone() };
    let batch_bytes = cx.o.batch_bytes;
    let mut body = String::with_capacity(batch_bytes + (1 << 20));
    let mut first_errors: Vec<String> = Vec::new();
    let sink = &mut cx.sink;
    let mut flush = |body: &mut String, t: &mut Totals, errs: &mut Vec<String>| -> Result<()> {
        if body.is_empty() {
            return Ok(());
        }
        if let Some(s) = sink.as_mut() {
            let r = s.bulk(&label, "?pipeline=_none", body)?;
            t.failed += r.errors.len() as u64;
            t.conflicts += r.conflicts as u64;
            for e in r.errors {
                if errs.len() < 10 {
                    errs.push(e);
                }
            }
        }
        body.clear();
        Ok(())
    };
    let (_, st) = state::scan_lucene_dir(&dir, |f| {
        let Some(id) = get_bytes(f, "_id") else {
            t.nested_children += 1;
            return Ok(());
        };
        let id = es::decode_id(id);
        let Some(src) = get_bytes(f, "_source") else {
            t.no_source += 1;
            return Ok(());
        };
        let src = match es::source_to_json(src) {
            Ok(s) => s,
            Err(e) => {
                t.failed += 1;
                if first_errors.len() < 10 {
                    first_errors.push(format!("[{}] {}", id, e));
                }
                return Ok(());
            }
        };
        let mut action = Map::new();
        action.insert("_index".into(), Value::String(dest.to_string()));
        action.insert("_id".into(), Value::String(id));
        if let Some(r) = get_str(f, "_routing") {
            action.insert("routing".into(), Value::String(r.to_string()));
        }
        body.push_str(&json!({ op: action }).to_string());
        body.push('\n');
        body.push_str(&src);
        body.push('\n');
        t.docs += 1;
        if body.len() >= batch_bytes {
            flush(&mut body, t, &mut first_errors)?;
        }
        Ok(())
    })?;
    flush(&mut body, t, &mut first_errors)?;
    println!(
        "  {}[{}]: {} segments, {} lucene docs, {} deleted ({} soft)",
        meta.name,
        shard,
        st.segments,
        st.max_doc,
        st.hard_deleted + st.soft_deleted,
        st.soft_deleted
    );
    for e in first_errors {
        warn(&mut cx.warnings, format!("{}: {}", meta.name, e));
    }
    Ok(())
}

fn run(o: &Opts, sink: Option<Sink>) -> Result<()> {
    let data = o.data.clone().ok_or_else(|| anyhow!("--data <path.data> is required"))?;
    let node = state::locate_node_dir(&data)?;
    let started = Instant::now();
    let cs = state::read_cluster_state(&node).context("reading the persisted cluster state")?;
    let g = state::global_meta(&cs).clone();
    let metas = state::index_metas(&cs)?;
    println!(
        "source: {} (cluster {}, node {}), {} indices",
        node.display(),
        g.get("cluster_uuid").and_then(|v| v.as_str()).unwrap_or("?"),
        cs.user_data.get("node_id").map(String::as_str).unwrap_or("?"),
        metas.len()
    );
    let mut cx = Ctx { o, node: node.clone(), sink, warnings: vec![] };

    // backing index -> data stream
    let mut backing: BTreeMap<String, String> = BTreeMap::new();
    let streams = g.pointer("/data_stream/data_stream").and_then(|d| d.as_object()).cloned().unwrap_or_default();
    for (name, ds) in &streams {
        for ix in ds.get("indices").and_then(|i| i.as_array()).into_iter().flatten() {
            if let Some(n) = ix.get("index_name").and_then(|n| n.as_str()) {
                backing.insert(n.to_string(), name.clone());
            }
        }
        for ix in ds.pointer("/failure_store/indices").or_else(|| ds.get("failure_indices")).and_then(|i| i.as_array()).into_iter().flatten() {
            if let Some(n) = ix.get("index_name").and_then(|n| n.as_str()) {
                backing.insert(n.to_string(), format!("{}::failures", name));
            }
        }
    }

    if !o.skip_metadata {
        import_metadata(&mut cx, &g)?;
    }

    let mut totals = Totals::default();
    let mut expected: BTreeMap<String, u64> = BTreeMap::new();
    let mut created_streams: Vec<String> = vec![];
    let mut mismatches = 0;
    println!("indices:");
    for meta in &metas {
        let ds = backing.get(&meta.name).cloned();
        let sel_name = ds.clone().unwrap_or_else(|| meta.name.clone());
        if !o.selected(&sel_name) {
            continue;
        }
        if let Some(r) = synthetic_source_reason(meta) {
            warn(&mut cx.warnings, format!("{}: skipped, uses synthetic _source ({}); reindex it in Elasticsearch to a stored-source index first", meta.name, r));
            continue;
        }
        if let Some(d) = &ds {
            if d.ends_with("::failures") {
                warn(&mut cx.warnings, format!("{}: failure-store index of {} skipped", meta.name, d.trim_end_matches("::failures")));
                continue;
            }
        }
        let before = totals.docs;
        let dest = ds.clone().unwrap_or_else(|| meta.name.clone());
        let (create, after) = split_settings(&meta.settings);
        if let Some(sink) = cx.sink.as_mut() {
            if let Some(d) = &ds {
                if !created_streams.contains(d) {
                    created_streams.push(d.clone());
                    if !sink.exists(&format!("_data_stream/{}", d))? {
                        let (st, r) = sink.request("PUT", &format!("_data_stream/{}", d), None)?;
                        if st >= 300 {
                            mismatches += 1;
                            warn(&mut cx.warnings, format!("data stream {}: create failed with {}: {}", d, st, truncate(&r.to_string(), 300)));
                        }
                    }
                }
            } else {
                let mut body = json!({ "settings": create, "mappings": meta.mappings });
                if !meta.aliases.is_empty() {
                    body["aliases"] = Value::Object(meta.aliases.clone());
                }
                let (st, r) = sink.request("PUT", &meta.name, Some(&body))?;
                if st >= 300 {
                    warn(&mut cx.warnings, format!("{}: create failed with {}: {}; skipped", meta.name, st, truncate(&r.to_string(), 400)));
                    mismatches += 1;
                    continue;
                }
            }
        }
        println!("  {}{} ({} shards, {})", meta.name, ds.as_ref().map(|d| format!(" -> data stream {}", d)).unwrap_or_default(), meta.num_shards, meta.state);
        for shard in 0..meta.num_shards {
            import_shard(&mut cx, meta, shard, &dest, ds.is_some(), &mut totals)?;
        }
        *expected.entry(dest.clone()).or_default() += totals.docs - before;
        if let Some(sink) = cx.sink.as_mut() {
            if ds.is_none() {
                sink.request("POST", &format!("{}/_refresh", meta.name), None)?;
                if !after.is_empty() {
                    let (st, r) = sink.request("PUT", &format!("{}/_settings", meta.name), Some(&Value::Object(after.clone())))?;
                    if st >= 300 {
                        warn(&mut cx.warnings, format!("{}: re-applying {:?} failed with {}: {}", meta.name, after.keys().collect::<Vec<_>>(), st, truncate(&r.to_string(), 300)));
                    }
                }
                if meta.state == "close" {
                    if !sink.is_dry() {
                        let (_, r) = sink.request("GET", &format!("{}/_count", meta.name), None)?;
                        let got = r.get("count").and_then(|c| c.as_u64());
                        let n = expected.remove(&dest).unwrap_or(0);
                        if got != Some(n) {
                            mismatches += 1;
                        }
                        println!("  {}: verified {} of {} docs before closing", meta.name, got.unwrap_or(0), n);
                    }
                    sink.request("POST", &format!("{}/_close", meta.name), None)?;
                }
            }
        }
    }

    // verify counts on the target
    if let Some(sink) = cx.sink.as_mut().filter(|s| !s.is_dry()) {
        println!("verify:");
        for (dest, n) in &expected {
            sink.request("POST", &format!("{}/_refresh", dest), None)?;
            let (st, r) = sink.request("GET", &format!("{}/_count", dest), None)?;
            let got = r.get("count").and_then(|c| c.as_u64());
            let ok = st == 200 && got == Some(*n);
            if !ok {
                mismatches += 1;
            }
            println!("  {}: expected {} got {} {}", dest, n, got.map(|g| g.to_string()).unwrap_or_else(|| format!("({})", st)), if ok { "ok" } else { "MISMATCH" });
        }
    }
    println!(
        "done in {:.1}s: {} docs, {} nested child docs (carried in parent _source), {} without _source (not importable), {} failed, {} conflicts; {} warnings",
        started.elapsed().as_secs_f64(),
        totals.docs,
        totals.nested_children,
        totals.no_source,
        totals.failed,
        totals.conflicts,
        cx.warnings.len()
    );
    if totals.failed > 0 || mismatches > 0 {
        bail!("import incomplete: {} failed docs, {} count mismatches", totals.failed, mismatches);
    }
    let _ = Path::new("");
    Ok(())
}
